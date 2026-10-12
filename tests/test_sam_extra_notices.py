"""sam-extra 결과·요청 알림 (P4) — core/sam_extra_notices + 백엔드·워커·UI 연결.

infotext 샘플은 확장 v0.30.0 이 쓰는 모양 그대로다(A1111 create_infotext: 쉼표·콜론이 든 값만 JSON 따옴표).
- scripts/!sam3.py: process() 검증 실패 → 'SAM3 Enable' 없이 'SAM3 Error: <pydantic 한 줄>',
  postprocess_image 실패 → 'SAM3 Enable' 을 걷고 'SAM3 Error: "<예외 이름>: <메시지>"', 성공 → 'SAM3 Enable: True'
  + extra_params() + 'SAM3 Version'
- scripts/anima_3_8b.py: 'Anima38: v2 bundle | v1 adapter | bypass | off: <이유>' (+ 'Anima38 …' 세부 키)
- scripts/anima_safe_pag.py: perturbation 이 붙었을 때만 'Anima Perturbation Guidance' + 'Anima PAG prefix dedup'
  (패스마다 지우고 다시 쓴다 — OOM 폴백 뒤 하이레스·다음 배치에는 없다)
마지막 클래스는 설치된 확장 소스에 이 키들이 그대로 있는지 AST 로 본다(없으면 skip).
"""
from __future__ import annotations

import ast
import base64
import copy
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from core import anima38, anima_guidance, sam3_args
from core import sam_extra_notices as sn
from core.sam_extra_capabilities import EP_SCRIPT_INFO, EP_SCRIPTS, HttpResult, build_capabilities
from tests._sam_extra_ext import EXT_ROOT, requires_extension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

# ── 녹화 모양의 infotext ──────────────────────────────────────────────────────
_HEAD = ("1girl, solo, looking at viewer, smile, upper body\n"
         "Negative prompt: worst quality, low quality, bad anatomy\n"
         "Steps: 28, Sampler: Euler a, Schedule type: Automatic, CFG scale: 4.5, Seed: 1234567890, "
         "Size: 832x1216, Model hash: 5c1e3f9a2b, Model: anima-preview3, ")
_TAIL = "Version: f2.0.1v1.10.1-previous-669-gdfdcbab6"
_SAM3_PARAMS = ('SAM3 Mode: Inpaint, SAM3 Mask Mode: Individual, SAM3 Prompt: face, SAM3 Threshold: 0.4, '
                'SAM3 Checkpoint: "C:\\\\sd-webui-forge-classic\\\\models\\\\sam3\\\\sam3.pt", SAM3 Device: cuda, ')
SAM3_OK = _HEAD + "SAM3 Enable: True, " + _SAM3_PARAMS + "SAM3 Version: 0.30.0, " + _TAIL
SAM3_CN_KEYERROR = (_HEAD + _SAM3_PARAMS + "SAM3 CN Enable: True, SAM3 CN Model: anima-lllite-inpainting-v2, "
                    "SAM3 CN Module: none, SAM3 Version: 0.30.0, SAM3 Error: \"KeyError: 'none'\", " + _TAIL)
SAM3_VALIDATION = (_HEAD + "SAM3 Error: 1 validation error for Sam3Args sam3_seed value is not a valid integer "
                   "(type=type_error.integer), " + _TAIL)
SAM3_OOM = (_HEAD + _SAM3_PARAMS + "SAM3 Version: 0.30.0, SAM3 Error: \"OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 1.50 GiB. GPU 0 has a total capacity of 15.99 GiB of which 312.00 MiB is free.\", "
            + _TAIL)
SAM3_GATED = (_HEAD + _SAM3_PARAMS + "SAM3 Version: 0.30.0, SAM3 Error: \"GatedRepoError: 401 Client Error. "
              "Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/sam3.pt.\", " + _TAIL)
NO_EXTENSION_TRACE = _HEAD + _TAIL

ANIMA38_V2 = (_HEAD + "Anima38: v2 bundle, Anima38 adapter: anima-3.8b-v2.safetensors, Anima38 strength: 1.0, "
              "Anima38 architecture: anima_qwen35_quality_anchored_semantic_connector_v2, "
              "Anima38 bundle: anima-3.8b-v2.safetensors, Anima38 negative: native, "
              "Anima38 encoder: qwen35_4b.safetensors, " + _TAIL)
ANIMA38_OFF_QWEN = (_HEAD + 'Anima38: "off: qwen35_4b.safetensors was not found in models/text_encoder.", ' + _TAIL)
ANIMA38_OFF_INSTALL = _HEAD + 'Anima38: "off: install failed (OutOfMemoryError)", ' + _TAIL
ANIMA38_LEGACY_OFF = _HEAD + 'Anima 3.8B: "off: install failed (RuntimeError)", ' + _TAIL
# 확장의 FileNotFoundError 문구 그대로 (sam3ext/anima38/files.py tokenizer_dir, runtime.py _qwen35_path·
# _load_adapter) — ExtensionSourceTests 가 설치된 확장의 문구와 같은 힌트로 가는지 다시 본다.
EXT_TOKENIZER_MISSING = ("Qwen3.5 tokenizer files are missing from the extension's assets/qwen35_tokenizer "
                         "directory. Reinstall the extension.")
EXT_QWEN35_MISSING = "qwen35_4b.safetensors was not found in models/text_encoder."
EXT_ADAPTER_MISSING = "Adapter 'anima38/qwen35_adapter.safetensors' is unavailable. Refresh Forge and select it again."


def anima38_off(message):
    return _HEAD + f"Anima38: {json.dumps('off: ' + message)}, " + _TAIL


# 부분 LoRA 추측 변환 기록 (scripts/anima_lora_blocks.py _record_infotext)
SPARSE_GUESS = _HEAD + 'Anima sparse LoRA: "Forge guess (char_a 28->40, style_b 40->52)", ' + _TAIL

_PAG_VALUE = ('"PAG mode=official scale=4.0 strength=0.75 blocks=[8, 18] heads=all; requested_range=0.00-0.70; '
              'effective_range=0.00-0.70; range_mode=continuous; rescale=0.2(full)"')
PAG_OK = _HEAD + f"Anima Perturbation Guidance: {_PAG_VALUE}, Anima PAG prefix dedup: True, " + _TAIL
PAG_GONE = _HEAD + "Anima APG: \"eta=0.0, norm=15.0, momentum=0.0\", " + _TAIL   # 가이던스는 돌았지만 PAG 키 없음


def info_of(*infotexts, extra=None, first=0):
    """Forge ``Processed.js()`` 모양의 info JSON 문자열."""
    return json.dumps({
        "prompt": "1girl", "seed": 1234567890, "cfg_scale": 4.5, "steps": 28,
        "extra_generation_params": extra if extra is not None else {},
        "index_of_first_image": first, "infotexts": list(infotexts), "version": "f2.0.1v1.10.1",
    })


def sam3_payload(**overrides):
    state = sam3_args.default_settings()
    state.update(overrides)
    state.update(sam3_enable=True, enabled=True)
    return {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {sam3_args.SCRIPT_SAM3: {"args": [state]}}}


def pag_payload(cfg_scale=4.5, **settings):
    payload = {"prompt": "1girl", "cfg_scale": cfg_scale, "alwayson_scripts": anima_guidance.build_alwayson(settings)}
    return payload


def codes(notices):
    return [n.code for n in notices]


class _QuietAssumption(unittest.TestCase):
    """스냅샷 없음 → 'v0.30 기준' 경고 로그는 프로세스당 한 번이다. 테스트 순서에 따라 나오지 않게 막아 둔다."""

    def setUp(self):
        patcher = mock.patch.object(sn, "_ASSUMED_LOGGED", True)
        patcher.start()
        self.addCleanup(patcher.stop)


# ── 요청 해석 ─────────────────────────────────────────────────────────────────
class RequestedFeatureTests(unittest.TestCase):
    def test_sam3_dict_form_is_read_like_the_extension(self):
        req = sn.requested_features(sam3_payload(sam3_checkpoint="sam3.pt"))
        self.assertTrue(req.sam3)
        self.assertEqual(req.sam3_checkpoint, "sam3.pt")
        off = sam3_payload()
        off["alwayson_scripts"]["SAM3 Mask"]["args"][0].update(sam3_enable=False, enabled=False)
        self.assertFalse(sn.requested_features(off).sam3)
        # 위치 형태 [True, state] 와 대소문자가 다른 제목도 Forge 처럼 받는다
        positional = {"alwayson_scripts": {"sam3 mask": {"args": [True, {"sam3_checkpoint": "x.pt"}]}}}
        self.assertTrue(sn.requested_features(positional).sam3)

    def test_pag_parts_and_cfg_bases_follow_extension_rules(self):
        req = sn.requested_features(pag_payload(guid_enabled=True, guid_slg_on=True))
        self.assertTrue(req.pag)
        self.assertEqual(req.pag_parts, ("PAG", "SLG"))
        self.assertEqual(req.cfg_bases, ())          # 62개 인자면 마스터 토글(57)이 명시적 — 프리셋 'Auto' 로 켜지지 않음
        req = sn.requested_features(pag_payload(guid_smc_master_enabled=True, guid_cwm_enabled=True,
                                                guid_cwm_alpha_low=0.3))
        self.assertFalse(req.pag)
        self.assertEqual(req.cfg_bases, ("SMC", "CWM"))
        # CWM 은 alpha 가 하나라도 0 이 아닐 때만 건다(원본 cwm_alpha_active — origin:
        # namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:817-821, 뜻만). 기본 alpha 0/0 이면 평범한 CFG 와 같다.
        self.assertEqual(sn.requested_features(pag_payload(guid_cwm_enabled=True)).cfg_bases, ())
        self.assertEqual(sn.requested_features(pag_payload(guid_cwm_enabled=True, guid_cwm_alpha_high=-0.2)).cfg_bases,
                         ("CWM",))
        self.assertEqual(sn.requested_features(pag_payload(guid_dcw_enabled=True, guid_cfg_mode="SMC + CWM")).cfg_bases,
                         ("SMC",))
        # 켜기만 하고 방법이 None 이거나 scale 0 이면 perturbation 없음
        self.assertFalse(sn.requested_features(pag_payload(guid_enabled=True, guid_attn_method="None")).pag)
        self.assertFalse(sn.requested_features(pag_payload(guid_enabled=True, guid_scale=0.0)).pag)
        # APG 체크박스 + cfg_mode 보존 → APG
        self.assertEqual(sn.requested_features(pag_payload(guid_apg_enabled=True)).cfg_bases, ("APG",))

    def test_old_build_truncation_turns_auto_preset_into_smc(self):
        """v0.21.2(57개): Forge 가 뒤 인자를 잘라 마스터 토글이 사라지고 'Auto' 프리셋이 SMC 를 켠다(나-5)."""
        payload = pag_payload(guid_dcw_enabled=True)
        self.assertEqual(sn.requested_features(payload).cfg_bases, ())
        self.assertEqual(sn.requested_features(payload, live_pag_argc=57).cfg_bases, ("SMC",))

    def test_anima38_block_is_optional(self):
        self.assertIsNone(sn.requested_features({"alwayson_scripts": {}}).anima38)
        block = {anima38.SCRIPT_NAME: {"args": [{"enabled": True}]}}
        self.assertTrue(sn.requested_features({"alwayson_scripts": block}).anima38_expected)
        block = {anima38.SCRIPT_NAME: {"args": [True, "a.safetensors", 1.0, False, 1.0, True]}}
        self.assertFalse(sn.requested_features({"alwayson_scripts": block}).anima38_expected)  # bypass

    def test_garbage_payloads_do_not_raise(self):
        for payload in (None, "x", {"alwayson_scripts": "x"}, {"alwayson_scripts": {"SAM3 Mask": "x"}},
                        {"alwayson_scripts": {"Anima Perturbation Guidance": {"args": [{"enabled": True}]}}}):
            req = sn.requested_features(payload)
            self.assertFalse(req.sam3 or req.pag)


# ── infotext 해석 ─────────────────────────────────────────────────────────────
class ImageParameterTests(unittest.TestCase):
    def test_infotexts_are_parsed_per_image(self):
        params = sn.image_parameters(info_of(SAM3_OK, SAM3_CN_KEYERROR))
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0]["SAM3 Enable"], "True")
        self.assertEqual(params[1]["SAM3 Error"], "KeyError: 'none'")
        self.assertNotIn("SAM3 Enable", params[1])

    def test_unquoted_validation_error_is_read_up_to_the_comma(self):
        params = sn.image_parameters(info_of(SAM3_VALIDATION))[0]
        self.assertEqual(params["SAM3 Error"], "1 validation error for Sam3Args sam3_seed value is not a valid "
                                               "integer (type=type_error.integer)")

    def test_grid_infotext_is_skipped(self):
        params = sn.image_parameters(info_of(SAM3_OK, SAM3_CN_KEYERROR, first=1))
        self.assertEqual(len(params), 1)
        self.assertIn("SAM3 Error", params[0])

    def test_extra_generation_params_fallback_and_dict_info(self):
        info = {"extra_generation_params": {"SAM3 Error": "KeyError: 'none'"}, "infotexts": []}
        self.assertEqual(sn.image_parameters(info), [{"SAM3 Error": "KeyError: 'none'"}])
        self.assertEqual(sn.image_parameters({"raw_info": "x"}), [])
        self.assertEqual(sn.image_parameters("not json"), [])


# ── 결과 알림 ─────────────────────────────────────────────────────────────────
class ResultNoticeTests(_QuietAssumption):
    def test_success_has_no_notice(self):
        self.assertEqual(sn.result_notices(info_of(SAM3_OK), sam3_payload()), [])
        self.assertEqual(sn.result_notices(info_of(NO_EXTENSION_TRACE), {"prompt": "x"}), [])

    def test_cn_module_keyerror_names_the_setting(self):
        notices = sn.result_notices(info_of(SAM3_CN_KEYERROR), sam3_payload(sam3_cn_enable=True, sam3_cn_module="none"))
        self.assertEqual(codes(notices), [sn.CODE_SAM3_ERROR])
        notice = notices[0]
        self.assertEqual(notice.level, sn.LEVEL_WARNING)
        self.assertIn("SAM3 없이 저장", notice.message)
        self.assertIn("sam3_cn_module", notice.message)
        self.assertIn("'None'", notice.message)
        self.assertEqual(notice.detail, "KeyError: 'none'")
        self.assertIn("sam3_cn_module", notice.hint)

    def test_validation_error_lists_fields(self):
        notice = sn.result_notices(info_of(SAM3_VALIDATION), sam3_payload())[0]
        self.assertIn("검증", notice.message)
        self.assertIn("Seed(sam3_seed)", notice.message)

    def test_oom_and_gated_download_hints(self):
        oom = sn.result_notices(info_of(SAM3_OOM), sam3_payload())[0]
        self.assertIn("VRAM", oom.message)
        gated = sn.result_notices(info_of(SAM3_GATED), sam3_payload())[0]
        self.assertIn("facebook/sam3", gated.message)
        self.assertIn("models/sam3", gated.message)

    def test_batch_counts_failed_images(self):
        notices = sn.result_notices(info_of(SAM3_CN_KEYERROR, SAM3_OK, SAM3_CN_KEYERROR), sam3_payload())
        self.assertEqual(len(notices), 1)
        self.assertIn("2/3장에서", notices[0].message)

    def test_error_is_reported_even_without_payload(self):
        self.assertEqual(codes(sn.result_notices(info_of(SAM3_OOM))), [sn.CODE_SAM3_ERROR])

    def test_sam3_requested_but_no_trace(self):
        notices = sn.result_notices(info_of(NO_EXTENSION_TRACE), sam3_payload())
        self.assertEqual(codes(notices), [sn.CODE_SAM3_NOT_APPLIED])
        self.assertIn("SAM3 Enable", notices[0].message)

    def test_extra_generation_params_dict_form(self):
        info = {"extra_generation_params": {"SAM3 Error": "OutOfMemoryError: CUDA out of memory.",
                                            "SAM3 Version": "0.30.0"}}
        self.assertEqual(codes(sn.result_notices(info, sam3_payload())), [sn.CODE_SAM3_ERROR])
        ok = {"extra_generation_params": {"SAM3 Enable": True, "SAM3 Version": "0.30.0"}}
        self.assertEqual(sn.result_notices(ok, sam3_payload()), [])

    def test_anima38_off_reasons(self):
        qwen = sn.result_notices(info_of(ANIMA38_OFF_QWEN))
        self.assertEqual(codes(qwen), [sn.CODE_ANIMA38_OFF])
        self.assertIn("qwen35_4b.safetensors", qwen[0].message)
        self.assertIn("text_encoder", qwen[0].message)
        install = sn.result_notices(info_of(ANIMA38_OFF_INSTALL))[0]
        self.assertIn("OutOfMemoryError", install.message)
        self.assertEqual(codes(sn.result_notices(info_of(ANIMA38_LEGACY_OFF))), [sn.CODE_ANIMA38_OFF])
        self.assertEqual(sn.result_notices(info_of(ANIMA38_V2)), [])

    def test_anima38_off_file_messages_point_at_the_right_fix(self):
        # 토크나이저 문구에도 'qwen35'('assets/qwen35_tokenizer')가 있다 — 텍스트 인코더 힌트로 새면
        # 있는 모델 파일을 찾게 만든다. 어댑터 이름에 qwen35 가 들어가도 어댑터 힌트여야 한다.
        tokenizer = sn.result_notices(info_of(anima38_off(EXT_TOKENIZER_MISSING)))[0].message
        self.assertIn("다시 설치", tokenizer)
        self.assertNotIn("qwen35_4b.safetensors 가 없습니다", tokenizer)
        encoder = sn.result_notices(info_of(anima38_off(EXT_QWEN35_MISSING)))[0].message
        self.assertIn("models/text_encoder 에 qwen35_4b.safetensors 가 없습니다", encoder)
        adapter = sn.anima38_off_hint("off: " + EXT_ADAPTER_MISSING)
        self.assertIn("어댑터", adapter)
        self.assertIn("(OutOfMemoryError)", sn.anima38_off_hint("off: install failed (OutOfMemoryError)"))

    def test_sam3_hints_do_not_guess_from_bare_substrings(self):
        generic = "Forge 콘솔의 '[-] SAM3: failed' 줄에서 원인을 확인하세요"
        # 텐서 모양의 4032 는 HTTP 403 이 아니다 (손상·불일치 체크포인트)
        shape = ("RuntimeError: Error(s) in loading state_dict for Sam3Image: size mismatch for "
                 "detector.backbone.pos: copying a param with shape torch.Size([1, 4032, 256]) from checkpoint")
        self.assertEqual(sn.sam3_error_hint(shape), generic)
        # 같은 장치 오류는 SAM3 Device 설정 문제가 아니다
        same_device = ("RuntimeError: Expected all tensors to be on the same device, but found at least two "
                       "devices, cuda:0 and cpu!")
        self.assertEqual(sn.sam3_error_hint(same_device), generic)
        # HF 캐시 경로가 든 로드 오류는 승인·로그인 문제가 아니다
        cache = ("RuntimeError: PytorchStreamReader failed reading zip archive: C:/Users/u/.cache/huggingface/hub/"
                 "models--facebook--sam3/snapshots/abc/sam3.pt")
        self.assertEqual(sn.sam3_error_hint(cache), generic)

    def test_sam3_hints_still_catch_the_real_cases(self):
        self.assertIn("facebook/sam3", sn.sam3_error_hint(
            "HfHubHTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt"))
        self.assertIn("facebook/sam3", sn.sam3_error_hint("GatedRepoError: 401 Client Error. Cannot access gated repo"))
        self.assertIn("facebook/sam3", sn.sam3_error_hint(
            "LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub"))
        self.assertIn("SAM3 Device", sn.sam3_error_hint("RuntimeError: CUDA error: invalid device ordinal"))
        self.assertIn("SAM3 Device", sn.sam3_error_hint("AssertionError: Torch not compiled with CUDA enabled"))
        # 확장의 '체크포인트 없음' 문구에는 huggingface.co URL 이 있다 — 다운로드 실패가 아니다
        missing = ("FileNotFoundError: SAM3 checkpoint not found: C:/forge/models/sam3/sam3_b.pt. Put it in "
                   "<webui>/models/sam3/ (official weights: https://huggingface.co/facebook/sam3).")
        self.assertIn("SAM3 Checkpoint", sn.sam3_error_hint(missing))

    def test_repeating_failures_are_config_problems_only(self):
        def first(infotext, **state):
            return sn.result_notices(info_of(infotext), sam3_payload(**state))[0]

        self.assertTrue(sn.sam3_failure_repeats(first(SAM3_CN_KEYERROR, sam3_cn_module="none")))
        self.assertTrue(sn.sam3_failure_repeats(first(SAM3_VALIDATION)))
        self.assertTrue(sn.sam3_failure_repeats(first(SAM3_GATED)))
        self.assertTrue(sn.sam3_failure_repeats(first(NO_EXTENSION_TRACE)))     # 'SAM3 Enable' 없음
        self.assertFalse(sn.sam3_failure_repeats(first(SAM3_OOM)))              # 이미지마다 다를 수 있다
        unknown = _HEAD + 'SAM3 Version: 0.30.0, SAM3 Error: "RuntimeError: boom", ' + _TAIL
        self.assertFalse(sn.sam3_failure_repeats(first(unknown)))
        self.assertFalse(sn.sam3_failure_repeats(None))

    def test_sparse_lora_guess_is_an_info_notice(self):
        notices = sn.result_notices(info_of(SPARSE_GUESS, SPARSE_GUESS))
        self.assertEqual(codes(notices), [sn.CODE_LORA_SPARSE_GUESS])
        self.assertEqual(notices[0].level, sn.LEVEL_INFO)
        self.assertIn("char_a 28->40", notices[0].message)
        self.assertIn("추측 변환", notices[0].message)
        self.assertEqual(sn.result_notices(info_of(NO_EXTENSION_TRACE)), [])
        # 같은 LoRA 묶음이면 생성마다 기록된다 — 생성 전 경고처럼 드물게
        self.assertEqual(sn.notice_ttl(notices[0], sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
        self.assertEqual(sn.notice_ttl(sn.Notice("x", sn.LEVEL_WARNING, "m"), 30.0), 30.0)

    def test_anima38_expected_but_missing(self):
        payload = {"alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [{"enabled": True}]}}}
        self.assertEqual(codes(sn.result_notices(info_of(NO_EXTENSION_TRACE), payload)),
                         [sn.CODE_ANIMA38_NOT_APPLIED])
        self.assertEqual(sn.result_notices(info_of(ANIMA38_V2), payload), [])

    def test_pag_dropped_after_oom_on_later_images(self):
        payload = pag_payload(guid_enabled=True)
        notices = sn.result_notices(info_of(PAG_OK, PAG_GONE), payload)
        self.assertEqual(codes(notices), [sn.CODE_PAG_DROPPED])
        self.assertIn("1/2장", notices[0].message)
        self.assertIn("OOM", notices[0].message)
        self.assertEqual(sn.result_notices(info_of(PAG_OK, PAG_OK), payload), [])

    def test_pag_missing_everywhere_with_hires_keeps_every_cause(self):
        # 모든 장에서 빠졌으면 하이레스여도 Anima 가 아닌 모델·잘못된 블록과 OOM 을 가를 수 없다
        # (앱은 모델과 무관하게 PAG 블록을 보낸다 — SDXL + 하이레스에서 매번 'VRAM 부족'이라 하면 틀린다).
        payload = pag_payload(guid_enabled=True)
        payload["enable_hr"] = True
        message = sn.result_notices(info_of(PAG_GONE), payload)[0].message
        self.assertIn("Anima 가 아니거나", message)
        self.assertIn("블록 번호", message)
        self.assertIn("VRAM 부족(OOM)", message)
        self.assertIn("하이레스", message)
        plain = sn.result_notices(info_of(PAG_GONE), pag_payload(guid_enabled=True))[0].message
        self.assertIn("Anima 가 아니거나", plain)
        self.assertNotIn("하이레스", plain)

    def test_pag_not_requested_means_no_check(self):
        self.assertEqual(sn.result_notices(info_of(PAG_GONE), pag_payload(guid_apg_enabled=True)), [])

    def test_old_pag_build_has_no_oom_fallback(self):
        old = SimpleNamespace(known=True, anima_guidance_argc=57)
        notice = sn.result_notices(info_of(PAG_OK, PAG_GONE), pag_payload(guid_enabled=True), capabilities=old)[0]
        self.assertNotIn("OOM", notice.message)
        self.assertIn("구버전", notice.message)


# ── 생성 전 알림 ──────────────────────────────────────────────────────────────
def _live_bodies():
    with open(os.path.join(FIXTURES, "sam_extra_live_scripts.json"), encoding="utf-8") as f:
        scripts = json.load(f)
    with open(os.path.join(FIXTURES, "sam_extra_script_info.json"), encoding="utf-8") as f:
        info = json.load(f)["scripts"]
    return scripts, info


def _capabilities(scripts, info):
    return build_capabilities({EP_SCRIPTS: HttpResult(200, scripts), EP_SCRIPT_INFO: HttpResult(200, info)})


class PreGenerationNoticeTests(_QuietAssumption):
    def test_cfg1_warns_about_every_active_base_on_both_v030_builds(self):
        """Forge 는 CFG 1 이면 네거티브를 인코딩하지 않는다(modules/processing.py:481-483 ``self.uc = None``).
        DCW-F 전 빌드(3522928)는 cond_scale≈1 이면 SMC/APG/CWM 전부를, DCW-F 빌드는 원본처럼
        disable_cfg1_optimization 을 걸어도 uncond 가 없어 SMC/CWM 까지 건너뛴다(확장 tests/test_dcw_origin.py
        test_forge_cfg_exactly_one_has_no_uncond_so_smc_cwm_step_aside). 원본 ComfyUI 노드(origin:
        namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:877-878 — 뜻만)는 CFG 1 에도 네거티브가 있어 SMC/CWM 이 돈다 —
        호스트 차이. 두 빌드는 버전·인자 수(0.30.0·62개)로 가를 수 없지만 답이 같다."""
        pre_dcwf = _capabilities(*_live_bodies())      # 픽스처 스냅샷(지금 91개 — v0.30 디테일 묶음 포함)
        self.assertEqual(pre_dcwf.anima_guidance_argc, 91)
        dcwf = SimpleNamespace(known=True, anima_guidance_argc=62, installed=True,
                               script=lambda _t: {"present": True, "img2img": True})
        for label, caps in (("unknown", None), ("pre-DCW-F", pre_dcwf), ("DCW-F", dcwf)):
            with self.subTest(build=label):
                notices = sn.pre_generation_notices(
                    pag_payload(cfg_scale=1.0, guid_smc_master_enabled=True, guid_apg_enabled=True,
                                guid_cwm_enabled=True, guid_cwm_alpha_low=0.3, guid_cwm_alpha_high=0.15),
                    capabilities=caps)
                self.assertEqual(codes(notices), [sn.CODE_CFG1_CFG_BASE])
                self.assertEqual(notices[0].level, sn.LEVEL_WARNING)
                self.assertEqual(notices[0].detail, "SMC/APG/CWM")
                self.assertIn("SMC/APG/CWM 가 효과가 없습니다", notices[0].message)
                self.assertIn("네거티브를 인코딩하지 않아", notices[0].message)
                self.assertIn("원본 ComfyUI", notices[0].message)
                self.assertNotIn("구버전", notices[0].message)
                for settings, detail in (({"guid_smc_master_enabled": True}, "SMC"),
                                         ({"guid_cwm_enabled": True, "guid_cwm_alpha_high": -0.2}, "CWM"),
                                         ({"guid_dcw_enabled": True, "guid_cfg_mode": "SMC + CWM",
                                           "guid_cwm_alpha_low": 0.3}, "SMC/CWM")):
                    notices = sn.pre_generation_notices(pag_payload(cfg_scale=1.0, **settings), capabilities=caps)
                    self.assertEqual([n.detail for n in notices], [detail], settings)
                apg = sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_apg_enabled=True), capabilities=caps)
                self.assertEqual([n.detail for n in apg], ["APG"])
                self.assertNotIn("원본 ComfyUI", apg[0].message)   # APG 는 원본에 없다
                # alpha 0 의 CWM 은 평범한 CFG 와 같아 알릴 것이 없다(원본 cwm_alpha_active)
                self.assertEqual(sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_cwm_enabled=True),
                                                           capabilities=caps), [])
                self.assertEqual(sn.pre_generation_notices(pag_payload(cfg_scale=4.5, guid_apg_enabled=True,
                                                                       guid_smc_master_enabled=True),
                                                           capabilities=caps), [])
                self.assertEqual(sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_enabled=True),
                                                           capabilities=caps), [])

    def test_cfg1_with_hires_cfg_is_informational(self):
        # 하이레스 CFG 가 1 이 아니면 Forge 가 하이레스 네거티브를 인코딩해(processing.py:1606-1608) 거기서는 적용된다
        payload = pag_payload(cfg_scale=1.0, guid_apg_enabled=True, guid_cwm_enabled=True)
        payload.update(enable_hr=True, hr_cfg=5.0)
        notice = sn.pre_generation_notices(payload)[0]
        self.assertEqual(notice.level, sn.LEVEL_INFO)
        self.assertIn("하이레스", notice.message)
        self.assertEqual(notice.detail, "APG@first")   # 억제 키가 CFG 1 인 패스로 갈린다
        cwm_only = pag_payload(cfg_scale=1.0, guid_cwm_enabled=True, guid_cwm_alpha_low=0.3)
        cwm_only.update(enable_hr=True, hr_cfg=5.0)
        notice = sn.pre_generation_notices(cwm_only)[0]
        self.assertEqual((notice.level, notice.detail), (sn.LEVEL_INFO, "CWM@first"))
        self.assertIn("CFG 5", notice.message)
        both_one = pag_payload(cfg_scale=1.0, guid_smc_master_enabled=True)
        both_one.update(enable_hr=True, hr_cfg=1.0)
        self.assertEqual(sn.pre_generation_notices(both_one)[0].level, sn.LEVEL_WARNING)

    def test_hires_only_cfg1_is_noticed(self):
        # 기본 CFG 가 1 이 아니어도 하이레스 CFG 가 1 이면 Forge 는 하이레스 네거티브를 인코딩하지 않고
        # (processing.py:1606-1608) 확장은 그 패스의 cond_scale(=p.hr_cfg, sd_samplers_cfg_denoiser.py:136-137)로
        # SMC/APG/CWM 을 건너뛴다(anima_safe_pag.py _pass_cfg_near_one · _cfg_base_skip_reason).
        payload = pag_payload(cfg_scale=7.0, guid_smc_master_enabled=True)
        payload.update(enable_hr=True, hr_cfg=1.0)
        notices = sn.pre_generation_notices(payload)
        self.assertEqual(codes(notices), [sn.CODE_CFG1_CFG_BASE])
        self.assertEqual((notices[0].level, notices[0].detail), (sn.LEVEL_INFO, "SMC@hires"))
        self.assertIn("하이레스 패스", notices[0].message)
        self.assertIn("첫 패스(CFG 7)", notices[0].message)
        self.assertIn("원본 ComfyUI", notices[0].message)
        self.assertNotIn("기본값", notices[0].message)
        # hr_cfg 를 안 보내면 Forge API 기본값 1.0 (processing.py:1226 ``hr_cfg: float = 1.0``,
        # api/models.py 가 __init__ 기본값으로 모델을 만든다) — 앱은 하이레스 CFG 가 0(끔)이면 hr_cfg 를 안 보낸다
        absent = pag_payload(cfg_scale=7.0, guid_apg_enabled=True)
        absent.update(enable_hr=True)
        notice = sn.pre_generation_notices(absent)[0]
        self.assertEqual((notice.level, notice.detail), (sn.LEVEL_INFO, "APG@hires"))
        self.assertIn("기본값 1", notice.message)
        self.assertNotIn("원본 ComfyUI", notice.message)   # APG 는 원본에 없다
        # 하이레스 CFG 가 1 이 아니거나 하이레스를 안 켜면 알릴 것이 없다
        for extra in ({"enable_hr": True, "hr_cfg": 5.0}, {}, {"enable_hr": False}, {"hr_cfg": 1.0}):
            with self.subTest(extra=extra):
                quiet = pag_payload(cfg_scale=7.0, guid_smc_master_enabled=True)
                quiet.update(extra)
                self.assertEqual(sn.pre_generation_notices(quiet), [])
        # 가드 없는 v0.21.2 계열은 하이레스 패스에서 빈 uncond 에 적용돼 망가진다
        old = SimpleNamespace(known=True, anima_guidance_argc=57, installed=True, script=lambda _t: {})
        notice = sn.pre_generation_notices(payload, capabilities=old)[0]
        self.assertEqual(notice.level, sn.LEVEL_WARNING)
        self.assertIn("구버전", notice.message)
        self.assertIn("하이레스 패스", notice.message)

    def test_cfg1_on_old_build_warns_about_breakage(self):
        # v0.21.2 계열에는 CFG 1 가드가 없다 — SMC/CWM 도 빈 uncond 에 적용되니 켠 base 전부를 알린다
        old = SimpleNamespace(known=True, anima_guidance_argc=57, installed=True, script=lambda _t: {})
        notice = sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_apg_enabled=True), capabilities=old)[0]
        self.assertIn("구버전", notice.message)
        notice = sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_cwm_enabled=True, guid_cwm_alpha_low=0.3),
                                           capabilities=old)[0]
        self.assertIn("구버전", notice.message)
        # 57개 빌드는 프리셋 'Auto' 로 SMC 도 켠다(나-5) — CWM 과 함께 알린다
        self.assertEqual(notice.detail, "SMC/CWM")

    def test_unknown_capabilities_assume_current_version_and_log(self):
        with mock.patch.object(sn, "_ASSUMED_LOGGED", False), \
                self.assertLogs("core.sam_extra_notices", level="WARNING") as logs:
            notice = sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_apg_enabled=True))[0]
        self.assertIn("v0.30", "\n".join(logs.output))
        self.assertIn("효과가 없습니다", notice.message)

    def test_bare_sam3_checkpoint_warns_about_hf_download(self):
        notices = sn.pre_generation_notices(sam3_payload(sam3_checkpoint="sam3.pt"))
        self.assertEqual(codes(notices), [sn.CODE_SAM3_HF_DOWNLOAD])
        self.assertIn("3.4 GB", notices[0].message)
        self.assertIn("facebook/sam3", notices[0].message)
        remote = sn.pre_generation_notices(sam3_payload(sam3_checkpoint="sam3.pt"), remote=True)[0]
        self.assertIn("원격 Forge", remote.message)
        for value in ("C:\\forge\\models\\sam3\\sam3.pt", "/data/models/sam3/sam3.pt", "sam3_v2.pt", "auto",
                      "huggingface", "sam3/sam3.pt"):
            with self.subTest(value=value):
                self.assertEqual(sn.pre_generation_notices(sam3_payload(sam3_checkpoint=value)), [])

    def test_missing_script_uses_known_snapshot_only(self):
        scripts, info = _live_bodies()
        payload = sam3_payload(sam3_checkpoint="C:\\m\\sam3.pt")
        payload["alwayson_scripts"].update(anima_guidance.build_alwayson({"guid_enabled": True}))
        self.assertEqual(sn.pre_generation_notices(payload, capabilities=_capabilities(scripts, info)), [])

        scripts = copy.deepcopy(scripts)
        scripts["txt2img"].remove("sam3 mask")
        notices = sn.pre_generation_notices(payload, capabilities=_capabilities(scripts, info))
        self.assertEqual(codes(notices), [sn.CODE_SCRIPT_MISSING])
        self.assertEqual(notices[0].level, sn.LEVEL_ERROR)
        self.assertIn("'SAM3 Mask'", notices[0].message)
        self.assertIn("422", notices[0].message)
        # img2img 요청은 img2img 목록을 본다 (거기엔 있다)
        payload["init_images"] = ["x"]
        self.assertEqual(sn.pre_generation_notices(payload, capabilities=_capabilities(scripts, info)), [])

    def test_extension_missing_entirely(self):
        caps = build_capabilities({EP_SCRIPTS: HttpResult(200, {"txt2img": ["controlnet"], "img2img": []}),
                                   EP_SCRIPT_INFO: HttpResult(200, [])})
        notices = sn.pre_generation_notices(sam3_payload(sam3_checkpoint="C:\\m\\sam3.pt"), capabilities=caps)
        self.assertEqual(codes(notices), [sn.CODE_EXTENSION_MISSING])
        self.assertIn("SAM3 Mask", notices[0].message)
        # 스냅샷이 없으면(확인 전) 아무 말도 하지 않는다
        self.assertEqual(sn.pre_generation_notices(sam3_payload(sam3_checkpoint="C:\\m\\sam3.pt")), [])


# ── HTTP 422 ──────────────────────────────────────────────────────────────────
class RejectedRequestTests(unittest.TestCase):
    def _forge_body(self, detail):
        # modules/api/api.py handle_exception 의 모양
        return {"error": "HTTPException", "detail": detail, "body": "", "errors": f"422: {detail}"}

    def test_missing_sam_extra_script(self):
        text = sn.explain_rejected_request(422, self._forge_body("Script 'SAM3 Mask' not found"), sam3_payload())
        self.assertIn("'SAM3 Mask'", text)
        self.assertIn("sam-extra", text)
        self.assertIn("켠 기능: SAM3 Mask", text)

    def test_missing_other_extension_and_old_message(self):
        text = sn.explain_rejected_request(422, json.dumps(self._forge_body("Script 'ADetailer' not found")))
        self.assertIn("ADetailer 확장", text)
        text = sn.explain_rejected_request(422, {"detail": "always on script Anima Perturbation Guidance not found"})
        self.assertIn("Anima 가이던스", text)

    def test_validation_detail_list(self):
        body = {"detail": [{"loc": ["body", "steps"], "msg": "value is not a valid integer",
                            "type": "type_error.integer"}]}
        self.assertIn("steps: value is not a valid integer", sn.explain_rejected_request(422, body))

    def test_other_status_is_left_alone(self):
        self.assertIsNone(sn.explain_rejected_request(500, {"detail": "x"}))
        self.assertIsNone(sn.explain_rejected_request(None, {}))


# ── 단독 경로 / 억제 ──────────────────────────────────────────────────────────
class StandaloneAndThrottleTests(unittest.TestCase):
    def test_noticed_image_is_a_plain_base64_string(self):
        value = sn.NoticedImage("QUJD", {"a": 1}, sn.result_notices(info_of(SAM3_OOM), sam3_payload()))
        self.assertEqual(value, "QUJD")
        self.assertEqual(base64.b64decode(value), b"ABC")
        self.assertEqual(sn.standalone_sam3_failure(value).code, sn.CODE_SAM3_ERROR)
        self.assertIsNone(sn.standalone_sam3_failure("QUJD"))
        text = sn.standalone_failure_text(sn.standalone_sam3_failure(value))
        self.assertIn("저장하지 않았습니다", text)
        self.assertIn("VRAM", text)
        # 앞 단계(업스케일·ADetailer)가 적용된 뒤의 SAM3 실패 — '원본과 같다'고 하지 않는다
        kept = sn.standalone_failure_text(sn.standalone_sam3_failure(value), saved_without_sam3=True)
        self.assertIn("SAM3 없이 앞 단계 결과만 저장", kept)
        self.assertNotIn("원본과 같아", kept)
        not_applied = sn.Notice(sn.CODE_SAM3_NOT_APPLIED, sn.LEVEL_WARNING, "m", feature="sam3")
        self.assertNotIn("원본과 같아", sn.standalone_failure_text(not_applied, saved_without_sam3=True))

    def test_notice_dict_roundtrip(self):
        notice = sn.result_notices(info_of(SAM3_CN_KEYERROR), sam3_payload(sam3_cn_module="none"))[0]
        self.assertEqual(sn.Notice.from_dict(json.loads(json.dumps(notice.to_dict()))), notice)
        info = {sn.INFO_KEY: [notice.to_dict(), {"message": ""}, "junk"]}
        self.assertEqual(sn.notices_from_info(info), [notice])

    def test_throttle(self):
        now = [100.0]
        throttle = sn.NoticeThrottle(lambda: now[0])
        notice = sn.Notice("x", sn.LEVEL_WARNING, "m")
        self.assertTrue(throttle.allow(notice, 30))
        self.assertFalse(throttle.allow(notice, 30))
        now[0] += 31
        self.assertTrue(throttle.allow(notice, 30))

    def test_loopback(self):
        self.assertTrue(sn.is_loopback_url("http://127.0.0.1:7860"))
        self.assertTrue(sn.is_loopback_url("http://localhost:7860/"))
        self.assertFalse(sn.is_loopback_url("http://192.168.0.5:7860"))


# ── 백엔드 연결 ───────────────────────────────────────────────────────────────
def _png_b64(size=(8, 8)):
    out = io.BytesIO()
    Image.new("RGB", size, "gray").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.closed = False

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        self.closed = True


class BackendWiringTests(unittest.TestCase):
    def setUp(self):
        from backends.webui_backend import WebUIBackend
        self.backend = WebUIBackend("http://127.0.0.1:7860")
        patcher = mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _generate(self, response, payload):
        with mock.patch.object(self.backend, "_switch_model_if_needed"), \
                mock.patch("backends.webui_backend.requests.post", return_value=response):
            return self.backend.txt2img("anima", payload)

    def test_generation_info_carries_notices(self):
        response = _Response({"images": [_png_b64()], "info": info_of(SAM3_CN_KEYERROR)})
        result = self._generate(response, sam3_payload(sam3_cn_module="none"))
        self.assertTrue(result.success)
        self.assertEqual([n["code"] for n in result.info[sn.INFO_KEY]], [sn.CODE_SAM3_ERROR])

    def test_clean_generation_adds_nothing(self):
        result = self._generate(_Response({"images": [_png_b64()], "info": info_of(SAM3_OK)}), sam3_payload())
        self.assertNotIn(sn.INFO_KEY, result.info)

    def test_422_is_explained(self):
        response = _Response({"error": "HTTPException", "detail": "Script 'SAM3 Mask' not found"}, 422)
        with self.assertLogs("backends.webui_backend", level="WARNING"):
            result = self._generate(response, sam3_payload())
        self.assertFalse(result.success)
        self.assertIn("sam-extra", result.error)
        self.assertTrue(response.closed)

    def test_standalone_sam3_returns_noticed_image(self):
        image = _png_b64((16, 16))
        response = _Response({"images": [image], "info": info_of(SAM3_OOM)})
        with mock.patch("backends.webui_backend.requests.post", return_value=response):
            result = self.backend.sam3(image, {"sam3_prompt": "face"})
        self.assertEqual(result, image)
        self.assertEqual(sn.standalone_sam3_failure(result).code, sn.CODE_SAM3_ERROR)

    def test_standalone_422_raises_explained_error(self):
        image = _png_b64((16, 16))
        response = _Response({"detail": "Script 'SAM3 Mask' not found"}, 422)
        with mock.patch("backends.webui_backend.requests.post", return_value=response), \
                self.assertLogs("backends.webui_backend", level="WARNING"):
            with self.assertRaises(RuntimeError) as ctx:
                self.backend.refine(image, {"target": "face"})
        self.assertIn("sam-extra", str(ctx.exception))


# ── 워커 ──────────────────────────────────────────────────────────────────────
class _FailingSam3Backend:
    """Forge 가 'SAM3 Error' 를 남긴 결과(입력 그대로)를 돌려주는 가짜 백엔드."""

    def __init__(self, infotext):
        self.infotext = infotext

    def _result(self, image_b64):
        return sn.NoticedImage(image_b64, {}, sn.result_notices(info_of(self.infotext), sam3_payload()))

    def sam3(self, image_b64, _settings):
        return self._result(image_b64)

    def refine(self, image_b64, _settings):
        return self._result(image_b64)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = os.path.join(temp.name, "out")
        self.paths = []
        for name in ("a.png", "b.png"):
            path = os.path.join(temp.name, name)
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(_png_b64()))
            self.paths.append(path)
        # 워커 모듈은 release_before_backend_job 을 import 때 묶는다 — 그 이름을 패치하면 처음 import 한
        # 테스트의 가짜가 모듈에 남는다. 그래서 그 아래(편집기 번들 반납)만 막는다(test_backend_job_vram_release 와 같다).
        from core import model_cache
        patcher = mock.patch.object(model_cache, "release_for_generation", return_value=0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _with_backend(self, backend):
        patcher = mock.patch("backends.get_backend", return_value=backend)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_single_sam3_failure_is_an_error_and_nothing_is_saved(self):
        from workers.sam3_worker import Sam3SingleWorker
        self._with_backend(_FailingSam3Backend(SAM3_CN_KEYERROR))
        results = []
        worker = Sam3SingleWorker(self.paths[0], {"output_folder": self.out})
        worker.finished.connect(results.append)
        worker.run()
        data = json.loads(results[0])
        self.assertIn("SAM3 실패", data["error"])
        self.assertFalse(os.path.isdir(self.out) and os.listdir(self.out))

    def test_batch_continues_after_a_failed_item(self):
        from workers.sam3_worker import Sam3BatchWorker
        self._with_backend(_FailingSam3Backend(SAM3_OOM))
        results, progress = [], []
        worker = Sam3BatchWorker(self.paths, {"output_folder": self.out})
        worker.single_done.connect(results.append)
        worker.progress.connect(lambda cur, tot: progress.append(cur))
        worker.run()
        self.assertEqual([json.loads(r)["index"] for r in results], [0, 1])
        self.assertTrue(all("VRAM" in json.loads(r)["error"] for r in results))
        self.assertTrue(not any(json.loads(r).get("batch_stopped") for r in results))   # OOM 은 이미지마다 다르다
        self.assertEqual(progress, [1, 2])
        kind, message = worker.completion_notice()
        self.assertEqual(kind, "error")
        self.assertIn("2개 모두 실패", message)

    def test_batch_stops_when_every_image_would_fail_the_same_way(self):
        from workers.sam3_worker import Sam3BatchWorker
        backend = _FailingSam3Backend(SAM3_VALIDATION)
        backend.calls = 0
        original = backend.sam3

        def counting(image_b64, settings):
            backend.calls += 1
            return original(image_b64, settings)

        backend.sam3 = counting
        self._with_backend(backend)
        results, progress = [], []
        worker = Sam3BatchWorker(self.paths + [self.paths[0]], {"output_folder": self.out})
        worker.single_done.connect(results.append)
        worker.progress.connect(lambda cur, tot: progress.append((cur, tot)))
        worker.run()
        self.assertEqual(backend.calls, 1, "설정 검증 실패 뒤에는 남은 이미지를 Forge 에 보내지 않는다")
        data = json.loads(results[0])
        self.assertEqual((data["index"], data["batch_stopped"], data["skipped"]), (0, True, 2))
        self.assertIn("검증", data["error"])
        self.assertEqual(progress, [(1, 3)])
        kind, message = worker.completion_notice()
        self.assertEqual(kind, "error")
        self.assertIn("건너뜀 2", message)
        self.assertFalse(os.path.isdir(self.out) and os.listdir(self.out))

    def test_batch_completion_counts_real_successes(self):
        from workers.sam3_worker import Sam3BatchWorker

        class _Mixed:
            def __init__(self):
                self.n = 0

            def sam3(self, image_b64, _settings):
                self.n += 1
                infotext = SAM3_OK if self.n == 1 else SAM3_OOM
                return sn.NoticedImage(image_b64, {}, sn.result_notices(info_of(infotext), sam3_payload()))

        self._with_backend(_Mixed())
        worker = Sam3BatchWorker(self.paths, {"output_folder": self.out})
        worker.run()
        kind, message = worker.completion_notice()
        self.assertEqual(kind, "warning")
        self.assertIn("성공 1", message)
        self.assertIn("실패 1", message)
        self.assertEqual(len(os.listdir(self.out)), 1)

    def _run_upscale(self, mode, backend):
        from workers import upscale_worker
        results = []
        worker = upscale_worker.BatchUpscaleWorker([self.paths[0]], {"mode": mode, "output_folder": self.out})
        worker.single_finished.connect(lambda i, ok, msg: results.append((i, ok, msg)))
        with mock.patch.object(upscale_worker, "get_backend", return_value=backend):
            worker.run()
        return results

    def test_upscale_both_keeps_the_upscale_when_only_sam3_fails(self):
        upscaled = _png_b64((16, 16))
        backend = _FailingSam3Backend(SAM3_CN_KEYERROR)
        backend.upscale = lambda _image, _settings: upscaled
        backend.adetailer = lambda image, _settings: image
        results = self._run_upscale("both", backend)
        self.assertEqual(len(results), 1)
        index, ok, message = results[0]
        self.assertEqual((index, ok), (0, True))
        self.assertTrue(message.startswith("a_upscaled_ad.png"), message)   # '_sam3' 는 붙지 않는다
        self.assertIn("SAM3 실패", message)
        self.assertIn("앞 단계 결과만 저장", message)
        self.assertNotIn("원본과 같아", message)
        with open(os.path.join(self.out, "a_upscaled_ad.png"), "rb") as fh:
            self.assertEqual(fh.read(), base64.b64decode(upscaled))   # SAM3 전(업스케일·AD) 결과

    def test_upscale_sam3_only_failure_saves_nothing(self):
        results = self._run_upscale("sam3_only", _FailingSam3Backend(SAM3_OOM))
        self.assertEqual(results[0][:2], (0, False))
        self.assertIn("저장하지 않았습니다", results[0][2])
        self.assertIn("VRAM", results[0][2])
        self.assertFalse(os.listdir(self.out))

    def test_refine_failure(self):
        from workers.refine_worker import RefineWorker
        self._with_backend(_FailingSam3Backend(NO_EXTENSION_TRACE))
        results = []
        worker = RefineWorker(self.paths[0], {"output_folder": self.out, "main_prompt": "1girl"})
        worker.finished.connect(results.append)
        worker.run()
        self.assertIn("SAM3 Enable", json.loads(results[0])["error"])

    def test_success_passes_other_notices_to_the_gui(self):
        from workers.sam3_worker import Sam3SingleWorker

        class _Backend:
            def sam3(self, image_b64, _settings):
                notices = sn.result_notices(info_of(ANIMA38_OFF_QWEN))
                return sn.NoticedImage(image_b64, {}, notices)

        self._with_backend(_Backend())
        results = []
        worker = Sam3SingleWorker(self.paths[0], {"output_folder": self.out})
        worker.finished.connect(results.append)
        worker.run()
        data = json.loads(results[0])
        self.assertNotIn("error", data)
        self.assertEqual([n["code"] for n in data["notices"]], [sn.CODE_ANIMA38_OFF])


# ── UI (토스트) ───────────────────────────────────────────────────────────────
class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


def _host(**attrs):
    bridge = SimpleNamespace(showNotification=_Signal(), sam3Result=_Signal(), refineResult=_Signal())
    return SimpleNamespace(vue_bridge=bridge, **attrs)


class UiTests(unittest.TestCase):
    def test_result_notices_become_toasts_once(self):
        from ui.sam_extra_notices_ui import show_result_notices
        host = _host()
        notices = sn.result_notices(info_of(SAM3_CN_KEYERROR), sam3_payload(sam3_cn_module="none"))
        info = {sn.INFO_KEY: sn.notices_to_dicts(notices)}
        with self.assertLogs("ui.sam_extra_notices_ui", level="WARNING"):
            self.assertEqual(show_result_notices(host, info), 1)
            self.assertEqual(show_result_notices(host, info), 0)     # 같은 실패는 잠시 한 번만
        level, message = host.vue_bridge.showNotification.calls[0]
        self.assertEqual(level, "warning")
        self.assertIn("sam3_cn_module", message)
        self.assertEqual(show_result_notices(host, {}), 0)

    def test_sparse_lora_info_is_shown_rarely(self):
        from ui.sam_extra_notices_ui import show_result_notices
        now = [1000.0]
        host = _host(_sam_extra_notice_throttle=sn.NoticeThrottle(lambda: now[0]))
        info = {sn.INFO_KEY: sn.notices_to_dicts(sn.result_notices(info_of(SPARSE_GUESS)))}
        with self.assertLogs("ui.sam_extra_notices_ui", level="INFO") as logs:
            self.assertEqual(show_result_notices(host, info), 1)
            now[0] += sn.RESULT_NOTICE_TTL_S + 1
            self.assertEqual(show_result_notices(host, info), 0)   # 같은 LoRA 묶음 — 생성마다 띄우지 않는다
            now[0] += sn.PRE_GENERATION_NOTICE_TTL_S
            self.assertEqual(show_result_notices(host, info), 1)
        self.assertTrue(all(record.levelname == "INFO" for record in logs.records))
        self.assertEqual(host.vue_bridge.showNotification.calls[0][0], "info")

    def test_relay_shows_notices_and_forwards_json(self):
        from ui.sam_extra_notices_ui import relay_worker_result
        host = _host()
        text = json.dumps({"after": "x.png", "notices": sn.notices_to_dicts(sn.result_notices(info_of(ANIMA38_OFF_QWEN)))})
        with self.assertLogs("ui.sam_extra_notices_ui", level="WARNING"):
            relay_worker_result(host, "sam3Result", text)
        self.assertEqual(host.vue_bridge.sam3Result.calls, [(text,)])
        self.assertEqual(len(host.vue_bridge.showNotification.calls), 1)
        relay_worker_result(host, "refineResult", "not json")
        self.assertEqual(host.vue_bridge.refineResult.calls, [("not json",)])

    def test_before_generation_only_on_webui(self):
        from backends import BackendType
        from ui import sam_extra_notices_ui as ui_mod
        payload = pag_payload(cfg_scale=1.0, guid_apg_enabled=True)
        backend = SimpleNamespace(api_url="http://127.0.0.1:7860")
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI):
            self.assertEqual(ui_mod.check_before_generation(_host(), payload), 0)
        host = _host(sam_extra_capabilities=None)
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None), \
                self.assertLogs("ui.sam_extra_notices_ui", level="WARNING"):
            self.assertEqual(ui_mod.check_before_generation(host, payload), 1)
            self.assertEqual(ui_mod.check_before_generation(host, payload), 0)   # 10분에 한 번
        self.assertIn("CFG 1", host.vue_bridge.showNotification.calls[0][1])

    def test_pass_specific_cfg1_notice_does_not_hide_the_other_pass(self):
        """CFG 1 알림은 어느 패스 얘기인지로 억제 키가 갈린다. 앱 기본 하이레스(하이레스 CFG 0 → hr_cfg 안 보냄 →
        Forge 1.0)면 '하이레스 패스만 CFG 1' INFO 가 하이레스 생성마다 뜨는데, 키가 첫 패스 CFG 1 WARNING 과 같으면
        뒤이은 'SMC 를 전부 건너뜁니다' WARNING 이 10분(PRE_GENERATION_NOTICE_TTL_S) 동안 가려진다."""
        from backends import BackendType
        from ui import sam_extra_notices_ui as ui_mod
        backend = SimpleNamespace(api_url="http://127.0.0.1:7860")
        old = SimpleNamespace(known=True, anima_guidance_argc=57, installed=True, script=lambda _t: {})
        hires_default = {"enable_hr": True}                  # hr_cfg 없음 → Forge 기본 1.0
        first_only = {"enable_hr": True, "hr_cfg": 5.0}      # 첫 패스만 CFG 1
        scenarios = (
            # (이름, 스냅샷, 먼저 뜬 알림의 (cfg, 추가 필드), 1분 뒤 (cfg, 추가 필드), 그 뒤 수준)
            ("hires INFO → base WARNING", None, (7.0, hires_default), (1.0, hires_default), "warning"),
            ("hires INFO → no-hires WARNING", None, (7.0, hires_default), (1.0, {}), "warning"),
            ("first-pass INFO → base WARNING", None, (1.0, first_only), (1.0, {}), "warning"),
            ("base WARNING → hires INFO", None, (1.0, {}), (7.0, hires_default), "info"),
            ("old build hires → old build base", old, (7.0, hires_default), (1.0, {}), "warning"),
        )
        for name, caps, (cfg1, extra1), (cfg2, extra2), level2 in scenarios:
            with self.subTest(name), mock.patch.object(sn, "_ASSUMED_LOGGED", True), \
                    mock.patch("backends.get_backend", return_value=backend), \
                    mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                    mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None), \
                    self.assertLogs("ui.sam_extra_notices_ui", level="INFO"):
                now = [1000.0]
                host = _host(sam_extra_capabilities=caps,
                             _sam_extra_notice_throttle=sn.NoticeThrottle(lambda: now[0]))

                def send(cfg, extra):
                    payload = pag_payload(cfg_scale=cfg, guid_smc_master_enabled=True)
                    payload.update(extra)
                    return ui_mod.check_before_generation(host, payload)

                self.assertEqual(send(cfg1, extra1), 1)
                self.assertEqual(send(cfg1, extra1), 0)       # 같은 상황의 반복은 여전히 10분에 한 번
                now[0] += 60
                self.assertEqual(send(cfg2, extra2), 1)       # 다른 패스의 알림은 가려지지 않는다
                self.assertEqual(host.vue_bridge.showNotification.calls[-1][0], level2)
                self.assertEqual(send(cfg2, extra2), 0)
        # 하이레스 기본값으로 첫 패스도 CFG 1 인 것과 하이레스를 안 켠 CFG 1 은 같은 경고다 — 한 번만
        self.assertEqual(sn.pre_generation_notices(pag_payload(cfg_scale=1.0, guid_smc_master_enabled=True))[0].key,
                         sn.pre_generation_notices({**pag_payload(cfg_scale=1.0, guid_smc_master_enabled=True),
                                                    **hires_default})[0].key)

    def test_standalone_checkpoint_check_resolves_locally_first(self):
        from backends import BackendType
        from ui import sam_extra_notices_ui as ui_mod
        backend = SimpleNamespace(api_url="http://127.0.0.1:7860")
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda v: v):
            host = _host()
            with self.assertLogs("ui.sam_extra_notices_ui", level="WARNING"):
                self.assertEqual(ui_mod.check_standalone_sam3(host, {}), 1)
            self.assertIn("3.4 GB", host.vue_bridge.showNotification.calls[0][1])
        with mock.patch("backends.get_backend", return_value=backend), \
                mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("core.forge_modules.resolve_sam3_checkpoint",
                           return_value="C:\\forge\\models\\sam3\\sam3.pt"):
            self.assertEqual(ui_mod.check_standalone_sam3(_host(), {"sam3_checkpoint": "sam3.pt"}), 0)


# ── 설치된 확장 소스와 키가 같은가 ────────────────────────────────────────────
def _assigned_string_keys(path):
    """``x[...] = …`` 의 문자열 첨자와 ``x.update({...})`` 의 문자열 키."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) \
                        and isinstance(target.slice.value, str):
                    keys.add(target.slice.value)
    return keys


@requires_extension
class ExtensionSourceTests(unittest.TestCase):
    """알림이 기대는 infotext 키·상수·가드가 설치된 확장에 그대로 있는가 (이름이 바뀌면 알림이 조용히 멈춘다)."""

    def test_sam3_infotext_keys(self):
        keys = _assigned_string_keys(os.path.join(EXT_ROOT, "scripts", "!sam3.py"))
        for key in (sn.KEY_SAM3_ERROR, sn.KEY_SAM3_ENABLE, sn.KEY_SAM3_VERSION):
            self.assertIn(key, keys)

    def test_pag_infotext_key_and_guards(self):
        path = os.path.join(EXT_ROOT, "scripts", "anima_safe_pag.py")
        self.assertIn(sn.KEY_PAG, _assigned_string_keys(path))
        with open(path, encoding="utf-8") as fh:
            names = {node.name for node in ast.walk(ast.parse(fh.read())) if isinstance(node, ast.FunctionDef)}
        # v0.30 의미: CFG 1 가드(Forge 는 CFG 1 에 uncond 가 없어 어느 빌드든 켠 base 를 건너뜀)와 확장 배치 OOM 폴백
        self.assertIn("_cfg_base_skip_reason", names)
        self.assertIn("_disable_perturbation_after_oom", names)

    def test_face_crop_infotext_keys_and_detector_messages(self):
        """얼굴 안에서만 찾기 — 인자 infotext 'SAM3 Face Crop…'(sam3ext/args.py ALL_ARGS)·결과 기록 상수·검출기 오류 문구가
        알림이 기대는 그대로인가. FaceDetectorUnavailable 문구는 모두 얼굴 검출기 힌트로 간다(SAM3 패키지·체크포인트로 읽지 않음)."""
        with open(os.path.join(EXT_ROOT, "sam3ext", "args.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        arg_names = {node.args[1].value for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Arg" and len(node.args) == 2
                     and isinstance(node.args[1], ast.Constant)}
        for key in (sn.KEY_SAM3_FACE_CROP, sn.KEY_SAM3_FACE_CROP_MODEL, sn.KEY_SAM3_FACE_CROP_CONFIDENCE):
            self.assertIn(key, arg_names)
        from core.sam_extra_scan import ExtensionSource
        src = ExtensionSource(EXT_ROOT)
        self.assertEqual(src.module_constant("scripts/!sam3.py", "INFOTEXT_FACE_CROP_FACES"), sn.KEY_SAM3_FACE_CROP_FACES)
        with open(os.path.join(EXT_ROOT, "sam3ext", "face_detect.py"), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        self.assertIn(sn.FACE_DETECTOR_ERROR, {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)})
        messages = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.JoinedStr, ast.Constant)) and isinstance(getattr(node, "value", None), str) \
                    and str(node.value).startswith("face crop:"):
                messages.append(node.value)
            elif isinstance(node, ast.JoinedStr):
                text = "".join(str(v.value) if isinstance(v, ast.Constant) else "x.pt" for v in node.values)
                if text.startswith("face crop:"):
                    messages.append(text)
        self.assertGreaterEqual(len(messages), 3, messages)
        generic = sn.sam3_error_hint("RuntimeError: ???")
        for message in messages:
            with self.subTest(message=message[:50]):
                hint = sn.sam3_error_hint(f"{sn.FACE_DETECTOR_ERROR}: {message}")
                self.assertNotEqual(hint, generic)
                self.assertIn("얼굴" if "ultralytics" not in message or "rename" in message else "ultralytics", hint)

    def test_constants(self):
        from core.sam_extra_scan import ExtensionSource
        src = ExtensionSource(EXT_ROOT)
        self.assertEqual(src.module_constant("scripts/anima_3_8b.py", "STATUS_KEY"), sn.KEY_ANIMA38_STATUS)
        self.assertEqual(src.module_constant("sam3ext/core.py", "HF_CHECKPOINT_NAME"), sn.HF_CHECKPOINT_NAME)
        self.assertEqual(src.module_constant("sam3ext/core.py", "HF_CHECKPOINT_REPO"), sn.HF_CHECKPOINT_REPO)
        self.assertEqual(src.module_constant("sam3ext/anima_lora_blocks.py", "INFOTEXT_SPARSE_GUESS_KEY"),
                         sn.KEY_SPARSE_LORA_GUESS)

    def test_anima38_file_errors_map_to_their_own_hint(self):
        """확장이 'Anima38: off: <FileNotFoundError 문구>' 로 남기는 문구마다 맞는 힌트로 가는가."""
        messages = []
        for rel in ("sam3ext/anima38/files.py", "sam3ext/anima38/runtime.py"):
            with open(os.path.join(EXT_ROOT, *rel.split("/")), encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) \
                        and getattr(node.exc.func, "id", "") == "FileNotFoundError" and node.exc.args:
                    arg = node.exc.args[0]
                    if isinstance(arg, ast.Constant):
                        messages.append(str(arg.value))
                    elif isinstance(arg, ast.JoinedStr):   # f"Adapter '{name}' …" — 이름 자리에 qwen35 를 넣어 본다
                        messages.append("".join(str(v.value) if isinstance(v, ast.Constant) else "qwen35_adapter"
                                                for v in arg.values))
        self.assertIn(EXT_TOKENIZER_MISSING, messages)
        self.assertIn(EXT_QWEN35_MISSING, messages)
        generic = sn.anima38_off_hint("off: ???")
        for message in messages:
            hint = sn.anima38_off_hint("off: " + message)
            self.assertNotEqual(hint, generic, message)
            lower = message.lower()
            if "tokenizer" in lower:
                self.assertIn("다시 설치", hint, message)
            elif "adapter" in lower:
                self.assertIn("어댑터", hint, message)
            else:
                self.assertIn("qwen35_4b", hint, message)


# ── 얼굴 안에서만 찾기 (sam-extra 0.34.0) ─────────────────────────────────────────
_FACE_ON = ("SAM3 Face Crop: True, SAM3 Face Crop Model: face_yolov8n.pt, SAM3 Face Crop Confidence: 0.3, ")
# 확장 sam3ext/face_detect.py 의 FaceDetectorUnavailable 문구 그대로(ExtensionSourceTests 가 설치된 확장과 다시 본다)
EXT_FACE_MODEL_MISSING = ("face crop: face detector model 'face_yolov8n.pt' not found — put it in "
                          "C:\\sd\\models\\adetailer (the same folder ADetailer uses; it is never downloaded automatically).")
EXT_FACE_NO_ULTRALYTICS = ("face crop: the 'ultralytics' package is not installed in the Forge venv — install ADetailer "
                           "(it brings ultralytics) or `pip install ultralytics` in the Forge venv.")


def face_infotext(faces=None, *, on=True):
    keys = _FACE_ON if on else ""
    if faces is not None:
        keys += f"SAM3 Face Crop Faces: {faces}, "
    return _HEAD + "SAM3 Enable: True, " + _SAM3_PARAMS + keys + "SAM3 Version: 0.34.0, " + _TAIL


def face_error(message):
    return _HEAD + _SAM3_PARAMS + _FACE_ON + "SAM3 Version: 0.34.0, SAM3 Error: " + json.dumps(
        "FaceDetectorUnavailable: " + message) + ", " + _TAIL


class FaceCropNoticeTests(_QuietAssumption):
    def test_request_reads_the_face_crop_switch(self):
        self.assertTrue(sn.requested_features(sam3_payload(sam3_face_crop=True)).sam3_face_crop)
        self.assertFalse(sn.requested_features(sam3_payload(sam3_face_crop=False)).sam3_face_crop)
        # 앱 build_state 는 끈 요청에 키를 싣지 않는다
        payload = {"alwayson_scripts": sam3_args.build_alwayson({})}
        self.assertFalse(sn.requested_features(payload).sam3_face_crop)

    def test_faces_found_or_face_crop_off_is_quiet(self):
        on = sam3_payload(sam3_face_crop=True)
        self.assertEqual(sn.result_notices(info_of(face_infotext(2)), on), [])
        self.assertEqual(sn.result_notices(info_of(face_infotext(on=False)), sam3_payload()), [])

    def test_old_extension_without_face_crop_record_warns(self):
        notices = sn.result_notices(info_of(SAM3_OK), sam3_payload(sam3_face_crop=True))
        self.assertEqual(codes(notices), [sn.CODE_SAM3_FACE_CROP_UNSUPPORTED])
        self.assertIn("0.34.0", notices[0].message)
        self.assertIn("전체 이미지", notices[0].message)

    def test_no_faces_warns_with_model_confidence_and_scope(self):
        payload = sam3_payload(sam3_face_crop=True, sam3_face_crop_confidence=0.5)
        notices = sn.result_notices(info_of(face_infotext(0)), payload)
        self.assertEqual(codes(notices), [sn.CODE_SAM3_FACE_CROP_NO_FACES])
        self.assertIn("face_yolov8n.pt", notices[0].message)
        self.assertIn("0.3", notices[0].message)          # 그 장의 기록(0.3)이 보낸 값보다 먼저
        batch = sn.result_notices(info_of(face_infotext(0), face_infotext(3), face_infotext(0)), payload)
        self.assertIn("2/3장", batch[0].message)
        self.assertNotIn(sn.CODE_SAM3_FACE_CROP_NO_FACES, sn.SAM3_FAILURE_CODES, "얼굴 없음은 '검출 없음' 처리 — 실패가 아니다")

    def test_face_detector_errors_get_their_own_hint_and_stop_a_batch(self):
        cases = ((EXT_FACE_MODEL_MISSING, "models/adetailer"), (EXT_FACE_NO_ULTRALYTICS, "ultralytics"))
        for message, needle in cases:
            with self.subTest(message=message[:40]):
                notices = sn.result_notices(info_of(face_error(message)), sam3_payload(sam3_face_crop=True))
                self.assertEqual(codes(notices), [sn.CODE_SAM3_ERROR])
                self.assertIn(needle, notices[0].hint)
                self.assertTrue(sn.sam3_failure_repeats(notices[0]), "설치 문제 — 같은 설정의 다음 장도 실패한다")
        # 'package is not installed' 를 SAM3 패키지로 읽지 않는다
        self.assertNotIn("install.py", sn.sam3_error_hint("FaceDetectorUnavailable: " + EXT_FACE_NO_ULTRALYTICS))
        self.assertIn("my_face.pt", sn.sam3_error_hint("FaceDetectorUnavailable: " + EXT_FACE_MODEL_MISSING,
                                                       {"sam3_face_crop_model": "my_face.pt"}))


# ── 샘플링 블록 게이트·보조 패스 전달 알림 (P7, T17) ─────────────────────────────
class PropagationNoticeTests(unittest.TestCase):
    def test_reencoding_i2i_with_empty_init_images_is_an_img2img_request(self):
        # core/i2i_payload.py 는 워커가 채울 때까지 init_images=[] 로 둔다 — 키가 있으면 img2img 목록으로 본다
        self.assertTrue(sn.requested_features({"init_images": [], "alwayson_scripts": {}}).img2img)
        self.assertFalse(sn.requested_features({"alwayson_scripts": {}}).img2img)
        caps = SimpleNamespace(known=True, installed=True,
                               script=lambda _t: {"present": True, "img2img": False})
        payload = {"init_images": [], **pag_payload(guid_enabled=True)}
        self.assertEqual(codes(sn.pre_generation_notices(payload, capabilities=caps)), [sn.CODE_SCRIPT_MISSING])

    def test_propagated_features_are_marked_in_result_notices(self):
        payload = pag_payload(guid_enabled=True)
        plain = sn.result_notices(info_of(SAM3_OK), payload)
        marked = sn.result_notices(info_of(SAM3_OK), payload,
                                   propagated_titles=("anima perturbation guidance",))
        self.assertEqual(codes(plain), codes(marked))
        self.assertFalse(plain[0].message.endswith(sn.PROPAGATED_SUFFIX))
        self.assertEqual(marked[0].message, plain[0].message + sn.PROPAGATED_SUFFIX)
        # 다른 기능의 알림은 그대로
        sam3 = sn.result_notices(info_of(SAM3_OOM), sam3_payload(), propagated_titles=("Anima Detail Daemon",))
        self.assertFalse(sam3[0].message.endswith(sn.PROPAGATED_SUFFIX))

    def test_new_codes_are_throttled_like_pre_generation_warnings(self):
        for code in (sn.CODE_BLOCK_NOT_SENT, sn.CODE_BLOCK_DEFERRED, sn.CODE_PROPAGATION_DROPPED,
                     sn.CODE_PROPAGATION_RETRIED):
            with self.subTest(code=code):
                notice = sn.Notice(code, sn.LEVEL_INFO, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
        self.assertEqual(sn.block_not_sent_notice("Anima Skimmed CFG", img2img=True).feature, "skimmed_cfg")
        self.assertEqual(sn.propagation_retried_notice("NegPiP", "SAM3").level, sn.LEVEL_WARNING)

    def test_retried_notice_refreshes_the_gui_snapshot_once(self):
        """(A1) 캐시를 버리지 않고 GUI 스냅샷을 워커에서 다시 받는다 — 억제된 반복은 다시 부르지 않는다."""
        from ui.sam_extra_notices_ui import show_notices
        now = [1000.0]
        host = _host(_sam_extra_notice_throttle=sn.NoticeThrottle(lambda: now[0]),
                     _refresh_sam_extra_capabilities=mock.Mock())
        retried = sn.propagation_retried_notice("Anima Detail Daemon", "Refine")
        with self.assertLogs("ui.sam_extra_notices_ui", "INFO"):
            show_notices(host, [sn.propagation_dropped_notice("Anima Skimmed CFG", "SAM3")])
            host._refresh_sam_extra_capabilities.assert_not_called()
            show_notices(host, [retried])
            now[0] += sn.RESULT_NOTICE_TTL_S + 1
            self.assertEqual(show_notices(host, [retried]), 0)          # 배치 100장이 30초마다 띄우지 않는다
        host._refresh_sam_extra_capabilities.assert_called_once_with(force=True)

    def test_before_generation_shows_pending_sampling_notices_on_any_backend(self):
        from backends import BackendType
        from ui import sam_extra_notices_ui as ui_mod
        from ui.sampling_blocks import remember_sampling_notices
        host = _host()
        payload = {"prompt": "x"}
        remember_sampling_notices(host, payload, [sn.block_not_sent_notice("Anima Skimmed CFG", img2img=False)])
        with mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI), \
                self.assertLogs("ui.sam_extra_notices_ui", "WARNING"):
            self.assertEqual(ui_mod.check_before_generation(host, payload), 1)
        self.assertEqual(ui_mod.check_before_generation(host, payload), 0)   # 한 번만


class MainRetryAndForgeOptionNoticeTests(unittest.TestCase):
    """P10 — 메인 생성 재시도·Forge 옵션 알림은 결과 info 로 오고, 거절 알림을 띄울 때 GUI 스냅샷을 다시 받는다(A1)."""

    def test_main_result_notices_refresh_the_gui_snapshot_once_per_ttl(self):
        from ui.sam_extra_notices_ui import show_result_notices
        now = [1000.0]
        host = _host(_sam_extra_notice_throttle=sn.NoticeThrottle(lambda: now[0]),
                     _refresh_sam_extra_capabilities=mock.Mock())
        retried = sn.app_block_retried_notice("DoRA Inference Mode")
        rejected = sn.forge_option_rejected_notice(["PAG 앞 블록 중복 계산 건너뛰기"], frozen=False)
        info = {sn.INFO_KEY: sn.notices_to_dicts([retried, rejected])}
        with self.assertLogs("ui.sam_extra_notices_ui", "INFO"):
            self.assertEqual(show_result_notices(host, info), 2)
            now[0] += sn.RESULT_NOTICE_TTL_S + 1
            self.assertEqual(show_result_notices(host, info), 0)   # 600초 억제(B14)
        host._refresh_sam_extra_capabilities.assert_called_once_with(force=True)

    def test_missing_and_unverified_notices_do_not_refresh(self):
        from ui.sam_extra_notices_ui import show_notices
        host = _host(_refresh_sam_extra_capabilities=mock.Mock())
        with self.assertLogs("ui.sam_extra_notices_ui", "INFO"):
            show_notices(host, [sn.forge_option_missing_notice(["x"]),
                                sn.forge_option_unverified_notice(["x"], no_config=True)])
        host._refresh_sam_extra_capabilities.assert_not_called()

    def test_app_block_retried_notice_names_the_block(self):
        notice = sn.app_block_retried_notice("DoRA Inference Mode")
        self.assertEqual((notice.code, notice.level), (sn.CODE_APP_BLOCK_RETRIED, sn.LEVEL_INFO))
        self.assertIn("'DoRA Inference Mode'", notice.message)
        self.assertIn("HTTP 422", notice.message)

    def test_block_retried_notice_is_a_neutral_throttled_warning_that_refreshes(self):
        """(P10 검토 2) 사용자가 켰거나 출처를 모르는 블록 — 누가 넣었는지 말하지 않고, 이번 결과에 빠졌다고 경고한다."""
        notice = sn.block_retried_notice("Anima 3.8B (Qwen3.5 / v2)")
        self.assertEqual((notice.code, notice.level), (sn.CODE_BLOCK_RETRIED, sn.LEVEL_WARNING))
        self.assertIn("'Anima 3.8B (Qwen3.5 / v2)'", notice.message)
        self.assertIn("Anima 3.8B 설정이 적용되지 않았습니다", notice.message)
        self.assertNotIn("앱이", notice.message)
        self.assertEqual(sn.NOTICE_MIN_TTL_S[sn.CODE_BLOCK_RETRIED], sn.PRE_GENERATION_NOTICE_TTL_S)
        self.assertIn(sn.CODE_BLOCK_RETRIED, sn.REFRESH_CAPABILITIES_CODES)



# ── VAE DeGrid 결과 알림 (core/vae_degrid) ─────────────────────────────────────
def degrid_payload(enabled=True, **arg):
    from core import vae_degrid
    block = vae_degrid.as_block(vae_degrid.DegridSettings(enabled=enabled, **arg))
    return {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {vae_degrid.SCRIPT_NAME: block}}


DEGRID_OK = _HEAD + ("Anima DeGrid model: qwenVAEDegridNafnet_v11, Anima DeGrid mode: Full, Anima DeGrid strength: 1, "
                     "Anima DeGrid tile: 512, Anima DeGrid precision: fp32, ") + _TAIL
DEGRID_ZERO = _HEAD + ("Anima DeGrid model: qwenVAEDegridNafnet_v11, Anima DeGrid mode: Full, Anima DeGrid strength: 0, "
                       "Anima DeGrid tile: 512, ") + _TAIL          # 강도 0 — 기록하되 정밀도 없음(건너뜀)
DEGRID_NOT_FOUND = _HEAD + "Anima DeGrid error: model not found: auto, " + _TAIL
# 쉼표가 든 값은 따옴표로(create_infotext). 숫자는 장마다 다르다
_NOT_RESIDUAL = ("not a DeGrid residual model: restore_x output does not look like a residual - it follows the input "
                 "like an image (mean |output| {:.1f}/255; correlation with the input +0.97) - use a VAE DeGrid NAFNet")
DEGRID_NOT_RESIDUAL = [_HEAD + f'Anima DeGrid error: "{_NOT_RESIDUAL.format(v)}", ' + _TAIL for v in (80.2, 91.7)]
DEGRID_BLEW_UP = _HEAD + ('Anima DeGrid error: "output blew up (mean |residual| 183.0/255 > 100/255 - v11 does not handle '
                          'this image, e.g. full-frame 1px stripes) - image kept without DeGrid", ') + _TAIL


class DegridResultNoticeTests(_QuietAssumption):
    def test_success_strength_zero_off_and_absent_blocks_are_quiet(self):
        for info, payload in ((info_of(DEGRID_OK), degrid_payload()), (info_of(DEGRID_ZERO), degrid_payload(strength=0)),
                              (info_of(NO_EXTENSION_TRACE), degrid_payload(enabled=False)),
                              (info_of(DEGRID_NOT_FOUND), {"prompt": "x"}),          # 보내지 않았다
                              (info_of(DEGRID_OK, DEGRID_OK), degrid_payload())):
            with self.subTest(info=info[:60]):
                self.assertEqual([n for n in sn.result_notices(info, payload) if n.feature == "degrid"], [])

    def test_error_only_infotext_warns_with_the_model_hint(self):
        [notice] = sn.result_notices(info_of(DEGRID_NOT_FOUND), degrid_payload())
        self.assertEqual((notice.code, notice.level, notice.feature), (sn.CODE_DEGRID_ERROR, sn.LEVEL_WARNING, "degrid"))
        self.assertIn("models/ESRGAN", notice.hint)
        self.assertIn("model not found: auto", notice.message)
        self.assertNotIn("장이", notice.message)                                 # 한 장이면 개수 없음

    def test_batch_counts_per_image_and_groups_numbers_only_differences(self):
        info = info_of(DEGRID_NOT_RESIDUAL[0], DEGRID_OK, DEGRID_NOT_RESIDUAL[1])
        [notice] = sn.result_notices(info, degrid_payload())
        self.assertIn("2/3장이", notice.message)
        self.assertEqual(notice.detail, "not a DeGrid residual model")         # 숫자는 억제 키에 들지 않는다
        self.assertIn("일반 복원 NAFNet", notice.hint)
        mixed = sn.result_notices(info_of(DEGRID_NOT_FOUND, DEGRID_BLEW_UP), degrid_payload())
        self.assertEqual([(n.code, n.detail) for n in mixed], [(sn.CODE_DEGRID_ERROR, "model not found: auto"),
                                                               (sn.CODE_DEGRID_ERROR, "output blew up")])
        self.assertTrue(all("1/2장이" in n.message for n in mixed))

    def test_quoted_error_with_commas_is_read_whole(self):
        params = sn.image_parameters(info_of(DEGRID_BLEW_UP))
        self.assertTrue(params[0][sn.KEY_DEGRID_ERROR].startswith("output blew up (mean |residual| 183.0/255"))
        self.assertTrue(params[0][sn.KEY_DEGRID_ERROR].endswith("image kept without DeGrid"))

    def test_error_kinds(self):
        cases = {"model not found: DeGrid/x": ("model not found: DeGrid/x", "models/DeGrid"),
                 "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB": ("out of memory", "CPU"),
                 "RuntimeError: boom 12": ("RuntimeError", "[AnimaDeGrid]"),
                 "strange 12 thing": ("strange # thing", "[AnimaDeGrid]")}
        for reason, (key, hint_part) in cases.items():
            with self.subTest(reason=reason):
                got_key, hint = sn.degrid_error_kind(reason)
                self.assertEqual(got_key, key)
                self.assertIn(hint_part, hint)

    def test_sent_but_no_trace_means_an_old_sam_extra(self):
        [notice] = sn.result_notices(info_of(NO_EXTENSION_TRACE, NO_EXTENSION_TRACE), degrid_payload())
        self.assertEqual((notice.code, notice.level), (sn.CODE_DEGRID_NOT_APPLIED, sn.LEVEL_WARNING))
        self.assertIn(sn.KEY_DEGRID_MODEL, notice.message)

    def test_requested_features_read_the_block_like_the_extension(self):
        from core import vae_degrid
        request = sn.requested_features(degrid_payload(mode="dark", tile=64))
        self.assertEqual(request.degrid, vae_degrid.DegridSettings(True, "", "dark", 1.0, 128))
        positional = {"alwayson_scripts": {"anima vae degrid (nafnet)": {"args": [True, "None", "Bright"]}}}
        self.assertEqual(sn.requested_features(positional).degrid.mode, "bright")
        self.assertIsNone(sn.requested_features({"prompt": "x"}).degrid)

    def test_comfy_reports_use_the_same_rules(self):
        """(A9) ComfyUI 노드 리포트 → Forge infotext 키 → 같은 알림."""
        from core import vae_degrid
        reports = [{"status": "ok", "model": "v11", "mode": "Full", "strength": 1.0, "tile": 512, "precision": "fp32",
                    "error": ""},
                   {"status": "skipped", "model": "restore_x", "mode": "Full", "strength": 1.0, "tile": 512,
                    "precision": "-", "error": _NOT_RESIDUAL.format(80.2)}]
        params = [vae_degrid.report_params(r) for r in reports]
        [notice] = sn.degrid_result_notices(degrid_payload(), params)
        self.assertEqual((notice.code, notice.detail), (sn.CODE_DEGRID_ERROR, "not a DeGrid residual model"))
        self.assertIn("1/2장이", notice.message)
        self.assertEqual(sn.degrid_result_notices(degrid_payload(), []), [])     # 리포트가 없으면 말하지 않는다
        self.assertEqual(sn.degrid_result_notices({"prompt": "x"}, params), [])
        self.assertEqual(sn.degrid_result_notices(degrid_payload(), [vae_degrid.report_params(reports[0])]), [])

    def test_comfy_unavailable_notice_and_throttling(self):
        notice = sn.degrid_comfy_unavailable_notice("ComfyUI 팩 1.5.0 필요")
        self.assertEqual((notice.code, notice.level, notice.feature),
                         (sn.CODE_DEGRID_COMFY_UNAVAILABLE, sn.LEVEL_WARNING, "degrid"))
        self.assertIn("팩 1.5.0", notice.message)
        for name in ("CODE_DEGRID_ERROR", "CODE_DEGRID_NOT_APPLIED", "CODE_DEGRID_COMFY_UNAVAILABLE"):
            with self.subTest(code=name):
                notice = sn.Notice(getattr(sn, name), sn.LEVEL_WARNING, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
                self.assertIn(name, sn.__all__)

    def test_rejected_request_and_gate_notices_name_the_feature(self):
        text = sn.explain_rejected_request(
            422, {"detail": "always on script Anima VAE DeGrid (NAFNet) not found"}, degrid_payload())
        self.assertIn("켠 기능: VAE DeGrid", text)
        self.assertIn("sam-extra", text)
        self.assertEqual(sn.block_not_sent_notice("Anima VAE DeGrid (NAFNet)", img2img=True).feature, "degrid")


@requires_extension
class DegridExtensionSourceTests(unittest.TestCase):
    """DeGrid 알림이 기대는 infotext 키·실패 문구 앞부분이 설치된 확장에 그대로 있는가."""

    def test_infotext_keys_and_failure_prefixes(self):
        from core import vae_degrid
        from core.sam_extra_scan import ExtensionSource
        src = ExtensionSource(EXT_ROOT)
        for name, value in (("KEY_MODEL", sn.KEY_DEGRID_MODEL), ("KEY_ERROR", sn.KEY_DEGRID_ERROR),
                            ("KEY_MODE", vae_degrid.KEY_MODE), ("KEY_STRENGTH", vae_degrid.KEY_STRENGTH),
                            ("KEY_TILE", vae_degrid.KEY_TILE), ("KEY_PRECISION", vae_degrid.KEY_PRECISION),
                            ("TITLE", vae_degrid.SCRIPT_NAME)):
            with self.subTest(name=name):
                self.assertEqual(src.module_constant("sam3ext/ui_vae_degrid.py", name), value)
        texts = []
        for rel in ("sam3ext/vae_degrid_runtime.py", "scripts/anima_vae_degrid.py"):
            with open(os.path.join(EXT_ROOT, *rel.split("/")), encoding="utf-8") as fh:
                texts.append(fh.read())
        joined = "\n".join(texts)
        for prefix in (vae_degrid.ERROR_MODEL_NOT_FOUND, vae_degrid.ERROR_NOT_RESIDUAL, vae_degrid.ERROR_BLEW_UP):
            with self.subTest(prefix=prefix):
                self.assertIn(f'"{prefix}', joined.replace("f\"", "\""))


# ── Extra Schedulers (계획 A-XS3) ──────────────────────────────────────────────
def xsched_payload(**values):
    from core import extra_schedulers
    block = extra_schedulers.as_block(extra_schedulers.ExtraSchedulerSettings(**values))
    return {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {extra_schedulers.SCRIPT_NAME: block}}


def _xsched_head(schedule: str, hires: str = "") -> str:
    head = _HEAD.replace("Schedule type: Automatic", f"Schedule type: {schedule}")
    return head + (f"Hires schedule type: {hires}, " if hires else "")


XSCHED_LAPLACE_OK = _xsched_head("Laplace") + "Laplace mu: 0.5, Laplace beta: 0.5, " + _TAIL
XSCHED_LAPLACE_FOREIGN = _xsched_head("Laplace") + _TAIL                  # 다른 확장의 'Laplace' 가 돌았다
XSCHED_CUSTOM_OK = _xsched_head("custom") + 'Custom scheduler sigmas: "[1.0, 0.6, 0.25, 0.1, 0.0]", ' + _TAIL
XSCHED_HIRES_CUSTOM_FOREIGN = _xsched_head("Karras", hires="custom") + _TAIL
XSCHED_KARRAS = _xsched_head("Karras") + _TAIL
XSCHED_COSMOS_OK = (_xsched_head("Flow Cosmos rho7", hires="Flow Cosmos Dynamic")
                    + "Flow Cosmos rho: 7.0, Flow Cosmos sigma max: 80.0, Flow Cosmos sigma min: 0.002, " + _TAIL)
XSCHED_COSMOS_DYNAMIC_FOREIGN = _xsched_head("Flow Cosmos Dynamic") + _TAIL   # 같은 라벨의 다른 스케줄러가 돌았다


class ExtraSchedulersNoticeTests(_QuietAssumption):
    def test_quiet_when_the_values_were_recorded_or_not_used_or_not_sent(self):
        for info, payload in ((info_of(XSCHED_LAPLACE_OK), xsched_payload()),
                              (info_of(XSCHED_CUSTOM_OK), xsched_payload(custom_mode="sigmas")),
                              (info_of(XSCHED_KARRAS), xsched_payload()),
                              (info_of(XSCHED_COSMOS_OK), xsched_payload(flow_cosmos_rho=7.0)),
                              (info_of(XSCHED_LAPLACE_FOREIGN), {"prompt": "x"})):     # 블록을 보내지 않았다
            with self.subTest(info=info[150:230]):
                self.assertEqual([n for n in sn.result_notices(info, payload) if n.feature == "extra_schedulers"], [])

    def test_laplace_or_custom_without_their_keys_warn_once(self):
        [notice] = sn.result_notices(info_of(XSCHED_LAPLACE_FOREIGN, XSCHED_LAPLACE_OK, XSCHED_LAPLACE_FOREIGN),
                                     xsched_payload())
        self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                         (sn.CODE_XSCHED_NOT_APPLIED, sn.LEVEL_WARNING, "extra_schedulers", "Laplace"))
        self.assertIn("'Laplace mu'·'Laplace beta'", notice.message)
        self.assertIn("aoleg/Neo_ExtraSchedulers", notice.message)
        [hires] = sn.result_notices(info_of(XSCHED_HIRES_CUSTOM_FOREIGN), xsched_payload())
        self.assertEqual(hires.detail, "custom")
        self.assertIn("Custom scheduler expression", hires.message)
        both = sn.xsched_result_notices(xsched_payload(), [
            {"Schedule type": "Laplace", "Hires schedule type": "custom"}])
        self.assertEqual([n.detail for n in both], ["Laplace · custom"])
        self.assertEqual(sn.xsched_result_notices(xsched_payload(), []), [])

    def test_flow_cosmos_without_its_keys_warns_like_laplace(self):
        """v0.33.0 D19/D20: Flow Cosmos rho7·Dynamic 은 돈 장에 'Flow Cosmos rho'·'sigma max'·'sigma min' 을 늘 남긴다."""
        [notice] = sn.result_notices(info_of(XSCHED_COSMOS_DYNAMIC_FOREIGN), xsched_payload())
        self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                         (sn.CODE_XSCHED_NOT_APPLIED, sn.LEVEL_WARNING, "extra_schedulers", "Flow Cosmos Dynamic"))
        self.assertIn("'Flow Cosmos rho'·'Flow Cosmos sigma max'·'Flow Cosmos sigma min'", notice.message)
        self.assertIn("ρ/σ̃", notice.message)
        both = sn.xsched_result_notices(xsched_payload(), [
            {"Schedule type": "flow_cosmos_rho7", "Hires schedule type": "Laplace"}])
        self.assertEqual([n.detail for n in both], ["Laplace · Flow Cosmos rho7"])
        self.assertEqual(sn.xsched_result_notices(xsched_payload(), [
            {"Schedule type": "Flow Cosmos Dynamic", "Flow Cosmos rho": "7.0", "Flow Cosmos sigma max": "80.0",
             "Flow Cosmos sigma min": "0.002"}]), [])
        self.assertEqual(sn.xsched_result_notices({"prompt": "x"}, [{"Schedule type": "Flow Cosmos rho7"}]), [])

    def test_requested_features_read_the_block_like_the_extension(self):
        from core import extra_schedulers
        request = sn.requested_features(xsched_payload(laplace_mu=12.0))
        self.assertEqual(request.extra_schedulers, extra_schedulers.ExtraSchedulerSettings(laplace_mu=10.0))
        positional = {"alwayson_scripts": {"extra schedulers (sam-extra)": {"args": ["Sigma list", " x "]}}}
        self.assertEqual(sn.requested_features(positional).extra_schedulers,
                         extra_schedulers.ExtraSchedulerSettings(custom_mode="sigmas", custom_expression="x"))
        self.assertIsNone(sn.requested_features({"prompt": "x"}).extra_schedulers)

    def test_invalid_text_notice_and_scheduler_error_text(self):
        notice = sn.xsched_invalid_text_notice("name 'q' is not allowed in a scheduler expression")
        self.assertEqual((notice.code, notice.level, notice.feature),
                         (sn.CODE_XSCHED_INVALID_TEXT, sn.LEVEL_WARNING, "extra_schedulers"))
        self.assertIn("custom 식을", notice.message)
        self.assertIn("시그마 목록을", sn.xsched_invalid_text_notice("x", sigmas=True).message)
        body = ('{"error": "CustomSchedulerError", "detail": "", "errors": "Extra Schedulers: the custom scheduler\'s '
                'sigma list gives sigma 0.5 at steps 3 and 4 (no change)"}')
        text = sn.explain_scheduler_error(body)
        self.assertTrue(text.startswith(sn.XSCHED_ERROR_LEAD + ": the custom scheduler's sigma list"), text)
        self.assertEqual(sn.explain_scheduler_error("Extra Schedulers:"), sn.XSCHED_ERROR_LEAD)
        for other in ("", None, "RuntimeError: boom", {"detail": "Not Found"}):
            self.assertIsNone(sn.explain_scheduler_error(other))

    def test_throttling_exports_and_feature_name(self):
        for name in ("CODE_XSCHED_NOT_APPLIED", "CODE_XSCHED_INVALID_TEXT"):
            with self.subTest(code=name):
                notice = sn.Notice(getattr(sn, name), sn.LEVEL_WARNING, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
                self.assertIn(name, sn.__all__)
        text = sn.explain_rejected_request(
            422, {"detail": "always on script Extra Schedulers (sam-extra) not found"}, xsched_payload())
        self.assertIn("켠 기능: Extra Schedulers 값", text)
        self.assertEqual(sn.block_not_sent_notice("Extra Schedulers (sam-extra)", img2img=False).feature,
                         "extra_schedulers")
        [propagated] = sn.result_notices(info_of(XSCHED_LAPLACE_FOREIGN), xsched_payload(),
                                         propagated_titles=("extra schedulers (sam-extra)",))
        self.assertTrue(propagated.message.endswith(sn.PROPAGATED_SUFFIX))


# ── Extra Samplers (계획 A-XSa2) ───────────────────────────────────────────────
def xsamp_payload(**values):
    from core import extra_samplers
    block = extra_samplers.as_block(extra_samplers.ExtraSamplersSettings(**values))
    return {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {extra_samplers.SCRIPT_NAME: block}}


def _quote(value: str) -> str:
    """A1111 create_infotext 의 따옴표 규칙(쉼표·콜론·줄바꿈이 든 값만 JSON 따옴표)."""
    return json.dumps(value) if any(ch in value for ch in ",:\n") else value


def xsamp_infotext(sampler: str = "ER SDE (Reverse-time)", *, hires: str = "", status: str = "", keys: str = "") -> str:
    """확장이 쓰는 모양 — 'Sampler'·'Hires sampler'(다를 때만)·값 키·'Extra Samplers status'."""
    text = _HEAD.replace("Sampler: Euler a", f"Sampler: {sampler}")
    if hires:
        text += f"Hires sampler: {hires}, "
    text += keys
    if status:
        text += f"Extra Samplers status: {_quote(status)}, "
    return text + _TAIL


def _caps_with_xsamp_argc(argc):
    from core import extra_samplers
    from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
    detail = {"present": True, "img2img": True, "live_argc": argc, "spec_argc": 5, "trailing_unmapped": [],
              "extra_live_args": 0}
    return SamExtraCapabilities(status="ok", installed=True,
                                scripts=_freeze({extra_samplers.SCRIPT_NAME.lower(): detail}))


_DY_SPECTRUM = "dy sub-steps skipped (Spectrum)"
_DY_SPECTRUM_PID = "dy sub-steps skipped (Spectrum + PiD lq_latent)"
_WINDOW_SKIPPED = "noise window skipped (no percent_to_sigma)"
XSAMP_OK = xsamp_infotext(keys="ER SDE max stage: 2, ER SDE eta: 0.5, ER SDE noise window: 0.2-0.8, ")
XSAMP_DY_OK = xsamp_infotext("Euler SMEA Dy CFG++")
XSAMP_DY_SPECTRUM = xsamp_infotext("Euler SMEA Dy CFG++", status=_DY_SPECTRUM)
XSAMP_DY_SPECTRUM_PID = xsamp_infotext("Euler Dy", status=_DY_SPECTRUM_PID)
XSAMP_WINDOW_SKIPPED = xsamp_infotext("ER SDE (Tunable)", hires="Euler SMEA Dy", status=f"{_DY_SPECTRUM}; {_WINDOW_SKIPPED}",
                                      keys="ER SDE eta: 0.5, ")
XSAMP_NO_WINDOW_KEY = xsamp_infotext(keys="ER SDE eta: 0.5, ")      # 인자 2개 sam-extra — 구간 키가 없다


class ExtraSamplersNoticeTests(_QuietAssumption):
    def _xsamp(self, info, payload=None, capabilities=None):
        return [n for n in sn.result_notices(info, xsamp_payload() if payload is None else payload,
                                             capabilities=capabilities) if n.feature == "extra_samplers"]

    def test_quiet_without_a_status_part_or_a_window_problem(self):
        window = xsamp_payload(max_stage=2, eta=0.5, noise_window=True)
        five, two = _caps_with_xsamp_argc(5), _caps_with_xsamp_argc(2)
        for info, payload, capabilities in (
                (info_of(XSAMP_OK), window, five),                                  # 0.33.0 이 구간을 적었다
                (info_of(XSAMP_DY_OK), None, None),                                 # Dy 가 보조 스텝으로 돌았다
                (info_of(XSAMP_NO_WINDOW_KEY), window, None),                       # 스냅샷을 모르면 말하지 않는다
                (info_of(XSAMP_NO_WINDOW_KEY), window, five),
                (info_of(XSAMP_NO_WINDOW_KEY), xsamp_payload(eta=0.5), two),        # 구간을 켜지 않았다
                (info_of(xsamp_infotext("Euler a")), window, two),                  # 구간을 읽는 샘플러가 돌지 않았다
                (info_of(xsamp_infotext(LABEL_ODE)), window, two),
                (info_of(XSAMP_NO_WINDOW_KEY), {"prompt": "x"}, two),               # 블록을 보내지 않았다
                (info_of(xsamp_infotext(status="something new (x)")), None, None)):  # 모르는 부분
            with self.subTest(info=info[150:260], capabilities=capabilities):
                self.assertEqual(self._xsamp(info, payload, capabilities), [])
        self.assertEqual(sn.extra_samplers_result_notices(xsamp_payload(), []), [])

    def test_dy_substeps_skipped_is_one_info_notice_with_korean_reasons(self):
        [notice] = self._xsamp(info_of(XSAMP_DY_SPECTRUM, XSAMP_DY_OK, XSAMP_DY_SPECTRUM_PID), {"prompt": "x"})
        self.assertEqual((notice.code, notice.level, notice.feature),
                         (sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, sn.LEVEL_INFO, "extra_samplers"))
        self.assertEqual(notice.detail, "Spectrum + PiD lq_latent")
        self.assertIn("2/3장에서", notice.message)
        self.assertIn("Forge Spectrum Integrated 가 켜져 있습니다", notice.message)
        self.assertIn("PiD 저화질 latent", notice.message)
        self.assertIn("'Extra Samplers status'", notice.message)
        [single] = sn.extra_samplers_result_notices(None, [{"Extra Samplers status": _DY_SPECTRUM}])
        self.assertNotIn("장", single.message.split("건너뛰고")[0])
        self.assertEqual(single.detail, "Spectrum")
        [unknown] = sn.extra_samplers_result_notices(None, [{"Extra Samplers status": "dy sub-steps skipped (Future)"}])
        self.assertIn("Future", unknown.message)                          # 모르는 이유는 원문 그대로

    def test_window_skipped_is_read_after_splitting_the_status_parts(self):
        notices = self._xsamp(info_of(XSAMP_WINDOW_SKIPPED, XSAMP_OK), xsamp_payload(eta=0.5, noise_window=True),
                              _caps_with_xsamp_argc(5))
        self.assertEqual([(n.code, n.level) for n in notices],
                         [(sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, sn.LEVEL_INFO),
                          (sn.CODE_EXTRA_SAMPLERS_WINDOW_SKIPPED, sn.LEVEL_INFO)])
        window = notices[1]
        self.assertEqual(window.detail, _WINDOW_SKIPPED)
        self.assertIn("1/2장에서", window.message)
        self.assertIn("percent_to_sigma", window.message)
        for status in (_WINDOW_SKIPPED, f"{_WINDOW_SKIPPED}; {_DY_SPECTRUM}", f" {_WINDOW_SKIPPED} ;something new (x)"):
            with self.subTest(status=status):
                codes_ = [n.code for n in sn.extra_samplers_result_notices(None, [{"Extra Samplers status": status}])]
                self.assertIn(sn.CODE_EXTRA_SAMPLERS_WINDOW_SKIPPED, codes_)

    def test_window_on_a_two_argument_sam_extra_is_a_warning(self):
        from core import extra_samplers
        window = xsamp_payload(eta=0.5, noise_window=True)
        two = _caps_with_xsamp_argc(2)
        for info in (info_of(XSAMP_NO_WINDOW_KEY), info_of(xsamp_infotext("Euler a", hires=LABEL_TUNABLE)),
                     info_of(xsamp_infotext("er_sde_reverse_time"))):
            with self.subTest(info=info[150:260]):
                [notice] = self._xsamp(info, window, two)
                self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                                 (sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED, sn.LEVEL_WARNING, "extra_samplers",
                                  "live_argc=2"))
                self.assertIn(extra_samplers.WINDOW_UNSUPPORTED_TEXT, notice.message)
                self.assertIn("인자가 2개", notice.message)
                self.assertIn(f"'{extra_samplers.KEY_NOISE_WINDOW}'", notice.message)
                self.assertIn(extra_samplers.WINDOW_MIN_VERSION, notice.message)
        # 0.31·0.32 는 구간을 모르므로 건너뜀 부분도 쓰지 않는다 — Dy 부분과는 따로 나온다
        notices = self._xsamp(info_of(xsamp_infotext(LABEL_TUNABLE, hires="Euler Dy", status=_DY_SPECTRUM)), window, two)
        self.assertEqual([n.code for n in notices],
                         [sn.CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED, sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED])

    def test_window_warning_counts_adetailer_and_sam3_passes_of_the_request(self):
        """FIX-forge(claude:forge-notices): ADetailer·SAM3 패스만 Reverse-time·Tunable 을 써도 그 안쪽 패스가 같은 값을
        읽는다 — 0.31·0.32 는 구간을 버리므로 경고한다(메인·Hires 는 결과 infotext 로 본다). ODE 만이면 조용하다."""
        two = _caps_with_xsamp_argc(2)
        info = info_of(xsamp_infotext("Euler", keys="ADetailer sampler: ER SDE (Reverse-time), "))

        def payload(**scripts):
            out = xsamp_payload(eta=1.0, noise_window=True)
            out["sampler_name"] = "Euler"
            out["alwayson_scripts"].update(scripts)
            return out

        adetailer = {"ADetailer": {"args": [True, False, {"ad_model": "face_yolov8n.pt", "ad_use_sampler": True,
                                                          "ad_sampler": "ER SDE (Reverse-time)"}]}}
        sam3 = {"SAM3 Mask": {"args": [{"sam3_enable": True, "sam3_mode": "Inpaint", "sam3_sampler": "er_sde_tunable"}]}}
        for scripts in (adetailer, sam3):
            with self.subTest(scripts=list(scripts)):
                [notice] = self._xsamp(info, payload(**scripts), two)
                self.assertEqual((notice.code, notice.level), (sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED, sn.LEVEL_WARNING))
        ode = {"ADetailer": {"args": [True, False, {"ad_model": "face_yolov8n.pt", "ad_use_sampler": True,
                                                    "ad_sampler": LABEL_ODE}]}}
        mask_only = {"SAM3 Mask": {"args": [{"sam3_enable": True, "sam3_mode": "Mask only",
                                             "sam3_sampler": "er_sde_tunable"}]}}
        for scripts in (ode, mask_only, {}):
            with self.subTest(quiet=list(scripts)):
                self.assertEqual(self._xsamp(info, payload(**scripts), two), [])
        self.assertEqual(self._xsamp(info, payload(**adetailer), _caps_with_xsamp_argc(5)), [])

    def test_window_warning_is_quiet_at_eta_zero(self):
        """FIX-forge(claude:forge-notices): η 0 이면 구간은 어느 판에서도 아무 일도 하지 않는다 — 업데이트 경고 없음."""
        two = _caps_with_xsamp_argc(2)
        info = info_of(xsamp_infotext())
        self.assertEqual(self._xsamp(info, xsamp_payload(eta=0.0, noise_window=True), two), [])
        [notice] = self._xsamp(info, xsamp_payload(eta=0.01, noise_window=True), two)
        self.assertEqual(notice.code, sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED)

    def test_window_warning_is_quiet_when_a_result_carries_the_window_key(self):
        """FIX-forge(claude:forge-notices): 낡은 스냅샷(인자 2개)인데 결과에 'ER SDE noise window' 가 있으면 구간이 돌았다 —
        '기록 없음'이라고 말하지 않는다(어느 장이든 하나라도)."""
        two = _caps_with_xsamp_argc(2)
        window = xsamp_payload(eta=0.5, noise_window=True)
        self.assertEqual(self._xsamp(info_of(XSAMP_OK), window, two), [])
        self.assertEqual(self._xsamp(info_of(XSAMP_NO_WINDOW_KEY, XSAMP_OK), window, two), [])
        [notice] = self._xsamp(info_of(XSAMP_NO_WINDOW_KEY, XSAMP_NO_WINDOW_KEY), window, two)
        self.assertEqual(notice.code, sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED)

    def test_window_warning_is_quiet_when_a_result_says_the_window_was_skipped(self):
        """FIX-recheck: 'noise window skipped' 부분은 0.33.0 부터만 쓴다 — 연결된 sam-extra 는 구간을 안다(인자 2개
        스냅샷이 낡았다). 구간 건너뜀 정보만 내고 '업데이트하세요' 경고는 내지 않는다(어느 장이든 하나라도)."""
        two = _caps_with_xsamp_argc(2)
        window = xsamp_payload(eta=0.5, noise_window=True)
        skipped = xsamp_infotext(LABEL_TUNABLE, status=_WINDOW_SKIPPED, keys="ER SDE eta: 0.5, ")
        for info in (info_of(skipped), info_of(XSAMP_NO_WINDOW_KEY, skipped)):
            with self.subTest(images=info.count("Steps:")):
                self.assertEqual([n.code for n in self._xsamp(info, window, two)],
                                 [sn.CODE_EXTRA_SAMPLERS_WINDOW_SKIPPED])

    def test_window_warning_for_inner_passes_only_says_it_rests_on_the_snapshot(self):
        """FIX-recheck(claude:forge-notices #1+#3): sam-extra 0.33.0 은 ADetailer·SAM3 패스의 구간을 결과 정보에 적지 않는다
        (실제 v0.33.0 Forge, GPU part 2 ad_rt__a — ER SDE 키가 하나도 없다, O3). 그 패스만 Reverse-time·Tunable 이면
        결과로는 구간이 빠졌는지 알 수 없다 — 경고는 연결 때 받은 기능 정보가 근거라고 말하고 '기록 없음'을 근거로
        대지 않는다. 메인·Hires 가 돈 장이 있으면 결과가 근거다(기존 문구)."""
        from core import extra_samplers
        two = _caps_with_xsamp_argc(2)
        payload = xsamp_payload(eta=0.5, noise_window=True)
        payload["sampler_name"] = "Res Multistep"
        payload["alwayson_scripts"]["ADetailer"] = {"args": [True, False, {
            "ad_model": "face_yolov8n.pt", "ad_use_sampler": True, "ad_sampler": "ER SDE (Reverse-time)"}]}
        ad_rt = info_of(xsamp_infotext("Res Multistep", keys="ADetailer model: face_yolov8n.pt, "
                                       "ADetailer use separate sampler: True, ADetailer sampler: ER SDE (Reverse-time), "))
        [notice] = self._xsamp(ad_rt, payload, two)
        self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                         (sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED, sn.LEVEL_WARNING, "extra_samplers",
                          "live_argc=2; snapshot"))
        self.assertIn(extra_samplers.WINDOW_UNSUPPORTED_TEXT, notice.message)
        self.assertIn("연결 때 받은 기능 정보", notice.message)
        self.assertIn("ADetailer 1 패스", notice.message)
        self.assertIn("결과로는 확인할 수 없습니다", notice.message)
        self.assertIn(extra_samplers.WINDOW_MIN_VERSION, notice.message)
        self.assertNotIn("기록 없음", notice.message)
        payload["alwayson_scripts"]["SAM3 Mask"] = {"args": [{"sam3_enable": True, "sam3_mode": "Inpaint",
                                                             "sam3_sampler": "er_sde_tunable"}]}
        [both] = self._xsamp(ad_rt, payload, two)
        self.assertIn("ADetailer 1·SAM3 패스", both.message)
        # 메인이 Reverse-time 으로 돈 장이 있으면 결과가 근거다 — 기존 문구·detail
        [main] = self._xsamp(info_of(XSAMP_NO_WINDOW_KEY), payload, two)
        self.assertEqual(main.detail, "live_argc=2")
        self.assertIn(f"'{extra_samplers.KEY_NOISE_WINDOW}' 기록 없음", main.message)
        # 새 스냅샷(0.33.0, 인자 다섯)이면 같은 결과가 조용하다
        self.assertEqual(self._xsamp(ad_rt, payload, _caps_with_xsamp_argc(5)), [])

    def test_window_warning_refreshes_the_gui_snapshot_once_per_ttl(self):
        """FIX-recheck(claude:forge-notices #3 선택 부분): 앱은 연결·수동 새로고침·REFRESH_CAPABILITIES_CODES 에서만 스냅샷을
        다시 받는다 — 연결 뒤 sam-extra 를 0.33.0 으로 올리고 Forge 를 다시 켜면 인자 2개 스냅샷이 남아 ADetailer·SAM3 만
        구간 샘플러인 생성마다(TTL 마다) 경고가 되풀이됐다. 경고를 띄우면 GUI 가 스냅샷(공용 캐시 포함)을 다시 받는다."""
        from ui.sam_extra_notices_ui import show_result_notices
        self.assertIn(sn.CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED, sn.REFRESH_CAPABILITIES_CODES)
        now = [1000.0]
        host = _host(_sam_extra_notice_throttle=sn.NoticeThrottle(lambda: now[0]),
                     _refresh_sam_extra_capabilities=mock.Mock())
        notices = self._xsamp(info_of(XSAMP_NO_WINDOW_KEY), xsamp_payload(eta=0.5, noise_window=True),
                              _caps_with_xsamp_argc(2))
        info = {sn.INFO_KEY: sn.notices_to_dicts(notices)}
        with self.assertLogs("ui.sam_extra_notices_ui", "WARNING"):
            self.assertEqual(show_result_notices(host, info), 1)
            now[0] += 1
            self.assertEqual(show_result_notices(host, info), 0)          # 억제된 반복은 다시 받지 않는다
        host._refresh_sam_extra_capabilities.assert_called_once_with(force=True)

    def test_requested_features_read_the_block_like_the_extension(self):
        from core import extra_samplers
        request = sn.requested_features(xsamp_payload(max_stage=2, eta=0.5, noise_window=True))
        self.assertEqual(request.extra_samplers,
                         extra_samplers.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True))
        loose = {"alwayson_scripts": {"EXTRA SAMPLERS": {"args": ["2", "12", "on", -1, "0.6"]}}}
        self.assertEqual(sn.requested_features(loose).extra_samplers,
                         extra_samplers.ExtraSamplersSettings(max_stage=2, eta=10.0, noise_window=True, noise_start=0.0,
                                                              noise_end=0.6))
        old = {"alwayson_scripts": {"Extra Samplers": {"args": [1, 0.25]}}}          # 0.32 까지의 두 인자
        self.assertEqual(sn.requested_features(old).extra_samplers,
                         extra_samplers.ExtraSamplersSettings(max_stage=1, eta=0.25))
        self.assertEqual(sn.requested_features({"alwayson_scripts": {"Extra Samplers": "x"}}).extra_samplers,
                         extra_samplers.APP_DEFAULTS)
        self.assertIsNone(sn.requested_features({"prompt": "x"}).extra_samplers)

    def test_throttling_exports_and_feature_name(self):
        from core import extra_samplers
        for name in ("CODE_EXTRA_SAMPLERS_SUBSTEPS_SKIPPED", "CODE_EXTRA_SAMPLERS_WINDOW_SKIPPED",
                     "CODE_EXTRA_SAMPLERS_WINDOW_UNSUPPORTED"):
            with self.subTest(code=name):
                notice = sn.Notice(getattr(sn, name), sn.LEVEL_WARNING, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
                self.assertIn(name, sn.__all__)
        self.assertIn("extra_samplers_result_notices", sn.__all__)
        self.assertEqual((sn.KEY_ER_SDE_MAX_STAGE, sn.KEY_ER_SDE_ETA, sn.KEY_ER_SDE_NOISE_WINDOW, sn.KEY_EXTRA_SAMPLERS_STATUS),
                         (extra_samplers.KEY_MAX_STAGE, extra_samplers.KEY_ETA, extra_samplers.KEY_NOISE_WINDOW,
                          extra_samplers.STATUS_KEY))
        self.assertEqual(set(sn._XSAMP_DY_REASON_TEXTS), set(extra_samplers.DY_REASONS))
        text = sn.explain_rejected_request(422, {"detail": "always on script Extra Samplers not found"}, xsamp_payload())
        self.assertIn("켠 기능: Extra Samplers 값", text)
        self.assertEqual(sn.block_not_sent_notice("Extra Samplers", img2img=False).feature, "extra_samplers")
        [propagated] = sn.result_notices(info_of(XSAMP_DY_SPECTRUM), xsamp_payload(),
                                         propagated_titles=("extra samplers",))
        self.assertTrue(propagated.message.endswith(sn.PROPAGATED_SUFFIX))


LABEL_ODE, LABEL_TUNABLE = "ER SDE (ODE)", "ER SDE (Tunable)"


@requires_extension
class ExtraSamplersExtensionSourceTests(unittest.TestCase):
    """Extra Samplers 알림이 기대는 상태 줄 모양 — 부분을 '; ' 로 잇고 Dy 부분은 '<머리> (<이유> + <이유>)'(핀은 상수만 본다)."""

    def test_status_shapes(self):
        with open(os.path.join(EXT_ROOT, "sam3ext", "extra_samplers", "substep_guard.py"), encoding="utf-8") as fh:
            guard = fh.read()
        with open(os.path.join(EXT_ROOT, "sam3ext", "extra_samplers", "registry.py"), encoding="utf-8") as fh:
            registry = fh.read()
        self.assertIn('value = "; ".join(parts.values())', guard)
        self.assertIn("text = f\"{_STATUS_PREFIX} ({' + '.join(seen)})\"", guard)
        self.assertIn('substep_guard.record_status_part(p, "noise_window", sampler_params.WINDOW_SKIPPED_STATUS)',
                      registry)


# ── Anima SPEED (계획 A-SP3) ───────────────────────────────────────────────────
def speed_payload(**values):
    from core import anima_speed
    block = anima_speed.as_block(anima_speed.SpeedSettings(**{"enabled": True, **values}))
    return {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {anima_speed.SCRIPT_NAME: block}}


_SPEED_SUMMARY = ("mode=transition; threshold=neo_shift; preset=anima; scales=0.5,1; delta=0.01; divisor=1.03; "
                  "manual=0.7; adaptive=True; transform=dct; seed=-1; hires=False")


def speed_infotext(status: str = "", *, summary: str = _SPEED_SUMMARY) -> str:
    """확장이 쓰는 모양(쉼표·콜론이 든 값은 JSON 따옴표 — create_infotext)."""
    text = _HEAD + f"Anima SPEED: {json.dumps(summary)}, "
    if status:
        text += f"Anima SPEED status: {json.dumps(status)}, "
    return text + _TAIL


SPEED_APPLIED = speed_infotext("base: applied (transition) - 5/28 steps coarse, 52x76 -> 104x152, step 5 sigma "
                               "0.7208->0.7208 | hires: not applied - Apply to Hires pass is off")
SPEED_NOT_FLOW = speed_infotext("base: skipped - not a flow-matching model (prediction_type=eps) - SDXL/SD1.x eps/v "
                                "models are not supported (kappa and the sigma* presets assume flow sigmas in (0, 1])")
SPEED_NO_COARSE = speed_infotext("img2img: skipped - no coarse steps: the pass starts at sigma 0.6000, at or below the "
                                 "first transition sigma 0.7208")
SPEED_INVALID = speed_infotext("base: invalid settings - spectrum A must be a positive number; got -1.0")
SPEED_FALLBACK = speed_infotext("base: fell back to plain sampling after RuntimeError: shape mismatch [1, 16, 52, 76]")
SPEED_PENDING = speed_infotext("base: pending - sampler wrapped, waiting for sampling")
SPEED_INTERRUPTED = speed_infotext("base: interrupted")


class SpeedNoticeTests(_QuietAssumption):
    def _speed(self, info, payload=None):
        return [n for n in sn.result_notices(info, speed_payload() if payload is None else payload)
                if n.feature == "speed"]

    def test_quiet_when_applied_hires_off_interrupted_off_or_not_sent(self):
        for info, payload in ((info_of(SPEED_APPLIED), None), (info_of(SPEED_INTERRUPTED), None),
                              (info_of(SPEED_APPLIED, SPEED_APPLIED), None),
                              (info_of(NO_EXTENSION_TRACE), speed_payload(enabled=False)),
                              (info_of(SPEED_INVALID), {"prompt": "x"})):              # 블록을 보내지 않았다
            with self.subTest(info=info[200:260]):
                self.assertEqual(self._speed(info, payload), [])
        self.assertEqual(sn.speed_result_notices(speed_payload(), []), [])

    def test_skips_are_one_info_notice_with_korean_reasons(self):
        [notice] = self._speed(info_of(SPEED_NOT_FLOW))
        self.assertEqual((notice.code, notice.level, notice.detail), (sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, "not_flow"))
        self.assertIn("기본: flow 모델이 아닙니다", notice.message)
        self.assertNotIn("장에서", notice.message)                              # 한 장이면 개수 없음
        [both] = self._speed(info_of(SPEED_NOT_FLOW, SPEED_NO_COARSE, SPEED_APPLIED))
        self.assertEqual(both.detail, "not_flow · no_coarse_steps")
        self.assertIn("1/3장에서", both.message)
        self.assertIn("img2img: 이 패스는 전환 σ 아래에서 시작해", both.message)
        [pending] = self._speed(info_of(SPEED_PENDING))
        self.assertEqual((pending.code, pending.detail), (sn.CODE_SPEED_SKIPPED, "pending"))
        self.assertEqual(sn.speed_skip_hint("Restart re-plans its own schedule")[0], "sampler")
        self.assertEqual(sn.speed_skip_hint("CompVisSampler is not a k-diffusion sampler (DDIM/PLMS-style samplers are "
                                            "not supported)")[0], "sampler")
        self.assertEqual(sn.speed_skip_hint("brand new 12 reason 0.5"), ("brand new # reason #", ""))

    def test_invalid_settings_and_fallback_warn(self):
        [invalid] = self._speed(info_of(SPEED_INVALID, SPEED_INVALID))
        self.assertEqual((invalid.code, invalid.level), (sn.CODE_SPEED_INVALID, sn.LEVEL_WARNING))
        self.assertEqual(invalid.detail, "spectrum A must be a positive number; got -1.0")
        self.assertIn("기본 패스", invalid.message)
        self.assertNotIn("장이", invalid.message)                               # 모든 장이면 개수 없음
        [fallback] = self._speed(info_of(SPEED_FALLBACK, SPEED_APPLIED))
        self.assertEqual((fallback.code, fallback.level, fallback.detail),
                         (sn.CODE_SPEED_FALLBACK, sn.LEVEL_WARNING, "RuntimeError"))
        self.assertIn("1/2장에서 기본 패스에서 실패해", fallback.message)
        self.assertIn("shape mismatch [1, 16, 52, 76]", fallback.message)

    def test_sent_but_no_trace_means_an_old_sam_extra_or_a_script_error(self):
        [old] = self._speed(info_of(NO_EXTENSION_TRACE, NO_EXTENSION_TRACE))
        self.assertEqual((old.code, old.level, old.detail),
                         (sn.CODE_SPEED_NOT_APPLIED, sn.LEVEL_WARNING, "Anima SPEED"))
        self.assertIn("0.31.0", old.message)
        # sam-extra 0.33.0: a/0 is an ordinary invalid setting (no notice of its own) — an older sam-extra's script crash
        # leaves no trace, so it reads as the same 'no trace' notice
        [script] = self._speed(info_of(NO_EXTENSION_TRACE), speed_payload(scales="1/0,1"))
        self.assertEqual((script.code, script.detail), (sn.CODE_SPEED_NOT_APPLIED, "Anima SPEED"))
        [invalid] = self._speed(info_of(speed_infotext("base: invalid settings - scales: division by zero: '1/0'")),
                                speed_payload(scales="1/0,1"))
        self.assertEqual((invalid.code, invalid.detail), (sn.CODE_SPEED_INVALID, "scales: division by zero: '1/0'"))

    def test_requested_features_read_the_block_like_the_extension(self):
        from core import anima_speed
        request = sn.requested_features(speed_payload(delta=0.02))
        self.assertEqual(request.speed, anima_speed.ScriptSettings(enabled=True, delta=0.02))
        alias = {"alwayson_scripts": {"anima speed": {"args": [{"enabled": True, "divisor": 1.5, "A": 7}]}}}
        got = sn.requested_features(alias).speed
        self.assertEqual((got.enabled, got.sigma_divisor, got.spectrum_A), (True, 1.5, 7.0))
        # a/0: the field's settings error like any bad value (sam-extra 0.33.0); the manual text only under 'manual'
        self.assertEqual(sn.requested_features(speed_payload(scales="0.5,1/0")).speed.error,
                         "scales: division by zero: '1/0'")
        quiet = sn.requested_features(speed_payload(manual_sigmas="0.7/0")).speed
        self.assertEqual((quiet.error, quiet.manual_sigmas), (None, (0.7,)))
        self.assertIsNone(sn.requested_features({"prompt": "x"}).speed)

    def test_static_skip_notice(self):
        from core import anima_speed
        for code in anima_speed.SKIP_MESSAGES:
            with self.subTest(code=code):
                notice = sn.speed_static_skip_notice(code)
                self.assertEqual((notice.code, notice.level, notice.feature, notice.detail),
                                 (sn.CODE_SPEED_SKIPPED, sn.LEVEL_INFO, "speed", code))
                self.assertIn(anima_speed.SKIP_MESSAGES[code], notice.message)

    def test_throttling_exports_and_feature_name(self):
        for name in ("CODE_SPEED_INVALID", "CODE_SPEED_FALLBACK", "CODE_SPEED_SKIPPED", "CODE_SPEED_NOT_APPLIED"):
            with self.subTest(code=name):
                notice = sn.Notice(getattr(sn, name), sn.LEVEL_WARNING, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
                self.assertIn(name, sn.__all__)
        for name in ("speed_result_notices", "speed_skip_hint", "speed_static_skip_notice"):
            self.assertIn(name, sn.__all__)
        text = sn.explain_rejected_request(422, {"detail": "always on script Anima SPEED not found"}, speed_payload())
        self.assertIn("켠 기능: Anima SPEED", text)
        self.assertEqual(sn.block_not_sent_notice("Anima SPEED", img2img=False).feature, "speed")


def colorcraft_payload(mapping=None, *, hires=False):
    """앱 카드가 보내는 모양 — 첫 인자 하나(to_infotext). 기본은 수정자 I 노출 0.3(Base)."""
    from core import colorcraft
    config = colorcraft.config_from_mapping(mapping or {"enabled": True, "I.exposure": 0.3})
    payload = {"prompt": "1girl", "cfg_scale": 4.5, "alwayson_scripts": {colorcraft.SCRIPT_NAME: colorcraft.as_block(config)}}
    if hires:
        payload["enable_hr"] = True
    return payload


def colorcraft_infotext(status: str = "", *, value: str = "v1;mods=I;I.exposure=0.3") -> str:
    """확장이 쓰는 모양(쉼표·콜론·세미콜론이 든 값은 JSON 따옴표 — create_infotext)."""
    text = _HEAD
    if value:
        text += f"SAM Extra Colorcraft: {json.dumps(value)}, "
    if status:
        text += f"SAM Extra Colorcraft status: {json.dumps(status)}, "
    return text + _TAIL


CC_APPLIED = colorcraft_infotext("base: krea2 (Wan21); mods I; 30 evals, 28 edited")
CC_NO_BASIS = colorcraft_infotext("base: no basis (SDXL); mods I; 30 evals, 0 edited; vector controls off on this "
                                  "model: only Contrast and Color Shift apply")
CC_VECTORS_MISSING = colorcraft_infotext("base: no basis (Wan21); mods I; 30 evals, 0 edited; colorcraft-krea2.safetensors "
                                         "missing: vector controls off; vector controls off on this model: only "
                                         "Contrast and Color Shift apply")
CC_ERROR = colorcraft_infotext("base: krea2 (Wan21); mods I; 30 evals, 27 edited; error KeyError x3 (input passed "
                               "through) | hires: krea2 (Wan21); mods I; 12 evals, 12 edited; non-finite result passed "
                               "through x1")
CC_NOT_ATTACHED = colorcraft_infotext("base: krea2 (Wan21); mods I; not attached (AttributeError)")
CC_REFUSED = colorcraft_infotext("not applied: unreadable panel state (expected a JSON object)", value="")


class ColorcraftNoticeTests(_QuietAssumption):
    def _cc(self, info, payload=None):
        return [n for n in sn.result_notices(info, colorcraft_payload() if payload is None else payload)
                if n.feature == "colorcraft"]

    def test_quiet_when_applied_off_not_sent_or_nothing_to_run(self):
        nothing = colorcraft_payload({"enabled": True, "I.strength": 0.0, "I.exposure": 0.3})   # 스케줄 0 → 항목 없음
        hires_only = colorcraft_payload({"enabled": True, "I.pass": "Hires", "I.exposure": 0.3})  # Hires 끈 요청
        for info, payload in ((info_of(CC_APPLIED), None), (info_of(CC_APPLIED, CC_APPLIED), None),
                              # 문자열에는 켬이 없다(받으면 켠 것) — 끈 블록은 dict 형식으로만 쓸 수 있다
                              (info_of(NO_EXTENSION_TRACE), {"alwayson_scripts": {"Colorcraft (sam-extra)": {
                                  "args": [{"enabled": False, "I.exposure": 0.3}]}}}),
                              (info_of(NO_EXTENSION_TRACE), {"alwayson_scripts": {"Colorcraft (sam-extra)": {
                                  "args": ["v1;mods="]}}}),                              # 켠 수정자 없음
                              (info_of(NO_EXTENSION_TRACE), {"prompt": "x"}),                    # 블록을 보내지 않았다
                              (info_of(NO_EXTENSION_TRACE), nothing), (info_of(NO_EXTENSION_TRACE), hires_only),
                              (info_of(CC_ERROR), {"alwayson_scripts": {"colorcraft (sam-extra)": {"args": [True]}}})):
            with self.subTest(info=info[200:260], payload=str(payload)[:120]):
                self.assertEqual(self._cc(info, payload), [])
        self.assertEqual(sn.colorcraft_result_notices(colorcraft_payload(), []), [])

    def test_sent_but_no_trace_means_an_old_sam_extra(self):
        [old] = self._cc(info_of(NO_EXTENSION_TRACE, NO_EXTENSION_TRACE))
        self.assertEqual((old.code, old.level, old.detail),
                         (sn.CODE_COLORCRAFT_MISSING, sn.LEVEL_WARNING, "SAM Extra Colorcraft"))
        self.assertIn("0.31.0", old.message)
        self.assertNotIn("장의", old.message)                                   # 모든 장이면 개수 없음
        [some] = self._cc(info_of(CC_APPLIED, NO_EXTENSION_TRACE))
        self.assertIn("1/2장의 결과에", some.message)
        hires_only = colorcraft_payload({"enabled": True, "I.pass": "Hires", "I.exposure": 0.3}, hires=True)
        [hires] = self._cc(info_of(NO_EXTENSION_TRACE), hires_only)            # Hires 패스만 도는 수정자도 기록을 남긴다
        self.assertEqual(hires.code, sn.CODE_COLORCRAFT_MISSING)

    def test_refused_arguments_are_a_contract_alarm(self):
        [refused] = self._cc(info_of(CC_REFUSED, CC_REFUSED))
        self.assertEqual((refused.code, refused.level, refused.detail),
                         (sn.CODE_COLORCRAFT_NOT_APPLIED, sn.LEVEL_WARNING, "refused"))
        self.assertIn("unreadable panel state", refused.message)
        self.assertNotIn(sn.CODE_COLORCRAFT_MISSING, codes(self._cc(info_of(CC_REFUSED))))   # status 가 있으면 흔적 있음

    def test_evaluation_errors_warn_per_kind_with_their_passes(self):
        notices = self._cc(info_of(CC_ERROR, CC_APPLIED), colorcraft_payload(hires=True))
        self.assertEqual([(n.code, n.level, n.detail) for n in notices],
                         [(sn.CODE_COLORCRAFT_ERROR, sn.LEVEL_WARNING, "error:KeyError"),
                          (sn.CODE_COLORCRAFT_ERROR, sn.LEVEL_WARNING, "nonfinite")])
        self.assertIn("1/2장에서 (기본 패스) 일부 스텝", notices[0].message)
        self.assertIn("(Hires 패스)", notices[1].message)
        [attach] = self._cc(info_of(CC_NOT_ATTACHED))
        self.assertEqual((attach.code, attach.detail), (sn.CODE_COLORCRAFT_ERROR, "not_attached:AttributeError"))

    def test_models_without_a_basis_inform_and_missing_vectors_warn(self):
        [info] = self._cc(info_of(CC_NO_BASIS))
        self.assertEqual((info.code, info.level, info.detail), (sn.CODE_COLORCRAFT_NO_BASIS, sn.LEVEL_INFO, "no_basis"))
        self.assertIn("대비·색 이동만", info.message)
        vectors = self._cc(info_of(CC_VECTORS_MISSING))
        self.assertEqual([(n.code, n.level, n.detail) for n in vectors],
                         [(sn.CODE_COLORCRAFT_NO_BASIS, sn.LEVEL_WARNING, "vectors_missing"),
                          (sn.CODE_COLORCRAFT_NO_BASIS, sn.LEVEL_INFO, "no_basis")])

    def test_status_parts(self):
        self.assertEqual(sn.colorcraft_status_parts(""), [])
        self.assertEqual(sn.colorcraft_status_parts("not applied: x; y | z"), [("", "not applied: x; y | z")])
        self.assertEqual(sn.colorcraft_status_parts("base: a; b | hires: c"),
                         [("base", "a"), ("base", "b"), ("hires", "c")])

    def test_requested_features_read_the_block_like_the_extension(self):
        from core import colorcraft
        request = sn.requested_features(colorcraft_payload({"enabled": True, "III.kind": "Punch", "III.active": True,
                                                            "III.clarity": 0.5}))
        self.assertEqual(request.colorcraft, colorcraft.parse_own("v1;mods=I,III;III.kind=Punch;III.clarity=0.5"))
        upper = {"alwayson_scripts": {"COLORCRAFT (SAM-EXTRA)": {"args": ["v1;mods=II;II.contrast=0.4"]}}}
        self.assertEqual(sn.requested_features(upper).colorcraft.modifiers[1]["contrast"], 0.4)
        self.assertIsNone(sn.requested_features({"alwayson_scripts": {colorcraft.SCRIPT_NAME: {"args": [True]}}})
                          .colorcraft)                                       # 위치 인자 목록은 앱이 만들지 않는다
        self.assertIsNone(sn.requested_features({"prompt": "x"}).colorcraft)

    def test_throttling_exports_and_feature_name(self):
        for name in ("CODE_COLORCRAFT_NOT_APPLIED", "CODE_COLORCRAFT_ERROR", "CODE_COLORCRAFT_NO_BASIS",
                     "CODE_COLORCRAFT_MISSING"):
            with self.subTest(code=name):
                notice = sn.Notice(getattr(sn, name), sn.LEVEL_WARNING, "m")
                self.assertEqual(sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S), sn.PRE_GENERATION_NOTICE_TTL_S)
                self.assertIn(name, sn.__all__)
        for name in ("colorcraft_result_notices", "colorcraft_status_parts"):
            self.assertIn(name, sn.__all__)
        text = sn.explain_rejected_request(422, {"detail": "always on script Colorcraft (sam-extra) not found"},
                                           colorcraft_payload())
        self.assertIn("켠 기능: Colorcraft", text)
        self.assertEqual(sn.block_not_sent_notice("Colorcraft (sam-extra)", img2img=False).feature, "colorcraft")


@requires_extension
class ColorcraftExtensionSourceTests(unittest.TestCase):
    """Colorcraft 알림이 기대는 status 문구가 설치된 확장에 그대로 있는가(f-string 이라 핀 대신 원문 검색)."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(EXT_ROOT, "sam3ext", "colorcraft", "hook.py"), encoding="utf-8") as fh:
            cls.source = fh.read()

    def test_status_shapes(self):
        for needle in ('_params(p)[spec.STATUS_KEY] = f"not applied: {reason}"',
                       'text = " | ".join(passes[name].render() for name in ("base", "hires") if name in passes)',
                       'return f"{self.name}: " + "; ".join(parts)',
                       'parts.append(f"error {self.last_error} x{self.errors} (input passed through)")',
                       'parts.append(f"non-finite result passed through x{self.nonfinite}")',
                       'status.notes.append(f"not attached ({type(exc).__name__})")',
                       'status.notes.append(f"anchors failed ({type(exc).__name__}): Contrast/Color Shift off")',
                       'status.notes.append(f"colorcraft-{family}.safetensors missing: vector controls off")',
                       'status.notes.append("vector controls off on this model: only Contrast and Color Shift apply")',
                       'params[spec.INFOTEXT_KEY] = spec.to_infotext(config)',
                       'print(f"[Colorcraft] {message}", file=sys.stderr)'):
            with self.subTest(needle=needle[:60]):
                self.assertIn(needle, self.source)

    def test_every_status_rule_matches_what_the_extension_writes(self):
        samples = {"error": "error KeyError x3 (input passed through)", "nonfinite": "non-finite result passed through x2",
                   "not_attached": "not attached (AttributeError)",
                   "anchors": "anchors failed (RuntimeError): Contrast/Color Shift off",
                   "vectors_missing": "colorcraft-flux2.safetensors missing: vector controls off",
                   "no_basis": "vector controls off on this model: only Contrast and Color Shift apply"}
        self.assertEqual(set(samples), {key for _pattern, _code, key in sn._COLORCRAFT_PART_RULES})
        for _pattern, _code, key in sn._COLORCRAFT_PART_RULES:
            with self.subTest(key=key):
                self.assertTrue(_pattern.match(samples[key]))


@requires_extension
class SpeedExtensionSourceTests(unittest.TestCase):
    """SPEED 알림이 기대는 status 문구 머리·건너뜀 이유가 설치된 확장에 그대로 있는가(f-string 이라 핀 대신 원문 검색)."""

    @classmethod
    def setUpClass(cls):
        texts = []
        for rel in ("scripts/anima_speed.py", "sam3ext/speed/forge_host.py", "sam3ext/speed/schedule.py"):
            with open(os.path.join(EXT_ROOT, *rel.split("/")), encoding="utf-8") as fh:
                texts.append(fh.read())
        cls.source = "\n".join(texts)

    def test_status_heads(self):
        from core import anima_speed as sp
        heads = {
            sp.STATUS_APPLIED: 'f"applied ({settings.mode}) - ',
            sp.STATUS_INVALID: 'f"invalid settings - {settings.error}"',
            sp.STATUS_SKIPPED: 'f"skipped - {reason}"',
            sp.STATUS_NOT_APPLIED: '"not applied - Apply to Hires pass is off"',
            sp.STATUS_PENDING: '"pending - sampler wrapped, waiting for sampling"',
            sp.STATUS_FALLBACK: 'f"fell back to plain sampling after {type(exc).__name__}: {exc}"',
            sp.STATUS_INTERRUPTED: 'record_status(p, label, "interrupted")',
        }
        self.assertEqual(set(heads), set(sp.STATUS_KINDS))
        for kind, needle in heads.items():
            with self.subTest(kind=kind):
                self.assertIn(needle, self.source)
        self.assertIn('prefix = "invalid settings" if exc.kind == "invalid" else "skipped"', self.source)
        self.assertIn('" | ".join(f"{name}: {value}" for name, value in state.statuses.items())', self.source)
        self.assertEqual(tuple(sp.PASS_LABELS), ("base", "img2img", "hires"))
        for label in sp.PASS_LABELS:
            self.assertIn(f'return "{label}"', self.source)

    def test_every_skip_hint_matches_a_reason_the_extension_writes(self):
        for needles, key, _hint in sn._SPEED_SKIP_HINTS:      # 문구마다 한 리터럴 안에 있다(줄 이음 밖)
            with self.subTest(key=key):
                self.assertTrue(any(needle in self.source for needle in needles), needles)

if __name__ == "__main__":
    unittest.main()
