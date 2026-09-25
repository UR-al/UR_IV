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
        pre_dcwf = _capabilities(*_live_bodies())      # 픽스처 = DCW-F 전 소스(62개, arg30 max 0.5)
        self.assertEqual(pre_dcwf.anima_guidance_argc, 62)
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


if __name__ == "__main__":
    unittest.main()
