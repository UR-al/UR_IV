"""Forge 옵션 덮어쓰기(P10)와 메인 생성 재시도 — 가짜 requests.post (D3, critic A1·A2·B10·B11).

- 기본('Forge 설정 따름')은 요청 JSON 이 P10 전과 바이트 단위로 같다.
- 스냅샷이 키를 확인한 것만 ``override_settings`` 로 보낸다(메인·보조 요청 모두).
- 앱이 덧붙인 부분(DoRA·Anima38 블록 422, 옵션 500 KeyError·설정 잠금)을 Forge 가 거절하면 그것만 빼고 다시 보낸다 —
  시도마다 새 task id, 재시도 전 취소 확인, 캐시 무효화 없이 워커에서 새로 받기. 앱이 넣은 제목 목록은 요청 JSON 에 없다.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import threading
import unittest
from unittest import mock

import requests
from PIL import Image

import core.forge_override_settings as fos
from backends.webui_backend import WebUIBackend
from core import alwayson_propagation as ap
from core import sam_extra_notices as sn
from core.sam_extra_capabilities import SamExtraCapabilities, _freeze
from tests.test_alwayson_propagation import A38, DORA, PAG, SKIM, DD
from tests.test_dora_infer_mode import LYCORIS_BLOCK
from tests.test_forge_override_settings import isolate_pushed_setting
from tests.test_sam_extra_notices import SAM3_OK, info_of

URL = "http://127.0.0.1:7860"
DEDUP, DAVE, KEEP_RESIDENT = fos.OPT_PREFIX_DEDUP, fos.OPT_DAVE_PRE_DD, fos.OPT_KEEP_RESIDENT
PLAIN_INFOTEXT = "1girl\nSteps: 28, Sampler: Euler, CFG scale: 4.5, Seed: 1, Size: 16x16, Model: anima"


def png_b64(size=(16, 16)):
    out = io.BytesIO()
    Image.new("RGB", size, "gray").save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class _Response:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.closed = False
        self.reason = "Internal Server Error" if status_code == 500 else "OK"

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} Server Error: {self.reason}", response=self)

    def close(self):
        self.closed = True


def ok(infotext=PLAIN_INFOTEXT):
    return _Response({"images": [png_b64()], "info": info_of(infotext)})


def key_error(key):
    return _Response({"error": "KeyError", "detail": "", "body": "", "errors": f"'{key}'"}, 500)


def frozen():
    return _Response({"error": "AssertionError", "detail": "", "body": "", "errors": "changing settings is disabled"},
                     500)


def script_missing(title):
    return _Response({"error": "HTTPException", "detail": f"Script '{title}' not found"}, 422)


SPARSE = fos.OPT_SPARSE_FORGE_GUESS
# Forge 가 옵션 일부만 잠갔을 때의 문구(modules/options.py — 섹션 잠금·키 잠금). 부분 LoRA 옵션은 'SAM Extra LoRA' 섹션이다.
PARTIAL_FREEZE_TEXTS = (
    f"not possible to set '{SPARSE}' because settings in section 'SAM Extra LoRA' (sam3_lora) are frozen with "
    "--freeze-settings-in-sections",
    f"not possible to set '{SPARSE}' because this setting is frozen with --freeze-specific-settings",
)


def partially_frozen_forge(text, infotext=PLAIN_INFOTEXT):
    """SPARSE 만 잠긴 Forge — 요청에 그 키가 있을 때만 잠금 500 으로 거절한다(다른 앱 키는 받아들인다)."""
    locked = _Response({"error": "AssertionError", "detail": "", "body": "", "errors": text}, 500)
    return lambda body: locked if SPARSE in (body.get("override_settings") or {}) else ok(infotext)


def caps(options=(DEDUP, KEEP_RESIDENT, DAVE), *, options_known=True):
    titles = {t.lower() for t in (PAG, SKIM, DD, A38, DORA)}
    scripts = {t: {"present": True, "img2img": True} for t in titles}
    return SamExtraCapabilities(status="ok", installed=True, scripts=_freeze(scripts),
                                options=_freeze({key: True for key in options}), options_known=options_known)


class _ForgeCase(unittest.TestCase):
    def setUp(self):
        isolate_pushed_setting(self)
        fos.set_forge_option_overrides({})
        self.backend = WebUIBackend(URL)
        self.caps = caps()
        patches = (
            mock.patch("core.sam_extra_probe.peek_capabilities", side_effect=lambda _url: self.caps),
            mock.patch("core.forge_output_policy.forge_save_outputs_setting", return_value=False),
            mock.patch.object(WebUIBackend, "_switch_model_if_needed"),
            mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        refresh = mock.patch("core.sam_extra_probe.get_capabilities", return_value=self.caps)
        self.refreshed = refresh.start()
        self.addCleanup(refresh.stop)
        invalidate = mock.patch("core.sam_extra_probe.invalidate_capabilities")
        self.invalidated = invalidate.start()
        self.addCleanup(invalidate.stop)
        self.sent, self.responses = [], []
        self.responder = None                  # 요청 → 응답(설정이 잠긴 Forge 처럼 내용으로 답할 때). 없으면 큐

    def post(self, url, json=None, **_kw):
        if url.endswith("/internal/progress"):
            return _Response({}, 404)          # 취소 감시 — 이 테스트는 작업 상태를 모른다(옛 방식 폴백)
        if url.endswith("/sdapi/v1/interrupt"):
            return _Response({})
        self.sent.append(copy.deepcopy(json))
        if self.responder is not None:
            return self.responder(json)
        return self.responses.pop(0)

    def run_main(self, payload, responses, *, method="txt2img", cancel_check=None):
        self.responses = list(responses)
        with mock.patch("backends.webui_backend.requests.post", side_effect=self.post):
            return getattr(self.backend, method)("anima", payload, cancel_check=cancel_check)

    def run_aux(self, method, settings, responses, **kwargs):
        self.responses = list(responses)
        with mock.patch("backends.webui_backend.requests.post", side_effect=self.post):
            return getattr(self.backend, method)(png_b64(), settings, **kwargs)

    @staticmethod
    def without_task(sent):
        return {key: value for key, value in sent.items() if key != "force_task_id"}

    @staticmethod
    def notice_codes(result):
        return [n.code for n in sn.notices_from_info(result.info)]


class MainGenerationTests(_ForgeCase):
    def payload(self, **scripts):
        return {"prompt": "1girl", "steps": 28, "save_images": True,
                "alwayson_scripts": {PAG: {"args": [True, 4.0]}, **scripts}}

    def test_default_request_is_byte_identical_to_before_p10(self):
        """D3 — 'Forge 설정 따름'(빈 설정)이면 요청은 payload + 저장 정책 + task id 뿐이다(P10 전과 같다).
        스냅샷을 모르는 채로 값을 정해 둔 경우도 요청은 같다(알림만)."""
        payload = self.payload(**{DORA: LYCORIS_BLOCK})
        expected = json.dumps({**payload, "save_images": False}, ensure_ascii=False)
        for overrides, snapshot in (({}, caps()), ({}, None), ({DEDUP: False}, None)):
            with self.subTest(overrides=overrides, snapshot=snapshot is not None):
                fos.set_forge_option_overrides(overrides)
                self.caps = snapshot
                before = copy.deepcopy(payload)
                result = self.run_main(payload, [ok()])
                self.assertTrue(result.success, result.error)
                self.assertEqual(len(self.sent), 1)
                self.assertEqual(json.dumps(self.without_task(self.sent[0]), ensure_ascii=False), expected)
                self.assertEqual(list(self.sent[0])[-1], "force_task_id")
                self.assertEqual(payload, before)
                self.sent.clear()
        self.refreshed.assert_not_called()

    def test_a_setting_that_can_not_be_verified_only_adds_a_notice(self):
        fos.set_forge_option_overrides({DEDUP: False})
        self.caps = None
        result = self.run_main(self.payload(), [ok()])
        self.assertNotIn("override_settings", self.sent[0])
        self.assertEqual(self.notice_codes(result), [sn.CODE_FORGE_OPTION_UNVERIFIED])
        self.assertIn("확인이 아직", sn.notices_from_info(result.info)[0].message)

    def test_confirmed_keys_are_sent_per_request_and_the_caller_wins(self):
        fos.set_forge_option_overrides({DEDUP: False, DAVE: False, fos.OPT_SEG_SEPARABLE: False})
        payload = {**self.payload(), "override_settings": {"sd_vae": "v.safetensors", DAVE: True}}
        result = self.run_main(payload, [ok()])
        self.assertTrue(result.success, result.error)
        sent = self.sent[0]
        self.assertEqual(sent["override_settings"], {"sd_vae": "v.safetensors", DAVE: True, DEDUP: False})
        self.assertNotIn("override_settings_restore_afterwards", sent)
        self.assertEqual(payload["override_settings"], {"sd_vae": "v.safetensors", DAVE: True})
        self.assertEqual(self.notice_codes(result), [sn.CODE_FORGE_OPTION_MISSING])   # SEG 분리 블러는 이 Forge 에 없다

    def test_unread_config_sends_nothing(self):
        fos.set_forge_option_overrides({DEDUP: False})
        self.caps = caps(options_known=False)
        result = self.run_main(self.payload(), [ok()])
        self.assertNotIn("override_settings", self.sent[0])
        self.assertEqual(self.notice_codes(result), [sn.CODE_FORGE_OPTION_UNVERIFIED])

    def test_key_error_retries_once_without_app_keys_and_a_new_task_id(self):
        fos.set_forge_option_overrides({DEDUP: False, DAVE: False})
        result = self.run_main(self.payload(), [key_error(DAVE), ok()])
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.sent[0]["override_settings"], {DEDUP: False, DAVE: False})
        self.assertNotIn("override_settings", self.sent[1])
        self.assertNotEqual(self.sent[0]["force_task_id"], self.sent[1]["force_task_id"])
        self.refreshed.assert_called_once_with(URL, refresh=True)     # 워커에서 새로 받는다(A1)
        self.invalidated.assert_not_called()                           # 캐시를 버리지 않는다(A1)
        codes = self.notice_codes(result)
        self.assertEqual(codes, [sn.CODE_FORGE_OPTION_REJECTED])
        rejected = sn.notices_from_info(result.info)[0]
        self.assertIn("다시 생성했습니다", rejected.message)
        self.assertIn(fos.SPEC_BY_KEY[DEDUP].label, rejected.message)

    def test_frozen_settings_also_retry(self):
        fos.set_forge_option_overrides({KEEP_RESIDENT: False})
        result = self.run_main(self.payload(), [frozen(), ok()])
        self.assertTrue(result.success, result.error)
        self.assertNotIn("override_settings", self.sent[1])
        self.assertIn("--freeze-settings", sn.notices_from_info(result.info)[0].message)

    def test_frozen_forge_is_remembered_so_the_next_generation_sends_no_app_keys(self):
        """(P10 검토 3) --freeze-settings 는 Forge 시작 인자라 새로 받은 스냅샷에도 옵션이 '있다'로 남는다. 잠금 거절을
        주소별로 기억하지 않으면 생성마다 거절 → (블로킹) 새로고침 → 다시 보내기를 되풀이한다(배치 100장 = POST 200·
        새로고침 100). 기억한 뒤로는 앱 키 없이 한 번에 보내고 '보내지 않음' 알림을 싣는다."""
        fos.set_forge_option_overrides({DEDUP: False})
        self.responder = lambda body: frozen() if "override_settings" in body else ok()   # 잠긴 Forge
        first = self.run_main(self.payload(), [])
        second = self.run_main(self.payload(), [])
        self.assertEqual(len(self.sent), 3)                              # 예전: 생성마다 2(= 4)
        self.assertTrue(first.success and second.success, (first.error, second.error))
        self.assertEqual(self.sent[0]["override_settings"], {DEDUP: False})
        self.assertNotIn("override_settings", self.sent[1])
        self.assertNotIn("override_settings", self.sent[2])
        self.refreshed.assert_not_called()      # 잠금은 스냅샷에 드러나지 않는다 — 새로 받아도 소용없다
        self.assertEqual(self.notice_codes(first), [sn.CODE_FORGE_OPTION_REJECTED])
        self.assertEqual(self.notice_codes(second), [sn.CODE_FORGE_OPTION_FROZEN])
        self.assertEqual(fos.frozen_options(URL), frozenset({DEDUP}))

    def test_a_partial_freeze_remembers_only_the_key_forge_named(self):
        """(P10 검토 4) --freeze-settings-in-sections·--freeze-specific-settings 는 확장 옵션 일부만 잠근다(섹션이 여럿).
        재시도는 앱 키를 한 번에 모두 빼지만 기억은 Forge 가 이름을 댄 키만 — 잠기지 않은 설정은 다음 생성부터 다시 보낸다.
        예전: 보낸 앱 키를 모두 잠겼다고 기억해 dedup 도 재연결 전까지 보내지 않고 '--freeze-settings 로 잠겼다'고 알렸다."""
        sparse, dedup = fos.SPEC_BY_KEY[SPARSE].label, fos.SPEC_BY_KEY[DEDUP].label
        for text in PARTIAL_FREEZE_TEXTS:
            with self.subTest(text=text):
                fos.forget_frozen_options()
                self.sent.clear()
                self.caps = caps(options=(DEDUP, KEEP_RESIDENT, DAVE, SPARSE))
                fos.set_forge_option_overrides({SPARSE: True, DEDUP: False})
                self.responder = partially_frozen_forge(text)
                first = self.run_main(self.payload(), [])
                self.assertEqual(len(self.sent), 2)
                second = self.run_main(self.payload(), [])
                self.assertEqual(len(self.sent), 3)                          # 기억한 뒤로는 한 번에
                self.assertTrue(first.success and second.success, (first.error, second.error))
                self.assertEqual(self.sent[0]["override_settings"], {DEDUP: False, SPARSE: True})
                self.assertNotIn("override_settings", self.sent[1])        # 재시도 한 번 — 앱 키를 모두 뺀다
                self.assertEqual(self.sent[2]["override_settings"], {DEDUP: False})   # 잠기지 않은 설정은 다시
                self.assertEqual(fos.frozen_options(URL), frozenset({SPARSE}))
                self.refreshed.assert_not_called()
                (rejected,) = sn.notices_from_info(first.info)
                self.assertEqual((rejected.code, rejected.detail), (sn.CODE_FORGE_OPTION_REJECTED, f"frozen:'{sparse}'@"))
                self.assertIn(f"설정 '{sparse}' 을(를) 거절해", rejected.message)
                self.assertIn(f"'{dedup}' 도 함께 뺐습니다", rejected.message)       # 이번 한 번만 — 다음부터 다시
                (held,) = sn.notices_from_info(second.info)
                self.assertEqual((held.code, held.detail), (sn.CODE_FORGE_OPTION_FROZEN, f"'{sparse}'"))
                self.assertNotIn(dedup, held.message)
                self.assertIn("--freeze-settings…", held.message)                  # 어느 잠금 인자인지 단정하지 않는다
                self.assertNotIn("(--freeze-settings)", held.message)

    def test_two_keys_in_one_frozen_section_are_learned_one_generation_each(self):
        """Forge 는 override 를 차례로 적용하다 처음 잠긴 키에서 멈춘다(modules/sysinfo.set_config) — 같은 섹션의 두 키가
        잠겼으면 한 번에 하나씩 이름이 나온다. 생성마다 시도는 2번을 넘지 않고, 둘 다 배운 뒤로는 한 번에 보낸다."""
        locked_keys = (DEDUP, DAVE)

        def forge(body):
            for key in body.get("override_settings") or {}:
                if key in locked_keys:
                    return _Response({"error": "AssertionError", "errors": (
                        f"not possible to set '{key}' because settings in section 'Anima Guidance' (anima_safe_pag) "
                        "are frozen with --freeze-settings-in-sections")}, 500)
            return ok()

        fos.set_forge_option_overrides({DEDUP: False, DAVE: False})
        self.responder = forge
        counts = []
        for _ in range(3):
            before = len(self.sent)
            self.assertTrue(self.run_main(self.payload(), []).success)
            counts.append(len(self.sent) - before)
        self.assertEqual(counts, [2, 2, 1])
        self.assertEqual(fos.frozen_options(URL), frozenset(locked_keys))
        self.assertNotIn("override_settings", self.sent[-1])

    def test_a_partial_freeze_seen_by_an_aux_pass_is_remembered_the_same_way(self):
        """단독 보조 패스(``_run_img2img_postprocess``)도 같은 기억을 쓴다 — 이름을 댄 키만."""
        self.caps = caps(options=(DEDUP, KEEP_RESIDENT, DAVE, SPARSE))
        fos.set_forge_option_overrides({SPARSE: True, DEDUP: False})
        self.responder = partially_frozen_forge(PARTIAL_FREEZE_TEXTS[0], SAM3_OK)
        self.run_aux("refine", {"target": "face", "sam3_prompt": "face"}, [], cancel_check=lambda: False)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(fos.frozen_options(URL), frozenset({SPARSE}))
        self.run_aux("refine", {"target": "face", "sam3_prompt": "face"}, [], cancel_check=lambda: False)
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(self.sent[-1]["override_settings"], {DEDUP: False})

    def test_the_frozen_verdict_covers_aux_passes_and_stays_with_its_address(self):
        fos.set_forge_option_overrides({DEDUP: False})
        self.responder = lambda body: frozen() if "override_settings" in body else ok(SAM3_OK)   # 잠긴 Forge
        self.run_main(self.payload(), [])
        result = self.run_aux("sam3", {"target": "face", "sam3_prompt": "face"}, [])
        self.assertEqual(len(self.sent), 3)
        self.assertNotIn("override_settings", self.sent[-1])             # 보조 요청도 보내지 않는다
        self.assertIn(sn.CODE_FORGE_OPTION_FROZEN, [n.code for n in sn.notices_of(result)])
        self.responder = None
        self.backend = WebUIBackend("http://127.0.0.1:7861")             # 다른 Forge 는 모른다 — 보낸다
        self.run_main(self.payload(), [ok()])
        self.assertEqual(self.sent[-1]["override_settings"], {DEDUP: False})
        fos.forget_frozen_options(URL)                                   # 재연결·수동 새로고침
        self.backend = WebUIBackend(URL)
        self.run_main(self.payload(), [ok()])
        self.assertEqual(self.sent[-1]["override_settings"], {DEDUP: False})

    def test_a_key_error_is_not_remembered(self):
        """KeyError 는 낡은 스냅샷 탓 — 새로 받은 스냅샷이 게이트하므로 기억하지 않는다(다음 요청은 스냅샷대로)."""
        fos.set_forge_option_overrides({DEDUP: False})
        self.run_main(self.payload(), [key_error(DEDUP), ok()])
        self.run_main(self.payload(), [ok()])
        self.assertEqual(self.sent[-1]["override_settings"], {DEDUP: False})
        self.assertEqual(fos.frozen_options(URL), frozenset())

    def test_a_second_rejection_fails_like_before(self):
        fos.set_forge_option_overrides({DEDUP: False})
        result = self.run_main(self.payload(), [key_error(DEDUP), key_error(DEDUP)])
        self.assertFalse(result.success)
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(result.error.startswith("API 요청 실패: 500"), result.error)

    def test_other_500_is_not_retried(self):
        fos.set_forge_option_overrides({DEDUP: False})
        oom = _Response({"error": "OutOfMemoryError", "errors": "CUDA out of memory"}, 500)
        result = self.run_main(self.payload(), [oom])
        self.assertFalse(result.success)
        self.assertEqual(len(self.sent), 1)
        self.assertTrue(result.error.startswith("API 요청 실패: 500"), result.error)
        self.refreshed.assert_not_called()

    def test_422_on_an_app_block_retries_without_it(self):
        """A2 — 앱이 넣은 블록(앱 기본값 DoRA — 체인이 비공개 출처로 적는다)을 낡은 스냅샷 탓에 Forge 가 모르면 그 블록만
        빼고 생성한다(정보 알림)."""
        payload = {**self.payload(**{DORA: LYCORIS_BLOCK}), ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        result = self.run_main(payload, [script_missing(DORA), ok()])
        self.assertTrue(result.success, result.error)
        self.assertEqual([list(p["alwayson_scripts"]) for p in self.sent], [[PAG, DORA], [PAG]])
        self.assertNotEqual(self.sent[0]["force_task_id"], self.sent[1]["force_task_id"])
        self.assertEqual(self.notice_codes(result), [sn.CODE_APP_BLOCK_RETRIED])
        self.assertEqual(sn.notices_from_info(result.info)[0].level, sn.LEVEL_INFO)
        self.refreshed.assert_called_once_with(URL, refresh=True)
        self.invalidated.assert_not_called()
        self.assertIn(DORA, payload["alwayson_scripts"])               # 호출자 payload 불변

    def test_422_on_a_user_block_fails_with_the_explanation(self):
        result = self.run_main(self.payload(**{DORA: LYCORIS_BLOCK}), [script_missing(PAG)])
        self.assertFalse(result.success)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("HTTP 422", result.error)
        self.assertIn(PAG, result.error)
        self.refreshed.assert_not_called()

    def test_422_and_500_together_end_within_three_attempts(self):
        fos.set_forge_option_overrides({DEDUP: False})
        app_dora = {**self.payload(**{DORA: LYCORIS_BLOCK}), ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        result = self.run_main(app_dora, [script_missing(DORA), key_error(DEDUP), ok()])
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(self.sent), 3)
        self.assertNotIn(DORA, self.sent[2]["alwayson_scripts"])
        self.assertNotIn("override_settings", self.sent[2])
        self.assertEqual(self.notice_codes(result), [sn.CODE_APP_BLOCK_RETRIED, sn.CODE_FORGE_OPTION_REJECTED])
        self.assertEqual(len({p["force_task_id"] for p in self.sent}), 3)
        # 부분이 다 빠진 뒤의 거절은 다시 보내지 않는다 — 최대 3회(1 + 제목 1 + 옵션 1)
        result = self.run_main(self.payload(**{DORA: LYCORIS_BLOCK}),
                               [script_missing(DORA), key_error(DEDUP), script_missing(PAG)])
        self.assertFalse(result.success)
        self.assertEqual(len(self.sent), 6)
        self.assertIn("HTTP 422", result.error)

    def test_cancel_before_the_retry_sends_nothing_more(self):
        """B10 — 첫 시도가 거절된 뒤 취소됐으면 다시 보내지 않는다."""
        fos.set_forge_option_overrides({DEDUP: False})
        cancelled = lambda: len(self.sent) >= 1   # noqa: E731 — 첫 요청을 보낸 뒤부터 취소
        result = self.run_main(self.payload(**{DORA: LYCORIS_BLOCK}), [script_missing(DORA), ok()],
                               cancel_check=cancelled)
        self.assertFalse(result.success)
        self.assertEqual(len(self.sent), 1)
        self.assertIn("취소", result.error)
        self.refreshed.assert_not_called()

    def test_cancel_during_the_capability_refresh_sends_nothing_more(self):
        """(P10 검토 1) 새로고침 중에 취소돼도 메인 경로는 ``_generate_once`` 의 잠금 안 확인이 막는다(보조 경로와 같은 결과)."""
        fos.set_forge_option_overrides({DEDUP: False})
        stop = threading.Event()

        def refresh(*_args, **_kwargs):
            stop.set()
            return self.caps

        self.refreshed.side_effect = refresh
        result = self.run_main(self.payload(), [key_error(DEDUP), ok()], cancel_check=stop.is_set)
        self.assertFalse(result.success)
        self.assertIn("취소", result.error)
        self.assertEqual(len(self.sent), 1)

    def test_app_titles_are_never_marked_in_the_request_json(self):
        """A2 — 블록 출처는 비공개 키로 백엔드까지 오고 백엔드가 떼어 쓴다. 요청에는 원래 키(비공개 키 뺀)와 task id·저장
        정책만 더해진다. 호출자 payload 는 그대로다."""
        payload = {**self.payload(**{DORA: LYCORIS_BLOCK}), ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        before = copy.deepcopy(payload)
        self.run_main(payload, [script_missing(DORA), ok()])
        for sent in self.sent:
            self.assertLessEqual(set(sent) - set(payload), {"force_task_id"})
            self.assertNotIn("_", "".join(key[0] for key in sent))     # 비공개('_') 키 없음
        self.assertEqual(payload, before)

    def test_the_private_provenance_key_leaves_the_request_byte_identical(self):
        """(P10 검토 2) 앱 기본값 블록이 있을 때만 붙는 비공개 키 — 떼고 나면 요청은 키가 없던 때와 바이트 단위로 같다."""
        plain = self.payload(**{DORA: LYCORIS_BLOCK})
        self.run_main(plain, [ok()])
        marked = {**plain, ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        self.run_main(marked, [ok()], method="txt2img")
        self.assertEqual(json.dumps(self.without_task(self.sent[0])), json.dumps(self.without_task(self.sent[1])))
        self.assertNotIn(ap.PROVENANCE_KEY, self.sent[1])

    def test_422_on_a_user_block_the_app_retries_is_a_warning_with_neutral_wording(self):
        """(P10 검토 2, A2·A6) 스냅샷을 모를 때 게이트를 지나는 Anima38 은 사용자가 켠 블록뿐이다(앱 기본값은 SKIP).
        출처가 사용자 값이거나 없으면(= 사용자 값으로 본다) 빼고 다시 생성하되 '앱이 넣은'이라 하지 않고 경고로 알린다."""
        self.caps = None
        for provenance in (None, {A38: ap.PROVENANCE_USER}, {DORA: ap.PROVENANCE_APP_DEFAULT}):
            with self.subTest(provenance=provenance):
                self.sent.clear()
                payload = {"prompt": "x", "alwayson_scripts": {PAG: {"args": [True]}, A38: {"args": [True]}}}
                if provenance is not None:
                    payload[ap.PROVENANCE_KEY] = provenance
                result = self.run_main(payload, [script_missing(A38), ok()])
                self.assertTrue(result.success, result.error)
                self.assertEqual([list(p["alwayson_scripts"]) for p in self.sent], [[PAG, A38], [PAG]])
                (notice,) = sn.notices_from_info(result.info)
                self.assertEqual((notice.code, notice.level), (sn.CODE_BLOCK_RETRIED, sn.LEVEL_WARNING))
                self.assertNotIn("앱이 넣은", notice.message)
                self.assertIn("Anima 3.8B", notice.message)
                self.assertIn("HTTP 422", notice.message)

    def test_a_chain_built_forge_payload_brings_its_app_default_provenance(self):
        """(P10 검토 2) 끝에서 끝으로: 메인 체인이 적은 앱 기본값 출처가 백엔드까지 와서 422 재시도를 정보로 알린다.
        Forge 요청에는 그 키가 없다."""
        from types import SimpleNamespace

        from backends import BackendType
        from tests.test_sampling_blocks import ALL_PRESENT, ChainHost
        from ui import sampling_blocks as sb

        def app_default_anima38(_host, _ctx):
            return sb.Contribution().add(A38, {"args": [{"negative": True}]}, provenance=ap.PROVENANCE_APP_DEFAULT)

        with mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
                mock.patch("backends.get_backend", return_value=SimpleNamespace(api_url=URL)), \
                mock.patch.object(sb, "CONTRIBUTORS", (*sb.CONTRIBUTORS, app_default_anima38)):
            payload, error = ChainHost(capabilities=ALL_PRESENT)._build_generation_payload(snapshot=True)
        self.assertIsNone(error)
        self.assertEqual(payload[ap.PROVENANCE_KEY], {A38: ap.PROVENANCE_APP_DEFAULT})
        result = self.run_main(payload, [script_missing(A38), ok()])
        self.assertTrue(result.success, result.error)
        retried = {sn.CODE_APP_BLOCK_RETRIED, sn.CODE_BLOCK_RETRIED}       # (PAG 누락 결과 알림은 이 테스트 밖)
        self.assertEqual([(n.code, n.level) for n in sn.notices_from_info(result.info) if n.code in retried],
                         [(sn.CODE_APP_BLOCK_RETRIED, sn.LEVEL_INFO)])
        self.assertFalse(any(ap.PROVENANCE_KEY in sent for sent in self.sent))

    def test_each_rejected_block_is_announced_by_its_own_provenance(self):
        """앱 기본값 DoRA 와 사용자 Anima38 이 차례로 거절되면 — DoRA 는 정보, Anima38 은 경고."""
        payload = {**self.payload(**{A38: {"args": [True]}, DORA: LYCORIS_BLOCK}),
                   ap.PROVENANCE_KEY: {DORA: ap.PROVENANCE_APP_DEFAULT}}
        result = self.run_main(payload, [script_missing(DORA), script_missing(A38), ok()])
        self.assertTrue(result.success, result.error)
        self.assertEqual([(n.code, n.level, n.detail) for n in sn.notices_from_info(result.info)],
                         [(sn.CODE_APP_BLOCK_RETRIED, sn.LEVEL_INFO, DORA),
                          (sn.CODE_BLOCK_RETRIED, sn.LEVEL_WARNING, A38)])

    def test_img2img_gets_the_same_policy(self):
        """손 재구성·I2I·인페인트는 img2img 로 온다 — 허용 목록이 호출자 override 를 빼도 앱 키는 백엔드가 붙인다."""
        fos.set_forge_option_overrides({DEDUP: False})
        payload = {"prompt": "hand", "init_images": [png_b64()], "denoising_strength": 0.4}
        result = self.run_main(payload, [ok()], method="img2img")
        self.assertTrue(result.success, result.error)
        self.assertEqual(self.sent[0]["override_settings"], {DEDUP: False})


class AuxPassTests(_ForgeCase):
    SETTINGS = {"target": "face", "sam3_prompt": "face", "seed": 7}

    def test_default_aux_requests_are_unchanged(self):
        for method in ("refine", "sam3", "adetailer"):
            with self.subTest(method=method):
                fos.set_forge_option_overrides({})
                self.run_aux(method, dict(self.SETTINGS), [ok(SAM3_OK)])
                before = json.dumps(self.sent[-1])
                fos.set_forge_option_overrides({DEDUP: False})
                self.caps = None                                        # 모르면 보내지 않는다
                self.run_aux(method, dict(self.SETTINGS), [ok(SAM3_OK)])
                self.assertEqual(json.dumps(self.sent[-1]), before)
                self.assertNotIn("override_settings", self.sent[-1])
                self.caps = caps()

    def test_aux_requests_carry_confirmed_keys(self):
        fos.set_forge_option_overrides({DEDUP: False, KEEP_RESIDENT: False})
        for method in ("refine", "sam3", "adetailer"):
            with self.subTest(method=method):
                result = self.run_aux(method, dict(self.SETTINGS), [ok(SAM3_OK)])
                self.assertEqual(self.sent[-1]["override_settings"], {KEEP_RESIDENT: False, DEDUP: False})
                self.assertFalse({n.code for n in sn.notices_of(result)} & {
                    sn.CODE_FORGE_OPTION_MISSING, sn.CODE_FORGE_OPTION_REJECTED})

    def test_aux_key_error_retries_once_and_says_which_job(self):
        fos.set_forge_option_overrides({DEDUP: False})
        result = self.run_aux("refine", dict(self.SETTINGS), [key_error(DEDUP), ok(SAM3_OK)],
                              cancel_check=lambda: False)
        self.assertEqual(len(self.sent), 2)
        self.assertNotIn("override_settings", self.sent[1])
        notices = [n for n in sn.notices_of(result) if n.code == sn.CODE_FORGE_OPTION_REJECTED]
        self.assertEqual(len(notices), 1)
        self.assertIn("Refine 을(를) 다시 실행", notices[0].message)
        self.refreshed.assert_called_once_with(URL, refresh=True)
        self.invalidated.assert_not_called()

    def test_aux_cancel_before_retry_fails_with_the_original_error(self):
        fos.set_forge_option_overrides({DEDUP: False})
        with self.assertRaises(requests.exceptions.HTTPError):
            self.run_aux("sam3", dict(self.SETTINGS), [key_error(DEDUP), ok(SAM3_OK)], cancel_check=lambda: True)
        self.assertEqual(len(self.sent), 1)
        self.refreshed.assert_not_called()

    def test_aux_cancel_during_the_capability_refresh_sends_nothing_more(self):
        """(P10 검토 1, B10) 재시도 전 확인을 지난 뒤 스냅샷 새로고침(블로킹 GET 여러 개) 중에 취소되면 다시 보내지 않고
        원래 거절로 실패한다 — 단독 경로는 Forge 를 끊지 않으므로 다시 보내면 중지 뒤에도 인페인트가 끝까지 돈다."""
        fos.set_forge_option_overrides({DEDUP: False})
        stop = threading.Event()

        def refresh(*_args, **_kwargs):
            stop.set()                                   # 사용자가 새로고침 도중 중지를 눌렀다
            return self.caps

        self.refreshed.side_effect = refresh
        for method in ("refine", "sam3", "adetailer"):
            with self.subTest(method=method):
                stop.clear()
                self.sent.clear()
                with self.assertRaises(requests.exceptions.HTTPError):
                    self.run_aux(method, dict(self.SETTINGS), [key_error(DEDUP), ok(SAM3_OK)],
                                 cancel_check=stop.is_set)
                self.assertEqual(len(self.sent), 1)
                self.assertIn("override_settings", self.sent[0])


if __name__ == "__main__":
    unittest.main()
