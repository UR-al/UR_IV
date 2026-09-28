"""단독 SAM3·Refine 의 원본 기준 요청(드리프트 수정) — core/sam3_args.with_init_source · WebUIBackend · 결과 알림.

denoise 0 부모 img2img 도 VAE 인코드·디코드로 픽셀을 조금씩 바꾼다. SAM3 가 그 출력을 검출·인페인트하면 결과의 마스크
밖까지 원본과 달라진다. 단독 SAM3·Refine 은 SAM3 state 에 'sam3_source_image': 'init' 을 실어 확장이 init 이미지를 쓰게
하고, 결과 infotext 'SAM3 Source' 로 확장이 따랐는지 본다(없음 = 모르는 예전 확장 → 정보, 'output (<이유>)' → 경고).
생성 안 SAM3(t2i·i2i)는 부모 출력이 곧 결과라 이 키를 보내지 않는다 — 그 요청은 바이트 단위로 예전과 같아야 한다.
단독 경로의 요청 크기는 Forge 가 EXIF 방향을 적용한 init 크기다(아니면 부모 패스가 찌그러뜨리고 요청이 'size' 로 폴백).
크기만 다른 폴백 경고는 한 종류로 묶여 배치에서 장마다 뜨지 않는다.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import unittest
from unittest import mock

from PIL import Image

from backends.webui_backend import WebUIBackend
from core import sam3_args
from core import sam_extra_notices as sn
from tests.test_sam_extra_notices import _HEAD, _SAM3_PARAMS, _TAIL, SAM3_OK, SAM3_OOM, _png_b64, _Response, info_of

KEY = sam3_args.SOURCE_STATE_KEY
URL = "http://127.0.0.1:7860"
SOURCE_CODES = {sn.CODE_SAM3_SOURCE_UNSUPPORTED, sn.CODE_SAM3_SOURCE_FALLBACK}


def sam3_infotext(source=None):
    """확장이 남기는 모양 그대로 — 'SAM3 Source' 는 요청을 받았을 때만 'SAM3 Version' 뒤에 붙는다."""
    tail = "" if source is None else f"SAM3 Source: {source}, "
    return _HEAD + "SAM3 Enable: True, " + _SAM3_PARAMS + "SAM3 Version: 0.30.0, " + tail + _TAIL


def sam3_state(payload):
    return payload["alwayson_scripts"][sam3_args.SCRIPT_SAM3]["args"][0]


def standalone_payload(**overrides):
    """단독 SAM3·Refine 이 보내는 모양(원본 기준 요청 포함)."""
    state = sam3_args.with_init_source(sam3_args.build_state(overrides))
    return {"init_images": ["x"], "prompt": "1girl", "denoising_strength": 0.0,
            "alwayson_scripts": {sam3_args.SCRIPT_SAM3: {"args": [state]}}}


def source_codes(notices):
    return [n.code for n in notices if n.code in SOURCE_CODES]


# ── core/sam3_args ────────────────────────────────────────────────────────────
class WithInitSourceTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_adds_the_request_to_a_copy(self):
        state = sam3_args.build_state({"sam3_prompt": "hand"})
        before = copy.deepcopy(state)
        out = sam3_args.with_init_source(state)
        self.assertEqual(out[KEY], "init")
        self.assertEqual(state, before, "입력 state 는 그대로 — 생성 안 경로와 공유되는 dict 일 수 있다")
        self.assertEqual({k: v for k, v in out.items() if k != KEY}, state)
        self.assertEqual(list(out)[-1], KEY, "끝에 붙는다 — 나머지 키 순서는 예전 그대로")
        self.assertEqual(sam3_args.with_init_source(None), {KEY: "init"})

    def test_in_generation_builders_never_carry_the_request(self):
        """생성 안 SAM3(t2i·i2i)는 부모 출력이 곧 결과다 — init 이미지로 돌리면 생성 결과가 버려진다."""
        self.assertNotIn(KEY, sam3_args.SAM3_KEYS)
        self.assertNotIn(KEY, sam3_args.default_settings())
        self.assertNotIn(KEY, sam3_args.build_state({KEY: "init"}), "설정에 섞여 들어와도 스펙 밖 키는 버린다")
        block = sam3_args.build_alwayson({"sam3_prompt": "face"}, prompt="1girl")[sam3_args.SCRIPT_SAM3]
        self.assertNotIn(KEY, block["args"][0])
        payload = sam3_args.apply_to_payload({"prompt": "1girl", "init_images": ["x"]}, {"sam3_prompt": "face"})
        self.assertNotIn(KEY, sam3_state(payload))
        self.assertEqual(set(sam3_state(payload)), {*sam3_args.SAM3_KEYS, "sam3_enable", "enabled"},
                         "생성 안 SAM3 state 의 키 집합은 예전과 같다")


# ── WebUIBackend 가 실제로 보내는 요청 ─────────────────────────────────────────
class BackendPayloadTests(unittest.TestCase):
    def setUp(self):
        self.backend = WebUIBackend(URL)
        for patcher in (
            mock.patch("core.sam_extra_probe.peek_capabilities", return_value=None),
            mock.patch("core.forge_modules.resolve_sam3_checkpoint", side_effect=lambda name: name),
            mock.patch("core.forge_output_policy.forge_save_outputs_setting", return_value=False),
            mock.patch.object(self.backend, "_switch_model_if_needed"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _post(self, call, infotext=SAM3_OK):
        """call() 이 보낸 요청 JSON 들(보낸 그대로 직렬화)과 결과."""
        sent: list = []

        def post(url, json=None, **_kw):
            sent.append((url, _json_text(json)))
            return _Response({"images": [_png_b64()], "info": info_of(infotext)})

        with mock.patch("backends.webui_backend.requests.post", side_effect=post):
            result = call()
        return sent, result

    def _standalone(self, method, settings, **kw):
        return self._post(lambda: getattr(self.backend, method)(_png_b64((16, 16)), copy.deepcopy(settings)), **kw)

    def test_refine_and_sam3_send_the_request(self):
        for method in ("refine", "sam3"):
            with self.subTest(method=method):
                sent, _ = self._standalone(method, {"target": "face", "sam3_prompt": "face"})
                self.assertEqual(len(sent), 1)
                url, body = sent[0]
                self.assertTrue(url.endswith("/sdapi/v1/img2img"))
                payload = json.loads(body)
                self.assertEqual(payload["denoising_strength"], 0.0)
                self.assertEqual(sam3_state(payload)[KEY], "init")

    def test_standalone_request_differs_from_before_only_by_the_key(self):
        """원본 기준 요청을 뺀 요청은 예전(with_init_source 없음)과 바이트 단위로 같다 — 다른 값은 건드리지 않는다."""
        settings = {"target": "shirt", "replacement": "red shirt", "sam3_prompt": "face", "steps": 12,
                    "cfg_scale": 5.0, "seed": 7, "sampler": "Euler", "scheduler": "Simple", "sam3_mode": "Mask only"}
        for method in ("refine", "sam3"):
            with self.subTest(method=method):
                new, _ = self._standalone(method, settings)
                with mock.patch("core.sam3_args.with_init_source", side_effect=lambda state: dict(state)):
                    old, _ = self._standalone(method, settings)
                new_payload = json.loads(new[0][1])
                self.assertEqual(sam3_state(new_payload).pop(KEY), "init")
                self.assertEqual(_json_text(new_payload), old[0][1])

    def test_in_generation_requests_are_byte_identical(self):
        """t2i·i2i 의 SAM3 는 원본 기준 요청을 보내지 않는다 — 요청 본문이 만든 payload 와 바이트 단위로 같다."""
        for endpoint, extra in (("txt2img", {}), ("img2img", {"init_images": [_png_b64()], "denoising_strength": 0.4})):
            with self.subTest(endpoint=endpoint):
                payload = {"prompt": "1girl", "negative_prompt": "bad", "steps": 20, "cfg_scale": 5.0, "seed": 1,
                           **extra, "save_images": False, "force_task_id": "task-fixed"}
                sam3_args.apply_to_payload(payload, {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
                expected = _json_text(payload)
                sent, result = self._post(lambda: getattr(self.backend, endpoint)("anima", copy.deepcopy(payload)))
                self.assertTrue(result.success, result.error)
                self.assertEqual(len(sent), 1)
                self.assertTrue(sent[0][0].endswith(f"/sdapi/v1/{endpoint}"))
                self.assertEqual(sent[0][1], expected)
                self.assertNotIn(KEY, sam3_state(json.loads(sent[0][1])))
                # 결과 infotext 에 'SAM3 Source' 가 없어도(요청하지 않았으니) 알림이 없다
                self.assertNotIn(sn.INFO_KEY, result.info)

    def test_size_follows_the_exif_orientation_forge_applies(self):
        """Forge 는 API init 이미지를 images.read → fix_image → ImageOps.exif_transpose 로 세운다. 요청 크기를 방향 적용
        전 크기로 보내면 부모 패스가 세운 이미지를 가로·세로 바뀐 크기로 늘려 전체가 찌그러지고, 원본 기준 요청도
        'size WxH != HxW' 로 폴백한다. 방향이 가로·세로를 바꾸지 않으면(1·3) 크기가 그대로다."""
        for orientation, expected in ((6, [48, 64]), (8, [48, 64]), (5, [48, 64]), (3, [64, 48]), (1, [64, 48])):
            image_b64 = _jpeg_b64((64, 48), orientation)
            for method in ("refine", "sam3"):
                with self.subTest(orientation=orientation, method=method):
                    sent, _ = self._post(lambda: getattr(self.backend, method)(image_b64, {"target": "face"}))
                    payload = json.loads(sent[0][1])
                    self.assertEqual([payload["width"], payload["height"]], expected)
                    self.assertEqual(payload["init_images"], [image_b64], "보내는 바이트는 그대로 — Forge 가 세운다")
            with self.subTest(orientation=orientation, method="adetailer payload"):
                payload = self.backend._build_postprocess_payload(image_b64, {}, prompt="", negative_prompt="")
                self.assertEqual([payload["width"], payload["height"]], expected)

    def test_standalone_result_notices(self):
        cases = (
            (SAM3_OK, [sn.CODE_SAM3_SOURCE_UNSUPPORTED]),                          # 요청을 모르는 확장
            (sam3_infotext("init image"), []),                                     # 원본으로 돌았다
            (sam3_infotext("output (size 1000x1000 != 1008x1000)"), [sn.CODE_SAM3_SOURCE_FALLBACK]),
            (SAM3_OOM, []),                                                        # 실패는 SAM3 실패 알림만
        )
        for method in ("refine", "sam3"):
            for infotext, expected in cases:
                with self.subTest(method=method, infotext=infotext[-80:]):
                    _, result = self._standalone(method, {"target": "face"}, infotext=infotext)
                    self.assertEqual(source_codes(sn.notices_of(result)), expected)


# ── 결과 알림 ─────────────────────────────────────────────────────────────────
class SourceNoticeTests(unittest.TestCase):
    def notices(self, infotext, payload=None):
        return sn.result_notices(info_of(infotext), standalone_payload() if payload is None else payload)

    def test_missing_key_means_the_extension_does_not_support_it(self):
        notices = self.notices(SAM3_OK)
        self.assertEqual([n.code for n in notices], [sn.CODE_SAM3_SOURCE_UNSUPPORTED])
        notice = notices[0]
        self.assertEqual(notice.level, sn.LEVEL_INFO, "결과는 예전과 같다 — 경고가 아니라 안내")
        self.assertEqual(notice.feature, "sam3")
        for text in ("원본 기준 SAM3", "마스크 밖", "sam-extra 를 업데이트", "Reload UI", "재시작"):
            self.assertIn(text, notice.message)
        self.assertIsNone(sn.standalone_sam3_failure(sn.NoticedImage("x", {}, notices)), "실패가 아니다 — 결과를 저장한다")

    def test_init_image_is_quiet(self):
        self.assertEqual(self.notices(sam3_infotext("init image")), [])

    def test_output_fallback_warns_with_the_reason(self):
        for value, reason, hint in (
            ("output (size 1000x1000 != 1008x1000)", "size 1000x1000 != 1008x1000", "크기"),
            ('"output (face restoration)"', "face restoration", "얼굴 복원"),     # 따옴표로 감싼 값도 읽는다
            ("output (denoising > 0)", "denoising > 0", None),
        ):
            with self.subTest(value=value):
                notices = self.notices(sam3_infotext(value))
                self.assertEqual([n.code for n in notices], [sn.CODE_SAM3_SOURCE_FALLBACK])
                notice = notices[0]
                self.assertEqual(notice.level, sn.LEVEL_WARNING)
                self.assertIn(f"(이유: {reason})", notice.message)
                self.assertIn("마스크 밖", notice.message)
                if hint:
                    self.assertIn(hint, notice.hint)
                else:
                    self.assertEqual(notice.hint, "")
                self.assertIsNone(sn.standalone_sam3_failure(sn.NoticedImage("x", {}, notices)))

    def test_only_when_the_request_was_sent_and_sam3_ran(self):
        # 생성 안 SAM3(요청 없음)는 'SAM3 Source' 가 없어도 조용하다
        plain = sam3_args.apply_to_payload({"prompt": "1girl"}, {"sam3_prompt": "face"})
        self.assertEqual(self.notices(SAM3_OK, plain), [])
        # 값이 'init' 이 아니면 요청이 아니다(확장과 같은 규칙) — ' INIT ' 은 요청
        other = standalone_payload()
        sam3_state(other)[KEY] = "output"
        self.assertEqual(self.notices(SAM3_OK, other), [])
        spaced = standalone_payload()
        sam3_state(spaced)[KEY] = " INIT "
        self.assertEqual(source_codes(self.notices(SAM3_OK, spaced)), [sn.CODE_SAM3_SOURCE_UNSUPPORTED])
        # SAM3 가 꺼진 요청
        off = standalone_payload()
        sam3_state(off).update(sam3_enable=False, enabled=False)
        self.assertEqual(self.notices(SAM3_OK, off), [])
        # 실패·미적용은 그 알림만 — 원본 기준 알림을 겹쳐 띄우지 않는다
        self.assertEqual([n.code for n in self.notices(SAM3_OOM)], [sn.CODE_SAM3_ERROR])
        not_applied = _HEAD + _TAIL
        self.assertEqual([n.code for n in self.notices(not_applied)], [sn.CODE_SAM3_NOT_APPLIED])

    def test_batch_does_not_spam(self):
        """배치 SAM3 100장이 같은 안내를 30초마다 다시 띄우지 않는다 — 생성 전 경고와 같은 억제 시간."""
        now = [0.0]
        throttle = sn.NoticeThrottle(clock=lambda: now[0])
        for infotext in (SAM3_OK, sam3_infotext("output (size 1000x1000 != 1008x1000)")):
            notice = self.notices(infotext)[0]
            with self.subTest(code=notice.code):
                ttl = sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S)
                self.assertEqual(ttl, sn.PRE_GENERATION_NOTICE_TTL_S)
                shown = 0
                for index in range(100):
                    now[0] = index * 5.0   # 장당 5초
                    shown += throttle.allow(self.notices(infotext)[0], ttl)
                self.assertEqual(shown, 1)

    def test_batch_of_different_sizes_does_not_spam(self):
        """크롭·스크린샷 폴더처럼 장마다 크기가 다르면 'size WxH != WxH' 원문도 장마다 다르다 — 그래도 한 번만 띄운다
        (Notice.key 가 원문을 쓰면 크기마다 억제 키가 새로 생겨 100장에 토스트 100번). 정확한 이유는 문구에 남는다."""
        now = [0.0]
        throttle = sn.NoticeThrottle(clock=lambda: now[0])
        shown, keys = 0, set()
        for index in range(100):
            now[0] = index * 5.0
            w, h = 1001 + 2 * index, 1403 + 2 * index
            reason = f"size {w}x{h} != {w // 8 * 8}x{h // 8 * 8}"
            notices = self.notices(sam3_infotext(f"output ({reason})"))
            self.assertEqual([n.code for n in notices], [sn.CODE_SAM3_SOURCE_FALLBACK])
            notice = notices[0]
            self.assertIn(f"(이유: {reason})", notice.message, "정확한 크기는 문구에 그대로")
            keys.add(notice.key)
            shown += throttle.allow(notice, sn.notice_ttl(notice, sn.RESULT_NOTICE_TTL_S))
        self.assertEqual(len(keys), 1, f"크기만 다른 폴백은 한 종류여야 한다: {sorted(keys)[:3]}")
        self.assertEqual(shown, 1)
        # 이유 종류가 다르면 따로 띄운다(얼굴 복원 경고가 크기 경고에 묻히지 않게)
        other = self.notices(sam3_infotext("output (face restoration)"))[0]
        self.assertNotEqual(other.key, next(iter(keys)))
        self.assertTrue(throttle.allow(other, sn.notice_ttl(other, sn.RESULT_NOTICE_TTL_S)))

    def test_one_result_with_several_sizes_warns_once(self):
        """한 결과에 크기만 다른 폴백이 여러 장이어도 알림은 한 건(첫 원문을 보인다)."""
        infotexts = [sam3_infotext(f"output (size {w}x1403 != 1000x1400)") for w in (1001, 1003, 1005)]
        notices = [n for n in sn.result_notices(info_of(*infotexts), standalone_payload()) if n.code in SOURCE_CODES]
        self.assertEqual([n.code for n in notices], [sn.CODE_SAM3_SOURCE_FALLBACK])
        self.assertIn("(이유: size 1001x1403 != 1000x1400)", notices[0].message)


def _json_text(payload) -> str:
    """requests.post(json=…) 가 보내는 본문처럼 직렬화(키 순서 유지)."""
    return json.dumps(payload, ensure_ascii=False)


def _jpeg_b64(size, orientation) -> str:
    """EXIF 방향(0x0112)이 붙은 JPEG — 카메라·휴대폰 사진처럼 저장된 픽셀은 눕혀 있고 방향 태그로 세운다."""
    image = Image.new("RGB", size, "white")
    exif = image.getexif()
    exif[0x0112] = orientation
    out = io.BytesIO()
    image.save(out, format="JPEG", exif=exif.tobytes())
    return base64.b64encode(out.getvalue()).decode("ascii")


if __name__ == "__main__":
    unittest.main()
