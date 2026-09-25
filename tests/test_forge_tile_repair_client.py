"""sam-extra Tile & Repair 라우트 클라이언트(core/forge_tile_repair_client.py) — 가짜 HTTP 로 계약을 본다.

확장 라우트(sam3ext/tile_repair_api.py)의 모양: 같은 출처 헤더 X-SAM3-Notebook: 1, 404 = 옛 확장,
400/413/422/503 은 사용자에게 보일 문구로. 결과 PNG 는 base64 로 오고 infotext 가 함께 온다. 네트워크 없음.
"""
from __future__ import annotations

import base64
import io
import unittest

from PIL import Image

from core import forge_tile_repair_client as mod
from core.forge_tile_repair_client import (
    ForgeTileRepairClient, TileRepairError, TileRepairUnavailable,
)
from core.sam_extra_capabilities import (
    EP_TILE_REPAIR, FEATURE_FLAGS, HttpResult, SamExtraCapabilities, build_capabilities,
)

BASE = "http://127.0.0.1:7860"


def png_bytes(size=(64, 96)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


class Response:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeRequest:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def client(*responses):
    fake = FakeRequest(*responses)
    return ForgeTileRepairClient(BASE + "/", request=fake), fake


class ClientTests(unittest.TestCase):
    def test_empty_base_url_is_refused(self):
        with self.assertRaises(ValueError):
            ForgeTileRepairClient("  ")

    def test_run_posts_the_body_with_the_same_origin_header_and_a_long_timeout(self):
        png = png_bytes()
        c, fake = client(Response(200, {
            "version": 1, "interrupted": False, "image": base64.b64encode(png).decode(), "info": "p\nSeed: 7",
            "seed": 7, "width": 64, "height": 96, "model": "animaTileRepair_v20.safetensors",
        }))
        result = c.run({"image": "AAAA", "steps": 50})
        method, url, kwargs = fake.calls[0]
        self.assertEqual((method, url), ("POST", BASE + "/sam-extra/tile-repair"))
        self.assertEqual(kwargs["headers"]["X-SAM3-Notebook"], "1")
        self.assertEqual(kwargs["json"], {"image": "AAAA", "steps": 50})
        # 루프백이면 (connect 짧게, read 길게) — 첫 실행은 모델 로딩 + 50 스텝, 큐 대기까지 기다린다
        self.assertEqual(kwargs["timeout"][1], mod.RUN_TIMEOUT)
        self.assertEqual((result.png, result.info, result.seed, result.width, result.height, result.model),
                         (png, "p\nSeed: 7", 7, 64, 96, "animaTileRepair_v20.safetensors"))
        self.assertFalse(result.interrupted)

    def test_interrupted_run_is_an_empty_result(self):
        c, _ = client(Response(200, {"version": 1, "interrupted": True, "image": None, "model": "m"}))
        result = c.run({})
        self.assertTrue(result.interrupted)
        self.assertEqual((result.png, result.model), (b"", "m"))

    def test_bad_result_image_is_an_error(self):
        for image in ("not base64 !", base64.b64encode(b"GIF89a").decode(), None):
            with self.subTest(image=image):
                c, _ = client(Response(200, {"interrupted": False, "image": image}))
                with self.assertRaises(TileRepairError):
                    c.run({})

    def test_status_codes_become_user_messages(self):
        cases = {
            404: (TileRepairUnavailable, "업데이트"),
            405: (TileRepairUnavailable, "업데이트"),
            401: (TileRepairError, "--gradio-auth"),
            403: (TileRepairError, "거부"),
            413: (TileRepairError, "64 MB"),
            400: (TileRepairError, "multiplier must be between"),
            422: (TileRepairError, "Text Encoder"),
            503: (TileRepairError, "벤더"),
            500: (TileRepairError, "HTTP 500"),
        }
        details = {400: "multiplier must be between -10.0 and 10.0 (got 11.0)",
                   422: "Anima는 Qwen3 Text Encoder가 필수입니다.", 500: "KeyError: 'x'"}
        for status, (kind, text) in cases.items():
            with self.subTest(status=status):
                c, _ = client(Response(status, {"detail": details.get(status, "")}))
                with self.assertRaises(kind) as raised:
                    c.run({})
                self.assertIn(text, str(raised.exception))
                self.assertNotIn(BASE, str(raised.exception))
                if status in (401, 403):   # 401 은 --gradio-auth 로그인과 --api·--nowebui 의 --api-auth 둘 다에서 온다
                    for flag in ("--gradio-auth", "--api-auth"):
                        self.assertIn(flag, str(raised.exception))

    def test_detail_is_truncated(self):
        c, _ = client(Response(400, {"detail": "x" * 1000}))
        with self.assertRaises(TileRepairError) as raised:
            c.run({})
        self.assertLess(len(str(raised.exception)), 300)

    def test_connection_failure_hides_the_url(self):
        c, _ = client(ConnectionError(f"refused {BASE}"))
        with self.assertRaises(TileRepairError) as raised:
            c.options()
        self.assertNotIn("127.0.0.1", str(raised.exception))
        self.assertIn("ConnectionError", str(raised.exception))

    def test_options_and_stop(self):
        options = {"version": 1, "available": True, "models": ["animaTileRepair_v20.safetensors"],
                   "default_model": "animaTileRepair_v20.safetensors", "dit": [], "text_encoder": [], "vae": [],
                   "defaults": {"steps": 50}, "ranges": {}, "increments": {}}
        c, fake = client(Response(200, options), Response(200, {"stopped": True}), Response(200, {"stopped": False}))
        self.assertEqual(c.options(), options)
        self.assertTrue(c.stop())
        self.assertFalse(c.stop())
        self.assertEqual([(m, u) for m, u, _ in fake.calls], [
            ("GET", BASE + "/sam-extra/tile-repair/options"),
            ("POST", BASE + "/sam-extra/tile-repair/stop"),
            ("POST", BASE + "/sam-extra/tile-repair/stop"),
        ])
        self.assertTrue(all(kw["headers"]["X-SAM3-Notebook"] == "1" for _m, _u, kw in fake.calls))
        self.assertNotIn("json", fake.calls[0][2])

    def test_malformed_options_are_an_error(self):
        for body in ({"models": "x", "defaults": {}}, [1], ValueError("no json")):
            with self.subTest(body=body):
                c, _ = client(Response(200, body))
                with self.assertRaises(TileRepairError):
                    c.options()


class CapabilityFlagTests(unittest.TestCase):
    """기능 스냅샷: POST 전용 라우트에 GET → 405 면 있음, 404 면 옛 확장."""

    def _caps(self, status) -> SamExtraCapabilities:
        responses = {
            "/sdapi/v1/scripts": HttpResult(200, {"txt2img": ["sam3 mask"], "img2img": []}),
            "/sdapi/v1/script-info": HttpResult(200, []),
            EP_TILE_REPAIR: HttpResult(status),
        }
        return build_capabilities(responses)

    def test_flag_follows_the_route(self):
        self.assertIn("tile_repair_route", FEATURE_FLAGS)
        self.assertEqual(EP_TILE_REPAIR, mod.TILE_REPAIR_API_PATH)
        self.assertTrue(self._caps(405).tile_repair_route)
        self.assertTrue(self._caps(405).may_use("tile_repair_route"))
        self.assertNotIn(EP_TILE_REPAIR, self._caps(405).errors)
        self.assertFalse(self._caps(404).tile_repair_route)
        self.assertFalse(self._caps(404).may_use("tile_repair_route"))
        self.assertTrue(self._caps(405).to_dict()["features"]["tile_repair_route"])

    def test_probe_asks_without_a_body(self):
        from core import sam_extra_probe as probe
        seen = []

        def fake_get(url, *, headers=None, timeout=None, want_body=True):
            seen.append((url[len(BASE):], want_body))
            return HttpResult(405 if url.endswith(EP_TILE_REPAIR) else 404)

        caps = probe.fetch_capabilities(BASE, http_get=fake_get)
        self.assertIn((EP_TILE_REPAIR, False), seen)
        self.assertEqual(caps.status, "error")   # 스크립트 목록이 없는 가짜 서버


if __name__ == "__main__":
    unittest.main()
