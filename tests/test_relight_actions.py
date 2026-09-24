"""Exercise the local relight action boundary without a window, GPU or downloads."""
import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image, PngImagePlugin

from ui.relight_actions import (RelightActionsMixin, decode_relight_image,
                                export_relight_png, render_relight_preview)


def image_url(size=(24, 16), *, mode="RGB", metadata=False):
    image = Image.new(mode, size, (90, 100, 120) if mode == "RGB" else 128)
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("parameters", "forest\nNegative prompt: blur\nSteps: 20")
        info.add_text("workflow", '{"nodes":[]}')
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", pnginfo=info)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class Signal:
    def __init__(self):
        self.items = []
        self.ready = threading.Event()

    def emit(self, raw):
        self.items.append(json.loads(raw))
        self.ready.set()


class Host(RelightActionsMixin):
    def __init__(self):
        self.vue_bridge = SimpleNamespace(relightEvent=Signal())
        self.web_mode = False


class RelightActionTests(unittest.TestCase):
    def setUp(self):
        self.host = Host()
        self.signal = self.host.vue_bridge.relightEvent
        self.errors = mock.patch("core.error_handler.handle_error")
        self.errors.start()
        self.addCleanup(self.errors.stop)

    def wait(self):
        self.assertTrue(self.signal.ready.wait(5), "relight worker did not finish")
        self.signal.ready.clear()
        return self.signal.items[-1]

    def test_full_preview_retains_dimensions_metadata_and_does_not_write(self):
        source = image_url(metadata=True)
        result = render_relight_preview({"image": source, "settings": {"strength": 0}})
        self.assertEqual((result["width"], result["height"]), (24, 16))
        self.assertEqual(result["geometry"], "luminance-approximation")
        with Image.open(io.BytesIO(result["png"])) as output:
            self.assertEqual(output.size, (24, 16))
            self.assertIn("forest", output.info["parameters"])
            self.assertEqual(output.info["workflow"], '{"nodes":[]}')
            provenance = json.loads(output.info["ai_studio_relight"])
            self.assertFalse(provenance["cast_shadows"])
            self.assertEqual(len(provenance["source_sha256"]), 64)
            np.testing.assert_array_equal(np.asarray(output)[0, 0], [90, 100, 120])
        self.assertEqual(set(result["diagnostics"]), {"light", "normals", "shadow"})

    def test_paths_external_urls_svg_corrupt_and_oversized_images_are_rejected(self):
        for value in ("C:/secret.png", "file:///C:/secret.png", "https://example.com/a.png",
                      "data:image/svg+xml;base64,AAAA", "data:image/png;base64,???", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_relight_image(value, "원본")
        with self.assertRaisesRegex(ValueError, "16 MP"):
            decode_relight_image(image_url((4097, 4097), mode="L"), "원본")   # 헤더에서 거부(픽셀 디코드 전)
        with mock.patch("ui.relight_actions.MAX_PIXELS", 100):
            # 문구는 실제 적용 중인 한도를 보여 준다 (예전에는 한도와 무관하게 '16 MP' 고정)
            with self.assertRaisesRegex(ValueError, "최대 100 px"):
                decode_relight_image(image_url(), "원본")
        with mock.patch("ui.relight_actions.MAX_FILE_BYTES", 16):
            with self.assertRaises(ValueError):
                decode_relight_image(image_url(), "원본")

    def test_validation_matches_hand_reconstruction_policy(self):
        """relight가 hand와 갈라졌던 네 가지 검증이 공용 정책(core.local_image_io)으로 통일됐다."""
        spoofed = image_url().replace("image/png", "image/jpeg")
        with self.assertRaisesRegex(ValueError, "MIME"):
            decode_relight_image(spoofed, "원본")
        with self.assertRaisesRegex(ValueError, "원본: 이미지를 읽을 수 없습니다"):
            decode_relight_image("data:image/png;base64,YQ==", "원본")   # PIL 원문 오류를 노출하지 않는다
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "p")
        info.add_text("workflow", "x" * (1024 * 1024 + 10))
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="PNG", pnginfo=info)
        oversized = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        with self.assertRaisesRegex(ValueError, "1 MB"):
            decode_relight_image(oversized, "원본")   # 예전: workflow만 조용히 버림
        with mock.patch("ui.relight_actions.MAX_METADATA_BYTES", 1):
            with self.assertRaises(ValueError):
                decode_relight_image(image_url(metadata=True), "원본")

    def test_maps_ignore_metadata_they_never_write(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("workflow", "x" * (1024 * 1024 + 10))
        buffer = io.BytesIO()
        Image.new("L", (24, 16), 128).save(buffer, format="PNG", pnginfo=info)
        depth = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        result = render_relight_preview({"image": image_url(), "depth": depth})
        self.assertEqual(result["geometry"], "depth")

    def test_grayscale_icc_is_not_attached_to_rgb_output(self):
        buffer = io.BytesIO()
        Image.new("L", (24, 16), 128).save(buffer, format="PNG", icc_profile=b"gray-profile")
        source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        _pixels, metadata = decode_relight_image(source, "원본")
        self.assertIsNone(metadata["icc"])

    def test_wrong_map_shape_and_non_rgb_normal_fail_without_resizing(self):
        with self.assertRaisesRegex(ValueError, "같은 해상도"):
            render_relight_preview({"image": image_url(), "depth": image_url((8, 8))})
        with self.assertRaisesRegex(ValueError, "RGB"):
            render_relight_preview({"image": image_url(), "normals": image_url(mode="L")})

    def test_actual_depth_and_normal_maps_are_accepted(self):
        result = render_relight_preview({"image": image_url(), "depth": image_url(mode="L"), "normals": image_url(), "mask": image_url(mode="L")})
        self.assertEqual(result["geometry"], "normal")

    def test_web_mode_and_invalid_identity_never_launch_worker(self):
        with mock.patch("ui.relight_actions.threading.Thread") as thread:
            self.host.web_mode = True
            self.host._handle_relight_action("relight_preview", {"requestId": "web", "image": image_url()})
            self.assertIn("로컬 앱", self.signal.items[-1]["error"])
            self.host.web_mode = False
            self.host._handle_relight_action("relight_preview", {"requestId": "../../bad", "image": image_url()})
            self.assertFalse(self.signal.items[-1]["ok"])
            thread.assert_not_called()

    def test_preview_runs_off_caller_thread_and_export_requires_exact_cached_id(self):
        caller = threading.get_ident()
        worker_ids = []
        original = render_relight_preview
        def render(request):
            worker_ids.append(threading.get_ident())
            return original(request)
        with mock.patch("ui.relight_actions.render_relight_preview", side_effect=render):
            self.host._handle_relight_action("relight_preview", {"requestId": "preview_1", "image": image_url()})
            event = self.wait()
        self.assertTrue(event["ok"])
        self.assertNotEqual(worker_ids, [caller])
        self.assertTrue(event["image"].startswith("data:image/png;base64,"))
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.dict(sys.modules, {"config": SimpleNamespace(OUTPUT_DIR=output_dir)}):
            self.host._handle_relight_action("relight_export", {"requestId": "export_wrong", "previewRequestId": "other"})
            self.assertFalse(self.wait()["ok"])
            self.assertFalse(list(Path(output_dir).iterdir()))
            self.host._handle_relight_action("relight_export", {"requestId": "export_1", "previewRequestId": "preview_1"})
            saved = self.wait()
            self.assertTrue(saved["ok"])
            path = Path(saved["path"])
            self.assertEqual(path.parent, Path(output_dir) / "relight")
            self.assertEqual(path.read_bytes(), self.host._relight_preview["png"])

    def test_single_job_duplicate_delivery_and_cancellation_reject_stale_results(self):
        entered, release = threading.Event(), threading.Event()
        original = render_relight_preview
        def slow(request):
            entered.set()
            release.wait(5)
            return original(request)
        with mock.patch("ui.relight_actions.render_relight_preview", side_effect=slow) as render:
            self.host._handle_relight_action("relight_preview", {"requestId": "first", "image": image_url()})
            self.assertTrue(entered.wait(2))
            self.host._handle_relight_action("relight_preview", {"requestId": "first", "image": image_url()})
            self.assertFalse(self.signal.items)
            self.host._handle_relight_action("relight_preview", {"requestId": "second", "image": image_url()})
            self.assertIn("하나", self.wait()["error"])
            self.host._handle_relight_action("relight_cancel", {"requestId": "first"})
            self.assertTrue(self.wait()["canceled"])
            release.set()
            done = self.wait()
            self.assertEqual(done["requestId"], "first")
            self.assertFalse(done["ok"])
            self.assertIsNone(self.host._relight_preview)
            self.assertEqual(render.call_count, 1)

    def test_export_is_exclusive_and_cancel_discards_cached_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(export_relight_png(b"first", directory))
            b = Path(export_relight_png(b"second", directory))
            self.assertNotEqual(a, b)
            self.assertEqual(a.read_bytes(), b"first")
            self.assertEqual(b.read_bytes(), b"second")
        self.host._ensure_relight_runtime()
        self.host._relight_preview = {"requestId": "cached", "png": b"test"}
        self.host._handle_relight_action("relight_cancel", {"requestId": "other"})
        self.assertIsNotNone(self.host._relight_preview)
        self.host._handle_relight_action("relight_cancel", {"requestId": "cached"})
        self.assertIsNone(self.host._relight_preview)

    def test_shutdown_cancels_worker_clears_cache_and_suppresses_late_emission(self):
        self.host._ensure_relight_runtime()
        cancel = threading.Event()
        self.host._relight_job = {"requestId": "working", "cancel": cancel}
        self.host._relight_preview = {"requestId": "old", "png": b"cached"}
        self.host._shutdown_relight()
        self.assertTrue(cancel.is_set())
        self.assertIsNone(self.host._relight_preview)
        self.host._relight_emit({"requestId": "working", "ok": True})
        with mock.patch("ui.relight_actions.threading.Thread") as thread:
            self.host._handle_relight_action("relight_preview", {"requestId": "late", "image": image_url()})
            thread.assert_not_called()
        self.assertFalse(self.signal.items)

    def test_combined_input_budget_is_checked_before_decoding(self):
        with mock.patch("ui.relight_actions.MAX_REQUEST_CHARS", 100):
            with mock.patch("ui.relight_actions.decode_relight_image") as decode:
                with self.assertRaisesRegex(ValueError, "총 전송 크기"):
                    render_relight_preview({"image": image_url()})
                decode.assert_not_called()

    def test_broken_node_pack_ends_as_a_relight_error_event(self):
        """노드 팩은 render_relight_preview 안에서만 import 한다 — 예전 최상단 import 는 노드 팩 한 파일의
        SyntaxError·ImportError 로 앱 기동 자체를 막았다. 지금은 미리보기 한 건의 오류 이벤트로 끝난다."""
        with mock.patch.dict(sys.modules, {"comfy_custom_nodes.ai_studio_forge_parity.relight": None}):
            with self.assertRaises(ImportError):
                render_relight_preview({"image": image_url()})
            self.host._handle_relight_action("relight_preview", {"requestId": "broken_pack", "image": image_url()})
            event = self.wait()
        self.assertEqual(event["requestId"], "broken_pack")
        self.assertFalse(event["ok"])
        self.assertTrue(event["error"])
        self.assertIsNone(self.host._relight_job, "실패한 작업이 다음 미리보기를 막지 않는다")


if __name__ == "__main__":
    unittest.main()
