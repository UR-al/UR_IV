"""T2I 결과 처리(GenerationMixin._process_new_image) — 생성마다 하던 헛작업 제거 회귀 가드.

예전엔 장마다 GUI 스레드에서 QPixmap 풀 디코드 + 스무스 축소를 no-op 뷰어 더미에 넘기고,
아무도 읽지 않는 150px 평면 썸네일(_create_thumbnail)과 보이지 않는 ThumbnailItem
(add_image_to_gallery, 최대 100개)을 만들고, 읽는 곳 없는 generation_data 를 쌓았다.
지금은 파일 저장 → 헤더에서 읽은 해상도로 send_image → (XYZ 항목이면) 결과 통지만 한다.
"""
from __future__ import annotations

import ast
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

import ui.generator_generation as generation
from ui.generator_generation import GenerationMixin

ROOT = Path(__file__).resolve().parents[1]


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (5, 6, 7)).save(buffer, "PNG")
    return buffer.getvalue()


class _Host:
    """_process_new_image 가 쓰는 것만 가진 창 — 레거시 갤러리 속성(gallery_items·viewer_label·
    exif_display·generation_data·add_image_to_gallery·_create_thumbnail)은 일부러 없다."""

    def __init__(self):
        self.vue_bridge = mock.Mock()
        self._xyz_emit = mock.Mock()
        self.current_image_path = None

    _process_new_image = GenerationMixin._process_new_image


class ProcessNewImageTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.TemporaryDirectory()
        self.addCleanup(self.out.cleanup)
        self.enterContext(mock.patch.object(generation, "OUTPUT_DIR", self.out.name))

    def test_saves_and_sends_the_header_size_not_the_requested_size(self):
        host = _Host()
        data = _png(96, 40)
        # gen_info 크기는 요청값(hires 전) — 해상도 표시는 결과 헤더를 따라야 한다
        host._process_new_image(data, {"seed": 1234, "width": 48, "height": 20})
        path, width, height, seed = host.vue_bridge.send_image.call_args.args
        self.assertEqual((width, height, seed), (96, 40, 1234))
        self.assertEqual(os.path.dirname(path), self.out.name)
        with open(path, "rb") as handle:
            self.assertEqual(handle.read(), data)
        self.assertEqual(host.current_image_path, path)
        host._xyz_emit.assert_not_called()

    def test_does_not_decode_the_result_on_the_gui_thread(self):
        host = _Host()
        data = _png(64, 32)
        with mock.patch.object(Image.Image, "load", side_effect=AssertionError("픽셀 디코드")):
            host._process_new_image(data, {"seed": 1})
        self.assertEqual(host.vue_bridge.send_image.call_args.args[1:3], (64, 32))

    def test_unreadable_result_falls_back_to_the_request_size(self):
        host = _Host()
        host._process_new_image(b"not a png", {"width": 832, "height": 1216})
        self.assertEqual(host.vue_bridge.send_image.call_args.args[1:], (832, 1216, 0))

    def test_non_dict_gen_info_is_tolerated(self):
        host = _Host()
        host._process_new_image(_png(10, 20), None)
        self.assertEqual(host.vue_bridge.send_image.call_args.args[1:], (10, 20, 0))

    def test_xyz_result_is_reported(self):
        host = _Host()
        info = {"seed": 3, "_xyz_info": {"requestId": "r1", "label": "steps=20", "axes": {"x": 20}}}
        host._process_new_image(_png(8, 8), info)
        name, payload = host._xyz_emit.call_args.args
        self.assertEqual(name, "xyzPlotEvent")
        self.assertEqual(payload["type"], "result")
        self.assertTrue(payload["ok"])
        self.assertEqual((payload["requestId"], payload["label"], payload["axes"]), ("r1", "steps=20", {"x": 20}))
        self.assertEqual(payload["path"], host.current_image_path)

    def test_source_has_no_dead_legacy_gallery_work(self):
        source = (ROOT / "ui" / "generator_generation.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_process_new_image")
        body = fn.body[1:] if ast.get_docstring(fn) is not None else fn.body   # 독스트링은 설명일 뿐
        nodes = [node for stmt in body for node in ast.walk(stmt)]
        called = {
            (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", ""))
            for node in nodes if isinstance(node, ast.Call)
        }
        referenced = (
            {node.attr for node in nodes if isinstance(node, ast.Attribute)}
            | {node.id for node in nodes if isinstance(node, ast.Name)}
            | {node.value for node in nodes if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        )
        for dead in ("QPixmap", "loadFromData", "scaled", "setPixmap", "_create_thumbnail",
                     "add_image_to_gallery", "exif_for_display", "add_result_image"):
            self.assertNotIn(dead, called)
        for dead_attr in ("generation_data", "exif_display", "viewer_info_bar", "viewer_label",
                          "_pending_xyz_info", "xyz_plot_tab"):
            self.assertNotIn(dead_attr, referenced)
        self.assertIn("result_image_size", called)


if __name__ == "__main__":
    unittest.main()
