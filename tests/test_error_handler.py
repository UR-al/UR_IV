"""core.error_handler — UI 로 보내는 오류 문구의 경로 가림과 토스트 정책."""
from __future__ import annotations

import io
import re
import unittest
from pathlib import Path
from unittest import mock

from core import error_handler
from core.error_handler import ERROR_CODES, handle_error, sanitize_for_ui

ROOT = Path(__file__).resolve().parents[1]


class SanitizeForUiTests(unittest.TestCase):
    def test_backend_urls_are_preserved(self):
        for message in (
            "500 Server Error: Internal Server Error for url: http://127.0.0.1:7860/sdapi/v1/txt2img",
            "HTTPConnectionPool(host='127.0.0.1', port=8188): Max retries exceeded with url: /prompt",
            "https://example.com/home/page 응답 없음",
            "ws://localhost:8188/ws?clientId=abc",
            "http://host:7860/tmp/x.png 404",
        ):
            with self.subTest(message=message):
                self.assertEqual(sanitize_for_ui(message, 400), message)

    def test_windows_drive_and_unc_paths_are_masked(self):
        cases = {
            r"[Errno 2] No such file: 'C:\Users\me\secret\model.safetensors'":
                "[Errno 2] No such file: '[path]'",
            "D:/models/Stable-diffusion/x.ckpt 을 열 수 없음": "[path] 을 열 수 없음",
            "파일C:\\data\\x.png": "파일[path]",
            r"공유 폴더 \\nas\share\models\x.pt 권한 없음": "공유 폴더 [path] 권한 없음",
            "file:///C:/Users/me/x.png": "file:///[path]",
        }
        for message, expected in cases.items():
            with self.subTest(message=message):
                self.assertEqual(sanitize_for_ui(message, 400), expected)

    def test_posix_absolute_paths_are_masked_but_not_url_paths(self):
        self.assertEqual(sanitize_for_ui("open '/home/me/.ssh/id' failed", 400), "open '[path]' failed")
        self.assertEqual(sanitize_for_ui("/Users/me/x.png", 400), "[path]")
        self.assertEqual(sanitize_for_ui("file:///home/me/x.png", 400), "file://[path]")
        self.assertEqual(
            sanitize_for_ui("GET http://127.0.0.1:7860/home/x failed", 400),
            "GET http://127.0.0.1:7860/home/x failed",
        )

    def test_length_is_limited_and_empty_is_empty(self):
        self.assertEqual(sanitize_for_ui("", 10), "")
        self.assertEqual(sanitize_for_ui(None, 10), "")
        out = sanitize_for_ui("x" * 50, 10)
        self.assertEqual(len(out), 10)
        self.assertTrue(out.endswith("…"))


class HandleErrorToastTests(unittest.TestCase):
    def setUp(self):
        self.original_bridge = error_handler._vue_bridge
        self.addCleanup(error_handler.set_bridge, self.original_bridge)
        self.bridge = mock.Mock()
        error_handler.set_bridge(self.bridge)

    def _call(self, exc, **kwargs):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", io.StringIO()):
            result = handle_error("E030", "설정 저장", exc, **kwargs)
        return result, out.getvalue()

    def test_toast_masks_paths_but_console_keeps_the_original(self):
        exc = PermissionError(r"[Errno 13] Permission denied: 'C:\Users\me\config\ui_prefs.json'")
        result, console = self._call(exc)
        level, toast = self.bridge.showNotification.emit.call_args.args
        self.assertEqual(level, "error")
        self.assertTrue(toast.startswith("[E030] 설정 저장: "))
        self.assertIn("[path]", toast)
        self.assertNotIn("ui_prefs.json", toast)
        self.assertIn(r"C:\Users\me\config\ui_prefs.json", console)   # 콘솔 로그는 원문
        self.assertIn(r"C:\Users\me\config\ui_prefs.json", result)

    def test_toast_keeps_backend_urls_and_is_length_limited(self):
        exc = RuntimeError("500 Server Error for url: http://127.0.0.1:7860/sdapi/v1/options " + "x" * 300)
        self._call(exc)
        _level, toast = self.bridge.showNotification.emit.call_args.args
        self.assertIn("http://127.0.0.1:7860/sdapi/v1/options", toast)
        self.assertLessEqual(len(toast), 120)

    def test_notify_false_sends_no_toast(self):
        self._call(ValueError("x"), notify=False)
        self.bridge.showNotification.emit.assert_not_called()


class ErrorCodeCatalogTests(unittest.TestCase):
    def test_every_code_used_in_code_is_defined(self):
        """handle_error('Exxx', …) 로 쓰는 코드는 ERROR_CODES 에 설명이 있어야 한다(없으면 'Unknown Error')."""
        used = set()
        pattern = re.compile(r"handle_error\(\s*['\"](E\d{3})['\"]")
        for folder in ("core", "ui", "backends", "workers", "utils", "tabs", "widgets"):
            for path in (ROOT / folder).rglob("*.py"):
                used.update(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
        self.assertTrue(used, "handle_error 호출을 하나도 찾지 못했다 — 패턴 확인")
        self.assertEqual(sorted(used - set(ERROR_CODES)), [])


if __name__ == "__main__":
    unittest.main()
