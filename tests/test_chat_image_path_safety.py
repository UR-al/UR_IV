"""대화 첨부 경로 → base64 인라인이 safe_input_path 를 거치는지 (임의 파일 유출 회귀).

inline_image_paths 는 경로처럼 보이는 첨부를 읽어 base64 로 싣고, 그 요청은 payload 의
url(Ollama/LM Studio)로 나간다. 예전엔 isfile·크기만 봐서 토큰 보유 웹 클라이언트가
config JSON 같은 이미지가 아닌 파일을 외부 호스트로 빼낼 수 있었다.
"""
from __future__ import annotations

import base64
import os
import pathlib
import tempfile
import unittest

from core.chat_store import inline_image_paths

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class InlineImagePathSafetyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def _images(self, *items):
        out = inline_image_paths([{"role": "user", "content": "?", "images": list(items)}])
        return out[0].get("images", [])

    def test_non_image_files_are_never_read(self):
        secret = self.root / "config" / "ui_prefs.json"
        secret.parent.mkdir()
        secret.write_text('{"token": "top-secret"}', encoding="utf-8")
        script = self.root / "notes.txt"
        script.write_text("private", encoding="utf-8")
        disguised = self.root / "archive.png.json"
        disguised.write_bytes(PNG)
        self.assertEqual(self._images(str(secret), str(script), str(disguised)), [])

    def test_image_files_still_inline_with_either_separator(self):
        image = self.root / "gallery" / "a.png"
        image.parent.mkdir()
        image.write_bytes(PNG)
        encoded = base64.b64encode(PNG).decode("ascii")
        posix = image.as_posix()
        self.assertEqual(self._images(str(image), posix), [encoded, encoded])

    @unittest.skipUnless(os.name == "nt", "Windows 시스템 폴더 전용")
    def test_system_folder_images_are_refused(self):
        system_root = os.environ.get("SystemRoot") or r"C:\Windows"
        candidates = [
            p for p in pathlib.Path(system_root, "Web").rglob("*.jpg")
        ][:1] if pathlib.Path(system_root, "Web").is_dir() else []
        if not candidates:
            self.skipTest("시스템 폴더에 샘플 이미지가 없다")
        sample = str(candidates[0])
        self.assertEqual(self._images(sample, "\\\\?\\" + sample), [])

    def test_oversized_image_is_dropped(self):
        image = self.root / "big.png"
        image.write_bytes(PNG + b"\x00" * 64)
        out = inline_image_paths(
            [{"role": "user", "content": "x", "images": [str(image)]}], max_bytes=16
        )
        self.assertNotIn("images", out[0])


if __name__ == "__main__":
    unittest.main()
