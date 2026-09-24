"""에디터 클립보드 붙여넣기 저장 — 경로 탈출·임의 바이트 쓰기 회귀 테스트.

예전 editorPasteImage 는 mime 의 '/' 뒤를 그대로 확장자로 붙여
``image/..\\..\\x.bat`` 로 임시 폴더 밖에 호출자가 정한 바이트를 쓸 수 있었다.
"""
from __future__ import annotations

import base64
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from core.clipboard_paste import (
    PASTE_EXTENSIONS,
    ClipboardPasteError,
    decode_paste_payload,
    paste_extension_for_mime,
    save_clipboard_image,
    sniff_image_extension,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
BMP = b"BM" + b"\x00" * 40
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


class MimeMapTests(unittest.TestCase):
    def test_frontend_whitelist_maps_to_fixed_extensions(self):
        self.assertEqual(
            {m: paste_extension_for_mime(m) for m in PASTE_EXTENSIONS},
            {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/bmp": "bmp", "image/webp": "webp"},
        )
        self.assertEqual(paste_extension_for_mime(" IMAGE/PNG "), "png")

    def test_everything_else_is_refused(self):
        for mime in (
            "image/..\\..\\..\\..\\..\\x.bat",
            "image/../../x.png",
            "image/png/../../x",
            "image/svg+xml",
            "image/heic",
            "text/plain",
            "png",
            "",
            None,
        ):
            with self.subTest(mime=mime):
                self.assertIsNone(paste_extension_for_mime(mime))


class PayloadTests(unittest.TestCase):
    def test_signatures(self):
        self.assertEqual(sniff_image_extension(PNG), "png")
        self.assertEqual(sniff_image_extension(JPG), "jpg")
        self.assertEqual(sniff_image_extension(BMP), "bmp")
        self.assertEqual(sniff_image_extension(WEBP), "webp")
        self.assertIsNone(sniff_image_extension(b"@echo off\r\ncalc.exe\r\n"))
        self.assertIsNone(sniff_image_extension(b""))

    def test_strict_base64_and_data_url_prefix(self):
        self.assertEqual(decode_paste_payload(_b64(PNG)), PNG)
        self.assertEqual(decode_paste_payload("data:image/png;base64," + _b64(PNG)), PNG)
        for bad in ("", "   ", "not base64 !!!", "QUJ"):
            with self.subTest(bad=bad):
                with self.assertRaises(ClipboardPasteError):
                    decode_paste_payload(bad)

    def test_oversized_payload_is_refused_before_decoding(self):
        with mock.patch("core.clipboard_paste.MAX_PASTE_BYTES", 16):
            with self.assertRaises(ClipboardPasteError):
                decode_paste_payload(_b64(PNG))


class SaveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.paste_dir = self.root / "sandbox" / "AIStudioPro_editor"

    def test_saves_inside_directory_with_sniffed_extension(self):
        saved = Path(save_clipboard_image(_b64(PNG), "image/png", directory=self.paste_dir))
        self.assertEqual(saved.parent, self.paste_dir.resolve())
        self.assertEqual(saved.suffix, ".png")
        self.assertTrue(saved.name.startswith("clipboard_"))
        self.assertEqual(saved.read_bytes(), PNG)
        # mime 이 jpeg 라고 해도 내용이 webp 면 확장자는 내용을 따른다.
        other = Path(save_clipboard_image(_b64(WEBP), "image/jpeg", directory=self.paste_dir))
        self.assertEqual(other.suffix, ".webp")

    def test_same_second_pastes_never_overwrite_each_other(self):
        first = save_clipboard_image(_b64(PNG), "image/png", directory=self.paste_dir)
        second = save_clipboard_image(_b64(JPG), "image/jpeg", directory=self.paste_dir)
        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).read_bytes(), PNG)
        self.assertEqual(Path(second).read_bytes(), JPG)

    def test_traversal_mime_writes_nothing_anywhere(self):
        payload = _b64(b"@echo off\r\n")
        for mime in ("image/..\\..\\..\\x.bat", "image/../../x.bat", "image/..\\..\\x.png"):
            with self.subTest(mime=mime):
                with self.assertRaises(ClipboardPasteError):
                    save_clipboard_image(payload, mime, directory=self.paste_dir)
        written = [p for p in self.root.rglob("*") if p.is_file()]
        self.assertEqual(written, [])

    def test_allowed_mime_with_non_image_bytes_is_refused_and_cleaned_up(self):
        with self.assertRaises(ClipboardPasteError):
            save_clipboard_image(_b64(b"@echo off\r\ncalc\r\n"), "image/png", directory=self.paste_dir)
        leftovers = list(self.paste_dir.glob("*")) if self.paste_dir.exists() else []
        self.assertEqual(leftovers, [])


class StoreAndPruneTests(unittest.TestCase):
    """붙여넣기 임시 파일은 %TEMP% 에 끝없이 쌓였다 — 오래된 clipboard_* 만 치운다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "AIStudioPro_editor"
        self.dir.mkdir()

    def _make(self, name: str, age_hours: float, now: float) -> Path:
        import os
        path = self.dir / name
        path.write_bytes(PNG)
        stamp = now - age_hours * 3600
        os.utime(path, (stamp, stamp))
        return path

    def test_prune_removes_only_old_clipboard_images_beyond_the_newest_few(self):
        from core.clipboard_paste import prune_clipboard_images
        now = 2_000_000_000.0
        old = [self._make(f"clipboard_old{i}.png", 48 + i, now) for i in range(5)]
        fresh = self._make("clipboard_fresh.jpg", 1, now)
        autosave = self._make("_autosave_session.png", 999, now)
        other = self._make("notes.png", 999, now)
        odd_ext = self._make("clipboard_keep.txt", 999, now)
        removed = prune_clipboard_images(self.dir, keep_newest=2, now=now)
        self.assertEqual(removed, 4)
        self.assertTrue(fresh.exists())
        self.assertTrue(old[0].exists(), "최신 keep_newest 장은 오래됐어도 남아야 한다")
        self.assertFalse(any(p.exists() for p in old[1:]))
        for keep in (autosave, other, odd_ext):
            self.assertTrue(keep.exists(), f"{keep.name} 은 붙여넣기 파일이 아니다 — 지우면 안 된다")

    def test_prune_respects_exclude_and_missing_directory(self):
        from core.clipboard_paste import prune_clipboard_images
        now = 2_000_000_000.0
        target = self._make("clipboard_open.png", 72, now)
        self.assertEqual(prune_clipboard_images(self.dir, keep_newest=0, now=now, exclude=[target]), 0)
        self.assertTrue(target.exists())
        self.assertEqual(prune_clipboard_images(self.dir / "nope"), 0)

    def test_store_bytes_validates_and_prunes(self):
        from core.clipboard_paste import store_clipboard_bytes
        import os
        now = time.time()
        stale = self._make("clipboard_stale.png", 100, now)
        keep = [self._make(f"clipboard_new{i}.png", 0.1, now) for i in range(3)]
        with mock.patch("core.clipboard_paste.CLIPBOARD_KEEP_NEWEST", 1):
            saved = Path(store_clipboard_bytes(PNG, directory=self.dir))
        self.assertEqual(saved.read_bytes(), PNG)
        self.assertFalse(stale.exists())
        self.assertTrue(all(p.exists() for p in keep), "24시간 안의 붙여넣기는 남아야 한다")
        for bad in (b"", b"@echo off"):
            with self.subTest(bad=bad):
                with self.assertRaises(ClipboardPasteError):
                    store_clipboard_bytes(bad, directory=self.dir)
        with mock.patch("core.clipboard_paste.MAX_PASTE_BYTES", 8):
            with self.assertRaises(ClipboardPasteError):
                store_clipboard_bytes(PNG, directory=self.dir)
        self.assertTrue(os.path.isdir(self.dir))

    def test_pick_clipboard_image_file(self):
        from core.clipboard_paste import pick_clipboard_image_file
        img = self.dir / "탐색기에서 복사.webp"
        img.write_bytes(WEBP)
        txt = self.dir / "readme.txt"
        txt.write_text("x", encoding="utf-8")
        self.assertEqual(pick_clipboard_image_file([str(txt), str(img)]), str(img.resolve()))
        self.assertIsNone(pick_clipboard_image_file([str(txt), str(self.dir / "없음.png"), "", None]))
        self.assertIsNone(pick_clipboard_image_file([]))


try:
    from ui.vue_bridge import VueBridge
    _IMPORT_ERROR = ""
except ModuleNotFoundError as exc:  # pragma: no cover - PyQt 없는 환경
    if not (exc.name or "").startswith("PyQt6"):
        raise
    VueBridge = None
    _IMPORT_ERROR = str(exc)


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class EditorPasteSlotTests(unittest.TestCase):
    def test_slot_returns_error_for_traversal_mime_and_path_for_valid_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            paste_dir = Path(tmp) / "AIStudioPro_editor"
            with mock.patch("core.clipboard_paste.default_paste_dir", return_value=paste_dir):
                bridge = VueBridge()
                bad = json.loads(bridge.editorPasteImage(_b64(b"payload"), "image/..\\..\\..\\x.bat"))
                good = json.loads(bridge.editorPasteImage(_b64(PNG), "image/png"))
            self.assertIn("error", bad)
            self.assertNotIn("path", bad)
            self.assertNotIn("\\", good["path"])
            saved = Path(good["path"])
            self.assertEqual(saved.parent, paste_dir.resolve())
            self.assertEqual(saved.read_bytes(), PNG)
            self.assertEqual([p.name for p in Path(tmp).rglob("*.bat")], [])


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class CompareGifExportTests(unittest.TestCase):
    """비교 GIF 는 초 단위 이름이라 같은 초의 두 번째 내보내기가 첫 GIF 를 덮어썼다.

    예전 동기 슬롯 exportCompareGif 는 비동기 requestCompareGif(→ compareGifReady)로 바뀌었다.
    여기서는 워커가 도는 본체(_compare_gif_json)를 호출 스레드에서 바로 부른다."""

    def test_same_second_exports_never_overwrite(self):
        from PIL import Image
        import ui.vue_bridge as vue_bridge_module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after = root / "전.png", root / "후.png"
            Image.new("RGB", (16, 12), (255, 0, 0)).save(before)
            Image.new("RGB", (16, 12), (0, 0, 255)).save(after)
            fake_module_file = root / "ui" / "vue_bridge.py"   # out_dir = <root>/gif
            with mock.patch.object(vue_bridge_module, "__file__", str(fake_module_file)), \
                    mock.patch("time.time", return_value=1_700_000_000.4):
                bridge = VueBridge()
                first = json.loads(bridge._compare_gif_json(str(before), str(after), 80, 0, "a"))
                second = json.loads(bridge._compare_gif_json(str(before), str(after), 80, 0, "b"))
            self.assertIn("path", first, first)
            self.assertIn("path", second, second)
            self.assertNotEqual(first["path"], second["path"])
            for result in (first, second):
                path = Path(result["path"])
                self.assertEqual(path.parent, root / "gif")
                with Image.open(path) as gif:
                    self.assertEqual(gif.format, "GIF")
                    self.assertGreater(getattr(gif, "n_frames", 1), 1)


_SYSTEM_CLIPBOARD_PROBE = r'''
import json, sys
from pathlib import Path
from unittest import mock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor
from PyQt6.QtCore import QMimeData, QUrl
from ui.vue_bridge import VueBridge   # QtWebEngine 은 QApplication 보다 먼저 import 해야 한다
app = QApplication([])
paste_dir = Path(sys.argv[1]) / "AIStudioPro_editor"
copied = Path(sys.argv[1]) / "탐색기 복사.png"
copied.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
out = {}
with mock.patch("core.clipboard_paste.default_paste_dir", return_value=paste_dir):
    bridge = VueBridge()
    app.clipboard().clear()
    out["empty"] = json.loads(bridge.editorPasteFromSystemClipboard())
    img = QImage(64, 32, QImage.Format.Format_ARGB32)
    img.fill(QColor(255, 0, 0, 128))
    app.clipboard().setImage(img)
    out["image"] = json.loads(bridge.editorPasteFromSystemClipboard())
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(copied))])
    app.clipboard().setMimeData(mime)
    out["file"] = json.loads(bridge.editorPasteFromSystemClipboard())
out["copied"] = str(copied.resolve()).replace("\\", "/")
print("RESULT:" + json.dumps(out, ensure_ascii=True))
'''


@unittest.skipIf(VueBridge is None, f"PyQt6 unavailable: {_IMPORT_ERROR}")
class SystemClipboardSlotTests(unittest.TestCase):
    """데스크톱 붙여넣기 — QtWebEngine 에는 clipboard.read() 권한이 없어 Qt 로 직접 읽는다.

    QApplication 은 테스트 프로세스를 오염시키지 않게 offscreen 하위 프로세스에서 만든다.
    """

    def test_reads_image_data_and_copied_files_from_the_qt_clipboard(self):
        import os
        import subprocess
        import sys
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", _SYSTEM_CLIPBOARD_PROBE, tmp],
                cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "QT_OPENGL": "software",
                     "PYTHONIOENCODING": "utf-8"},
            )
            line = next((l for l in result.stdout.splitlines() if l.startswith("RESULT:")), None)
            self.assertIsNotNone(line, result.stdout[-2000:] + result.stderr[-2000:])
            out = json.loads(line[len("RESULT:"):])
            self.assertTrue(out["empty"].get("empty"), out["empty"])
            self.assertNotIn("path", out["empty"])
            saved = Path(out["image"]["path"])
            self.assertEqual(saved.parent, (Path(tmp) / "AIStudioPro_editor").resolve())
            self.assertTrue(saved.name.startswith("clipboard_") and saved.suffix == ".png")
            self.assertTrue(saved.read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(out["file"]["path"], out["copied"], "탐색기에서 복사한 원본 파일을 그대로 열어야 한다")


if __name__ == "__main__":
    unittest.main()
