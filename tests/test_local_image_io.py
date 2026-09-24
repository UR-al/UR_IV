"""core.local_image_io — 손 재구성·조명 편집이 공유하는 업로드 I/O 정책.

두 모듈이 복제하며 갈라졌던 검증(MIME 위장, 깨진 이미지 문구, 메타 1 MB 초과, ICC/EXIF 한도)을
한 곳에서 고정한다. 파일시스템은 export_exclusive_png 테스트의 임시 폴더만 쓴다.
"""
import base64
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageOps, PngImagePlugin

from core.local_image_io import (bytes_label, capture_png_metadata, decode_image_data_url, encode_png,
                                 export_exclusive_png, open_still_raster, pixels_label, png_data_url,
                                 png_save_options)

MB = 1024 * 1024


def encoded(image, fmt="PNG", **kwargs):
    stream = io.BytesIO()
    image.save(stream, format=fmt, **kwargs)
    return stream.getvalue()


def data_url(data, mime="png"):
    return f"data:image/{mime};base64," + base64.b64encode(data).decode("ascii")


class LabelTests(unittest.TestCase):
    def test_labels_reflect_the_limit_in_use(self):
        self.assertEqual(bytes_label(64 * MB), "64 MB")
        self.assertEqual(bytes_label(MB), "1 MB")
        self.assertEqual(bytes_label(16), "16 bytes")
        self.assertEqual(pixels_label(16_777_216), "16 MP")
        self.assertEqual(pixels_label(100), "100 px")


class DecodeDataUrlTests(unittest.TestCase):
    def test_accepts_real_png_jpeg_webp(self):
        image = Image.new("RGB", (4, 4), "red")
        for fmt, mime in (("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp")):
            with self.subTest(fmt=fmt):
                raw = encoded(image, fmt)
                self.assertEqual(decode_image_data_url(data_url(raw, mime), "원본", max_bytes=64 * MB), raw)

    def test_rejects_paths_urls_svg_spoofed_mime_corrupt_and_oversized(self):
        png = encoded(Image.new("RGB", (4, 4)))
        cases = {
            "C:/secret.png": "경로나 외부 URL",
            "https://example.com/a.png": "경로나 외부 URL",
            "data:image/svg+xml;base64,PHN2Zz4=": "경로나 외부 URL",
            "data:image/png;base64,%%%%": "경로나 외부 URL",
            "data:image/png;base64,YQ=": "base64",
            "data:image/png;base64,YQ==": "읽을 수 없습니다",
            data_url(png, "jpeg"): "MIME",
            data_url(b""): "경로나 외부 URL",
        }
        for value, message in cases.items():
            with self.subTest(value=value[:40]), self.assertRaisesRegex(ValueError, message):
                decode_image_data_url(value, "원본", max_bytes=64 * MB)
        with self.assertRaisesRegex(ValueError, "16 bytes"):
            decode_image_data_url(data_url(png), "원본", max_bytes=16)
        with self.assertRaises(ValueError):
            decode_image_data_url(None, "원본", max_bytes=64 * MB)


class OpenStillRasterTests(unittest.TestCase):
    def test_rejects_animation_bmp_tiny_and_over_pixel_limit(self):
        stream = io.BytesIO()
        Image.new("RGB", (4, 4)).save(stream, format="PNG", save_all=True,
                                      append_images=[Image.new("RGB", (4, 4), "blue")])
        for data, message in ((stream.getvalue(), "정지"), (encoded(Image.new("RGB", (4, 4)), "BMP"), "정지"),
                              (encoded(Image.new("RGB", (1, 4))), "2×2")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                with open_still_raster(data, "원본", max_bytes=64 * MB, max_pixels=16_777_216):
                    pass
        with self.assertRaisesRegex(ValueError, "최대 10 px"):
            with open_still_raster(encoded(Image.new("RGB", (4, 4))), "원본", max_bytes=64 * MB, max_pixels=10):
                pass

    def test_pil_errors_inside_the_body_become_korean_value_errors(self):
        truncated = encoded(Image.new("RGB", (64, 64), "red"))[:-40]
        with self.assertRaisesRegex(ValueError, "손상되었거나"):
            with open_still_raster(truncated, "원본", max_bytes=64 * MB, max_pixels=16_777_216) as opened:
                opened.load()

    def test_caller_value_errors_pass_through_unchanged(self):
        with self.assertRaisesRegex(ValueError, "caller"):
            with open_still_raster(encoded(Image.new("RGB", (4, 4))), "원본", max_bytes=64 * MB,
                                   max_pixels=16_777_216):
                raise ValueError("caller")


class MetadataTests(unittest.TestCase):
    def _open(self, data):
        opened = Image.open(io.BytesIO(data))
        opened.load()
        return opened

    def test_text_icc_exif_round_trip(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "prompt\nSteps: 20")
        info.add_itxt("workflow", '{"nodes":[]}')
        exif = Image.Exif()
        exif[315] = "artist"
        data = encoded(Image.new("RGB", (4, 4)), pnginfo=info, icc_profile=b"rgb-profile", exif=exif)
        opened = self._open(data)
        meta = capture_png_metadata(ImageOps.exif_transpose(opened), limit=MB, source_mode=opened.mode)
        self.assertEqual(meta["text"], {"parameters": "prompt\nSteps: 20", "workflow": '{"nodes":[]}'})
        self.assertEqual(meta["icc"], b"rgb-profile")
        round_trip = Image.open(io.BytesIO(encode_png(Image.new("RGB", (4, 4)), meta)))
        self.assertEqual(round_trip.info["parameters"], "prompt\nSteps: 20")
        self.assertEqual(round_trip.info["icc_profile"], b"rgb-profile")
        self.assertEqual(round_trip.getexif().get(315), "artist")

    def test_text_after_idat_is_captured_after_load(self):
        # IDAT 뒤 텍스트 청크는 load() 이후에만 info에 나타난다 — 호출자가 load()를 먼저 부른다.
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "trail")
        data = encoded(Image.new("RGB", (4, 4)), pnginfo=info)
        opened = self._open(data)
        self.assertEqual(capture_png_metadata(opened, limit=MB, source_mode=opened.mode)["text"],
                         {"parameters": "trail"})

    def test_oversized_text_fails_instead_of_silently_dropping(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", "p")
        info.add_text("workflow", "w" * (MB + 1))
        opened = self._open(encoded(Image.new("RGB", (4, 4)), pnginfo=info))
        with self.assertRaisesRegex(ValueError, "1 MB"):
            capture_png_metadata(opened, limit=MB, source_mode=opened.mode)

    def test_key_bytes_count_toward_the_limit(self):
        info = PngImagePlugin.PngInfo()
        info.add_text("k" * 10, "v" * 10)
        opened = self._open(encoded(Image.new("RGB", (4, 4)), pnginfo=info))
        self.assertTrue(capture_png_metadata(opened, limit=20, source_mode="RGB")["text"])
        with self.assertRaises(ValueError):
            capture_png_metadata(opened, limit=19, source_mode="RGB")

    def test_oversized_icc_or_exif_fails(self):
        opened = self._open(encoded(Image.new("RGB", (4, 4)), icc_profile=b"x" * 64))
        with self.assertRaisesRegex(ValueError, "ICC/EXIF"):
            capture_png_metadata(opened, limit=32, source_mode="RGB")

    def test_non_rgb_icc_profiles_are_not_attached_to_rgb_png(self):
        for mode in ("L", "LA", "CMYK", "I;16"):
            with self.subTest(mode=mode):
                opened = self._open(encoded(Image.new("RGB", (4, 4)), icc_profile=b"x" * 64))
                self.assertIsNone(capture_png_metadata(opened, limit=32, source_mode=mode)["icc"],
                                  "회색·CMYK 프로파일은 버리며 크기 오류도 내지 않는다")
        for mode in ("RGB", "RGBA", "P", "PA"):
            with self.subTest(mode=mode):
                opened = self._open(encoded(Image.new("RGB", (4, 4)), icc_profile=b"rgb"))
                self.assertEqual(capture_png_metadata(opened, limit=MB, source_mode=mode)["icc"], b"rgb")

    def test_keys_that_cannot_be_png_keywords_are_excluded(self):
        opened = self._open(encoded(Image.new("RGB", (4, 4))))
        opened.info.update({"k" * 80: "too long", "한글키": "not latin-1", "ok": "kept", "dpi": (72, 72)})
        self.assertEqual(capture_png_metadata(opened, limit=MB, source_mode="RGB")["text"], {"ok": "kept"})

    def test_png_save_options_always_pin_the_icc_profile(self):
        # Pillow은 icc_profile 옵션이 없으면 image.info로 폴백한다 → 항상 명시(None 포함)
        self.assertEqual(png_save_options(None), {"icc_profile": None})
        self.assertEqual(png_save_options({}), {"icc_profile": None})
        self.assertEqual(set(png_save_options({"text": {}, "icc": None, "exif": None})), {"pnginfo", "icc_profile"})

    def test_encode_png_never_leaks_the_images_own_icc(self):
        image = Image.new("RGB", (4, 4))
        image.info["icc_profile"] = b"stale-gray-profile"       # convert/copy가 원본 info를 유지한 상황
        for metadata in (None, {"text": {}, "icc": None, "exif": None}):
            with self.subTest(metadata=metadata):
                self.assertNotIn("icc_profile", Image.open(io.BytesIO(encode_png(image, metadata))).info)
        approved = Image.open(io.BytesIO(encode_png(image, {"text": {}, "icc": b"rgb", "exif": None})))
        self.assertEqual(approved.info["icc_profile"], b"rgb")

    def test_png_data_url(self):
        self.assertEqual(png_data_url(b"abc"), "data:image/png;base64,YWJj")


class ExclusiveExportTests(unittest.TestCase):
    def export(self, root, data, token="same"):
        return export_exclusive_png(
            data, root, subdir="out", prefix="test",
            now=lambda: datetime(2026, 9, 24, 12, 0, 0), token=lambda: token,
            outside_message="outside", exhausted_message="이름 소진")

    def test_never_overwrites_and_reports_exhaustion(self):
        with tempfile.TemporaryDirectory() as root:
            first = Path(self.export(root, b"first"))
            self.assertEqual(first.name, "test_20260924_120000_same.png")
            with self.assertRaisesRegex(ValueError, "이름 소진"):
                self.export(root, b"second")
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(list(first.parent.iterdir()), [first])

    def test_retries_with_a_new_token(self):
        tokens = iter(["same", "same", "other"])
        with tempfile.TemporaryDirectory() as root:
            self.export(root, b"first")
            path = export_exclusive_png(
                b"second", root, subdir="out", prefix="test",
                now=lambda: datetime(2026, 9, 24, 12, 0, 0), token=lambda: next(tokens),
                outside_message="outside", exhausted_message="이름 소진")
            self.assertTrue(path.endswith("_other.png"))

    def test_subdir_escaping_the_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "outside"):
                export_exclusive_png(b"x", Path(root) / "inner", subdir="..", prefix="t",
                                     now=datetime.now, token=lambda: "a",
                                     outside_message="outside", exhausted_message="x")

    def test_failed_write_removes_only_the_new_partial_file(self):
        with tempfile.TemporaryDirectory() as root:
            keep = Path(self.export(root, b"keep", token="keep"))
            with self.assertRaises(TypeError):
                self.export(root, object(), token="partial")
            self.assertEqual(sorted(p.name for p in keep.parent.iterdir()), [keep.name])


if __name__ == "__main__":
    unittest.main()
