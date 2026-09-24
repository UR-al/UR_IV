"""생성 결과 해상도(core.result_image) — 헤더만 읽고, 못 읽을 때만 요청 크기로 폴백."""
from __future__ import annotations

import io
import unittest
from unittest import mock

from PIL import Image

from core.result_image import result_image_size


def _encoded(width: int, height: int, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, fmt)
    return buffer.getvalue()


class ResultImageSizeTests(unittest.TestCase):
    def test_reads_the_real_size_from_the_header_not_the_request(self):
        # hires fix·업스케일 결과는 요청 크기(gen_info)와 다르다 — 헤더가 이긴다
        data = _encoded(96, 40)
        self.assertEqual(result_image_size(data, {"width": 48, "height": 20}), (96, 40))
        self.assertEqual(result_image_size(bytearray(data)), (96, 40))
        self.assertEqual(result_image_size(memoryview(data)), (96, 40))

    def test_other_formats(self):
        for fmt in ("JPEG", "WEBP"):
            with self.subTest(fmt=fmt):
                self.assertEqual(result_image_size(_encoded(33, 17, fmt)), (33, 17))

    def test_does_not_decode_pixels(self):
        data = _encoded(64, 32)
        with mock.patch.object(Image.Image, "load", side_effect=AssertionError("픽셀 디코드")):
            self.assertEqual(result_image_size(data), (64, 32))

    def test_unreadable_bytes_fall_back_to_the_request_size(self):
        self.assertEqual(result_image_size(b"not an image", {"width": 832, "height": "1216"}), (832, 1216))
        self.assertEqual(result_image_size(b"", {"width": 512.0, "height": 768}), (512, 768))
        self.assertEqual(result_image_size(None, {"width": 640, "height": 480}), (640, 480))

    def test_missing_or_bad_fallback_is_zero(self):
        for info in (None, {}, {"width": 512}, {"width": "x", "height": 5},
                     {"width": -3, "height": 4}, {"width": True, "height": 4}, "not a dict"):
            with self.subTest(info=info):
                self.assertEqual(result_image_size(b"garbage", info), (0, 0))


if __name__ == "__main__":
    unittest.main()
