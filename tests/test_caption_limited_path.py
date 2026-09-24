"""_limited_image_path — 원본을 그대로 넘기는 경로(1MP 이하·RGB·회전 없음)의 두 회귀.

1) 컨텍스트 안(=Ollama 요청 중)에서 난 ConnectionError/TimeoutError(둘 다 OSError)가
   'Unable to open image for captioning' 으로 잘못 포장되면 안 된다.
2) 요청 내내 원본 파일 핸들이 열려 있으면 JPEG/WebP 원본을 옮기거나 지울 때 WinError 32 가 난다.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.image_captioning import ImageCaptioningError, _limited_image_path


class LimitedImagePathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def _image(self, name, size=(768, 1024), mode='RGB', fmt=None):
        path = self.root / name
        Image.new(mode, size, 'navy' if mode == 'RGB' else None).save(path, format=fmt)
        return path

    def test_errors_raised_inside_the_context_propagate_unchanged(self):
        for name in ('small.png', 'small.jpg', 'small.webp'):
            source = self._image(name)
            for error in (ConnectionError('Ollama 서버에 연결할 수 없습니다.'), TimeoutError('캡션 응답 시간 초과')):
                with self.subTest(name=name, error=type(error).__name__):
                    with self.assertRaises(type(error)) as caught:
                        with _limited_image_path(source, 1_000_000) as limited:
                            self.assertEqual(Path(limited), source, '작은 RGB 원본은 그대로 넘긴다')
                            raise error
                    self.assertNotIsInstance(caught.exception, ImageCaptioningError)

    def test_passthrough_source_is_not_locked_during_the_request(self):
        for name in ('small.jpg', 'small.webp', 'small.png'):
            source = self._image(name)
            moved = self.root / f'moved_{name}'
            with self.subTest(name=name):
                with _limited_image_path(source, 1_000_000) as limited:
                    self.assertEqual(Path(limited), source)
                    os.replace(source, moved)   # WinError 32 없이 이동돼야 한다
                    os.replace(moved, source)

    def test_normalised_copy_is_still_used_for_large_or_non_rgb_images(self):
        large = self._image('large.jpg', size=(1400, 1400))
        with _limited_image_path(large, 1_000_000) as limited:
            self.assertNotEqual(Path(limited), large)
            with Image.open(limited) as copy:
                self.assertLessEqual(copy.width * copy.height, 1_000_000)
                self.assertEqual(copy.mode, 'RGB')
        rgba = self._image('alpha.png', size=(64, 64), mode='RGBA')
        with _limited_image_path(rgba, 1_000_000) as limited:
            self.assertNotEqual(Path(limited), rgba)

    def test_unreadable_image_is_still_reported_as_open_failure(self):
        broken = self.root / 'broken.png'
        broken.write_bytes(b'not an image')
        with self.assertRaisesRegex(ImageCaptioningError, 'Unable to open image'):
            with _limited_image_path(broken, 1_000_000):
                pass


if __name__ == '__main__':
    unittest.main()
