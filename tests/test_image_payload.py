"""생성 입력 이미지 인코딩 (core/image_payload) — #120.

경로로 온 I2I/Inpaint 입력은 원본 바이트를 그대로 base64 로 보낸다(메인 스레드에서
QPixmap 디코드·스케일·PNG 재인코딩 ~250ms 를 하지 않는다). 알파·방향 태그 등만 다시 인코딩.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import unittest

from PIL import Image

from core.image_payload import (
    ImagePayloadError,
    decode_base64_image,
    encode_image_bytes,
    encode_image_file,
    image_size_from_base64,
    image_size_from_bytes,
    resolve_target_size,
    strip_data_url,
)


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


class PassThroughTests(unittest.TestCase):
    def test_plain_rgb_png_jpeg_webp_are_sent_byte_for_byte(self):
        for fmt in ('PNG', 'JPEG', 'WEBP'):
            with self.subTest(fmt=fmt):
                data = _encode(Image.new('RGB', (40, 24), (10, 20, 30)), fmt)
                encoded = encode_image_bytes(data)
                self.assertFalse(encoded.reencoded)
                self.assertEqual(base64.b64decode(encoded.b64), data)
                self.assertEqual((encoded.width, encoded.height), (40, 24))

    def test_png_metadata_survives_because_bytes_are_untouched(self):
        from PIL import PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text('parameters', 'a cat\nSteps: 20')
        data = _encode(Image.new('RGB', (8, 8)), 'PNG', pnginfo=info)
        encoded = encode_image_bytes(data)
        with Image.open(io.BytesIO(base64.b64decode(encoded.b64))) as image:
            self.assertEqual(image.info.get('parameters'), 'a cat\nSteps: 20')

    def test_file_with_korean_path(self):
        data = _encode(Image.new('RGB', (12, 6), 'red'), 'PNG')
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, '한글 폴더', '원본.png')
            os.makedirs(os.path.dirname(path))
            with open(path, 'wb') as handle:
                handle.write(data)
            encoded = encode_image_file(path)
        self.assertEqual(base64.b64decode(encoded.b64), data)
        self.assertEqual((encoded.width, encoded.height), (12, 6))


class ReencodeTests(unittest.TestCase):
    def test_alpha_is_flattened_on_white_like_forge(self):
        rgba = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
        rgba.putpixel((0, 0), (255, 0, 0, 255))
        encoded = encode_image_bytes(_encode(rgba, 'PNG'))
        self.assertTrue(encoded.reencoded)
        with Image.open(io.BytesIO(base64.b64decode(encoded.b64))) as image:
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(image.getpixel((3, 3)), (255, 255, 255))

    def test_exif_orientation_is_applied_so_the_mask_matches(self):
        image = Image.new('RGB', (30, 10), 'blue')
        exif = Image.Exif()
        exif[0x0112] = 6   # 90° 회전 — 브라우저는 세워서 보여 준다
        encoded = encode_image_bytes(_encode(image, 'JPEG', exif=exif.tobytes()))
        self.assertTrue(encoded.reencoded)
        self.assertEqual((encoded.width, encoded.height), (10, 30))

    def test_bmp_and_cmyk_are_reencoded_to_png(self):
        for data in (_encode(Image.new('RGB', (5, 5)), 'BMP'),
                     _encode(Image.new('CMYK', (5, 5)), 'JPEG')):
            encoded = encode_image_bytes(data)
            self.assertTrue(encoded.reencoded)
            with Image.open(io.BytesIO(base64.b64decode(encoded.b64))) as image:
                self.assertEqual((image.format, image.mode), ('PNG', 'RGB'))


class ErrorTests(unittest.TestCase):
    def test_broken_or_missing_inputs_raise_user_errors(self):
        with self.assertRaises(ImagePayloadError):
            encode_image_bytes(b'not an image')
        with self.assertRaises(ImagePayloadError):
            encode_image_bytes(b'')
        with self.assertRaises(ImagePayloadError):
            encode_image_file(os.path.join(tempfile.gettempdir(), 'missing-이미지.png'))
        with self.assertRaises(ImagePayloadError):
            decode_base64_image('data:image/png;base64,@@@')
        with self.assertRaises(ImagePayloadError):
            image_size_from_bytes(b'xx')

    def test_strip_data_url(self):
        self.assertEqual(strip_data_url('data:image/png;base64,QUJD'), 'QUJD')
        self.assertEqual(strip_data_url(' QU\nJD '), 'QUJD')
        self.assertEqual(strip_data_url(None), '')


class TargetSizeTests(unittest.TestCase):
    """I2I 요청 크기 — 칸을 비우면(parseInt → null) 보낼 이미지 크기로 (예전 _load_image 동작)."""

    def test_valid_request_size_wins_and_image_is_not_read(self):
        def _never():
            raise AssertionError('크기가 다 있으면 이미지를 읽지 않는다')
        self.assertEqual(resolve_target_size(768, '1024', _never), (768, 1024))
        self.assertEqual(resolve_target_size(512.9, 640, _never), (512, 640))

    def test_missing_or_invalid_sides_fall_back_to_the_image(self):
        for width, height in ((None, None), ('', ''), (float('nan'), 0), (-5, 'abc'), (True, None),
                              (99999, None)):
            with self.subTest(width=width, height=height):
                self.assertEqual(resolve_target_size(width, height, (1344, 768)), (1344, 768))
        # 한 변만 비었으면 그 변만 이미지 크기로 — 예전처럼 이미지 크기를 깔고 payload 로 덮은 결과
        self.assertEqual(resolve_target_size(None, 900, (1344, 768)), (1344, 900))
        self.assertEqual(resolve_target_size(1000, None, lambda: (1344, 768)), (1000, 768))

    def test_unknown_image_size_leaves_the_side_undecided(self):
        def _broken():
            raise ImagePayloadError('I2I: 이미지를 읽을 수 없습니다.')
        self.assertEqual(resolve_target_size(None, None, _broken), (None, None))
        self.assertEqual(resolve_target_size(640, None, None), (640, None))

    def test_image_size_from_base64_reads_the_header(self):
        data = _encode(Image.new('RGB', (37, 21)), 'PNG')
        b64 = base64.b64encode(data).decode('ascii')
        self.assertEqual(image_size_from_base64(b64), (37, 21))
        self.assertEqual(image_size_from_base64('data:image/png;base64,' + b64), (37, 21))
        with self.assertRaises(ImagePayloadError):
            image_size_from_base64('QUJD')


class _Field:
    def __init__(self, value=''):
        self.value = value

    def setText(self, value):
        self.value = value

    def text(self):
        return self.value

    def setPlainText(self, value):
        self.value = value

    def toPlainText(self):
        return self.value

    def setCurrentIndex(self, index):
        self.value = index


class I2IGenerateFromPayloadSizeTests(unittest.TestCase):
    """tabs/i2i_tab.Img2ImgTab.generate_from_payload 가 빈 크기 칸을 이미지 크기로 채운다 (#120)."""

    def _host(self):
        host = type('_Host', (), {})()
        host.main_window = None
        host.current_base64 = None
        host.current_image_path = None
        host.current_reference_base64 = None
        for name in ('prompt_text', 'neg_prompt_text', 'denoise_input', 'steps_input', 'cfg_input',
                     'seed_input', 'resize_combo'):
            setattr(host, name, _Field())
        host.width_input = _Field('1024')    # 숨은 입력칸의 옛 값
        host.height_input = _Field('1024')
        host.generated = 0

        def _on_generate():
            host.generated += 1
        host._on_generate = _on_generate
        return host

    def _run(self, host, payload):
        from tabs.i2i_tab import Img2ImgTab
        Img2ImgTab.generate_from_payload(host, payload)

    def test_path_input_with_cleared_size_uses_the_image_size(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, '갤러리.png')
            Image.new('RGB', (832, 1216)).save(path)
            host = self._host()
            self._run(host, {'image': '', 'image_path': path, 'width': None, 'height': None})
            self.assertEqual((host.width_input.value, host.height_input.value), ('832', '1216'))
            self.assertEqual(host.generated, 1)

    def test_data_url_input_with_cleared_size_uses_the_image_size(self):
        data = _encode(Image.new('RGB', (640, 360)), 'PNG')
        host = self._host()
        self._run(host, {'image': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii'),
                         'image_path': '', 'width': None, 'height': 480})
        self.assertEqual((host.width_input.value, host.height_input.value), ('640', '480'))

    def test_explicit_size_is_kept(self):
        data = _encode(Image.new('RGB', (640, 360)), 'PNG')
        host = self._host()
        self._run(host, {'image': base64.b64encode(data).decode('ascii'), 'width': 1024, 'height': 576})
        self.assertEqual((host.width_input.value, host.height_input.value), ('1024', '576'))


if __name__ == '__main__':
    unittest.main()
