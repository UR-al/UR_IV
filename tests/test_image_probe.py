"""core.image_payload.probe_image_bytes — encode_image_bytes 와 같은 검사·크기를 PNG 재압축 없이.

I2I 는 GUI 스레드에서 이걸로 요청 크기와 '원본 그대로 보내도 되는지'만 정하고, 재압축은 생성 워커가
한다(core/i2i_payload ``I2IRequest.prepare_payload``). 두 함수가 어긋나면 요청 크기와 실제로 보낸
이미지 크기가 달라진다 — 여기서 짝을 고정한다.
"""
from __future__ import annotations

import io
import unittest

from PIL import Image

from core.image_payload import (
    MAX_INPUT_BYTES,
    ImagePayloadError,
    encode_image_bytes,
    probe_image_bytes,
)


def _save(image: Image.Image, fmt: str, **kwargs) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


def _rotated_jpeg(orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return _save(Image.new('RGB', (40, 20), 'green'), 'JPEG', exif=exif.tobytes())


class ProbeMatchesEncodeTests(unittest.TestCase):
    def test_size_and_passthrough_agree_with_encode(self):
        cases = {
            'rgb png': _save(Image.new('RGB', (30, 10), 'red'), 'PNG'),
            'gray jpeg': _save(Image.new('L', (12, 7)), 'JPEG'),
            'rgba png': _save(Image.new('RGBA', (30, 10), (0, 0, 0, 0)), 'PNG'),
            'palette transparency': _save(Image.new('P', (9, 5)), 'PNG', transparency=0),
            'tiff': _save(Image.new('RGB', (16, 8)), 'TIFF'),
            'bmp': _save(Image.new('RGB', (16, 8)), 'BMP'),
            'gif': _save(Image.new('P', (16, 8)), 'GIF'),
            'apng': _save(Image.new('RGB', (16, 8)), 'PNG', save_all=True,
                          append_images=[Image.new('RGB', (16, 8), 'blue')]),
        }
        for orientation in range(2, 9):
            cases[f'jpeg orientation {orientation}'] = _rotated_jpeg(orientation)
        for name, data in cases.items():
            with self.subTest(name=name):
                probe = probe_image_bytes(data)
                encoded = encode_image_bytes(data)
                self.assertEqual((probe.width, probe.height), (encoded.width, encoded.height))
                self.assertEqual(probe.passthrough, not encoded.reencoded)

    def test_rotation_swaps_the_size(self):
        self.assertEqual((probe_image_bytes(_rotated_jpeg(6)).width, probe_image_bytes(_rotated_jpeg(6)).height),
                         (20, 40))
        probe = probe_image_bytes(_rotated_jpeg(3))
        self.assertEqual((probe.width, probe.height, probe.passthrough), (40, 20, False))


class ProbeErrorTests(unittest.TestCase):
    def test_same_user_messages_as_encode(self):
        for data in (b'', b'not an image', None):
            with self.subTest(data=data):
                with self.assertRaises(ImagePayloadError) as probe_error:
                    probe_image_bytes(data, label='I2I')
                with self.assertRaises(ImagePayloadError) as encode_error:
                    encode_image_bytes(data, label='I2I')
                self.assertEqual(str(probe_error.exception), str(encode_error.exception))

    def test_too_large_input_is_refused_before_decoding(self):
        with self.assertRaisesRegex(ImagePayloadError, '너무 큽니다'):
            probe_image_bytes(b'\0' * (MAX_INPUT_BYTES + 1))


if __name__ == '__main__':
    unittest.main()
