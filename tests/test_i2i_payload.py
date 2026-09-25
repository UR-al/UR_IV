"""Vue I2I 페이로드 → 백엔드 요청 (core/i2i_payload) + Qt 접착층 (ui/i2i_actions).

예전엔 숨은 PyQt Img2ImgTab 을 거쳐 (1) 이전 요청의 입력 이미지가 탭에 남아 이미지가 빠진 요청도
그 이미지로 생성됐고 (2) seed 칸의 글이 int() 예외로 요청을 멈췄고 (3) 실행 중 재요청은 앞 워커의
신호를 끊고 GUI 를 최대 2초 막은 뒤 참조를 버렸다. 크기 칸을 비우면 보낼 이미지 크기를 쓰는 규칙
(#120 — 예전 tests/test_image_payload.I2IGenerateFromPayloadSizeTests)은 그대로 지킨다.
"""
from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core.image_payload import MAX_TARGET_DIMENSION, ImagePayloadError
from core.i2i_payload import (
    DEFAULT_FIDELITY,
    FAMILY_KREA2,
    FAMILY_STANDARD,
    build_i2i_request,
    generation_family,
    input_image_exts,
    parse_i2i_seed,
    resize_mode,
)

ROOT = Path(__file__).resolve().parents[1]

# 예전 Img2ImgTab._on_generate 가 보내던 키 — 표준 경로는 이 모양을 그대로 지킨다.
LEGACY_STANDARD_KEYS = {
    "init_images", "prompt", "negative_prompt", "denoising_strength", "resize_mode", "steps",
    "cfg_scale", "seed", "width", "height", "send_images", "save_images", "alwayson_scripts",
}


def _png(width: int, height: int, color='gray', mode='RGB') -> bytes:
    out = io.BytesIO()
    Image.new(mode, (width, height), color).save(out, format='PNG')
    return out.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + _b64(data)


def _no_path(raw):
    raise AssertionError(f'data URL 입력이면 경로를 보지 않는다: {raw!r}')


class SizeTests(unittest.TestCase):
    """빈 크기 칸은 보낼 이미지의 크기 (#120)."""

    def test_path_input_with_cleared_size_uses_the_image_size(self):
        data = _png(832, 1216)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, '갤러리.png')
            with open(path, 'wb') as handle:
                handle.write(data)
            request = build_i2i_request({'image': '', 'image_path': path, 'width': None, 'height': None},
                                        resolve_path=lambda raw: path)
        self.assertEqual((request.payload['width'], request.payload['height']), (832, 1216))
        # 경로 입력은 원본 바이트 그대로(메인 스레드 재인코딩 없음)
        self.assertEqual(base64.b64decode(request.payload['init_images'][0]), data)
        self.assertEqual(request.source_path, path)

    def test_data_url_input_with_cleared_size_uses_the_image_size(self):
        request = build_i2i_request({'image': _data_url(_png(640, 360)), 'image_path': '',
                                     'width': None, 'height': 480}, resolve_path=_no_path)
        self.assertEqual((request.payload['width'], request.payload['height']), (640, 480))
        self.assertEqual((request.width, request.height), (640, 480))
        self.assertEqual(request.source_path, '')

    def test_explicit_size_is_kept(self):
        request = build_i2i_request({'image': _b64(_png(640, 360)), 'width': 1024, 'height': 576})
        self.assertEqual((request.payload['width'], request.payload['height']), (1024, 576))

    def test_image_side_beyond_the_request_limit_is_clamped(self):
        request = build_i2i_request({'image': _data_url(_png(MAX_TARGET_DIMENSION + 100, 8)),
                                     'width': None, 'height': None})
        self.assertEqual((request.payload['width'], request.payload['height']), (MAX_TARGET_DIMENSION, 8))


class StandardRequestTests(unittest.TestCase):
    def test_defaults_match_the_retired_tab(self):
        request = build_i2i_request({'image': _data_url(_png(8, 8))})
        payload = request.payload
        self.assertEqual(set(payload), LEGACY_STANDARD_KEYS)
        self.assertEqual(request.family, FAMILY_STANDARD)
        self.assertEqual(payload['denoising_strength'], 0.75)
        self.assertEqual(payload['resize_mode'], 0)
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (20, 7.0, -1))
        self.assertIs(payload['send_images'], True)
        self.assertIs(payload['save_images'], True)
        self.assertEqual(payload['alwayson_scripts'], {})

    def test_vue_values_are_used_and_clamped(self):
        payload = build_i2i_request({
            'image': _data_url(_png(8, 8)), 'denoising': 0.42, 'resize_mode': 2,
            'steps': 35, 'cfg': 5.5, 'seed': '1234',
        }).payload
        self.assertEqual((payload['denoising_strength'], payload['resize_mode']), (0.42, 2))
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (35, 5.5, 1234))
        payload = build_i2i_request({
            'image': _data_url(_png(8, 8)), 'denoising': 1.7, 'steps': '999', 'cfg': -3,
        }).payload
        self.assertEqual((payload['denoising_strength'], payload['steps'], payload['cfg_scale']), (1.0, 150, 0.0))

    def test_seed_text_that_is_not_a_number_is_random_instead_of_a_crash(self):
        # 예전 탭은 int('') / int('abc') 로 예외가 나 요청이 통째로 멈췄다
        for raw in ('', 'abc', None, '-1', '-7', float('nan'), True):
            with self.subTest(seed=raw):
                self.assertEqual(build_i2i_request({'image': _data_url(_png(8, 8)), 'seed': raw}).payload['seed'], -1)

    def test_resize_mode(self):
        for raw, expected in ((0, 0), (3, 3), ('1', 1), (4, 0), (-1, 0), (None, 0), ('x', 0), (float('nan'), 0)):
            with self.subTest(raw=raw):
                self.assertEqual(resize_mode(raw), expected)

    def test_empty_prompts_fall_back_to_t2i(self):
        request = build_i2i_request(
            {'image': _data_url(_png(8, 8)), 'prompt': '  ', 'negative_prompt': ''},
            main_prompt='1girl, solo', main_negative='lowres')
        self.assertEqual((request.payload['prompt'], request.payload['negative_prompt']), ('1girl, solo', 'lowres'))
        request = build_i2i_request(
            {'image': _data_url(_png(8, 8)), 'prompt': ' red hat \n', 'negative_prompt': 'blur'},
            main_prompt='1girl', main_negative='lowres')
        self.assertEqual((request.payload['prompt'], request.payload['negative_prompt']), ('red hat', 'blur'))

    def test_transparent_data_url_is_flattened_like_path_input(self):
        rgba = _png(6, 4, (0, 0, 0, 0), mode='RGBA')
        request = build_i2i_request({'image': _data_url(rgba)})
        payload = dict(request.payload)
        request.prepare_payload(payload)   # 워커 스레드가 하는 일
        with Image.open(io.BytesIO(base64.b64decode(payload['init_images'][0]))) as sent:
            self.assertEqual(sent.mode, 'RGB')
            self.assertEqual(sent.getpixel((0, 0)), (255, 255, 255))

    def test_standard_family_ignores_reference_fields(self):
        request = build_i2i_request({'generation_family': 'standard', 'image': _data_url(_png(8, 8)),
                                     'reference_image': 'data:image/png;base64,QUJD',
                                     'reference_path': 'C:/missing.png', 'fidelity': 9},
                                    resolve_path=_no_path)
        self.assertEqual(set(request.payload), LEGACY_STANDARD_KEYS)


class Krea2RequestTests(unittest.TestCase):
    def test_family_value(self):
        for raw, expected in (('krea2', FAMILY_KREA2), (' KREA2 ', FAMILY_KREA2), ('standard', FAMILY_STANDARD),
                              ('', FAMILY_STANDARD), (None, FAMILY_STANDARD), ('flux', FAMILY_STANDARD)):
            with self.subTest(raw=raw):
                self.assertEqual(generation_family(raw), expected)

    def test_krea2_fields_and_fidelity(self):
        base = {'generation_family': 'krea2', 'image': _data_url(_png(8, 8)), 'steps': 15, 'cfg': 1}
        request = build_i2i_request(base)
        self.assertEqual(request.family, FAMILY_KREA2)
        self.assertEqual(request.payload['_generation_family'], 'krea2')
        self.assertEqual(request.payload['krea2_fidelity'], DEFAULT_FIDELITY)
        self.assertNotIn('krea2_reference_image', request.payload)
        self.assertEqual((request.payload['steps'], request.payload['cfg_scale']), (15, 1.0))
        for raw, expected in (('7.5', 7.5), (99, 12.0), (0.1, 0.5), (float('nan'), DEFAULT_FIDELITY), (None, DEFAULT_FIDELITY)):
            with self.subTest(fidelity=raw):
                self.assertEqual(build_i2i_request({**base, 'fidelity': raw}).payload['krea2_fidelity'], expected)

    def test_inline_reference_is_sent_as_is(self):
        reference = _b64(_png(5, 5, 'red'))
        request = build_i2i_request({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)),
                                     'reference_image': 'data:image/png;base64,' + reference})
        self.assertEqual(request.payload['krea2_reference_image'], reference)

    def test_reference_path_sends_the_original_bytes(self):
        reference = _png(5, 5, (1, 2, 3, 128), mode='RGBA')   # 다시 인코딩하지 않는다(예전과 같다)
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, '참조.png')
            with open(path, 'wb') as handle:
                handle.write(reference)
            request = build_i2i_request({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)),
                                         'reference_image': '', 'reference_path': path},
                                        resolve_path=lambda raw: path)
        self.assertEqual(base64.b64decode(request.payload['krea2_reference_image']), reference)

    def test_bad_reference_is_an_error_not_a_silent_drop(self):
        base = {'generation_family': 'krea2', 'image': _data_url(_png(8, 8))}
        with self.assertRaisesRegex(ImagePayloadError, '참조'):
            build_i2i_request({**base, 'reference_path': 'C:/nope.png'}, resolve_path=lambda raw: None)
        with self.assertRaisesRegex(ImagePayloadError, '참조'):
            build_i2i_request({**base, 'reference_image': 'data:image/png;base64,QUJD'})


class ErrorTests(unittest.TestCase):
    def test_errors_are_user_facing(self):
        with self.assertRaisesRegex(ImagePayloadError, '입력 이미지가 없습니다'):
            build_i2i_request({'image': '', 'image_path': ''})
        with self.assertRaisesRegex(ImagePayloadError, '입력 이미지가 없습니다'):
            build_i2i_request(None)
        with self.assertRaisesRegex(ImagePayloadError, '찾을 수 없습니다'):
            build_i2i_request({'image_path': 'C:/nope.png'}, resolve_path=lambda raw: None)
        with self.assertRaisesRegex(ImagePayloadError, 'I2I'):
            build_i2i_request({'image': 'data:image/png;base64,QUJD'})

    def test_default_resolver_refuses_non_image_paths(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, 'secret.txt')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('not an image')
            with self.assertRaises(ImagePayloadError) as caught:
                build_i2i_request({'image_path': path})
        # 파일은 있다 — '찾을 수 없습니다'가 아니라 막힌 형식이라고 알린다
        self.assertEqual(str(caught.exception), 'I2I: 지원하지 않는 파일 형식입니다 (.txt)')


def _write_image(path: str, fmt: str, *, size=(32, 16), mode='RGB', **save) -> bytes:
    Image.new(mode, size, 'red').save(path, format=fmt, **save)
    with open(path, 'rb') as handle:
        return handle.read()


class GalleryFormatPathTests(unittest.TestCase):
    """갤러리가 보여 주는 형식(.tif·.apng·.avif …)은 기본 경로 판정(check_input_path)으로도 열린다.

    예전 _default_resolver 는 safe_input_path 기본 목록(.tiff 는 있고 .tif·.apng 는 없다)을 써서
    send_to_i2i 로 온 .tif 가 'I2I: 입력 이미지 파일을 찾을 수 없습니다'로 실패했다.
    """

    def test_input_extensions_cover_every_gallery_image_format(self):
        exts = input_image_exts()
        for ext in ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.apng', '.bmp', '.tif', '.tiff', '.avif'):
            with self.subTest(ext=ext):
                self.assertIn(ext, exts)
        from core.thumb_cache import THUMB_SOURCE_EXTS
        self.assertEqual(exts, THUMB_SOURCE_EXTS)

    def test_tif_and_apng_paths_generate(self):
        with tempfile.TemporaryDirectory() as root:
            cases = {
                '한글 폴더.tif': _write_image(os.path.join(root, '한글 폴더.tif'), 'TIFF'),
                'scan.tiff': _write_image(os.path.join(root, 'scan.tiff'), 'TIFF'),
                'anim.apng': _write_image(os.path.join(root, 'anim.apng'), 'PNG', save_all=True,
                                          append_images=[Image.new('RGB', (32, 16), 'blue')]),
            }
            from PIL import features
            if features.check('avif'):
                cases['photo.avif'] = _write_image(os.path.join(root, 'photo.avif'), 'AVIF')
            for name in cases:
                with self.subTest(name=name):
                    path = os.path.join(root, name)
                    request = build_i2i_request({'image_path': path, 'width': None, 'height': None})
                    self.assertEqual(os.path.normcase(request.source_path), os.path.normcase(os.path.realpath(path)))
                    self.assertEqual((request.width, request.height), (32, 16))
                    # TIFF·애니메이션은 RGB PNG 로 다시 인코딩 — 워커 스레드 몫
                    self.assertTrue(request.needs_prepare)
                    payload = dict(request.payload)
                    request.prepare_payload(payload)
                    with Image.open(io.BytesIO(base64.b64decode(payload['init_images'][0]))) as sent:
                        self.assertEqual((sent.format, sent.mode, sent.size), ('PNG', 'RGB', (32, 16)))

    def test_krea2_reference_tif_path_is_sent_as_is(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, '참조.tif')
            raw = _write_image(path, 'TIFF', size=(5, 5))
            request = build_i2i_request({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)),
                                         'reference_image': '', 'reference_path': path})
        self.assertEqual(base64.b64decode(request.payload['krea2_reference_image']), raw)

    def test_messages_tell_missing_apart_from_blocked(self):
        with tempfile.TemporaryDirectory() as root:
            folder = os.path.join(root, 'folder.png')
            os.mkdir(folder)
            text = os.path.join(root, 'notes.txt')
            with open(text, 'w', encoding='utf-8') as handle:
                handle.write('x')
            image = os.path.join(root, 'ok.png')
            _write_image(image, 'PNG')
            cases = (
                ({'image_path': os.path.join(root, 'gone.png')}, 'I2I: 입력 이미지 파일을 찾을 수 없습니다'),
                ({'image_path': text}, 'I2I: 지원하지 않는 파일 형식입니다 (.txt)'),
                ({'image_path': os.path.join(root, 'no_extension')}, 'I2I: 지원하지 않는 파일 형식입니다 (확장자 없음)'),
                ({'image_path': folder}, 'I2I: 이미지 파일이 아닙니다'),
                ({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)),
                  'reference_path': os.path.join(root, 'gone.tif')}, 'I2I 아이덴티티 참조: 파일을 찾을 수 없습니다'),
                ({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)), 'reference_path': text},
                 'I2I 아이덴티티 참조: 지원하지 않는 파일 형식입니다 (.txt)'),
            )
            for payload, message in cases:
                with self.subTest(message=message):
                    with self.assertRaises(ImagePayloadError) as caught:
                        build_i2i_request(payload)
                    self.assertEqual(str(caught.exception), message)
            with mock.patch('core.path_safety._is_forbidden', return_value=True):
                with self.assertRaises(ImagePayloadError) as caught:
                    build_i2i_request({'image_path': image})
            self.assertEqual(str(caught.exception), 'I2I: 시스템 폴더의 파일은 열 수 없습니다')


class DeferredEncodingTests(unittest.TestCase):
    """PNG 재압축(2048² RGBA 0.4초+)은 GUI 스레드의 build_i2i_request 가 아니라 워커의 prepare_payload 몫."""

    def test_passthrough_input_is_ready_without_prepare(self):
        data = _png(12, 8)
        request = build_i2i_request({'image': _data_url(data)})
        self.assertFalse(request.needs_prepare)
        self.assertEqual(request.payload['init_images'], [_b64(data)])
        payload = dict(request.payload)
        request.prepare_payload(payload)   # 할 일이 없다
        self.assertEqual(payload, request.payload)

    def test_reencode_is_not_done_while_building(self):
        rgba = _png(40, 30, (10, 20, 30, 128), mode='RGBA')
        with mock.patch('core.i2i_payload.encode_image_bytes',
                        side_effect=AssertionError('GUI 스레드에서 재인코딩하면 안 된다')):
            request = build_i2i_request({'image': _data_url(rgba), 'width': None, 'height': None})
        self.assertTrue(request.needs_prepare)
        self.assertEqual(request.payload['init_images'], [])
        self.assertEqual((request.width, request.height), (40, 30))
        payload = dict(request.payload)
        request.prepare_payload(payload)
        self.assertEqual(len(payload['init_images']), 1)
        self.assertEqual(request.payload['init_images'], [])   # 원본 요청은 그대로

    def test_rotated_exif_size_matches_the_reencoded_image(self):
        exif = Image.Exif()
        exif[0x0112] = 6   # 90° 회전 — 브라우저·백엔드가 보는 크기는 가로세로가 바뀐다
        out = io.BytesIO()
        Image.new('RGB', (40, 20), 'green').save(out, format='JPEG', exif=exif.tobytes())
        request = build_i2i_request({'image': _b64(out.getvalue()), 'width': None, 'height': None})
        self.assertEqual((request.width, request.height), (20, 40))
        payload = dict(request.payload)
        request.prepare_payload(payload)
        with Image.open(io.BytesIO(base64.b64decode(payload['init_images'][0]))) as sent:
            self.assertEqual(sent.size, (20, 40))

    def test_broken_pixels_fail_in_prepare_with_a_user_message(self):
        noisy = Image.frombytes('RGBA', (64, 64), bytes(range(256)) * 64)
        out = io.BytesIO()
        noisy.save(out, format='PNG')
        truncated = out.getvalue()[:len(out.getvalue()) // 2]   # 헤더는 멀쩡, 픽셀 데이터가 잘림
        request = build_i2i_request({'image': _b64(truncated)})
        with self.assertRaisesRegex(ImagePayloadError, '^I2I: '):
            request.prepare_payload(dict(request.payload))


class SeedTests(unittest.TestCase):
    """상한을 넘는 seed 는 조용히 자르지 않고 알린다 — 예전 탭은 int() 값 그대로 보냈다."""

    def test_seeds_within_the_family_range_pass_through_unchanged(self):
        for raw, family, expected in (
            ('1234', 'standard', 1234), (2 ** 40, 'standard', 2 ** 40), ('5000000000', 'standard', 5000000000),
            (str(2 ** 64 - 1), 'standard', 2 ** 64 - 1), ('12.0', 'standard', 12),
            (2 ** 32 - 1, 'krea2', 2 ** 32 - 1), ('-1', 'krea2', -1), ('', 'krea2', -1),
        ):
            with self.subTest(raw=raw, family=family):
                self.assertEqual(parse_i2i_seed(raw, family), expected)

    def test_seed_above_the_limit_is_an_error_not_a_clamp(self):
        for raw, family, limit in ((2 ** 64, 'standard', 2 ** 64 - 1), ('4294967296', 'krea2', 2 ** 32 - 1)):
            with self.subTest(raw=raw, family=family):
                with self.assertRaises(ImagePayloadError) as caught:
                    parse_i2i_seed(raw, family)
                self.assertEqual(str(caught.exception),
                                 f'I2I: 시드는 -1(랜덤) 또는 0~{limit} 이어야 합니다 (받은 값: {int(raw)})')

    def test_request_keeps_large_standard_seed_and_rejects_large_krea2_seed(self):
        self.assertEqual(build_i2i_request({'image': _data_url(_png(8, 8)), 'seed': 5000000000}).payload['seed'],
                         5000000000)
        with self.assertRaisesRegex(ImagePayloadError, '시드'):
            build_i2i_request({'generation_family': 'krea2', 'image': _data_url(_png(8, 8)), 'seed': 2 ** 40})


class VuePayloadContractTests(unittest.TestCase):
    def test_every_field_i2iview_sends_is_read(self):
        view = (ROOT / 'frontend' / 'src' / 'views' / 'I2IView.vue').read_text(encoding='utf-8')
        start = view.index("requestAction('generate_i2i', {")
        body = view[start:view.index('})', start)]
        sent = set(re.findall(r'^\s*([a-z_]+):', body, re.MULTILINE))
        self.assertEqual(sent, {
            'generation_family', 'image', 'image_path', 'reference_image', 'reference_path', 'prompt',
            'negative_prompt', 'denoising', 'fidelity', 'resize_mode', 'width', 'height', 'steps', 'cfg', 'seed',
        })
        source = (ROOT / 'core' / 'i2i_payload.py').read_text(encoding='utf-8')
        unread = sorted(key for key in sent if f'data.get("{key}"' not in source)
        self.assertEqual(unread, [])


# ── Qt 접착층 (ui/i2i_actions) ────────────────────────────────────────────────

class _Signal:
    def __init__(self):
        self.calls = []
        self.slots = []

    def emit(self, *args):
        self.calls.append(args)

    def connect(self, slot):
        self.slots.append(slot)


class _Bridge:
    def __init__(self):
        self.showNotification = _Signal()
        self.i2iJobState = _Signal()
        self.sent = []

    def send_image(self, path, width, height, seed):
        self.sent.append((path, width, height, seed))

    def job_states(self):
        import json
        return [json.loads(args[0]) for args in self.i2iJobState.calls]


class _Text:
    def __init__(self, value):
        self.value = value

    def toPlainText(self):
        return self.value

    def currentText(self):
        return self.value


class _Worker:
    instances = []

    def __init__(self, model, payload, *, prepare=None):
        self.model, self.payload, self.prepare = model, payload, prepare
        self.finished = _Signal()
        self.running = False
        self.cancel_calls = 0
        self.wait_calls = []
        self.wait_finishes = True   # wait() 안에 run() 이 돌아오는지
        _Worker.instances.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running

    def cancel(self):
        self.cancel_calls += 1

    def wait(self, ms):
        self.wait_calls.append(ms)
        if self.wait_finishes:
            self.running = False
        return not self.running


class _Window:
    def __init__(self):
        self.vue_bridge = _Bridge()
        self.total_prompt_display = _Text('main prompt')
        self.neg_prompt_text = _Text('main neg')
        self.model_combo = _Text('model.safetensors')
        self.extensions_applied = []

    def apply_alwayson_extensions(self, payload):
        self.extensions_applied.append(payload)
        payload['alwayson_scripts']['NegPiP'] = {'args': [True]}


class I2IActionTests(unittest.TestCase):
    def setUp(self):
        _Worker.instances.clear()

    def _start(self, window, payload):
        from ui.i2i_actions import start_vue_i2i
        return start_vue_i2i(window, payload, worker_factory=_Worker)

    def test_start_builds_request_applies_extensions_and_refuses_while_running(self):
        from ui.i2i_actions import WORKER_ATTR
        window = _Window()
        payload = {'image': _data_url(_png(96, 64)), 'steps': 30, 'width': None, 'height': None}
        self.assertTrue(self._start(window, payload))
        worker = getattr(window, WORKER_ATTR)
        self.assertIs(worker, _Worker.instances[0])
        self.assertTrue(worker.running)
        self.assertEqual(worker.model, 'model.safetensors')
        self.assertEqual(worker.payload['steps'], 30)
        self.assertEqual((worker.payload['width'], worker.payload['height']), (96, 64))
        self.assertEqual((worker.payload['prompt'], worker.payload['negative_prompt']), ('main prompt', 'main neg'))
        self.assertIn('NegPiP', worker.payload['alwayson_scripts'])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('info', 'I2I 생성 중… (96×64)'))
        # 실행 중 재요청은 거절 — 앞 워커를 끊거나 버리지 않는다
        self.assertFalse(self._start(window, payload))
        self.assertEqual(len(_Worker.instances), 1)
        self.assertIs(getattr(window, WORKER_ATTR), worker)
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'warning')
        # 끝나면 다음 요청을 받는다
        worker.running = False
        self.assertTrue(self._start(window, payload))
        self.assertEqual(len(_Worker.instances), 2)

    def test_krea2_request_reaches_the_worker(self):
        window = _Window()
        self.assertTrue(self._start(window, {'generation_family': 'krea2', 'image': _data_url(_png(8, 8)),
                                             'fidelity': 6, 'width': 512, 'height': 512}))
        payload = _Worker.instances[0].payload
        self.assertEqual((payload['_generation_family'], payload['krea2_fidelity']), ('krea2', 6.0))
        self.assertEqual(window.vue_bridge.showNotification.calls[-1],
                         ('info', 'Krea2 아이덴티티 편집 중… (512×512)'))

    def test_invalid_payload_notifies_without_starting(self):
        window = _Window()
        self.assertFalse(self._start(window, {'image': '', 'image_path': ''}))
        self.assertEqual(_Worker.instances, [])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('error', 'I2I: 입력 이미지가 없습니다'))

    def test_extension_failure_does_not_block_generation(self):
        window = _Window()

        def _boom(payload):
            raise RuntimeError('adetailer missing')
        window.apply_alwayson_extensions = _boom
        self.assertTrue(self._start(window, {'image': _data_url(_png(8, 8))}))
        self.assertEqual(_Worker.instances[0].payload['alwayson_scripts'], {})

    def test_window_without_optional_parts_still_works(self):
        window = type('_Bare', (), {})()
        self.assertTrue(self._start(window, {'image': _data_url(_png(8, 8)), 'prompt': 'cat'}))
        self.assertEqual(_Worker.instances[0].model, '')
        self.assertEqual(_Worker.instances[0].payload['prompt'], 'cat')

    def test_result_is_saved_as_new_file_and_sent_to_history(self):
        from ui import i2i_actions
        window = _Window()
        self._start(window, {'image': _data_url(_png(20, 10))})
        slot = _Worker.instances[0].finished.slots[0]
        result = _png(40, 20, 'red')
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(i2i_actions, '_output_dir', return_value=folder), \
                mock.patch.object(i2i_actions.time, 'time', return_value=1700000000), \
                mock.patch.object(i2i_actions.random, 'randint', return_value=123):
            slot(result, {'seed': 77})
            slot(bytearray(result), {'seed': 78})   # 같은 이름이어도 덮어쓰지 않는다
            files = sorted(os.listdir(folder))
            self.assertEqual(len(files), 2)
            for name in files:
                with open(os.path.join(folder, name), 'rb') as handle:
                    self.assertEqual(handle.read(), result)
        path, width, height, seed = window.vue_bridge.sent[0]
        self.assertTrue(os.path.basename(path).startswith('i2i_1700000000_123'))
        # 해상도는 결과 헤더(40×20) — 요청 크기(20×10)가 아니다
        self.assertEqual((width, height, seed), (40, 20, 77))
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('success', 'I2I 생성 완료'))

    def test_unreadable_result_falls_back_to_the_request_size(self):
        from ui import i2i_actions
        window = _Window()
        self._start(window, {'image': _data_url(_png(20, 10))})
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(i2i_actions, '_output_dir', return_value=folder):
            _Worker.instances[0].finished.slots[0](b'not an image', {'seed': 5, 'width': 64, 'height': 48})
        _path, width, height, seed = window.vue_bridge.sent[0]
        self.assertEqual((width, height, seed), (64, 48, 5))

    def test_save_failure_is_reported(self):
        from ui import i2i_actions
        window = _Window()
        self._start(window, {'image': _data_url(_png(8, 8))})
        with mock.patch.object(i2i_actions, 'save_i2i_result', side_effect=OSError('disk full')):
            _Worker.instances[0].finished.slots[0](_png(8, 8), {})
        self.assertEqual(window.vue_bridge.sent, [])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('error', 'I2I 결과 저장 실패: disk full'))

    def test_failure_and_cancel_notify(self):
        window = _Window()
        self._start(window, {'image': _data_url(_png(8, 8))})
        slot = _Worker.instances[0].finished.slots[0]
        slot('backend down', {})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('error', 'I2I 생성 실패: backend down'))
        slot('생성 취소됨', {'cancelled': True})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('info', 'I2I 생성이 취소되었습니다'))
        slot(b'', None)
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'error')
        self.assertEqual(window.vue_bridge.sent, [])

    def test_dispatcher_routes_generate_i2i_to_the_new_glue(self):
        source = (ROOT / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        start = source.index("elif action == 'generate_i2i':")
        block = source[start:source.index('elif action', start + 10)]
        self.assertIn('start_vue_i2i(self, payload)', block)
        self.assertNotIn('i2i_tab', block)

    def test_reencode_is_handed_to_the_worker(self):
        window = _Window()
        self.assertTrue(self._start(window, {'image': _data_url(_png(8, 8, (0, 0, 0, 0), mode='RGBA'))}))
        worker = _Worker.instances[0]
        self.assertEqual(worker.payload['init_images'], [])
        self.assertTrue(callable(worker.prepare))
        payload = dict(worker.payload)
        worker.prepare(payload)
        self.assertEqual(len(payload['init_images']), 1)
        # 원본 그대로 보내는 입력이면 워커가 할 일이 없다
        _Worker.instances.clear()
        self.assertTrue(self._start(_Window(), {'image': _data_url(_png(8, 8))}))
        self.assertIsNone(_Worker.instances[0].prepare)


class I2IJobStateAndCancelTests(unittest.TestCase):
    """실행 중 재요청은 거절한다 — 그래서 화면에 진행 상태와 취소 버튼(cancel_i2i)이 있어야 한다.

    예전엔 거절만 있고 _vue_i2i_worker 를 취소할 길이 없어(cancel_generation 은 gen_worker 만 본다),
    멈춘 백엔드 호출 하나가 이후 I2I 를 백엔드 타임아웃(Forge img2img 600초)까지 막았다.
    """

    def setUp(self):
        _Worker.instances.clear()

    def _start(self, window, payload=None):
        from ui.i2i_actions import start_vue_i2i
        return start_vue_i2i(window, payload or {'image': _data_url(_png(8, 8))}, worker_factory=_Worker)

    def test_state_follows_start_and_finish(self):
        window = _Window()
        self.assertTrue(self._start(window))
        self.assertEqual(window.vue_bridge.job_states()[-1], {'running': True, 'cancelling': False})
        _Worker.instances[0].finished.slots[0]('backend down', {})
        self.assertEqual(window.vue_bridge.job_states()[-1], {'running': False, 'cancelling': False})

    def test_refusal_resends_running_state_and_says_how_to_stop(self):
        from ui.i2i_actions import BUSY_MESSAGE
        window = _Window()
        self._start(window)
        window.vue_bridge.i2iJobState.calls.clear()
        self.assertFalse(self._start(window))
        self.assertEqual(window.vue_bridge.job_states(), [{'running': True, 'cancelling': False}])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('warning', BUSY_MESSAGE))
        self.assertIn('취소', BUSY_MESSAGE)

    def test_cancel_stops_the_running_worker_and_unblocks_the_next_request(self):
        from ui.i2i_actions import cancel_vue_i2i
        window = _Window()
        self._start(window)
        worker = _Worker.instances[0]
        self.assertTrue(cancel_vue_i2i(window))
        self.assertEqual(worker.cancel_calls, 1)   # 플래그 + 백엔드 interrupt(_CancellableMixin)
        self.assertEqual(window.vue_bridge.job_states()[-1], {'running': True, 'cancelling': True})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('info', 'I2I 생성을 취소하는 중…'))
        # 취소 중 재요청도 거절 — 상태는 '취소 중'으로 다시 알린다
        self.assertFalse(self._start(window))
        self.assertEqual(window.vue_bridge.job_states()[-1], {'running': True, 'cancelling': True})
        # 워커가 cancelled 로 끝나면 풀린다
        worker.finished.slots[0]('생성 취소됨', {'cancelled': True})
        self.assertEqual(window.vue_bridge.job_states()[-1], {'running': False, 'cancelling': False})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('info', 'I2I 생성이 취소되었습니다'))
        worker.running = False
        self.assertTrue(self._start(window))
        self.assertEqual(len(_Worker.instances), 2)

    def test_cancel_without_a_running_job_resets_the_view(self):
        from ui.i2i_actions import cancel_vue_i2i
        window = _Window()
        self.assertFalse(cancel_vue_i2i(window))
        self.assertEqual(window.vue_bridge.job_states(), [{'running': False, 'cancelling': False}])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('info', '취소할 I2I 생성이 없습니다'))

    def test_finished_worker_still_returning_from_run_is_awaited_not_refused(self):
        # finished 는 run() 안에서 나온다 — 그 직후 요청을 '진행 중'으로 거절하면 화면이 다시 '실행 중'이
        # 되고 풀어 줄 끝 알림은 이미 지나갔다. 대신 run() 복귀를 기다렸다가 교체한다(GC 크래시 방지).
        from ui.i2i_actions import FINISHING_WAIT_MS, WORKER_ATTR, cancel_vue_i2i, is_running
        window = _Window()
        self._start(window)
        first = _Worker.instances[0]
        first.finished.slots[0]('backend down', {})
        self.assertTrue(first.running)            # 스레드는 아직 돌아오는 중
        self.assertFalse(is_running(window))
        self.assertFalse(cancel_vue_i2i(window))  # 취소할 것도 없다
        self.assertEqual(first.cancel_calls, 0)
        self.assertTrue(self._start(window))
        self.assertEqual(first.wait_calls, [FINISHING_WAIT_MS])
        self.assertIs(getattr(window, WORKER_ATTR), _Worker.instances[1])

    def test_worker_that_does_not_return_is_kept_and_the_request_refused(self):
        from ui.i2i_actions import FINISHING_MESSAGE, WORKER_ATTR
        window = _Window()
        self._start(window)
        first = _Worker.instances[0]
        first.finished.slots[0]('backend down', {})
        first.wait_finishes = False
        window.vue_bridge.i2iJobState.calls.clear()
        self.assertFalse(self._start(window))
        self.assertIs(getattr(window, WORKER_ATTR), first)   # 실행 중 QThread 참조를 버리지 않는다
        self.assertEqual(len(_Worker.instances), 1)
        # 작업은 이미 끝났다 — 화면을 '실행 중'으로 묶으면 풀어 줄 알림이 다시 오지 않는다
        self.assertEqual(window.vue_bridge.job_states(), [{'running': False, 'cancelling': False}])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('warning', FINISHING_MESSAGE))

    def test_stale_worker_finish_does_not_clear_a_newer_job(self):
        from ui import i2i_actions
        window = _Window()
        self._start(window)
        old = _Worker.instances[0]
        old.running = False
        self._start(window)
        window.vue_bridge.i2iJobState.calls.clear()
        i2i_actions._on_finished(window, old, 'late', {})
        self.assertEqual(window.vue_bridge.job_states(), [])

    def test_dispatcher_and_shutdown_cover_the_i2i_worker(self):
        source = (ROOT / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        start = source.index("elif action == 'cancel_i2i':")
        block = source[start:source.index('elif action', start + 10)]
        self.assertIn('cancel_vue_i2i(self)', block)
        # 종료 때도 I2I·인페인트 워커를 취소(+interrupt)하고 잠깐 기다린다 — 예전엔 gen_worker 만 봤다
        shutdown = source[source.index("for _name in ('gen_worker'"):]
        shutdown = shutdown[:shutdown.index(':')]
        self.assertIn("'_vue_i2i_worker'", shutdown)
        self.assertIn("'_vue_inpaint_worker'", shutdown)

    def test_i2i_view_listens_to_the_state_and_sends_cancel(self):
        view = (ROOT / 'frontend' / 'src' / 'views' / 'I2IView.vue').read_text(encoding='utf-8')
        self.assertIn("onBackendEvent('i2iJobState'", view)
        self.assertIn("requestAction('cancel_i2i'", view)
        self.assertIn(':disabled="!imageSrc || jobState.running"', view)


class Img2ImgWorkerPrepareTests(unittest.TestCase):
    """Img2ImgFlowWorker(prepare=…) — 입력 이미지 재압축을 워커 스레드에서 백엔드 호출 전에 한다."""

    def _run(self, worker, backend):
        from contextlib import contextmanager

        class _Coordinator:
            @contextmanager
            def reserve(self, *args, **kwargs):
                yield

        emitted = []
        worker.finished.connect(lambda result, info: emitted.append((result, info)))
        with mock.patch('workers.generation_worker.get_backend', return_value=backend), \
                mock.patch('workers.generation_worker.get_generation_coordinator', return_value=_Coordinator()):
            worker.run()
        self.assertEqual(len(emitted), 1)
        return emitted[0]

    def _backend(self):
        from backends.base import GenerationResult

        class _Backend:
            payloads = []

            def img2img(self, model, payload, progress_callback=None, cancel_check=None):
                self.payloads.append(dict(payload))
                return GenerationResult(success=True, image_data=b'result', info={})
        backend = _Backend()
        backend.payloads = []
        return backend

    def test_prepared_payload_reaches_the_backend(self):
        from workers.generation_worker import Img2ImgFlowWorker
        request = build_i2i_request({'image': _data_url(_png(8, 8, (0, 0, 0, 0), mode='RGBA'))})
        backend = self._backend()
        worker = Img2ImgFlowWorker('model', request.payload, prepare=request.prepare_payload)
        result, _info = self._run(worker, backend)
        self.assertEqual(result, b'result')
        self.assertEqual(len(backend.payloads[0]['init_images']), 1)
        self.assertEqual(request.payload['init_images'], [])   # 워커는 사본을 채운다

    def test_prepare_failure_is_reported_without_calling_the_backend(self):
        from workers.generation_worker import Img2ImgFlowWorker

        def _broken(payload):
            raise ImagePayloadError('I2I: 이미지를 읽을 수 없습니다.')
        backend = self._backend()
        result, info = self._run(Img2ImgFlowWorker('model', {'init_images': []}, prepare=_broken), backend)
        self.assertEqual((result, info), ('I2I: 이미지를 읽을 수 없습니다.', {}))
        self.assertEqual(backend.payloads, [])


if __name__ == '__main__':
    unittest.main()
