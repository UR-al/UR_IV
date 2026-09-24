"""Vue 인페인트 페이로드 → 백엔드 요청 (core/inpaint_payload) + Qt 접착층 (ui/inpaint_actions) — #33.

예전엔 숨은 InpaintTab 을 거쳐 (1) 마스크 초기값·인페인트 범위가 무시되고 (2) 숨은
steps/cfg/seed 가 쓰이고 (3) data URL 입력에도 이전 이미지 크기(또는 1024²)가 나갔다.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from core.image_payload import ImagePayloadError
from core.inpaint_payload import (
    DEFAULT_INPAINT_AREA,
    DEFAULT_MASK_CONTENT,
    build_inpaint_request,
    parse_seed,
)


def _png(width: int, height: int, color='gray') -> bytes:
    out = io.BytesIO()
    Image.new('RGB', (width, height), color).save(out, format='PNG')
    return out.getvalue()


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


MASK = _data_url(_png(64, 48, 'white'))


class BuildRequestTests(unittest.TestCase):
    def test_data_url_input_uses_its_own_size_not_a_previous_path(self):
        request = build_inpaint_request(
            {'image': _data_url(_png(640, 480)), 'image_path': '', 'mask': MASK},
            resolve_path=lambda raw: self.fail('data URL 입력이면 경로를 보지 않는다'),
        )
        self.assertEqual((request.payload['width'], request.payload['height']), (640, 480))
        self.assertEqual(request.source_path, '')

    def test_path_input_sends_original_bytes_and_real_size(self):
        data = _png(300, 200)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, '생성.png')
            with open(path, 'wb') as handle:
                handle.write(data)
            request = build_inpaint_request(
                {'image': '', 'image_path': path.replace('\\', '/'), 'mask': MASK},
                resolve_path=lambda raw: path,
            )
        self.assertEqual(base64.b64decode(request.payload['init_images'][0]), data)
        self.assertEqual((request.width, request.height), (300, 200))
        self.assertEqual(request.source_path, path)

    def test_vue_mask_options_map_to_forge_fields(self):
        base = {'image': _data_url(_png(8, 8)), 'mask': MASK}
        for content in range(4):
            request = build_inpaint_request({**base, 'mask_content': content, 'inpaint_area': 0})
            self.assertEqual(request.payload['inpainting_fill'], content)
            self.assertIs(request.payload['inpaint_full_res'], False)
        request = build_inpaint_request({**base, 'mask_content': 0, 'inpaint_area': 1})
        self.assertIs(request.payload['inpaint_full_res'], True)

    def test_defaults_match_what_the_hidden_tab_actually_sent(self):
        request = build_inpaint_request({'image': _data_url(_png(8, 8)), 'mask': MASK})
        payload = request.payload
        self.assertEqual(DEFAULT_MASK_CONTENT, 1)
        self.assertEqual(DEFAULT_INPAINT_AREA, 1)
        self.assertEqual(payload['inpainting_fill'], 1)
        self.assertIs(payload['inpaint_full_res'], True)
        self.assertEqual(payload['inpaint_full_res_padding'], 32)
        self.assertEqual(payload['mask_blur'], 4)
        self.assertEqual((payload['steps'], payload['cfg_scale'], payload['seed']), (20, 7.0, -1))
        self.assertEqual(payload['denoising_strength'], 0.75)
        self.assertEqual(payload['alwayson_scripts'], {})

    def test_vue_sampling_values_are_used_and_clamped(self):
        request = build_inpaint_request({
            'image': _data_url(_png(8, 8)), 'mask': MASK,
            'steps': '999', 'cfg': 5.5, 'seed': '1234', 'denoising': 1.7,
            'mask_blur': -3, 'padding': '48',
        })
        payload = request.payload
        self.assertEqual(payload['steps'], 150)
        self.assertEqual(payload['cfg_scale'], 5.5)
        self.assertEqual(payload['seed'], 1234)
        self.assertEqual(payload['denoising_strength'], 1.0)
        self.assertEqual(payload['mask_blur'], 0)
        self.assertEqual(payload['inpaint_full_res_padding'], 48)

    def test_empty_prompts_fall_back_to_t2i(self):
        request = build_inpaint_request(
            {'image': _data_url(_png(8, 8)), 'mask': MASK, 'prompt': '  ', 'negative_prompt': ''},
            main_prompt='1girl', main_negative='lowres',
        )
        self.assertEqual((request.payload['prompt'], request.payload['negative_prompt']), ('1girl', 'lowres'))
        request = build_inpaint_request(
            {'image': _data_url(_png(8, 8)), 'mask': MASK, 'prompt': 'red hat', 'negative_prompt': 'blur'},
            main_prompt='1girl', main_negative='lowres',
        )
        self.assertEqual((request.payload['prompt'], request.payload['negative_prompt']), ('red hat', 'blur'))

    def test_errors_are_user_facing(self):
        with self.assertRaisesRegex(ImagePayloadError, '입력 이미지가 없습니다'):
            build_inpaint_request({'image': '', 'image_path': '', 'mask': MASK})
        with self.assertRaisesRegex(ImagePayloadError, '찾을 수 없습니다'):
            build_inpaint_request({'image_path': 'C:/nope.png', 'mask': MASK}, resolve_path=lambda raw: None)
        with self.assertRaisesRegex(ImagePayloadError, '마스크'):
            build_inpaint_request({'image': _data_url(_png(8, 8)), 'mask': ''})
        with self.assertRaises(ImagePayloadError):
            build_inpaint_request({'image': _data_url(_png(8, 8)), 'mask': 'data:image/png;base64,QUJD'})

    def test_parse_seed(self):
        self.assertEqual(parse_seed('-1'), -1)
        self.assertEqual(parse_seed(''), -1)
        self.assertEqual(parse_seed('abc'), -1)
        self.assertEqual(parse_seed('-7'), -1)
        self.assertEqual(parse_seed(42), 42)
        self.assertEqual(parse_seed(2 ** 40), 2 ** 32 - 1)


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
        self.sent = []

    def send_image(self, path, width, height, seed):
        self.sent.append((path, width, height, seed))


class _Text:
    def __init__(self, value):
        self.value = value

    def toPlainText(self):
        return self.value

    def currentText(self):
        return self.value


class _Worker:
    instances = []

    def __init__(self, model, payload):
        self.model, self.payload = model, payload
        self.finished = _Signal()
        self.running = False
        _Worker.instances.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running


class _Window:
    def __init__(self):
        self.vue_bridge = _Bridge()
        self.total_prompt_display = _Text('main prompt')
        self.neg_prompt_text = _Text('main neg')
        self.model_combo = _Text('model.safetensors')
        self.extensions_applied = []
        self.gallery = []

    def apply_alwayson_extensions(self, payload):
        self.extensions_applied.append(payload)
        payload['alwayson_scripts']['NegPiP'] = {'args': [True]}

    def add_image_to_gallery(self, path):
        # 은퇴한 숨은 PyQt 갤러리 훅 — 결과 처리는 이것을 부르지 않아야 한다(아래 assert)
        self.gallery.append(path)


class InpaintActionTests(unittest.TestCase):
    def setUp(self):
        _Worker.instances.clear()

    def test_start_builds_request_applies_extensions_and_refuses_while_running(self):
        from ui.inpaint_actions import WORKER_ATTR, start_vue_inpaint
        window = _Window()
        payload = {'image': _data_url(_png(96, 64)), 'mask': MASK, 'steps': 30}
        self.assertTrue(start_vue_inpaint(window, payload, worker_factory=_Worker))
        worker = getattr(window, WORKER_ATTR)
        self.assertEqual(worker.model, 'model.safetensors')
        self.assertEqual(worker.payload['steps'], 30)
        self.assertEqual((worker.payload['width'], worker.payload['height']), (96, 64))
        self.assertEqual(worker.payload['prompt'], 'main prompt')
        self.assertIn('NegPiP', worker.payload['alwayson_scripts'])
        # 실행 중 재요청은 거절 — 워커가 둘이 되지 않는다
        self.assertFalse(start_vue_inpaint(window, payload, worker_factory=_Worker))
        self.assertEqual(len(_Worker.instances), 1)
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'warning')

    def test_invalid_payload_notifies_without_starting(self):
        from ui.inpaint_actions import start_vue_inpaint
        window = _Window()
        self.assertFalse(start_vue_inpaint(window, {'image': _data_url(_png(8, 8)), 'mask': ''},
                                           worker_factory=_Worker))
        self.assertEqual(_Worker.instances, [])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'error')

    def test_result_is_saved_as_new_file_and_sent_to_history(self):
        from ui import inpaint_actions
        window = _Window()
        inpaint_actions.start_vue_inpaint(window, {'image': _data_url(_png(20, 10)), 'mask': MASK},
                                          worker_factory=_Worker)
        worker = _Worker.instances[0]
        result = _png(40, 20, 'red')
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(inpaint_actions, '_output_dir', return_value=folder):
            worker.finished.slots[0](result, {'seed': 77})
            files = os.listdir(folder)
            self.assertEqual(len(files), 1)
            with open(os.path.join(folder, files[0]), 'rb') as handle:
                self.assertEqual(handle.read(), result)
        path, width, height, seed = window.vue_bridge.sent[0]
        self.assertTrue(path.endswith('.png'))
        # 해상도는 결과 헤더(40×20) — 요청 크기(20×10)가 아니다
        self.assertEqual((width, height, seed), (40, 20, 77))
        # 숨은 PyQt 갤러리(풀해상도 QPixmap 을 최대 100장 들고 있던 ThumbnailItem)로 보내지 않는다
        self.assertEqual(window.gallery, [])
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'success')

    def test_unreadable_result_falls_back_to_the_request_size(self):
        from ui import inpaint_actions
        window = _Window()
        inpaint_actions.start_vue_inpaint(window, {'image': _data_url(_png(20, 10)), 'mask': MASK},
                                          worker_factory=_Worker)
        worker = _Worker.instances[0]
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(inpaint_actions, '_output_dir', return_value=folder):
            worker.finished.slots[0](b'not an image', {'seed': 5, 'width': 64, 'height': 48})
        _path, width, height, seed = window.vue_bridge.sent[0]
        self.assertEqual((width, height, seed), (64, 48, 5))

    def test_failure_and_cancel_notify(self):
        from ui import inpaint_actions
        window = _Window()
        inpaint_actions.start_vue_inpaint(window, {'image': _data_url(_png(20, 10)), 'mask': MASK},
                                          worker_factory=_Worker)
        slot = _Worker.instances[0].finished.slots[0]
        slot('backend down', {})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1], ('error', '인페인트 실패: backend down'))
        slot('생성 취소됨', {'cancelled': True})
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'info')

    def test_dispatcher_no_longer_routes_through_the_hidden_inpaint_tab(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        start = source.index("elif action == 'generate_inpaint':")
        block = source[start:source.index('elif action', start + 10)]
        self.assertIn('start_vue_inpaint', block)
        self.assertNotIn('inpaint_tab', block)


if __name__ == '__main__':
    unittest.main()
