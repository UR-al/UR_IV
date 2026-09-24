"""Vue 업스케일 설정·실행 (core/upscale_settings, ui/upscale_actions, workers/upscale_worker) — #7.

예전엔 숨은 UpscaleTab 을 거쳐 늘 upscaler='(로드 필요)'·2배로 나가 모든 파일이 실패했는데도
성공 토스트와 QMessageBox 가 떴다.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from PIL import Image

from core.upscale_settings import (
    build_upscale_settings,
    comfy_upscale_model_names,
    input_files,
    scale_factor,
    upscaler_choices,
    upscaler_name,
)


class SettingsTests(unittest.TestCase):
    def test_settings_come_only_from_the_vue_payload(self):
        settings = build_upscale_settings(
            {'files': ['a.png'], 'upscaler': 'R-ESRGAN 4x+', 'scale': 2.5}, 'C:/out/upscale')
        self.assertEqual(settings['mode'], 'upscale_only')
        self.assertEqual(settings['upscaler_name'], 'R-ESRGAN 4x+')
        self.assertEqual(settings['scale_mode'], 'factor')
        self.assertEqual(settings['scale_factor'], 2.5)   # 0.5 단위 값도 그대로
        self.assertEqual(settings['output_folder'], 'C:/out/upscale')
        self.assertFalse(settings['ad_enabled'])
        self.assertFalse(settings['sam3_enabled'])

    def test_placeholder_or_empty_upscaler_falls_back_to_lanczos(self):
        for value in ('', None, '  ', '(로드 필요)', 'None'):
            self.assertEqual(upscaler_name(value), 'Lanczos')

    def test_scale_is_parsed_and_clamped(self):
        self.assertEqual(scale_factor('3'), 3.0)
        self.assertEqual(scale_factor(None), 2.0)
        self.assertEqual(scale_factor('abc'), 2.0)
        self.assertEqual(scale_factor(float('nan')), 2.0)
        self.assertEqual(scale_factor(0.2), 1.0)
        self.assertEqual(scale_factor(40), 8.0)

    def test_input_files_drop_blanks_duplicates_and_non_strings(self):
        self.assertEqual(input_files({'files': ['a.png', None, '', 'a.png', 3, ' b.png ']}), ['a.png', 'b.png'])
        self.assertEqual(input_files({}), [])


class ChoiceTests(unittest.TestCase):
    def test_forge_list_drops_none(self):
        self.assertEqual(upscaler_choices('webui', ['None', 'Lanczos', 'Nearest', 'R-ESRGAN 4x+', 'Lanczos']),
                         ['Lanczos', 'Nearest', 'R-ESRGAN 4x+'])

    def test_comfy_list_has_builtins_plus_models(self):
        choices = upscaler_choices('comfyui', ['4x-UltraSharp.pth'])
        self.assertEqual(choices[:5], ['Lanczos', 'Nearest', 'Bilinear', 'Bicubic', 'Area'])
        self.assertEqual(choices[-1], '4x-UltraSharp.pth')

    def test_comfy_object_info_both_formats(self):
        legacy = {'UpscaleModelLoader': {'input': {'required': {'model_name': [['a.pth', 'b.pth']]}}}}
        combo = {'UpscaleModelLoader': {'input': {'required': {'model_name': ['COMBO', {'options': ['c.pth']}]}}}}
        self.assertEqual(comfy_upscale_model_names(legacy), ['a.pth', 'b.pth'])
        self.assertEqual(comfy_upscale_model_names(combo), ['c.pth'])
        self.assertEqual(comfy_upscale_model_names({}), [])
        self.assertEqual(comfy_upscale_model_names(None), [])

    def test_bridge_loader_asks_comfy_for_upscale_models(self):
        from backends import BackendType
        from ui.vue_bridge import VueBridge

        class _Resp:
            status_code = 200

            def json(self):
                return {'UpscaleModelLoader': {'input': {'required': {'model_name': [['x.pth']]}}}}

        backend = mock.Mock(api_url='http://127.0.0.1:8188')
        with mock.patch('backends.get_backend', return_value=backend), \
                mock.patch('backends.get_backend_type', return_value=BackendType.COMFYUI), \
                mock.patch('requests.get', return_value=_Resp()) as get:
            names = json.loads(VueBridge._load_upscalers_json(mock.Mock()))
        self.assertIn('/object_info/UpscaleModelLoader', get.call_args.args[0])
        self.assertEqual(names[0], 'Lanczos')
        self.assertEqual(names[-1], 'x.pth')


def _png(color='red', size=(8, 6)) -> str:
    out = io.BytesIO()
    Image.new('RGB', size, color).save(out, format='PNG')
    return base64.b64encode(out.getvalue()).decode('ascii')


class WorkerTests(unittest.TestCase):
    """BatchUpscaleWorker — 결과를 덮어쓰지 않고, 실패 이유를 실제로 알린다."""

    def _run(self, folder, backend):
        from workers import upscale_worker
        src = os.path.join(folder, '원본.png')
        with open(src, 'wb') as handle:
            handle.write(base64.b64decode(_png()))
        out_dir = os.path.join(folder, 'upscale')
        worker = upscale_worker.BatchUpscaleWorker(
            [src], build_upscale_settings({'upscaler': 'Lanczos', 'scale': 2}, out_dir))
        results = []
        worker.single_finished.connect(lambda i, ok, msg: results.append((i, ok, msg)))
        with mock.patch.object(upscale_worker, 'get_backend', return_value=backend), \
                mock.patch.object(upscale_worker, 'release_before_backend_job'):
            worker.run()
        return src, out_dir, results

    def test_second_run_writes_a_new_file_instead_of_overwriting(self):
        backend = mock.Mock()
        backend.upscale.side_effect = [_png('blue', (16, 12)), _png('green', (16, 12))]
        with tempfile.TemporaryDirectory() as folder:
            _src, out_dir, first = self._run(folder, backend)
            _src, out_dir, second = self._run(folder, backend)
            self.assertEqual(sorted(os.listdir(out_dir)), ['원본_upscaled.png', '원본_upscaled_2.png'])
        self.assertEqual(first[0][:2], (0, True))
        self.assertEqual(second[0][2], '원본_upscaled_2.png')
        settings = backend.upscale.call_args.args[1]
        self.assertEqual((settings['upscaler_name'], settings['scale_factor']), ('Lanczos', 2.0))

    def test_failure_message_reports_the_real_cause(self):
        backend = mock.Mock()
        backend.upscale.side_effect = RuntimeError('허용되지 않은 선택: UpscaleModelLoader.model_name')
        with tempfile.TemporaryDirectory() as folder:
            _src, _out, results = self._run(folder, backend)
        self.assertEqual(results[0][:2], (0, False))
        self.assertIn('허용되지 않은 선택', results[0][2])


class _Signal:
    def __init__(self):
        self.calls = []
        self.slots = []

    def emit(self, *args):
        self.calls.append(args)
        for slot in list(self.slots):
            slot(*args)

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self):
        self.slots.clear()


class _Bridge:
    def __init__(self):
        self.showNotification = _Signal()
        self.batchJobState = _Signal()


class _Window:
    def __init__(self):
        self.vue_bridge = _Bridge()


class _UpscaleWorker:
    instances = []

    def __init__(self, files, settings):
        self.files, self.settings = files, settings
        self.single_finished = _Signal()
        self.all_finished = _Signal()
        self.progress = _Signal()
        self.running = False
        _UpscaleWorker.instances.append(self)

    def start(self):
        self.running = True

    def isRunning(self):
        return self.running


class UpscaleActionTests(unittest.TestCase):
    def setUp(self):
        _UpscaleWorker.instances.clear()

    def test_start_uses_payload_and_notifies_real_outcome_once(self):
        from ui.upscale_actions import WORKER_ATTR, start_vue_upscale
        window = _Window()
        with tempfile.TemporaryDirectory() as root:
            started = start_vue_upscale(window, {'files': ['a.png', 'b.png'], 'upscaler': '', 'scale': 3},
                                        worker_factory=_UpscaleWorker, output_root=root)
            self.assertTrue(os.path.isdir(os.path.join(root, 'upscale')))
        self.assertTrue(started)
        worker = getattr(window, WORKER_ATTR)
        self.assertEqual(worker.settings['upscaler_name'], 'Lanczos')
        self.assertEqual(worker.settings['scale_factor'], 3.0)
        state = json.loads(window.vue_bridge.batchJobState.calls[-1][0])
        self.assertEqual((state['job'], state['running'], state['total']), ('upscale', True, 2))

        # 실행 중 재클릭 — 두 번째 워커를 만들지 않는다
        self.assertFalse(start_vue_upscale(window, {'files': ['c.png']}, worker_factory=_UpscaleWorker,
                                           output_root=tempfile.gettempdir()))
        self.assertEqual(len(_UpscaleWorker.instances), 1)
        self.assertEqual(window.vue_bridge.showNotification.calls[-1][0], 'warning')

        # 전부 실패하면 '완료'가 아니라 error 한 번
        worker.single_finished.emit(0, False, '허용되지 않은 선택')
        worker.single_finished.emit(1, False, '허용되지 않은 선택')
        worker.running = False
        worker.all_finished.emit()
        kinds = [call[0] for call in window.vue_bridge.showNotification.calls]
        self.assertEqual(kinds.count('error'), 1)
        self.assertNotIn('success', kinds)
        self.assertIn('허용되지 않은 선택', window.vue_bridge.showNotification.calls[-1][1])
        final = json.loads(window.vue_bridge.batchJobState.calls[-1][0])
        self.assertEqual((final['running'], final['failed']), (False, 2))

    def test_no_files_is_an_error_without_worker(self):
        from ui.upscale_actions import start_vue_upscale
        window = _Window()
        self.assertFalse(start_vue_upscale(window, {'files': []}, worker_factory=_UpscaleWorker,
                                           output_root=tempfile.gettempdir()))
        self.assertEqual(_UpscaleWorker.instances, [])

    def test_dispatcher_no_longer_routes_through_hidden_tabs(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        for action, func, tab in (('start_upscale', 'start_vue_upscale', 'upscale_tab'),
                                  ('start_batch', 'start_vue_batch', 'batch_tab')):
            start = source.index(f"elif action == '{action}':")
            block = source[start:source.index('elif action', start + 10)]
            self.assertIn(func, block)
            self.assertNotIn(tab, block)


if __name__ == '__main__':
    unittest.main()
