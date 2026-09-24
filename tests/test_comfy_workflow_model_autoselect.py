"""ComfyUI 연결 때 워크플로 모델 자동 선택이 방금 되살린 사용자 모델을 덮지 않는다.

예전 on_webui_info_loaded 는 restore_backend_combos 로 연결 전 선택·저장값을 되살린 직후
_auto_select_workflow_model 이 워크플로 로더의 체크포인트로 모델 콤보를 무조건 바꿨다. 컴파일러가
그 콤보 값을 로더에 써 넣으므로 시작·재시도·게이트·PRIMARY 전환마다 다음 생성이 사용자가 고르지
않은 워크플로 기본 모델로 돌았고, 종료 저장으로 그 값이 굳었다. 이제 되살린 선택이 있으면 유지하고,
워크플로를 새로 골랐을 때만 워크플로 모델을 따른다.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import config
from core.combo_selection import workflow_checkpoint_index, workflow_model_wins
from ui.combo_restore import restore_backend_combos, snapshot_backend_combos
from ui.widget_proxies import ComboBoxProxy

WORKFLOW_MODEL = 'workflow_A.safetensors'
USER_MODEL = 'selected_B.safetensors'


class _FakeBridge:
    def __init__(self):
        self.values = []

    def _register_proxy(self, widget_id, proxy):
        pass

    def pushWidgetValue(self, widget_id, value):
        self.values.append((widget_id, value))

    def pushWidgetProperty(self, widget_id, prop, value):
        pass


def _api_workflow(loader_class: str, param: str) -> dict:
    return {
        '1': {'class_type': loader_class, 'inputs': {param: WORKFLOW_MODEL}},
        '2': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'a', 'clip': ['1', 1]}},
        '3': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'b', 'clip': ['1', 1]}},
        '4': {'class_type': 'EmptyLatentImage', 'inputs': {'width': 512, 'height': 512, 'batch_size': 1}},
        '5': {'class_type': 'KSampler', 'inputs': {
            'model': ['1', 0], 'positive': ['2', 0], 'negative': ['3', 0], 'latent_image': ['4', 0],
            'seed': 1, 'steps': 20, 'cfg': 7, 'sampler_name': 'euler', 'scheduler': 'normal', 'denoise': 1}},
        '6': {'class_type': 'VAEDecode', 'inputs': {'samples': ['5', 0], 'vae': ['1', 2]}},
        '7': {'class_type': 'SaveImage', 'inputs': {'images': ['6', 0], 'filename_prefix': 'x'}},
    }


class WorkflowCheckpointIndexTests(unittest.TestCase):
    def test_exact_name_then_file_name_ignoring_folder_and_case(self):
        items = ['other.safetensors', 'SDXL\\Anima_Base.safetensors', 'anima_base.safetensors']
        self.assertEqual(workflow_checkpoint_index(items, 'anima_base.safetensors'), 2)
        self.assertEqual(workflow_checkpoint_index(items[:2], 'models/anima_base.safetensors'), 1)
        self.assertEqual(workflow_checkpoint_index(items, 'missing.safetensors'), -1)
        self.assertEqual(workflow_checkpoint_index([], 'a'), -1)
        self.assertEqual(workflow_checkpoint_index(items, ''), -1)


class WorkflowModelWinsTests(unittest.TestCase):
    def test_nothing_restored_lets_the_workflow_model_win(self):
        self.assertTrue(workflow_model_wins(kept_selection=False, workflow_path='a.json',
                                            previous_workflow_path='a.json'))

    def test_restored_selection_is_kept_for_the_same_or_unknown_workflow(self):
        self.assertFalse(workflow_model_wins(kept_selection=True, workflow_path='C:\\wf\\a.json',
                                             previous_workflow_path='c:/wf/a.json'))
        self.assertFalse(workflow_model_wins(kept_selection=True, workflow_path='a.json',
                                             previous_workflow_path=None))

    def test_a_newly_chosen_workflow_brings_its_model(self):
        self.assertTrue(workflow_model_wins(kept_selection=True, workflow_path='b.json',
                                            previous_workflow_path='a.json'))
        self.assertTrue(workflow_model_wins(kept_selection=True, workflow_path='b.json',
                                            previous_workflow_path=''))


class AutoSelectOnConnectTests(unittest.TestCase):
    """실제 ComboBoxProxy·restore_backend_combos·analyze_workflow 로 on_webui_info_loaded 의 순서를 따른다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.selectable = self._write('selectable.json', _api_workflow('CheckpointLoaderSimple', 'ckpt_name'))
        self.locked = self._write('locked_gguf.json', _api_workflow('UnetLoaderGGUF', 'unet_name'))
        self.notices = []
        self.host = SimpleNamespace(
            is_programmatic_change=False,
            model_combo=ComboBoxProxy(_FakeBridge(), 'model_combo'),
            vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(
                emit=lambda level, text: self.notices.append((level, text)))),
        )

    def _write(self, name: str, data: dict) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(data, handle)
        return path

    def _connect(self, workflow_path: str, saved: dict) -> list[str]:
        """on_webui_info_loaded: 스냅숏 → 비우고 채움 → 복원 → 워크플로 모델 자동 선택."""
        from ui.generator_webui import WebUIMixin

        models = [USER_MODEL, WORKFLOW_MODEL]
        preserved = snapshot_backend_combos(self.host)
        self.host.model_combo.clear()
        self.host.model_combo.addItems(models)
        restored = restore_backend_combos(self.host, preserved, load_saved=lambda: saved)
        with mock.patch.object(config, 'COMFYUI_WORKFLOW_PATH', workflow_path):
            WebUIMixin._auto_select_workflow_model(self.host, models, keep_current='model_combo' in restored)
        return restored

    def test_saved_model_survives_startup_connect_with_the_same_workflow(self):
        self.host._model_workflow_path = self.selectable          # load_settings 가 남긴 경로
        restored = self._connect(self.selectable, {'model': USER_MODEL})
        self.assertIn('model_combo', restored)
        self.assertEqual(self.host.model_combo.currentText(), USER_MODEL)
        self.assertTrue(self.host.model_combo.isEnabled())

    def test_live_choice_survives_a_reconnect(self):
        self.host._model_workflow_path = self.selectable
        self._connect(self.selectable, {})                        # 첫 연결 — 되살릴 값 없음 → 워크플로 모델
        self.assertEqual(self.host.model_combo.currentText(), WORKFLOW_MODEL)
        self.host.model_combo._on_vue_changed(USER_MODEL)         # 사용자가 Vue 에서 B 를 고름(저장 전)
        self._connect(self.selectable, {'model': WORKFLOW_MODEL})  # 재시도·PRIMARY 전환
        self.assertEqual(self.host.model_combo.currentText(), USER_MODEL)

    def test_newly_chosen_workflow_brings_its_model_once(self):
        self.host._model_workflow_path = os.path.join(self._tmp.name, 'old.json')
        self._connect(self.selectable, {'model': USER_MODEL})
        self.assertEqual(self.host.model_combo.currentText(), WORKFLOW_MODEL)
        self.host.model_combo._on_vue_changed(USER_MODEL)
        self._connect(self.selectable, {'model': USER_MODEL})     # 같은 워크플로로 다시 연결 — 이제 유지
        self.assertEqual(self.host.model_combo.currentText(), USER_MODEL)

    def test_locked_loader_still_disables_the_combo_when_the_selection_is_kept(self):
        self.host._model_workflow_path = self.locked
        self._connect(self.locked, {'model': USER_MODEL})
        self.assertFalse(self.host.model_combo.isEnabled())
        self.assertEqual(self.host.model_combo.currentText(), USER_MODEL)
        self.assertEqual(self.notices[-1][0], 'warning')

    def test_load_settings_records_the_workflow_the_saved_model_belongs_to(self):
        import backends
        from ui.generator_settings import SettingsMixin

        mixin = object.__new__(SettingsMixin)
        settings = {'backend_type': 'comfyui', 'webui_url': 'http://forge.test:7860',
                    'comfyui_url': 'http://comfy.test:8188', 'comfyui_workflow_path': self.selectable,
                    'comfyui_workflow_img2img_path': ''}
        with mock.patch.object(config, 'WEBUI_API_URL', config.WEBUI_API_URL), \
                mock.patch.object(config, 'COMFYUI_API_URL', config.COMFYUI_API_URL), \
                mock.patch.object(config, 'COMFYUI_WORKFLOW_PATH', config.COMFYUI_WORKFLOW_PATH), \
                mock.patch.object(config, 'COMFYUI_WORKFLOW_IMG2IMG_PATH', config.COMFYUI_WORKFLOW_IMG2IMG_PATH), \
                mock.patch.object(backends, 'is_active_backend', return_value=True):
            mixin._restore_backend_settings(settings)
        self.assertEqual(mixin._model_workflow_path, self.selectable)

    # ── Codex R6#2: 기준 워크플로가 재시작을 건너간다 ──

    def _restore(self, settings: dict):
        """새 세션의 load_settings(_restore_backend_settings) — 전역 config·백엔드는 건드리지 않는다."""
        import backends
        from ui.generator_settings import SettingsMixin

        mixin = object.__new__(SettingsMixin)
        base = {'backend_type': 'comfyui', 'webui_url': 'http://forge.test:7860',
                'comfyui_url': 'http://comfy.test:8188', 'comfyui_workflow_img2img_path': ''}
        with mock.patch.object(config, 'WEBUI_API_URL', config.WEBUI_API_URL), \
                mock.patch.object(config, 'COMFYUI_API_URL', config.COMFYUI_API_URL), \
                mock.patch.object(config, 'COMFYUI_WORKFLOW_PATH', config.COMFYUI_WORKFLOW_PATH), \
                mock.patch.object(config, 'COMFYUI_WORKFLOW_IMG2IMG_PATH', config.COMFYUI_WORKFLOW_IMG2IMG_PATH), \
                mock.patch.object(backends, 'is_active_backend', return_value=True):
            mixin._restore_backend_settings({**base, **settings})
        return mixin

    def _saved_backend_keys(self, *, model_workflow_path, chosen_workflow: str) -> dict:
        """save_settings 의 _build_settings_dict 중 워크플로 두 키 — JSON 을 한 번 거친다(디스크 왕복)."""
        from ui.generator_settings import SettingsMixin

        class _Host(SettingsMixin):
            def __getattr__(self, name):   # 이 테스트가 보지 않는 위젯은 MagicMock
                if name.startswith('__'):
                    raise AttributeError(name)
                value = mock.MagicMock(name=name)
                setattr(self, name, value)
                return value

        host = _Host()
        if model_workflow_path is not None:
            host._model_workflow_path = model_workflow_path
        with mock.patch.object(config, 'COMFYUI_WORKFLOW_PATH', chosen_workflow):
            settings = SettingsMixin._build_settings_dict(host)
        keys = ('comfyui_workflow_path', 'comfyui_model_workflow_path')
        return json.loads(json.dumps({key: settings[key] for key in keys}))

    def test_gate_saved_new_workflow_then_failed_connect_still_brings_its_model_after_restart(self):
        """게이트가 새 워크플로 W2 를 연결 전에 저장 → 연결 실패·종료 → 재시작. 예전엔 기준을
        comfyui_workflow_path(W2)로 다시 만들어 옛 모델이 W2 로더에 써졌다."""
        old_workflow = os.path.join(self._tmp.name, 'old.json')
        first = self._restore({'comfyui_workflow_path': old_workflow})
        self.assertEqual(first._model_workflow_path, old_workflow)

        saved = self._saved_backend_keys(model_workflow_path=first._model_workflow_path,
                                         chosen_workflow=self.selectable)
        self.assertEqual(saved, {'comfyui_workflow_path': self.selectable,
                                 'comfyui_model_workflow_path': old_workflow})

        restarted = self._restore(saved)
        self.assertEqual(restarted._model_workflow_path, old_workflow)
        self.host._model_workflow_path = restarted._model_workflow_path
        self._connect(self.selectable, {'model': USER_MODEL})     # 저장된 옛 모델이 목록에 있어도
        self.assertEqual(self.host.model_combo.currentText(), WORKFLOW_MODEL)
        self.assertEqual(self.host._model_workflow_path, self.selectable)

        # 연결 뒤 저장 → 다음 재시작은 같은 워크플로라 사용자가 고른 모델을 지킨다
        self.host.model_combo._on_vue_changed(USER_MODEL)
        saved = self._saved_backend_keys(model_workflow_path=self.host._model_workflow_path,
                                         chosen_workflow=self.selectable)
        self.host._model_workflow_path = self._restore(saved)._model_workflow_path
        self._connect(self.selectable, {'model': USER_MODEL})
        self.assertEqual(self.host.model_combo.currentText(), USER_MODEL)

    def test_unknown_baseline_is_saved_as_null_and_old_files_fall_back_to_the_workflow_path(self):
        saved = self._saved_backend_keys(model_workflow_path=None, chosen_workflow=self.selectable)
        self.assertIsNone(saved['comfyui_model_workflow_path'])
        self.assertEqual(self._restore(saved)._model_workflow_path, self.selectable)
        self.assertEqual(self._restore({'comfyui_workflow_path': self.selectable})._model_workflow_path,
                         self.selectable)   # 키가 없는 옛 설정 파일 — 예전 동작
        self.assertEqual(self._restore({'comfyui_workflow_path': self.selectable,
                                        'comfyui_model_workflow_path': ''})._model_workflow_path, '')


if __name__ == '__main__':
    unittest.main()
