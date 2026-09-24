"""설정 저장·연결이 load_settings 전체를 다시 돌리지 않는다 (감사 #29).

- 저장 200ms 뒤·백엔드 연결마다 load_settings 가 다시 돌아 같은 URL 백엔드를 재생성(BACKEND_URL_CHANGED
  → XYZ 축 초기화, Comfy preflight 초기화)했고, _sync_slider 가 int() 로 잘라 cfg 5.3→5.0 처럼
  값을 한 칸씩 내렸고, 레거시 cleaning_options 가 Vue 정리 토글을 덮었다.
"""
from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import backends
from backends import BackendType
from core.combo_selection import pick_combo_index
from core.prompt_cleaner_prefs import (
    LEGACY_ONLY_DEFAULTS,
    UI_PREF_DEFAULTS,
    cleaner_options_from_ui_prefs,
    legacy_cleaning_options,
)
from ui.combo_restore import BACKEND_COMBOS, restore_backend_combos, snapshot_backend_combos
from ui.generation_settings_apply import (
    match_checkpoint_index,
    match_plain_index,
    match_sampler_index,
    match_scheduler_index,
)
from ui.generator_settings import SettingsMixin
from ui.widget_proxies import ComboBoxProxy, SliderProxy
from utils.prompt_cleaner import PromptCleaner

ROOT = Path(__file__).resolve().parents[1]


class _FakeBridge:
    def __init__(self):
        self.values = []

    def _register_proxy(self, widget_id, proxy):
        pass

    def pushWidgetValue(self, widget_id, value):
        self.values.append((widget_id, value))

    def pushWidgetProperty(self, widget_id, prop, value):
        pass


class SliderDriftTests(unittest.TestCase):
    def test_slider_proxy_values_do_not_drift_on_restore(self):
        bridge = _FakeBridge()
        cfg = SliderProxy(bridge, 'cfg_input', multiplier=2)
        denoise = SliderProxy(bridge, 'hires_denoising_input', multiplier=100)
        cfg.setText('5.3')
        denoise.setText('0.29')
        SettingsMixin._sync_slider(cfg)
        SettingsMixin._sync_slider(denoise)
        self.assertEqual(cfg.text(), '5.3')          # 예전: int(10.6)/2 → '5.0'
        self.assertEqual(denoise.text(), '0.29')     # 예전: int(28.999…)/100 → '0.28'

    def test_separate_slider_is_rounded_not_truncated(self):
        slider = mock.Mock()
        line = SimpleNamespace(text=lambda: '0.29', _slider=slider, _multiplier=100)
        SettingsMixin._sync_slider(line)
        slider.setValue.assert_called_once_with(29)


class BackendRestoreIdempotencyTests(unittest.TestCase):
    def _settings(self, backend_type='webui'):
        return {
            'backend_type': backend_type,
            'webui_url': 'http://forge.test:7860',
            'comfyui_url': 'http://comfy.test:8188/',
            'comfyui_workflow_path': '',
            'comfyui_workflow_img2img_path': '',
        }

    def test_same_type_and_url_does_not_recreate_the_backend(self):
        class WebUIBackend:   # 이름으로 타입을 확인한다 — HTTP 없는 가짜
            api_url = 'http://forge.test:7860/'

        mixin = object.__new__(SettingsMixin)
        with mock.patch.object(backends, '_current_backend', WebUIBackend()), \
                mock.patch.object(backends, '_current_type', BackendType.WEBUI), \
                mock.patch('backends.set_backend') as set_backend:
            mixin._restore_backend_settings(self._settings())
        set_backend.assert_not_called()

    def test_other_url_or_type_still_switches(self):
        class ComfyUIBackend:
            api_url = 'http://comfy.test:8188'

        mixin = object.__new__(SettingsMixin)
        with mock.patch.object(backends, '_current_backend', ComfyUIBackend()), \
                mock.patch.object(backends, '_current_type', BackendType.COMFYUI), \
                mock.patch('backends.set_backend') as set_backend:
            mixin._restore_backend_settings(self._settings('webui'))
            set_backend.assert_called_once_with(BackendType.WEBUI, 'http://forge.test:7860')
            set_backend.reset_mock()
            mixin._restore_backend_settings(self._settings('comfyui'))
            set_backend.assert_not_called()   # 끝 '/' 만 다른 같은 URL

    def test_is_active_backend_requires_an_existing_instance(self):
        with mock.patch.object(backends, '_current_backend', None):
            self.assertFalse(backends.is_active_backend(BackendType.WEBUI, 'http://x'))


class ComboSelectionTests(unittest.TestCase):
    def test_pick_prefers_first_matching_candidate(self):
        items = ['a', 'b', 'c']
        self.assertEqual(pick_combo_index(items, ['zz', 'b', 'c']), 1)
        self.assertEqual(pick_combo_index(items, ['', None, 'c']), 2)
        self.assertEqual(pick_combo_index(items, ['nope']), -1)
        self.assertEqual(pick_combo_index([], ['a']), -1)

    def test_checkpoint_matching_ignores_hash(self):
        items = ['animagine.safetensors', 'pony.safetensors']
        self.assertEqual(pick_combo_index(items, ['pony.safetensors [abc123]'], match=match_checkpoint_index), 1)
        self.assertEqual(pick_combo_index(items, ['pony.safetensors [abc123]']), -1)   # matcher 없음 = 정확히 같음

    def test_sampler_and_scheduler_matching_follows_forge_comfy_aliases(self):
        self.assertEqual(pick_combo_index(['euler', 'dpmpp_2m'], ['DPM++ 2M'], match=match_sampler_index), 1)
        self.assertEqual(pick_combo_index(['normal', 'karras'], ['Karras'], match=match_scheduler_index), 1)
        # 앞 후보(지금 선택)가 별칭으로 맞으면 뒤 후보(디스크)는 보지 않는다
        self.assertEqual(pick_combo_index(['euler', 'dpmpp_2m'], ['DPM++ 2M', 'euler'], match=match_sampler_index), 1)

    def test_every_backend_combo_uses_the_load_settings_matcher(self):
        """연결 경로와 load_settings(generation_settings_apply/_set_slot_settings)가 같은 비교를 쓴다."""
        expected = {
            'model_combo': match_checkpoint_index, 'hires_checkpoint_combo': match_checkpoint_index,
            'vae_main_combo': match_checkpoint_index,
            's1.checkpoint_combo': match_checkpoint_index, 's1.vae_combo': match_checkpoint_index,
            's2.checkpoint_combo': match_checkpoint_index, 's2.vae_combo': match_checkpoint_index,
            'sam3.checkpoint': match_checkpoint_index,
            'sampler_combo': match_sampler_index, 'hires_sampler_combo': match_sampler_index,
            's1.sampler_combo': match_sampler_index, 's2.sampler_combo': match_sampler_index,
            'scheduler_combo': match_scheduler_index, 'hires_scheduler_combo': match_scheduler_index,
            's1.scheduler_combo': match_scheduler_index, 's2.scheduler_combo': match_scheduler_index,
            'upscaler_combo': match_plain_index,
        }
        self.assertEqual({key: match for key, _r, _s, match in BACKEND_COMBOS}, expected)

    def _host(self):
        bridge = _FakeBridge()
        host = SimpleNamespace(is_programmatic_change=False)
        for key, _resolve, _read, _ckpt in BACKEND_COMBOS:
            if '.' in key:
                continue
            setattr(host, key, ComboBoxProxy(bridge, key))
        host.s1_widgets = {k: ComboBoxProxy(bridge, f's1_{k}') for k in ('checkpoint_combo', 'vae_combo', 'sampler_combo', 'scheduler_combo')}
        host.s2_widgets = {k: ComboBoxProxy(bridge, f's2_{k}') for k in ('checkpoint_combo', 'vae_combo', 'sampler_combo', 'scheduler_combo')}
        host.sam3_widgets = {'checkpoint': ComboBoxProxy(bridge, 'sam3_ckpt')}
        return host

    def test_reconnect_keeps_the_live_selection_and_falls_back_to_disk_only_when_empty(self):
        host = self._host()
        host.sampler_combo.addItems(['Euler a', 'DPM++ 2M'])
        host.sampler_combo.setCurrentIndex(1)            # 사용자가 Vue 에서 고른 값(저장 전)
        host.model_combo.setText('pony.safetensors')      # 목록 도착 전 설정값(fallback)
        preserved = snapshot_backend_combos(host)
        self.assertEqual(preserved['sampler_combo'], 'DPM++ 2M')
        self.assertEqual(preserved['model_combo'], 'pony.safetensors')

        # 연결: 비우고 새 목록으로 채운다
        host.model_combo.clear(); host.model_combo.addItems(['animagine.safetensors', 'pony.safetensors [4a458d26b2]'])
        host.sampler_combo.clear(); host.sampler_combo.addItems(['Euler a', 'DPM++ 2M'])
        host.upscaler_combo.clear(); host.upscaler_combo.addItems(['None', 'R-ESRGAN 4x+'])
        saved = {'sampler': 'Euler a', 'model': 'animagine.safetensors', 'hires_upscaler': 'R-ESRGAN 4x+'}
        load_saved = mock.Mock(return_value=saved)
        restored = restore_backend_combos(host, preserved, load_saved=load_saved)

        self.assertEqual(host.sampler_combo.currentText(), 'DPM++ 2M')            # 디스크의 'Euler a' 로 되돌리지 않음
        self.assertEqual(host.model_combo.currentText(), 'pony.safetensors [4a458d26b2]')  # 해시가 붙어도 같은 파일
        self.assertEqual(host.upscaler_combo.currentText(), 'R-ESRGAN 4x+')       # 기억한 값이 없으면 디스크
        self.assertIn('upscaler_combo', restored)
        load_saved.assert_called_once()
        self.assertFalse(host.is_programmatic_change)

    def test_disk_is_not_read_when_every_selection_survives(self):
        host = self._host()
        host.sampler_combo.addItems(['Euler a'])
        preserved = snapshot_backend_combos(host)
        host.sampler_combo.clear(); host.sampler_combo.addItems(['Euler a'])
        load_saved = mock.Mock(return_value={})
        restore_backend_combos(host, preserved, load_saved=load_saved)
        load_saved.assert_not_called()

    def test_forge_to_comfy_switch_keeps_sampler_scheduler_and_vae_via_aliases(self):
        """Forge 목록에서 고른 값이 ComfyUI 목록으로 바뀌어도 첫 항목('euler'/'normal')으로 새지 않는다."""
        host = self._host()
        host.sampler_combo.addItems(['Euler a', 'DPM++ 2M']); host.sampler_combo.setCurrentIndex(1)
        host.scheduler_combo.addItems(['Automatic', 'Karras']); host.scheduler_combo.setCurrentIndex(1)
        host.hires_sampler_combo.addItems(['Use same sampler', 'DPM++ 2M']); host.hires_sampler_combo.setCurrentIndex(1)
        host.vae_main_combo.addItems(['Automatic', 'sdxl_vae.safetensors']); host.vae_main_combo.setCurrentIndex(1)
        host.s1_widgets['scheduler_combo'].addItems(['Automatic', 'Karras'])
        host.s1_widgets['scheduler_combo'].setCurrentIndex(1)
        preserved = snapshot_backend_combos(host)

        # ComfyUI 로 연결(또는 PRIMARY 전환): 비우고 ComfyUI 이름으로 채운다
        comfy_samplers = ['euler', 'euler_ancestral', 'dpmpp_2m']
        comfy_schedulers = ['normal', 'karras', 'sgm_uniform']
        host.sampler_combo.clear(); host.sampler_combo.addItems(comfy_samplers)
        host.scheduler_combo.clear(); host.scheduler_combo.addItems(comfy_schedulers)
        host.hires_sampler_combo.clear(); host.hires_sampler_combo.addItems(['Use same sampler'] + comfy_samplers)
        host.vae_main_combo.clear(); host.vae_main_combo.addItems(['Automatic', 'sdxl/sdxl_vae.safetensors'])
        host.s1_widgets['scheduler_combo'].clear(); host.s1_widgets['scheduler_combo'].addItems(comfy_schedulers)
        load_saved = mock.Mock(return_value={})
        restore_backend_combos(host, preserved, load_saved=load_saved)

        self.assertEqual(host.sampler_combo.currentText(), 'dpmpp_2m')
        self.assertEqual(host.scheduler_combo.currentText(), 'karras')
        self.assertEqual(host.hires_sampler_combo.currentText(), 'dpmpp_2m')
        self.assertEqual(host.vae_main_combo.currentText(), 'sdxl/sdxl_vae.safetensors')   # 하위 폴더만 다름
        self.assertEqual(host.s1_widgets['scheduler_combo'].currentText(), 'karras')

    def test_disk_value_is_matched_with_aliases_too(self):
        host = self._host()
        host.sampler_combo.addItems(['euler', 'dpmpp_2m'])   # 기억한 값 없음(첫 연결)
        restore_backend_combos(host, {}, load_saved=lambda: {'sampler': 'DPM++ 2M', 'scheduler': 'Karras'})
        self.assertEqual(host.sampler_combo.currentText(), 'dpmpp_2m')

    def _error_clear(self, host):
        """on_webui_info_error 가 비우는 콤보(ui/generator_webui.py)."""
        host.model_combo.clear()
        host.hires_checkpoint_combo.clear()
        host.vae_main_combo.clear()
        for slot in (host.s1_widgets, host.s2_widgets):
            slot['checkpoint_combo'].clear()
            slot['vae_combo'].clear()

    def test_error_clear_then_reconnect_keeps_the_users_later_choice_not_the_boot_fallback(self):
        host = self._host()
        vae_list = ['Automatic', 'A.vae', 'B.vae']
        host.vae_main_combo.setText('A.vae')                 # 부팅 load_settings(defer_when_empty) — 목록 전
        # 첫 연결
        preserved = snapshot_backend_combos(host)
        self.assertEqual(preserved['vae_main_combo'], 'A.vae')
        host.vae_main_combo.clear(); host.vae_main_combo.addItems(vae_list)
        restore_backend_combos(host, preserved, load_saved=lambda: {'vae_main': 'A.vae'})
        self.assertEqual(host.vae_main_combo.currentText(), 'A.vae')
        # 사용자가 Vue 에서 B 를 고르고 저장했다
        host.vae_main_combo._on_vue_changed('B.vae')
        disk = {'vae_main': 'B.vae'}
        # 연결 오류 → 콤보를 비움 — 비운 콤보는 옛 이름을 생성 요청에 내보내지 않는다
        self._error_clear(host)
        self.assertEqual(host.vae_main_combo.currentText(), '')
        # 다시 연결
        preserved = snapshot_backend_combos(host)
        self.assertEqual(preserved['vae_main_combo'], 'B.vae')   # 예전: 부팅 fallback 'A.vae'
        host.vae_main_combo.clear(); host.vae_main_combo.addItems(vae_list)
        restore_backend_combos(host, preserved, load_saved=lambda: disk)
        self.assertEqual(host.vae_main_combo.currentText(), 'B.vae')

    def test_error_clear_keeps_an_unsaved_choice_and_a_missing_one_falls_back_to_disk(self):
        host = self._host()
        host.model_combo.addItems(['a.safetensors', 'b.safetensors', 'c.safetensors'])
        host.model_combo.setCurrentIndex(2)                   # 저장 전 선택(디스크는 b)
        host.hires_checkpoint_combo.addItems(['Use same checkpoint', 'x.safetensors'])
        host.hires_checkpoint_combo.setCurrentIndex(1)
        self._error_clear(host)
        self._error_clear(host)                               # 오류가 두 번 나도 마지막 실제 선택은 남는다
        preserved = snapshot_backend_combos(host)
        host.model_combo.clear(); host.model_combo.addItems(['a.safetensors', 'b.safetensors', 'c.safetensors [4a458d26b2]'])
        host.hires_checkpoint_combo.clear(); host.hires_checkpoint_combo.addItems(['Use same checkpoint', 'y.safetensors'])
        restore_backend_combos(host, preserved,
                               load_saved=lambda: {'model': 'b.safetensors', 'hires_checkpoint': 'Use same checkpoint'})
        self.assertEqual(host.model_combo.currentText(), 'c.safetensors [4a458d26b2]')           # 저장 전 선택 유지
        self.assertEqual(host.hires_checkpoint_combo.currentText(), 'Use same checkpoint')  # 새 백엔드에 없음 → 디스크


class ComboBoxProxySelectionStateTests(unittest.TestCase):
    """ComboBoxProxy 의 fallback·마지막 선택 — 연결 오류 뒤 재연결이 옛 부팅 값으로 되돌아가지 않게."""

    def _combo(self):
        return ComboBoxProxy(_FakeBridge(), 'vae_main_combo')

    def test_real_selection_drops_the_pending_fallback(self):
        combo = self._combo()
        combo.setText('A')                                    # 목록 전 — fallback
        combo.addItems(['A', 'B'])
        self.assertEqual(combo.currentText(), 'A')            # addItems 가 fallback 을 고름
        combo._on_vue_changed('B')                            # 사용자 선택
        self.assertEqual(combo._fallback_text, '')
        combo.setText('C')                                    # 목록에 없는 값 — 다시 미룸
        combo.setCurrentIndex(0)                              # Python 에서 실제 항목을 고름
        self.assertEqual(combo._fallback_text, '')

    def test_clear_hides_the_old_name_from_current_text_but_preserves_it_for_restore(self):
        combo = self._combo()
        combo.addItems(['A', 'B'])
        combo.setCurrentIndex(1)
        combo.clear()
        self.assertEqual(combo.currentText(), '')             # 생성 요청으로 옛 이름이 새지 않는다
        self.assertEqual(combo.preservedText(), 'B')
        combo.addItems(['A'])                                 # B 가 없는 목록 — 첫 항목(기존 규칙)
        self.assertEqual(combo.currentText(), 'A')

    def test_value_deferred_after_clear_is_newer_than_the_last_selection(self):
        combo = self._combo()
        combo.addItems(['A', 'B'])
        combo.setCurrentIndex(1)
        combo.clear()
        combo.setText('C')                                    # 끊긴 동안 프리셋·Vue 가 준 값
        self.assertEqual(combo.preservedText(), 'C')
        self.assertEqual(ComboBoxProxy(_FakeBridge(), 'x').preservedText(), '')


class CleanerSingleSourceTests(unittest.TestCase):
    def test_ui_prefs_own_the_three_vue_toggles_with_vue_defaults(self):
        self.assertEqual(cleaner_options_from_ui_prefs({}), {
            'remove_duplicates': True, 'auto_space': True, 'underscore_to_space': True,
        })
        self.assertEqual(
            cleaner_options_from_ui_prefs({'cleanDuplicates': False, 'cleanSpaces': 'yes'}),
            {'remove_duplicates': False, 'auto_space': True, 'underscore_to_space': True},
        )
        self.assertEqual(set(UI_PREF_DEFAULTS), {'cleanDuplicates', 'cleanSpaces', 'cleanUnderscore'})

    def test_legacy_cleaning_options_keep_only_the_toggles_vue_does_not_have(self):
        legacy = {'auto_comma': False, 'auto_escape': True, 'remove_duplicates': False,
                  'auto_space': False, 'underscore_to_space': False}
        self.assertEqual(legacy_cleaning_options(legacy), {'auto_comma': False, 'auto_escape': True})
        self.assertEqual(legacy_cleaning_options(None), dict(LEGACY_ONLY_DEFAULTS))

    def test_save_ui_prefs_then_load_settings_keeps_the_vue_value(self):
        """save_ui_prefs(cleanDuplicates=True) 뒤 load_settings 의 레거시 적용이 되돌리지 않는다."""
        from ui.generator_main import GeneratorMainUI

        host = SimpleNamespace(prompt_cleaner=PromptCleaner())
        GeneratorMainUI._apply_ui_prefs_to_cleaner(host, {'cleanDuplicates': True, 'cleanUnderscore': False})
        # load_settings 가 하는 일(레거시 두 옵션만) — 현재 사용자 파일은 remove_duplicates=False 였다
        host.prompt_cleaner.set_options(**legacy_cleaning_options(
            {'remove_duplicates': False, 'underscore_to_space': True, 'auto_escape': True}))
        self.assertTrue(host.prompt_cleaner.remove_duplicates)
        self.assertFalse(host.prompt_cleaner.underscore_to_space)
        self.assertTrue(host.prompt_cleaner.auto_escape)

    def test_settings_code_paths_use_the_single_source(self):
        source = (ROOT / 'ui' / 'generator_settings.py').read_text(encoding='utf-8')
        self.assertNotIn('self.prompt_cleaner.set_options(**cleaning_options)', source)
        # 레거시 두 옵션의 읽기·쓰기는 core/prompt_settings_extras 한 곳(legacy_cleaning_options)을 거친다
        self.assertIn('self.prompt_cleaner.set_options(**extras.cleaning)', source)
        self.assertIn('**extras_of(self).to_settings()', source)
        extras = (ROOT / 'core' / 'prompt_settings_extras.py').read_text(encoding='utf-8')
        self.assertIn('cleaning=legacy_cleaning_options(source.get("cleaning_options"))', extras)
        self.assertIn('"cleaning_options": legacy_cleaning_options(self.cleaning)', extras)


class NoReloadAfterSaveOrConnectTests(unittest.TestCase):
    def test_save_settings_action_does_not_reload(self):
        source = (ROOT / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        self.assertNotIn('QTimer.singleShot(200, self.load_settings)', source)

    def test_connect_restores_combos_instead_of_load_settings(self):
        from ui.generator_webui import WebUIMixin
        source = inspect.getsource(WebUIMixin.on_webui_info_loaded)
        self.assertNotIn('self.load_settings()', source)
        self.assertIn('snapshot_backend_combos(self)', source)
        self.assertIn('restore_backend_combos(self, preserved_combos)', source)
        # 스냅숏은 첫 clear() 보다 먼저
        self.assertLess(source.index('snapshot_backend_combos(self)'), source.index('.clear()'))
        # ComfyUI 워크플로 모델 자동 선택은 방금 되살린 모델을 덮지 않는다
        # (tests/test_comfy_workflow_model_autoselect.py 가 동작을 본다)
        self.assertIn('restored = restore_backend_combos(self, preserved_combos)', source)
        self.assertIn("_auto_select_workflow_model(models, keep_current='model_combo' in restored)", source)
        self.assertLess(source.index('restored = restore_backend_combos('),
                        source.index('_auto_select_workflow_model('))


if __name__ == '__main__':
    unittest.main()
