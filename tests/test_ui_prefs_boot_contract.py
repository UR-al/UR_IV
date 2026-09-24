"""부팅 설정 전달·ui_prefs 쓰기 경로의 계약 (감사 #107 · #41 · #145 · #130).

- Qt 모드도 웹 모드처럼 바인딩 직후 getInitialConfig 를 당겨 온다 — 예전엔 부팅 1초 타이머의 1회
  emit 이 JS connect 전에 터지면 uiPrefs·condRules·globalWeights 가 영구히 유실됐다.
- Python 쪽 ui_prefs 적용(클리너·테마·런타임)은 __init__ 에서 동기로 한 번 — 늦은 타이머가 Vue 가 먼저
  보낸 값을 덮지 않게.
- 블록 모드·갤러리 메타는 localStorage 폴링(setInterval) 대신 공유 ref, SAVE GLOBAL 은 ui_prefs 를
  통째로 다시 보내지 않는다.
"""
from __future__ import annotations

import inspect
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FRONT = ROOT / 'frontend' / 'src'


def _read(*parts: str) -> str:
    return (ROOT.joinpath(*parts)).read_text(encoding='utf-8')


class QtInitialConfigPullTests(unittest.TestCase):
    def test_qt_branch_pulls_initial_config_after_binding(self):
        bridge = _read('frontend', 'src', 'bridge.js')
        qt_branch = bridge[bridge.index('new window.QWebChannel(window.qt.webChannelTransport'):]
        qt_branch = qt_branch[:qt_branch.index('resolve(_backendProxy)')]
        self.assertIn('_bindBackend(channel.objects.backend', qt_branch)
        self.assertIn('_requestInitialConfig(channel.objects.backend)', qt_branch)
        self.assertLess(qt_branch.index('_bindBackend('), qt_branch.index('_requestInitialConfig('))

    def test_python_no_longer_broadcasts_the_boot_config_on_a_timer(self):
        source = _read('ui', 'generator_main.py')
        for signal in ('uiPrefsLoaded', 'condRulesLoaded', 'globalWeightsLoaded'):
            self.assertNotIn(f'{signal}.emit(', source, signal)
        self.assertNotIn('singleShot(1000, self._load_saved_configs)', source)
        self.assertIn('self._apply_saved_configs()', source)

    def test_initial_config_reply_carries_all_three_payloads(self):
        from ui.vue_bridge import VueBridge
        payload = json.loads(VueBridge().getInitialConfig())
        self.assertIn('uiPrefs', payload)
        self.assertIn('condRules', payload)
        self.assertIn('globalWeights', payload)

    def test_apply_saved_configs_applies_prefs_without_emitting(self):
        from ui.generator_main import GeneratorMainUI
        prefs = {'schema_version': 1, 'cleanDuplicates': False, 'ratingFilter': [True, False, False, False]}
        host = SimpleNamespace(
            vue_bridge=SimpleNamespace(
                uiPrefsLoaded=mock.Mock(), condRulesLoaded=mock.Mock(), globalWeightsLoaded=mock.Mock()),
            _migrate_legacy_cond_rules=mock.Mock(),
            _migrate_legacy_lora_stack=mock.Mock(),
            _apply_tab_defaults_to_empty_widgets=mock.Mock(),
            _apply_ui_prefs_to_cleaner=mock.Mock(),
            _apply_theme_prefs=mock.Mock(),
            _restore_runtime_prefs=mock.Mock(),
        )
        with mock.patch('core.config_migration.load_ui_prefs', return_value=dict(prefs)), \
                mock.patch('os.path.exists', return_value=False):
            GeneratorMainUI._apply_saved_configs(host)
        host._apply_ui_prefs_to_cleaner.assert_called_once()
        host._restore_runtime_prefs.assert_called_once()
        host.vue_bridge.uiPrefsLoaded.emit.assert_not_called()
        host.vue_bridge.condRulesLoaded.emit.assert_not_called()
        host.vue_bridge.globalWeightsLoaded.emit.assert_not_called()

    def test_boot_runtime_pushes_wait_for_the_file_values(self):
        app = _read('frontend', 'src', 'App.vue')
        self.assertNotIn('\n  pushRatingFilter()\n', app)
        high_res = _read('frontend', 'src', 'composables', 'useHighRes.js')
        self.assertNotIn('setTimeout(_syncHighRes', high_res)


class SharedFlagTests(unittest.TestCase):
    def test_no_localstorage_polling_for_block_mode_or_gallery_metadata(self):
        for rel in (('components', 'PromptPanel.vue'), ('views', 'GalleryView.vue'), ('views', 'SettingsView.vue')):
            source = FRONT.joinpath(*rel).read_text(encoding='utf-8')
            with self.subTest(rel=rel):
                self.assertNotIn("getItem('tagBlockMode')", source)
                self.assertNotIn("getItem('galleryShowMetadata')", source)
                self.assertNotIn("setItem('tagBlockMode'", source)
                self.assertNotIn("setItem('galleryShowMetadata'", source)
        self.assertIn("from '../composables/uiPrefs'", _read('frontend', 'src', 'components', 'PromptPanel.vue'))
        self.assertIn("from '../composables/uiPrefs'", _read('frontend', 'src', 'views', 'GalleryView.vue'))

    def test_save_global_does_not_resend_ui_prefs(self):
        settings = _read('frontend', 'src', 'views', 'SettingsView.vue')
        start = settings.index('const act = (name: ActionName) =>')
        body = settings[start:settings.index('\n}\n', start)]
        self.assertNotIn('save_ui_prefs', body)
        self.assertNotIn('persistUiPrefs', body)

    def test_ollama_fields_are_saved_separately(self):
        settings = _read('frontend', 'src', 'views', 'SettingsView.vue')
        self.assertNotIn('saveOllamaSettings', settings)
        for fn, key in (('saveOllamaUrl', 'ollamaUrl'), ('saveOllamaModel', 'ollamaModel'), ('saveOllamaUnloadOnGen', 'ollamaUnloadOnGen')):
            match = re.search(rf'function {fn}\(\) \{{\n(.*?)\n\}}', settings, re.S)
            self.assertIsNotNone(match, fn)
            self.assertEqual(re.findall(r'(\w+):', match.group(1)), [key], fn)


class ProductionConsoleTests(unittest.TestCase):
    def test_request_action_logs_only_in_dev_builds(self):
        store = _read('frontend', 'src', 'stores', 'widgetStore.js')
        block = store[store.index('export function requestAction'):]
        block = block[:block.index('\n}\n')]
        self.assertNotIn('else console.log', block)
        self.assertIn('if (import.meta.env.DEV) console.log', block)


if __name__ == '__main__':
    unittest.main()
