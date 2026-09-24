"""모드별 자동화 설정이 부팅마다 Vue 기본값으로 덮이지 않는다 (감사 #42).

예전엔 Vue 가 마운트 직후 하드코딩 기본값을 set_automation_settings 로 보냈고 Python 은 곧바로
automation_settings_<mode>.json 에 썼다. 파일 복원은 1.5초 타이머의 1회 emit 뿐이라 순서와 무관하게
사용자 값이 기본값으로 바뀌었다. cleanupEveryN 은 영속 키에 아예 없었다.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.mode_aware_automation import (
    ModeAwareAutomationSettings,
    _DEFAULT_AUTOMATION_SETTINGS,
    apply_automation_payload,
    merge_automation_payload,
    normalize_automation_settings,
)


def _host():
    return SimpleNamespace(
        _vue_automation_settings=dict(_DEFAULT_AUTOMATION_SETTINGS),
        vue_bridge=SimpleNamespace(automationSettingsLoaded=mock.Mock()),
        queue_manager=SimpleNamespace(cleanup_every_n=0, delay_seconds=1.0),
    )


class _TempPathMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / 'automation_settings_webui.json'
        patcher = mock.patch.object(ModeAwareAutomationSettings, '_settings_path', lambda _self, mode=None: self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        sub = mock.patch.object(ModeAwareAutomationSettings, 'auto_subscribe_to_mode_change')
        self.subscribe = sub.start()
        self.addCleanup(sub.stop)


class NormalizeTests(unittest.TestCase):
    def test_includes_cleanup_every_n_and_fills_defaults(self):
        self.assertEqual(normalize_automation_settings({'limit': '30', 'cleanupEveryN': '5', 'junk': 1}), {
            'mode': 'count', 'limit': 30.0, 'repeat': 1, 'delay': 1.0, 'allowDupes': False,
            'autoResetDeck': False, 'maxRetries': 2, 'cleanupEveryN': 5,
        })
        self.assertEqual(normalize_automation_settings({'cleanupEveryN': -3, 'repeat': 'x'})['cleanupEveryN'], 0)
        self.assertEqual(normalize_automation_settings(None), normalize_automation_settings({}))

    def test_same_values_normalize_equal(self):
        vue = {'mode': 'count', 'limit': 10, 'repeat': 1, 'delay': 1, 'allowDupes': False,
               'autoResetDeck': False, 'maxRetries': 2, 'cleanupEveryN': 0}
        self.assertEqual(normalize_automation_settings(vue), normalize_automation_settings(_DEFAULT_AUTOMATION_SETTINGS))


class InitializeAndSnapshotTests(_TempPathMixin, unittest.TestCase):
    def test_initialize_loads_the_file_without_emitting_and_applies_the_queue(self):
        self.path.write_text(json.dumps({'mode': 'timer', 'limit': 45, 'repeat': 3, 'delay': 2.5,
                                         'maxRetries': 1, 'cleanupEveryN': 20}), encoding='utf-8')
        host = _host()
        persistence = ModeAwareAutomationSettings(host)
        persistence.initialize()
        self.assertEqual(host._vue_automation_settings['limit'], 45.0)
        self.assertEqual(host._vue_automation_settings['cleanupEveryN'], 20)
        self.assertEqual(host.queue_manager.cleanup_every_n, 20)
        self.assertEqual(host.queue_manager.delay_seconds, 2.5)
        host.vue_bridge.automationSettingsLoaded.emit.assert_not_called()   # Vue 는 당겨 간다
        self.subscribe.assert_called_once()
        # Vue 가 마운트 때 당겨 가는 값 = 파일 값
        self.assertEqual(persistence.snapshot()['mode'], 'timer')
        self.assertEqual(persistence.snapshot()['cleanupEveryN'], 20)

    def test_first_run_writes_defaults_including_cleanup(self):
        host = _host()
        ModeAwareAutomationSettings(host).initialize()
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')),
                         normalize_automation_settings(_DEFAULT_AUTOMATION_SETTINGS))

    def test_resync_with_the_same_values_does_not_rewrite_the_file(self):
        self.path.write_text(json.dumps(normalize_automation_settings({'limit': 45})), encoding='utf-8')
        host = _host()
        persistence = ModeAwareAutomationSettings(host)
        persistence.initialize()
        # Vue hydrate 뒤 동기화 — 같은 값(정수/실수 표기 차이 포함)
        host._vue_automation_settings = {'mode': 'count', 'limit': 45, 'repeat': 1, 'delay': 1,
                                         'allowDupes': False, 'autoResetDeck': False, 'maxRetries': 2,
                                         'cleanupEveryN': 0}
        with mock.patch('utils.atomic_json.atomic_write_json') as write:
            self.assertTrue(persistence.save_mode_settings())
        write.assert_not_called()
        host._vue_automation_settings['cleanupEveryN'] = 7
        self.assertTrue(persistence.save_mode_settings())
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['cleanupEveryN'], 7)

    def test_mode_switch_load_emits_to_vue(self):
        host = _host()
        persistence = ModeAwareAutomationSettings(host)
        persistence.apply_settings({'limit': 12, 'cleanupEveryN': 3})
        payload = json.loads(host.vue_bridge.automationSettingsLoaded.emit.call_args.args[0])
        self.assertEqual(payload['limit'], 12.0)
        self.assertEqual(payload['cleanupEveryN'], 3)


_USER_FILE = {'mode': 'timer', 'limit': 42, 'repeat': 3, 'delay': 5, 'allowDupes': True,
              'autoResetDeck': True, 'maxRetries': 0, 'cleanupEveryN': 7}


class MergePayloadTests(unittest.TestCase):
    """R2b#1 — 파일 값을 아직 못 받은 Vue 의 동기화(자동 NL·Ollama 만, 또는 고친 키만)가
    빠진 키를 기본값으로 덮어 파일에 쓰던 문제."""

    def test_payload_without_persisted_keys_changes_nothing(self):
        current = normalize_automation_settings(_USER_FILE)
        self.assertEqual(merge_automation_payload(current, {'autoNl': True, 'ollamaUrl': 'x'}), current)
        self.assertEqual(merge_automation_payload(current, None), current)

    def test_only_sent_keys_change(self):
        merged = merge_automation_payload(_USER_FILE, {'delay': 2.5})
        self.assertEqual(merged, {**normalize_automation_settings(_USER_FILE), 'delay': 2.5})

    def test_bad_values_keep_the_current_value_not_the_default(self):
        merged = merge_automation_payload(_USER_FILE, {
            'limit': 'x', 'repeat': None, 'delay': float('nan'), 'maxRetries': [], 'cleanupEveryN': -4,
            'mode': '', 'allowDupes': None,
        })
        self.assertEqual(merged['limit'], 42.0)
        self.assertEqual(merged['repeat'], 3)
        self.assertEqual(merged['delay'], 5.0)
        self.assertEqual(merged['maxRetries'], 0)
        self.assertEqual(merged['cleanupEveryN'], 0)   # 음수는 0 으로 자른다(정규화와 같게)
        self.assertEqual(merged['mode'], 'timer')
        self.assertTrue(merged['allowDupes'])

    def test_full_payload_still_replaces_every_key(self):
        full = {'mode': 'count', 'limit': 10, 'repeat': 1, 'delay': 1, 'allowDupes': False,
                'autoResetDeck': False, 'maxRetries': 2, 'cleanupEveryN': 0}
        self.assertEqual(merge_automation_payload(_USER_FILE, full), normalize_automation_settings(full))


class ApplyPayloadTests(_TempPathMixin, unittest.TestCase):
    def _host_with_file(self):
        self.path.write_text(json.dumps(normalize_automation_settings(_USER_FILE)), encoding='utf-8')
        host = _host()
        host.automation_persistence = ModeAwareAutomationSettings(host)
        host.automation_persistence.initialize()
        return host

    def test_unhydrated_sync_does_not_rewrite_the_file_with_defaults(self):
        host = self._host_with_file()
        with mock.patch('utils.atomic_json.atomic_write_json') as write:
            apply_automation_payload(host, {'autoNl': True, 'ollamaUrl': 'http://o', 'ollamaModel': 'm'})
        write.assert_not_called()
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['limit'], 42.0)
        self.assertEqual(host._vue_automation_settings, normalize_automation_settings(_USER_FILE))
        self.assertTrue(host._auto_nl_enabled)
        self.assertEqual((host._auto_nl_url, host._auto_nl_model), ('http://o', 'm'))
        self.assertEqual(host.queue_manager.cleanup_every_n, 7)
        self.assertEqual(host.queue_manager.delay_seconds, 5.0)

    def test_an_edit_saves_only_that_key(self):
        host = self._host_with_file()
        apply_automation_payload(host, {'delay': 2.5, 'autoNl': False})
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(saved, {**normalize_automation_settings(_USER_FILE), 'delay': 2.5})
        self.assertEqual(host.queue_manager.delay_seconds, 2.5)
        self.assertEqual(host.queue_manager.cleanup_every_n, 7)

    def test_missing_nl_keys_keep_their_previous_values(self):
        host = self._host_with_file()
        host._auto_nl_enabled, host._auto_nl_url, host._auto_nl_model = True, 'http://keep', 'keep'
        apply_automation_payload(host, {'limit': 12})
        self.assertEqual((host._auto_nl_enabled, host._auto_nl_url, host._auto_nl_model), (True, 'http://keep', 'keep'))
        self.assertEqual(host._vue_automation_settings['limit'], 12.0)

    def test_empty_ollama_url_falls_back_to_the_default(self):
        from core.ollama_client import DEFAULT_OLLAMA_URL
        host = self._host_with_file()
        apply_automation_payload(host, {'ollamaUrl': '', 'ollamaModel': ''})
        self.assertEqual((host._auto_nl_url, host._auto_nl_model), (DEFAULT_OLLAMA_URL, ''))

    def test_handler_delegates_to_the_merge(self):
        source = (Path(__file__).resolve().parents[1] / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        start = source.index("action == 'set_automation_settings'")
        block = source[start:source.index('elif action ==', start + 10)]
        self.assertIn('apply_automation_payload(self, payload)', block)
        self.assertNotIn("payload.get('limit', 10)", block)


class BridgeSlotTests(unittest.TestCase):
    def test_get_automation_settings_returns_the_host_snapshot(self):
        from ui.vue_bridge import VueBridge
        bridge = VueBridge()
        persistence = SimpleNamespace(snapshot=lambda: {'mode': 'unlimited', 'cleanupEveryN': 4})
        with mock.patch.object(VueBridge, 'parent', return_value=SimpleNamespace(automation_persistence=persistence)):
            self.assertEqual(json.loads(bridge.getAutomationSettings()), {'mode': 'unlimited', 'cleanupEveryN': 4})
        with mock.patch.object(VueBridge, 'parent', return_value=SimpleNamespace()):
            self.assertEqual(bridge.getAutomationSettings(), '{}')

    def test_boot_initializes_synchronously_not_on_a_timer(self):
        source = (Path(__file__).resolve().parents[1] / 'ui' / 'generator_main.py').read_text(encoding='utf-8')
        self.assertIn('self.automation_persistence.initialize()', source)
        self.assertNotIn('QTimer.singleShot(1500, self.automation_persistence.initialize)', source)


if __name__ == '__main__':
    unittest.main()
