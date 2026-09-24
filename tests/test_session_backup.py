"""크래시 복구 백업 — 정상 종료 표시와 브리지 왕복 (감사 #160)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core import session_backup


class SessionBackupStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / 'session_backup.json'

    def test_edit_backups_are_never_clean(self):
        written = session_backup.write_session_backup({'prompt': 'p', 'clean': True}, self.path)
        self.assertFalse(written['clean'])
        self.assertEqual(session_backup.read_session_backup(self.path), {'prompt': 'p', 'clean': False})

    def test_clean_shutdown_marks_the_backup(self):
        session_backup.write_session_backup({'prompt': 'p'}, self.path)
        self.assertTrue(session_backup.mark_session_clean(self.path))
        self.assertTrue(session_backup.read_session_backup(self.path)['clean'])
        self.assertEqual(session_backup.read_session_backup(self.path)['prompt'], 'p')
        self.assertFalse(session_backup.mark_session_clean(self.path))   # 이미 clean

    def test_missing_or_broken_backup(self):
        self.assertEqual(session_backup.read_session_backup(self.path), {})
        self.assertFalse(session_backup.mark_session_clean(self.path))
        self.assertFalse(self.path.exists())
        self.path.write_text('{broken', encoding='utf-8')
        self.assertEqual(session_backup.read_session_backup(self.path), {})
        with self.assertRaises(ValueError):
            session_backup.write_session_backup(['not', 'a', 'dict'], self.path)

    def test_bridge_round_trip(self):
        from ui.vue_bridge import VueBridge
        bridge = VueBridge()
        with mock.patch('core.session_backup.session_backup_path', return_value=self.path):
            self.assertEqual(json.loads(bridge.saveSession(json.dumps({'tab': 't2i', 'prompt': 'x'}))), {'ok': True})
            self.assertEqual(json.loads(bridge.getSession()), {'tab': 't2i', 'prompt': 'x', 'clean': False})
            self.assertIn('error', json.loads(bridge.saveSession('[1]')))


class ShutdownMarksCleanTests(unittest.TestCase):
    def _host(self, saved):
        return SimpleNamespace(
            _preserve_imported_settings_on_quit=False,
            save_settings=mock.Mock(return_value=saved),
            ui_state=SimpleNamespace(save_all=mock.Mock()),
        )

    def test_marks_clean_only_after_the_prompt_was_saved(self):
        from ui.generator_main import GeneratorMainUI
        with mock.patch('core.session_backup.mark_session_clean') as mark:
            self.assertTrue(GeneratorMainUI._save_shutdown_state(self._host(True)))
            mark.assert_called_once_with()
            mark.reset_mock()
            GeneratorMainUI._save_shutdown_state(self._host(False))   # 저장 실패 — 백업이 유일한 사본
            mark.assert_not_called()

    def test_imported_settings_path_marks_the_backup_clean_without_saving(self):
        """설정 백업 가져오기 → 재시작은 크래시가 아니다. 백업(가져오기 전 프롬프트)을 정상 종료로 두지
        않으면 다음 부팅이 가져온 설정과 다른 그 프롬프트를 복구로 제안해, 누르면 가져온 설정을 덮는다."""
        from ui.generator_main import GeneratorMainUI
        host = self._host(True)
        host._preserve_imported_settings_on_quit = True
        with mock.patch('core.session_backup.mark_session_clean') as mark:
            self.assertFalse(GeneratorMainUI._save_shutdown_state(host))
        mark.assert_called_once_with()
        host.save_settings.assert_not_called()        # 가져온 prompt_settings 는 그대로
        host.ui_state.save_all.assert_not_called()

    def test_import_restart_leaves_a_clean_backup_so_no_recovery_is_offered(self):
        """실제 파일로: 편집 중 백업(clean:false) → 가져오기 잠금 상태로 종료 → clean:true."""
        from ui.generator_main import GeneratorMainUI
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'session_backup.json'
            session_backup.write_session_backup({'tab': 't2i', 'prompt': 'before import', 'negative': ''}, path)
            host = self._host(True)
            host._preserve_imported_settings_on_quit = True
            with mock.patch('core.session_backup.session_backup_path', return_value=path):
                GeneratorMainUI._save_shutdown_state(host)
            self.assertIs(session_backup.read_session_backup(path)['clean'], True)
            self.assertEqual(session_backup.read_session_backup(path)['prompt'], 'before import')


if __name__ == '__main__':
    unittest.main()
