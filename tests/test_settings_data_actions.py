"""Settings '데이터 · 백업' 액션 — 백업/복원·재시작·프리셋 공유(audit #179). Qt 대화상자는 가짜로 대신한다."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core import app_restart
from core.app_restart import (
    RestartError,
    parse_args,
    relaunch_command,
    spawn_restart_waiter,
    wait_then_launch,
    waiter_command,
)
from ui import settings_data_actions as actions


def _host(**extra):
    notices = []
    host = SimpleNamespace(
        vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(emit=lambda *a: notices.append(a))),
        _quit_app=mock.Mock(),
        save_settings=mock.Mock(return_value=True),
        **extra,
    )
    return host, notices


class SettingsBackupActionTests(unittest.TestCase):
    def test_export_saves_current_state_then_archives_with_chat_flag(self):
        host, notices = _host()
        export = mock.Mock(return_value=7)
        path = actions.export_settings(host, {'includeChat': True},
                                       ask_save=lambda *a: 'D:/b.zip', export=export)
        self.assertEqual(path, 'D:/b.zip')
        host.save_settings.assert_called_once()
        export.assert_called_once_with('D:/b.zip', include_chat=True)
        self.assertEqual(notices[-1][0], 'success')
        self.assertIn('대화 기록 포함', notices[-1][1])

    def test_export_does_not_overwrite_a_just_imported_backup(self):
        host, _ = _host(_preserve_imported_settings_on_quit=True)
        actions.export_settings(host, {}, ask_save=lambda *a: 'x.zip', export=mock.Mock(return_value=1))
        host.save_settings.assert_not_called()

    def test_cancelled_dialogs_do_nothing(self):
        host, notices = _host()
        self.assertIsNone(actions.export_settings(host, {}, ask_save=lambda *a: ''))
        self.assertIsNone(actions.import_settings(host, {}, ask_open=lambda *a: ''))
        self.assertEqual(notices, [])
        self.assertFalse(getattr(host, '_preserve_imported_settings_on_quit', False))

    def test_import_locks_persistence_and_restarts(self):
        host, notices = _host()
        restart = mock.Mock()
        count = actions.import_settings(host, {}, ask_open=lambda *a: 'b.zip',
                                        importer=mock.Mock(return_value=12), restart=restart)
        self.assertEqual(count, 12)
        self.assertTrue(host._preserve_imported_settings_on_quit)
        restart.assert_called_once_with(host)
        self.assertTrue(actions.should_skip_persist_action(host, 'save_cond_rules'))

    def test_zip_without_settings_entries_neither_locks_nor_restarts(self):
        """엉뚱한 ZIP — 복원한 것이 없으면 저장을 잠그고 재시작해 현재 설정만 잃으면 안 된다."""
        import zipfile

        from core.settings_backup import import_settings_archive

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'holiday.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('holiday/pic.txt', 'not a backup')
                bundle.writestr('dataset/img_001.txt', 'caption')
            host, notices = _host()
            restart = mock.Mock()
            count = actions.import_settings(
                host, {}, ask_open=lambda *a: str(archive),
                importer=lambda path: import_settings_archive(path, project_root=root / 'app'),
                restart=restart)
        self.assertEqual(count, 0)
        self.assertFalse(getattr(host, '_preserve_imported_settings_on_quit', False))
        self.assertFalse(actions.should_skip_persist_action(host, 'save_settings'))
        restart.assert_not_called()
        self.assertEqual(notices[-1][0], 'warning')
        self.assertIn('복원할 설정 파일을 찾지 못했습니다', notices[-1][1])

    def test_restart_failure_keeps_the_app_running(self):
        host, notices = _host()

        def fail():
            raise RestartError('no python')

        self.assertFalse(actions.restart_app(host, spawn=fail, schedule=mock.Mock()))
        host._quit_app.assert_not_called()
        self.assertEqual(notices[-1][0], 'error')
        self.assertFalse(actions.restart_pending(host), '실패하면 다시 시도할 수 있어야 한다')

    def test_second_restart_does_not_spawn_another_waiter(self):
        """1.2초 창 안의 두 번째 재시작 — 대기 프로세스가 둘이면 앱이 두 개 뜬다."""
        host, notices = _host()
        spawn = mock.Mock()
        scheduled = []
        schedule = lambda ms, fn: scheduled.append((ms, fn))
        self.assertTrue(actions.restart_app(host, spawn=spawn, schedule=schedule))
        self.assertTrue(actions.restart_pending(host))
        self.assertFalse(actions.restart_app(host, spawn=spawn, schedule=schedule))
        spawn.assert_called_once()
        self.assertEqual(len(scheduled), 1)
        self.assertEqual(notices[-1], ('info', '이미 앱을 재시작하는 중입니다'))

    def test_dispatcher_refuses_every_action_while_a_restart_is_pending(self):
        """재시작 대기 중엔 대화상자를 열지 않는다 — _quit_app 이 모달 대화상자 도중 앱을 끈다."""
        host, notices = _host(_restart_pending=True)
        targets = ('restart_app', 'import_settings', 'export_settings', 'export_presets', 'import_presets',
                   'export_character_presets', 'import_character_presets')
        patches = [mock.patch.object(actions, name) for name in targets]
        mocks = [patcher.start() for patcher in patches]
        for patcher in patches:
            self.addCleanup(patcher.stop)
        for name in actions.SETTINGS_DATA_ACTIONS:
            actions.handle_settings_data_action(host, name, {})
        for fn in mocks:
            fn.assert_not_called()
        self.assertEqual(len(notices), len(actions.SETTINGS_DATA_ACTIONS))
        self.assertTrue(all(level == 'info' for level, _ in notices))

    def test_dispatcher_turns_exceptions_into_error_toasts(self):
        host, notices = _host()
        with mock.patch.object(actions, 'export_presets', side_effect=ValueError('boom')):
            actions.handle_settings_data_action(host, 'presets_export', None)
        self.assertEqual(notices[-1], ('error', '작업 실패: boom'))

    def test_dispatcher_covers_every_declared_action(self):
        host, _ = _host()
        for name in actions.SETTINGS_DATA_ACTIONS:
            target = {'restart_app': 'restart_app'}.get(name) or {
                'settings_export': 'export_settings', 'settings_import': 'import_settings',
                'presets_export': 'export_presets', 'presets_import': 'import_presets',
                'character_presets_export': 'export_character_presets',
                'character_presets_import': 'import_character_presets',
            }[name]
            with mock.patch.object(actions, target) as fn:
                actions.handle_settings_data_action(host, name, {})
                fn.assert_called_once()


class PresetSharingActionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.presets = self.root / 'presets'
        patcher = mock.patch('core.generation_presets.default_presets_dir', return_value=self.presets)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_generation_presets_round_trip_through_a_json_file(self):
        from core.generation_presets import list_presets, read_preset, write_preset

        write_preset('portrait', {'steps': '28', 'model': 'm.safetensors', 'theme': 'x'})
        host, notices = _host()
        out = self.root / 'share.json'
        self.assertEqual(actions.export_presets(host, {}, ask_save=lambda *a: str(out)), str(out))
        bundle = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(bundle['presets']['portrait'], {'model': 'm.safetensors', 'steps': '28'})

        (self.presets / 'portrait.json').unlink()
        result = actions.import_presets(host, {'overwrite': False}, ask_open=lambda *a: str(out))
        self.assertEqual(result['added'], ['portrait'])
        self.assertEqual(list_presets(), ['portrait'])
        self.assertEqual(read_preset('portrait')['steps'], '28')

    def test_case_duplicates_inside_the_file_are_counted_and_warned_not_added_twice(self):
        from core.generation_presets import list_presets

        incoming = self.root / 'dupes.json'
        incoming.write_text(json.dumps({'Alpha': {'steps': '20'}, 'alpha': {'steps': '30'}}), encoding='utf-8')
        host, notices = _host()
        result = actions.import_presets(host, {'overwrite': True}, ask_open=lambda *a: str(incoming))
        self.assertEqual((result['added'], result['duplicate']), (['Alpha'], ['alpha']))
        self.assertEqual(list_presets(), ['Alpha'])
        level, message = notices[-1]
        self.assertEqual(level, 'warning')
        self.assertIn('추가 1', message)
        self.assertIn("파일 안 중복 이름 1개 건너뜀('alpha')", message)

    def test_export_with_no_presets_informs_instead_of_opening_a_dialog(self):
        host, notices = _host()
        ask = mock.Mock()
        self.assertIsNone(actions.export_presets(host, {}, ask_save=ask))
        ask.assert_not_called()
        self.assertEqual(notices[-1][0], 'info')

    def test_character_presets_merge_and_replace_keep_the_cache_in_sync(self):
        from utils import character_presets as cp

        store = self.root / 'character_presets.json'
        with mock.patch.object(cp, '_FILE', str(store)), mock.patch.object(cp, '_cache', None):
            cp.save_character_preset('Hatsune Miku', 'twintails')
            host, _ = _host()
            share = self.root / 'chars.json'
            actions.export_character_presets(host, {}, ask_save=lambda *a: str(share))
            data = json.loads(share.read_text(encoding='utf-8'))
            self.assertIn('hatsune miku', data)

            incoming = self.root / 'incoming.json'
            incoming.write_text(json.dumps({
                'Rem': {'extra_prompt': 'blue hair', 'display_name': 'Rem'},
                'broken': 'nope',
            }), encoding='utf-8')
            result = actions.import_character_presets(host, {'replace': False}, ask_open=lambda *a: str(incoming))
            self.assertEqual(result, {'imported': 1, 'total': 2})
            # 캐시도 갱신 — 생성·브리지가 쓰는 get_character_preset_full 로 읽는다
            self.assertEqual(cp.get_character_preset_full('rem')['extra_prompt'], 'blue hair')
            self.assertEqual(cp.get_character_preset_full('hatsune miku')['extra_prompt'], 'twintails')

            actions.import_character_presets(host, {'replace': True}, ask_open=lambda *a: str(incoming))
            self.assertIsNone(cp.get_character_preset_full('hatsune miku'))
            self.assertEqual(json.loads(store.read_text(encoding='utf-8')).keys(), {'rem'})

    def test_character_import_rejects_non_preset_files(self):
        from utils import character_presets as cp

        with mock.patch.object(cp, '_FILE', str(self.root / 'c.json')), mock.patch.object(cp, '_cache', {}):
            with self.assertRaises(ValueError):
                cp.import_character_presets([1, 2])
            with self.assertRaises(ValueError):
                cp.import_character_presets({'x': 'y'})


class AppRestartTests(unittest.TestCase):
    def test_relaunch_command_uses_absolute_script_and_args(self):
        script = os.path.abspath(__file__)
        self.assertEqual(relaunch_command('py.exe', [script, '--flag']), ['py.exe', script, '--flag'])
        with self.assertRaises(RestartError):
            relaunch_command('py.exe', [])
        with self.assertRaises(RestartError):
            relaunch_command('py.exe', ['missing_script_xyz.py'])

    def test_waiter_command_round_trips_through_parse_args(self):
        cmd = waiter_command(4321, ['py.exe', 'C:/app/new_main_ui.py', '--x'], 'C:/app', executable='py.exe')
        self.assertEqual(cmd[:3], ['py.exe', '-m', 'core.app_restart'])
        self.assertEqual(parse_args(cmd[3:]), (4321, 'C:/app', ['py.exe', 'C:/app/new_main_ui.py', '--x']))
        for bad in (['--wait-pid', '1'], ['--', 'x'], ['--wait-pid', 'abc', '--', 'x'], ['--wait-pid', '1', '--']):
            with self.assertRaises(RestartError):
                parse_args(bad)

    def test_launches_only_after_the_old_process_exits(self):
        alive = iter([True, True, False])
        launched = []
        ok = wait_then_launch(7, ['app'], 'C:/cwd', exists=lambda pid: next(alive),
                              launch=lambda cmd, cwd: launched.append((cmd, cwd)), sleep=lambda s: None)
        self.assertTrue(ok)
        self.assertEqual(launched, [(['app'], 'C:/cwd')])

    def test_gives_up_when_the_old_process_never_exits(self):
        clock = iter(range(0, 1000, 50))
        launched = []
        ok = wait_then_launch(7, ['app'], '', exists=lambda pid: True, launch=lambda *a: launched.append(a),
                              sleep=lambda s: None, clock=lambda: next(clock), timeout=120)
        self.assertFalse(ok)
        self.assertEqual(launched, [])

    def test_spawn_is_detached_and_failures_raise(self):
        popen = mock.Mock()
        spawn_restart_waiter(99, ['py', 'app.py'], cwd='C:/w', popen=popen)
        args, kwargs = popen.call_args
        self.assertIn('--wait-pid', args[0])
        self.assertIn('99', args[0])
        self.assertEqual(kwargs['cwd'], app_restart._project_root())
        if os.name == 'nt':
            self.assertTrue(kwargs['creationflags'] & app_restart._CREATE_NO_WINDOW)
        with self.assertRaises(RestartError):
            spawn_restart_waiter(99, ['py'], cwd='C:/w', popen=mock.Mock(side_effect=OSError('denied')))

    def test_module_is_runnable_with_python_dash_m(self):
        self.assertEqual(app_restart.main(['--wait-pid', 'x', '--', 'a']), 2)
        with mock.patch.object(app_restart, 'wait_then_launch', return_value=True) as wait:
            self.assertEqual(app_restart.main(['--wait-pid', '5', '--cwd', 'C:/', '--', sys.executable]), 0)
            wait.assert_called_once_with(5, [sys.executable], 'C:/')


if __name__ == '__main__':
    unittest.main()
