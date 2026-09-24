"""처리되지 않은 예외 훅(core.crash_hooks)과 '랜덤 해상도' 토글.

Vue '랜덤 해상도'를 끌 때 CheckBoxProxy.toggled → toggle_random_resolution_editor(False) 가
더미 LblProxy.clear() 를 불러 AttributeError 가 났다. 데스크톱은 excepthook 이 트레이스백만
남겼지만 웹 모드는 훅이 없어 PyQt6 기본 동작(qFatal)으로 프로세스가 끝날 수 있었다.
그 PyQt 편집기 배선(더미 라벨·컨테이너·목록)은 아무 효과가 없어 통째로 은퇴했다(audit #175) —
토글은 이제 값만 바꾸고, 목록은 Vue 편집기가 set_random_resolutions 로 보낸다.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core import crash_hooks

ROOT = Path(__file__).resolve().parents[1]


def _raise_and_capture():
    try:
        raise AttributeError("'LblProxy' object has no attribute 'clear'")
    except AttributeError:
        return sys.exc_info()


class ExceptHookTests(unittest.TestCase):
    def test_hook_prints_and_appends_traceback_to_crash_file(self):
        crash = io.StringIO()
        hook = crash_hooks.make_excepthook(crash, label="web")
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", out):
            hook(*_raise_and_capture())
        self.assertIn("UNHANDLED EXCEPTION (web):", out.getvalue())
        self.assertIn("LblProxy", out.getvalue())
        self.assertIn("=== UNHANDLED PYTHON EXCEPTION ===", crash.getvalue())
        self.assertIn("AttributeError", crash.getvalue())

    def test_hook_never_raises_even_if_the_crash_file_is_broken(self):
        broken = mock.Mock()
        broken.write.side_effect = OSError("disk full")
        hook = crash_hooks.make_excepthook(broken)
        with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            hook(*_raise_and_capture())   # 예외가 새지 않는다

    def test_install_sets_excepthook_and_faulthandler_on_the_log_file(self):
        original = sys.excepthook
        self.addCleanup(setattr, sys, "excepthook", original)
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "logs", "last_crash.log")
            with mock.patch("faulthandler.enable") as enable:
                fp = crash_hooks.install_crash_handlers(path, label="web")
            try:
                self.assertIsNotNone(fp)
                enable.assert_called_once_with(fp)
                self.assertIsNot(sys.excepthook, original)
                with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    sys.excepthook(*_raise_and_capture())
                fp.flush()
                self.assertIn("LblProxy", Path(path).read_text(encoding="utf-8"))
            finally:
                fp.close()

    def test_install_still_hooks_when_the_log_cannot_be_opened(self):
        original = sys.excepthook
        self.addCleanup(setattr, sys, "excepthook", original)
        with mock.patch("core.crash_hooks.open_crash_log", return_value=None), \
                mock.patch("faulthandler.enable") as enable:
            fp = crash_hooks.install_crash_handlers("unused.log")
        self.assertIsNone(fp)
        enable.assert_not_called()
        self.assertIsNot(sys.excepthook, original)

    def test_both_entry_points_install_the_shared_hook(self):
        for entry in ("new_main_ui.py", "web_main_ui.py"):
            source = (ROOT / entry).read_text(encoding="utf-8")
            self.assertIn("install_crash_handlers(", source, f"{entry} 가 예외 훅을 설치하지 않는다")


def _interrupt_exc_info():
    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        return sys.exc_info()


class KeyboardInterruptTests(unittest.TestCase):
    """Ctrl+C 는 크래시가 아니다 — 훅이 기록하고 삼키면 웹 서버를 콘솔에서 멈출 수 없었다."""

    def test_interrupt_requests_quit_and_is_not_logged_as_a_crash(self):
        crash = io.StringIO()
        on_interrupt = mock.Mock(return_value=True)
        hook = crash_hooks.make_excepthook(crash, label="web", on_interrupt=on_interrupt)
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", out), \
                mock.patch("sys.__excepthook__") as default_hook:
            hook(*_interrupt_exc_info())
        on_interrupt.assert_called_once_with()
        default_hook.assert_not_called()
        self.assertNotIn("UNHANDLED EXCEPTION", out.getvalue())
        self.assertIn("Ctrl+C", out.getvalue())
        self.assertEqual(crash.getvalue(), "")

    def test_interrupt_outside_the_event_loop_falls_back_to_the_default_hook(self):
        crash = io.StringIO()
        hook = crash_hooks.make_excepthook(crash, on_interrupt=lambda: False)
        info = _interrupt_exc_info()
        with mock.patch("sys.__excepthook__") as default_hook, mock.patch("sys.stdout", io.StringIO()):
            hook(*info)
        default_hook.assert_called_once_with(*info)
        self.assertEqual(crash.getvalue(), "")

    def test_a_failing_interrupt_handler_never_escapes_the_hook(self):
        hook = crash_hooks.make_excepthook(None, on_interrupt=mock.Mock(side_effect=RuntimeError("no qt")))
        with mock.patch("sys.__excepthook__") as default_hook, mock.patch("sys.stdout", io.StringIO()):
            hook(*_interrupt_exc_info())
        default_hook.assert_called_once()

    def test_default_interrupt_handler_is_the_console_quit_request(self):
        hook = crash_hooks.make_excepthook(None)
        with mock.patch("core.console_interrupt.request_quit", return_value=True) as request, \
                mock.patch("sys.stdout", io.StringIO()):
            hook(*_interrupt_exc_info())
        request.assert_called_once_with()

    def test_other_exceptions_still_do_not_request_quit(self):
        on_interrupt = mock.Mock(return_value=True)
        hook = crash_hooks.make_excepthook(io.StringIO(), on_interrupt=on_interrupt)
        with mock.patch("sys.stdout", io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
            hook(*_raise_and_capture())
        on_interrupt.assert_not_called()


class CrashLogNameTests(unittest.TestCase):
    """데스크톱과 웹 모드는 동시에 뜰 수 있다 — 같은 파일을 'w' 로 열면 나중 쪽이 앞선 기록을 지운다."""

    def test_names_per_entry_point(self):
        self.assertEqual(crash_hooks.crash_log_name(), "last_crash.log")
        self.assertEqual(crash_hooks.crash_log_name(""), "last_crash.log")
        self.assertEqual(crash_hooks.crash_log_name("web"), "last_crash_web.log")
        self.assertEqual(crash_hooks.crash_log_name(" Web Mode/2 "), "last_crash_web_mode_2.log")
        self.assertEqual(crash_hooks.crash_log_name("../.."), "last_crash.log")

    def test_web_label_resolves_to_a_different_file_than_the_desktop_default(self):
        original = sys.excepthook
        self.addCleanup(setattr, sys, "excepthook", original)
        resolved = []

        def fake_log_file(name, *, legacy_paths=None):
            resolved.append((name, legacy_paths))
            return Path("logs") / name

        opened = []
        with mock.patch("core.storage_paths.log_file", side_effect=fake_log_file), \
                mock.patch("core.crash_hooks.open_crash_log", side_effect=lambda p: opened.append(p)):
            crash_hooks.install_crash_handlers()
            crash_hooks.install_crash_handlers(label="web")
        self.assertEqual(resolved, [("last_crash.log", "config/last_crash.log"), ("last_crash_web.log", None)])
        self.assertEqual([Path(p).name for p in opened], ["last_crash.log", "last_crash_web.log"])
        self.assertNotEqual(opened[0], opened[1])

    def test_web_launcher_shows_the_web_crash_log(self):
        launcher = (ROOT / "run_WEB_gui.bat").read_text(encoding="utf-8")
        self.assertIn(r"type logs\last_crash_web.log", launcher)


class RandomResolutionToggleTests(unittest.TestCase):
    """실제 _init_settings_proxies 가 만든 프록시로 Vue 토글 on→off 를 돌린다."""

    def test_turning_random_resolution_on_and_off_only_changes_the_value(self):
        from ui.generator_actions import ActionsMixin
        from ui.generator_ui_setup import UISetupMixin
        from ui.vue_bridge import VueBridge

        class Host(UISetupMixin, ActionsMixin):
            pass

        host = Host.__new__(Host)
        host.vue_bridge = VueBridge()
        UISetupMixin._init_settings_proxies(host)
        host.random_resolutions = [(512, 768, "portrait")]

        host.random_res_check._on_vue_changed("true")
        self.assertTrue(host.random_res_check.isChecked())
        host.random_res_check._on_vue_changed("false")   # 예전: AttributeError('LblProxy' .clear)
        self.assertFalse(host.random_res_check.isChecked())
        self.assertEqual(host.random_resolutions, [(512, 768, "portrait")])

        # 효과 없던 PyQt 편집기 배선은 돌아오지 않는다
        for name in ('toggle_random_resolution_editor', 'add_resolution_item', '_update_resolution_list',
                     'delete_resolution_item', '_update_random_res_label'):
            self.assertFalse(hasattr(ActionsMixin, name), name)
        for name in ('random_res_label', 'resolution_editor_container', 'resolution_list_widget',
                     'res_width_input', 'res_height_input', 'btn_add_res', '_res_presets'):
            self.assertFalse(hasattr(host, name), name)


if __name__ == "__main__":
    unittest.main()
