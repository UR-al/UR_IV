"""core.console_interrupt — 콘솔 Ctrl+C(SIGINT·KeyboardInterrupt)를 Qt 앱의 정상 종료로 잇는다.

공용 크래시 훅이 슬롯 예외를 기록만 하고 삼키게 되자, 웹 모드에서 Ctrl+C 가 슬롯(1초 VRAM 타이머)
안에서 KeyboardInterrupt 로 나도 'UNHANDLED EXCEPTION' 만 찍히고 서버는 계속 돌았다(예전에는 PyQt
기본 동작이 프로세스를 끝냈다). 단위 테스트는 가짜 앱으로 분기를, 마지막 클래스는 실제
QCoreApplication 을 자식 프로세스에서 돌려 끝까지 본다(이 프로세스의 QApplication 을 건드리지 않게).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from core import console_interrupt as ci

ROOT = Path(__file__).resolve().parents[1]


class _FakeApp:
    def __init__(self):
        self.calls = []
        self.running_flags = []

    def quit(self):
        self.calls.append("quit")

    def exit(self, code=0):
        self.calls.append(("exit", code))

    def exec(self):
        self.running_flags.append(ci.main_loop_running())
        return 7


class _StateReset(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.multiple(ci, _main_loop_running=False, _quit_requested=False)
        patcher.start()
        self.addCleanup(patcher.stop)


class RequestQuitTests(_StateReset):
    def test_without_an_app_nothing_is_requested(self):
        with mock.patch.object(ci, "_qt_app", return_value=None):
            self.assertFalse(ci.request_quit(loop_level=1))

    def test_outside_any_event_loop_the_caller_keeps_the_default(self):
        app = _FakeApp()
        self.assertFalse(ci.request_quit(app, loop_level=0))
        self.assertEqual(app.calls, [])
        self.assertFalse(ci.quit_requested())

    def test_main_loop_quits_through_the_graceful_path(self):
        app = _FakeApp()
        with mock.patch.object(ci, "_main_loop_running", True):
            self.assertTrue(ci.request_quit(app, loop_level=1))
        self.assertEqual(app.calls, ["quit"])
        self.assertFalse(ci.quit_requested())

    def test_nested_loop_before_the_main_loop_exits_and_remembers(self):
        """Qt 6 은 exec 밖의 quit() 을 무시한다 — 중첩 루프를 끝내고 요청을 기억한다."""
        app = _FakeApp()
        self.assertTrue(ci.request_quit(app, loop_level=1))
        self.assertEqual(app.calls, [("exit", 0)])
        self.assertTrue(ci.quit_requested())


class RunMainLoopTests(_StateReset):
    def test_marks_the_main_loop_and_returns_exec_code(self):
        app = _FakeApp()
        self.assertEqual(ci.run_main_loop(app), 7)
        self.assertEqual(app.running_flags, [True])
        self.assertFalse(ci.main_loop_running())

    def test_flag_is_cleared_even_if_exec_raises(self):
        app = _FakeApp()
        app.exec = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            ci.run_main_loop(app)
        self.assertFalse(ci.main_loop_running())

    def test_a_quit_requested_during_startup_is_applied_as_soon_as_the_loop_starts(self):
        app = _FakeApp()
        ci.request_quit(app, loop_level=1)   # 시작 대화상자(중첩 루프) 중 Ctrl+C
        with mock.patch("PyQt6.QtCore.QTimer.singleShot") as single_shot:
            ci.run_main_loop(app)
        single_shot.assert_called_once_with(0, app.quit)
        self.assertFalse(ci.quit_requested())

    def test_no_quit_is_scheduled_without_a_request(self):
        with mock.patch("PyQt6.QtCore.QTimer.singleShot") as single_shot:
            ci.run_main_loop(_FakeApp())
        single_shot.assert_not_called()


class SigintHandlerTests(_StateReset):
    def test_inside_the_event_loop_ctrl_c_requests_quit(self):
        app = _FakeApp()
        said = []
        handler = ci.make_sigint_handler(app, loop_level=lambda: 1, notify=said.append)
        with mock.patch.object(ci, "_main_loop_running", True):
            handler(signal.SIGINT, None)   # 예외 없이 돌아온다
        self.assertEqual(app.calls, ["quit"])
        self.assertEqual(len(said), 1)

    def test_outside_the_event_loop_ctrl_c_is_a_keyboard_interrupt(self):
        app = _FakeApp()
        handler = ci.make_sigint_handler(app, loop_level=lambda: 0, notify=lambda _t: None)
        with self.assertRaises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
        self.assertEqual(app.calls, [])

    def test_install_registers_the_handler(self):
        original = signal.getsignal(signal.SIGINT)
        self.addCleanup(signal.signal, signal.SIGINT, original)
        self.assertIsNone(ci.install_sigint_quit(_FakeApp(), keepalive_ms=0))
        self.assertTrue(callable(signal.getsignal(signal.SIGINT)))
        self.assertIsNot(signal.getsignal(signal.SIGINT), signal.default_int_handler)

    def test_web_entry_point_installs_it_and_both_entry_points_run_the_marked_loop(self):
        web = (ROOT / "web_main_ui.py").read_text(encoding="utf-8")
        self.assertIn("install_sigint_quit(app)", web)
        for entry in ("web_main_ui.py", "new_main_ui.py"):
            source = (ROOT / entry).read_text(encoding="utf-8")
            with self.subTest(entry=entry):
                self.assertIn("sys.exit(run_main_loop(app))", source)
                self.assertNotIn("sys.exit(app.exec())", source)


class WebStartupQuitTests(unittest.TestCase):
    """웹 모드 '브라우저 접속 대기' 중 Ctrl+C — 종료 정리 뒤에 백엔드 연결 워커를 새로 띄우지 않는다."""

    def _run_startup(self, during_wait):
        import io
        from types import SimpleNamespace

        import web_main_ui

        window = SimpleNamespace(
            _startup_backend_check=mock.Mock(), _apply_backend_startup_result=mock.Mock(),
            _restore_search_deck=mock.Mock(),
        )
        server = SimpleNamespace(_had_client=False, _on_first_client=None)
        timers = []

        class FakeLoop:
            def quit(self):
                pass

            def exec(self):
                during_wait(server, timers)
                return 0

        with mock.patch.object(web_main_ui, "QEventLoop", FakeLoop), \
                mock.patch.object(web_main_ui.QTimer, "singleShot",
                                  side_effect=lambda ms, fn: timers.append((ms, fn))), \
                mock.patch("sys.stdout", io.StringIO()):
            web_main_ui._web_startup(window, object(), server)
        return window

    def test_quit_while_waiting_skips_backend_apply(self):
        window = self._run_startup(lambda server, timers: None)   # 앱 종료가 대기 루프를 끝냈다
        window._startup_backend_check.assert_called_once_with()
        window._apply_backend_startup_result.assert_not_called()

    def test_first_client_continues_startup(self):
        def connect(server, timers):
            server._had_client = True

        self._run_startup(connect)._apply_backend_startup_result.assert_called_once_with()

    def test_wait_timeout_still_continues_startup(self):
        def time_out(server, timers):
            [fn for ms, fn in timers if ms == 60000][0]()

        self._run_startup(time_out)._apply_backend_startup_result.assert_called_once_with()


_CHILD = textwrap.dedent(
    r"""
    import os, signal, sys
    sys.path.insert(0, os.environ["URIV_ROOT"])
    from PyQt6.QtCore import QCoreApplication, QEventLoop, QTimer
    from core.crash_hooks import install_crash_handlers
    from core.console_interrupt import install_sigint_quit, run_main_loop

    app = QCoreApplication(sys.argv)
    crash_fp = install_crash_handlers(os.environ["URIV_CRASH_LOG"], label="web")
    keepalive = install_sigint_quit(app, keepalive_ms=100)
    app.aboutToQuit.connect(lambda: print("ABOUT_TO_QUIT", flush=True))

    def watchdog(name):
        QTimer.singleShot(4000, lambda: (print("STILL RUNNING " + name, flush=True), app.exit(3)))

    def raise_in_slot():
        raise KeyboardInterrupt   # 기본 SIGINT 처리기가 슬롯 안에서 내는 모양 그대로

    # 1) 메인 루프 슬롯 안의 KeyboardInterrupt → 크래시 훅이 종료 요청으로 넘긴다
    QTimer.singleShot(0, raise_in_slot)
    watchdog("slot")
    print("RC slot", run_main_loop(app), flush=True)

    # 2) 메인 루프 중 실제 SIGINT → 처리기가 quit
    QTimer.singleShot(50, lambda: signal.raise_signal(signal.SIGINT))
    watchdog("sigint")
    print("RC sigint", run_main_loop(app), flush=True)

    # 3) 메인 루프 전 중첩 루프(시작 대화상자 등) 중 SIGINT → 중첩 루프를 끝내고 메인 루프가 곧바로 quit
    loop = QEventLoop()
    QTimer.singleShot(50, lambda: signal.raise_signal(signal.SIGINT))
    QTimer.singleShot(4000, lambda: (print("STILL RUNNING nested", flush=True), loop.quit()))
    loop.exec()
    print("NESTED RETURNED", flush=True)
    watchdog("pending")
    print("RC pending", run_main_loop(app), flush=True)
    crash_fp.close()
    """
)


class RealEventLoopTests(unittest.TestCase):
    """실제 Qt 이벤트 루프 — 리뷰 재현(훅 설치 시 KeyboardInterrupt 뒤 'STILL RUNNING')이 사라졌는지."""

    def test_ctrl_c_quits_a_real_event_loop_in_every_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            crash_log = os.path.join(temp, "last_crash_web.log")
            env = dict(os.environ, URIV_ROOT=str(ROOT), URIV_CRASH_LOG=crash_log, PYTHONIOENCODING="utf-8")
            proc = subprocess.run(
                [sys.executable, "-c", _CHILD], cwd=str(ROOT), env=env, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
            )
            crash_text = Path(crash_log).read_text(encoding="utf-8") if os.path.exists(crash_log) else ""
        out = proc.stdout
        self.assertEqual(proc.returncode, 0, out + proc.stderr)
        self.assertNotIn("STILL RUNNING", out, out)
        self.assertNotIn("UNHANDLED EXCEPTION", out + proc.stderr)
        for phase in ("slot", "sigint", "pending"):
            self.assertIn(f"RC {phase} 0", out)
        self.assertIn("NESTED RETURNED", out)
        self.assertEqual(out.count("ABOUT_TO_QUIT"), 3, out)
        self.assertNotIn("KeyboardInterrupt", crash_text, "Ctrl+C 는 크래시가 아니다 — 크래시 로그에 남기지 않는다")


if __name__ == "__main__":
    unittest.main()
