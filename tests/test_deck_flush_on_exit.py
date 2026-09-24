"""덱 진행도 모아 저장(DeckSaveThrottle)의 빈틈 — 웹 모드 종료·강제 종료에도 진행이 남는지.

뽑을 때마다 쓰지 않고 20장·60초마다 저장하게 바꾼 뒤, 데스크톱 _quit_app 만 모아 둔 진행을
flush 했다. 웹 모드 종료(aboutToQuit)는 그 경로를 타지 않아 마지막 저장 뒤 뽑은 진행(최대 19장)을
잃었고, 콘솔을 닫는 식의 강제 종료는 어떤 훅도 돌지 않는다.
"""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QTimer

import web_main_ui
from tests.test_search_deck import _DeckHost, _rows
from ui.generator_actions import ActionsMixin


class WebQuitFlushTests(unittest.TestCase):
    def test_quit_hook_flushes_the_pending_deck_progress(self):
        calls = []
        web_main_ui._flush_window_state_on_quit(SimpleNamespace(_flush_deck_state=lambda: calls.append(1)))
        self.assertEqual(calls, [1])

    def test_quit_hook_writes_a_dirty_deck_and_skips_a_clean_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last_deck.json'
            host = _DeckHost(path, _rows(*('g' * 10)))
            host.is_automating = True
            with mock.patch('PyQt6.QtCore.QCoreApplication.instance', return_value=None):
                host.apply_random_prompt()               # 리필 저장
                host.apply_random_prompt()               # 모아 둔 뽑기 1장
            self.assertTrue(host._deck_throttle().dirty)
            with mock.patch.object(ActionsMixin, '_save_deck_state', autospec=True,
                                   side_effect=ActionsMixin._save_deck_state) as save:
                web_main_ui._flush_window_state_on_quit(host)
                web_main_ui._flush_window_state_on_quit(host)
            self.assertEqual(save.call_count, 1)
            self.assertEqual(len(json.loads(path.read_text(encoding='utf-8'))['remaining']), 8)

    def test_quit_hook_never_blocks_shutdown(self):
        def boom():
            raise OSError('disk full')
        web_main_ui._flush_window_state_on_quit(SimpleNamespace(_flush_deck_state=boom))
        web_main_ui._flush_window_state_on_quit(SimpleNamespace())   # 메서드가 없어도 조용히

    def test_about_to_quit_handler_calls_the_flush_first(self):
        src = inspect.getsource(web_main_ui.main)
        body = src[src.index('def _shutdown_servers'):src.index('app.aboutToQuit.connect(_shutdown_servers)')]
        self.assertIn('_flush_window_state_on_quit(window)', body)
        self.assertLess(body.index('_flush_window_state_on_quit(window)'), body.index('web_server.close()'))


class TrailingFlushTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / 'last_deck.json'
        self.timers = []
        patches = (
            mock.patch.object(QTimer, 'singleShot', side_effect=lambda ms, fn: self.timers.append((ms, fn))),
            mock.patch('PyQt6.QtCore.QCoreApplication.instance', return_value=object()),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        self._tmp.cleanup()

    def test_batched_draws_are_written_once_the_interval_passes_without_more_draws(self):
        host = _DeckHost(self.path, _rows(*('g' * 30)))
        host.is_automating = True
        with mock.patch.object(ActionsMixin, '_save_deck_state', autospec=True,
                               side_effect=ActionsMixin._save_deck_state) as save:
            for _ in range(3):
                host.apply_random_prompt()
            self.assertEqual(save.call_count, 1)                      # 첫 리필 저장만
            self.assertEqual(len(self.timers), 1)                     # 뒤늦은 저장은 하나만 걸린다
            ms, fire = self.timers.pop()
            self.assertEqual(ms, int(host._deck_throttle().interval * 1000))
            fire()                                                    # 그 사이 더 안 뽑았다
            self.assertEqual(save.call_count, 2)
        stored = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertEqual(len(stored['remaining']), 27)

        host.apply_random_prompt()                                    # 다시 모으기 시작하면 다시 건다
        self.assertEqual(len(self.timers), 1)

    def test_no_timer_without_an_event_loop(self):
        host = _DeckHost(self.path, _rows(*('g' * 5)))
        host.is_automating = True   # 리필 안내 모달 없이
        with mock.patch('PyQt6.QtCore.QCoreApplication.instance', return_value=None):
            host.apply_random_prompt()
            host.apply_random_prompt()
        self.assertEqual(self.timers, [])
        self.assertFalse(getattr(host, '_deck_flush_scheduled', False))


if __name__ == '__main__':
    unittest.main()
