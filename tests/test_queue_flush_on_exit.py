"""대기열 모아 저장(PERSIST_DELAY_MS)의 종료 배선 — 마지막 변경이 앱 종료에 사라지지 않는가.

대기열은 변경을 250ms 모았다가 한 번 쓴다(widgets/queue_panel.py). 그래서 종료 직전의 변경(XYZ 일괄
추가 · 이벤트 시나리오 등)은 종료 훅이 flush 해야 남는다. 데스크톱은 _quit_app(os._exit 전),
웹 모드는 aboutToQuit(_flush_window_state_on_quit)이 그 일을 한다 — 호출이 빠지거나 os._exit 뒤로
밀려도 조용히 사라지므로 배선을 여기서 지킨다.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import web_main_ui
from ui.generator_main import GeneratorMainUI
from widgets.queue_panel import QueuePanel


class _Exited(Exception):
    pass


class _QueueHostCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / 'queue_state.json'
        # 예약된 저장이 절대 스스로 돌지 않는 스케줄러 — flush 만이 쓴다
        self.panel = QueuePanel(state_path=self.state, prefs_path=Path(self._tmp.name) / 'p.json',
                                schedule=lambda _ms, _fn: None, restore=False)

    def stored_prompts(self):
        return [item['prompt'] for item in json.loads(self.state.read_text(encoding='utf-8'))['items']]


class FlushQueueStateTests(_QueueHostCase):
    def test_flush_writes_the_pending_queue(self):
        self.panel.add_single_item({'prompt': 'xyz-1'})
        self.panel.add_single_item({'prompt': 'xyz-2'})
        self.assertFalse(self.state.exists())
        GeneratorMainUI._flush_queue_state(SimpleNamespace(queue_panel=self.panel))
        self.assertEqual(self.stored_prompts(), ['xyz-1', 'xyz-2'])

    def test_flush_without_a_queue_is_a_no_op(self):
        GeneratorMainUI._flush_queue_state(SimpleNamespace())


class WebQuitHookTests(_QueueHostCase):
    def test_quit_hook_flushes_the_queue(self):
        self.panel.add_single_item({'prompt': 'last'})
        host = SimpleNamespace(queue_panel=self.panel)
        host._flush_queue_state = lambda: GeneratorMainUI._flush_queue_state(host)
        web_main_ui._flush_window_state_on_quit(host)
        self.assertEqual(self.stored_prompts(), ['last'])

    def test_a_failing_deck_flush_does_not_skip_the_queue_and_nothing_raises(self):
        calls = []

        def deck_boom():
            calls.append('deck')
            raise OSError('disk full')

        def queue_boom():
            calls.append('queue')
            raise OSError('disk full')

        web_main_ui._flush_window_state_on_quit(
            SimpleNamespace(_flush_deck_state=deck_boom, _flush_queue_state=queue_boom))
        self.assertEqual(calls, ['deck', 'queue'])


class DesktopQuitTests(_QueueHostCase):
    def _host(self, flush_queue):
        host = mock.MagicMock()
        host.queue_panel = self.panel
        host._flush_queue_state = flush_queue
        host.gen_worker = host._search_worker = host.info_worker = None
        host.vue_bridge = SimpleNamespace(_stale_search_workers=[])
        return host

    def _quit(self, host):
        written_at_exit = []

        def fake_exit(_code):
            written_at_exit.append(self.state.exists())
            raise _Exited()

        with mock.patch('core.app_instance.unregister_app_instance'), \
                mock.patch('ui.generator_main.os._exit', side_effect=fake_exit) as exit_:
            with self.assertRaises(_Exited):
                GeneratorMainUI._quit_app(host)
        exit_.assert_called_once_with(0)
        return written_at_exit

    def test_quit_writes_the_pending_queue_before_the_process_exits(self):
        self.panel.add_single_item({'prompt': 'queued just before quit'})
        host = self._host(None)
        host._flush_queue_state = lambda: GeneratorMainUI._flush_queue_state(host)
        self.assertEqual(self._quit(host), [True])
        self.assertEqual(self.stored_prompts(), ['queued just before quit'])

    def test_a_failing_queue_flush_still_exits(self):
        host = self._host(mock.Mock(side_effect=OSError('disk full')))
        self.assertEqual(self._quit(host), [False])
        host._flush_queue_state.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
