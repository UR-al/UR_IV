"""대기열 매니저 ↔ 자동화 '큐 우선' — 같은 대기열을 두 소비자가 다투지 않는가 (감사 #12 후속).

회귀(재현: scratchpad p2_mark_clobber.py)
- 자동화가 큐 우선 항목 Q 를 생성하는 동안 대기열 '시작'을 누르면(자동화는 isRunning 을 세우지 않아
  버튼이 보였다) 대기열 매니저가 Q 를 다시 보냈고, start_generation 이 '생성 중'으로 거절하자 자동화가
  건 '생성 중' 표시를 풀었다. 그 뒤 Q 를 지울 수 있게 되었고, 자동화가 끝나며 맨 앞 항목을 지워
  생성하지 않은 R 이 사라졌다. 같은 디스패치가 생성 중에 Q 의 값을 UI 에 덮어쓰기도 했다.
- 지금은 표시에 주인이 있고, 자동화가 도는 동안 대기열 시작을, 대기열이 도는 동안 자동화 시작을
  거절하며, 자동화는 맨 앞이 아니라 **그 항목**(id)을 소비한다.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QTimer

from core.queue_model import AUTOMATION_OWNER, QUEUE_OWNER
from tests.test_automation_pause_edit import _StatusHost
from ui.generator_generation import GenerationMixin
from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin
from ui.queue_coordination import (
    AUTOMATION_BLOCKED_BY_QUEUE,
    QUEUE_BLOCKED_BY_AUTOMATION,
    RUNNING_ITEM_KEPT,
    announce,
    automation_start_refusal,
    queue_start_refusal,
    running_item_kept,
)
from widgets.queue_manager import AUTOMATION_NOTICE, QueueManager
from widgets.queue_panel import QueuePanel


class _Host(_StatusHost):
    """자동화(ActionsMixin) + 실제 QueuePanel/QueueManager — ui.generator_main._setup_queue 와 같은 배선."""

    _on_generation_requested = GeneratorMainUI._on_generation_requested
    _generation_matches_queue = GenerationMixin._generation_matches_queue

    def __init__(self, prompts, tmp, **settings):
        super().__init__(prompts, **settings)
        self.queue_panel = QueuePanel(state_path=Path(tmp) / 'q.json', prefs_path=Path(tmp) / 'p.json',
                                      schedule=lambda _ms, _fn: None, restore=False)
        self.queue_manager = QueueManager(self.queue_panel)
        self.queue_manager.delay_seconds = 0
        self.queue_manager.generation_requested.connect(self._on_generation_requested)
        self.queue_manager.generation_active = self._auto_generation_in_flight
        self.queue_manager.start_blocker = lambda: queue_start_refusal(self)
        self.queue_manager.notice.connect(lambda message: announce(self, message))

    def start_generation(self, **kwargs):
        # 실제 start_generation 처럼 워커가 결과를 내기 전이면 거절한다
        if self._auto_generation_in_flight():
            return False
        return super().start_generation(**kwargs)

    def finish(self, success):
        """on_generation_finished 와 같은 순서 — 자동화 집계 → 대기열 매니저 → 자동화 계속."""
        worker = getattr(self, 'gen_worker', None)
        if worker is not None:
            worker._result_emitted = True
        if success:
            if self.is_automating:
                self._automation_after_generation(True)
        elif self.is_automating and self._automation_after_generation(False):
            return
        if self._generation_matches_queue({}):
            self.queue_manager.on_generation_completed(success)
        if self.is_automating:
            self._continue_automation()

    def notifications(self):
        return [args[1] for args in self.vue_bridge.showNotification.calls]


class _Case(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        timers = mock.patch.object(QTimer, 'singleShot',
                                   side_effect=lambda ms, fn: self.host.timers.append((ms, fn)))
        timers.start()
        self.addCleanup(timers.stop)
        box = mock.patch('ui.generator_actions.QMessageBox')
        box.start()
        self.addCleanup(box.stop)

    def host_with(self, prompts, *queued, **settings):
        self.host = _Host(prompts, self._tmp.name, **settings)
        ids = [self.host.queue_panel.add_single_item({'prompt': p}) for p in queued]
        return self.host, ids

    def start_automation(self):
        self.host._start_automation()
        self.host.run_timers()

    def step(self, success=True):
        self.host.finish(success)
        self.host.run_timers()

    def queued(self):
        return [item['prompt'] for item in self.host.queue_panel.queue_items]


class QueueStartDuringAutomationTests(_Case):
    def test_queue_start_during_a_queue_first_item_cannot_lose_the_next_item(self):
        host, (q, r) = self.host_with(['P1', 'P2'], 'Q', 'R')
        self.start_automation()                                   # 덱 P1
        self.step()                                               # → 큐 우선 Q
        self.assertEqual(host.started[-1], ('Q', True))
        panel = host.queue_panel
        self.assertEqual((panel.current_processing_id, panel.processing_owner), (q, AUTOMATION_OWNER))
        applied = len(host.ui_applied)

        # 사용자가 대기열 '시작' — 자동화가 대기열을 먼저 처리하므로 거절
        self.assertFalse(host.queue_manager.start())
        self.assertEqual(host.queue_manager.last_start_refusal, QUEUE_BLOCKED_BY_AUTOMATION)
        self.assertFalse(host.queue_manager.is_running)

        # 가드를 지나온 경우라도(경합) Q 를 다시 보내지도, 자동화의 표시를 풀지도 않는다
        host.queue_manager.start_blocker = None
        self.assertTrue(host.queue_manager.start())
        self.assertTrue(host.queue_manager.is_paused)
        self.assertIn(AUTOMATION_NOTICE, host.notifications())
        self.assertEqual(len(host.ui_applied), applied)           # 생성 중에 UI 를 덮어쓰지 않았다
        self.assertEqual((panel.current_processing_id, panel.processing_owner), (q, AUTOMATION_OWNER))
        host.queue_manager.stop()                                 # 중지도 남의 표시는 풀지 않는다
        self.assertEqual(panel.current_processing_id, q)

        self.assertEqual(panel.remove_items_by_ids([q]), 0)      # 생성 중 — 지우지 못한다
        self.step()                                               # 자동화의 Q 완료
        self.assertEqual(host.started[-1], ('R', True))           # R 은 살아남아 다음으로 생성된다
        self.assertEqual(self.queued(), ['R'])
        self.assertEqual((panel.current_processing_id, panel.processing_owner), (r, AUTOMATION_OWNER))
        self.step()                                               # R 완료 → 덱으로
        self.assertEqual(self.queued(), [])
        self.assertEqual(host.started[-1], ('P2', False))

    def test_automation_consumes_the_item_it_generated_not_the_first_one(self):
        host, (q, r) = self.host_with(['P1', 'P2'], 'Q', 'R')
        self.start_automation()
        self.step()                                               # → Q 생성 중
        # 예전 경로(표시가 풀린 뒤 사용자가 Q 를 지움)를 재현 — 자동화 정리가 R 을 지우면 안 된다
        host.queue_panel.release_processing(q, AUTOMATION_OWNER)
        host.queue_panel.remove_items_by_ids([q])
        self.step()
        self.assertEqual(host.started[-1], ('R', True))
        self.assertEqual(self.queued(), ['R'])

    def test_a_failed_queue_first_start_releases_only_its_own_mark(self):
        host, (q, _r) = self.host_with(['P1', 'P2'], 'Q', 'R')
        self.start_automation()
        host.start_generation = lambda **_kw: False                # 검증 실패 등
        self.step()
        self.assertFalse(host._auto_processing_queue)
        self.assertIsNone(host._auto_queue_item_id)
        self.assertFalse(host.queue_panel.is_processing)
        self.assertEqual(host.queue_panel.remove_items_by_ids([q]), 1)

    def test_stopping_automation_releases_its_mark_and_keeps_the_item(self):
        host, (q, _r) = self.host_with(['P1', 'P2'], 'Q', 'R')
        self.start_automation()
        self.step()                                               # → Q 생성 중
        host._stop_automation()
        self.assertFalse(host.queue_panel.is_processing)
        self.assertEqual(self.queued(), ['Q', 'R'])


class AutomationStartDuringQueueTests(_Case):
    def test_automation_does_not_start_while_the_queue_runs(self):
        host, (a,) = self.host_with(['P1'], 'A')
        self.assertTrue(host.queue_manager.start())
        self.assertEqual((host.queue_panel.current_processing_id, host.queue_panel.processing_owner),
                         (a, QUEUE_OWNER))
        self.start_automation()
        self.assertFalse(host.is_automating)
        self.assertIn(AUTOMATION_BLOCKED_BY_QUEUE, host.notifications())
        self.assertEqual(host.queue_panel.processing_owner, QUEUE_OWNER)

    def test_a_queue_first_item_owned_by_the_queue_stops_automation_without_touching_the_mark(self):
        host, (a,) = self.host_with(['P1'], 'A')
        host.queue_manager.start()                                # A 를 대기열이 생성 중
        host.is_automating = True                                 # 가드를 지나온 경우(경합)
        host._start_automation_queue_item(host.queue_panel, host.queue_panel.get_first_item())
        self.assertFalse(host.is_automating)
        self.assertFalse(host._auto_processing_queue)
        self.assertEqual((host.queue_panel.current_processing_id, host.queue_panel.processing_owner),
                         (a, QUEUE_OWNER))


class GenerationRoutingTests(unittest.TestCase):
    """_generation_matches_queue — 중지 뒤에도 워커가 만드는 항목의 결과는 대기열 매니저로 간다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.panel = QueuePanel(state_path=Path(self._tmp.name) / 'q.json',
                                prefs_path=Path(self._tmp.name) / 'p.json',
                                schedule=lambda _ms, _fn: None, restore=False)
        self.manager = QueueManager(self.panel)
        self.active = [False]
        self.manager.generation_active = lambda: self.active[0]
        self.manager.generation_requested.connect(lambda _item: self.active.__setitem__(0, True))
        self.host = SimpleNamespace(queue_manager=self.manager)

    def matches(self, info):
        return GenerationMixin._generation_matches_queue(self.host, info)

    def test_idle_stopped_manager_ignores_results(self):
        self.panel.add_single_item({'prompt': 'a'})
        self.assertFalse(self.matches({}))

    def test_result_of_an_item_left_generating_after_stop_is_routed(self):
        self.panel.add_single_item({'prompt': 'a'})
        self.manager.start()
        self.manager.stop()
        self.assertTrue(self.matches({}))
        self.active[0] = False
        self.manager.on_generation_completed(True)
        self.assertTrue(self.panel.is_empty())
        self.assertFalse(self.matches({}))

    def test_xyz_result_is_compared_with_the_awaited_item(self):
        info = {'_xyz_info': {'requestId': 'plot', 'index': 0}}
        self.panel.add_single_item({'prompt': 'x', **info})
        self.manager.start()
        self.manager.stop()
        self.assertTrue(self.matches(info))
        self.assertFalse(self.matches({'_xyz_info': {'requestId': 'plot', 'index': 1}}))
        self.assertFalse(self.matches({}))


class _Recorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _ActionHost(ModelDownloadActionsMixin):
    """generator_main._handle_vue_action 의 대기열 분기만 — 실제 QueuePanel/QueueManager 배선."""

    _handle_vue_action = GeneratorMainUI._handle_vue_action

    def __init__(self, tmp):
        self.vue_bridge = SimpleNamespace(showNotification=_Recorder())
        self.statuses = []
        self.synced = 0
        self.is_automating = False
        self.queue_panel = QueuePanel(state_path=Path(tmp) / 'q.json', prefs_path=Path(tmp) / 'p.json',
                                      schedule=lambda _ms, _fn: None, restore=False)
        self.queue_manager = QueueManager(self.queue_panel)
        self.queue_manager.start_blocker = lambda: queue_start_refusal(self)
        self.queue_manager.generation_requested.connect(lambda _item: None)

    def show_status(self, message, *_args):
        self.statuses.append(message)

    def _sync_queue_to_vue(self):
        self.synced += 1

    def _handle_chat_action(self, _action, _payload):
        return False

    def _handle_creator_action(self, _action, _payload):
        return False

    def toasts(self):
        return [args[1] for args in self.vue_bridge.showNotification.calls]


class QueueActionHandlerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.host = _ActionHost(self._tmp.name)

    def add(self, *prompts):
        return [self.host.queue_panel.add_single_item({'prompt': p}) for p in prompts]

    def test_start_queue_explains_why_it_did_not_start(self):
        self.host._handle_vue_action('start_queue', {})
        self.assertEqual(self.host.toasts(), ['대기열이 비어 있어 시작할 항목이 없습니다'])
        self.add('a')
        self.host.is_automating = True
        self.host._handle_vue_action('start_queue', {})
        self.assertEqual(self.host.toasts()[-1], QUEUE_BLOCKED_BY_AUTOMATION)
        self.assertFalse(self.host.queue_manager.is_running)
        self.host.is_automating = False
        self.host._handle_vue_action('start_queue', {})
        self.assertTrue(self.host.queue_manager.is_running)
        self.assertEqual(self.host.statuses[-1], 'Queue started.')

    def test_clear_queue_says_the_generating_row_was_kept(self):
        a, _b, _c = self.add('a', 'b', 'c')
        self.host.queue_panel.claim_processing(a, AUTOMATION_OWNER)
        self.host._handle_vue_action('clear_queue', {})
        self.assertEqual([item['id'] for item in self.host.queue_panel.queue_items], [a])
        self.assertEqual(self.host.toasts(), [RUNNING_ITEM_KEPT])

    def test_clear_queue_without_a_generating_row_is_quiet(self):
        self.add('a', 'b')
        self.host._handle_vue_action('clear_queue', {})
        self.assertTrue(self.host.queue_panel.is_empty())
        self.assertEqual(self.host.toasts(), [])
        self.assertEqual(self.host.statuses[-1], '2개 항목 삭제')

    def test_removing_the_generating_row_warns_and_keeps_it(self):
        a, b = self.add('a', 'b')
        self.host.queue_panel.claim_processing(a, QUEUE_OWNER)
        self.host._handle_vue_action('remove_queue_items', {'item_ids': [a, b]})
        self.assertEqual([item['id'] for item in self.host.queue_panel.queue_items], [a])
        self.assertEqual(self.host.toasts(), [RUNNING_ITEM_KEPT])
        self.host._handle_vue_action('remove_queue_items', {'item_ids': ['missing']})
        self.assertEqual(self.host.statuses[-1], '0개 항목 삭제')

    def test_sync_queue_state_pushes_the_current_queue(self):
        self.host._handle_vue_action('sync_queue_state', {})
        self.assertEqual(self.host.synced, 1)


class CoordinationHelperTests(unittest.TestCase):
    def test_refusals_follow_the_other_consumer(self):
        self.assertEqual(queue_start_refusal(SimpleNamespace(is_automating=True)), QUEUE_BLOCKED_BY_AUTOMATION)
        self.assertEqual(queue_start_refusal(SimpleNamespace()), '')
        running = SimpleNamespace(queue_manager=SimpleNamespace(is_running=True))
        self.assertEqual(automation_start_refusal(running), AUTOMATION_BLOCKED_BY_QUEUE)
        self.assertEqual(automation_start_refusal(SimpleNamespace()), '')

    def test_running_item_kept_only_reports_a_surviving_generating_row(self):
        panel = SimpleNamespace(current_processing_id='a',
                                get_item_by_id=lambda i: {'id': i} if i == 'a' else None)
        self.assertTrue(running_item_kept(panel))
        self.assertTrue(running_item_kept(panel, ['a', 'b']))
        self.assertFalse(running_item_kept(panel, ['b']))
        self.assertFalse(running_item_kept(SimpleNamespace(current_processing_id=None)))
        gone = SimpleNamespace(current_processing_id='a', get_item_by_id=lambda _i: None)
        self.assertFalse(running_item_kept(gone))

    def test_announce_reaches_status_line_and_toast(self):
        shown, toasts = [], []
        host = SimpleNamespace(show_status=shown.append,
                               vue_bridge=SimpleNamespace(showNotification=SimpleNamespace(
                                   emit=lambda level, text: toasts.append((level, text)))))
        announce(host, RUNNING_ITEM_KEPT)
        announce(host, '')
        self.assertEqual(shown, [RUNNING_ITEM_KEPT])
        self.assertEqual(toasts, [('warning', RUNNING_ITEM_KEPT)])
        announce(SimpleNamespace(), 'no bridge')                 # 조용히 넘어간다


if __name__ == '__main__':
    unittest.main()
