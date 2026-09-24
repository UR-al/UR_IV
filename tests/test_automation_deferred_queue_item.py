"""자동화 '큐 우선' 동결 항목이 백엔드에 닿기 전에 멈췄을 때(_queue_deferred) — 회귀.

동결 항목(시드 탐색 · XYZ · ComfyUI 스냅숏)은 backend_override 로 나가고, 워커가 GPU 리스를 못 잡거나
(인페인트·I2I·캡션 등이 쥐고 있음) 백엔드가 바뀌었으면 {'_queue_deferred': True} 로 돌아온다.
예전엔 on_generation_finished 가 이 결과를 대기열 매니저에게만 넘겼다 — 자동화 중엔 매니저가 멈춰 있어
무시됐고, 자동화는 '실행 중'인 채 새 생성도 타이머도 없이 멈췄으며 재개할 지점(_auto_resume_pending)이
비어 일시정지→재개로도 살릴 수 없었다(중지만 풀었다). 이제 대기열 매니저와 같은 규칙으로 항목은 남기고
자동화를 일시정지하며, 재개하면 같은 항목부터 다시 낸다.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core.queue_model import AUTOMATION_OWNER
from core.seed_explore import build_seed_explore_jobs
from tests.test_automation_pause_edit import _Flow, _Signal, _StatusHost
from tests.test_automation_retry import _Queue
from ui.generator_generation import GenerationMixin

_BUSY = '다른 생성 작업이 GPU 리소스를 사용 중입니다'


def _seed_items(backend, count=3):
    """시드 탐색 동결 항목(test_automation_pause_edit 와 같은 모양) — 첫 장은 덱이 대신한다."""
    from core.xyz_capabilities import backend_identity
    base = {'prompt': 'frozen <lora:x:0.7>', 'seed': -1, 'steps': 28, 'cfg_scale': 5,
            'width': 832, 'height': 1216}
    jobs = build_seed_explore_jobs(base, base_seed=777, model='anima.safetensors',
                                   backend_id=backend_identity('webui', backend),
                                   supports_subseed=True, count=count)
    return [{**job, 'id': i} for i, job in enumerate(jobs[1:], start=1)]


class _DeferHost(_StatusHost):
    """자동화 흐름(_StatusHost) + 실제 on_generation_finished 의 '보내지 못함' 분기."""

    on_generation_finished = GenerationMixin.on_generation_finished
    _restore_generate_button = GenerationMixin._restore_generate_button
    _generation_matches_queue = GenerationMixin._generation_matches_queue

    def __init__(self, prompts, **settings):
        super().__init__(prompts, **settings)
        self.vue_bridge.generationError = _Signal()
        # (더미 gen_progress_bar·viewer_label 은 없다 — 완료 처리가 닿으면 AttributeError 로 드러난다)
        self.refused = []

    def setWindowTitle(self, _title):
        pass

    def start_generation(self, **kwargs):
        # 실제 start_generation 처럼 워커가 결과를 내기 전이면 거절한다
        if self._auto_generation_in_flight():
            self.refused.append(kwargs)
            return False
        return super().start_generation(**kwargs)

    def defer(self, reason=_BUSY):
        """워커가 리스를 못 잡고 {'_queue_deferred': True} 를 낸 것과 같다."""
        self.gen_worker._result_emitted = True
        self.on_generation_finished(reason, {'_queue_deferred': True})


class AutomationDeferredQueueItemTests(_Flow):
    def setUp(self):
        super().setUp()
        self.backend = SimpleNamespace(api_url='http://127.0.0.1:7860')
        for patcher in (mock.patch('backends.get_backend', return_value=self.backend),
                        mock.patch('backends.get_backend_type', return_value=BackendType.WEBUI)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _host_at_frozen_item(self):
        host = _DeferHost(['P1', 'P2', 'P3'])
        items = _seed_items(self.backend, count=3)
        host.queue_panel = _Queue(items)
        self.start(host)                           # 덱 P1
        self.step(True)                            # → 동결 항목 1 (자동화가 '생성 중' 표시)
        self.assertEqual(host.queue_panel.processing, (1, AUTOMATION_OWNER))
        self.assertTrue(host._auto_processing_queue)
        return host, items

    def test_deferred_frozen_item_pauses_automation_and_keeps_the_item(self):
        host, items = self._host_at_frozen_item()
        sent = len(host.started)
        host.defer()
        host.run_timers()
        self.assertTrue(host.is_automating)
        self.assertTrue(host._auto_paused)
        self.assertEqual(host._auto_resume_pending, 'continue')
        self.assertFalse(host._auto_processing_queue)
        self.assertIsNone(host.queue_panel.processing, "자동화가 건 '생성 중' 표시를 푼다")
        self.assertEqual(host.queue_panel.items, items, '항목은 소비하지 않는다')
        self.assertEqual(len(host.started), sent, '멈춘 채 새로 보내지 않는다')
        self.assertEqual(host.timers, [])
        status = host.status()
        self.assertTrue(status['running'])
        self.assertTrue(status['paused'])
        self.assertTrue(any('자동화 일시정지' in s and _BUSY in s for s in host.statuses))
        self.assertFalse(any(s.startswith('대기열 일시정지') for s in host.statuses))
        self.assertEqual(host.vue_bridge.generationError.calls[-1], (_BUSY,))   # Vue 스피너 리셋
        self.assertEqual(host.auto_gen_count, 1)
        self.assertEqual(getattr(host, '_auto_retry_count', 0), 0, '실패가 아니다 — 재시도로 세지 않는다')

    def test_resume_sends_the_same_frozen_item_and_the_deck_continues(self):
        host, items = self._host_at_frozen_item()
        host.defer()
        host._resume_automation()
        host.run_timers()
        self.assertEqual(host.started[-1], ('frozen <lora:x:0.7>', True))
        self.assertEqual(host.queue_panel.processing, (1, AUTOMATION_OWNER))
        self.assertEqual([o['payload_override']['subseed_strength'] for o in host.overrides], [0.05, 0.05])
        self.assertIs(host.overrides[-1]['backend_override'], self.backend)
        self.step(True)                            # 항목 1 완료 → 소비, 항목 2
        self.assertEqual([item['id'] for item in host.queue_panel.items], [2])
        self.step(True)                            # 항목 2 완료 → 덱 P2
        self.assertEqual(host.started[-1], ('P2', False))
        self.assertEqual(host.queue_panel.items, [])
        self.assertEqual(host.auto_gen_count, 1, '큐 항목은 세지 않는다')
        self.assertEqual(host.refused, [])

    def test_stop_after_the_deferral_leaves_the_item_queued(self):
        host, items = self._host_at_frozen_item()
        host.defer()
        host._stop_automation()
        self.assertFalse(host.is_automating)
        self.assertIsNone(host.queue_panel.processing)
        self.assertEqual(host.queue_panel.items, items)

    def test_deferral_is_not_claimed_by_automation_outside_a_queue_first_item(self):
        host = _DeferHost(['P1'])
        host.is_automating = True
        host._auto_processing_queue = False
        self.assertFalse(host._automation_queue_item_deferred(_BUSY))
        self.assertFalse(getattr(host, '_auto_paused', False))


class QueueManagerDeferralUnchangedTests(unittest.TestCase):
    """대기열 매니저가 보낸 항목의 '보내지 못함'은 여전히 매니저가 맡는다(자동화 상태는 그대로)."""

    def test_running_queue_manager_still_receives_the_deferral(self):
        manager = mock.Mock(is_running=True)
        manager.expects_result.return_value = True
        manager.awaited_item.return_value = {'prompt': 'q'}
        host = _DeferHost(['P1'])
        host.queue_manager = manager
        host.gen_worker = SimpleNamespace(_result_emitted=True, isRunning=lambda: False)
        host.on_generation_finished(_BUSY, {'_queue_deferred': True})
        manager.on_generation_deferred.assert_called_once_with()
        self.assertFalse(getattr(host, '_auto_paused', False))
        self.assertEqual(host.statuses[-1], f'대기열 일시정지: {_BUSY}')


if __name__ == '__main__':
    unittest.main()
