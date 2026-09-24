"""대기열 상태 저장소(widgets.queue_panel) · 대기열 매니저 · Vue 동기화.

회귀
- 숨은 QWidget QueuePanel 이 변경마다 카드 N장을 다시 만들고, 전체 JSON 을 쓰고, 전체 목록을 Vue 로
  보냈다 — XYZ 256칸이면 약 20초 정지·RSS 수 GB. 지금은 화면 없는 QObject + 모아 쓰기 + 턴당 한 번 전송.
- 실행 중 항목 보호 가드가 무력해, 실행 중 항목을 지우면 완료 때 '다음' 항목이 생성 없이 사라졌다.
- 대기열 정기 정리가 VRAM 을 회수하지 않는 cleanup_models(LoRA 재스캔 + options POST)를 불렀다.
"""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QObject

import widgets.queue_manager as queue_manager_module
from core.queue_model import AUTOMATION_OWNER, QUEUE_OWNER
from ui.queue_vue_sync import QueueVueSync
from widgets.queue_manager import QueueManager
from widgets.queue_panel import PERSIST_DELAY_MS, QueuePanel


class _Timers:
    """QTimer.singleShot 대역 — 예약만 모으고 fire_all() 로 한 번에 돌린다."""

    def __init__(self):
        self.pending = []

    def __call__(self, ms, fn):
        self.pending.append((ms, fn))

    def fire_all(self):
        while self.pending:
            _ms, fn = self.pending.pop(0)
            fn()


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


class _PanelCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / 'queue_state.json'
        self.prefs = Path(self._tmp.name) / 'ui_prefs.json'
        self.timers = _Timers()

    def panel(self, *, restore=True):
        return QueuePanel(state_path=self.state, prefs_path=self.prefs, schedule=self.timers,
                          restore=restore)


class QueuePanelTests(_PanelCase):
    def test_is_a_screenless_qobject(self):
        panel = self.panel()
        self.assertIsInstance(panel, QObject)
        self.assertFalse(hasattr(panel, 'card_widgets'))
        self.assertFalse(hasattr(panel, 'show'))

    def test_bulk_add_writes_once_and_stays_linear(self):
        panel = self.panel()
        changes = []
        panel.queue_changed.connect(changes.append)
        writes = []
        with mock.patch('widgets.queue_panel.atomic_write_json',
                        side_effect=lambda path, doc, indent=None: writes.append(len(doc['items']))):
            started = time.perf_counter()
            for i in range(2000):
                panel.add_single_item({'prompt': f'p{i}'})
            elapsed = time.perf_counter() - started
            self.assertEqual(writes, [])                         # 아직 모으는 중
            self.assertEqual([ms for ms, _ in self.timers.pending], [PERSIST_DELAY_MS])
            self.timers.fire_all()
        self.assertEqual(writes, [2000])                         # 2000건 → 저장 1번
        self.assertEqual(len(changes), 2000)                     # 알림은 싸다(목록 없음)
        self.assertEqual(changes[-1], 2000)
        # 예전 구현은 256건에 약 20초였다(카드 재생성 O(N²)). 넉넉한 상한으로 선형성만 확인한다.
        self.assertLess(elapsed, 2.0)

    def test_flush_to_disk_writes_pending_state_and_restore_reads_it_back(self):
        panel = self.panel()
        first_id = panel.add_single_item({'prompt': 'a', 'seed': 1})
        panel.add_single_item({'prompt': 'b'})
        self.assertFalse(self.state.exists())
        self.assertTrue(panel.flush_to_disk())
        self.assertFalse(panel.flush_to_disk())                  # 쓸 것이 없으면 다시 안 쓴다
        stored = json.loads(self.state.read_text(encoding='utf-8'))
        self.assertEqual([item['prompt'] for item in stored['items']], ['a', 'b'])
        self.timers.fire_all()                                   # 늦은 예약은 두 번 쓰지 않는다

        restored = self.panel()
        self.assertEqual(restored.count(), 2)
        self.assertEqual(restored.get_first_item()['id'], first_id)
        self.assertFalse(restored.is_processing)

    def test_restore_respects_the_manual_opt_out(self):
        self.state.write_text(json.dumps({'items': [{'id': 'x', 'prompt': 'p'}]}), encoding='utf-8')
        self.prefs.write_text(json.dumps({'queue_auto_restore': False}), encoding='utf-8')
        from PyQt6.QtWidgets import QMessageBox
        with mock.patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            panel = self.panel()
        self.assertTrue(panel.is_empty())
        self.assertFalse(self.state.exists())

    def test_running_item_cannot_be_removed_moved_or_cleared(self):
        panel = self.panel()
        ids = [panel.add_single_item({'prompt': str(i)}) for i in range(3)]
        panel.set_processing(True, ids[0])
        self.assertEqual(panel.remove_items_by_ids([ids[0]]), 0)
        self.assertFalse(panel.move_item_up(ids[1]))
        self.assertFalse(panel.move_item_down(ids[0]))
        self.assertEqual(panel.clear_items(), 2)
        self.assertEqual([item['id'] for item in panel.queue_items], [ids[0]])
        self.assertEqual(panel.processing_index(), 0)

    def test_set_processing_without_id_marks_the_first_item(self):
        panel = self.panel()
        first = panel.add_single_item({'prompt': 'a'})
        panel.set_processing(True)
        self.assertEqual(panel.current_processing_id, first)
        panel.set_processing(False)
        self.assertFalse(panel.is_processing)

    def test_release_processing_only_clears_the_named_item(self):
        panel = self.panel()
        a = panel.add_single_item({'prompt': 'a'})
        b = panel.add_single_item({'prompt': 'b'})
        panel.set_processing(True, a)
        self.assertFalse(panel.release_processing(b))
        self.assertEqual(panel.current_processing_id, a)
        self.assertTrue(panel.release_processing(a))
        self.assertEqual(panel.remove_items_by_ids([a]), 1)

    def test_owned_marks_notify_vue_and_resist_the_other_consumer(self):
        panel = self.panel(restore=False)
        a = panel.add_single_item({'prompt': 'a'})
        b = panel.add_single_item({'prompt': 'b'})
        self.timers.fire_all()                                   # 추가분 저장은 끝냈다
        changes = []
        panel.queue_changed.connect(changes.append)
        self.assertTrue(panel.claim_processing(a, AUTOMATION_OWNER))
        self.assertEqual(len(changes), 1)                        # Vue 가 그 행을 잠근다
        self.assertTrue(panel.claim_processing(a, AUTOMATION_OWNER))
        self.assertEqual(len(changes), 1)                        # 바뀐 게 없으면 알리지 않는다
        self.assertFalse(panel.claim_processing(b, QUEUE_OWNER))
        self.assertTrue(panel.processing_held_by_other(QUEUE_OWNER))
        self.assertFalse(panel.release_processing(a, QUEUE_OWNER))
        self.assertEqual((panel.current_processing_id, panel.processing_owner), (a, AUTOMATION_OWNER))
        self.assertTrue(panel.release_processing(a, AUTOMATION_OWNER))
        self.assertEqual(len(changes), 2)
        self.assertIsNone(panel.processing_owner)
        self.assertEqual(self.timers.pending, [])                # 표시만 바뀌면 디스크에 쓰지 않는다

    def test_update_item_and_consume_item(self):
        panel = self.panel()
        a = panel.add_single_item({'prompt': 'a'})
        b = panel.add_single_item({'prompt': 'b'})
        self.assertTrue(panel.update_item(b, {'prompt': 'edited'}))
        self.assertEqual(panel.get_item_by_id(b)['prompt'], 'edited')
        self.assertEqual(panel.consume_item(a)['id'], a)
        self.assertIsNone(panel.consume_item(a))                 # 다음 항목을 대신 지우지 않는다
        self.assertEqual([item['id'] for item in panel.queue_items], [b])


class QueueManagerTests(_PanelCase):
    def setUp(self):
        super().setUp()
        self.panel_obj = self.panel(restore=False)
        self.manager = QueueManager(self.panel_obj)
        self.manager.delay_seconds = 0
        self.requested = []
        self.dispatch_ok = True
        self.manager.generation_requested.connect(self._on_requested)
        self.completed = []
        self.manager.queue_completed.connect(self.completed.append)

    def _on_requested(self, item):
        self.requested.append(item['id'])
        if not self.dispatch_ok:
            self.manager.pause()          # _on_generation_requested 가 시작하지 못하면 이렇게 한다

    def _fill(self, n):
        return [self.panel_obj.add_single_item({'prompt': str(i)}) for i in range(n)]

    def test_empty_queue_does_not_start(self):
        self.assertFalse(self.manager.start())
        self.assertFalse(self.manager.is_running)

    def test_completion_consumes_the_dispatched_item_and_advances(self):
        ids = self._fill(2)
        self.assertTrue(self.manager.start())
        self.assertEqual(self.requested, [ids[0]])
        self.assertEqual(self.panel_obj.current_processing_id, ids[0])
        self.manager.on_generation_completed(True)
        self.assertEqual(self.requested, [ids[0], ids[1]])
        self.manager.on_generation_completed(True)
        self.assertFalse(self.manager.is_running)
        self.assertTrue(self.manager.last_stop_natural)
        self.assertEqual(self.completed, [2])
        self.assertTrue(self.panel_obj.is_empty())

    def test_deleting_the_running_item_is_refused_so_the_next_item_survives(self):
        ids = self._fill(3)
        self.manager.start()
        self.assertEqual(self.panel_obj.remove_items_by_ids([ids[0]]), 0)
        self.manager.on_generation_completed(True)
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], ids[1:])
        self.assertEqual(self.requested[-1], ids[1])

    def test_completion_without_a_dispatch_consumes_nothing(self):
        ids = self._fill(2)
        self.manager.start()
        self.manager.pause()
        self.manager.on_generation_completed(True)               # 보낸 항목의 결과
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [ids[1]])
        self.manager.on_generation_completed(True)               # 일시정지 중 수동 생성의 결과
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [ids[1]])
        self.assertEqual(self.manager.generated_count, 1)

    def test_failed_dispatch_pauses_and_keeps_the_item_resumable(self):
        ids = self._fill(2)
        self.dispatch_ok = False
        self.manager.start()
        self.assertTrue(self.manager.is_paused)
        self.assertFalse(self.panel_obj.is_processing)           # 보낸 표시가 풀렸다 → 지울 수도 있다
        self.manager.on_generation_completed(True)               # 다른 생성이 끝나도 소비하지 않는다
        self.assertEqual(self.panel_obj.count(), 2)
        self.dispatch_ok = True
        self.manager.resume()
        self.assertEqual(self.requested, [ids[0], ids[0]])
        self.assertEqual(self.panel_obj.current_processing_id, ids[0])

    def test_resume_while_the_item_is_still_generating_does_not_resend_it(self):
        ids = self._fill(2)
        worker = self._worker()
        self.manager.start()
        self.assertTrue(worker['active'])                        # 보낸 항목이 GPU 에 있다
        self.manager.pause()
        self.manager.resume()
        self.assertEqual(self.requested, [ids[0]])               # 두 번 보내지 않는다
        self.assertFalse(self.manager.is_paused)
        self._finish(worker)
        self.assertEqual(self.requested, [ids[0], ids[1]])

    # ── 워커 흉내: 보낸 항목을 '생성 중'으로 두고, 끝내면 결과를 낸다 ──

    def _worker(self):
        """generation_active 를 연결하고, 보낸 항목이 워커에 올라가게 한다(바쁘면 거절 → pause)."""
        worker = {'active': False, 'item': None}
        self.manager.generation_active = lambda: worker['active']
        self.manager.generation_requested.disconnect(self._on_requested)

        def on_requested(item):
            self.requested.append(item['id'])
            if worker['active']:
                self.manager.pause()      # start_generation() 이 '생성 중' 으로 거절
                return
            worker['active'], worker['item'] = True, item['id']

        self.manager.generation_requested.connect(on_requested)
        return worker

    def _finish(self, worker, success=True):
        """on_generation_finished 흉내 — 결과를 내고 대기열 매니저에 알린다(_generation_matches_queue)."""
        worker['active'] = False
        if self.manager.is_running or self.manager.expects_result():
            self.manager.on_generation_completed(success)

    def test_stop_then_start_while_generating_adopts_the_item_instead_of_resending(self):
        # 재현(p2q/redispatch.py): 예전엔 generated=['A','A'] — 다시 보냈다가 거절돼 일시정지하고, 원래
        # 생성이 끝나도 추적을 잃어 재개 때 A 를 한 번 더 만들었다.
        a, b = self._fill(2)
        worker = self._worker()
        self.manager.start()
        self.manager.stop()
        self.assertEqual(self.panel_obj.current_processing_id, a)  # 아직 생성 중 — 지우지 못한다
        self.assertEqual(self.panel_obj.remove_items_by_ids([a]), 0)
        self.assertTrue(self.manager.start())
        self.assertFalse(self.manager.is_paused)
        self.assertEqual(self.requested, [a])                      # 다시 보내지 않는다
        self._finish(worker)                                       # 원래 A 완료
        self.assertEqual(self.requested, [a, b])
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [b])
        self._finish(worker)
        self.assertFalse(self.manager.is_running)
        self.assertTrue(self.manager.last_stop_natural)
        self.assertEqual(self.manager.generated_count, 2)

    def test_item_finished_after_stop_is_consumed_and_a_failed_one_is_kept(self):
        a, b = self._fill(2)
        worker = self._worker()
        self.manager.start()
        self.manager.stop()
        self.assertTrue(self.manager.expects_result())
        self.assertEqual(self.manager.awaited_item()['id'], a)
        self._finish(worker, success=True)                          # 멈춘 채 끝났다 — 만들어졌다
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [b])
        self.assertFalse(self.panel_obj.is_processing)
        self.assertFalse(self.manager.expects_result())

        self.manager.start()                                        # B
        self.manager.stop()
        self._finish(worker, success=False)                         # 실패 · 취소 — 항목은 남는다
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [b])
        self.assertFalse(self.panel_obj.is_processing)
        self.assertEqual(self.panel_obj.remove_items_by_ids([b]), 1)

    def test_stop_without_a_running_generation_releases_only_its_own_mark(self):
        a, _b = self._fill(2)
        self.manager.start()                                         # 대역 워커 없음 — 곧바로 끝난 셈
        self.manager.stop()
        self.assertFalse(self.panel_obj.is_processing)
        self.assertFalse(self.manager.expects_result())
        self.assertEqual(self.panel_obj.remove_items_by_ids([a]), 1)

    def test_stale_delay_timer_does_not_resend_into_a_new_run(self):
        # 재현(p2q/stale_timer.py): 다음 장까지 기다리는 사이 중지→시작하면 옛 타이머가 이미 보낸
        # B 를 다시 보내 일시정지·추적 해제 → generated=['A','B','B'].
        a, b, c = self._fill(3)
        worker = self._worker()
        self.manager.delay_seconds = 0.05
        timers = []
        with mock.patch('widgets.queue_manager.QTimer.singleShot',
                        side_effect=lambda ms, fn: timers.append(fn)):
            self.manager.start()                                     # A
            self._finish(worker)                                     # A 완료 → 지연 타이머
            self.assertEqual(len(timers), 1)
            self.manager.stop()
            self.manager.start()                                     # B 즉시
            self.assertEqual(self.requested, [a, b])
            timers.pop(0)()                                          # 옛 타이머
            self.assertEqual(self.requested, [a, b])
            self.assertFalse(self.manager.is_paused)
            self._finish(worker)                                     # B 완료
            timers.pop(0)()                                          # 이번 회차 타이머 → C
        self.assertEqual(self.requested, [a, b, c])
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [c])

    def test_a_busy_worker_holds_the_queue_without_dispatching(self):
        # 수동 생성이 돌고 있으면 보내지 않는다 — 예전엔 보냈다가 거절되며 항목 값을 UI 에 덮어썼다
        ids = self._fill(1)
        worker = self._worker()
        worker['active'] = True
        notices = []
        self.manager.notice.connect(notices.append)
        self.assertTrue(self.manager.start())
        self.assertTrue(self.manager.is_paused)
        self.assertEqual(self.requested, [])
        self.assertFalse(self.panel_obj.is_processing)
        self.assertEqual(notices, [queue_manager_module.BUSY_NOTICE])
        self._finish(worker)                                         # 수동 생성 결과 — 소비하지 않는다
        self.assertEqual(self.panel_obj.count(), 1)
        self.manager.resume()
        self.assertEqual(self.requested, ids)

    def test_a_lost_result_is_resent_on_resume(self):
        # 보낸 항목의 결과가 오지 않은 채 워커가 끝났다(짝이 맞지 않는 결과) — 재개 때 같은 항목을 다시 보낸다
        ids = self._fill(2)
        worker = self._worker()
        self.manager.start()
        self.manager.pause()
        worker['active'] = False                                     # 결과는 이 매니저에 오지 않았다
        self.manager.resume()
        self.assertEqual(self.requested, [ids[0], ids[0]])
        self.assertEqual(self.panel_obj.current_processing_id, ids[0])

    def test_start_is_refused_while_automation_owns_the_queue(self):
        self._fill(1)
        self.manager.start_blocker = lambda: '자동화 중'
        self.assertFalse(self.manager.start())
        self.assertEqual(self.manager.last_start_refusal, '자동화 중')
        self.assertFalse(self.manager.is_running)
        self.assertEqual(self.requested, [])
        self.manager.start_blocker = lambda: ''
        self.assertTrue(self.manager.start())
        self.assertEqual(self.manager.last_start_refusal, '')

    def test_an_item_generated_by_automation_is_neither_resent_nor_released(self):
        # 재현(p2_mark_clobber.py): 자동화가 Q 를 생성 중일 때 대기열 '시작' → 예전엔 Q 를 다시 보냈다가
        # 거절되며 자동화의 표시를 풀었고, Q 를 지운 뒤 자동화의 맨 앞 정리가 R 을 지웠다.
        q, r = self._fill(2)
        self.assertTrue(self.panel_obj.claim_processing(q, AUTOMATION_OWNER))
        notices = []
        self.manager.notice.connect(notices.append)
        self.assertTrue(self.manager.start())                        # 시작 가드를 지나온 경우라도
        self.assertTrue(self.manager.is_paused)
        self.assertEqual(self.requested, [])
        self.assertEqual(notices, [queue_manager_module.AUTOMATION_NOTICE])
        self.assertEqual((self.panel_obj.current_processing_id, self.panel_obj.processing_owner),
                         (q, AUTOMATION_OWNER))
        self.manager.stop()                                          # 중지도 남의 표시는 풀지 않는다
        self.assertEqual(self.panel_obj.current_processing_id, q)
        self.assertEqual(self.panel_obj.remove_items_by_ids([q]), 0)
        self.assertEqual(self.panel_obj.consume_item(q)['id'], q)    # 자동화가 끝나면 그 항목만
        self.assertEqual([item['id'] for item in self.panel_obj.queue_items], [r])

    def test_deferred_request_after_stop_only_releases_the_mark(self):
        a, _b = self._fill(2)
        worker = self._worker()
        self.manager.start()
        self.manager.stop()
        worker['active'] = False
        self.manager.on_generation_deferred()
        self.assertFalse(self.manager.expects_result())
        self.assertFalse(self.panel_obj.is_processing)
        self.assertEqual(self.panel_obj.count(), 2)
        self.assertEqual(self.panel_obj.get_first_item()['id'], a)

    def test_deferred_generation_pauses_without_consuming(self):
        ids = self._fill(1)
        self.manager.start()
        self.manager.on_generation_deferred()
        self.assertTrue(self.manager.is_paused)
        self.assertFalse(self.panel_obj.is_processing)
        self.assertEqual(self.panel_obj.count(), 1)
        self.manager.resume()
        self.assertEqual(self.requested, [ids[0], ids[0]])

    def test_start_while_running_joins_instead_of_resending(self):
        self._fill(1)
        self.manager.start()
        self._fill(3)
        self.assertTrue(self.manager.start())
        self.assertEqual(len(self.requested), 1)
        self.assertEqual(self.manager.total_count, 4)

    def test_resume_on_an_emptied_queue_stops_instead_of_hanging(self):
        ids = self._fill(1)
        self.dispatch_ok = False
        self.manager.start()
        self.panel_obj.remove_items_by_ids(ids)
        self.manager.resume()
        self.assertFalse(self.manager.is_running)
        self.assertFalse(self.manager.last_stop_natural)

    def test_periodic_cleanup_unloads_the_checkpoint_between_items_only(self):
        self._fill(3)
        self.manager.cleanup_every_n = 1
        backend = object()
        with mock.patch('backends.get_backend', return_value=backend), \
                mock.patch('core.post_generation.start_post_generation_unload') as unload:
            self.manager.start()
            self.manager.on_generation_completed(True)
            self.manager.on_generation_completed(True)
            self.manager.on_generation_completed(True)           # 마지막 장 — 정리하지 않는다
        self.assertEqual(unload.call_count, 2)
        self.assertIs(unload.call_args.args[0], backend)

    def test_periodic_cleanup_respects_the_interval_and_no_longer_uses_cleanup_models(self):
        self._fill(5)
        self.manager.cleanup_every_n = 2
        backend = SimpleNamespace(cleanup_models=mock.Mock())
        with mock.patch('backends.get_backend', return_value=backend), \
                mock.patch('core.post_generation.start_post_generation_unload') as unload:
            self.manager.start()
            for _ in range(5):
                self.manager.on_generation_completed(True)
        self.assertEqual(unload.call_count, 2)                   # 2장째 · 4장째
        backend.cleanup_models.assert_not_called()


class QueueVueSyncTests(_PanelCase):
    def _host(self):
        panel = self.panel(restore=False)
        manager = QueueManager(panel)
        bridge = SimpleNamespace(queueUpdated=_Signal(), queueItemAdded=_Signal())
        host = SimpleNamespace(queue_panel=panel, queue_manager=manager, vue_bridge=bridge,
                               show_status=lambda *_: None)
        return host, QueueVueSync(host, schedule=self.timers)

    def test_many_requests_in_one_turn_emit_one_state(self):
        host, sync = self._host()
        for i in range(256):
            host.queue_panel.add_single_item({'prompt': str(i)})
            sync.request_state()
            sync.item_added(host.queue_panel.queue_items[-1])
        self.assertEqual(host.vue_bridge.queueUpdated.values, [])
        self.timers.fire_all()
        self.assertEqual(len(host.vue_bridge.queueUpdated.values), 1)
        state = json.loads(host.vue_bridge.queueUpdated.values[0][0])
        self.assertEqual(len(state['items']), 256)
        self.assertEqual(state['processing_index'], -1)
        added = host.vue_bridge.queueItemAdded.values
        self.assertEqual(len(added), 1)
        self.assertEqual(json.loads(added[0][0])['prompt'], '255')

    def test_state_reports_the_generating_row(self):
        host, sync = self._host()
        a = host.queue_panel.add_single_item({'prompt': 'a'})
        host.queue_panel.set_processing(True, a)
        sync.request_state()
        sync.flush()
        state = json.loads(host.vue_bridge.queueUpdated.values[-1][0])
        self.assertEqual((state['processing_index'], state['running'], state['current_index']), (0, False, -1))

    def test_no_bridge_is_a_quiet_no_op(self):
        host, sync = self._host()
        del host.vue_bridge
        sync.request_state()
        sync.item_added({'id': 'x'})
        self.timers.fire_all()


if __name__ == '__main__':
    unittest.main()
