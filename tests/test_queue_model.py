"""대기열 상태 모델(core.queue_model) · 합쳐 부르기(core.coalesced_call) — 순수 로직.

회귀
- 실행 중 항목 보호 가드가 존재하지 않는 속성(_processing_item_id)을 읽어 무력했다 — 실행 중 항목을
  지우면 생성이 끝날 때 '다음' 항목이 생성 없이 지워졌다.
- 숨은 QWidget 이 변경마다 카드 N장을 다시 만들어 일괄 추가가 O(N²)였다(모델은 화면과 무관).
"""
import itertools
import unittest

from core.coalesced_call import CoalescedCall
from core.queue_model import (
    AUTOMATION_OWNER,
    PROMPT_TEXT_LIMIT,
    QUEUE_OWNER,
    QueueModel,
    auto_restore_enabled,
    periodic_unload_due,
    queue_item_added_payload,
    restorable_items,
    state_document,
    vue_queue_state,
)


def _model(n=0):
    counter = itertools.count(1)
    model = QueueModel(id_factory=lambda: f"id{next(counter)}")
    for i in range(n):
        model.add({'prompt': f'p{i}'})
    return model


class AddTests(unittest.TestCase):
    def test_add_assigns_unique_ids_and_single_item_group_defaults(self):
        model = _model()
        first = model.add({'prompt': 'a'})
        self.assertEqual(first['id'], 'id1')
        self.assertEqual(
            {k: first[k] for k in ('group_id', 'group_index', 'group_total', 'is_last_of_group')},
            {'group_id': '', 'group_index': 1, 'group_total': 1, 'is_last_of_group': True})
        self.assertIs(model.get('id1'), first)
        self.assertEqual(model.count(), 1)

    def test_payload_id_is_ignored_so_ids_never_collide(self):
        model = _model()
        a = model.add({'prompt': 'a'})
        b = model.add({'prompt': 'b', 'id': a['id']})      # 복제·재등록 payload
        self.assertNotEqual(a['id'], b['id'])
        self.assertEqual(model.count(), 2)

    def test_colliding_factory_values_are_retried(self):
        values = iter(['same', 'same', 'other'])
        model = QueueModel(id_factory=lambda: next(values))
        self.assertEqual(model.add({})['id'], 'same')
        self.assertEqual(model.add({})['id'], 'other')

    def test_payload_group_fields_are_kept_and_caller_dict_is_not_mutated(self):
        model = _model()
        payload = {'prompt': 'x', 'is_last_of_group': False, 'group_id': 'A'}
        item = model.add(payload)
        self.assertFalse(item['is_last_of_group'])
        self.assertEqual(item['group_id'], 'A')
        self.assertNotIn('id', payload)

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(TypeError):
            _model().add(['prompt'])

    def test_many_adds_are_linear(self):
        model = QueueModel()
        for i in range(10_000):
            model.add({'prompt': str(i)})
        self.assertEqual(model.count(), 10_000)
        self.assertEqual(len({item['id'] for item in model.items}), 10_000)


class RunningItemProtectionTests(unittest.TestCase):
    def test_remove_ids_never_removes_the_generating_item(self):
        model = _model(3)
        model.set_processing('id1')
        self.assertEqual(model.remove_ids(['id1', 'id2']), 1)
        self.assertEqual([item['id'] for item in model.items], ['id1', 'id3'])
        self.assertEqual(model.processing_id, 'id1')

    def test_remove_ids_ignores_unknown_and_unhashable_values(self):
        model = _model(2)
        self.assertEqual(model.remove_ids(['nope', None, ['id1'], {'x': 1}]), 0)
        self.assertEqual(model.count(), 2)

    def test_clear_keeps_only_the_generating_item(self):
        model = _model(4)
        model.set_processing('id1')
        self.assertEqual(model.clear(), 3)
        self.assertEqual([item['id'] for item in model.items], ['id1'])
        self.assertIsNotNone(model.get('id1'))
        self.assertIsNone(model.get('id2'))

    def test_clear_without_running_item_empties_the_queue(self):
        model = _model(2)
        self.assertEqual(model.clear(), 2)
        self.assertTrue(model.is_empty())

    def test_moves_never_swap_with_the_generating_item(self):
        model = _model(3)
        model.set_processing('id1')
        self.assertFalse(model.move('id2', 'up'))       # 0번(생성 중) 자리로 못 온다
        self.assertFalse(model.move('id1', 'down'))     # 생성 중인 항목은 못 내린다
        self.assertTrue(model.move('id3', 'up'))
        self.assertEqual([item['id'] for item in model.items], ['id1', 'id3', 'id2'])

    def test_moves_are_free_when_nothing_is_generating_and_respect_bounds(self):
        model = _model(2)
        self.assertTrue(model.move('id2', 'up'))
        self.assertEqual([item['id'] for item in model.items], ['id2', 'id1'])
        self.assertFalse(model.move('id2', 'up'))
        self.assertFalse(model.move('id1', 'down'))
        self.assertFalse(model.move('id1', 'sideways'))
        self.assertFalse(model.move('missing', 'up'))

    def test_consume_removes_only_that_item_and_releases_the_mark(self):
        model = _model(3)
        model.set_processing('id1')
        done = model.consume('id1')
        self.assertEqual(done['id'], 'id1')
        self.assertIsNone(model.processing_id)
        # 이미 없는 항목을 소비하면 아무것도 지우지 않는다 — 예전엔 '다음' 항목이 사라졌다
        self.assertIsNone(model.consume('id1'))
        self.assertEqual([item['id'] for item in model.items], ['id2', 'id3'])

    def test_pop_first_releases_the_running_mark_of_that_item(self):
        model = _model(2)
        model.set_processing('id1')
        self.assertEqual(model.pop_first()['id'], 'id1')
        self.assertFalse(model.is_processing)
        self.assertEqual(model.processing_index(), -1)
        self.assertIsNone(_model().pop_first())

    def test_processing_index_tracks_position(self):
        model = _model(3)
        self.assertEqual(model.processing_index(), -1)
        model.set_processing('id2')
        self.assertEqual(model.processing_index(), 1)
        self.assertFalse(model.set_processing('id2'))   # 같은 값은 변화 없음
        self.assertTrue(model.set_processing(None))


class ProcessingOwnerTests(unittest.TestCase):
    """'생성 중' 표시의 주인 — 대기열 매니저와 자동화 '큐 우선'이 서로의 표시를 빼앗거나 풀지 못한다.

    회귀: 주인 없는 한 칸이라, 자동화가 생성 중인 Q 를 대기열 '시작'이 다시 보냈다가 실패하며 표시를
    풀었고, Q 를 지운 뒤 자동화가 맨 앞(생성하지 않은 R)을 지웠다.
    """

    def test_claim_marks_the_item_for_its_owner(self):
        model = _model(2)
        self.assertTrue(model.claim('id1', AUTOMATION_OWNER))
        self.assertEqual((model.processing_id, model.processing_owner), ('id1', AUTOMATION_OWNER))
        self.assertEqual(model.remove_ids(['id1']), 0)                 # 보호된다

    def test_another_owner_can_neither_claim_nor_release(self):
        model = _model(2)
        model.claim('id1', AUTOMATION_OWNER)
        self.assertFalse(model.claim('id1', QUEUE_OWNER))
        self.assertFalse(model.claim('id2', QUEUE_OWNER))
        self.assertFalse(model.release('id1', QUEUE_OWNER))
        self.assertTrue(model.held_by_other(QUEUE_OWNER))
        self.assertFalse(model.held_by_other(AUTOMATION_OWNER))
        self.assertEqual(model.processing_id, 'id1')
        self.assertTrue(model.release('id1', AUTOMATION_OWNER))
        self.assertFalse(model.is_processing)
        self.assertIsNone(model.processing_owner)

    def test_same_owner_moves_on_and_missing_items_are_not_marked(self):
        model = _model(2)
        model.claim('id1', QUEUE_OWNER)
        self.assertTrue(model.claim('id2', QUEUE_OWNER))                # 다음 항목으로
        self.assertEqual(model.processing_id, 'id2')
        self.assertFalse(model.claim('missing', QUEUE_OWNER))
        self.assertFalse(model.claim('id1', ''))
        self.assertEqual(model.processing_id, 'id2')

    def test_legacy_unowned_mark_is_taken_over_and_released_by_anyone(self):
        model = _model(2)
        model.set_processing('id1')
        self.assertFalse(model.held_by_other(QUEUE_OWNER))
        self.assertTrue(model.claim('id1', QUEUE_OWNER))
        model.set_processing('id2')
        self.assertTrue(model.release('id2', AUTOMATION_OWNER))
        self.assertFalse(model.release('id2'))                          # 이미 풀렸다

    def test_release_ignores_other_items(self):
        model = _model(2)
        model.claim('id1', QUEUE_OWNER)
        self.assertFalse(model.release('id2', QUEUE_OWNER))
        self.assertFalse(model.release(None))
        self.assertEqual(model.processing_id, 'id1')

    def test_consuming_or_replacing_drops_the_owner_with_the_mark(self):
        model = _model(2)
        model.claim('id1', AUTOMATION_OWNER)
        model.consume('id1')
        self.assertIsNone(model.processing_owner)
        self.assertFalse(model.held_by_other(QUEUE_OWNER))
        model.claim('id2', AUTOMATION_OWNER)
        model.replace([{'id': 'x'}])
        self.assertIsNone(model.processing_owner)


class UpdateAndRestoreTests(unittest.TestCase):
    def test_update_fields_keeps_the_id(self):
        model = _model(1)
        self.assertTrue(model.update_fields('id1', {'prompt': 'new', 'id': 'hijack'}))
        self.assertEqual(model.get('id1')['prompt'], 'new')
        self.assertFalse(model.update_fields('missing', {'prompt': 'x'}))

    def test_replace_drops_non_dicts_and_reissues_missing_or_duplicate_ids(self):
        model = _model()
        model.set_processing('old')
        count = model.replace([{'id': 'a', 'prompt': 1}, 'junk', {'id': 'a'}, {'prompt': 'no id'}, {'id': 7}])
        self.assertEqual(count, 4)
        ids = [item['id'] for item in model.items]
        self.assertEqual(ids[0], 'a')
        self.assertEqual(len(set(ids)), 4)
        self.assertTrue(all(isinstance(i, str) for i in ids))
        self.assertIsNone(model.processing_id)

    def test_restorable_items_and_state_document(self):
        self.assertEqual(restorable_items({'items': [{'id': 'a'}, 3, None]}), [{'id': 'a'}])
        self.assertEqual(restorable_items({'items': 'bad'}), [])
        self.assertEqual(restorable_items(['not a dict']), [])
        self.assertEqual(state_document([{'id': 'a'}]), {'items': [{'id': 'a'}]})

    def test_auto_restore_defaults_on(self):
        self.assertTrue(auto_restore_enabled(None))
        self.assertTrue(auto_restore_enabled({}))
        self.assertFalse(auto_restore_enabled({'queue_auto_restore': False}))


class VuePayloadTests(unittest.TestCase):
    def test_state_keeps_the_legacy_shape_and_adds_processing_index(self):
        items = [{'id': 'a', 'prompt': 'x' * (PROMPT_TEXT_LIMIT + 10), 'other': 'y' * 900,
                  'seed': 3, 'hires': True, 'nested': {'k': 1}}]
        state = vue_queue_state(items, running=True, paused=False, completed=2, processing_index=0)
        self.assertEqual(set(state), {'items', 'running', 'paused', 'current_index', 'completed',
                                      'processing_index', 'automation'})
        item = state['items'][0]
        self.assertEqual(len(item['prompt']), PROMPT_TEXT_LIMIT)
        self.assertEqual(len(item['other']), 500)
        self.assertNotIn('nested', item)
        self.assertEqual((item['seed'], item['hires']), (3, True))
        self.assertEqual((state['current_index'], state['processing_index'], state['completed']), (0, 0, 2))
        self.assertIs(state['automation'], False)

    def test_state_reports_whether_automation_owns_the_queue(self):
        # 자동화가 돌면 대기열 '시작'은 거절된다 — Vue 는 이 값으로 버튼을 끈다
        state = vue_queue_state([{'id': 'a'}], running=False, paused=False, completed=0, automating=1)
        self.assertIs(state['automation'], True)

    def test_current_index_follows_the_queue_manager_only(self):
        items = [{'id': 'a'}]
        idle = vue_queue_state(items, running=False, paused=False, completed=0, processing_index=0)
        self.assertEqual(idle['current_index'], -1)     # 자동화 '큐 우선' 처리 — 대기열 매니저는 멈춤
        self.assertEqual(idle['processing_index'], 0)
        self.assertEqual(vue_queue_state([], running=True, paused=False, completed=0)['current_index'], -1)
        self.assertEqual(vue_queue_state(items, running=True, paused=False, completed=0,
                                         processing_index=5)['processing_index'], -1)

    def test_item_added_payload_is_short_and_primitive(self):
        payload = queue_item_added_payload({'id': 'a', 'prompt': 'p' * 500, 'n': 2, 'obj': []})
        self.assertEqual(payload, {'id': 'a', 'prompt': 'p' * 200, 'n': '2'})


class PeriodicUnloadTests(unittest.TestCase):
    def test_every_n_with_items_left(self):
        self.assertTrue(periodic_unload_due(2, 2, remaining=3))
        self.assertTrue(periodic_unload_due(1, 1, remaining=1))
        self.assertFalse(periodic_unload_due(3, 2, remaining=3))

    def test_disabled_or_last_item(self):
        self.assertFalse(periodic_unload_due(2, 0, remaining=3))
        self.assertFalse(periodic_unload_due(4, 2, remaining=0))   # 마지막 장 뒤엔 내리지 않는다
        self.assertFalse(periodic_unload_due(0, 2, remaining=3))
        self.assertFalse(periodic_unload_due('x', 2, remaining=3))


class CoalescedCallTests(unittest.TestCase):
    def setUp(self):
        self.scheduled = []
        self.calls = 0

    def _fn(self):
        self.calls += 1

    def _schedule(self, ms, fn):
        self.scheduled.append((ms, fn))

    def test_many_requests_run_once(self):
        call = CoalescedCall(self._fn, schedule=self._schedule, delay_ms=250)
        self.assertTrue(call.request())
        for _ in range(255):
            self.assertFalse(call.request())
        self.assertEqual(len(self.scheduled), 1)
        self.assertEqual(self.scheduled[0][0], 250)
        self.scheduled[0][1]()
        self.assertEqual(self.calls, 1)
        self.assertFalse(call.pending)
        call.request()                         # 실행 뒤에는 다시 예약된다
        self.assertEqual(len(self.scheduled), 2)

    def test_flush_runs_now_and_the_stale_timer_does_nothing(self):
        call = CoalescedCall(self._fn, schedule=self._schedule)
        call.request()
        self.assertTrue(call.flush())
        self.assertEqual(self.calls, 1)
        self.scheduled[0][1]()                 # 늦게 온 예약 — 두 번 실행하지 않는다
        self.assertEqual(self.calls, 1)
        self.assertFalse(call.flush())

    def test_cancel_and_exception_do_not_wedge(self):
        call = CoalescedCall(self._fn, schedule=self._schedule)
        call.request()
        call.cancel()
        self.scheduled[0][1]()
        self.assertEqual(self.calls, 0)

        def boom():
            raise RuntimeError('x')
        failing = CoalescedCall(boom, schedule=self._schedule)
        failing.request()
        with self.assertRaises(RuntimeError):
            self.scheduled[-1][1]()
        self.assertTrue(failing.request())     # 대기 표시가 남지 않았다


if __name__ == '__main__':
    unittest.main()
