"""자동화 실패 재시도 · 장 수 세기 · 일회성 덮어쓰기 수명 회귀 (ActionsMixin 흐름)."""
import unittest
from unittest import mock

from PyQt6.QtCore import QTimer

from core.automation_retry import plan_retry, retry_backoff_seconds
from ui.generator_actions import ActionsMixin


class PlanRetryTests(unittest.TestCase):
    def test_backoff_and_exhaustion(self):
        self.assertEqual(plan_retry(0, 2), (1, 2.0))
        self.assertEqual(plan_retry(1, 2), (2, 4.0))
        self.assertIsNone(plan_retry(2, 2))
        self.assertIsNone(plan_retry(0, 0))
        self.assertIsNone(plan_retry(0, "bad"))
        self.assertEqual(retry_backoff_seconds(10), 30.0)


class _Text:
    def __init__(self, value=""):
        self.value = value

    def toPlainText(self):
        return self.value

    def setPlainText(self, value):
        self.value = value


class _Button:
    def __init__(self, checked=True):
        self.checked = checked

    def setText(self, _t):
        pass

    def setStyleSheet(self, _s):
        pass

    def setEnabled(self, _e):
        pass

    def isChecked(self):
        return self.checked


class _Queue:
    """QueuePanel 의 자동화 '큐 우선'용 표면 — 항목 목록 + id 로 찾기·소비 + 주인 있는 '생성 중' 표시."""

    def __init__(self, items=()):
        self.items = list(items)
        self.processing = None      # (item_id, owner)

    def get_first_item(self):
        return self.items[0] if self.items else None

    def get_item_by_id(self, item_id):
        return next((item for item in self.items if item.get('id') == item_id), None)

    def consume_item(self, item_id):
        item = self.get_item_by_id(item_id)
        if item is not None:
            self.items.remove(item)
            if self.processing and self.processing[0] == item_id:
                self.processing = None
        return item

    def claim_processing(self, item_id, owner):
        if self.get_item_by_id(item_id) is None:
            return False
        if self.processing and self.processing[1] != owner:
            return False
        self.processing = (item_id, owner)
        return True

    def release_processing(self, item_id, owner=None):
        if not self.processing or self.processing[0] != item_id:
            return False
        if owner is not None and self.processing[1] != owner:
            return False
        self.processing = None
        return True


class _Host(ActionsMixin):
    def __init__(self, prompts, **settings):
        self.settings = {
            'termination_mode': 'count', 'termination_limit': 99, 'delay': 0,
            'repeat_per_prompt': 1, 'max_retries': 0, 'allow_duplicates': False,
            'auto_reset_deck': False, **settings,
        }
        self.automation_widget = mock.Mock(get_settings=lambda: dict(self.settings))
        self.filtered_results = [{'general': p} for p in prompts]
        self.shuffled_prompt_deck = list(reversed(self.filtered_results))   # pop → 첫 프롬프트부터
        self.is_automating = False
        self.is_programmatic_change = False
        self.total_prompt_display = _Text()
        self.main_prompt_text = _Text()
        self.btn_random_prompt = _Button()
        self.btn_generate = _Button()
        self.btn_auto_toggle = _Button()
        self.queue_panel = _Queue()
        self.started = []          # (프롬프트, 큐 항목 여부)
        self.statuses = []
        self.timers = []

    # ── 대역 ──
    def show_status(self, message, *_a):
        self.statuses.append(message)

    def apply_random_prompt(self):
        if not self.shuffled_prompt_deck:
            return False
        bundle = self.shuffled_prompt_deck.pop()
        self._current_auto_bundle = bundle
        self.total_prompt_display.setPlainText(bundle['general'])
        return True

    def apply_prompt_from_data(self, bundle):
        self.total_prompt_display.setPlainText(bundle['general'])

    def update_total_prompt_display(self):
        bundle = getattr(self, '_current_auto_bundle', None)
        self.total_prompt_display.setPlainText(bundle['general'] if bundle else '')

    def _apply_payload_to_ui(self, item):
        self.total_prompt_display.setPlainText(item['prompt'])

    def _ensure_wait_timer(self):
        if getattr(self, '_wait_timer', None) is None:
            self._wait_timer = mock.Mock(isActive=lambda: False)

    def start_generation(self):
        self.started.append((self.total_prompt_display.toPlainText(),
                             bool(getattr(self, '_auto_processing_queue', False))))
        return True

    # ── 구동 ──
    def run_timers(self):
        while self.timers:
            _ms, fn = self.timers.pop(0)
            fn()

    def finish(self, success):
        """on_generation_finished 의 자동화 부분과 같은 순서."""
        if self._automation_after_generation(success):
            return
        if self.is_automating:
            self._continue_automation()


class AutomationFlowTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(QTimer, 'singleShot',
                                    side_effect=lambda ms, fn: self.host.timers.append((ms, fn)))
        patcher.start()
        self.addCleanup(patcher.stop)
        box = mock.patch('ui.generator_actions.QMessageBox')
        box.start()
        self.addCleanup(box.stop)

    def start(self, host):
        self.host = host
        host._start_automation()
        host.run_timers()

    def step(self, success):
        self.host.finish(success)
        self.host.run_timers()

    def test_failed_image_is_retried_without_losing_a_repeat(self):
        # repeat=3, 실패 1회 → 같은 프롬프트로 성공 3장 (예전엔 2장만 나왔다)
        host = _Host(['P1', 'P2'], repeat_per_prompt=3, termination_limit=3, max_retries=2)
        self.start(host)
        self.step(True)
        self.step(False)      # 재시도 예약
        self.step(True)
        self.step(True)
        self.assertEqual([p for p, _q in host.started], ['P1', 'P1', 'P1', 'P1'])
        self.assertEqual(host.auto_gen_count, 3)
        self.assertFalse(host.is_automating)

    def test_stale_retry_does_not_fire_into_a_new_run(self):
        host = _Host(['P1', 'P2', 'P3'], max_retries=2)
        self.start(host)
        host.finish(False)                    # 재시도 타이머만 걸린 상태
        stale = list(host.timers)
        host.timers.clear()
        host._stop_automation()
        host._start_automation()
        host.run_timers()
        before = len(host.started)
        for _ms, fn in stale:
            fn()
        self.assertEqual(len(host.started), before)

    def test_queue_items_are_never_counted_even_when_they_fail(self):
        host = _Host(['P1', 'P2', 'P3'], termination_limit=2)
        host.queue_panel = _Queue([{'id': 1, 'prompt': 'Q1'}, {'id': 2, 'prompt': 'Q2'}])
        self.start(host)                      # P1 (덱)
        self.step(True)                       # 덱 1장 → 큐 Q1 시작
        self.step(False)                      # Q1 실패(재시도 없음) → 큐 Q2
        self.step(True)                       # Q2 성공 → 덱 P2
        self.assertEqual(host.auto_gen_count, 1)
        self.step(True)                       # P2 → 2장 → 종료
        self.assertEqual(host.auto_gen_count, 2)
        self.assertEqual([p for p, _q in host.started], ['P1', 'Q1', 'Q2', 'P2'])
        self.assertFalse(host.is_automating)

    def test_queue_item_does_not_consume_the_next_prompt_override(self):
        host = _Host(['P1', 'P2'], repeat_per_prompt=2)
        host.queue_panel = _Queue([{'id': 1, 'prompt': 'Q1'}])
        self.start(host)                      # P1 생성 중
        host._set_prompt_override('P1 edited')
        self.step(True)                       # → 큐 Q1 (덮어쓰기는 그대로)
        self.assertEqual(host.started[-1], ('Q1', True))
        self.step(True)                       # → P1 반복 2번째 — 편집본으로
        self.assertEqual(host.started[-1], ('P1 edited', False))

    def test_override_for_previous_prompt_is_dropped_when_a_new_one_is_drawn(self):
        host = _Host(['P1', 'P2'])
        self.start(host)                      # P1 생성 중 — 패널은 P1 을 보여 준다
        host._set_prompt_override('P1 edited')
        self.step(True)                       # 새 프롬프트 P2 를 뽑는다
        self.assertEqual(host.started[-1], ('P2', False))
        self.assertEqual(host._auto_prompt_override, '')
        self.assertTrue(any('버렸습니다' in s for s in host.statuses))

    def test_override_set_during_wait_applies_to_the_drawn_prompt(self):
        host = _Host(['P1', 'P2'], delay=1)
        self.start(host)
        host.finish(True)                     # P2 를 뽑고 대기 타이머 시작(실행 전)
        host._set_prompt_override('P2 edited')
        host._automation_generate()           # 대기 종료
        self.assertEqual(host.started[-1], ('P2 edited', False))


if __name__ == "__main__":
    unittest.main()
