"""자동화 조종석 — 일시정지 후 '다음 프롬프트' 편집 · 편집 가능 판정 · 큐 우선 동결 항목 회귀.

- 생성 중에 멈추면(repeat=1) 예전엔 방금 생성한 프롬프트를 '다음 프롬프트'로 보여 준 채 섰고,
  거기 건 편집은 재개하자마자 새로 뽑은 프롬프트 앞에서 버려져 본 적 없는 프롬프트가 나갔다.
  이제는 다음 프롬프트를 뽑아 보여 준 뒤 서고, 그 편집이 그 장에 쓰인다.
- 생성 중인 마지막 반복 장의 프롬프트는 prompt_is_next=False 로 알려 패널이 편집을 막는다.
- 멈춘 대기열에 남은 시드 탐색 항목을 자동화 '큐 우선' 경로가 UI 에서 다시 만들어 subseed 를
  잃고 seed 칸을 기준 시드로 고정하던 문제(대기열 매니저 경로와 같은 준비 규칙을 쓴다).
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QTimer

from backends import BackendType
from core.automation_prompt_state import prompt_is_next
from core.seed_explore import build_seed_explore_jobs
from tests.test_automation_retry import _Host, _Queue, _Text


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Worker:
    """gen_worker 대역 — 결과를 내기 전까지 '생성 중'."""

    def __init__(self):
        self._result_emitted = False

    def isRunning(self):
        return True


class _StatusHost(_Host):
    def __init__(self, prompts, **settings):
        super().__init__(prompts, **settings)
        self.vue_bridge = SimpleNamespace(automationStatus=_Signal(), showNotification=_Signal())
        self.seed_input = _Text('-1')
        self.aborted = []
        self.ui_applied = []
        self.overrides = []

    # 생성이 들어가면 워커가 '돌고', 끝나면 결과를 낸다
    def start_generation(self, **kwargs):
        if kwargs:
            self.overrides.append(kwargs)
            self.started.append((kwargs['payload_override'].get('prompt'),
                                 bool(getattr(self, '_auto_processing_queue', False))))
        else:
            super().start_generation()
        self.gen_worker = _Worker()
        return True

    def finish(self, success):
        worker = getattr(self, 'gen_worker', None)
        if worker is not None:
            worker._result_emitted = True
        super().finish(success)

    def _apply_payload_to_ui(self, item):
        self.ui_applied.append(item)
        super()._apply_payload_to_ui(item)
        if item.get('seed') is not None:
            self.seed_input.setPlainText(str(item['seed']))

    def _abort_generation(self, msg):
        self.aborted.append(msg)
        if self.is_automating:
            self._stop_automation(msg)

    def status(self):
        return json.loads(self.vue_bridge.automationStatus.calls[-1][0])


class PromptIsNextRuleTests(unittest.TestCase):
    def rule(self, **kw):
        base = dict(generating=False, held_after_generation=False, processing_queue=False,
                    queue_dirtied=False, override_pending=False, override_used=False,
                    current_repeat=1, repeat_per_prompt=1)
        base.update(kw)
        return prompt_is_next(**base)

    def test_waiting_or_held_after_the_draw_is_always_next(self):
        self.assertTrue(self.rule())
        self.assertTrue(self.rule(current_repeat=5, override_used=True))

    def test_last_repeat_in_flight_is_not_next(self):
        self.assertFalse(self.rule(generating=True))
        self.assertFalse(self.rule(held_after_generation=True))
        self.assertFalse(self.rule(generating=True, override_pending=True))

    def test_remaining_repeats_keep_the_prompt_editable(self):
        self.assertTrue(self.rule(generating=True, current_repeat=1, repeat_per_prompt=3))
        self.assertTrue(self.rule(generating=True, current_repeat=1, repeat_per_prompt=3,
                                  processing_queue=True, override_pending=True))

    def test_what_is_on_screen_differs_from_the_next_repeat(self):
        for flag in ('processing_queue', 'queue_dirtied', 'override_used'):
            self.assertFalse(self.rule(generating=True, current_repeat=1, repeat_per_prompt=3,
                                       **{flag: True}), flag)

    def test_bad_repeat_setting_is_treated_as_exhausted(self):
        self.assertFalse(self.rule(generating=True, repeat_per_prompt='x'))


class _Flow(unittest.TestCase):
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


class PauseToEditTests(_Flow):
    def test_pause_during_generation_shows_and_applies_to_the_next_prompt(self):
        host = _StatusHost(['P1', 'P2', 'P3'], delay=1)
        self.start(host)                               # P1 생성 중
        self.assertFalse(host.status()['prompt_is_next'])
        host._pause_automation()
        self.step(True)                                # P1 끝 → P2 를 뽑아 보여 준 채 선다
        st = host.status()
        self.assertTrue(st['paused'])
        self.assertEqual(st['prompt'], 'P2')
        self.assertTrue(st['prompt_is_next'])
        self.assertEqual(len(host.started), 1)         # 아직 안 나갔다

        host._set_prompt_override('P2 edited')         # 멈춘 채 다음 장을 손본다
        host._resume_automation()                      # 얼린 간격부터 다시 센다
        self.assertTrue(st['waiting'])
        host._automation_generate()                    # 간격 끝
        self.assertEqual(host.started[-1], ('P2 edited', False))
        self.assertFalse(any('버렸습니다' in s for s in host.statuses))

    def test_pause_with_no_delay_holds_right_before_generation(self):
        host = _StatusHost(['P1', 'P2'])
        self.start(host)
        host._pause_automation()
        self.step(True)
        self.assertEqual(host._auto_resume_pending, 'generate')
        self.assertEqual(host.status()['prompt'], 'P2')
        host._set_prompt_override('P2 edited')
        host._resume_automation()
        host.run_timers()
        self.assertEqual(host.started[-1], ('P2 edited', False))

    def test_count_limit_reached_while_paused_finishes_instead_of_holding(self):
        host = _StatusHost(['P1', 'P2'], termination_limit=1)
        self.start(host)
        host._pause_automation()
        self.step(True)
        self.assertFalse(host.is_automating)

    def test_pause_with_a_queue_item_waiting_holds_before_the_queue(self):
        host = _StatusHost(['P1', 'P2'])
        host.queue_panel = _Queue([{'id': 1, 'prompt': 'Q1'}])
        self.start(host)
        host._pause_automation()
        self.step(True)
        self.assertEqual(host._auto_resume_pending, 'continue')
        self.assertFalse(host.status()['prompt_is_next'])     # P1 은 다시 안 나간다
        host._resume_automation()
        host.run_timers()
        self.assertEqual(host.started[-1], ('Q1', True))


class PromptIsNextStatusTests(_Flow):
    def test_last_repeat_in_flight_is_locked_and_remaining_repeats_are_not(self):
        host = _StatusHost(['P1', 'P2'])
        self.start(host)
        self.assertFalse(host.status()['prompt_is_next'])     # repeat=1 — 다음 장은 새로 뽑는다

        host = _StatusHost(['P1', 'P2'], repeat_per_prompt=2)
        self.start(host)
        self.assertTrue(host.status()['prompt_is_next'])      # 다음 반복 장이 같은 프롬프트
        self.step(True)                                       # 두 번째 반복 장 생성 중
        self.assertFalse(host.status()['prompt_is_next'])

    def test_the_drawn_prompt_is_editable_during_the_wait(self):
        host = _StatusHost(['P1', 'P2'], delay=1)
        self.start(host)
        self.step(True)
        st = host.status()
        self.assertTrue(st['waiting'])
        self.assertEqual(st['prompt'], 'P2')
        self.assertTrue(st['prompt_is_next'])


class AutomationQueueDispatchTests(_Flow):
    def _seed_items(self, backend, count=3):
        from core.xyz_capabilities import backend_identity
        base = {'prompt': 'frozen <lora:x:0.7>', 'seed': -1, 'steps': 28, 'cfg_scale': 5,
                'width': 832, 'height': 1216, 'enable_hr': True,
                'alwayson_scripts': {'ADetailer': {'args': [True]}}}
        jobs = build_seed_explore_jobs(base, base_seed=777, model='anima.safetensors',
                                       backend_id=backend_identity('webui', backend),
                                       supports_subseed=True, count=count)
        return [{**job, 'id': i} for i, job in enumerate(jobs[1:], start=1)]

    def test_leftover_seed_explore_items_keep_their_frozen_payload(self):
        backend = SimpleNamespace(api_url='http://127.0.0.1:7860')
        host = _StatusHost(['P1', 'P2', 'P3'])
        host.queue_panel = _Queue(self._seed_items(backend))
        with mock.patch('backends.get_backend', return_value=backend), \
             mock.patch('backends.get_backend_type', return_value=BackendType.WEBUI):
            self.start(host)                       # P1 (덱)
            self.step(True)                        # → 시드 탐색 항목 1
            self.step(True)                        # → 시드 탐색 항목 2
            self.step(True)                        # → 덱 P2
        self.assertEqual([o['payload_override']['subseed_strength'] for o in host.overrides], [0.05, 0.1])
        for o in host.overrides:
            self.assertEqual(o['payload_override']['seed'], 777)
            self.assertEqual(o['model_override'], 'anima.safetensors')
            self.assertIs(o['backend_override'], backend)
            self.assertTrue(o['payload_override']['enable_hr'])
        self.assertEqual(host.ui_applied, [])                  # UI 에서 다시 만들지 않는다
        self.assertEqual(host.seed_input.toPlainText(), '-1')  # seed 칸은 그대로
        self.assertEqual(host.started[-1], ('P2', False))      # 동결 항목 뒤 덱이 그대로 이어진다
        self.assertEqual(host.auto_gen_count, 1)               # 큐 항목은 세지 않는다

    def test_a_failed_frozen_item_is_retried_with_its_frozen_payload(self):
        # 재시도는 _automation_start_generation 으로 들어온다 — UI 에서 만들면 덱 프롬프트가 대신 나간다
        backend = SimpleNamespace(api_url='http://127.0.0.1:7860')
        host = _StatusHost(['P1', 'P2'], max_retries=1)
        host.queue_panel = _Queue(self._seed_items(backend, count=2))
        with mock.patch('backends.get_backend', return_value=backend), \
             mock.patch('backends.get_backend_type', return_value=BackendType.WEBUI):
            self.start(host)                       # P1
            self.step(True)                        # → 시드 탐색 항목
            self.step(False)                       # 실패 → 같은 항목 재시도
            self.step(True)                        # 성공 → 덱 P2
        self.assertEqual([o['payload_override']['subseed_strength'] for o in host.overrides], [0.05, 0.05])
        self.assertEqual([p for p, _q in host.started], ['P1', 'frozen <lora:x:0.7>',
                                                         'frozen <lora:x:0.7>', 'P2'])
        self.assertEqual(host.queue_panel.items, [])
        self.assertEqual(host.auto_gen_count, 1)

    def test_item_from_another_backend_stops_automation_and_stays_queued(self):
        backend = SimpleNamespace(api_url='http://127.0.0.1:7860')
        host = _StatusHost(['P1', 'P2'])
        items = self._seed_items(backend, count=2)
        host.queue_panel = _Queue(items)
        with mock.patch('backends.get_backend', return_value=SimpleNamespace(api_url='http://elsewhere')), \
             mock.patch('backends.get_backend_type', return_value=BackendType.WEBUI):
            self.start(host)
            self.step(True)
        self.assertEqual(len(host.aborted), 1)
        self.assertFalse(host.is_automating)
        self.assertFalse(host._auto_processing_queue)          # 다음 회차가 이 항목을 지우지 않는다
        self.assertEqual(host.queue_panel.items, items)

    def test_plain_items_still_go_through_the_ui(self):
        host = _StatusHost(['P1', 'P2'])
        host.queue_panel = _Queue([{'id': 1, 'prompt': 'Q1', 'seed': 5}])
        self.start(host)
        self.step(True)
        self.assertEqual(host.started[-1], ('Q1', True))
        self.assertEqual(len(host.ui_applied), 1)
        self.assertTrue(host._queue_dirtied_prompt)

    def test_a_new_run_does_not_inherit_a_stale_queue_processing_flag(self):
        host = _StatusHost(['P1', 'P2'])
        host._auto_processing_queue = True
        self.start(host)
        self.assertFalse(host._auto_processing_queue)
        self.step(True)
        self.assertEqual(host.auto_gen_count, 1)


if __name__ == '__main__':
    unittest.main()
