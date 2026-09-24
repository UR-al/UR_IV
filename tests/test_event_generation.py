"""Event Gen Vue-to-queue contract regression tests."""

from __future__ import annotations

import math
import unittest

from core.event_generation import (
    MAX_EVENT_SCENARIOS,
    EventGenerationPlanError,
    plan_event_generation,
)
from ui.event_search_actions import EventSearchActionsMixin, normalize_event_ratings
from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin


class _Signal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _Bridge:
    def __init__(self):
        self.showNotification = _Signal()
        self.eventSearchResults = _Signal()
        self.eventSearchProgress = _Signal()
        self.eventLoadStatus = _Signal()
        self.searchStatus = _Signal()


class _RunningWorker:
    def __init__(self):
        self.cancelled = False

    def isRunning(self):
        return True

    def cancel(self):
        self.cancelled = True


class _QueueManager:
    def __init__(self):
        self.start_count = 0

    def start(self):
        self.start_count += 1


class _HiddenEventTab:
    def __getattribute__(self, name):
        raise AssertionError(f"hidden EventGen tab was accessed: {name}")


class _Harness(EventSearchActionsMixin, ModelDownloadActionsMixin):
    _handle_vue_action = GeneratorMainUI._handle_vue_action
    _handle_event_generation_request = (
        GeneratorMainUI._handle_event_generation_request
    )
    _start_event_search = GeneratorMainUI._start_event_search

    def __init__(self):
        self.vue_bridge = _Bridge()
        self.queue_manager = _QueueManager()
        self.event_gen_tab = _HiddenEventTab()
        self.received = []
        self.statuses = []
        self.search_runs = []
        self.load_requests = []

    def _handle_creator_action(self, _action, _payload):
        return False

    def _handle_chat_action(self, _action, _payload):
        return False

    def receive_event_scenarios(self, scenarios):
        self.received.append(scenarios)

    def show_status(self, message):
        self.statuses.append(message)

    def _run_event_search_worker(self, loader, payload):
        self.search_runs.append((loader, payload))

    def _auto_load_event_data(self, ratings):
        self.load_requests.append(ratings)


def _request(prompt="1girl"):
    return {
        "scenarios": [
            {
                "name": "Step 0",
                "payload": {
                    "prompt": prompt,
                    "negative_prompt": "low quality",
                    "steps": 25,
                    "alwayson_scripts": {"NegPiP": {"args": [True]}},
                },
            }
        ]
    }


class EventGenerationPlanTests(unittest.TestCase):
    def test_plan_validates_detaches_and_preserves_queue_payload(self):
        request = _request(" 1girl, blue hair ")

        plan = plan_event_generation(request)

        self.assertEqual(plan.count, 1)
        self.assertEqual(plan.scenarios[0]["name"], "Step 0")
        self.assertEqual(
            plan.scenarios[0]["payload"],
            {
                "prompt": "1girl, blue hair",
                "negative_prompt": "low quality",
                "steps": 25,
                "alwayson_scripts": {"NegPiP": {"args": [True]}},
            },
        )
        request["scenarios"][0]["payload"]["prompt"] = "changed"
        request["scenarios"][0]["payload"]["alwayson_scripts"]["NegPiP"][
            "args"
        ][0] = False
        self.assertEqual(plan.scenarios[0]["payload"]["prompt"], "1girl, blue hair")
        self.assertEqual(
            plan.scenarios[0]["payload"]["alwayson_scripts"]["NegPiP"][
                "args"
            ],
            [True],
        )

    def test_missing_or_invalid_scenarios_fail_the_whole_plan(self):
        invalid_requests = (
            None,
            {},
            {"scenarios": []},
            {"scenarios": [None]},
            {"scenarios": [{}]},
            {"scenarios": [{"payload": {"prompt": "   "}}]},
            {"scenarios": [{"payload": {"prompt": "ok", "negative_prompt": 3}}]},
            {"scenarios": [{"payload": {"prompt": "ok", "cfg_scale": math.inf}}]},
        )

        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(EventGenerationPlanError):
                    plan_event_generation(request)

    def test_plan_has_a_bounded_scenario_count(self):
        scenario = {"payload": {"prompt": "ok"}}
        with self.assertRaises(EventGenerationPlanError):
            plan_event_generation(
                {"scenarios": [scenario] * (MAX_EVENT_SCENARIOS + 1)}
            )


class EventGenerationActionTests(unittest.TestCase):
    def test_generate_now_enqueues_exact_vue_plan_and_starts_queue(self):
        harness = _Harness()
        request = _request("vue prompt")

        harness._handle_vue_action("event_generate_now", request)

        self.assertEqual(len(harness.received), 1)
        self.assertEqual(
            harness.received[0][0]["payload"]["prompt"], "vue prompt"
        )
        self.assertEqual(harness.queue_manager.start_count, 1)

    def test_add_to_queue_does_not_start_processing(self):
        harness = _Harness()

        harness._handle_vue_action("event_add_to_queue", _request())

        self.assertEqual(len(harness.received), 1)
        self.assertEqual(harness.queue_manager.start_count, 0)

    def test_invalid_request_neither_enqueues_nor_starts(self):
        harness = _Harness()

        harness._handle_vue_action("event_generate_now", {"scenarios": []})

        self.assertEqual(harness.received, [])
        self.assertEqual(harness.queue_manager.start_count, 0)
        self.assertEqual(harness.vue_bridge.showNotification.calls[0][0], "warning")

    def test_event_search_uses_window_owned_loader_not_hidden_tab(self):
        harness = _Harness()
        loader = object()
        harness._event_loader = loader
        harness._event_loader_ratings = ("g",)
        payload = {"ratings": ["g"], "prompt": "running"}

        harness._start_event_search(payload)

        self.assertEqual(harness.search_runs, [(loader, payload)])
        self.assertEqual(harness.load_requests, [])

    def test_event_search_reloads_when_rating_selection_changes(self):
        harness = _Harness()
        harness._event_loader = object()
        harness._event_loader_ratings = ("g",)
        payload = {"ratings": ["s", "q"], "prompt": "running"}

        harness._start_event_search(payload)

        self.assertEqual(harness.search_runs, [])
        self.assertEqual(harness.load_requests, [("s", "q")])
        self.assertIs(harness._pending_event_payload, payload)


class EventSearchOrderingTests(unittest.TestCase):
    """중복 적재·검색 방지와 옛 결과 버리기 (EventSearchActionsMixin)."""

    def test_ratings_are_normalised(self):
        self.assertEqual(normalize_event_ratings(["s", "x", 3, "e"]), ("s", "e"))
        self.assertEqual(normalize_event_ratings([]), ("g",))
        self.assertEqual(normalize_event_ratings("g"), ("g",))
        self.assertEqual(normalize_event_ratings(None), ("g",))

    def test_generator_main_uses_the_mixin_implementation(self):
        self.assertIs(
            GeneratorMainUI._start_event_search,
            EventSearchActionsMixin._start_event_search,
        )
        self.assertIs(
            GeneratorMainUI._run_event_search_worker,
            EventSearchActionsMixin._run_event_search_worker,
        )

    def test_second_request_while_loading_reuses_the_running_load(self):
        harness = _Harness()
        first = {"ratings": ["s"], "prompt": "first"}
        second = {"ratings": ["s"], "prompt": "second"}

        harness._start_event_search(first)
        harness._start_event_search(second)

        self.assertEqual(harness.load_requests, [("s",)], "같은 등급을 두 번 읽지 않는다")
        loader = object()
        harness._on_event_data_loaded(loader)

        self.assertEqual(harness.search_runs, [(loader, second)], "마지막 요청만 검색한다")
        self.assertEqual(harness._event_loader_ratings, ("s",))
        self.assertEqual(harness._pending_event_payload, {})

    def test_rating_change_during_load_triggers_a_reload_of_the_new_rating(self):
        harness = _Harness()
        harness._start_event_search({"ratings": ["g"], "prompt": "a"})
        latest = {"ratings": ["e"], "prompt": "b"}
        harness._start_event_search(latest)
        self.assertEqual(harness.load_requests, [("g",)])

        harness._on_event_data_loaded(object())

        # 적재가 끝난 g 가 아니라 마지막 요청의 e 를 다시 적재한다
        self.assertEqual(harness.load_requests, [("g",), ("e",)])
        self.assertEqual(harness.search_runs, [])
        self.assertIs(harness._pending_event_payload, latest)
        self.assertIsNone(harness._event_loader, "옛 등급 적재본을 들고 새 등급을 읽지 않는다")

    def test_loaded_ratings_come_from_the_load_not_from_a_later_request(self):
        harness = _Harness()
        harness._start_event_search({"ratings": ["q"], "prompt": "a"})
        loader = object()
        harness._on_event_data_loaded(loader)
        self.assertEqual(harness._event_loader_ratings, ("q",))
        self.assertIs(harness._event_loader, loader)

    def test_load_error_is_reported_and_clears_pending_state(self):
        harness = _Harness()
        harness._start_event_search({"ratings": ["g"], "prompt": "a"})

        harness._on_event_data_loaded("오류: broken")

        self.assertEqual(harness.vue_bridge.eventSearchResults.calls, [('{"error": "\\uc624\\ub958: broken"}',)])
        self.assertEqual(harness._pending_event_payload, {})
        self.assertIsNone(harness._event_loading_ratings)
        # 다음 요청은 다시 적재를 시작할 수 있다
        harness._start_event_search({"ratings": ["g"], "prompt": "b"})
        self.assertEqual(harness.load_requests, [("g",), ("g",)])

    def test_stale_search_results_and_progress_are_dropped(self):
        harness = _Harness()
        old_worker = _RunningWorker()
        harness._event_search_worker = old_worker
        stale_id = harness._begin_event_search_request()
        current_id = harness._begin_event_search_request()

        self.assertTrue(old_worker.cancelled, "새 요청은 돌고 있던 검색을 취소한다")
        harness._on_event_search_progress(stale_id, 5, 10)
        harness._on_event_search_finished(stale_id, '[{"old": true}]')
        harness._on_event_search_progress(current_id, 7, 10)
        harness._on_event_search_finished(current_id, "[]")

        self.assertEqual(harness.vue_bridge.eventSearchProgress.calls, [(7, 10)])
        self.assertEqual(harness.vue_bridge.eventSearchResults.calls, [("[]",)])

    def test_stale_event_shards_are_announced(self):
        harness = _Harness()
        harness._start_event_search({"ratings": ["g"], "prompt": "a"})
        loader = type("L", (), {"stale_shards": ["danbooru_sorted/danbooru_g.parquet"]})()

        harness._on_event_data_loaded(loader)

        calls = harness.vue_bridge.showNotification.calls
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "warning")
        self.assertIn("danbooru_sorted/danbooru_g.parquet", calls[0][1])


if __name__ == "__main__":
    unittest.main()
