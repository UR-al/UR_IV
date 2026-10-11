"""연관 태그 브리지 믹스인 — 액션 분기 · 페이로드 검사 · 시그널 · 종료(ui/related_tags_actions.py). Qt 없음."""
from __future__ import annotations

import json
import unittest

from ui.related_tags_actions import RelatedTagsActionsMixin

FIELDS = {"main": "armor", "character": "", "copyright": "", "count": "", "prefix": "", "suffix": ""}


class _Signal:
    def __init__(self, fail=False):
        self.sent: list[dict] = []
        self.fail = fail

    def emit(self, text):
        if self.fail:
            raise RuntimeError("wrapped C/C++ object has been deleted")
        self.sent.append(json.loads(text))


class _Bridge:
    def __init__(self, signal):
        self.relatedTagsEvent = signal


class _Service:
    def __init__(self, emit):
        self.emit = emit
        self.queries = []
        self.builds = []
        self.cancels = []
        self.closed = False
        self.running = None

    def submit_query(self, request, ratings):
        self.queries.append((request, ratings))

    def start_build(self, request_id, shards):
        self.builds.append((request_id, shards))
        if self.running:
            return False
        self.running = request_id
        return True

    def build_event(self):
        return {"action": "related_tags_build", "requestId": self.running, "state": "building",
                "progress": {"shard": "g", "index": 0, "count": 1, "progress": 0.5, "overall": 0.5}}

    def cancel(self, request_id=None):
        self.cancels.append(request_id)
        return True

    def close(self):
        self.closed = True


class _Host(RelatedTagsActionsMixin):
    def __init__(self, signal=None):
        self.vue_bridge = _Bridge(signal or _Signal())
        self.services = []

        def factory(emit):
            service = _Service(emit)
            self.services.append(service)
            return service

        self._related_tags_service_factory = factory

    @property
    def sent(self):
        return self.vue_bridge.relatedTagsEvent.sent


class RelatedTagsActionsTests(unittest.TestCase):
    def test_only_related_tags_actions_are_handled(self):
        host = _Host()
        self.assertFalse(host._handle_related_tags_action("tile_repair_run", {}))
        self.assertEqual(host.services, [])
        self.assertTrue(host._handle_related_tags_action("related_tags_query",
                                                         {"requestId": "q1", "fields": FIELDS}))
        self.assertEqual(len(host.services), 1)

    def test_query_uses_the_rating_filter_with_a_gs_default(self):
        host = _Host()
        host._handle_related_tags_action("related_tags_query", {"requestId": "q1", "fields": FIELDS, "limit": 4})
        request, ratings = host.services[0].queries[-1]
        self.assertEqual((request.request_id, request.limit, ratings), ("q1", 4, {"g", "s"}))
        host._rating_filter = {"q", "e"}
        host._handle_related_tags_action("related_tags_query", {"requestId": "q2", "fields": FIELDS})
        self.assertEqual(host.services[0].queries[-1][1], {"q", "e"})
        host._rating_filter = set()
        host._handle_related_tags_action("related_tags_query", {"requestId": "q3", "fields": FIELDS})
        self.assertEqual(host.services[0].queries[-1][1], {"g", "s"})
        self.assertEqual(len(host.services), 1)        # 서비스는 한 번만 만든다

    def test_bad_payloads_answer_with_error_events(self):
        host = _Host()
        host._handle_related_tags_action("related_tags_query", {"requestId": "q1", "fields": {"main": 1}})
        event = host.sent[-1]
        self.assertEqual((event["action"], event["requestId"], event["state"]), ("related_tags_query", "q1", "error"))
        for key in ("tags", "context", "ratings", "missing", "stale", "unavailable"):
            self.assertEqual(event[key], [])
        self.assertIsNone(event["build"])
        host._handle_related_tags_action("related_tags_query", {"requestId": 5, "fields": FIELDS})
        self.assertEqual(host.sent[-1]["requestId"], "")
        host._handle_related_tags_action("related_tags_build", {"requestId": "b1", "shards": ["x"]})
        self.assertEqual(host.sent[-1]["action"], "related_tags_build")
        self.assertEqual(host.sent[-1]["state"], "error")
        count = len(host.sent)
        host._handle_related_tags_action("related_tags_cancel", {})      # 취소는 따로 답하지 않는다
        self.assertEqual(len(host.sent), count)
        self.assertEqual(host.services[0].queries, [])

    def test_build_cancel_and_a_second_build_gets_the_running_progress(self):
        host = _Host()
        host._handle_related_tags_action("related_tags_build", {"requestId": "b1", "shards": ["s", "g"]})
        service = host.services[0]
        self.assertEqual(service.builds, [("b1", ("g", "s"))])
        self.assertEqual(host.sent, [])
        host._handle_related_tags_action("related_tags_build", {"requestId": "b2", "shards": ["q"]})
        self.assertEqual(host.sent[-1]["requestId"], "b1")
        self.assertEqual(host.sent[-1]["state"], "building")
        host._handle_related_tags_action("related_tags_cancel", {"requestId": "b1"})
        self.assertEqual(service.cancels, ["b1"])

    def test_shutdown_stops_the_service_and_silences_events(self):
        host = _Host()
        host._handle_related_tags_action("related_tags_query", {"requestId": "q1", "fields": FIELDS})
        service = host.services[0]
        host._shutdown_related_tags()
        self.assertTrue(service.closed)
        service.emit({"action": "related_tags_query", "requestId": "late"})
        host._handle_related_tags_action("related_tags_query", {"requestId": "q2", "fields": FIELDS})
        self.assertEqual(len(service.queries), 1)
        self.assertEqual(host.sent, [])
        _Host()._shutdown_related_tags()     # 서비스를 만든 적 없어도 된다

    def test_emit_survives_a_deleted_bridge(self):
        host = _Host(_Signal(fail=True))
        host._related_tags_emit({"action": "related_tags_query"})
        host.vue_bridge = None
        host._related_tags_emit({"action": "related_tags_query"})

    def test_default_factory_builds_the_real_service_without_starting_work(self):
        host = _Host()
        del host._related_tags_service_factory
        service = host._related_tags_service()
        from core.fetch_data import DATA_DIR
        from core.related_tags_service import RelatedTagsService

        self.assertIsInstance(service, RelatedTagsService)
        self.assertEqual(service._data_dir, DATA_DIR)
        self.assertIsNone(service.build_progress())
        self.assertIs(host._related_tags_service(), service)


if __name__ == "__main__":
    unittest.main()
