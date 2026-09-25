"""메모 액션 — core.memo_sync_service(스케줄·상태) + ui.memo_actions(브리지 배선).

memoState 모양과 액션 이름은 공유 메모 계약 그대로다(프론트 패키지가 같은 이름을 쓴다).
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.forge_memo_client import MemoRemoteError, MemoRoutesUnavailable
from core.memo_store import MemoStore
from core.memo_sync_service import (
    CONFLICT_MESSAGE, LOCAL_ONLY_MESSAGE, MemoSyncService, MemoSyncTarget,
)
from tests.test_memo_store import Clock, Ids
from tests.test_memo_sync import FakeForge
from ui.memo_actions import MemoActionsMixin

SERVER = "http://127.0.0.1:7860"
SYNC_KEYS = {"available", "target", "syncing", "last_synced_at", "error"}
MEMO_KEYS = {"id", "title", "text", "created_at", "updated_at"}


class FakeTimer:
    def __init__(self, delay, fn):
        self.delay, self.fn = delay, fn
        self.started = self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if not self.cancelled:
            self.fn()


class ServiceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.store = MemoStore(Path(self.tmp.name) / "memos.json", clock=self.clock, id_factory=Ids())
        self.server = FakeForge(self.clock)
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=self.server, connected=True)
        self.states, self.notes, self.timers, self.provided = [], [], [], 0
        self.service = MemoSyncService(
            self.store, target_provider=self.provide, emit=self.states.append,
            notify=lambda level, msg: self.notes.append((level, msg)),
            debounce_s=2.0, timer_factory=self.make_timer, start_thread=lambda fn: fn())

    def provide(self):
        self.provided += 1
        return self.target

    def make_timer(self, delay, fn):
        timer = FakeTimer(delay, fn)
        self.timers.append(timer)
        return timer

    @property
    def last(self):
        return self.states[-1]


class PayloadTests(ServiceCase):
    def test_memo_state_matches_the_contract(self):
        self.store.save("m1", "제목", "본문")
        self.store.save("gone", "t", "x")
        self.store.delete("gone")
        self.service.emit_state()
        state = self.last
        self.assertEqual(set(state), {"memos", "sync"})
        self.assertEqual(set(state["sync"]), SYNC_KEYS)
        self.assertEqual([m["id"] for m in state["memos"]], ["m1"])
        self.assertEqual(set(state["memos"][0]), MEMO_KEYS)
        self.assertEqual(state["sync"], {"available": False, "target": "local", "syncing": False,
                                         "last_synced_at": None, "error": None})
        json.dumps(state, ensure_ascii=False)   # 브리지로 그대로 나간다


class ActionTests(ServiceCase):
    def test_list_emits_then_syncs(self):
        self.server.edit("r1", "from forge")
        self.service.handle_list()
        self.assertEqual(self.states[0]["memos"], [])          # 로컬 상태를 먼저
        self.assertTrue(any(s["sync"]["syncing"] for s in self.states))
        self.assertEqual([m["text"] for m in self.last["memos"]], ["from forge"])
        self.assertEqual(self.last["sync"]["target"], "forge")
        self.assertTrue(self.last["sync"]["available"])
        self.assertFalse(self.last["sync"]["syncing"])
        self.assertIsNotNone(self.last["sync"]["last_synced_at"])
        self.assertIsNone(self.last["sync"]["error"])

    def test_save_is_local_first_and_sync_is_debounced(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "a", "base_updated_at": None})
        self.assertEqual([m["text"] for m in self.last["memos"]], ["a"])
        self.assertEqual(self.server.memos, {})
        self.assertEqual(len(self.timers), 1)
        self.assertEqual(self.timers[0].delay, 2.0)
        base = self.last["memos"][0]["updated_at"]
        self.service.handle_save({"id": "m1", "title": "t", "text": "ab", "base_updated_at": base})
        self.assertTrue(self.timers[0].cancelled)               # 타자마다 요청하지 않는다
        self.timers[-1].fire()
        self.assertEqual(self.server.memos["m1"]["text"], "ab")
        self.assertEqual(self.provided, 1)

    def test_unchanged_save_does_not_schedule(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "a"})
        base = self.last["memos"][0]["updated_at"]
        self.timers.clear()
        self.service.handle_save({"id": "m1", "title": "t", "text": "a", "base_updated_at": base})
        self.assertEqual(self.timers, [])

    def test_delete_hides_and_syncs(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "a"})
        self.timers[-1].fire()
        self.service.handle_delete({"id": "m1"})
        self.assertEqual(self.last["memos"], [])
        self.timers[-1].fire()
        self.assertTrue(self.server.memos["m1"]["deleted"])
        before = len(self.timers)
        self.service.handle_delete({"id": "unknown"})
        self.assertEqual(len(self.timers), before)

    def test_validation_error_is_notified_not_raised(self):
        self.service.handle_save({"id": "../bad", "title": "t", "text": "a"})
        self.assertEqual(self.notes[-1][0], "error")
        self.assertEqual(self.last["memos"], [])
        self.service.handle_save("not a dict")
        self.assertEqual(self.notes[-1][0], "error")

    def test_stale_save_notifies_conflict_copy(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "v1"})
        first = self.last["memos"][0]["updated_at"]
        self.service.handle_save({"id": "m1", "title": "t", "text": "v2", "base_updated_at": first})
        self.service.handle_save({"id": "m1", "title": "t", "text": "mine", "base_updated_at": first})
        self.assertEqual(self.notes[-1], ("info", CONFLICT_MESSAGE))
        self.assertEqual(len(self.last["memos"]), 2)

    def test_sync_conflict_is_notified(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "v1"})
        self.timers[-1].fire()
        base = self.last["memos"][0]["updated_at"]
        self.server.edit("m1", "forge")
        self.service.handle_save({"id": "m1", "title": "t", "text": "app", "base_updated_at": base})
        self.timers[-1].fire()
        self.assertIn(("info", CONFLICT_MESSAGE), self.notes)
        self.assertEqual(sorted(m["text"] for m in self.last["memos"]), ["app", "forge"])


class TargetTests(ServiceCase):
    def test_comfy_is_local_only(self):
        self.target = MemoSyncTarget.local()
        self.service.handle_list()
        self.assertEqual((self.last["sync"]["available"], self.last["sync"]["target"],
                          self.last["sync"]["error"]), (False, "local", None))
        self.service.handle_sync()
        self.assertEqual(self.last["sync"]["error"], LOCAL_ONLY_MESSAGE)
        self.assertEqual(self.server.writes, [])

    def test_not_connected_skips_auto_but_explicit_tries(self):
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=self.server, connected=False)
        self.store.save("m1", "t", "x")
        self.service.handle_list()
        self.assertEqual(self.server.writes, [])
        self.assertFalse(any(s["sync"]["syncing"] for s in self.states))
        self.service.handle_sync()
        self.assertEqual(self.server.writes, [("PUT", "m1")])
        self.assertTrue(self.last["sync"]["available"])

    def test_old_extension_is_quiet_unless_asked(self):
        self.server.available = False
        self.service.backend_changed()
        self.assertEqual((self.last["sync"]["available"], self.last["sync"]["error"]), (False, None))
        self.service.handle_sync()
        self.assertEqual(self.last["sync"]["error"], "old extension")   # 클라이언트 문구를 그대로
        self.assertEqual(self.last["sync"]["target"], "local")

    def test_network_error_is_reported_and_recovers(self):
        broken = mock.Mock()
        broken.list_memos.side_effect = MemoRemoteError("Forge 에 연결하지 못했습니다")
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=broken, connected=True)
        self.service.handle_list()
        self.assertEqual(self.last["sync"]["error"], "Forge 에 연결하지 못했습니다")
        self.assertFalse(self.last["sync"]["syncing"])
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=self.server, connected=True)
        self.service.handle_sync()
        self.assertIsNone(self.last["sync"]["error"])
        self.assertTrue(self.last["sync"]["available"])

    def test_rejected_memo_is_reported_while_available(self):
        self.store.save("big", "t", "x")
        self.server.reject.add("big")
        self.service.handle_sync()
        self.assertTrue(self.last["sync"]["available"])
        self.assertIn("1개", self.last["sync"]["error"])

    def test_unexpected_exception_does_not_wedge_the_worker(self):
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=object(), connected=True)
        with mock.patch("traceback.print_exc"):
            self.service.handle_sync()
        self.assertFalse(self.last["sync"]["syncing"])
        self.assertTrue(self.last["sync"]["error"])
        self.target = MemoSyncTarget(kind="forge", server=SERVER, client=self.server, connected=True)
        self.service.handle_sync()
        self.assertTrue(self.last["sync"]["available"])


class SchedulingTests(ServiceCase):
    def test_request_during_a_run_coalesces_into_one_rerun(self):
        calls = []
        real_list = self.server.list_memos

        def list_and_poke(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                self.service.request_sync()
                self.service.request_sync()
            return real_list(**kwargs)
        self.server.list_memos = list_and_poke
        self.service.handle_sync()
        self.assertEqual(len(calls), 2)

    def test_shutdown_cancels_pending_timer_and_stops_emitting(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "a"})
        timer = self.timers[-1]
        self.service.shutdown()
        self.assertTrue(timer.cancelled)
        count = len(self.states)
        self.service.handle_sync()
        timer.fn()
        self.assertEqual(len(self.states), count)
        self.assertEqual(self.server.writes, [])


class Signal:
    def __init__(self):
        self.events = []
        self.condition = threading.Condition()

    def emit(self, *args):
        with self.condition:
            self.events.append(args)
            self.condition.notify_all()

    def wait_for(self, predicate, timeout=3):
        with self.condition:
            if not self.condition.wait_for(lambda: any(predicate(e) for e in self.events), timeout):
                raise AssertionError(f"event not received: {self.events}")
            return next(e for e in self.events if predicate(e))


class Host(MemoActionsMixin):
    def __init__(self, store, target):
        self.vue_bridge = SimpleNamespace(memoState=Signal(), showNotification=Signal())
        self._memo_store_factory = lambda: store
        self._target = target

    def _memo_sync_target(self):
        return self._target


class MixinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = MemoStore(Path(self.tmp.name) / "memos.json")
        self.host = Host(self.store, MemoSyncTarget.local())
        self.addCleanup(self.host._shutdown_memo_sync)

    def states(self):
        return [json.loads(args[0]) for args in self.host.vue_bridge.memoState.events]

    def test_only_memo_actions_are_claimed(self):
        self.assertFalse(self.host._handle_memo_action("chat_send", {}))
        self.assertFalse(hasattr(self.host, "_memo_sync_service"))

    def test_actions_round_trip_through_the_bridge_signal(self):
        self.assertTrue(self.host._handle_memo_action("memo_save", {
            "id": "m1", "title": "제목", "text": "본문", "base_updated_at": None}))
        state = json.loads(self.host.vue_bridge.memoState.wait_for(
            lambda e: json.loads(e[0])["memos"])[0])
        self.assertEqual(state["memos"][0]["title"], "제목")
        self.assertEqual(set(state["sync"]), SYNC_KEYS)
        self.assertTrue(self.host._handle_memo_action("memo_list", None))
        self.assertTrue(self.host._handle_memo_action("memo_sync", {}))
        self.assertTrue(self.host._handle_memo_action("memo_delete", {"id": "m1"}))
        self.host.vue_bridge.memoState.wait_for(lambda e: json.loads(e[0])["memos"] == [])
        self.assertTrue(self.store.get("m1")["deleted"])

    def test_explicit_sync_on_comfy_reports_local_only(self):
        self.host._handle_memo_action("memo_sync", {})
        self.host.vue_bridge.memoState.wait_for(
            lambda e: json.loads(e[0])["sync"]["error"] == LOCAL_ONLY_MESSAGE)

    def test_shutdown_makes_actions_inert(self):
        self.host._shutdown_memo_sync()
        self.assertTrue(self.host._handle_memo_action("memo_save", {"id": "m1", "title": "", "text": "x"}))
        self.assertIsNone(self.store.get("m1"))
        self.host._memo_backend_connected()   # 종료 뒤 연결 통보도 무시

    def test_backend_connected_triggers_a_sync(self):
        clock = Clock()
        server = FakeForge(clock)
        server.edit("r1", "from forge")
        self.host._target = MemoSyncTarget(kind="forge", server=SERVER, client=server, connected=True)
        self.host._memo_backend_connected()
        self.host.vue_bridge.memoState.wait_for(
            lambda e: [m["id"] for m in json.loads(e[0])["memos"]] == ["r1"]
            and json.loads(e[0])["sync"]["available"])


class TargetSelectionTests(unittest.TestCase):
    """_memo_sync_target — 활성 백엔드로 대상을 고른다(네트워크 없음)."""

    def host(self, connected):
        host = MemoActionsMixin()
        host._backend_connected = connected
        return host

    def test_comfy_backend_is_local(self):
        import backends
        with mock.patch.object(backends, "get_backend_type", return_value=backends.BackendType.COMFYUI):
            target = self.host(True)._memo_sync_target()
        self.assertEqual(target.kind, "local")

    def test_webui_backend_uses_its_api_url(self):
        import backends
        fake_backend = SimpleNamespace(api_url="http://127.0.0.1:7860/")
        with mock.patch.object(backends, "get_backend_type", return_value=backends.BackendType.WEBUI), \
                mock.patch.object(backends, "get_backend", return_value=fake_backend):
            target = self.host(True)._memo_sync_target()
            offline = self.host(False)._memo_sync_target()
        self.assertEqual((target.kind, target.server, target.connected), ("forge", SERVER, True))
        self.assertEqual(target.client.base_url, SERVER)
        self.assertFalse(offline.connected)

    def test_blank_url_falls_back_to_local(self):
        import backends
        with mock.patch.object(backends, "get_backend_type", return_value=backends.BackendType.WEBUI), \
                mock.patch.object(backends, "get_backend", return_value=SimpleNamespace(api_url="")):
            self.assertEqual(self.host(True)._memo_sync_target().kind, "local")


if __name__ == "__main__":
    unittest.main()
