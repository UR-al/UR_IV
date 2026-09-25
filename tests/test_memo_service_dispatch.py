"""memoState 순서와 저장 오류 — core.memo_sync_service + ui.memo_actions 의 GUI 스레드 통로.

워커가 만든 memoState 를 큐로 넘기면, 그 사이 GUI 스레드에서 처리된 저장의 더 새 memoState 보다 늦게
도착해 화면이 옛 목록으로 돌아간다. 그래서 워커는 'GUI 스레드에서 보내 달라'고만 하고, payload 는
보내는 순간 GUI 스레드에서 만든다(대기 중인 알림은 하나로 합친다).
"""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PyQt6.QtCore import QCoreApplication, QObject

from core.memo_store import MemoStore
from core.memo_sync_service import MemoSyncService, MemoSyncTarget
from tests.test_memo_actions import FakeTimer
from tests.test_memo_store import Clock, Ids
from tests.test_memo_sync import FakeForge
from ui.memo_actions import MemoActionsMixin

SERVER = "http://127.0.0.1:7860"


class DispatchCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.store = MemoStore(Path(self.tmp.name) / "memos.json", clock=self.clock, id_factory=Ids())
        self.server = FakeForge(self.clock)
        self.states, self.notes, self.queue = [], [], []
        self.service = MemoSyncService(
            self.store,
            target_provider=lambda: MemoSyncTarget(kind="forge", server=SERVER, client=self.server,
                                                   connected=True),
            emit=self.states.append, notify=lambda level, msg: self.notes.append((level, msg)),
            timer_factory=FakeTimer, start_thread=lambda fn: fn(), dispatch=self.queue.append)

    def drain(self):
        while self.queue:
            self.queue.pop(0)()


class WorkerEmitOrderTests(DispatchCase):
    def test_worker_state_is_built_when_the_gui_thread_sends_it(self):
        self.server.edit("r1", "from forge")
        self.service.handle_sync()                      # 워커: syncing → 끝, 알림은 하나로 합쳐 대기
        self.assertEqual(len(self.queue), 1)
        self.assertEqual(self.states, [])
        # 그 알림이 GUI 스레드에서 돌기 전에 저장이 먼저 처리된다
        self.service.handle_save({"id": "m1", "title": "t", "text": "just typed", "base_updated_at": None})
        after_save = len(self.states)
        self.drain()
        for state in self.states[after_save - 1:]:      # 저장 뒤의 어떤 memoState 도 옛 목록이 아니다
            self.assertIn("m1", [m["id"] for m in state["memos"]])
        self.assertEqual(sorted(m["id"] for m in self.states[-1]["memos"]), ["m1", "r1"])
        self.assertTrue(self.states[-1]["sync"]["available"])
        self.assertFalse(self.states[-1]["sync"]["syncing"])

    def test_change_after_the_queued_emit_ran_schedules_another(self):
        self.service.handle_sync()
        self.drain()
        count = len(self.states)
        self.service.handle_sync()
        self.assertEqual(len(self.queue), 1)
        self.drain()
        self.assertEqual(len(self.states), count + 1)

    def test_failed_dispatch_does_not_wedge_later_emits(self):
        calls = []

        def flaky(fn):
            calls.append(fn)
            if len(calls) == 1:
                raise RuntimeError("wrapped C/C++ object has been deleted")
            fn()
        self.service._dispatch = flaky
        with mock.patch("builtins.print"):
            self.service.handle_sync()
            self.service.handle_sync()
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(self.states and self.states[-1]["sync"]["available"])

    def test_queued_emit_after_shutdown_is_dropped(self):
        self.service.handle_sync()
        self.service.shutdown()
        self.drain()
        self.assertEqual(self.states, [])


class SaveErrorTests(DispatchCase):
    def test_unexpected_write_error_is_reported_and_later_saves_work(self):
        import core.memo_store as store_module
        real_write = store_module.atomic_write_json
        calls = []

        def fail_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise UnicodeEncodeError("utf-8", "\ud83d", 0, 1, "surrogates not allowed")
            return real_write(*args, **kwargs)
        with mock.patch("core.memo_store.atomic_write_json", side_effect=fail_once), \
                mock.patch("traceback.print_exc"):
            self.service.handle_save({"id": "m1", "title": "t", "text": "x"})
            self.assertEqual(self.notes[-1][0], "error")
            self.assertIn("UnicodeEncodeError", self.notes[-1][1])
            self.assertEqual(self.states[-1]["memos"], [])        # 저장된 것처럼 보이지 않는다
            self.service.handle_save({"id": "m2", "title": "t", "text": "fine"})
        self.assertEqual([m["id"] for m in self.states[-1]["memos"]], ["m2"])
        self.assertEqual([m["id"] for m in MemoStore(self.store.path).list_public()], ["m2"])

    def test_lone_surrogate_from_the_bridge_is_saved(self):
        import json
        payload = json.loads('{"id":"m1","title":"t","text":"emoji cut \\ud83d","base_updated_at":null}')
        self.service.handle_save(payload)
        self.service.handle_save({"id": "m2", "title": "t", "text": "other"})
        self.assertEqual(self.notes, [])
        self.assertEqual(sorted(m["id"] for m in self.states[-1]["memos"]), ["m1", "m2"])
        self.assertTrue(self.store.path.exists())

    def test_disk_error_on_delete_is_reported_and_memo_stays(self):
        self.service.handle_save({"id": "m1", "title": "t", "text": "x"})
        with mock.patch("core.memo_store.atomic_write_json", side_effect=OSError("locked")):
            self.service.handle_delete({"id": "m1"})
        self.assertEqual(self.notes[-1][0], "error")
        self.assertEqual([m["id"] for m in self.states[-1]["memos"]], ["m1"])


class QtHost(QObject, MemoActionsMixin):
    def __init__(self):
        super().__init__()
        self.vue_bridge = SimpleNamespace(memoState=mock.Mock(), showNotification=mock.Mock())


class GuiRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_relay_runs_worker_requests_on_the_window_thread(self):
        host = QtHost()
        dispatch = host._memo_dispatcher()
        ran = []
        worker = threading.Thread(target=lambda: dispatch(lambda: ran.append(threading.current_thread())))
        worker.start()
        worker.join()
        self.assertEqual(ran, [])                        # 큐에 들어갔을 뿐 — GUI 스레드가 돌 때 실행
        deadline = time.monotonic() + 3
        while not ran and time.monotonic() < deadline:
            self.app.processEvents()
        self.assertEqual(ran, [threading.main_thread()])

    def test_plain_host_has_no_relay(self):
        self.assertIsNone(MemoActionsMixin()._memo_dispatcher())


if __name__ == "__main__":
    unittest.main()
