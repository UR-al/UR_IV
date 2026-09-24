"""ui.event_search_actions — 끝난·밀려난 워커가 옛 등급 적재본을 붙들지 않는지 (실제 QThread).

예전엔 EventSearchWorker 가 메인 창을 Qt 부모로 둔 채 지워지지 않고 `_loader` 를 계속
들고 있어, 등급을 바꿀 때 `_event_loader = None` 을 해도 옛 적재본(수백 MB)이 그대로
상주했고 검색마다 워커가 창의 자식으로 쌓였다.
"""
from __future__ import annotations

import gc
import threading
import time
import unittest
import weakref
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent, QObject, QThread, pyqtSignal

from core.event_data_loader import EventSearchCancelled
from ui.event_search_actions import EventSearchActionsMixin, thread_finished_signal


class _QtBridge(QObject):
    eventLoadStatus = pyqtSignal(str)
    eventSearchResults = pyqtSignal(str)
    eventSearchProgress = pyqtSignal(int, int)
    showNotification = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.results: list[str] = []
        self.load_status: list[str] = []
        self.eventSearchResults.connect(self.results.append)
        self.eventLoadStatus.connect(self.load_status.append)


class _Host(QObject, EventSearchActionsMixin):
    """GeneratorMainUI 대역 — 워커의 Qt 부모가 되는 QObject."""

    def __init__(self):
        super().__init__()
        self.vue_bridge = _QtBridge()
        self.statuses: list[str] = []

    def show_status(self, message):
        self.statuses.append(message)


class _SearchLoader:
    """search_by_prompt 만 있는 적재본 대역. block=True 면 취소될 때까지 붙잡고 있다."""

    def __init__(self, block: bool = False):
        self.block = block
        self.started = threading.Event()

    def search_by_prompt(self, **kwargs):
        self.started.set()
        if self.block:
            cancel_check = kwargs["cancel_check"]
            deadline = time.monotonic() + 5
            while not cancel_check():
                if time.monotonic() > deadline:
                    raise RuntimeError("search was never cancelled")
                time.sleep(0.005)
            raise EventSearchCancelled()
        return []


class _FakeEventDataLoader(_SearchLoader):
    """workers.event_data_load_worker.EventDataLoader 대역 (parquet 없이)."""

    instances: list = []

    def __init__(self, parquet_dir):
        super().__init__()
        self.parquet_dir = parquet_dir
        self.ratings = None
        _FakeEventDataLoader.instances.append(weakref.ref(self))

    def load_parquets_by_rating(self, ratings, progress_callback=None):
        self.ratings = tuple(ratings)
        if progress_callback is not None:
            progress_callback(1, 1, ratings[0])


def _flush(app) -> None:
    app.processEvents()
    # deleteLater 는 exec() 루프 밖의 processEvents 로는 처리되지 않는다
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _pump(app, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _flush(app)
        if predicate():
            return True
        time.sleep(0.005)
    _flush(app)
    return predicate()


class EventSearchReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self.host = _Host()

    def tearDown(self):
        # 실패한 테스트가 스레드를 남기면 파괴 전에 끝까지 기다린다
        for thread in self.host.findChildren(QThread):
            thread.wait(5000)
        _flush(self.app)

    def _threads(self):
        return self.host.findChildren(QThread)

    def test_thread_finished_signal_is_the_qthread_one_even_when_shadowed(self):
        from workers.event_data_load_worker import EventDataLoadWorker

        worker = EventDataLoadWorker("unused", ("g",))
        self.assertEqual(thread_finished_signal(worker).signal, QThread.finished.__get__(worker).signal)
        self.assertNotEqual(worker.finished.signal, thread_finished_signal(worker).signal,
                            "커스텀 finished(object) 가 아니라 QThread 자체의 finished() 여야 한다")

    def test_finished_search_worker_is_deleted_and_drops_the_loader(self):
        loader = _SearchLoader()
        loader_ref = weakref.ref(loader)

        self.host._run_event_search_worker(loader, {"prompt": "1girl"})
        self.assertEqual(len(self._threads()), 1)
        del loader

        self.assertTrue(_pump(self.app, lambda: not self._threads()), "끝난 워커는 창의 자식으로 남지 않는다")
        gc.collect()
        self.assertEqual(self.host.vue_bridge.results, ["[]"])
        self.assertIsNone(self.host._event_search_worker)
        self.assertIsNone(loader_ref(), "끝난 검색은 적재본을 붙들지 않는다")

    def test_repeated_searches_do_not_pile_up_workers(self):
        loader = _SearchLoader()
        for index in range(3):
            self.host._run_event_search_worker(loader, {"prompt": f"p{index}"})
            self.assertTrue(_pump(self.app, lambda: not self._threads()))
        self.assertEqual(self.host.vue_bridge.results, ["[]", "[]", "[]"])

    def test_rating_switch_frees_the_previous_loader(self):
        old = _SearchLoader()
        old_ref = weakref.ref(old)
        self.host._event_loader = old
        self.host._event_loader_ratings = ("g",)
        self.host._start_event_search({"ratings": ["g"], "prompt": "a"})
        self.assertTrue(_pump(self.app, lambda: bool(self.host.vue_bridge.results)))

        load_requests = []
        self.host._auto_load_event_data = load_requests.append
        self.host._start_event_search({"ratings": ["s"], "prompt": "b"})
        del old

        self.assertEqual(load_requests, [("s",)])
        self.assertTrue(_pump(self.app, lambda: not self._threads()))
        gc.collect()
        self.assertIsNone(self.host._event_loader)
        self.assertIsNone(old_ref(), "새 등급을 읽는 동안 옛 등급 적재본이 함께 상주하지 않는다")

    def test_superseded_running_search_is_cancelled_and_releases_its_loader(self):
        old = _SearchLoader(block=True)
        old_ref = weakref.ref(old)
        self.host._event_loader = old
        self.host._event_loader_ratings = ("g",)
        self.host._start_event_search({"ratings": ["g"], "prompt": "slow"})
        self.assertTrue(old.started.wait(5))

        self.host._auto_load_event_data = lambda _ratings: None
        self.host._start_event_search({"ratings": ["q"], "prompt": "new"})
        self.assertIsNone(self.host._event_search_worker, "밀려난 워커 참조는 바로 놓는다")
        del old

        self.assertTrue(_pump(self.app, lambda: not self._threads()))
        gc.collect()
        self.assertIsNone(old_ref())
        self.assertEqual(self.host.vue_bridge.results, [], "취소된 옛 검색 결과는 Vue 로 가지 않는다")

    def test_rating_change_during_load_reloads_without_leaking_threads_or_loaders(self):
        _FakeEventDataLoader.instances = []
        with patch("workers.event_data_load_worker.EventDataLoader", _FakeEventDataLoader):
            self.host._start_event_search({"ratings": ["g"], "prompt": "first"})
            first_worker = self.host._event_load_worker
            self.assertIs(first_worker.parent(), self.host,
                          "바꿔 끼울 때 실행 중인 QThread 가 파괴되지 않도록 창을 부모로 둔다")
            self.host._start_event_search({"ratings": ["e"], "prompt": "latest"})
            del first_worker

            self.assertTrue(_pump(
                self.app,
                lambda: bool(self.host.vue_bridge.results) and not self._threads(),
            ))
        gc.collect()

        self.assertEqual(self.host.vue_bridge.results, ["[]"], "마지막 요청만 검색한다")
        self.assertEqual(self.host._event_loader_ratings, ("e",))
        loaders = [ref() for ref in _FakeEventDataLoader.instances]
        self.assertEqual(len(loaders), 2)
        self.assertIsNone(loaders[0], "적재가 끝난 g 적재본은 e 로 바꾼 뒤 풀린다")
        self.assertIs(loaders[1], self.host._event_loader)
        self.assertEqual(loaders[1].ratings, ("e",))
        self.assertIsNone(self.host._event_load_worker)
        self.assertIsNone(self.host._event_search_worker)


if __name__ == "__main__":
    unittest.main()
