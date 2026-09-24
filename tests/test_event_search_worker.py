"""workers.event_search_worker — Vue payload 가 검색 인자로 빠짐없이 전달되는지."""
from __future__ import annotations

import gc
import json
import unittest
import weakref

from PyQt6.QtCore import QObject, QThread

from core.event_data_loader import EventSearchCancelled
from workers.event_search_worker import EventSearchWorker


class _Loader:
    def __init__(self, results=None, raise_cancel=False):
        self.kwargs = None
        self.results = results or []
        self.raise_cancel = raise_cancel

    def search_by_prompt(self, **kwargs):
        self.kwargs = kwargs
        if self.raise_cancel:
            raise EventSearchCancelled()
        kwargs["progress_callback"](1, 2)
        return self.results


def _payload(**overrides):
    payload = {
        "character": "hatsune_miku", "copyright": "vocaloid", "artist": "wlop",
        "ratings": ["g"], "prompt": "1girl, smile", "min_steps": 3, "max_steps": 9,
        "exclude_tags": "blush", "limit": True,
    }
    payload.update(overrides)
    return payload


class EventSearchWorkerTests(unittest.TestCase):
    def _run(self, loader, payload):
        worker = EventSearchWorker(loader, payload)
        finished, progress = [], []
        worker.search_finished.connect(finished.append)
        worker.progress.connect(lambda cur, total: progress.append((cur, total)))
        worker.run()
        return worker, finished, progress

    def test_character_copyright_artist_reach_the_loader(self):
        loader = _Loader()
        worker, finished, progress = self._run(loader, _payload())
        kwargs = loader.kwargs
        self.assertEqual(kwargs["character"], "hatsune_miku")
        self.assertEqual(kwargs["copyright"], "vocaloid")
        self.assertEqual(kwargs["artist"], "wlop")
        self.assertEqual(kwargs["prompt"], "1girl, smile")
        self.assertEqual(kwargs["exclude_tags"], "blush")
        self.assertEqual((kwargs["min_children"], kwargs["max_children"]), (3, 9))
        self.assertEqual(kwargs["limit"], 100)
        self.assertEqual(kwargs["cancel_check"], worker.is_cancelled)
        self.assertEqual(progress, [(1, 2)])
        self.assertEqual(json.loads(finished[0]), [])

    def test_bad_payload_values_fall_back_instead_of_crashing(self):
        loader = _Loader()
        self._run(loader, _payload(min_steps="", max_steps=None, character=None, limit=False))
        self.assertEqual((loader.kwargs["min_children"], loader.kwargs["max_children"]), (2, 20))
        self.assertEqual(loader.kwargs["character"], "")
        self.assertEqual(loader.kwargs["limit"], 5000)

    def test_results_become_step_lists(self):
        loader = _Loader(results=[{
            "parent": {"tag_string_general": "1girl smile", "tag_string_character": "hatsune_miku",
                       "tag_string_copyright": "vocaloid"},
            "children": [{"tag_string_general": "1girl blush"}],
            "child_count": 1,
            "similarity": 0.8,
        }])
        _worker, finished, _progress = self._run(loader, _payload())
        event = json.loads(finished[0])[0]
        self.assertEqual(event["character"], "hatsune miku")
        self.assertEqual(event["steps"][1]["added"], ["blush"])
        self.assertEqual(event["steps"][1]["removed"], ["smile"])

    def test_cancelled_search_reports_cancelled(self):
        _worker, finished, _progress = self._run(_Loader(raise_cancel=True), _payload())
        self.assertEqual(json.loads(finished[0]), {"cancelled": True})

    def test_cancel_sticks_even_before_the_thread_runs(self):
        # QThread.requestInterruption() 은 돌지 않는 스레드에선 무시된다 — 자체 플래그로 남아야
        worker = EventSearchWorker(_Loader(), _payload())
        self.assertFalse(worker.is_cancelled())
        worker.cancel()
        self.assertTrue(worker.is_cancelled())
        finished = []
        worker.search_finished.connect(finished.append)
        worker.run()
        self.assertEqual(json.loads(finished[0]), {"cancelled": True})

    def test_run_releases_the_loader_even_while_a_qt_parent_keeps_the_worker(self):
        # 메인 창이 부모면 워커 객체는 deleteLater 까지 남는다 — 적재본(수백 MB)은 그 전에 풀려야
        parent = QObject()
        factories = {
            "results": _Loader,
            "cancelled": lambda: _Loader(raise_cancel=True),
            "error": _BrokenLoader,
        }
        for name, factory in factories.items():
            with self.subTest(outcome=name):
                loader = factory()
                loader_ref = weakref.ref(loader)
                worker = EventSearchWorker(loader, _payload(), parent)
                finished = []
                worker.search_finished.connect(finished.append)
                del loader
                worker.run()
                gc.collect()
                self.assertEqual(len(finished), 1)
                self.assertIsNone(loader_ref(), "run() 이 끝나면 워커가 적재본을 붙들지 않는다")
                self.assertEqual(worker._params, {})

    def test_result_signal_does_not_shadow_qthread_finished(self):
        # 커스텀 결과 시그널이 `finished` 면 QThread.finished 에 deleteLater 를 걸 수 없다
        worker = EventSearchWorker(_Loader(), _payload())
        self.assertEqual(worker.finished.signal, QThread.finished.__get__(worker).signal)
        self.assertNotEqual(worker.search_finished.signal, worker.finished.signal)


class _BrokenLoader:
    def search_by_prompt(self, **_kwargs):
        raise RuntimeError("broken")


if __name__ == "__main__":
    unittest.main()
