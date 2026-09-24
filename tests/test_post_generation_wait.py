"""'생성 후 언로드' 대기 핸들 — UI 스레드가 아니라 생성 워커가 run() 초입에서 기다린다.

예전엔 start_generation 이 UI 스레드에서 join(timeout=30)을 했고(Forge unload-checkpoint 최대
30초 동안 창이 멈춤), PNG Info 즉시 생성·img2img 경로는 아예 기다리지 않아 경합 방지가 빠져 있었다.
"""
from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core import post_generation
from core.post_generation import start_post_generation_unload, unload_in_progress, wait_for_pending_unload


class _SlowBackend:
    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()
        self.calls = 0

    def unload_checkpoint(self):
        self.calls += 1
        self.entered.set()
        self.release.wait(5)
        return True


class PendingUnloadTests(unittest.TestCase):
    def tearDown(self):
        thread = post_generation._pending_unload
        if thread is not None:
            thread.join(5)
        post_generation._pending_unload = None

    def test_nothing_pending_returns_immediately(self):
        started = time.monotonic()
        self.assertTrue(wait_for_pending_unload(timeout=5))
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(unload_in_progress())

    def test_wait_blocks_until_the_unload_finishes_and_reports_the_result(self):
        backend = _SlowBackend()
        reported = []
        self.assertIsNotNone(start_post_generation_unload(backend, on_done=reported.append))
        self.assertTrue(backend.entered.wait(5))
        self.assertTrue(unload_in_progress())
        self.assertIsNone(start_post_generation_unload(backend), '진행 중이면 두 번 보내지 않는다')
        threading.Timer(0.1, backend.release.set).start()
        self.assertTrue(wait_for_pending_unload(timeout=5))
        post_generation._pending_unload.join(5)
        self.assertEqual(reported, [True])
        self.assertEqual(backend.calls, 1)
        self.assertFalse(unload_in_progress())

    def test_cancel_and_timeout_stop_waiting_without_hanging(self):
        backend = _SlowBackend()
        start_post_generation_unload(backend)
        self.assertTrue(backend.entered.wait(5))
        flag = threading.Event()
        threading.Timer(0.05, flag.set).start()
        started = time.monotonic()
        self.assertFalse(wait_for_pending_unload(timeout=5, cancelled=flag.is_set, poll=0.01))
        self.assertLess(time.monotonic() - started, 2)
        self.assertFalse(wait_for_pending_unload(timeout=0.05, poll=0.01))
        backend.release.set()

    def test_failing_callback_never_escapes_the_daemon_thread(self):
        def boom(_ok):
            raise RuntimeError('callback failure')
        thread = start_post_generation_unload(SimpleNamespace(unload=lambda: True), on_done=boom)
        thread.join(5)
        self.assertFalse(thread.is_alive())


class WorkerWaitTests(unittest.TestCase):
    """생성 워커가 run() 초입에서 기다리고, 기다리는 동안의 취소를 존중한다."""

    def tearDown(self):
        post_generation._pending_unload = None

    def _result(self, worker):
        results = []
        worker.finished.connect(lambda result, info: results.append((result, info)))
        worker.run()
        return results

    def test_txt2img_worker_waits_on_its_own_thread_and_rechecks_cancel(self):
        from workers.generation_worker import GenerationFlowWorker
        seen = []

        def fake_wait(timeout=30.0, *, cancelled=None, poll=0.1):
            seen.append(threading.current_thread().name)
            worker._cancelled = True   # 기다리는 동안 사용자가 취소
            return False

        backend = SimpleNamespace(txt2img=lambda *a, **k: self.fail('취소됐으면 보내지 않는다'),
                                  interrupt=lambda: None)
        worker = GenerationFlowWorker('model', {'prompt': 'x'}, backend=backend)
        with patch('workers.generation_worker.wait_for_pending_unload', side_effect=fake_wait), \
                patch('workers.generation_worker.get_backend', return_value=backend):
            results = self._result(worker)
        self.assertEqual(len(seen), 1)
        self.assertEqual(results[0][1].get('cancelled'), True)

    def test_img2img_worker_also_waits_before_reserving(self):
        from workers.generation_worker import Img2ImgFlowWorker
        order = []

        class Reserve:
            def __enter__(self):
                order.append('reserve')

            def __exit__(self, *_):
                return False

        backend = SimpleNamespace(img2img=lambda *a, **k: SimpleNamespace(success=False, error='stop here'),
                                  interrupt=lambda: None)
        worker = Img2ImgFlowWorker('model', {'prompt': 'x'})
        coordinator = SimpleNamespace(reserve=lambda *a, **k: Reserve())
        with patch('workers.generation_worker.wait_for_pending_unload',
                   side_effect=lambda **kw: order.append('wait') or True), \
                patch('workers.generation_worker.get_backend', return_value=backend), \
                patch('workers.generation_worker.get_generation_coordinator', return_value=coordinator):
            results = self._result(worker)
        self.assertEqual(order, ['wait', 'reserve'])
        self.assertEqual(results[0][0], 'stop here')

    def test_start_generation_no_longer_joins_on_the_ui_thread(self):
        import inspect
        from ui import generator_generation
        source = inspect.getsource(generator_generation)
        self.assertNotIn('.join(timeout', source)
        self.assertNotIn('_wait_for_post_gen_unload', source)


if __name__ == '__main__':
    unittest.main()
