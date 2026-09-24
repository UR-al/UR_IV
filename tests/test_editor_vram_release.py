"""VRAM 게이지 수동 언로드가 앱이 쥔 편집기 비전 모델(YOLO/SAM3 캐시)까지 반납하는지.

회귀: unload_model_request가 백엔드만 언로드해 에디터 SAM3 번들(~3.4GB)이 그대로 남았다.
"""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from core import post_generation
from ui.generator_main import GeneratorMainUI
from ui.model_download_actions import ModelDownloadActionsMixin


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, *args):
        self.values.append(args)


class _Harness(ModelDownloadActionsMixin):
    _handle_vue_action = GeneratorMainUI._handle_vue_action

    def __init__(self):
        self.statuses = []
        self.vue_bridge = SimpleNamespace(showNotification=_Signal())

    def show_status(self, message):
        self.statuses.append(message)

    def _handle_chat_action(self, _action, _payload):
        return False

    def _handle_creator_action(self, _action, _payload):
        return False


def _blocking_scheduler(ms, fn):
    """메인 스레드 QTimer 대신 — 잠깐 쉬고 바로 폴링을 이어 간다."""
    time.sleep(ms / 1000.0)
    fn()


class UnloadRequestReleasesEditorModelsTests(unittest.TestCase):
    def setUp(self):
        post_generation._pending_unload = None
        patcher = mock.patch("ui.manual_model_unload._default_scheduler", return_value=_blocking_scheduler)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        thread = post_generation._pending_unload
        if thread is not None:
            thread.join(5)
        post_generation._pending_unload = None

    def test_manual_unload_clears_editor_caches_and_backend(self):
        backend = SimpleNamespace(unload_models=mock.Mock(return_value=True))
        host = _Harness()
        with mock.patch("core.model_cache.clear_all", return_value=2) as clear_all, \
                mock.patch("backends.get_backend", return_value=backend):
            host._handle_vue_action("unload_model_request", {})
        clear_all.assert_called_once_with()
        backend.unload_models.assert_called_once_with()
        self.assertIn("Model unload requested.", host.statuses)
        self.assertEqual(host.statuses[-1], "Model unloaded.")
        self.assertEqual(host.vue_bridge.showNotification.values[-1][0], "success")

    def test_cache_release_failure_does_not_block_backend_unload(self):
        backend = SimpleNamespace(unload_models=mock.Mock(return_value=True))
        host = _Harness()
        with mock.patch("core.model_cache.clear_all", side_effect=RuntimeError("boom")), \
                mock.patch("backends.get_backend", return_value=backend):
            host._handle_vue_action("unload_model_request", {})
        backend.unload_models.assert_called_once_with()

    def test_failed_backend_unload_is_reported_as_failure(self):
        backend = SimpleNamespace(unload_models=mock.Mock(return_value=False))
        host = _Harness()
        with mock.patch("core.model_cache.clear_all", return_value=0), \
                mock.patch("backends.get_backend", return_value=backend):
            host._handle_vue_action("unload_model_request", {})
        self.assertEqual(host.statuses[-1], "Model unload failed.")
        self.assertEqual(host.vue_bridge.showNotification.values[-1][0], "error")

    def test_manual_unload_prefers_full_cleanup_over_checkpoint_only(self):
        backend = SimpleNamespace(
            unload_checkpoint=mock.Mock(return_value=True),
            unload_models=mock.Mock(return_value=True),
        )
        host = _Harness()
        with mock.patch("core.model_cache.clear_all", return_value=0), \
                mock.patch("backends.get_backend", return_value=backend):
            host._handle_vue_action("unload_model_request", {})
        backend.unload_models.assert_called_once_with()
        backend.unload_checkpoint.assert_not_called()

    def test_backend_http_runs_off_the_calling_thread_and_is_awaited_by_generation(self):
        release = threading.Event()
        callers = []

        def slow_unload():
            callers.append(threading.current_thread())
            release.wait(5)
            return True

        backend = SimpleNamespace(unload_models=slow_unload)
        host = _Harness()
        scheduled = []
        from ui.manual_model_unload import request_manual_backend_unload

        thread = request_manual_backend_unload(host, backend, schedule=lambda ms, fn: scheduled.append(fn))
        self.assertIsNotNone(thread)
        # 호출 스레드는 HTTP 를 기다리지 않고 바로 돌아온다 — 생성 워커는 이 언로드를 기다린다.
        self.assertTrue(post_generation.unload_in_progress())
        self.assertIsNone(
            request_manual_backend_unload(host, backend, schedule=lambda ms, fn: None),
            "진행 중이면 두 번 보내지 않는다",
        )
        self.assertEqual(host.statuses[-1], "Model unload already in progress.")
        release.set()
        thread.join(5)
        self.assertIsNot(callers[0], threading.current_thread())
        scheduled[0]()
        self.assertEqual(host.statuses[-1], "Model unloaded.")


if __name__ == "__main__":
    unittest.main()
