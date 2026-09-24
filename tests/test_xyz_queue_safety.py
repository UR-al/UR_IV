"""Real queue/worker completion boundaries with synthetic generation only."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from backends import BackendType
from core.resource_coordinator import GenerationResourceCoordinator
from ui.generator_generation import GenerationMixin
from ui.generator_main import GeneratorMainUI
from ui.xyz_actions import XYZActionsMixin
from widgets.queue_manager import QueueManager
from workers.generation_worker import GenerationFlowWorker


class _Panel:
    """QueuePanel 의 대기열 매니저용 표면(항목 목록 + 실행 중 표시)만 흉내 낸다."""

    def __init__(self):
        self.items = []
        self._next_id = 0

    def add_single_item(self, item):
        self._next_id += 1
        self.items.append({"id": str(self._next_id), **item})

    def is_empty(self):
        return not self.items

    def count(self):
        return len(self.items)

    def get_first_item(self):
        return self.items[0] if self.items else None

    def get_item_by_id(self, item_id):
        return next((item for item in self.items if item["id"] == item_id), None)

    def remove_first_item(self):
        return self.items.pop(0)

    def consume_item(self, item_id):
        for index, item in enumerate(self.items):
            if item["id"] == item_id:
                return self.items.pop(index)
        return None

    def claim_processing(self, item_id, _owner):
        return self.get_item_by_id(item_id) is not None

    def release_processing(self, *_args):
        return False

    def processing_held_by_other(self, _owner):
        return False


class _Host(GenerationMixin, XYZActionsMixin):
    _on_generation_requested = GeneratorMainUI._on_generation_requested

    def __init__(self, backend):
        self.queue_panel = _Panel()
        self.queue_manager = QueueManager(self.queue_panel)
        self.queue_manager.delay_seconds = 0
        self.queue_manager.generation_requested.connect(self._on_generation_requested)
        self._xyz_lock = threading.RLock()
        self._xyz_seen_requests = set()
        self._xyz_capabilities = {
            "backend": backend, "context": {"hires": False, "family": "standard"},
            "data": {"capabilityId": "cap", "axes": [
                {"id": "steps", "label": "Steps", "type": "integer", "min": 1, "max": 150},
            ]},
        }
        self._xyz_emit = mock.Mock()
        self._build_generation_payload = lambda **_: ({
            "prompt": "synthetic", "steps": 20, "width": 512, "height": 512,
            "cfg_scale": 7, "seed": 1,
        }, None)
        self.model_combo = SimpleNamespace(currentText=lambda: "model")
        self.gen_worker = None
        self.is_automating = False
        self._maybe_unload_ollama = mock.Mock()
        self._backend_needs_checkpoint = lambda: False
        self._abort_generation = mock.Mock()
        for name in (
            "setWindowTitle", "btn_generate", "show_status", "vue_bridge",
            "_restore_generate_button", "_process_new_image", "isActiveWindow",
        ):
            setattr(self, name, mock.MagicMock())


class XYZQueueSafetyTests(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace(api_url="http://synthetic.invalid", txt2img=mock.Mock())
        self.host = _Host(self.backend)
        self.enterContext(mock.patch("backends.get_backend", return_value=self.backend))
        self.enterContext(mock.patch("backends.get_backend_type", return_value=BackendType.COMFYUI))
        self.enterContext(mock.patch("core.gen_stats.get_gen_stats", return_value=mock.MagicMock()))

    def _start_plot(self):
        self.host._xyz_start_plot({"requestId": "plot", "capabilityId": "cap",
                                   "axes": [{"id": "steps", "values": [25]}]})

    def test_starting_xyz_does_not_cancel_an_existing_manual_generation(self):
        active = mock.MagicMock()
        active.isRunning.return_value = True
        active.wait.return_value = True
        self.host.gen_worker = active
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as replacement:
            self._start_plot()
        active.cancel.assert_not_called()
        active.finished.disconnect.assert_not_called()
        replacement.assert_not_called()
        self.assertEqual(self.host.queue_panel.count(), 0)
        self.assertFalse(self.host._xyz_emit.call_args.args[1]["ok"])

    def _queued_worker(self):
        with mock.patch("ui.generator_generation.GenerationFlowWorker"):
            self._start_plot()
        item = self.host.queue_panel.get_first_item()
        self.assertIsNotNone(item)
        payload, model, backend = self.host._xyz_prepare_queue_generation(item)
        worker = GenerationFlowWorker(model, payload, backend=backend)
        self.host.gen_worker = worker
        worker.finished.connect(self.host.on_generation_finished)
        return worker

    def test_backend_change_before_dispatch_keeps_the_job_paused_and_resumable(self):
        worker = self._queued_worker()
        with mock.patch("workers.generation_worker.get_backend", return_value=object()):
            worker.run()
        self.backend.txt2img.assert_not_called()
        self.assertEqual(self.host.queue_panel.count(), 1)
        self.assertTrue(self.host.queue_manager.is_paused)
        self.assertEqual(self.host.queue_manager._fail_count, 0)
        # The original backend is selected again; Resume submits the same
        # snapshot exactly once, then ordinary completion consumes it.
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as factory:
            self.host.queue_manager.resume()
        self.assertEqual(factory.call_count, 1)
        model, payload = factory.call_args.args
        self.assertEqual(payload["steps"], 25)
        self.backend.txt2img.return_value = SimpleNamespace(success=True, image_data=b"synthetic", info={})
        resumed = GenerationFlowWorker(model, payload, backend=self.backend)
        resumed.finished.connect(self.host.on_generation_finished)
        with mock.patch("workers.generation_worker.get_backend", return_value=self.backend):
            resumed.run()
        self.assertEqual(self.backend.txt2img.call_count, 1)
        self.assertEqual(self.host.queue_panel.count(), 0)
        self.assertEqual(self.host.queue_manager._success_count, 1)

    def test_a_busy_resource_before_dispatch_does_not_discard_the_job(self):
        worker = self._queued_worker()
        coordinator = GenerationResourceCoordinator()
        with coordinator.reserve("synthetic-creator", unload_llm=False), mock.patch(
            "workers.generation_worker.get_generation_coordinator", return_value=coordinator
        ):
            worker.run()
        self.backend.txt2img.assert_not_called()
        self.assertEqual(self.host.queue_panel.count(), 1)
        self.assertTrue(self.host.queue_manager.is_paused)
        self.assertEqual(self.host.queue_manager._fail_count, 0)

    def test_resume_during_manual_generation_neither_cancels_it_nor_consumes_xyz(self):
        self._queued_worker()
        self.host.queue_manager.pause()
        active = mock.MagicMock()
        active.isRunning.return_value = True
        active.wait.return_value = True
        self.host.gen_worker = active
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as replacement:
            self.host.queue_manager.resume()
        active.cancel.assert_not_called()
        replacement.assert_not_called()
        self.assertTrue(self.host.queue_manager.is_paused)
        # Completion of that unrelated manual image must not delete XYZ #0.
        self.host.on_generation_finished(b"manual-result", {})
        self.assertEqual(self.host.queue_panel.count(), 1)
        self.assertEqual(self.host.queue_manager.generated_count, 0)

    def test_unrelated_or_stale_result_does_not_consume_the_current_xyz_item(self):
        self._queued_worker()
        self.host.queue_manager.pause()
        for info in ({}, {"_xyz_info": {"requestId": "old-plot", "index": 0}}):
            with self.subTest(info=info):
                self.host.on_generation_finished(b"another-result", info)
                self.assertEqual(self.host.queue_panel.count(), 1)
                self.assertEqual(self.host.queue_manager.generated_count, 0)

    def test_actual_xyz_failure_and_user_cancel_still_complete_the_owned_item(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled):
                self.host = _Host(self.backend)
                worker = self._queued_worker()
                self.backend.txt2img.reset_mock()
                self.backend.txt2img.return_value = SimpleNamespace(success=False, error="synthetic failure")
                if cancelled:
                    worker._cancelled = True
                with mock.patch("workers.generation_worker.get_backend", return_value=self.backend):
                    worker.run()
                self.assertEqual(self.backend.txt2img.call_count, 0 if cancelled else 1)
                self.assertEqual(self.host.queue_panel.count(), 0)
                self.assertEqual(self.host.queue_manager.generated_count, 1)
                self.assertFalse(self.host.queue_manager.is_paused)

    def test_finishing_worker_can_advance_without_an_implicit_cancel(self):
        finishing = mock.MagicMock()
        finishing.isRunning.return_value = True
        finishing._result_emitted = True
        finishing.wait.return_value = True
        self.host.gen_worker = finishing
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as replacement:
            self._start_plot()
        self.assertEqual(replacement.return_value.start.call_count, 1)
        finishing.cancel.assert_not_called()

    def test_existing_running_queue_accepts_additional_xyz_without_restarting_it(self):
        self.host.queue_manager.is_running = True
        self.host.queue_manager.total_count = 1
        active = mock.MagicMock()
        active.isRunning.return_value = True
        self.host.gen_worker = active
        with mock.patch("ui.generator_generation.GenerationFlowWorker") as replacement:
            self._start_plot()
        active.cancel.assert_not_called()
        replacement.assert_not_called()
        self.assertEqual(self.host.queue_panel.count(), 1)
        self.assertEqual(self.host.queue_manager.total_count, 2)


if __name__ == "__main__":
    unittest.main()
