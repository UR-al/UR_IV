import unittest
from contextlib import contextmanager
from unittest.mock import patch

from backends.base import GenerationResult
from workers.generation_worker import GenerationFlowWorker, Img2ImgFlowWorker


class _Coordinator:
    @contextmanager
    def reserve(self, *args, **kwargs):
        yield


class _Backend:
    """AbstractBackend 시그니처(cancel_check 포함)를 따르는 fake."""

    def __init__(self):
        self.txt_payload = None
        self.i2i_payload = None
        self.cancel_checks = []

    def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
        self.txt_payload = dict(payload)
        self.cancel_checks.append(cancel_check)
        return GenerationResult(success=True, image_data=b"standard-t2i", info={})

    def img2img(self, model, payload, progress_callback=None, cancel_check=None):
        self.i2i_payload = dict(payload)
        self.cancel_checks.append(cancel_check)
        return GenerationResult(success=True, image_data=b"standard-i2i", info={})


class _LegacyBackend:
    """cancel_check 를 모르는 옛 어댑터 — 넘기면 TypeError 가 나야 정상."""

    def __init__(self):
        self.calls = 0

    def txt2img(self, model, payload, progress_callback=None):
        self.calls += 1
        return GenerationResult(success=True, image_data=b"legacy", info={})


class Krea2WorkerRoutingTests(unittest.TestCase):
    def _run_worker(self, worker):
        emitted = []
        worker.finished.connect(lambda result, info: emitted.append((result, info)))
        worker.run()
        self.assertEqual(len(emitted), 1)
        return emitted[0]

    def test_standard_marker_is_consumed_before_txt2img_backend(self):
        backend = _Backend()
        worker = GenerationFlowWorker(
            "checkpoint.safetensors",
            {"prompt": "test", "_generation_family": "standard"},
        )
        with (
            patch("workers.generation_worker.get_backend", return_value=backend),
            patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
        ):
            result, _info = self._run_worker(worker)
        self.assertEqual(result, b"standard-t2i")
        self.assertNotIn("_generation_family", backend.txt_payload)

    def test_standard_workers_pass_live_cancel_check_to_backend(self):
        # 감사 #31: 모델 전환 중·발송 직후 취소가 사라지지 않게 백엔드가 직접 확인할 수 있어야 한다
        for worker_cls, payload, calls_attr in (
            (GenerationFlowWorker, {"prompt": "test"}, "txt_payload"),
            (Img2ImgFlowWorker, {"init_images": ["abc"]}, "i2i_payload"),
        ):
            with self.subTest(worker=worker_cls.__name__):
                backend = _Backend()
                worker = worker_cls("checkpoint.safetensors", payload)
                with (
                    patch("workers.generation_worker.get_backend", return_value=backend),
                    patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
                ):
                    self._run_worker(worker)
                self.assertIsNotNone(getattr(backend, calls_attr))
                cancel_check = backend.cancel_checks[0]
                self.assertTrue(callable(cancel_check))
                self.assertFalse(cancel_check())
                worker._cancelled = True   # cancel() 은 interrupt 스레드도 띄우므로 플래그만
                self.assertTrue(cancel_check())

    def test_cancel_reported_by_backend_is_emitted_as_cancelled(self):
        worker = GenerationFlowWorker("checkpoint.safetensors", {"prompt": "test"})

        class _SwitchingBackend(_Backend):
            def txt2img(inner, model, payload, progress_callback=None, cancel_check=None):
                worker._cancelled = True   # 체크포인트 전환 중에 사용자가 취소
                self.assertTrue(cancel_check())
                return GenerationResult(success=False, error="사용자가 작업을 취소했습니다")

        with (
            patch("workers.generation_worker.get_backend", return_value=_SwitchingBackend()),
            patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
        ):
            result, info = self._run_worker(worker)
        self.assertEqual(result, "생성 취소됨")
        self.assertTrue(info.get("cancelled"))

    def test_legacy_adapter_without_cancel_check_still_runs(self):
        backend = _LegacyBackend()
        worker = GenerationFlowWorker("checkpoint.safetensors", {"prompt": "test"})
        with (
            patch("workers.generation_worker.get_backend", return_value=backend),
            patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
        ):
            result, _info = self._run_worker(worker)
        self.assertEqual(result, b"legacy")
        self.assertEqual(backend.calls, 1)

    def test_krea_t2i_routes_to_family_runner(self):
        backend = _Backend()
        calls = []

        def _run(adapter, operation, payload, progress_callback=None, cancel_check=None):
            calls.append((adapter, operation, dict(payload), cancel_check))
            return GenerationResult(success=True, image_data=b"krea-t2i", info={"seed": 1})

        worker = GenerationFlowWorker(
            "ignored-checkpoint",
            {"prompt": "test", "_generation_family": "krea2"},
        )
        with (
            patch("workers.generation_worker.get_backend", return_value=backend),
            patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
            patch("core.krea2_generation.run_krea2_generation", side_effect=_run),
        ):
            result, info = self._run_worker(worker)
        self.assertEqual(result, b"krea-t2i")
        self.assertEqual(info["seed"], 1)
        self.assertEqual(calls[0][1], "t2i")
        self.assertNotIn("_generation_family", calls[0][2])
        self.assertTrue(callable(calls[0][3]))   # Krea2 도 취소 확인을 받는다
        self.assertIsNone(backend.txt_payload)

    def test_krea_i2i_routes_to_family_runner(self):
        backend = _Backend()
        calls = []

        def _run(adapter, operation, payload, progress_callback=None, cancel_check=None):
            calls.append((adapter, operation, dict(payload), cancel_check))
            return GenerationResult(success=True, image_data=b"krea-i2i", info={})

        worker = Img2ImgFlowWorker(
            "ignored-checkpoint",
            {"init_images": ["abc"], "_generation_family": "krea2"},
        )
        with (
            patch("workers.generation_worker.get_backend", return_value=backend),
            patch("workers.generation_worker.get_generation_coordinator", return_value=_Coordinator()),
            patch("core.krea2_generation.run_krea2_generation", side_effect=_run),
        ):
            result, _info = self._run_worker(worker)
        self.assertEqual(result, b"krea-i2i")
        self.assertEqual(calls[0][1], "i2i")
        self.assertTrue(callable(calls[0][3]))
        self.assertIsNone(backend.i2i_payload)


if __name__ == "__main__":
    unittest.main()
