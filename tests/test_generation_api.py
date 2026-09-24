import base64
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

from PIL import Image

from backends.base import GenerationResult, MediaArtifact
from core.generation_api import (
    GenerationApiManager,
    GenerationQueueFullError,
    GenerationValidationError,
    MAX_PERSISTED_REQUEST_BYTES,
    _canonical_endpoint_host,
)
from core.resource_coordinator import GenerationResourceCoordinator


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _image_b64(width=8, height=8, image_format="PNG"):
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format=image_format)
    return base64.b64encode(output.getvalue()).decode("ascii")


PRIMARY_IMAGE = base64.b64decode(_image_b64(image_format="PNG"))
SECOND_IMAGE = base64.b64decode(_image_b64(image_format="WEBP"))


class FakeBackend:
    def __init__(self, engine="webui", *, entered=None, release=None, calls=None):
        self.engine = engine
        self.entered = entered
        self.release = release
        self.calls = calls if calls is not None else []
        self.interrupted = threading.Event()

    def get_backend_type(self):
        return self.engine

    def test_connection(self):
        self.calls.append(("test_connection", self.engine))
        return True

    def interrupt(self):
        self.interrupted.set()
        if self.release is not None:
            self.release.set()

    def _generate(self, mode, model, payload, progress):
        self.calls.append((mode, model, payload))
        if self.entered is not None:
            self.entered.set()
        if progress is not None:
            progress(1, 2, b"preview")
        if self.release is not None:
            self.release.wait(3)
        if progress is not None:
            progress(2, 2, None)
        return GenerationResult(
            success=True,
            image_data=PRIMARY_IMAGE,
            info={"mode": mode, "model": model},
            artifacts=[
                MediaArtifact(kind="image", data=PRIMARY_IMAGE, filename="first.png", mime="image/png"),
                MediaArtifact(kind="image", data=SECOND_IMAGE, filename="second.webp", mime="image/webp"),
                MediaArtifact(kind="audio", data=b"audio", filename="sound.wav", mime="audio/wav"),
            ],
        )

    def txt2img(self, model, payload, progress=None):
        return self._generate("txt2img", model, payload, progress)

    def img2img(self, model, payload, progress=None):
        return self._generate("img2img", model, payload, progress)

    def generate_workflow(self, mode, workflow, model, payload, progress=None):
        self.calls.append(("workflow", mode, workflow, model, payload))
        if progress is not None:
            progress(1, 1, None)
        return GenerationResult(
            success=True,
            image_data=b"comfy-image",
            artifacts=[MediaArtifact(kind="image", data=b"comfy-image", filename="comfy.png", mime="image/png")],
        )

    def get_object_info(self):
        return {}

    def run_workflow(self, workflow, progress=None):
        self.calls.append(("run_workflow", workflow))
        return GenerationResult(success=True, image_data=b"krea-image")


class GenerationApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.config_path = root / "generation_api.json"
        self.results_path = root / "results"
        self.calls = []
        self.backends = {}

        def factory(profile):
            target_id = profile["id"]
            backend = self.backends.get(target_id)
            if backend is None:
                engine = profile.get("engine", "active")
                if engine == "active":
                    engine = "webui"
                backend = FakeBackend(engine, calls=self.calls)
                self.backends[target_id] = backend
            return backend

        self.manager = GenerationApiManager(
            config_path=self.config_path,
            storage_root=self.results_path,
            target_factory=factory,
            coordinator=GenerationResourceCoordinator(),
        )

    def tearDown(self):
        self.manager.shutdown()
        self.temp.cleanup()

    def _wait_completed(self, job, timeout=3):
        result = self.manager.wait(job["id"], timeout)
        self.assertEqual(result["state"], "completed", result)
        return result

    def test_defaults_are_disabled_loopback_and_token_is_redacted(self):
        private = self.manager.snapshot(include_secret=True)
        public = self.manager.snapshot(include_secret=False)
        self.assertFalse(private["config"]["enabled"])
        self.assertEqual(private["config"]["bindHost"], "127.0.0.1")
        self.assertGreaterEqual(len(private["config"]["token"]), 16)
        self.assertEqual(public["config"]["token"], "")
        persisted = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["token"], private["config"]["token"])

    def test_execute_contract_saves_profiles_rotates_token_and_tests_target(self):
        workflow = Path(self.temp.name) / "workflow.json"
        workflow.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
        state = self.manager.execute("save_config", {
            "enabled": False,
            "bindHost": "127.0.0.1",
            "port": _free_port(),
            "token": "a" * 24,
            "defaultTarget": "remote-comfy",
            "targets": [{
                "id": "remote-comfy",
                "name": "Remote Comfy",
                "type": "comfyui",
                "url": "http://192.0.2.10:8188",
                "workflowPath": str(workflow),
                "img2imgWorkflowPath": str(workflow),
            }],
        })
        target = state["state"]["config"]["targets"][0]
        self.assertEqual(target["engine"], "comfyui")
        self.assertEqual(target["type"], "comfyui")
        tested = self.manager.execute("test_target", {"targetId": "remote-comfy"})
        self.assertTrue(tested["ok"])
        draft = self.manager.execute("test_target", {"target": {
            "id": "unsaved-forge",
            "name": "Unsaved Forge",
            "engine": "webui",
            "url": "http://192.0.2.44:7860",
            "enabled": True,
        }})
        self.assertTrue(draft["ok"])
        self.assertEqual(draft["targetId"], "unsaved-forge")
        old_token = state["state"]["config"]["token"]
        rotated = self.manager.execute("rotate_token", {})
        self.assertNotEqual(rotated["state"]["config"]["token"], old_token)

    def test_request_accepts_only_approved_targets_without_urls_paths_or_workflows(self):
        with self.assertRaises(GenerationValidationError):
            self.manager.submit({"target": "missing", "mode": "txt2img", "payload": {}})
        for request in (
            {"mode": "txt2img", "url": "http://example.test", "payload": {}},
            {"mode": "txt2img", "workflow": {"1": {}}, "payload": {}},
            {"mode": "txt2img", "payload": {"workflow_path": "C:/secret.json"}},
            {"mode": "txt2img", "payload": {"alwayson_scripts": {"extension": {}}}},
            {"mode": "txt2img", "payload": {"script_name": "extension", "script_args": []}},
            {"mode": "txt2img", "payload": {"webhook_url": "http://example.test/callback"}},
            {"mode": "txt2img", "payload": {"source_url": "http://169.254.169.254/latest/meta-data"}},
            {"mode": "txt2img", "payload": {"input_image": "http://example.test/input.png"}},
            {"mode": "txt2img", "payload": {"controlnet_input_images": ["http://example.test/input.png"]}},
            {"mode": "img2img", "payload": {"init_images": ["C:/image.png"]}},
        ):
            with self.subTest(request=request):
                with self.assertRaises(GenerationValidationError):
                    self.manager.submit(request)

    def test_external_payload_resource_bounds_are_enforced(self):
        valid_image = _image_b64()
        oversized_side = _image_b64(16385, 1, "BMP")
        invalid_payloads = [
            {"prompt": "가" * 6000},  # UTF-8 byte length exceeds 16 KiB.
            {"width": 4104, "height": 512},
            {"width": 513, "height": 512},
            {"steps": 151},
            {"cfg_scale": 30.1},
            {"batch_size": 2, "n_iter": 3},
            {"n_iter": 1, "batch_count": 999999},
            {"batch_size": 2, "n_iter": 1, "batch_count": 4},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(GenerationValidationError):
                    self.manager.submit({"mode": "txt2img", "payload": payload})
        with self.assertRaises(GenerationValidationError):
            self.manager.submit({
                "mode": "img2img",
                "payload": {"init_images": [valid_image] * 5},
            })
        with self.assertRaises(GenerationValidationError):
            self.manager.submit({
                "mode": "img2img",
                "payload": {"init_images": [oversized_side]},
            })

    def test_disabled_target_cannot_be_default_or_receive_jobs_and_dot_id_is_valid(self):
        config = self.manager.snapshot(True)["config"]
        config.update({
            "defaultTarget": "active",
            "targets": [{
                "id": "office.forge-1", "name": "Office", "engine": "webui",
                "url": "http://192.0.2.77:7860", "enabled": False,
            }],
        })
        saved = self.manager.save_config(config)
        self.assertFalse(saved["config"]["targets"][0]["enabled"])
        with self.assertRaises(GenerationValidationError):
            self.manager.submit({"target": "office.forge-1", "mode": "txt2img", "payload": {}})
        config["defaultTarget"] = "office.forge-1"
        with self.assertRaises(GenerationValidationError):
            self.manager.save_config(config)

    def test_target_cannot_relay_back_into_same_api_server(self):
        port = _free_port()
        with self.assertRaises(GenerationValidationError):
            self.manager.save_config({
                "enabled": False,
                "bindHost": "127.0.0.1",
                "port": port,
                "token": "b" * 24,
                "defaultTarget": "loop",
                "targets": [{
                    "id": "loop",
                    "name": "Loop",
                    "engine": "webui",
                    "url": f"http://localhost:{port}",
                }],
            })

    def test_wildcard_bind_rejects_this_hosts_lan_address_and_uses_gpu_lease(self):
        port = _free_port()
        lan_ip = "192.168.219.128"
        with patch("core.generation_api._local_ipv4_addresses", return_value={"127.0.0.1", lan_ip}):
            with self.assertRaises(GenerationValidationError):
                self.manager.save_config({
                    "enabled": False,
                    "bindHost": "0.0.0.0",
                    "port": port,
                    "token": "l" * 24,
                    "defaultTarget": "active",
                    "targets": [{
                        "id": "self-lan", "name": "Self LAN", "engine": "webui",
                        "url": f"http://{lan_ip}:{port}",
                    }],
                })
            self.assertTrue(self.manager._uses_local_generation_lease({
                "id": "local-forge", "engine": "webui", "url": f"http://{lan_ip}:7860",
            }))

    def test_app_startup_preference_is_independent_from_manual_start_stop(self):
        port = _free_port()
        config = self.manager.snapshot(True)["config"]
        config.update({"enabled": True, "port": port})
        saved = self.manager.save_config(config)
        self.assertTrue(saved["config"]["enabled"])
        self.assertFalse(saved["running"])

        started = self.manager.execute("start", {})["state"]
        self.assertTrue(started["running"])
        self.assertTrue(started["config"]["enabled"])
        stopped = self.manager.execute("stop", {})["state"]
        self.assertFalse(stopped["running"])
        self.assertTrue(stopped["config"]["enabled"])

        self.assertTrue(self.manager.start_if_enabled()["running"])
        config = self.manager.snapshot(True)["config"]
        config["enabled"] = False
        saved = self.manager.save_config(config)
        self.assertTrue(saved["running"])
        self.assertFalse(saved["config"]["enabled"])
        self.manager.execute("stop", {})

    def test_active_backend_is_snapshotted_when_job_is_submitted(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        second_entered = threading.Event()
        first_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=[])
        submitted_backend = FakeBackend("comfyui", entered=second_entered, calls=[])
        later_backend = FakeBackend("webui", calls=[])
        self.backends["active"] = first_backend
        first = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "first"}})
        self.assertTrue(first_entered.wait(1))
        self.backends["active"] = submitted_backend
        second = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "second"}})
        self.backends["active"] = later_backend
        first_release.set()
        self._wait_completed(first)
        self.assertTrue(second_entered.wait(1))
        self._wait_completed(second)
        self.assertEqual(len(submitted_backend.calls), 1)
        self.assertEqual(later_backend.calls, [])

    def test_cancel_interrupt_cannot_spill_into_next_job_on_same_endpoint(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        interrupt_entered = threading.Event()
        allow_interrupt_return = threading.Event()
        second_entered = threading.Event()
        first_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=[])
        second_backend = FakeBackend("webui", entered=second_entered, calls=[])

        def slow_interrupt():
            first_backend.interrupted.set()
            first_release.set()
            interrupt_entered.set()
            allow_interrupt_return.wait(2)

        first_backend.interrupt = slow_interrupt
        self.backends["active"] = first_backend
        first = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "first"}})
        self.assertTrue(first_entered.wait(1))
        self.backends["active"] = second_backend
        second = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "second"}})
        cancel_thread = threading.Thread(target=self.manager.cancel, args=(first["id"],), daemon=True)
        cancel_thread.start()
        self.assertTrue(interrupt_entered.wait(1))
        self.assertFalse(second_entered.wait(0.15))
        allow_interrupt_return.set()
        cancel_thread.join(2)
        self.assertFalse(cancel_thread.is_alive())
        self.assertEqual(self.manager.wait(first["id"], 3)["state"], "cancelled")
        self.assertTrue(second_entered.wait(1))
        self._wait_completed(second)

    def test_job_persists_every_artifact_and_can_be_loaded_after_restart(self):
        job = self.manager.submit({
            "mode": "txt2img",
            "model": "model.safetensors",
            "payload": {"prompt": "test"},
        })
        result = self._wait_completed(job)
        self.assertEqual(len(result["artifacts"]), 3)
        self.assertEqual(self.manager.artifact(job["id"], 0)[0], PRIMARY_IMAGE)
        self.assertEqual(self.manager.artifact(job["id"], 1)[0], SECOND_IMAGE)
        self.assertEqual(self.manager.artifact(job["id"], 2)[0], b"audio")
        manifest = self.results_path / job["id"] / "job.json"
        self.assertTrue(manifest.is_file())

        reloaded = GenerationApiManager(
            config_path=self.config_path,
            storage_root=self.results_path,
            target_factory=lambda profile: FakeBackend(),
            coordinator=GenerationResourceCoordinator(),
        )
        try:
            old = reloaded.inspect(job["id"])
            self.assertEqual(old["state"], "completed")
            self.assertEqual(reloaded.artifact(job["id"], 1)[0], SECOND_IMAGE)
        finally:
            reloaded.shutdown()

    def test_completed_job_persists_a_bounded_large_request_summary(self):
        payload = {
            "prompt": "large request",
            **{f"custom_{index}": "x" * 1000 for index in range(300)},
        }
        job = self.manager.submit({"mode": "txt2img", "payload": payload})
        self._wait_completed(job)

        manifest_path = self.results_path / job["id"] / "job.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        request_summary = manifest["request"]
        encoded_summary = json.dumps(
            request_summary,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertLessEqual(len(encoded_summary), MAX_PERSISTED_REQUEST_BYTES)
        self.assertIn("_truncated", request_summary["payload"])

    def test_completed_img2img_job_releases_internal_payload(self):
        job = self.manager.submit({
            "mode": "img2img",
            "payload": {
                "prompt": "convert this image",
                "init_images": [_image_b64()],
            },
        })
        self._wait_completed(job)

        with self.manager._lock:
            self.assertEqual(self.manager._jobs[job["id"]]["_payload"], {})

    def test_artifact_persistence_does_not_hold_global_manager_lock(self):
        entered = threading.Event()
        release = threading.Event()
        original = self.manager._persist_result_artifacts

        def slow_persist(job, result):
            entered.set()
            release.wait(2)
            return original(job, result)

        self.manager._persist_result_artifacts = slow_persist
        job = self.manager.submit({"mode": "txt2img", "payload": {}})
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        snapshot = self.manager.inspect(job["id"])
        self.assertLess(time.monotonic() - started, 0.25)
        self.assertEqual(snapshot["state"], "running")
        release.set()
        self._wait_completed(job)

    def test_artifact_cap_preflight_leaves_no_partial_files(self):
        backend = FakeBackend("webui", calls=self.calls)
        backend.txt2img = lambda _model, _payload, _progress=None: GenerationResult(
            success=True,
            artifacts=[
                MediaArtifact(kind="file", data=b"1234", filename="first.bin"),
                MediaArtifact(kind="file", data=b"5", filename="second.bin"),
            ],
        )
        self.backends["active"] = backend

        with patch("core.generation_api.MAX_ARTIFACT_BYTES_PER_JOB", 4):
            job = self.manager.submit({"mode": "txt2img", "payload": {}})
            result = self.manager.wait(job["id"], 3)

        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["artifacts"], [])
        job_files = sorted(
            path.name for path in (self.results_path / job["id"]).iterdir()
        )
        self.assertEqual(job_files, ["job.json"])

    def test_artifact_metadata_redacts_path_and_url_keys_and_values(self):
        backend = FakeBackend("webui", calls=self.calls)
        backend.txt2img = lambda _model, _payload, _progress=None: GenerationResult(
            success=True,
            artifacts=[MediaArtifact(
                kind="image",
                data=PRIMARY_IMAGE,
                filename="result.png",
                mime="image/png",
                metadata={
                    "source_url": "https://private.example.test/result.png",
                    "local_path": "C:\\private\\results\\result.png",
                    "https://private.example.test/key": "C:\\private\\metadata.json",
                },
            )],
        )
        self.backends["active"] = backend

        job = self.manager.submit({"mode": "txt2img", "payload": {}})
        result = self._wait_completed(job)
        manifest = json.loads(
            (self.results_path / job["id"] / "job.json").read_text(encoding="utf-8")
        )

        for public_value in (result["artifacts"][0]["metadata"], manifest["artifacts"][0]["metadata"]):
            encoded = json.dumps(public_value, ensure_ascii=False)
            self.assertNotIn("private.example.test", encoded)
            self.assertNotIn("C:\\\\private", encoded)
            self.assertIn("[backend endpoint]", encoded)
            self.assertIn("[local path]", encoded)

    def test_queue_is_bounded_and_queued_job_can_be_cancelled(self):
        entered = threading.Event()
        release = threading.Event()
        blocking = FakeBackend("webui", entered=entered, release=release, calls=self.calls)
        self.backends["active"] = blocking
        self.manager.save_config({
            **self.manager.snapshot(True)["config"],
            "maxQueue": 2,
        })
        first = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "one"}})
        self.assertTrue(entered.wait(1))
        second = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "two"}})
        with self.assertRaises(GenerationQueueFullError):
            self.manager.submit({"mode": "txt2img", "payload": {"prompt": "three"}})
        cancelled = self.manager.cancel(second["id"])
        self.assertEqual(cancelled["state"], "cancelled")
        with self.assertRaises(GenerationQueueFullError):
            self.manager.submit({"mode": "txt2img", "payload": {"prompt": "still bounded"}})
        release.set()
        self._wait_completed(first)

    def test_cancelled_queue_tombstone_survives_trim_until_worker_releases_slot(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        blocking = FakeBackend("webui", entered=entered, release=release, calls=self.calls)
        self.backends["active"] = blocking
        self.manager.save_config({
            **self.manager.snapshot(True)["config"],
            "maxQueue": 3,
        })

        first = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "first"}})
        self.assertTrue(entered.wait(1))
        tombstone = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "cancel me"}})
        self.assertEqual(self.manager.cancel(tombstone["id"])["state"], "cancelled")

        with patch("core.generation_api.MAX_RECENT_JOBS", 2):
            third = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "third"}})
        self.assertEqual(self.manager.inspect(tombstone["id"])["state"], "cancelled")

        release.set()
        self._wait_completed(first)
        self._wait_completed(third)
        with self.manager._lock:
            self.assertEqual(self.manager._active_slots, 0)

        self.manager.save_config({
            **self.manager.snapshot(True)["config"],
            "maxQueue": 1,
        })
        final = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "slot reused"}})
        self._wait_completed(final)

    def test_running_cancel_interrupts_backend_and_finishes_cancelled(self):
        entered = threading.Event()
        release = threading.Event()
        blocking = FakeBackend("webui", entered=entered, release=release, calls=self.calls)
        self.backends["active"] = blocking
        job = self.manager.submit({"mode": "txt2img", "payload": {}})
        self.assertTrue(entered.wait(1))
        cancelling = self.manager.cancel(job["id"])
        self.assertTrue(cancelling["cancelRequested"])
        finished = self.manager.wait(job["id"], 3)
        self.assertEqual(finished["state"], "cancelled")
        self.assertTrue(blocking.interrupted.is_set())

    def test_cancel_during_backend_type_probe_never_dispatches_generation(self):
        type_probe_entered = threading.Event()
        release_type_probe = threading.Event()

        class SlowTypeBackend(FakeBackend):
            def get_backend_type(self):
                type_probe_entered.set()
                release_type_probe.wait(2)
                return self.engine

        backend = SlowTypeBackend("webui", calls=[])
        self.backends["active"] = backend
        job = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "race"}})
        self.assertTrue(type_probe_entered.wait(1))

        self.manager.cancel(job["id"])
        release_type_probe.set()
        finished = self.manager.wait(job["id"], 3)

        self.assertEqual(finished["state"], "cancelled")
        self.assertEqual(backend.calls, [])

    def test_cancel_while_waiting_for_local_gpu_lease_does_not_interrupt_owner(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        first_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=self.calls)
        second_backend = FakeBackend("comfyui", calls=self.calls)
        workflow = Path(self.temp.name) / "local-comfy.json"
        workflow.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
        config = self.manager.snapshot(True)["config"]
        config.update({
            "targets": [
                {"id": "local-forge", "name": "Forge", "engine": "webui", "url": "http://127.0.0.1:7860"},
                {
                    "id": "local-comfy", "name": "Comfy", "engine": "comfyui", "url": "http://127.0.0.1:8188",
                    "workflowPath": str(workflow), "img2imgWorkflowPath": str(workflow),
                },
            ],
        })
        self.manager.save_config(config)
        self.backends["local-forge"] = first_backend
        self.backends["local-comfy"] = second_backend
        owner = self.manager.submit({"target": "local-forge", "mode": "txt2img", "payload": {}})
        self.assertTrue(first_entered.wait(1))
        waiting = self.manager.submit({"target": "local-comfy", "mode": "txt2img", "payload": {}})
        deadline = time.monotonic() + 1
        while self.manager.inspect(waiting["id"])["state"] != "running" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.manager.cancel(waiting["id"])
        self.assertFalse(second_backend.interrupted.is_set())
        self.assertFalse(first_backend.interrupted.is_set())
        first_release.set()
        self.assertEqual(self.manager.wait(owner["id"], 3)["state"], "completed")
        self.assertEqual(self.manager.wait(waiting["id"], 3)["state"], "cancelled")

    def test_alias_profiles_for_same_endpoint_share_one_serial_lane(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        second_entered = threading.Event()
        first_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=self.calls)
        second_backend = FakeBackend("webui", entered=second_entered, calls=self.calls)
        config = self.manager.snapshot(True)["config"]
        config.update({
            "targets": [
                {"id": "alias-a", "name": "Alias A", "engine": "webui", "url": "http://192.0.2.88:7860"},
                {"id": "alias-b", "name": "Alias B", "engine": "webui", "url": "http://192.0.2.88:7860"},
            ],
        })
        self.manager.save_config(config)
        self.backends["alias-a"] = first_backend
        self.backends["alias-b"] = second_backend
        first = self.manager.submit({"target": "alias-a", "mode": "txt2img", "payload": {}})
        self.assertTrue(first_entered.wait(1))
        second = self.manager.submit({"target": "alias-b", "mode": "txt2img", "payload": {}})
        self.assertFalse(second_entered.wait(0.15))
        first_release.set()
        self._wait_completed(first)
        self.assertTrue(second_entered.wait(1))
        self._wait_completed(second)

    def test_active_and_named_alias_of_same_endpoint_share_one_serial_lane(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        second_entered = threading.Event()
        active_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=[])
        active_backend.api_url = "http://192.0.2.89:7860/"
        named_backend = FakeBackend("webui", entered=second_entered, calls=[])
        named_backend.api_url = "http://192.0.2.89:7860"
        config = self.manager.snapshot(True)["config"]
        config["targets"] = [{
            "id": "same-forge", "name": "Same Forge", "engine": "webui",
            "url": "http://192.0.2.89:7860",
        }]
        self.manager.save_config(config)
        self.backends["active"] = active_backend
        self.backends["same-forge"] = named_backend
        first = self.manager.submit({"target": "active", "mode": "txt2img", "payload": {}})
        self.assertTrue(first_entered.wait(1))
        second = self.manager.submit({"target": "same-forge", "mode": "txt2img", "payload": {}})
        self.assertFalse(second_entered.wait(0.15))
        first_release.set()
        self._wait_completed(first)
        self.assertTrue(second_entered.wait(1))
        self._wait_completed(second)

    def test_dns_and_ip_aliases_of_same_endpoint_share_one_serial_lane(self):
        first_entered = threading.Event()
        first_release = threading.Event()
        second_entered = threading.Event()
        active_backend = FakeBackend("webui", entered=first_entered, release=first_release, calls=[])
        active_backend.api_url = "http://forge-api.test:7860"
        named_backend = FakeBackend("webui", entered=second_entered, calls=[])
        config = self.manager.snapshot(True)["config"]
        config["targets"] = [{
            "id": "forge-ip", "name": "Forge IP", "engine": "webui",
            "url": "http://192.0.2.90:7860",
        }]
        self.manager.save_config(config)
        self.backends["active"] = active_backend
        self.backends["forge-ip"] = named_backend
        _canonical_endpoint_host.cache_clear()
        resolved = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.90", 0))]
        with patch("core.generation_api.socket.getaddrinfo", return_value=resolved):
            first = self.manager.submit({"target": "active", "mode": "txt2img", "payload": {}})
        self.assertTrue(first_entered.wait(1))
        second = self.manager.submit({"target": "forge-ip", "mode": "txt2img", "payload": {}})
        self.assertFalse(second_entered.wait(0.15))
        first_release.set()
        self._wait_completed(first)
        self.assertTrue(second_entered.wait(1))
        self._wait_completed(second)
        _canonical_endpoint_host.cache_clear()

    def test_backend_error_and_metadata_redact_endpoints_and_local_paths(self):
        backend = FakeBackend("webui", calls=self.calls)

        def fail(_model, _payload, _progress=None):
            return GenerationResult(
                success=False,
                error="POST http://127.0.0.1:7860 failed at C:\\models\\private.safetensors",
                info={"endpoint": "http://127.0.0.1:7860", "output": "C:\\secret\\out.png"},
            )

        backend.txt2img = fail
        self.backends["active"] = backend
        job = self.manager.submit({"mode": "txt2img", "payload": {}})
        result = self.manager.wait(job["id"], 3)
        self.assertEqual(result["state"], "failed")
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("127.0.0.1:7860", encoded)
        self.assertNotIn("private.safetensors", encoded)
        self.assertNotIn("secret", encoded)

    def test_malformed_manifest_fields_do_not_break_manager_initialization(self):
        self.manager.shutdown()
        job_id = "f" * 32
        folder = self.results_path / job_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(json.dumps({
            "id": job_id,
            "state": "completed",
            "progress": {"bad": True},
            "currentStep": "not-a-number",
            "totalSteps": None,
            "artifacts": [{"file": "result.png", "size": {"bad": True}}],
        }), encoding="utf-8")
        self.manager = GenerationApiManager(
            config_path=self.config_path,
            storage_root=self.results_path,
            target_factory=lambda profile: FakeBackend(),
            coordinator=GenerationResourceCoordinator(),
        )
        loaded = self.manager.inspect(job_id)
        self.assertEqual(loaded["progress"], 0.0)
        self.assertEqual(loaded["currentStep"], 0)

    def _reload_with_config(self, values):
        self.manager.shutdown()
        self.config_path.write_text(json.dumps(values), encoding="utf-8")
        self.manager = GenerationApiManager(
            config_path=self.config_path,
            storage_root=self.results_path,
            target_factory=lambda profile: FakeBackend(),
            coordinator=GenerationResourceCoordinator(),
        )
        return self.manager.snapshot(True)["config"]

    def test_default_port_stays_outside_managed_runtime_port_ranges(self):
        from core.backend_runtime import ENGINE_DEFINITIONS
        from core.generation_api import DEFAULT_PORT

        for engine, definition in ENGINE_DEFINITIONS.items():
            with self.subTest(engine=engine):
                self.assertNotIn(DEFAULT_PORT, range(definition.preferred_port, definition.preferred_port + 50))
        self.assertEqual(self.manager.snapshot(True)["config"]["port"], DEFAULT_PORT)

    def test_schema1_untouched_default_port_moves_off_the_forge_port(self):
        from core.generation_api import DEFAULT_PORT, SCHEMA_VERSION

        config = self._reload_with_config({
            "schemaVersion": 1, "enabled": False, "port": 17860, "token": "m" * 24,
        })
        self.assertEqual(config["port"], DEFAULT_PORT)
        self.assertEqual(config["token"], "m" * 24)
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual((saved["schemaVersion"], saved["port"]), (SCHEMA_VERSION, DEFAULT_PORT))

    def test_schema1_enabled_or_custom_ports_are_preserved(self):
        for values in (
            {"schemaVersion": 1, "enabled": True, "port": 17860, "token": "e" * 24},
            {"schemaVersion": 1, "enabled": False, "port": 18000, "token": "c" * 24},
        ):
            with self.subTest(values=values):
                config = self._reload_with_config(values)
                self.assertEqual(config["port"], values["port"])
                self.assertEqual(config["enabled"], values["enabled"])

    def test_schema2_explicit_legacy_port_is_not_migrated_again(self):
        from core.generation_api import SCHEMA_VERSION

        config = self._reload_with_config({
            "schemaVersion": SCHEMA_VERSION, "enabled": False, "port": 17860, "token": "s" * 24,
        })
        self.assertEqual(config["port"], 17860)

    def test_recovered_manifests_do_not_resolve_target_hostnames_at_startup(self):
        config = self.manager.snapshot(True)["config"]
        config["targets"] = [{
            "id": "remote-forge", "name": "Remote Forge", "engine": "webui",
            "url": "http://remote-forge.invalid:7860",
        }]
        self.manager.save_config(config)
        self.manager.shutdown()
        job_ids = []
        for index, state in enumerate(("completed", "running", "queued")):
            job_id = f"{index:x}" * 32
            job_ids.append((job_id, state))
            folder = self.results_path / job_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "job.json").write_text(json.dumps({
                "id": job_id, "state": state, "target": "remote-forge",
            }), encoding="utf-8")

        def no_dns(*_args, **_kwargs):
            raise AssertionError("manifest recovery must not resolve target hostnames")

        with patch("core.generation_api.socket.getaddrinfo", side_effect=no_dns):
            self.manager = GenerationApiManager(
                config_path=self.config_path,
                storage_root=self.results_path,
                target_factory=lambda profile: FakeBackend(),
                coordinator=GenerationResourceCoordinator(),
            )
            for job_id, state in job_ids:
                loaded = self.manager.inspect(job_id)
                # 비종료 상태는 복구 시 failed 로 닫히므로 직렬화 키가 다시 쓰이지 않는다.
                self.assertEqual(loaded["state"], "completed" if state == "completed" else "failed")
                self.assertEqual(loaded["target"], "remote-forge")
                # 이미 끝난 작업의 취소는 직렬화 키를 읽지 않고 그대로 돌아온다.
                self.assertEqual(self.manager.cancel(job_id)["state"], loaded["state"])

    def test_named_comfy_uses_only_saved_workflow_profile(self):
        workflow = Path(self.temp.name) / "api-workflow.json"
        workflow_data = {"1": {"class_type": "SaveImage", "inputs": {}}}
        workflow.write_text(json.dumps(workflow_data), encoding="utf-8")
        config = self.manager.snapshot(True)["config"]
        config.update({
            "defaultTarget": "comfy",
            "targets": [{
                "id": "comfy",
                "name": "Comfy",
                "engine": "comfyui",
                "url": "http://192.0.2.25:8188",
                "workflowPath": str(workflow),
                "img2imgWorkflowPath": str(workflow),
            }],
        })
        self.manager.save_config(config)
        backend = FakeBackend("comfyui", calls=self.calls)
        self.backends["comfy"] = backend
        job = self.manager.submit({
            "target": "comfy",
            "mode": "txt2img",
            "model": "checkpoint.safetensors",
            "payload": {"prompt": "remote"},
        })
        self._wait_completed(job)
        workflow_call = next(call for call in self.calls if call[0] == "workflow")
        self.assertEqual(workflow_call[1], "txt2img")
        self.assertEqual(workflow_call[2], workflow_data)
        self.assertEqual(workflow_call[3], "checkpoint.safetensors")

    def test_krea2_dispatches_only_through_comfy_backend(self):
        with self.assertRaises(GenerationValidationError):
            self.manager.save_config({
                **self.manager.snapshot(True)["config"],
                "defaultTarget": "forge",
                "targets": [{
                    "id": "forge", "name": "Forge", "engine": "webui",
                    "url": "http://192.0.2.20:7860",
                }],
            })
            self.manager.submit({"target": "forge", "family": "krea2", "mode": "txt2img", "payload": {}})

        # Restore a Comfy target. Krea2 does not consume the standard workflow
        # profile, but target configuration remains explicit and approved.
        workflow = Path(self.temp.name) / "krea-fallback.json"
        workflow.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
        config = self.manager.snapshot(True)["config"]
        config.update({
            "defaultTarget": "comfy",
            "targets": [{
                "id": "comfy", "name": "Comfy", "engine": "comfyui",
                "url": "http://192.0.2.30:8188",
                "workflowPath": str(workflow), "img2imgWorkflowPath": str(workflow),
            }],
        })
        self.manager.save_config(config)
        self.backends["comfy"] = FakeBackend("comfyui", calls=self.calls)
        with self.assertRaises(GenerationValidationError):
            self.manager.submit({
                "target": "comfy", "family": "krea2", "mode": "txt2img",
                "payload": {"prompt": "krea batch", "batch_size": 2},
            })
        fake_result = GenerationResult(success=True, image_data=b"krea")
        with patch("core.krea2_generation.run_krea2_generation", return_value=fake_result) as runner:
            job = self.manager.submit({
                "target": "comfy", "request_family": "krea2", "mode": "t2i",
                "payload": {"prompt": "krea"},
            })
            self._wait_completed(job)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[1], "t2i")


class GenerationApiHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.backend = FakeBackend("webui")
        self.manager = GenerationApiManager(
            config_path=root / "generation_api.json",
            storage_root=root / "results",
            target_factory=lambda profile: self.backend,
            coordinator=GenerationResourceCoordinator(),
            max_body_bytes=1024,
        )
        self.token = "http-test-token-0123456789"
        self.port = _free_port()
        self.manager.save_config({
            "enabled": True,
            "bindHost": "127.0.0.1",
            "port": self.port,
            "token": self.token,
            "defaultTarget": "active",
            "maxQueue": 8,
            "targets": [],
        })
        self.manager.start(persist_enabled=False)

    def tearDown(self):
        self.manager.shutdown()
        self.temp.cleanup()

    def _request(self, method, path, body=None, *, auth=True, extra_headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = dict(extra_headers or {})
        if auth:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        content_type = response.getheader("Content-Type", "")
        connection.close()
        if content_type.startswith("application/json"):
            payload = json.loads(payload.decode("utf-8"))
        return response.status, payload, dict(response.getheaders())

    def test_health_is_public_but_generation_requires_bearer_token(self):
        status, payload, _headers = self._request("GET", "/api/v1/health", auth=False)
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        status, payload, headers = self._request("GET", "/api/v1/targets", auth=False)
        self.assertEqual(status, 401)
        self.assertIn("Bearer", headers["WWW-Authenticate"])

    def test_native_submit_poll_and_authenticated_artifact_download(self):
        status, job, _headers = self._request(
            "POST",
            "/api/v1/generations?wait=3",
            {"mode": "txt2img", "model": "api-model", "payload": {"prompt": "hello"}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(job["state"], "completed")
        self.assertEqual(len(job["artifacts"]), 3)
        status, data, headers = self._request(
            "GET", f"/api/v1/generations/{job['id']}/artifacts/1"
        )
        self.assertEqual(status, 200)
        self.assertEqual(data, SECOND_IMAGE)
        self.assertEqual(headers["Content-Type"], "image/webp")

    def test_a1111_txt2img_response_progress_and_interrupt_routes(self):
        status, payload, _headers = self._request(
            "POST", "/sdapi/v1/txt2img", {"prompt": "compat", "steps": 2}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["images"]), 2)
        self.assertEqual(base64.b64decode(payload["images"][0]), PRIMARY_IMAGE)
        self.assertEqual(json.loads(payload["info"])["mode"], "txt2img")
        status, progress, _headers = self._request("GET", "/sdapi/v1/progress")
        self.assertEqual(status, 200)
        self.assertIn("progress", progress)
        status, interrupted, _headers = self._request("POST", "/sdapi/v1/interrupt")
        self.assertEqual(status, 200)
        self.assertEqual(interrupted["cancelled"], 0)

    def test_delete_cancels_a_queued_job(self):
        entered = threading.Event()
        release = threading.Event()
        self.backend.entered = entered
        self.backend.release = release
        first = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "first"}})
        self.assertTrue(entered.wait(1))
        second = self.manager.submit({"mode": "txt2img", "payload": {"prompt": "second"}})
        status, cancelled, _headers = self._request(
            "DELETE", f"/api/v1/generations/{second['id']}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["state"], "cancelled")
        release.set()
        self.assertEqual(self.manager.wait(first["id"], 3)["state"], "completed")

    def test_request_body_limit_returns_413_without_dispatch(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        body = b"x" * 2048
        connection.request(
            "POST",
            "/api/v1/generations",
            body=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 413)

    def test_invalid_wait_query_returns_400_without_submitting_a_job(self):
        for wait_value in ("NaN", "invalid"):
            with self.subTest(wait=wait_value):
                jobs_before = [job["id"] for job in self.manager.list_jobs()]
                calls_before = len(self.backend.calls)
                status, payload, _headers = self._request(
                    "POST",
                    f"/api/v1/generations?wait={wait_value}",
                    {"mode": "txt2img", "payload": {"prompt": "must not run"}},
                )

                self.assertEqual(status, 400)
                self.assertIn("wait", payload["error"])
                self.assertEqual(
                    [job["id"] for job in self.manager.list_jobs()],
                    jobs_before,
                )
                self.assertEqual(len(self.backend.calls), calls_before)

    def test_artifact_mime_cannot_inject_response_headers(self):
        def malicious(_model, _payload, _progress=None):
            return GenerationResult(
                success=True,
                artifacts=[MediaArtifact(
                    kind="image",
                    data=b"unsafe-header-test",
                    filename="result.png",
                    mime="image/png\r\nX-Injected: yes",
                )],
            )

        self.backend.txt2img = malicious
        status, job, _headers = self._request(
            "POST", "/api/v1/generations?wait=3", {"mode": "txt2img", "payload": {}}
        )
        self.assertEqual(status, 200)
        self.assertEqual(job["artifacts"][0]["kind"], "file")
        status, data, headers = self._request(
            "GET", f"/api/v1/generations/{job['id']}/artifacts/0"
        )
        self.assertEqual(status, 200)
        self.assertEqual(data, b"unsafe-header-test")
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertTrue(headers["Content-Disposition"].startswith("attachment;"))
        self.assertNotIn("X-Injected", headers)

    def test_http_artifact_download_streams_without_path_read_bytes(self):
        status, job, _headers = self._request(
            "POST", "/api/v1/generations?wait=3", {"mode": "txt2img", "payload": {}}
        )
        self.assertEqual(status, 200)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must stream")):
            status, data, _headers = self._request(
                "GET", f"/api/v1/generations/{job['id']}/artifacts/0"
            )
        self.assertEqual(status, 200)
        self.assertEqual(data, PRIMARY_IMAGE)

    def test_unexpected_http_error_does_not_expose_local_path(self):
        status, job, _headers = self._request(
            "POST", "/api/v1/generations?wait=3", {"mode": "txt2img", "payload": {}}
        )
        self.assertEqual(status, 200)
        with patch.object(
            self.manager,
            "artifact_path",
            side_effect=PermissionError("C:\\private\\models\\secret.safetensors"),
        ):
            status, payload, _headers = self._request(
                "GET", f"/api/v1/generations/{job['id']}/artifacts/0"
            )
        self.assertEqual(status, 500)
        self.assertEqual(payload["error"], "내부 서버 오류")
        self.assertNotIn("private", json.dumps(payload))

    def test_a1111_compat_returns_bad_gateway_when_backend_has_no_image(self):
        self.backend.txt2img = lambda _model, _payload, _progress=None: GenerationResult(
            success=True, info={"message": "no artifact"}
        )
        status, payload, _headers = self._request(
            "POST", "/sdapi/v1/txt2img", {"prompt": "no image"}
        )
        self.assertEqual(status, 502)
        self.assertIn("이미지 결과", payload["error"])


if __name__ == "__main__":
    unittest.main()
