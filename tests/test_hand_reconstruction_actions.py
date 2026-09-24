"""Native hand-reconstruction boundary tests; real pixels, no server or GPU.

The fake backend returns a PNG at the exact requested working resolution. The
real preparation and composition functions therefore run in every generation
test, including cancellation, cache identity and exclusive export checks.
"""
import base64
import copy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image, ImageDraw, PngImagePlugin

from backends.base import GenerationResult
from core import anima38, anima_guidance
from core.hand_reconstruction import prepare_hand_repair
from core.resource_coordinator import GenerationResourceCoordinator, ResourceBusyError
from ui.hand_reconstruction_actions import (
    HandReconstructionActionsMixin,
    export_hand_candidate,
    freeze_hand_backend,
    hand_generation_payload,
)


GENERATE = "hand_reconstruction_generate"
EXPORT = "hand_reconstruction_export"
CANCEL = "hand_reconstruction_cancel"


def png_bytes(image, *, metadata=False):
    stream = io.BytesIO()
    info = PngImagePlugin.PngInfo()
    if metadata:
        info.add_text("parameters", "original scene\nNegative prompt: blur\nSteps: 20")
        info.add_text("workflow", '{"nodes":[]}')
    image.save(stream, format="PNG", pnginfo=info)
    return stream.getvalue()


def image_url(image, *, metadata=False):
    return "data:image/png;base64," + base64.b64encode(png_bytes(image, metadata=metadata)).decode("ascii")


def url_bytes(value):
    return base64.b64decode(value.split(",", 1)[1], validate=True)


def repair_request(request_id="preview", **settings):
    source = Image.new("RGB", (96, 64), (70, 100, 130))
    ImageDraw.Draw(source).rectangle((0, 0, 20, 63), fill=(25, 40, 55))
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((34, 20, 53, 43), fill=255)
    return {
        "requestId": request_id,
        "image": image_url(source, metadata=True),
        "mask": image_url(mask),
        "prompt": "relaxed open hand",
        "settings": {"enabled": True, "candidates": 2, "strength": 0.9,
                     "padding": 12, "feather": 2, "resolution": 512, **settings},
    }


def generation_snapshot():
    scripts = {name: {"args": [True, 0.75]} for name in anima_guidance.SPECS}
    scripts.update({
        anima38.SCRIPT_NAME: {"args": [True, "adapter-v2", 1.0, "", 1.0, False]},
        "NegPiP": {"args": [True]},
        "ADetailer": {"args": [True]},
        "SAM3": {"args": [True]},
        "Unrelated extension": {"args": [True]},
    })
    return {
        "prompt": "open hand, natural pose, <lora:custom-anima:0.7>",
        "negative_prompt": "blur",
        "sampler_name": "Euler", "scheduler": "normal", "steps": 12,
        "cfg_scale": 5.0, "distilled_cfg_scale": 3.0, "seed": 42,
        "forge_additional_modules": ["vae.safetensors", "qwen-text-encoder.safetensors"],
        "alwayson_scripts": scripts,
        "enable_hr": True, "hr_scale": 2, "_postprocess_chain": ["sam3"],
        "_comfy_detail_passes": ["face"], "adetailer_enabled": True,
        "sam3_enabled": True, "comfy_workflow_controls": {"node": "external"},
        "override_settings": {"sd_vae": "conflicting-vae"},
    }


class Signal:
    def __init__(self):
        self.items = []
        self.ready = threading.Event()
        self.condition = threading.Condition()

    def emit(self, raw):
        with self.condition:
            self.items.append(json.loads(raw))
            self.ready.set()
            self.condition.notify_all()

    def wait_for(self, predicate, timeout=5):
        with self.condition:
            if not self.condition.wait_for(lambda: any(predicate(item) for item in self.items), timeout):
                raise AssertionError(f"Expected hand event did not arrive: {self.items!r}")
            return next(item for item in reversed(self.items) if predicate(item))


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.gates = {}
        self.outcomes = {}
        self.returned_pngs = []
        self.interrupt = mock.Mock(side_effect=AssertionError("Global interrupt is forbidden"))

    def get_backend_type(self):
        return "webui"

    def block_call(self, index):
        gate = (threading.Event(), threading.Event())
        self.gates[index] = gate
        return gate

    # Intentionally no cancel_check or **kwargs: the adapter may not install a
    # watcher that interrupts a shared Forge server after local cancellation.
    def img2img(self, model, payload, progress_callback=None):
        index = len(self.calls)
        self.calls.append({"model": model, "payload": copy.deepcopy(payload),
                           "thread": threading.get_ident()})
        if index in self.gates:
            entered, release = self.gates[index]
            entered.set()
            if not release.wait(5):
                raise AssertionError("Test did not release the fake backend")
        if progress_callback:
            progress_callback(1, payload["steps"], None)
        outcome = self.outcomes.get(index)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            return outcome
        generated = png_bytes(Image.new("RGB", (payload["width"], payload["height"]),
                                        (180 + index * 10, 65, 40)))
        self.returned_pngs.append(generated)
        return GenerationResult(success=True, image_data=generated)


class Host(HandReconstructionActionsMixin):
    def __init__(self, backend=None):
        self.vue_bridge = SimpleNamespace(handReconstructionEvent=Signal())
        self.web_mode = False
        self.backend = backend or FakeBackend()
        self.snapshot = generation_snapshot()
        self.snapshot_threads = []

    def _hand_snapshot(self, request):
        self.snapshot_threads.append(threading.get_ident())
        return "anima-model.safetensors", copy.deepcopy(self.snapshot), self.backend


class SnapshotHost(HandReconstructionActionsMixin):
    """Minimal proxy host that exercises the real UI-thread snapshot method."""
    def __init__(self, snapshot=None):
        self.vue_bridge = SimpleNamespace(handReconstructionEvent=Signal())
        self.web_mode = False
        self._is_krea2_generation = mock.Mock(return_value=False)
        self.total_prompt_display = SimpleNamespace(toPlainText=lambda: "existing pose")
        self._chat_generation_snapshot = mock.Mock(return_value=("selected-model", snapshot or generation_snapshot()))


class HandReconstructionActionTests(unittest.TestCase):
    def setUp(self):
        self.host = Host()
        self.backend = self.host.backend
        self.signal = self.host.vue_bridge.handReconstructionEvent
        self.coordinator = GenerationResourceCoordinator()
        self.threads = []
        self.real_thread = threading.Thread
        self.error_patch = mock.patch("core.error_handler.handle_error")
        self.error_patch.start()
        self.coordinator_patch = mock.patch("core.resource_coordinator.get_generation_coordinator", return_value=self.coordinator)
        self.coordinator_patch.start()
        self.thread_patch = mock.patch("ui.hand_reconstruction_actions.threading.Thread", side_effect=self.make_thread)
        self.thread_patch.start()

    def make_thread(self, *args, **kwargs):
        thread = self.real_thread(*args, **kwargs)
        self.threads.append(thread)
        return thread

    def tearDown(self):
        for _, release in self.backend.gates.values():
            release.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(5)
        self.thread_patch.stop()
        self.coordinator_patch.stop()
        self.error_patch.stop()
        self.assertFalse(any(thread.is_alive() for thread in self.threads), "Hand worker leaked from test")

    def complete(self, request_id="preview", *, action=GENERATE):
        event = self.signal.wait_for(lambda item: item.get("requestId") == request_id
                                     and item.get("action") == action and item.get("phase") == "complete")
        # Error/export signals can precede the worker's finally cleanup.
        for thread in self.threads:
            if thread.ident is not None and thread is not threading.current_thread():
                thread.join(5)
        return event

    def generate(self, request=None):
        request = request or repair_request()
        self.assertTrue(self.host._handle_hand_reconstruction_action(GENERATE, request))
        return self.complete(request["requestId"])

    def test_unknown_action_is_not_claimed(self):
        self.assertFalse(self.host._handle_hand_reconstruction_action("unrelated", {}))
        self.assertFalse(self.signal.items)
        self.assertFalse(self.threads)

    def test_native_only_opt_in_and_request_identity_fail_before_worker_or_backend(self):
        request = repair_request()
        self.host.web_mode = True
        self.host._handle_hand_reconstruction_action(GENERATE, request)
        self.assertIn("로컬 앱", self.signal.items[-1]["error"])
        self.host.web_mode = False
        for value in (None, {}, {"enabled": False}, {"enabled": 1}, {"enabled": "true"}):
            with self.subTest(settings=value):
                self.host._handle_hand_reconstruction_action(GENERATE, {**request, "settings": value})
                self.assertFalse(self.signal.items[-1]["ok"])
        for identity in ("", "../../bad", "space here", "x" * 101, None, 12, {}):
            with self.subTest(identity=identity):
                self.host._handle_hand_reconstruction_action(GENERATE, {**request, "requestId": identity})
                self.assertFalse(self.signal.items[-1]["ok"])
        self.host._handle_hand_reconstruction_action(GENERATE, [request])
        self.assertFalse(self.signal.items[-1]["ok"])
        self.assertFalse(self.host.snapshot_threads)
        self.assertFalse(self.threads)
        self.assertFalse(self.backend.calls)

    def test_invalid_image_requests_never_reach_the_backend(self):
        for index, value in enumerate(("C:/private.png", "file:///C:/private.png",
                                       "https://example.com/image.png", "data:image/png;base64,???")):
            with self.subTest(value=value):
                request = repair_request(f"invalid_{index}")
                request["image"] = value
                result = self.generate(request)
                self.assertFalse(result["ok"])
        self.assertFalse(self.backend.calls)
        self.assertIsNone(self.host._hand_job)
        self.assertIsNone(self.host._hand_preview)

    def test_full_generation_uses_masked_reset_payload_and_preserves_source_pixels(self):
        caller = threading.get_ident()
        request = repair_request(candidates=4)
        original_request = copy.deepcopy(request)
        original_snapshot = copy.deepcopy(self.host.snapshot)
        with mock.patch("ui.hand_reconstruction_actions.export_hand_candidate") as save:
            result = self.generate(request)
            save.assert_not_called()
        self.assertTrue(result["ok"])
        self.assertFalse(result["canceled"])
        self.assertEqual([item["seed"] for item in result["candidates"]], [42, 43, 44, 45])
        self.assertEqual(len(self.backend.calls), 4)
        self.assertEqual(self.host.snapshot_threads, [caller])
        self.assertTrue(all(call["thread"] != caller for call in self.backend.calls))
        self.assertEqual(request, original_request)
        self.assertEqual(self.host.snapshot, original_snapshot)
        prepared = prepare_hand_repair(request)
        allowed_scripts = {anima38.SCRIPT_NAME, *anima_guidance.SPECS, "NegPiP"}
        for call in self.backend.calls:
            payload = call["payload"]
            self.assertEqual(call["model"], "anima-model.safetensors")
            self.assertEqual((payload["width"], payload["height"]), prepared.working_size)
            self.assertEqual(base64.b64decode(payload["init_images"][0]), prepared.init_png)
            self.assertEqual(base64.b64decode(payload["mask"]), prepared.mask_png)
            self.assertEqual(payload["inpainting_fill"], 1)
            self.assertFalse(payload["inpaint_full_res"])
            self.assertEqual(payload["inpainting_mask_invert"], 0)
            self.assertEqual(payload["mask_blur"], 0)
            self.assertEqual(payload["mask_dilation"], 0)
            self.assertEqual(payload["grow_mask_by"], 0)
            self.assertEqual(payload["resize_mode"], 0)
            self.assertEqual(payload["denoising_strength"], 0.9)
            self.assertEqual((payload["batch_size"], payload["n_iter"]), (1, 1))
            self.assertFalse(payload["save_images"])
            self.assertTrue(payload["send_images"])
            self.assertTrue(payload["do_not_save_samples"])
            self.assertTrue(payload["do_not_save_grid"])
            self.assertEqual(payload["forge_additional_modules"], original_snapshot["forge_additional_modules"])
            self.assertIn("<lora:custom-anima:0.7>", payload["prompt"])
            self.assertEqual(set(payload["alwayson_scripts"]), allowed_scripts)
            for key in ("enable_hr", "hr_scale", "_postprocess_chain", "_comfy_detail_passes",
                        "adetailer_enabled", "sam3_enabled", "comfy_workflow_controls", "override_settings"):
                self.assertNotIn(key, payload)
        # The completion event no longer re-sends the source: the panel keeps the
        # exact image it sent with this requestId (display-only re-encode removed).
        self.assertNotIn("source", result)
        self.assertTrue(result["prepared"].startswith("data:image/png;base64,"))
        self.assertEqual(url_bytes(result["prepared"]), prepared.prepared_png)
        with Image.open(io.BytesIO(url_bytes(request["image"]))) as source, \
                Image.open(io.BytesIO(url_bytes(result["candidates"][0]["image"]))) as output:
            self.assertEqual(output.size, source.size)
            self.assertEqual(output.getpixel((0, 0)), source.getpixel((0, 0)))
            self.assertNotEqual(output.getpixel((44, 32)), source.getpixel((44, 32)))
            self.assertEqual(output.info["parameters"], source.info["parameters"])
            self.assertEqual(output.info["workflow"], source.info["workflow"])
            record = json.loads(output.info["ai_studio_hand_reconstruction"])
            self.assertFalse(record["run"]["anatomy_verified"])
            self.assertEqual(record["run"]["modules"], original_snapshot["forge_additional_modules"])
            self.assertEqual(record["run"]["seed"], 42)
        self.backend.interrupt.assert_not_called()
        self.assertEqual(self.coordinator.state.phase, "idle")

    def test_payload_nested_settings_are_copied_not_aliased(self):
        prepared = prepare_hand_repair(repair_request())
        snapshot = generation_snapshot()
        before = copy.deepcopy(snapshot)
        payload = hand_generation_payload(prepared, snapshot, 20)
        payload["forge_additional_modules"].append("later-change")
        payload["alwayson_scripts"]["NegPiP"]["args"].append(False)
        self.assertEqual(snapshot, before)

    def test_random_seed_is_selected_once_then_candidates_increment_with_uint32_wrap(self):
        self.host.snapshot["seed"] = -1
        with mock.patch("ui.hand_reconstruction_actions.secrets.randbits", return_value=2 ** 32 - 1) as random_seed:
            result = self.generate(repair_request(candidates=3))
        self.assertTrue(result["ok"])
        random_seed.assert_called_once_with(32)
        self.assertEqual([item["seed"] for item in result["candidates"]], [2 ** 32 - 1, 0, 1])

    def test_deferred_prompt_runs_in_worker_and_is_not_sent_to_backend(self):
        self.host.snapshot["_chat_deferred_prompt"] = {"wildcards": False}
        threads = []
        def prepare(snapshot):
            threads.append(threading.get_ident())
            values = copy.deepcopy(snapshot)
            values.pop("_chat_deferred_prompt")
            values["prompt"] += ", resolved tag"
            return values
        with mock.patch("core.chat_generation.prepare_prompt_payload", side_effect=prepare):
            result = self.generate(repair_request(candidates=1))
        self.assertTrue(result["ok"])
        self.assertNotEqual(threads, [threading.get_ident()])
        self.assertIn("resolved tag", self.backend.calls[0]["payload"]["prompt"])
        self.assertNotIn("_chat_deferred_prompt", self.backend.calls[0]["payload"])

    def test_duplicate_single_job_cancel_discards_late_result_and_holds_resource_lease(self):
        entered, release = self.backend.block_call(0)
        request = repair_request(candidates=3)
        self.host._handle_hand_reconstruction_action(GENERATE, request)
        self.assertTrue(entered.wait(5))
        count = len(self.signal.items)
        self.host._handle_hand_reconstruction_action(GENERATE, request)
        self.assertEqual(len(self.signal.items), count)
        self.assertEqual(len(self.threads), 1)
        self.host._handle_hand_reconstruction_action(GENERATE, repair_request("other"))
        self.assertIn("실행 중", self.signal.items[-1]["error"])
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "unrelated"})
        self.assertFalse(self.host._hand_job["cancel"].is_set())
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "preview"})
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "preview"})
        self.assertTrue(self.host._hand_job["cancel"].is_set())
        self.assertEqual(self.coordinator.state.phase, "running")
        with self.assertRaises(ResourceBusyError):
            with self.coordinator.reserve("other-ui-generation", timeout=0):
                self.fail("Canceled HTTP request released the lease before returning")
        release.set()
        result = self.complete()
        self.assertFalse(result["ok"])
        self.assertTrue(result["canceled"])
        self.assertFalse(result["candidates"])
        self.assertIsNone(self.host._hand_preview)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertIsNone(self.host._hand_job)
        self.assertEqual(self.coordinator.state.phase, "idle")
        self.backend.interrupt.assert_not_called()

    def test_cancel_during_later_candidate_keeps_completed_candidate_only(self):
        entered, release = self.backend.block_call(1)
        self.host._handle_hand_reconstruction_action(GENERATE, repair_request(candidates=4))
        self.assertTrue(entered.wait(5))
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "preview"})
        release.set()
        result = self.complete()
        self.assertTrue(result["ok"])
        self.assertTrue(result["canceled"])
        self.assertEqual([item["seed"] for item in result["candidates"]], [42])
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(len(self.host._hand_preview["candidates"]), 1)
        self.backend.interrupt.assert_not_called()

    def test_later_failure_retains_prior_candidate_and_reports_warning(self):
        self.backend.outcomes[1] = GenerationResult(success=False, error="second candidate failed")
        result = self.generate(repair_request(candidates=4))
        self.assertTrue(result["ok"])
        self.assertFalse(result["canceled"])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertIn("second candidate failed", result["warning"])
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(len(self.host._hand_preview["candidates"]), 1)

    def test_first_failure_releases_job_and_lease_without_candidate(self):
        self.backend.outcomes[0] = RuntimeError("backend failed")
        result = self.generate()
        self.assertFalse(result["ok"])
        self.assertIn("backend failed", result["error"])
        self.assertIsNone(self.host._hand_preview)
        self.assertIsNone(self.host._hand_job)
        self.assertEqual(self.coordinator.state.phase, "idle")

    def test_wrong_backend_resolution_is_rejected_before_cache_or_export(self):
        self.backend.outcomes[0] = GenerationResult(True, png_bytes(Image.new("RGB", (64, 64))))
        with mock.patch("ui.hand_reconstruction_actions.export_hand_candidate") as save:
            result = self.generate()
            save.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("해상도", result["error"])
        self.assertIsNone(self.host._hand_preview)

    def test_busy_shared_coordinator_prevents_any_backend_submission(self):
        with self.coordinator.reserve("ordinary-generation", unload_llm=False):
            result = self.generate()
            self.assertFalse(result["ok"])
            self.assertFalse(self.backend.calls)
            self.assertEqual(self.coordinator.state.owner, "ordinary-generation")
        self.assertEqual(self.coordinator.state.phase, "idle")

    def test_cache_budget_failure_never_publishes_an_oversized_candidate(self):
        with mock.patch("ui.hand_reconstruction_actions.MAX_CACHED_BYTES", 1):
            result = self.generate()
        self.assertFalse(result["ok"])
        self.assertIn("128 MB", result["error"])
        self.assertIsNone(self.host._hand_preview)
        self.assertEqual(len(self.backend.calls), 1)

    def generate_on_fresh_host(self, request):
        host = Host()   # fresh FakeBackend → identical candidate bytes for identical input
        host._handle_hand_reconstruction_action(GENERATE, request)
        event = host.vue_bridge.handReconstructionEvent.wait_for(
            lambda item: item.get("requestId") == request["requestId"] and item.get("phase") == "complete")
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(5)
        return host, event

    def test_cache_budget_counts_only_prepared_crop_and_candidates(self):
        """회귀: 표시 전용 source_png가 예산을 잡아먹어 JPEG 원본에서 후보가 거부되던 문제."""
        request = repair_request(candidates=1)
        host, event = self.generate_on_fresh_host(request)
        self.assertTrue(event["ok"])
        needed = len(prepare_hand_repair(request).prepared_png) + len(host._hand_preview["candidates"][0]["png"])
        with mock.patch("ui.hand_reconstruction_actions.MAX_CACHED_BYTES", needed):
            _host, exact = self.generate_on_fresh_host(request)
        self.assertTrue(exact["ok"], "원본 재인코딩이 예산에 포함되면 정확히 맞춘 예산에서 실패한다")
        with mock.patch("ui.hand_reconstruction_actions.MAX_CACHED_BYTES", needed - 1):
            _host, over = self.generate_on_fresh_host(request)
        self.assertFalse(over["ok"])
        self.assertIn("128 MB", over["error"])

    def test_export_requires_current_cache_and_strict_index_and_ignores_client_paths(self):
        generated = self.generate(repair_request(candidates=1))
        self.assertTrue(generated["ok"])
        cached_png = self.host._hand_preview["candidates"][0]["png"]
        with tempfile.TemporaryDirectory(prefix="hand-action-test-") as temporary, \
                mock.patch.dict(sys.modules, {"config": SimpleNamespace(OUTPUT_DIR=temporary)}):
            source = Path(temporary) / "original.png"
            source.write_bytes(b"preserve original file")
            invalid = [("old-preview", 0), ("preview", -1), ("preview", 1),
                       ("preview", True), ("preview", 0.0), ("preview", "0"), ("preview", None)]
            for number, (preview_id, index) in enumerate(invalid):
                request_id = f"bad_export_{number}"
                self.host._handle_hand_reconstruction_action(EXPORT, {
                    "requestId": request_id, "previewRequestId": preview_id, "candidateIndex": index,
                })
                self.assertFalse(self.complete(request_id, action=EXPORT)["ok"])
            self.assertEqual(list(Path(temporary).iterdir()), [source])
            paths = []
            for number in range(2):
                request_id = f"save_{number}"
                self.host._handle_hand_reconstruction_action(EXPORT, {
                    "requestId": request_id, "previewRequestId": "preview", "candidateIndex": 0,
                    "path": str(source), "outputPath": str(source), "outputRoot": "C:/not-allowed",
                    "image": "data:image/png;base64,AAAA",
                })
                saved = self.complete(request_id, action=EXPORT)
                self.assertTrue(saved["ok"])
                path = Path(saved["path"])
                self.assertEqual(path.parent, Path(temporary) / "hand_reconstruction")
                self.assertEqual(path.read_bytes(), cached_png)
                paths.append(path)
            self.assertNotEqual(*paths)
            self.assertEqual(source.read_bytes(), b"preserve original file")

    def test_exclusive_export_collision_never_overwrites_existing_file(self):
        with tempfile.TemporaryDirectory(prefix="hand-exclusive-test-") as temporary, \
                mock.patch("ui.hand_reconstruction_actions.datetime") as clock, \
                mock.patch("ui.hand_reconstruction_actions.secrets.token_hex", return_value="same-name"):
            clock.now.return_value = datetime(2026, 9, 6, 12, 30, 0)
            original = Path(export_hand_candidate(b"existing output", temporary))
            with self.assertRaisesRegex(ValueError, "이름"):
                export_hand_candidate(b"replacement must not win", temporary)
            self.assertEqual(original.read_bytes(), b"existing output")
            self.assertEqual(list(original.parent.iterdir()), [original])

    def test_cancel_clears_matching_cached_preview_but_not_unrelated_preview(self):
        self.generate(repair_request(candidates=1))
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "unrelated"})
        self.assertIsNotNone(self.host._hand_preview)
        self.host._handle_hand_reconstruction_action(CANCEL, {"requestId": "preview"})
        self.assertIsNone(self.host._hand_preview)
        self.host._handle_hand_reconstruction_action(EXPORT, {
            "requestId": "stale_export", "previewRequestId": "preview", "candidateIndex": 0,
        })
        self.assertFalse(self.complete("stale_export", action=EXPORT)["ok"])

    def test_shutdown_suppresses_inflight_result_clears_cache_and_rejects_new_work(self):
        entered, release = self.backend.block_call(0)
        self.host._handle_hand_reconstruction_action(GENERATE, repair_request(candidates=3))
        self.assertTrue(entered.wait(5))
        self.host._shutdown_hand_reconstruction()
        after_close = len(self.signal.items)
        self.assertTrue(self.host._hand_job["cancel"].is_set())
        self.assertIsNone(self.host._hand_preview)
        self.host._handle_hand_reconstruction_action(GENERATE, repair_request("late"))
        self.assertEqual(len(self.threads), 1)
        release.set()
        self.threads[0].join(5)
        self.assertFalse(self.threads[0].is_alive())
        self.assertEqual(len(self.signal.items), after_close)
        self.assertIsNone(self.host._hand_preview)
        self.assertIsNone(self.host._hand_job)
        self.assertEqual(self.coordinator.state.phase, "idle")
        self.backend.interrupt.assert_not_called()

    def test_thread_start_failure_clears_job_and_reports_error(self):
        with mock.patch.object(self.real_thread, "start", side_effect=RuntimeError("thread unavailable")):
            self.host._handle_hand_reconstruction_action(GENERATE, repair_request())
        result = self.complete()
        self.assertFalse(result["ok"])
        self.assertIn("thread unavailable", result["error"])
        self.assertIsNone(self.host._hand_job)
        self.assertFalse(self.backend.calls)


class HandReconstructionCapabilityTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch("core.error_handler.handle_error")
        patch.start()
        self.addCleanup(patch.stop)

    def test_custom_comfy_workflow_is_rejected_before_constructing_or_submitting_backend(self):
        backend = mock.Mock(api_url="http://127.0.0.1:8188")
        backend.get_backend_type.return_value = "comfyui"
        backend._configured_workflow_path.return_value = "custom-i2i.json"
        factory = mock.Mock()
        with mock.patch.dict(sys.modules, {"backends.comfyui_backend": SimpleNamespace(ComfyUIBackend=factory)}):
            with self.assertRaisesRegex(ValueError, "사용자 워크플로"):
                freeze_hand_backend(backend)
        backend._configured_workflow_path.assert_called_once_with("img2img")
        factory.assert_not_called()
        backend.img2img.assert_not_called()

    def test_default_comfy_backend_freezes_endpoint_and_explicit_empty_workflow_paths(self):
        backend = mock.Mock(api_url="http://127.0.0.1:8188")
        backend.get_backend_type.return_value = "comfyui"
        backend._configured_workflow_path.return_value = ""
        frozen = object()
        factory = mock.Mock(return_value=frozen)
        with mock.patch.dict(sys.modules, {"backends.comfyui_backend": SimpleNamespace(ComfyUIBackend=factory)}):
            result = freeze_hand_backend(backend)
        self.assertIs(result, frozen)
        factory.assert_called_once_with("http://127.0.0.1:8188", workflow_path="", img2img_workflow_path="")
        backend.img2img.assert_not_called()

    def test_webui_identity_is_preserved_and_unknown_backend_is_rejected(self):
        backend = FakeBackend()
        self.assertIs(freeze_hand_backend(backend), backend)
        backend.get_backend_type = lambda: "other"
        with self.assertRaisesRegex(ValueError, "기본 경로"):
            freeze_hand_backend(backend)

    def test_krea_ui_selection_and_snapshot_family_are_both_rejected_without_api(self):
        for use_flag in (True, False):
            with self.subTest(ui_flag=use_flag):
                snapshot = generation_snapshot()
                snapshot["_generation_family"] = "krea2"
                host = SnapshotHost(snapshot)
                host._is_krea2_generation.return_value = use_flag
                with mock.patch("backends.get_backend") as backend, \
                        mock.patch("ui.hand_reconstruction_actions.threading.Thread") as thread:
                    host._handle_hand_reconstruction_action(GENERATE, repair_request())
                    event = host.vue_bridge.handReconstructionEvent.items[-1]
                    self.assertFalse(event["ok"])
                    self.assertIn("Krea2", event["error"])
                    backend.assert_not_called()
                    thread.assert_not_called()
                if use_flag:
                    host._chat_generation_snapshot.assert_not_called()

    def test_real_snapshot_appends_instruction_and_negative_without_losing_lora_or_modules(self):
        snapshot = generation_snapshot()
        original = copy.deepcopy(snapshot)
        host = SnapshotHost(snapshot)
        backend = FakeBackend()
        with mock.patch("backends.get_backend", return_value=backend):
            model, result, selected = host._hand_snapshot({"prompt": "  spread fingers naturally  "})
        self.assertEqual(model, "selected-model")
        self.assertIs(selected, backend)
        host._chat_generation_snapshot.assert_called_once_with("existing pose\nspread fingers naturally")
        self.assertEqual(result["forge_additional_modules"], original["forge_additional_modules"])
        self.assertIn("<lora:custom-anima:0.7>", result["prompt"])
        self.assertEqual(result["negative_prompt"], "blur, extra fingers, fused fingers, duplicated hands, malformed hands")
        self.assertEqual(set(result["alwayson_scripts"]), {anima38.SCRIPT_NAME, *anima_guidance.SPECS, "NegPiP"})
        self.assertNotIn("enable_hr", result)
        self.assertEqual(snapshot, original)

    def test_invalid_or_empty_instruction_is_rejected_before_snapshot_or_backend(self):
        for text in (None, 12, "x" * 4001):
            host = SnapshotHost()
            with self.subTest(text_type=type(text).__name__), mock.patch("backends.get_backend") as backend:
                with self.assertRaises(ValueError):
                    host._hand_snapshot({"prompt": text})
                host._chat_generation_snapshot.assert_not_called()
                backend.assert_not_called()
        host = SnapshotHost()
        host.total_prompt_display = SimpleNamespace(toPlainText=lambda: "  ")
        with mock.patch("backends.get_backend") as backend:
            with self.assertRaisesRegex(ValueError, "프롬프트"):
                host._hand_snapshot({"prompt": "  "})
            host._chat_generation_snapshot.assert_not_called()
            backend.assert_not_called()


class HandReconstructionIntegrationBoundaryTests(unittest.TestCase):
    def test_actual_hand_payload_compiles_to_one_masked_comfy_pass_with_modules_and_loras(self):
        from core.comfy_workflow_compiler import ComfyWorkflowCompiler
        from tests.test_comfy_workflow_compiler import _capabilities, _classes, _node

        prepared = prepare_hand_repair(repair_request(candidates=1))
        snapshot = generation_snapshot()
        # This fixture is a normal full checkpoint, not a semantic Anima 3.8
        # checkpoint. Keep real fixture resource names and independent LoRA
        # model/CLIP weights; Anima script preservation is covered above.
        snapshot.update({
            "prompt": "open hand, <lora:ink:0.7:0.4>, <lora:alice:0.3:0.2>",
            "forge_additional_modules": ["text/base.safetensors", "vae/image_vae.safetensors"],
            "alwayson_scripts": {
                "NegPiP": {"args": [True]},
                "ADetailer": {"args": [True]},
                "SAM3": {"args": [True]},
            },
        })
        snapshot_before = copy.deepcopy(snapshot)
        payload = hand_generation_payload(prepared, snapshot, 1234)
        payload_before = copy.deepcopy(payload)
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "inpaint", "checkpoint.safetensors", payload,
            uploaded_image="source.png", uploaded_mask="mask.png",
        )

        latent_id, latent_node = _node(graph, "ForgeNeoLatentInput")
        latent = latent_node["inputs"]
        self.assertEqual(latent["mode"], "inpaint")
        self.assertEqual((latent["width"], latent["height"]), prepared.working_size)
        self.assertEqual(latent["batch_size"], 1)
        self.assertEqual(latent["fit"], "stretch")
        self.assertFalse(latent["mask_invert"])
        self.assertEqual(latent["mask_blur"], 0)
        self.assertEqual(latent["grow_mask_by"], 0)
        self.assertFalse(latent["reference_enabled"])
        image_nodes = {node["inputs"]["image"]: node_id for node_id, node in graph.items()
                       if node.get("class_type") == "LoadImage"}
        self.assertEqual(set(image_nodes), {"source.png", "mask.png"})
        self.assertEqual(latent["inpaint_image"], [image_nodes["source.png"], 0])
        self.assertEqual(latent["inpaint_mask_image"], [image_nodes["mask.png"], 0])
        self.assertNotIn("img2img_image", latent)

        sampler_id, sampler_node = _node(graph, "ForgeNeoKSamplerCNS")
        sampler = sampler_node["inputs"]
        self.assertEqual(sampler["latent_image"], [latent_id, 0])
        self.assertEqual(sampler["denoise"], 0.9)
        self.assertEqual(sampler["seed"], 1234)
        self.assertEqual(sampler["steps"], snapshot["steps"])
        self.assertEqual(sampler["cfg"], snapshot["cfg_scale"])
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("euler", "normal"))

        self.assertEqual(_node(graph, "CheckpointLoaderSimple")[1]["inputs"]["ckpt_name"], "checkpoint.safetensors")
        self.assertEqual(_node(graph, "CLIPLoader")[1]["inputs"]["clip_name"], "text/base.safetensors")
        vae_id, vae = _node(graph, "VAELoader")
        self.assertEqual(vae["inputs"]["vae_name"], "vae/image_vae.safetensors")
        self.assertEqual(latent["vae"], [vae_id, 0])
        loras = [node["inputs"] for node in graph.values() if node.get("class_type") == "LoraLoader"]
        self.assertEqual([(item["lora_name"], item["strength_model"], item["strength_clip"]) for item in loras], [
            ("styles/ink.safetensors", 0.7, 0.4), ("characters/alice.safetensors", 0.3, 0.2),
        ])
        self.assertIn("ForgeNeoNegPip", _classes(graph))
        self.assertEqual(_node(graph, "CLIPTextEncode")[1]["inputs"]["text"], "open hand")
        for class_type in ("ForgeNeoHiresFix", "ForgeNeoADetailer", "ForgeNeoSAM3Mask",
                           "ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine", "LatentUpscale"):
            self.assertNotIn(class_type, _classes(graph))
        self.assertEqual(_classes(graph).count("ForgeNeoKSamplerCNS"), 1)
        self.assertEqual(_classes(graph).count("VAEDecode"), 1)
        decode_id, decode = _node(graph, "VAEDecode")
        self.assertEqual(decode["inputs"]["samples"], [sampler_id, 0])
        self.assertEqual(decode["inputs"]["vae"], [vae_id, 0])
        # 손 재구성 payload 는 save_images=False — Forge 처럼 ComfyUI/output 에
        # 사본을 남기지 않고 temp 미리보기로 결과만 돌려받는다.
        self.assertNotIn("SaveImage", _classes(graph))
        self.assertEqual(_classes(graph).count("PreviewImage"), 1)
        self.assertEqual(_node(graph, "PreviewImage")[1]["inputs"]["images"], [decode_id, 0])
        self.assertEqual(payload, payload_before)
        self.assertEqual(snapshot, snapshot_before)

    def test_native_original_byte_loader_preserves_png_metadata_and_prepare_source_hash(self):
        from ui.vue_bridge import VueBridge

        request = repair_request(candidates=1)
        original_bytes = url_bytes(request["image"])
        original_hash = hashlib.sha256(original_bytes).hexdigest()
        with tempfile.TemporaryDirectory(prefix="hand-native-source-") as temporary:
            path = Path(temporary) / "원본 image with metadata.png"
            path.write_bytes(original_bytes)
            # Invoke the existing slot unbound: no QApplication/QObject/window,
            # replacement loader or patched path-normalization implementation.
            host = SimpleNamespace(web_mode=False)
            for original_path in (str(path), path.as_uri()):
                with self.subTest(path=original_path):
                    loaded = VueBridge.loadImageBase64(host, original_path)
                    self.assertTrue(loaded.startswith("data:image/png;base64,"))
                    self.assertEqual(url_bytes(loaded), original_bytes)
                    prepared = prepare_hand_repair({**request, "image": loaded})
                    self.assertEqual(prepared.source_sha256, original_hash)
                    self.assertEqual(prepared.source_metadata["text"]["workflow"], '{"nodes":[]}')
                    self.assertIn("original scene", prepared.source_metadata["text"]["parameters"])
                    self.assertFalse(hasattr(prepared, "source_png"),
                                     "표시 전용 원본 재인코딩은 만들지 않는다 (패널이 보낸 원본을 쓴다)")
                    with Image.open(io.BytesIO(original_bytes)) as original:
                        self.assertEqual(prepared.source.size, original.size)
                        self.assertEqual(prepared.source.tobytes(), original.convert(prepared.source.mode).tobytes())
                        self.assertEqual(prepared.source_metadata["text"]["parameters"], original.info["parameters"])
                        self.assertEqual(prepared.source_metadata["text"]["workflow"], original.info["workflow"])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hash)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
