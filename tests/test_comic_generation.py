"""Comic 컷 생성이 메인 T2I payload 빌더를 재사용한다 (감사 #32).

예전엔 {prompt, negative_prompt, seed, width, height} 만 담아 backend.txt2img 를 직접 불러
sampler/steps/cfg·VAE/TE·Hires·LoRA·NegPiP·ADetailer/SAM3·Comfy 프리셋이 빠지고, T2I 에서
KREA2 를 골라도 일반 체크포인트로 생성됐다.
"""
from __future__ import annotations

import json
import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from backends.base import GenerationResult
from core import comic_generation as cg
from ui.creator_actions import CreatorActionsMixin


class MergeAndRequestTests(unittest.TestCase):
    def test_negative_merge_keeps_t2i_text_and_appends_only_new_panel_tags(self):
        self.assertEqual(cg.merge_negative("lowres, (bad hands:1.2)", "Lowres, blurry"),
                         "lowres, (bad hands:1.2), blurry")
        self.assertEqual(cg.merge_negative("", "blurry, blurry"), "blurry")
        self.assertEqual(cg.merge_negative("worst quality,", ""), "worst quality")
        self.assertEqual(cg.merge_negative(None, None), "")

    def test_explicit_size_only_when_both_positive(self):
        self.assertEqual(cg.explicit_size({"width": "832", "height": 1216}), (832, 1216))
        self.assertIsNone(cg.explicit_size({"width": 832}))
        self.assertIsNone(cg.explicit_size({"width": "x", "height": 5}))
        self.assertIsNone(cg.explicit_size({}))

    def test_panel_request_overlays_panel_seed_negative_and_fixed_size_on_snapshot(self):
        snapshot = {"prompt": "hero, <lora:ink:0.7>", "negative_prompt": "lowres", "seed": 99,
                    "sampler_name": "Euler a", "steps": 30, "cfg_scale": 5.5, "width": 640, "height": 960,
                    "enable_hr": True, "alwayson_scripts": {"ADetailer": {"args": [True, False]}}}
        request = cg.panel_request(snapshot, {"prompt": "hero", "negative_prompt": "text", "seed": 7},
                                   size=(832, 1216))
        self.assertEqual(request["prompt"], "hero, <lora:ink:0.7>")
        self.assertEqual(request["negative_prompt"], "lowres, text")
        self.assertEqual(request["seed"], 7)
        self.assertEqual((request["width"], request["height"]), (832, 1216))
        self.assertEqual((request["sampler_name"], request["steps"], request["cfg_scale"]), ("Euler a", 30, 5.5))
        self.assertTrue(request["enable_hr"])
        # 원본 스냅샷은 건드리지 않는다
        self.assertEqual(snapshot["seed"], 99)
        request["alwayson_scripts"]["ADetailer"]["args"].append("x")
        self.assertEqual(snapshot["alwayson_scripts"]["ADetailer"]["args"], [True, False])

    def test_panel_request_drops_t2i_main_prompt_overrides(self):
        """T2I 의 Hires/ADetailer/SAM3 prompt 는 T2I 메인 프롬프트용 — 컷 패스는 컷 프롬프트를 따른다."""
        snapshot = {
            "prompt": "a castle", "negative_prompt": "lowres", "enable_hr": True, "hr_scale": 1.5,
            "denoising_strength": 0.45, "hr_prompt": "T2I hires prompt: 1girl, red dress",
            "hr_negative_prompt": "T2I hires negative",
            "alwayson_scripts": {
                "ADetailer": {"args": [True, False, {"ad_model": "face_yolov8n.pt", "ad_prompt": "1girl",
                                                     "ad_negative_prompt": "bad face", "ad_denoising_strength": 0.4}]},
                "SAM3 Mask": {"args": [{"sam3_prompt": "face", "sam3_exclude_prompt": "eyes",
                                        "sam3_inpaint_prompt": "a castle", "sam3_negative_prompt": "lowres"}]},
            },
        }
        request = cg.panel_request(snapshot, {"prompt": "a castle", "negative_prompt": "text", "seed": 1})
        self.assertNotIn("hr_prompt", request)
        self.assertNotIn("hr_negative_prompt", request)
        self.assertEqual((request["enable_hr"], request["hr_scale"], request["denoising_strength"]), (True, 1.5, 0.45))
        slot = request["alwayson_scripts"]["ADetailer"]["args"][2]
        self.assertEqual((slot["ad_prompt"], slot["ad_negative_prompt"]), ("", ""))
        self.assertEqual((slot["ad_model"], slot["ad_denoising_strength"]), ("face_yolov8n.pt", 0.4))
        sam = request["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual((sam["sam3_inpaint_prompt"], sam["sam3_negative_prompt"]), ("", ""))
        self.assertEqual((sam["sam3_prompt"], sam["sam3_exclude_prompt"]), ("face", "eyes"))
        self.assertEqual(request["negative_prompt"], "lowres, text")
        # 원본 스냅샷은 그대로
        self.assertEqual(snapshot["hr_prompt"], "T2I hires prompt: 1girl, red dress")
        self.assertEqual(snapshot["alwayson_scripts"]["ADetailer"]["args"][2]["ad_prompt"], "1girl")
        self.assertEqual(snapshot["alwayson_scripts"]["SAM3 Mask"]["args"][0]["sam3_inpaint_prompt"], "a castle")


class RunPanelRoutingTests(unittest.TestCase):
    def test_standard_family_runs_deferred_prompt_and_strips_private_keys(self):
        calls = []

        class _Backend:
            def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
                calls.append((model, dict(payload), cancel_check))
                return GenerationResult(success=True, image_data=b"png")

        payload = {"prompt": "__hair__", "negative_prompt": "bad", "_generation_family": "standard",
                   "_chat_deferred_prompt": {"wildcards": False}, "_comfy_model_snapshot": "m",
                   "_xyz_info": {"x": 1}, "steps": 30}
        with mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=lambda text: text + "!"):
            result = cg.run_panel(_Backend(), "model.safetensors", payload, None, cancel_check=lambda: False)
        self.assertTrue(result.success)
        model, sent, cancel_check = calls[0]
        self.assertEqual(model, "model.safetensors")
        self.assertEqual(sent["prompt"], "__hair__!")
        self.assertEqual(sent["steps"], 30)
        self.assertFalse([key for key in sent if key.startswith("_")], sent)
        self.assertTrue(callable(cancel_check))
        self.assertIn("_chat_deferred_prompt", payload)  # 입력은 그대로

    def test_krea2_family_routes_to_krea2_runner_not_checkpoint_txt2img(self):
        backend = SimpleNamespace(txt2img=mock.Mock(side_effect=AssertionError("standard checkpoint used")))
        expected = GenerationResult(success=True, image_data=b"krea")
        seen = {}

        def _run(backend_arg, operation, params, progress_callback=None, cancel_check=None):
            seen.update(operation=operation, params=params, cancel=cancel_check)
            return expected

        with mock.patch("core.krea2_generation.run_krea2_generation", side_effect=_run):
            result = cg.run_panel(backend, "", {"prompt": "hero", "_generation_family": "krea2", "seed": 3},
                                  None, cancel_check=lambda: False)
        self.assertIs(result, expected)
        self.assertEqual(seen["operation"], "t2i")
        self.assertNotIn("_generation_family", seen["params"])
        self.assertTrue(callable(seen["cancel"]))

    def test_sam3_fallback_uses_the_final_prompt_after_deferred_hooks(self):
        calls = []

        class _Backend:
            def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
                calls.append(payload)
                return GenerationResult(success=True, image_data=b"png")

        payload = {"prompt": "hero", "negative_prompt": "lowres, text",
                   "_chat_deferred_prompt": {"wildcards": False},
                   "alwayson_scripts": {"SAM3 Mask": {"args": [{"sam3_prompt": "face", "sam3_inpaint_prompt": "",
                                                                "sam3_negative_prompt": ""}]}}}
        with mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=lambda text: text + " [hooked]"):
            cg.run_panel(_Backend(), "m", payload, None, cancel_check=lambda: False)
        state = calls[0]["alwayson_scripts"]["SAM3 Mask"]["args"][0]
        self.assertEqual(state["sam3_inpaint_prompt"], "hero [hooked]")
        self.assertEqual(state["sam3_negative_prompt"], "lowres, text [hooked]")
        # 입력 payload 는 건드리지 않는다
        self.assertEqual(payload["alwayson_scripts"]["SAM3 Mask"]["args"][0]["sam3_inpaint_prompt"], "")

    def test_legacy_adapter_without_cancel_check_still_works(self):
        class _Legacy:
            def txt2img(self, model, payload, progress_callback=None):
                return GenerationResult(success=True, image_data=b"ok")

        self.assertTrue(cg.run_panel(_Legacy(), "m", {"prompt": "p"}, None, cancel_check=lambda: False).success)


# ── Creator 액션 통합: UI 스레드 스냅샷 → 워커 ─────────────────────────────


class _Signal:
    def __init__(self):
        self.values = []

    def emit(self, value):
        self.values.append(json.loads(value))


class _Bridge:
    def __init__(self):
        for name in ("creatorResult", "creatorProgress", "creatorStateChanged", "comicDocumentChanged"):
            setattr(self, name, _Signal())


class _ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class _Panel:
    def __init__(self, panel_id):
        self.id = panel_id
        self.image_path = ""


class _Document:
    def __init__(self, count):
        self.panels = [_Panel(f"panel-{i + 1}") for i in range(count)]

    def to_dict(self):
        return {"panels": [{"id": p.id, "imagePath": p.image_path} for p in self.panels]}


def _panel_payloads(document):
    return [{"panel_index": i, "panel_id": p.id, "prompt": f"scene {i}", "negative_prompt": "text" if i else "",
             "seed": 100 + i} for i, p in enumerate(document.panels)]


class _Host(CreatorActionsMixin):
    """GenerationMixin 의 빌더를 흉내 — 컷 프롬프트에 LoRA 를 붙이고 T2I 설정을 채운다."""

    def __init__(self, document, *, family="standard", sizes=None, model="anima.safetensors",
                 error=None, needs_checkpoint=True):
        self.vue_bridge = _Bridge()
        self._creator_state_lock = threading.RLock()
        self._creator_job_local = threading.local()
        self._creator_running = False
        self._creator_mode = ""
        self._creator_active_job = None
        self._creator_cancel_event = threading.Event()
        self._creator_coordinator = SimpleNamespace(reserve=lambda *_a, **_k: nullcontext())
        self._creator_should_unload_ollama = lambda: False
        self._comic_studio = lambda: SimpleNamespace(normalize=lambda _d: document, save=lambda d: d)
        self.written = []
        self._creator_write_bytes = lambda data, name, feature: self.written.append(name) or f"out/{name}"
        self.model_combo = SimpleNamespace(currentText=lambda: model)
        self._family = family
        self._sizes = list(sizes or [(832, 1216)])
        self._error = error
        self._needs = needs_checkpoint
        self.builder_calls = []

    def _backend_needs_checkpoint(self):
        return self._needs

    def _build_generation_payload(self, *, prompt_override=None, snapshot=False, comfy_workflow_snapshot=None):
        self.builder_calls.append((prompt_override, snapshot))
        if self._error:
            return None, self._error
        width, height = self._sizes[min(len(self.builder_calls) - 1, len(self._sizes) - 1)]
        payload = {
            "prompt": prompt_override if self._family == "krea2" else f"{prompt_override}, <lora:ink:0.7>",
            "negative_prompt": "lowres", "sampler_name": "DPM++ 2M", "scheduler": "Karras",
            "steps": 30, "cfg_scale": 5.5, "seed": -1, "width": width, "height": height,
            "forge_additional_modules": ["vae.safetensors"], "enable_hr": True, "hr_scale": 1.5,
            "alwayson_scripts": {"NegPiP": {"args": [True]}},
            "_chat_deferred_prompt": {"wildcards": False},
        }
        if self._family == "krea2":
            payload["_generation_family"] = "krea2"
        return payload, None


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def txt2img(self, model, payload, progress_callback=None, cancel_check=None):
        self.calls.append((model, payload))
        return GenerationResult(success=True, image_data=b"png")

    def interrupt(self):
        pass


def _run(host, action, payload, backend, backend_type=None):
    from backends import BackendType

    with mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread), \
         mock.patch("backends.get_backend", return_value=backend), \
         mock.patch("backends.get_backend_type", return_value=backend_type or BackendType.WEBUI), \
         mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads), \
         mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=lambda text: text):
        host._handle_creator_action(action, payload)
    return host.vue_bridge.creatorResult.values[-1]


class ComicUsesT2IBuilderTests(unittest.TestCase):
    def test_generate_all_sends_full_t2i_settings_lora_and_fixed_size(self):
        host = _Host(_Document(3), sizes=[(832, 1216), (1024, 1024), (640, 640)])  # random_res 흉내
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "comic_generate_all", {"document": {}}, backend)
        self.assertTrue(result["ok"], result)
        self.assertEqual([call[0] for call in host.builder_calls], ["scene 0", "scene 1", "scene 2"])
        self.assertTrue(all(snapshot for _prompt, snapshot in host.builder_calls))
        self.assertEqual(len(backend.calls), 3)
        for index, (model, sent) in enumerate(backend.calls):
            self.assertEqual(model, "anima.safetensors")
            self.assertEqual(sent["prompt"], f"scene {index}, <lora:ink:0.7>")
            self.assertEqual((sent["sampler_name"], sent["scheduler"], sent["steps"], sent["cfg_scale"]),
                             ("DPM++ 2M", "Karras", 30, 5.5))
            self.assertEqual(sent["forge_additional_modules"], ["vae.safetensors"])
            self.assertTrue(sent["enable_hr"])
            self.assertEqual(sent["alwayson_scripts"], {"NegPiP": {"args": [True]}})
            self.assertEqual(sent["seed"], 100 + index)
            self.assertEqual((sent["width"], sent["height"]), (832, 1216))  # 첫 스냅샷 크기로 고정
            self.assertFalse([key for key in sent if key.startswith("_")], sent)
        self.assertEqual(backend.calls[1][1]["negative_prompt"], "lowres, text")
        self.assertEqual(host.written, ["comic_panel_1.png", "comic_panel_2.png", "comic_panel_3.png"])

    def test_explicit_request_size_and_model_override_the_t2i_values(self):
        host = _Host(_Document(1))
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "comic_generate_all",
                          {"document": {}, "width": 512, "height": 768, "model": "other.safetensors"}, backend)
        self.assertTrue(result["ok"], result)
        model, sent = backend.calls[0]
        self.assertEqual(model, "other.safetensors")
        self.assertEqual((sent["width"], sent["height"]), (512, 768))

    def test_single_panel_snapshots_only_that_panel(self):
        host = _Host(_Document(3))
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "creator_generate", {"mode": "comic_panel", "panel": {"id": "panel-2"},
                                                      "document": {}}, backend)
        self.assertTrue(result["ok"], result)
        self.assertEqual(host.builder_calls, [("scene 1", True)])
        self.assertEqual(backend.calls[0][1]["prompt"], "scene 1, <lora:ink:0.7>")
        self.assertEqual(backend.calls[0][1]["seed"], 101)
        self.assertEqual(host.written, ["comic_panel_2.png"])

    def test_krea2_t2i_family_runs_the_krea2_runner(self):
        host = _Host(_Document(2), family="krea2", model="")
        backend = _RecordingBackend()
        seen = []

        def _krea(backend_arg, operation, params, progress_callback=None, cancel_check=None):
            seen.append((operation, params))
            return GenerationResult(success=True, image_data=b"krea")

        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None), \
             mock.patch("core.krea2_generation.run_krea2_generation", side_effect=_krea):
            result = _run(host, "comic_generate_all", {"document": {}}, backend)
        self.assertTrue(result["ok"], result)
        self.assertEqual(backend.calls, [])  # 숨은 일반 체크포인트로 조용히 생성하지 않는다
        self.assertEqual([op for op, _ in seen], ["t2i", "t2i"])
        self.assertEqual(seen[0][1]["prompt"], "scene 0")
        self.assertNotIn("_generation_family", seen[0][1])

    def test_builder_validation_error_is_reported_without_starting_a_job(self):
        host = _Host(_Document(2), error="steps는 1~150 범위여야 합니다")
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "comic_generate_all", {"document": {}}, backend)
        self.assertFalse(result["ok"])
        self.assertEqual(result["mode"], "comic_generate")
        self.assertIn("steps", result["error"])
        self.assertEqual(backend.calls, [])
        self.assertIsNone(host._creator_active_job)

    def test_missing_checkpoint_on_webui_is_reported_before_generation(self):
        host = _Host(_Document(1), model="")
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "creator_generate", {"mode": "comic_panel", "panelId": "panel-1",
                                                      "document": {}}, backend)
        self.assertFalse(result["ok"])
        self.assertIn("모델", result["error"])
        self.assertEqual(backend.calls, [])

    def test_frontend_cannot_inject_its_own_t2i_snapshot(self):
        host = _Host(_Document(1))
        backend = _RecordingBackend()
        forged = {"model": "evil.ckpt", "panels": {"panel-1": {"prompt": "x", "override_settings": {"a": 1}}}}
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "comic_generate_all", {"document": {}, "_comicT2I": forged}, backend)
        self.assertTrue(result["ok"], result)
        model, sent = backend.calls[0]
        self.assertEqual(model, "anima.safetensors")
        self.assertNotIn("override_settings", sent)

    def test_comfy_workflow_controls_are_frozen_at_click_time(self):
        from backends import BackendType

        host = _Host(_Document(2))
        backend = _RecordingBackend()
        frozen = {"version": 1, "mode": "txt2img", "controls": None}
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None), \
             mock.patch("core.comfy_workflow_controls.snapshot_comfy_payload",
                        return_value={"_comfy_workflow_snapshot": frozen}) as snap:
            result = _run(host, "comic_generate_all", {"document": {}}, backend, BackendType.COMFYUI)
        self.assertTrue(result["ok"], result)
        snap.assert_called_once()
        self.assertTrue(all(sent["_comfy_workflow_snapshot"] == frozen for _m, sent in backend.calls))

    def test_worker_refuses_to_silently_fall_back_when_a_panel_snapshot_is_missing(self):
        host = _Host(_Document(2))
        backend = _RecordingBackend()
        payload = {"document": {}, "_comicT2I": {"model": "m", "panels": [
            cg.panel_entry(0, "panel-1", {"prompt": "p"}),
        ]}}
        with mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread), \
             mock.patch("backends.get_backend", return_value=backend), \
             mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads):
            host._creator_run_thread("comic_generate", host._comic_generate_all, payload)
        result = host.vue_bridge.creatorResult.values[-1]
        self.assertFalse(result["ok"])
        self.assertIn("컷 2", result["error"])
        self.assertEqual(len(backend.calls), 1)

    def test_worker_refuses_a_snapshot_whose_panel_id_does_not_match_its_index(self):
        # 인덱스만 맞고 id 가 다른 스냅샷(= 다른 컷 설정)을 조용히 쓰지 않는다
        host = _Host(_Document(1))
        backend = _RecordingBackend()
        payload = {"document": {}, "_comicT2I": {"model": "m", "panels": [
            cg.panel_entry(0, "someone-else", {"prompt": "p"}),
        ]}}
        with mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread), \
             mock.patch("backends.get_backend", return_value=backend), \
             mock.patch("core.comic_studio.panel_generation_payloads", side_effect=_panel_payloads):
            host._creator_run_thread("comic_generate", host._comic_generate_all, payload)
        result = host.vue_bridge.creatorResult.values[-1]
        self.assertFalse(result["ok"])
        self.assertIn("컷 1", result["error"])
        self.assertEqual(backend.calls, [])

    def test_t2i_hires_adetailer_and_sam3_prompt_overrides_do_not_reach_panels(self):
        class _HiresHost(_Host):
            def _build_generation_payload(self, **kwargs):
                payload, error = super()._build_generation_payload(**kwargs)
                payload.update({"hr_prompt": "T2I hires prompt: 1girl, red dress",
                                "hr_negative_prompt": "T2I hires negative"})
                payload["alwayson_scripts"].update({
                    "ADetailer": {"args": [True, False, {"ad_model": "face_yolov8n.pt",
                                                         "ad_prompt": "1girl, red eyes",
                                                         "ad_negative_prompt": "T2I face negative"}]},
                    "SAM3 Mask": {"args": [{"sam3_prompt": "face", "sam3_inpaint_prompt": "T2I inpaint",
                                            "sam3_negative_prompt": "lowres", "sam3_enable": True}]},
                })
                return payload, error

        host = _HiresHost(_Document(2))
        backend = _RecordingBackend()
        with mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
            result = _run(host, "comic_generate_all", {"document": {}}, backend)
        self.assertTrue(result["ok"], result)
        for index, (_model, sent) in enumerate(backend.calls):
            self.assertNotIn("hr_prompt", sent)
            self.assertNotIn("hr_negative_prompt", sent)
            self.assertTrue(sent["enable_hr"])  # Hires 설정 자체는 T2I 를 따른다
            slot = sent["alwayson_scripts"]["ADetailer"]["args"][2]
            self.assertEqual((slot["ad_prompt"], slot["ad_negative_prompt"]), ("", ""))
            self.assertEqual(slot["ad_model"], "face_yolov8n.pt")
            sam = sent["alwayson_scripts"]["SAM3 Mask"]["args"][0]
            self.assertEqual(sam["sam3_prompt"], "face")  # 검출 대상은 설정 — 유지
            self.assertEqual(sam["sam3_inpaint_prompt"], f"scene {index}, <lora:ink:0.7>")
            self.assertEqual(sam["sam3_negative_prompt"], sent["negative_prompt"])
        self.assertEqual(backend.calls[1][1]["alwayson_scripts"]["SAM3 Mask"]["args"][0]["sam3_negative_prompt"],
                         "lowres, text")


class _RealStudioHost(_Host):
    """실제 ComicStudio.normalize/save 를 쓰는 호스트 — UI 스레드와 워커가 각자 정규화하는 경로."""

    def __init__(self, state_path):
        from core.comic_studio import ComicStudio

        super().__init__(None)
        studio = ComicStudio(state_path)
        self._comic_studio = lambda: studio

    def _build_generation_payload(self, *, prompt_override=None, snapshot=False, comfy_workflow_snapshot=None):
        self.builder_calls.append((prompt_override, snapshot))
        return {"prompt": prompt_override, "negative_prompt": "", "steps": 30, "seed": -1,
                "width": 832, "height": 1216, "alwayson_scripts": {},
                "_chat_deferred_prompt": {"wildcards": False}}, None


def _run_real(host, action, payload, backend):
    from backends import BackendType

    with mock.patch("ui.creator_actions.threading.Thread", _ImmediateThread), \
         mock.patch("backends.get_backend", return_value=backend), \
         mock.patch("backends.get_backend_type", return_value=BackendType.WEBUI), \
         mock.patch("core.standard_hooks.run_pipeline_on_text", side_effect=lambda text: text), \
         mock.patch.object(CreatorActionsMixin, "_ensure_creator_runtime", lambda self: None):
        host._handle_creator_action(action, payload)
    return host.vue_bridge.creatorResult.values[-1]


class PanelIdentityTests(unittest.TestCase):
    """스냅샷은 컷 인덱스 + id 로 묶이고, 워커는 UI 스레드가 정규화한 문서를 그대로 쓴다."""

    def _document(self, *ids):
        prompts = ["panel one: a castle", "panel two: a dragon", "panel three: a sea"]
        return {"title": "t", "panels": [{"id": panel_id, "prompt": prompts[i], "seed": 10 + i}
                                         for i, panel_id in enumerate(ids)]}

    def test_duplicate_panel_ids_each_panel_keeps_its_own_prompt(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            host = _RealStudioHost(Path(tmp) / "comic.json")
            backend = _RecordingBackend()
            result = _run_real(host, "comic_generate_all", {"document": self._document("p", "p")}, backend)
        self.assertTrue(result["ok"], result)
        self.assertEqual([sent["prompt"] for _m, sent in backend.calls],
                         ["panel one: a castle", "panel two: a dragon"])
        self.assertEqual([sent["seed"] for _m, sent in backend.calls], [10, 11])
        self.assertEqual(host.written, ["comic_panel_1.png", "comic_panel_2.png"])
        # 겹친 id 는 결정적으로 풀려 저장·결과 문서에 실린다
        self.assertEqual([p["id"] for p in result["document"]["panels"]], ["p", "p-2"])
        self.assertEqual([p["imagePath"] for p in result["document"]["panels"]],
                         ["out/comic_panel_1.png", "out/comic_panel_2.png"])

    def test_panel_id_that_sanitizes_to_empty_generates_all_panels(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            host = _RealStudioHost(Path(tmp) / "comic.json")
            backend = _RecordingBackend()
            result = _run_real(host, "comic_generate_all", {"document": self._document("컷", "둘")}, backend)
        self.assertTrue(result["ok"], result)
        self.assertEqual([sent["prompt"] for _m, sent in backend.calls],
                         ["panel one: a castle", "panel two: a dragon"])

    def test_single_panel_by_raw_id_that_normalization_rewrites(self):
        import tempfile
        from pathlib import Path

        for ids, wanted, expected_prompt in ((("컷", "둘"), "둘", "panel two: a dragon"),
                                             (("a b", "ab"), "ab", "panel two: a dragon"),
                                             (("a b", "ab"), "a b", "panel one: a castle")):
            with self.subTest(ids=ids, wanted=wanted), tempfile.TemporaryDirectory() as tmp:
                host = _RealStudioHost(Path(tmp) / "comic.json")
                backend = _RecordingBackend()
                document = self._document(*ids)
                result = _run_real(host, "creator_generate",
                                   {"mode": "comic_panel", "panel": {"id": wanted}, "document": document},
                                   backend)
                self.assertTrue(result["ok"], result)
                self.assertEqual([sent["prompt"] for _m, sent in backend.calls], [expected_prompt])
                # 결과 panelId 는 함께 보낸 (정규화된) 문서에 실제로 있는 id
                self.assertIn(result["panelId"], [p["id"] for p in result["document"]["panels"]])
                target = next(p for p in result["document"]["panels"] if p["id"] == result["panelId"])
                self.assertEqual(target["prompt"], expected_prompt)
                self.assertTrue(target["imagePath"])
                # 결과 문서의 컷 id 는 유일하다 — 프론트가 panelId 로 엉뚱한 컷에 이미지를 붙이지 않게
                ids = [p["id"] for p in result["document"]["panels"]]
                self.assertEqual(len(ids), len(set(ids)), ids)
                self.assertEqual([p["imagePath"] != "" for p in result["document"]["panels"]],
                                 [p["prompt"] == expected_prompt for p in result["document"]["panels"]])

    def test_unknown_single_panel_id_is_rejected_without_generating(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            host = _RealStudioHost(Path(tmp) / "comic.json")
            backend = _RecordingBackend()
            result = _run_real(host, "creator_generate",
                               {"mode": "comic_panel", "panelId": "nope", "document": self._document("a", "b")},
                               backend)
        self.assertFalse(result["ok"])
        self.assertIn("찾을 수 없습니다", result["error"])
        self.assertEqual(backend.calls, [])
        self.assertEqual(host.builder_calls, [])


class PanelLookupHelperTests(unittest.TestCase):
    def test_find_panel_request_needs_both_index_and_id(self):
        entries = [cg.panel_entry(0, "p", {"prompt": "one"}), cg.panel_entry(1, "p", {"prompt": "two"})]
        self.assertEqual(cg.find_panel_request(entries, 1, "p"), {"prompt": "two"})
        self.assertEqual(cg.find_panel_request(entries, 0, "p"), {"prompt": "one"})
        self.assertIsNone(cg.find_panel_request(entries, 1, "q"))
        self.assertIsNone(cg.find_panel_request(entries, 2, "p"))
        self.assertIsNone(cg.find_panel_request({"p": {"prompt": "old dict format"}}, 0, "p"))
        self.assertIsNone(cg.find_panel_request([{"index": True, "panelId": "p", "request": {}}], 1, "p"))
        found = cg.find_panel_request(entries, 0, "p")
        found["prompt"] = "mutated"
        self.assertEqual(entries[0]["request"]["prompt"], "one")  # 사본을 돌려준다

    def test_resolve_panel_index_prefers_the_raw_request_ids(self):
        raw = {"panels": [{"id": "a b"}, {"id": "ab"}, {"id": "컷"}]}
        normalized = ["ab", "ab", "panel_1234"]
        self.assertEqual(cg.resolve_panel_index(raw, normalized, "ab"), 1)
        self.assertEqual(cg.resolve_panel_index(raw, normalized, "a b"), 0)
        self.assertEqual(cg.resolve_panel_index(raw, normalized, "컷"), 2)
        # 원문에 없으면 정규화된 id 로(이미 정규화된 문서를 다시 받은 경우)
        self.assertEqual(cg.resolve_panel_index(raw, normalized, "panel_1234"), 2)
        self.assertIsNone(cg.resolve_panel_index(raw, normalized, "zzz"))
        self.assertIsNone(cg.resolve_panel_index(raw, normalized, ""))
        # 원문 컷 수가 다르면(원문 해석 불가) 정규화된 id 만 본다
        self.assertEqual(cg.resolve_panel_index({"panels": [{"id": "ab"}]}, normalized, "ab"), 0)
        self.assertEqual(cg.resolve_panel_index(None, normalized, "ab"), 0)

    def test_unique_panel_ids_is_deterministic_idempotent_and_keeps_unique_ids(self):
        self.assertEqual(cg.unique_panel_ids(["a", "b"]), ["a", "b"])
        self.assertEqual(cg.unique_panel_ids(["p", "p", "p"]), ["p", "p-2", "p-3"])
        # 뒤에 원래 있는 id('p-2')와 부딪히지 않는다 — 원래 유일한 id 는 바꾸지 않는다
        self.assertEqual(cg.unique_panel_ids(["p", "p", "p-2"]), ["p", "p-3", "p-2"])
        for ids in (["p", "p", "p-2"], ["x", "x", "x", "x-2", "x-3"], ["a" * 80, "a" * 80]):
            once = cg.unique_panel_ids(ids)
            self.assertEqual(len(once), len(set(once)), once)
            self.assertEqual(cg.unique_panel_ids(once), once)  # 멱등 — 워커 재정규화와 어긋나지 않게
            self.assertTrue(all(len(item) <= cg.PANEL_ID_MAX for item in once))
        self.assertEqual(cg.unique_panel_ids([]), [])

    def test_unique_ids_survive_comic_studio_normalize_round_trip(self):
        import tempfile
        from pathlib import Path
        from core.comic_studio import ComicStudio

        with tempfile.TemporaryDirectory() as tmp:
            studio = ComicStudio(Path(tmp) / "comic.json")
            document = studio.normalize({"panels": [{"id": "a" * 80, "prompt": "x"}, {"id": "a" * 80, "prompt": "y"}]})
            ids = cg.unique_panel_ids([panel.id for panel in document.panels])
            for panel, panel_id in zip(document.panels, ids):
                panel.id = panel_id
            again = studio.normalize(document.to_dict())
        self.assertEqual([panel.id for panel in again.panels], ids)
        self.assertEqual(again.content_hash, document.compute_content_hash())


if __name__ == "__main__":
    unittest.main()
