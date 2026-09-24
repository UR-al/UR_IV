"""Regression coverage for standalone operations retaining generation context."""
import base64
import copy
import io
import json
import unittest
from unittest import mock

from PIL import Image

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core import anima38
from core.comfy_workflow_compiler import ComfyWorkflowCompiler
from tests.test_comfy_anima38_compiler import _anima_capabilities, _modules, V2_MODEL


def _png(width=64, height=64):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TestStandaloneComfyContext(unittest.TestCase):
    def _backend(self, scripts, *, qwen=True):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="")
        backend._last_generation_context = {
            "model_name": V2_MODEL,
            "payload": {
                "prompt": "portrait", "negative_prompt": "bad anatomy",
                "forge_additional_modules": _modules(qwen="" if not qwen else "text/qwen35_4b.safetensors"),
                "alwayson_scripts": scripts,
            },
        }
        compiler = ComfyWorkflowCompiler(_anima_capabilities(qwen=qwen))
        backend._workflow_compiler = mock.Mock(return_value=compiler)
        backend._upload_image = mock.Mock(return_value="source.png")
        backend._queue_and_wait = mock.Mock(return_value=GenerationResult(success=True, image_data=b"result"))
        return backend

    def test_all_detail_modes_preserve_bypass_without_qwen(self):
        for kind in ("adetailer", "sam3", "refine"):
            with self.subTest(kind=kind):
                backend = self._backend({anima38.SCRIPT_NAME: {"args": [{"bypass": True}]}}, qwen=False)
                before = copy.deepcopy(backend._last_generation_context)
                backend._standalone_detail(_png(), {}, kind)
                graph = backend._queue_and_wait.call_args.args[0]
                self.assertNotIn("ForgeNeoAnimaQwen35Loader", {n["class_type"] for n in graph.values()})
                self.assertEqual(backend._last_generation_context, before)

    def test_standalone_detail_does_not_inherit_previous_generation_eye_pass(self):
        for kind in ("adetailer", "sam3", "refine"):
            with self.subTest(kind=kind):
                backend = self._backend({})
                backend._last_generation_context["payload"]["_comfy_detail_passes"] = ["eyes"]
                before = copy.deepcopy(backend._last_generation_context)
                getattr(backend, kind)(_png(), {"sam3_prompt": "face"})
                graph = backend._queue_and_wait.call_args.args[0]
                targets = [node["inputs"]["prompt"] for node in graph.values()
                           if node["class_type"] == "ForgeNeoSAM3Mask"]
                self.assertEqual(targets, [] if kind == "adetailer" else ["face"])
                self.assertEqual(backend._last_generation_context, before)

    def test_standalone_sam3_samples_only_masked_at_the_input_image_size(self):
        # Forge 단독 SAM3/Refine 은 width/height = 입력 이미지 크기로 보내고(webui_backend
        # sam3()/refine()), 확장은 p.width/p.height 로 샘플링한다. 직전 txt2img 의 크기·종횡비
        # (Hires 이전 기본 크기 포함)나 세션 이력이 결과를 바꾸면 안 된다.
        previous_generations = (
            {},  # 생성 크기를 모름
            {"width": 832, "height": 1216},  # 같은 종횡비, Hires 이전 기본 크기
            {"width": 1216, "height": 832},  # 가로 생성 뒤 세로 이미지
            {"width": 1024, "height": 1536,
             "_sam3_processing_width": 1024, "_sam3_processing_height": 1536},  # 남은 키
        )
        for kind in ("sam3", "refine"):
            for previous in previous_generations:
                with self.subTest(kind=kind, previous=previous):
                    backend = self._backend({})
                    backend._last_generation_context["payload"].update(previous)
                    before = copy.deepcopy(backend._last_generation_context)
                    getattr(backend, kind)(_png(96, 128), {"sam3_prompt": "face"})
                    graph = backend._queue_and_wait.call_args.args[0]
                    detail = next(node for node in graph.values() if node["class_type"] in {
                        "ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine",
                    })
                    self.assertEqual(
                        (detail["inputs"]["target_width"], detail["inputs"]["target_height"]),
                        (96, 128),
                    )
                    self.assertEqual(backend._last_generation_context, before)

    def test_standalone_detail_never_leaves_a_comfy_output_copy(self):
        # 직전 메인 생성 payload 는 save_images=True 지만 단독 후처리는 Forge 처럼 False.
        for kind in ("adetailer", "sam3", "refine"):
            with self.subTest(kind=kind):
                backend = self._backend({})
                backend._last_generation_context["payload"]["save_images"] = True
                getattr(backend, kind)(_png(), {"sam3_prompt": "face"})
                graph = backend._queue_and_wait.call_args.args[0]
                classes = [node["class_type"] for node in graph.values()]
                self.assertIn("PreviewImage", classes)
                self.assertNotIn("SaveImage", classes)
                self.assertTrue(backend._last_generation_context["payload"]["save_images"])

    def test_negative_semantic_survives_but_old_image_passes_do_not(self):
        backend = self._backend({
            anima38.SCRIPT_NAME: {"args": [{"negative": True}]},
            "SAM3 Mask": {"args": [{"sam3_enabled": True}]},
        })
        backend._standalone_detail(_png(), {}, "adetailer")
        graph = backend._queue_and_wait.call_args.args[0]
        classes = [n["class_type"] for n in graph.values()]
        self.assertEqual(classes.count("ForgeNeoAnima38V2Prompt"), 2)
        self.assertNotIn("ForgeNeoSAM3Detailer", classes)

    def test_explicit_script_overrides_are_used_without_mutating_settings(self):
        backend = self._backend({}, qwen=False)
        settings = {"alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [{"bypass": True}]}}}
        before = copy.deepcopy(settings)
        backend._standalone_detail(_png(), settings, "adetailer")
        self.assertEqual(settings, before)

    def test_persisted_semantic_and_shift_survive_a_new_backend_instance(self):
        backend = ComfyUIBackend("http://127.0.0.1:1")
        scripts = {anima38.SCRIPT_NAME: {"args": [{"bypass": True}]}}
        saved = {"model": V2_MODEL, "alwayson_scripts": scripts, "shift": "3.0"}
        with mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(saved))):
            _, payload = backend._saved_generation_context({})
        self.assertEqual(payload.get("alwayson_scripts"), scripts)
        self.assertEqual(payload.get("distilled_cfg_scale"), 3.0)

    def test_upscale_supplies_input_dimensions_for_factor_mode(self):
        backend = self._backend({})
        compiler = mock.Mock()
        backend._workflow_compiler.return_value = compiler
        settings = {"upscaler_name": "4x-UltraSharp.pth", "scale_mode": "factor", "scale_factor": 2}
        backend.upscale("data:image/png;base64," + _png(80, 48), settings)
        compiler.compile_upscale.assert_called_once_with(
            "source.png", settings, source_width=80, source_height=48,
        )

    def test_invalid_upscale_image_fails_before_upload(self):
        backend = self._backend({})
        with self.assertRaises(ValueError):
            backend.upscale("not an image", {})
        backend._upload_image.assert_not_called()

    def test_upscale_dimensions_follow_comfy_exif_orientation(self):
        buffer = io.BytesIO()
        photo = Image.new("RGB", (80, 48))
        exif = Image.Exif()
        exif[274] = 6
        photo.save(buffer, format="JPEG", exif=exif)
        backend = self._backend({})
        compiler = mock.Mock()
        backend._workflow_compiler.return_value = compiler
        backend.upscale(base64.b64encode(buffer.getvalue()).decode("ascii"), {})
        self.assertEqual(compiler.compile_upscale.call_args.kwargs,
                         {"source_width": 48, "source_height": 80})
