"""Custom workflow model loaders, bundled Anima LoRA nodes and API-only format.

감사 #119: the bundled ForgeNeoLoraBlockWeight / ForgeNeoAnimaLoraLoaderModelOnly
nodes are usable in an Anima custom workflow.
감사 #157: the picker and the compiler agree — web-format JSON is refused with
one instruction everywhere, loaders the picker calls "model selectable" are
rewritten, and a workflow with a custom (locked) loader keeps its own model.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from backends.comfyui_backend import ComfyUIBackend
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_format import WEB_WORKFLOW_MESSAGE, WorkflowFormatError, require_api_workflow
from tests.test_comfy_anima38_compiler import (
    NATIVE_CLIP, V1_MODEL, VAE, _anima_capabilities, _split_custom_workflow,
)
from tests.test_comfy_workflow_compiler import _capabilities, _choice, _custom_workflow


def _anima_pack_capabilities() -> dict:
    capabilities = _anima_capabilities()
    for name in ("ForgeNeoLoraBlockWeight", "ForgeNeoAnimaLoraLoaderModelOnly"):
        capabilities[name] = {"input": {"required": {}}}  # class-only probe
    return capabilities


def _block_weight(model, clip, lora="styles/ink.safetensors"):
    return {"class_type": "ForgeNeoLoraBlockWeight", "inputs": {
        "model": model, "clip": clip, "enabled": True, "lora_name": lora,
        "strength_model": 0.7, "strength_clip": 0.7, "inverse": False, "seed": 0,
        "A": 4.0, "B": 1.0, "preset": "Preset", "block_vector": "",
    }}


class AnimaBundledLoraNodeTests(unittest.TestCase):
    def test_block_weight_and_model_only_nodes_pass_through_unchanged(self):
        workflow = _split_custom_workflow()
        workflow["11"] = _block_weight(["1", 0], ["8", 0])
        workflow["12"] = {"class_type": "ForgeNeoAnimaLoraLoaderModelOnly", "inputs": {
            "model": ["11", 0], "lora_name": "characters/alice.safetensors", "strength_model": 0.5,
        }}
        workflow["13"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["12", 0], "clip": ["11", 1], "lora_name": "styles/ink.safetensors",
            "strength_model": 0.4, "strength_clip": 0.4,
        }}
        workflow["5"]["inputs"]["model"] = ["13", 0]
        workflow["2"]["inputs"]["clip"] = ["13", 1]
        workflow["3"]["inputs"]["clip"] = ["13", 1]
        graph = ComfyWorkflowCompiler(_anima_pack_capabilities()).compile(
            "txt2img", V1_MODEL, {"prompt": "portrait"}, workflow=workflow,
        )
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoLoraBlockWeight")
        self.assertEqual(graph["12"]["class_type"], "ForgeNeoAnimaLoraLoaderModelOnly")
        # Core LoraLoader in the same chain is still remapped to the Anima loader.
        self.assertEqual(graph["13"]["class_type"], "ForgeNeoAnimaLoraLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], V1_MODEL)

    def test_text_encoder_override_rebases_the_block_weight_clip_chain(self):
        workflow = _split_custom_workflow()
        workflow["11"] = _block_weight(["1", 0], ["8", 0])
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        graph = ComfyWorkflowCompiler(_anima_pack_capabilities()).compile(
            "txt2img", V1_MODEL, {"forge_additional_modules": [VAE, NATIVE_CLIP]}, workflow=workflow,
        )
        clip_loader = next(node_id for node_id, node in graph.items()
                           if node["class_type"] == "CLIPLoader" and node_id != "8")
        # The new text encoder feeds the block-weight LoRA, whose CLIP feeds the prompts.
        self.assertEqual(graph["11"]["inputs"]["clip"], [clip_loader, 0])
        self.assertEqual(graph["2"]["inputs"]["clip"], ["11", 1])
        self.assertEqual(graph["3"]["inputs"]["clip"], ["11", 1])

    def test_block_weight_shared_with_an_inactive_branch_is_rejected(self):
        workflow = _split_custom_workflow()
        workflow["11"] = _block_weight(["1", 0], ["8", 0])
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        workflow["12"] = {"class_type": "SaveImage", "inputs": {
            "images": ["6", 0], "filename_prefix": "x", "debug": ["11", 1],
        }}
        with self.assertRaisesRegex(WorkflowCompileError, "공유"):
            ComfyWorkflowCompiler(_anima_pack_capabilities()).compile(
                "txt2img", V1_MODEL, {}, workflow=workflow,
            )

    def test_unknown_lora_nodes_still_fail_with_the_supported_list(self):
        workflow = _split_custom_workflow()
        workflow["11"] = {"class_type": "Power Lora Loader (rgthree)", "inputs": {"model": ["1", 0]}}
        workflow["5"]["inputs"]["model"] = ["11", 0]
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoLoraBlockWeight"):
            ComfyWorkflowCompiler(_anima_pack_capabilities()).compile(
                "txt2img", V1_MODEL, {}, workflow=workflow,
            )


class CustomWorkflowLoaderTests(unittest.TestCase):
    def test_kj_and_legacy_checkpoint_loaders_receive_the_selected_model(self):
        for loader, key in (("DiffusionModelLoaderKJ", "model_name"), ("CheckpointLoader", "ckpt_name")):
            with self.subTest(loader=loader):
                capabilities = _capabilities()
                capabilities[loader] = {"input": {"required": {key: _choice("old.safetensors", "new.safetensors")}}}
                workflow = _custom_workflow()
                workflow["1"] = {"class_type": loader, "inputs": {key: "old.safetensors"}}
                if loader == "DiffusionModelLoaderKJ":
                    workflow["8"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "text/base.safetensors", "type": "stable_diffusion"}}
                    workflow["9"] = {"class_type": "VAELoader", "inputs": {"vae_name": "vae/image_vae.safetensors"}}
                    workflow["2"]["inputs"]["clip"] = ["8", 0]
                    workflow["3"]["inputs"]["clip"] = ["8", 0]
                    workflow["6"]["inputs"]["vae"] = ["9", 0]
                graph = ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "new.safetensors", {}, workflow=workflow,
                )
                self.assertEqual(graph["1"]["inputs"][key], "new.safetensors")

    def test_custom_loader_keeps_its_model_and_identifies_the_family(self):
        workflow = _split_custom_workflow()
        workflow["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "anima-preview-q8.gguf"}}
        workflow["11"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["8", 0], "lora_name": "styles/ink.safetensors",
            "strength_model": 0.5, "strength_clip": 0.5,
        }}
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        capabilities = _anima_capabilities()
        capabilities["UnetLoaderGGUF"] = {"input": {"required": {"unet_name": _choice("anima-preview-q8.gguf")}}}
        # The UI still sends its (disabled) model combo value.
        graph = ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", "checkpoint.safetensors", {"prompt": "portrait, <lora:alice:0.3>"}, workflow=workflow,
        )
        self.assertEqual(graph["1"], workflow["1"])
        # The workflow's own Anima model drives the Anima LoRA path.
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoAnimaLoraLoader")
        inserted = [node for node in graph.values()
                    if node["class_type"] == "ForgeNeoAnimaLoraLoader"
                    and node["inputs"].get("lora_name") == "characters/alice.safetensors"]
        self.assertEqual(len(inserted), 1)

    def _gguf_lora_workflow(self, on_disk: str) -> dict:
        workflow = _split_custom_workflow()
        workflow["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": on_disk}}
        workflow["11"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["8", 0], "lora_name": "styles/ink.safetensors",
            "strength_model": 0.5, "strength_clip": 0.5,
        }}
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        return workflow

    def _compile_with_model_control(self, on_disk: str, override: str,
                                    selected: str = "checkpoint.safetensors") -> dict:
        from core.comfy_workflow_controls import describe_controls

        workflow = self._gguf_lora_workflow(on_disk)
        capabilities = _anima_capabilities()
        capabilities["UnetLoaderGGUF"] = {"input": {"required": {
            "unet_name": _choice("sdxl-q8.gguf", "anima-preview-q8.gguf"),
        }}}
        schema = describe_controls(workflow, capabilities)
        binding = {
            "workflowFingerprint": schema["workflowFingerprint"],
            "schemaFingerprint": schema["schemaFingerprint"],
            # A locked loader's model is not app-managed: the user may pick it.
            "overrides": [{"nodeId": "1", "classType": "UnetLoaderGGUF",
                           "name": "unet_name", "value": override}],
        }
        return ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", selected, {"prompt": "portrait, <lora:alice:0.3>"},
            workflow=workflow, workflow_controls=binding,
        )

    def test_workflow_control_model_override_decides_the_family(self):
        # 감사 #157: the family comes from the model the queued graph loads,
        # not from the file on disk the control replaces.
        graph = self._compile_with_model_control("sdxl-q8.gguf", "anima-preview-q8.gguf")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "anima-preview-q8.gguf")
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoAnimaLoraLoader")
        self.assertTrue(any(
            node["class_type"] == "ForgeNeoAnimaLoraLoader"
            and node["inputs"].get("lora_name") == "characters/alice.safetensors"
            for node in graph.values()
        ))

        graph = self._compile_with_model_control("anima-preview-q8.gguf", "sdxl-q8.gguf")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "sdxl-q8.gguf")
        self.assertEqual(graph["11"]["class_type"], "LoraLoader")
        self.assertFalse(any(node["class_type"] == "ForgeNeoAnimaLoraLoader" for node in graph.values()))

    # 모델 선택이 비어도(.gguf 만 있는 ComfyUI 는 모델 콤보가 비고, ComfyUI 는 빈 선택을 허용)
    # 워크플로 자신의 모델로 계열을 판별한다 — 예전엔 판별을 건너뛰어 Anima 워크플로가 core
    # LoraLoader(블록 remap 없음)로 남았다.
    _PROMPT = {"prompt": "portrait, <lora:alice:0.3>"}

    @staticmethod
    def _gguf_capabilities() -> dict:
        capabilities = _anima_capabilities()
        capabilities["UnetLoaderGGUF"] = {"input": {"required": {
            "unet_name": _choice("sdxl-q8.gguf", "anima-preview-q8.gguf"),
        }}}
        return capabilities

    @staticmethod
    def _lora_classes(graph: dict) -> list[str]:
        return [node["class_type"] for node in graph.values()
                if node["class_type"] in {"LoraLoader", "ForgeNeoAnimaLoraLoader"}]

    def test_locked_anima_loader_identifies_the_family_without_a_selection(self):
        workflow = self._gguf_lora_workflow("anima-preview-q8.gguf")
        graph = ComfyWorkflowCompiler(self._gguf_capabilities()).compile(
            "txt2img", "", dict(self._PROMPT), workflow=workflow,
        )
        self.assertEqual(graph["1"], workflow["1"])
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoAnimaLoraLoader")
        inserted = [node for node in graph.values()
                    if node["inputs"].get("lora_name") == "characters/alice.safetensors"]
        self.assertEqual([node["class_type"] for node in inserted], ["ForgeNeoAnimaLoraLoader"])

    def test_locked_non_anima_loader_keeps_core_lora_loader_without_a_selection(self):
        workflow = self._gguf_lora_workflow("sdxl-q8.gguf")
        graph = ComfyWorkflowCompiler(self._gguf_capabilities()).compile(
            "txt2img", "", dict(self._PROMPT), workflow=workflow,
        )
        self.assertEqual(graph["1"], workflow["1"])
        self.assertEqual(graph["11"]["class_type"], "LoraLoader")
        self.assertNotIn("ForgeNeoAnimaLoraLoader", self._lora_classes(graph))

    def test_workflow_control_override_decides_the_family_without_a_selection(self):
        graph = self._compile_with_model_control("sdxl-q8.gguf", "anima-preview-q8.gguf", selected="")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "anima-preview-q8.gguf")
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoAnimaLoraLoader")

    def test_rewritable_anima_loader_uses_its_own_model_without_a_selection(self):
        workflow = _split_custom_workflow()  # UNETLoader(V1_MODEL)
        workflow["11"] = {"class_type": "LoraLoader", "inputs": {
            "model": ["1", 0], "clip": ["8", 0], "lora_name": "styles/ink.safetensors",
            "strength_model": 0.5, "strength_clip": 0.5,
        }}
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        empty = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", "", {**self._PROMPT, "seed": 5}, workflow=workflow,
        )
        self.assertEqual(empty["1"], workflow["1"])
        self.assertEqual(empty["11"]["class_type"], "ForgeNeoAnimaLoraLoader")
        self.assertEqual(set(self._lora_classes(empty)), {"ForgeNeoAnimaLoraLoader"})
        # 워크플로 자신의 모델을 고른 것과 같은 그래프.
        selected = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V1_MODEL, {**self._PROMPT, "seed": 5}, workflow=workflow,
        )
        self.assertEqual(empty, selected)


class SharedTextEncoderTests(unittest.TestCase):
    """A negative derived from the positive encoder (ConditioningZeroOut)."""

    def _workflow(self) -> dict:
        workflow = _custom_workflow()
        workflow["3"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
        return workflow

    def _capabilities(self) -> dict:
        capabilities = _capabilities()
        capabilities["ConditioningZeroOut"] = {"input": {"required": {"conditioning": ["CONDITIONING"]}}}
        return capabilities

    def test_positive_prompt_is_kept_when_the_negative_is_empty(self):
        workflow = self._workflow()
        graph = ComfyWorkflowCompiler(self._capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {"prompt": "castle", "negative_prompt": ""},
            workflow=workflow,
        )
        self.assertEqual(graph["2"]["inputs"]["text"], "castle")
        self.assertEqual(graph["3"], workflow["3"])

    def test_a_negative_prompt_without_its_own_encoder_is_an_explicit_error(self):
        with self.assertRaisesRegex(WorkflowCompileError, "ConditioningZeroOut"):
            ComfyWorkflowCompiler(self._capabilities()).compile(
                "txt2img", "checkpoint.safetensors",
                {"prompt": "castle", "negative_prompt": "blurry"}, workflow=self._workflow(),
            )


class ApiFormatOnlyTests(unittest.TestCase):
    def test_require_api_workflow(self):
        self.assertEqual(require_api_workflow({"1": {}}), {"1": {}})
        with self.assertRaisesRegex(WorkflowFormatError, "Export \\(API\\)"):
            require_api_workflow({"nodes": [], "links": []})
        with self.assertRaises(WorkflowFormatError):
            require_api_workflow([1, 2])

    def test_backend_configured_workflow_refuses_web_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "web.json"
            path.write_text(json.dumps({"nodes": [{"id": 1, "type": "KSampler"}], "links": []}), encoding="utf-8")
            backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path=str(path))
            with mock.patch("core.path_safety.safe_config_file", return_value=str(path)):
                with self.assertRaisesRegex(RuntimeError, "Export \\(API\\)"):
                    backend._load_configured_workflow("txt2img")
                backend._workflow_compiler = mock.Mock()
                result = backend.txt2img("model.safetensors", {})
            self.assertFalse(result.success)
            self.assertEqual(result.error, WEB_WORKFLOW_MESSAGE)
            backend._workflow_compiler.assert_not_called()

    def test_generation_api_profile_refuses_web_format(self):
        from core.generation_api import GenerationApiManager

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "web.json"
            path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Export \\(API\\)"):
                GenerationApiManager._load_workflow(str(path))

    def test_generate_workflow_refuses_web_format_without_uploading(self):
        backend = ComfyUIBackend("http://127.0.0.1:1")
        backend._upload_image = mock.Mock()
        backend.run_workflow = mock.Mock()
        result = backend.generate_workflow("img2img", {"nodes": []}, "", {"init_images": ["x"]})
        self.assertFalse(result.success)
        self.assertEqual(result.error, WEB_WORKFLOW_MESSAGE)
        backend._upload_image.assert_not_called()
        backend.run_workflow.assert_not_called()

    def test_legacy_converters_are_gone(self):
        for name in ("_convert_web_to_api", "_map_widget_values", "_apply_params",
                     "_find_ksampler_node", "_trace_clip_nodes", "_find_clip_encode_node",
                     "_load_workflow", "_load_img2img_workflow"):
            self.assertFalse(hasattr(ComfyUIBackend, name), name)


if __name__ == "__main__":
    unittest.main()
