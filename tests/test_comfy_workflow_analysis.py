"""The workflow picker must recognise the graphs produced by our compiler."""
import json
import unittest
from unittest import mock

from backends.comfyui_backend import analyze_workflow
from core import anima38
from core.comfy_workflow_compiler import ComfyWorkflowCompiler
from tests.test_comfy_anima38_compiler import _anima_capabilities, _modules, V2_MODEL
from tests.test_comfy_workflow_compiler import _capabilities


def _analyze(graph):
    with mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock.mock_open(read_data=json.dumps(graph)),
    ):
        return analyze_workflow("in-memory-workflow.json")


class TestComfyWorkflowAnalysis(unittest.TestCase):
    def test_bundled_anima_graph_with_two_semantic_encoders(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile("txt2img", V2_MODEL, {
            "prompt": "portrait", "negative_prompt": "bad",
            "forge_additional_modules": _modules(), "width": 768, "height": 512,
            "distilled_cfg_scale": 3,
            "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [{"negative": True}]}},
        })
        result = _analyze(graph)
        self.assertTrue(result["valid"], result["error"])
        self.assertEqual(result["ksampler_type"], "ForgeNeoKSamplerCNS")
        self.assertEqual(result["checkpoint"], V2_MODEL)
        self.assertEqual(result["classification"], "native_unet")
        self.assertFalse(result["is_locked"])
        self.assertEqual(result["model_param"], "model_name")
        self.assertEqual((result["width"], result["height"]), (768, 512))

    def test_default_checkpoint_graph_is_still_valid(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {"prompt": "cat"},
        )
        result = _analyze(graph)
        self.assertTrue(result["valid"], result["error"])
        self.assertEqual(result["classification"], "native_checkpoint")

    def test_web_graph_is_summarised_but_rejected_with_export_api_guidance(self):
        # 편집기 저장(웹) 포맷은 실행할 수 없다 — 요약은 보여 주되 선택 단계에서
        # 'Export (API)' 로 저장하라고 알린다(생성에서 매번 실패하던 것을 앞당김).
        result = _analyze({"nodes": [
            {"id": 1, "type": "ForgeNeoAnima38V2Loader", "widgets_values": [V2_MODEL]},
            {"id": 2, "type": "ForgeNeoAnima38V2Prompt", "widgets_values": ["portrait"]},
            {"id": 3, "type": "ForgeNeoLatentInput", "widgets_values": ["txt2img", 768, 512, 1]},
            {"id": 4, "type": "ForgeNeoKSamplerCNS", "widgets_values": []},
            {"id": 5, "type": "SaveImage", "widgets_values": []},
        ]})
        self.assertFalse(result["valid"])
        self.assertEqual(result["format"], "web")
        self.assertIn("Export (API)", result["error"])
        self.assertEqual(result["checkpoint"], V2_MODEL)
        self.assertEqual((result["width"], result["height"]), (768, 512))
        self.assertEqual(result["node_count"], 5)

    def test_structure_the_compiler_always_rejects_is_reported_as_a_blocker(self):
        base = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "p"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "n"}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": 1, "steps": 20, "cfg": 7,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            }},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
        }
        clean = _analyze(base)
        self.assertTrue(clean["valid"], clean["error"])
        self.assertIsNone(clean["generation_blocker"])
        self.assertTrue(clean["model_selectable"])
        self.assertFalse(clean["is_locked"])

        two_pass = json.loads(json.dumps(base))
        two_pass["8"] = {"class_type": "KSampler", "inputs": {
            **base["5"]["inputs"], "latent_image": ["5", 0], "denoise": 0.5,
        }}
        two_pass["6"]["inputs"]["samples"] = ["8", 0]
        result = _analyze(two_pass)
        self.assertTrue(result["valid"])  # 필수 노드는 다 있다
        self.assertIn("sampler 노드가 여러 개", result["generation_blocker"])

        custom = json.loads(json.dumps(base))
        custom["5"]["class_type"] = "SamplerCustom"
        result = _analyze(custom)
        self.assertIn("SamplerCustom", result["generation_blocker"])

    def test_custom_model_loader_is_reported_as_workflow_owned_model(self):
        graph = {
            "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "anima-q8.gguf"}},
            "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen.safetensors", "type": "stable_diffusion"}},
            "10": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "p"}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "n"}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": 1, "steps": 20, "cfg": 7,
                "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
            }},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["10", 0]}},
            "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
        }
        result = _analyze(graph)
        self.assertTrue(result["valid"], result["error"])
        self.assertIsNone(result["generation_blocker"])
        self.assertFalse(result["model_selectable"])
        self.assertTrue(result["is_locked"])

    def test_kj_and_legacy_checkpoint_loaders_are_model_selectable(self):
        for loader, key in (("DiffusionModelLoaderKJ", "model_name"), ("CheckpointLoader", "ckpt_name")):
            with self.subTest(loader=loader):
                graph = {
                    "1": {"class_type": loader, "inputs": {key: "model.safetensors"}},
                    "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors", "type": "stable_diffusion"}},
                    "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "p"}},
                    "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["9", 0], "text": "n"}},
                    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
                    "5": {"class_type": "KSampler", "inputs": {
                        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                        "latent_image": ["4", 0], "seed": 1, "steps": 20, "cfg": 7,
                        "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                    }},
                    "7": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
                }
                result = _analyze(graph)
                self.assertTrue(result["model_selectable"], result)
                self.assertFalse(result["is_locked"])

    def test_missing_sampler_is_not_made_valid(self):
        result = _analyze({"1": {"class_type": "SaveImage", "inputs": {}}})
        self.assertFalse(result["valid"])
        self.assertIn("KSampler", result["error"])

    def test_malformed_node_inputs_are_reported_not_raised(self):
        # The connect handler and the startup picker call analyze_workflow
        # unguarded: a node whose "inputs" is null or a list must not raise.
        from core.comfy_workflow_structure import describe_generation_structure

        for malformed in (None, [], "x"):
            with self.subTest(inputs=malformed):
                graph = {
                    "1": {"class_type": "CheckpointLoaderSimple", "inputs": malformed},
                    "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "p"}},
                    "4": {"class_type": "EmptyLatentImage", "inputs": malformed},
                    "5": {"class_type": "KSampler", "inputs": malformed},
                    "7": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
                }
                result = _analyze(graph)
                self.assertEqual(result["ksampler_type"], "KSampler")
                self.assertFalse(result["model_selectable"])
                self.assertTrue(result["generation_blocker"])
                structure = describe_generation_structure(graph)
                self.assertFalse(structure["model_selectable"])
                self.assertTrue(structure["generation_blocker"])
