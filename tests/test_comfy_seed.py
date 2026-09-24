"""Forge-style random seed on the ComfyUI path (감사 #48).

seed=-1 must become ONE concrete seed per job: base sampler, Hires.fix,
ADetailer and SAM3 all use it, the queued graph replays exactly, and the
backend reports it as info['seed'] for the viewer, '시드 탐색' and gen_stats.
"""
from __future__ import annotations

import base64
import io
import json
import unittest
from unittest import mock

from PIL import Image

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core import sam3_args
from core.comfy_seed import (
    RANDOM_SEED_MAX, concrete_seed, graph_main_seed, with_concrete_seed,
)
from core.comfy_workflow_compiler import ComfyWorkflowCompiler
from tests.test_comfy_workflow_compiler import _capabilities, _custom_workflow, _node


def _png(width=64, height=64) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _full_payload(seed=-1, **sam):
    return {
        "prompt": "portrait", "negative_prompt": "bad", "seed": seed,
        "enable_hr": True, "hr_scale": 1.5, "hr_seed_delta": 0,
        "alwayson_scripts": {
            "ADetailer": {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt"}]},
            **sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam}),
        },
        "_comfy_detail_passes": ["eyes"],
    }


def _seeds(graph: dict) -> dict:
    found = {}
    for node_id, node in graph.items():
        cls = node["class_type"]
        if cls == "ForgeNeoADetailer":
            found[f"{cls}:{node_id}"] = json.loads(node["inputs"]["settings_json"])["seed"]
        elif "seed" in node["inputs"] and not isinstance(node["inputs"]["seed"], list):
            found[f"{cls}:{node_id}"] = node["inputs"]["seed"]
        elif "noise_seed" in node["inputs"]:
            found[f"{cls}:{node_id}"] = node["inputs"]["noise_seed"]
    return found


class ConcreteSeedTests(unittest.TestCase):
    def test_negative_and_garbage_draw_once_explicit_values_pass_through(self):
        draw = mock.Mock(return_value=1234)
        self.assertEqual(concrete_seed(-1, randint=draw), 1234)
        draw.assert_called_once_with(0, RANDOM_SEED_MAX)
        self.assertEqual(concrete_seed("junk", randint=lambda a, b: 7), 7)
        self.assertEqual(concrete_seed("1e400", randint=lambda a, b: 8), 8)
        self.assertEqual(concrete_seed(42), 42)
        self.assertEqual(concrete_seed("42"), 42)
        # uint64 seeds stay exact (no float round trip).
        self.assertEqual(concrete_seed("18446744073709551615"), 18446744073709551615)
        self.assertEqual(concrete_seed(2**70), 0xFFFFFFFFFFFFFFFF)

    def test_with_concrete_seed_copies_and_never_mutates_the_caller(self):
        payload = {"seed": -1, "prompt": "x"}
        resolved = with_concrete_seed(payload)
        self.assertEqual(payload["seed"], -1)
        self.assertGreaterEqual(resolved["seed"], 0)
        self.assertEqual(resolved["prompt"], "x")

    def test_graph_main_seed_reads_the_single_sampler_only(self):
        self.assertEqual(graph_main_seed({"1": {"class_type": "KSampler", "inputs": {"seed": 5}}}), 5)
        self.assertEqual(graph_main_seed({"1": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 6}}}), 6)
        self.assertIsNone(graph_main_seed({"1": {"class_type": "SaveImage", "inputs": {}}}))
        self.assertIsNone(graph_main_seed({
            "1": {"class_type": "KSampler", "inputs": {"seed": 1}},
            "2": {"class_type": "KSampler", "inputs": {"seed": 2}},
        }))
        self.assertIsNone(graph_main_seed({"1": {"class_type": "KSampler", "inputs": {"seed": ["9", 0]}}}))


class CompilerSeedTests(unittest.TestCase):
    def test_random_seed_is_resolved_once_for_every_pass(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", _full_payload(-1),
        )
        seeds = _seeds(graph)
        classes = {key.split(":")[0] for key in seeds}
        self.assertTrue({
            "ForgeNeoKSamplerCNS", "ForgeNeoHiresFix", "ForgeNeoADetailer",
            "ForgeNeoSAM3Mask", "ForgeNeoSAM3Detailer",
        } <= classes, seeds)
        self.assertEqual(len(set(seeds.values())), 1, seeds)
        (value,) = set(seeds.values())
        self.assertGreaterEqual(value, 0)
        # The second (eyes) SAM3 pass reuses the same seed.
        self.assertEqual(sum(1 for key in seeds if key.startswith("ForgeNeoSAM3Detailer")), 2)
        self.assertEqual(graph_main_seed(graph), value)

    def test_explicit_seed_and_hires_delta_are_kept(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {**_full_payload(77), "hr_seed_delta": 3},
        )
        _id, hires = _node(graph, "ForgeNeoHiresFix")
        self.assertEqual(hires["inputs"]["seed"], 77)
        self.assertEqual(hires["inputs"]["seed_delta"], 3)
        self.assertEqual(set(_seeds(graph).values()), {77})

    def test_sam3_own_seed_only_with_use_seed_and_shared_by_mask_and_detailer(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors",
            _full_payload(10, sam3_use_seed=True, sam3_seed=99),
        )
        _mask_id, mask = _node(graph, "ForgeNeoSAM3Mask")
        _detail_id, detail = _node(graph, "ForgeNeoSAM3Detailer")
        self.assertEqual((mask["inputs"]["seed"], detail["inputs"]["seed"]), (99, 99))
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["seed"], 10)

        random_own = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors",
            _full_payload(10, sam3_use_seed=True, sam3_seed=-1),
        )
        _mask_id, mask = _node(random_own, "ForgeNeoSAM3Mask")
        _detail_id, detail = _node(random_own, "ForgeNeoSAM3Detailer")
        self.assertGreaterEqual(detail["inputs"]["seed"], 0)
        self.assertEqual(mask["inputs"]["seed"], detail["inputs"]["seed"])

    def test_custom_workflow_advanced_sampler_gets_the_concrete_seed(self):
        workflow = _custom_workflow()
        workflow["5"]["class_type"] = "KSamplerAdvanced"
        workflow["5"]["inputs"].pop("seed")
        workflow["5"]["inputs"]["noise_seed"] = 1
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {"seed": -1}, workflow=workflow,
        )
        self.assertGreaterEqual(graph["5"]["inputs"]["noise_seed"], 0)
        self.assertEqual(graph_main_seed(graph), graph["5"]["inputs"]["noise_seed"])

    def test_postprocess_passes_share_one_seed(self):
        payload = _full_payload(-1)
        payload.pop("enable_hr")
        graph = ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
            "checkpoint.safetensors", payload, uploaded_image="source.png",
        )
        seeds = set(_seeds(graph).values())
        self.assertEqual(len(seeds), 1)
        self.assertGreaterEqual(next(iter(seeds)), 0)


class BackendSeedReportTests(unittest.TestCase):
    def _backend(self):
        backend = ComfyUIBackend("http://127.0.0.1:1", workflow_path="", img2img_workflow_path="")
        backend._workflow_compiler = mock.Mock(return_value=ComfyWorkflowCompiler(_capabilities()))
        backend._queue_and_wait = mock.Mock(return_value=GenerationResult(success=True, image_data=b"img"))
        return backend

    def test_txt2img_reports_the_queued_seed_and_keeps_the_caller_payload(self):
        backend = self._backend()
        payload = _full_payload(-1)
        result = backend.txt2img("checkpoint.safetensors", payload)
        self.assertTrue(result.success, result.error)
        graph = backend._queue_and_wait.call_args.args[0]
        self.assertEqual(result.info["seed"], graph_main_seed(graph))
        self.assertGreaterEqual(result.info["seed"], 0)
        self.assertEqual(payload["seed"], -1)
        # The saved context stays "random" so a later standalone pass draws anew.
        self.assertEqual(backend._last_generation_context["payload"]["seed"], -1)

    def test_img2img_reports_explicit_seed(self):
        backend = self._backend()
        backend._upload_image = mock.Mock(return_value="source.png")
        result = backend.img2img("checkpoint.safetensors", {"init_images": [_png()], "seed": 321})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.info["seed"], 321)

    def test_failed_generation_is_not_given_a_seed(self):
        backend = self._backend()
        backend._queue_and_wait.return_value = GenerationResult(success=False, error="boom")
        result = backend.txt2img("checkpoint.safetensors", {"seed": -1})
        self.assertFalse(result.success)
        self.assertNotIn("seed", result.info)

    def test_standalone_detail_resolves_one_seed_for_all_passes(self):
        backend = self._backend()
        backend._upload_image = mock.Mock(return_value="source.png")
        backend._last_generation_context = {
            "model_name": "checkpoint.safetensors",
            "payload": {"prompt": "p", "seed": -1, "alwayson_scripts": {}},
        }
        backend.sam3(_png(), {"sam3_prompt": "face", "sam3_mode": "Inpaint"})
        graph = backend._queue_and_wait.call_args.args[0]
        seeds = set(_seeds(graph).values())
        self.assertEqual(len(seeds), 1)
        self.assertGreaterEqual(next(iter(seeds)), 0)


if __name__ == "__main__":
    unittest.main()
