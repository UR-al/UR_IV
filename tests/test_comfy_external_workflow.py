"""Generation API ComfyUI profiles apply payloads with the compiler's rules (감사 #185).

ComfyUIBackend.generate_workflow used its own legacy mapper: txt2img got a
partial denoise, Forge sampler names and <lora:> tags were sent verbatim,
ForgeNeoKSamplerCNS/UNETLoader/ForgeNeoLatentInput were not recognised.  It now
uses ComfyWorkflowCompiler.map_external_workflow (no node-pack nodes added).

Repairs on top: multi-pass graphs (Hires 2-pass, SDXL refiner) that the legacy
mapper ran keep running — the payload goes to the first pass only; a prompt
LoRA the target lacks is skipped with a warning like Forge; a negative derived
from the positive encoder (ConditioningZeroOut) never overwrites the prompt;
the schema read is bounded, cancellable and shared per endpoint.
"""
from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

from backends.base import GenerationResult
from backends.comfyui_backend import ComfyUIBackend
from core.comfy_object_info_cache import ObjectInfoCacheRegistry
from core.comfy_schema_fetch import SchemaFetchCancelled
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError


def _choice(*values):
    return [list(values), {}]


def _schema() -> dict:
    return {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": _choice("base.safetensors", "new.safetensors")}}},
        "UNETLoader": {"input": {"required": {"unet_name": _choice("anima.safetensors", "other.safetensors")}}},
        "LoraLoader": {"input": {"required": {"lora_name": _choice("styles/ink.safetensors")}}},
        "KSampler": {"input": {"required": {
            "sampler_name": _choice(
                "euler", "euler_ancestral", "dpmpp_2m",
                "dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2",
            ),
            "scheduler": _choice("normal", "karras"),
        }}},
    }


# ComfyUI comfy/samplers.py KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS — 컴파일러 상수와
# 따로 적어 둔다(같은 표를 import 하면 둘이 함께 틀려도 테스트가 통과한다).
_COMFY_DISCARD_PENULTIMATE = frozenset({"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"})


def _karras_sigmas(n: int, sigma_min: float = 0.0292, sigma_max: float = 14.6146) -> list:
    """k_diffusion get_sigmas_karras (rho 7) + append_zero, SDXL sigma range."""
    rho = 7.0
    ramp = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    min_inv, max_inv = sigma_min ** (1 / rho), sigma_max ** (1 / rho)
    return [(max_inv + r * (min_inv - max_inv)) ** rho for r in ramp] + [0.0]


def _pass_sigmas(sampler: str, steps: int) -> list:
    """KSampler.calculate_sigmas for karras: DPM2/UniPC use steps+1 and drop the penultimate."""
    if sampler in _COMFY_DISCARD_PENULTIMATE:
        sigmas = _karras_sigmas(steps + 1)
        return sigmas[:-2] + sigmas[-1:]
    return _karras_sigmas(steps)


def _workflow(sampler="KSampler") -> dict:
    inputs = {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal",
    }
    if sampler == "KSamplerAdvanced":
        inputs.update({"noise_seed": 1, "add_noise": "enable", "start_at_step": 0,
                       "end_at_step": 20, "return_with_leftover_noise": "disable"})
    else:
        inputs.update({"seed": 1, "denoise": 1.0})
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "old"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "old neg"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": sampler, "inputs": inputs},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0]}},
    }


def _map(workflow, mode="txt2img", model="", payload=None, schema=None):
    return ComfyWorkflowCompiler(_schema() if schema is None else schema).map_external_workflow(
        workflow, mode, model, payload or {},
    )


class MapExternalWorkflowTests(unittest.TestCase):
    def test_txt2img_ignores_client_hires_denoise_and_converts_sampler(self):
        workflow = _workflow()
        original = copy.deepcopy(workflow)
        graph = _map(workflow, payload={
            "denoising_strength": 0.7, "sampler_name": "Euler a", "scheduler": "Automatic",
            "steps": 30, "cfg_scale": 5, "seed": 9, "width": 768, "height": 1024,
            "batch_size": 2, "n_iter": 2,
        })
        self.assertEqual(workflow, original)
        sampler = graph["5"]["inputs"]
        self.assertEqual(sampler["denoise"], 1.0)
        self.assertEqual((sampler["sampler_name"], sampler["scheduler"]), ("euler_ancestral", "normal"))
        self.assertEqual((sampler["steps"], sampler["cfg"], sampler["seed"]), (30, 5.0, 9))
        self.assertEqual(graph["4"]["inputs"], {"width": 768, "height": 1024, "batch_size": 4})

    def test_img2img_denoise_and_advanced_step_window(self):
        graph = _map(_workflow(), mode="img2img", payload={"denoising_strength": 0.4})
        self.assertEqual(graph["5"]["inputs"]["denoise"], 0.4)
        advanced = _map(_workflow("KSamplerAdvanced"), mode="img2img",
                        payload={"denoising_strength": 0.25, "steps": 20})
        self.assertEqual(advanced["5"]["inputs"]["start_at_step"], 15)
        self.assertEqual(advanced["5"]["inputs"]["end_at_step"], 20)
        self.assertNotIn("denoise", advanced["5"]["inputs"])
        self.assertEqual(_map(_workflow("KSamplerAdvanced"))["5"]["inputs"]["start_at_step"], 0)

    def test_lora_tags_become_loader_nodes_and_leave_the_prompt(self):
        graph = _map(_workflow(), payload={
            "prompt": "portrait, <lora:ink:0.6>, smile", "negative_prompt": "bad",
        })
        loras = [node_id for node_id, node in graph.items() if node["class_type"] == "LoraLoader"]
        self.assertEqual(len(loras), 1)
        lora = graph[loras[0]]["inputs"]
        self.assertEqual(lora["lora_name"], "styles/ink.safetensors")
        self.assertEqual((lora["model"], lora["clip"]), (["1", 0], ["1", 1]))
        self.assertEqual(graph["5"]["inputs"]["model"], [loras[0], 0])
        self.assertEqual(graph["2"]["inputs"], {"clip": [loras[0], 1], "text": "portrait, smile"})
        self.assertEqual(graph["3"]["inputs"]["text"], "bad")

    def test_model_is_written_to_rewritable_loaders_only(self):
        graph = _map(_workflow(), model="new")
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], "new.safetensors")
        unet = _workflow()
        unet["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "anima.safetensors", "weight_dtype": "default"}}
        unet["8"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors", "type": "stable_diffusion"}}
        unet["2"]["inputs"]["clip"] = ["8", 0]
        unet["3"]["inputs"]["clip"] = ["8", 0]
        self.assertEqual(_map(unet, model="other.safetensors")["1"]["inputs"]["unet_name"], "other.safetensors")
        locked = copy.deepcopy(unet)
        locked["1"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "q8.gguf"}}
        self.assertEqual(_map(locked, model="other.safetensors")["1"], locked["1"])

    def test_forge_parity_graph_nodes_are_recognised(self):
        workflow = _workflow()
        workflow["5"]["class_type"] = "ForgeNeoKSamplerCNS"
        workflow["4"] = {"class_type": "ForgeNeoLatentInput", "inputs": {
            "vae": ["1", 2], "mode": "img2img", "width": 512, "height": 512, "batch_size": 1,
        }}
        workflow["2"] = {"class_type": "ForgeNeoAnima38V2Prompt", "inputs": {
            "model": ["1", 0], "native_clip": ["1", 1], "qwen35_clip": ["9", 0], "prompt": "old",
        }}
        graph = _map(workflow, payload={"prompt": "new prompt", "width": 832, "height": 1216})
        self.assertEqual(graph["5"]["inputs"]["denoise"], 1.0)
        self.assertEqual(graph["4"]["inputs"]["mode"], "txt2img")
        self.assertEqual((graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]), (832, 1216))
        self.assertEqual(graph["2"]["inputs"]["prompt"], "new prompt")
        self.assertNotIn("text", graph["2"]["inputs"])

    def test_sdxl_encoder_gets_both_texts(self):
        workflow = _workflow()
        workflow["2"] = {"class_type": "CLIPTextEncodeSDXL", "inputs": {"clip": ["1", 1], "text_g": "", "text_l": ""}}
        graph = _map(workflow, payload={"prompt": "castle"})
        self.assertEqual((graph["2"]["inputs"]["text_g"], graph["2"]["inputs"]["text_l"]), ("castle", "castle"))

    def test_unrepresentable_requests_fail_before_queueing(self):
        with self.assertRaisesRegex(WorkflowCompileError, "Hires"):
            _map(_workflow(), payload={"enable_hr": True})
        custom = _workflow()
        custom["5"]["class_type"] = "SamplerCustom"
        with self.assertRaisesRegex(WorkflowCompileError, "SamplerCustom"):
            _map(custom)
        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 값"):
            _map(_workflow(), payload={"sampler_name": "Restart"})
        split_clip = _workflow()
        split_clip["9"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "x", "type": "sd"}}
        split_clip["3"]["inputs"]["clip"] = ["9", 0]
        with self.assertRaisesRegex(WorkflowCompileError, "LoRA|lora"):
            _map(split_clip, payload={"prompt": "<lora:ink:1>"})
        malformed = _workflow()
        malformed["5"]["inputs"] = None
        with self.assertRaisesRegex(WorkflowCompileError, "inputs"):
            _map(malformed)

    def test_random_seed_is_concrete(self):
        graph = _map(_workflow(), payload={"seed": -1})
        self.assertGreaterEqual(graph["5"]["inputs"]["seed"], 0)


def _two_pass_workflow() -> dict:
    """Hires 2-pass graph whose refine sampler is listed first (not dict order)."""
    workflow = {"10": {"class_type": "KSampler", "inputs": {
        "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
        "latent_image": ["9", 0], "seed": 77, "steps": 12, "cfg": 4.5,
        "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.5,
    }}}
    workflow.update(_workflow())
    workflow["9"] = {"class_type": "LatentUpscale", "inputs": {
        "samples": ["5", 0], "upscale_method": "nearest-exact",
        "width": 1024, "height": 1024, "crop": "disabled",
    }}
    workflow["6"]["inputs"]["samples"] = ["10", 0]
    return workflow


class MultiPassExternalWorkflowTests(unittest.TestCase):
    """Graphs the legacy mapper ran (it wrote the first sampler) keep running."""

    def test_two_pass_graph_maps_the_first_pass_and_keeps_the_refine_pass(self):
        workflow = _two_pass_workflow()
        graph = _map(workflow, payload={
            "steps": 30, "seed": 9, "denoising_strength": 0.7, "sampler_name": "Euler a",
            "width": 768, "height": 768,
        })
        base = graph["5"]["inputs"]
        self.assertEqual(
            (base["steps"], base["seed"], base["denoise"], base["sampler_name"]),
            (30, 9, 1.0, "euler_ancestral"),
        )
        self.assertEqual(graph["10"], workflow["10"])
        self.assertEqual(graph["9"], workflow["9"])
        self.assertEqual((graph["4"]["inputs"]["width"], graph["4"]["inputs"]["height"]), (768, 768))

    def test_prompt_lora_patches_every_pass_and_encoder(self):
        graph = _map(_two_pass_workflow(), payload={"prompt": "cat <lora:ink:0.5>"})
        lora_id = next(node_id for node_id, node in graph.items() if node["class_type"] == "LoraLoader")
        self.assertEqual((graph[lora_id]["inputs"]["model"], graph[lora_id]["inputs"]["clip"]), (["1", 0], ["1", 1]))
        for sampler_id in ("5", "10"):
            self.assertEqual(graph[sampler_id]["inputs"]["model"], [lora_id, 0])
        for encoder_id in ("2", "3"):
            self.assertEqual(graph[encoder_id]["inputs"]["clip"], [lora_id, 1])
        self.assertEqual(graph["6"]["inputs"]["vae"], ["1", 2])
        self.assertEqual(graph["2"]["inputs"]["text"], "cat")

    @staticmethod
    def _split_schedule_workflow(scheduler: str = "normal", sampler: str = "euler",
                                 refiner_sampler=None) -> dict:
        """SDXL base + refiner: base 0→20 of 25 hands leftover noise to the refiner 20→end.

        ``refiner_sampler`` 는 이어받는 패스의 샘플러(기본: base 와 같음).
        """
        workflow = _workflow("KSamplerAdvanced")
        workflow["5"]["inputs"].update({
            "steps": 25, "end_at_step": 20, "return_with_leftover_noise": "enable",
            "scheduler": scheduler, "sampler_name": sampler,
        })
        workflow["10"] = {"class_type": "KSamplerAdvanced", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["5", 0],
            "noise_seed": 3, "add_noise": "disable", "steps": 25, "cfg": 7,
            "sampler_name": sampler if refiner_sampler is None else refiner_sampler,
            "scheduler": scheduler,
            "start_at_step": 20, "end_at_step": 10000, "return_with_leftover_noise": "disable",
        }}
        workflow["6"]["inputs"]["samples"] = ["10", 0]
        return workflow

    def test_split_schedule_refiner_keeps_its_step_window(self):
        workflow = self._split_schedule_workflow()
        graph = _map(workflow, payload={
            "steps": 30, "seed": 5, "cfg_scale": 6, "sampler_name": "DPM++ 2M", "scheduler": "Karras",
        })
        base = graph["5"]["inputs"]
        # 시드·CFG·샘플러 알고리즘은 페이로드를 따르지만, 스케줄러는 다음 패스와 한 시그마
        # 스케줄이라 작성된 값 그대로다(예전엔 첫 패스만 karras 로 바뀌어 경계에서 끊겼다).
        self.assertEqual(
            (base["noise_seed"], base["cfg"], base["sampler_name"], base["scheduler"]),
            (5, 6.0, "dpmpp_2m", "normal"),
        )
        self.assertEqual(
            (base["steps"], base["start_at_step"], base["end_at_step"], base["return_with_leftover_noise"]),
            (25, 0, 20, "enable"),
        )
        self.assertEqual(graph["10"], workflow["10"])
        # Without a later pass, leftover noise is not a split schedule: the payload
        # scheduler and steps apply as usual.
        single = _workflow("KSamplerAdvanced")
        single["5"]["inputs"]["return_with_leftover_noise"] = "enable"
        mapped = _map(single, payload={"steps": 30, "scheduler": "Karras"})["5"]["inputs"]
        self.assertEqual(
            (mapped["steps"], mapped["end_at_step"], mapped["return_with_leftover_noise"], mapped["scheduler"]),
            (30, 30, "disable", "karras"),
        )

    def test_split_schedule_default_scheduler_never_overwrites_the_authored_one(self):
        # 스케줄러가 없는 페이로드(또는 'Automatic')는 'normal' 로 풀린다 — 분할 패스에 쓰면
        # karras/karras 워크플로의 첫 패스만 normal 이 됐다.
        for payload in ({}, {"scheduler": "Automatic"}, {"scheduler": "Normal"}):
            with self.subTest(payload=payload):
                workflow = self._split_schedule_workflow("karras")
                graph = _map(workflow, payload=payload)
                self.assertEqual(graph["5"]["inputs"]["scheduler"], "karras")
                self.assertEqual(graph["10"], workflow["10"])

    def test_split_schedule_passes_stay_on_one_sigma_schedule(self):
        # 불변식: 주 패스의 leftover noise 를 이어받는(add_noise=disable) 패스와 주 패스는 작성 때
        # 같았던 (steps, scheduler) 를 매핑 뒤에도 공유한다.
        payloads = (
            {}, {"scheduler": "Karras"}, {"scheduler": "Automatic", "steps": 40},
            {"sampler_name": "DPM++ 2M Karras", "steps": 8}, {"sampler_name": "Euler a", "scheduler": "normal"},
        )
        for authored in ("normal", "karras"):
            for payload in payloads:
                with self.subTest(authored=authored, payload=payload):
                    graph = _map(self._split_schedule_workflow(authored), payload=payload)
                    main = graph["5"]["inputs"]
                    continuation = graph["10"]["inputs"]
                    self.assertEqual(continuation["add_noise"], "disable")
                    self.assertEqual(
                        (main["steps"], main["scheduler"]),
                        (continuation["steps"], continuation["scheduler"]),
                    )
                    self.assertEqual(main["end_at_step"], continuation["start_at_step"])

    def test_split_schedule_boundary_is_one_noise_level(self):
        # 같은 step 번호라도 DPM2/UniPC 는 steps+1 로 시그마를 만들고 끝에서 두 번째를 버려서
        # 다른 노이즈 수준이다. 주 패스가 멈춘 시그마 == 이어받는 패스가 시작하는 시그마.
        payloads = (
            {}, {"sampler_name": "DPM2"}, {"sampler_name": "DPM2 a"}, {"sampler_name": "UniPC"},
            {"sampler_name": "Euler a"}, {"sampler_name": "DPM++ 2M"}, {"sampler_name": "Euler"},
        )
        for authored in ("euler", "uni_pc", "dpm_2"):
            for payload in payloads:
                with self.subTest(authored=authored, payload=payload):
                    graph = _map(self._split_schedule_workflow("karras", authored), payload=payload)
                    main, continuation = graph["5"]["inputs"], graph["10"]["inputs"]
                    stop = _pass_sigmas(main["sampler_name"], main["steps"])[main["end_at_step"]]
                    start = _pass_sigmas(
                        continuation["sampler_name"], continuation["steps"],
                    )[continuation["start_at_step"]]
                    self.assertTrue(math.isclose(stop, start, rel_tol=1e-9), (stop, start))

    def test_split_schedule_sampler_of_another_sigma_layout_keeps_the_authored_one(self):
        for label in ("DPM2", "DPM2 a", "UniPC"):
            with self.subTest(label=label):
                warnings: list = []
                workflow = self._split_schedule_workflow("karras", "euler")
                graph = ComfyWorkflowCompiler(_schema()).map_external_workflow(
                    workflow, "txt2img", "", {"sampler_name": label, "seed": 5}, warnings=warnings,
                )
                self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler")
                self.assertEqual(graph["5"]["inputs"]["noise_seed"], 5)
                self.assertEqual(graph["10"], workflow["10"])
                self.assertEqual(len(warnings), 1)
                self.assertIn("euler", warnings[0])
        # 반대 방향: uni_pc 분할 워크플로에 Euler 를 보내도, 샘플러 없는 페이로드(예전엔
        # 기본값 euler 로 덮였다)도 작성된 uni_pc 를 지킨다. 페이로드가 고른 게 없으면 경고도 없다.
        for payload, expected_warnings in (({}, 0), ({"sampler_name": "Euler"}, 1)):
            with self.subTest(payload=payload):
                warnings = []
                graph = ComfyWorkflowCompiler(_schema()).map_external_workflow(
                    self._split_schedule_workflow("karras", "uni_pc"), "txt2img", "", payload,
                    warnings=warnings,
                )
                self.assertEqual(graph["5"]["inputs"]["sampler_name"], "uni_pc")
                self.assertEqual(len(warnings), expected_warnings)

    def test_split_schedule_sampler_of_the_same_sigma_layout_still_applies(self):
        for authored, label, expected in (
            ("euler", "DPM++ 2M", "dpmpp_2m"), ("euler", "Euler a", "euler_ancestral"),
            ("dpm_2", "UniPC", "uni_pc"), ("uni_pc", "DPM2 a", "dpm_2_ancestral"),
        ):
            with self.subTest(authored=authored, label=label):
                warnings: list = []
                graph = ComfyWorkflowCompiler(_schema()).map_external_workflow(
                    self._split_schedule_workflow("karras", authored), "txt2img", "",
                    {"sampler_name": label}, warnings=warnings,
                )
                self.assertEqual(graph["5"]["inputs"]["sampler_name"], expected)
                self.assertEqual(warnings, [])

    def _split_warnings(self, workflow, payload):
        warnings: list = []
        graph = ComfyWorkflowCompiler(_schema()).map_external_workflow(
            workflow, "txt2img", "", payload, warnings=warnings,
        )
        return graph, warnings

    def test_split_schedule_boundary_follows_the_continuation_pass_sampler(self):
        # 노이즈 경계는 '이어받는 패스'와의 경계라 그 패스의 샘플러가 시그마 배열을 정한다.
        # 예전엔 주 패스의 작성된 샘플러와 비교해, base=uni_pc / refiner=euler 워크플로에 Euler 를
        # 보내면 uni_pc 를 지키고 "Euler 는 경계가 어긋난다"고 경고했다 — 실제론 반대였다.
        workflow = self._split_schedule_workflow("karras", "uni_pc", refiner_sampler="euler")
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "Euler"})
        main, continuation = graph["5"]["inputs"], graph["10"]["inputs"]
        self.assertEqual(main["sampler_name"], "euler")
        self.assertEqual(warnings, [])
        self.assertEqual(graph["10"], workflow["10"])
        stop = _pass_sigmas(main["sampler_name"], main["steps"])[main["end_at_step"]]
        start = _pass_sigmas(continuation["sampler_name"], continuation["steps"])[
            continuation["start_at_step"]
        ]
        self.assertTrue(math.isclose(stop, start, rel_tol=1e-9), (stop, start))

    def test_split_schedule_mapping_never_moves_the_boundary_further_off(self):
        # 불변식(작성된 두 패스가 같든 다르든): 매핑 뒤 경계는 맞거나, 작성된 그대로다. 그리고
        # 이어받는 패스와 같은 배열의 페이로드 샘플러는 언제나 들어간다.
        payloads = (
            {}, {"sampler_name": "DPM2"}, {"sampler_name": "DPM2 a"}, {"sampler_name": "UniPC"},
            {"sampler_name": "Euler a"}, {"sampler_name": "DPM++ 2M"}, {"sampler_name": "Euler"},
        )
        labels = {
            "DPM2": "dpm_2", "DPM2 a": "dpm_2_ancestral", "UniPC": "uni_pc",
            "Euler a": "euler_ancestral", "DPM++ 2M": "dpmpp_2m", "Euler": "euler",
        }
        authored_samplers = ("euler", "uni_pc", "dpm_2")

        def boundary(inputs_main, inputs_next):
            stop = _pass_sigmas(inputs_main["sampler_name"], inputs_main["steps"])[
                inputs_main["end_at_step"]
            ]
            start = _pass_sigmas(inputs_next["sampler_name"], inputs_next["steps"])[
                inputs_next["start_at_step"]
            ]
            return stop, start

        for base in authored_samplers:
            for refiner in authored_samplers:
                for payload in payloads:
                    with self.subTest(base=base, refiner=refiner, payload=payload):
                        workflow = self._split_schedule_workflow("karras", base, refiner_sampler=refiner)
                        authored_stop, _ = boundary(workflow["5"]["inputs"], workflow["10"]["inputs"])
                        graph, warnings = self._split_warnings(workflow, payload)
                        stop, start = boundary(graph["5"]["inputs"], graph["10"]["inputs"])
                        self.assertTrue(
                            math.isclose(stop, start, rel_tol=1e-9)
                            or math.isclose(stop, authored_stop, rel_tol=1e-9),
                            (stop, start, authored_stop),
                        )
                        wanted = labels.get(payload.get("sampler_name"))
                        if wanted is None:
                            self.assertEqual(graph["5"]["inputs"]["sampler_name"], base)
                            self.assertEqual(warnings, [])
                            continue
                        matches_next = (wanted in _COMFY_DISCARD_PENULTIMATE) == (
                            refiner in _COMFY_DISCARD_PENULTIMATE
                        )
                        matches_base = (wanted in _COMFY_DISCARD_PENULTIMATE) == (
                            base in _COMFY_DISCARD_PENULTIMATE
                        )
                        if matches_next or matches_base:
                            self.assertEqual(graph["5"]["inputs"]["sampler_name"], wanted)
                        else:
                            self.assertEqual(graph["5"]["inputs"]["sampler_name"], base)
                        # 경고는 경계가 어긋난 채 남을 때만: 샘플러를 지켰거나, 작성된 두 패스가
                        # 이미 달라 페이로드도 첫 패스 배열로 들어갔을 때.
                        self.assertEqual(len(warnings), 0 if matches_next else 1, warnings)
                        if matches_next:
                            continue
                        if matches_base:
                            self.assertIn("이미 시그마 배열이 달라", warnings[0])
                        else:
                            self.assertIn(f"다음 패스 샘플러({refiner})", warnings[0])

    def test_split_schedule_ignores_a_later_pass_that_does_not_take_the_leftover_noise(self):
        # base → refiner(leftover noise 이어받음) → 디코드·인코드 뒤 Hires KSampler(uni_pc).
        # 경계는 refiner 와의 것뿐이라 Hires 패스의 샘플러는 판정에 끼지 않는다.
        workflow = self._split_schedule_workflow("karras", "euler")
        workflow["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}}
        workflow["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["1", 2]}}
        workflow["13"] = {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["12", 0],
            "seed": 8, "steps": 12, "cfg": 5, "sampler_name": "uni_pc", "scheduler": "karras",
            "denoise": 0.4,
        }}
        workflow["6"]["inputs"]["samples"] = ["13", 0]
        self.assertEqual(
            ComfyWorkflowCompiler._continuation_sampler_names(workflow, "5"), ["euler"],
        )
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "Euler a"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(warnings, [])
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "UniPC"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(len(warnings), 1)
        self.assertIn("다음 패스 샘플러(euler)", warnings[0])
        self.assertEqual((graph["10"], graph["13"]), (workflow["10"], workflow["13"]))

    def test_split_schedule_linked_sampler_is_judged_by_the_continuation_pass(self):
        # 주 패스 샘플러가 다른 노드에 연결돼 있어도, 이어받는 패스의 샘플러를 알면 그 배열로
        # 판정한다 — 같은 배열이면 페이로드가 들어가고(경계가 맞는다), 다르면 링크를 지키고 알린다.
        workflow = self._split_schedule_workflow("karras")
        workflow["5"]["inputs"]["sampler_name"] = ["30", 0]
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "Euler a"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(warnings, [])
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "UniPC"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], ["30", 0])
        self.assertEqual(len(warnings), 1)
        self.assertIn("다른 노드에서 연결됨", warnings[0])
        self.assertIn("다음 패스 샘플러(euler)", warnings[0])
        self.assertEqual(
            _map(workflow, payload={})["5"]["inputs"]["sampler_name"], ["30", 0],
        )
        # 두 패스가 같은 노드(샘플러 선택기)에 연결돼 있으면 어느 쪽 배열도 알 수 없다 — 지키고 알린다.
        workflow["10"]["inputs"]["sampler_name"] = ["30", 0]
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "Euler a"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], ["30", 0])
        self.assertEqual(len(warnings), 1)
        self.assertIn("확인할 수 없습니다", warnings[0])
        # 이어받는 패스만 연결돼 있으면: 작성된 첫 패스와 같은 배열은 경계가 그대로라 들어가고,
        # 다른 배열은 확인할 수 없으니 작성된 샘플러를 지킨다.
        workflow["5"]["inputs"]["sampler_name"] = "euler"
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "Euler a"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(warnings, [])
        graph, warnings = self._split_warnings(workflow, {"sampler_name": "UniPC"})
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(len(warnings), 1)
        self.assertIn("확인할 수 없습니다", warnings[0])
        workflow["5"]["inputs"]["sampler_name"] = ["30", 0]
        # 작성된 샘플러가 아예 없으면 지킬 게 없으니 페이로드(없으면 기본 euler)를 쓴다.
        del workflow["5"]["inputs"]["sampler_name"]
        self.assertEqual(_map(workflow, payload={})["5"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(
            _map(workflow, payload={"sampler_name": "UniPC"})["5"]["inputs"]["sampler_name"], "uni_pc",
        )

    def test_split_schedule_still_rejects_an_unsupported_sampler(self):
        # 작성된 샘플러를 지키는 경우에도 대상에 없는 페이로드 샘플러는 여전히 오류다.
        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 값"):
            _map(self._split_schedule_workflow("karras"), payload={"sampler_name": "Restart"})

    def test_split_schedule_still_rejects_an_unsupported_scheduler(self):
        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 값"):
            _map(self._split_schedule_workflow(), payload={"scheduler": "exponential"})

    def test_custom_sampler_is_rejected_only_as_the_main_sampler(self):
        workflow = _two_pass_workflow()
        workflow["10"]["class_type"] = "SamplerCustom"
        self.assertEqual(_map(workflow)["10"], workflow["10"])
        workflow = _two_pass_workflow()
        workflow["5"]["class_type"] = "SamplerCustom"
        with self.assertRaisesRegex(WorkflowCompileError, "SamplerCustom"):
            _map(workflow)

    def test_first_pass_that_reaches_the_output_beats_an_unused_branch(self):
        workflow = {"20": copy.deepcopy(_workflow()["5"])}  # dangling sampler, listed first
        workflow["20"]["inputs"]["seed"] = 42
        workflow.update(_workflow())
        graph = _map(workflow, payload={"seed": 9})
        self.assertEqual(graph["5"]["inputs"]["seed"], 9)
        self.assertEqual(graph["20"]["inputs"]["seed"], 42)


class ExternalPromptRulesTests(unittest.TestCase):
    def test_missing_prompt_lora_is_skipped_with_a_warning_like_forge(self):
        warnings: list = []
        graph = ComfyWorkflowCompiler(_schema()).map_external_workflow(
            _workflow(), "txt2img", "", {"prompt": "a, <lora:ghost:1>, <lora:ink:0.6>"},
            warnings=warnings,
        )
        loras = [node for node in graph.values() if node["class_type"] == "LoraLoader"]
        self.assertEqual([node["inputs"]["lora_name"] for node in loras], ["styles/ink.safetensors"])
        self.assertEqual(graph["2"]["inputs"]["text"], "a")
        self.assertEqual(len(warnings), 1)
        self.assertIn("ghost", warnings[0])
        # Nothing to insert: a split CLIP is then not an error either.
        split_clip = _workflow()
        split_clip["9"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "x", "type": "sd"}}
        split_clip["3"]["inputs"]["clip"] = ["9", 0]
        self.assertEqual(_map(split_clip, payload={"prompt": "<lora:ghost:1> a"})["2"]["inputs"]["text"], "a")

    def test_zero_out_negative_never_overwrites_the_positive_prompt(self):
        workflow = _workflow()
        workflow["3"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["2", 0]}}
        graph = _map(workflow, payload={"prompt": "castle", "negative_prompt": ""})
        self.assertEqual(graph["2"]["inputs"]["text"], "castle")
        self.assertEqual(graph["3"], workflow["3"])
        with self.assertRaisesRegex(WorkflowCompileError, "ConditioningZeroOut"):
            _map(workflow, payload={"prompt": "castle", "negative_prompt": "blurry"})


def _backend(schema=None, registry=None) -> ComfyUIBackend:
    backend = ComfyUIBackend("http://remote.invalid:8188")
    backend.get_object_info_bounded = mock.Mock(return_value=_schema() if schema is None else schema)
    backend.get_object_info = mock.Mock(side_effect=AssertionError("unbounded /object_info"))
    backend._external_object_info_caches = registry if registry is not None else ObjectInfoCacheRegistry()
    backend._preflight_bundled_node_pack = mock.Mock(side_effect=AssertionError("no preflight"))
    backend.run_workflow = mock.Mock(return_value=GenerationResult(success=True, image_data=b"x"))
    return backend


def _schema_without(**removed) -> dict:
    schema = _schema()
    for class_type, input_name in removed.items():
        schema[class_type]["input"]["required"][input_name] = _choice()
    return schema


class GenerateWorkflowBackendTests(unittest.TestCase):
    def test_reports_the_queued_seed_and_does_not_preflight_the_node_pack(self):
        backend = _backend()
        result = backend.generate_workflow("t2i", _workflow(), "new", {"seed": -1, "denoising_strength": 0.7})
        self.assertTrue(result.success, result.error)
        queued = backend.run_workflow.call_args.args[0]
        self.assertEqual(queued["5"]["inputs"]["denoise"], 1.0)
        self.assertEqual(result.info["seed"], queued["5"]["inputs"]["seed"])
        self.assertNotIn("warnings", result.info)

    def test_mapping_errors_become_generation_errors(self):
        backend = _backend()
        result = backend.generate_workflow("t2i", _workflow(), "", {"enable_hr": True})
        self.assertFalse(result.success)
        self.assertIn("Hires", result.error)
        backend.run_workflow.assert_not_called()

    def test_two_pass_profile_workflow_still_runs(self):
        backend = _backend()
        result = backend.generate_workflow("t2i", _two_pass_workflow(), "", {"seed": 4})
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.info["seed"], 4)  # several samplers: the payload seed

    def test_repeated_jobs_share_one_bounded_schema_read_per_endpoint(self):
        registry = ObjectInfoCacheRegistry()
        first, second = _backend(registry=registry), _backend(registry=registry)
        check = mock.Mock(return_value=False)
        self.assertTrue(first.generate_workflow("t2i", _workflow(), "", {}, None, check).success)
        self.assertTrue(second.generate_workflow("t2i", _workflow(), "", {}, None, check).success)
        first.get_object_info_bounded.assert_called_once_with(check)
        second.get_object_info_bounded.assert_not_called()

    def test_missing_lora_on_a_cached_schema_is_rechecked_once_fresh(self):
        registry = ObjectInfoCacheRegistry()
        stale = _schema_without(LoraLoader="lora_name")
        registry.for_url("http://remote.invalid:8188").store("http://remote.invalid:8188", stale)
        backend = _backend(registry=registry)
        result = backend.generate_workflow("t2i", _workflow(), "", {"prompt": "a <lora:ink:1>"})
        self.assertTrue(result.success, result.error)
        self.assertNotIn("warnings", result.info)
        backend.get_object_info_bounded.assert_called_once_with(None)
        queued = backend.run_workflow.call_args.args[0]
        self.assertTrue(any(node["class_type"] == "LoraLoader" for node in queued.values()))

    def test_compile_error_on_a_cached_schema_is_retried_once_fresh(self):
        registry = ObjectInfoCacheRegistry()
        registry.for_url("http://remote.invalid:8188").store(
            "http://remote.invalid:8188", _schema_without(CheckpointLoaderSimple="ckpt_name"),
        )
        backend = _backend(registry=registry)
        result = backend.generate_workflow("t2i", _workflow(), "new", {})
        self.assertTrue(result.success, result.error)
        self.assertEqual(backend.run_workflow.call_args.args[0]["1"]["inputs"]["ckpt_name"], "new.safetensors")
        backend.get_object_info_bounded.assert_called_once_with(None)

    def test_missing_lora_warning_is_reported_in_the_result(self):
        backend = _backend(schema=_schema_without(LoraLoader="lora_name"))
        result = backend.generate_workflow("t2i", _workflow(), "", {"prompt": "a <lora:ghost:1>"})
        self.assertTrue(result.success, result.error)
        self.assertEqual(len(result.info["warnings"]), 1)
        self.assertIn("ghost", result.info["warnings"][0])
        backend.get_object_info_bounded.assert_called_once_with(None)  # fresh: no retry

    def test_split_sampler_warning_on_a_cached_schema_costs_no_refetch(self):
        # 스키마와 무관한 경고(분할 패스 샘플러 유지)는 캐시된 스키마를 다시 받을 이유가 아니다 —
        # 재확인은 대상에 없던 LoRA 가 있을 때만.
        registry = ObjectInfoCacheRegistry()
        registry.for_url("http://remote.invalid:8188").store("http://remote.invalid:8188", _schema())
        backend = _backend(registry=registry)
        workflow = MultiPassExternalWorkflowTests._split_schedule_workflow("karras")
        result = backend.generate_workflow("t2i", workflow, "", {"sampler_name": "UniPC"})
        self.assertTrue(result.success, result.error)
        backend.get_object_info_bounded.assert_not_called()
        self.assertEqual(len(result.info["warnings"]), 1)
        self.assertIn("euler", result.info["warnings"][0])
        self.assertEqual(backend.run_workflow.call_args.args[0]["5"]["inputs"]["sampler_name"], "euler")

    def test_cancel_during_or_right_after_the_schema_read_skips_the_prompt(self):
        backend = _backend()
        backend.get_object_info_bounded.side_effect = SchemaFetchCancelled("취소")
        result = backend.generate_workflow("t2i", _workflow(), "", {}, None, lambda: False)
        self.assertFalse(result.success)
        self.assertIn("취소", result.error)
        backend.run_workflow.assert_not_called()

        backend = _backend()
        cancelled = {"flag": False}

        def fetch(cancel_check=None):
            cancelled["flag"] = True  # the user cancels while the schema downloads
            return _schema()

        backend.get_object_info_bounded = fetch
        result = backend.generate_workflow("t2i", _workflow(), "", {}, None, lambda: cancelled["flag"])
        self.assertFalse(result.success)
        self.assertIn("취소", result.error)
        backend.run_workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
