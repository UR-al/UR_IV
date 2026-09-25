"""Pure contracts for Forge payload -> ComfyUI graph compilation."""
from __future__ import annotations

import copy
import json
import unittest

from core import anima_guidance, sam3_args
from core.comfy_workflow_compiler import (
    ComfyWorkflowCompiler,
    WorkflowCompileError,
    parse_lora_tags,
)


def _choice(*values):
    return [list(values), {}]


def _capabilities(*, include_forge: bool = True) -> dict:
    names = {
        "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader",
        "TripleCLIPLoader", "VAELoader", "CLIPTextEncode", "EmptyLatentImage",
        "LoadImage", "LoraLoader", "ModelSamplingSD3", "KSampler", "KSamplerAdvanced",
        "SamplerCustom", "VAEDecode", "SaveImage", "PreviewImage", "LatentUpscale",
        "MaskToImage", "ImageScale", "ImageScaleBy", "UpscaleModelLoader",
        "ImageUpscaleWithModel",
    }
    if include_forge:
        names.update({
            "ForgeNeoLatentInput", "ForgeNeoKSamplerCNS",
            "ForgeNeoModelSamplingShift", "ForgeNeoNegPip",
            "ForgeNeoHiresFix", "ForgeNeoADetailer", "ForgeNeoSAM3Mask",
            "ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine", "ForgeNeoAnimaGuidanceSuite",
            "ForgeNeoSkimmedCFG", "ForgeNeoAnimaDetailDaemon",
            # The current (1.4.0) pack's per-step CNS SAMPLER node — the compiler's
            # stale-pack marker for CNS (TestCnsCompilation).
            "ForgeNeoCNSSamplerPatch",
        })
    result = {name: {"input": {"required": {}}} for name in names}
    result["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = _choice(
        "checkpoint.safetensors"
    )
    result["UNETLoader"]["input"]["required"]["unet_name"] = _choice(
        "anima.safetensors"
    )
    result["CLIPLoader"]["input"]["required"]["clip_name"] = _choice(
        "text/base.safetensors", "text/adapter.safetensors", "text/third.safetensors"
    )
    result["VAELoader"]["input"]["required"]["vae_name"] = _choice(
        "vae/image_vae.safetensors"
    )
    result["LoraLoader"]["input"]["required"]["lora_name"] = _choice(
        "styles/ink.safetensors", "characters/alice.safetensors"
    )
    result["UpscaleModelLoader"]["input"]["required"]["model_name"] = _choice(
        "4x-UltraSharp.pth"
    )
    result["KSampler"]["input"]["required"].update({
        "sampler_name": _choice("euler", "euler_ancestral", "dpmpp_2m"),
        "scheduler": _choice("normal", "karras", "simple"),
    })
    if include_forge:
        # ForgeNeoADetailer calls Impact Pack/Subpack providers internally.
        result["FaceDetailer"] = {"input": {"required": {}}}
        result["UltralyticsDetectorProvider"] = {"input": {"required": {
            "model_name": _choice(
                "bbox/face_yolov8n.pt", "bbox/face.pt", "bbox/hand_yolov8n.pt",
                "bbox/person_yolov8n-seg.pt", "segm/person_yolov8n-seg.pt",
            ),
        }}}
    return result


def _classes(graph: dict) -> list[str]:
    return [node["class_type"] for node in graph.values() if isinstance(node, dict)]


def _node(graph: dict, class_type: str, index: int = 0) -> tuple[str, dict]:
    matches = [(node_id, value) for node_id, value in graph.items()
               if isinstance(value, dict) and value.get("class_type") == class_type]
    return matches[index]


def _custom_workflow() -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "checkpoint.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "old"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "old neg"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": 1, "steps": 20, "cfg": 7,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "custom"}},
    }


def _ksampler_advanced_workflow() -> dict:
    workflow = _custom_workflow()
    workflow["5"]["class_type"] = "KSamplerAdvanced"
    inputs = workflow["5"]["inputs"]
    inputs["noise_seed"] = inputs.pop("seed")
    inputs.pop("denoise")
    inputs.update({
        "add_noise": "enable", "start_at_step": 0, "end_at_step": 20,
        "return_with_leftover_noise": "disable",
    })
    return workflow


# The CNS node's own defaults — the compiler's fallbacks for every CNS input.
# origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437
_ORIGIN_CNS_DEFAULTS = {"cns_strength": 1.0, "cns_gamma_power": 0.5, "cns_gamma_scale": 2.0}


def _cns_only_alwayson() -> dict:
    """CNS on, every CNS value left to the compiler's fallback."""
    return {anima_guidance.SCRIPT_PERTURBATION: {"args": {"guid_cns_enabled": True}}}


class TestLoraParsing(unittest.TestCase):
    def test_parses_order_and_independent_clip_weight(self):
        specs, cleaned = parse_lora_tags(
            "portrait, <lora:ink:0.7>, <lora:alice:1.2:0.4>",
            "bad, <lora:negative-helper:-0.2>",
        )
        self.assertEqual([item.name for item in specs], ["ink", "alice", "negative-helper"])
        self.assertEqual(specs[0].strength_clip, 0.7)
        self.assertEqual(specs[1].strength_clip, 0.4)
        self.assertEqual(cleaned, ["portrait", "bad"])


class TestDefaultCompilation(unittest.TestCase):
    def test_explicit_lora_path_wins_over_an_earlier_matching_filename(self):
        capabilities = _capabilities()
        capabilities["LoraLoader"]["input"]["required"]["lora_name"] = _choice(
            "forge/style.safetensors", "comfy/style.safetensors",
        )
        graph = ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", "checkpoint.safetensors", {
                "prompt": "portrait, <lora:comfy/style.safetensors:0.7>",
            },
        )
        self.assertEqual(
            _node(graph, "LoraLoader")[1]["inputs"]["lora_name"],
            "comfy/style.safetensors",
        )

    def test_ambiguous_lora_basename_and_stem_fail_before_queueing(self):
        capabilities = _capabilities()
        capabilities["LoraLoader"]["input"]["required"]["lora_name"] = _choice(
            "forge/style.safetensors", "comfy/style.safetensors",
        )
        for name in ("style.safetensors", "style"):
            with self.subTest(name=name), self.assertRaisesRegex(
                WorkflowCompileError, "여러.*리소스|리소스.*여러",
            ):
                ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "checkpoint.safetensors", {
                        "prompt": f"portrait, <lora:{name}:0.7>",
                    },
                )

    def test_lora_path_without_extension_and_exact_filename_are_unambiguous(self):
        cases = (
            ("comfy/style", ("forge/style.safetensors", "comfy/style.safetensors"),
             "comfy/style.safetensors"),
            ("style.safetensors", ("forge/style.pt", "comfy/style.safetensors"),
             "comfy/style.safetensors"),
        )
        for requested, choices, expected in cases:
            with self.subTest(requested=requested):
                capabilities = _capabilities()
                capabilities["LoraLoader"]["input"]["required"]["lora_name"] = _choice(*choices)
                graph = ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "checkpoint.safetensors", {
                        "prompt": f"portrait, <lora:{requested}:0.7>",
                    },
                )
                self.assertEqual(_node(graph, "LoraLoader")[1]["inputs"]["lora_name"], expected)

    def test_main_and_hires_modules_preserve_explicit_relative_paths(self):
        capabilities = _capabilities()
        capabilities["CLIPLoader"]["input"]["required"]["clip_name"] = _choice(
            "forge/base.safetensors", "comfy/base.safetensors",
        )
        capabilities["VAELoader"]["input"]["required"]["vae_name"] = _choice(
            "forge/image_vae.safetensors", "comfy/image_vae.safetensors",
        )
        graph = ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", "checkpoint.safetensors", {
                "forge_additional_modules": [
                    "comfy\\base.safetensors", "comfy\\image_vae.safetensors",
                ],
                "enable_hr": True,
                "hr_additional_modules": [
                    "comfy/base.safetensors", "comfy/image_vae.safetensors",
                ],
            },
        )
        self.assertEqual(_node(graph, "CLIPLoader")[1]["inputs"]["clip_name"], "comfy/base.safetensors")
        self.assertEqual(_node(graph, "VAELoader")[1]["inputs"]["vae_name"], "comfy/image_vae.safetensors")
        hires = _node(graph, "ForgeNeoHiresFix")[1]["inputs"]
        self.assertEqual(hires["text_encoder_name"], "comfy/base.safetensors")
        self.assertEqual(hires["vae_name"], "comfy/image_vae.safetensors")

    def test_checkpoint_graph_needs_no_user_workflow_and_orders_loras(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        graph = compiler.compile("t2i", "checkpoint.safetensors", {
            "prompt": "subject, <lora:ink:0.5>, <lora:alice:0.8:0.6>",
            "negative_prompt": "bad anatomy",
            "width": 768, "height": 1024, "batch_size": 2, "n_iter": 2,
            "seed": 123, "steps": 30, "cfg_scale": 5.5,
        })

        self.assertIn("ForgeNeoLatentInput", _classes(graph))
        self.assertIn("ForgeNeoKSamplerCNS", _classes(graph))
        loras = [(node_id, node) for node_id, node in graph.items()
                 if node.get("class_type") == "LoraLoader"]
        self.assertEqual([item[1]["inputs"]["lora_name"] for item in loras], [
            "styles/ink.safetensors", "characters/alice.safetensors",
        ])
        self.assertEqual(loras[1][1]["inputs"]["model"], [loras[0][0], 0])
        _pos_id, pos = _node(graph, "CLIPTextEncode", 0)
        self.assertEqual(pos["inputs"]["text"], "subject")
        _latent_id, latent = _node(graph, "ForgeNeoLatentInput")
        self.assertEqual(latent["inputs"]["batch_size"], 4)
        self.assertEqual((latent["inputs"]["width"], latent["inputs"]["height"]), (768, 1024))

    def test_split_unet_maps_three_text_encoders_and_vae(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "img2img", "anima.safetensors", {
                "prompt": "edit", "forge_additional_modules": [
                    "C:/models/vae/image_vae.safetensors",
                    "C:/models/text/base.safetensors",
                    "C:/models/text/adapter.safetensors",
                    "C:/models/text/third.safetensors",
                ],
            }, uploaded_image="api/source.png",
        )
        _clip_id, clip = _node(graph, "TripleCLIPLoader")
        self.assertEqual(clip["inputs"]["clip_name1"], "text/base.safetensors")
        _latent_id, latent = _node(graph, "ForgeNeoLatentInput")
        self.assertEqual(latent["inputs"]["mode"], "img2img")
        load_id, _load = _node(graph, "LoadImage")
        self.assertEqual(latent["inputs"]["img2img_image"], [load_id, 0])

    def test_full_checkpoint_can_override_text_encoder_and_vae_modules(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "forge_additional_modules": [
                    "C:/models/text/base.safetensors",
                    "C:/models/vae/image_vae.safetensors",
                ],
            },
        )
        self.assertNotIn("UNETLoader", _classes(graph))
        self.assertIn("CheckpointLoaderSimple", _classes(graph))
        self.assertIn("CLIPLoader", _classes(graph))
        self.assertIn("VAELoader", _classes(graph))

    def test_inpaint_requires_and_wires_uploaded_mask(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        with self.assertRaisesRegex(WorkflowCompileError, "마스크"):
            compiler.compile(
                "inpaint", "checkpoint.safetensors", {}, uploaded_image="source.png"
            )
        graph = compiler.compile(
            "inpaint", "checkpoint.safetensors", {},
            uploaded_image="source.png", uploaded_mask="mask.png",
        )
        load_nodes = [(node_id, node) for node_id, node in graph.items()
                      if node.get("class_type") == "LoadImage"]
        self.assertEqual(len(load_nodes), 2)
        _latent_id, latent = _node(graph, "ForgeNeoLatentInput")
        self.assertEqual(latent["inputs"]["inpaint_mask_image"], [load_nodes[1][0], 0])

    def test_missing_custom_node_is_a_pre_queue_error(self):
        compiler = ComfyWorkflowCompiler(_capabilities(include_forge=False))
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoKSamplerCNS"):
            compiler.compile("txt2img", "checkpoint.safetensors", {})

    def test_translates_webui_sampler_suffix_and_validates_runtime_choices(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        graph = compiler.compile("txt2img", "checkpoint.safetensors", {
            "sampler_name": "DPM++ 2M Karras", "scheduler": "Automatic",
        })
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertEqual(sampler["inputs"]["scheduler"], "karras")
        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 값"):
            compiler.compile("txt2img", "checkpoint.safetensors", {
                "sampler_name": "Not A Real Sampler",
            })

    def test_translates_forge_er_sde_and_exact_res4lyf_beta57(self):
        capabilities = _capabilities()
        capabilities["KSampler"]["input"]["required"].update({
            "sampler_name": _choice("euler", "er_sde", "er_sde_cns"),
            "scheduler": _choice("normal", "beta", "beta_1_1", "beta57"),
        })
        compiler = ComfyWorkflowCompiler(capabilities)

        graph = compiler.compile("txt2img", "checkpoint.safetensors", {
            "sampler_name": "ER SDE",
            "scheduler": "Beta57 (RES4LYF)",
        })
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["sampler_name"], "er_sde")
        self.assertEqual(sampler["inputs"]["scheduler"], "beta57")

        explicit_cns = compiler.compile("txt2img", "checkpoint.safetensors", {
            "sampler_name": "er_sde_cns",
            "scheduler": "beta57",
        })
        _sampler_id, cns_sampler = _node(explicit_cns, "ForgeNeoKSamplerCNS")
        self.assertEqual(cns_sampler["inputs"]["sampler_name"], "er_sde_cns")

    def test_enabled_extension_chain_is_explicit(self):
        guidance = anima_guidance.default_settings()
        guidance.update({
            "guid_enabled": True, "guid_apg_enabled": True,
            "guid_adg_enabled": True, "guid_cns_enabled": True,
            "skim_enabled": True, "dd_enabled": True,
        })
        payload = {
            "prompt": "portrait", "negative_prompt": "bad",
            "enable_hr": True, "hr_scale": 1.5,
            "alwayson_scripts": {
                "NegPiP": {"args": [True]},
                "ADetailer": {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face_yolov8n.pt",
                }]},
                **anima_guidance.build_alwayson(guidance),
                **sam3_args.build_alwayson({
                    "sam3_prompt": "face/eyes", "sam3_mode": "Inpaint",
                    "sam3_save_artifacts": True, "sam3_cn_enable": True,
                    "sam3_cn_model": "controlnet.safetensors", "sam3_restore_face": True,
                }, prompt="retouch", negative_prompt="bad"),
            },
        }
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", payload
        )
        classes = _classes(graph)
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["denoise"], 1.0)
        # CNS reaches both passes with the guidance values (base sampler and Hires).
        for _node_id, node in (_node(graph, "ForgeNeoKSamplerCNS"), _node(graph, "ForgeNeoHiresFix")):
            self.assertIs(node["inputs"]["cns_enabled"], True)
            self.assertEqual(
                {key: node["inputs"][key] for key in _ORIGIN_CNS_DEFAULTS},
                {key: guidance[f"guid_{key}"] for key in _ORIGIN_CNS_DEFAULTS},
            )
        for expected in (
            "ForgeNeoNegPip", "ForgeNeoAnimaGuidanceSuite", "ForgeNeoSkimmedCFG",
            "ForgeNeoAnimaDetailDaemon", "ForgeNeoHiresFix", "ForgeNeoADetailer",
            "ForgeNeoSAM3Mask", "ForgeNeoSAM3Detailer",
        ):
            self.assertIn(expected, classes)
        _suite_id, suite = _node(graph, "ForgeNeoAnimaGuidanceSuite")
        settings = json.loads(suite["inputs"]["settings_json"])
        self.assertTrue(settings["guid_adg_enabled"])
        self.assertEqual(settings["cfg_scale"], 7.0)
        _sam_id, sam = _node(graph, "ForgeNeoSAM3Detailer")
        self.assertEqual(sam["inputs"]["inpaint_prompt"], "retouch")
        self.assertTrue(sam["inputs"]["controlnet_enable"])
        self.assertFalse(sam["inputs"]["controlnet_override_external"])
        self.assertTrue(sam["inputs"]["restore_face"])
        _mask_id, mask = _node(graph, "ForgeNeoSAM3Mask")
        self.assertEqual(mask["inputs"]["artifact_directory"], "")

    def test_generated_forge_nodes_match_bundled_input_contracts(self):
        from comfy_custom_nodes.ai_studio_forge_parity import (
            generation, guidance, sam3_nodes,
        )

        mappings = {
            **generation.NODE_CLASS_MAPPINGS,
            **guidance.NODE_CLASS_MAPPINGS,
            **sam3_nodes.NODE_CLASS_MAPPINGS,
        }
        guidance_settings = anima_guidance.default_settings()
        guidance_settings.update({
            "guid_enabled": True, "guid_apg_enabled": True,
            "guid_adg_enabled": True, "guid_cns_enabled": True,
            "skim_enabled": True, "dd_enabled": True,
        })
        graph = ComfyWorkflowCompiler().compile("txt2img", "model.safetensors", {
            "prompt": "portrait", "enable_hr": True,
            "hr_upscaler": "latent (bilinear)", "clip_type": "krea2",
            "alwayson_scripts": {
                "NegPiP": {"args": [True]},
                "ADetailer": {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face.pt",
                }]},
                **anima_guidance.build_alwayson(guidance_settings),
                **sam3_args.build_alwayson({
                    "sam3_mode": "Inpaint", "sam3_cn_enable": True,
                    "sam3_cn_override_external": True,
                    "sam3_restore_face": True,
                }),
            },
        })
        for node_id, node in graph.items():
            class_type = node["class_type"]
            if class_type not in mappings:
                continue
            schema = mappings[class_type].INPUT_TYPES()
            required = set(schema.get("required", {}))
            optional = set(schema.get("optional", {}))
            supplied = set(node["inputs"])
            self.assertFalse(required - supplied, (node_id, class_type, required - supplied))
            self.assertFalse(supplied - required - optional, (
                node_id, class_type, supplied - required - optional,
            ))
        _hires_id, hires = _node(graph, "ForgeNeoHiresFix")
        self.assertEqual(hires["inputs"]["upscale_method"], "latent:bilinear")
        self.assertEqual(hires["inputs"]["clip_type"], "krea2")

    def test_stale_forge_node_schema_is_rejected_before_queue(self):
        capabilities = _capabilities()
        capabilities["ForgeNeoKSamplerCNS"]["input"]["required"] = {
            "model": ["MODEL", {}],
        }
        with self.assertRaisesRegex(WorkflowCompileError, "노드 계약"):
            ComfyWorkflowCompiler(capabilities).compile(
                "txt2img", "checkpoint.safetensors", {},
            )

    def test_disabled_scripts_are_not_inserted(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "alwayson_scripts": {
                    "ADetailer": {"args": [False, False, {
                        "ad_tab_enable": True, "ad_model": "face.pt",
                    }]},
                    "SAM3 Mask": {"args": [{
                        "sam3_enable": False, "sam3_mode": "Inpaint",
                    }]},
                },
            },
        )
        self.assertNotIn("ForgeNeoADetailer", _classes(graph))
        self.assertNotIn("ForgeNeoSAM3Mask", _classes(graph))

    def test_unrepresentable_adetailer_option_fails_before_queue(self):
        with self.assertRaisesRegex(WorkflowCompileError, "separate checkpoint"):
            ComfyWorkflowCompiler(_capabilities()).compile(
                "txt2img", "checkpoint.safetensors", {
                    "alwayson_scripts": {"ADetailer": {"args": [True, False, {
                        "ad_tab_enable": True, "ad_model": "face.pt",
                        "ad_use_checkpoint": True,
                    }]}},
                },
            )


class TestAdvancedWorkflowCompilation(unittest.TestCase):
    def test_preserves_caller_graph_and_augments_supported_seams(self):
        workflow = _custom_workflow()
        original = copy.deepcopy(workflow)
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "prompt": "new, <lora:ink:0.75>", "negative_prompt": "new neg",
                "width": 640, "height": 832, "batch_size": 2,
                "alwayson_scripts": {"NegPiP": {"args": [True]}},
            }, workflow=workflow,
        )
        self.assertEqual(workflow, original)
        self.assertEqual(graph["2"]["inputs"]["text"], "new")
        self.assertEqual(graph["3"]["inputs"]["text"], "new neg")
        self.assertEqual(graph["4"]["inputs"]["width"], 640)
        self.assertEqual(graph["4"]["inputs"]["height"], 832)
        self.assertIn("LoraLoader", _classes(graph))
        self.assertIn("ForgeNeoNegPip", _classes(graph))
        self.assertEqual(graph["7"]["inputs"]["images"], ["6", 0])

    def test_ksampler_advanced_maps_img2img_denoise_to_step_window(self):
        workflow = _ksampler_advanced_workflow()
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "img2img", "checkpoint.safetensors", {
                "steps": 40, "denoising_strength": 0.25,
            }, workflow=workflow, uploaded_image="source.png",
        )
        advanced = graph["5"]["inputs"]
        self.assertEqual(advanced["start_at_step"], 30)
        self.assertEqual(advanced["end_at_step"], 40)
        self.assertNotIn("denoise", advanced)

    def test_multiple_custom_samplers_fail_instead_of_using_insertion_order(self):
        workflow = _custom_workflow()
        workflow["8"] = copy.deepcopy(workflow["5"])
        with self.assertRaisesRegex(WorkflowCompileError, "sampler.*여러"):
            ComfyWorkflowCompiler(_capabilities()).compile(
                "txt2img", "checkpoint.safetensors", {}, workflow=workflow,
            )

    def test_hires_preserves_intermediary_latent_chain(self):
        workflow = _custom_workflow()
        workflow["8"] = {
            "class_type": "LatentUpscale",
            "inputs": {
                "samples": ["5", 0], "upscale_method": "bislerp",
                "width": 768, "height": 768, "crop": "disabled",
            },
        }
        workflow["6"]["inputs"]["samples"] = ["8", 0]
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors",
            {"enable_hr": True, "hr_scale": 1.5}, workflow=workflow,
        )
        hires_id, hires = _node(graph, "ForgeNeoHiresFix")
        self.assertEqual(hires["inputs"]["samples"], ["8", 0])
        self.assertEqual(graph["6"]["inputs"]["samples"], [hires_id, 0])

    def test_custom_module_and_postprocess_rewrites_stay_on_selected_branch(self):
        workflow = _custom_workflow()
        workflow.update({
            "8": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "unrelated"},
            },
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["4", 0], "vae": ["1", 2]},
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9", 0], "filename_prefix": "unrelated"},
            },
        })
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "prompt": "selected", "negative_prompt": "selected negative",
                "forge_additional_modules": [
                    "text/base.safetensors", "vae/image_vae.safetensors",
                ],
                "alwayson_scripts": {"ADetailer": {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face_yolov8n.pt",
                }]}},
            }, workflow=workflow,
        )
        _vae_id, vae = _node(graph, "VAELoader")
        _clip_id, clip = _node(graph, "CLIPLoader")
        self.assertEqual(graph["2"]["inputs"]["clip"], [_clip_id, 0])
        self.assertEqual(graph["3"]["inputs"]["clip"], [_clip_id, 0])
        self.assertEqual(graph["6"]["inputs"]["vae"], [_vae_id, 0])
        self.assertEqual(graph["8"]["inputs"]["clip"], ["1", 1])
        self.assertEqual(graph["9"]["inputs"]["vae"], ["1", 2])
        self.assertEqual(graph["10"]["inputs"]["images"], ["9", 0])
        adetailer_id, _adetailer = _node(graph, "ForgeNeoADetailer")
        self.assertEqual(graph["7"]["inputs"]["images"], [adetailer_id, 0])

    def test_postprocess_preserves_selected_branch_image_intermediary(self):
        workflow = _custom_workflow()
        workflow["8"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["6", 0], "upscale_method": "lanczos",
                "width": 640, "height": 640, "crop": "disabled",
            },
        }
        workflow["7"]["inputs"]["images"] = ["8", 0]
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "alwayson_scripts": {"ADetailer": {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face_yolov8n.pt",
                }]}},
            }, workflow=workflow,
        )
        adetailer_id, adetailer = _node(graph, "ForgeNeoADetailer")
        self.assertEqual(adetailer["inputs"]["image"], ["8", 0])
        self.assertEqual(graph["7"]["inputs"]["images"], [adetailer_id, 0])

    def test_custom_sampler_graph_fails_instead_of_ignoring_forge_parameters(self):
        workflow = _custom_workflow()
        workflow["5"]["class_type"] = "SamplerCustom"
        with self.assertRaisesRegex(WorkflowCompileError, "안전하게 자동 매핑"):
            ComfyWorkflowCompiler(_capabilities()).compile(
                "txt2img", "checkpoint.safetensors", {}, workflow=workflow,
            )

    def test_distinct_custom_negative_clip_survives_plain_prompt_updates(self):
        workflow = _custom_workflow()
        workflow["8"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "text/base.safetensors",
                "type": "stable_diffusion", "device": "default",
            },
        }
        workflow["3"]["inputs"]["clip"] = ["8", 0]
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "prompt": "updated positive", "negative_prompt": "updated negative",
            }, workflow=workflow,
        )
        self.assertEqual(graph["2"]["inputs"]["clip"], ["1", 1])
        self.assertEqual(graph["3"]["inputs"]["clip"], ["8", 0])
        self.assertEqual(graph["3"]["inputs"]["text"], "updated negative")

    def test_pixel_and_model_upscale_graphs(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        pixel = compiler.compile_upscale("source.png", {
            "upscaler_name": "Lanczos", "scale_factor": 2,
        })
        self.assertIn("ImageScaleBy", _classes(pixel))
        model = compiler.compile_upscale("source.png", {
            "upscaler_name": "4x-UltraSharp.pth", "scale_mode": "size",
            "target_width": 1200, "target_height": 800,
        })
        self.assertIn("ImageUpscaleWithModel", _classes(model))
        self.assertIn("ImageScale", _classes(model))

    def test_model_upscale_factor_uses_original_image_dimensions(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        for factor, width, height in ((2.0, 640, 480), (3.0, 960, 720), (1.5, 480, 360)):
            with self.subTest(factor=factor):
                graph = compiler.compile_upscale("source.png", {
                    "upscaler_name": "4x-UltraSharp.pth",
                    "scale_mode": "factor", "scale_factor": factor,
                }, source_width=320, source_height=240)
                resize_id, resize = _node(graph, "ImageScale")
                upscale_id, _ = _node(graph, "ImageUpscaleWithModel")
                self.assertEqual(resize["inputs"]["image"], [upscale_id, 0])
                self.assertEqual((resize["inputs"]["width"], resize["inputs"]["height"]), (width, height))
                # Forge extras API 처럼 서버 output 에 사본을 남기지 않는다(temp 미리보기로 반환).
                self.assertEqual(_node(graph, "PreviewImage")[1]["inputs"]["images"], [resize_id, 0])
                self.assertNotIn("SaveImage", _classes(graph))

    def test_model_upscale_factor_rejects_missing_source_dimensions(self):
        with self.assertRaisesRegex(WorkflowCompileError, "원본 너비와 높이"):
            ComfyWorkflowCompiler(_capabilities()).compile_upscale("source.png", {
                "upscaler_name": "4x-UltraSharp.pth", "scale_factor": 2,
            })

    def test_standalone_postprocess_has_no_wasteful_base_sampler(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
            "checkpoint.safetensors",
            {
                "prompt": "portrait",
                "alwayson_scripts": {"ADetailer": {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face_yolov8n.pt",
                }]}},
            },
            uploaded_image="source.png",
        )
        self.assertNotIn("ForgeNeoKSamplerCNS", _classes(graph))
        self.assertIn("ForgeNeoADetailer", _classes(graph))
        _ad_id, ad = _node(graph, "ForgeNeoADetailer")
        load_id, _load = _node(graph, "LoadImage")
        self.assertEqual(ad["inputs"]["image"], [load_id, 0])

    def test_refine_postprocess_uses_independent_forge_refine_node(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
            "checkpoint.safetensors",
            {
                "prompt": "portrait",
                "alwayson_scripts": sam3_args.build_alwayson({
                    "sam3_mode": "Inpaint", "sam3_prompt": "face/eyes",
                }),
            },
            uploaded_image="source.png",
            sam3_detailer_class="ForgeNeoSAM3Refine",
        )
        self.assertIn("ForgeNeoSAM3Mask", _classes(graph))
        self.assertIn("ForgeNeoSAM3Refine", _classes(graph))
        self.assertNotIn("ForgeNeoSAM3Detailer", _classes(graph))

        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 SAM3"):
            ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
                "checkpoint.safetensors",
                {"alwayson_scripts": sam3_args.build_alwayson({})},
                uploaded_image="source.png",
                sam3_detailer_class="UnknownDetailer",
            )

    def test_sam3_detailer_samples_at_forge_generation_resolution(self):
        scripts = sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face"})
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors",
            {"prompt": "portrait", "width": 832, "height": 1216, "enable_hr": True,
             "hr_scale": 1.5, "alwayson_scripts": scripts},
        )
        _sam_id, sam = _node(graph, "ForgeNeoSAM3Detailer")
        # Forge p.width/p.height = 기본 생성 해상도(Hires 이전)
        self.assertEqual((sam["inputs"]["target_width"], sam["inputs"]["target_height"]), (832, 1216))

        # 단독 후처리: Forge 단독 img2img 처럼 payload 크기(= 입력 이미지 크기)로 샘플링한다.
        graph = ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
            "checkpoint.safetensors",
            {"prompt": "portrait", "width": 1248, "height": 1824, "alwayson_scripts": scripts},
            uploaded_image="source.png", sam3_detailer_class="ForgeNeoSAM3Refine",
        )
        _sam_id, sam = _node(graph, "ForgeNeoSAM3Refine")
        self.assertEqual((sam["inputs"]["target_width"], sam["inputs"]["target_height"]), (1248, 1824))

        # 명시한 _sam3_processing_* 가 payload 크기보다 우선한다(백엔드가 이미지 크기로 고정).
        graph = ComfyWorkflowCompiler(_capabilities()).compile_postprocess(
            "checkpoint.safetensors",
            {"prompt": "portrait", "width": 1248, "height": 1824,
             "_sam3_processing_width": 832, "_sam3_processing_height": 1216,
             "alwayson_scripts": scripts},
            uploaded_image="source.png", sam3_detailer_class="ForgeNeoSAM3Refine",
        )
        _sam_id, sam = _node(graph, "ForgeNeoSAM3Refine")
        self.assertEqual((sam["inputs"]["target_width"], sam["inputs"]["target_height"]), (832, 1216))

        for payload_size, expected in (({}, (0, 0)), ({"width": 32, "height": 20000}, (64, 8192))):
            with self.subTest(payload_size=payload_size):
                self.assertEqual(
                    ComfyWorkflowCompiler._sam3_processing_size(payload_size), expected,
                )

    def test_sam3_mask_cache_model_follows_keep_in_ram_setting_like_forge(self):
        # Forge: 'Unload after' 를 끄면 번들을 장치에 계속 두고, 켜면 설정
        # sam3_unload_keep_in_ram 이 CPU RAM 보관 여부를 정한다. 앱 설정이 그 역할이다.
        cases = (
            (True, True, True),    # 기본: 언로드 후 RAM 보관
            (False, True, False),  # 설정 끔: 매번 로드, 보관본 해제
            (False, False, True),  # Unload after 끔: 설정과 무관하게 장치에 유지
            (True, False, True),
        )
        for keep_in_ram, unload_after, expected in cases:
            with self.subTest(keep_in_ram=keep_in_ram, unload_after=unload_after):
                compiler = ComfyWorkflowCompiler(_capabilities(), sam3_keep_in_ram=keep_in_ram)
                inpaint = compiler.compile(
                    "txt2img", "checkpoint.safetensors",
                    {"prompt": "portrait", "alwayson_scripts": sam3_args.build_alwayson({
                        "sam3_mode": "Inpaint", "sam3_unload_after": unload_after,
                    })},
                )
                mask_only = compiler.compile_sam3_mask_only(
                    {"alwayson_scripts": sam3_args.build_alwayson({
                        "sam3_mode": "Mask only", "sam3_unload_after": unload_after,
                    })},
                    uploaded_image="source.png",
                )
                for graph in (inpaint, mask_only):
                    _mask_id, mask = _node(graph, "ForgeNeoSAM3Mask")
                    self.assertIs(mask["inputs"]["cache_model"], expected)
        # 설정을 모르는 호출부(사전 검증 등)는 노드 기본값과 같은 보관을 쓴다.
        self.assertTrue(ComfyWorkflowCompiler().sam3_keep_in_ram)

    def test_sam3_mask_only_does_not_load_a_diffusion_model(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile_sam3_mask_only(
            {"alwayson_scripts": sam3_args.build_alwayson({
                "sam3_mode": "Mask only", "sam3_prompt": "face",
                "sam3_preview_overlay": True,
            })},
            uploaded_image="source.png",
        )
        self.assertNotIn("CheckpointLoaderSimple", _classes(graph))
        self.assertNotIn("ForgeNeoSAM3Detailer", _classes(graph))
        self.assertIn("ForgeNeoSAM3Mask", _classes(graph))
        mask_id, _mask = _node(graph, "ForgeNeoSAM3Mask")
        _save_id, save = _node(graph, "PreviewImage")
        self.assertEqual(save["inputs"]["images"], [mask_id, 3])

    def test_only_explicit_save_images_leaves_a_comfy_output_copy(self):
        base = {"prompt": "portrait"}
        compiler = ComfyWorkflowCompiler(_capabilities())
        for payload, expected in (
            ({**base, "save_images": True, "filename_prefix": "AIStudio/main"}, "SaveImage"),
            ({**base, "save_images": False}, "PreviewImage"),
            (base, "PreviewImage"),  # Forge API 기본값(save_images=False)
        ):
            with self.subTest(save_images=payload.get("save_images")):
                graph = compiler.compile("txt2img", "checkpoint.safetensors", payload)
                classes = _classes(graph)
                self.assertIn(expected, classes)
                self.assertEqual(
                    sum(cls in {"SaveImage", "PreviewImage"} for cls in classes), 1,
                )
                _id, output = _node(graph, expected)
                decode_id, _decode = _node(graph, "VAEDecode")
                self.assertEqual(output["inputs"]["images"], [decode_id, 0])
                if expected == "SaveImage":
                    self.assertEqual(output["inputs"]["filename_prefix"], "AIStudio/main")
                else:
                    self.assertNotIn("filename_prefix", output["inputs"])

        postprocess = compiler.compile_postprocess(
            "checkpoint.safetensors",
            {**base, "save_images": False, "alwayson_scripts": sam3_args.build_alwayson({
                "sam3_mode": "Inpaint", "sam3_prompt": "face",
            })},
            uploaded_image="source.png",
        )
        self.assertIn("PreviewImage", _classes(postprocess))
        self.assertNotIn("SaveImage", _classes(postprocess))


class TestCnsCompilation(unittest.TestCase):
    """CNS is the suite MODEL's sampler wrapper; the sampler nodes only carry its values."""

    def _values(self, node: dict) -> dict:
        return {key: node["inputs"][key] for key in _ORIGIN_CNS_DEFAULTS}

    def test_every_fallback_is_the_original_node_default(self):
        compiler = ComfyWorkflowCompiler(_capabilities())
        for scripts, enabled in (({}, False), (_cns_only_alwayson(), True)):
            with self.subTest(enabled=enabled):
                graph = compiler.compile("txt2img", "checkpoint.safetensors", {
                    "enable_hr": True, "hr_scale": 1.5, "alwayson_scripts": scripts,
                })
                for class_type in ("ForgeNeoKSamplerCNS", "ForgeNeoHiresFix"):
                    _node_id, node = _node(graph, class_type)
                    self.assertIs(node["inputs"]["cns_enabled"], enabled, class_type)
                    self.assertEqual(self._values(node), _ORIGIN_CNS_DEFAULTS, class_type)
        custom = compiler.compile("txt2img", "checkpoint.safetensors", {
            "alwayson_scripts": _cns_only_alwayson(),
        }, workflow=_custom_workflow())
        self.assertEqual(custom["5"]["class_type"], "ForgeNeoKSamplerCNS")
        self.assertIs(custom["5"]["inputs"]["cns_enabled"], True)
        self.assertEqual(self._values(custom["5"]), _ORIGIN_CNS_DEFAULTS)

    def test_fallbacks_equal_the_bundled_node_defaults(self):
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_cns

        self.assertEqual(
            {f"cns_{name}": value for name, value in guidance_cns.CNS_DEFAULTS.items()},
            _ORIGIN_CNS_DEFAULTS,
        )

    def test_guidance_values_win_over_the_fallbacks(self):
        guidance = anima_guidance.default_settings()
        guidance.update({
            "guid_cns_enabled": True, "guid_cns_strength": 0.5,
            "guid_cns_gamma_power": 0.75, "guid_cns_gamma_scale": 3.0,
        })
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "enable_hr": True, "hr_scale": 1.5,
                "alwayson_scripts": anima_guidance.build_alwayson(guidance),
            },
        )
        for class_type in ("ForgeNeoKSamplerCNS", "ForgeNeoHiresFix"):
            _node_id, node = _node(graph, class_type)
            self.assertEqual(self._values(node), {
                "cns_strength": 0.5, "cns_gamma_power": 0.75, "cns_gamma_scale": 3.0,
            }, class_type)

    def test_cns_keeps_a_ksampler_advanced_graph_on_its_own_sampler(self):
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {
                "alwayson_scripts": _cns_only_alwayson(),
            }, workflow=_ksampler_advanced_workflow(),
        )
        sampler = graph["5"]
        self.assertEqual(sampler["class_type"], "KSamplerAdvanced")
        self.assertFalse([
            key for key in sampler["inputs"]
            if key.startswith(("cns_", "spectrum_", "speed_"))
        ])
        # The sampler runs the suite's MODEL, which carries the CNS wrapper.
        suite_id, suite = _node(graph, "ForgeNeoAnimaGuidanceSuite")
        self.assertTrue(json.loads(suite["inputs"]["settings_json"])["guid_cns_enabled"])
        self.assertEqual(sampler["inputs"]["model"], [suite_id, 0])

    def test_spectrum_and_speed_still_need_a_ksampler(self):
        capabilities = _capabilities()
        capabilities["DiTSpectrumPatch"] = {"input": {"required": {}}}
        for flag in ("spectrum_enabled", "speed_enabled"):
            with self.subTest(flag=flag), self.assertRaisesRegex(
                WorkflowCompileError, "Spectrum/SPEED",
            ):
                ComfyWorkflowCompiler(capabilities).compile(
                    "txt2img", "checkpoint.safetensors", {
                        flag: True, "alwayson_scripts": _cns_only_alwayson(),
                    }, workflow=_ksampler_advanced_workflow(),
                )

    @staticmethod
    def _stale_pack_capabilities() -> dict:
        """/object_info of pack 1.3.0: same suite/sampler contracts, no per-step CNS node.

        (1.3.0's ForgeNeoKSamplerCNS/ForgeNeoAnimaGuidanceSuite take the same input
        names as 1.4.0's, so validate()'s contract check alone passes them.)
        """
        capabilities = _capabilities()
        del capabilities["ForgeNeoCNSSamplerPatch"]
        return capabilities

    def test_the_stale_pack_marker_is_a_bundled_node(self):
        from comfy_custom_nodes.ai_studio_forge_parity import generation
        from core import comfy_workflow_compiler

        self.assertIn(comfy_workflow_compiler._PER_STEP_CNS_MARKER, generation.NODE_CLASS_MAPPINGS)

    def test_a_stale_pack_refuses_cns_instead_of_dropping_it(self):
        # Pack 1.3.0 passes the contract check but colours only the initial noise on
        # ForgeNeoKSamplerCNS/HiresFix and nothing on KSamplerAdvanced (plan §5.2 #1).
        compiler = ComfyWorkflowCompiler(self._stale_pack_capabilities())
        payload = {"enable_hr": True, "hr_scale": 1.5, "alwayson_scripts": _cns_only_alwayson()}
        for label, workflow in (
            ("main", None), ("KSampler", _custom_workflow()),
            ("KSamplerAdvanced", _ksampler_advanced_workflow()),
        ):
            with self.subTest(workflow=label), self.assertRaisesRegex(
                WorkflowCompileError, "ForgeNeoCNSSamplerPatch.*번들 노드 팩을 갱신",
            ):
                compiler.compile("txt2img", "checkpoint.safetensors", payload, workflow=workflow)

    def test_a_stale_pack_still_compiles_without_cns(self):
        compiler = ComfyWorkflowCompiler(self._stale_pack_capabilities())
        guidance = anima_guidance.default_settings()
        guidance.update({"guid_enabled": True, "guid_cns_enabled": False})
        for workflow in (None, _ksampler_advanced_workflow()):
            graph = compiler.compile("txt2img", "checkpoint.safetensors", {
                "enable_hr": True, "hr_scale": 1.5,
                "alwayson_scripts": anima_guidance.build_alwayson(guidance),
            }, workflow=workflow)
            self.assertIn("ForgeNeoAnimaGuidanceSuite", _classes(graph))

    def test_a_hand_built_cns_sampler_on_a_stale_pack_is_refused(self):
        workflow = _custom_workflow()
        workflow["5"]["class_type"] = "ForgeNeoKSamplerCNS"
        workflow["5"]["inputs"].update({"cns_enabled": True, **_ORIGIN_CNS_DEFAULTS})
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoCNSSamplerPatch"):
            ComfyWorkflowCompiler(self._stale_pack_capabilities()).compile(
                "txt2img", "checkpoint.safetensors", {}, workflow=workflow,
            )
        workflow["5"]["inputs"]["cns_enabled"] = False
        ComfyWorkflowCompiler(self._stale_pack_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {}, workflow=workflow,
        )

    def test_sampler_custom_is_refused_for_payload_mapping_not_for_cns(self):
        # DELIBERATE OPEN ITEM, not parity: plan §5.3 A / §8.3 APP-COMP want
        # SamplerCustom(Advanced) allowed — the original CNS node's own host is
        # KSamplerSelect -> CNSSamplerPatch -> SamplerCustomAdvanced
        # (origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:441-609).
        # Allowing it needs a new payload mapping onto BasicScheduler/CFGGuider/
        # RandomNoise (steps/CFG/seed/denoise), and the workflow picker and
        # frontend/src/utils/gateWorkflowInfo.ts share this refusal.  Deferred until the
        # user/lead decides the scope; until then this pins today's behaviour.
        for class_type in ("SamplerCustom", "SamplerCustomAdvanced"):
            workflow = _custom_workflow()
            workflow["5"]["class_type"] = class_type
            with self.subTest(class_type=class_type), self.assertRaisesRegex(
                WorkflowCompileError, "steps/CFG/denoise",
            ):
                ComfyWorkflowCompiler(_capabilities()).compile(
                    "txt2img", "checkpoint.safetensors", {
                        "alwayson_scripts": _cns_only_alwayson(),
                    }, workflow=workflow,
                )


class TestStalePackPag(unittest.TestCase):
    """PAG legacy/head indices need pack 1.4.0 (the original PAG node).

    Pack 1.3.0's ForgeNeoAnimaGuidanceSuite/ForgeNeoAnimaSafePAG take the same inputs as
    1.4.0's but raise inside patch() on legacy/heads (1.3.0 guidance.py:1078-1085 and
    :862-865), so the contract check alone lets a stale remote pack fail at run time.
    """

    @staticmethod
    def _capabilities(*, stale: bool) -> dict:
        capabilities = _capabilities()
        capabilities["ForgeNeoAnimaSafePAG"] = {"input": {"required": {}}}
        if stale:
            del capabilities["ForgeNeoCNSSamplerPatch"]  # 1.3.0 publishes no 1.4.0 class
        return capabilities

    @staticmethod
    def _pag_payload(**settings) -> dict:
        guidance = anima_guidance.default_settings()
        guidance.update({"guid_enabled": True, "guid_attn_method": "PAG", **settings})
        return {"alwayson_scripts": anima_guidance.build_alwayson(guidance)}

    @staticmethod
    def _safe_pag_workflow(**inputs) -> dict:
        workflow = _custom_workflow()
        workflow["8"] = {"class_type": "ForgeNeoAnimaSafePAG", "inputs": {
            "model": ["1", 0], "enabled": True, "scale": 4.0, "block_indices": "18",
            "perturbation_strength": 0.75, "head_indices": "", "start_percent": 0.0,
            "end_percent": 0.7, "rescale": 0.2, "rescale_mode": "full", **inputs,
        }}
        workflow["5"]["inputs"]["model"] = ["8", 0]
        return workflow

    def test_the_marker_is_a_bundled_node(self):
        from comfy_custom_nodes.ai_studio_forge_parity import generation
        from core import comfy_workflow_compiler

        self.assertIn(comfy_workflow_compiler._PAG_ORIGIN_MARKER, generation.NODE_CLASS_MAPPINGS)

    def test_a_stale_pack_refuses_pag_legacy_and_heads(self):
        compiler = ComfyWorkflowCompiler(self._capabilities(stale=True))
        for label, settings in (
            ("legacy", {"guid_legacy_attn": True}),
            ("heads", {"guid_head_indices": "0,2"}),
            ("both", {"guid_legacy_attn": True, "guid_head_indices": "7-4"}),
        ):
            for workflow_label, workflow in (("main", None), ("KSampler", _custom_workflow())):
                with self.subTest(settings=label, workflow=workflow_label), self.assertRaisesRegex(
                    WorkflowCompileError, "PAG legacy.*1\\.4\\.0.*번들 노드 팩을 갱신",
                ):
                    compiler.compile(
                        "txt2img", "checkpoint.safetensors", self._pag_payload(**settings),
                        workflow=workflow,
                    )

    def test_the_current_pack_compiles_pag_legacy_and_heads(self):
        graph = ComfyWorkflowCompiler(self._capabilities(stale=False)).compile(
            "txt2img", "checkpoint.safetensors",
            self._pag_payload(guid_legacy_attn=True, guid_head_indices="0,2"),
        )
        _, suite = _node(graph, "ForgeNeoAnimaGuidanceSuite")
        settings = json.loads(suite["inputs"]["settings_json"])
        self.assertTrue(settings["guid_legacy_attn"])
        self.assertEqual(settings["guid_head_indices"], "0,2")

    def test_a_stale_pack_still_compiles_what_it_can_run(self):
        # 1.3.0 checks legacy/heads only when attention guidance is on (its method PAG/SEG).
        compiler = ComfyWorkflowCompiler(self._capabilities(stale=True))
        for label, payload in (
            ("plain PAG", self._pag_payload()),
            ("attention off", self._pag_payload(
                guid_enabled=False, guid_slg_on=True,
                guid_legacy_attn=True, guid_head_indices="0,2",
            )),
            ("method none", self._pag_payload(
                guid_attn_method="None", guid_legacy_attn=True, guid_head_indices="0,2",
            )),
        ):
            with self.subTest(payload=label):
                compiler.compile("txt2img", "checkpoint.safetensors", payload)

    def test_a_hand_built_safe_pag_with_heads_on_a_stale_pack_is_refused(self):
        stale = ComfyWorkflowCompiler(self._capabilities(stale=True))
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoCNSSamplerPatch"):
            stale.compile(
                "txt2img", "checkpoint.safetensors", {},
                workflow=self._safe_pag_workflow(head_indices="0,2"),
            )
        # What 1.3.0's node returns from before its head check, or cannot be read statically.
        for label, inputs in (
            ("no heads", {}), ("disabled", {"enabled": False, "head_indices": "0,2"}),
            ("scale 0", {"scale": 0.0, "head_indices": "0,2"}),
            ("strength 0", {"perturbation_strength": 0.0, "head_indices": "0,2"}),
            ("linked heads", {"head_indices": ["9", 0]}),
        ):
            with self.subTest(inputs=label):
                stale.compile(
                    "txt2img", "checkpoint.safetensors", {},
                    workflow=self._safe_pag_workflow(**inputs),
                )
        ComfyWorkflowCompiler(self._capabilities(stale=False)).compile(
            "txt2img", "checkpoint.safetensors", {},
            workflow=self._safe_pag_workflow(head_indices="0,2"),
        )


if __name__ == "__main__":
    unittest.main()
