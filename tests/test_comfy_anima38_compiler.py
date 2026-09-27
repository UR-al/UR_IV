"""Focused contracts for ANIMA Semantic Connector workflow compilation."""
from __future__ import annotations

import copy
from graphlib import TopologicalSorter
import unittest

from core import anima38, anima_guidance
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from tests.test_comfy_workflow_compiler import (
    _capabilities,
    _choice,
    _classes,
    _custom_workflow,
    _node,
)


V1_MODEL = "anima-native.safetensors"
V2_MODEL = "Anima-3.8B-v1.1.safetensors"
NATIVE_CLIP = "text/qwen_3_06b_base.safetensors"
QWEN35 = "text/qwen35_4b.safetensors"
QWEN35_ALT = "text/qwen3.5-4b-alt.safetensors"
ADAPTER = "text/Anima-3.8B-expanded_adapter.safetensors"
VAE = "vae/image_vae.safetensors"


def _typed(name: str):
    return [name, {}]


def _anima_capabilities(*, qwen: bool = True) -> dict:
    result = _capabilities()
    result["UNETLoader"]["input"]["required"]["unet_name"] = _choice(
        V1_MODEL, V2_MODEL,
    )
    result["CLIPLoader"]["input"]["required"]["clip_name"] = _choice(
        NATIVE_CLIP,
        QWEN35,
        QWEN35_ALT,
        ADAPTER,
        "text/base.safetensors",
        "text/adapter.safetensors",
        "text/third.safetensors",
    )
    result.update({
        "ForgeNeoAnima38V2Loader": {"input": {"required": {
            "model_name": _choice(V2_MODEL),
        }}},
        "ForgeNeoAnimaQwen35Loader": {"input": {"required": {
            "qwen35_model": _choice(*(QWEN35, QWEN35_ALT) if qwen else ()),
        }}},
        "ForgeNeoAnimaQwen35Prompt": {"input": {"required": {
            "model": _typed("MODEL"),
            "native_clip": _typed("CLIP"),
            "qwen35_clip": _typed("CLIP"),
            "adapter_name": _choice(ADAPTER),
            "prompt": _typed("STRING"),
            "adapter_strength": _typed("FLOAT"),
        }}},
        "ForgeNeoAnima38V2Prompt": {"input": {"required": {
            "model": _typed("MODEL"),
            "native_clip": _typed("CLIP"),
            "qwen35_clip": _typed("CLIP"),
            "prompt": _typed("STRING"),
        }}},
        "ForgeNeoAnimaLoraLoader": {"input": {"required": {
            "model": _typed("MODEL"),
            "clip": _typed("CLIP"),
            "lora_name": _choice(
                "styles/ink.safetensors", "characters/alice.safetensors",
            ),
            "strength_model": _typed("FLOAT"),
            "strength_clip": _typed("FLOAT"),
        }}},
    })
    return result


def _modules(*, qwen: str = QWEN35, adapter: str = "") -> list[str]:
    values = [VAE, NATIVE_CLIP]
    if qwen:
        values.append(qwen)
    if adapter:
        values.append(adapter)
    return values


def _script(*args):
    return {anima38.SCRIPT_NAME: {"args": list(args)}}


def _split_custom_workflow() -> dict:
    workflow = _custom_workflow()
    workflow["1"] = {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": V1_MODEL, "weight_dtype": "default"},
    }
    workflow["8"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": NATIVE_CLIP, "type": "stable_diffusion", "device": "default"},
    }
    workflow["9"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": VAE},
    }
    workflow["2"]["inputs"]["clip"] = ["8", 0]
    workflow["3"]["inputs"]["clip"] = ["8", 0]
    workflow["6"]["inputs"]["vae"] = ["9", 0]
    return workflow


class TestAnima38Settings(unittest.TestCase):
    def test_positional_and_dict_forms_follow_the_six_arg_contract(self):
        positional = anima38.parse_args([
            "yes", "custom.safetensors", 9, "true", -1, 1,
        ])
        self.assertEqual(positional.as_args(), [
            True, "custom.safetensors", 2.0, True, 0.0, True,
        ])
        mapped = anima38.parse_script_block({"args": [{
            "enabled": True, "negative_strength": "1.25", "bypass": "off",
        }]})
        self.assertTrue(mapped.enabled)
        self.assertEqual(mapped.adapter, anima38.DEFAULT_ADAPTER)
        self.assertEqual(mapped.negative_strength, 1.25)
        self.assertFalse(mapped.bypass)

    def test_invalid_values_use_forge_defaults(self):
        parsed = anima38.parse_args([False, "", float("nan"), False, "bad"])
        self.assertEqual(parsed.adapter, anima38.DEFAULT_ADAPTER)
        self.assertEqual(parsed.strength, 1.0)
        self.assertEqual(parsed.negative_strength, 1.0)


class TestAnima38DefaultCompilation(unittest.TestCase):
    def test_generated_semantic_workflow_with_guidance_roundtrips_without_a_cycle(self):
        compiler = ComfyWorkflowCompiler(_anima_capabilities())
        guidance = anima_guidance.default_settings()
        guidance["guid_apg_enabled"] = True
        payload = {
            "prompt": "original", "negative_prompt": "artifact",
            "forge_additional_modules": _modules(),
            "alwayson_scripts": anima_guidance.build_alwayson(guidance),
        }
        original = compiler.compile("txt2img", V2_MODEL, payload)
        self.assertIn("ForgeNeoAnimaGuidanceSuite", _classes(original))
        updated = compiler.compile("txt2img", V2_MODEL, {
            **payload, "prompt": "updated", "alwayson_scripts": {},
        }, workflow=original)
        dependencies = {
            node_id: [
                str(value[0]) for value in node.get("inputs", {}).values()
                if isinstance(value, list) and len(value) == 2 and str(value[0]) in updated
            ]
            for node_id, node in updated.items()
        }
        self.assertEqual(len(tuple(TopologicalSorter(dependencies).static_order())), len(updated))

    def test_generated_v2_workflow_roundtrip_updates_prompts_dimensions_and_bypass(self):
        compiler = ComfyWorkflowCompiler(_anima_capabilities())
        payload = {
            "prompt": "original", "negative_prompt": "old negative",
            "forge_additional_modules": _modules(),
            "width": 512, "height": 512, "batch_size": 1,
        }
        original = compiler.compile("txt2img", V2_MODEL, payload)
        original_copy = copy.deepcopy(original)
        updated_payload = {
            **payload, "prompt": "updated", "negative_prompt": "new negative",
            "width": 768, "height": 1024, "batch_size": 2,
            "alwayson_scripts": _script(False, ADAPTER, 1, True, 1, False),
        }
        updated = compiler.compile("txt2img", V2_MODEL, updated_payload, workflow=original)
        self.assertEqual(original, original_copy)
        self.assertEqual(
            {n["inputs"]["prompt"] for n in updated.values() if n["class_type"] == "ForgeNeoAnima38V2Prompt"},
            {"updated", "new negative"},
        )
        latent = _node(updated, "ForgeNeoLatentInput")[1]["inputs"]
        self.assertEqual((latent["width"], latent["height"], latent["batch_size"]), (768, 1024, 2))
        bypassed = compiler.compile("txt2img", V2_MODEL, {
            **updated_payload, "alwayson_scripts": _script(False, ADAPTER, 1, False, 1, True),
        }, workflow=updated)
        self.assertNotIn("ForgeNeoAnima38V2Prompt", _classes(bypassed))
        self.assertEqual(
            {n["inputs"]["text"] for n in bypassed.values() if n["class_type"] == "CLIPTextEncode"},
            {"updated", "new negative"},
        )

    def test_generated_v2_workflow_can_switch_to_a_native_anima_model(self):
        compiler = ComfyWorkflowCompiler(_anima_capabilities())
        payload = {"prompt": "portrait", "forge_additional_modules": _modules()}
        original = compiler.compile("txt2img", V2_MODEL, payload)
        updated = compiler.compile("txt2img", V1_MODEL, {
            **payload, "forge_additional_modules": _modules(qwen=""),
        }, workflow=original)
        self.assertNotIn("ForgeNeoAnima38V2Loader", _classes(updated))
        self.assertNotIn("ForgeNeoAnima38V2Prompt", _classes(updated))
        self.assertEqual(_node(updated, "UNETLoader")[1]["inputs"]["unet_name"], V1_MODEL)

    def test_v2_shift_preserves_the_anima_unit_timestep_scale(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {
                "prompt": "portrait",
                "forge_additional_modules": _modules(),
                "distilled_cfg_scale": 3.0,
            },
        )

        loader_id, _loader = _node(graph, "ForgeNeoAnima38V2Loader")
        shift_id, shift = _node(graph, "ForgeNeoModelSamplingShift")
        self.assertEqual(shift["inputs"], {
            "model": [loader_id, 0], "shift": 3.0,
        })
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["model"], [shift_id, 0])
        self.assertNotIn("ModelSamplingSD3", _classes(graph))

    def test_v2_auto_plan_prioritises_bundle_loader_and_partitions_modules(self):
        payload = {
            "prompt": "portrait, <lora:ink:0.5>",
            "negative_prompt": "bad anatomy",
            "forge_additional_modules": _modules(adapter=ADAPTER),
            "alwayson_scripts": {"NegPiP": {"args": [True]}},
        }
        original = copy.deepcopy(payload)
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, payload,
        )
        self.assertEqual(payload, original)
        self.assertNotIn("UNETLoader", _classes(graph))
        loader_id, loader = _node(graph, "ForgeNeoAnima38V2Loader")
        self.assertEqual(loader["inputs"], {"model_name": V2_MODEL})

        clip_id, clip = _node(graph, "CLIPLoader")
        self.assertEqual(clip["inputs"]["clip_name"], NATIVE_CLIP)
        self.assertNotIn("DualCLIPLoader", _classes(graph))
        self.assertNotIn("TripleCLIPLoader", _classes(graph))

        lora_id, lora = _node(graph, "ForgeNeoAnimaLoraLoader")
        self.assertEqual(lora["inputs"]["model"], [loader_id, 0])
        self.assertEqual(lora["inputs"]["clip"], [clip_id, 0])
        negpip_id, negpip = _node(graph, "ForgeNeoNegPip")
        self.assertEqual(negpip["inputs"]["model"], [lora_id, 0])

        qwen_id, qwen = _node(graph, "ForgeNeoAnimaQwen35Loader")
        self.assertEqual(qwen["inputs"], {"qwen35_model": QWEN35})
        prompt_id, prompt = _node(graph, "ForgeNeoAnima38V2Prompt")
        self.assertEqual(prompt["inputs"], {
            "model": [negpip_id, 0],
            "native_clip": [negpip_id, 1],
            "qwen35_clip": [qwen_id, 0],
            "prompt": "portrait",
        })
        _negative_id, negative = _node(graph, "CLIPTextEncode")
        self.assertEqual(negative["inputs"], {
            "clip": [negpip_id, 1], "text": "bad anatomy",
        })
        _sampler_id, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["positive"], [prompt_id, 0])

    def test_v2_semantic_negative_is_fixed_strength_and_bypass_keeps_loader(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {
                "prompt": "positive", "negative_prompt": "negative",
                "forge_additional_modules": _modules(),
                "alwayson_scripts": _script(
                    False, ADAPTER, 0.2, True, 0.25, False,
                ),
            },
        )
        prompts = [
            node for node in graph.values()
            if node.get("class_type") == "ForgeNeoAnima38V2Prompt"
        ]
        self.assertEqual([node["inputs"]["prompt"] for node in prompts], [
            "positive", "negative",
        ])
        self.assertTrue(all("adapter_strength" not in node["inputs"] for node in prompts))
        self.assertNotIn("CLIPTextEncode", _classes(graph))

        bypass_caps = _anima_capabilities(qwen=False)
        bypass = ComfyWorkflowCompiler(bypass_caps).compile(
            "txt2img", V2_MODEL, {
                "prompt": "positive", "negative_prompt": "negative",
                "forge_additional_modules": _modules(qwen=""),
                "alwayson_scripts": _script(
                    False, ADAPTER, 1.0, False, 1.0, True,
                ),
            },
        )
        self.assertIn("ForgeNeoAnima38V2Loader", _classes(bypass))
        self.assertNotIn("UNETLoader", _classes(bypass))
        self.assertNotIn("ForgeNeoAnimaQwen35Loader", _classes(bypass))
        self.assertNotIn("ForgeNeoAnima38V2Prompt", _classes(bypass))
        self.assertEqual(_classes(bypass).count("CLIPTextEncode"), 2)

    def test_v1_maps_detected_qwen_adapter_and_both_strengths(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V1_MODEL, {
                "prompt": "positive", "negative_prompt": "negative",
                "forge_additional_modules": _modules(
                    qwen=QWEN35_ALT, adapter=ADAPTER,
                ),
                "alwayson_scripts": _script(
                    True, ADAPTER, 0.65, True, 0.25, False,
                ),
            },
        )
        self.assertIn("UNETLoader", _classes(graph))
        self.assertNotIn("ForgeNeoAnima38V2Loader", _classes(graph))
        _qwen_id, qwen = _node(graph, "ForgeNeoAnimaQwen35Loader")
        self.assertEqual(qwen["inputs"]["qwen35_model"], QWEN35_ALT)
        prompts = [
            node for node in graph.values()
            if node.get("class_type") == "ForgeNeoAnimaQwen35Prompt"
        ]
        self.assertEqual([node["inputs"]["adapter_strength"] for node in prompts], [
            0.65, 0.25,
        ])
        self.assertTrue(all(node["inputs"]["adapter_name"] == ADAPTER for node in prompts))

    def test_v1_module_pair_alone_stays_native_like_forge(self):
        """(P9) Forge 는 v1 을 enabled 일 때만 켠다 — Qwen3.5·어댑터 모듈 쌍만으로는 순정(예전 자동 켜짐 없음).
        그 모듈은 여전히 네이티브 CLIP 입력으로 새지 않는다."""
        for label, payload in (("블록 없음", {}),
                               ("v1 꺼짐", {"alwayson_scripts": _script(False, ADAPTER, 1.0, True, 1.0, False)})):
            with self.subTest(label):
                graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
                    "txt2img", V1_MODEL, {"forge_additional_modules": _modules(adapter=ADAPTER), **payload},
                )
                self.assertNotIn("ForgeNeoAnimaQwen35Prompt", _classes(graph))
                self.assertNotIn("ForgeNeoAnimaQwen35Loader", _classes(graph))
                self.assertEqual(_classes(graph).count("CLIPTextEncode"), 2)
                _clip_id, clip = _node(graph, "CLIPLoader")
                self.assertEqual(clip["inputs"]["clip_name"], NATIVE_CLIP)

    def test_v1_with_a_differently_named_comfy_adapter(self):
        """(P9 리뷰 2) 기본 이름이 아닌 어댑터만 있는 ComfyUI: 카드에서 그 어댑터를 고르면 v1 이 된다(카드 선택지는
        object_info 에서 온다 — core.anima38.comfy_adapter_choices). 고르지 않고 켜면 무엇을 골라야 하는지 말하고 멈춘다."""
        alt_adapter = "text/MyAnima_v1_adapter.safetensors"
        capabilities = _anima_capabilities()
        capabilities["ForgeNeoAnimaQwen35Prompt"]["input"]["required"]["adapter_name"] = _choice(alt_adapter)
        base = {"forge_additional_modules": _modules(adapter=alt_adapter)}
        self.assertEqual(anima38.comfy_adapter_choices(capabilities), [alt_adapter])
        picked = anima38.as_dict(anima38.Anima38Settings(enabled=True, adapter=alt_adapter))
        graph = ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", V1_MODEL, {**base, "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [picked]}}})
        prompts = [node for node in graph.values() if node["class_type"] == "ForgeNeoAnimaQwen35Prompt"]
        self.assertEqual([node["inputs"]["adapter_name"] for node in prompts], [alt_adapter])
        unpicked = anima38.as_dict(anima38.Anima38Settings(enabled=True))
        with self.assertRaisesRegex(WorkflowCompileError, r"어댑터 칸.*MyAnima_v1_adapter"):
            ComfyWorkflowCompiler(capabilities).compile(
                "txt2img", V1_MODEL, {**base, "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [unpicked]}}})

    def test_v1_when_the_pack_found_no_adapter_names_what_to_install(self):
        """(P9 리뷰 2차 1) 팩이 v1 어댑터를 못 찾으면 콤보가 자리표시자 이름 하나뿐이다(vendor prompt.py
        adapter_candidates — `sorted(candidates) or [COMFY_ADAPTER_FALLBACK]`). 그 이름은 파일이 아니라 고르면 큐에
        들어간 뒤 실행 때 FileNotFoundError 가 난다 — 고르라고 권하지 말고 무엇을 설치해야 하는지 말하며 큐 전에 멈춘다.
        같은 이름의 진짜 파일(text_encoders 목록 = CLIPLoader.clip_name 에 보임)은 자리표시자가 아니다."""
        fallback = anima38.COMFY_ADAPTER_FALLBACK
        capabilities = _anima_capabilities()
        capabilities["ForgeNeoAnimaQwen35Prompt"]["input"]["required"]["adapter_name"] = _choice(fallback)
        base = {"forge_additional_modules": _modules()}
        for label, adapter in (("기본 어댑터", anima38.DEFAULT_ADAPTER), ("예전 안내대로 자리표시자를 고름", fallback)):
            with self.subTest(label):
                block = anima38.as_dict(anima38.Anima38Settings(enabled=True, adapter=adapter))
                with self.assertRaises(WorkflowCompileError) as caught:
                    ComfyWorkflowCompiler(capabilities).compile(
                        "txt2img", V1_MODEL, {**base, "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [block]}}})
                message = str(caught.exception)
                self.assertIn("찾지 못했습니다", message)
                self.assertIn("자리표시자", message)
                self.assertIn("anima_progressive_qwen35_cross_adapter_v1", message)
                self.assertIn("text_encoders", message)
                self.assertNotIn("고르세요", message)                       # 없는 파일을 고르라고 하지 않는다
        # v1 을 끄면 자리표시자는 상관없다 — 순정
        graph = ComfyWorkflowCompiler(capabilities).compile("txt2img", V1_MODEL, base)
        self.assertNotIn("ForgeNeoAnimaQwen35Prompt", _classes(graph))

        # 같은 이름의 진짜 파일이 text_encoders 에 있으면(업스트림 팩 기본 이름) 보통 어댑터처럼 고를 수 있다
        real = copy.deepcopy(capabilities)
        real["CLIPLoader"]["input"]["required"]["clip_name"][0].append(fallback)
        picked = anima38.as_dict(anima38.Anima38Settings(enabled=True, adapter=fallback))
        graph = ComfyWorkflowCompiler(real).compile(
            "txt2img", V1_MODEL, {**base, "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [picked]}}})
        prompts = [node for node in graph.values() if node["class_type"] == "ForgeNeoAnimaQwen35Prompt"]
        self.assertEqual([node["inputs"]["adapter_name"] for node in prompts], [fallback])
        unpicked = anima38.as_dict(anima38.Anima38Settings(enabled=True))
        with self.assertRaisesRegex(WorkflowCompileError, "어댑터 칸.*" + fallback.replace(".", r"\.")):
            ComfyWorkflowCompiler(real).compile(
                "txt2img", V1_MODEL, {**base, "alwayson_scripts": {anima38.SCRIPT_NAME: {"args": [unpicked]}}})

    def test_v1_needs_enabled_and_uses_the_card_adapter_not_the_module(self):
        """enabled → v1. 어댑터는 카드 값(settings.adapter) — Forge 는 모듈 목록의 어댑터를 보지 않는다."""
        alt_adapter = "text/Anima-alt_adapter.safetensors"
        capabilities = _anima_capabilities()
        capabilities["ForgeNeoAnimaQwen35Prompt"]["input"]["required"]["adapter_name"] = _choice(ADAPTER, alt_adapter)
        graph = ComfyWorkflowCompiler(capabilities).compile(
            "txt2img", V1_MODEL, {
                "forge_additional_modules": _modules(adapter=alt_adapter),
                "alwayson_scripts": _script(True, ADAPTER, 0.8, False, 1.0, False),
            },
        )
        prompts = [node for node in graph.values() if node.get("class_type") == "ForgeNeoAnimaQwen35Prompt"]
        self.assertEqual([node["inputs"]["adapter_name"] for node in prompts], [ADAPTER])   # 부정은 순정(negative 끔)
        self.assertEqual(prompts[0]["inputs"]["adapter_strength"], 0.8)

    def test_plain_anima_needs_no_qwen(self):
        native = ComfyWorkflowCompiler(_anima_capabilities(qwen=False)).compile(
            "txt2img", V1_MODEL, {
                "prompt": "native, <lora:ink:1>",
                "forge_additional_modules": _modules(qwen=""),
            },
        )
        self.assertIn("ForgeNeoAnimaLoraLoader", _classes(native))
        self.assertNotIn("ForgeNeoAnimaQwen35Loader", _classes(native))
        self.assertNotIn("ForgeNeoAnimaQwen35Prompt", _classes(native))
        self.assertEqual(_classes(native).count("CLIPTextEncode"), 2)

    def test_semantic_plan_without_qwen_choice_fails_before_graph_queue(self):
        with self.assertRaisesRegex(WorkflowCompileError, "Qwen3.5"):
            ComfyWorkflowCompiler(_anima_capabilities(qwen=False)).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(qwen=""),
                },
            )

    def test_semantic_plan_prefers_runtime_qwen_when_module_is_omitted(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {
                "forge_additional_modules": _modules(qwen=""),
            },
        )
        _qwen_id, qwen = _node(graph, "ForgeNeoAnimaQwen35Loader")
        self.assertEqual(qwen["inputs"], {"qwen35_model": QWEN35})

    def test_disabled_or_partial_semantic_modules_never_leak_into_native_clip(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V1_MODEL, {
                "forge_additional_modules": [VAE, NATIVE_CLIP, QWEN35],
            },
        )
        self.assertNotIn("ForgeNeoAnimaQwen35Loader", _classes(graph))
        self.assertNotIn("ForgeNeoAnimaQwen35Prompt", _classes(graph))
        self.assertNotIn("DualCLIPLoader", _classes(graph))
        self.assertNotIn("TripleCLIPLoader", _classes(graph))
        _clip_id, clip = _node(graph, "CLIPLoader")
        self.assertEqual(clip["inputs"]["clip_name"], NATIVE_CLIP)

    def test_unverified_v2_release_name_never_falls_back_to_unet_loader(self):
        capabilities = _anima_capabilities()
        capabilities["ForgeNeoAnima38V2Loader"]["input"]["required"][
            "model_name"
        ] = _choice("other-v2-bundle.safetensors")
        with self.assertRaisesRegex(WorkflowCompileError, "metadata.*loader"):
            ComfyWorkflowCompiler(capabilities).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(),
                },
            )

    def test_postprocess_uses_the_same_semantic_conditioning_plan(self):
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile_postprocess(
            V2_MODEL,
            {
                "prompt": "retouch, <lora:ink:0.4>",
                "negative_prompt": "artifact",
                "forge_additional_modules": _modules(),
                "alwayson_scripts": {
                    "ADetailer": {"args": [True, False, {
                        "ad_tab_enable": True, "ad_model": "face.pt",
                    }]},
                },
            },
            uploaded_image="source.png",
        )
        self.assertNotIn("ForgeNeoKSamplerCNS", _classes(graph))
        self.assertIn("ForgeNeoAnima38V2Loader", _classes(graph))
        self.assertIn("ForgeNeoAnimaLoraLoader", _classes(graph))
        prompt_id, _prompt = _node(graph, "ForgeNeoAnima38V2Prompt")
        negative_id, _negative = _node(graph, "CLIPTextEncode")
        lora_id, _lora = _node(graph, "ForgeNeoAnimaLoraLoader")
        _ad_id, adetailer = _node(graph, "ForgeNeoADetailer")
        self.assertEqual(adetailer["inputs"]["model"], [lora_id, 0])
        self.assertEqual(adetailer["inputs"]["clip"], [lora_id, 1])
        self.assertEqual(adetailer["inputs"]["positive"], [prompt_id, 0])
        self.assertEqual(adetailer["inputs"]["negative"], [negative_id, 0])

    def test_semantic_hires_allows_reuse_and_rejects_unsafe_overrides(self):
        base = {
            "forge_additional_modules": _modules(),
            "enable_hr": True,
            "hr_additional_modules": ["Use same choices"],
        }
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, base,
        )
        self.assertIn("ForgeNeoHiresFix", _classes(graph))
        for patch in (
            {"hr_prompt": "different"},
            {"hr_checkpoint_name": V1_MODEL},
            {"hr_additional_modules": [NATIVE_CLIP]},
        ):
            with self.subTest(patch=patch), self.assertRaisesRegex(
                WorkflowCompileError, "Hires.*override",
            ):
                ComfyWorkflowCompiler(_anima_capabilities()).compile(
                    "txt2img", V2_MODEL, {**base, **patch},
                )


class TestAnima38CustomCompilation(unittest.TestCase):
    def test_only_active_existing_lora_is_upgraded_and_rebased(self):
        workflow = _split_custom_workflow()
        workflow["11"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0], "clip": ["8", 0],
                "lora_name": "styles/ink.safetensors",
                "strength_model": 0.5, "strength_clip": 0.5,
            },
        }
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        workflow["20"] = copy.deepcopy(workflow["1"])
        workflow["21"] = copy.deepcopy(workflow["8"])
        workflow["22"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["20", 0], "clip": ["21", 0],
                "lora_name": "characters/alice.safetensors",
                "strength_model": 0.4, "strength_clip": 0.4,
            },
        }
        unrelated = copy.deepcopy(workflow["22"])
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {
                "forge_additional_modules": _modules(),
            }, workflow=workflow,
        )
        self.assertEqual(graph["11"]["class_type"], "ForgeNeoAnimaLoraLoader")
        self.assertEqual(graph["2"]["inputs"]["native_clip"], ["11", 1])
        self.assertEqual(graph["3"]["inputs"]["clip"], ["11", 1])
        self.assertEqual(graph["22"], unrelated)

    def test_shared_active_lora_and_model_only_lora_fail_explicitly(self):
        workflow = _split_custom_workflow()
        workflow["11"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0], "clip": ["8", 0],
                "lora_name": "styles/ink.safetensors",
                "strength_model": 0.5, "strength_clip": 0.5,
            },
        }
        workflow["5"]["inputs"]["model"] = ["11", 0]
        workflow["2"]["inputs"]["clip"] = ["11", 1]
        workflow["3"]["inputs"]["clip"] = ["11", 1]
        workflow["12"] = {
            "class_type": "SaveImage", "inputs": {
                "images": ["6", 0], "filename_prefix": "unrelated",
                "debug_model": ["11", 0],
            },
        }
        with self.assertRaisesRegex(WorkflowCompileError, "LoRA node.*공유"):
            ComfyWorkflowCompiler(_anima_capabilities()).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(),
                }, workflow=workflow,
            )

        workflow = _split_custom_workflow()
        workflow["11"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0], "lora_name": "styles/ink.safetensors",
                "strength_model": 0.5,
            },
        }
        workflow["5"]["inputs"]["model"] = ["11", 0]
        with self.assertRaisesRegex(WorkflowCompileError, "호환 remap"):
            ComfyWorkflowCompiler(_anima_capabilities()).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(),
                }, workflow=workflow,
            )

    def test_selected_split_branch_is_rewritten_in_place_only(self):
        workflow = _split_custom_workflow()
        workflow["10"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["8", 0], "text": "unrelated"},
        }
        original = copy.deepcopy(workflow)
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, {
                "prompt": "selected, <lora:ink:0.7>",
                "negative_prompt": "selected negative",
                "forge_additional_modules": _modules(),
            }, workflow=workflow,
        )
        self.assertEqual(workflow, original)
        self.assertEqual(graph["1"]["class_type"], "ForgeNeoAnima38V2Loader")
        self.assertEqual(graph["1"]["inputs"], {"model_name": V2_MODEL})
        self.assertEqual(graph["2"]["class_type"], "ForgeNeoAnima38V2Prompt")
        self.assertEqual(graph["3"]["class_type"], "CLIPTextEncode")
        self.assertEqual(graph["10"], original["10"])
        lora_id, _lora = _node(graph, "ForgeNeoAnimaLoraLoader")
        qwen_id, _qwen = _node(graph, "ForgeNeoAnimaQwen35Loader")
        self.assertEqual(graph["2"]["inputs"], {
            "model": [lora_id, 0],
            "native_clip": [lora_id, 1],
            "qwen35_clip": [qwen_id, 0],
            "prompt": "selected",
        })
        self.assertEqual(graph["3"]["inputs"], {
            "clip": [lora_id, 1], "text": "selected negative",
        })
        self.assertEqual(graph["5"]["inputs"]["model"], [lora_id, 0])
        self.assertEqual(graph["5"]["inputs"]["positive"], ["2", 0])
        self.assertEqual(graph["5"]["inputs"]["negative"], ["3", 0])

    def test_v1_custom_branch_uses_exact_prompt_contract_for_both_sides(self):
        workflow = _split_custom_workflow()
        graph = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V1_MODEL, {
                "prompt": "selected, <lora:ink:0.6>",
                "negative_prompt": "selected negative",
                "forge_additional_modules": _modules(adapter=ADAPTER),
                "alwayson_scripts": _script(
                    True, ADAPTER, 0.7, True, 0.3, False,
                ),
            }, workflow=workflow,
        )
        self.assertEqual(graph["1"]["class_type"], "UNETLoader")
        self.assertEqual(graph["1"]["inputs"]["unet_name"], V1_MODEL)
        lora_id, _lora = _node(graph, "ForgeNeoAnimaLoraLoader")
        qwen_id, _qwen = _node(graph, "ForgeNeoAnimaQwen35Loader")
        expected_common = {
            "model": [lora_id, 0],
            "native_clip": [lora_id, 1],
            "qwen35_clip": [qwen_id, 0],
            "adapter_name": ADAPTER,
        }
        self.assertEqual(graph["2"]["class_type"], "ForgeNeoAnimaQwen35Prompt")
        self.assertEqual(graph["2"]["inputs"], {
            **expected_common, "prompt": "selected", "adapter_strength": 0.7,
        })
        self.assertEqual(graph["3"]["class_type"], "ForgeNeoAnimaQwen35Prompt")
        self.assertEqual(graph["3"]["inputs"], {
            **expected_common,
            "prompt": "selected negative",
            "adapter_strength": 0.3,
        })
        self.assertEqual(graph["5"]["inputs"]["model"], [lora_id, 0])

    def test_shared_checkpoint_outputs_make_v2_rewrite_ambiguous(self):
        workflow = _custom_workflow()
        workflow["8"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": "unrelated"},
        }
        with self.assertRaisesRegex(WorkflowCompileError, "CLIP/VAE 출력을 계속"):
            ComfyWorkflowCompiler(_anima_capabilities()).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(),
                }, workflow=workflow,
            )

    def test_empty_selection_keeps_the_v2_loader_workflow_on_semantic_connector(self):
        # 모델 콤보가 비면(ComfyUI 는 허용) 워크플로 자신의 v2 번들이 모델이다 — 예전엔
        # 판별을 건너뛰어 Semantic Connector v2 없이 native 조건부로 컴파일됐다.
        workflow = _split_custom_workflow()
        workflow["1"] = {"class_type": "ForgeNeoAnima38V2Loader", "inputs": {"model_name": V2_MODEL}}
        payload = {"prompt": "selected, <lora:ink:0.7>", "forge_additional_modules": _modules()}
        empty = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", "", copy.deepcopy(payload), workflow=copy.deepcopy(workflow),
        )
        self.assertIn("ForgeNeoAnima38V2Prompt", _classes(empty))
        self.assertIn("ForgeNeoAnimaQwen35Loader", _classes(empty))
        self.assertEqual(empty["1"]["class_type"], "ForgeNeoAnima38V2Loader")
        self.assertEqual(empty["1"]["inputs"], {"model_name": V2_MODEL})
        self.assertIn("ForgeNeoAnimaLoraLoader", _classes(empty))
        # 워크플로 자신의 모델을 고른 것과 같은 그래프.
        selected = ComfyWorkflowCompiler(_anima_capabilities()).compile(
            "txt2img", V2_MODEL, copy.deepcopy(payload), workflow=copy.deepcopy(workflow),
        )
        for graph in (empty, selected):
            for node in graph.values():
                node["inputs"].pop("seed", None)
                node["inputs"].pop("noise_seed", None)
        self.assertEqual(empty, selected)

    def test_stale_v2_prompt_schema_is_rejected_before_queue(self):
        capabilities = _anima_capabilities()
        del capabilities["ForgeNeoAnima38V2Prompt"]["input"]["required"]["prompt"]
        with self.assertRaisesRegex(WorkflowCompileError, "노드 계약"):
            ComfyWorkflowCompiler(capabilities).compile(
                "txt2img", V2_MODEL, {
                    "forge_additional_modules": _modules(),
                },
            )


if __name__ == "__main__":
    unittest.main()
