"""VAE DeGrid in the ComfyUI compiler (L3): one ``ForgeNeoAnimaVAEDeGrid`` on the final image.

Forge runs the extension once per finished image after every other
post-process (``postprocess_image_after_composite``); the compiler puts the
pack node at the same place — after the last ADetailer/SAM3 pass, right before
the save/preview node — on main generations only.  A missing node (pack before
1.5.0), a contract mismatch, a model the node does not list, or a custom
workflow without a place for it leaves DeGrid out with a warning instead of
failing the job (D8).  No ComfyUI is contacted; the node schema is the real
pack node's ``INPUT_TYPES`` with its model list patched.
"""
from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import degrid_nodes
from core import comfy_node_pack, sam3_args, sam_extra_notices as sn, vae_degrid as vdg
from core.comfy_compatibility import check_recipes
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import feature_preflight
from tests.test_comfy_workflow_compiler import _capabilities, _classes, _custom_workflow, _node

ROOT = Path(__file__).resolve().parent.parent
NODE = vdg.COMFY_NODE_CLASS
MODEL_FILES = ("qwenVAEDegridNafnet_v11.safetensors", "NAFNet-QwenVAE-DeGrid.safetensors")


def node_schema(*choices: str) -> dict:
    """The real pack node's ``INPUT_TYPES`` with ``model_choices`` patched (no ComfyUI, no files)."""
    with mock.patch.object(degrid_nodes, "model_choices", return_value=[degrid_nodes.AUTO, *choices]):
        return {"input": degrid_nodes.ForgeNeoAnimaVAEDeGrid.INPUT_TYPES(), "output": ["IMAGE", "STRING"]}


def caps_with_degrid(*choices: str) -> dict:
    caps = _capabilities()
    caps[NODE] = node_schema(*(choices or MODEL_FILES))
    return caps


def degrid_block(**values) -> dict:
    settings = vdg.DegridSettings(enabled=True, **values)
    return vdg.as_block(settings)


def payload(*, degrid=None, sam3=False, adetailer=False, passes=(), save=True, **extra) -> dict:
    scripts: dict = {}
    if adetailer:
        scripts["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face_yolov8n.pt"}]}
    if sam3:
        scripts.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face"}))
    if degrid is not None:
        scripts[vdg.SCRIPT_NAME] = degrid
    body = {"prompt": "portrait", "negative_prompt": "bad", "seed": 7, "save_images": save,
            "alwayson_scripts": scripts, **extra}
    if passes:
        body["_comfy_detail_passes"] = list(passes)
    return body


def compile_t2i(caps, body, *, workflow=None, model="checkpoint.safetensors"):
    warnings: list = []
    graph = ComfyWorkflowCompiler(caps).compile("txt2img", model, body, workflow=workflow, warnings=warnings)
    return graph, warnings


def degrid_nodes_of(graph: dict) -> list:
    return [(node_id, node) for node_id, node in graph.items() if node.get("class_type") == NODE]


def output_input(graph: dict) -> list:
    for class_type in ("SaveImage", "PreviewImage"):
        found = [node for node in graph.values() if node.get("class_type") == class_type]
        if found:
            return found[0]["inputs"]["images"]
    raise AssertionError("no output node")


class DefaultGraphTests(unittest.TestCase):
    def test_node_sits_after_the_last_sam3_pass_and_right_before_save(self):
        graph, warnings = compile_t2i(caps_with_degrid(), payload(
            degrid=degrid_block(mode="dark", strength=0.8, tile=300), sam3=True, adetailer=True, passes=("eyes",)))
        self.assertEqual(warnings, [])
        found = degrid_nodes_of(graph)
        self.assertEqual(len(found), 1)                                         # once, not per SAM3 pass
        degrid_id, node = found[0]
        detailers = [node_id for node_id, n in graph.items() if n["class_type"] == "ForgeNeoSAM3Detailer"]
        self.assertEqual(len(detailers), 2)
        self.assertEqual(node["inputs"]["image"], [max(detailers, key=int), 0])  # after the last pass
        self.assertEqual(output_input(graph), [degrid_id, 0])
        self.assertEqual(node["_meta"]["title"], "VAE DeGrid (final image)")
        self.assertNotIn(vdg.SCRIPT_NAME.lower(), node["_meta"]["title"].lower())

    def test_input_values_follow_the_extension_rules_and_comfy_constants(self):
        graph, _ = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block(mode="bright", strength=0.85, tile=300)))
        inputs = degrid_nodes_of(graph)[0][1]["inputs"]
        self.assertEqual(set(inputs), set(vdg.COMFY_INPUTS))
        self.assertEqual(inputs["model_name"], vdg.COMFY_AUTO)
        self.assertEqual(inputs["mode"], "Bright Pixels Mainly")
        self.assertEqual((inputs["strength"], inputs["tile"]), (0.85, 300))
        self.assertIs(inputs["enabled"], True)
        self.assertEqual((inputs["device"], inputs["precision"], inputs["keep_loaded"]),
                         ("auto", "fp32", False))                                # D6 — never Forge's options
        self.assertEqual(dict(vdg.COMFY_OPTIONS), {"device": "auto", "precision": "fp32", "keep_loaded": False})
        self.assertIs(inputs["forge_quantize"], True)                             # D9

    def test_values_are_coerced_like_the_extension_reads_the_block(self):
        cases = (
            ({"args": [True, "", "Dark Pixels Mainly (어두운 점 위주)", 5, 50]}, ("Dark Pixels Mainly", 1.5, 128)),
            ({"args": [{"enabled": True, "mode": "weird", "strength": "nan", "tile": -3}]}, ("Full", 1.0, 0)),
            ({"args": [{"enabled": True, "mode": "BRIGHT", "strength": -1, "tile": 99999}]}, ("Bright Pixels Mainly", 0.0, 4096)),
        )
        for block, expected in cases:
            with self.subTest(block=block):
                graph, _ = compile_t2i(caps_with_degrid(), payload(degrid=block))
                inputs = degrid_nodes_of(graph)[0][1]["inputs"]
                self.assertEqual((inputs["mode"], inputs["strength"], inputs["tile"]), expected)

    def test_off_or_absent_block_builds_the_graph_byte_identical(self):
        plain, _ = compile_t2i(caps_with_degrid(), payload(sam3=True))
        for block in ({"args": [{"enabled": False}]}, {"args": [False, "", "full", 1, 512]}, {"args": []}):
            with self.subTest(block=block):
                graph, warnings = compile_t2i(caps_with_degrid(), payload(sam3=True, degrid=block))
                self.assertEqual(graph, plain)
                self.assertEqual(warnings, [])

    def test_title_matches_case_insensitively_like_forge(self):
        body = payload()
        body["alwayson_scripts"]["  anima vae degrid (NAFNET) ".strip()] = degrid_block()
        graph, _ = compile_t2i(caps_with_degrid(), body)
        self.assertEqual(len(degrid_nodes_of(graph)), 1)

    def test_preview_output_and_img2img_inpaint_also_end_with_degrid(self):
        graph, _ = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block(), save=False))
        self.assertEqual(output_input(graph), [degrid_nodes_of(graph)[0][0], 0])
        for mode, extra in (("img2img", {}), ("inpaint", {"uploaded_mask": "mask.png"})):
            with self.subTest(mode=mode):
                graph = ComfyWorkflowCompiler(caps_with_degrid()).compile(
                    mode, "checkpoint.safetensors", payload(degrid=degrid_block(), denoising_strength=0.5),
                    uploaded_image="input.png", **extra)
                self.assertEqual(output_input(graph), [degrid_nodes_of(graph)[0][0], 0])

    def test_offline_compile_adds_the_requested_name_unchecked(self):
        compiler = ComfyWorkflowCompiler(None)
        for model, expected in (("", "auto"), ("qwenVAEDegridNafnet_v11", "qwenVAEDegridNafnet_v11")):
            with self.subTest(model=model):
                graph = compiler.compile("txt2img", "checkpoint.safetensors", payload(degrid=degrid_block(model=model)))
                self.assertEqual(degrid_nodes_of(graph)[0][1]["inputs"]["model_name"], expected)

    def test_class_only_capability_probe_is_not_treated_as_an_old_pack(self):
        caps = _capabilities()
        caps[NODE] = {"input": {"required": {}}}
        graph, warnings = compile_t2i(caps, payload(degrid=degrid_block(model="x")))
        self.assertEqual(warnings, [])
        self.assertEqual(degrid_nodes_of(graph)[0][1]["inputs"]["model_name"], "x")


class ModelResolutionTests(unittest.TestCase):
    CHOICES = ("upscale_models/anzhc.safetensors", "degrid/anzhc.safetensors",
               "qwenVAEDegridNafnet_v11.safetensors", "Other-Name.pth")

    def resolved(self, model: str):
        graph, warnings = compile_t2i(caps_with_degrid(*self.CHOICES), payload(degrid=degrid_block(model=model)))
        found = degrid_nodes_of(graph)
        return (found[0][1]["inputs"]["model_name"] if found else None), warnings

    def test_forge_style_names_map_to_the_node_choice(self):
        for stem, expected in (
            ("", "auto"),
            ("qwenVAEDegridNafnet_v11", "qwenVAEDegridNafnet_v11.safetensors"),
            ("QWENVAEDEGRIDNAFNET_V11", "qwenVAEDegridNafnet_v11.safetensors"),
            ("ESRGAN/anzhc", "upscale_models/anzhc.safetensors"),               # Forge clash prefix
            ("DeGrid/anzhc", "degrid/anzhc.safetensors"),
            ("other-name", "Other-Name.pth"),
            ("qwenVAEDegridNafnet_v11.safetensors", "qwenVAEDegridNafnet_v11.safetensors"),
        ):
            with self.subTest(stem=stem):
                self.assertEqual(self.resolved(stem), (expected, []))

    def test_missing_model_is_left_out_with_forge_wording_and_the_graph_is_otherwise_identical(self):
        plain, _ = compile_t2i(caps_with_degrid(*self.CHOICES), payload(sam3=True))
        graph, warnings = compile_t2i(caps_with_degrid(*self.CHOICES),
                                      payload(sam3=True, degrid=degrid_block(model="4x-UltraSharp")))
        self.assertEqual(graph, plain)
        self.assertEqual(warnings, [{"code": sn.CODE_DEGRID_ERROR, "reason": "model not found: 4x-UltraSharp",
                                     "feature": "degrid", "cause": vdg.COMFY_OMIT_MODEL_MISSING}])

    def test_auto_without_any_model_file_is_model_not_found_auto(self):
        for listed in ((), ("None",)):
            with self.subTest(listed=listed):
                graph, warnings = compile_t2i(caps_with_degrid(*listed) if listed else _with_empty_list(),
                                              payload(degrid=degrid_block()))
                self.assertEqual(degrid_nodes_of(graph), [])
                self.assertEqual([w["reason"] for w in warnings], ["model not found: auto"])
                self.assertEqual(warnings[0]["code"], sn.CODE_DEGRID_ERROR)
                self.assertEqual(warnings[0]["cause"], vdg.COMFY_OMIT_MODEL_MISSING)

    def test_without_a_warnings_list_the_same_graph_is_built(self):
        caps = caps_with_degrid()
        body = payload(degrid=degrid_block(model="missing"))
        graph = ComfyWorkflowCompiler(caps).compile("txt2img", "checkpoint.safetensors", body)
        self.assertEqual(degrid_nodes_of(graph), [])


def _with_empty_list() -> dict:
    caps = _capabilities()
    caps[NODE] = node_schema()
    return caps


class OldPackTests(unittest.TestCase):
    def test_missing_class_leaves_degrid_out_and_validate_passes(self):
        plain, _ = compile_t2i(_capabilities(), payload(adetailer=True))
        graph, warnings = compile_t2i(_capabilities(), payload(adetailer=True, degrid=degrid_block()))
        self.assertEqual(graph, plain)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["code"], sn.CODE_DEGRID_COMFY_UNAVAILABLE)
        self.assertIn(NODE, warnings[0]["reason"])
        self.assertIn(vdg.COMFY_MIN_PACK_VERSION, warnings[0]["reason"])
        self.assertEqual(warnings[0]["cause"], vdg.COMFY_OMIT_NODE_MISSING)          # 새 스키마로 풀릴 수 있다

    def test_input_contract_mismatch_is_an_old_pack_too(self):
        caps = caps_with_degrid()
        caps[NODE]["input"]["required"].pop("forge_quantize")
        graph, warnings = compile_t2i(caps, payload(degrid=degrid_block()))
        self.assertEqual(degrid_nodes_of(graph), [])
        self.assertEqual(warnings[0]["code"], sn.CODE_DEGRID_COMFY_UNAVAILABLE)
        self.assertIn("입력", warnings[0]["reason"])
        self.assertEqual(warnings[0]["cause"], vdg.COMFY_OMIT_NODE_CONTRACT)

    def test_pack_version_carries_the_node(self):
        def parts(text):
            return tuple(int(x) for x in text.split("."))
        self.assertGreaterEqual(parts(comfy_node_pack.PACK_VERSION), parts(vdg.COMFY_MIN_PACK_VERSION))
        card = (ROOT / "frontend/src/utils/vaeDegrid.ts").read_text(encoding="utf-8")
        self.assertIn(f"앱 노드 팩 {vdg.COMFY_MIN_PACK_VERSION}", card)


class MainRequestsOnlyTests(unittest.TestCase):
    """D1: post-processing, upscale and mask-only graphs never build DeGrid, even when the payload carries it."""

    def test_postprocess_upscale_and_mask_only_graphs_have_no_degrid(self):
        compiler = ComfyWorkflowCompiler(caps_with_degrid())
        body = payload(sam3=True, adetailer=True, degrid=degrid_block())
        for kind in ("ForgeNeoSAM3Detailer", "ForgeNeoSAM3Refine"):
            with self.subTest(kind=kind):
                graph = compiler.compile_postprocess("checkpoint.safetensors", body, uploaded_image="in.png",
                                                     sam3_detailer_class=kind)
                self.assertNotIn(NODE, _classes(graph))
        graph = compiler.compile_sam3_mask_only(body, uploaded_image="in.png")
        self.assertNotIn(NODE, _classes(graph))
        graph = compiler.compile_upscale("in.png", {"upscaler_name": "Lanczos", "scale_factor": 2})
        self.assertNotIn(NODE, _classes(graph))

    def test_degrid_is_not_an_image_script(self):
        """D11/C6: DeGrid needs no VAE link and is no post-processing request on its own."""
        compiler = ComfyWorkflowCompiler(caps_with_degrid())
        body = payload(degrid=degrid_block())
        self.assertFalse(compiler._has_image_scripts(body))
        with self.assertRaisesRegex(WorkflowCompileError, "ADetailer 또는 SAM3 후처리 설정이 없습니다"):
            compiler.compile_postprocess("checkpoint.safetensors", body, uploaded_image="in.png")


class CustomWorkflowTests(unittest.TestCase):
    def test_outputs_are_rewired_to_degrid_after_the_extensions_tail(self):
        workflow = _custom_workflow()
        graph, warnings = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block(), adetailer=True),
                                      workflow=workflow)
        self.assertEqual(warnings, [])
        degrid_id, node = degrid_nodes_of(graph)[0]
        adetailer_id, _ = _node(graph, "ForgeNeoADetailer")
        self.assertEqual(node["inputs"]["image"], [adetailer_id, 0])
        self.assertEqual(graph["7"]["inputs"]["images"], [degrid_id, 0])

    def test_degrid_only_custom_workflow_takes_the_decode_output(self):
        graph, warnings = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block()), workflow=_custom_workflow())
        self.assertEqual(warnings, [])
        degrid_id, node = degrid_nodes_of(graph)[0]
        self.assertEqual(node["inputs"]["image"], ["6", 0])
        self.assertEqual(graph["7"]["inputs"]["images"], [degrid_id, 0])

    def test_degrid_only_does_not_need_a_vae_link(self):
        workflow = _custom_workflow()
        del workflow["6"], workflow["7"]                       # no decode: latent-only workflow
        workflow["1"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "anima.safetensors",
                                                               "weight_dtype": "default"}}
        workflow["2"]["inputs"]["clip"] = workflow["3"]["inputs"]["clip"] = ["9", 0]
        workflow["9"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "text/base.safetensors"}}
        graph, warnings = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block()), workflow=workflow,
                                      model="anima.safetensors")
        self.assertNotIn(NODE, _classes(graph))
        self.assertEqual([w["code"] for w in warnings], [sn.CODE_DEGRID_COMFY_UNAVAILABLE])
        self.assertIn("VAEDecode", warnings[0]["reason"])
        self.assertEqual(warnings[0]["cause"], vdg.COMFY_OMIT_PLACEMENT)              # 새 스키마로도 같다

    def test_no_output_or_ambiguous_outputs_skip_degrid_with_a_warning(self):
        no_output = _custom_workflow()
        del no_output["7"]
        ambiguous = _custom_workflow()
        ambiguous["8"] = {"class_type": "ImageInvert", "inputs": {"image": ["6", 0]}}
        ambiguous["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}}
        caps = caps_with_degrid()
        caps["ImageInvert"] = {"input": {"required": {"image": ["IMAGE"]}}}
        for name, workflow in (("no output", no_output), ("ambiguous", ambiguous)):
            with self.subTest(name):
                plain, _ = compile_t2i(caps, payload(), workflow=copy.deepcopy(workflow))
                graph, warnings = compile_t2i(caps, payload(degrid=degrid_block()), workflow=copy.deepcopy(workflow))
                self.assertEqual(graph, plain)
                self.assertEqual([w["code"] for w in warnings], [sn.CODE_DEGRID_COMFY_UNAVAILABLE])
                self.assertEqual([w["cause"] for w in warnings], [vdg.COMFY_OMIT_PLACEMENT])

    def test_with_adetailer_the_existing_errors_stand_and_name_degrid(self):
        ambiguous = _custom_workflow()
        ambiguous["8"] = {"class_type": "ImageInvert", "inputs": {"image": ["6", 0]}}
        ambiguous["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "x"}}
        caps = caps_with_degrid()
        caps["ImageInvert"] = {"input": {"required": {"image": ["IMAGE"]}}}
        with self.assertRaisesRegex(WorkflowCompileError, "ADetailer/SAM3/VAE DeGrid"):
            compile_t2i(caps, payload(adetailer=True, degrid=degrid_block()), workflow=ambiguous)
        with self.assertRaisesRegex(WorkflowCompileError, "ADetailer/SAM3 자동"):
            compile_t2i(caps, payload(adetailer=True), workflow=ambiguous)

    def test_missing_model_in_a_custom_workflow_keeps_the_original_wiring(self):
        graph, warnings = compile_t2i(caps_with_degrid(), payload(degrid=degrid_block(model="gone")),
                                      workflow=_custom_workflow())
        self.assertEqual(graph["7"]["inputs"]["images"], ["6", 0])
        self.assertEqual([w["reason"] for w in warnings], ["model not found: gone"])


class PreflightTests(unittest.TestCase):
    def test_ready_row(self):
        result = feature_preflight(ComfyWorkflowCompiler(caps_with_degrid()), "checkpoint.safetensors",
                                   payload(degrid=degrid_block()))
        row = next(r for r in result["features"] if r["id"] == "degrid")
        self.assertTrue(result["ok"], result)
        self.assertEqual((row["label"], row["state"]), ("VAE DeGrid", "ready"))

    def test_left_out_degrid_is_skipped_with_its_reason_and_keeps_ok(self):
        result = feature_preflight(ComfyWorkflowCompiler(_capabilities()), "checkpoint.safetensors",
                                   payload(degrid=degrid_block(), enable_hr=True))
        row = next(r for r in result["features"] if r["id"] == "degrid")
        self.assertTrue(result["ok"], result)
        self.assertEqual(row["state"], "skipped")
        self.assertIn(NODE, row["reason"])
        self.assertEqual(next(r for r in result["features"] if r["id"] == "hires")["state"], "ready")

    def test_off_row(self):
        result = feature_preflight(ComfyWorkflowCompiler(caps_with_degrid()), "checkpoint.safetensors", payload())
        self.assertEqual(next(r for r in result["features"] if r["id"] == "degrid")["state"], "off")

    def test_vue_lists_the_skipped_state_and_reason(self):
        source = (ROOT / "frontend/src/components/ComfyWorkflowControls.vue").read_text(encoding="utf-8")
        self.assertIn("skipped: '건너뜀", source)
        self.assertIn("feature.reason", source)
        self.assertIn("reason?: string", source)


class CompatibilityRecipeTests(unittest.TestCase):
    def recipe(self, schema):
        return next(r for r in check_recipes(schema) if r["id"] == "degrid")

    def test_recipe_uses_the_app_constants(self):
        from core.comfy_compatibility import MODEL_FIELDS, RECIPES
        recipe = next(r for r in RECIPES if r["id"] == "degrid")
        self.assertIs(recipe["title"], vdg.SCRIPT_NAME)
        self.assertEqual(set(recipe["nodes"][NODE]) | {"enabled"}, set(vdg.COMFY_INPUTS))
        self.assertEqual(MODEL_FIELDS["degrid"], [(NODE, "model_name")])

    def test_available_missing_model_and_old_pack(self):
        self.assertEqual(self.recipe(caps_with_degrid())["status"], "available")
        only_auto = self.recipe(_with_empty_list())                              # 'auto' is no file
        self.assertEqual(only_auto["status"], "missing")
        self.assertIn("missing", [c["status"] for c in only_auto["checks"] if c["label"] == "VAE DeGrid 모델"])
        self.assertEqual(self.recipe(_capabilities())["status"], "missing")      # node absent
        self.assertEqual(self.recipe(None)["status"], "unknown")


class PackNodeContractTests(unittest.TestCase):
    """L1-P2's node and L1-P1's constants describe one contract (the compiler relies on both)."""

    def test_node_inputs_and_choices_equal_the_app_constants(self):
        required = node_schema(*MODEL_FILES)["input"]["required"]
        self.assertEqual(tuple(required), vdg.COMFY_INPUTS)
        self.assertEqual(list(required["mode"][0]), list(vdg.MODE_LABELS.values()))
        self.assertEqual(tuple(required["device"][0]), vdg.DEVICE_CHOICES)
        self.assertEqual(tuple(required["precision"][0]), vdg.PRECISION_CHOICES)
        self.assertEqual(required["model_name"][0][0], vdg.COMFY_AUTO)
        self.assertEqual(degrid_nodes.UI_KEY, vdg.COMFY_UI_KEY)
        self.assertEqual(vdg.comfy_model_choices(caps_with_degrid()),
                         ["qwenVAEDegridNafnet_v11", "NAFNet-QwenVAE-DeGrid"])


class TitleSourceTests(unittest.TestCase):
    """C15: the L3 modules read the title from ``vae_degrid.SCRIPT_NAME`` — no copy of the literal."""

    MODULES = ("core/comfy_workflow_compiler.py", "core/comfy_workflow_controls.py", "core/comfy_degrid_report.py",
               "core/comfy_metadata.py", "core/comfy_compatibility.py", "backends/comfyui_backend.py")

    def test_no_title_literal_and_the_compiler_uses_the_constant(self):
        for rel in self.MODULES:
            with self.subTest(module=rel):
                source = (ROOT / rel).read_text(encoding="utf-8")
                tree = ast.parse(source)
                literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                            and isinstance(node.value, str) and vdg.SCRIPT_NAME.lower() in node.value.lower()]
                self.assertEqual(literals, [])
                self.assertNotIn(f'"{NODE}"', source)                             # node id from the constant too
        compiler = (ROOT / "core/comfy_workflow_compiler.py").read_text(encoding="utf-8")
        self.assertIn("vae_degrid.SCRIPT_NAME", compiler)
        self.assertIn("vae_degrid.COMFY_NODE_CLASS", compiler)


if __name__ == "__main__":
    unittest.main()
