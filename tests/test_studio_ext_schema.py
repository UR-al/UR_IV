"""The studio's extension table (``web/ai_studio/ext_schema.json``) against the app's specs, and its JS mirror.

The pack cannot import ``core``, so ``tools/build_studio_ext_schema.py`` writes the table from the app's specs; these
tests (venv: ``core`` importable) pin the table to those specs. ``ext_compiler`` cases of the golden fixture hold what
the app compiler builds for a panel state: ``compiled_roles`` rebuilds them with the real compiler methods here, and
``js_tests/studio_ext.test.mjs`` asserts ``studio_ext.js`` gives the same inputs and modes.
"""
from __future__ import annotations

import importlib.util
import json
import random
import re
import tempfile
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS
from comfy_custom_nodes.ai_studio_forge_parity.studio_state import normalize_state
from comfy_custom_nodes.ai_studio_forge_parity.xyz_plan import plan_xyz
from core import anima_guidance as ag
from core import anima_speed, extra_samplers, vae_degrid
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, _cns_inputs, _Graph

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity"
SCHEMA = PACK / "web" / "ai_studio" / "ext_schema.json"
SCHEMA_JS = PACK / "web" / "ai_studio" / "ext_schema.js"
GOLDEN = PACK / "js_tests" / "fixtures" / "studio_logic_golden.json"
BUILDER = ROOT / "tools" / "build_studio_ext_schema.py"

EXPECTED_ORDER = [
    ("guidance", "ForgeNeoAnimaGuidanceSuite", "가이던스"),
    ("optimal_scale", "ForgeNeoAnimaOptimalScale", "가이던스"),
    ("skim", "ForgeNeoSkimmedCFG", "가이던스"),
    ("detail_daemon", "ForgeNeoAnimaDetailDaemon", "디테일"),
    ("lora_block", "ForgeNeoLoraBlockWeight", "디테일"),
    ("speed", "ForgeNeoAnimaSpeed", "샘플러·VAE"),
    ("er_sde", "ForgeNeoExtraSamplerSettings", "샘플러·VAE"),
    ("vae_2x", "ForgeNeoAnimaVAE2x", "샘플러·VAE"),
    ("degrid", "ForgeNeoAnimaVAEDeGrid", "샘플러·VAE"),
    ("pid", "ForgeNeoAnimaPiD", "샘플러·VAE"),
    ("sam3_refine", "ForgeNeoSAM3Refine", "SAM3"),
    ("sam3_tile", "ForgeNeoSAM3TileRepair", "SAM3"),
]
LEFT_OUT = {"ForgeNeoADetailer", "ForgeNeoCharacterReference", "ForgeNeoReferencePrompt", "ForgeNeoReferenceOutput",
            "ForgeNeoAnimaSafePAG", "ForgeNeoAnimaDAVE", "ForgeNeoAnimaModGuidance", "ForgeNeoDCWCWMSMC",
            "ForgeNeoColorcraft"}
MAX_SAFE_INT = 2 ** 53 - 1  # the JS side holds the table's numbers as doubles


def load_schema() -> dict:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))


def _extension(key: str) -> dict:
    return next(item for item in load_schema()["extensions"] if item["key"] == key)


def _fields(key: str) -> dict:
    return {field["key"]: field for field in _extension(key)["fields"]}


# ------------------------------------------------------------------------------------------- compiler oracle

SUITE_ROLES = {"ForgeNeoAnimaOptimalScale": "ext_optimal_scale", "ForgeNeoAnimaGuidanceSuite": "ext_guidance",
               "ForgeNeoSkimmedCFG": "ext_skim"}
MODEL, CLIP, POS, NEG = ["model", 0], ["clip", 0], ["positive", 0], ["negative", 0]


def _values(node: dict) -> dict:
    """A node's literal inputs, ``settings_json`` parsed."""
    inputs = {name: value for name, value in node["inputs"].items() if not isinstance(value, list)}
    if "settings_json" in inputs:
        inputs["settings_json"] = json.loads(inputs["settings_json"])
    return inputs


def _only_node(build) -> dict | None:
    graph = _Graph()
    build(graph)
    nodes = list(graph.nodes.values())
    if len(nodes) > 1:
        raise AssertionError(f"expected one node, got {nodes}")
    return _values(nodes[0]) if nodes else None


def _script(ext: dict, on_key: str) -> dict:
    return {**ext, on_key: ext["on"]}


def payload_of(state: dict) -> tuple[dict, str]:
    """The app's request for a normalized panel state: (payload, compiler mode)."""
    ext = state["ext"]
    mode = "txt2img" if state["tab"] == "t2i" else "inpaint" if state["i2i_mode"] == "inpaint" else "img2img"
    scripts = {}
    if ext["guidance"]["on"]:
        scripts[ag.SCRIPT_PERTURBATION] = {"args": ag.build_args(ag.SCRIPT_PERTURBATION, ext["guidance"])}
    for key, title, on_key in (("optimal_scale", ag.SCRIPT_OPTIMAL_SCALE, "ocfg_enabled"),
                               ("skim", ag.SCRIPT_SKIMMED_CFG, "skim_enabled"),
                               ("detail_daemon", ag.SCRIPT_DETAIL_DAEMON, "dd_enabled")):
        scripts[title] = {"args": ag.build_args(title, _script(ext[key], on_key))}
    if ext["er_sde"]["on"]:
        values = [ext["er_sde"][f"{extra_samplers.SPEC_PREFIX}{name}"] for name in extra_samplers.ARG_NAMES]
        scripts[extra_samplers.SCRIPT_NAME] = {"args": values}
    speed = anima_speed.parse_settings({**ext["speed"], "enabled": ext["speed"]["on"]})
    plan = anima_speed.plan(speed, target="t2i" if mode == "txt2img" else "i2i", backend=anima_speed.BACKEND_COMFY)
    if plan.block is not None:
        scripts[anima_speed.SCRIPT_NAME] = plan.block
    return {"alwayson_scripts": scripts, "cfg_scale": state["cfg"], "seed": state["seed"],
            "sampler_name": state["sampler"], "enable_hr": state["hires"]["on"]}, mode


def compiled_roles(raw: dict) -> dict:
    """role -> the inputs of what the compiler builds for this state (a role it leaves out is absent), plus ``cns``."""
    state = normalize_state(raw)
    compiler = ComfyWorkflowCompiler.__new__(ComfyWorkflowCompiler)
    payload, mode = payload_of(state)
    t2i_hires = mode == "txt2img" and state["hires"]["on"]
    roles = {}

    if state["ext"]["er_sde"]["on"]:
        node = _only_node(lambda graph: compiler._add_extra_samplers(graph, MODEL, payload))
        if node is not None:
            roles["ext_er_sde"] = node
    graph = _Graph()
    _, sampler_options = compiler._add_anima_guidance(graph, MODEL, CLIP, POS, NEG, payload, hiflow_pass=t2i_hires)
    for node in graph.nodes.values():
        roles[SUITE_ROLES[node["class_type"]]] = _values(node)
    if "ext_guidance" in roles:
        baseline = {key: field["default"] for key, field in _fields("guidance").items()}
        settings = roles["ext_guidance"].pop("settings_json")
        roles["ext_guidance"]["settings_delta"] = {
            key: value for key, value in settings.items() if key not in baseline or baseline[key] != value}
    roles["cns"] = _cns_inputs(sampler_options)

    speed = compiler._speed_plan(payload, mode, [])
    pass_label = compiler._speed_pass_label(mode)
    for role, build in (
            ("ext_dd", lambda graph: compiler._add_detail_daemon(graph, MODEL, payload)),
            ("ext_speed", lambda graph: compiler._add_speed(graph, MODEL, speed, payload, pass_label=pass_label))):
        node = _only_node(build)
        if node is not None:
            roles[role] = node
    if state["hires"]["on"] and mode == "txt2img":
        for role, build in (
                ("ext_dd_hires", lambda graph: compiler._add_detail_daemon(graph, MODEL, payload, hires_pass=True)),
                ("ext_speed_hires", lambda graph: compiler._add_speed(
                    graph, MODEL, speed, payload, pass_label=anima_speed.PASS_HIRES))):
            node = _only_node(build)
            if node is not None:
                roles[role] = node
    if state["sam3"]["on"]:
        cfg = state["sam3"]["cfg"] if state["sam3"]["cfg"] > 0 else state["cfg"]
        node = _only_node(lambda graph: compiler._add_detail_daemon(
            graph, MODEL, payload, cfg_scale=cfg, sampler_name=state["sampler"], title="sam3"))
        if node is not None:
            roles["ext_dd_sam3"] = node
    return roles


# ------------------------------------------------------------------------------------------------------ tests


class SchemaShapeTests(unittest.TestCase):
    def test_extensions_in_display_order(self):
        found = [(item["key"], item["class"], item["group"]) for item in load_schema()["extensions"]]
        self.assertEqual(found, EXPECTED_ORDER)

    def test_adetailer_reference_and_superseded_nodes_left_out(self):
        self.assertFalse({item["class"] for item in load_schema()["extensions"]} & LEFT_OUT)

    def test_every_extension_has_a_role_and_a_form(self):
        roles = [item["role"] for item in load_schema()["extensions"]]
        self.assertEqual(len(roles), len(set(roles)))
        for item in load_schema()["extensions"]:
            with self.subTest(key=item["key"]):
                self.assertTrue(item["role"].startswith("ext_"))
                self.assertIn(item["form"], ("json", "widgets"))
                self.assertTrue(item["label"])
                self.assertTrue(item["fields"])
                self.assertNotIn("on", {field["key"] for field in item["fields"]})

    def test_fields_are_well_formed(self):
        for item in load_schema()["extensions"]:
            keys = [field["key"] for field in item["fields"]]
            self.assertEqual(len(keys), len(set(keys)), item["key"])
            for field in item["fields"]:
                with self.subTest(key=field["key"]):
                    self.assertTrue(field["label"])
                    kind, default = field["kind"], field["default"]
                    self.assertIn(kind, ("bool", "int", "float", "choice", "text"))
                    if kind == "bool":
                        self.assertIsInstance(default, bool)
                    elif kind in ("int", "float"):
                        self.assertIsInstance(default, int if kind == "int" else float)
                        self.assertLessEqual(field["min"], default)
                        self.assertLessEqual(default, field["max"])
                        self.assertLessEqual(abs(field["min"]), MAX_SAFE_INT)
                        self.assertLessEqual(abs(field["max"]), MAX_SAFE_INT)
                    elif kind == "choice":
                        self.assertIsInstance(field["choices"], list)
                        if field["choices"]:
                            self.assertIn(default, field["choices"])
                        else:  # a file list from the user's ComfyUI: /object_info fills it
                            self.assertIsInstance(default, str)
                    else:
                        self.assertIsInstance(default, str)

    def test_widget_fields_are_node_inputs(self):
        for item in load_schema()["extensions"]:
            spec = NODE_CLASS_MAPPINGS[item["class"]].INPUT_TYPES()
            inputs = {**spec.get("required", {}), **spec.get("optional", {})}
            for name in item.get("fixed", {}):
                self.assertIn(name, inputs, item["key"])
            for field in item["fields"]:
                if item["form"] == "widgets" and "input" in field:
                    self.assertIn(field["input"], inputs, field["key"])
                    options = inputs[field["input"]][1] if len(inputs[field["input"]]) > 1 else {}
                    if field["kind"] in ("int", "float"):
                        self.assertGreaterEqual(field["min"], options.get("min", field["min"]), field["key"])
                        self.assertLessEqual(field["max"], options.get("max", field["max"]), field["key"])
                    if field["kind"] == "choice" and field["choices"]:
                        self.assertLessEqual(set(field["choices"]), set(inputs[field["input"]][0]), field["key"])

    def test_json_twin_equals_json(self):
        # studio_logic.js is imported by node:test and the browser, which cannot both import a .json file: the
        # generator writes the same object as a side-effect-free ES module next to the JSON Python reads.
        source = SCHEMA_JS.read_text(encoding="utf-8")
        match = re.fullmatch(r"(?://[^\n]*\n)+export const EXT_SCHEMA = (.*);\n", source, re.S)
        self.assertIsNotNone(match, "ext_schema.js must only export EXT_SCHEMA")
        self.assertEqual(json.loads(match.group(1)), load_schema())

    def test_generator_reproduces_files(self):
        spec = importlib.util.spec_from_file_location("build_studio_ext_schema", BUILDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            json_path, js_path = module.write_schema(Path(folder))
            for made, saved in ((json_path, SCHEMA), (js_path, SCHEMA_JS)):
                with self.subTest(file=saved.name):
                    self.assertFalse(made.read_bytes().startswith(b"\xef\xbb\xbf"))
                    self.assertEqual(made.read_text(encoding="utf-8"), saved.read_text(encoding="utf-8"))


class SchemaMatchesAppSpecsTests(unittest.TestCase):
    def _check_spec(self, key: str, spec, skip=()):
        fields = _fields(key)
        used = [item for item in spec if item[1] != ag.FIXED and item[0] not in skip]
        self.assertEqual(sorted(fields), sorted(item[0] for item in used))
        for name, kind, default, extra in used:
            with self.subTest(key=name):
                field = fields[name]
                self.assertEqual(field["kind"], kind)
                self.assertEqual(field["default"], default)
                if kind in ("int", "float"):
                    self.assertEqual((field["min"], field["max"]), tuple(extra))
                if kind == "choice":
                    self.assertEqual(field["choices"], list(extra))

    def test_guidance_suite_has_every_perturbation_field(self):
        self.assertNotIn("on_key", _extension("guidance"))
        self.assertEqual(_extension("guidance")["form"], "json")
        self._check_spec("guidance", ag.PERTURBATION_SPEC)

    def test_guidance_sections_cover_the_fields(self):
        sections = {field.get("section") for field in _extension("guidance")["fields"]}
        self.assertNotIn(None, sections)
        self.assertLessEqual({"PAG·SEG", "SLG·S²", "APG", "SMC", "CWM", "DCW·RDC", "DAVE", "CNS", "Mod", "TSR", "MG",
                              "HiGS", "HiFlow"}, sections)

    def test_optimal_scale(self):
        self.assertEqual(_extension("optimal_scale")["on_key"], "ocfg_enabled")
        self._check_spec("optimal_scale", ag.OPTIMAL_SCALE_SPEC, skip=("ocfg_enabled",))

    def test_skim(self):
        self.assertEqual(_extension("skim")["on_key"], "skim_enabled")
        self._check_spec("skim", ag.SKIMMED_SPEC, skip=("skim_enabled",))

    def test_detail_daemon(self):
        self.assertEqual(_extension("detail_daemon")["on_key"], "dd_enabled")
        self.assertEqual(_extension("detail_daemon")["form"], "json")
        self._check_spec("detail_daemon", ag.DETAIL_DAEMON_SPEC, skip=("dd_enabled",))

    def test_er_sde(self):
        self._check_spec("er_sde", extra_samplers.ARG_SPEC)

    def test_speed_has_the_card_values(self):
        fields = _fields("speed")
        wanted = {key for key in anima_speed.DEFAULTS if key != "enabled"} | {"apply_img2img"}
        self.assertEqual(set(fields), wanted)
        self.assertEqual(_extension("speed")["on_key"], "enabled")
        for key, default in anima_speed.DEFAULTS.items():
            if key != "enabled":
                self.assertEqual(fields[key]["default"], default, key)
        self.assertFalse(fields["apply_img2img"]["default"])
        self.assertEqual(fields["mode"]["choices"], list(anima_speed.MODES))
        self.assertEqual(fields["threshold"]["choices"], list(anima_speed.THRESHOLDS))
        self.assertEqual(fields["preset"]["choices"], list(anima_speed.PRESET_NAMES))
        self.assertEqual(fields["transform"]["choices"], list(anima_speed.TRANSFORMS))
        self.assertEqual(_extension("speed")["fixed"], {"enabled": True,
                                                        "img2img_rescale": anima_speed.COMFY_IMG2IMG_RESCALE})

    def test_degrid_has_the_card_values(self):
        fields = _fields("degrid")
        self.assertEqual(set(fields), {"model", "mode", "strength", "tile", "apply_img2img"})
        self.assertEqual(_extension("degrid")["on_key"], "enabled")
        self.assertEqual(fields["model"]["default"], vae_degrid.COMFY_AUTO)
        self.assertEqual(fields["mode"]["default"], vae_degrid.MODE_LABELS[vae_degrid.DEFAULT_MODE])
        self.assertEqual(fields["mode"]["choices"], list(vae_degrid.MODE_LABELS.values()))
        self.assertEqual((fields["strength"]["default"], fields["strength"]["min"], fields["strength"]["max"]),
                         (vae_degrid.DEFAULT_STRENGTH, vae_degrid.STRENGTH_MIN, vae_degrid.STRENGTH_MAX))
        self.assertEqual((fields["tile"]["default"], fields["tile"]["max"]),
                         (vae_degrid.DEFAULT_TILE, vae_degrid.MAX_TILE))
        self.assertEqual(_extension("degrid")["fixed"], {
            "enabled": True, **vae_degrid.COMFY_OPTIONS, "forge_quantize": vae_degrid.COMFY_FORGE_QUANTIZE})

    def test_guidance_active_keys_are_the_compilers(self):
        active = set(_extension("guidance")["active_keys"])
        self.assertEqual(active, {*ag._ACTIVATION_KEYS[ag.SCRIPT_PERTURBATION], "guid_rdc_enabled"})

    def test_tile_settings_are_keys_the_node_takes(self):
        spec = NODE_CLASS_MAPPINGS["ForgeNeoSAM3TileRepair"].INPUT_TYPES()
        defaults = json.loads(spec["required"]["settings_json"][1]["default"])
        for key, field in _fields("sam3_tile").items():
            self.assertIn(key, defaults)
            self.assertEqual(field["default"], defaults[key], key)


class XyzCellsTests(unittest.TestCase):
    def test_every_cell_keeps_the_extensions(self):
        ext = {"skim": {"on": True, "skim_skimming_cfg": 2.5}, "speed": {"on": True, "preset": "flux"}}
        plan = plan_xyz({"ext": ext}, [{"axis": "cfg", "text": "1, 2"}], {}, rng=random.Random(0), families={})
        self.assertEqual(len(plan.cells), 2)
        for cell in plan.cells:
            self.assertTrue(cell.state["ext"]["skim"]["on"])
            self.assertEqual(cell.state["ext"]["skim"]["skim_skimming_cfg"], 2.5)
            self.assertEqual(cell.state["ext"]["speed"]["preset"], "flux")


class CompilerParityTests(unittest.TestCase):
    """The golden ``ext_compiler`` cases are what the real compiler methods build; studio_ext.js must match them."""

    def test_cases_cover_the_passes(self):
        cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["ext_compiler"]
        seen = {role for case in cases for role in case["expected"]}
        self.assertLessEqual({"ext_er_sde", "ext_guidance", "ext_optimal_scale", "ext_skim", "ext_dd", "ext_speed",
                              "ext_dd_hires", "ext_speed_hires", "ext_dd_sam3", "cns"}, seen)

    def test_golden_is_what_the_compiler_builds(self):
        for case in json.loads(GOLDEN.read_text(encoding="utf-8"))["ext_compiler"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(compiled_roles(case["state"]), case["expected"])


if __name__ == "__main__":
    unittest.main()
