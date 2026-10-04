"""Colorcraft (sam-extra) on ComfyUI — the compiler side (plan C5, design_colorcraft §7).

The bundled pack (1.7.0) has ``ForgeNeoColorcraft`` (MODEL -> MODEL): it grades the CFG-combined prediction at every
step with the modifier chain of one pass, as sam-extra's Forge hook does.  The compiler adds one node per **main**
sampling pass that has chain entries (``core/colorcraft.build_chain`` — base, which also serves an img2img request, and
the hires pass), as the last patch on that pass's sampler MODEL (plan R9: Detail Daemon -> SPEED -> Colorcraft ->
sampler), so its post-CFG function is the last one (Forge's order).  Detailer passes never get it (Forge skips the
SAM3/ADetailer inner passes).

**How the payloads are built.**  Since plan task C6 the app's ``core/colorcraft.plan()`` follows the Forge rule on
ComfyUI too (tests/test_colorcraft_settings.py checks the contributor), so these tests build the payload the same way:
``alwayson_scripts["Colorcraft (sam-extra)"] = core/colorcraft.as_block(config)`` (one compact infotext string — the
block ``plan()`` returns on both backends, main request only, and only when a pass has entries).

No ComfyUI is contacted: the pack node's schema is its real ``INPUT_TYPES``.
"""
from __future__ import annotations

import copy
import itertools
import json
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import colorcraft_host, colorcraft_nodes, speed_nodes
from core import anima_guidance, sam3_args
from core import anima_speed as asp
from core import colorcraft as cc
from core import comfy_workflow_compiler as compiler_module
from core.comfy_compatibility import RECIPES, check_recipes
from core.comfy_metadata import parse_comfy_metadata
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import feature_preflight
from tests.test_comfy_sampler_aliases import _owner_caps
from tests.test_comfy_workflow_compiler import _custom_workflow, _ksampler_advanced_workflow, _node

NODE = cc.COMFY_NODE_CLASS
BASE_ONLY = "v1;mods=I;I.kind=Luma;I.exposure=0.3"
BOTH = ("v1;mods=I,III;masking=1;masks=M1;combos=;I.kind=Luma;I.mask=M1;I.exposure=0.3;I.tone_compression=-0.2;"
        "III.kind=Punch;III.pass=Hires;III.clarity=0.5;III.sharpness=0.25;M1.mask_mode=lows;M1.mask_hardness=2;"
        "M1.blur=8")
EVERY_PASS = "v1;mods=II;II.kind=Basic;II.pass=Both;II.contrast=0.4"
HIRES_ONLY = "v1;mods=II;II.kind=Punch;II.pass=Hires;II.clarity=0.5"
NOOP = "v1;mods=I;I.kind=Luma"
SKIMMED = {anima_guidance.SCRIPT_SKIMMED_CFG: {"args": {"skim_enabled": True}}}
DETAIL_DAEMON = {anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True, "dd_amount": 0.3}}}
DETAIL_DAEMON_HIRES = {anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True, "dd_amount": 0.3,
                                                                       "dd_hires": True}}}


def caps(*, colorcraft_node: bool = True) -> dict:
    result = _owner_caps()
    result["DiTSpectrumPatch"] = {"input": {"required": {}}}
    result[asp.COMFY_NODE_CLASS] = {"input": speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES(), "output": ["MODEL"]}
    if colorcraft_node:
        result[NODE] = {"input": colorcraft_nodes.ForgeNeoColorcraft.INPUT_TYPES(), "output": ["MODEL"]}
    return result


def config(text: str) -> cc.Config:
    return cc.parse_own(text)


def block(text: str = BASE_ONLY) -> dict:
    """What ``core/colorcraft.plan()`` sends (C6: also on ComfyUI)."""
    return cc.as_block(config(text))


def payload(*, cc_block=None, speed=None, adetailer=None, sam3=None, scripts=None, **extra) -> dict:
    found: dict = dict(scripts or {})
    if adetailer is not None:
        found["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face.pt", **adetailer}]}
    if sam3 is not None:
        found.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    if speed is not None:
        found[asp.SCRIPT_NAME] = asp.as_block(speed)
    if cc_block is not None:
        found[cc.SCRIPT_NAME] = cc_block
    return {"prompt": "portrait", "negative_prompt": "bad", "seed": 7, "steps": 28, "sampler_name": "euler",
            "alwayson_scripts": found, **extra}


def compile_graph(body, *, mode="txt2img", workflow=None, capabilities=None, warnings=None) -> dict:
    uploads = {} if mode == "txt2img" else {"uploaded_image": "in.png"}
    if mode == "inpaint":
        uploads["uploaded_mask"] = "mask.png"
    return ComfyWorkflowCompiler(capabilities or caps()).compile(
        mode, "checkpoint.safetensors", body, workflow=workflow, warnings=warnings, **uploads)


def cc_nodes(graph: dict) -> list:
    return [(node_id, node) for node_id, node in graph.items() if node.get("class_type") == NODE]


def model_chain(graph: dict, link) -> list:
    chain, seen = [], set()
    while isinstance(link, list) and len(link) >= 2 and str(link[0]) in graph and str(link[0]) not in seen:
        seen.add(str(link[0]))
        node = graph[str(link[0])]
        chain.append(node["class_type"])
        link = node.get("inputs", {}).get("model")
    return chain


def consumers(graph: dict, node_id: str) -> list:
    return [(node["class_type"], key) for node in graph.values()
            for key, value in node.get("inputs", {}).items()
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == str(node_id)]


def settings_of(node: dict) -> dict:
    return json.loads(node["inputs"]["settings_json"])


def expected_settings(text: str, *, hires: bool) -> dict:
    chain = cc.build_chain(config(text), hires)
    return {"v": 1, "pass": "hires" if hires else "base", "entries": chain.entries,
            "infotext": cc.to_infotext(config(text)), "tags": chain.tags}


def dumps(graph: dict) -> str:
    return json.dumps(graph, sort_keys=True)


class ColorcraftInsertionTests(unittest.TestCase):
    """Where ``ForgeNeoColorcraft`` goes in and what it receives."""

    def test_nothing_to_grade_keeps_the_graph_byte_identical(self):
        variants = {"plain": {}, "hires": {"enable_hr": True}, "guidance": {"scripts": {**SKIMMED, **DETAIL_DAEMON}},
                    "adetailer": {"adetailer": {}}}
        for label, extra in variants.items():
            without = compile_graph(payload(**extra))
            self.assertEqual(cc_nodes(without), [])
            empty = [block(NOOP), {"args": [{"enabled": False, "I.exposure": 0.3}]}, {"args": [{"enabled": True}]}]
            if not extra.get("enable_hr"):
                empty.append(block(HIRES_ONLY))
            for raw in empty:
                with self.subTest(variant=label, block=raw):
                    warnings: list = []
                    self.assertEqual(dumps(compile_graph(payload(cc_block=raw, **extra), warnings=warnings)),
                                     dumps(without))
                    self.assertEqual([(w["feature"], w["code"], w["cause"]) for w in warnings],
                                     [("colorcraft", "colorcraft_no_entries", None)])
                    self.assertIn("도는 수정자가 없어", warnings[0]["reason"])

    def test_the_base_sampler_gets_the_node_last(self):
        schema = colorcraft_nodes.ForgeNeoColorcraft.INPUT_TYPES()["required"]
        for guidance, speed in itertools.product(({}, {**SKIMMED, **DETAIL_DAEMON}), (None, asp.SpeedSettings(enabled=True))):
            with self.subTest(guidance=bool(guidance), speed=bool(speed)):
                graph = compile_graph(payload(cc_block=block(), scripts=guidance, speed=speed))
                (cc_id, node), = cc_nodes(graph)
                _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                self.assertEqual(sampler["inputs"]["model"], [cc_id, 0])
                self.assertEqual(consumers(graph, cc_id), [("ForgeNeoKSamplerCNS", "model")])
                expected_chain = (["ForgeNeoAnimaSpeed"] if speed else []) + (
                    ["ForgeNeoAnimaDetailDaemon", "ForgeNeoSkimmedCFG"] if guidance else [])
                self.assertEqual(model_chain(graph, node["inputs"]["model"])[:len(expected_chain)], expected_chain)
                inputs = node["inputs"]
                self.assertEqual(tuple(inputs), cc.COMFY_INPUTS)
                self.assertEqual(set(inputs), set(schema))
                _lid, loader = _node(graph, "CheckpointLoaderSimple")
                self.assertEqual(inputs["vae"], [_lid, 2])
                self.assertEqual((inputs["contract"], inputs["pre_dd_sigma"]), (cc.COMFY_CONTRACT, True))
                self.assertIn(inputs["contract"], schema["contract"][0])
                self.assertEqual(settings_of(node), expected_settings(BASE_ONLY, hires=False))
                self.assertEqual(node["_meta"]["title"], "Colorcraft (base pass)")
                parsed = colorcraft_host.parse_settings(inputs["settings_json"])
                self.assertEqual((parsed["entries"], parsed["tags"], parsed["pass"]),
                                 (cc.build_chain(config(BASE_ONLY), False).entries, ["I"], "base"))
                self.assertNotIn("seed", inputs["settings_json"])

    def test_each_pass_gets_its_own_chain(self):
        cases = ((BOTH, ["base", "hires"]), (EVERY_PASS, ["base", "hires"]), (HIRES_ONLY, ["hires"]),
                 (BASE_ONLY, ["base"]))
        for text, passes in cases:
            for dd, speed in itertools.product(({}, DETAIL_DAEMON_HIRES), (None, asp.SpeedSettings(enabled=True,
                                                                                                 hires=True))):
                with self.subTest(config=text, dd=bool(dd), speed=bool(speed)):
                    graph = compile_graph(payload(cc_block=block(text), enable_hr=True, scripts=dd, speed=speed))
                    nodes = cc_nodes(graph)
                    self.assertEqual([settings_of(node)["pass"] for _nid, node in nodes], passes)
                    for _nid, node in nodes:
                        self.assertEqual(settings_of(node),
                                         expected_settings(text, hires=settings_of(node)["pass"] == "hires"))
                    _hid, hires = _node(graph, "ForgeNeoHiresFix")
                    chain = model_chain(graph, hires["inputs"]["model"])
                    if "hires" in passes:
                        self.assertEqual(hires["inputs"]["model"], [nodes[-1][0], 0])
                        self.assertEqual(nodes[-1][1]["_meta"]["title"], "Colorcraft (hires pass)")
                        expected = ([NODE] + (["ForgeNeoAnimaSpeed"] if speed else [])
                                    + (["ForgeNeoAnimaDetailDaemon"] if dd else []))
                        self.assertEqual(chain[:len(expected)], expected)
                    else:
                        self.assertNotIn(NODE, chain)

    def test_detailer_passes_never_get_colorcraft(self):
        bodies = (
            payload(cc_block=block(EVERY_PASS), enable_hr=True, adetailer={}, scripts=DETAIL_DAEMON),
            payload(cc_block=block(BASE_ONLY), adetailer={"ad_prompt": "face <lora:characters/alice:1>"}),
            payload(cc_block=block(BASE_ONLY), sam3={}, _comfy_detail_passes=["hand"]),
            payload(cc_block=block(BASE_ONLY), sam3={"sam3_inpaint_prompt": "eye <lora:styles/ink:0.5>"}),
        )
        for body in bodies:
            with self.subTest(body=body):
                graph = compile_graph(body)
                for node in graph.values():
                    if node["class_type"] in ("ForgeNeoADetailer", "ForgeNeoSAM3Detailer"):
                        self.assertNotIn(NODE, model_chain(graph, node["inputs"]["model"]))
                for cc_id, _node_ in cc_nodes(graph):
                    self.assertIn(consumers(graph, cc_id),
                                  ([("ForgeNeoKSamplerCNS", "model")], [("ForgeNeoHiresFix", "model")]))

    def test_img2img_and_inpaint_use_the_base_chain(self):
        for mode in ("img2img", "inpaint"):
            with self.subTest(mode=mode):
                graph = compile_graph(payload(cc_block=block(BOTH)), mode=mode)
                (_cid, node), = cc_nodes(graph)
                self.assertEqual(settings_of(node), expected_settings(BOTH, hires=False))

    def test_custom_workflows(self):
        for workflow in (_custom_workflow(), _ksampler_advanced_workflow()):
            with self.subTest(sampler=workflow["5"]["class_type"]):
                graph = compile_graph(payload(cc_block=block(BOTH), enable_hr=True, scripts=DETAIL_DAEMON,
                                              speed=asp.SpeedSettings(enabled=True)), workflow=workflow)
                nodes = cc_nodes(graph)
                self.assertEqual([settings_of(node)["pass"] for _nid, node in nodes], ["base", "hires"])
                self.assertEqual(graph["5"]["inputs"]["model"], [nodes[0][0], 0])
                self.assertEqual(model_chain(graph, nodes[0][1]["inputs"]["model"])[:2],
                                 ["ForgeNeoAnimaSpeed", "ForgeNeoAnimaDetailDaemon"])
                self.assertEqual(nodes[0][1]["inputs"]["vae"], ["1", 2])     # the workflow's own decode VAE
                _hid, hires = _node(graph, "ForgeNeoHiresFix")
                self.assertEqual(hires["inputs"]["model"], [nodes[1][0], 0])
        workflow = _custom_workflow()
        del workflow["6"], workflow["7"]
        with mock.patch.object(ComfyWorkflowCompiler, "_infer_vae_from_model", return_value=None):
            compile_graph(payload(), workflow=workflow)
            with self.assertRaisesRegex(WorkflowCompileError, "Colorcraft 에 필요한 VAE 연결을 사용자 워크플로에서 찾지"):
                compile_graph(payload(cc_block=block()), workflow=workflow)

    def test_block_forms_and_unreadable_blocks(self):
        mapping = {"enabled": True, "I.kind": "Luma", "I.exposure": 0.3}
        graph = compile_graph(payload(cc_block={"args": [mapping]}))
        (_cid, node), = cc_nodes(graph)
        self.assertEqual(settings_of(node)["entries"], cc.build_chain(cc.config_from_mapping(mapping), False).entries)
        self.assertEqual(settings_of(node)["infotext"], cc.to_infotext(cc.config_from_mapping(mapping)))
        for raw in ({"args": ["garbage"]}, {"args": [[1, 2]]}, {"args": []}, {}):
            with self.subTest(block=raw), self.assertRaisesRegex(WorkflowCompileError, "Colorcraft 블록을 읽지 못했습니다"):
                compile_graph(payload(cc_block=raw))

    def test_a_stale_pack_names_the_feature(self):
        with self.assertRaisesRegex(WorkflowCompileError, "필요한 노드가 없습니다: ForgeNeoColorcraft\\..*Colorcraft에는 "
                                    "AI Studio Forge Parity 노드 팩 1\\.7\\.0 이상이 필요합니다"):
            compile_graph(payload(cc_block=block()), capabilities=caps(colorcraft_node=False))
        compile_graph(payload(cc_block=block(NOOP)), capabilities=caps(colorcraft_node=False))

    def test_plan_r9_model_chain_with_all_four_sam_extra_features(self):
        """R9: loader -> flow shift -> Extra Schedulers -> LoRA -> Extra Samplers -> guidance stack -> per pass:
        Detail Daemon -> SPEED -> Colorcraft -> sampler / Hires.fix."""
        from core import extra_samplers as xsa, extra_schedulers as xs
        body = payload(cc_block=block(EVERY_PASS), speed=asp.SpeedSettings(enabled=True, hires=True), enable_hr=True,
                       scheduler="Laplace", distilled_cfg_scale=3.0, prompt="portrait <lora:styles/ink:0.5>",
                       scripts={**SKIMMED, anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True}}})
        body["alwayson_scripts"][xs.SCRIPT_NAME] = xs.as_block(xs.APP_DEFAULTS)
        body["alwayson_scripts"][xsa.SCRIPT_NAME] = xsa.as_block(xsa.ExtraSamplersSettings(eta=0.5))
        capabilities = {**caps(), **{name: schema for name, schema in _owner_caps(schedulers=("laplace",)).items()
                                     if name in ("KSampler", xs.COMFY_NODE_CLASS)}}
        graph = compile_graph(body, capabilities=capabilities)
        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(model_chain(graph, sampler["inputs"]["model"]), [
            NODE, "ForgeNeoAnimaSpeed", "ForgeNeoAnimaDetailDaemon", "ForgeNeoSkimmedCFG", xsa.COMFY_SETTINGS_NODE,
            "LoraLoader", xs.COMFY_NODE_CLASS, "ForgeNeoModelSamplingShift", "CheckpointLoaderSimple"])
        _hid, hires = _node(graph, "ForgeNeoHiresFix")
        self.assertEqual(model_chain(graph, hires["inputs"]["model"]), [
            NODE, "ForgeNeoAnimaSpeed", "ForgeNeoSkimmedCFG", xsa.COMFY_SETTINGS_NODE,
            "LoraLoader", xs.COMFY_NODE_CLASS, "ForgeNeoModelSamplingShift", "CheckpointLoaderSimple"])

    def test_graph_invariant(self):
        count = 0
        for text, enable_hr, speed, dd, extra, custom in itertools.product(
                (BASE_ONLY, BOTH, HIRES_ONLY, EVERY_PASS), (False, True),
                (None, asp.SpeedSettings(enabled=True, hires=True)), ({}, DETAIL_DAEMON, DETAIL_DAEMON_HIRES),
                ({}, {"adetailer": {}}, {"sam3": {}}), (False, True)):
            body = payload(cc_block=block(text), enable_hr=enable_hr, speed=speed, scripts=dd, **extra)
            warnings: list = []
            graph = compile_graph(body, workflow=_custom_workflow() if custom else None, warnings=warnings)
            expected = [pass_ for pass_, hires in (("base", False), ("hires", True))
                        if (enable_hr or not hires) and cc.build_chain(config(text), hires).entries]
            with self.subTest(config=text, hr=enable_hr, speed=bool(speed), dd=dd, extra=extra, custom=custom):
                nodes = cc_nodes(graph)
                self.assertEqual([settings_of(node)["pass"] for _nid, node in nodes], expected)
                self.assertEqual(bool([w for w in warnings if w["feature"] == "colorcraft"]), not expected)
                for cc_id, _node_ in nodes:
                    (reader,) = consumers(graph, cc_id)
                    self.assertIn(reader, (("ForgeNeoKSamplerCNS", "model"), ("KSampler", "model"),
                                           ("ForgeNeoHiresFix", "model")))
                for node in graph.values():
                    if node["class_type"] in ("ForgeNeoADetailer", "ForgeNeoSAM3Detailer"):
                        self.assertNotIn(NODE, model_chain(graph, node["inputs"]["model"]))
            count += 1
        self.assertEqual(count, 4 * 2 * 2 * 3 * 3 * 2)


def _infotext(graph: dict):
    return parse_comfy_metadata(graph)["parameters"].get(cc.INFOTEXT_KEY)


class ColorcraftMetadataTests(unittest.TestCase):
    def test_the_infotext_equals_forges(self):
        for text, extra in ((BASE_ONLY, {}), (BOTH, {"enable_hr": True}), (HIRES_ONLY, {"enable_hr": True}),
                            (EVERY_PASS, {"enable_hr": True}), (BOTH, {})):
            with self.subTest(config=text, extra=extra):
                params = parse_comfy_metadata(compile_graph(payload(cc_block=block(text), **extra)))["parameters"]
                self.assertEqual(params[cc.INFOTEXT_KEY], cc.to_infotext(config(text)))
                for key in (cc.STATUS_KEY, cc.PRE_DD_INFOTEXT_KEY):
                    self.assertNotIn(key, params)
                restored = cc.from_infotext(params)
                for hires in (False, True):
                    self.assertEqual(cc.build_chain(restored, hires).entries, cc.build_chain(config(text), hires).entries)

    def test_no_node_no_key(self):
        for body in (payload(), payload(cc_block=block(NOOP)), payload(cc_block=block(HIRES_ONLY))):
            with self.subTest(body=body):
                self.assertIsNone(_infotext(compile_graph(body)))

    def test_hand_made_graphs(self):
        graph = compile_graph(payload(cc_block=block(BOTH), enable_hr=True))
        (base_id, base), (hires_id, _hires) = cc_nodes(graph)
        linked = copy.deepcopy(graph)
        linked["90"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": base["inputs"]["settings_json"]}}
        linked[base_id]["inputs"]["settings_json"] = ["90", 0]
        self.assertEqual(_infotext(linked), cc.to_infotext(config(BOTH)))
        broken = copy.deepcopy(graph)
        broken[base_id]["inputs"]["settings_json"] = "{not json"
        result = parse_comfy_metadata(broken)
        self.assertNotIn(cc.INFOTEXT_KEY, result["parameters"])
        self.assertIn("Colorcraft 노드 설정을 읽지 못해 Colorcraft 값을 기록하지 않았습니다.", result["warnings"])
        two = copy.deepcopy(graph)
        settings = json.loads(two[hires_id]["inputs"]["settings_json"])
        settings["infotext"] = BASE_ONLY
        two[hires_id]["inputs"]["settings_json"] = json.dumps(settings)
        result = parse_comfy_metadata(two)
        self.assertNotIn(cc.INFOTEXT_KEY, result["parameters"])
        self.assertIn("Colorcraft 노드가 여럿이고 설정이 달라 Colorcraft 값을 고르지 않았습니다.", result["warnings"])
        empty = copy.deepcopy(graph)
        settings = json.loads(empty[hires_id]["inputs"]["settings_json"])
        settings.update(entries=[], tags=[], infotext=BASE_ONLY)
        empty[hires_id]["inputs"]["settings_json"] = json.dumps(settings)
        self.assertEqual(_infotext(empty), cc.to_infotext(config(BOTH)))

    def test_web_workflow_format(self):
        node = cc_nodes(compile_graph(payload(cc_block=block(BOTH))))[0][1]["inputs"]
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["checkpoint.safetensors"], "inputs": []},
                {"id": 2, "type": NODE, "widgets_values": [node["settings_json"], "v1", True],
                 "inputs": [{"name": "model", "link": 1}, {"name": "vae", "link": 4}]},
                {"id": 5, "type": "KSampler", "widgets_values": [7, "fixed", 20, 5.0, "euler", "normal", 1.0],
                 "inputs": [{"name": "model", "link": 2}]},
                {"id": 8, "type": "SaveImage", "widgets_values": ["out"], "inputs": [{"name": "images", "link": 3}]},
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 5, 0, "MODEL"], [3, 5, 0, 8, 0, "IMAGE"],
                      [4, 1, 2, 2, 1, "VAE"]],
        }
        self.assertEqual(parse_comfy_metadata(None, workflow)["parameters"][cc.INFOTEXT_KEY],
                         cc.to_infotext(config(BOTH)))


class ColorcraftPreflightTests(unittest.TestCase):
    def _row(self, result: dict) -> dict:
        return next(row for row in result["features"] if row["id"] == "colorcraft")

    def test_rows(self):
        compiler = ComfyWorkflowCompiler(caps())
        ready = feature_preflight(compiler, "checkpoint.safetensors", payload(cc_block=block()))
        self.assertTrue(ready["ok"], ready)
        self.assertEqual((self._row(ready)["label"], self._row(ready)["state"]), ("Colorcraft", "ready"))
        skipped = feature_preflight(compiler, "checkpoint.safetensors", payload(cc_block=block(HIRES_ONLY)))
        self.assertTrue(skipped["ok"], skipped)
        self.assertEqual(self._row(skipped)["state"], "skipped")
        self.assertIn("Hires 전용 수정자인데 Hires 가 꺼짐", self._row(skipped)["reason"])
        off = feature_preflight(compiler, "checkpoint.safetensors", payload())
        self.assertEqual((self._row(off)["state"], self._row(off)["requested"]), ("off", False))
        for capabilities, body, error in (
                (caps(colorcraft_node=False), payload(cc_block=block()), "Colorcraft에는 AI Studio Forge Parity 노드 팩"),
                (caps(), payload(cc_block={"args": ["garbage"]}), "Colorcraft 블록을 읽지 못했습니다")):
            with self.subTest(error=error):
                result = feature_preflight(ComfyWorkflowCompiler(capabilities), "checkpoint.safetensors", body)
                self.assertFalse(result["ok"])
                self.assertEqual(self._row(result)["state"], "blocked")
                self.assertIn(error, result["error"])


class ColorcraftContractTests(unittest.TestCase):
    def test_compiler_constants_match_the_pack(self):
        schema = colorcraft_nodes.ForgeNeoColorcraft.INPUT_TYPES()["required"]
        self.assertEqual(tuple(schema), cc.COMFY_INPUTS)
        self.assertEqual(colorcraft_nodes.NODE_ID, NODE)
        self.assertIn(cc.COMFY_CONTRACT, colorcraft_nodes.CONTRACTS)
        self.assertIs(schema["pre_dd_sigma"][1]["default"], cc.COMFY_PRE_DD_SIGMA)
        self.assertEqual(colorcraft_host.SETTINGS_VERSION, compiler_module._COLORCRAFT_SETTINGS_VERSION)
        self.assertEqual(colorcraft_host.PASSES,
                         (compiler_module._COLORCRAFT_PASS_BASE, compiler_module._COLORCRAFT_PASS_HIRES))

    def test_the_recipe_lists_the_pack_node_inputs(self):
        recipe = next(item for item in RECIPES if item["id"] == "colorcraft")
        self.assertEqual(recipe["title"], cc.SCRIPT_NAME)
        fields = recipe["nodes"][NODE]
        schema = colorcraft_nodes.ForgeNeoColorcraft.INPUT_TYPES()["required"]
        self.assertEqual(tuple(fields), cc.COMFY_INPUTS)
        for name, kind in fields.items():
            spec = schema[name][0]
            self.assertEqual("CHOICE" if isinstance(spec, list) else spec, kind, name)
        self.assertIn(cc.COMFY_MIN_PACK_VERSION, recipe["note"])
        self.assertEqual(recipe["models"], [])
        results = {item["id"]: item for item in check_recipes({NODE: {"input": {"required": schema}}})}
        self.assertEqual(results["colorcraft"]["status"], "available")
        self.assertEqual({item["id"]: item for item in check_recipes({})}["colorcraft"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
