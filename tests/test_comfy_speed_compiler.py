"""Anima SPEED (sam-extra v0.33.0) on ComfyUI — the compiler side (plan C4a part 1, C4b part 2).

The bundled pack (1.7.0) has ``ForgeNeoAnimaSpeed`` (MODEL -> MODEL): its sampler wrapper runs the early,
noise-dominated steps of a pass on a DCT-truncated coarse latent like sam-extra's Forge script.  The compiler adds one
node per **main** sampling pass (base or img2img; Hires only when the block's ``hires`` is on) right after that pass's
Detail Daemon — detailer passes never get it (Forge's inner passes are masked or start below the transition sigma) —
and leaves it out with a warning when sam-extra would not run SPEED at all: invalid settings, or a skip known before
sampling (inpaint, ControlNet, ComfyUI Spectrum — D6, a single scale).

**How the payloads are built.**  Since plan task C6 the app's ``core/anima_speed.plan()`` follows the Forge rule on
ComfyUI too (tests/test_anima_speed_ui.py checks the contributor), so these tests build the payload the same way:
``alwayson_scripts["Anima SPEED"] = core/anima_speed.as_block(settings)`` (one dict with the fourteen keys — the block
``plan()`` returns on both backends, main request only); positional and alias forms come from hand-written payloads.

No ComfyUI is contacted: the pack node's schema is its real ``INPUT_TYPES``.
"""
from __future__ import annotations

import copy
import itertools
import json
import unittest
from dataclasses import replace

from comfy_custom_nodes.ai_studio_forge_parity import speed_nodes, speed_schedule
from core import anima_guidance, sam3_args
from core import anima_speed as asp
from core import comfy_workflow_compiler as compiler_module
from core import sam_extra_notices as notices
from core.comfy_compatibility import RECIPES, check_recipes
from core.comfy_metadata import parse_comfy_metadata
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import feature_preflight
from core.spectrum_settings import skips_anima_speed
from tests.test_comfy_sampler_aliases import _owner_caps
from tests.test_comfy_workflow_compiler import _custom_workflow, _ksampler_advanced_workflow, _node

NODE = asp.COMFY_NODE_CLASS
SPEED = asp.SpeedSettings(enabled=True)
SKIMMED = {anima_guidance.SCRIPT_SKIMMED_CFG: {"args": {"skim_enabled": True}}}
DETAIL_DAEMON = {anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True, "dd_amount": 0.3}}}
DETAIL_DAEMON_HIRES = {anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True, "dd_amount": 0.3,
                                                                       "dd_hires": True}}}


def caps(*, speed_node: bool = True) -> dict:
    result = _owner_caps()
    result["DiTSpectrumPatch"] = {"input": {"required": {}}}
    if speed_node:
        result[NODE] = {"input": speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES(), "output": ["MODEL"]}
    return result


def block(settings: asp.SpeedSettings = SPEED) -> dict:
    """What ``core/anima_speed.plan()`` sends (C6: also on ComfyUI)."""
    return asp.as_block(settings)


def payload(*, speed_block=None, adetailer=None, sam3=None, scripts=None, **extra) -> dict:
    found: dict = dict(scripts or {})
    if adetailer is not None:
        found["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face.pt", **adetailer}]}
    if sam3 is not None:
        found.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    if speed_block is not None:
        found[asp.SCRIPT_NAME] = speed_block
    return {"prompt": "portrait", "negative_prompt": "bad", "seed": 7, "steps": 28, "sampler_name": "euler",
            "alwayson_scripts": found, **extra}


def compile_graph(body, *, mode="txt2img", workflow=None, capabilities=None, warnings=None) -> dict:
    uploads = {} if mode == "txt2img" else {"uploaded_image": "in.png"}
    if mode == "inpaint":
        uploads["uploaded_mask"] = "mask.png"
    return ComfyWorkflowCompiler(capabilities or caps()).compile(
        mode, "checkpoint.safetensors", body, workflow=workflow, warnings=warnings, **uploads)


def speed_nodes_of(graph: dict) -> list:
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
    """(consumer class, input name) of every input that reads ``node_id``."""
    return [(node["class_type"], key) for node in graph.values()
            for key, value in node.get("inputs", {}).items()
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == str(node_id)]


def dumps(graph: dict) -> str:
    return json.dumps(graph, sort_keys=True)


def expected_inputs(script: asp.ScriptSettings) -> dict:
    """The node's settings inputs for ``script`` (all values inside the node's ranges)."""
    return {
        "enabled": True, "mode": script.mode, "threshold": script.threshold, "preset": script.preset,
        "scales": ",".join(repr(float(v)) for v in script.scales), "delta": script.delta,
        "sigma_divisor": script.sigma_divisor, "manual_sigmas": ",".join(repr(float(v)) for v in script.manual_sigmas),
        "adaptive": script.adaptive, "transform": script.transform, "spectrum_A": script.spectrum_A,
        "spectrum_beta": script.spectrum_beta, "seed": script.seed,
    }


def node_script(node: dict) -> asp.ScriptSettings:
    """What the pack node's own coercion (= sam-extra's) reads from the node's inputs (without ``hires``)."""
    inputs = node["inputs"]
    return asp.coerce([{name: inputs[name] for name in asp.ARG_NAMES if name in inputs}])


class SpeedInsertionTests(unittest.TestCase):
    """C4a: where ``ForgeNeoAnimaSpeed`` goes in and what it receives."""

    def test_no_block_or_a_disabled_block_keeps_the_graph_byte_identical(self):
        variants = {"plain": {}, "hires": {"enable_hr": True}, "guidance": {"scripts": {**SKIMMED, **DETAIL_DAEMON}},
                    "adetailer": {"adetailer": {}}, "sam3": {"sam3": {}}}
        disabled = (block(asp.SpeedSettings()), {"args": [False]}, {"args": [{"enabled": "false", "hires": True}]},
                    {"args": []}, {})
        for label, extra in variants.items():
            without = compile_graph(payload(**extra))
            self.assertEqual(speed_nodes_of(without), [])
            for raw in disabled:
                with self.subTest(variant=label, block=raw):
                    warnings: list = []
                    self.assertEqual(dumps(compile_graph(payload(speed_block=raw, **extra), warnings=warnings)),
                                     dumps(without))
                    self.assertEqual(warnings, [])
        workflow = _custom_workflow()
        self.assertEqual(dumps(compile_graph(payload(speed_block=block(asp.SpeedSettings())), workflow=workflow)),
                         dumps(compile_graph(payload(), workflow=workflow)))

    def test_the_base_sampler_gets_one_node_right_after_detail_daemon(self):
        schema = speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES()["required"]
        for guidance in ({}, {**SKIMMED, **DETAIL_DAEMON}):
            with self.subTest(guidance=bool(guidance)):
                graph = compile_graph(payload(speed_block=block(), scripts=guidance))
                (speed_id, speed), = speed_nodes_of(graph)
                _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                self.assertEqual(sampler["inputs"]["model"], [speed_id, 0])
                self.assertEqual(consumers(graph, speed_id), [("ForgeNeoKSamplerCNS", "model")])
                chain = model_chain(graph, speed["inputs"]["model"])
                self.assertEqual(chain[:2], ["ForgeNeoAnimaDetailDaemon", "ForgeNeoSkimmedCFG"] if guidance
                                 else ["CheckpointLoaderSimple"])
                inputs = speed["inputs"]
                self.assertEqual(tuple(inputs), asp.COMFY_INPUTS)
                self.assertEqual(set(inputs), set(schema))
                script = asp.parse_script_block(block())
                self.assertEqual({key: value for key, value in inputs.items() if key in expected_inputs(script)},
                                 expected_inputs(script))
                self.assertEqual((inputs["pass_label"], inputs["image_seed"], inputs["img2img_rescale"]),
                                 ("base", 7, True))
                self.assertEqual(speed["_meta"]["title"], "Anima SPEED (base pass)")
                self.assertIs(sampler["inputs"]["speed_enabled"], False)      # legacy provider input stays off
                self.assertEqual(node_script(speed), script)

    def test_hires_gets_its_own_node_only_with_the_block_hires_flag(self):
        for hires_flag, dd in itertools.product((False, True), ({}, DETAIL_DAEMON_HIRES)):
            with self.subTest(hires=hires_flag, dd=bool(dd)):
                graph = compile_graph(payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=hires_flag)),
                                              enable_hr=True, scripts=dd))
                nodes = speed_nodes_of(graph)
                labels = [node["inputs"]["pass_label"] for _nid, node in nodes]
                self.assertEqual(labels, ["base", "hires"] if hires_flag else ["base"])
                _hid, hires = _node(graph, "ForgeNeoHiresFix")
                chain = model_chain(graph, hires["inputs"]["model"])
                if hires_flag:
                    self.assertEqual(hires["inputs"]["model"], [nodes[1][0], 0])
                    self.assertEqual(chain[1], "ForgeNeoAnimaDetailDaemon" if dd else "CheckpointLoaderSimple")
                    self.assertEqual(nodes[1][1]["_meta"]["title"], "Anima SPEED (hires pass)")
                    self.assertEqual(nodes[1][1]["inputs"]["image_seed"], 7)
                else:
                    self.assertNotIn(NODE, chain)
        no_hires = compile_graph(payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=True))))
        self.assertEqual([node["inputs"]["pass_label"] for _nid, node in speed_nodes_of(no_hires)], ["base"])

    def test_detailer_passes_never_get_speed(self):
        bodies = {
            "adetailer": payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=True)), enable_hr=True,
                                 adetailer={}, scripts=DETAIL_DAEMON),
            "adetailer own LoRA": payload(speed_block=block(), adetailer={"ad_prompt": "face <lora:characters/alice:1>"},
                                          scripts=DETAIL_DAEMON),
            "sam3 + detail pass": payload(speed_block=block(), sam3={}, _comfy_detail_passes=["hand"]),
            "sam3 own LoRA": payload(speed_block=block(), sam3={"sam3_inpaint_prompt": "eye <lora:styles/ink:0.5>"}),
        }
        for label, body in bodies.items():
            with self.subTest(case=label):
                graph = compile_graph(body)
                for node in graph.values():
                    if node["class_type"] in ("ForgeNeoADetailer", "ForgeNeoSAM3Detailer"):
                        self.assertNotIn(NODE, model_chain(graph, node["inputs"]["model"]))
                for speed_id, _speed in speed_nodes_of(graph):
                    self.assertIn(consumers(graph, speed_id),
                                  ([("ForgeNeoKSamplerCNS", "model")], [("ForgeNeoHiresFix", "model")]))

    def test_img2img_label_and_inpaint_skip(self):
        graph = compile_graph(payload(speed_block=block()), mode="img2img")
        (_sid, speed), = speed_nodes_of(graph)
        self.assertEqual((speed["inputs"]["pass_label"], speed["_meta"]["title"]), ("img2img", "Anima SPEED (img2img pass)"))
        warnings: list = []
        inpaint = compile_graph(payload(speed_block=block()), mode="inpaint", warnings=warnings)
        self.assertEqual(speed_nodes_of(inpaint), [])
        self.assertEqual(warnings, [{"code": notices.CODE_SPEED_SKIPPED, "feature": "speed", "cause": None,
                                     "reason": asp.SKIP_MESSAGES[asp.SKIP_INPAINT]}])
        self.assertEqual(dumps(inpaint), dumps(compile_graph(payload(), mode="inpaint")))

    def test_custom_workflows(self):
        for workflow in (_custom_workflow(), _ksampler_advanced_workflow()):
            with self.subTest(sampler=workflow["5"]["class_type"]):
                body = payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=True)), enable_hr=True,
                               scripts=DETAIL_DAEMON)
                graph = compile_graph(body, workflow=workflow)
                nodes = speed_nodes_of(graph)
                self.assertEqual([node["inputs"]["pass_label"] for _nid, node in nodes], ["base", "hires"])
                self.assertEqual(graph["5"]["inputs"]["model"], [nodes[0][0], 0])
                self.assertEqual(model_chain(graph, nodes[0][1]["inputs"]["model"])[0], "ForgeNeoAnimaDetailDaemon")
                _hid, hires = _node(graph, "ForgeNeoHiresFix")
                self.assertEqual(hires["inputs"]["model"], [nodes[1][0], 0])
        img2img = compile_graph(payload(speed_block=block()), mode="img2img", workflow=_custom_workflow())
        self.assertEqual([node["inputs"]["pass_label"] for _nid, node in speed_nodes_of(img2img)], ["img2img"])

    def test_every_block_form_gives_the_values_sam_extra_reads(self):
        card = asp.SpeedSettings(enabled=True, mode="respace", threshold="delta_optimal", preset="custom",
                                 scales="1/4, 1/2; 1", delta=0.02, sigma_divisor=1.2, manual_sigmas="0.8,0.4",
                                 adaptive=False, transform="fft", spectrum_A=150.5, spectrum_beta=2.25, seed=1234,
                                 hires=True)
        positional = [card.as_arg()[name] for name in asp.ARG_NAMES]
        aliased = {**{k: v for k, v in card.as_arg().items() if k not in ("sigma_divisor", "manual_sigmas",
                                                                         "spectrum_A", "spectrum_beta")},
                   "divisor": 1.2, "manual": "0.8,0.4", "A": 150.5, "beta": 2.25}
        texts = {key: (str(value).lower() if isinstance(value, bool) else str(value))
                 for key, value in card.as_arg().items()}
        script = asp.coerce([card.as_arg()])
        self.assertIsNone(script.error)
        for label, raw in (("dict", block(card)), ("positional", {"args": positional}), ("aliases", {"args": [aliased]}),
                           ("strings", {"args": [texts]})):
            with self.subTest(block=label):
                graph = compile_graph(payload(speed_block=raw, enable_hr=True))
                for _nid, node in speed_nodes_of(graph):
                    self.assertEqual({key: node["inputs"][key] for key in expected_inputs(script)},
                                     expected_inputs(script))
                    self.assertEqual(node_script(node), asp.coerce([{**card.as_arg(), "hires": False}]))
                    self.assertEqual(node["inputs"]["scales"], "0.25,0.5,1.0")

    def test_values_outside_the_node_ranges(self):
        schema = speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES()["required"]
        unused = (
            ("manual threshold", {"threshold": "manual", "delta": 5.0, "sigma_divisor": -3.0, "preset": "custom",
                                  "spectrum_A": -5e12, "spectrum_beta": 5e12},
             {"delta": 0.01, "sigma_divisor": 1.03, "spectrum_A": asp.CUSTOM_DEFAULT_A,
              "spectrum_beta": asp.CUSTOM_DEFAULT_BETA}),
            ("not custom", {"preset": "flux", "spectrum_A": 1e15, "spectrum_beta": -2e12},
             {"spectrum_A": asp.CUSTOM_DEFAULT_A, "spectrum_beta": asp.CUSTOM_DEFAULT_BETA}),
            ("delta_optimal divisor", {"threshold": "delta_optimal", "sigma_divisor": 500.0}, {"sigma_divisor": 1.03}),
        )
        for label, values, sent in unused:
            with self.subTest(case=label):
                graph = compile_graph(payload(speed_block={"args": [{"enabled": True, **values}]}))
                (_sid, node), = speed_nodes_of(graph)
                for name, value in sent.items():
                    self.assertEqual(node["inputs"][name], value, name)
                for name in compiler_module._SPEED_NODE_RANGES:
                    self.assertGreaterEqual(node["inputs"][name], schema[name][1]["min"], name)
                    self.assertLessEqual(node["inputs"][name], schema[name][1]["max"], name)
        used = (("sigma_divisor", {"sigma_divisor": 150.0}),
                ("spectrum_A", {"preset": "custom", "spectrum_A": 1e13}),
                ("spectrum_beta", {"preset": "custom", "spectrum_beta": -2e12}),
                ("seed", {"seed": 2 ** 70}))
        for name, values in used:
            with self.subTest(used=name), self.assertRaisesRegex(
                    WorkflowCompileError, f"Anima SPEED 값 {name}=.*ComfyUI 노드가 받는 범위"):
                compile_graph(payload(speed_block={"args": [{"enabled": True, **values}]}))

    def test_settings_sam_extra_refuses_are_left_out_with_a_warning(self):
        refused = (
            ({"scales": "0.5,0.9"}, None),
            ({"mode": "zoom"}, None),
            ({"scales": "1/0, 1"}, None),
            ({"threshold": "manual", "manual_sigmas": "0.2,0.6"}, None),
            ({"preset": "custom", "spectrum_A": 0.0}, asp.ERR_SPECTRUM_A),
            ({"threshold": "delta_optimal", "sigma_divisor": -1.0}, asp.ERR_DIVISOR),
            ({"threshold": "manual", "scales": "0.25,0.5,1.0", "manual_sigmas": "0.7"}, asp.ERR_MANUAL_COUNT),
        )
        for values, code in refused:
            with self.subTest(values=values):
                script = asp.coerce([{"enabled": True, **values}])
                warnings: list = []
                graph = compile_graph(payload(speed_block={"args": [{"enabled": True, **values}]}), warnings=warnings)
                self.assertEqual(speed_nodes_of(graph), [])
                detail = script.error if code is None else asp.VALIDATION_MESSAGES[code]
                self.assertTrue(detail)
                self.assertEqual(warnings, [{"code": notices.CODE_SPEED_INVALID, "feature": "speed", "cause": None,
                                             "reason": f"Anima SPEED 설정 오류로 적용하지 않았습니다 — {detail}"}])
                self.assertEqual(dumps(graph), dumps(compile_graph(payload())))
        # depends on which transitions a pass reaches: the node decides at sampling (sam-extra's runtime refusal)
        for values in ({"transform": "dwt", "scales": "0.3,1.0"},
                       {"threshold": "manual", "mode": "respace", "scales": "0.25,0.5,1.0", "manual_sigmas": "0.7"}):
            with self.subTest(kept=values):
                self.assertEqual(len(speed_nodes_of(compile_graph(payload(speed_block={"args": [
                    {"enabled": True, **values}]})))), 1)

    def test_static_skips(self):
        cases = (
            ({"spectrum_enabled": True}, asp.SKIP_COMFY_SPECTRUM),
            ({"spectrum_enabled": "yes"}, asp.SKIP_COMFY_SPECTRUM),
            ({"scripts": {"ControlNet": {"args": [{"enabled": True, "module": "canny"}]}}}, asp.SKIP_CONTROLNET),
            ({"use_image_alpha_as_mask": True}, asp.SKIP_INPAINT),
        )
        for extra, reason in cases:
            with self.subTest(extra=extra):
                warnings: list = []
                graph = compile_graph(payload(speed_block=block(), **extra), warnings=warnings)
                self.assertEqual(speed_nodes_of(graph), [])
                self.assertEqual(warnings, [{"code": notices.CODE_SPEED_SKIPPED, "feature": "speed", "cause": None,
                                             "reason": asp.SKIP_MESSAGES[reason]}])
        warnings = []
        compile_graph(payload(speed_block=block(asp.SpeedSettings(enabled=True, scales="1.0"))), warnings=warnings)
        self.assertEqual([w["reason"] for w in warnings], [asp.SKIP_MESSAGES[asp.SKIP_NO_COARSE_STAGE]])
        for extra in ({"spectrum_enabled": False}, {"spectrum_enabled": "off"},
                      {"scripts": {"ControlNet": {"args": [{"enabled": False}]}}}):
            with self.subTest(kept=extra):
                self.assertEqual(len(speed_nodes_of(compile_graph(payload(speed_block=block(), **extra)))), 1)

    def test_the_spectrum_rule_is_the_samplers_own(self):
        for value in (True, False, None, 1, 0, 0.0, 2.5, "true", "yes", "On", "1", "0", "no", "off", "none", "",
                      " FALSE ", "enabled"):
            with self.subTest(value=value):
                body = payload(speed_block=block(), spectrum_enabled=value)
                if value is None:
                    body.pop("spectrum_enabled")
                graph = compile_graph(body)
                _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                self.assertEqual(skips_anima_speed(body), sampler["inputs"]["spectrum_enabled"])
                self.assertEqual(bool(speed_nodes_of(graph)), not sampler["inputs"]["spectrum_enabled"])

    def test_legacy_speed_keys_are_still_refused_with_spectrum(self):
        with self.assertRaisesRegex(WorkflowCompileError, "옛 SPEED 입력\\(speed_enabled"):
            compile_graph(payload(spectrum_enabled=True, speed_enabled=True))
        compile_graph(payload(spectrum_enabled=True, speed_block=block()))      # the sam-extra block: a warning

    def test_a_stale_pack_names_the_feature(self):
        with self.assertRaisesRegex(WorkflowCompileError, "필요한 노드가 없습니다: ForgeNeoAnimaSpeed\\..*Anima SPEED에는 "
                                    "AI Studio Forge Parity 노드 팩 1\\.7\\.0 이상이 필요합니다"):
            compile_graph(payload(speed_block=block()), capabilities=caps(speed_node=False))
        compile_graph(payload(), capabilities=caps(speed_node=False))

    def test_graph_invariant_main_passes_only(self):
        count = 0
        for enable_hr, hires_flag, dd, extra, custom in itertools.product(
                (False, True), (False, True), ({}, DETAIL_DAEMON, DETAIL_DAEMON_HIRES),
                ({}, {"adetailer": {}}, {"sam3": {}}, {"adetailer": {"ad_prompt": "<lora:styles/ink:1>"}}),
                (False, True)):
            body = payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=hires_flag)), enable_hr=enable_hr,
                           scripts=dd, **extra)
            graph = compile_graph(body, workflow=_custom_workflow() if custom else None)
            with self.subTest(hr=enable_hr, hires=hires_flag, dd=dd, extra=extra, custom=custom):
                nodes = speed_nodes_of(graph)
                self.assertEqual(len(nodes), 1 + (enable_hr and hires_flag))
                for speed_id, _speed in nodes:
                    (reader,) = consumers(graph, speed_id)
                    self.assertIn(reader, (("ForgeNeoKSamplerCNS", "model"), ("KSampler", "model"),
                                           ("ForgeNeoHiresFix", "model")))
                for node in graph.values():
                    if node["class_type"] in ("ForgeNeoADetailer", "ForgeNeoSAM3Detailer"):
                        self.assertNotIn(NODE, model_chain(graph, node["inputs"]["model"]))
            count += 1
        self.assertEqual(count, 2 * 2 * 3 * 4 * 2)


class SpeedNodeContractTests(unittest.TestCase):
    def test_compiler_constants_match_the_pack_node(self):
        schema = speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES()["required"]
        self.assertEqual(tuple(schema), asp.COMFY_INPUTS)
        self.assertEqual({name: (schema[name][1]["min"], schema[name][1]["max"])
                          for name in compiler_module._SPEED_NODE_RANGES}, dict(compiler_module._SPEED_NODE_RANGES))
        self.assertEqual(tuple(schema["pass_label"][0]), asp.COMFY_PASS_LABELS)
        self.assertEqual(tuple(schema["mode"][0]), asp.MODES)
        self.assertEqual(tuple(schema["threshold"][0]), asp.THRESHOLDS)
        self.assertEqual(tuple(schema["preset"][0]), asp.PRESET_NAMES)
        self.assertEqual(tuple(schema["transform"][0]), asp.TRANSFORMS)
        self.assertEqual(tuple(speed_schedule.PRESET_NAMES), asp.PRESET_NAMES)
        for name in asp.ARG_NAMES:
            if name in schema:
                self.assertEqual(schema[name][1]["default"], asp.DEFAULTS[name], name)
        self.assertIs(schema["img2img_rescale"][1]["default"], asp.COMFY_IMG2IMG_RESCALE)


def _summary(graph: dict):
    return parse_comfy_metadata(graph)["parameters"].get(asp.INFOTEXT_KEY)


class SpeedMetadataTests(unittest.TestCase):
    """C4b: a ComfyUI PNG's PNG Info carries the 'Anima SPEED' summary Forge writes (not the run-time status)."""

    CARDS = (
        SPEED,
        asp.SpeedSettings(enabled=True, hires=True),
        asp.SpeedSettings(enabled=True, mode="respace", threshold="delta_optimal", preset="custom",
                          scales="1/4,1/2,1", delta=0.02, adaptive=False, transform="fft", spectrum_A=150.5,
                          spectrum_beta=2.25, seed=1234),
        asp.SpeedSettings(enabled=True, threshold="manual", manual_sigmas="0.8; 0.6", scales="0.25,0.5,1.0",
                          hires=True),
        asp.SpeedSettings(enabled=True, preset="flux", sigma_divisor=1.5, delta=0.005),
    )

    def test_the_summary_equals_forges(self):
        for card in self.CARDS:
            with self.subTest(card=card):
                params = parse_comfy_metadata(compile_graph(payload(speed_block=block(card), enable_hr=True)))[
                    "parameters"]
                self.assertEqual(params[asp.INFOTEXT_KEY], asp.summary(card))
                self.assertNotIn(asp.STATUS_KEY, params)
                self.assertNotIn(asp.INFOTEXT_IMG2IMG_RESCALE, params)
                restored = asp.from_infotext(params)
                self.assertEqual(asp.script_settings(restored), asp.script_settings(card))
        img2img = compile_graph(payload(speed_block=block(self.CARDS[2])), mode="img2img")
        self.assertEqual(_summary(img2img), asp.summary(self.CARDS[2]))

    def test_hires_reads_the_hires_pass_node(self):
        card = asp.SpeedSettings(enabled=True, hires=True)
        # without Hires the graph has no hires pass: hires=False (Forge writes the block's flag - a registry gap)
        self.assertEqual(_summary(compile_graph(payload(speed_block=block(card)))),
                         asp.summary(replace(card, hires=False)))
        self.assertEqual(_summary(compile_graph(payload(speed_block=block(card), enable_hr=True))), asp.summary(card))

    def test_no_node_no_summary(self):
        for body in (payload(), payload(speed_block=block(asp.SpeedSettings())),
                     payload(speed_block=block(), spectrum_enabled=True),
                     payload(speed_block={"args": [{"enabled": True, "scales": "0.5,0.9"}]})):
            with self.subTest(body=body):
                self.assertIsNone(_summary(compile_graph(body)))

    def test_hand_made_graphs(self):
        graph = compile_graph(payload(speed_block=block(), enable_hr=True))
        (speed_id, _speed), = speed_nodes_of(graph)
        linked = copy.deepcopy(graph)
        linked["90"] = {"class_type": "PrimitiveString", "inputs": {"value": "0.25,1.0"}}
        linked[speed_id]["inputs"]["scales"] = ["90", 0]
        self.assertEqual(_summary(linked), asp.summary(replace(SPEED, scales="0.25,1.0")))
        linked["91"] = {"class_type": "PrimitiveNode", "inputs": {"value": 0.5}}
        linked[speed_id]["inputs"]["delta"] = ["91", 0]
        result = parse_comfy_metadata(linked)
        self.assertNotIn(asp.INFOTEXT_KEY, result["parameters"])
        self.assertIn("Anima SPEED 값이 연결로 들어와 그 값을 기록하지 않았습니다.", result["warnings"])
        off = copy.deepcopy(graph)
        off[speed_id]["inputs"]["enabled"] = False
        self.assertIsNone(_summary(off))
        two = compile_graph(payload(speed_block=block(asp.SpeedSettings(enabled=True, hires=True)), enable_hr=True))
        (_base_id, _b), (hires_id, _h) = speed_nodes_of(two)
        self.assertEqual(_summary(two), asp.summary(asp.SpeedSettings(enabled=True, hires=True)))
        two[hires_id]["inputs"]["delta"] = 0.02
        result = parse_comfy_metadata(two)
        self.assertNotIn(asp.INFOTEXT_KEY, result["parameters"])
        self.assertIn("Anima SPEED 노드가 여럿이고 값이 달라 Anima SPEED 값을 고르지 않았습니다.", result["warnings"])

    def test_unread_out_of_range_values_show_what_the_node_got(self):
        raw = {"enabled": True, "threshold": "manual", "delta": 5.0}
        summary = _summary(compile_graph(payload(speed_block={"args": [raw]})))
        self.assertEqual(summary, asp.coerce([{**raw, "delta": 0.01}]).summary())
        self.assertNotEqual(summary, asp.coerce([raw]).summary())      # Forge writes delta=5 (unused by its plan)

    def test_web_workflow_format(self):
        card = asp.SpeedSettings(enabled=True, mode="respace", preset="custom", spectrum_A=99.5, seed=42)
        node = speed_nodes_of(compile_graph(payload(speed_block=block(card))))[0][1]["inputs"]
        widgets = []
        for name in asp.COMFY_INPUTS[1:]:
            widgets.append(node[name])
            if name == "seed":
                widgets.append("fixed")
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["checkpoint.safetensors"], "inputs": []},
                {"id": 2, "type": NODE, "widgets_values": widgets, "inputs": [{"name": "model", "link": 1}]},
                {"id": 5, "type": "KSampler", "widgets_values": [7, "fixed", 20, 5.0, "euler", "normal", 1.0],
                 "inputs": [{"name": "model", "link": 2}]},
                {"id": 8, "type": "SaveImage", "widgets_values": ["out"], "inputs": [{"name": "images", "link": 3}]},
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 5, 0, "MODEL"], [3, 5, 0, 8, 0, "IMAGE"]],
        }
        self.assertEqual(parse_comfy_metadata(None, workflow)["parameters"][asp.INFOTEXT_KEY], asp.summary(card))


class SpeedPreflightTests(unittest.TestCase):
    def _row(self, result: dict) -> dict:
        return next(row for row in result["features"] if row["id"] == "speed")

    def test_rows(self):
        compiler = ComfyWorkflowCompiler(caps())
        ready = feature_preflight(compiler, "checkpoint.safetensors", payload(speed_block=block()))
        self.assertTrue(ready["ok"], ready)
        self.assertEqual((self._row(ready)["label"], self._row(ready)["state"]), ("Anima SPEED", "ready"))
        for body, reason in ((payload(speed_block=block(), spectrum_enabled=True),
                              asp.SKIP_MESSAGES[asp.SKIP_COMFY_SPECTRUM]),
                             (payload(speed_block={"args": [{"enabled": True, "preset": "custom", "spectrum_A": -1}]}),
                              "Anima SPEED 설정 오류로 적용하지 않았습니다 — "
                              + asp.VALIDATION_MESSAGES[asp.ERR_SPECTRUM_A])):
            with self.subTest(reason=reason):
                result = feature_preflight(compiler, "checkpoint.safetensors", body)
                self.assertTrue(result["ok"], result)
                self.assertEqual((self._row(result)["state"], self._row(result)["reason"]), ("skipped", reason))
        for body in (payload(), payload(speed_block=block(asp.SpeedSettings()))):
            result = feature_preflight(compiler, "checkpoint.safetensors", body)
            self.assertEqual((self._row(result)["state"], self._row(result)["requested"]), ("off", False))
        stale = feature_preflight(ComfyWorkflowCompiler(caps(speed_node=False)), "checkpoint.safetensors",
                                  payload(speed_block=block()))
        self.assertFalse(stale["ok"])
        self.assertEqual(self._row(stale)["state"], "blocked")
        self.assertIn("Anima SPEED에는 AI Studio Forge Parity 노드 팩 1.7.0", stale["error"])


class SpeedRecipeTests(unittest.TestCase):
    def test_the_recipe_lists_the_pack_node_inputs(self):
        recipe = next(item for item in RECIPES if item["id"] == "anima-speed")
        self.assertEqual(recipe["title"], asp.SCRIPT_NAME)
        fields = recipe["nodes"][NODE]
        schema = speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES()["required"]
        self.assertEqual(tuple(fields), asp.COMFY_INPUTS)
        for name, kind in fields.items():
            spec = schema[name][0]
            self.assertEqual("CHOICE" if isinstance(spec, list) else spec, kind, name)
        self.assertIn(asp.COMFY_MIN_PACK_VERSION, recipe["note"])
        self.assertEqual(recipe["models"], [])
        results = {item["id"]: item for item in check_recipes({NODE: {"input": {"required": schema}}})}
        self.assertEqual(results["anima-speed"]["status"], "available")
        self.assertEqual({item["id"]: item for item in check_recipes({})}["anima-speed"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
