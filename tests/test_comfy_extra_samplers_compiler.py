"""Extra Samplers (sam-extra v0.33.0) on ComfyUI — the compiler side (plan C3, PLAN_amend C3).

The bundled pack (1.7.0) registers twelve of sam-extra's eighteen samplers in ComfyUI under their Forge aliases (the
other six are ComfyUI core's own) and ``ForgeNeoExtraSamplerSettings`` (MODEL -> MODEL) carries the block's five ER SDE
values (``max_stage``, ``eta``, ``noise_window``, ``noise_start``, ``noise_end``) in the MODEL's ``model_options`` so
every later sampler of that MODEL reads them.  The compiler adds the node only when a value differs from the defaults
(plan R5/X5, ComfyUI rule — every other graph stays byte-identical), once on each model stack the passes share, right
before the guidance stack (default graph, each ADetailer/SAM3 LoRA branch, a custom workflow).  ``validate`` checks
that every pack sampler name a node runs is the pack's own; informational compile warnings mirror Forge's Spectrum +
Dy rule (plan R3) and its CFG++ "CFG 1~2" warning; PNG metadata gets Forge's ER SDE infotext keys.

**How the payloads are built.**  Since plan task C6 the app's ``core/extra_samplers.plan()`` sends the block on ComfyUI
always (``REASON_COMFY`` — tests/test_extra_samplers_settings.py checks the contributor), so these tests build the
payload the same way: ``alwayson_scripts["Extra Samplers"] = core/extra_samplers.as_block(settings)`` (five positional
values — the block ``plan()`` also returns for Forge); older two-value blocks come from presets and history.

No ComfyUI is contacted: the pack node's schema is its real ``INPUT_TYPES`` (``owned_names`` patched, see
``tests/test_comfy_sampler_aliases._owner_caps``).
"""
from __future__ import annotations

import ast
import copy
import itertools
import json
import unittest
from pathlib import Path

from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes, extra_samplers_register
from core import anima_guidance, sam3_args
from core import extra_samplers as xsa
from core import comfy_workflow_compiler as compiler_module
from core.comfy_metadata import parse_comfy_metadata
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import feature_preflight
from tests._sam_extra_ext import EXT_ROOT, requires_extension
from tests.test_comfy_sampler_aliases import _owner_caps
from tests.test_comfy_workflow_compiler import _custom_workflow, _node

NODE = xsa.COMFY_SETTINGS_NODE
NON_DEFAULT = xsa.ExtraSamplersSettings(max_stage=2, eta=0.5, noise_window=True, noise_start=0.3, noise_end=0.7)
REVERSE, ODE, TUNABLE = (xsa.COMFY_NAMES[label] for label in xsa.ER_SDE_LABELS)
SKIMMED = {anima_guidance.SCRIPT_SKIMMED_CFG: {"args": {"skim_enabled": True}}}
DETAIL_DAEMON = {anima_guidance.SCRIPT_DETAIL_DAEMON: {"args": {"dd_enabled": True, "dd_amount": 0.3}}}


def caps(*, owned=None) -> dict:
    """Capabilities with all eighteen names in KSampler's list; the pack owns its twelve (or ``owned``)."""
    result = _owner_caps(samplers=(*xsa.PACK_SAMPLERS, *xsa.CORE_SAMPLERS), owned_samplers=owned)
    result["KSamplerSelect"] = {"input": {"required": {"sampler_name": result["KSampler"]["input"]["required"]
                                                       ["sampler_name"]}}}
    result["DiTSpectrumPatch"] = {"input": {"required": {}}}
    return result


def block(settings: xsa.ExtraSamplersSettings = xsa.APP_DEFAULTS) -> dict:
    """What ``core/extra_samplers.plan()`` sends (C6: also on ComfyUI)."""
    return xsa.as_block(settings)


def payload(*, sampler="euler", xsa_block=None, adetailer=None, sam3=None, scripts=None, **extra) -> dict:
    found: dict = dict(scripts or {})
    if adetailer is not None:
        found["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face.pt", **adetailer}]}
    if sam3 is not None:
        found.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    if xsa_block is not None:
        found[xsa.SCRIPT_NAME] = xsa_block
    return {"prompt": "portrait", "negative_prompt": "bad", "seed": 7, "steps": 28, "sampler_name": sampler,
            "alwayson_scripts": found, **extra}


def compile_t2i(body, *, workflow=None, capabilities=None, warnings=None) -> dict:
    return ComfyWorkflowCompiler(capabilities or caps()).compile(
        "txt2img", "checkpoint.safetensors", body, workflow=workflow, warnings=warnings)


def carriers(graph: dict) -> list:
    return [(node_id, node) for node_id, node in graph.items() if node.get("class_type") == NODE]


def model_chain(graph: dict, link) -> list:
    """(id, class) up the ``model`` inputs from ``link`` (nearest first)."""
    chain, seen = [], set()
    while isinstance(link, list) and len(link) >= 2 and str(link[0]) in graph and str(link[0]) not in seen:
        seen.add(str(link[0]))
        node = graph[str(link[0])]
        chain.append((str(link[0]), node["class_type"]))
        link = node.get("inputs", {}).get("model")
    return chain


def classes(chain: list) -> list:
    return [class_type for _node_id, class_type in chain]


def nearest_values(graph: dict, link):
    """The Extra Samplers values the sampler on ``link`` reads (nearest node up the chain), None without one."""
    for node_id, class_type in model_chain(graph, link):
        if class_type == NODE:
            inputs = graph[node_id]["inputs"]
            return xsa.coerce_args([inputs[name] for name in xsa.ARG_NAMES])
    return None


def sampling_nodes(graph: dict) -> list:
    """(node id, node, sampler name) for every node that samples — ADetailer's sampler is in its settings JSON."""
    found = []
    for node_id, node in graph.items():
        inputs = node.get("inputs", {})
        if node.get("class_type") == "ForgeNeoADetailer":
            found.append((node_id, node, json.loads(inputs["settings_json"]).get("sampler_name")))
        elif isinstance(inputs.get("sampler_name"), str) and "model" in inputs:
            found.append((node_id, node, inputs["sampler_name"]))
    return found


def dumps(graph: dict) -> str:
    return json.dumps(graph, sort_keys=True)


class ExtraSamplersInsertionTests(unittest.TestCase):
    """Where and when ``ForgeNeoExtraSamplerSettings`` goes in, and what it receives."""

    VARIANTS = {
        "plain": {},
        "ER SDE main": {"sampler": "ER SDE (Reverse-time)"},
        "hires Dy": {"enable_hr": True, "hr_sampler_name": "Euler SMEA Dy CFG++"},
        "adetailer": {"adetailer": {"ad_use_sampler": True, "ad_sampler": "ER SDE (Tunable)"}},
        "adetailer own LoRA": {"adetailer": {"ad_prompt": "face <lora:characters/alice:0.7>"}},
        "sam3": {"sam3": {"sam3_use_sampler": True, "sam3_sampler": "er_sde_ode"}},
        "guidance": {"scripts": {**SKIMMED, **DETAIL_DAEMON}},
    }

    def test_default_or_absent_values_keep_the_graph_byte_identical(self):
        default_blocks = (block(), {"args": [3, 1.0]}, {"args": []}, {}, {"args": [3, 1.0, False, 0.2, 0.8, "x"]},
                          {"args": ["3", "1", "0", "0.2", "0.8"]}, {"args": [3.4, True, "off"]})
        for label, extra in self.VARIANTS.items():
            without = compile_t2i(payload(**extra))
            self.assertEqual(carriers(without), [])
            for raw in default_blocks:
                with self.subTest(variant=label, block=raw):
                    self.assertTrue(xsa.is_default(xsa.parse_script_block(raw)))
                    self.assertEqual(dumps(compile_t2i(payload(xsa_block=raw, **extra))), dumps(without))
        workflow = _custom_workflow()
        without = compile_t2i(payload(sampler="er_sde_tunable"), workflow=workflow)
        self.assertEqual(dumps(compile_t2i(payload(sampler="er_sde_tunable", xsa_block=block()), workflow=workflow)),
                         dumps(without))

    def test_values_other_than_the_defaults_insert_one_node_right_before_the_guidance_stack(self):
        schema = extra_samplers_nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES()
        for settings in (NON_DEFAULT, xsa.ExtraSamplersSettings(max_stage=1), xsa.ExtraSamplersSettings(eta=0.0),
                         xsa.ExtraSamplersSettings(noise_window=True),
                         xsa.ExtraSamplersSettings(noise_start=0.25),            # window off: still a value
                         xsa.ExtraSamplersSettings(noise_end=0.9, eta=10.0)):
            for guidance in ({}, {**SKIMMED, **DETAIL_DAEMON}):
                with self.subTest(settings=settings, guidance=bool(guidance)):
                    graph = compile_t2i(payload(xsa_block=block(settings), scripts=guidance,
                                                prompt="portrait <lora:styles/ink:0.5>", distilled_cfg_scale=3.0))
                    (carrier_id, carrier), = carriers(graph)
                    inputs = carrier["inputs"]
                    self.assertEqual(tuple(inputs), xsa.COMFY_INPUTS)
                    self.assertEqual({key: value for key, value in inputs.items() if key != "model"},
                                     dict(zip(xsa.ARG_NAMES, settings.as_args())))
                    self.assertEqual([type(inputs[name]) for name in xsa.ARG_NAMES], [int, float, bool, float, float])
                    self.assertEqual(set(inputs), set(schema["required"]))
                    self.assertNotIn(xsa.COMFY_REPORT_INPUT, inputs)
                    self.assertEqual(carrier["_meta"]["title"], "Extra Samplers values (sam-extra)")
                    _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                    chain = classes(model_chain(graph, sampler["inputs"]["model"]))
                    index = chain.index(NODE)
                    expected_above = ["ForgeNeoAnimaDetailDaemon", "ForgeNeoSkimmedCFG"] if guidance else []
                    self.assertEqual(chain[:index], expected_above)
                    self.assertEqual(chain[index + 1:], ["LoraLoader", "ForgeNeoModelSamplingShift",
                                                         "CheckpointLoaderSimple"])

    def test_values_are_read_like_the_extension_from_blocks_of_every_shape(self):
        schema = extra_samplers_nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES()["required"]
        cases = (
            ("old two values", [2, 0.5]),
            ("five values", [1, 0.25, True, 0.1, 0.9]),
            ("empty window", [3, 1.0, True, 0.8, 0.2]),
            ("out of range", [9, 50, "on", -1, 2]),
            ("strings", ["2.6", "7.5", "yes", "0.45", "0.55"]),
            ("non-finite", ["nan", float("inf"), None, "x", 10 ** 400]),
            ("bool stage", [True, 1.0]),
        )
        for label, args in cases:
            expected = xsa.coerce_args(args)
            with self.subTest(block=label):
                graph = compile_t2i(payload(sampler="er_sde_tunable", xsa_block={"args": args}))
                if xsa.is_default(expected):
                    self.assertEqual(carriers(graph), [])
                    continue
                (_cid, carrier), = carriers(graph)
                values = [carrier["inputs"][name] for name in xsa.ARG_NAMES]
                self.assertEqual(values, expected.as_args())
                for name, value in zip(xsa.ARG_NAMES, values):
                    spec = schema[name]
                    if spec[0] in ("INT", "FLOAT"):
                        self.assertGreaterEqual(value, spec[1]["min"], name)
                        self.assertLessEqual(value, spec[1]["max"], name)

    def test_every_pass_reads_the_values(self):
        cases = {
            "main": (payload(sampler="ER SDE (Reverse-time)", xsa_block=block(NON_DEFAULT)), "ForgeNeoKSamplerCNS"),
            "hires": (payload(xsa_block=block(NON_DEFAULT), enable_hr=True, hr_sampler_name="er_sde_tunable",
                              hr_checkpoint_name="other.safetensors"), "ForgeNeoHiresFix"),
            "adetailer": (payload(xsa_block=block(NON_DEFAULT), adetailer={
                "ad_use_sampler": True, "ad_sampler": "er_sde_ode"}), "ForgeNeoADetailer"),
            "adetailer own LoRA branch": (payload(xsa_block=block(NON_DEFAULT), adetailer={
                "ad_use_sampler": True, "ad_sampler": "er_sde_ode", "ad_prompt": "face <lora:characters/alice:0.7>"}),
                "ForgeNeoADetailer"),
            "sam3": (payload(xsa_block=block(NON_DEFAULT), sam3={"sam3_use_sampler": True,
                                                                 "sam3_sampler": "ER SDE (Tunable)"}),
                     "ForgeNeoSAM3Detailer"),
            "sam3 own LoRA branch": (payload(xsa_block=block(NON_DEFAULT), sam3={
                "sam3_inpaint_prompt": "face <lora:characters/alice:0.7>"}, sampler="er_sde_reverse_time"),
                "ForgeNeoSAM3Detailer"),
        }
        for label, (body, class_type) in cases.items():
            with self.subTest(case=label):
                graph = compile_t2i(body)
                _nid, node = _node(graph, class_type)
                self.assertEqual(nearest_values(graph, node["inputs"]["model"]), NON_DEFAULT)
                branch = "own LoRA" in label
                titles = [carrier["_meta"]["title"] for _cid, carrier in carriers(graph)]
                self.assertEqual(len(titles), 2 if branch else 1, titles)
                if branch:
                    self.assertRegex(titles[1], r"\((ADetailer|SAM3) LoRA branch\)$")

    def test_custom_workflow_and_standalone_postprocess(self):
        workflow = _custom_workflow()
        graph = compile_t2i(payload(sampler="er_sde_tunable", xsa_block=block(NON_DEFAULT), scripts=SKIMMED),
                            workflow=workflow)
        (carrier_id, _carrier), = carriers(graph)
        chain = classes(model_chain(graph, graph["5"]["inputs"]["model"]))
        self.assertEqual(chain[:2], ["ForgeNeoSkimmedCFG", NODE])
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], TUNABLE)
        compiler = ComfyWorkflowCompiler(caps())
        for extra in ({"adetailer": {"ad_use_sampler": True, "ad_sampler": "er_sde_ode"}},
                      {"sam3": {"sam3_use_sampler": True, "sam3_sampler": "er_sde_reverse_time"}}):
            with self.subTest(postprocess=extra):
                body = payload(xsa_block=block(NON_DEFAULT), **extra)
                post = compiler.compile_postprocess("checkpoint.safetensors", body, uploaded_image="in.png")
                for _nid, node, _name in sampling_nodes(post):
                    self.assertEqual(nearest_values(post, node["inputs"]["model"]), NON_DEFAULT)
                plain = compiler.compile_postprocess("checkpoint.safetensors", payload(**extra),
                                                     uploaded_image="in.png")
                self.assertEqual(carriers(plain), [])

    def test_graph_invariant_every_sampling_pass_reaches_its_values(self):
        count = 0
        for main, hires, ad_branch, sam, values in itertools.product(
                ("euler", "er_sde_tunable"), (None, "er_sde_reverse_time"), (False, True),
                (None, "Use same sampler", "er_sde_ode"), (xsa.APP_DEFAULTS, NON_DEFAULT)):
            extra: dict = {}
            if hires:
                extra.update(enable_hr=True, hr_sampler_name=hires)
            ad = {"ad_use_sampler": True, "ad_sampler": "er_sde_ode",
                  **({"ad_prompt": "face <lora:characters/alice:0.7>"} if ad_branch else {})}
            body = payload(sampler=main, xsa_block=block(values), adetailer=ad, scripts=DETAIL_DAEMON,
                           **({"sam3": {"sam3_use_sampler": True, "sam3_sampler": sam}} if sam else {}), **extra)
            graph = compile_t2i(body)
            with self.subTest(main=main, hires=hires, branch=ad_branch, sam=sam, values=values):
                if xsa.is_default(values):
                    self.assertEqual(carriers(graph), [])
                for _nid, node, _name in sampling_nodes(graph):
                    expected = None if xsa.is_default(values) else values
                    self.assertEqual(nearest_values(graph, node["inputs"]["model"]), expected)
            count += 1
        self.assertEqual(count, 2 * 2 * 2 * 3 * 2)


class PackSamplerNameTests(unittest.TestCase):
    """``validate``: a pack sampler name any node runs must be the pack's own (plan R6/X12, PLAN_amend C3)."""

    def test_stale_pack_without_the_settings_node_refuses_the_values(self):
        stale = _owner_caps(sampler_node=False)
        with self.assertRaisesRegex(WorkflowCompileError, "필요한 노드가 없습니다: ForgeNeoExtraSamplerSettings\\..*"
                                    "Extra Samplers 값\\(ER SDE\\)에는 AI Studio Forge Parity 노드 팩 1\\.7\\.0 이상"):
            compile_t2i(payload(xsa_block=block(NON_DEFAULT)), capabilities=stale)
        compile_t2i(payload(xsa_block=block()), capabilities=stale)      # defaults: no node, nothing to refuse
        both = _owner_caps(sampler_node=False, scheduler_node=False)
        both.pop("ForgeNeoNegPip")
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoExtraSamplerSettings, ForgeNeoNegPip\\..* "
                                    "Extra Samplers 값\\(ER SDE\\)에는"):
            compile_t2i(payload(xsa_block=block(NON_DEFAULT), scripts={"NegPiP": {"args": [True]}}),
                        capabilities=both)

    def test_every_pack_name_is_checked_on_any_node_and_core_names_are_not(self):
        compiler = ComfyWorkflowCompiler(caps(owned=()))
        for name in xsa.PACK_SAMPLERS:
            for graph in ({"1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": name}}},
                          {"1": {"class_type": "ForgeNeoHiresFix", "inputs": {"sampler_name": "euler",
                                                                             "base_sampler_name": name}}},
                          {"1": {"class_type": "ForgeNeoADetailer",
                                 "inputs": {"settings_json": json.dumps({"sampler_name": name})}}}):
                with self.subTest(sampler=name, node=graph["1"]["class_type"]), self.assertRaisesRegex(
                        WorkflowCompileError, f"다른 노드 팩이 '{name}' 샘플러를 먼저 등록"):
                    compiler.validate(graph)
        for name in xsa.CORE_SAMPLERS:
            with self.subTest(core=name):
                compiler.validate({"1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": name}}})
        owned = ComfyWorkflowCompiler(caps())
        for name in xsa.PACK_SAMPLERS:
            owned.validate({"1": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": name}}})
        owned.validate({"1": {"class_type": "ForgeNeoADetailer", "inputs": {"settings_json": "not json"}}})

    def test_a_workflow_sampler_the_app_does_not_rewrite_is_checked(self):
        workflow = _custom_workflow()
        workflow["9"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_dy"}}
        with self.assertRaisesRegex(WorkflowCompileError, "다른 노드 팩이 'euler_dy' 샘플러를 먼저 등록"):
            compile_t2i(payload(), workflow=workflow, capabilities=caps(owned=("er_sde_ode",)))
        stale = caps()
        stale.pop(NODE)
        with self.assertRaisesRegex(WorkflowCompileError, "'euler_dy' 샘플러는 앱 노드 팩 1\\.7\\.0 의 것이 아닙니다"):
            compile_t2i(payload(), workflow=workflow, capabilities=stale)
        graph = compile_t2i(payload(), workflow=workflow)
        self.assertEqual(graph["9"]["inputs"]["sampler_name"], "euler_dy")

    def test_class_name_only_probes_and_no_capabilities_are_not_checked(self):
        probe = caps(owned=())
        probe[NODE] = {"input": {"required": {}}}
        ComfyWorkflowCompiler(probe).validate({"1": {"class_type": "KSamplerSelect",
                                                     "inputs": {"sampler_name": "restart_flow"}}})
        ComfyWorkflowCompiler().validate({"1": {"class_type": "KSamplerSelect",
                                                "inputs": {"sampler_name": "restart_flow"}}})


class ExtraSamplersWarningTests(unittest.TestCase):
    """Informational compile warnings (the graph is the same with or without the list)."""

    def _warnings(self, body, **kwargs) -> list:
        warnings: list = []
        graph = compile_t2i(body, warnings=warnings, **kwargs)
        self.assertEqual(dumps(graph), dumps(compile_t2i(body, **kwargs)))
        return warnings

    def test_spectrum_with_a_dy_sampler_warns_for_that_sampler(self):
        dy = [label for label in xsa.DY_KIND]
        for text in (*dy, *(xsa.COMFY_NAMES[label] for label in dy)):
            with self.subTest(sampler=text):
                warnings = self._warnings(payload(sampler=text, spectrum_enabled=True, cfg_scale=1.5))
                (warning,) = warnings
                self.assertEqual((warning["feature"], warning["code"], warning["cause"]),
                                 ("extra_samplers", "extra_samplers_substeps_skipped", None))
                self.assertIn("Euler (SMEA) Dy (CFG++)", warning["reason"])
                self.assertIn(f"'{xsa.comfy_name(text)}'", warning["reason"])
                self.assertIn("Spectrum", warning["reason"])
                self.assertIn("Forge-compatible sampler", warning["reason"])
        for body in (payload(sampler="Euler Dy"),                                       # no Spectrum
                     payload(sampler="er_sde_tunable", spectrum_enabled=True),
                     payload(sampler="euler", spectrum_enabled=True, enable_hr=True,    # Spectrum is on the base only
                             hr_sampler_name="euler_smea_dy_cfg_pp", hr_cfg=1.5)):
            with self.subTest(body=body):
                self.assertEqual(self._warnings(body), [])

    def test_cfg_pp_samplers_above_cfg_two_warn(self):
        names = [xsa.COMFY_NAMES[label] for label in xsa.CFG_PP_LABELS]
        self.assertEqual(sorted(names), ["cfgpp_ud10_ab", "euler_dy_cfg_pp", "euler_smea_dy_cfg_pp"])
        for name in names:
            with self.subTest(sampler=name):
                (warning,) = self._warnings(payload(sampler=name, cfg_scale=2.5))
                self.assertEqual((warning["feature"], warning["code"]), ("extra_samplers", "extra_samplers_cfg_pp"))
                self.assertIn(f"CFG++ 샘플러 '{name}' 는 CFG 1~2 를 권장합니다", warning["reason"])
                self.assertIn("CFG 가 2.5 입니다", warning["reason"])
                for cfg in (2.0, 1.0):
                    self.assertEqual(self._warnings(payload(sampler=name, cfg_scale=cfg)), [])
        hires = self._warnings(payload(cfg_scale=7.0, enable_hr=True, hr_sampler_name="CFG++ UD10 AB", hr_cfg=3.0))
        self.assertEqual([w["reason"] for w in hires],
                         ["CFG++ 샘플러 'cfgpp_ud10_ab' 는 CFG 1~2 를 권장합니다 — Forge-compatible Hires.fix(node "
                          + _node(compile_t2i(payload(cfg_scale=7.0, enable_hr=True, hr_sampler_name="CFG++ UD10 AB",
                                                      hr_cfg=3.0)), "ForgeNeoHiresFix")[0]
                          + ")의 CFG 가 3 입니다(Forge 도 같은 경고를 남깁니다)"])
        detailer = self._warnings(payload(cfg_scale=1.5, sampler="euler_dy_cfg_pp", adetailer={
            "ad_use_cfg_scale": True, "ad_cfg_scale": 4.0}))
        self.assertEqual(len(detailer), 1)
        self.assertIn("ADetailer slot 1", detailer[0]["reason"])
        self.assertEqual(self._warnings(payload(sampler="euler_dy", cfg_scale=7.0)), [])
        # a Hires node runs its own sampler; base_sampler_name (the main pass's) is only its fallback
        self.assertEqual(self._warnings(payload(sampler="cfgpp_ud10_ab", cfg_scale=1.5, enable_hr=True,
                                                hr_sampler_name="euler", hr_cfg=7.0)), [])

    def test_without_a_list_the_warnings_are_logged(self):
        with self.assertLogs(compiler_module._logger, "WARNING") as logged:
            compile_t2i(payload(sampler="Euler Dy CFG++", cfg_scale=5.0, spectrum_enabled=True))
        self.assertEqual(len(logged.output), 2)
        compiler = ComfyWorkflowCompiler(caps())
        with self.assertLogs(compiler_module._logger, "WARNING") as logged:
            compiler.compile_postprocess("checkpoint.safetensors", payload(adetailer={
                "ad_use_sampler": True, "ad_sampler": "cfgpp_ud10_ab", "ad_use_cfg_scale": True,
                "ad_cfg_scale": 6.0}), uploaded_image="in.png")
        self.assertIn("CFG++ 샘플러 'cfgpp_ud10_ab'", logged.output[0])

    def test_names_match_the_pack_and_the_extension(self):
        self.assertEqual(compiler_module._DY_SAMPLER_NAMES, frozenset(extra_samplers_register.DY_SAMPLERS))
        self.assertEqual(compiler_module._CFG_PP_SAMPLER_NAMES - {"cfgpp_ud10_ab"},
                         {name for name, cfg_pp in extra_samplers_register.DY_SAMPLERS.items() if cfg_pp})
        self.assertEqual(compiler_module._PACK_SAMPLER_NAMES, frozenset(extra_samplers_register.PACK_SAMPLERS))


@requires_extension
class ForgeCfgPpWarningTests(unittest.TestCase):
    def test_the_threshold_is_forges(self):
        """sam3ext/extra_samplers/registry._warn_cfg warns when ``cfg > 2.0`` — the compiler's threshold."""
        tree = ast.parse((Path(EXT_ROOT) / "sam3ext/extra_samplers/registry.py").read_text(encoding="utf-8"))
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "_warn_cfg")
        compares = [node for node in ast.walk(function) if isinstance(node, ast.Compare)
                    and isinstance(node.ops[0], ast.Gt) and isinstance(node.comparators[0], ast.Constant)]
        self.assertEqual([node.comparators[0].value for node in compares], [compiler_module._CFG_PP_MAX_CFG])


def _metadata(graph: dict) -> dict:
    return parse_comfy_metadata(graph)


def _hand_graph(*chain: str, sampler: str = REVERSE, settings=NON_DEFAULT) -> dict:
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "checkpoint.safetensors"}}}
    link = ["1", 0]
    for index, class_type in enumerate(chain, start=2):
        inputs = dict(zip(xsa.ARG_NAMES, settings.as_args())) if class_type == NODE else {}
        inputs["model"] = link
        graph[str(index)] = {"class_type": class_type, "inputs": inputs}
        link = [str(index), 0]
    graph["99"] = {"class_type": "KSampler", "inputs": {"model": link, "sampler_name": sampler}}
    graph["98"] = {"class_type": "SaveImage", "inputs": {"images": ["99", 0]}}
    return graph


def _keys(params: dict) -> dict:
    return {key: params[key] for key in (xsa.KEY_MAX_STAGE, xsa.KEY_ETA, xsa.KEY_NOISE_WINDOW) if key in params}


class ExtraSamplersMetadataTests(unittest.TestCase):
    """PNG Info of a ComfyUI image gets the ER SDE keys Forge writes for its main/Hires pass."""

    SETTINGS = (NON_DEFAULT, xsa.ExtraSamplersSettings(max_stage=1),
                xsa.ExtraSamplersSettings(eta=0.123456789, noise_window=True),
                xsa.ExtraSamplersSettings(eta=0.0, noise_window=True),
                xsa.ExtraSamplersSettings(noise_window=True, noise_start=0.8, noise_end=0.2),
                xsa.ExtraSamplersSettings(noise_start=0.3))

    def test_main_pass_keys_equal_forge_infotext_items(self):
        for settings in self.SETTINGS:
            for label in xsa.ER_SDE_LABELS:
                with self.subTest(settings=settings, sampler=label):
                    params = _metadata(compile_t2i(payload(sampler=label, xsa_block=block(settings))))["parameters"]
                    expected = xsa.infotext_items(settings, [label])
                    self.assertEqual(_keys(params), expected)
                    self.assertEqual(params["Sampler"], xsa.COMFY_NAMES[label])
        tunable = _keys(_metadata(compile_t2i(payload(sampler="er_sde_tunable", xsa_block=block(NON_DEFAULT))))
                        ["parameters"])
        self.assertEqual(tunable, {xsa.KEY_MAX_STAGE: 2, xsa.KEY_ETA: 0.5, xsa.KEY_NOISE_WINDOW: "0.3-0.7"})
        ode = _keys(_metadata(compile_t2i(payload(sampler="er_sde_ode", xsa_block=block(NON_DEFAULT))))["parameters"])
        self.assertEqual(ode, {xsa.KEY_MAX_STAGE: 2})

    def test_hires_pass_counts_detailer_passes_do_not(self):
        params = _metadata(compile_t2i(payload(xsa_block=block(NON_DEFAULT), enable_hr=True,
                                               hr_sampler_name="ER SDE (Tunable)")))["parameters"]
        self.assertEqual(_keys(params), xsa.infotext_items(NON_DEFAULT, [TUNABLE]))
        for extra in ({"adetailer": {"ad_use_sampler": True, "ad_sampler": "er_sde_reverse_time"}},
                      {"sam3": {"sam3_use_sampler": True, "sam3_sampler": "er_sde_tunable"}}):
            with self.subTest(extra=extra):
                params = _metadata(compile_t2i(payload(xsa_block=block(NON_DEFAULT), **extra)))["parameters"]
                self.assertEqual(_keys(params), {})

    def test_no_node_or_another_sampler_writes_nothing(self):
        for body in (payload(sampler=REVERSE, xsa_block=block()), payload(sampler=REVERSE),
                     payload(sampler="dpmpp_4m_sde", xsa_block=block(NON_DEFAULT)),
                     payload(sampler="euler", xsa_block=block(NON_DEFAULT))):
            with self.subTest(body=body):
                self.assertEqual(_keys(_metadata(compile_t2i(body))["parameters"]), {})

    def test_paste_restores_what_the_sampler_reads(self):
        for settings in self.SETTINGS:
            for name in (REVERSE, TUNABLE):
                with self.subTest(settings=settings, sampler=name):
                    params = _metadata(compile_t2i(payload(sampler=name, xsa_block=block(settings))))["parameters"]
                    restored = xsa.from_infotext({key: str(value) for key, value in params.items()})
                    # Forge rounds the written eta to 6 decimals (params.record_infotext) — so does the key here
                    self.assertEqual((restored.max_stage, restored.eta), (settings.max_stage, round(settings.eta, 6)))
                    if settings.noise_window and settings.eta > 0:
                        self.assertEqual((restored.noise_window, restored.noise_start, restored.noise_end),
                                         (True, settings.noise_start, settings.noise_end))

    def test_nearest_node_linked_values_and_conflicts(self):
        nearest = _hand_graph(NODE, "LoraLoader", NODE)
        nearest["2"]["inputs"].update(max_stage=1, eta=4.0)
        self.assertEqual(_keys(_metadata(nearest)["parameters"]), xsa.infotext_items(NON_DEFAULT, [REVERSE]))
        linked = _hand_graph(NODE)
        linked["50"] = {"class_type": "PrimitiveNode", "inputs": {"value": 0.5}}
        linked["2"]["inputs"]["eta"] = ["50", 0]
        result = _metadata(linked)
        self.assertEqual(_keys(result["parameters"]), {})
        self.assertIn("Extra Samplers 값이 연결로 들어와 그 값을 기록하지 않았습니다.", result["warnings"])
        two = _hand_graph(NODE)
        two["3"] = {"class_type": NODE, "inputs": {"model": ["1", 0], **dict(zip(xsa.ARG_NAMES, [1, 2.0, False,
                                                                                               0.2, 0.8]))}}
        two["97"] = {"class_type": "KSampler", "inputs": {"model": ["3", 0], "sampler_name": TUNABLE}}
        two["96"] = {"class_type": "SaveImage", "inputs": {"images": ["97", 0]}}
        result = _metadata(two)
        self.assertEqual(_keys(result["parameters"]), {})
        self.assertIn("Extra Samplers 값 노드가 여럿이고 값이 달라 ER SDE 값을 고르지 않았습니다.", result["warnings"])
        same = copy.deepcopy(two)
        same["3"]["inputs"].update(dict(zip(xsa.ARG_NAMES, NON_DEFAULT.as_args())))
        self.assertEqual(_keys(_metadata(same)["parameters"]), xsa.infotext_items(NON_DEFAULT, [REVERSE, TUNABLE]))

    def test_web_workflow_format(self):
        widgets = [*NON_DEFAULT.as_args(), "er_sde_reverse_time"]
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["checkpoint.safetensors"], "inputs": []},
                {"id": 2, "type": NODE, "widgets_values": widgets, "inputs": [{"name": "model", "link": 1}]},
                {"id": 5, "type": "KSampler", "widgets_values": [7, "fixed", 20, 5.0, TUNABLE, "normal", 1.0],
                 "inputs": [{"name": "model", "link": 2}]},
                {"id": 8, "type": "SaveImage", "widgets_values": ["out"], "inputs": [{"name": "images", "link": 3}]},
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 5, 0, "MODEL"], [3, 5, 0, 8, 0, "IMAGE"]],
        }
        params = parse_comfy_metadata(None, workflow)["parameters"]
        self.assertEqual(_keys(params), xsa.infotext_items(NON_DEFAULT, [TUNABLE]))


class ExtraSamplersPreflightTests(unittest.TestCase):
    def _row(self, result: dict) -> dict:
        return next(row for row in result["features"] if row["id"] == "extra_samplers")

    def test_rows(self):
        compiler = ComfyWorkflowCompiler(caps())
        values = feature_preflight(compiler, "checkpoint.safetensors", payload(xsa_block=block(NON_DEFAULT)))
        self.assertTrue(values["ok"], values)
        self.assertEqual((self._row(values)["state"], self._row(values)["label"]),
                         ("ready", "Extra Samplers · ER SDE 값"))
        names = feature_preflight(compiler, "checkpoint.safetensors", payload(
            sampler="ER SDE (Tunable)", xsa_block=block(), enable_hr=True, hr_sampler_name="Restart (flow)",
            adetailer={"ad_use_sampler": True, "ad_sampler": "deis"}, spectrum_enabled=False))
        self.assertTrue(names["ok"], names)
        self.assertEqual((self._row(names)["state"], self._row(names)["label"]),
                         ("ready", "Extra Samplers — er_sde_tunable, restart_flow"))
        both = feature_preflight(compiler, "checkpoint.safetensors", payload(
            sampler="Euler Dy", xsa_block=block(NON_DEFAULT), spectrum_enabled=True))
        self.assertTrue(both["ok"], both)
        self.assertEqual((self._row(both)["state"], self._row(both)["label"]),
                         ("ready", "Extra Samplers — euler_dy · ER SDE 값"))
        for body in (payload(), payload(sampler="IPNDM_V", xsa_block=block())):
            with self.subTest(body=body):
                result = feature_preflight(compiler, "checkpoint.safetensors", body)
                self.assertTrue(result["ok"])
                self.assertEqual((self._row(result)["state"], self._row(result)["requested"]), ("off", False))
        foreign = feature_preflight(ComfyWorkflowCompiler(caps(owned=())), "checkpoint.safetensors",
                                    payload(sampler="DPM++ 4M SDE"))
        self.assertFalse(foreign["ok"])
        self.assertEqual(self._row(foreign)["state"], "blocked")
        self.assertIn("다른 노드 팩이 'dpmpp_4m_sde' 샘플러를 먼저 등록", foreign["error"])
        stale = feature_preflight(ComfyWorkflowCompiler(_owner_caps(sampler_node=False)), "checkpoint.safetensors",
                                  payload(xsa_block=block(NON_DEFAULT)))
        self.assertFalse(stale["ok"])
        self.assertEqual(self._row(stale)["state"], "blocked")

    def test_the_pass_scan_names_what_the_graph_runs(self):
        names = ("er_sde_tunable", "Euler SMEA Dy", "Use same sampler", "")
        count = 0
        for main, hires, ad, sam in itertools.product(("euler", "restart_flow", "ER SDE (ODE)"), (None, *names),
                                                      (None, *names), (None, *names)):
            body = payload(sampler=main)
            if hires is not None:
                body.update(enable_hr=True, hr_sampler_name=hires)
            if ad is not None:
                body["alwayson_scripts"]["ADetailer"] = {"args": [True, False, {
                    "ad_tab_enable": True, "ad_model": "face.pt", "ad_use_sampler": bool(ad), "ad_sampler": ad}]}
            if sam is not None:
                body["alwayson_scripts"].update(sam3_args.build_alwayson({
                    "sam3_mode": "Inpaint", "sam3_prompt": "face", "sam3_use_sampler": True, "sam3_sampler": sam}))
            try:
                graph = compile_t2i(body)
            except WorkflowCompileError:
                continue          # 'Use same sampler' as a Hires sampler is not a KSampler name
            ran = {name for _nid, node, name in sampling_nodes(graph)} & set(xsa.PACK_SAMPLERS)
            with self.subTest(main=main, hires=hires, ad=ad, sam=sam):
                self.assertEqual(ran, set(ComfyWorkflowCompiler._pass_sampler_names(body)) & set(xsa.PACK_SAMPLERS))
            count += 1
        self.assertGreater(count, 200)


class ExtraSamplersNodeContractTests(unittest.TestCase):
    def test_compiler_constants_match_the_pack_node(self):
        schema = extra_samplers_nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES()
        self.assertEqual(tuple(schema["required"]), xsa.COMFY_INPUTS)
        self.assertEqual(list(schema["optional"]), [xsa.COMFY_REPORT_INPUT])
        self.assertEqual(extra_samplers_nodes.NODE_NAME, NODE)
        required = schema["required"]
        self.assertEqual((required["max_stage"][1]["min"], required["max_stage"][1]["max"]),
                         (xsa.MAX_STAGE_MIN, xsa.MAX_STAGE_MAX))
        self.assertEqual((required["eta"][1]["min"], required["eta"][1]["max"]), (xsa.ETA_MIN, xsa.ETA_MAX))
        for name in ("noise_start", "noise_end"):
            self.assertEqual((required[name][1]["min"], required[name][1]["max"]),
                             (xsa.NOISE_PERCENT_MIN, xsa.NOISE_PERCENT_MAX))
        self.assertEqual(required["noise_window"][0], "BOOLEAN")
        self.assertEqual([required[name][1]["default"] for name in xsa.ARG_NAMES], xsa.DEFAULTS.as_args())


if __name__ == "__main__":
    unittest.main()
