"""Extra Schedulers (sam-extra v0.33.0) on ComfyUI — the compiler side (plan C2a, C2b).

The bundled pack (1.7.0) registers sam-extra's nine schedulers under the same names, and ``ForgeNeoExtraSchedulers``
(MODEL -> MODEL) carries the block's eight values on the MODEL's ``model_sampling`` so every later sampler builds
Forge's schedule.  The compiler inserts that node right after the flow shift when the request carries the block and a
pass uses one of the four schedulers that read values (``laplace``, ``custom``, ``flow_cosmos_rho7``,
``flow_cosmos_dynamic``).

**How the payloads are built.**  Since plan task C6 the app's ``core/extra_schedulers.plan()`` sends the block on
ComfyUI always (``REASON_COMFY`` — tests/test_extra_schedulers_ui.py checks the contributor), so these tests build the
payload the same way: ``alwayson_scripts["Extra Schedulers (sam-extra)"] = core/extra_schedulers.as_block(settings)`` —
the block ``plan()`` also returns for Forge (one dict with all eight keys; on ComfyUI it is sent always, plan
R5/X5) — next to the app's ADetailer/SAM3 scripts; propagation copies the same block into aux payloads (standalone
ADetailer/SAM3), as on Forge.  Older blocks (five keys, positional values) come from presets and history.

No ComfyUI is contacted: the pack nodes' schemas are their real ``INPUT_TYPES`` (``owned_names`` patched, see
``tests/test_comfy_sampler_aliases._owner_caps``).
"""
from __future__ import annotations

import copy
import json
import math
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers as pack_schedulers
from core import sam3_args
from core import extra_schedulers as xs
from core.comfy_metadata import parse_comfy_metadata
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.comfy_workflow_controls import feature_preflight
from tests.test_comfy_sampler_aliases import _owner_caps
from tests.test_comfy_workflow_compiler import _custom_workflow, _node

NODE = xs.COMFY_NODE_CLASS
NON_DEFAULT = xs.ExtraSchedulerSettings(
    custom_mode=xs.MODE_SIGMAS, custom_expression="M * (m / M) ** (x ** 1.2)", custom_sigmas="[1.0, 0.5, 0.0]",
    laplace_mu=1.5, laplace_beta=0.75, flow_cosmos_rho=5.0, flow_cosmos_sigma_max=240.0, flow_cosmos_sigma_min=0.006,
)


def caps() -> dict:
    """Capabilities with the nine schedulers in KSampler's list, all owned by the pack."""
    return _owner_caps(schedulers=xs.SCHEDULER_NAMES)


def block(settings: xs.ExtraSchedulerSettings = xs.APP_DEFAULTS) -> dict:
    """What ``core/extra_schedulers.plan()`` sends (C6: also on ComfyUI)."""
    return xs.as_block(settings)


def payload(*, scheduler="karras", xs_block=None, adetailer=None, sam3=None, **extra) -> dict:
    scripts: dict = {}
    if adetailer is not None:
        scripts["ADetailer"] = {"args": [True, False, {"ad_tab_enable": True, "ad_model": "face.pt", **adetailer}]}
    if sam3 is not None:
        scripts.update(sam3_args.build_alwayson({"sam3_mode": "Inpaint", "sam3_prompt": "face", **sam3}))
    if xs_block is not None:
        scripts[xs.SCRIPT_NAME] = xs_block
    return {"prompt": "portrait", "negative_prompt": "bad", "seed": 7, "scheduler": scheduler,
            "alwayson_scripts": scripts, **extra}


def compile_t2i(body, *, workflow=None, capabilities=None) -> dict:
    return ComfyWorkflowCompiler(capabilities or caps()).compile(
        "txt2img", "checkpoint.safetensors", body, workflow=workflow)


def carriers(graph: dict) -> list:
    return [(node_id, node) for node_id, node in graph.items() if node.get("class_type") == NODE]


def model_chain(graph: dict, link) -> list:
    """Class types up the ``model`` inputs from ``link`` (nearest first)."""
    chain, seen = [], set()
    while isinstance(link, list) and len(link) >= 2 and str(link[0]) in graph and str(link[0]) not in seen:
        seen.add(str(link[0]))
        node = graph[str(link[0])]
        chain.append(node["class_type"])
        link = node.get("inputs", {}).get("model")
    return chain


def sampling_nodes(graph: dict) -> list:
    """(node, scheduler) for every pass the graph samples with — ADetailer's scheduler is in its settings JSON."""
    found = []
    for node in graph.values():
        inputs = node.get("inputs", {})
        if node.get("class_type") == "ForgeNeoADetailer":
            found.append((node, json.loads(inputs["settings_json"]).get("scheduler")))
            continue
        for key in ("scheduler", "base_scheduler"):
            if isinstance(inputs.get(key), str) and "model" in inputs:
                found.append((node, inputs[key]))
    return found


class ExtraSchedulersInsertionTests(unittest.TestCase):
    """C2a: where and when ``ForgeNeoExtraSchedulers`` goes in, and what it receives."""

    def test_no_block_or_no_value_scheduler_keeps_the_graph_byte_identical(self):
        others = [name for name in xs.SCHEDULER_NAMES if name not in xs.VALUE_SCHEDULERS]
        variants = {
            "plain": {},
            "hires": {"enable_hr": True, "hr_scheduler": "Karras Dynamic"},
            "adetailer": {"adetailer": {"ad_use_sampler": True, "ad_scheduler": "Phi"}},
            "sam3": {"sam3": {"sam3_use_scheduler": True, "sam3_scheduler": "cosine"}},
            "adetailer laplace but slot off": {
                "adetailer": {"ad_tab_enable": False, "ad_use_sampler": True, "ad_scheduler": "laplace"}},
        }
        for scheduler in ("karras", "Automatic", *others, *(xs.SCHEDULER_LABELS[name] for name in others)):
            for label, extra in variants.items():
                with self.subTest(scheduler=scheduler, variant=label):
                    without = compile_t2i(payload(scheduler=scheduler, **extra))
                    for settings in (xs.APP_DEFAULTS, NON_DEFAULT):
                        with_block = compile_t2i(payload(scheduler=scheduler, xs_block=block(settings), **extra))
                        self.assertEqual(json.dumps(with_block, sort_keys=True), json.dumps(without, sort_keys=True))
                    self.assertEqual(carriers(without), [])

    def test_each_value_scheduler_inserts_one_node_right_after_the_flow_shift(self):
        for name in xs.VALUE_SCHEDULERS:
            for text in (name, xs.SCHEDULER_LABELS[name]):
                for shift in (0.0, 3.0):
                    with self.subTest(scheduler=text, shift=shift):
                        graph = compile_t2i(payload(
                            scheduler=text, xs_block=block(), distilled_cfg_scale=shift,
                            prompt="portrait <lora:styles/ink:0.5>"))
                        (carrier_id, carrier), = carriers(graph)
                        upstream = model_chain(graph, carrier["inputs"]["model"])
                        self.assertEqual(upstream[0], "ForgeNeoModelSamplingShift" if shift else "CheckpointLoaderSimple")
                        _lid, lora = _node(graph, "ForgeNeoAnimaLoraLoader" if "ForgeNeoAnimaLoraLoader" in str(graph)
                                           else "LoraLoader")
                        self.assertEqual(lora["inputs"]["model"], [carrier_id, 0])
                        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                        self.assertEqual(sampler["inputs"]["scheduler"], name)
                        self.assertIn(NODE, model_chain(graph, sampler["inputs"]["model"]))
                        self.assertEqual(carrier["_meta"]["title"], "Extra Schedulers values (sam-extra)")

    def test_the_node_gets_the_eight_core_values(self):
        schema = pack_schedulers.ForgeNeoExtraSchedulers.INPUT_TYPES()
        for settings in (xs.APP_DEFAULTS, NON_DEFAULT,
                         xs.ExtraSchedulerSettings(custom_expression="", flow_cosmos_sigma_min=1.0,
                                                   flow_cosmos_sigma_max=1.0, flow_cosmos_rho=1.0)):
            with self.subTest(settings=settings):
                (_cid, carrier), = carriers(compile_t2i(payload(scheduler="laplace", xs_block=block(settings))))
                inputs = carrier["inputs"]
                self.assertEqual({key: value for key, value in inputs.items() if key != "model"}, settings.as_arg())
                self.assertEqual(tuple(inputs), xs.COMFY_INPUTS)
                self.assertEqual(set(inputs), set(schema["required"]))
                self.assertNotIn(xs.COMFY_REPORT_INPUT, inputs)
                self.assertIn(inputs["custom_mode"], schema["required"]["custom_mode"][0])
                for key in ("laplace_mu", "laplace_beta", "flow_cosmos_rho", "flow_cosmos_sigma_max",
                            "flow_cosmos_sigma_min"):
                    self.assertIs(type(inputs[key]), float)

    def test_values_are_coerced_into_the_node_input_ranges(self):
        schema = pack_schedulers.ForgeNeoExtraSchedulers.INPUT_TYPES()["required"]
        raw = {"custom_mode": "Sigma list", "custom_expression": "  M * x  ", "custom_sigmas": None,
               "laplace_mu": 99, "laplace_beta": -3, "flow_cosmos_rho": 0.2, "flow_cosmos_sigma_max": 5000,
               "flow_cosmos_sigma_min": 1e-9}
        for label, args in (("dict", [raw]),
                            ("non-finite", [{"laplace_mu": "nan", "flow_cosmos_rho": float("inf"),
                                             "flow_cosmos_sigma_max": 10 ** 400, "laplace_beta": True}])):
            with self.subTest(block=label):
                (_cid, carrier), = carriers(compile_t2i(payload(scheduler="laplace", xs_block={"args": args})))
                inputs = carrier["inputs"]
                self.assertEqual({key: value for key, value in inputs.items() if key != "model"},
                                 xs.coerce_args(args).as_arg())
                for key, value in inputs.items():
                    spec = schema[key]
                    if spec[0] == "FLOAT":
                        self.assertTrue(math.isfinite(value))
                        self.assertGreaterEqual(value, spec[1]["min"], key)
                        self.assertLessEqual(value, spec[1]["max"], key)
        (_cid, carrier), = carriers(compile_t2i(payload(scheduler="laplace", xs_block={"args": [raw]})))
        self.assertEqual((carrier["inputs"]["custom_mode"], carrier["inputs"]["custom_expression"],
                          carrier["inputs"]["custom_sigmas"], carrier["inputs"]["laplace_mu"],
                          carrier["inputs"]["laplace_beta"], carrier["inputs"]["flow_cosmos_rho"],
                          carrier["inputs"]["flow_cosmos_sigma_max"], carrier["inputs"]["flow_cosmos_sigma_min"]),
                         ("sigmas", "M * x", xs.DEFAULT_SIGMAS, 10.0, 0.0, 1.0, 1000.0, 0.0001))

    def test_older_and_positional_blocks_give_flow_cosmos_its_defaults(self):
        five = {"custom_mode": "expression", "custom_expression": "M * x", "custom_sigmas": "[1.0, 0.0]",
                "laplace_mu": 2.0, "laplace_beta": 1.0}
        cases = (
            ("five keys", {"args": [five]}, xs.ExtraSchedulerSettings(custom_expression="M * x",
                                                                       custom_sigmas="[1.0, 0.0]",
                                                                       laplace_mu=2.0, laplace_beta=1.0)),
            ("five positional", {"args": list(five.values())},
             xs.ExtraSchedulerSettings(custom_expression="M * x", custom_sigmas="[1.0, 0.0]", laplace_mu=2.0,
                                       laplace_beta=1.0)),
            ("eight positional", {"args": [*five.values(), 4.0, 120.0, 0.01]},
             xs.ExtraSchedulerSettings(custom_expression="M * x", custom_sigmas="[1.0, 0.0]", laplace_mu=2.0,
                                       laplace_beta=1.0, flow_cosmos_rho=4.0, flow_cosmos_sigma_max=120.0,
                                       flow_cosmos_sigma_min=0.01)),
            ("no args", {"args": []}, xs.APP_DEFAULTS),
            ("no args key", {}, xs.APP_DEFAULTS),
        )
        for label, raw_block, expected in cases:
            with self.subTest(block=label):
                (_cid, carrier), = carriers(compile_t2i(payload(scheduler="Flow Cosmos rho7", xs_block=raw_block)))
                self.assertEqual({key: value for key, value in carrier["inputs"].items() if key != "model"},
                                 expected.as_arg())

    def test_every_pass_that_reads_the_values_gets_them(self):
        cases = {
            "hires only": (payload(xs_block=block(), enable_hr=True, hr_scheduler="Laplace"), "ForgeNeoHiresFix"),
            "hires follows main": (payload(scheduler="custom", xs_block=block(), enable_hr=True), "ForgeNeoHiresFix"),
            "adetailer slot": (payload(xs_block=block(), adetailer={
                "ad_use_sampler": True, "ad_scheduler": "Flow Cosmos Dynamic"}), "ForgeNeoADetailer"),
            "adetailer follows main": (payload(scheduler="flow_cosmos_rho7", xs_block=block(), adetailer={}),
                                       "ForgeNeoADetailer"),
            "adetailer own LoRA branch": (payload(xs_block=block(), adetailer={
                "ad_use_sampler": True, "ad_scheduler": "laplace", "ad_prompt": "face <lora:characters/alice:0.7>"}),
                "ForgeNeoADetailer"),
            "sam3": (payload(xs_block=block(), sam3={"sam3_use_scheduler": True, "sam3_scheduler": "custom"}),
                     "ForgeNeoSAM3Detailer"),
            "sam3 same scheduler": (payload(scheduler="Laplace", xs_block=block(), sam3={}), "ForgeNeoSAM3Detailer"),
        }
        for label, (body, class_type) in cases.items():
            with self.subTest(case=label):
                graph = compile_t2i(body)
                self.assertEqual(len(carriers(graph)), 1)
                _pid, node = _node(graph, class_type)
                self.assertIn(NODE, model_chain(graph, node["inputs"]["model"]))
                used = [scheduler for found, scheduler in sampling_nodes(graph) if found is node]
                self.assertTrue(set(used) & set(xs.VALUE_SCHEDULERS), used)

    def test_one_node_serves_every_pass(self):
        body = payload(scheduler="laplace", xs_block=block(NON_DEFAULT), enable_hr=True, hr_scheduler="custom",
                       adetailer={"ad_use_sampler": True, "ad_scheduler": "flow_cosmos_rho7"},
                       sam3={"sam3_use_scheduler": True, "sam3_scheduler": "flow_cosmos_dynamic"},
                       _comfy_detail_passes=["hand"])
        graph = compile_t2i(body)
        (carrier_id, _carrier), = carriers(graph)
        for node, scheduler in sampling_nodes(graph):
            with self.subTest(node=node["class_type"], scheduler=scheduler):
                self.assertIn(NODE, model_chain(graph, node["inputs"]["model"]))

    def test_custom_workflow_gets_the_node_on_top_of_its_own_model_sampling(self):
        workflow = _custom_workflow()
        workflow["8"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": 3.0}}
        workflow["5"]["inputs"]["model"] = ["8", 0]
        capabilities = caps()
        capabilities["ModelSamplingSD3"] = {"input": {"required": {}}}
        for shift in (0.0, 2.0):
            with self.subTest(shift=shift):
                graph = compile_t2i(payload(scheduler="Laplace", xs_block=block(), distilled_cfg_scale=shift),
                                    workflow=workflow, capabilities=capabilities)
                (carrier_id, carrier), = carriers(graph)
                chain = model_chain(graph, graph["5"]["inputs"]["model"])
                self.assertEqual(graph["5"]["inputs"]["scheduler"], "laplace")
                self.assertLess(chain.index(NODE), chain.index("ModelSamplingSD3"))
                upstream = model_chain(graph, carrier["inputs"]["model"])
                self.assertEqual(upstream[0], "ForgeNeoModelSamplingShift" if shift else "ModelSamplingSD3")
        without = compile_t2i(payload(scheduler="karras", xs_block=block()), workflow=workflow,
                              capabilities=capabilities)
        self.assertEqual(json.dumps(without, sort_keys=True), json.dumps(compile_t2i(
            payload(scheduler="karras"), workflow=workflow, capabilities=capabilities), sort_keys=True))

    def test_standalone_postprocess_gets_the_node(self):
        compiler = ComfyWorkflowCompiler(caps())
        body = payload(xs_block=block(NON_DEFAULT), adetailer={"ad_use_sampler": True, "ad_scheduler": "Laplace"})
        graph = compiler.compile_postprocess("checkpoint.safetensors", body, uploaded_image="in.png")
        (_cid, carrier), = carriers(graph)
        self.assertEqual(carrier["inputs"]["laplace_mu"], 1.5)
        _aid, adetailer = _node(graph, "ForgeNeoADetailer")
        self.assertIn(NODE, model_chain(graph, adetailer["inputs"]["model"]))
        plain = compiler.compile_postprocess("checkpoint.safetensors", payload(adetailer={}), uploaded_image="in.png")
        self.assertEqual(carriers(plain), [])

    def test_the_pass_scan_and_the_built_graph_agree(self):
        """Graph invariant: whenever a built pass samples with a value scheduler the node is there (never fewer),
        and the node is there exactly when ``_extra_schedulers_needed`` names a pass."""
        choices = ("karras", "Laplace", "custom", "Flow Cosmos rho7", "flow_cosmos_dynamic", "Use same scheduler")
        count = 0
        for main in ("karras", "laplace", "Flow Cosmos Dynamic"):
            for hires in (None, *choices):
                for ad in (None, *choices):
                    for sam in (None, "karras", "custom", "Use same scheduler"):
                        body = payload(scheduler=main, xs_block=block())
                        if hires is not None:
                            body.update(enable_hr=True, **({} if hires == "Use same scheduler" else
                                                           {"hr_scheduler": hires}))
                        scripts = body["alwayson_scripts"]
                        if ad is not None:
                            scripts["ADetailer"] = {"args": [True, False, {
                                "ad_tab_enable": True, "ad_model": "face.pt", "ad_use_sampler": True,
                                "ad_scheduler": ad}]}
                        if sam is not None:
                            scripts.update(sam3_args.build_alwayson({
                                "sam3_mode": "Inpaint", "sam3_prompt": "face", "sam3_use_scheduler": True,
                                "sam3_scheduler": sam}))
                        graph = compile_t2i(body)
                        used = {scheduler for _node_, scheduler in sampling_nodes(graph)} & set(xs.VALUE_SCHEDULERS)
                        present = bool(carriers(graph))
                        needed = bool(ComfyWorkflowCompiler._extra_schedulers_needed(body))
                        with self.subTest(main=main, hires=hires, ad=ad, sam=sam):
                            self.assertEqual(present, needed)
                            if used:
                                self.assertTrue(present)
                        count += 1
        self.assertEqual(count, 3 * 7 * 7 * 4)



def _carrier_inputs(settings: xs.ExtraSchedulerSettings = xs.APP_DEFAULTS, model=("1", 0)) -> dict:
    return {"model": list(model), **settings.as_arg()}


def _hand_graph(*chain: str, scheduler: str = "laplace", sampler_class: str = "KSampler") -> dict:
    """loader -> the given MODEL nodes in order -> one sampler (hand-made graph for the reach rule)."""
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "checkpoint.safetensors"}}}
    link = ["1", 0]
    for index, class_type in enumerate(chain, start=2):
        inputs = _carrier_inputs() if class_type == NODE else {"shift": 3.0} if "Sampling" in class_type else {}
        inputs["model"] = link
        graph[str(index)] = {"class_type": class_type, "inputs": inputs}
        link = [str(index), 0]
    graph["99"] = {"class_type": sampler_class, "inputs": {"model": link, "scheduler": scheduler}}
    return graph


class ExtraSchedulersReachTests(unittest.TestCase):
    """C2b: with the block present every pass that samples with a value scheduler must get the values."""

    def test_hires_with_shift_and_a_checkpoint_override_keeps_the_values(self):
        for override in ({}, {"hr_checkpoint_name": "other.safetensors"}):
            with self.subTest(override=override):
                graph = compile_t2i(payload(xs_block=block(NON_DEFAULT), enable_hr=True, hr_scheduler="Laplace",
                                            distilled_cfg_scale=3.0, **override))
                _hid, hires = _node(graph, "ForgeNeoHiresFix")
                self.assertEqual((hires["inputs"]["shift"], hires["inputs"]["scheduler"]), (3.0, "laplace"))
                chain = model_chain(graph, hires["inputs"]["model"])
                self.assertLess(chain.index(NODE), chain.index("ForgeNeoModelSamplingShift"))

    def test_the_rule_on_hand_made_chains(self):
        compiler = ComfyWorkflowCompiler(caps())
        body = payload(xs_block=block())
        ok = (
            (NODE,), (NODE, "LoraLoader"), (NODE, "ForgeNeoAnimaGuidanceSuite", "ForgeNeoAnimaDetailDaemon"),
            ("ModelSamplingSD3", NODE), ("ForgeNeoModelSamplingShift", NODE, "LoraLoader"),
        )
        bad = (
            (), ("LoraLoader",), (NODE, "ModelSamplingSD3"), (NODE, "ModelSamplingFlux", "LoraLoader"),
            (NODE, "ForgeNeoModelSamplingShift"), ("ModelSamplingAuraFlow",),
        )
        for sampler_class in ("KSampler", "KSamplerAdvanced", "ForgeNeoKSamplerCNS", "ForgeNeoSAM3Detailer",
                              "FaceDetailer"):
            for scheduler in (*xs.VALUE_SCHEDULERS, "Laplace", "Flow Cosmos Dynamic"):
                for chain in ok:
                    with self.subTest(sampler=sampler_class, scheduler=scheduler, chain=chain):
                        compiler._check_extra_schedulers_reach(
                            _hand_graph(*chain, scheduler=scheduler, sampler_class=sampler_class), body)
                for chain in bad:
                    with self.subTest(sampler=sampler_class, scheduler=scheduler, chain=chain), \
                            self.assertRaisesRegex(WorkflowCompileError,
                                                   f"Extra Schedulers 값이 닿지 않는 샘플러가 있습니다: {sampler_class} \\(node 99\\)"):
                        compiler._check_extra_schedulers_reach(
                            _hand_graph(*chain, scheduler=scheduler, sampler_class=sampler_class), body)
                    # without the block there are no values to carry — Forge and the pack use the defaults
                    compiler._check_extra_schedulers_reach(
                        _hand_graph(*chain, scheduler=scheduler, sampler_class=sampler_class), payload())

    def test_other_schedulers_hires_base_and_adetailer_settings(self):
        compiler = ComfyWorkflowCompiler(caps())
        body = payload(xs_block=block())
        for scheduler in ("karras", "normal", "cosine", "karras_dynamic", "react_cosinusoidal_dynsf", "beta57"):
            with self.subTest(scheduler=scheduler):
                compiler._check_extra_schedulers_reach(_hand_graph("LoraLoader", scheduler=scheduler), body)
        hires = _hand_graph("ModelSamplingSD3", scheduler="karras", sampler_class="ForgeNeoHiresFix")
        hires["99"]["inputs"]["base_scheduler"] = "Laplace"
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoHiresFix \\(node 99\\)"):
            compiler._check_extra_schedulers_reach(hires, body)
        detailer = _hand_graph("LoraLoader", scheduler="karras", sampler_class="ForgeNeoADetailer")
        del detailer["99"]["inputs"]["scheduler"]
        detailer["99"]["_meta"] = {"title": "ADetailer slot 1"}
        for settings_json, refused in (('{"scheduler": "flow_cosmos_dynamic"}', True),
                                       ('{"scheduler": "karras"}', False), ("{}", False), ("not json", False),
                                       ("[1, 2]", False)):
            detailer["99"]["inputs"]["settings_json"] = settings_json
            with self.subTest(settings=settings_json):
                if refused:
                    with self.assertRaisesRegex(WorkflowCompileError, "닿지 않는 샘플러가 있습니다: ADetailer slot 1"):
                        compiler._check_extra_schedulers_reach(detailer, body)
                else:
                    compiler._check_extra_schedulers_reach(detailer, body)
        looped = _hand_graph(NODE, scheduler="custom")
        looped["2"]["inputs"]["model"] = ["99", 0]
        looped["99"]["inputs"]["model"] = ["99", 0]
        with self.assertRaises(WorkflowCompileError):
            compiler._check_extra_schedulers_reach(looped, body)

    def test_a_workflow_sampler_the_app_does_not_rewire_is_refused(self):
        workflow = _custom_workflow()
        workflow["9"] = {"class_type": "FaceDetailer", "inputs": {"model": ["1", 0], "scheduler": "laplace"},
                         "_meta": {"title": "My face pass"}}
        for scheduler in ("Laplace", "karras", "Flow Cosmos rho7"):
            with self.subTest(main=scheduler), self.assertRaisesRegex(
                    WorkflowCompileError, "Extra Schedulers 값이 닿지 않는 샘플러가 있습니다: My face pass \\(node 9\\)"):
                compile_t2i(payload(scheduler=scheduler, xs_block=block()), workflow=workflow)
        compile_t2i(payload(scheduler="Laplace"), workflow=workflow)      # no block: nothing to carry

    def test_a_scan_that_misses_a_pass_is_caught(self):
        with mock.patch.object(ComfyWorkflowCompiler, "_extra_schedulers_needed", return_value=()):
            for body, title in (
                    (payload(scheduler="laplace", xs_block=block()), "Forge-compatible sampler"),
                    (payload(xs_block=block(), enable_hr=True, hr_scheduler="custom"), "Forge-compatible Hires.fix"),
                    (payload(xs_block=block(), adetailer={"ad_use_sampler": True, "ad_scheduler": "laplace"}),
                     "ADetailer slot 1"),
                    (payload(xs_block=block(), sam3={"sam3_use_scheduler": True,
                                                     "sam3_scheduler": "flow_cosmos_dynamic"}), "SAM3 detailer")):
                with self.subTest(title=title), self.assertRaisesRegex(
                        WorkflowCompileError, f"닿지 않는 샘플러가 있습니다: {title}"):
                    compile_t2i(body)
            with self.assertRaisesRegex(WorkflowCompileError, "닿지 않는 샘플러가 있습니다: ADetailer slot 1"):
                ComfyWorkflowCompiler(caps()).compile_postprocess(
                    "checkpoint.safetensors",
                    payload(xs_block=block(), adetailer={"ad_use_sampler": True, "ad_scheduler": "custom"}),
                    uploaded_image="in.png")


def _metadata(graph: dict) -> dict:
    return parse_comfy_metadata(graph)


def _saved(graph: dict) -> dict:
    graph["98"] = {"class_type": "SaveImage", "inputs": {"images": ["99", 0]}}
    return graph


class ExtraSchedulersMetadataTests(unittest.TestCase):
    """C2b: PNG Info of a ComfyUI image gets the Forge infotext keys of the values its main/Hires pass used."""

    def test_main_pass_keys_equal_forge_infotext_items(self):
        for settings in (xs.APP_DEFAULTS, NON_DEFAULT, xs.ExtraSchedulerSettings(custom_expression="  M *   x  ")):
            for name in xs.VALUE_SCHEDULERS:
                with self.subTest(settings=settings, scheduler=name):
                    graph = compile_t2i(payload(scheduler=xs.SCHEDULER_LABELS[name], xs_block=block(settings)))
                    params = _metadata(graph)["parameters"]
                    expected = xs.infotext_items(settings, [name])
                    self.assertTrue(expected)
                    self.assertEqual({key: params[key] for key in xs.INFOTEXT_KEYS if key in params}, expected)
                    self.assertEqual(params["Schedule type"], name)

    def test_either_flow_cosmos_label_writes_all_three_values(self):
        for name in xs.FLOW_COSMOS_SCHEDULERS:
            with self.subTest(scheduler=name):
                params = _metadata(compile_t2i(payload(scheduler=name, xs_block=block(NON_DEFAULT))))["parameters"]
                self.assertEqual((params[xs.KEY_FLOW_COSMOS_RHO], params[xs.KEY_FLOW_COSMOS_SIGMA_MAX],
                                  params[xs.KEY_FLOW_COSMOS_SIGMA_MIN]), ("5.0", "240.0", "0.006"))
                for key in (xs.KEY_LAPLACE_MU, xs.KEY_CUSTOM_SIGMAS, xs.KEY_CUSTOM_EXPRESSION):
                    self.assertNotIn(key, params)

    def test_hires_pass_counts_detailer_passes_do_not(self):
        hires = _metadata(compile_t2i(payload(xs_block=block(NON_DEFAULT), enable_hr=True, hr_scheduler="Laplace")))
        self.assertEqual({key: hires["parameters"][key] for key in (xs.KEY_LAPLACE_MU, xs.KEY_LAPLACE_BETA)},
                         {xs.KEY_LAPLACE_MU: "1.5", xs.KEY_LAPLACE_BETA: "0.75"})
        both = _metadata(compile_t2i(payload(scheduler="custom", xs_block=block(NON_DEFAULT), enable_hr=True,
                                             hr_scheduler="flow_cosmos_rho7")))["parameters"]
        self.assertEqual({key: both[key] for key in xs.INFOTEXT_KEYS if key in both},
                         xs.infotext_items(NON_DEFAULT, ["custom", "flow_cosmos_rho7"]))
        for extra in ({"adetailer": {"ad_use_sampler": True, "ad_scheduler": "laplace"}},
                      {"sam3": {"sam3_use_scheduler": True, "sam3_scheduler": "flow_cosmos_rho7"}}):
            with self.subTest(extra=extra):
                params = _metadata(compile_t2i(payload(xs_block=block(NON_DEFAULT), **extra)))["parameters"]
                self.assertFalse(set(params) & set(xs.INFOTEXT_KEYS))

    def test_no_block_or_no_reach_writes_nothing(self):
        params = _metadata(compile_t2i(payload(scheduler="laplace")))["parameters"]
        self.assertFalse(set(params) & set(xs.INFOTEXT_KEYS))
        self.assertFalse(set(_metadata(_saved(_hand_graph(NODE, "ModelSamplingSD3")))["parameters"]) & set(xs.INFOTEXT_KEYS))
        self.assertFalse(set(_metadata(_saved(_hand_graph(NODE, "ForgeNeoModelSamplingShift")))["parameters"])
                         & set(xs.INFOTEXT_KEYS))
        params = _metadata(_saved(_hand_graph("ModelSamplingSD3", NODE, "LoraLoader")))["parameters"]
        self.assertEqual({key: value for key, value in params.items() if key in xs.INFOTEXT_KEYS},
                         xs.infotext_items(xs.APP_DEFAULTS, ["laplace"]))
        params = _metadata(_saved(_hand_graph(NODE, scheduler="karras")))["parameters"]
        self.assertFalse(set(params) & set(xs.INFOTEXT_KEYS))

    def test_paste_restores_the_values_from_comfy_metadata(self):
        for name in xs.VALUE_SCHEDULERS:
            with self.subTest(scheduler=name):
                params = _metadata(compile_t2i(payload(scheduler=name, xs_block=block(NON_DEFAULT))))["parameters"]
                restored = xs.from_infotext(params)
                if name == "laplace":
                    self.assertEqual((restored.laplace_mu, restored.laplace_beta), (1.5, 0.75))
                elif name == "custom":
                    self.assertEqual((restored.custom_mode, restored.custom_sigmas), (xs.MODE_SIGMAS, "[1.0, 0.5, 0.0]"))
                else:
                    self.assertEqual((restored.flow_cosmos_rho, restored.flow_cosmos_sigma_max,
                                      restored.flow_cosmos_sigma_min), (5.0, 240.0, 0.006))

    def test_linked_and_conflicting_values(self):
        graph = _saved(_hand_graph(NODE, scheduler="custom"))
        graph["50"] = {"class_type": "PrimitiveString", "inputs": {"value": "M * (m / M) ** (x * 2)"}}
        graph["2"]["inputs"]["custom_expression"] = ["50", 0]
        params = _metadata(graph)["parameters"]
        self.assertEqual(params[xs.KEY_CUSTOM_EXPRESSION], "M * (m / M) ** (x * 2)")
        linked = copy.deepcopy(graph)
        linked["51"] = {"class_type": "PrimitiveNode", "inputs": {"value": 2.0}}
        linked["2"]["inputs"]["laplace_mu"] = ["51", 0]
        result = _metadata(linked)
        self.assertFalse(set(result["parameters"]) & set(xs.INFOTEXT_KEYS))
        self.assertIn("Extra Schedulers 값이 연결로 들어와 그 값을 기록하지 않았습니다.", result["warnings"])
        two = copy.deepcopy(graph)
        two["3"] = {"class_type": NODE, "inputs": _carrier_inputs(NON_DEFAULT)}
        two["97"] = {"class_type": "KSampler", "inputs": {"model": ["3", 0], "scheduler": "custom"}}
        two["96"] = {"class_type": "SaveImage", "inputs": {"images": ["97", 0]}}
        result = _metadata(two)
        self.assertFalse(set(result["parameters"]) & set(xs.INFOTEXT_KEYS))
        self.assertIn("Extra Schedulers 노드가 여럿이고 값이 달라 Extra Schedulers 값을 고르지 않았습니다.", result["warnings"])
        same = copy.deepcopy(two)
        same["3"]["inputs"] = _carrier_inputs()
        same["3"]["inputs"]["custom_expression"] = "M * (m / M) ** (x * 2)"
        self.assertEqual(_metadata(same)["parameters"][xs.KEY_CUSTOM_EXPRESSION], "M * (m / M) ** (x * 2)")

    def test_web_workflow_format(self):
        widgets = [*NON_DEFAULT.as_arg().values(), "laplace"]
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "widgets_values": ["checkpoint.safetensors"], "inputs": []},
                {"id": 2, "type": NODE, "widgets_values": widgets, "inputs": [{"name": "model", "link": 1}]},
                {"id": 5, "type": "KSampler", "widgets_values": [7, "fixed", 20, 5.0, "euler", "laplace", 1.0],
                 "inputs": [{"name": "model", "link": 2}]},
                {"id": 8, "type": "SaveImage", "widgets_values": ["out"], "inputs": [{"name": "images", "link": 3}]},
            ],
            "links": [[1, 1, 0, 2, 0, "MODEL"], [2, 2, 0, 5, 0, "MODEL"], [3, 5, 0, 8, 0, "IMAGE"]],
        }
        params = parse_comfy_metadata(None, workflow)["parameters"]
        self.assertEqual({key: params[key] for key in xs.INFOTEXT_KEYS if key in params},
                         xs.infotext_items(NON_DEFAULT, ["laplace"]))


class ExtraSchedulersPreflightTests(unittest.TestCase):
    def _row(self, result: dict) -> dict:
        return next(row for row in result["features"] if row["id"] == "extra_schedulers")

    def test_rows(self):
        compiler = ComfyWorkflowCompiler(caps())
        ready = feature_preflight(compiler, "checkpoint.safetensors", payload(scheduler="laplace", xs_block=block()))
        self.assertTrue(ready["ok"])
        self.assertEqual((self._row(ready)["state"], self._row(ready)["label"]), ("ready", "Extra Schedulers 값"))
        for body in (payload(scheduler="laplace"), payload(scheduler="karras", xs_block=block())):
            with self.subTest(body=body):
                result = feature_preflight(compiler, "checkpoint.safetensors", body)
                self.assertTrue(result["ok"])
                self.assertEqual((self._row(result)["state"], self._row(result)["requested"]), ("off", False))
        foreign = feature_preflight(ComfyWorkflowCompiler(_owner_caps(schedulers=xs.SCHEDULER_NAMES, owned_schedulers=())),
                                    "checkpoint.safetensors", payload(scheduler="laplace", xs_block=block()))
        self.assertFalse(foreign["ok"])
        self.assertEqual(self._row(foreign)["state"], "blocked")
        self.assertIn("다른 노드 팩이 'laplace' 스케줄러를 먼저 등록", foreign["error"])


if __name__ == "__main__":
    unittest.main()
