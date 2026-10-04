"""Forge sampler labels resolve the same in the app compiler and the node pack (감사 #174).

The bundled node pack is copied into ComfyUI and cannot import core, so the
ADetailer node keeps its own alias table.  These golden values pin both
tables to the same ComfyUI names, including labels the compiler used to reject
("DPM++ 2S a", "UniPC", "DPM fast", "DPM adaptive") and the one the node used
to mangle ("DPM++ 3M SDE").
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import compat as pack_compat
from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_nodes, extra_samplers_register
from comfy_custom_nodes.ai_studio_forge_parity import extra_schedulers as pack_schedulers
from comfy_custom_nodes.ai_studio_forge_parity import generation
from core import comfy_node_pack, sam3_args
from core import extra_samplers as xsa
from core import extra_schedulers as xs
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.lenient_numbers import finite_float, lenient_int
from tests.test_comfy_workflow_compiler import _capabilities, _choice, _custom_workflow, _node

# Forge label -> (ComfyUI sampler_name, scheduler implied by the label or None)
GOLDEN_SAMPLERS = {
    "Euler": ("euler", None),
    "Euler a": ("euler_ancestral", None),
    "LMS": ("lms", None),
    "Heun": ("heun", None),
    "DPM2": ("dpm_2", None),
    "DPM2 a": ("dpm_2_ancestral", None),
    "DPM++ 2S a": ("dpmpp_2s_ancestral", None),
    "DPM++ 2M": ("dpmpp_2m", None),
    "DPM++ SDE": ("dpmpp_sde", None),
    "DPM++ 2M SDE": ("dpmpp_2m_sde", None),
    "DPM++ 2M SDE Heun": ("dpmpp_2m_sde_heun", None),
    "DPM++ 3M SDE": ("dpmpp_3m_sde", None),
    "DPM fast": ("dpm_fast", None),
    "DPM adaptive": ("dpm_adaptive", None),
    "UniPC": ("uni_pc", None),
    "LCM": ("lcm", None),
    "DDIM": ("ddim", None),
    "ER SDE": ("er_sde", None),
    "Res Multistep": ("res_multistep", None),
    # Old A1111 combined labels carry the scheduler.
    "DPM++ 2M Karras": ("dpmpp_2m", "karras"),
    "DPM++ SDE Karras": ("dpmpp_sde", "karras"),
    "DPM++ 2M SDE Exponential": ("dpmpp_2m_sde", "exponential"),
    "DPM++ 3M SDE Karras": ("dpmpp_3m_sde", "karras"),
    "DPM2 a Karras": ("dpm_2_ancestral", "karras"),
    # sam-extra Extra Samplers v0.33.0 (PLAN_amend §3) — twelve pack names, five ComfyUI core names
    # ("DPM++ 2M SDE Heun" above is the sixth core one).
    "ER SDE (Reverse-time)": ("er_sde_reverse_time", None),
    "ER SDE (ODE)": ("er_sde_ode", None),
    "ER SDE (Tunable)": ("er_sde_tunable", None),
    "DPM++ 4M SDE": ("dpmpp_4m_sde", None),
    "Euler Dy CFG++": ("euler_dy_cfg_pp", None),
    "Euler SMEA Dy CFG++": ("euler_smea_dy_cfg_pp", None),
    "Euler Dy": ("euler_dy", None),
    "Euler SMEA Dy": ("euler_smea_dy", None),
    "DPM++ 2M (flow ODE)": ("dpmpp_2m_flow_ode", None),
    "DPM++ 2M Heun (flow ODE)": ("dpmpp_2m_heun_flow_ode", None),
    "DPM++ 3M (flow ODE)": ("dpmpp_3m_flow_ode", None),
    "UniPC bh2": ("uni_pc_bh2", None),
    "CFG++ UD10 AB": ("cfgpp_ud10_ab", None),
    "IPNDM": ("ipndm", None),
    "IPNDM_V": ("ipndm_v", None),
    "DEIS": ("deis", None),
    "Restart (flow)": ("restart_flow", None),
}

GOLDEN_SCHEDULERS = {
    "Automatic": "normal", "Use same scheduler": "normal", "Karras": "karras",
    "Exponential": "exponential", "SGM Uniform": "sgm_uniform",
    "DDIM Uniform": "ddim_uniform", "Simple": "simple", "Normal": "normal",
    "Beta": "beta", "Beta57 (RES4LYF)": "beta57", "beta57": "beta57",
    # sam-extra Extra Schedulers v0.33.0 — labels and README aliases resolve to the names the pack registers.
    "Cosine": "cosine", "CosineExponential blend": "cosine_exponential",
    "cosine-exponential blend": "cosine_exponential", "Phi": "phi", "Laplace": "laplace",
    "Karras Dynamic": "karras_dynamic", "karras dynamic": "karras_dynamic", "custom": "custom",
    "React Cosinusoidal DynSF": "react_cosinusoidal_dynsf", "Flow Cosmos rho7": "flow_cosmos_rho7",
    "Flow Cosmos Dynamic": "flow_cosmos_dynamic",
}


class SamplerAliasGoldenTests(unittest.TestCase):
    def test_compiler_and_node_resolve_every_forge_label_identically(self):
        for label, (sampler, implied) in GOLDEN_SAMPLERS.items():
            with self.subTest(label=label):
                self.assertEqual(ComfyWorkflowCompiler._comfy_sampler(label), sampler)
                node_sampler, node_scheduler = generation._normalize_ad_sampler(label, "normal")
                self.assertEqual(node_sampler, sampler)
                compiler_scheduler = ComfyWorkflowCompiler._comfy_scheduler("normal", sampler_text=label)
                self.assertEqual(compiler_scheduler, implied or "normal")
                self.assertEqual(node_scheduler, implied or "normal")

    def test_scheduler_labels_resolve_identically(self):
        for label, expected in GOLDEN_SCHEDULERS.items():
            with self.subTest(label=label):
                self.assertEqual(ComfyWorkflowCompiler._comfy_scheduler(label, sampler_text="Euler"), expected)
                self.assertEqual(generation._normalize_ad_sampler("Euler", label)[1], expected)

    def test_alias_tables_are_identical(self):
        from core import comfy_workflow_compiler as compiler_module

        self.assertEqual(compiler_module._FORGE_SAMPLER_ALIASES, generation._AD_SAMPLER_ALIASES)
        self.assertEqual(compiler_module._FORGE_SCHEDULER_ALIASES, generation._AD_SCHEDULER_ALIASES)
        self.assertEqual(
            compiler_module._FORGE_SAMPLER_SCHEDULER_SUFFIXES,
            generation._AD_SAMPLER_SCHEDULER_SUFFIXES,
        )
        self.assertEqual(compiler_module._FORGE_SAME_SCHEDULER, generation._AD_SAME_SCHEDULER)

    def test_forge_labels_pass_live_choice_validation_everywhere(self):
        capabilities = _capabilities()
        capabilities["KSampler"]["input"]["required"].update({
            "sampler_name": _choice("euler", "dpmpp_2s_ancestral", "uni_pc", "dpm_fast", "dpm_adaptive"),
            "scheduler": _choice("normal", "karras"),
        })
        compiler = ComfyWorkflowCompiler(capabilities)
        for label, native in (("DPM++ 2S a", "dpmpp_2s_ancestral"), ("UniPC", "uni_pc"),
                              ("DPM fast", "dpm_fast"), ("DPM adaptive", "dpm_adaptive")):
            with self.subTest(label=label):
                graph = compiler.compile("txt2img", "checkpoint.safetensors", {
                    "sampler_name": label, "enable_hr": True,
                    "alwayson_scripts": {"ADetailer": {"args": [True, False, {
                        "ad_tab_enable": True, "ad_model": "face.pt",
                    }]}},
                })
                _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
                _hid, hires = _node(graph, "ForgeNeoHiresFix")
                self.assertEqual(sampler["inputs"]["sampler_name"], native)
                self.assertEqual(hires["inputs"]["sampler_name"], native)
        with self.assertRaisesRegex(WorkflowCompileError, "지원하지 않는 값"):
            compiler.compile("txt2img", "checkpoint.safetensors", {"sampler_name": "Restart"})


class SamExtraNameTests(unittest.TestCase):
    """sam-extra v0.33.0 names on ComfyUI (plan C1): every Forge label/alias the app knows resolves to the ComfyUI
    name in the compiler and in the pack's ADetailer node alike; the discard set and 'Automatic' follow the pack."""

    def test_every_extra_sampler_label_resolves_to_its_comfy_name(self):
        self.assertEqual(len(xsa.LABELS), 18)
        for label in xsa.LABELS:
            name = xsa.COMFY_NAMES[label]
            for text in (label, label.upper(), f"  {label}  ", name):
                with self.subTest(text=text):
                    self.assertEqual(ComfyWorkflowCompiler._comfy_sampler(text), name)
                    self.assertEqual(generation._normalize_ad_sampler(text, "normal")[0], name)

    def test_every_extra_scheduler_name_label_and_alias_resolves_to_its_name(self):
        self.assertEqual(len(xs.SCHEDULER_NAMES), 9)
        for name in xs.SCHEDULER_NAMES:
            texts = (name, xs.SCHEDULER_LABELS[name], xs.SCHEDULER_LABELS[name].upper(),
                     *((xs.README_LABELS[name],) if name in xs.README_LABELS else ()))
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(ComfyWorkflowCompiler._comfy_scheduler(text, sampler_text="Euler"), name)
                    self.assertEqual(generation._normalize_ad_sampler("Euler", text)[1], name)
                    self.assertEqual(xs.resolve_scheduler(text), name)

    def test_alias_rows_name_only_known_comfy_names(self):
        from core import comfy_workflow_compiler as compiler_module

        sampler_rows = {key: value for key, value in compiler_module._FORGE_SAMPLER_ALIASES.items()
                        if value in xsa.COMFY_NAMES.values()}
        self.assertEqual(set(sampler_rows.values()), set(xsa.COMFY_NAMES.values()))
        app_spellings = {"uni pc bh2": "uni_pc_bh2"}   # like "uni pc" -> uni_pc above; not an extension alias
        for key, value in sampler_rows.items():
            with self.subTest(row=key):
                self.assertEqual(xsa.comfy_name(key) or app_spellings.get(key), value,
                                 "별칭 행이 확장 표기와 다른 이름을 고른다")
        scheduler_rows = {key: value for key, value in compiler_module._FORGE_SCHEDULER_ALIASES.items()
                          if value in xs.SCHEDULER_NAMES}
        self.assertEqual(set(scheduler_rows.values()), set(xs.SCHEDULER_NAMES))
        for key, value in scheduler_rows.items():
            with self.subTest(row=key):
                self.assertEqual(xs.resolve_scheduler(key), value)

    def test_discard_set_includes_what_the_pack_adds(self):
        from core import comfy_workflow_compiler as compiler_module

        core_comfy = {"dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"}   # ComfyUI samplers.py
        self.assertEqual(set(compiler_module._PENULTIMATE_SIGMA_DISCARD_SAMPLERS),
                         core_comfy | set(xsa.COMFY_DISCARD_PENULTIMATE))
        self.assertEqual(tuple(extra_samplers_register.DISCARD_PENULTIMATE_SIGMA), tuple(xsa.COMFY_DISCARD_PENULTIMATE))

    def test_automatic_is_exponential_for_dpmpp_4m_sde_only(self):
        self.assertEqual(dict(generation._AD_AUTOMATIC_SCHEDULERS), dict(xsa.COMFY_AUTOMATIC_SCHEDULER))
        from core import comfy_workflow_compiler as compiler_module

        self.assertEqual(compiler_module._FORGE_AUTOMATIC_SCHEDULER, generation._AD_AUTOMATIC_SCHEDULER)
        cases = (
            ("DPM++ 4M SDE", "Automatic", "exponential"), ("DPM++ 4M SDE", "auto", "exponential"),
            ("dpmpp_4m_sde", "Automatic", "exponential"), ("DPM++ 4M SDE", "Karras", "karras"),
            ("DPM++ 4M SDE", "Use same scheduler", "normal"), ("DPM++ 4M SDE", "normal", "normal"),
            ("DPM++ 4M SDE Karras", "Automatic", "karras"),
            ("DPM++ 2M SDE", "Automatic", "normal"), ("DPM++ 3M SDE", "Automatic", "normal"),
            ("DPM++ 3M (flow ODE)", "Automatic", "normal"), ("Euler", "Automatic", "normal"),
        )
        for sampler, scheduler, expected in cases:
            with self.subTest(sampler=sampler, scheduler=scheduler):
                self.assertEqual(ComfyWorkflowCompiler._comfy_scheduler(scheduler, sampler_text=sampler), expected)
                self.assertEqual(generation._normalize_ad_sampler(sampler, scheduler)[1], expected)

    def test_automatic_reaches_every_pass_for_dpmpp_4m_sde(self):
        compiler = ComfyWorkflowCompiler(_owner_caps(samplers=("dpmpp_4m_sde",), schedulers=("exponential",)))
        graph = compiler.compile("txt2img", "checkpoint.safetensors", {
            "sampler_name": "DPM++ 4M SDE", "scheduler": "Automatic", "enable_hr": True,
        })
        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        _hid, hires = _node(graph, "ForgeNeoHiresFix")
        self.assertEqual((sampler["inputs"]["sampler_name"], sampler["inputs"]["scheduler"]),
                         ("dpmpp_4m_sde", "exponential"))
        self.assertEqual((hires["inputs"]["scheduler"], hires["inputs"]["base_scheduler"]), ("exponential", "exponential"))

    def test_pack_contract_constants_match_the_bundled_nodes(self):
        self.assertEqual(tuple(extra_samplers_register.PACK_SAMPLERS), tuple(xsa.PACK_SAMPLERS))
        self.assertEqual(tuple(pack_schedulers.SCHEDULER_NAMES), tuple(xs.SCHEDULER_NAMES))
        version = tuple(int(part) for part in comfy_node_pack.PACK_VERSION.split("."))
        for core_module, node_class, schema in (
                (xsa, xsa.COMFY_SETTINGS_NODE, extra_samplers_nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES()),
                (xs, xs.COMFY_NODE_CLASS, pack_schedulers.ForgeNeoExtraSchedulers.INPUT_TYPES())):
            with self.subTest(node=node_class):
                self.assertIn(node_class, generation_mappings())
                self.assertEqual(tuple(schema["required"]), tuple(core_module.COMFY_INPUTS))
                self.assertEqual(tuple(schema["optional"]), (core_module.COMFY_REPORT_INPUT,))
                self.assertEqual(core_module.COMFY_REPORT_INPUT, pack_compat.OWNED_NAMES_INPUT)
                self.assertLessEqual(tuple(int(part) for part in core_module.COMFY_MIN_PACK_VERSION.split(".")), version)


def generation_mappings() -> dict:
    from comfy_custom_nodes.ai_studio_forge_parity import NODE_CLASS_MAPPINGS

    return NODE_CLASS_MAPPINGS


def _owner_caps(*, samplers=(), schedulers=(), owned_samplers=None, owned_schedulers=None,
                sampler_node=True, scheduler_node=True) -> dict:
    """Capabilities with extra KSampler choices and the pack's two report nodes (their real ``INPUT_TYPES``,
    ``owned_names`` patched — no ComfyUI). ``owned_*`` None = the pack owns every pack name in the choices."""
    caps = _capabilities()
    required = caps["KSampler"]["input"]["required"]
    required["sampler_name"] = _choice("euler", "euler_ancestral", "dpmpp_2m", *samplers)
    required["scheduler"] = _choice("normal", "karras", "simple", *schedulers)
    if owned_samplers is None:
        owned_samplers = tuple(name for name in samplers if name in xsa.PACK_SAMPLERS)
    if owned_schedulers is None:
        owned_schedulers = tuple(name for name in schedulers if name in xs.SCHEDULER_NAMES)
    if sampler_node:
        with mock.patch.object(extra_samplers_nodes, "owned_names", return_value=tuple(owned_samplers)):
            caps[xsa.COMFY_SETTINGS_NODE] = {
                "input": extra_samplers_nodes.ForgeNeoExtraSamplerSettings.INPUT_TYPES(), "output": ["MODEL"]}
    if scheduler_node:
        with mock.patch.object(pack_schedulers, "owned_names", return_value=tuple(owned_schedulers)):
            caps[xs.COMFY_NODE_CLASS] = {
                "input": pack_schedulers.ForgeNeoExtraSchedulers.INPUT_TYPES(), "output": ["MODEL"]}
    return caps


def _pass_payloads(sampler: str, scheduler: str) -> dict:
    """One payload per app pass that resolves its own sampler/scheduler (``_runtime_sampler_values``)."""
    adetailer = {"ADetailer": {"args": [True, False, {
        "ad_tab_enable": True, "ad_model": "face.pt", "ad_use_sampler": True,
        "ad_sampler": sampler, "ad_scheduler": scheduler,
    }]}}
    sam3 = sam3_args.build_alwayson({
        "sam3_mode": "Inpaint", "sam3_prompt": "face", "sam3_use_sampler": True, "sam3_sampler": sampler,
        "sam3_use_scheduler": True, "sam3_scheduler": scheduler,
    })
    return {
        "base": {"sampler_name": sampler, "scheduler": scheduler},
        "hires": {"enable_hr": True, "hr_sampler_name": sampler, "hr_scheduler": scheduler},
        "adetailer": {"alwayson_scripts": adetailer},
        "sam3": {"alwayson_scripts": sam3},
    }


class PackOwnerCheckTests(unittest.TestCase):
    """Plan R6: a sam-extra name the bundled pack registers runs only when ComfyUI's entry is the pack's — the pack
    node's ``owned_names`` COMBO (read from /object_info) lists it — for every pass and a custom workflow."""

    SAMPLER, SCHEDULER = "er_sde_tunable", "flow_cosmos_rho7"

    def _compile(self, caps, payload, *, workflow=None):
        return ComfyWorkflowCompiler(caps).compile("txt2img", "checkpoint.safetensors", payload, workflow=workflow)

    def _every_pass(self, sampler, scheduler):
        yield from _pass_payloads(sampler, scheduler).items()
        yield "custom", {"sampler_name": sampler, "scheduler": scheduler}

    def _run(self, label, caps, payload):
        return self._compile(caps, payload, workflow=_custom_workflow() if label == "custom" else None)

    def test_owned_names_compile_on_every_pass(self):
        caps = _owner_caps(samplers=(self.SAMPLER,), schedulers=(self.SCHEDULER,))
        for label, payload in self._every_pass("ER SDE (Tunable)", "Flow Cosmos rho7"):
            with self.subTest(label=label):
                graph = self._run(label, caps, payload)
                used = set()
                for node in graph.values():
                    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
                    if node.get("class_type") == "ForgeNeoADetailer":
                        inputs = json.loads(inputs["settings_json"])
                    if "sampler_name" in inputs:
                        used.add((inputs.get("sampler_name"), inputs.get("scheduler")))
                self.assertIn((self.SAMPLER, self.SCHEDULER), used)

    def test_a_foreign_sampler_is_refused_on_every_pass(self):
        caps = _owner_caps(samplers=(self.SAMPLER,), owned_samplers=())
        for label, payload in self._every_pass("ER SDE (Tunable)", "normal"):
            with self.subTest(label=label), self.assertRaisesRegex(
                    WorkflowCompileError, "다른 노드 팩이 'er_sde_tunable' 샘플러를 먼저 등록"):
                self._run(label, caps, payload)

    def test_a_foreign_scheduler_is_refused_on_every_pass(self):
        caps = _owner_caps(schedulers=(self.SCHEDULER,), owned_schedulers=("laplace",))
        for label, payload in self._every_pass("Euler", "Flow Cosmos rho7"):
            with self.subTest(label=label), self.assertRaisesRegex(
                    WorkflowCompileError, "다른 노드 팩이 'flow_cosmos_rho7' 스케줄러를 먼저 등록"):
                self._run(label, caps, payload)

    def test_every_pack_name_is_checked_and_core_names_are_not(self):
        caps = _owner_caps(samplers=(*xsa.PACK_SAMPLERS, *xsa.CORE_SAMPLERS), schedulers=xs.SCHEDULER_NAMES,
                           owned_samplers=(), owned_schedulers=())
        for name in xsa.PACK_SAMPLERS:
            with self.subTest(sampler=name), self.assertRaisesRegex(WorkflowCompileError, f"'{name}' 샘플러"):
                self._compile(caps, {"sampler_name": name})
        for name in xs.SCHEDULER_NAMES:
            with self.subTest(scheduler=name), self.assertRaisesRegex(WorkflowCompileError, f"'{name}' 스케줄러"):
                self._compile(caps, {"scheduler": name})
        for name in xsa.CORE_SAMPLERS:
            with self.subTest(core=name):
                _sid, sampler = _node(self._compile(caps, {"sampler_name": name}), "ForgeNeoKSamplerCNS")
                self.assertEqual(sampler["inputs"]["sampler_name"], name)

    def test_a_stale_pack_without_the_report_node_is_refused(self):
        for kind, caps, payload, pattern in (
                ("sampler", _owner_caps(samplers=(self.SAMPLER,), sampler_node=False),
                 {"sampler_name": "ER SDE (Tunable)"}, "'er_sde_tunable' 샘플러는 앱 노드 팩 1.7.0 의 것이 아닙니다"),
                ("scheduler", _owner_caps(schedulers=(self.SCHEDULER,), scheduler_node=False),
                 {"scheduler": "Flow Cosmos rho7"}, "'flow_cosmos_rho7' 스케줄러는 앱 노드 팩 1.7.0 의 것이 아닙니다")):
            with self.subTest(kind=kind), self.assertRaisesRegex(WorkflowCompileError, pattern + ".*재시작"):
                self._compile(caps, payload)

    def test_a_report_node_without_owned_names_is_a_stale_contract(self):
        caps = _owner_caps(schedulers=(self.SCHEDULER,))
        del caps[xs.COMFY_NODE_CLASS]["input"]["optional"][xs.COMFY_REPORT_INPUT]
        with self.assertRaisesRegex(WorkflowCompileError, "ForgeNeoExtraSchedulers\\(owned_names\\)가 없어"):
            self._compile(caps, {"scheduler": "flow_cosmos_rho7"})

    def test_class_name_only_probes_and_no_capabilities_are_not_checked(self):
        caps = _owner_caps(samplers=(self.SAMPLER,), schedulers=(self.SCHEDULER,))
        caps[xsa.COMFY_SETTINGS_NODE] = {"input": {"required": {}}}
        caps[xs.COMFY_NODE_CLASS] = {"input": {}}
        graph = self._compile(caps, {"sampler_name": "ER SDE (Tunable)", "scheduler": "Flow Cosmos rho7"})
        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual((sampler["inputs"]["sampler_name"], sampler["inputs"]["scheduler"]),
                         (self.SAMPLER, self.SCHEDULER))
        graph = ComfyWorkflowCompiler().compile("txt2img", "checkpoint.safetensors", {
            "sampler_name": "ER SDE (Tunable)", "scheduler": "Flow Cosmos rho7"})
        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual((sampler["inputs"]["sampler_name"], sampler["inputs"]["scheduler"]),
                         (self.SAMPLER, self.SCHEDULER))

    def test_missing_names_get_the_pack_and_comfy_version_hints(self):
        caps = _owner_caps()
        for payload, pattern in (
                ({"sampler_name": "ER SDE (Tunable)"},
                 "지원하지 않는 값입니다: er_sde_tunable \\(앱 노드 팩 1\\.7\\.0 의 sam-extra 샘플러입니다"),
                ({"sampler_name": "Restart (flow)"}, "restart_flow \\(앱 노드 팩 1\\.7\\.0 의 sam-extra 샘플러"),
                ({"scheduler": "Flow Cosmos Dynamic"},
                 "지원하지 않는 값입니다: flow_cosmos_dynamic \\(앱 노드 팩 1\\.7\\.0 의 sam-extra 스케줄러입니다"),
                ({"scheduler": "CosineExponential blend"}, "cosine_exponential \\(앱 노드 팩 1\\.7\\.0"),
                ({"sampler_name": "CFG++ UD10 AB"},
                 "지원하지 않는 값입니다: cfgpp_ud10_ab \\(ComfyUI v0\\.35\\.0 이상이 필요한 코어 샘플러입니다\\)")):
            with self.subTest(payload=payload), self.assertRaisesRegex(WorkflowCompileError, pattern):
                self._compile(caps, payload)
        with self.assertRaises(WorkflowCompileError) as caught:
            self._compile(caps, {"sampler_name": "IPNDM"})
        self.assertTrue(str(caught.exception).endswith("지원하지 않는 값입니다: ipndm"))


class LenientNumberTests(unittest.TestCase):
    def test_overflowing_and_non_finite_values_fall_back(self):
        for value in ("1e400", float("inf"), float("-inf"), float("nan"), "nan", None, "x", [], {}):
            with self.subTest(value=value):
                self.assertEqual(lenient_int(value, 7), 7)
                self.assertEqual(finite_float(value, 1.5), 1.5)

    def test_regular_values(self):
        self.assertEqual(lenient_int("12", 0), 12)
        self.assertEqual(lenient_int(" 12 ", 0), 12)
        self.assertEqual(lenient_int("12.9", 0), 12)
        self.assertEqual(lenient_int(True, 0), 1)
        self.assertEqual(lenient_int(2**64 - 1, 0), 2**64 - 1)
        self.assertEqual(finite_float("0.25", 0.0), 0.25)
        # The compiler no longer crashes on overflow in any numeric field.
        graph = ComfyWorkflowCompiler(_capabilities()).compile(
            "txt2img", "checkpoint.safetensors", {"steps": "1e400", "width": "1e400"},
        )
        _sid, sampler = _node(graph, "ForgeNeoKSamplerCNS")
        self.assertEqual(sampler["inputs"]["steps"], 20)


if __name__ == "__main__":
    unittest.main()
