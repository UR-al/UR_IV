"""Forge sampler labels resolve the same in the app compiler and the node pack (감사 #174).

The bundled node pack is copied into ComfyUI and cannot import core, so the
ADetailer node keeps its own alias table.  These golden values pin both
tables to the same ComfyUI names, including labels the compiler used to reject
("DPM++ 2S a", "UniPC", "DPM fast", "DPM adaptive") and the one the node used
to mangle ("DPM++ 3M SDE").
"""
from __future__ import annotations

import unittest

from comfy_custom_nodes.ai_studio_forge_parity import generation
from core.comfy_workflow_compiler import ComfyWorkflowCompiler, WorkflowCompileError
from core.lenient_numbers import finite_float, lenient_int
from tests.test_comfy_workflow_compiler import _capabilities, _choice, _node

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
}

GOLDEN_SCHEDULERS = {
    "Automatic": "normal", "Use same scheduler": "normal", "Karras": "karras",
    "Exponential": "exponential", "SGM Uniform": "sgm_uniform",
    "DDIM Uniform": "ddim_uniform", "Simple": "simple", "Normal": "normal",
    "Beta": "beta", "Beta57 (RES4LYF)": "beta57", "beta57": "beta57",
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
