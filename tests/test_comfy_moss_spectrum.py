"""Pack 1.9.0 MOSS Spectrum (``moss_spectrum``, ``ForgeNeoMossSpectrum``) without ComfyUI.

* The schedule is upstream's (sorryhyun/ComfyUI-Spectrum-KSampler@ff25e70 ``moss.py`` + ``spectrum.py``): at 50 steps
  ``flex_window`` 0.25 / 0.5 / 1.0 run 22 / 19 / 16 actual steps, with upstream's defaults.
* The forecaster equals an explicit Chebyshev ridge fit blended with the first-order Taylor step (written out here
  from ``_vendor/networks/spectrum_forecast.py``'s ``ChebyshevForecaster``/``SpectrumPredictor``, not imported).
* A run (``SpectrumRun`` through the real APPLY_MODEL / OUTER_SAMPLE wrapper functions, on a small torch stand-in with
  MOSS-Image's head attributes): the step index from the run's sigma list (Detail Daemon's noted sigma first),
  off-grid / behind-the-step / Dy sub-step calls run the model and change nothing, per-branch forecasters, the hook
  only during an actual forward (also when the forward raises), every tensor released when the run ends, a fresh
  state per run, a failing forecast that leaves the rest of the run to the model.
* The SPEED and Euler (SMEA) Dy guards find the node's wrappers; the node leaves non-MOSS models and its input alone;
  registration, version, README and notices.

The same node on the real ComfyUI runtime with a tiny MOSS-Image model: ``tests/test_comfy_moss_spectrum_runtime.py``.
"""
from __future__ import annotations

import gc
import logging
import math
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import comfy_custom_nodes.ai_studio_forge_parity as bundled_pack
from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy_host as dy_host
from comfy_custom_nodes.ai_studio_forge_parity import guidance_common
from comfy_custom_nodes.ai_studio_forge_parity import moss_spectrum as ms
from comfy_custom_nodes.ai_studio_forge_parity import moss_spectrum_nodes as nodes
from core.comfy_node_pack import PACK_VERSION
from tests._moss_spectrum_oracle import explicit_forecast
from tests._optional_deps import bind_torch, requires_torch

torch = None  # filled by bind_torch()

PACK_DIR = Path(bundled_pack.__file__).resolve().parent
UPSTREAM_COMMIT = "ff25e70203cfd9b83c20307474ceff29dfb2139c"
# The actual steps of a plain 50-step run (each step evaluated once, in order) at upstream's defaults.
PLANNED_50 = {
    0.25: [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 16, 19, 22, 25, 29, 33, 37, 41, 46, 47, 48, 49],
    0.5: [0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 19, 23, 28, 33, 39, 45, 47, 48, 49],
    1.0: [0, 1, 2, 3, 4, 5, 7, 10, 14, 19, 25, 32, 40, 47, 48, 49],
}


# --------------------------------------------------------------------------------------------------
# Schedule, grid lookup, branch keys (no torch)
# --------------------------------------------------------------------------------------------------


class ScheduleTests(unittest.TestCase):
    def test_defaults_are_upstreams(self):
        # moss.py:255-265 (apply_moss_spectrum_patch) and :425-432 (the node's flex_window 0.5).
        self.assertEqual(ms.DEFAULTS, {"window_size": 2.0, "flex_window": 0.5, "warmup_steps": 6,
                                       "tail_actual_steps": 3, "blend_w": 0.3, "cheby_degree": 3,
                                       "ridge_lambda": 0.1, "history_size": 100})
        self.assertEqual(ms.SpectrumSettings().ready_after, 5)    # len(t) >= max(2, m + 2), moss.py:112-114

    def test_actual_step_counts_at_50_steps(self):
        for flex, expected in ((0.25, 22), (0.5, 19), (1.0, 16)):
            with self.subTest(flex_window=flex):
                actual = ms.planned_actual_steps(50, ms.SpectrumSettings(flex_window=flex))
                self.assertEqual(len(actual), expected)
                self.assertEqual(actual, PLANNED_50[flex])

    def test_window_rules(self):
        settings = ms.SpectrumSettings(warmup_steps=2, tail_actual_steps=2)
        schedule = ms.WindowSchedule(settings, 10)
        schedule.advance(0)
        self.assertFalse(schedule.should_cache(True))             # warmup
        schedule.advance(1)
        schedule.advance(2)
        self.assertTrue(schedule.should_cache(True))              # (0 + 1) % floor(2.0) != 0
        self.assertFalse(schedule.should_cache(False))            # forecaster not ready
        schedule.mode = ms.CACHED
        schedule.advance(3)
        self.assertEqual(schedule.consec_cached, 1)
        self.assertFalse(schedule.should_cache(True))             # (1 + 1) % 2 == 0: an actual step
        schedule.mode = ms.ACTUAL
        schedule.advance(4)
        self.assertEqual((schedule.consec_cached, schedule.curr_ws, schedule.fwd_count), (0, 2.5, 3))
        schedule.advance(8)
        self.assertFalse(schedule.should_cache(True))             # the last 2 steps run the model
        # the window grows by flex_window rounded to 3 places (spectrum.py:1084)
        grow = ms.WindowSchedule(ms.SpectrumSettings(flex_window=0.1, warmup_steps=0), 50)
        for index in range(4):
            grow.advance(index)
        self.assertEqual(grow.curr_ws, 2.3)

    def test_history_must_exceed_the_degree(self):
        with self.assertRaisesRegex(ValueError, "history_size >= cheby_degree \\+ 2"):
            ms.SpectrumSettings(cheby_degree=3, history_size=4)
        self.assertEqual(ms.SpectrumSettings(cheby_degree=3, history_size=5).history_size, 5)


class LocateStepTests(unittest.TestCase):
    GRID = [1.0, 0.75, 0.5, 0.25, 0.0]

    def test_on_grid_off_grid_and_the_final_sigma(self):
        self.assertEqual(ms.locate_step(1.0, self.GRID, -1), 0)
        self.assertEqual(ms.locate_step(0.5, self.GRID, 1), 2)
        self.assertEqual(ms.locate_step(0.5 + 1e-8, self.GRID, 1), 2)   # float32 round trip
        self.assertIsNone(ms.locate_step(0.625, self.GRID, 1))          # a second-order midpoint
        self.assertIsNone(ms.locate_step(0.0, self.GRID, 3))            # the final sigma is no step
        self.assertIsNone(ms.locate_step(0.4999, self.GRID, 1))

    def test_same_step_and_behind_the_step(self):
        self.assertEqual(ms.locate_step(0.75, self.GRID, 1), 1)         # another call of the current step
        self.assertIsNone(ms.locate_step(0.75, self.GRID, 2))           # back up (restart): not a step

    def test_repeated_sigmas_prefer_the_current_then_the_next_step(self):
        grid = [1.0, 0.5, 0.5, 0.25, 0.0]
        self.assertEqual(ms.locate_step(0.5, grid, 0), 1)
        self.assertEqual(ms.locate_step(0.5, grid, 1), 1)
        self.assertEqual(ms.locate_step(0.5, grid, 2), 2)
        self.assertIsNone(ms.locate_step(0.5, grid, 3))

    def test_a_first_sigma_moved_below_the_grid_start(self):
        # ComfyUI's offset_first_sigma_for_snr: percent_to_sigma(1e-4) on a flow model, about 0.9999 instead of 1.0
        self.assertTrue(ms.offset_start(0.9999, self.GRID, -1))
        self.assertTrue(ms.offset_start(0.8, self.GRID, -1))
        self.assertFalse(ms.offset_start(0.9999, self.GRID, 0))         # a step began: a midpoint of it
        self.assertFalse(ms.offset_start(1.0, self.GRID, -1))           # grid[0] itself (locate_step's)
        self.assertFalse(ms.offset_start(0.75, self.GRID, -1))          # grid[1] itself
        self.assertFalse(ms.offset_start(1.2, self.GRID, -1))           # above the start
        self.assertFalse(ms.offset_start(0.6, self.GRID, -1))           # below grid[1]
        self.assertFalse(ms.offset_start(0.5, [1.0], -1))               # no step


class BranchKeyTests(unittest.TestCase):
    def test_keys_like_upstream(self):
        self.assertEqual(ms.batch_keys(["u1", "u2"], [1, 0]), [(1, "u1"), (0, "u2")])
        self.assertEqual(ms.batch_keys(None, [1, 0]), [1, 0])
        self.assertEqual(ms.batch_keys(["u1"], [1, 0]), [1, 0])        # length mismatch: the ints
        self.assertEqual(ms.batch_keys(5, [0]), [0])                    # not iterable
        self.assertEqual(ms.normalize_cond_or_uncond([0, 1], 4), ([0, 1], True))
        self.assertEqual(ms.normalize_cond_or_uncond([0, 1, 0], 4), ([0, 1, 0], False))
        self.assertEqual(ms.normalize_cond_or_uncond(None, 3), ([0], True))
        self.assertEqual(ms.normalize_cond_or_uncond(1, 2), ([1], True))


# --------------------------------------------------------------------------------------------------
# Guards, node, registration, notices (no torch)
# --------------------------------------------------------------------------------------------------


def _wrapped_options(wrappers: dict) -> dict:
    return {"transformer_options": {"wrappers": wrappers}}


class GuardTests(unittest.TestCase):
    def test_registered_spectrum_wrappers_are_found(self):
        patch = ms.SpectrumPatch(ms.SpectrumSettings())
        ours = {"apply_model": {ms.WRAPPER_KEY: [ms.spectrum_apply_model_wrapper(patch)]},
                "outer_sample": {ms.WRAPPER_KEY: [ms.spectrum_outer_sample_wrapper(patch)]}}
        self.assertTrue(guidance_common.spectrum_wrapper_registered(_wrapped_options(ours)))
        by_mark = {"apply_model": {None: [ms.spectrum_apply_model_wrapper(patch)]}}     # the callable's own mark
        self.assertTrue(guidance_common.spectrum_wrapper_registered(_wrapped_options(by_mark)))
        self.assertEqual((ms.WRAPPER_KEY, guidance_common.MOSS_SPECTRUM_WRAPPER_KEY),
                         ("ai_studio_forge_parity.moss_spectrum", "ai_studio_forge_parity.moss_spectrum"))

        def other_wrapper(executor, *args, **kwargs):
            return executor(*args, **kwargs)

        others = {"sampler_sample": {"ai_studio_forge_parity.speed": [other_wrapper],
                                     "ai_studio_forge_parity.detail_daemon": [other_wrapper]},
                  "apply_model": {None: [other_wrapper]}}
        self.assertFalse(guidance_common.spectrum_wrapper_registered(_wrapped_options(others)))
        for malformed in (None, {}, {"transformer_options": None}, {"transformer_options": {"wrappers": []}},
                          _wrapped_options({"apply_model": [other_wrapper]}),
                          _wrapped_options({"apply_model": {None: other_wrapper}})):
            with self.subTest(options=malformed):
                self.assertFalse(guidance_common.spectrum_wrapper_registered(malformed))

    def test_other_wrappers_that_say_spectrum_are_not_the_node(self):
        """Only the node's own patch counts. Upstream Spectrum-KSampler's Anima modulation guidance (an APPLY_MODEL
        wrapper keyed ``spectrum_mod_guidance_apply``, mod_guidance.py:45, which ForgeNeoAnimaModGuidance calls),
        its DCW / FSG CALC_COND_BATCH wrappers (ComfyUI names a folder node's modules after the folder path,
        nodes.py), and any wrapper under an install path that says "spectrum" forecast no steps: SPEED runs and
        Euler Dy keeps its sub-steps."""
        upstream = r"C:\ComfyUI\custom_nodes\ComfyUI-Spectrum-KSampler"

        def wrapper(module, qualname):
            def hook(executor, *args, **kwargs):
                return executor(*args, **kwargs)

            hook.__module__, hook.__qualname__ = module, qualname
            return hook

        cases = {
            "mod guidance": {"apply_model": {"spectrum_mod_guidance_apply": [
                wrapper(f"{upstream}.mod_guidance", "_mod_apply_wrapper")]}},
            "dcw": {"calc_cond_batch": {None: [wrapper(f"{upstream}.dcw", "_dcw_calc_cond_batch_wrapper")]}},
            "install path": {"sampler_sample": {"ai_studio_forge_parity.speed": [
                wrapper(r"C:\spectrum\ComfyUI\custom_nodes\ai_studio_forge_parity.speed_nodes", "speed_wrapper")]},
                             "outer_sample": {"other.key": [wrapper(r"C:\spectrum\ComfyUI\custom_nodes\x.y", "f")]}},
            "empty entry": {"apply_model": {ms.WRAPPER_KEY: []}},
        }
        model = SimpleNamespace(inner_model=SimpleNamespace(conds={}))
        for name, wrappers in cases.items():
            with self.subTest(case=name):
                self.assertFalse(guidance_common.spectrum_wrapper_registered(_wrapped_options(wrappers)))
                self.assertEqual(dy_host.blocking_reasons(model, {"model_options": _wrapped_options(wrappers)}), [])

    def test_dy_sub_steps_step_aside(self):
        patch = ms.SpectrumPatch(ms.SpectrumSettings())
        extra = {"model_options": _wrapped_options({"apply_model": {ms.WRAPPER_KEY: [
            ms.spectrum_apply_model_wrapper(patch)]}})}
        model = SimpleNamespace(inner_model=SimpleNamespace(conds={}))
        self.assertEqual(dy_host.blocking_reasons(model, extra), [dy_host.REASON_SPECTRUM])
        self.assertIn("Spectrum", dy_host.explain(dy_host.REASON_SPECTRUM, extra))
        self.assertEqual(dy_host.blocking_reasons(model, {"model_options": _wrapped_options({})}), [])

    @requires_torch
    def test_speed_steps_aside(self):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_nodes
        from tests.test_comfy_speed_host import _extra, _guider

        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        extra = _extra()
        self.assertIsNone(speed_nodes.comfy_runtime_guard(_guider(), torch.zeros(1, 4, 8, 8), sigmas, extra))
        patch = ms.SpectrumPatch(ms.SpectrumSettings())
        extra["model_options"]["transformer_options"]["wrappers"] = {
            "apply_model": {ms.WRAPPER_KEY: [ms.spectrum_apply_model_wrapper(patch)]},
            "outer_sample": {ms.WRAPPER_KEY: [ms.spectrum_outer_sample_wrapper(patch)]},
        }
        self.assertEqual(speed_nodes.comfy_runtime_guard(_guider(), torch.zeros(1, 4, 8, 8), sigmas, extra),
                         speed_nodes.SPECTRUM_REASON)
        self.assertIn("Forge Neo MOSS Spectrum", speed_nodes.SPECTRUM_REASON)


class _FakePatcher:
    """ModelPatcher's clone and wrapper API (comfy/model_patcher.py) around a stand-in BaseModel."""

    def __init__(self, base):
        self.model = base
        self.model_options = {"transformer_options": {}}
        self.wrappers: dict = {}

    def clone(self):
        clone = _FakePatcher(self.model)
        clone.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()} for kind, keyed in self.wrappers.items()}
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


def _moss_like_base():
    dit = SimpleNamespace(**{name: object() for name in ms.MOSS_HEAD})
    return SimpleNamespace(diffusion_model=dit, model_sampling=object())


class NodeTests(unittest.TestCase):
    def test_inputs(self):
        types = nodes.ForgeNeoMossSpectrum.INPUT_TYPES()
        self.assertEqual(list(types["required"]), ["model", "enabled", "flex_window", "warmup_steps",
                                                   "tail_actual_steps"])
        self.assertEqual(tuple(types["optional"]), nodes.ADVANCED_INPUTS)
        self.assertEqual(types["required"]["enabled"], ("BOOLEAN", {"default": True,
                                                                    "tooltip": "Off = the MODEL is returned unchanged."}))
        flex = types["required"]["flex_window"][1]
        self.assertEqual((flex["default"], flex["min"], flex["max"], flex["step"]), (0.5, 0.0, 2.0, 0.05))
        self.assertIn("22 / 19 / 16", flex["tooltip"])
        self.assertEqual(types["required"]["warmup_steps"][1]["default"], 6)
        self.assertEqual(types["required"]["tail_actual_steps"][1]["default"], 3)
        for name in nodes.ADVANCED_INPUTS:
            with self.subTest(advanced=name):
                options = types["optional"][name][1]
                self.assertIs(options["advanced"], True)
                self.assertEqual(options["default"], ms.DEFAULTS[name])
                self.assertTrue(options["tooltip"])
        for name, spec in types["required"].items():
            with self.subTest(tooltip=name):
                self.assertTrue(spec[1]["tooltip"])
        self.assertEqual((nodes.ForgeNeoMossSpectrum.RETURN_TYPES, nodes.ForgeNeoMossSpectrum.CATEGORY),
                         (("MODEL",), "AI Studio/Forge Neo parity/Generation"))

    def test_disabled_and_non_moss_models_are_returned_unchanged(self):
        model = _FakePatcher(SimpleNamespace(diffusion_model=SimpleNamespace(blocks=[]), model_sampling=object()))
        node = nodes.ForgeNeoMossSpectrum()
        moss = _FakePatcher(_moss_like_base())
        with self.assertNoLogs("ai_studio_forge_parity", level="WARNING"):
            self.assertIs(node.patch(model, False)[0], model)
            self.assertIs(node.patch(moss, False)[0], moss)
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            self.assertIs(node.patch(model, True)[0], model)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("not MOSS-Image", logs.records[0].getMessage())
        self.assertEqual(model.wrappers, {})
        self.assertTrue(ms.is_moss_model(moss))
        no_sampling = SimpleNamespace(diffusion_model=_moss_like_base().diffusion_model, model_sampling=None)
        self.assertFalse(ms.is_moss_model(SimpleNamespace(model=no_sampling)))
        self.assertFalse(ms.is_moss_model(object()))

    def test_moss_model_is_cloned_with_both_wrappers(self):
        source = _FakePatcher(_moss_like_base())
        before = {kind: {key: list(items) for key, items in keyed.items()} for kind, keyed in source.wrappers.items()}
        patched = nodes.ForgeNeoMossSpectrum().patch(source, True, 0.25, 4, 2, 3.0, 0.5, 2, 0.2)[0]
        self.assertIsNot(patched, source)
        self.assertEqual(source.wrappers, before)                       # the input MODEL is never changed
        self.assertEqual(sorted(patched.wrappers), ["apply_model", "outer_sample"])
        for kind in ("apply_model", "outer_sample"):
            self.assertEqual(list(patched.wrappers[kind]), [ms.WRAPPER_KEY])
            self.assertEqual(len(patched.wrappers[kind][ms.WRAPPER_KEY]), 1)
            self.assertIn("spectrum", patched.wrappers[kind][ms.WRAPPER_KEY][0].__name__)
        patch = ms.model_spectrum_patch(patched)
        self.assertEqual(patch.settings, ms.SpectrumSettings(window_size=3.0, flex_window=0.25, warmup_steps=4,
                                                             tail_actual_steps=2, blend_w=0.5, cheby_degree=2,
                                                             ridge_lambda=0.2))
        again = nodes.ForgeNeoMossSpectrum().patch(patched, True)[0]   # replaces, never stacks
        self.assertEqual([len(again.wrappers[kind][ms.WRAPPER_KEY]) for kind in ("apply_model", "outer_sample")],
                         [1, 1])
        self.assertEqual(ms.model_spectrum_patch(again).settings, ms.SpectrumSettings())
        self.assertIsNone(ms.model_spectrum_patch(source))
        self.assertNotIn(ms.RUN_KEY, patched.model_options["transformer_options"])


class RegistrationTests(unittest.TestCase):
    def test_node_is_registered_and_named(self):
        self.assertIs(bundled_pack.NODE_CLASS_MAPPINGS[nodes.NODE_ID], nodes.ForgeNeoMossSpectrum)
        self.assertEqual(nodes.NODE_ID, "ForgeNeoMossSpectrum")
        self.assertEqual(bundled_pack.NODE_DISPLAY_NAME_MAPPINGS[nodes.NODE_ID],
                         "Forge Neo MOSS Spectrum (Spectrum-KSampler)")
        self.assertEqual(nodes.NODE_CLASS_MAPPINGS, {nodes.NODE_ID: nodes.ForgeNeoMossSpectrum})

    def test_module_names(self):
        names = sorted(path.name for path in PACK_DIR.glob("*spectrum*.py"))
        self.assertEqual(names, ["moss_spectrum.py", "moss_spectrum_nodes.py"])
        self.assertFalse(any(name.startswith("speed_") for name in names))   # not a sam-extra SPEED port module

    def test_version_lockstep(self):
        self.assertEqual(bundled_pack.__version__, "1.9.0")
        self.assertEqual(PACK_VERSION, bundled_pack.__version__)
        source = (Path(__file__).resolve().parent.parent / "core" / "comfy_node_pack.py").read_text(encoding="utf-8")
        self.assertIn("# 1.9.0: new ForgeNeoMossSpectrum", source)

    def test_readme(self):
        readme = (PACK_DIR / "README.md").read_text(encoding="utf-8")
        custom = readme.split("### Custom ComfyUI workflows only", 1)[1].split("##", 1)[0]
        self.assertIn("`ForgeNeoMossSpectrum`", custom)
        changes = readme.split("## 1.9.0 changes", 1)[1].split("\n## ", 1)[0]
        for text in ("ff25e70", "22 / 19 / 16", "`flex_window`", "`APPLY_MODEL`", "`OUTER_SAMPLE`",
                     "not MOSS-Image is returned unchanged", "SkipLayerGuidanceDiTSimple"):
            with self.subTest(text=text):
                self.assertIn(text, " ".join(changes.split()))
        self.assertLess(readme.index("## 1.9.0 changes"), readme.index("## 1.8.0 changes"))

    def test_notices_and_licenses(self):
        notices = (PACK_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        spectrum = notices.split("## MOSS Spectrum (pack 1.9.0)", 1)[1].split("\n## ", 1)[0]
        for text in (UPSTREAM_COMMIT, "`moss.py`", "`spectrum.py`", "`_vendor/networks/spectrum_forecast.py`",
                     "MIT", "`LICENSES/ComfyUI-Spectrum-KSampler-MIT.txt`", "`moss_spectrum.py`",
                     "`moss_spectrum_nodes.py`", "never imported"):
            with self.subTest(spectrum=text):
                self.assertIn(text, spectrum)
        head = notices.split("## MOSS-Image head (pack 1.9.0)", 1)[1]
        for text in ("OpenMOSS/MOSS-Image", "0.1.1", "198-209", "242-245", "Apache License, Version 2.0",
                     "`LICENSES/moss-image-Apache-2.0.txt`", "`moss_spectrum.moss_head`"):
            with self.subTest(head=text):
                self.assertIn(text, " ".join(head.split()))
        apache = (PACK_DIR / "LICENSES" / "moss-image-Apache-2.0.txt").read_text(encoding="utf-8")
        self.assertIn("Apache License", apache)
        self.assertIn("Version 2.0, January 2004", apache)
        self.assertIn("END OF TERMS AND CONDITIONS", apache)
        mit = (PACK_DIR / "LICENSES" / "ComfyUI-Spectrum-KSampler-MIT.txt").read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2026 sorryhyun", mit)
        # the installed node's model.py against this hash: tests/test_comfy_moss_spectrum_runtime.py
        self.assertRegex(" ".join(head.split()), r"`model.py`\s+SHA-256 `[0-9a-f]{64}`")

    def test_module_docstring_names_the_origins(self):
        doc = " ".join((ms.__doc__ or "").split())
        for text in (f"sorryhyun/ComfyUI-Spectrum-KSampler@{UPSTREAM_COMMIT}:moss.py:57-103", "spectrum.py:139-154",
                     ":488-506", ":1036-1049", ":1052-1092", "moss-image 0.1.1", "lines 198-209", "242-245",
                     "Apache-2.0"):
            with self.subTest(text=text):
                self.assertIn(text, doc)


# --------------------------------------------------------------------------------------------------
# Forecaster and a run on a torch stand-in
# --------------------------------------------------------------------------------------------------


def _explicit_forecast(ts, hs, t_star, m, lam, w, total):
    """The Chebyshev ridge fit blended with the Newton forward difference, in float64 (tests/_moss_spectrum_oracle)."""
    return explicit_forecast(torch, ts, hs, t_star, m, lam, w, total)


@requires_torch
class ForecasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _history(self, count, dtype=None):
        generator = torch.Generator().manual_seed(3)
        base = torch.randn(1, 6, 4, generator=generator)
        drift = torch.randn(1, 6, 4, generator=generator)
        return [(base + 0.1 * i * drift + 0.01 * torch.randn(1, 6, 4, generator=generator)).to(dtype or torch.float32)
                for i in range(count)]

    def test_prediction_equals_the_explicit_fit_and_blend(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0]
        hs = self._history(len(ts))
        for m, lam, w, t_star in ((3, 0.1, 0.3, 10.0), (3, 0.1, 1.0, 11.0), (2, 0.5, 0.0, 12.0), (4, 0.01, 0.6, 8.0)):
            with self.subTest(m=m, lam=lam, w=w, t_star=t_star):
                forecaster = ms.Forecaster(m, lam, w, 50, 100)
                for t, h in zip(ts, hs):
                    forecaster.update(t, h)
                predicted = forecaster.predict(t_star)
                expected = _explicit_forecast(ts, hs, t_star, m, lam, w, 50)
                self.assertEqual(predicted.dtype, torch.float32)
                self.assertTrue(torch.allclose(predicted.double(), expected, rtol=1e-6, atol=1e-6),
                                (predicted - expected).abs().max())

    def test_history_window_and_low_precision(self):
        forecaster = ms.Forecaster(3, 0.1, 0.3, 50, 5)
        hs = self._history(8, torch.bfloat16)
        for t, h in enumerate(hs):
            forecaster.update(float(t), h)
        self.assertEqual(forecaster.t, [3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(len(forecaster.h), 5)
        predicted = forecaster.predict(8.0)
        self.assertEqual(predicted.dtype, torch.bfloat16)                # fp32 sum, cast back (moss.py:98-103)
        expected = _explicit_forecast([3.0, 4.0, 5.0, 6.0, 7.0], hs[3:], 8.0, 3, 0.1, 0.3, 50)
        self.assertTrue(torch.allclose(predicted.double(), expected, rtol=1e-2, atol=1e-2))
        forecaster.clear()
        self.assertEqual((forecaster.t, forecaster.h, forecaster.shape), ([], [], None))


def _stand_in_classes():
    class StandInMossDiT(torch.nn.Module):
        """MOSS-Image's head attributes and per-sample forward shape; one timestep-dependent block."""

        def __init__(self, channels=4, dim=8, freq_dim=16):
            super().__init__()
            generator = torch.Generator().manual_seed(11)

            def linear(fan_in, fan_out, bias=True):
                layer = torch.nn.Linear(fan_in, fan_out, bias=bias)
                with torch.no_grad():
                    layer.weight.copy_(torch.randn(fan_out, fan_in, generator=generator) * 0.3)
                    if bias:
                        layer.bias.copy_(torch.randn(fan_out, generator=generator) * 0.1)
                return layer

            self.freq_dim = freq_dim
            self.video_embedder = linear(channels, dim)
            self.layers = torch.nn.ModuleList([linear(dim, dim)])
            self.time_embedding = torch.nn.Sequential(linear(freq_dim, dim), torch.nn.SiLU(), linear(dim, dim))
            self.video_head_modulation = linear(dim, 2 * dim)
            self.final_norm_video = torch.nn.RMSNorm(dim, eps=1e-6)
            self.final_linear_video = linear(dim, channels, bias=False)
            self.forwards = 0
            self.fail = False

        def forward(self, x, timestep):
            if self.fail:
                raise RuntimeError("model failed")
            self.forwards += 1
            batch, _, height, width = x.shape
            times = timestep.to(torch.float32).reshape(-1).expand(batch)
            half = self.freq_dim // 2
            freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
            phase = times.double()[:, None] * freq
            time = self.time_embedding(torch.cat((phase.cos(), phase.sin()), -1).float()).float()
            head = self.video_head_modulation(time).view(batch, 2, -1)
            outputs = []
            for index in range(batch):
                tokens = self.video_embedder(x[index:index + 1].flatten(2).transpose(1, 2))
                tokens = tokens + torch.tanh(self.layers[0](tokens)) * time[index:index + 1, None]
                y = self.final_norm_video(tokens)
                y = y * (1 + head[index:index + 1, 1, None]) + head[index:index + 1, 0, None]
                outputs.append(self.final_linear_video(y).transpose(1, 2).reshape(1, -1, height, width))
            return torch.cat(outputs)

    class FlowSampling:
        def timestep(self, sigma):
            return sigma * 1000.0

        def calculate_denoised(self, sigma, model_output, model_input):
            return model_input - model_output * sigma.reshape(-1, *([1] * (model_output.ndim - 1)))

    class StandInBase:
        def __init__(self):
            self.diffusion_model = StandInMossDiT()
            self.model_sampling = FlowSampling()

        def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
            out = self.diffusion_model(x, self.model_sampling.timestep(t).float())
            return self.model_sampling.calculate_denoised(t, out.float(), x)

    class Executor:
        """``WrapperExecutor`` at the end of the APPLY_MODEL chain: ``class_obj`` is the BaseModel."""

        def __init__(self, base):
            self.class_obj = base

        def __call__(self, *args, **kwargs):
            return self.class_obj._apply_model(*args, **kwargs)

    return StandInBase, Executor


@requires_torch
class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.Base, cls.Executor = _stand_in_classes()

    def setUp(self):
        self.base = self.Base()
        self.executor = self.Executor(self.base)
        self.settings = ms.SpectrumSettings(warmup_steps=3, tail_actual_steps=2)
        self.patch = ms.SpectrumPatch(self.settings)
        self.apply = ms.spectrum_apply_model_wrapper(self.patch)
        self.grid = torch.linspace(1.0, 0.0, 21)

    def _options(self, run, **extra):
        options = {ms.RUN_KEY: run, "sample_sigmas": self.grid, "cond_or_uncond": [0, 1], "uuids": ["c", "u"]}
        options.update(extra)
        return options

    def _call(self, x, sigma, options):
        t = torch.full((x.shape[0],), float(sigma))
        return self.apply(self.executor, x, t, None, None, None, options)

    def _euler(self, run, x, steps=None, extra=None):
        """Euler over the grid, cond and uncond batched (batch 2 = the two branches); (model input, output) per step."""
        outputs = []
        for index in range(len(self.grid) - 1 if steps is None else steps):
            sigma, following = self.grid[index], self.grid[index + 1]
            batched = torch.cat([x, x])
            denoised = self._call(batched, sigma, self._options(run, **(extra or {})))
            outputs.append((batched, denoised))
            cond = denoised[:1]
            x = x + (x - cond) / sigma * (following - sigma) if sigma > 0 else x
        return x, outputs

    def _x(self):
        return torch.randn(1, 4, 3, 2, generator=torch.Generator().manual_seed(9))

    def test_a_run_follows_the_planned_schedule_and_forecasts_without_the_model(self):
        run = ms.SpectrumRun(self.settings)
        self._euler(run, self._x())
        planned = ms.planned_actual_steps(20, self.settings)
        self.assertEqual(run.report.actual_steps(), planned)
        self.assertEqual(run.report.cached_steps(), sorted(set(range(20)) - set(planned)))
        self.assertEqual(self.base.diffusion_model.forwards, len(planned))   # cached steps never ran the model
        self.assertEqual((run.report.steps, run.report.model, run.report.forecast, run.report.branches),
                         (20, len(planned), 20 - len(planned), 2))
        self.assertEqual(sorted(run.forecasters), [(0, "c"), (1, "u")])
        self.assertEqual(len(run.forecasters[(0, "c")].t), len(planned))
        self.assertEqual(run.forecasters[(0, "c")].shape, (1, 6, 8))
        self.assertEqual(dict(self.base.diffusion_model.final_norm_video._forward_pre_hooks), {})

    def test_each_forecast_is_mosses_head_on_the_explicit_branch_forecasts(self):
        run = ms.SpectrumRun(self.settings)
        _, outputs = self._euler(run, self._x())
        cached = run.report.cached_steps()
        self.assertGreaterEqual(len(cached), 5)
        for step in cached:
            with self.subTest(step=step):
                feats = []
                for key in [(0, "c"), (1, "u")]:     # each branch from its own history before the step
                    forecaster = run.forecasters[key]
                    ts = [t for t in forecaster.t if t < step]
                    hs = [h for t, h in zip(forecaster.t, forecaster.h) if t < step]
                    feats.append(_explicit_forecast(ts, hs, float(step), 3, 0.1, 0.3, 20).float())
                model_input, out = outputs[step]
                sigma = torch.full((2,), float(self.grid[step]))
                head = ms.moss_head(self.base.diffusion_model, sigma * 1000.0, torch.cat(feats), 3, 2)
                expected = model_input - head.float() * sigma.reshape(-1, 1, 1, 1)
                self.assertTrue(torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (out - expected).abs().max())

    def test_head_equals_the_models_own_output(self):
        dit = self.base.diffusion_model
        x = torch.randn(2, 4, 3, 2, generator=torch.Generator().manual_seed(4))
        timestep = torch.tensor([420.0, 420.0])
        feats = []
        handle = dit.final_norm_video.register_forward_pre_hook(lambda module, inputs: feats.append(inputs[0].clone()))
        try:
            expected = dit(x, timestep)
        finally:
            handle.remove()
        actual = ms.moss_head(dit, timestep, torch.cat(feats), 3, 2)
        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-6))
        feats.clear()
        handle = dit.final_norm_video.register_forward_pre_hook(lambda module, inputs: feats.append(inputs[0].clone()))
        try:
            single = dit(x[:1], timestep[:1])
        finally:
            handle.remove()
        self.assertTrue(torch.equal(ms.moss_head(dit, timestep[:1], feats[0], 3, 2), single))   # same ops, same bits

    def test_off_grid_and_behind_the_step_calls_run_the_model_and_change_nothing(self):
        run = ms.SpectrumRun(self.settings)
        x = torch.cat([self._x(), self._x()])
        self._call(x, self.grid[0], self._options(run))
        self._call(x, self.grid[1], self._options(run))
        before = (run.schedule.step_idx, run.schedule.mode, run.schedule.consec_cached,
                  [len(f.t) for f in run.forecasters.values()])
        forwards = self.base.diffusion_model.forwards
        midpoint = (self.grid[1] + self.grid[2]) / 2
        out = self._call(x, midpoint, self._options(run))
        self._call(x, self.grid[0], self._options(run))                  # back up: a restart segment
        self.assertEqual(self.base.diffusion_model.forwards, forwards + 2)
        self.assertEqual(run.report.off_grid, 2)
        self.assertEqual((run.schedule.step_idx, run.schedule.mode, run.schedule.consec_cached,
                          [len(f.t) for f in run.forecasters.values()]), before)
        expected = self.base._apply_model(x, torch.full((2,), float(midpoint)))
        self.assertTrue(torch.equal(out, expected))
        self.assertEqual(dict(self.base.diffusion_model.final_norm_video._forward_pre_hooks), {})

    def test_a_branch_keeps_one_observation_per_step(self):
        """Heun's pattern: the corrector calls the next step's sigma (beginning that step) with another input, then
        that step's own call follows. Each branch records the step once; the forecasts stay finite."""
        run = ms.SpectrumRun(self.settings)
        x = torch.cat([self._x(), self._x()])
        outputs = []
        for index in range(20):
            outputs.append(self._call(x, self.grid[index], self._options(run)))
            if index + 1 < 20:
                outputs.append(self._call(x * 0.9, self.grid[index + 1], self._options(run)))
        planned = ms.planned_actual_steps(20, self.settings)
        self.assertEqual(run.report.actual_steps(), planned)
        for forecaster in run.forecasters.values():
            self.assertEqual(forecaster.t, [float(step) for step in planned])
        self.assertEqual(run.report.forecast, 2 * (20 - len(planned)))       # a cached step's corrector and own call
        self.assertTrue(all(torch.isfinite(out).all() for out in outputs))
        # the first call of a step is its observation: the corrector's features at step 1, not step 1's own call
        first = run.forecasters[(0, "c")].h[1]
        replay = []
        handle = self.base.diffusion_model.final_norm_video.register_forward_pre_hook(
            lambda module, inputs: replay.append(inputs[0].clone()))
        try:
            self.base._apply_model(x * 0.9, torch.full((2,), float(self.grid[1])))
        finally:
            handle.remove()
        self.assertTrue(torch.equal(first, replay[0]))

    def _separate(self, run, x, sigma, branches=((0, "c"), (1, "u"))):
        """One call per branch (MOSS's cond and uncond texts differ in length: ComfyUI cannot batch them)."""
        return [self._call(x, sigma, self._options(run, cond_or_uncond=[branch], uuids=[uid]))
                for branch, uid in branches]

    def test_a_first_sigma_moved_below_the_grid_start_is_step_zero(self):
        """ComfyUI's SDE and exponential-integrator samplers start a full-denoise flow run at
        ``percent_to_sigma(1e-4)`` (offset_first_sigma_for_snr), about 0.9999 instead of 1.0. Upstream counts each
        new sigma, so those calls are its step 0: here as well, with the planned schedule and a t = 0 observation."""
        run = ms.SpectrumRun(self.settings)
        x = self._x()
        for index in range(20):
            self._separate(run, x, 0.9999 if index == 0 else float(self.grid[index]))
        planned = ms.planned_actual_steps(20, self.settings)
        self.assertEqual(run.report.actual_steps(), planned)
        self.assertEqual((run.report.off_grid, run.report.steps, sorted(run.report.step_modes)),
                         (0, 20, list(range(20))))
        self.assertEqual(sorted(run.forecasters), [(0, "c"), (1, "u")])
        for forecaster in run.forecasters.values():
            self.assertEqual(forecaster.t, [float(step) for step in planned])
        self.assertTrue(run.report.summary().startswith(
            f"{len(planned)}/20 steps ran the model, {20 - len(planned)} forecast"))     # all 20 steps accounted

    def test_only_the_first_calls_sigma_stands_for_step_zero(self):
        run = ms.SpectrumRun(self.settings)
        x = self._x()
        self._separate(run, x, 0.9999)                                    # step 0 (cond, uncond)
        self._separate(run, x, 0.98, ((0, "c"),))                         # a midpoint of step 0: off the grid
        self._separate(run, x, 0.9999, ((1, "u"),))                       # step 0 again
        self.assertEqual((run.schedule.step_idx, run.report.off_grid), (0, 1))
        self._separate(run, x, float(self.grid[1]))
        self._separate(run, x, 0.9999, ((0, "c"),))                       # behind the step: off the grid
        self.assertEqual((run.schedule.step_idx, run.report.off_grid), (1, 2))
        self.assertEqual([len(f.t) for f in run.forecasters.values()], [2, 2])
        late = ms.SpectrumRun(self.settings)
        self._separate(late, x, float(self.grid[0]))
        self._separate(late, x, 0.9999, ((0, "c"),))                      # after grid[0] began step 0: a midpoint
        self.assertEqual((late.schedule.step_idx, late.report.off_grid), (0, 1))

    def test_a_branch_that_appears_mid_run_is_forecast_only_when_ready(self):
        """A negative that starts mid-run (ConditioningSetTimestepRange, prompt editing) is a new branch: on cached
        steps it runs the model until it holds ``ready_after`` observations (moss.py:112-114), never forecast from
        fewer (upstream's cached path checks only that the branch has a forecaster)."""
        run = ms.SpectrumRun(self.settings)
        x = self._x()
        predicted = []
        original = ms.Forecaster.predict

        def recording(forecaster, t_star):
            predicted.append((t_star, len(forecaster.t)))
            return original(forecaster, t_star)

        with mock.patch.object(ms.Forecaster, "predict", recording):
            for index in range(20):
                branches = ((0, "c"), (1, "late")) if index >= 6 else ((0, "c"),)
                outputs = self._separate(run, x, float(self.grid[index]), branches)
                self.assertTrue(all(torch.isfinite(out).all() for out in outputs))
        self.assertTrue(predicted)
        self.assertEqual([(t, n) for t, n in predicted if n < self.settings.ready_after], [])
        self.assertEqual(run.forecasters[(1, "late")].t[:5], [6.0, 7.0, 8.0, 9.0, 10.0])   # ran the model 5 times
        self.assertTrue(any(step > 10 for step in run.report.cached_steps()))               # then forecast again

    def test_detail_daemons_noted_sigma_is_the_step(self):
        run = ms.SpectrumRun(self.settings)
        x = torch.cat([self._x(), self._x()])
        for index in range(4):
            scaled = self.grid[index] * 0.93                                 # the sigma the model gets
            options = self._options(run, **{guidance_common.PRE_DD_SIGMAS_KEY: self.grid[index].reshape(1)})
            self._call(x, scaled, options)
        self.assertEqual((run.report.off_grid, run.schedule.step_idx), (0, 3))
        plain = ms.SpectrumRun(self.settings)
        self._call(x, self.grid[1] * 0.93, self._options(plain))
        self.assertEqual((plain.report.off_grid, plain.schedule.step_idx), (1, -1))

    def test_dy_sub_steps_and_runs_without_a_sigma_list_pass_through(self):
        run = ms.SpectrumRun(self.settings)
        x = torch.cat([self._x(), self._x()])
        self._call(x, self.grid[0], self._options(run, **{guidance_common.SUBSTEP_MARKER: "dy"}))
        self._call(x, self.grid[0], self._options(run, sample_sigmas=None))
        self.assertEqual((run.report.passed, run.schedule), (2, None))
        # no run under RUN_KEY (another sampling path, a different patch): the model, untouched
        other = ms.SpectrumRun(ms.SpectrumSettings())
        out = self._call(x, self.grid[0], self._options(other))
        self.assertTrue(torch.equal(out, self.base._apply_model(x, torch.full((2,), float(self.grid[0])))))
        self.assertEqual(other.report.evaluations, 0)

    def test_the_hook_is_removed_when_the_forward_raises(self):
        run = ms.SpectrumRun(self.settings)
        self.base.diffusion_model.fail = True
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            self._call(torch.cat([self._x(), self._x()]), self.grid[0], self._options(run))
        self.assertEqual(dict(self.base.diffusion_model.final_norm_video._forward_pre_hooks), {})

    def test_a_failing_forecast_hands_the_rest_of_the_run_to_the_model(self):
        run = ms.SpectrumRun(self.settings)
        original = ms.moss_head

        def broken(*args, **kwargs):
            raise ValueError("bad head")

        ms.moss_head = broken
        try:
            with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
                self._euler(run, self._x())
        finally:
            ms.moss_head = original
        self.assertEqual(len(logs.records), 1)
        self.assertIn("ValueError: bad head", run.report.error)
        self.assertEqual(run.forecasters, {})
        first_cached = min(set(range(20)) - set(ms.planned_actual_steps(20, self.settings)))
        self.assertEqual(self.base.diffusion_model.forwards, 20)          # every step ran the model
        self.assertEqual((run.report.forecast, run.report.passed, run.report.model), (0, 20 - first_cached,
                                                                                         first_cached))
        self.assertEqual(run.report.actual_steps(), list(range(first_cached)))

    def test_close_releases_every_tensor(self):
        run = ms.SpectrumRun(self.settings)
        self._euler(run, self._x())
        refs = [weakref.ref(h) for forecaster in run.forecasters.values() for h in forecaster.h]
        self.assertTrue(refs)
        report = run.close()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.assertEqual((run.forecasters, run.closed), ({}, True))
        self.assertIs(report, run.report)
        x = torch.cat([self._x(), self._x()])
        forwards = self.base.diffusion_model.forwards
        self._call(x, self.grid[5], self._options(run))                    # a closed run only passes through
        self.assertEqual((self.base.diffusion_model.forwards, run.report.passed), (forwards + 1, 1))

    def test_outer_sample_opens_a_fresh_run_and_restores_the_options(self):
        outer = ms.spectrum_outer_sample_wrapper(self.patch)
        model_options = {"transformer_options": {"keep": 1}, "other": [1]}
        guider = SimpleNamespace(model_options=model_options)
        seen = []
        test = self

        class OuterExecutor:
            class_obj = guider

            def __call__(self, *args, **kwargs):
                options = guider.model_options
                seen.append(options)
                run = options["transformer_options"][ms.RUN_KEY]
                test.assertIsNot(options, model_options)
                test.assertEqual(run.report.evaluations, 0)                  # fresh, step 0 again
                transformer = dict(options["transformer_options"], sample_sigmas=test.grid[:steps + 1],
                                   cond_or_uncond=[0], uuids=["c"])
                x = test._x()
                for index in range(steps):
                    test.apply(test.executor, x, test.grid[index].reshape(1), None, None, None, transformer)
                return "samples"

        for steps in (12, 6):
            self.assertEqual(outer(OuterExecutor()), "samples")
            self.assertIs(guider.model_options, model_options)
            self.assertNotIn(ms.RUN_KEY, model_options["transformer_options"])
            report = self.patch.last_report
            self.assertEqual(report.steps, steps)
            self.assertEqual(report.actual_steps(), ms.planned_actual_steps(steps, self.settings))
        self.assertIsNot(seen[0]["transformer_options"][ms.RUN_KEY], seen[1]["transformer_options"][ms.RUN_KEY])
        self.assertTrue(all(options["transformer_options"][ms.RUN_KEY].closed for options in seen))
        self.assertTrue(all(options["transformer_options"][ms.RUN_KEY].forecasters == {} for options in seen))

    def test_outer_sample_releases_on_an_exception(self):
        outer = ms.spectrum_outer_sample_wrapper(self.patch)
        guider = SimpleNamespace(model_options={"transformer_options": {}})
        runs = []

        class Failing:
            class_obj = guider

            def __call__(self, *args, **kwargs):
                runs.append(guider.model_options["transformer_options"][ms.RUN_KEY])
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            outer(Failing())
        self.assertTrue(runs[0].closed)
        self.assertEqual(guider.model_options, {"transformer_options": {}})

    def test_summary_line(self):
        report = ms.RunReport(steps=50, evaluations=100, forecast=62, model=38, off_grid=2,
                              step_modes={**{i: ms.ACTUAL for i in range(19)}, **{i: ms.CACHED for i in range(19, 50)}})
        self.assertEqual(report.summary(), "19/50 steps ran the model, 31 forecast (62 of 100 model calls); "
                                           "2 off-grid calls ran the model")
        self.assertTrue(math.isfinite(len(report.summary())))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
