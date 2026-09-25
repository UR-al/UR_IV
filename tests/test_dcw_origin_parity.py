"""The pack's DCW(+a) node (``guidance_dcw``) against the original ComfyUI-DCW.

origin: namemechan/ComfyUI-DCW@66aaf9dddb03bad031c1e8443e255a811008e477:dcw_node.py

The original is GPL-3.0, so this repository holds none of its code. The expected
values in ``tests/fixtures/dcw_origin_golden.json`` were produced by running the
original on the deterministic CPU inputs built here (``_recipe_values``). The
generator is the scratch script named in the fixture's ``generator`` field.

Torch-free checks (inputs, auto detection, which hooks are set, the conflict
warning, legacy suite keywords) run in ``--quick``. The tensor checks need torch
(CPU only) and are marked ``requires_torch``.
"""
from __future__ import annotations

import json
import math
import os
import unittest

from comfy_custom_nodes.ai_studio_forge_parity import guidance_dcw
from tests._optional_deps import bind_torch, requires_torch

torch = None  # filled by bind_torch() in the marked class

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "dcw_origin_golden.json")
ORIGIN = "namemechan/ComfyUI-DCW@66aaf9dddb03bad031c1e8443e255a811008e477:dcw_node.py"


def _golden():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


GOLDEN = _golden()


# --- deterministic inputs (same recipe as the fixture generator) ------------
_MASK64 = (1 << 64) - 1


def _recipe_values(count, seed):
    """``count`` values in [-1, 1) from a 64-bit LCG; each is exact in fp32."""
    state = (seed * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & _MASK64
    values = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & _MASK64
        values.append((state >> 40) / 8388608.0 - 1.0)
    return values


def _recipe_tensor(shape, seed, scale=1.0, dtype_name="float32"):
    value = torch.tensor(_recipe_values(math.prod(shape), seed), dtype=torch.float32).reshape(shape)
    if scale != 1.0:
        value = value * scale
    return value.to(getattr(torch, dtype_name))


# --- a fake Comfy model and sampler -----------------------------------------
class _Patcher:
    """Just enough of ModelPatcher: ``clone()``, ``model_options``, ``model``."""

    def __init__(self, inner=None):
        self.model_options = {}
        if inner is not None:
            self.model = inner

    def clone(self):
        twin = _Patcher(getattr(self, "model", None))
        twin.model_options = dict(self.model_options)
        return twin


def _fake_model(case):
    if case.get("no_inner"):
        return _Patcher()
    inner = type(case.get("base", "Anima"), (), {})()
    if case.get("model_type") is not None:
        inner.model_type = case["model_type"]
    if case.get("diffusion") is not None:
        inner.diffusion_model = type(case["diffusion"], (), {})()
    return _Patcher(inner)


def _existing_cfg(args):
    """The 'another node already set sampler_cfg_function' stand-in used by the generator."""
    return args["uncond"] + (args["cond_scale"] * 0.5) * (args["cond"] - args["uncond"])


def _scenario_model(entry):
    model = _fake_model(entry["model"])
    if entry.get("existing_cfg"):
        model.model_options["sampler_cfg_function"] = _existing_cfg
    return model


def _per_run_options(options):
    """Comfy copies model_options for each sampling run (nested dicts and lists)."""
    copied = dict(options)
    for key, value in options.items():
        if isinstance(value, dict):
            copied[key] = _per_run_options(value)
        elif isinstance(value, list):
            copied[key] = list(value)
    return copied


def _sample(options, steps):
    """Run the CFG hook and post-CFG hooks the way Comfy's cfg_function calls them."""
    run_options = _per_run_options(options)
    results = []
    for x, cond_x0, uncond_x0, sigma, scale in steps:
        if "sampler_cfg_function" in run_options:
            result = x - run_options["sampler_cfg_function"]({
                "cond": x - cond_x0, "uncond": x - uncond_x0, "cond_scale": scale,
                "timestep": sigma, "input": x, "sigma": sigma,
                "cond_denoised": cond_x0, "uncond_denoised": uncond_x0,
                "model": None, "model_options": run_options,
                "input_cond": None, "input_uncond": None,
            })
        else:
            result = uncond_x0 + (cond_x0 - uncond_x0) * scale
        for hook in run_options.get("sampler_post_cfg_function", []):
            result = hook({
                "denoised": result, "cond": None, "uncond": None, "cond_scale": scale,
                "model": None, "uncond_denoised": uncond_x0, "cond_denoised": cond_x0,
                "sigma": sigma, "model_options": run_options, "input": x,
            })
        results.append(result)
    return results, run_options


def _scenario_steps(entry):
    shape = [2, 3, 5, 4]
    steps = []
    for index, sigma in enumerate(entry["sigmas"]):
        steps.append((
            _recipe_tensor(shape, 100 + index, 2.0, entry["dtype"]),
            _recipe_tensor(shape, 200 + index, 1.0, entry["dtype"]),
            _recipe_tensor(shape, 300 + index, 1.0, entry["dtype"]),
            torch.tensor([sigma] * shape[0], dtype=torch.float32),
            float(entry["cfg"]),
        ))
    return steps


def _pack_inputs():
    """The pack node's INPUT_TYPES in the fixture's form (tooltips dropped)."""
    rows = []
    for name, spec in guidance_dcw.ForgeNeoDCWCWMSMC.INPUT_TYPES()["required"].items():
        options = dict(spec[1]) if len(spec) > 1 else {}
        options.pop("tooltip", None)
        kind = spec[0] if isinstance(spec[0], str) else list(spec[0])
        rows.append([name, kind, options])
    return rows


def _node(model, **inputs):
    return guidance_dcw.ForgeNeoDCWCWMSMC().patch(model, **inputs)[0]


def _scenario_node(test, entry):
    """Patch for a fixture scenario; the conflict scenario must log its warning."""
    model = _scenario_model(entry)
    if not entry.get("existing_cfg"):
        return model, _node(model, **entry["inputs"])
    with test.assertLogs("ai_studio_forge_parity", level="WARNING"):
        return model, _node(model, **entry["inputs"])


class TestDcwOriginContract(unittest.TestCase):
    """Torch-free: inputs, presets, auto detection, gating, conflict warning."""

    def test_fixture_names_the_original(self):
        self.assertEqual(GOLDEN["origin"], ORIGIN)
        self.assertIn("dcw_parity_check.py", GOLDEN["generator"])
        # Without these the golden tests quietly fall back from exact to tolerant comparison.
        self.assertTrue(GOLDEN.get("torch"))
        self.assertTrue(GOLDEN.get("cpu_capability"))

    def test_input_types_equal_original(self):
        # origin: dcw_node.py:630-817 (INPUT_TYPES: names, order, default/min/max/step/round)
        self.assertEqual(_pack_inputs(), GOLDEN["input_types"])
        self.assertEqual(guidance_dcw.ForgeNeoDCWCWMSMC.RETURN_NAMES, ("model",))

    def test_smc_preset_values(self):
        # origin: dcw_node.py:75-85 (preset table and choice order, Off first)
        self.assertEqual(guidance_dcw._SMC_PRESETS, {
            "SD1.5 / SD2": (5.0, 0.10), "SDXL": (5.0, 0.10), "SD3 / SD3.5": (6.0, 0.10),
            "Flux": (6.0, 0.70), "Qwen-Image": (6.0, 0.10), "Cosmos / Wan": (6.0, 0.20),
            "Custom": (6.0, 0.10),
        })
        self.assertEqual(guidance_dcw.SMC_PRESET_CHOICES[:2], ["Off", "Auto"])

    def test_auto_detection_matches_original(self):
        # origin: dcw_node.py:103-130 (class name, then model_type, then diffusion_model)
        for case in GOLDEN["smc_detect"]:
            with self.subTest(case=case):
                self.assertEqual(guidance_dcw._detect_smc(_fake_model(case)), case["preset"])

    def test_hooks_set_like_original(self):
        # origin: dcw_node.py:854-866 (gating), :868-875 (conflict), :914-915, :962-963
        for entry in GOLDEN["node_runs"]:
            with self.subTest(scenario=entry["name"]):
                model, patched = _scenario_node(self, entry)
                options = patched.model_options
                self.assertEqual(patched is model, entry["returns_input_model"])
                self.assertEqual("sampler_cfg_function" in options, entry["has_cfg_function"])
                self.assertEqual(options.get("sampler_cfg_function") is _existing_cfg,
                                 entry["cfg_function_is_existing"])
                self.assertEqual(bool(options.get("disable_cfg1_optimization", False)),
                                 entry["disable_cfg1_optimization"])
                self.assertEqual(len(options.get("sampler_post_cfg_function", [])),
                                 entry["post_cfg_count"])

    def test_existing_cfg_function_warns_and_keeps_dcw(self):
        # origin: dcw_node.py:868-875 — warn, skip CWM/SMC, still add DCW (:918-963)
        model = _fake_model({"base": "Anima"})
        model.model_options["sampler_cfg_function"] = _existing_cfg
        with self.assertLogs("ai_studio_forge_parity", level="WARNING") as logs:
            patched = _node(
                model, dcw_enabled=True, lambda_l=0.05, lambda_h=0.01,
                cwm_enabled=True, alpha_l=0.3, alpha_h=0.15, smc_preset="Auto",
            )
        self.assertIn("sampler_cfg_function", "\n".join(logs.output))
        self.assertIs(patched.model_options["sampler_cfg_function"], _existing_cfg)
        self.assertNotIn("disable_cfg1_optimization", patched.model_options)
        self.assertEqual(len(patched.model_options["sampler_post_cfg_function"]), 1)
        self.assertNotIn("sampler_post_cfg_function", model.model_options)

    def test_post_cfg_hook_is_appended_after_earlier_hooks(self):
        # origin: dcw_node.py:919, :962-963 — appended to the existing list, input untouched
        model = _fake_model({"base": "Anima"})
        earlier = [lambda args: args["denoised"]]
        model.model_options["sampler_post_cfg_function"] = earlier
        patched = _node(model, dcw_enabled=True, lambda_l=0.05, lambda_h=0.01)
        hooks = patched.model_options["sampler_post_cfg_function"]
        self.assertEqual(len(hooks), 2)
        self.assertIs(hooks[0], earlier[0])
        self.assertEqual(len(earlier), 1)

    def test_rdc_needs_dcw_enabled(self):
        # origin: dcw_node.py:855-856 — rdc_on = tau > 0, and DCW (with RDC) needs dcw_enabled
        model = _fake_model({"base": "Anima"})
        self.assertIs(_node(model, dcw_enabled=False, rdc_tau=0.15), model)
        patched = _node(model, dcw_enabled=True, lambda_l=0.0, lambda_h=0.0, rdc_tau=0.15)
        self.assertEqual(len(patched.model_options["sampler_post_cfg_function"]), 1)
        self.assertIs(_node(model, dcw_enabled=True, lambda_l=0.0, lambda_h=0.0, rdc_tau=0.0), model)

    def test_cwm_needs_a_nonzero_alpha(self):
        # origin: dcw_node.py:858-859 — cwm_enabled with alpha 0/0 sets no CFG hook
        model = _fake_model({"base": "Anima"})
        self.assertIs(_node(model, cwm_enabled=True, alpha_l=0.0, alpha_h=0.0), model)
        patched = _node(model, cwm_enabled=True, alpha_l=0.0, alpha_h=0.1)
        self.assertTrue(patched.model_options["disable_cfg1_optimization"])

    def test_legacy_suite_keywords(self):
        # The guidance suite still sends the older pack spellings.
        model = _fake_model({"base": "Anima"})
        patched = _node(model, dcw_enabled=True, lambda_low=0.1, lambda_high=0.02,
                        rdc_enabled=False, rdc_tau=0.15)
        self.assertNotIn("sampler_cfg_function", patched.model_options)
        self.assertEqual(len(patched.model_options["sampler_post_cfg_function"]), 1)
        # RDC without DCW is off, as in the original.
        self.assertIs(_node(model, dcw_enabled=False, rdc_enabled=True, rdc_tau=0.15), model)
        # smc_enabled=False turns the preset Off.
        self.assertIs(_node(model, smc_enabled=False, smc_preset="Auto"), model)
        patched = _node(model, smc_enabled=True, smc_preset="cosmos / wan")
        self.assertIn("sampler_cfg_function", patched.model_options)
        patched = _node(model, cwm_enabled=True, alpha_low=0.3, alpha_high=0.15)
        self.assertIn("sampler_cfg_function", patched.model_options)
        patched = _node(model, apg_enabled=True)
        self.assertIn("sampler_cfg_function", patched.model_options)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            _node(_fake_model({"base": "Anima"}), smc_preset="Turbo")

    def test_bare_patch_is_identity(self):
        model = object()
        self.assertIs(guidance_dcw.ForgeNeoDCWCWMSMC().patch(model)[0], model)


@requires_torch
class TestDcwOriginGolden(unittest.TestCase):
    """CPU tensors against the numbers the original produced."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        # The fixture stores the original's outputs as 9-significant-digit decimals, which
        # round-trip fp32 exactly (and bf16 widens to fp32 exactly), so on the torch build and
        # CPU kernel set that wrote it the pack must match bit for bit. A looser check hid a
        # bf16 energy weight (the original uses fp32, dcw_node.py:284) behind atol 1e-2.
        # Another build may round reductions differently; there the tolerances are the fallback.
        get_capability = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
        cls.exact = (GOLDEN["torch"] == torch.__version__
                     and get_capability is not None
                     and GOLDEN.get("cpu_capability") == get_capability())

    def assertMatchesGolden(self, actual, expected, dtype_name, label):
        want = torch.tensor(expected, dtype=torch.float32)
        got = actual.detach().float().flatten()
        self.assertEqual(got.shape, want.shape, label)

        def message(detail):
            return f"{label}: {detail}"

        if self.exact:
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=message)
        elif dtype_name == "float32":
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5, msg=message)
        else:
            torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-2, msg=message)

    def test_input_recipe_matches_generator(self):
        sums = GOLDEN["input_checksums"]
        self.assertEqual(_recipe_values(8, 12), sums["seed12_first8"])
        self.assertAlmostEqual(
            float(_recipe_tensor([2, 3, 5, 4], 11).double().sum()),
            sums["seed11_shape_2x3x5x4"], places=9,
        )

    def test_dcw_one_step(self):
        # origin: dcw_node.py:297-418 (apply_dcw), :137-151 (reflect pad), :219-236 (sigma norm)
        for case in GOLDEN["dcw_single"]:
            with self.subTest(case=case["name"]):
                (seed_denoised, seed_live), (scale_denoised, scale_live) = case["seeds"], case["scales"]
                denoised = _recipe_tensor(case["shape"], seed_denoised, scale_denoised, case["dtype"])
                live = _recipe_tensor(case["shape"], seed_live, scale_live, case["dtype"])
                sigma = case["sigma"]
                sigma = torch.tensor(sigma, dtype=torch.float32) if isinstance(sigma, list) else float(sigma)
                out = guidance_dcw.apply_dcw(
                    denoised, live, sigma, case["lambda_l"], case["lambda_h"],
                    rdc_tau=case.get("rdc_tau", 0.0),
                )
                self.assertEqual(str(out.dtype).replace("torch.", ""), case["output_dtype"])
                self.assertMatchesGolden(out, case["output"], case["dtype"], case["name"])

    def test_zero_lambdas_return_the_same_tensor(self):
        # origin: dcw_node.py:338-340 — no RDC and both lambdas 0: denoised itself
        denoised = _recipe_tensor([2, 3, 5, 4], 11)
        live = _recipe_tensor([2, 3, 5, 4], 12, 2.0)
        sigma = torch.tensor([0.5, 0.5])
        self.assertIs(guidance_dcw.apply_dcw(denoised, live, sigma, 0.0, 0.0), denoised)

    def test_rdc_eight_steps(self):
        # origin: dcw_node.py:372-410 (band EMA, beta = 1 - exp(-|ds|/tau), first step seeds)
        spec = GOLDEN["dcw_rdc_steps"]
        state = {}
        for index, sigma in enumerate(spec["sigmas"]):
            out = guidance_dcw.apply_dcw(
                _recipe_tensor(spec["shape"], 400 + index, 1.0),
                _recipe_tensor(spec["shape"], 500 + index, 2.0),
                torch.tensor([sigma] * spec["shape"][0], dtype=torch.float32),
                spec["lambda_l"], spec["lambda_h"], rdc_tau=spec["rdc_tau"],
                rdc_alpha_low=spec["rdc_alpha_ll"], rdc_alpha_high=spec["rdc_alpha_hh"],
                state=state,
            )
            self.assertMatchesGolden(out, spec["outputs"][index], "float32", f"RDC step {index}")
        self.assertEqual(sorted(state), ["HH", "HL", "LH", "LL", "_s_prev"])

    def test_node_runs_match_original(self):
        # origin: dcw_node.py:425-579 (SMC -> CWM, noise space), :821-965 (hooks, per-run state)
        for entry in GOLDEN["node_runs"]:
            with self.subTest(scenario=entry["name"]):
                _, patched = _scenario_node(self, entry)
                steps = _scenario_steps(entry)
                run1, run_options = _sample(patched.model_options, steps)
                self.assertEqual(sorted(k for k in run_options if k.startswith("_")), entry["state_keys"])
                for index, (out, expected) in enumerate(zip(run1, entry["outputs"])):
                    self.assertMatchesGolden(out, expected, entry["dtype"], f"{entry['name']} step {index}")

    def test_two_runs_are_identical(self):
        # origin: dcw_node.py:87-96, :897-899, :946-949 — state lives in the run's model_options
        entry = next(e for e in GOLDEN["node_runs"] if e["name"] == "full_auto_anima")
        patched = _node(_scenario_model(entry), **entry["inputs"])
        steps = _scenario_steps(entry)
        run1, _ = _sample(patched.model_options, steps)
        run2, _ = _sample(patched.model_options, steps)
        for index, (first, second) in enumerate(zip(run1, run2)):
            self.assertTrue(torch.equal(first, second), f"step {index} leaked state between runs")
        self.assertFalse([k for k in patched.model_options if k.startswith("_dcw")])

    def test_cfg_hook_step_failure_falls_back_to_plain_cfg(self):
        # origin: dcw_node.py:901-912 — an error in one step gives plain CFG for that step
        patched = _node(_fake_model({"base": "Anima"}), cwm_enabled=True, alpha_l=0.3, alpha_h=0.15)
        hook = patched.model_options["sampler_cfg_function"]
        cond = _recipe_tensor([2, 3, 5, 4], 1)
        uncond = _recipe_tensor([2, 3, 5, 4], 2)
        options = {}
        with self.assertLogs("ai_studio_forge_parity", level="WARNING"):
            out = hook({"cond": cond, "uncond": uncond, "cond_scale": 4.0, "sigma": "not a sigma",
                        "input": cond, "model_options": options})
        self.assertTrue(torch.equal(out, uncond + 4.0 * (cond - uncond)))
        self.assertIn("_dcw_smc_state", options)

    def test_dcw_hook_step_failure_returns_denoised(self):
        # origin: dcw_node.py:942-960 — missing input returns denoised; an error returns it too
        patched = _node(_fake_model({"base": "Anima"}), dcw_enabled=True, lambda_l=0.05,
                        lambda_h=0.01, rdc_tau=0.15)
        hook = patched.model_options["sampler_post_cfg_function"][-1]
        denoised = _recipe_tensor([2, 3, 5, 4], 3)
        self.assertIs(hook({"denoised": denoised, "input": None, "sigma": torch.tensor([0.5, 0.5]),
                            "model_options": {}}), denoised)
        with self.assertLogs("ai_studio_forge_parity", level="WARNING"):
            out = hook({"denoised": denoised, "input": _recipe_tensor([2, 3, 5, 4], 4),
                        "sigma": "not a sigma", "model_options": {}})
        self.assertIs(out, denoised)

    def test_combined_output_is_finite(self):
        # origin: dcw_node.py:474, :495-504, :536, :574 — nan_to_num(nan/inf -> 0)
        patched = _node(_fake_model({"base": "Anima"}), cwm_enabled=True, alpha_l=0.3,
                        alpha_h=0.15, smc_preset="Custom", smc_lambda=4.0, smc_k=0.3)
        hook = patched.model_options["sampler_cfg_function"]
        cond = _recipe_tensor([2, 3, 5, 4], 5)
        cond[0, 0, 0, 0] = float("nan")
        cond[1, 1, 1, 1] = float("inf")
        uncond = _recipe_tensor([2, 3, 5, 4], 6)
        uncond[0, 2, 3, 1] = float("-inf")
        out = hook({"cond": cond, "uncond": uncond, "cond_scale": 4.0,
                    "sigma": torch.tensor([0.5, 0.5]), "input": cond, "model_options": {}})
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_apg_in_noise_space_equals_x0_space(self):
        # APG is the pack's addition: noise-space hook must equal the x0-space formula.
        patched = _node(_fake_model({"base": "Anima"}), apg_enabled=True, apg_eta=0.3,
                        apg_norm=15.0, apg_momentum=0.5)
        steps = _scenario_steps({"sigmas": [1.0, 0.75, 0.5], "dtype": "float32", "cfg": 4.5})
        results, run_options = _sample(patched.model_options, steps)
        self.assertIn(guidance_dcw.APG_STATE_KEY, run_options)
        state = {}
        for index, ((x, cond_x0, uncond_x0, sigma, scale), got) in enumerate(zip(steps, results)):
            guided = guidance_dcw._apply_apg_error(
                cond_x0 - uncond_x0, cond_x0, eta=0.3, norm_threshold=15.0,
                momentum=0.5, sigma=sigma, state=state,
            )
            expected = uncond_x0 + scale * guided
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5, msg=f"APG step {index}")


if __name__ == "__main__":
    unittest.main()
