"""The pack's CNS (``guidance_cns``) against the original CNS Sampler Patch.

origin: namemechan/comfyui-cns_sampler_patch@42278b138284f7a8685ef174af0a50fe03246dd0:cns_sampler_patch.py

The original is GPL-3.0, so this repository holds none of its code. The numbers
in ``tests/fixtures/cns_origin_golden.json`` were produced by running the
original's ``CNSSamplerPatch`` on the fake k-diffusion sampler and deterministic
CPU inputs defined here (``_pseudo_normal``, ``_smooth_pattern``,
``_make_kd_module``); the scratch generator named in the fixture's
``generator`` field holds an identical copy of that harness. The fixture's
``nested`` numbers come from the original on keyword-delegating samplers
(``_make_nested_kd_module``), and ``dd_chain`` from the original's working graph
CNSSamplerPatch -> DetailDaemonSampler
(Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b, MIT).

Torch-free checks (node inputs, wrapper registration and swapping, KSampler /
Hires / SPEED wiring) run in ``--quick``. The tensor checks need torch (CPU
only) and are marked ``requires_torch``.
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from comfy_custom_nodes.ai_studio_forge_parity import generation, guidance_cns
from tests._optional_deps import bind_torch, requires_torch

torch = None  # filled by bind_torch() in the marked classes

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "cns_origin_golden.json")
ORIGIN = "namemechan/comfyui-cns_sampler_patch@42278b138284f7a8685ef174af0a50fe03246dd0:cns_sampler_patch.py"

# origin: cns_sampler_patch.py:396-437 (INPUT_TYPES: default, min, max, step, round)
ORIGIN_INPUTS = {
    "strength": {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01},
    "gamma_power": {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "round": 0.01},
    "gamma_scale": {"default": 2.0, "min": 0.1, "max": 25.0, "step": 0.1, "round": 0.01},
}
RANGE_KEYS = ("default", "min", "max", "step", "round")


def _golden():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


GOLDEN = _golden()
CASES = {case["name"]: case for case in GOLDEN["cases"]}


# --- fakes that need no torch -------------------------------------------------
class _FakePatcher:
    """ModelPatcher's clone/wrapper API (comfy/model_patcher.py add/remove_wrappers_with_key)."""

    def __init__(self):
        self.wrappers = {}
        self.model_options = {}

    def clone(self):
        clone = _FakePatcher()
        clone.wrappers = {kind: {key: list(items) for key, items in keyed.items()}
                          for kind, keyed in self.wrappers.items()}
        clone.model_options = dict(self.model_options)
        return clone

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


def _cns_wrappers(model):
    return model.wrappers.get("sampler_sample", {}).get(guidance_cns.CNS_WRAPPER_KEY, [])


class _Executor:
    """WrapperExecutor stand-in: ``class_obj`` is the KSAMPLER, calling runs ``sample``."""

    def __init__(self, sampler, sample):
        self.class_obj = sampler
        self._sample = sample

    def __call__(self, *args, **kwargs):
        return self._sample(self.class_obj, *args, **kwargs)


class TestCnsNodeContract(unittest.TestCase):
    """Inputs, registration and wiring — no torch."""

    def assertOriginInput(self, spec, name, label):
        self.assertEqual(spec[0], "FLOAT", label)
        self.assertEqual({key: spec[1][key] for key in RANGE_KEYS}, ORIGIN_INPUTS[name], label)

    def test_sampler_patch_node_has_the_original_inputs(self):
        node = guidance_cns.ForgeNeoCNSSamplerPatch
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(tuple(required), ("sampler", "strength", "gamma_power", "gamma_scale"))
        self.assertEqual(required["sampler"], ("SAMPLER",))
        for name in ORIGIN_INPUTS:
            self.assertOriginInput(required[name], name, name)
        self.assertEqual(node.RETURN_TYPES, ("SAMPLER",))
        self.assertEqual(node.RETURN_NAMES, ("sampler",))
        self.assertIs(generation.NODE_CLASS_MAPPINGS["ForgeNeoCNSSamplerPatch"], node)

    def test_ksampler_and_hires_cns_inputs_are_the_original_ranges(self):
        for node in (generation.ForgeNeoKSamplerCNS, generation.ForgeNeoHiresFix):
            required = node.INPUT_TYPES()["required"]
            for name in ORIGIN_INPUTS:
                with self.subTest(node=node.__name__, name=name):
                    self.assertOriginInput(required[f"cns_{name}"], name, name)
        self.assertEqual(guidance_cns.CNS_DEFAULTS,
                         {name: spec["default"] for name, spec in ORIGIN_INPUTS.items()})

    def test_apply_cns_puts_one_sampler_sample_wrapper_on_a_clone(self):
        model = _FakePatcher()
        first = guidance_cns.apply_cns(model, 0.5, 0.75, 3.0)
        self.assertIsNot(first, model)
        self.assertEqual(model.wrappers, {})
        self.assertEqual(len(_cns_wrappers(first)), 1)
        self.assertEqual(guidance_cns.model_cns_settings(first),
                         {"strength": 0.5, "gamma_power": 0.75, "gamma_scale": 3.0})
        second = guidance_cns.apply_cns(first)
        self.assertEqual(len(_cns_wrappers(second)), 1, "CNS again replaces, never stacks")
        self.assertEqual(guidance_cns.model_cns_settings(second),
                         {"strength": 1.0, "gamma_power": 0.5, "gamma_scale": 2.0})
        self.assertEqual(guidance_cns.model_cns_settings(first)["gamma_scale"], 3.0)
        self.assertIsNone(guidance_cns.model_cns_settings(model))

    def test_wrapper_swaps_sampler_function_only_while_the_run_lasts(self):
        def sample_original(model, x, sigmas, **kwargs):
            return "original"

        sampler = SimpleNamespace(sampler_function=sample_original)
        seen = []

        def run(obj, *args, **kwargs):
            seen.append((obj.sampler_function, args, kwargs))
            return "samples"

        wrapper = guidance_cns.cns_sampler_sample_wrapper(0.5, 0.5, 2.0)
        self.assertEqual(wrapper(_Executor(sampler, run), "guider", sigmas="s"), "samples")
        swapped, args, kwargs = seen[0]
        self.assertIsNot(swapped, sample_original)
        self.assertEqual(getattr(swapped, "forge_neo_cns"),
                         {"strength": 0.5, "gamma_power": 0.5, "gamma_scale": 2.0})
        self.assertEqual((args, kwargs), (("guider",), {"sigmas": "s"}))
        self.assertIs(sampler.sampler_function, sample_original)

        def boom(obj, *args, **kwargs):
            raise RuntimeError("interrupted")

        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            wrapper(_Executor(sampler, boom))
        self.assertIs(sampler.sampler_function, sample_original)

    def test_wrapper_refuses_a_sampler_without_sampler_function(self):
        # origin: cns_sampler_patch.py:448 — patch() reads sampler.sampler_function, so a
        # non-KSAMPLER sampler fails loudly; CNS is never dropped silently.
        sampler = SimpleNamespace()
        ran = []
        wrapper = guidance_cns.cns_sampler_sample_wrapper()
        with self.assertRaisesRegex(RuntimeError, "KSAMPLER samplers only.*SimpleNamespace"):
            wrapper(_Executor(sampler, lambda obj: ran.append(obj)))
        self.assertEqual(ran, [])
        self.assertFalse(hasattr(sampler, "sampler_function"))

    def test_apply_cns_is_the_outermost_sampler_wrapper(self):
        # comfy/patcher_extension.py:83-90 lists wrappers in dict order and WrapperExecutor
        # (:92-130) calls the first one first. The compiler puts the Detail Daemon node's
        # wrapper on the MODEL before CNS; CNS must still wrap the run's real KSAMPLER.
        model = _FakePatcher()
        model.add_wrapper_with_key("sampler_sample", "ai_studio_forge_parity.detail_daemon", "dd")
        model.add_wrapper_with_key("sampler_sample", "other.pack", "other")
        model.add_wrapper_with_key("predict_noise", "other.pack", "predict")
        patched = guidance_cns.apply_cns(model, 0.5, 0.5, 2.0)
        self.assertEqual(list(patched.wrappers["sampler_sample"]),
                         [guidance_cns.CNS_WRAPPER_KEY, "ai_studio_forge_parity.detail_daemon", "other.pack"])
        self.assertEqual(patched.wrappers["predict_noise"], {"other.pack": ["predict"]})
        self.assertEqual(list(model.wrappers["sampler_sample"]),
                         ["ai_studio_forge_parity.detail_daemon", "other.pack"])
        # Applying CNS again (Hires on a CNS model) keeps it first with the new settings.
        again = guidance_cns.apply_cns(patched, 1.0, 0.5, 3.0)
        self.assertEqual(list(again.wrappers["sampler_sample"])[0], guidance_cns.CNS_WRAPPER_KEY)
        self.assertEqual(len(_cns_wrappers(again)), 1)
        self.assertEqual(guidance_cns.model_cns_settings(again)["gamma_scale"], 3.0)

    def test_patched_function_is_not_a_kd_function_so_a_second_patch_uses_path_b(self):
        # origin: cns_sampler_patch.py:486, :531-540 — the patched function carries no __wrapped__
        def sample_x(model, x, sigmas, noise_sampler=None, callback=None):
            return x

        patched = guidance_cns.cns_sampler_function(sample_x)
        self.assertFalse(hasattr(patched, "__wrapped__"))

    def test_ode_sampler_is_called_unchanged(self):
        # origin: cns_sampler_patch.py:544-580 — Path A, no noise_sampler parameter: no CNS
        calls = []

        def sample_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
            calls.append((extra_args, callback, disable))
            return "ode"

        kd_module = types.ModuleType("fake_kd")
        kd_module.sample_ode = sample_ode
        user_callback = object()
        patched = guidance_cns.cns_sampler_function(sample_ode, kd_module=kd_module)
        self.assertEqual(
            patched("model", "x", "sigmas", extra_args={}, callback=user_callback, disable=True),
            "ode",
        )
        self.assertEqual(calls, [({}, user_callback, True)])

    def test_ksampler_node_applies_cns_and_allows_speed(self):
        model = _FakePatcher()
        received = []

        def fake_provider(name, *, method, feature, args=(), kwargs=None):
            received.append((name, args[0]))
            return ({"samples": "speed"},)

        with mock.patch.object(generation, "invoke_provider", side_effect=fake_provider):
            result = generation.ForgeNeoKSamplerCNS().sample(
                model, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                "euler_ancestral", "simple", cns_enabled=True, cns_strength=0.5,
                cns_gamma_power=0.5, cns_gamma_scale=3.0, speed_enabled=True,
            )
        self.assertEqual(result, ({"samples": "speed"},))
        (name, speed_model), = received
        self.assertEqual(name, "SpectrumSPDKSampler")
        self.assertEqual(guidance_cns.model_cns_settings(speed_model),
                         {"strength": 0.5, "gamma_power": 0.5, "gamma_scale": 3.0})
        self.assertEqual(model.wrappers, {})

    def test_ksampler_node_without_cns_samples_the_incoming_model(self):
        model = _FakePatcher()
        with mock.patch.object(generation, "_common_sample",
                               side_effect=lambda m, *a, **k: (m,)) as sample:
            (used,) = generation.ForgeNeoKSamplerCNS().sample(
                model, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                "euler_ancestral", "simple",
            )
        self.assertIs(used, model)
        self.assertNotIn("cns", sample.call_args.kwargs)

    def test_legacy_suite_handoff_becomes_the_wrapper(self):
        model = _FakePatcher()
        model.model_options["forge_neo_cns"] = {"strength": 0.8, "gamma_power": 0.6, "gamma_scale": 3.0}
        with mock.patch.object(generation, "_common_sample",
                               side_effect=lambda m, *a, **k: (m,)):
            (used,) = generation.ForgeNeoKSamplerCNS().sample(
                model, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                "euler_ancestral", "simple",
            )
            self.assertEqual(guidance_cns.model_cns_settings(used),
                             {"strength": 0.8, "gamma_power": 0.6, "gamma_scale": 3.0})
            # A MODEL that already carries the wrapper keeps it as it is.
            (again,) = generation.ForgeNeoKSamplerCNS().sample(
                used, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                "euler_ancestral", "simple",
            )
            self.assertIs(again, used)
            # Explicit node inputs replace it.
            (explicit,) = generation.ForgeNeoKSamplerCNS().sample(
                used, "positive", "negative", {"samples": None}, 1, 28, 5.0,
                "euler_ancestral", "simple", cns_enabled=True,
            )
        self.assertEqual(guidance_cns.model_cns_settings(explicit), guidance_cns.CNS_DEFAULTS)
        self.assertEqual(len(_cns_wrappers(explicit)), 1)

    def test_hires_colours_its_own_sampling_run(self):
        model = _FakePatcher()
        comfy_stub = types.ModuleType("comfy")
        utils_stub = types.ModuleType("comfy.utils")
        utils_stub.common_upscale = lambda tensor, w, h, method, crop: ("upscaled", w, h, method)
        comfy_stub.utils = utils_stub
        samples = {"samples": SimpleNamespace(shape=(1, 16, 8, 10))}
        with (
            mock.patch.dict(sys.modules, {"comfy": comfy_stub, "comfy.utils": utils_stub}),
            mock.patch.object(generation, "_common_sample",
                              side_effect=lambda m, *a, **k: ({"model": m, "args": a},)) as sample,
        ):
            output, _vae = generation.ForgeNeoHiresFix().run(
                model, "positive", "negative", samples, seed=5, enabled=True,
                scale_by=1.5, steps=10, denoise=0.4, shift=0.0,
                cns_enabled=True, cns_strength=1.0, cns_gamma_power=0.5, cns_gamma_scale=2.0,
            )
        self.assertEqual(guidance_cns.model_cns_settings(output["model"]), guidance_cns.CNS_DEFAULTS)
        self.assertEqual(output["args"][7]["samples"], ("upscaled", 15, 12, "bislerp"))
        self.assertNotIn("cns", sample.call_args.kwargs)
        self.assertEqual(model.wrappers, {})


# --- harness shared with the scratch generator (keep identical) ---------------
def _pseudo_normal(shape, seed):
    """Irwin-Hall(12) from an integer LCG: no torch RNG, exact in float64."""
    count = 1
    for size in shape:
        count *= size
    state = (seed * 2654435761 + 12345) % 4294967296
    values = []
    for _ in range(count):
        acc = 0
        for _ in range(12):
            state = (1664525 * state + 1013904223) % 4294967296
            acc += state
        values.append(acc / 4294967296.0 - 6.0)
    return torch.tensor(values, dtype=torch.float64).reshape(shape).to(torch.float32)


def _smooth_pattern(shape):
    """Low-frequency 'image' made of polynomials only (exact in float64)."""
    *lead, height, width = shape
    planes = 1
    for size in lead:
        planes *= size
    rows = []
    for plane in range(planes):
        gain = 3.0 * (1.0 + 0.25 * (plane % 4)) * (1.0 if plane % 2 == 0 else -1.0)
        for i in range(height):
            ty = (i + 0.5) / height
            for j in range(width):
                tx = (j + 0.5) / width
                rows.append(gain * (4.0 * ty * (1.0 - ty) * (2.0 * tx - 1.0) + 0.5 * 4.0 * tx * (1.0 - tx)))
    return torch.tensor(rows, dtype=torch.float64).reshape(shape).to(torch.float32)


def _make_kd_module(record):
    kds = types.ModuleType("fake_kd_sampling")

    @torch.no_grad()
    def sample_fake_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
        extra_args = {} if extra_args is None else extra_args
        if noise_sampler is None:
            noise_sampler = lambda sigma, sigma_next: torch.zeros_like(x)
        for i in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            denoised = model(x, sigma, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
            if float(sigma_next) == 0.0:
                x = denoised
                continue
            noise = noise_sampler(sigma, sigma_next)
            record.append(noise.detach().clone())
            ratio = float(sigma_next) / float(sigma)
            x = denoised + (x - denoised) * (ratio * 0.5) + noise * (float(sigma_next) * s_noise * 0.5)
        return x

    @torch.no_grad()
    def sample_fake_ode(model, x, sigmas, extra_args=None, callback=None, disable=None):
        extra_args = {} if extra_args is None else extra_args
        for i in range(len(sigmas) - 1):
            denoised = model(x, sigmas[i], **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
            x = denoised + (x - denoised) * (float(sigmas[i + 1]) / float(sigmas[i]))
        return x

    kds.sample_fake_ancestral = sample_fake_ancestral
    kds.sample_fake_ode = sample_fake_ode
    return kds


def _make_model(shape):
    clean = _smooth_pattern(shape)

    def model(x, sigma, **_extra):
        s = float(sigma)
        return clean * (1.0 - 0.5 * s) + x * (0.25 * s)

    return model


def _make_base_noise_sampler(shape, seed):
    counter = [0]

    def base(sigma, sigma_next):
        counter[0] += 1
        return _pseudo_normal(shape, seed + counter[0])

    return base


def _make_nested_kd_module(record):
    """_make_kd_module + a *_gpu-style sample_* that delegates to another by keyword."""
    kds = _make_kd_module(record)

    @torch.no_grad()
    def sample_fake_delegate(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
        return kds.sample_fake_ancestral(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
            eta=eta, s_noise=s_noise, noise_sampler=noise_sampler,
        )

    kds.sample_fake_delegate = sample_fake_delegate
    return kds


class _KSampler:
    """comfy.samplers.KSAMPLER (comfy/samplers.py:977-1007): ``sample`` calls
    ``sampler_function(model, x, sigmas, ..., **self.extra_options)``."""

    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model, x, sigmas, **kwargs):
        return self.sampler_function(model, x, sigmas, **kwargs, **self.extra_options)


class _WrapperExecutor:
    """comfy.patcher_extension.WrapperExecutor semantics (ComfyUI@387f98aa
    comfy/patcher_extension.py:92-130): wrappers[0] runs first; calling the executor
    runs the next wrapper, then ``original``."""

    def __init__(self, original, class_obj, wrappers, idx):
        self.original = original
        self.class_obj = class_obj
        self.wrappers = list(wrappers)
        self.idx = idx
        self.is_last = idx == len(wrappers)

    def __call__(self, *args, **kwargs):
        return type(self).new_class_executor(
            self.original, self.class_obj, self.wrappers, self.idx + 1,
        ).execute(*args, **kwargs)

    def execute(self, *args, **kwargs):
        if self.is_last:
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)

    @classmethod
    def new_class_executor(cls, original, class_obj, wrappers, idx=0):
        return cls(original, class_obj, wrappers, idx=idx)


def _run_case(case, sampler_function, record, **extra):
    """Run a (patched) sampler function the way the generator ran the original."""
    shape = tuple(case["shape"])
    x0 = _pseudo_normal(shape, GOLDEN["x0_seed"]) * GOLDEN["sigmas"][0]
    kwargs = dict(extra)
    if case["base_noise_seed"] is not None:
        kwargs["noise_sampler"] = _make_base_noise_sampler(shape, case["base_noise_seed"])
    else:
        torch.manual_seed(case["torch_manual_seed"])
    kwargs.setdefault("callback", None)
    final = sampler_function(
        _make_model(shape), x0.clone(), torch.tensor(GOLDEN["sigmas"]),
        extra_args={}, disable=True, **kwargs,
    )
    return final, x0


def _randn_golden_usable():
    return str(torch.__version__) == GOLDEN["torch"]


@requires_torch
class TestCnsOriginGolden(unittest.TestCase):
    """Every step's noise_sampler output against the numbers the original produced."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def assertMatchesGolden(self, actual, expected, label):
        want = torch.tensor(expected, dtype=torch.float32)
        got = actual.detach().float().flatten()
        self.assertEqual(tuple(got.shape), tuple(want.shape), label)
        diff = (got - want).abs().max().item()
        self.assertLessEqual(diff, 1e-6, f"{label}: max|d| {diff:.3e}")

    def assertCaseMatches(self, case, record, final, label):
        self.assertEqual(len(record), len(case["noise"]), f"{label}: noise draws")
        for step, (got, want) in enumerate(zip(record, case["noise"])):
            self.assertMatchesGolden(got, want, f"{label} step {step}")
        self.assertMatchesGolden(final, case["final"], f"{label} final x")

    def _cases(self):
        for case in GOLDEN["cases"]:
            if case["base_noise_seed"] is None and not _randn_golden_usable():
                continue  # torch.randn_like stream of another torch build; see the structural test
            yield case

    def test_harness_matches_the_generator(self):
        for case in GOLDEN["cases"]:
            with self.subTest(case=case["name"]):
                shape = tuple(case["shape"])
                check = case["input_check"]
                x0 = _pseudo_normal(shape, GOLDEN["x0_seed"]) * GOLDEN["sigmas"][0]
                self.assertAlmostEqual(float(x0.double().sum()), check["x0_sum"], places=9)
                self.assertAlmostEqual(float(_smooth_pattern(shape).double().sum()), check["clean_sum"], places=9)
                if case["base_noise_seed"] is not None:
                    first = _pseudo_normal(shape, case["base_noise_seed"] + 1)
                    self.assertAlmostEqual(float(first.double().sum()), check["first_base_noise_sum"], places=9)

    def test_path_a_every_step_matches_origin(self):
        # origin: cns_sampler_patch.py:169-288 (colouring), :295-347 (noise sampler),
        #         :544-580 (Path A: noise_sampler + callback x tracking)
        for case in self._cases():
            with self.subTest(case=case["name"]):
                record = []
                kds = _make_kd_module(record)
                patched = guidance_cns.cns_sampler_function(
                    kds.sample_fake_ancestral, case["strength"], case["gamma_power"],
                    case["gamma_scale"], kd_module=kds,
                )
                final, _x0 = _run_case(case, patched, record)
                self.assertCaseMatches(case, record, final, case["name"])

    def test_path_b_wrapper_matches_origin_and_restores_the_module(self):
        # origin: cns_sampler_patch.py:454-484, :582-602 (Path B: swap sample_* while the wrapper runs)
        self.assertTrue(GOLDEN["path_b_equals_path_a"])
        for case in self._cases():
            with self.subTest(case=case["name"]):
                record = []
                kds = _make_kd_module(record)
                real = kds.sample_fake_ancestral, kds.sample_fake_ode

                def speed_like(model, x, sigmas, **kwargs):
                    return kds.sample_fake_ancestral(model, x, sigmas, **kwargs)

                patched = guidance_cns.cns_sampler_function(
                    speed_like, case["strength"], case["gamma_power"], case["gamma_scale"],
                    kd_module=kds,
                )
                final, _x0 = _run_case(case, patched, record)
                self.assertCaseMatches(case, record, final, case["name"] + " (Path B)")
                self.assertEqual((kds.sample_fake_ancestral, kds.sample_fake_ode), real)

    def test_model_wrapper_route_matches_origin_and_keeps_the_initial_noise(self):
        # The SAMPLER_SAMPLE wrapper swaps the KSAMPLER's function for the run
        # (comfy/samplers.py KSAMPLER.sample -> sampler_function(model_k, noise, sigmas, ...)).
        case = CASES["odd_default"]
        record, first_x = [], []
        kds = _make_kd_module(record)
        sampler = SimpleNamespace(sampler_function=kds.sample_fake_ancestral)
        shape = tuple(case["shape"])
        x0 = _pseudo_normal(shape, GOLDEN["x0_seed"]) * GOLDEN["sigmas"][0]

        def ksampler_sample(obj, noise):
            def user_callback(info):
                if info["i"] == 0:
                    first_x.append(info["x"])
            return obj.sampler_function(
                _make_model(shape), noise, torch.tensor(GOLDEN["sigmas"]), extra_args={},
                callback=user_callback, disable=True,
                noise_sampler=_make_base_noise_sampler(shape, case["base_noise_seed"]),
            )

        wrapper = guidance_cns.cns_sampler_sample_wrapper(
            case["strength"], case["gamma_power"], case["gamma_scale"], kd_module=kds,
        )
        noise = x0.clone()
        final = wrapper(_Executor(sampler, ksampler_sample), noise)
        self.assertCaseMatches(case, record, final, "SAMPLER_SAMPLE wrapper")
        self.assertTrue(torch.equal(first_x[0], x0), "the sampler starts from the uncoloured noise")
        self.assertTrue(torch.equal(noise, x0))
        self.assertIs(sampler.sampler_function, kds.sample_fake_ancestral)

    def test_stacked_patches_colour_once_and_the_inner_patch_wins(self):
        # origin: a CNS patch around a CNS-patched sampler takes Path B (:531-540); the inner
        # patch then finds the real sample_* (inspect.unwrap) and colours through Path A.
        case = CASES["odd_default"]
        record = []
        kds = _make_kd_module(record)
        inner = guidance_cns.cns_sampler_function(
            kds.sample_fake_ancestral, case["strength"], case["gamma_power"], case["gamma_scale"],
            kd_module=kds,
        )
        outer = guidance_cns.cns_sampler_function(inner, 0.5, 2.0, 0.1, kd_module=kds)
        final, _x0 = _run_case(case, outer, record)
        self.assertCaseMatches(case, record, final, "stacked")

    def _nested_samplers(self, record):
        kds = _make_nested_kd_module(record)

        @torch.no_grad()
        def sample_fake_dispatch(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0, noise_sampler=None):
            # flow-model *_RF style: the next sample_* is called positionally
            return kds.sample_fake_ancestral(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler)

        kds.sample_fake_dispatch = sample_fake_dispatch
        return kds

    def test_path_a_nested_calls_reach_the_real_sample_functions(self):
        # origin: cns_sampler_patch.py:544-580 — Path A patches nothing, so a sample_* the
        # sampler calls by name (positionally or by keyword) is the real one: coloured once.
        case = CASES["odd_default"]
        record = []
        kds = self._nested_samplers(record)
        for label in ("sample_fake_dispatch", "sample_fake_delegate"):
            with self.subTest(sampler=label):
                record.clear()
                patched = guidance_cns.cns_sampler_function(
                    getattr(kds, label), case["strength"], case["gamma_power"], case["gamma_scale"],
                    kd_module=kds,
                )
                final, _x0 = _run_case(case, patched, record)
                self.assertCaseMatches(case, record, final, f"Path A {label}")

    def test_path_b_positional_calls_colour_once(self):
        # Deliberate difference, only where the original raises: its Path B stand-in is
        # ``wrapped(model, x, sigmas, **kwargs)`` (cns_sampler_patch.py:462), so a sample_*
        # calling another positionally (flow-model *_RF dispatch) or a wrapper calling one
        # with positional extras (dpm_fast) raised TypeError. Here both colour once.
        case = CASES["odd_default"]
        record = []
        kds = self._nested_samplers(record)

        def speed_dispatch(model, x, sigmas, **kwargs):
            return kds.sample_fake_dispatch(model, x, sigmas, **kwargs)

        def speed_positional(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
            return kds.sample_fake_ancestral(
                model, x, sigmas, extra_args, callback, disable, 1.0, 1.0, noise_sampler=noise_sampler,
            )

        for label, function in (("dispatch", speed_dispatch), ("positional", speed_positional)):
            with self.subTest(path=label):
                record.clear()
                patched = guidance_cns.cns_sampler_function(
                    function, case["strength"], case["gamma_power"], case["gamma_scale"],
                    kd_module=kds,
                )
                final, _x0 = _run_case(case, patched, record)
                self.assertCaseMatches(case, record, final, f"Path B {label}")

    def test_keyword_nested_calls_colour_again_like_the_original(self):
        # origin: cns_sampler_patch.py:454-484, :582-602 — every swapped sample_* recolours on
        # every call, so a *_gpu-style keyword delegation under Path B, or a second CNS patch
        # around a sampler that delegates, colours each draw twice (golden from the original).
        nested = GOLDEN["nested"]
        case = CASES[nested["case"]]
        outer = nested["stack_outer"]
        record = []
        kds = _make_nested_kd_module(record)
        real = kds.sample_fake_ancestral, kds.sample_fake_delegate

        def speed_delegate(model, x, sigmas, **kwargs):
            return kds.sample_fake_delegate(model, x, sigmas, **kwargs)

        settings = (case["strength"], case["gamma_power"], case["gamma_scale"])
        inner = guidance_cns.cns_sampler_function(kds.sample_fake_delegate, *settings, kd_module=kds)
        variants = {
            "path_b_delegate": guidance_cns.cns_sampler_function(speed_delegate, *settings, kd_module=kds),
            "stacked_delegate": guidance_cns.cns_sampler_function(
                inner, outer["strength"], outer["gamma_power"], outer["gamma_scale"], kd_module=kds,
            ),
        }
        for label, patched in variants.items():
            with self.subTest(variant=label):
                record.clear()
                final, _x0 = _run_case(case, patched, record)
                self.assertCaseMatches(nested[label], record, final, label)
                self.assertGreater(
                    max(abs(a - b) for twice, once in zip(nested[label]["noise"], case["noise"])
                        for a, b in zip(twice, once)),
                    1e-3, "the double colouring differs from one colouring",
                )
                self.assertEqual((kds.sample_fake_ancestral, kds.sample_fake_delegate), real)

    def test_detail_daemon_and_cns_model_wrappers_match_the_original_chain(self):
        # The compiler puts the Detail Daemon node's SAMPLER_SAMPLE wrapper on the MODEL, then
        # apply_cns. Run the MODEL's wrappers the way ComfyUI does (get_all_wrappers order,
        # WrapperExecutor) on a sampler that does not re-dispatch: every draw and the final x
        # must equal the original's working graph KSamplerSelect -> CNSSamplerPatch ->
        # DetailDaemonSampler (golden: Jonseed/ComfyUI-Detail-Daemon@3394e44 go() around the
        # original CNSSamplerPatch; origin detail_daemon_node.py:265-310 calls
        # dds_wrapped_sampler.sampler_function at run time).
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_dd

        chain = GOLDEN["dd_chain"]
        case = CASES[chain["case"]]
        shape = tuple(case["shape"])
        record = []
        kds = _make_kd_module(record)
        kd_package = types.ModuleType("comfy.k_diffusion")
        kd_package.sampling = kds
        comfy_stub = types.ModuleType("comfy")
        samplers_stub = types.ModuleType("comfy.samplers")
        samplers_stub.KSAMPLER = _KSampler
        extension_stub = types.ModuleType("comfy.patcher_extension")
        extension_stub.WrappersMP = SimpleNamespace(SAMPLER_SAMPLE="sampler_sample")
        extension_stub.WrapperExecutor = _WrapperExecutor
        comfy_stub.samplers, comfy_stub.patcher_extension = samplers_stub, extension_stub
        comfy_stub.k_diffusion = kd_package
        modules = {
            "comfy": comfy_stub, "comfy.samplers": samplers_stub,
            "comfy.patcher_extension": extension_stub, "comfy.k_diffusion": kd_package,
            "comfy.k_diffusion.sampling": kds,
        }
        with mock.patch.dict(sys.modules, modules):
            model = _FakePatcher()
            model.add_wrapper_with_key(
                "sampler_sample", guidance_dd.DD_WRAPPER_KEY,
                guidance_dd.detail_daemon_sampler_sample_wrapper(dict(chain["dd_inputs"])),
            )
            patched = guidance_cns.apply_cns(
                model, case["strength"], case["gamma_power"], case["gamma_scale"],
            )
            # comfy/patcher_extension.py:83-90 get_all_wrappers
            wrappers = [item for items in patched.wrappers["sampler_sample"].values() for item in items]
            sampler = _KSampler(kds.sample_fake_ancestral)
            x0 = _pseudo_normal(shape, GOLDEN["x0_seed"]) * GOLDEN["sigmas"][0]
            final = _WrapperExecutor.new_class_executor(sampler.sample, sampler, wrappers).execute(
                _make_model(shape), x0.clone(), torch.tensor(GOLDEN["sigmas"]), extra_args={},
                callback=None, disable=True,
                noise_sampler=_make_base_noise_sampler(shape, case["base_noise_seed"]),
            )
        self.assertCaseMatches(chain, record, final, "DD + CNS wrappers")
        self.assertIs(sampler.sampler_function, kds.sample_fake_ancestral)
        base = [_pseudo_normal(shape, case["base_noise_seed"] + step + 1) for step in range(len(record))]
        for step, (got, white) in enumerate(zip(record, base)):
            self.assertFalse(torch.equal(got, white), f"step {step} noise is coloured")

    def test_ode_sampler_output_is_unchanged(self):
        # origin: README / cns_sampler_patch.py:50-53 — ODE samplers never call noise_sampler
        case = dict(CASES["odd_default"], base_noise_seed=None, torch_manual_seed=0)
        kds = _make_kd_module([])
        steps = []
        plain, _ = _run_case(case, kds.sample_fake_ode, [])
        patched = guidance_cns.cns_sampler_function(kds.sample_fake_ode, kd_module=kds)
        coloured, _ = _run_case(case, patched, [], callback=lambda info: steps.append(info["i"]))
        self.assertTrue(torch.equal(plain, coloured))
        self.assertEqual(steps, list(range(len(GOLDEN["sigmas"]) - 1)))

    def test_default_base_noise_is_randn_like_of_the_live_x(self):
        # origin: cns_sampler_patch.py:327-334 — no noise_sampler given: torch.randn_like(x_t)
        case = dict(CASES["default_randn"])
        record, xs = [], []
        kds = _make_kd_module(record)
        patched = guidance_cns.cns_sampler_function(
            kds.sample_fake_ancestral, case["strength"], case["gamma_power"], case["gamma_scale"],
            kd_module=kds,
        )
        _run_case(case, patched, record, callback=lambda info: xs.append(info["x"]))
        torch.manual_seed(case["torch_manual_seed"])
        for step, got in enumerate(record):
            live = xs[step]
            want = guidance_cns.color_noise_wavelet(
                torch.randn_like(live), live, case["strength"], case["gamma_power"], case["gamma_scale"],
            )
            self.assertTrue(torch.equal(got, want), f"step {step}")

    def test_strength_zero_returns_the_base_noise_itself(self):
        # origin: cns_sampler_patch.py:192-193
        noise = _pseudo_normal((1, 4, 5, 7), 7)
        live = _smooth_pattern((1, 4, 5, 7))
        self.assertIs(guidance_cns.color_noise_wavelet(noise, live, strength=0.0), noise)

    def test_strength_half_is_a_plain_lerp_without_renormalising(self):
        # origin: cns_sampler_patch.py:284-288 — lerp(noise, coloured, strength), no std fix after it
        noise = _pseudo_normal((2, 4, 9, 11), 8)
        live = _smooth_pattern((2, 4, 9, 11)) + 0.1 * _pseudo_normal((2, 4, 9, 11), 9)
        full = guidance_cns.color_noise_wavelet(noise, live, 1.0, 0.5, 2.0)
        half = guidance_cns.color_noise_wavelet(noise, live, 0.5, 0.5, 2.0)
        self.assertTrue(torch.equal(half, torch.lerp(noise, full, 0.5)))
        self.assertAlmostEqual(float(full.std()), float(noise.std()), places=5)
        self.assertLess(float(half.std()), float(noise.std()))

    def test_half_precision_noise_keeps_its_dtype(self):
        noise = _pseudo_normal((1, 4, 5, 7), 10).to(torch.float16)
        live = _smooth_pattern((1, 4, 5, 7))
        out = guidance_cns.color_noise_wavelet(noise, live)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (1, 4, 5, 7))

    def test_colouring_error_falls_back_to_the_base_noise(self):
        # origin: cns_sampler_patch.py:336-345 — an exception skips colouring for that draw
        base_noise = _pseudo_normal((3, 4, 6, 6), 11)
        sampler, _state = guidance_cns.make_cns_noise_sampler(
            _pseudo_normal((2, 4, 6, 6), 12), 1.0, 0.5, 2.0, lambda s, sn: base_noise,
        )
        with self.assertLogs("ai_studio_forge_parity", level="WARNING"):
            self.assertIs(sampler(1.0, 0.5), base_noise)

    def test_failing_base_sampler_falls_back_to_randn_like(self):
        # origin: cns_sampler_patch.py:328-332
        def broken(sigma, sigma_next):
            raise RuntimeError("no noise")

        live = _pseudo_normal((1, 4, 6, 6), 13)
        sampler, _state = guidance_cns.make_cns_noise_sampler(live, 0.0, 0.5, 2.0, broken)
        torch.manual_seed(3)
        got = sampler(1.0, 0.5)
        torch.manual_seed(3)
        self.assertTrue(torch.equal(got, torch.randn_like(live)))


@requires_torch
class TestCommonSampleKeepsInitialNoise(unittest.TestCase):
    """generation._common_sample hands prepare_noise's tensor to comfy.sample untouched."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def test_prepared_noise_reaches_the_sampler_uncoloured(self):
        noise = _pseudo_normal((1, 4, 5, 7), 14)
        received = {}
        comfy_stub = types.ModuleType("comfy")
        sample_stub = types.ModuleType("comfy.sample")
        sample_stub.fix_empty_latent_channels = lambda model, latent, *a: latent
        sample_stub.prepare_noise = lambda latent, seed, inds: noise

        def fake_sample(model, given_noise, *args, **kwargs):
            received["noise"] = given_noise
            return given_noise

        sample_stub.sample = fake_sample
        utils_stub = types.ModuleType("comfy.utils")
        utils_stub.PROGRESS_BAR_ENABLED = False
        preview_stub = types.ModuleType("latent_preview")
        preview_stub.prepare_callback = lambda model, steps: None
        comfy_stub.sample, comfy_stub.utils = sample_stub, utils_stub
        model = guidance_cns.apply_cns(_FakePatcher())
        with mock.patch.dict(sys.modules, {
            "comfy": comfy_stub, "comfy.sample": sample_stub, "comfy.utils": utils_stub,
            "latent_preview": preview_stub,
        }):
            generation._common_sample(
                model, 1, 4, 5.0, "euler_ancestral", "simple", "p", "n",
                {"samples": _smooth_pattern((1, 4, 5, 7))}, 1.0,
            )
        self.assertIs(received["noise"], noise)


if __name__ == "__main__":
    unittest.main()
