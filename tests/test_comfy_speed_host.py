r"""Pack 1.7.0 Anima SPEED on ComfyUI — the pack's own host rules (no extension needed).

``speed_nodes`` puts ``speed_runner.run_speed`` into a ComfyUI sampling run. What sam-extra
decides is checked against the installed extension in ``tests/test_comfy_speed_parity.py``; this
file checks the ComfyUI side with fake ComfyUI objects on the CPU:

* ``NodeSettingsTests`` (no torch): the settings reader (names, aliases, positional lists, the
  one-dict form), refusals and the ``Anima SPEED`` summary text.
* ``RuntimeGuardTests`` (torch): every ComfyUI condition SPEED steps aside for, and the order the
  checks run in (sam-extra's order, ComfyUI-only checks last); the Extra Schedulers private
  ``model_sampling`` copy still counts as flow matching; unsupported sampler functions are found
  through CNS's ``_forge_neo_base``.
* ``HostSeedAndNoiseTests`` (torch): spectral seeds (fixed vs image seeds); one noise stream
  across the segments, made like ComfyUI's ``default_noise_sampler`` (CPU ``seed + 1``);
  Brownian-tree samplers get per-image coarse trees at ``seed + (stage + 1) * 1_000_003`` and a
  plain-run tree for the tail (with the range raised to an aligned hand-off sigma above the pass
  range); samplers without a ``noise_sampler`` parameter get none; a caller's noise sampler stays.
* ``HostPublishTests`` (torch): the patched schedule is published as ``sample_sigmas`` (tagged
  ``republished_from`` = the run's list, also for an offset list) while the run lasts and the
  original list object is put back, also after a failure.
* ``ComfyTableTests``: ``BROWNIAN_SAMPLERS`` and the ComfyUI names in ``UNSUPPORTED_SAMPLERS``
  equal ComfyUI's own source (read by AST; skipped unless ``AISTUDIO_COMFY_TEST_ROOT`` is set —
  nothing is imported from ComfyUI).
* ``SpeedWrapperTests`` (torch): one sampling run through the SAMPLER_SAMPLE wrapper with stand-ins
  for ComfyUI's ``WrapperExecutor``/``KSAMPLER``/``CFGGuider`` — applied (== ``run_speed`` by
  hand), the console lines, img2img rescale off, guard/plan refusals sample plainly, UniPC is not
  wrapped, a SPEED failure falls back to the plain run, an interrupt goes through, only the first
  call of a run is segmented, the step offset reaches the sampler through CNS.
* ``SpeedNodeTests`` (torch): explicit node inputs with the extension's defaults; off / invalid /
  ``1/0`` leave the MODEL as it is; the patch replaces an earlier one; wrapper order CNS -> SPEED
  -> Detail Daemon whatever the chain order; the Hires checkpoint override keeps the patch; the
  ControlNet-LLLite marker makes SPEED step aside.
* ``DetailDaemonFollowsSpeedTests`` (torch): Detail Daemon inside SPEED looks each sigma up in the
  re-published schedule (transition: the hand-off step's own schedule value at the aligned sigma;
  respace: every tail step), coarse steps and plain runs use the run's own list, and the node's
  lookup list is the closure cell the pack writes into.

torch is imported lazily (tests/_optional_deps).
"""
from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from comfy_custom_nodes.ai_studio_forge_parity import guidance_cns, guidance_common, speed_nodes, speed_schedule
from tests._optional_deps import bind_torch, requires_torch

torch = None  # filled by bind_torch() in the torch classes

_COMFY_ROOT = os.environ.get("AISTUDIO_COMFY_TEST_ROOT")


# --------------------------------------------------------------------------------------------------
# Fake ComfyUI pieces (class names are what the pack reads)
# --------------------------------------------------------------------------------------------------


class EPS:
    pass


class V_PREDICTION(EPS):
    pass


class EDM(V_PREDICTION):
    pass


class CONST:
    pass


class ModelSamplingDiscrete:
    pass


class ModelSamplingDiscreteFlow:
    pass


class FlowSampling(ModelSamplingDiscreteFlow, CONST):
    pass


class EpsSampling(ModelSamplingDiscrete, EPS):
    pass


class EdmSampling(ModelSamplingDiscrete, EDM):
    pass


class Anima:
    """Stands for ``comfy.model_base.Anima`` (only the class name is read)."""


class Flux:
    pass


class _Patcher:
    def __init__(self, sampling):
        self.sampling = sampling

    def get_model_object(self, name):
        if name != "model_sampling":
            raise AssertionError(name)
        return self.sampling


def _guider(sampling=None, *, conds=None, model=None, latent_shapes=None):
    inner = model if model is not None else Anima()
    inner.latent_shapes = latent_shapes
    return SimpleNamespace(
        model_patcher=_Patcher(FlowSampling() if sampling is None else sampling),
        inner_model=inner,
        conds={"positive": [{"model_conds": {"c_crossattn": object()}}], "negative": [{"model_conds": {}}]}
        if conds is None else conds,
    )


def _extra(**transformer):
    return {"model_options": {"transformer_options": dict(transformer)}, "seed": 5, "denoise_mask": None}


def _flow_sigmas(steps: int = 12, shift: float = 3.0, start: float = 1.0):
    ts = [start - (start * i) / steps for i in range(steps)]
    return torch.tensor([shift * t / (1.0 + (shift - 1.0) * t) for t in ts] + [0.0])


# --------------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------------


class NodeSettingsTests(unittest.TestCase):
    def test_defaults_and_names(self):
        settings = speed_nodes.coerce_settings({})
        self.assertEqual(settings, speed_nodes.SpeedSettings())
        self.assertIsNone(settings.error)
        self.assertEqual(speed_nodes.coerce_settings([]), settings)
        self.assertEqual(speed_nodes.coerce_settings(None), settings)
        self.assertEqual(tuple(speed_nodes.DEFAULTS), speed_nodes.ARG_NAMES)

    def test_named_aliases_positional_and_one_dict_forms(self):
        named = {"enabled": True, "mode": "Respace", "threshold": "manual", "preset": "custom", "scales": "1/4; 1/2; 1",
                 "divisor": "2", "manual": "0.9, 0.8", "A": 300, "beta": "2.5", "seed": "17.9", "transform": " FFT "}
        settings = speed_nodes.coerce_settings(named)
        self.assertEqual(
            (settings.enabled, settings.mode, settings.threshold, settings.preset, settings.scales, settings.sigma_divisor,
             settings.manual_sigmas, settings.spectrum_A, settings.spectrum_beta, settings.seed, settings.transform),
            (True, "respace", "manual", "custom", (0.25, 0.5, 1.0), 2.0, (0.9, 0.8), 300.0, 2.5, 17, "fft"),
        )
        self.assertIsNone(settings.error)
        self.assertEqual(speed_nodes.coerce_settings([named]), settings)
        positional = [named.get(name, speed_nodes.DEFAULTS[name]) for name in speed_nodes.ARG_NAMES]
        positional[speed_nodes.ARG_NAMES.index("sigma_divisor")] = "2"
        positional[speed_nodes.ARG_NAMES.index("manual_sigmas")] = "0.9, 0.8"
        positional[speed_nodes.ARG_NAMES.index("spectrum_A")] = 300
        positional[speed_nodes.ARG_NAMES.index("spectrum_beta")] = "2.5"
        self.assertEqual(speed_nodes.coerce_settings(positional), settings)
        short = speed_nodes.coerce_settings([True, "respace"])
        self.assertEqual((short.enabled, short.mode, short.threshold), (True, "respace", "neo_shift"))

    def test_refusals_in_order(self):
        settings = speed_nodes.coerce_settings({"mode": "x", "threshold": "y", "preset": "z", "transform": "w",
                                                "scales": "0.5", "delta": 0, "sigma_divisor": -1})
        self.assertEqual(settings.error, "; ".join([
            "mode must be one of transition, respace (got 'x')",
            "threshold must be one of neo_shift, delta_optimal, manual (got 'y')",
            "unknown spectrum preset 'z'",
            "transform must be one of dct, dwt, fft (got 'w')",
            "last scale must equal 1.0 (full resolution); got 0.5",
            "delta must be in (0, 1); got 0.0",
            "sigma divisor must be positive; got -1.0",
        ]))
        self.assertEqual((settings.mode, settings.threshold, settings.preset, settings.transform, settings.scales),
                         ("transition", "neo_shift", "anima", "dct", (0.5, 1.0)))
        manual = speed_nodes.coerce_settings({"threshold": "manual", "manual_sigmas": "0.5, 0.6", "delta": 7})
        self.assertIn("strictly decreasing", manual.error)
        self.assertEqual(manual.manual_sigmas, (0.7,))
        respace = speed_nodes.coerce_settings({"mode": "respace", "threshold": "manual", "manual_sigmas": "0.5, 0.6"})
        self.assertIsNone(respace.error)
        quiet = speed_nodes.coerce_settings({"threshold": "neo_shift", "manual_sigmas": "nonsense"})
        self.assertIsNone(quiet.error)
        # a zero denominator is a settings error, not an exception (sam-extra 0.33.0)
        self.assertEqual(speed_nodes.coerce_settings({"scales": "0.5,1/0"}).error,
                         "scales: division by zero: '1/0'")
        self.assertEqual(speed_nodes.coerce_settings({"threshold": "manual", "manual_sigmas": "0.7/0"}).error,
                         "manual sigmas: division by zero: '0.7/0'")
        unused = speed_nodes.coerce_settings({"threshold": "neo_shift", "manual_sigmas": "1/0"})
        self.assertEqual((unused.error, unused.manual_sigmas), (None, (0.7,)))

    def test_lenient_values(self):
        for raw, enabled in ((None, False), ("on", True), ("Disabled", False), (0, False), (2.5, True), ("maybe", False)):
            with self.subTest(raw=raw):
                self.assertIs(speed_nodes.coerce_settings({"enabled": raw}).enabled, enabled)
        self.assertIs(speed_nodes.coerce_settings({"adaptive": "maybe"}).adaptive, True)
        self.assertEqual(speed_nodes.coerce_settings({"delta": "nan"}).delta, 0.01)
        self.assertEqual(speed_nodes.coerce_settings({"seed": -5}).seed, -1)
        self.assertEqual(speed_nodes.coerce_settings({"seed": ""}).seed, -1)
        self.assertEqual(speed_nodes.coerce_settings({"preset": ""}).preset, "anima")
        # an integer too large for a float reads as the default (sam-extra 0.33.0; 0.32.1 raised OverflowError)
        huge = speed_nodes.coerce_settings({"delta": 10 ** 400, "sigma_divisor": -10 ** 400, "seed": 10 ** 400,
                                            "spectrum_A": 10 ** 400, "spectrum_beta": -10 ** 400})
        self.assertEqual((huge.delta, huge.sigma_divisor, huge.seed, huge.spectrum_A, huge.spectrum_beta, huge.error),
                         (0.01, 1.03, -1, speed_schedule.CUSTOM_DEFAULT_A, speed_schedule.CUSTOM_DEFAULT_BETA, None))

    def test_summary(self):
        settings = speed_nodes.coerce_settings({"preset": "custom", "A": 0.1 + 0.2, "scales": "0.25,0.5,1"})
        self.assertEqual(
            settings.summary(),
            "mode=transition; threshold=neo_shift; preset=custom; scales=0.25,0.5,1; delta=0.01; divisor=1.03; "
            "manual=0.7; adaptive=True; transform=dct; A=0.3; beta=1.915461; seed=-1; hires=False",
        )
        self.assertNotIn("A=", speed_nodes.SpeedSettings().summary())


# --------------------------------------------------------------------------------------------------
# Runtime guard
# --------------------------------------------------------------------------------------------------


@requires_torch
class RuntimeGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _guard(self, guider=None, x=None, sigmas=None, extra=None):
        return speed_nodes.comfy_runtime_guard(
            _guider() if guider is None else guider,
            torch.zeros(1, 4, 8, 8) if x is None else x,
            _flow_sigmas() if sigmas is None else sigmas,
            _extra() if extra is None else extra,
        )

    def test_plain_flow_run_passes(self):
        self.assertIsNone(self._guard())
        self.assertIsNone(self._guard(x=torch.zeros(2, 16, 1, 8, 8)))
        self.assertIsNone(self._guard(guider=_guider(latent_shapes=[(1, 4, 8, 8)])))
        self.assertIsNone(self._guard(extra={}))

    def test_extra_schedulers_private_copy_counts_as_flow(self):
        copy_class = type("FlowSampling", (FlowSampling,), {})
        sampling = copy_class()
        sampling._ai_studio_extra_schedulers = ("laplace",)
        self.assertEqual(speed_nodes.prediction_kind(sampling), "const")
        self.assertIsNone(self._guard(guider=_guider(sampling)))

    def test_prediction_kinds(self):
        self.assertEqual(speed_nodes.prediction_kind(EpsSampling()), "eps")
        self.assertEqual(speed_nodes.prediction_kind(EdmSampling()), "edm")
        self.assertEqual(speed_nodes.prediction_kind(SimpleNamespace()), "simplenamespace")
        self.assertIsNone(speed_nodes.prediction_kind(None))
        without_patcher = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=FlowSampling(), latent_shapes=None),
                                          conds={})
        self.assertIsNone(self._guard(guider=without_patcher))

    def test_each_condition_and_its_wording(self):
        anima_ref = {"positive": [{"model_conds": {"ref_latents": object()}}]}
        rows = [
            ("2-D sigmas", dict(sigmas=torch.zeros(2, 3)), speed_nodes.NO_SCHEDULE_REASON),
            ("one sigma", dict(sigmas=torch.ones(1)), speed_nodes.NO_SCHEDULE_REASON),
            ("not a tensor", dict(sigmas=[1.0, 0.0]), speed_nodes.NO_SCHEDULE_REASON),
            ("3-D latent", dict(x=torch.zeros(4, 8, 8)), "unsupported latent shape (4, 8, 8)"),
            ("eps model", dict(guider=_guider(EpsSampling())),
             "not a flow-matching model (prediction_type=eps) - SDXL/SD1.x eps/v models are not supported "
             "(kappa and the sigma* presets assume flow sigmas in (0, 1])"),
            ("no sampling", dict(guider=SimpleNamespace(conds={})),
             "not a flow-matching model (prediction_type=None) - SDXL/SD1.x eps/v models are not supported "
             "(kappa and the sigma* presets assume flow sigmas in (0, 1])"),
            ("sigma max", dict(sigmas=torch.tensor([1.5, 0.5, 0.0])), "sigma_max 1.500 > 1 is not a flow schedule"),
            ("denoise mask", dict(extra={**_extra(), "denoise_mask": torch.ones(1, 1, 8, 8)}), speed_nodes.MASK_REASON),
            ("anima reference", dict(guider=_guider(conds=anima_ref)), speed_nodes.ANIMA_REFERENCE_REASON),
            ("other reference", dict(guider=_guider(conds=anima_ref, model=Flux())), speed_nodes.REFERENCE_REASON),
            ("concat", dict(guider=_guider(conds={"negative": [{"model_conds": {"c_concat": object()}}]})),
             speed_nodes.CONCAT_REASON),
            ("controlnet", dict(guider=_guider(conds={"positive": [{"model_conds": {}, "control": object()}]})),
             speed_nodes.CONTROLNET_REASON),
            ("lllite", dict(extra={"model_options": {speed_nodes.LLLITE_MARKER: True}}), speed_nodes.LLLITE_REASON),
            ("spectrum flag", dict(extra=_extra(**{guidance_common.DY_SUBSTEPS_OFF_KEY: "Spectrum"})),
             speed_nodes.SPECTRUM_REASON),
            ("legacy speed flag", dict(extra=_extra(**{guidance_common.DY_SUBSTEPS_OFF_KEY: "SPEED (SpectrumSPDKSampler)"})),
             "SPEED (SpectrumSPDKSampler) is on - SPEED steps aside (that sampler already changes how this pass samples)"),
            ("area cond", dict(guider=_guider(conds={"positive": [{"model_conds": {}, "area": (8, 8, 0, 0)}]})),
             speed_nodes.AREA_REASON),
            ("mask cond", dict(guider=_guider(conds={"positive": [{"model_conds": {}, "mask": object()}]})),
             speed_nodes.AREA_REASON),
            ("latent shapes", dict(guider=_guider(latent_shapes=[(1, 4, 8, 8), (1, 8, 4)])), speed_nodes.LATENT_SHAPES_REASON),
        ]
        for name, kwargs, reason in rows:
            with self.subTest(row=name):
                self.assertEqual(self._guard(**kwargs), reason)

    def test_spectrum_function_wrapper(self):
        def spectrum_unet_wrapper(apply, args):
            return apply(args["input"], args["timestep"], **args["c"])

        extra = {"model_options": {"model_function_wrapper": spectrum_unet_wrapper, "transformer_options": {}}}
        self.assertEqual(self._guard(extra=extra), speed_nodes.SPECTRUM_REASON)
        extra["model_options"]["model_function_wrapper"] = lambda apply, args: apply(args["input"], args["timestep"])
        self.assertIsNone(self._guard(extra=extra))

    def test_checks_run_in_the_extension_order(self):
        """Everything wrong at once: each fix reveals the next reason, in sam-extra's order."""
        cond = {"model_conds": {"ref_latents": 1, "c_concat": 1}, "control": 1, "area": (1,)}
        guider = _guider(EpsSampling(), conds={"positive": [cond]}, latent_shapes=[1, 2])
        extra = {"denoise_mask": 1, "model_options": {speed_nodes.LLLITE_MARKER: 1, "transformer_options": {
            guidance_common.DY_SUBSTEPS_OFF_KEY: "Spectrum"}}}
        state = {"sigmas": torch.tensor([1.5, 0.0])}
        fixes = {
            "not a flow-matching model (prediction_type=eps)":
                lambda: setattr(guider.model_patcher, "sampling", FlowSampling()),
            "sigma_max 1.500": lambda: state.update(sigmas=_flow_sigmas()),
            speed_nodes.MASK_REASON: lambda: extra.update(denoise_mask=None),
            speed_nodes.ANIMA_REFERENCE_REASON: lambda: cond["model_conds"].pop("ref_latents"),
            speed_nodes.CONCAT_REASON: lambda: cond["model_conds"].pop("c_concat"),
            speed_nodes.CONTROLNET_REASON: lambda: cond.pop("control"),
            speed_nodes.LLLITE_REASON: lambda: extra["model_options"].pop(speed_nodes.LLLITE_MARKER),
            speed_nodes.SPECTRUM_REASON: lambda: extra["model_options"]["transformer_options"].clear(),
            speed_nodes.AREA_REASON: lambda: cond.pop("area"),
            speed_nodes.LATENT_SHAPES_REASON: lambda: setattr(guider.inner_model, "latent_shapes", None),
        }
        order = list(fixes)
        seen = []
        while True:
            reason = self._guard(guider=guider, sigmas=state["sigmas"], extra=extra)
            if reason is None:
                break
            key = next(key for key in fixes if reason.startswith(key))
            seen.append(key)
            fixes.pop(key)()
        self.assertEqual(seen, order)

    def test_unsupported_samplers_are_found_through_cns(self):
        def sample_unipc_bh2(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return x

        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            return noise

        def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return x

        kd = SimpleNamespace(sample_unipc_bh2=sample_unipc_bh2, sample_euler=sample_euler)
        for function, reason in ((sample_unipc_bh2, "UniPC is not segment-safe"),
                                 (dpm_fast_function, "DPM fast has no sigma schedule"),
                                 (sample_euler, None)):
            for wrapped in (function, guidance_cns.cns_sampler_function(function, kd_module=kd)):
                with self.subTest(function=function.__name__, wrapped=wrapped is not function):
                    sampler = SimpleNamespace(sampler_function=wrapped, extra_options={})
                    self.assertEqual(speed_nodes.unsupported_sampler_reason(sampler), reason)

        class UNIPC:
            pass

        self.assertEqual(speed_nodes.unsupported_sampler_reason(UNIPC()), "UNIPC is not a k-diffusion sampler")


# --------------------------------------------------------------------------------------------------
# Host: seeds and sampler noise
# --------------------------------------------------------------------------------------------------


def _sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * x.new_ones([x.shape[0]]), **(extra_args or {}))
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "denoised": denoised})
        x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
    return x


def _ancestral(draws: list, *, name="sample_euler_ancestral"):
    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0,
                noise_sampler=None):
        if noise_sampler is None:
            raise AssertionError("the host gives every segment a noise sampler")
        for i in range(len(sigmas) - 1):
            denoised = model(x, sigmas[i] * x.new_ones([x.shape[0]]), **(extra_args or {}))
            x = denoised + (x - denoised) * (sigmas[i + 1] / sigmas[i])
            if sigmas[i + 1] > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1])
                draws.append(noise.clone())
                x = x + 0.01 * noise
        return x

    sampler.__name__ = name
    return sampler


class _Model:
    def __init__(self):
        self.seen = []

    def __call__(self, x, sigma, **extra):
        options = (extra.get("model_options") or {}).get("transformer_options") or {}
        listed = options.get("sample_sigmas")
        self.seen.append((tuple(x.shape), None if listed is None else (listed.clone(), guidance_common.republished_from(listed))))
        return x * 0.5


class _Tree:
    """Records how a BrownianTreeNoiseSampler was built; draws a fixed pattern."""

    made: list = []

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=None, cpu=False):
        _Tree.made.append(dict(shape=tuple(x.shape), low=sigma_min, high=sigma_max, seed=seed, cpu=cpu))
        self.tree = SimpleNamespace(cpu_tree=cpu)
        self.shape, self.dtype = tuple(x.shape), x.dtype

    def __call__(self, sigma, sigma_next):
        return torch.ones(self.shape, dtype=self.dtype)


_Tree.__name__ = "BrownianTreeNoiseSampler"


@requires_torch
class HostSeedAndNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_runner

        cls.runner = speed_runner

    def setUp(self):
        _Tree.made = []

    def _host(self, function, sigmas=None, *, seed=5, settings_seed=-1, image_seed=100, plan=None, extra=None):
        sigmas = _flow_sigmas() if sigmas is None else sigmas
        extra = {"seed": seed, "model_options": {"transformer_options": {"sample_sigmas": sigmas}}} if extra is None else extra
        return speed_nodes.ComfySpeedHost(
            sampler_function=function, sigmas=sigmas, extra_args=extra, settings_seed=settings_seed,
            image_seed=image_seed, plan=plan, kd_module=SimpleNamespace(BrownianTreeNoiseSampler=_Tree),
        ), sigmas, extra

    def test_seeds(self):
        host, _, _ = self._host(_sample_euler)
        self.assertEqual(host.seeds(3), [100, 101, 102])
        self.assertEqual(host.batch_image_seeds(2), [100, 101])
        fixed, _, _ = self._host(_sample_euler, settings_seed=7)
        self.assertEqual(fixed.seeds(2), [7, 7])
        self.assertEqual(fixed.batch_image_seeds(2), [100, 101])

    def test_one_stream_across_segments(self):
        for seed in (5, 0, 2 ** 40):
            with self.subTest(seed=seed):
                host, sigmas, _ = self._host(_ancestral([]), seed=seed)
                x = torch.zeros(2, 4, 16, 16)
                kwargs = host.sampler_kwargs({"eta": 0.5}, x)
                self.assertEqual(kwargs["eta"], 0.5)
                tail = kwargs["noise_sampler"]
                coarse0 = host.coarse_noise_sampler(tail, torch.zeros(2, 4, 4, 4), 0, sigmas)
                coarse1 = host.coarse_noise_sampler(tail, torch.zeros(2, 4, 8, 8, dtype=torch.bfloat16), 1, sigmas)
                got = [coarse0(1, 0), coarse0(1, 0), coarse1(1, 0), tail(1, 0), tail(1, 0)]
                generator = torch.Generator(device="cpu").manual_seed(seed + 1)   # ComfyUI: CPU seed + 1
                want = [torch.randn((2, 4, 4, 4), generator=generator) for _ in range(2)]
                want.append(torch.randn((2, 4, 8, 8), generator=generator, dtype=torch.bfloat16))
                want += [torch.randn((2, 4, 16, 16), generator=generator) for _ in range(2)]
                for a, b in zip(got, want):
                    self.assertEqual((a.dtype, tuple(a.shape)), (b.dtype, tuple(b.shape)))
                    self.assertTrue(torch.equal(a, b))
        unseeded, _, _ = self._host(_ancestral([]), extra={"model_options": {}})
        self.assertIn("noise_sampler", unseeded.sampler_kwargs({}, torch.zeros(1, 4, 8, 8)))
        self.assertIsNone(unseeded._stream.generator)

    def test_one_stream_through_a_whole_run(self):
        draws: list = []
        sampler = _ancestral(draws)
        sigmas = _flow_sigmas(16)
        plan = speed_schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                         scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.95, 0.8))
        host, _, extra = self._host(sampler, sigmas, plan=plan)
        x = torch.randn(1, 4, 16, 16, generator=torch.Generator().manual_seed(3))
        self.runner.run_speed(sampler, _Model(), x, sigmas, run=self.runner.SpeedRun(plan=plan, seeds=host.seeds(1)),
                              host=host, extra_args=extra, sampler_kwargs=host.sampler_kwargs({}, x))
        generator = torch.Generator().manual_seed(6)
        want = [torch.randn(tuple(d.shape), generator=generator) for d in draws]
        self.assertEqual([tuple(d.shape[-2:]) for d in draws],
                         [(4, 4)] * plan.transitions[0].step + [(8, 8)] * (plan.transitions[1].step - plan.transitions[0].step)
                         + [(16, 16)] * (len(sigmas) - 2 - plan.transitions[1].step))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(draws, want)))

    def test_samplers_without_a_noise_sampler_get_none(self):
        def sample_star(model, x, sigmas, **kwargs):
            return x

        for function in (_sample_euler, sample_star):
            with self.subTest(function=function.__name__):
                host, sigmas, _ = self._host(function)
                self.assertFalse(host.takes_noise_sampler)
                self.assertEqual(host.sampler_kwargs({"s_noise": 1.0, "noise_sampler": None}, torch.zeros(1, 4, 8, 8)),
                                 {"s_noise": 1.0})

    def test_cns_wrapped_sampler_is_read_through_its_base(self):
        draws: list = []
        sampler = _ancestral(draws)
        cns = guidance_cns.cns_sampler_function(sampler, kd_module=SimpleNamespace(sample_euler_ancestral=sampler))
        host, _, _ = self._host(cns)
        self.assertTrue(host.takes_noise_sampler)
        self.assertEqual(host.sampler_name, "sample_euler_ancestral")
        self.assertIn("noise_sampler", host.sampler_kwargs({}, torch.zeros(1, 4, 8, 8)))

    def test_a_callers_noise_sampler_is_kept(self):
        own = lambda sigma, sigma_next: None  # noqa: E731
        host, sigmas, _ = self._host(_ancestral([]))
        self.assertIs(host.sampler_kwargs({"noise_sampler": own}, torch.zeros(1, 4, 8, 8))["noise_sampler"], own)
        coarse = host.coarse_noise_sampler(own, torch.zeros(1, 4, 4, 4), 0, sigmas)
        self.assertEqual(tuple(coarse(1, 0).shape), (1, 4, 4, 4))   # coarse stages draw the run's stream

    def test_brownian_family(self):
        sigmas = _flow_sigmas(10)
        for name, cpu in speed_nodes.BROWNIAN_SAMPLERS.items():
            with self.subTest(sampler=name):
                _Tree.made = []
                host, _, _ = self._host(_ancestral([], name=name), sigmas, seed=9, image_seed=40)
                self.assertEqual(host.brownian_cpu, cpu)
                tail = host.sampler_kwargs({}, torch.zeros(2, 4, 16, 16))["noise_sampler"]
                host.coarse_noise_sampler(tail, torch.zeros(2, 4, 8, 8), 0, sigmas)
                host.coarse_noise_sampler(tail, torch.zeros(2, 4, 12, 12), 1, sigmas)
                stride = speed_nodes.BROWNIAN_STAGE_SEED_STRIDE
                self.assertEqual([(m["shape"], m["seed"], m["cpu"]) for m in _Tree.made], [
                    ((2, 4, 16, 16), 9, cpu),
                    ((2, 4, 8, 8), [40 + stride, 41 + stride], cpu),
                    ((2, 4, 12, 12), [40 + 2 * stride, 41 + 2 * stride], cpu),
                ])
                for made in _Tree.made:      # a plain ComfyUI SDE run's range, as tensors
                    self.assertTrue(torch.equal(made["low"], sigmas[sigmas > 0].min()))
                    self.assertTrue(torch.equal(made["high"], sigmas.max()))
        self.assertEqual(stride, 1_000_003)

    def test_brownian_template_from_the_caller(self):
        sigmas = _flow_sigmas(10)
        host, _, _ = self._host(_ancestral([]), sigmas, image_seed=3)
        template = _Tree(torch.zeros(1, 4, 16, 16), 0.1, 1.0, seed=1, cpu=False)
        _Tree.made = []
        self.assertIsInstance(host.coarse_noise_sampler(template, torch.zeros(1, 4, 8, 8), 0, sigmas), _Tree)
        self.assertEqual((_Tree.made[0]["seed"], _Tree.made[0]["cpu"]), ([3 + 1_000_003], False))

    def test_tree_range_covers_aligned_hand_offs(self):
        tail = _flow_sigmas(20)[-9:]                       # an img2img/hires pass: starts below 1
        for mode in speed_schedule.MODES:
            with self.subTest(mode=mode):
                plan = speed_schedule.build_plan(mode=mode, transform="dct", sigmas=tail, full_grid=(16, 16),
                                                 scales=(0.5, 1.0), threshold="manual",
                                                 manual_sigmas=(float(tail[1]) - 1e-4,))
                aligned = max(t.aligned for t in plan.transitions)
                self.assertGreater(aligned, float(tail.max()))
                _Tree.made = []
                host, _, _ = self._host(_ancestral([], name="sample_dpmpp_2m_sde"), tail, plan=plan)
                host.sampler_kwargs({}, torch.zeros(1, 4, 16, 16))
                high = _Tree.made[0]["high"]
                self.assertEqual(high.dtype, tail.dtype)
                self.assertEqual(float(high), float(torch.tensor(aligned, dtype=torch.float32)))
                self.assertTrue(torch.equal(_Tree.made[0]["low"], tail[tail > 0].min()))
        full = _flow_sigmas(20)
        plan = speed_schedule.build_plan(mode="transition", transform="dct", sigmas=full, full_grid=(16, 16),
                                         scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        _Tree.made = []
        host, _, _ = self._host(_ancestral([], name="sample_dpmpp_3m_sde"), full, plan=plan)
        host.sampler_kwargs({}, torch.zeros(1, 4, 16, 16))
        self.assertIs(_Tree.made[0]["high"].dtype, full.dtype)
        self.assertTrue(torch.equal(_Tree.made[0]["high"], full.max()))     # txt2img: the plain tree


# --------------------------------------------------------------------------------------------------
# Host: the published schedule
# --------------------------------------------------------------------------------------------------


@requires_torch
class HostPublishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_runner

        cls.runner = speed_runner

    def _host(self, sigmas, whole):
        extra = {"seed": 1, "model_options": {"transformer_options": {"sample_sigmas": whole}}}
        host = speed_nodes.ComfySpeedHost(sampler_function=_sample_euler, sigmas=sigmas, extra_args=extra)
        return host, extra["model_options"]["transformer_options"]

    def test_publish_and_restore(self):
        sigmas = _flow_sigmas(8)
        host, options = self._host(sigmas, sigmas)
        patched = sigmas.clone()
        patched[3] = 0.9
        host.publish_sigmas(patched)
        first = options["sample_sigmas"]
        self.assertIsNot(first, sigmas)
        self.assertTrue(torch.equal(first, patched))
        self.assertIs(guidance_common.republished_from(first), sigmas)
        patched[5] = 0.7
        host.publish_sigmas(patched.double())
        second = options["sample_sigmas"]
        self.assertEqual(second.dtype, sigmas.dtype)
        self.assertIs(guidance_common.republished_from(second), sigmas)
        self.assertTrue(torch.equal(first[5:6], sigmas[5:6]))     # an earlier copy is not changed
        host.restore_sigmas()
        self.assertIs(options["sample_sigmas"], sigmas)

    def test_offset_list_and_mismatch(self):
        whole = _flow_sigmas(12)
        tail = whole[4:].clone()
        host, options = self._host(tail, whole)
        patched = tail.clone()
        patched[1] = 0.5
        host.publish_sigmas(patched)
        published = options["sample_sigmas"]
        self.assertTrue(torch.equal(published[:4], whole[:4]) and torch.equal(published[4:], patched))
        self.assertIs(guidance_common.republished_from(published), whole)
        host.restore_sigmas()
        self.assertIs(options["sample_sigmas"], whole)
        other = whole.clone()
        other[-2] += 1e-3
        host, options = self._host(tail, other)
        host.publish_sigmas(patched)
        host.restore_sigmas()
        self.assertIs(options["sample_sigmas"], other)
        bare = speed_nodes.ComfySpeedHost(sampler_function=_sample_euler, sigmas=tail, extra_args=None)
        bare.publish_sigmas(patched)
        bare.restore_sigmas()

    def test_model_sees_the_published_list_during_the_run(self):
        sigmas = _flow_sigmas(12)
        plan = speed_schedule.build_plan(mode="respace", transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                         scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        host, options = self._host(sigmas, sigmas)
        extra = {"seed": 1, "model_options": {"transformer_options": options}}
        model = _Model()
        self.runner.run_speed(_sample_euler, model, torch.zeros(1, 4, 16, 16), sigmas,
                              run=self.runner.SpeedRun(plan=plan, seeds=[1]), host=host, extra_args=extra)
        step = plan.transitions[0].step
        coarse, full = model.seen[:step], model.seen[step:]
        self.assertTrue(all(listed[0] is not None and torch.equal(listed[0], sigmas) and listed[1] is None
                            for _shape, listed in coarse))
        for _shape, (listed, origin) in full:
            self.assertIs(origin, sigmas)
            self.assertFalse(torch.equal(listed, sigmas))
        self.assertIs(options["sample_sigmas"], sigmas)

    def test_restored_after_a_failure(self):
        sigmas = _flow_sigmas(12)
        plan = speed_schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                         scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        host, options = self._host(sigmas, sigmas)
        calls = []

        def failing(model, x, sigmas, extra_args=None, callback=None, disable=None):
            calls.append(tuple(x.shape))
            if len(calls) == 2:
                raise RuntimeError("tail failed")
            return x

        with self.assertRaises(RuntimeError):
            self.runner.run_speed(failing, _Model(), torch.zeros(1, 4, 16, 16), sigmas,
                                  run=self.runner.SpeedRun(plan=plan, seeds=[1]), host=host,
                                  extra_args={"model_options": {"transformer_options": options}})
        self.assertIs(options["sample_sigmas"], sigmas)


# --------------------------------------------------------------------------------------------------
# ComfyUI's own tables (AST of the ComfyUI source; nothing imported)
# --------------------------------------------------------------------------------------------------


@unittest.skipUnless(_COMFY_ROOT, "Set AISTUDIO_COMFY_TEST_ROOT to read ComfyUI's sampler source")
class ComfyTableTests(unittest.TestCase):
    def _module(self, relative: str) -> ast.Module:
        return ast.parse((Path(_COMFY_ROOT) / relative).read_text(encoding="utf-8"))

    def test_brownian_table_matches_comfy(self):
        tree = self._module("comfy/k_diffusion/sampling.py")
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        table: dict[str, bool] = {}
        for name, node in functions.items():
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "BrownianTreeNoiseSampler":
                    flags = [kw.value.value for kw in call.keywords if kw.arg == "cpu"]
                    self.assertEqual(len(flags), 1, name)
                    table[name] = flags[0]
        for name, node in functions.items():     # one-line delegations: return sample_x(..., noise_sampler=...)
            body = [item for item in node.body if not isinstance(item, ast.Expr)]
            if name not in table and len(body) == 1 and isinstance(body[0], ast.Return):
                call = body[0].value
                target = getattr(getattr(call, "func", None), "id", None)
                if isinstance(call, ast.Call) and target in table:
                    table[name] = table[target]
        pack = {name: cpu for name, cpu in speed_nodes.BROWNIAN_SAMPLERS.items() if name != "sample_dpmpp_4m_sde"}
        self.assertEqual(table, pack)

    def test_unsupported_comfy_names_exist(self):
        samplers = self._module("comfy/samplers.py")
        inner = {node.name for node in ast.walk(samplers) if isinstance(node, ast.FunctionDef)}
        self.assertLessEqual({"dpm_fast_function", "dpm_adaptive_function"}, inner)
        unipc = self._module("comfy/extra_samplers/uni_pc.py")
        self.assertLessEqual({"sample_unipc", "sample_unipc_bh2"},
                             {node.name for node in unipc.body if isinstance(node, ast.FunctionDef)})


# --------------------------------------------------------------------------------------------------
# Part 2 (PB-SP5): the sampler wrapper, the node, the Hires carry, Detail Daemon and LLLite
# --------------------------------------------------------------------------------------------------


class _Executor:
    """``comfy.patcher_extension.WrapperExecutor`` as the wrappers use it (ComfyUI 0.36.0)."""

    def __init__(self, original, class_obj, wrappers, idx=0):
        self.original, self.class_obj, self.wrappers, self.idx = original, class_obj, list(wrappers), idx

    @classmethod
    def new_class_executor(cls, original, class_obj, wrappers, idx=0):
        return cls(original, class_obj, wrappers, idx)

    def execute(self, *args, **kwargs):
        if self.idx >= len(self.wrappers):
            return self.original(*args, **kwargs)
        return self.wrappers[self.idx](self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return type(self)(self.original, self.class_obj, self.wrappers, self.idx + 1).execute(*args, **kwargs)


class _KSampler:
    """``comfy.samplers.KSAMPLER`` (0.36.0): the flow noise scaling, then ``sampler_function``."""

    def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
        self.sampler_function = sampler_function
        self.extra_options = {} if extra_options is None else extra_options
        self.inpaint_options = {} if inpaint_options is None else inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None,
               disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = _X0Inpaint(model_wrap, sigmas)
        if latent_image is not None:
            noise = sigmas[0] * noise + (1.0 - sigmas[0]) * latent_image
        total = len(sigmas) - 1
        k_callback = None if callback is None else (lambda d: callback(d["i"], d["denoised"], d["x"], total))
        return self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback,
                                     disable=disable_pbar, **self.extra_options)


class _X0Inpaint:
    def __init__(self, inner_model, sigmas):
        self.inner_model, self.sigmas = inner_model, sigmas

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None):
        return self.inner_model(x, sigma, model_options=model_options or {}, seed=seed)


class _Guider:
    """A CFGGuider stand-in on a flow model: records every model call (shape, sigma, schedule seen)."""

    def __init__(self, sampling=None, cfg=4.0):
        self.model_patcher = _Patcher(FlowSampling() if sampling is None else sampling)
        self.inner_model = Anima()
        self.inner_model.latent_shapes = None
        self.conds = {"positive": [{"model_conds": {}}], "negative": [{"model_conds": {}}]}
        self.cfg = cfg
        self.calls = []

    def __call__(self, x, sigma, model_options=None, seed=None):
        options = (model_options or {}).get("transformer_options") or {}
        listed = options.get("sample_sigmas")
        self.calls.append((tuple(x.shape), sigma.clone(), None if listed is None else listed.clone(),
                           options.get(guidance_common.PRE_DD_SIGMAS_KEY)))
        return x * (1.0 / (1.0 + sigma.reshape(-1, *([1] * (x.ndim - 1)))))


class _ModelPatcher:
    """The ModelPatcher wrapper surface (clone, add/remove by key) the pack's patches use."""

    def __init__(self):
        self.wrappers: dict = {}
        self.model_options: dict = {"transformer_options": {}}

    def clone(self):
        copy = _ModelPatcher()
        copy.wrappers = {kind: {key: list(items) for key, items in table.items()} for kind, table in self.wrappers.items()}
        copy.model_options = {**self.model_options, "transformer_options": dict(self.model_options["transformer_options"])}
        return copy

    def add_wrapper_with_key(self, wrapper_type, key, function):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(function)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def flat(self, wrapper_type="sampler_sample"):
        return [function for items in self.wrappers.get(wrapper_type, {}).values() for function in items]


class InterruptProcessingException(BaseException):
    """Stands for ``comfy.model_management.InterruptProcessingException`` (only the name is read)."""


def _fake_comfy():
    import types

    comfy = types.ModuleType("comfy")
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSAMPLER = _KSampler
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = SimpleNamespace(SAMPLER_SAMPLE="sampler_sample")
    comfy.samplers, comfy.patcher_extension = samplers, extension
    return {"comfy": comfy, "comfy.samplers": samplers, "comfy.patcher_extension": extension}


def _probe_sampler(log: list, name: str = "sample_probe"):
    def probe(model, x, sigmas, extra_args=None, callback=None, disable=None, sam_extra_step_offset=None):
        log.append(sam_extra_step_offset)
        return _sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

    probe.__name__ = name
    return probe


@requires_torch
class SpeedWrapperTests(unittest.TestCase):
    """One sampling run through the wrapper: applied, skipped, refused, fallback, interrupt, once."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_runner

        cls.runner = speed_runner

    def _patch(self, label="base", **values):
        settings = speed_nodes.coerce_settings({"enabled": True, "threshold": "manual", "manual_sigmas": "0.8", **values})
        self.assertIsNone(settings.error)
        return speed_nodes.SpeedPatch(settings=settings, label=label, image_seed=40)

    def _run(self, sampler_function, patch, *, guider=None, sigmas=None, x=None, wrappers=None, seed=7):
        guider = _Guider() if guider is None else guider
        sigmas = _flow_sigmas(12) if sigmas is None else sigmas
        noise = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(2)) if x is None else x
        ksampler = _KSampler(sampler_function)
        wrapper = speed_nodes.speed_sampler_sample_wrapper(patch)
        extra = {"model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "seed": seed}
        steps = []
        executor = _Executor.new_class_executor(ksampler.sample, ksampler, [wrapper] if wrappers is None else wrappers)
        out = executor.execute(guider, sigmas, extra, lambda i, d, xx, total: steps.append((i, total)), noise, None,
                               None, True)
        self.assertIs(ksampler.sampler_function, sampler_function, "the run's sampler function is restored")
        self.assertIs(extra["model_options"]["transformer_options"]["sample_sigmas"], sigmas, "schedule restored")
        return out, wrapper, guider, steps

    def _plain(self, sampler_function, *, sigmas=None, x=None):
        sigmas = _flow_sigmas(12) if sigmas is None else sigmas
        noise = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(2)) if x is None else x
        guider = _Guider()
        ksampler = _KSampler(sampler_function)
        out = ksampler.sample(guider, sigmas, {"model_options": {"transformer_options": {"sample_sigmas": sigmas}},
                                               "seed": 7}, None, noise)
        return out, guider

    def test_applied_run_equals_run_speed(self):
        patch = self._patch()
        out, wrapper, guider, steps = self._run(_sample_euler, patch)
        status = wrapper._ai_studio_speed_status.text
        self.assertTrue(status.startswith("applied (transition) - "), status)
        shapes = [call[0] for call in guider.calls]
        self.assertIn((1, 4, 8, 8), shapes)
        self.assertEqual(shapes[-1], (1, 4, 16, 16))
        self.assertEqual([i for i, _ in steps], list(range(12)))          # progress counts the whole pass
        self.assertEqual({total for _, total in steps}, {12})
        # the same run by hand: run_speed with the host the wrapper builds
        sigmas = _flow_sigmas(12)
        noise = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(2))
        plan = speed_nodes.plan_for(patch.settings, noise, sigmas)
        extra = {"model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "seed": 7, "denoise_mask": None}
        host = speed_nodes.ComfySpeedHost(sampler_function=_sample_euler, sigmas=sigmas, extra_args=extra,
                                          image_seed=40, plan=plan)
        by_hand, report = self.runner.run_speed(
            _sample_euler, _X0Inpaint(_Guider(), sigmas), noise, sigmas,
            run=self.runner.SpeedRun(plan=plan, seeds=host.seeds(1)), host=host, extra_args=extra)
        self.assertTrue(torch.equal(out, by_hand))
        self.assertEqual(status, f"applied (transition) - {report.summary(plan)}")

    def test_logs_the_extension_lines(self):
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            self._run(_sample_euler, self._patch(label="hires", manual_sigmas="0.6"), sigmas=_flow_sigmas(20)[-9:])
        lines = [record.getMessage() for record in logs.records]
        plan_lines = [line for line in lines if line.startswith("[AnimaSPEED] hires: transition plan - step ")]
        self.assertEqual(len(plan_lines), 1, lines)
        self.assertIn("steps coarse", plan_lines[0])
        self.assertTrue(lines[-1].startswith("[AnimaSPEED] hires: done - "), lines)
        self.assertIn("init=rescaled 1/2.000", lines[-1])                  # hires start: flow-rescaled

    def test_img2img_rescale_off(self):
        patch = speed_nodes.SpeedPatch(settings=self._patch(manual_sigmas="0.6").settings, label="img2img", image_seed=40,
                                       img2img_rescale=False)
        _out, wrapper, _guider, _steps = self._run(_sample_euler, patch, sigmas=_flow_sigmas(20)[-9:])
        self.assertIn("init=upstream (not rescaled)", wrapper._ai_studio_speed_status.text)

    def test_guard_and_plan_refusals_sample_plainly(self):
        tail = _flow_sigmas(20)[-9:]                     # starts at sigma 0.67, below the 0.8 hand-off
        rows = [
            (dict(guider=_Guider(EpsSampling())), self._patch(), "skipped - not a flow-matching model (prediction_type=eps)"),
            (dict(sigmas=tail), self._patch(), "skipped - no coarse steps: the pass starts at sigma 0.6667"),
            (dict(), self._patch(threshold="neo_shift", preset="custom", A=-1), "invalid settings - "),
        ]
        for kwargs, patch, prefix in rows:
            with self.subTest(status=prefix):
                plain, _ = self._plain(_sample_euler, sigmas=kwargs.get("sigmas"))
                out, wrapper, guider, _steps = self._run(_sample_euler, patch, **kwargs)
                self.assertTrue(wrapper._ai_studio_speed_status.text.startswith(prefix), wrapper._ai_studio_speed_status.text)
                self.assertTrue(all(call[0] == (1, 4, 16, 16) for call in guider.calls))
                self.assertTrue(torch.equal(out, plain))

    def test_unsupported_samplers_are_not_wrapped(self):
        seen = []

        def sample_unipc(model, x, sigmas, extra_args=None, callback=None, disable=None):
            seen.append(tuple(x.shape))
            return x

        _out, wrapper, _guider, _steps = self._run(sample_unipc, self._patch())
        self.assertEqual(wrapper._ai_studio_speed_status.text, "skipped - UniPC is not segment-safe")
        self.assertEqual(seen, [(1, 4, 16, 16)])

    def test_failure_falls_back_to_plain_sampling(self):
        def coarse_fails(model, x, sigmas, extra_args=None, callback=None, disable=None):
            if x.shape[-1] != 16:
                raise RuntimeError("coarse grid refused")
            return _sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

        plain, _ = self._plain(coarse_fails)
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            out, wrapper, _guider, _steps = self._run(coarse_fails, self._patch())
        self.assertEqual(wrapper._ai_studio_speed_status.text,
                         "fell back to plain sampling after RuntimeError: coarse grid refused")
        self.assertTrue(torch.equal(out, plain))
        self.assertTrue(any("SPEED failed, sampling again without it:" in record.getMessage()
                            and "Traceback" in record.getMessage() for record in logs.records))

    def test_interrupt_goes_through(self):
        def interrupted(model, x, sigmas, extra_args=None, callback=None, disable=None):
            raise InterruptProcessingException()

        sigmas = _flow_sigmas(12)
        ksampler = _KSampler(interrupted)
        wrapper = speed_nodes.speed_sampler_sample_wrapper(self._patch())
        extra = {"model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "seed": 1}
        with self.assertRaises(InterruptProcessingException):
            _Executor.new_class_executor(ksampler.sample, ksampler, [wrapper]).execute(
                _Guider(), sigmas, extra, None, torch.zeros(1, 4, 16, 16), None, None, True)
        self.assertEqual(wrapper._ai_studio_speed_status.text, "interrupted")
        self.assertIs(ksampler.sampler_function, interrupted)
        self.assertIs(extra["model_options"]["transformer_options"]["sample_sigmas"], sigmas)

    def test_only_the_first_call_of_a_run_is_segmented(self):
        guider = _Guider()
        function = speed_nodes.speed_sampler_function(_sample_euler, guider, self._patch())
        sigmas = _flow_sigmas(12)
        extra = {"model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "seed": 1, "denoise_mask": None}
        model = _X0Inpaint(guider, sigmas)
        function(model, torch.zeros(1, 4, 16, 16), sigmas, extra_args=extra)
        first = len(guider.calls)
        self.assertIn((1, 4, 8, 8), [call[0] for call in guider.calls])
        function(model, torch.zeros(1, 4, 16, 16), sigmas, extra_args=extra)
        self.assertTrue(all(call[0] == (1, 4, 16, 16) for call in guider.calls[first:]))
        self.assertIs(getattr(function, guidance_common.SAMPLER_BASE_ATTR), _sample_euler)

    def test_step_offset_reaches_the_sampler_through_cns(self):
        offsets: list = []
        probe = _probe_sampler(offsets)
        cns = guidance_cns.cns_sampler_sample_wrapper(kd_module=SimpleNamespace(sample_probe=probe))
        patch = self._patch()
        speed = speed_nodes.speed_sampler_sample_wrapper(patch)
        _out, _wrapper, _guider, _steps = self._run(probe, patch, wrappers=[cns, speed])
        plan = speed_nodes.plan_for(patch.settings, torch.zeros(1, 4, 16, 16), _flow_sigmas(12))
        self.assertEqual(offsets, [0] + [t.step for t in plan.transitions])
        self.assertTrue(speed._ai_studio_speed_status.text.startswith("applied"))


@requires_torch
class SpeedNodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def test_inputs_are_explicit_and_default_to_the_extension(self):
        required = speed_nodes.ForgeNeoAnimaSpeed.INPUT_TYPES()["required"]
        self.assertEqual(list(required), [
            "model", "enabled", "mode", "threshold", "preset", "scales", "delta", "sigma_divisor", "manual_sigmas",
            "adaptive", "transform", "spectrum_A", "spectrum_beta", "seed", "pass_label", "image_seed",
            "img2img_rescale",
        ])
        defaults = {name: spec[1]["default"] for name, spec in required.items() if name != "model"}
        settings = speed_nodes.coerce_settings({key: value for key, value in defaults.items()
                                                if key in speed_nodes.ARG_NAMES})
        self.assertEqual(settings, speed_nodes.SpeedSettings())
        self.assertEqual(required["mode"][0], list(speed_schedule.MODES))
        self.assertEqual(required["preset"][0], list(speed_schedule.PRESET_NAMES))
        self.assertEqual(required["pass_label"][0], list(speed_nodes.PASS_LABELS))
        self.assertEqual((required["seed"][1]["min"], required["image_seed"][1]["min"]), (-1, 0))
        self.assertEqual(speed_nodes.NODE_CLASS_MAPPINGS, {"ForgeNeoAnimaSpeed": speed_nodes.ForgeNeoAnimaSpeed})
        self.assertIn("ForgeNeoAnimaSpeed", speed_nodes.NODE_DISPLAY_NAME_MAPPINGS)

    def test_patch(self):
        node = speed_nodes.ForgeNeoAnimaSpeed()
        base = _ModelPatcher()
        self.assertIs(node.patch(base)[0], base)                               # off
        with self.assertLogs("ai_studio_forge_parity", level="INFO") as logs:
            self.assertIs(node.patch(base, enabled=True, mode="bogus")[0], base)
            self.assertIs(node.patch(base, enabled=True, scales="1/0,1")[0], base)
        self.assertIn("base: invalid settings, not applied - mode must be one of", logs.records[0].getMessage())
        self.assertIn("base: invalid settings, not applied - scales: division by zero: '1/0'",
                      logs.records[1].getMessage())
        with self.assertRaises(ValueError):
            node.patch(base, enabled=True, pass_label="refine")
        (patched,) = node.patch(base, enabled=True, mode="respace", pass_label="hires", image_seed=99, seed=5,
                                img2img_rescale=False)
        self.assertIsNot(patched, base)
        self.assertEqual(base.wrappers, {})
        patch = speed_nodes.model_speed_patch(patched)
        self.assertEqual((patch.label, patch.image_seed, patch.img2img_rescale, patch.settings.mode, patch.settings.seed),
                         ("hires", 99, False, "respace", 5))
        (again,) = node.patch(patched, enabled=True)
        self.assertEqual(len(again.wrappers["sampler_sample"][speed_nodes.SPEED_WRAPPER_KEY]), 1)   # replaced
        self.assertEqual(speed_nodes.model_speed_patch(again).label, "base")

    def test_wrapper_order_cns_speed_dd(self):
        from unittest import mock

        from comfy_custom_nodes.ai_studio_forge_parity import guidance_dd

        with mock.patch.dict(sys.modules, _fake_comfy()):
            dd_model = guidance_dd.ForgeNeoAnimaDetailDaemon().patch(_ModelPatcher(), True, "{}")[0]
            speed_model = speed_nodes.ForgeNeoAnimaSpeed().patch(dd_model, enabled=True)[0]
            cns_model = guidance_cns.apply_cns(speed_model)
            later_dd = guidance_dd.ForgeNeoAnimaDetailDaemon().patch(speed_nodes.ForgeNeoAnimaSpeed().patch(
                _ModelPatcher(), enabled=True)[0], True, "{}")[0]
        keys = list(cns_model.wrappers["sampler_sample"])
        self.assertEqual(keys, [guidance_cns.CNS_WRAPPER_KEY, speed_nodes.SPEED_WRAPPER_KEY, guidance_dd.DD_WRAPPER_KEY])
        self.assertEqual(list(later_dd.wrappers["sampler_sample"]),
                         [speed_nodes.SPEED_WRAPPER_KEY, guidance_dd.DD_WRAPPER_KEY])

    def test_hires_override_carries_the_patch(self):
        from comfy_custom_nodes.ai_studio_forge_parity import hires_carry

        self.assertIn(speed_nodes.CARRY_KEY, hires_carry.registered())
        source = speed_nodes.ForgeNeoAnimaSpeed().patch(_ModelPatcher(), enabled=True, pass_label="hires",
                                                         image_seed=12)[0]
        target = _ModelPatcher()
        target.add_wrapper_with_key("sampler_sample", "other", object())
        carried = speed_nodes._carry_to_override(source, target, None)
        self.assertIsNot(carried, target)
        self.assertEqual(list(carried.wrappers["sampler_sample"]), [speed_nodes.SPEED_WRAPPER_KEY, "other"])
        self.assertIs(carried.wrappers["sampler_sample"][speed_nodes.SPEED_WRAPPER_KEY][0],
                      source.wrappers["sampler_sample"][speed_nodes.SPEED_WRAPPER_KEY][0])
        self.assertEqual(list(target.wrappers["sampler_sample"]), ["other"])
        self.assertIsNone(speed_nodes._carry_to_override(_ModelPatcher(), target, None))
        self.assertEqual(speed_nodes.model_speed_patch(carried).image_seed, 12)

    def test_lllite_marker(self):
        from unittest import mock

        from comfy_custom_nodes.ai_studio_forge_parity import anima_lllite

        self.assertEqual(speed_nodes.LLLITE_MARKER, anima_lllite.LLLITE_MODEL_OPTION)
        self.assertEqual(anima_lllite.LLLITE_MODEL_OPTION, "forge_neo_lllite")
        base = _ModelPatcher()

        class _Node:
            def apply(self, model, *args, **kwargs):
                clone = model.clone()
                clone.model_options["model_function_wrapper"] = lambda apply, args: apply(args)
                return (clone,)

        with mock.patch.object(anima_lllite, "_vendored_nodes",
                               return_value=SimpleNamespace(AnimaLLLiteApply_sdscripts=_Node)):
            patched = anima_lllite.apply_lllite(base, "x.safetensors", None, 1.0, 0.0, 1.0)
        self.assertIs(patched.model_options[anima_lllite.LLLITE_MODEL_OPTION], True)
        self.assertNotIn(anima_lllite.LLLITE_MODEL_OPTION, base.model_options)
        reason = speed_nodes.comfy_runtime_guard(_Guider(), torch.zeros(1, 4, 8, 8), _flow_sigmas(),
                                                 {"model_options": patched.model_options})
        self.assertEqual(reason, speed_nodes.LLLITE_REASON)


def _dd_values(**overrides):
    from comfy_custom_nodes.ai_studio_forge_parity import guidance_dd

    settings = {"dd_amount": 0.4, "dd_start": 0.0, "dd_end": 1.0, "dd_bias": 0.5, "dd_smooth": True, **overrides}
    return guidance_dd.detail_daemon_node_values(settings, 0.0)


@requires_torch
class DetailDaemonFollowsSpeedTests(unittest.TestCase):
    """DD inside SPEED looks each sigma up in the schedule SPEED re-published (sam-extra's DD)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_dd

        cls.dd = guidance_dd

    def test_the_vendored_lookup_list_is_a_closure_cell(self):
        inner = [const for const in self.dd.detail_daemon_sampler.__code__.co_consts
                 if getattr(const, "co_name", None) == "model_wrapper"]
        self.assertEqual(len(inner), 1)
        self.assertIn("sigmas_cpu", inner[0].co_freevars)

    def test_published_schedule_unit(self):
        own = _flow_sigmas(8)

        def make():
            sigmas_cpu = own.clone()

            def model_wrapper(x, sigma, **extra_args):
                return sigmas_cpu

            return model_wrapper

        wrapper = make()
        follower = self.dd._PublishedSchedule(wrapper, own)
        table = wrapper(None, None)
        patched = guidance_common.mark_republished(own.clone().index_fill_(0, torch.tensor([3]), 0.9), own)
        follower.follow(patched)
        self.assertTrue(torch.equal(table, patched))
        follower.follow(own)
        self.assertTrue(torch.equal(table, own))
        untagged = patched.clone()                 # same length and first sigma, but not a re-published copy
        follower.follow(untagged)
        self.assertTrue(torch.equal(table, own))
        other = guidance_common.mark_republished(_flow_sigmas(9), own)          # another length: ignored
        follower.follow(other)
        self.assertTrue(torch.equal(table, own))
        handed_on = guidance_common.mark_republished(patched.clone(), own.clone())  # same length + first sigma
        follower.follow(handed_on)
        self.assertTrue(torch.equal(table, patched))
        follower.follow(None)
        self.assertTrue(torch.equal(table, own))
        self.assertIsNone(self.dd._PublishedSchedule(lambda x, s: None, own).table)

    def _stacked_run(self, mode, *, speed=True):
        from unittest import mock

        # shift 2.5: the aligned hand-off sigma (0.875) is not one of the run's own sigmas, so the run's
        # list would interpolate there (with shift 3 it would hit sigma[4] exactly and hide the rule)
        sigmas = _flow_sigmas(12, shift=2.5)
        guider = _Guider(cfg=4.0)
        values = _dd_values()
        patch = speed_nodes.SpeedPatch(settings=speed_nodes.coerce_settings({
            "enabled": True, "mode": mode, "threshold": "manual", "manual_sigmas": "0.8"}), image_seed=3)
        with mock.patch.dict(sys.modules, _fake_comfy()):
            dd = self.dd.detail_daemon_sampler_sample_wrapper(values)
            wrappers = [speed_nodes.speed_sampler_sample_wrapper(patch), dd] if speed else [dd]
            ksampler = _KSampler(_sample_euler)
            extra = {"model_options": {"transformer_options": {"sample_sigmas": sigmas}}, "seed": 1}
            _Executor.new_class_executor(ksampler.sample, ksampler, wrappers).execute(
                guider, sigmas, extra, None, torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(4)),
                None, None, True)
        schedule = torch.tensor(self.dd.detail_daemon_schedule(len(sigmas) - 1, values), dtype=torch.float32)
        plan = speed_nodes.plan_for(patch.settings, torch.zeros(1, 4, 16, 16), sigmas)
        return guider, schedule, plan, sigmas

    @staticmethod
    def _scaled(sigma_in, value, cfg=4.0):
        return sigma_in * max(1e-06, 1.0 - value * 0.1 * cfg)

    def test_transition_hand_off_step_uses_its_own_schedule_value(self):
        guider, schedule, plan, _sigmas = self._stacked_run("transition")
        step = plan.transitions[0].step
        shape, sigma_out, listed, pre_dd = guider.calls[step]
        self.assertEqual(shape, (1, 4, 16, 16))
        self.assertEqual(float(pre_dd), float(torch.tensor(plan.transitions[0].aligned, dtype=torch.float32)))
        self.assertTrue(torch.equal(sigma_out, self._scaled(pre_dd, schedule[step].item())))
        self.assertEqual(float(listed[step]), float(pre_dd))         # the model saw the published schedule
        self.assertNotEqual(float(schedule[step]), 0.0)
        own_list = self.dd.get_dd_schedule(float(pre_dd), _sigmas, schedule)
        self.assertNotEqual(own_list, schedule[step].item())          # the run's own list would interpolate
        self.assertFalse(torch.equal(sigma_out, self._scaled(pre_dd, own_list)))

    def test_respace_every_tail_step_uses_its_own_schedule_value(self):
        guider, schedule, plan, sigmas = self._stacked_run("respace")
        step = plan.transitions[0].step
        tail = guider.calls[step:]
        self.assertEqual(len(tail), len(sigmas) - 1 - step)
        for offset, (shape, sigma_out, _listed, pre_dd) in enumerate(tail):
            with self.subTest(step=step + offset):
                self.assertEqual(shape, (1, 4, 16, 16))
                self.assertTrue(torch.equal(sigma_out, self._scaled(pre_dd, schedule[step + offset].item())))

    def test_coarse_steps_and_plain_runs_use_the_runs_own_list(self):
        guider, schedule, plan, sigmas = self._stacked_run("transition")
        for index, (_shape, sigma_out, _listed, pre_dd) in enumerate(guider.calls[:plan.transitions[0].step]):
            self.assertTrue(torch.equal(pre_dd, sigmas[index:index + 1]))
            self.assertTrue(torch.equal(sigma_out, self._scaled(pre_dd, schedule[index].item())))
        plain, schedule, _plan, sigmas = self._stacked_run("transition", speed=False)
        for index, (_shape, sigma_out, _listed, pre_dd) in enumerate(plain.calls):
            self.assertTrue(torch.equal(sigma_out, self._scaled(pre_dd, schedule[index].item())))

if __name__ == "__main__":
    unittest.main()
