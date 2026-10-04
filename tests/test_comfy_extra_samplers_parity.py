"""Pack 1.7.0 Extra Samplers = sam-extra v0.33.0 Extra Samplers (CPU, no GPU, no ComfyUI).

The pack re-implements sam-extra's Extra Samplers for ComfyUI (``comfy_custom_nodes/ai_studio_forge_parity/
extra_samplers_*.py``). sam-extra is GPL-3.0 and is never imported by the pack; here the **installed**
extension (``tests/_sam_extra_ext.EXT_ROOT``) is the oracle: both run on **Forge's own k-diffusion module**
(``tests/_forge_k_diffusion.forge_k_sampling`` — the pack takes it as ``kd``, sam-extra finds it as
``k_diffusion.sampling``) with the same toy model, sigmas and deterministic noise, and the latents must be
equal bit for bit.

* ``SettingsCoercionTests`` — pure, no torch: the five values' coercion (``params.coerce_*``,
  ``settings_from_args``, ``resolve``'s dict form) and the constants, on a value grid.
* ``ErSdeParityTests`` — torch: ER SDE (Reverse-time) / (ODE) / (Tunable) over eps/flow, float32/float64,
  4-D/5-D, stages 1–3, η {0, 0.37, 1, 2.5, 10}, s_noise {0, 1} and the noise window {off, full, empty,
  0.2–0.8, …}; the run's values from ``model_options`` == the values Forge's sampler passes
  (``registry.runtime_kwargs``); the window's default noise, the callback, the guards.
* ``Dpmpp4mParityTests`` — torch: DPM++ 4M SDE over eps/flow, float32/float64, 6/8/13 steps, η {0, 0.5, 1,
  1.7}, s_noise {0, 1}; the first three orders == the host's ``sample_dpmpp_3m_sde``; the default noise
  (a CPU Brownian tree seeded from ``extra_args["seed"]``) and ComfyUI's ``noise_scale``.
* ``FlowOdeParityTests`` — torch: DPM++ 2M / 2M Heun / 3M (flow ODE) == sam-extra's ``dpmpp_flow`` (eps/flow,
  float32/float64, 4-D/5-D, 1/2/3/6/28 steps, latents, model calls, callback) and == the host's DPM++ 2M/3M SDE
  at η 0 with a real noise sampler that is never called; sam-extra's signatures (no ``eta``, no ``noise_sampler``).
* ``RestartFlowParityTests`` — torch: Restart (flow) (written from the paper) == sam-extra's ``restart_flow`` — the
  constants, the plan on ~150 grids (shift-3 flow, Karras, random, "no room"), the jump coefficients, and the sampler
  over eps/flow, float32/float64, 4-D/5-D, 19/20/28/36/50 steps, s_noise {1, 0.7} (latents, draws, model calls,
  callback); without a plan == the host's ``sample_heun``; the re-walk order; the default noise.
* ``EulerDyParityTests`` — torch: Euler Dy / Euler SMEA Dy and their CFG++ entries with a CFG stand-in that runs
  the post-CFG list (eps/flow, float32/float64, churn {0, 1.5}, ``sam_extra_step_offset`` {0, 2}, sub-steps on
  and off, 4-D/5-D, odd and 1×1 sizes), the core's other rules (upstream's ``max`` churn, ``s_tmin``/``s_tmax``,
  CFG++ without an unconditional x0), where the sub-steps sit and what they carry (``sam_extra_substep``), and
  ``substeps=False`` == the host's ``sample_euler`` / ``sample_euler_cfg_pp``.
* ``DyHostTests`` — torch, no extension (Forge's k-diffusion for the maths): ComfyUI's ``KSamplerX0Inpaint``
  inputs (denoise mask, inpainting latent, noise) follow each sub-step and come back (also when it raises),
  an all-ones mask gives the unmasked latent bit for bit; the guard's reasons (DY_SUBSTEPS_OFF_KEY, a
  Spectrum wrapper, concat conditioning, regional mask/area), logged once, and the guarded entry points.

The real-ComfyUI tier is ``tests/test_comfy_extra_samplers_runtime.py``.

Skipped without the extension, failing under ``AISTUDIO_REQUIRE_FORGE_EXT=1``; the torch classes also need
Forge's files (``SAM3_FORGE_ROOT`` or two levels above the extension). torch is imported lazily
(``tests/_optional_deps``). Until sam-extra v0.33.0 is installed, point the oracle at its work tree:
``AISTUDIO_FORGE_EXTENSION_DIR=<tree>`` and ``SAM3_FORGE_ROOT=<Forge>``.
"""
from __future__ import annotations

import ast
import itertools
import types
import unittest
from pathlib import Path
from typing import Any, Callable

from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_math as xs
from tests._forge_k_diffusion import (
    ComfyPatcher,
    Predictor,
    ReplayNoise,
    ToyModel,
    ext_literals,
    forge_k_sampling,
    import_ext,
    installed_k_sampling,
    load_ext_file,
)
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # filled by bind_torch() in the @requires_torch classes

PACK_DIR = Path(__file__).resolve().parents[1] / "comfy_custom_nodes" / "ai_studio_forge_parity"
_PARAMS = "sam3ext/extra_samplers/params.py"


def _outcome(function: Callable[[], Any]) -> tuple[str, Any]:
    """('ok', value) or ('error', class name) — what a caller of ``function`` sees."""
    try:
        return "ok", _key(function())
    except Exception as exc:
        return "error", type(exc).__name__


def _key(value: Any) -> Any:
    """Floats by ``repr`` (every bit, -0.0 ≠ 0.0); the type is part of the key."""
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, (list, tuple)):
        return type(value).__name__, tuple(_key(item) for item in value)
    return type(value).__name__, value


_VALUES = (
    None, True, False, 0, 1, 2, 3, 4, -1, 0.0, -0.0, 0.5, 1.0, 1.5, 2.5, 3.5, 2.6, 0.37, 0.004, 0.005, 0.995, 9.99,
    10, 10.0, 10.5, -0.2, 1e9, -1e9, float("nan"), float("inf"), float("-inf"), "nan", "inf", "", " ", "x", "3",
    "2.5", " 0.4 ", "-1", "1e9", "true", "TRUE", " on ", "Yes", "1", "0", "1.0", "off", "false", [], {}, (1,),
    10 ** 400, 1e-320,
)


def _settings_key(settings) -> tuple:
    return tuple(_key(getattr(settings, name)) for name in xs.ARG_NAMES)


@requires_extension
class SettingsCoercionTests(unittest.TestCase):
    """``extra_samplers_math`` coercion == sam-extra ``params`` (loaded alone — it needs no torch)."""

    @classmethod
    def setUpClass(cls):
        cls.params = load_ext_file(_PARAMS, "_aistudio_oracle_xsamp_params")

    def test_constants_equal_the_extension(self):
        for name in ("ARG_NAMES", "DEFAULT_MAX_STAGE", "MAX_STAGE_MIN", "MAX_STAGE_MAX", "DEFAULT_ETA", "ETA_MIN",
                     "ETA_MAX", "ETA_STEP", "DEFAULT_NOISE_WINDOW", "DEFAULT_NOISE_START", "DEFAULT_NOISE_END",
                     "NOISE_PERCENT_MIN", "NOISE_PERCENT_MAX", "NOISE_PERCENT_STEP"):
            with self.subTest(name=name):
                self.assertEqual(_key(getattr(xs, name)), _key(getattr(self.params, name)))
        self.assertEqual(xs.Settings._fields, self.params.ARG_NAMES)
        self.assertEqual(xs.DEFAULTS.as_dict(), self.params.ErSdeSettings().as_dict())

    def test_solver_type_names_equal_the_extension(self):
        tree = ast.parse((EXT_ROOT / "sam3ext/extra_samplers/er_sde.py").read_text(encoding="utf-8"))
        literals = {
            node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual((xs.ER_SDE, xs.REVERSE_TIME, xs.ODE),
                         (literals["ER_SDE"], literals["REVERSE_TIME"], literals["ODE"]))

    def test_each_value_is_coerced_alike(self):
        p = self.params
        pairs = (
            ("max_stage", xs.coerce_max_stage, p.coerce_max_stage),
            ("eta", xs.coerce_eta, p.coerce_eta),
            ("noise_window", xs.coerce_noise_window, p.coerce_noise_window),
            ("noise_start", lambda v: xs.coerce_noise_percent(v, 0.2), lambda v: p.coerce_noise_percent(v, 0.2)),
            ("noise_end", lambda v: xs.coerce_noise_percent(v, 0.8), lambda v: p.coerce_noise_percent(v, 0.8)),
        )
        for (name, ours, theirs), value in itertools.product(pairs, _VALUES):
            with self.subTest(name=name, value=repr(value)[:40]):
                self.assertEqual(_outcome(lambda: ours(value)), _outcome(lambda: theirs(value)))

    def test_positional_arguments_read_alike(self):
        samples = (3, 1.0, False, 0.2, 0.8, "extra", None)
        variants = (
            (1, 0.0, True, 0.1, 0.9), ("2", "2.5", "on", "0.35", "0.6"), (9, -3, 1, -1, 2), (2.5, float("nan"), 2,
            True, "x"), (None, None, None, None, None), (1, 0.37, "yes", 0.8, 0.2), (3, 10.5, "1", 1e-320, 1.0),
        )
        cases = [list(samples[:length]) for length in range(len(samples) + 1)]
        cases += [list(row) for row in variants] + [list(row[:3]) for row in variants]
        cases += [(), None, "", (1, 2)]
        for args in cases:
            with self.subTest(args=repr(args)):
                theirs = self.params.settings_from_args(args)
                self.assertEqual(_settings_key(xs.coerce_args(args)), _settings_key(theirs))

    def test_model_options_dict_reads_like_the_extensions_dict_form(self):
        # sam-extra's ``resolve`` accepts the same five keys as a dict (an API caller's form).
        dicts = (
            {}, {"max_stage": 1}, {"eta": "0.5", "noise_window": "on"}, {"noise_start": 0.9, "noise_end": 0.1},
            {"max_stage": "x", "eta": float("inf"), "noise_window": 2, "noise_start": True, "noise_end": None},
            {"max_stage": 2.5, "eta": 3, "noise_window": 1.0, "noise_start": "0.25", "noise_end": 0.75, "other": 1},
        )
        for values in dicts:
            with self.subTest(values=values):
                theirs = self.params.resolve(types.SimpleNamespace(sam_extra_samplers=dict(values)))
                ours = xs.settings_from_model_options({xs.MODEL_OPTIONS_KEY: dict(values)})
                self.assertEqual(_settings_key(ours), _settings_key(theirs))
                self.assertEqual(_settings_key(xs.settings_from_extra_args(
                    {"model_options": {xs.MODEL_OPTIONS_KEY: dict(values)}})), _settings_key(theirs))

    def test_absent_values_are_the_defaults(self):
        for model_options in (None, {}, {"transformer_options": {}}, {xs.MODEL_OPTIONS_KEY: None}, "x"):
            with self.subTest(model_options=model_options):
                self.assertEqual(xs.settings_from_model_options(model_options), xs.DEFAULTS)
        self.assertEqual(xs.settings_from_extra_args(None), xs.DEFAULTS)
        self.assertEqual(xs.settings_from_extra_args({}), xs.DEFAULTS)


# ── ER SDE on Forge's k-diffusion ─────────────────────────────────────────────────────────────


def _sigmas(flow: bool, steps: int, dtype):
    if flow:
        t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
        sigmas = 3.0 * t / (1 + 2.0 * t)
    else:
        ramp = torch.linspace(0, 1, steps, dtype=torch.float64)
        max_inv, min_inv = 14.6146 ** (1 / 7), 0.0292 ** (1 / 7)
        sigmas = torch.cat([(max_inv + ramp * (min_inv - max_inv)) ** 7, torch.zeros(1, dtype=torch.float64)])
    return sigmas.to(dtype)


def _latent(shape, dtype, seed=5):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)


def _same_bits(test: unittest.TestCase, a, b, message: str = "") -> None:
    test.assertEqual(a.dtype, b.dtype, message)
    test.assertEqual(a.shape, b.shape, message)
    integer = {torch.float32: torch.int32, torch.float64: torch.int64, torch.float16: torch.int16,
               torch.bfloat16: torch.int16}[a.dtype]
    test.assertTrue(torch.equal(a.contiguous().view(integer), b.contiguous().view(integer)),
                    message or "latents differ")


_SHAPES = ((1, 4, 5, 7), (2, 3, 2, 5, 6))
_WINDOWS = (None, (0.0, 1.0), (0.5, 0.5), (0.8, 0.2), (0.2, 0.8), (0.35, 0.6), (0.0, 0.3), (0.7, 1.0))


@requires_torch
@requires_extension
class ErSdeParityTests(unittest.TestCase):
    """The pack's three ER SDE samplers == sam-extra's on Forge's k-diffusion module, bit for bit."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.er = import_ext("sam3ext.extra_samplers.er_sde")
            cls.window = import_ext("sam3ext.extra_samplers.er_sde_window")
            cls.registry = import_ext("sam3ext.extra_samplers.registry")
            cls.params = import_ext("sam3ext.extra_samplers.params")
        cls.ours = {
            "er_sde_reverse_time": xs.sample_er_sde_reverse_time,
            "er_sde_ode": xs.sample_er_sde_ode,
            "er_sde_tunable": xs.sample_er_sde_tunable,
        }
        cls.theirs = {
            "er_sde_reverse_time": cls.er.sample_er_sde_reverse_time,
            "er_sde_ode": cls.er.sample_er_sde_ode,
            "er_sde_tunable": cls.er.sample_er_sde_tunable,
        }

    def _run(self, function, *, flow, dtype, shape, steps=8, kwargs=None, extra_args=None, noise=True,
             predictor=None, ours=True, seed=1234, callback=None):
        predictor = predictor or Predictor(flow)
        model = ToyModel(predictor)
        x = _latent(shape, dtype) * float(_sigmas(flow, steps, torch.float64)[0])
        sigmas = _sigmas(flow, steps, dtype)
        sampler = ReplayNoise(x, seed) if noise else None
        call = dict(kwargs or {})
        if ours:
            call["kd"] = self.kd
        with installed_k_sampling(self.kd):
            out = function(model, x, sigmas, extra_args=dict(extra_args or {}), callback=callback, disable=True,
                           noise_sampler=sampler, **call)
        return out, (sampler.draws if sampler is not None else None), [c[0] for c in model.calls]

    def _compare(self, name, *, flow, dtype, shape, kwargs, label):
        ours = self._run(self.ours[name], flow=flow, dtype=dtype, shape=shape, kwargs=kwargs)
        theirs = self._run(self.theirs[name], flow=flow, dtype=dtype, shape=shape, kwargs=kwargs, ours=False)
        _same_bits(self, ours[0], theirs[0], label)
        self.assertEqual(ours[1], theirs[1], f"{label}: noise draws")
        self.assertEqual(ours[2], theirs[2], f"{label}: model calls")
        return ours

    def test_sde_kinds_over_the_grid(self):
        cases = 0
        for name, flow, dtype, stage, eta, s_noise, window in itertools.product(
                ("er_sde_reverse_time", "er_sde_tunable"), (True, False), (torch.float32, torch.float64),
                (1, 2, 3), (0.0, 0.37, 1.0, 2.5, 10.0), (0.0, 1.0), (None, (0.2, 0.8))):
            shape = _SHAPES[cases % 2]
            cases += 1
            kwargs = {"max_stage": stage, "er_sde_eta": eta, "s_noise": s_noise, "er_sde_window": window}
            label = f"{name} flow={flow} {dtype} stage={stage} eta={eta} s_noise={s_noise} window={window}"
            with self.subTest(label):
                self._compare(name, flow=flow, dtype=dtype, shape=shape, kwargs=kwargs, label=label)
        self.assertEqual(cases, 480)

    def test_window_shapes(self):
        cases = 0
        for name, flow, dtype, eta, window in itertools.product(
                ("er_sde_reverse_time", "er_sde_tunable"), (True, False), (torch.float32, torch.float64),
                (0.37, 2.5), _WINDOWS):
            shape = _SHAPES[cases % 2]
            cases += 1
            kwargs = {"max_stage": 3, "er_sde_eta": eta, "s_noise": 1.0, "er_sde_window": window}
            label = f"{name} flow={flow} {dtype} eta={eta} window={window}"
            with self.subTest(label):
                ours = self._compare(name, flow=flow, dtype=dtype, shape=shape, kwargs=kwargs, label=label)
                if window is not None and window[0] >= window[1]:
                    self.assertEqual(ours[1], [], f"{label}: an empty window draws nothing")
        self.assertEqual(cases, 128)

    def test_ode_kind(self):
        for flow, dtype, stage, shape in itertools.product((True, False), (torch.float32, torch.float64),
                                                           (1, 2, 3), _SHAPES):
            label = f"er_sde_ode flow={flow} {dtype} stage={stage} {shape}"
            with self.subTest(label):
                ours = self._compare("er_sde_ode", flow=flow, dtype=dtype, shape=shape,
                                     kwargs={"max_stage": stage}, label=label)
                self.assertEqual(ours[1], [], "the ODE never draws noise")

    def test_full_window_equals_no_window_and_empty_window_equals_the_ode(self):
        for name, flow in itertools.product(("er_sde_reverse_time", "er_sde_tunable"), (True, False)):
            with self.subTest(name=name, flow=flow):
                common = {"max_stage": 3, "er_sde_eta": 0.6, "s_noise": 1.0}
                plain = self._run(self.ours[name], flow=flow, dtype=torch.float64, shape=_SHAPES[0],
                                  kwargs={**common, "er_sde_window": None})
                full = self._run(self.ours[name], flow=flow, dtype=torch.float64, shape=_SHAPES[0],
                                 kwargs={**common, "er_sde_window": (0.0, 1.0)})
                _same_bits(self, plain[0], full[0])
                empty = self._run(self.ours[name], flow=flow, dtype=torch.float64, shape=_SHAPES[0],
                                  kwargs={**common, "er_sde_window": (0.6, 0.6)})
                ode = self._run(xs.sample_er_sde_ode, flow=flow, dtype=torch.float64, shape=_SHAPES[0],
                                kwargs={"max_stage": 3})
                _same_bits(self, empty[0], ode[0])
                self.assertEqual(empty[1], [])

    def test_draws_happen_only_on_in_window_steps(self):
        predictor = Predictor(True)
        start, end = 0.2, 0.8
        hi, lo = predictor.percent_to_sigma(start), predictor.percent_to_sigma(end)
        ours = self._run(xs.sample_er_sde_tunable, flow=True, dtype=torch.float64, shape=_SHAPES[0], steps=20,
                         kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": (start, end)},
                         predictor=predictor)
        sigmas = _sigmas(True, 20, torch.float64)
        expected = [(float(sigmas[i]), float(sigmas[i + 1])) for i in range(20)
                    if lo <= float(sigmas[i + 1]) <= hi and float(sigmas[i + 1]) != 0.0]
        self.assertTrue(expected)
        self.assertEqual(ours[1], expected)

    def test_tunable_at_eta_one_is_the_hosts_default_er_sde(self):
        for flow, dtype in itertools.product((True, False), (torch.float32, torch.float64)):
            with self.subTest(flow=flow, dtype=dtype):
                ours = self._run(xs.sample_er_sde_tunable, flow=flow, dtype=dtype, shape=_SHAPES[1],
                                 kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": None})
                host = self._run(self.kd.sample_er_sde, flow=flow, dtype=dtype, shape=_SHAPES[1],
                                 kwargs={"max_stage": 3}, ours=False)
                _same_bits(self, ours[0], host[0])
                self.assertEqual(ours[1], host[1])

    def test_window_helpers_equal_the_extension(self):
        for flow in (True, False):
            predictor = Predictor(flow)
            for start, end in ((0.0, 1.0), (0.2, 0.8), (0.5, 0.5), (0.8, 0.2), (1e-4, 0.9999), (0.33, 0.34)):
                with self.subTest(flow=flow, start=start, end=end):
                    bounds = xs.window_bounds(predictor, start, end)
                    self.assertEqual(bounds, self.window.window_bounds(predictor, start, end))
                    for steps in (1, 2, 7, 28):
                        sigmas = _sigmas(flow, steps, torch.float32)
                        self.assertEqual(xs.in_window_steps(sigmas, start, end, *bounds),
                                         self.window.in_window_steps(sigmas, start, end, *bounds))
        with self.assertRaises(ValueError):
            xs.window_bounds(Predictor(True, percent_to_sigma=False), 0.2, 0.8)

    def test_scalers_equal_the_extension(self):
        grid = torch.tensor([0.0, 1e-6, 0.01, 0.3, 1.0, 2.0, 14.6, 150.0, 1e4], dtype=torch.float64)
        for eta in (0.0, 0.37, 1.0, 2.5, 10.0):
            with self.subTest(eta=eta):
                _same_bits(self, xs.tunable_scaler(eta)(grid), self.er.er_sde_noise_scaler(eta)(grid))
                _same_bits(self, xs.reverse_time_scaler(eta)(grid), self.er.reverse_time_sde_noise_scaler(eta)(grid))
        _same_bits(self, xs.ode_scaler(grid), self.er.ode_noise_scaler(grid))
        for solver_type, stage, eta, s_noise in itertools.product(xs.SOLVER_TYPES, (1, 3), (0.0, 0.5, 1.0),
                                                                  (0.0, 1.0)):
            with self.subTest(solver_type=solver_type, stage=stage, eta=eta, s_noise=s_noise):
                ours = xs.er_sde_kwargs(solver_type, stage, eta, s_noise)
                theirs = self.er.er_sde_kwargs(solver_type, stage, eta, s_noise)
                self.assertEqual(list(ours), list(theirs))
                self.assertEqual((ours["s_noise"], ours["max_stage"]), (theirs["s_noise"], theirs["max_stage"]))
                self.assertEqual(ours["noise_scaler"] is xs.ode_scaler, theirs["noise_scaler"] is self.er.ode_noise_scaler)
                _same_bits(self, ours["noise_scaler"](grid), theirs["noise_scaler"](grid))
        with self.assertRaises(ValueError):
            xs.er_sde_kwargs("Other", 3, 1.0, 1.0)

    def _forge_kwargs(self, name, settings, predictor):
        """What sam-extra's Forge sampler adds for ``settings`` (``registry.runtime_kwargs``)."""
        spec = next(spec for spec in self.registry.SPECS if spec.aliases[0] == name)
        p = types.SimpleNamespace(sam_extra_samplers=self.params.ErSdeSettings(**settings.as_dict()),
                                  extra_generation_params={})
        return self.registry.runtime_kwargs(spec, p, forge_eta=1.0, predictor=predictor)

    def test_model_options_drive_the_pack_like_forges_runtime_kwargs(self):
        grid = (
            xs.DEFAULTS, xs.Settings(1, 0.37, False, 0.2, 0.8), xs.Settings(2, 2.5, True, 0.2, 0.8),
            xs.Settings(3, 0.0, True, 0.2, 0.8), xs.Settings(3, 1.0, True, 0.0, 1.0),
            xs.Settings(3, 1.0, True, 0.7, 0.3), xs.Settings(2, 4.0, True, 0.35, 0.6),
        )
        for name, flow, settings in itertools.product(sorted(self.ours), (True, False), grid):
            label = f"{name} flow={flow} {settings}"
            with self.subTest(label):
                predictor = Predictor(flow)
                kwargs = self._forge_kwargs(name, settings, predictor)
                theirs = self._run(self.theirs[name], flow=flow, dtype=torch.float32, shape=_SHAPES[0],
                                   kwargs=kwargs, predictor=predictor, ours=False)
                ours = self._run(self.ours[name], flow=flow, dtype=torch.float32, shape=_SHAPES[0],
                                 extra_args={"model_options": {xs.MODEL_OPTIONS_KEY: settings.as_dict()}},
                                 predictor=predictor)
                _same_bits(self, ours[0], theirs[0], label)
                self.assertEqual(ours[1], theirs[1], label)

    def test_without_values_the_pack_runs_the_defaults(self):
        self.params.reset_active()
        for name, flow in itertools.product(sorted(self.ours), (True, False)):
            with self.subTest(name=name, flow=flow):
                predictor = Predictor(flow)
                kwargs = self._forge_kwargs(name, xs.DEFAULTS, predictor)
                theirs = self._run(self.theirs[name], flow=flow, dtype=torch.float64, shape=_SHAPES[1],
                                   kwargs=kwargs, predictor=predictor, ours=False)
                ours = self._run(self.ours[name], flow=flow, dtype=torch.float64, shape=_SHAPES[1],
                                 predictor=predictor)
                _same_bits(self, ours[0], theirs[0])

    def test_a_window_the_model_cannot_place_is_dropped_like_forge(self):
        # eps sigmas: a flow schedule starting at σ 1 would need percent_to_sigma for the first-σ offset.
        settings = xs.Settings(3, 0.8, True, 0.2, 0.8)
        for name in ("er_sde_reverse_time", "er_sde_tunable"):
            with self.subTest(name=name):
                predictor = Predictor(False, percent_to_sigma=False)
                kwargs = self._forge_kwargs(name, settings, predictor)
                self.assertNotIn("er_sde_window", kwargs)
                theirs = self._run(self.theirs[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                                   kwargs=kwargs, predictor=predictor, ours=False)
                xs._window_logged.discard(name)
                with self.assertLogs("ai_studio_forge_parity", "WARNING") as logs:
                    ours = self._run(self.ours[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                                     extra_args={"model_options": {xs.MODEL_OPTIONS_KEY: settings.as_dict()}},
                                     predictor=predictor)
                self.assertIn("noise window is skipped", logs.output[0])
                _same_bits(self, ours[0], theirs[0])
                with self.assertRaises(ValueError):
                    self._run(self.ours[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                              kwargs={"er_sde_window": (0.2, 0.8)}, predictor=predictor)

    def test_at_eta_zero_the_window_is_not_even_placed(self):
        # η 0 is the ODE: no window to place, so a model without percent_to_sigma neither fails nor warns.
        predictor = Predictor(False, percent_to_sigma=False)
        for name in ("er_sde_reverse_time", "er_sde_tunable"):
            with self.subTest(name=name):
                settings = xs.Settings(3, 0.0, True, 0.2, 0.8)
                p_kwargs = self._forge_kwargs(name, settings, predictor)
                self.assertNotIn("er_sde_window", p_kwargs)
                theirs = self._run(self.theirs[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                                   kwargs={"max_stage": 3, "er_sde_eta": 0.0, "er_sde_window": (0.2, 0.8)},
                                   predictor=predictor, ours=False)
                xs._window_logged.discard(name)
                with self.assertNoLogs("ai_studio_forge_parity", "WARNING"):
                    from_values = self._run(
                        self.ours[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                        extra_args={"model_options": {xs.MODEL_OPTIONS_KEY: settings.as_dict()}}, predictor=predictor)
                explicit = self._run(self.ours[name], flow=False, dtype=torch.float32, shape=_SHAPES[0], steps=6,
                                     kwargs={"max_stage": 3, "er_sde_eta": 0.0, "er_sde_window": (0.2, 0.8)},
                                     predictor=predictor)
                _same_bits(self, from_values[0], theirs[0])
                _same_bits(self, explicit[0], theirs[0])
                self.assertEqual(explicit[1], [])

    def test_explicit_arguments_override_the_model_options(self):
        settings = {"max_stage": 1, "eta": 5.0, "noise_window": True, "noise_start": 0.4, "noise_end": 0.6}
        explicit = {"max_stage": 2, "er_sde_eta": 0.5, "er_sde_window": None}
        ours = self._run(xs.sample_er_sde_reverse_time, flow=True, dtype=torch.float32, shape=_SHAPES[0],
                         kwargs=explicit, extra_args={"model_options": {xs.MODEL_OPTIONS_KEY: settings}})
        theirs = self._run(self.er.sample_er_sde_reverse_time, flow=True, dtype=torch.float32, shape=_SHAPES[0],
                           kwargs=explicit, ours=False)
        _same_bits(self, ours[0], theirs[0])

    def test_window_default_noise_is_the_hosts_own_stream(self):
        # noise_sampler None: both build the host's default sampler (Forge: randn_like on the global stream).
        for name, window in itertools.product(("er_sde_reverse_time", "er_sde_tunable"), (None, (0.2, 0.8))):
            with self.subTest(name=name, window=window):
                kwargs = {"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": window}
                torch.manual_seed(77)
                ours = self._run(self.ours[name], flow=True, dtype=torch.float32, shape=_SHAPES[0], kwargs=kwargs,
                                 noise=False)
                torch.manual_seed(77)
                theirs = self._run(self.theirs[name], flow=True, dtype=torch.float32, shape=_SHAPES[0],
                                   kwargs=kwargs, noise=False, ours=False)
                _same_bits(self, ours[0], theirs[0])

    def test_the_callback_sees_every_step_unchanged(self):
        for window in (None, (0.2, 0.8)):
            seen: dict = {"ours": [], "theirs": []}
            for side, function, ours in (("ours", xs.sample_er_sde_tunable, True),
                                         ("theirs", self.er.sample_er_sde_tunable, False)):
                self._run(function, flow=True, dtype=torch.float32, shape=_SHAPES[0], ours=ours,
                          kwargs={"max_stage": 3, "er_sde_eta": 1.0, "er_sde_window": window},
                          callback=lambda info, side=side: seen[side].append(
                              (info["i"], float(info["sigma"]), float(info["denoised"].sum()))))
            with self.subTest(window=window):
                self.assertEqual(seen["ours"], seen["theirs"])
                self.assertEqual([item[0] for item in seen["ours"]], list(range(8)))


@requires_torch
class ErSdeHostTests(unittest.TestCase):
    """ComfyUI-side plumbing with a recording stand-in for ``comfy.k_diffusion.sampling``."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())

    def _recording_kd(self, *, call_scaler_first=False):
        record: dict = {"seed": "unset", "kwargs": None}

        def default_noise_sampler(x, seed=None):
            record["seed"] = seed
            return lambda sigma, sigma_next: torch.ones_like(x)

        def sample_er_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0,
                          noise_sampler=None, noise_scaler=None, max_stage=3):
            record["kwargs"] = {"s_noise": s_noise, "max_stage": max_stage, "noise_scaler": noise_scaler,
                                "noise_sampler": noise_sampler, "callback": callback}
            if call_scaler_first:
                noise_scaler(torch.ones(1))
            for i in range(len(sigmas) - 1):
                if callback is not None:
                    callback({"i": i})
                noise_scaler(torch.ones(1))
                if noise_sampler is not None and s_noise > 0:
                    noise_sampler(sigmas[i], sigmas[i + 1])
            return x

        return types.SimpleNamespace(default_noise_sampler=default_noise_sampler, sample_er_sde=sample_er_sde), record

    def test_comfy_layout_places_the_window_with_the_patchers_model_sampling(self):
        kd, record = self._recording_kd()
        sampling = Predictor(True)
        model = ToyModel(sampling, comfy_layout=True)
        sigmas = torch.tensor([1.0, 0.9, 0.6, 0.3, 0.0])
        x = torch.zeros(1, 2, 3, 3)
        xs.sample_er_sde_tunable(model, x, sigmas, extra_args={"seed": 41}, kd=kd,
                                 er_sde_window=(0.2, 0.8), er_sde_eta=1.0)
        self.assertEqual(record["seed"], 41)   # ComfyUI's seeded default noise, as its own sample_er_sde builds it
        self.assertIsNot(record["kwargs"]["noise_scaler"], xs.ode_scaler)
        self.assertEqual(xs.model_sampling(model), sampling)

    def test_no_window_leaves_the_noise_sampler_to_the_host(self):
        kd, record = self._recording_kd()
        model = ToyModel(Predictor(True), comfy_layout=True)
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        xs.sample_er_sde_reverse_time(model, torch.zeros(1, 1, 2, 2), sigmas, extra_args={"seed": 3}, kd=kd)
        self.assertEqual(record["seed"], "unset")
        self.assertIsNone(record["kwargs"]["noise_sampler"])
        self.assertEqual(record["kwargs"]["max_stage"], 3)

    def test_a_scaler_before_the_steps_callback_fails_loudly(self):
        kd, _record = self._recording_kd(call_scaler_first=True)
        model = ToyModel(Predictor(True), comfy_layout=True)
        with self.assertRaisesRegex(RuntimeError, "before the step's callback"):
            xs.sample_er_sde_tunable(model, torch.zeros(1, 1, 2, 2), torch.tensor([0.9, 0.5, 0.0]), kd=kd,
                                     er_sde_window=(0.2, 0.8))

    def test_the_ode_ignores_a_window_in_the_values(self):
        kd, record = self._recording_kd()
        model = ToyModel(Predictor(True), comfy_layout=True)
        options = {"model_options": {xs.MODEL_OPTIONS_KEY: {"noise_window": True, "max_stage": 2}}}
        xs.sample_er_sde_ode(model, torch.zeros(1, 1, 2, 2), torch.tensor([0.9, 0.5, 0.0]), extra_args=options, kd=kd)
        self.assertIs(record["kwargs"]["noise_scaler"], xs.ode_scaler)
        self.assertEqual((record["kwargs"]["s_noise"], record["kwargs"]["max_stage"]), (0.0, 2))
        self.assertIsNone(record["kwargs"]["noise_sampler"])

    def test_sampler_functions_take_a_noise_sampler_and_no_eta(self):
        import inspect

        for function in (xs.sample_er_sde_reverse_time, xs.sample_er_sde_tunable, xs.sample_er_sde_ode):
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                self.assertIn("noise_sampler", parameters)
                self.assertNotIn("eta", parameters)   # no host may pass an ancestral η into it
                self.assertEqual(list(parameters)[:6], ["model", "x", "sigmas", "extra_args", "callback", "disable"])
        self.assertEqual(xs.sample_er_sde_tunable.__name__, "sample_er_sde_tunable")


class _RecordingTree:
    """Stand-in for the host's ``BrownianTreeNoiseSampler``: records how it was built, draws from a seeded stream."""

    built: list = []

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=None, cpu=False):
        type(self).built.append((tuple(x.shape), float(sigma_min), float(sigma_max), seed, cpu))
        self.noise = ReplayNoise(x, 99)

    def __call__(self, sigma, sigma_next):
        return self.noise(sigma, sigma_next)


def _module_with(module, **overrides):
    copy = types.ModuleType(module.__name__ + "_copy")
    copy.__dict__.update(module.__dict__)
    copy.__dict__.update(overrides)
    return copy


@requires_torch
@requires_extension
class Dpmpp4mParityTests(unittest.TestCase):
    """``dpmpp_4m_sde`` == sam-extra's DPM++ 4M SDE on Forge's k-diffusion module, bit for bit."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dpmpp_4m as m4

        cls.m4 = m4
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.ext = import_ext("sam3ext.extra_samplers.dpmpp_4m_sde")

    def _run(self, function, kd, *, flow, dtype, shape, steps, eta, s_noise, ours, noise=True, extra_args=None,
             predictor=None, sigmas=None):
        predictor = predictor or Predictor(flow)
        model = ToyModel(predictor)
        sigmas = _sigmas(flow, steps, dtype) if sigmas is None else sigmas
        x = _latent(shape, dtype) * float(sigmas[0])
        sampler = ReplayNoise(x, 4321) if noise else None
        call = {"kd": kd} if ours else {}
        with installed_k_sampling(kd):
            out = function(model, x, sigmas, extra_args=dict(extra_args or {}), disable=True, eta=eta,
                           s_noise=s_noise, noise_sampler=sampler, **call)
        return out, (sampler.draws if sampler else None), [c[0] for c in model.calls]

    def test_grid(self):
        cases = 0
        for flow, dtype, steps, eta, s_noise in itertools.product(
                (True, False), (torch.float32, torch.float64), (6, 8, 13), (0.0, 0.5, 1.0, 1.7), (0.0, 1.0)):
            shape = _SHAPES[cases % 2]
            cases += 1
            label = f"flow={flow} {dtype} steps={steps} eta={eta} s_noise={s_noise}"
            with self.subTest(label):
                common = dict(flow=flow, dtype=dtype, shape=shape, steps=steps, eta=eta, s_noise=s_noise)
                ours = self._run(self.m4.sample_dpmpp_4m_sde, self.kd, ours=True, **common)
                theirs = self._run(self.ext.sample_dpmpp_4m_sde, self.kd, ours=False, **common)
                _same_bits(self, ours[0], theirs[0], label)
                self.assertEqual(ours[1], theirs[1], label)
                self.assertEqual(ours[2], theirs[2], label)
                if eta == 0.0 or s_noise == 0.0:
                    self.assertEqual(ours[1], [], f"{label}: no noise drawn")
        self.assertEqual(cases, 96)

    def test_first_three_orders_are_the_hosts_3m_sde(self):
        for flow, dtype, eta in itertools.product((True, False), (torch.float32, torch.float64), (0.0, 1.0)):
            with self.subTest(flow=flow, dtype=dtype, eta=eta):
                full = _sigmas(flow, 10, dtype)
                for sigmas in (full[:4], torch.cat([full[:4], full.new_zeros(1)])):
                    common = dict(flow=flow, dtype=dtype, shape=_SHAPES[0], steps=0, eta=eta, s_noise=1.0,
                                  sigmas=sigmas)
                    ours = self._run(self.m4.sample_dpmpp_4m_sde, self.kd, ours=True, **common)
                    host = self._run(self.kd.sample_dpmpp_3m_sde, self.kd, ours=False, **common)
                    _same_bits(self, ours[0], host[0])
                # the fourth history term changes the result once it exists (the comparison is not vacuous)
                common = dict(flow=flow, dtype=dtype, shape=_SHAPES[0], steps=0, eta=eta, s_noise=1.0,
                              sigmas=full[:6])
                ours = self._run(self.m4.sample_dpmpp_4m_sde, self.kd, ours=True, **common)
                host = self._run(self.kd.sample_dpmpp_3m_sde, self.kd, ours=False, **common)
                self.assertFalse(torch.equal(ours[0], host[0]))

    def test_default_noise_is_a_cpu_brownian_tree_seeded_from_extra_args(self):
        for eta, s_noise in ((1.0, 1.0), (0.0, 1.0)):
            with self.subTest(eta=eta, s_noise=s_noise):
                kd = _module_with(self.kd, BrownianTreeNoiseSampler=_RecordingTree)
                _RecordingTree.built = []
                common = dict(flow=True, dtype=torch.float32, shape=_SHAPES[0], steps=7, eta=eta, s_noise=s_noise,
                              noise=False, extra_args={"seed": 1234})
                ours = self._run(self.m4.sample_dpmpp_4m_sde, kd, ours=True, **common)
                built_ours, _RecordingTree.built = _RecordingTree.built, []
                theirs = self._run(self.ext.sample_dpmpp_4m_sde, kd, ours=False, **common)
                self.assertEqual(built_ours, _RecordingTree.built)
                self.assertEqual(built_ours[0][3:], (1234, True))
                _same_bits(self, ours[0], theirs[0])

    def test_noise_scale_multiplies_s_noise(self):
        scaled = Predictor(True)
        scaled.noise_scale = 2.0
        one = self._run(self.m4.sample_dpmpp_4m_sde, self.kd, ours=True, flow=True, dtype=torch.float64,
                        shape=_SHAPES[0], steps=8, eta=1.0, s_noise=1.0, predictor=scaled)
        two = self._run(self.m4.sample_dpmpp_4m_sde, self.kd, ours=True, flow=True, dtype=torch.float64,
                        shape=_SHAPES[0], steps=8, eta=1.0, s_noise=2.0)
        _same_bits(self, one[0], two[0])

    def test_short_schedules(self):
        for length in (0, 1):
            with self.subTest(length=length):
                x = _latent(_SHAPES[0], torch.float32)
                sigmas = torch.ones(length)
                self.assertIs(self.m4.sample_dpmpp_4m_sde(ToyModel(Predictor(True)), x, sigmas, kd=self.kd), x)


class CfgToyModel(ToyModel):
    """A CFG denoiser stand-in: cond/uncond predictions, CFG, then the ``sampler_post_cfg_function`` list
    (so CFG++ gets a real unconditional x0). ``calls`` also records the sub-step marker."""

    def __init__(self, sampling, cfg: float = 1.7, **kwargs):
        super().__init__(sampling, **kwargs)
        self.cfg = cfg
        self.markers = []

    def __call__(self, x, sigma, **extra_args):
        options = extra_args.get("model_options") or {}
        self.markers.append(((options.get("transformer_options") or {}).get("sam_extra_substep"), tuple(x.shape)))
        s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        cond = torch.tanh(x) * 0.8 / (1.0 + s) + self.bias * s - 0.03 * x * x
        uncond = torch.tanh(0.7 * x) * 0.6 / (1.0 + s) - 0.02 * s
        denoised = uncond + (cond - uncond) * self.cfg
        for function in options.get("sampler_post_cfg_function", []):
            denoised = function({"denoised": denoised, "cond_denoised": cond, "uncond_denoised": uncond,
                                 "model": self, "sigma": sigma, "input": x, "model_options": options})
        self.calls.append((float(sigma.reshape(-1)[0]), tuple(x.shape), None))
        return denoised


_DY_SHAPES = ((1, 4, 8, 10), (1, 4, 7, 9), (2, 3, 2, 7, 9), (1, 2, 6, 5), (1, 4, 1, 1))
_DY_ENTRIES = ("euler_dy", "euler_smea_dy", "euler_dy_cfg_pp", "euler_smea_dy_cfg_pp")


@requires_torch
@requires_extension
class EulerDyParityTests(unittest.TestCase):
    """Euler (SMEA) Dy and their CFG++ entries == sam-extra's ``euler_dy`` on Forge's k-diffusion module."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy as dy

        cls.dy = dy
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.ext = import_ext("sam3ext.extra_samplers.euler_dy")

    def _run(self, function, *, ours, flow, dtype, shape, steps=8, kwargs=None, cfg=True, sigmas=None,
             noise=True, record_callback=True):
        model = CfgToyModel(Predictor(flow)) if cfg else ToyModel(Predictor(flow))
        sigmas = _sigmas(flow, steps, dtype) if sigmas is None else sigmas
        x = _latent(shape, dtype) * float(sigmas[0])
        seen = []
        call = dict(kwargs or {})
        if noise:
            call["noise_sampler"] = ReplayNoise(x, 2468)
        if ours:
            call["kd"] = self.kd
        with installed_k_sampling(self.kd):
            out = function(model, x, sigmas, extra_args={}, disable=True,
                           callback=(lambda info: seen.append((info["i"], info["x"].clone()))) if record_callback
                           else None, **call)
        return out, getattr(model, "markers", None), [c[:2] for c in model.calls], seen

    def _compare(self, ours_fn, theirs_fn, label, **common):
        ours = self._run(ours_fn, ours=True, **common)
        theirs = self._run(theirs_fn, ours=False, **common)
        _same_bits(self, ours[0], theirs[0], label)
        self.assertEqual(ours[1], theirs[1], f"{label}: evaluations (marker, shape)")
        self.assertEqual(ours[2], theirs[2], f"{label}: model calls")
        self.assertEqual([i for i, _ in ours[3]], [i for i, _ in theirs[3]], label)
        for (_, a), (_, b) in zip(ours[3], theirs[3]):
            _same_bits(self, a, b, f"{label}: callback x")
        return ours

    def test_registered_entries_over_the_grid(self):
        cases = 0
        for name, flow, dtype, s_churn, offset, substeps in itertools.product(
                _DY_ENTRIES, (True, False), (torch.float32, torch.float64), (0.0, 1.5), (0, 2), (True, False)):
            shape = _DY_SHAPES[cases % len(_DY_SHAPES)]
            cases += 1
            kwargs = {"s_churn": s_churn, "sam_extra_step_offset": offset, "substeps": substeps}
            label = f"{name} flow={flow} {dtype} churn={s_churn} offset={offset} substeps={substeps} {shape}"
            with self.subTest(label):
                self._compare(getattr(self.dy, "sample_" + name), getattr(self.ext, "sample_" + name), label,
                              flow=flow, dtype=dtype, shape=shape, kwargs=kwargs)
        self.assertEqual(cases, 128)

    def test_every_shape_for_each_entry(self):
        for name, shape, flow in itertools.product(_DY_ENTRIES, _DY_SHAPES, (True, False)):
            label = f"{name} {shape} flow={flow}"
            with self.subTest(label):
                self._compare(getattr(self.dy, "sample_" + name), getattr(self.ext, "sample_" + name), label,
                              flow=flow, dtype=torch.float32, shape=shape, kwargs={"s_churn": 1.5})

    def test_core_rules_beyond_the_entries(self):
        # upstream's ``max`` churn (γ ≥ √2−1 on every step), s_tmin/s_tmax windows, no CFG (CFG++ falls back)
        variants = (
            {"churn_rule": "max"}, {"churn_rule": "max", "s_churn": 3.0}, {"s_churn": 1.5, "s_tmin": 0.4},
            {"s_churn": 1.5, "s_tmax": 0.6}, {"s_churn": 1.5, "s_noise": 0.5}, {"sam_extra_step_offset": 1},
            {"sam_extra_step_offset": 3}, {"sam_extra_step_offset": 6},
        )
        for variant, smea, cfg_pp, flow, cfg in itertools.product(variants, (False, True), (False, True),
                                                                  (True, False), (True, False)):
            kwargs = {"smea": smea, "cfg_pp": cfg_pp, **variant}
            label = f"{kwargs} flow={flow} cfg_model={cfg}"
            with self.subTest(label):
                self._compare(self.dy.euler_dy, self.ext.euler_dy, label, flow=flow, dtype=torch.float64,
                              shape=_DY_SHAPES[1], kwargs=kwargs, cfg=cfg)

    def test_substeps_sit_where_the_pass_index_says(self):
        expected = {  # pass step → marker of the extra evaluation
            ("euler_dy", 0): {2: "dy", 3: "dy"}, ("euler_dy", 2): {0: "dy", 1: "dy"},
            ("euler_smea_dy", 0): {0: "smea", 1: "dy"}, ("euler_smea_dy", 2): {},
        }
        for (name, offset), where in expected.items():
            for suffix in ("", "_cfg_pp"):
                with self.subTest(name=name + suffix, offset=offset):
                    _out, markers, _calls, _seen = self._run(
                        getattr(self.dy, f"sample_{name}{suffix}"), ours=True, flow=True, dtype=torch.float32,
                        shape=(1, 4, 8, 10), kwargs={"sam_extra_step_offset": offset})
                    step, found = -1, {}
                    for marker, shape in markers:
                        if marker is None:
                            step += 1
                            self.assertEqual(shape, (1, 4, 8, 10))
                        else:
                            found[step] = marker
                            self.assertEqual(shape, (1, 4, 4, 5) if marker == "dy" else (1, 4, 10, 12))
                    self.assertEqual(found, where)

    def test_without_substeps_the_entries_are_the_hosts_euler_and_euler_cfg_pp(self):
        for flow, dtype in itertools.product((True, False), (torch.float32, torch.float64)):
            with self.subTest(flow=flow, dtype=dtype):
                for ours_name, host in (("sample_euler_dy", self.kd.sample_euler),
                                        ("sample_euler_smea_dy", self.kd.sample_euler),
                                        ("sample_euler_dy_cfg_pp", self.kd.sample_euler_cfg_pp),
                                        ("sample_euler_smea_dy_cfg_pp", self.kd.sample_euler_cfg_pp)):
                    ours = self._run(getattr(self.dy, ours_name), ours=True, flow=flow, dtype=dtype,
                                     shape=_DY_SHAPES[2], kwargs={"substeps": False}, record_callback=False)
                    theirs = self._run(host, ours=False, flow=flow, dtype=dtype, shape=_DY_SHAPES[2], noise=False,
                                       record_callback=False)
                    _same_bits(self, ours[0], theirs[0], f"{ours_name} vs host")

    def test_helpers_equal_the_extension(self):
        for rule, s_churn, steps, sigma in itertools.product(("min", "max"), (0.0, 0.3, 1.5, 9.0), (4, 28),
                                                             (0.05, 0.5, 0.99, 3.0)):
            for s_tmin, s_tmax in ((0.0, float("inf")), (0.4, 0.6), (1.0, 2.0)):
                self.assertEqual(self.dy.churn_gamma(rule, s_churn, steps, sigma, s_tmin, s_tmax),
                                 self.ext.churn_gamma(rule, s_churn, steps, sigma, s_tmin, s_tmax))
        with self.assertRaises(ValueError):
            self.dy.churn_gamma("other", 0.0, 4, 0.5, 0.0, 1.0)
        for shape, size, scale in (((3, 7, 9), None, (1.25, 1.25)), ((1, 4, 7, 9), (3, 4), None),
                                   ((2, 3, 2, 7, 9), None, (1.25, 1.25)), ((2, 3, 2, 10, 12), (7, 9), None)):
            with self.subTest(shape=shape):
                t = _latent(shape, torch.float32)
                _same_bits(self, self.dy.spatial_interpolate(t, size=size, scale_factor=scale),
                           self.ext.spatial_interpolate(t, size=size, scale_factor=scale))
        with self.assertRaises(ValueError):
            self.dy.spatial_interpolate(_latent((4, 4), torch.float32), size=(2, 2))
        self.assertEqual((self.dy.CHURN_MIN, self.dy.CHURN_MAX), (self.ext.CHURN_MIN, self.ext.CHURN_MAX))

    def test_the_cfg_pp_capture_is_appended_like_the_hosts(self):
        extra_args = {"model_options": {"sampler_post_cfg_function": ["existing"], "other": 1}}
        original = extra_args["model_options"]
        capture = self.dy.install_uncond_capture(extra_args, self.kd)
        self.assertEqual(extra_args["model_options"]["sampler_post_cfg_function"], ["existing", capture])
        self.assertTrue(extra_args["model_options"]["disable_cfg1_optimization"])
        self.assertEqual(original["sampler_post_cfg_function"], ["existing"])   # the caller's dict is not changed
        marked = self.dy.with_substep_marker(extra_args, "dy")
        self.assertEqual(marked["model_options"]["transformer_options"], {"sam_extra_substep": "dy"})
        self.assertNotIn("transformer_options", extra_args["model_options"])
        self.assertIs(marked["model_options"]["sampler_post_cfg_function"],
                      extra_args["model_options"]["sampler_post_cfg_function"])

    def test_on_comfyui_the_capture_goes_through_comfy_model_patcher(self):
        """ComfyUI's k-diffusion module has no ``set_model_options_post_cfg_function`` attribute (it calls
        ``comfy.model_patcher``'s): the capture is installed through that module's function."""
        bare = _module_with(self.kd)
        del bare.set_model_options_post_cfg_function
        calls = []

        def setter(model_options, function, disable_cfg1_optimization=False):
            calls.append((function, disable_cfg1_optimization))
            return self.kd.set_model_options_post_cfg_function(model_options, function, disable_cfg1_optimization)

        comfy = types.ModuleType("comfy")
        comfy.model_patcher = types.ModuleType("comfy.model_patcher")
        comfy.model_patcher.set_model_options_post_cfg_function = setter
        from tests._forge_k_diffusion import stub_modules

        with stub_modules({"comfy": comfy, "comfy.model_patcher": comfy.model_patcher}):
            self.assertIs(self.dy.post_cfg_setter(bare), setter)
            extra_args = {"model_options": {}}
            capture = self.dy.install_uncond_capture(extra_args, bare)
        self.assertEqual(calls, [(capture, True)])
        self.assertEqual(extra_args["model_options"]["sampler_post_cfg_function"], [capture])
        self.assertIs(self.dy.post_cfg_setter(self.kd), self.kd.set_model_options_post_cfg_function)


class InpaintModel:
    """ComfyUI's ``KSamplerX0Inpaint`` semantics (mask blend before and after the model) around a CFG
    stand-in; the ``CFGGuider`` it wraps is ``inner_model`` (``model_patcher``, ``conds``)."""

    def __init__(self, sampling, latent_image, noise, conds=None, raise_on_substep=False):
        self.inner_model = types.SimpleNamespace(model_patcher=ComfyPatcher(sampling), conds=conds or {})
        self.latent_image = latent_image
        self.noise = noise
        self.toy = CfgToyModel(sampling)
        self.raise_on_substep = raise_on_substep
        self.seen = []

    def __call__(self, x, sigma, denoise_mask=None, model_options={}, seed=None):
        marker = (model_options.get("transformer_options") or {}).get("sam_extra_substep")
        self.seen.append((marker, tuple(x.shape), None if denoise_mask is None else tuple(denoise_mask.shape),
                          tuple(self.latent_image.shape), tuple(self.noise.shape)))
        if marker and self.raise_on_substep:
            raise RuntimeError("interrupted")
        if denoise_mask is not None:
            s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
            latent_mask = 1.0 - denoise_mask
            x = x * denoise_mask + (self.latent_image * (1 - s) + self.noise * s) * latent_mask
        out = self.toy(x, sigma, model_options=model_options)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out


@requires_torch
class DyHostTests(unittest.TestCase):
    """ComfyUI host parts of Euler (SMEA) Dy: the sub-step inputs follow and come back; the guard."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy as dy
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_dy_host as host

        cls.dy, cls.host = dy, host
        cls.kd = forge_k_sampling()

    def _inpaint_run(self, name, shape, *, mask, flow=True, raise_on_substep=False, conds=None, offset=0):
        sigmas = _sigmas(flow, 6, torch.float32)
        x = _latent(shape, torch.float32, 3) * float(sigmas[0])
        latent_image, noise = _latent(shape, torch.float32, 4), _latent(shape, torch.float32, 6)
        model = InpaintModel(Predictor(flow), latent_image, noise, conds=conds, raise_on_substep=raise_on_substep)
        extra_args = {"denoise_mask": mask, "model_options": {}}
        out = getattr(self.dy, "sample_" + name)(
            model, x, sigmas, extra_args=extra_args, disable=True, noise_sampler=ReplayNoise(x, 5), kd=self.kd,
            substep_inputs=self.dy.SubstepInputs, sam_extra_step_offset=offset)
        return out, model, extra_args, (latent_image, noise, mask)

    def test_masked_sub_steps_see_inputs_of_their_own_size_and_give_them_back(self):
        for name, shape in itertools.product(_DY_ENTRIES, ((1, 4, 8, 10), (1, 4, 7, 9), (1, 3, 2, 7, 9))):
            with self.subTest(name=name, shape=shape):
                mask = (_latent(shape, torch.float32, 8) > 0).to(torch.float32)
                _out, model, extra_args, (latent_image, noise, mask) = self._inpaint_run(name, shape, mask=mask)
                substeps = [row for row in model.seen if row[0]]
                self.assertTrue(substeps)
                for marker, x_shape, mask_shape, latent_shape, noise_shape in model.seen:
                    self.assertEqual({x_shape}, {mask_shape, latent_shape, noise_shape})
                    if marker is None:
                        self.assertEqual(x_shape, shape)
                    else:
                        self.assertNotEqual(x_shape, shape)
                self.assertIs(model.latent_image, latent_image)
                self.assertIs(model.noise, noise)
                self.assertIs(extra_args["denoise_mask"], mask)

    def test_an_all_ones_mask_gives_the_unmasked_result(self):
        for name, shape in itertools.product(_DY_ENTRIES, ((1, 4, 7, 9), (1, 3, 2, 8, 6))):
            with self.subTest(name=name, shape=shape):
                masked = self._inpaint_run(name, shape, mask=torch.ones(shape))[0]
                plain = self._inpaint_run(name, shape, mask=None)[0]
                _same_bits(self, masked, plain)

    def test_inputs_come_back_when_a_sub_step_raises(self):
        shape = (1, 4, 8, 10)
        mask = torch.ones(shape)
        model_ref = {}
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            sigmas = _sigmas(True, 6, torch.float32)
            x = _latent(shape, torch.float32)
            latent_image, noise = _latent(shape, torch.float32, 4), _latent(shape, torch.float32, 6)
            model = InpaintModel(Predictor(True), latent_image, noise, raise_on_substep=True)
            model_ref.update(model=model, latent_image=latent_image, noise=noise)
            self.dy.sample_euler_smea_dy(model, x, sigmas, extra_args={"denoise_mask": mask}, disable=True,
                                         noise_sampler=ReplayNoise(x, 1), kd=self.kd,
                                         substep_inputs=self.dy.SubstepInputs)
        self.assertIs(model_ref["model"].latent_image, model_ref["latent_image"])
        self.assertIs(model_ref["model"].noise, model_ref["noise"])

    def test_inputs_of_another_size_are_left_alone(self):
        model = types.SimpleNamespace(latent_image=torch.zeros(1, 4, 5, 5), noise=torch.zeros(4, 4))
        mask = torch.zeros(1, 4, 9, 9)
        inputs = self.dy.SubstepInputs(model, {"denoise_mask": mask, "seed": 3}, (1, 4, 8, 8), (4, 4), "dy")
        with inputs as args:
            self.assertIs(args["denoise_mask"], mask)
            self.assertEqual(tuple(model.latent_image.shape), (1, 4, 5, 5))
            self.assertEqual(args["seed"], 3)
            self.assertEqual(args["model_options"]["transformer_options"]["sam_extra_substep"], "dy")
        latent_image, noise = _latent((1, 4, 8, 8), torch.float32, 1), _latent((1, 4, 8, 8), torch.float32, 2)
        mask = _latent((1, 4, 8, 8), torch.float32, 3)
        follows = types.SimpleNamespace(latent_image=latent_image, noise=noise)
        with self.dy.SubstepInputs(follows, {"denoise_mask": mask}, (1, 4, 8, 8), (10, 10), "smea") as args:
            for value, original in ((follows.latent_image, latent_image), (follows.noise, noise),
                                    (args["denoise_mask"], mask)):
                _same_bits(self, value, self.dy.spatial_interpolate(original, size=(10, 10)))
        self.assertIs(follows.noise, noise)
        frames = types.SimpleNamespace(latent_image=torch.zeros(1, 4, 2, 8, 8))
        with self.dy.SubstepInputs(frames, {}, (1, 4, 2, 8, 8), (4, 4), "dy") as args:
            self.assertEqual(tuple(frames.latent_image.shape), (1, 4, 2, 4, 4))
            self.assertNotIn("denoise_mask", args)

    # ── the guard ──

    def _guider_model(self, conds=None):
        return types.SimpleNamespace(inner_model=types.SimpleNamespace(conds=conds or {}))

    def test_each_reason_is_found(self):
        host = self.host

        def spectrum_unet_wrapper(apply_model, args):
            return apply_model(args["input"], args["timestep"], **args["c"])

        spectrum_unet_wrapper.__module__ = "lib_spectrum.forecaster"
        cases = (
            ({}, {}, []),
            ({}, {"model_options": {"transformer_options": {host.DY_SUBSTEPS_OFF_KEY: "Spectrum"}}},
             [host.REASON_FLAG]),
            ({}, {"model_options": {"model_function_wrapper": spectrum_unet_wrapper}}, [host.REASON_SPECTRUM]),
            ({"positive": [{"model_conds": {"c_concat": object()}}]}, {}, [host.REASON_CONCAT]),
            ({"negative": [{"concat_latent_image": object(), "model_conds": {}}]}, {}, [host.REASON_CONCAT]),
            ({"positive": [{"mask": object()}]}, {}, [host.REASON_REGIONAL]),
            ({"positive": [{"area": (8, 8, 0, 0)}]}, {}, [host.REASON_REGIONAL]),
            ({"positive": [{"area": None, "mask": None, "model_conds": {"c_crossattn": 1}}]}, {"model_options": {
                "model_function_wrapper": lambda apply, args: None}}, []),
            ({"positive": [{"mask": 1, "model_conds": {"c_concat": 1}}]},
             {"model_options": {"transformer_options": {host.DY_SUBSTEPS_OFF_KEY: "SPEED (SpectrumSPDKSampler)"}}},
             [host.REASON_FLAG, host.REASON_CONCAT, host.REASON_REGIONAL]),
        )
        for conds, extra_args, expected in cases:
            with self.subTest(conds=conds, extra_args=extra_args):
                self.assertEqual(host.blocking_reasons(self._guider_model(conds), extra_args), expected)
        self.assertEqual(host.blocking_reasons(object(), None), [])
        self.assertIn("SPEED (SpectrumSPDKSampler)", host.explain(host.REASON_FLAG, {"model_options": {
            "transformer_options": {host.DY_SUBSTEPS_OFF_KEY: "SPEED (SpectrumSPDKSampler)"}}}))

    def test_guarded_entries_turn_the_sub_steps_off_and_say_so_once(self):
        host = self.host
        calls = []

        def inner(model, x, sigmas, **kwargs):
            calls.append(kwargs)
            return x

        sampler = host.guarded(inner, "euler_dy_cfg_pp", cfg_pp=True)
        model = self._guider_model({"positive": [{"model_conds": {"c_concat": 1}}]})
        host._logged.clear()
        with self.assertLogs("ai_studio_forge_parity", "WARNING") as logs:
            sampler(model, "x", "sigmas", extra_args={}, sam_extra_step_offset=2)
            sampler(model, "x", "sigmas", extra_args={})
        self.assertEqual(len(logs.output), 1)
        self.assertIn("plain CFG++ Euler steps", logs.output[0])
        self.assertEqual([c["substeps"] for c in calls], [False, False])
        self.assertEqual(calls[0]["sam_extra_step_offset"], 2)
        self.assertIs(calls[0]["substep_inputs"], self.dy.SubstepInputs)
        calls.clear()
        sampler(self._guider_model(), "x", "sigmas", extra_args={}, s_churn=1.5, noise_sampler="n")
        self.assertEqual((calls[0]["substeps"], calls[0]["s_churn"], calls[0]["noise_sampler"]), (True, 1.5, "n"))
        calls.clear()
        sampler(model, "x", "sigmas", extra_args={}, substeps=False)
        self.assertFalse(calls[0]["substeps"])

    def test_a_guarded_entry_runs_the_real_sampler(self):
        # euler_dy under a Spectrum flag == the same entry with substeps=False; without it, the sub-steps run.
        shape = (1, 4, 8, 10)
        sigmas = _sigmas(True, 6, torch.float32)
        x = _latent(shape, torch.float32) * float(sigmas[0])
        flagged = {"model_options": {"transformer_options": {self.host.DY_SUBSTEPS_OFF_KEY: "Spectrum"}}}
        model = InpaintModel(Predictor(True), torch.zeros(shape), torch.zeros(shape))
        entry = self.host.guarded(_with_kd(self.dy.sample_euler_dy, self.kd), "euler_dy", cfg_pp=False)
        self.host._logged.add("euler_dy:flag")
        out = entry(model, x, sigmas, extra_args=dict(flagged), noise_sampler=ReplayNoise(x, 2), disable=True)
        self.assertFalse(any(row[0] for row in model.seen))
        plain = self.dy.sample_euler_dy(InpaintModel(Predictor(True), torch.zeros(shape), torch.zeros(shape)), x,
                                        sigmas, extra_args=dict(flagged), noise_sampler=ReplayNoise(x, 2),
                                        disable=True, substeps=False, kd=self.kd)
        _same_bits(self, out, plain)
        model = InpaintModel(Predictor(True), torch.zeros(shape), torch.zeros(shape))
        entry(model, x, sigmas, extra_args={}, noise_sampler=ReplayNoise(x, 2), disable=True)
        self.assertEqual([row[0] for row in model.seen if row[0]], ["dy", "dy"])


def _with_kd(function, kd):
    def call(*args, **kwargs):
        return function(*args, kd=kd, **kwargs)

    call.__name__ = function.__name__
    return call


class PureModuleImportTests(unittest.TestCase):
    """The pack's sampler modules stay importable by the app (no torch/ComfyUI at their top)."""

    def test_top_level_imports_are_standard_library_or_the_pack(self):
        for name in sorted(path.name for path in PACK_DIR.glob("extra_samplers*.py")):
            with self.subTest(module=name):
                tree = ast.parse((PACK_DIR / name).read_text(encoding="utf-8"))
                imported = set()
                for item in tree.body:
                    if isinstance(item, ast.Import):
                        imported.update(alias.name.split(".")[0] for alias in item.names)
                    elif isinstance(item, ast.ImportFrom):
                        imported.add("." if item.level else (item.module or "").split(".")[0])
                self.assertLessEqual(imported, {"__future__", ".", "functools", "importlib", "inspect", "logging",
                                                "math", "typing"})


# ── DPM++ flow ODE trio (sam-extra dpmpp_flow) on Forge's k-diffusion ───────────────────────────


_FLOW_ODE = ("dpmpp_2m_flow_ode", "dpmpp_2m_heun_flow_ode", "dpmpp_3m_flow_ode")
# the host call each entry stands for (function, solver_type) — sam-extra dpmpp_flow / DESIGN §3 #1
_FLOW_ODE_HOST = {"dpmpp_2m_flow_ode": ("sample_dpmpp_2m_sde", "midpoint"),
                  "dpmpp_2m_heun_flow_ode": ("sample_dpmpp_2m_sde", "heun"),
                  "dpmpp_3m_flow_ode": ("sample_dpmpp_3m_sde", None)}


@requires_torch
@requires_extension
class FlowOdeParityTests(unittest.TestCase):
    """``dpmpp_2m_flow_ode`` / ``dpmpp_2m_heun_flow_ode`` / ``dpmpp_3m_flow_ode`` == sam-extra's ``dpmpp_flow``
    entries on Forge's k-diffusion module, bit for bit, and == the host's DPM++ 2M/3M SDE at η 0."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_flow as flow

        cls.flow = flow
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.ext = import_ext("sam3ext.extra_samplers.dpmpp_flow")

    def _run(self, function, *, flow, dtype, shape, steps, ours, callback=True, kwargs=None):
        model = ToyModel(Predictor(flow))
        sigmas = _sigmas(flow, steps, dtype)
        x = _latent(shape, dtype) * float(sigmas[0])
        seen = []
        call = dict(kwargs or {})
        if ours:
            call["kd"] = self.kd
        with installed_k_sampling(self.kd):
            out = function(model, x, sigmas, extra_args={}, disable=True,
                           callback=(lambda info: seen.append((info["i"], info["x"].clone()))) if callback else None,
                           **call)
        return out, [c[0] for c in model.calls], seen

    def _same_run(self, ours, theirs, label):
        _same_bits(self, ours[0], theirs[0], label)
        self.assertEqual(ours[1], theirs[1], f"{label}: model calls")
        self.assertEqual([i for i, _x in ours[2]], [i for i, _x in theirs[2]], f"{label}: callback steps")
        for (i, a), (_j, b) in zip(ours[2], theirs[2]):
            _same_bits(self, a, b, f"{label}: callback x at step {i}")

    def test_entries_equal_the_extension_over_the_grid(self):
        cases = 0
        for name, flow, dtype, shape, steps in itertools.product(
                _FLOW_ODE, (True, False), (torch.float32, torch.float64), _SHAPES, (1, 2, 3, 6, 28)):
            label = f"{name} flow={flow} {dtype} {shape} steps={steps}"
            with self.subTest(label):
                common = dict(flow=flow, dtype=dtype, shape=shape, steps=steps)
                ours = self._run(getattr(self.flow, "sample_" + name), ours=True, **common)
                theirs = self._run(getattr(self.ext, "sample_" + name), ours=False, **common)
                self._same_run(ours, theirs, label)
                self.assertEqual(len(ours[1]), steps)
                cases += 1
        self.assertEqual(cases, 120)

    def test_each_entry_is_the_hosts_sde_at_eta_zero(self):
        """The ODE entry == the host's SDE function at η 0 with s_noise 1 and a real noise sampler, which is
        never called — "DPM++ 3M (flow ODE)" is Forge's "DPM++ 3M SDE" at Eta 0 (and 2M/2M Heun alike)."""
        for name, flow, dtype in itertools.product(_FLOW_ODE, (True, False), (torch.float32, torch.float64)):
            label = f"{name} flow={flow} {dtype}"
            with self.subTest(label):
                function, solver = _FLOW_ODE_HOST[name]
                host_kwargs = {"eta": 0.0, "s_noise": 1.0}
                if solver is not None:
                    host_kwargs["solver_type"] = solver
                common = dict(flow=flow, dtype=dtype, shape=_SHAPES[1], steps=9)
                ours = self._run(getattr(self.flow, "sample_" + name), ours=True, **common)
                model = ToyModel(Predictor(flow))
                sigmas = _sigmas(flow, 9, dtype)
                x = _latent(_SHAPES[1], dtype) * float(sigmas[0])
                noise = ReplayNoise(x, 99)   # a real sampler: at eta 0 never called
                host = getattr(self.kd, function)(model, x, sigmas, extra_args={}, disable=True, noise_sampler=noise,
                                                  **host_kwargs)
                _same_bits(self, ours[0], host, label)
                self.assertEqual(noise.draws, [], label)

    def test_the_solver_type_reaches_the_host(self):
        """midpoint and Heun differ from the second step on (the comparison above is not vacuous)."""
        common = dict(flow=True, dtype=torch.float64, shape=_SHAPES[0], steps=8, ours=True)
        midpoint = self._run(self.flow.sample_dpmpp_2m_flow_ode, **common)
        heun = self._run(self.flow.sample_dpmpp_2m_heun_flow_ode, **common)
        three = self._run(self.flow.sample_dpmpp_3m_flow_ode, **common)
        self.assertFalse(torch.equal(midpoint[0], heun[0]))
        self.assertFalse(torch.equal(midpoint[0], three[0]))

    def test_the_noise_sampler_is_never_called_and_says_so_if_it_were(self):
        self.assertEqual(self.flow.NOISE_DRAWN, "a flow ODE sampler drew noise")
        with self.assertRaises(RuntimeError) as raised:
            self.flow.no_noise(1.0, 0.5)
        with self.assertRaises(RuntimeError) as theirs:
            self.ext._no_noise(1.0, 0.5)
        self.assertEqual(str(raised.exception), str(theirs.exception))
        # the entries pass exactly that sampler with eta 0 / s_noise 0 (a recording host shows the call)
        calls = []

        def record(name):
            def host(model, x, sigmas, **kwargs):
                calls.append((name, kwargs))
                return x
            return host

        kd = _module_with(self.kd, sample_dpmpp_2m_sde=record("2m"), sample_dpmpp_3m_sde=record("3m"))
        x = _latent(_SHAPES[0], torch.float32)
        for name in _FLOW_ODE:
            getattr(self.flow, "sample_" + name)(ToyModel(Predictor(True)), x, _sigmas(True, 4, torch.float32), kd=kd)
        self.assertEqual([name for name, _kwargs in calls], ["2m", "2m", "3m"])
        for (_name, kwargs), solver in zip(calls, ("midpoint", "heun", None)):
            self.assertIs(kwargs["noise_sampler"], self.flow.no_noise)
            self.assertEqual((kwargs["eta"], kwargs["s_noise"], kwargs.get("solver_type")), (0.0, 0.0, solver))

    def test_signatures_are_the_extensions(self):
        """No ``eta`` and no ``noise_sampler`` — the pack's names are sam-extra's plus ``kd``."""
        import inspect

        for name in _FLOW_ODE:
            with self.subTest(name):
                ours = list(inspect.signature(getattr(self.flow, "sample_" + name)).parameters)
                theirs = list(inspect.signature(getattr(self.ext, "sample_" + name)).parameters)
                self.assertEqual(ours, [*theirs, "kd"])
                self.assertNotIn("eta", ours)
                self.assertNotIn("noise_sampler", ours)

    def test_short_schedules_return_the_input(self):
        for name in _FLOW_ODE:
            for length in (0, 1):
                with self.subTest(name=name, length=length):
                    x = _latent(_SHAPES[0], torch.float32)
                    out = getattr(self.flow, "sample_" + name)(ToyModel(Predictor(True)), x, torch.ones(length),
                                                               kd=self.kd)
                    self.assertIs(out, x)

# ── Restart (flow) (sam-extra restart_flow) on Forge's k-diffusion ──────────────────────────────

_RESTART = "sam3ext/extra_samplers/restart_flow.py"
_RESTART_CONSTANTS = ("RESTART_S_MIN", "RESTART_S_MAX", "RESTART_MIN_STEPS", "RESTART_TIMES", "RESTART_TIMES_LONG",
                      "RESTART_LONG_FROM")
_RESTART_STEPS = (19, 20, 28, 36, 50)


class StableToyModel(ToyModel):
    """A bounded x0 at every σ (``tanh(x/√(1+σ²))``): the ε runs from σ 14.6 stay finite, unlike ``ToyModel``'s."""

    def __call__(self, x, sigma, **extra_args):
        self.calls.append((float(sigma.reshape(-1)[0]), tuple(x.shape), extra_args.get("model_options")))
        s = sigma.reshape([-1] + [1] * (x.ndim - 1)).to(x.dtype)
        return torch.tanh(x / (1.0 + s * s) ** 0.5) * 0.8 + self.bias * torch.tanh(s)


def _restart_grids(seed=3):
    """(flow, sigmas) — the shift-3 flow and Karras ρ 7 grids of every step count plus seeded random,
    near-zero, flat and σ-1-then-tiny grids (the "no room" cases)."""
    import random

    grids = [(flow, _sigmas(flow, steps, dtype)) for flow, steps, dtype in itertools.product(
        (True, False), (*_RESTART_STEPS, 21, 35, 37, 60), (torch.float32, torch.float64))]
    rng = random.Random(seed)
    for steps, flow, kind in itertools.product(range(18, 62, 3), (True, False), ("random", "tiny", "flat", "one")):
        top = 1.0 if flow else rng.uniform(0.5, 20.0)
        if kind == "random":
            values = sorted((rng.uniform(0.0, top) for _ in range(steps)), reverse=True)
        elif kind == "tiny":
            values = sorted((rng.uniform(0.0, 0.05) for _ in range(steps)), reverse=True)
        elif kind == "flat":
            values = [0.5] * steps
        else:
            values = [top] + sorted((rng.uniform(0.0, 0.08) for _ in range(steps - 1)), reverse=True)
        if flow and kind != "one":
            values[0] = 1.0
        grids.append((flow, torch.tensor(values + [0.0], dtype=rng.choice((torch.float32, torch.float64)))))
    return grids


@requires_torch
@requires_extension
class RestartFlowParityTests(unittest.TestCase):
    """``restart_flow`` == sam-extra's ``restart_flow.sample_restart_flow`` on Forge's k-diffusion module, bit for bit
    (written from the paper; sam-extra is the oracle only)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import extra_samplers_restart as restart

        cls.restart = restart
        cls.kd = forge_k_sampling()
        with installed_k_sampling(cls.kd):
            cls.ext = import_ext("sam3ext.extra_samplers.restart_flow")

    @staticmethod
    def _plan(plan):
        return None if plan is None else (plan.lo, plan.hi, plan.times)

    def _run(self, function, *, flow, dtype, shape, sigmas, ours, s_noise=1.0, noise=True, extra_args=None):
        """``noise=None``: no ``noise_sampler`` argument at all (the host's ``sample_heun`` takes none)."""
        model = StableToyModel(Predictor(flow))
        x = _latent(shape, dtype) * float(sigmas[0])
        sampler = ReplayNoise(x, 77) if noise else None
        seen = []
        call = {"kd": self.kd} if ours else {}
        if noise is not None:
            call["noise_sampler"] = sampler
        with installed_k_sampling(self.kd):
            out = function(model, x, sigmas, extra_args=dict(extra_args or {}), disable=True, s_noise=s_noise,
                           callback=lambda info: seen.append((info["i"], float(info["sigma"]), info["x"].clone())),
                           **call)
        return out, (sampler.draws if sampler else None), [c[0] for c in model.calls], seen

    def _same_run(self, ours, theirs, label):
        _same_bits(self, ours[0], theirs[0], label)
        self.assertEqual(ours[1], theirs[1], f"{label}: noise draws")
        self.assertEqual(ours[2], theirs[2], f"{label}: model calls")
        self.assertEqual([c[:2] for c in ours[3]], [c[:2] for c in theirs[3]], f"{label}: callback steps")
        for a, b in zip(ours[3], theirs[3]):
            _same_bits(self, a[2], b[2], f"{label}: callback x at step {a[0]}")

    def test_constants_equal_the_extension(self):
        literals = ext_literals(_RESTART)
        for name in _RESTART_CONSTANTS:
            with self.subTest(name=name):
                self.assertEqual(_key(getattr(self.restart, name)), _key(literals[name]))
        self.assertEqual((self.restart.RESTART_S_MIN, self.restart.RESTART_S_MAX, self.restart.RESTART_MIN_STEPS,
                          self.restart.RESTART_TIMES, self.restart.RESTART_TIMES_LONG, self.restart.RESTART_LONG_FROM),
                         (0.1, 2.0, 20, 1, 2, 36))

    def test_plans_equal_the_extension(self):
        grids = _restart_grids()
        planned = 0
        for index, (flow, sigmas) in enumerate(grids):
            with self.subTest(index=index, flow=flow, steps=len(sigmas) - 1):
                ours = self._plan(self.restart.restart_plan(sigmas, flow))
                self.assertEqual(ours, self._plan(self.ext.restart_plan(sigmas, flow)))
                planned += ours is not None
        self.assertGreater(planned, 60)
        self.assertGreater(len(grids) - planned, 40)   # the "no restart" rules are exercised too
        # Anima shift 3, 28 steps: σ_j = 3t/(1+2t), t = 1 − j/28, s = 3t/(1−t) — s_27 = 0.111 (σ 0.1) is closest to
        # 0.1, s_17 = 1.94 (σ 0.66) to 2 (s_16 = 2.25)
        anima = self.restart.restart_plan(_sigmas(True, 28, torch.float32), True)
        self.assertEqual(self._plan(anima), (27, 17, 1))
        self.assertIsNone(self.restart.restart_plan(_sigmas(True, 19, torch.float32), True))
        self.assertEqual(self.restart.restart_plan(_sigmas(True, 36, torch.float32), True).times, 2)

    def test_noise_levels_and_kernels_equal_the_extension(self):
        generator = torch.Generator().manual_seed(11)
        for dtype in (torch.float32, torch.float64):
            values = torch.rand(4000, generator=generator, dtype=torch.float64).to(dtype) * 0.98 + 0.01
            for k in range(0, 4000, 2):   # enough pairs to see a 1-ulp difference of the float arithmetic
                lo, hi = sorted((values[k], values[k + 1]))
                for flow, pair in itertools.product((True, False), ((lo, hi), (float(lo), float(hi)))):
                    ours = self.restart.restart_kernel(*pair, flow)
                    theirs = self.ext.restart_kernel(*pair, flow)
                    for a, b in zip(ours, theirs):
                        self.assertEqual(torch.is_tensor(a), torch.is_tensor(b))
                        self.assertEqual(float(a), float(b), (dtype, flow, k))
                    self.assertEqual(self.restart.noise_level(float(lo), flow), self.ext.noise_level(float(lo), flow))
        for flow, sigma in itertools.product((True, False), (0.0, 0.5, 1.0, 1.5)):
            self.assertEqual(self.restart.noise_level(sigma, flow), self.ext.noise_level(sigma, flow))
        # Python-float inputs where ``v ** 0.5`` and ``math.sqrt(v)`` round differently (~1 in 1000 here): the
        # float arithmetic is the extension's too, not only the tensor one the sampler uses
        import math
        import random

        rng, hard = random.Random(7), []
        for _ in range(200000):
            if len(hard) >= 6:
                break
            lo, hi = sorted((rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)))
            for flow, value in ((False, hi ** 2 - lo ** 2), (True, (hi / (1 - hi)) ** 2 - (lo / (1 - lo)) ** 2)):
                if value > 0 and value ** 0.5 != math.sqrt(value):
                    hard.append((lo, hi, flow))
        self.assertGreaterEqual(len(hard), 6)
        self.assertTrue(any(not flow for _lo, _hi, flow in hard))
        for lo, hi, flow in hard:
            with self.subTest(lo=lo, hi=hi, flow=flow):
                self.assertEqual(self.restart.restart_kernel(lo, hi, flow), self.ext.restart_kernel(lo, hi, flow))
        self.assertEqual(self.restart.noise_level(1.0, True), float("inf"))

    def test_sampler_equals_the_extension_over_the_grid(self):
        cases = 0
        for flow, dtype, steps, s_noise, shape in itertools.product(
                (True, False), (torch.float32, torch.float64), _RESTART_STEPS, (1.0, 0.7), _SHAPES):
            label = f"flow={flow} {dtype} steps={steps} s_noise={s_noise} {shape}"
            with self.subTest(label):
                common = dict(flow=flow, dtype=dtype, shape=shape, sigmas=_sigmas(flow, steps, dtype), s_noise=s_noise)
                ours = self._run(self.restart.sample_restart_flow, ours=True, **common)
                theirs = self._run(self.ext.sample_restart_flow, ours=False, **common)
                self._same_run(ours, theirs, label)
                self.assertTrue(torch.isfinite(ours[0]).all(), label)
                plan = self.restart.restart_plan(common["sigmas"], flow)
                self.assertEqual(len(ours[1]), 0 if plan is None else plan.times, label)
                cases += 1
        self.assertEqual(cases, 80)

    def test_odd_grids_equal_the_extension(self):
        """Random and "no room" grids (plan or not) run alike too — the sampler follows its plan."""
        for index, (flow, sigmas) in enumerate(_restart_grids(seed=5)[::9]):
            with self.subTest(index=index, flow=flow, steps=len(sigmas) - 1):
                common = dict(flow=flow, dtype=sigmas.dtype, shape=_SHAPES[0], sigmas=sigmas)
                self._same_run(self._run(self.restart.sample_restart_flow, ours=True, **common),
                               self._run(self.ext.sample_restart_flow, ours=False, **common), str(index))

    def test_without_a_restart_it_is_the_hosts_heun(self):
        for flow, dtype in itertools.product((True, False), (torch.float32, torch.float64)):
            with self.subTest(flow=flow, dtype=dtype):
                sigmas = _sigmas(flow, 19, dtype)
                self.assertIsNone(self.restart.restart_plan(sigmas, flow))
                ours = self._run(self.restart.sample_restart_flow, flow=flow, dtype=dtype, shape=_SHAPES[1],
                                 sigmas=sigmas, ours=True)
                host = self._run(self.kd.sample_heun, flow=flow, dtype=dtype, shape=_SHAPES[1], sigmas=sigmas,
                                 ours=False, noise=None)
                self._same_run((ours[0], None, ours[2], ours[3]), (host[0], None, host[2], host[3]), "heun")
                self.assertEqual(ours[1], [])

    def test_the_restart_re_walks_the_schedule(self):
        """Callback ``i``: the main steps up to ``lo``, then ``hi … lo−1`` once per restart, then the rest; one draw
        per restart with ``(σ_lo, σ_hi)``; the restart is not vacuous (the latent differs from plain Heun)."""
        for flow, steps in ((True, 28), (True, 36), (False, 50)):
            with self.subTest(flow=flow, steps=steps):
                sigmas = _sigmas(flow, steps, torch.float64)
                plan = self.restart.restart_plan(sigmas, flow)
                ours = self._run(self.restart.sample_restart_flow, flow=flow, dtype=torch.float64,
                                 shape=_SHAPES[0], sigmas=sigmas, ours=True)
                expected = [*range(plan.lo), *list(range(plan.hi, plan.lo)) * plan.times, *range(plan.lo, steps)]
                self.assertEqual([c[0] for c in ours[3]], expected)
                self.assertEqual(ours[1], [(float(sigmas[plan.lo]), float(sigmas[plan.hi]))] * plan.times)
                heun = self._run(self.kd.sample_heun, flow=flow, dtype=torch.float64, shape=_SHAPES[0],
                                 sigmas=sigmas, ours=False, noise=None)
                self.assertFalse(torch.equal(ours[0], heun[0]))

    def test_the_default_noise_is_the_hosts_and_built_only_for_a_restart(self):
        built = []

        def default_noise_sampler(x, seed=None):
            built.append(seed)
            return ReplayNoise(x, 5)

        kd = _module_with(self.kd, default_noise_sampler=default_noise_sampler)
        for steps, expected in ((28, [1234]), (19, [])):
            with self.subTest(steps=steps):
                built.clear()
                x = _latent(_SHAPES[0], torch.float32)
                with installed_k_sampling(kd):
                    self.restart.sample_restart_flow(StableToyModel(Predictor(True)), x,
                                                     _sigmas(True, steps, torch.float32), extra_args={"seed": 1234},
                                                     disable=True, kd=kd)
                self.assertEqual(built, expected)

    def test_the_flow_kernel_is_the_exact_forward_transition(self):
        """On ``x = α·x₀ + σ·n`` the jump σ_lo → σ_hi gives ``α_hi·x₀`` (scale·α_lo = α_hi) plus noise of variance
        σ_hi² exactly (scale²·σ_lo² + std² = σ_hi²); on ε models (α = 1) it is ``x + √(σ_hi² − σ_lo²)·ε``."""
        for lo, hi in ((0.1, 0.66), (0.0909, 0.6667), (0.2, 0.5), (0.05, 0.95)):
            with self.subTest(lo=lo, hi=hi):
                scale, std = self.restart.restart_kernel(lo, hi, True)
                self.assertAlmostEqual(scale * (1.0 - lo), 1.0 - hi, places=14)
                self.assertAlmostEqual((scale * lo) ** 2 + std ** 2, hi ** 2, places=14)
                scale, std = self.restart.restart_kernel(lo, hi, False)
                self.assertEqual(scale, 1.0)
                self.assertAlmostEqual(lo ** 2 + std ** 2, hi ** 2, places=14)

    def test_signature_is_the_extensions(self):
        import inspect

        ours = list(inspect.signature(self.restart.sample_restart_flow).parameters)
        theirs = list(inspect.signature(self.ext.sample_restart_flow).parameters)
        self.assertEqual(ours, [*theirs, "kd"])

if __name__ == "__main__":
    unittest.main()
