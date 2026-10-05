"""Pack 1.7.0 Anima SPEED == sam-extra v0.33.0 ``sam3ext.speed`` — same inputs, same results (CPU).

The pack re-implements SPEED for ComfyUI (``comfy_custom_nodes/ai_studio_forge_parity/speed_*.py``)
from the MIT upstreams; sam-extra (GPL) is never imported by the pack. Here the installed extension's
modules (``tests/_sam_extra_ext.EXT_ROOT``; ``sam3ext.speed.schedule`` needs only torch) are the
oracle: both run on the same inputs and every result is compared exactly — floats by their bits,
refusals by exception type, message, ``reason`` and ``kind``. sam-extra 0.32.1 refuses custom
spectra with A <= 0 or a non-finite A/beta as ``PlanError(kind="invalid")``; those rows are ordinary
parity rows here.

The suite pins sam-extra **v0.33.0**, like the Extra Schedulers/Samplers suites: v0.33.0's reader makes
a zero denominator in the scales/manual text a ``ValueError`` (``division by zero: '1/0'`` — an invalid
setting; 0.32.1 let ``ZeroDivisionError`` escape ``coerce_settings``), reads an integer too large for a
float as the default (0.32.1: ``OverflowError``) and lists ``sample_unipc_bh2`` among the samplers SPEED
skips. Against 0.32.1 those rows fail by design. Until v0.33.0 is installed, point the oracle at its work
tree: ``AISTUDIO_FORGE_EXTENSION_DIR=<tree>``.

* ``ScheduleConstantsTests`` (no torch): presets, names and limits equal the extension's literals.
* ``ScheduleParityTests`` (torch, extension): the scalar maths, text parsing, thresholds, both step
  lookups, the grids and ``build_plan`` over a large deterministic grid of modes, thresholds,
  presets (custom with valid and invalid spectra), adaptive delta, divisors, ladders, latent sizes
  and schedules (flow shifts 1/3/8; simple, normal, beta-like, linear-quadratic-like and Karras-like
  spacing; 8-50 steps; img2img/hires tails at denoise 0.95-0.35).
* ``SpectralParityTests`` (torch, extension): DCT basis and cache, DCT pair, downscale, amplitude
  resize and the DCT/FFT/Haar expansions on float32/float64/bfloat16/float16, 4-D and 5-D latents —
  same dtype, shape and bits; inputs unchanged.
* ``RunnerParityTests`` (torch, extension): the per-seed draws (numpy and torch, negative and >= 2**63
  seeds), the img2img start, and whole ``run_speed`` runs on a deterministic toy model with toy
  k-diffusion samplers (Euler, Euler ancestral with explicit noise samplers, DPM++ 2M, a
  step-offset probe): both modes, all transforms, 4-D/5-D, batch 2, fixed spectral seed, img2img
  rescale on/off, refusals and sampler failures — output bits, report, status text, every model
  call, callback step, published schedule and host call compared. The pack's own rule (the
  step-offset keyword read through CNS's ``_forge_neo_base``) is tested on the pack alone.

torch is imported lazily (tests/_optional_deps).
"""
from __future__ import annotations

import ast
import importlib
import math
import random
import sys
import unittest
from typing import Any, Callable

from comfy_custom_nodes.ai_studio_forge_parity import speed_schedule, speed_spectral
from tests._optional_deps import bind_torch, requires_torch
from tests._sam_extra_ext import EXT_ROOT, requires_extension

torch = None  # filled by bind_torch() in the torch classes


def _ext_speed(name: str):
    root = str(EXT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(f"sam3ext.speed.{name}")


def _ext_literal(relative: str, name: str) -> Any:
    """A module-level literal of the installed extension, read by AST (nothing is imported)."""
    tree = ast.parse((EXT_ROOT / relative).read_text(encoding="utf-8"))
    for item in tree.body:
        if isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in item.targets
        ):
            return ast.literal_eval(item.value)
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == name:
            return ast.literal_eval(item.value)
    raise AssertionError(f"{relative} has no module-level {name}")


# --------------------------------------------------------------------------------------------------
# Canonical forms: floats by their bits (NaN == NaN, -0.0 != 0.0), exceptions by what a caller sees
# --------------------------------------------------------------------------------------------------


def _bits(value) -> str:
    if not isinstance(value, float):
        raise AssertionError(f"expected a float, got {type(value).__name__} {value!r}")
    return value.hex()


def _canon(value):
    if isinstance(value, float):
        return ("f", _bits(value))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canon(item) for item in value))
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return (type(value).__name__, value)
    raise AssertionError(f"no canonical form for {type(value).__name__} {value!r}")


def _canon_transition(t) -> tuple:
    return (
        ("step", t.step, type(t.step).__name__), ("k", t.k, type(t.k).__name__),
        _canon(t.scale_from), _canon(t.scale_to), _canon(t.grid_from), _canon(t.grid_to),
        _canon(t.sigma), _canon(t.aligned),
    )


def _canon_plan(plan) -> tuple:
    return (
        plan.mode, plan.transform, _canon(plan.full_grid), _canon(plan.first_scale), _canon(plan.first_grid),
        tuple(_canon_transition(t) for t in plan.transitions), _canon(plan.thresholds),
        ("n_steps", plan.n_steps, type(plan.n_steps).__name__), _canon(list(plan.notes)),
        ("coarse", plan.coarse_steps), type(plan.transitions).__name__, type(plan.notes).__name__,
    )


def _canon_error(exc: BaseException) -> tuple:
    return (
        "raise", type(exc).__name__, isinstance(exc, ValueError), str(exc),
        getattr(exc, "reason", None), getattr(exc, "kind", None),
    )


def _outcome(fn: Callable, canon: Callable, *args, **kwargs) -> tuple:
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # the oracle's refusals are part of the contract
        return _canon_error(exc)
    return ("ok", canon(result))


# --------------------------------------------------------------------------------------------------
# Schedules (flow-matching sigmas in [0, 1], as a discrete-flow model's scheduler would give them)
# --------------------------------------------------------------------------------------------------


def _flow(t: float, shift: float) -> float:
    return shift * t / (1.0 + (shift - 1.0) * t)


def _spacing(kind: str, steps: int, shift: float) -> list[float]:
    """``steps`` sigmas from ~1 downwards plus the final 0 (shapes like ComfyUI's schedulers)."""
    if kind == "simple":
        ts = [1.0 - i / steps for i in range(steps)]
    elif kind == "normal":
        ts = [1.0 - (1.0 - 0.001) * i / (steps - 1) for i in range(steps)] if steps > 1 else [1.0]
    elif kind == "beta":       # arcsine spacing (beta(0.5, 0.5)): crowded at both ends
        ts = [0.5 * (1.0 + math.cos(math.pi * i / steps)) for i in range(steps)]
    elif kind == "linear_quadratic":
        half = steps // 2
        out = []
        for i in range(steps):
            if i < half:
                out.append(1.0 - 0.025 * i / half)
            else:
                u = (i - half) / (steps - half)
                out.append(0.975 * (1.0 - u) ** 2)
        return out + [0.0]
    elif kind == "karras":
        rho, lo, hi = 7.0, 0.002 ** (1.0 / 7.0), 1.0
        return [(hi + i / max(1, steps - 1) * (lo - hi)) ** rho for i in range(steps)] + [0.0]
    else:
        raise AssertionError(kind)
    return [_flow(t, shift) for t in ts] + [0.0]


def _schedule_pool() -> list[tuple[str, Any]]:
    """Full passes and img2img/hires tails (ComfyUI denoise: the last ``steps + 1`` of a longer list)."""
    pool = []
    for shift in (1.0, 3.0, 8.0):
        for steps in (8, 20, 28, 50):
            for kind in ("simple", "normal", "beta", "linear_quadratic", "karras"):
                pool.append((f"{kind}/{steps}/s{shift:g}", torch.tensor(_spacing(kind, steps, shift))))
                for denoise in (0.95, 0.8, 0.5, 0.35):
                    longer = _spacing(kind, int(steps / denoise), shift)
                    pool.append((f"{kind}/{steps}/s{shift:g}/d{denoise:g}", torch.tensor(longer[-(steps + 1):])))
    # other carriers and odd lists
    pool.append(("float64/simple/28/s3", torch.tensor(_spacing("simple", 28, 3.0), dtype=torch.float64)))
    pool.append(("list/simple/20/s3", _spacing("simple", 20, 3.0)))
    pool.append(("one-step", torch.tensor([1.0, 0.0])))
    pool.append(("two-step", torch.tensor([1.0, 0.5, 0.0])))
    pool.append(("flat-start", torch.tensor([1.0, 1.0, 1.0, 0.9, 0.6, 0.3, 0.0])))
    pool.append(("no-final-zero", torch.tensor([0.99, 0.9, 0.7, 0.5, 0.3, 0.1])))
    pool.append(("eps-like", torch.tensor([14.6, 8.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.0])))
    return pool


LADDERS = ((0.5, 1.0), (0.25, 0.5, 1.0), (0.3, 1.0), (0.75, 1.0))
WIDE_LADDERS = LADDERS + ((0.5, 0.9999999), (0.1, 0.4, 1.0), (0.2, 1.0), (0.25, 1.0), (0.125, 0.25, 0.5, 1.0),
                          (0.6, 0.8, 1.0))
LATENTS = ((128, 128), (104, 152), (125, 97), (64, 64))
WIDE_LATENTS = LATENTS + ((152, 104), (16, 16), (9, 7), (256, 192), (2, 2), (1, 3), (33, 65))
# custom spectra: the published fit, tiny/huge A, and every refusal sam-extra 0.32.1 makes
CUSTOM_SPECTRA = (
    (203.615097, 1.915461), (1e-12, 2.0), (5.0, 0.0), (100.0, -3.0), (1e300, 2.0), (100.0, 1e6), (100.0, 400.0),
    (0.0, 2.0), (-1.0, 2.0), (-1e-9, 1.9), (math.nan, 2.0), (math.inf, 2.0), (-math.inf, 2.0),
    (100.0, math.nan), (100.0, math.inf), (100.0, -math.inf),
)


def _plan_kwargs(rng: random.Random, pool, *, exhaustive: dict | None = None) -> dict:
    """One ``build_plan`` call: the given dimensions, the rest drawn from ``rng``."""
    chosen = dict(exhaustive or {})
    kwargs = {
        "mode": chosen.get("mode", rng.choice(speed_schedule.MODES)),
        "transform": chosen.get("transform", rng.choice(("dct", "dct", "fft", "dwt"))),
        "threshold": chosen.get("threshold", rng.choice(("neo_shift", "delta_optimal"))),
        "scales": chosen.get("scales", rng.choice(WIDE_LADDERS)),
        "full_grid": chosen.get("full_grid", rng.choice(WIDE_LATENTS)),
        "adaptive": chosen.get("adaptive", rng.choice((True, False))),
        "sigma_divisor": chosen.get("sigma_divisor", rng.choice((1.0, 1.03, 1.5, 0.7))),
        "delta": chosen.get("delta", rng.choice((0.01, 0.01, 0.005, 0.05, 0.2))),
    }
    preset = chosen.get("preset", rng.choice(speed_schedule.PRESET_NAMES))
    A, beta = chosen.get("spectrum", rng.choice(CUSTOM_SPECTRA))
    A, beta, ref_latent, ref_shift = speed_schedule.preset_params(preset, A, beta)
    kwargs.update(A=A, beta=beta, ref_latent=ref_latent, ref_shift=ref_shift)
    name, sigmas = chosen.get("schedule", rng.choice(pool))
    kwargs["sigmas"] = sigmas
    kwargs["_label"] = f"{preset} {name}"
    return kwargs


def _call_both(case: unittest.TestCase, pack_fn, ext_fn, kwargs: dict, canon=_canon_plan) -> tuple:
    kwargs = dict(kwargs)
    label = kwargs.pop("_label", "")
    sigmas = kwargs.get("sigmas")
    before = sigmas.clone() if torch.is_tensor(sigmas) else list(sigmas or ())
    pack_kwargs, ext_kwargs = dict(kwargs), dict(kwargs)
    if torch.is_tensor(sigmas):
        pack_kwargs["sigmas"], ext_kwargs["sigmas"] = sigmas.clone(), sigmas.clone()
    got = _outcome(pack_fn, canon, **pack_kwargs)
    want = _outcome(ext_fn, canon, **ext_kwargs)
    if torch.is_tensor(sigmas):
        case.assertTrue(torch.equal(pack_kwargs["sigmas"], before), "the pack changed the caller's sigmas")
    shown = {k: v for k, v in kwargs.items() if k != "sigmas"}
    case.assertEqual(got, want, f"{label} {shown}")
    return got


@requires_extension
class ScheduleConstantsTests(unittest.TestCase):
    """Presets, names and limits are the extension's literals (AST — no torch, no import)."""

    def test_constants(self):
        for name in ("MODES", "THRESHOLDS", "TRANSFORMS", "REFERENCE_LATENT", "MAX_COARSE_FRACTION",
                     "RESPACE_PATCH", "CUSTOM_DEFAULT_A", "CUSTOM_DEFAULT_BETA", "PRESETS", "PRESET_NAMES"):
            with self.subTest(name=name):
                want = _ext_literal("sam3ext/speed/schedule.py", name)
                got = getattr(speed_schedule, name)
                self.assertEqual(got, want)
                self.assertEqual(type(got), type(want))

    def test_preset_names_cover_the_table(self):
        self.assertEqual(set(speed_schedule.PRESET_NAMES), set(speed_schedule.PRESETS))


@requires_torch
@requires_extension
class ScheduleParityTests(unittest.TestCase):
    """``speed_schedule`` / scalar ``speed_spectral`` == ``sam3ext.speed.schedule`` / ``.spectral``."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.ext = _ext_speed("schedule")
        cls.ext_spectral = _ext_speed("spectral")
        cls.pool = _schedule_pool()

    # -- scalar maths ------------------------------------------------------------------------------

    def test_scalar_maths(self):
        rng = random.Random(7)
        omegas = [0.5, 1.0, 3.0, 16.0, 32.0, 52.0, 64.0, 1e-3, 1e5] + [rng.uniform(0.1, 200) for _ in range(40)]
        spectra = [(8664.998524, 2.422687), (203.615097, 1.915461), (1.0, 0.0), (1e-12, 2.0), (100.0, 400.0),
                   (100.0, -3.0), (0.0, 2.0), (-1.0, 2.0)]
        deltas = [0.0, 1e-4, 0.005, 0.01, 0.2, 0.999999, 1.0, 1.5, -0.1]
        times = [-0.1, 0.0, 1e-9, 0.25, 0.5, 0.932, 0.96, 0.999999, 1.0, 1.2]
        for omega in omegas:
            for A, beta in spectra:
                with self.subTest(fn="power_spectrum", omega=omega, A=A, beta=beta):
                    self.assertEqual(_outcome(speed_schedule.power_spectrum, _canon, omega, A, beta),
                                     _outcome(self.ext.power_spectrum, _canon, omega, A, beta))
                for t in times:
                    with self.subTest(fn="equivalent_delta", omega=omega, A=A, beta=beta, t=t):
                        self.assertEqual(_outcome(speed_schedule.equivalent_delta, _canon, t, A, beta, omega),
                                         _outcome(self.ext.equivalent_delta, _canon, t, A, beta, omega))
        for P in [1e-30, 1e-3, 0.5, 1.0, 7.0, 1e3, 1e12, 0.0, -0.5]:
            for delta in deltas:
                with self.subTest(fn="activation_time", P=P, delta=delta):
                    self.assertEqual(_outcome(speed_schedule.activation_time, _canon, P, delta),
                                     _outcome(self.ext.activation_time, _canon, P, delta))
        for t in times:
            for shift in (3.0, 3.158193, 8.0, 0.5, 0.0, -1.0):
                with self.subTest(fn="reference_coarse_fraction", t=t, shift=shift):
                    self.assertEqual(_outcome(speed_schedule.reference_coarse_fraction, _canon, t, shift),
                                     _outcome(self.ext.reference_coarse_fraction, _canon, t, shift))
            for r in (2.0, 4.0, 1.5, 4.0 / 3.0, 1.0 / 0.3, 1.0 / 0.75, 1.0000001):
                for fn in ("kappa", "align_timestep"):
                    with self.subTest(fn=fn, t=t, r=r):
                        self.assertEqual(_outcome(getattr(speed_spectral, fn), _canon, t, r),
                                         _outcome(getattr(self.ext_spectral, fn), _canon, t, r))
                with self.subTest(fn="respace_alignment", t=t, r=r):
                    self.assertEqual(_outcome(speed_spectral.respace_alignment, _canon, t, r),
                                     _outcome(self.ext_spectral.respace_alignment, _canon, t, r))
        for value in [0.0, 0.4, 1.0, 1.5, 2.5, 3.0, 3.5, 51.9, 52.0, 62.5, 63.0, 76.0, 125 * 0.3, 97 * 0.75, 1e6]:
            for multiple in (1, 2, 4):
                with self.subTest(fn="snap", value=value, multiple=multiple):
                    self.assertEqual(_outcome(speed_spectral.snap, _canon, value, multiple),
                                     _outcome(self.ext_spectral.snap, _canon, value, multiple))

    def test_validate_and_delta_optimal_transitions(self):
        ladders = list(WIDE_LADDERS) + [(), (1.0,), (0.5,), (0.5, 0.5, 1.0), (0.6, 0.4, 1.0), (0.0, 1.0), (-0.5, 1.0),
                                        (0.5, 1.1), (0.5, 1.000001), (0.5, 0.9999989)]
        for scales in ladders:
            with self.subTest(fn="validate_scales", scales=scales):
                self.assertEqual(_outcome(speed_schedule.validate_scales, _canon, scales),
                                 _outcome(self.ext.validate_scales, _canon, scales))
            for H, W in WIDE_LATENTS:
                for delta in (0.01, 0.2, 1.0):
                    for A, beta in ((8664.998524, 2.422687), (520.495848, 1.915461), (100.0, 1e6), (0.0, 2.0)):
                        args = (scales, delta, A, beta, H, W)
                        with self.subTest(fn="delta_optimal_transitions", args=args):
                            self.assertEqual(_outcome(speed_schedule.delta_optimal_transitions, _canon, *args),
                                             _outcome(self.ext.delta_optimal_transitions, _canon, *args))

    def test_text_parsing(self):
        texts = ["0.5,1.0", "0.5, 1.0", "0.5;1.0", "1/2,1", "1/4; 1/2; 1", "0.25,0.5,1", " 0.5 ,, 1.0 ,", "1.0",
                 "0.9999999", "0.5,0.9999999", "1", "", None, "  ", "0.7", "0.9,0.6", "0.6,0.9", "0.7,0.7", "1/0",
                 "1/2/3", "x", "0.5,nan", "inf", "0.5,1e400", "-0.5,1", "0,1", "1/3,2/3,1", "0.5;;1", "0.5,1.0,",
                 [0.5, 1.0], (0.25, "0.5", 1), [], ["", "0.5", " ", 1], [True, 1.0], ["1/2", "1"], [float("nan")],
                 0.5, 1, "0.95,0.8,0.6", "0.0", "1.0,0.5", "0.3", "-0.2", "0.5 1.0",
                 # zero denominators (v0.33.0: a ValueError naming the token) and their neighbours
                 "0/0", "1/0.0", "-1/0", "1/-0", "0.5,1/0", " 1/0 ;1", "inf/0", "x/0", "1/0/0", "0.7/0", ["1/0"],
                 ("0.5", "1/0"), [0.9, "2/0"]]
        for text in texts:
            with self.subTest(fn="parse_scales", text=text):
                self.assertEqual(_outcome(speed_schedule.parse_scales, _canon, text),
                                 _outcome(self.ext.parse_scales, _canon, text))
            for decreasing in (True, False):
                with self.subTest(fn="parse_sigmas", text=text, decreasing=decreasing):
                    self.assertEqual(
                        _outcome(speed_schedule.parse_sigmas, _canon, text, strictly_decreasing=decreasing),
                        _outcome(self.ext.parse_sigmas, _canon, text, strictly_decreasing=decreasing))
        # v0.33.0: the first zero-denominator token is named; the rest of the text is not read
        for text, token in (("1/0", "1/0"), (" 0.5 , 1/0 ", "1/0"), ("0/0;1/0", "0/0"), (["1/-0"], "1/-0"),
                            ("0.9,0.7/0", "0.7/0")):
            for fn, prefix in (("parse_scales", "scales"), ("parse_sigmas", "manual sigmas")):
                want = ("raise", "ValueError", True, f"{prefix}: division by zero: {token!r}", None, None)
                with self.subTest(fn=fn, zero_denominator=text):
                    self.assertEqual(_outcome(getattr(speed_schedule, fn), _canon, text), want)
                    self.assertEqual(_outcome(getattr(self.ext, fn), _canon, text), want)

    def test_preset_params(self):
        names = list(speed_schedule.PRESET_NAMES) + ["unknown", "", None, "ANIMA"]
        for name in names:
            for A, beta in ((203.615097, 1.915461), (1.0, 2.0), ("3", "4"), (7, 2)):
                with self.subTest(name=name, A=A, beta=beta):
                    self.assertEqual(_outcome(speed_schedule.preset_params, _canon, name, A, beta),
                                     _outcome(self.ext.preset_params, _canon, name, A, beta))
            with self.subTest(name=name, defaults=True):
                self.assertEqual(_outcome(speed_schedule.preset_params, _canon, name),
                                 _outcome(self.ext.preset_params, _canon, name))

    # -- thresholds and lookups --------------------------------------------------------------------

    def test_delta_threshold_plan(self):
        def canon(tp):
            return (type(tp).__name__, _canon(tp.values), _canon(tp.caps), _canon(tp.notes))

        for preset in speed_schedule.PRESET_NAMES:
            for A_beta in ((203.615097, 1.915461), (1e-12, 2.0), (100.0, 1e6)):
                A, beta, ref_latent, ref_shift = speed_schedule.preset_params(preset, *A_beta)
                for scales in WIDE_LADDERS + ((1.0,),):
                    for H, W in WIDE_LATENTS:
                        for adaptive in (False, True):
                            for divisor in (1.0, 1.03, 1.5, 0.7):
                                for n_steps in (1, 8, 28, 50):
                                    args = (n_steps, scales, 0.01, A, beta, H, W)
                                    kwargs = dict(adaptive=adaptive, ref_latent=ref_latent, ref_shift=ref_shift,
                                                  sigma_divisor=divisor)
                                    got = _outcome(speed_schedule.delta_threshold_plan, canon, *args, **kwargs)
                                    want = _outcome(self.ext.delta_threshold_plan, canon, *args, **kwargs)
                                    if got != want:
                                        self.assertEqual(got, want, f"{preset} {A_beta} {args} {kwargs}")

    def test_step_lookups(self):
        rng = random.Random(11)
        for index in range(1500):
            name, sigmas = rng.choice(self.pool)
            scales = rng.choice(WIDE_LADDERS)
            n = len(scales) - 1
            values = [rng.choice((0.99, 0.96, 0.93, 0.9, 0.8, 0.7, 0.5, 0.2, 0.0, 1.0, 1.2)) for _ in range(n)]
            caps = rng.choice((None, [None] * n, [rng.choice((None, 0, 1, 3, 5, 12, 60)) for _ in range(n)]))
            with self.subTest(index=index, schedule=name, scales=scales, values=values, caps=caps):
                got = _outcome(speed_schedule.static_transitions, _canon, sigmas, scales, values, caps)
                want = _outcome(self.ext.static_transitions, _canon, sigmas, scales, values, caps)
                self.assertEqual(got, want)
                copy_pack = sigmas.clone() if torch.is_tensor(sigmas) else list(sigmas)
                copy_ext = sigmas.clone() if torch.is_tensor(sigmas) else list(sigmas)
                got = speed_schedule.respace_transitions(copy_pack, scales, values, caps)
                want = self.ext.respace_transitions(copy_ext, scales, values, caps)
                self.assertEqual(_canon(list(got[:2])), _canon(list(want[:2])))
                self.assertEqual(got[2].dtype, want[2].dtype)
                self.assertTrue(torch.equal(got[2], want[2]))
                if torch.is_tensor(sigmas):
                    self.assertTrue(torch.equal(copy_pack, sigmas))

    def test_stage_grids(self):
        ladders = list(WIDE_LADDERS) + [(1.0,), (0.33, 0.66, 1.0), (0.01, 1.0), (0.9, 0.95, 1.0)]
        for mode in speed_schedule.MODES + ("other",):
            for scales in ladders:
                for H, W in WIDE_LATENTS:
                    with self.subTest(mode=mode, scales=scales, grid=(H, W)):
                        self.assertEqual(_outcome(speed_schedule.stage_grids, _canon, mode, scales, H, W),
                                         _outcome(self.ext.stage_grids, _canon, mode, scales, H, W))

    # -- the plan ----------------------------------------------------------------------------------

    def test_build_plan_core_grid(self):
        """Every mode x threshold x preset x adaptive x divisor x ladder x latent, two schedules each."""
        rng = random.Random(20261004)
        count = outcomes = 0
        seen = set()
        for mode in speed_schedule.MODES:
            for threshold in ("neo_shift", "delta_optimal"):
                for preset in speed_schedule.PRESET_NAMES:
                    for adaptive in (True, False):
                        for divisor in (1.0, 1.03, 1.5):
                            for scales in LADDERS:
                                for grid in LATENTS:
                                    for _ in range(2):
                                        kwargs = _plan_kwargs(rng, self.pool, exhaustive=dict(
                                            mode=mode, threshold=threshold, preset=preset, adaptive=adaptive,
                                            sigma_divisor=divisor, scales=scales, full_grid=grid, transform="dct",
                                            delta=0.01, spectrum=(203.615097, 1.915461)))
                                        result = _call_both(self, speed_schedule.build_plan, self.ext.build_plan,
                                                            kwargs)
                                        seen.add(result[0] if result[0] == "ok" else (result[1], result[5]))
                                        count += 1
        self.assertEqual(count, 6144)
        # the grid exercises plans, skips and settings errors alike
        self.assertIn("ok", seen)
        self.assertIn(("PlanError", "skip"), seen)

    def test_build_plan_wide_random_grid(self):
        """Every other dimension too (transforms, deltas, custom spectra incl. the refused ones, odd grids)."""
        rng = random.Random(4242)
        kinds = {}
        for index in range(5000):
            kwargs = _plan_kwargs(rng, self.pool)
            with self.subTest(index=index):
                result = _call_both(self, speed_schedule.build_plan, self.ext.build_plan, kwargs)
                key = result[0] if result[0] == "ok" else (result[1], result[5])
                kinds[key] = kinds.get(key, 0) + 1
        self.assertGreater(kinds.get("ok", 0), 500, kinds)
        self.assertGreater(kinds.get(("PlanError", "invalid"), 0), 300, kinds)
        self.assertGreater(kinds.get(("PlanError", "skip"), 0), 300, kinds)

    def test_build_plan_custom_spectra(self):
        """sam-extra 0.32.1: A <= 0 and non-finite A/beta are settings errors — ordinary parity rows."""
        rng = random.Random(99)
        errors = set()
        for A, beta in CUSTOM_SPECTRA:
            for mode in speed_schedule.MODES:
                for threshold in ("neo_shift", "delta_optimal"):
                    for adaptive in (True, False):
                        kwargs = _plan_kwargs(rng, self.pool, exhaustive=dict(
                            mode=mode, threshold=threshold, preset="custom", spectrum=(A, beta), adaptive=adaptive,
                            scales=(0.5, 1.0), full_grid=(104, 152), transform="dct", delta=0.01,
                            schedule=("simple/28/s3", torch.tensor(_spacing("simple", 28, 3.0)))))
                        result = _call_both(self, speed_schedule.build_plan, self.ext.build_plan, kwargs)
                        if result[0] == "raise":
                            errors.add((A if math.isfinite(A) else str(A), result[5]))
        for A in (0.0, -1.0, -1e-9):
            self.assertIn((A, "invalid"), errors)

    def test_build_plan_manual(self):
        rng = random.Random(5)
        manuals = [(0.7,), (0.9, 0.6), (0.6, 0.9), (0.7, 0.7), (0.95, 0.8, 0.6), (1.0,), (0.0,), (), (0.5,),
                   (0.999,), (0.05,), ("0.7",), (0.9, 0.9, 0.9)]
        for mode in speed_schedule.MODES:
            for manual in manuals:
                for scales in WIDE_LADDERS:
                    for transform in speed_schedule.TRANSFORMS:
                        for _ in range(2):
                            kwargs = _plan_kwargs(rng, self.pool, exhaustive=dict(
                                mode=mode, threshold="manual", scales=scales, transform=transform))
                            kwargs["manual_sigmas"] = manual
                            with self.subTest(mode=mode, manual=manual, scales=scales, transform=transform):
                                _call_both(self, speed_schedule.build_plan, self.ext.build_plan, kwargs)

    def test_build_plan_refusals_and_odd_inputs(self):
        sigmas = torch.tensor(_spacing("simple", 28, 3.0))
        base = dict(mode="transition", transform="dct", sigmas=sigmas, full_grid=(104, 152), scales=(0.5, 1.0),
                    threshold="neo_shift", delta=0.01, A=8664.998524, beta=2.422687, ref_latent=128, ref_shift=3.0,
                    adaptive=True, sigma_divisor=1.03)
        rows = [
            {}, {"mode": "other"}, {"transform": "wavelet"}, {"threshold": "auto"}, {"scales": ()},
            {"scales": (1.0,)}, {"scales": (0.5, 0.5, 1.0)}, {"scales": (0.5, 0.9)}, {"scales": ("0.5", "1")},
            {"sigmas": torch.tensor([1.0])}, {"sigmas": torch.tensor([])}, {"sigmas": []},
            {"delta": 0.0}, {"delta": 1.0}, {"delta": -0.1}, {"delta": math.nan}, {"delta": "0.02"}, {"delta": "x"},
            {"delta": None}, {"sigma_divisor": 0.0}, {"sigma_divisor": -1.0}, {"sigma_divisor": math.nan},
            {"sigma_divisor": "1.2"}, {"sigma_divisor": 0.5}, {"sigma_divisor": 0.9},
            {"threshold": "delta_optimal", "sigma_divisor": 0.0}, {"A": 0.0}, {"A": -5}, {"A": "1e3"},
            {"A": math.nan}, {"beta": math.inf}, {"beta": "2"}, {"ref_shift": None}, {"ref_shift": 0.0},
            {"ref_latent": 64}, {"ref_latent": "128"}, {"adaptive": 0}, {"adaptive": "yes"},
            {"full_grid": (0, 152)}, {"full_grid": (1, 1)}, {"full_grid": ("104", "152")}, {"full_grid": (104.7, 152.2)},
            {"transform": "dwt"}, {"transform": "dwt", "full_grid": (104, 151)}, {"transform": "dwt", "scales": (0.25, 1.0)},
            {"transform": "dwt", "scales": (0.25, 0.5, 1.0), "full_grid": (128, 128)},
            {"transform": "dwt", "mode": "respace", "full_grid": (106, 150)},
            {"mode": "respace", "full_grid": (125, 97)}, {"mode": "respace", "full_grid": (3, 3), "scales": (0.1, 1.0)},
            {"threshold": "manual", "manual_sigmas": (0.7,)}, {"threshold": "manual", "manual_sigmas": [0.7, 0.3]},
            {"threshold": "manual", "manual_sigmas": (0.7,), "mode": "respace", "scales": (0.25, 0.5, 1.0)},
            {"threshold": "manual", "manual_sigmas": ("a",)}, {"threshold": "manual", "manual_sigmas": (math.nan,)},
            {"threshold": "manual", "manual_sigmas": (0.99,), "sigmas": sigmas[10:]},
            {"sigmas": sigmas[20:]}, {"sigmas": sigmas[25:]}, {"sigmas": sigmas.double()},
            {"sigmas": [float(s) for s in sigmas]}, {"sigmas": sigmas.flip(0)},
        ]
        for row in rows:
            kwargs = dict(base, **row)
            kwargs["_label"] = repr({k: (v if not torch.is_tensor(v) else f"tensor{tuple(v.shape)}") for k, v in row.items()})
            with self.subTest(row=kwargs["_label"]):
                _call_both(self, speed_schedule.build_plan, self.ext.build_plan, kwargs)

    def test_hand_off_sigmas_match_a_replay(self):
        """The plan's (sigma, aligned) are what the run sees: transition patches one step, respace re-spaces."""
        sigmas = torch.tensor(_spacing("beta", 28, 3.0))
        for mode in speed_schedule.MODES:
            plan = speed_schedule.build_plan(
                mode=mode, transform="dct", sigmas=sigmas, full_grid=(104, 152), scales=(0.25, 0.5, 1.0),
                threshold="manual", manual_sigmas=(0.95, 0.8))
            self.assertEqual(len(plan.transitions), 2)
            for t in plan.transitions:
                r = t.scale_to / t.scale_from
                expected = (speed_spectral.respace_alignment(t.sigma, r)[0] if mode == "respace"
                            else speed_spectral.align_timestep(t.sigma, r))
                self.assertEqual(t.aligned, float(torch.tensor(expected, dtype=sigmas.dtype)))



# --------------------------------------------------------------------------------------------------
# Spectral transforms (PB-SP2)
# --------------------------------------------------------------------------------------------------

SPECTRAL_SHAPES = ((2, 4, 13, 19), (1, 16, 26, 38), (1, 3, 1, 1), (1, 16, 1, 26, 38), (2, 4, 3, 13, 19), (1, 2, 8, 8))
SPECTRAL_DTYPES = ("float32", "float64", "bfloat16", "float16")


def _randn(shape, seed: int, dtype="float32"):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32).to(getattr(torch, dtype))


@requires_torch
@requires_extension
class SpectralParityTests(unittest.TestCase):
    """``speed_spectral`` == ``sam3ext.speed.spectral``: same dtype, shape and bits."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        cls.ext = _ext_speed("spectral")

    def _same(self, got, want, label=""):
        self.assertEqual(got.dtype, want.dtype, label)
        self.assertEqual(tuple(got.shape), tuple(want.shape), label)
        self.assertEqual(got.device, want.device, label)
        self.assertTrue(torch.equal(got, want), f"{label}: max diff {(got.double() - want.double()).abs().max()}")

    def _both(self, name, *args, label=""):
        """Run the pack and the extension function on clones of ``args``; inputs must stay unchanged."""
        def clones():
            return [a.clone() if torch.is_tensor(a) else a for a in args]

        pack_args, ext_args = clones(), clones()
        try:
            got = getattr(speed_spectral, name)(*pack_args)
        except Exception as exc:
            got = _canon_error(exc)
        try:
            want = getattr(self.ext, name)(*ext_args)
        except Exception as exc:
            want = _canon_error(exc)
        for before, after in zip(args, pack_args):
            if torch.is_tensor(before):
                self.assertTrue(torch.equal(before, after), f"{name} changed its input ({label})")
        if isinstance(want, tuple) or isinstance(got, tuple):
            self.assertEqual(got, want, f"{name} {label}")
        else:
            self._same(got, want, f"{name} {label}")
        return got

    def test_work_dtype(self):
        for device in ("cpu", "mps", "cuda", None, "cuda:1"):
            with self.subTest(device=device):
                self.assertEqual(speed_spectral.work_dtype(device), self.ext.work_dtype(device))
                if device is not None:
                    dev = torch.device(device)
                    self.assertEqual(speed_spectral.work_dtype(dev), self.ext.work_dtype(dev))

    def test_dct_basis(self):
        for n in list(range(1, 41)) + [52, 64, 76, 97, 104, 125, 128, 152]:
            for dtype in ("float64", "float32", "bfloat16"):
                with self.subTest(n=n, dtype=dtype):
                    got = speed_spectral.dct_basis(n, "cpu", getattr(torch, dtype))
                    want = self.ext.dct_basis(n, "cpu", getattr(torch, dtype))
                    self._same(got, want)
            with self.subTest(n=n, default=True):
                self._same(speed_spectral.dct_basis(n), self.ext.dct_basis(n))
        for n in (0, -3):
            with self.subTest(n=n):
                self.assertEqual(_outcome(speed_spectral.dct_basis, repr, n), _outcome(self.ext.dct_basis, repr, n))

    def test_basis_cache_is_shared_and_bounded(self):
        speed_spectral.clear_basis_cache()
        first = speed_spectral.dct_basis(11)
        self.assertIs(speed_spectral.dct_basis(11, "cpu", torch.float64), first)
        for n in range(20, 40):
            speed_spectral.dct_basis(n)
        self.assertLessEqual(len(speed_spectral._BASES), speed_spectral._BASES_LIMIT)
        self.assertIsNot(speed_spectral.dct_basis(11), first)       # evicted (least recently used)
        speed_spectral.clear_basis_cache()
        self.assertEqual(len(speed_spectral._BASES), 0)

    def test_dct_pair(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 100 + index, dtype)
                label = f"{shape} {dtype}"
                with self.subTest(label=label):
                    coeffs = self._both("dct2", x, label=label)
                    self._both("idct2", coeffs, label=label)
                    self._both("idct2", x, label=label)

    def test_dct_downscale(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            H, W = shape[-2], shape[-1]
            sizes = {(H, W), (max(1, H // 2), max(1, W // 2)), (max(1, round(H * 0.3)), max(1, round(W * 0.3))),
                     (1, 1), (H, max(1, W - 1)), (H + 1, W), (0, W), (H, -1)}
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 200 + index, dtype)
                for size in sorted(sizes):
                    label = f"{shape} {dtype} -> {size}"
                    with self.subTest(label=label):
                        self._both("dct_downscale", x, size, label=label)

    def test_amplitude_resize(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            H, W = shape[-2], shape[-1]
            sizes = {(H, W), (2 * H, 2 * W), (max(1, H // 2), max(1, W // 2)), (H + 3, max(1, W - 2)), (1, 1),
                     (3 * H, W)}
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 300 + index, dtype)
                for size in sorted(sizes):
                    label = f"{shape} {dtype} -> {size}"
                    with self.subTest(label=label):
                        self._both("amplitude_resize", x, size, label=label)

    @staticmethod
    def _targets(h, w):
        return sorted({(2 * h, 2 * w), (round(h / 0.3), round(w / 0.3)), (h + 1, w + 2), (h, w), (h + 7, w),
                       (max(1, h - 1), w)})

    def test_expand_dct(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            h, w = shape[-2], shape[-1]
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 400 + index, dtype)
                for target in self._targets(h, w):
                    noise = _randn(tuple(shape[:-2]) + target, 500 + index)
                    for sigma in (0.0, 0.3, 0.9335, 1.0):
                        label = f"{shape} {dtype} -> {target} sigma {sigma}"
                        with self.subTest(label=label):
                            self._both("expand_dct", x, target, sigma, noise, label=label)
                with self.subTest(label="noise of the wrong shape"):
                    bad = _randn(tuple(shape[:-2]) + (2 * h + 1, 2 * w), 9)
                    self._both("expand_dct", x, (2 * h, 2 * w), 0.5, bad, label="bad noise")
                    bad = _randn((7,) + tuple(shape[1:-2]) + (2 * h, 2 * w), 9)
                    self._both("expand_dct", x, (2 * h, 2 * w), 0.5, bad, label="bad lead")
                    noise64 = _randn(tuple(shape[:-2]) + (2 * h, 2 * w), 10, "float64")
                    self._both("expand_dct", x, (2 * h, 2 * w), 0.7, noise64, label="float64 noise")

    def test_expand_fft(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            h, w = shape[-2], shape[-1]
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 600 + index, dtype)
                for target in self._targets(h, w):
                    lead = tuple(shape[:-2]) + target
                    real, imag = _randn(lead, 700 + index), _randn(lead, 800 + index)
                    for sigma in (0.0, 0.5, 0.96):
                        label = f"{shape} {dtype} -> {target} sigma {sigma}"
                        with self.subTest(label=label):
                            self._both("expand_fft", x, target, sigma, real, imag, label=label)

    def test_expand_haar(self):
        for index, shape in enumerate(SPECTRAL_SHAPES):
            for dtype in SPECTRAL_DTYPES:
                x = _randn(shape, 900 + index, dtype)
                lh, hl, hh = (_randn(shape, 1000 + index + k) for k in range(3))
                for sigma in (0.0, 0.25, 0.97):
                    label = f"{shape} {dtype} sigma {sigma}"
                    with self.subTest(label=label):
                        self._both("expand_haar", x, sigma, lh, hl, hh, label=label)
                with self.subTest(label="float64 bands"):
                    self._both("expand_haar", x, 0.4, lh.double(), hl.double(), hh.double(), label="f64 bands")


# --------------------------------------------------------------------------------------------------
# The segmented run (PB-SP3)
# --------------------------------------------------------------------------------------------------
# Toy k-diffusion samplers in the k-diffusion style (crowsonkb/k-diffusion, MIT): Euler, Euler
# ancestral with an explicit noise sampler, DPM++ 2M (multi-step history, reset per segment because
# SPEED calls the sampler once per segment), and a probe naming the step-offset keyword. Both runners
# get the same functions, so only the runners are compared.


def _append_dims(value, ndim):
    return value[(...,) + (None,) * (ndim - value.ndim)]


class _ToyModel:
    """Deterministic denoiser on 4-D and 5-D latents; records every call (shape, dtype, sigma bits)."""

    def __init__(self, channels: int):
        gen = torch.Generator().manual_seed(1234)
        self.mix = torch.randn(channels, channels, generator=gen) * 0.1
        self.calls = []

    def __call__(self, x, sigma, **extra):
        self.calls.append((tuple(x.shape), str(x.dtype), tuple(float(s).hex() for s in sigma.flatten()),
                           tuple(sorted(extra))))
        work = x.float()
        s = _append_dims(sigma.float(), x.ndim)
        mixed = torch.einsum("ij,bj...->bi...", self.mix, work)
        smooth = (work.roll(1, -1) + work.roll(-1, -1) + work.roll(1, -2) + work.roll(-1, -2)) * 0.25
        return (work * (1.0 / (1.0 + s)) + 0.2 * smooth * s + 0.05 * mixed).to(x.dtype)


def _to_d(x, sigma, denoised):
    return (x - denoised) / _append_dims(sigma, x.ndim)


def _sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = _to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = x + d * (sigmas[i + 1] - sigmas[i])
    return x


def _sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0,
                            noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:       # the sampler's own default: a fixed generator on this grid
        gen = torch.Generator().manual_seed(4000 + int(x.shape[-1]))
        noise_sampler = lambda sigma, sigma_next: torch.randn(tuple(x.shape), generator=gen).to(x.dtype)  # noqa: E731
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_from, sigma_to = sigmas[i], sigmas[i + 1]
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        d = _to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = x + d * (sigma_down - sigmas[i])
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


def _sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    def t_fn(sigma):
        return sigma.log().neg()

    def sigma_fn(t):
        return t.neg().exp()

    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            r = (t - t_fn(sigmas[i - 1])) / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


def _offset_probe(log: list):
    def sample_offset_probe(model, x, sigmas, extra_args=None, callback=None, disable=None, sam_extra_step_offset=None):
        log.append(sam_extra_step_offset)
        return _sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

    return sample_offset_probe


def _make_sampler(name: str):
    log: list = []
    if name == "offset":
        return _offset_probe(log), log
    return {"euler": _sample_euler, "euler_ancestral": _sample_euler_ancestral,
            "dpmpp_2m": _sample_dpmpp_2m}[name], log


def _full_grid_noise(x):
    gen = torch.Generator().manual_seed(31337)
    shape = tuple(x.shape)
    return lambda sigma, sigma_next: torch.randn(shape, generator=gen).to(x.dtype)


def _recording_host(base_cls, *, none_stages=()):
    class RecordingHost(base_cls):
        def __init__(self):
            self.published, self.restored, self.coarse, self.scopes = [], 0, [], []

        def noise_scope(self, full_shape):
            self.scopes.append(tuple(full_shape))
            return super().noise_scope(full_shape)

        def coarse_noise_sampler(self, template, x, stage, sigmas):
            self.coarse.append((stage, tuple(x.shape), template is not None, len(sigmas)))
            if stage in none_stages:
                return None
            gen = torch.Generator().manual_seed(500 + stage)
            shape, dtype = tuple(x.shape), x.dtype
            return lambda sigma, sigma_next: torch.randn(shape, generator=gen).to(dtype)

        def publish_sigmas(self, sigmas):
            self.published.append((sigmas.clone(), str(sigmas.dtype)))

        def restore_sigmas(self):
            self.restored += 1

    return RecordingHost()


def _canon_report(report) -> tuple:
    events = tuple(tuple(sorted((key, _canon(value) if isinstance(value, (float, tuple)) else value)
                                for key, value in event.items())) for event in report.events)
    return (report.applied, report.coarse_steps, report.total_steps, events, report.init)


RUN_LATENTS = ((2, 4, 32, 32), (1, 4, 26, 38), (1, 4, 25, 19), (1, 4, 1, 26, 38), (2, 4, 2, 16, 16))
RUN_LADDERS = (
    dict(scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,)),
    dict(scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.95, 0.8)),
    dict(scales=(0.5, 1.0), threshold="neo_shift", A=8664.998524, beta=2.422687, ref_shift=3.0, sigma_divisor=1.03),
    dict(scales=(0.3, 1.0), threshold="delta_optimal", A=1963.48906, beta=1.915461, delta=0.05, adaptive=False),
    dict(scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.85,)),
)


@requires_torch
@requires_extension
class RunnerParityTests(unittest.TestCase):
    """``speed_runner.run_speed`` == ``sam3ext.speed.runner.run_speed`` on a toy model, bit for bit."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_runner
        cls.pack = speed_runner
        cls.ext = _ext_speed("runner")
        cls.ext_schedule = _ext_speed("schedule")

    # -- constants and draws -------------------------------------------------------------------------

    def test_constants(self):
        for name in ("EXPANSION_SEED_STRIDE", "RESPACE_SEED_OFFSET", "INIT_NOISE_SEED_OFFSET", "STEP_OFFSET_KWARG"):
            with self.subTest(name=name):
                self.assertEqual(getattr(self.pack, name), _ext_literal("sam3ext/speed/runner.py", name))
                self.assertEqual(getattr(self.pack, name), getattr(self.ext, name))

    def test_numpy_and_torch_draws(self):
        seeds = (0, 5, 11, -1, -5, -(2 ** 63), 2 ** 31, 2 ** 63, 2 ** 63 + 5, 2 ** 64 - 1, 2 ** 64 + 3)
        layouts = ((4,), (16,), (4, 3), (2, 1), (1,))
        shape_sets = ([(13, 19)], [(8, 8)] * 3, [(26, 38)] * 2, [(1, 1)])
        for seed in seeds:
            for lead in layouts:
                for shapes in shape_sets:
                    label = f"seed {seed} lead {lead} shapes {shapes}"
                    with self.subTest(label=label, fn="numpy"):
                        self._same_draws(lambda m: m._numpy_draws(seed, lead, shapes), label)
                    with self.subTest(label=label, fn="torch"):
                        def torch_draws(module):
                            gen = torch.Generator().manual_seed(seed + module.RESPACE_SEED_OFFSET)
                            first = module._torch_draws(gen, lead, shapes, torch.device("cpu"))
                            return first + module._torch_draws(gen, lead, shapes, "cpu")   # the stream continues
                        self._same_draws(torch_draws, label)

    def _same_draws(self, draw, label):
        outcomes = []
        for module in (self.pack, self.ext):
            try:
                outcomes.append(("ok", draw(module)))
            except Exception as exc:
                outcomes.append(_canon_error(exc))
        got, want = outcomes
        if got[0] != "ok" or want[0] != "ok":
            self.assertEqual(got, want, label)
            return
        self.assertEqual(len(got[1]), len(want[1]), label)
        for a, b in zip(got[1], want[1]):
            self.assertEqual((a.dtype, tuple(a.shape), a.is_contiguous()), (b.dtype, tuple(b.shape), b.is_contiguous()),
                             label)
            self.assertTrue(torch.equal(a, b), label)

    def test_draw_shapes(self):
        for transform in ("dct", "fft", "dwt", "other"):
            with self.subTest(transform=transform):
                self.assertEqual(self.pack._draw_shapes(transform, (13, 19), (26, 38)),
                                 self.ext._draw_shapes(transform, (13, 19), (26, 38)))

    def test_init_coarse(self):
        sigma0s = (1.0, 1.0 - 1e-7, 0.999, 0.95, 0.6, 0.0)
        for shape in ((2, 4, 32, 32), (1, 16, 26, 38), (1, 4, 1, 25, 19)):
            for scales in ((0.5, 1.0), (0.25, 0.5, 1.0), (1.0,)):
                for mode in speed_schedule.MODES:
                    grids = speed_schedule.stage_grids(mode, scales, shape[-2], shape[-1])
                    for init_latent in (False, True):
                        for rescale in (True, False):
                            for sigma0 in sigma0s:
                                for seeds in ([3] * shape[0], [-7] * shape[0], [2 ** 40 + i for i in range(shape[0])]):
                                    for dtype in ("float32", "bfloat16"):
                                        x = _randn(shape, 77, dtype)
                                        label = f"{shape} {scales} {mode} init={init_latent} rescale={rescale} s0={sigma0} {seeds} {dtype}"
                                        with self.subTest(label=label):
                                            self._same_init(x, mode, grids, init_latent, rescale, sigma0, seeds, label)

    def _same_init(self, x, mode, grids, init_latent, rescale, sigma0, seeds, label):
        results = []
        for module, schedule in ((self.pack, speed_schedule), (self.ext, self.ext_schedule)):
            plan = schedule.Plan(mode=mode, transform="dct", full_grid=tuple(x.shape[-2:]), first_scale=0.5,
                                 first_grid=grids[0], transitions=[], thresholds=(), n_steps=1)
            run = module.SpeedRun(plan=plan, seeds=list(seeds), init_latent=init_latent, img2img_rescale=rescale)
            report = module.RunReport()
            source = x.clone()
            out = module._init_coarse(source, run, sigma0, report)
            self.assertTrue(torch.equal(source, x), label)
            results.append((out, report.init, out is source))
        (got, got_init, got_same), (want, want_init, want_same) = results
        self.assertEqual((got_init, got_same, got.dtype, tuple(got.shape)), (want_init, want_same, want.dtype,
                                                                              tuple(want.shape)), label)
        self.assertTrue(torch.equal(got, want), label)

    # -- whole runs ------------------------------------------------------------------------------------

    def _plans(self, ladder: dict, mode: str, transform: str, sigmas, grid):
        kwargs = dict(ladder, mode=mode, transform=transform, sigmas=sigmas, full_grid=grid)
        try:
            pack_plan = speed_schedule.build_plan(**kwargs)
        except speed_schedule.PlanError as exc:
            with self.assertRaises(self.ext_schedule.PlanError) as ctx:
                self.ext_schedule.build_plan(**kwargs)
            self.assertEqual((ctx.exception.reason, ctx.exception.kind), (exc.reason, exc.kind))
            return None
        ext_plan = self.ext_schedule.build_plan(**kwargs)
        self.assertEqual(_canon_plan(pack_plan), _canon_plan(ext_plan))
        return pack_plan, ext_plan

    def _run_one(self, module, plan, x, sigmas, seeds, sampler, *, init_latent, rescale, none_stages, extra_kwargs,
                 base_fn=None):
        model = _ToyModel(x.shape[1])
        host = _recording_host(module.SpeedHost, none_stages=none_stages)
        steps: list = []
        fn, probe = _make_sampler(sampler)
        if base_fn is not None:
            fn = base_fn
        sampler_kwargs = dict(extra_kwargs)
        if sampler == "euler_ancestral":
            sampler_kwargs["noise_sampler"] = _full_grid_noise(x)
        source, schedule = x.clone(), sigmas.clone()
        try:
            out, report = module.run_speed(
                fn, model, source, schedule, run=module.SpeedRun(plan=plan, seeds=list(seeds), init_latent=init_latent,
                                                                  img2img_rescale=rescale),
                host=host, extra_args={"seed": 5}, callback=lambda d: steps.append((d["i"], tuple(d["x"].shape))),
                disable=True, sampler_kwargs=sampler_kwargs)
            result = ("ok", out, _canon_report(report), report.summary(plan))
        except Exception as exc:
            result = _canon_error(exc)
        self.assertTrue(torch.equal(source, x) and torch.equal(schedule, sigmas), "the run changed its inputs")
        return result, model.calls, host, steps, probe

    def _compare_runs(self, plans, x, sigmas, seeds, sampler, *, init_latent=False, rescale=True, none_stages=(),
                      extra_kwargs=None, base_fn=None, label=""):
        pack_plan, ext_plan = plans
        got = self._run_one(self.pack, pack_plan, x, sigmas, seeds, sampler, init_latent=init_latent, rescale=rescale,
                            none_stages=none_stages, extra_kwargs=extra_kwargs or {}, base_fn=base_fn)
        want = self._run_one(self.ext, ext_plan, x, sigmas, seeds, sampler, init_latent=init_latent, rescale=rescale,
                             none_stages=none_stages, extra_kwargs=extra_kwargs or {}, base_fn=base_fn)
        (g_result, g_calls, g_host, g_steps, g_probe), (w_result, w_calls, w_host, w_steps, w_probe) = got, want
        if g_result[0] != "ok" or w_result[0] != "ok":
            self.assertEqual(g_result, w_result, label)
        else:
            self.assertEqual(g_result[2:], w_result[2:], label)
            self.assertEqual((g_result[1].dtype, tuple(g_result[1].shape)), (w_result[1].dtype, tuple(w_result[1].shape)))
            self.assertTrue(torch.equal(g_result[1], w_result[1]),
                            f"{label}: max diff {(g_result[1].double() - w_result[1].double()).abs().max()}")
        self.assertEqual(g_calls, w_calls, label)
        self.assertEqual(g_steps, w_steps, label)
        self.assertEqual(g_probe, w_probe, label)
        self.assertEqual((g_host.restored, g_host.coarse, g_host.scopes), (w_host.restored, w_host.coarse, w_host.scopes),
                         label)
        self.assertEqual(len(g_host.published), len(w_host.published), label)
        for (a, a_dtype), (b, b_dtype) in zip(g_host.published, w_host.published):
            self.assertEqual(a_dtype, b_dtype, label)
            self.assertTrue(torch.equal(a, b), label)
        return g_result

    def test_runs_over_modes_transforms_samplers_and_latents(self):
        rng = random.Random(3)
        txt2img = torch.tensor(_spacing("simple", 28, 3.0))
        applied = refused = 0
        for mode in speed_schedule.MODES:
            for transform in speed_schedule.TRANSFORMS:
                for sampler in ("euler", "euler_ancestral", "dpmpp_2m", "offset"):
                    for shape in RUN_LATENTS:
                        ladder = rng.choice(RUN_LADDERS)
                        if transform == "dwt":
                            ladder = rng.choice((RUN_LADDERS[0], RUN_LADDERS[1], RUN_LADDERS[4]))
                        plans = self._plans(ladder, mode, transform, txt2img, tuple(shape[-2:]))
                        if plans is None:
                            refused += 1
                            continue
                        seeds = rng.choice(([11 + i for i in range(shape[0])], [42] * shape[0],
                                            [-5 - i for i in range(shape[0])], [2 ** 63 + 7 + i for i in range(shape[0])]))
                        x = _randn(shape, 1 + len(seeds) + shape[-1])
                        label = f"{mode} {transform} {sampler} {shape} {ladder} {seeds}"
                        with self.subTest(label=label):
                            result = self._compare_runs(plans, x, txt2img, seeds, sampler, label=label)
                            applied += result[0] == "ok"
        self.assertGreater(applied, 80)
        self.assertGreater(refused, 0)      # dwt on odd grids is refused by both plans

    def test_img2img_and_hires_starts(self):
        """sigma0 < 1 with an init latent: rescale on/off, txt2img-like start (sigma0 = 1) unchanged."""
        longer = _spacing("beta", int(20 / 0.6), 3.0)
        tail = torch.tensor(longer[-21:])
        full = torch.tensor(_spacing("beta", 20, 3.0))
        for mode in speed_schedule.MODES:
            for transform in ("dct", "fft"):
                for sampler in ("euler", "dpmpp_2m", "euler_ancestral"):
                    for shape in ((2, 4, 32, 32), (1, 4, 1, 26, 38)):
                        for sigmas, init_latent in ((tail, True), (full, True), (tail, False)):
                            ladder = dict(scales=(0.5, 1.0), threshold="manual", manual_sigmas=(float(sigmas[3]) - 1e-4,))
                            plans = self._plans(ladder, mode, transform, sigmas, tuple(shape[-2:]))
                            self.assertIsNotNone(plans)
                            for rescale in (True, False):
                                x = _randn(shape, 90 + shape[0])
                                label = f"{mode} {transform} {sampler} {shape} s0={float(sigmas[0]):.4f} init={init_latent} rescale={rescale}"
                                with self.subTest(label=label):
                                    self._compare_runs(plans, x, sigmas, [21 + i for i in range(shape[0])], sampler,
                                                       init_latent=init_latent, rescale=rescale, label=label)

    def test_batch_images_and_fixed_spectral_seed(self):
        """Per-seed draws: a batch of two equals the two seeds run alone (pack), and both runners agree."""
        sigmas = torch.tensor(_spacing("simple", 20, 3.0))
        for mode in speed_schedule.MODES:
            for transform in speed_schedule.TRANSFORMS:
                ladder = dict(scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.95, 0.8))
                plans = self._plans(ladder, mode, transform, sigmas, (32, 32))
                x = _randn((2, 4, 32, 32), 5)
                for seeds in ([11, 12], [42, 42]):
                    label = f"{mode} {transform} {seeds}"
                    with self.subTest(label=label):
                        batch = self._compare_runs(plans, x, sigmas, seeds, "euler", label=label)
                        alone = [self._compare_runs(plans, x[i:i + 1], sigmas, [seed], "euler", label=label)
                                 for i, seed in enumerate(seeds)]
                        self.assertTrue(torch.equal(batch[1], torch.cat([a[1] for a in alone])), label)

    def test_noise_sampler_routing_and_dtypes(self):
        """The full-grid noise sampler reaches only the tail; coarse stages ask the host (None = default)."""
        sigmas = torch.tensor(_spacing("simple", 24, 3.0))
        ladder = dict(scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.95, 0.8))
        for mode in speed_schedule.MODES:
            plans = self._plans(ladder, mode, "dct", sigmas, (32, 32))
            for none_stages in ((), (0,), (1,), (0, 1)):
                for dtype in ("float32", "bfloat16", "float16"):
                    x = _randn((1, 4, 32, 32), 8, dtype)
                    label = f"{mode} none={none_stages} {dtype}"
                    with self.subTest(label=label):
                        self._compare_runs(plans, x, sigmas, [3], "euler_ancestral", none_stages=none_stages,
                                           label=label, extra_kwargs={"eta": 0.7, "s_noise": 0.9})
            with self.subTest(mode=mode, schedule="float64"):
                self._compare_runs(plans, _randn((1, 4, 32, 32), 9), sigmas.double(), [3], "euler", label="f64 sigmas")

    def test_step_offset_follows_the_pack_wrapper_to_the_sampler(self):
        """The pack reads the keyword through ``_forge_neo_base`` (CNS); the offsets are each segment's start."""
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_common

        log: list = []
        probe = _offset_probe(log)

        def wrapper(model, x, sigmas, **kwargs):          # CNS-like: no __wrapped__, forwards everything
            return probe(model, x, sigmas, **kwargs)

        setattr(wrapper, guidance_common.SAMPLER_BASE_ATTR, probe)
        sigmas = torch.tensor(_spacing("simple", 20, 3.0))
        plan = speed_schedule.build_plan(mode="transition", transform="dct", sigmas=sigmas, full_grid=(16, 16),
                                         scales=(0.25, 0.5, 1.0), threshold="manual", manual_sigmas=(0.95, 0.8))
        self.pack.run_speed(wrapper, _ToyModel(4), _randn((1, 4, 16, 16), 1), sigmas,
                            run=self.pack.SpeedRun(plan=plan, seeds=[1]))
        self.assertEqual(log, [0] + [t.step for t in plan.transitions])
        log.clear()
        self.pack.run_speed(_sample_euler, _ToyModel(4), _randn((1, 4, 16, 16), 1), sigmas,
                            run=self.pack.SpeedRun(plan=plan, seeds=[1]))
        self.assertEqual(log, [])

    def test_refusals_and_failures(self):
        sigmas = torch.tensor(_spacing("simple", 20, 3.0))
        ladder = dict(scales=(0.5, 1.0), threshold="manual", manual_sigmas=(0.8,))
        for mode in speed_schedule.MODES:
            plans = self._plans(ladder, mode, "dct", sigmas, (16, 16))

            def boom(model, x, sigmas, extra_args=None, callback=None, disable=None):
                raise ZeroDivisionError("sampler failed")

            def shrink(model, x, sigmas, extra_args=None, callback=None, disable=None):
                return x[..., :-1, :]

            rows = [
                ("3-D latent", _randn((4, 16, 16), 1), [1], None),
                ("plan grid", _randn((1, 4, 16, 18), 1), [1], None),
                ("seed count", _randn((2, 4, 16, 16), 1), [1], None),
                ("sampler raises", _randn((1, 4, 16, 16), 1), [1], boom),
                ("sampler shrinks", _randn((1, 4, 16, 16), 1), [1], shrink),
            ]
            for name, x, seeds, base_fn in rows:
                with self.subTest(mode=mode, row=name):
                    result = self._compare_runs(plans, x, sigmas, seeds, "euler", base_fn=base_fn, label=name)
                    self.assertEqual(result[0], "raise")

# --------------------------------------------------------------------------------------------------
# ComfyUI glue (speed_nodes): settings, plan mapping and status wording vs the extension's Forge glue
# --------------------------------------------------------------------------------------------------

SETTINGS_VALUES = {
    "enabled": (True, False, "on", "Disabled", 0, 1, 2.5, None, "maybe", "", "TRUE", " yes "),
    "mode": ("transition", "respace", "Respace ", "x", None, "", 3),
    "threshold": ("neo_shift", "delta_optimal", "manual", "MANUAL", "bad", None),
    "preset": speed_schedule.PRESET_NAMES + ("Anima", "", None, "nope", 0, " flux "),
    "scales": ("0.5,1.0", "1/4; 1/2; 1", "0.25,0.5,1", [0.5, 1.0], ("0.3", "1"), "1.0", "0.5", "1,0.5", "0.5,,1",
               "x,1", "0,1", "", None, "1/0,1", "inf,1", "0.9999999", 1.0, "0.5,1.0000001", "0.5,1/0", "0/0,1"),
    "delta": (0.01, 0, 1, 0.5, "0.2", "nan", "x", None, -0.1, 1e-9, "inf", 10 ** 400, True),
    "sigma_divisor": (1.03, 0, -1, "2", None, "x", float("nan"), 1e308, 10 ** 400),
    "manual_sigmas": ("0.7", "0.9,0.8", "0.8,0.9", "0.5;0.3", "1.0", "0", "", None, "x", [0.9, 0.5], "0.9,0.9",
                      ("0.6", 0.2), "0.7/0", "0.9,1/0"),
    "adaptive": (True, False, "false", "0", None, "maybe"),
    "transform": ("dct", "dwt", "fft", "DWT", "x", None),
    "spectrum_A": (203.6, 0, -1, "1e3", "x", None, float("inf"), -10 ** 400),
    "spectrum_beta": (1.9, "2", "nan", None, -3, 10 ** 400),
    "seed": (-1, 0, 5, 3.7, "12", "", "x", None, -5, 2 ** 63, 2 ** 64 + 3, 1e20, "-0.5", 10 ** 400),
    "hires": (True, False, "yes", None),
}
SETTINGS_CONTEXTS = ({}, {"threshold": "manual"}, {"mode": "respace", "threshold": "manual"},
                     {"threshold": "delta_optimal"}, {"preset": "custom"})


def _settings_rows() -> list:
    rows: list = [[], {}, [{}], None]
    for name, values in SETTINGS_VALUES.items():
        for value in values:
            for context in SETTINGS_CONTEXTS:
                rows.append({**context, name: value})
    for alias, name in (("divisor", "sigma_divisor"), ("manual", "manual_sigmas"), ("A", "spectrum_A"),
                        ("beta", "spectrum_beta")):
        for value in SETTINGS_VALUES[name][:4]:
            rows.append({"threshold": "manual", alias: value})
            rows.append({name: SETTINGS_VALUES[name][-1], alias: value})       # the later key wins
    rng = random.Random(41)
    names = list(speed_schedule_arg_names())
    for _ in range(400):
        row = {name: rng.choice(SETTINGS_VALUES[name]) for name in names if rng.random() < 0.6}
        rows.append(row)
        rows.append(row if rng.random() < 0.5 else [row])
        length = rng.randint(0, len(names))
        rows.append([rng.choice(SETTINGS_VALUES[name]) for name in names[:length]])
    return rows


def speed_schedule_arg_names() -> tuple:
    from comfy_custom_nodes.ai_studio_forge_parity import speed_nodes

    return speed_nodes.ARG_NAMES


def _canon_settings(settings) -> tuple:
    from comfy_custom_nodes.ai_studio_forge_parity import speed_nodes

    fields = tuple(speed_nodes.SpeedSettings.__dataclass_fields__)
    return tuple((name, _canon(getattr(settings, name))) for name in fields)


@requires_torch
@requires_extension
class NodeSettingsParityTests(unittest.TestCase):
    """``speed_nodes`` settings == the extension's ``forge_host.coerce_settings`` (no XYZ on ComfyUI)."""

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import speed_nodes

        cls.pack = speed_nodes
        cls.fh = _ext_speed("forge_host")

    def test_constants(self):
        for name in ("ARG_NAMES", "ARG_ALIASES", "BROWNIAN_STAGE_SEED_STRIDE"):
            with self.subTest(name=name):
                self.assertEqual(getattr(self.pack, name), getattr(self.fh, name))
                self.assertEqual(type(getattr(self.pack, name)), type(getattr(self.fh, name)))
        self.assertEqual(list(self.pack.DEFAULTS), list(self.fh.DEFAULTS))
        self.assertEqual({k: _canon(v) for k, v in self.pack.DEFAULTS.items()},
                         {k: _canon(v) for k, v in self.fh.DEFAULTS.items()})
        for name, reason in self.fh.UNSUPPORTED_SAMPLERS.items():
            with self.subTest(sampler=name):
                self.assertEqual(self.pack.UNSUPPORTED_SAMPLERS[name], reason)
        extra = set(self.pack.UNSUPPORTED_SAMPLERS) - set(self.fh.UNSUPPORTED_SAMPLERS)
        self.assertEqual(extra, {"dpm_fast_function", "dpm_adaptive_function"})    # v0.33.0 lists sample_unipc_bh2
        self.assertEqual(list(self.pack.UNSUPPORTED_SAMPLERS)[:len(self.fh.UNSUPPORTED_SAMPLERS)],
                         list(self.fh.UNSUPPORTED_SAMPLERS))
        self.assertLessEqual(set(self.pack.UNSUPPORTED_SAMPLERS.values()), set(self.fh.UNSUPPORTED_SAMPLERS.values()))
        self.assertEqual(set(self.pack.SpeedSettings.__dataclass_fields__),
                         set(self.fh.SpeedSettings.__dataclass_fields__) - {"xyz"})

    def test_coerce_settings_rows(self):
        rows = _settings_rows()
        self.assertGreater(len(rows), 1500)
        errors = valid = 0
        for row in rows:
            ext_args = [row] if isinstance(row, dict) else row
            got = _outcome(self.pack.coerce_settings, _canon_settings, row)
            want = _outcome(self.fh.coerce_settings, _canon_settings, ext_args)
            with self.subTest(row=repr(row)[:160]):
                self.assertEqual(got, want)
                if got[0] == "ok":
                    pack_settings = self.pack.coerce_settings(row)
                    self.assertEqual(pack_settings.summary(), self.fh.coerce_settings(ext_args).summary())
                    errors += pack_settings.error is not None
                    valid += pack_settings.error is None
        self.assertGreater(errors, 300)
        self.assertGreater(valid, 300)
        # sam-extra v0.33.0: a zero denominator in the ladder/manual text is an invalid setting (0.32.1 let
        # ZeroDivisionError escape its reader) — the manual text counts only under the manual threshold — and
        # an integer too large for a float reads as the default (0.32.1: OverflowError). Neither reader raises.
        defaults = self.pack.DEFAULTS
        pinned = (
            ({"scales": "1/0,1"}, "scales: division by zero: '1/0'", {"scales": (0.5, 1.0)}),
            ({"scales": "0.5,1/0"}, "scales: division by zero: '1/0'", {"scales": (0.5, 1.0)}),
            ({"threshold": "manual", "manual_sigmas": "0.7/0"}, "manual sigmas: division by zero: '0.7/0'",
             {"manual_sigmas": (0.7,)}),
            ({"threshold": "neo_shift", "manual_sigmas": "0.5,1/0"}, None, {"manual_sigmas": (0.7,)}),
            ({"threshold": "delta_optimal", "manual_sigmas": "1/0"}, None, {"manual_sigmas": (0.7,)}),
            ({"delta": 10 ** 400, "sigma_divisor": -10 ** 400, "seed": 10 ** 400, "spectrum_A": 10 ** 400,
              "spectrum_beta": -10 ** 400}, None,
             {"delta": defaults["delta"], "sigma_divisor": defaults["sigma_divisor"], "seed": -1,
              "spectrum_A": defaults["spectrum_A"], "spectrum_beta": defaults["spectrum_beta"]}),
        )
        for row, error, fields in pinned:
            for form in (row, [row]):
                with self.subTest(pinned=repr(form)[:160]):
                    got = _outcome(self.pack.coerce_settings, _canon_settings, form)
                    self.assertEqual(got, _outcome(self.fh.coerce_settings, _canon_settings, [row]))
                    self.assertEqual(got[0], "ok")
                    settings = self.pack.coerce_settings(form)
                    self.assertEqual(settings.error, error)
                    self.assertEqual({name: getattr(settings, name) for name in fields}, fields)

    def test_plan_for(self):
        rng = random.Random(8)
        sigma_lists = [torch.tensor(_spacing(kind, steps, 3.0)) for kind in ("simple", "beta") for steps in (8, 20, 28)]
        sigma_lists.append(torch.tensor(_spacing("simple", 40, 3.0)[-15:]))
        latents = ((1, 16, 128, 128), (1, 16, 104, 152), (2, 4, 1, 125, 97), (1, 4, 64, 64))
        compared = 0
        for row in _settings_rows()[:900]:
            settings = self.pack.coerce_settings(row)      # never raises (sam-extra v0.33.0)
            if settings.error:
                continue
            ext_settings = self.fh.coerce_settings([row] if isinstance(row, dict) else row)
            sigmas = rng.choice(sigma_lists)
            x = torch.zeros(rng.choice(latents))
            with self.subTest(row=repr(row)[:160]):
                got = _outcome(self.pack.plan_for, _canon_plan, settings, x, sigmas)
                want = _outcome(self.fh._plan_for, _canon_plan, ext_settings, x, sigmas)
                self.assertEqual(got, want)
                compared += 1
        self.assertGreater(compared, 150)


class _ExtAnimaModel:
    """``p.sd_model`` of an Anima checkpoint for the extension's guard (its class name is read)."""


class _ExtFluxModel:
    pass


def _ext_spectrum_function():
    def spectrum_unet_wrapper(apply, args):
        return apply(args)

    return spectrum_unet_wrapper


@requires_torch
@requires_extension
class GuardWordingParityTests(unittest.TestCase):
    """Each ComfyUI skip reason says what the extension's Forge status says for the same condition.

    Equal text, except where the ComfyUI condition is a different object: the Anima reference line
    drops the Forge option it names (same first clause), c_concat and Spectrum name ComfyUI's
    providers (same explanation after `` - ``).
    """

    @classmethod
    def setUpClass(cls):
        bind_torch(globals())
        from comfy_custom_nodes.ai_studio_forge_parity import guidance_common, speed_nodes

        cls.pack = speed_nodes
        cls.common = guidance_common
        cls.fh = _ext_speed("forge_host")

    def _ext(self, *, x=None, sigmas=None, prediction="const", anima=True, image_mask=None, ref=False, concat=False,
             controlnet=False, lllite=False, spectrum=False):
        from unittest import mock
        from types import SimpleNamespace

        predictor = SimpleNamespace(prediction_type=prediction)
        unet = SimpleNamespace(
            model=SimpleNamespace(predictor=predictor, diffusion_model=SimpleNamespace(in_dim=36)),
            model_options={"model_function_wrapper": _ext_spectrum_function()} if spectrum else {},
            controlnet_linked_list=object() if controlnet else None,
        )
        sd_model = (_ExtAnimaModel if anima else _ExtFluxModel)()
        sd_model.forge_objects = SimpleNamespace(unet=unet)
        p = SimpleNamespace(sd_model=sd_model, image_mask=image_mask, mask=None)
        model = SimpleNamespace(inner_model=SimpleNamespace(predictor=predictor))
        dyn = SimpleNamespace(ref_latents=[torch.zeros(1)] if ref else None,
                              concat_latent=torch.zeros(1, 20, 8, 8) if concat else None,
                              lq_latent=None, pid=False, ACTIVE_LLLITE_DIT=lllite, anima=False)
        with mock.patch.dict(sys.modules, {"backend.args": SimpleNamespace(dynamic_args=dyn)}):
            return self.fh.runtime_guard(p, model, torch.zeros(1, 4, 8, 8) if x is None else x,
                                         _flow_tensor() if sigmas is None else sigmas)

    def _pack(self, *, x=None, sigmas=None, eps=False, anima=True, extra=None, conds=None):
        from types import SimpleNamespace

        class CONST:
            pass

        class EPS:
            pass

        class Anima:
            pass

        class Flux:
            pass

        sampling = type("ModelSampling", (EPS if eps else CONST,), {})()
        guider = SimpleNamespace(model_patcher=SimpleNamespace(get_model_object=lambda name: sampling),
                                 inner_model=(Anima if anima else Flux)(),
                                 conds={"positive": conds or [{"model_conds": {}}]})
        guider.inner_model.latent_shapes = None
        return self.pack.comfy_runtime_guard(guider, torch.zeros(1, 4, 8, 8) if x is None else x,
                                             _flow_tensor() if sigmas is None else sigmas,
                                             extra if extra is not None else {"model_options": {}})

    def test_same_wording(self):
        rows = [
            ("no schedule", dict(sigmas=torch.zeros(2, 3)), dict(sigmas=torch.zeros(2, 3))),
            ("3-D latent", dict(x=torch.zeros(4, 8, 8)), dict(x=torch.zeros(4, 8, 8))),
            ("5-D ok", dict(x=torch.zeros(1, 4, 2, 8, 8)), dict(x=torch.zeros(1, 4, 2, 8, 8))),
            ("eps", dict(prediction="eps"), dict(eps=True)),
            ("sigma max", dict(sigmas=torch.tensor([1.25, 0.5, 0.0])), dict(sigmas=torch.tensor([1.25, 0.5, 0.0]))),
            ("mask", dict(image_mask=object()), dict(extra={"denoise_mask": torch.ones(1), "model_options": {}})),
            ("other reference", dict(ref=True, anima=False),
             dict(anima=False, conds=[{"model_conds": {"ref_latents": object()}}])),
            ("controlnet", dict(controlnet=True), dict(conds=[{"model_conds": {}, "control": object()}])),
            ("lllite", dict(lllite=True), dict(extra={"model_options": {self.pack.LLLITE_MARKER: True}})),
            ("clean", dict(), dict()),
        ]
        for name, ext_kwargs, pack_kwargs in rows:
            with self.subTest(row=name):
                self.assertEqual(self._pack(**pack_kwargs), self._ext(**ext_kwargs))

    def test_comfy_specific_wording_keeps_the_explanation(self):
        ext_anima = self._ext(ref=True)
        pack_anima = self._pack(conds=[{"model_conds": {"ref_latents": object()}}])
        self.assertTrue(ext_anima.startswith(pack_anima.split("; ")[0] + "; "), (ext_anima, pack_anima))
        ext_concat = self._ext(concat=True)
        pack_concat = self._pack(conds=[{"model_conds": {"c_concat": object()}}])
        self.assertEqual(pack_concat.split(" - ", 1)[1], ext_concat.split(" - ", 1)[1])
        ext_spectrum = self._ext(spectrum=True)
        pack_spectrum = self._pack(extra={"model_options": {"transformer_options": {
            self.common.DY_SUBSTEPS_OFF_KEY: "Spectrum"}}})
        self.assertEqual(pack_spectrum.split(" - ", 1)[1], ext_spectrum.split(" - ", 1)[1])


def _flow_tensor():
    return torch.tensor([_flow(1.0 - i / 12, 3.0) for i in range(12)] + [0.0])

if __name__ == "__main__":
    unittest.main()
