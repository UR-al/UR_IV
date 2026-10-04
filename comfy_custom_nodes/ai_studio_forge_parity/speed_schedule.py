"""SPEED schedule maths for the pack's ``ForgeNeoAnimaSpeed`` (pack 1.7.0).

Where a SPEED pass hands off from one grid to the next, and on which grids it runs:

* ``transition`` mode (official / aoleg): a coarse segment ends at the first step whose sigma is at
  or below the transition sigma ``sigma*``; the expansion patches only that step's sigma to the
  aligned time ``sigma * kappa`` and keeps the rest of the schedule; grids are ``round(s * H)``.
* ``respace`` mode (sorryhyun): at the hand-off every remaining sigma is multiplied by
  ``aligned / sigma`` (paper Sec. 4.3); coarse grids are snapped to even sizes (the DiT patch).

``sigma*`` comes from the latent power spectrum ``P(w) = A w^-beta`` at the coarse grid's Nyquist
frequency (paper Eq. 9/10 with the noise tolerance ``delta``), optionally held at the 1024 px
reference (aoleg's adaptive delta, with its step caps), divided by the ``neo_shift`` divisor, or
given by hand (``manual``).

The behaviour is sam-extra's (``forge_sam3_extension`` ``sam3ext/speed/schedule.py``): the same
plan, the same refusals (:class:`PlanError` with the same ``reason``/``kind``) and the same notes, so
a ComfyUI pass reports what a Forge pass reports. The code is the pack's own, written from the MIT
upstreams (full notices in THIRD_PARTY_NOTICES.md, texts in LICENSES/):

* howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 ``utils.py`` (``power_spectrum``,
  ``activation_time``, ``delta_optimal_transitions``, ``validate_scales``) — Copyright (c) 2026
  Howard Xiao;
* aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 ``speed_core.py`` (the preset table,
  ``REFERENCE_LATENT``, ``MAX_COARSE_FRACTION``, adaptive delta, ``sigma_divisor``,
  ``_resolve_transitions`` / ``_resolve_manual``) and ``spectral_utils.py`` (``equivalent_delta``,
  ``reference_coarse_fraction``) — Copyright (c) 2026 A. Izzuddin Al Faruq (the presets, adaptive
  delta and neo_shift are Oleg Afonin's work in that fork);
* sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae ``spd.py``
  (``resolve_spd_schedule``'s one-sigma-per-hand-off rule, the respace loop's sigma arithmetic) —
  Copyright (c) 2026 sorryhyun.

sam-extra's documented changes to those sources, re-expressed here from their behaviour and checked
by ``tests/test_comfy_speed_parity.py`` against the extension (v0.33.0): the threshold maths and the
step lookup are separate so both plans use the same thresholds; a threshold the divisor lifts to
>= 1 gets no reference-share cap; the respace loop is replayed on the sigmas alone so the plan is
known before the first model call; stages whose hand-off fires at step 0 are dropped (not expanded);
a plan whose last hand-off is never reached is refused; respace's last stage is the exact full grid;
dwt is refused up front unless every hand-off doubles the grid; non-positive or non-finite custom
spectra are settings errors (sam-extra 0.32.1); a zero denominator in the text (``1/0``) is a
``ValueError`` naming the token, so the readers report it as a settings error (sam-extra 0.33.0);
each transition carries the sigma it runs at and the aligned one.

No torch at module level (the app imports the pack without it); only the respace replay and the
hand-off replay need it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .compat import require_torch
from .speed_spectral import align_timestep, respace_alignment, snap

__all__ = [
    "CUSTOM_DEFAULT_A",
    "CUSTOM_DEFAULT_BETA",
    "MAX_COARSE_FRACTION",
    "MODES",
    "PRESETS",
    "PRESET_NAMES",
    "REFERENCE_LATENT",
    "RESPACE_PATCH",
    "THRESHOLDS",
    "TRANSFORMS",
    "Plan",
    "PlanError",
    "ThresholdPlan",
    "Transition",
    "activation_time",
    "build_plan",
    "delta_optimal_transitions",
    "delta_threshold_plan",
    "equivalent_delta",
    "parse_scales",
    "parse_sigmas",
    "power_spectrum",
    "preset_params",
    "reference_coarse_fraction",
    "respace_transitions",
    "stage_grids",
    "static_transitions",
    "validate_scales",
]

MODES = ("transition", "respace")
THRESHOLDS = ("neo_shift", "delta_optimal", "manual")
TRANSFORMS = ("dct", "dwt", "fft")

# aoleg speed_core.py: the latent short side of the 1024x1024 reference every preset sigma* is quoted at.
REFERENCE_LATENT = 128
# aoleg speed_core.py: backstop share of coarse steps for adaptive delta without ``ref_shift``.
MAX_COARSE_FRACTION = 0.8
# sorryhyun spd.py ``_PATCH``: respace snaps its coarse grids to this multiple (Anima DiT patch).
RESPACE_PATCH = 2
# Official configs / comfyui speed_sampler.py: the published Flux fit, the ``custom`` defaults.
CUSTOM_DEFAULT_A = 203.615097
CUSTOM_DEFAULT_BETA = 1.915461

# aoleg speed_core.py ``_PRESETS`` (values unchanged). ``ref_shift`` switches the adaptive cap to the
# reference share (raw multiplier for discrete-flow models, exp(mu) for Flux-style ones).
PRESETS: dict[str, dict[str, Any] | None] = {
    "flux": {"A": 1963.48906, "beta": 1.915461, "ref_latent": 128},
    "wan21": {"A": 219.484718, "beta": 2.422687, "ref_latent": 128, "ref_shift": 8.0},
    "krea-2": {"A": 2357.960557, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "krea-2-raw": {"A": 887.402358, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.158193},
    "z-image": {"A": 877.001476, "beta": 1.915461, "ref_latent": 128, "ref_shift": 3.0},
    "flux2": {"A": 520.495848, "beta": 1.915461, "ref_latent": 64},
    "anima": {"A": 8664.998524, "beta": 2.422687, "ref_latent": 128, "ref_shift": 3.0},
    "custom": None,
}
# sam-extra's UI order (its model first); the node offers the same list.
PRESET_NAMES = ("anima", "flux", "flux2", "krea-2", "krea-2-raw", "z-image", "wan21", "custom")


class PlanError(ValueError):
    """No SPEED run for this pass. ``reason`` is shown to the user; ``kind`` is ``"skip"`` (nothing
    to do, the pass samples plainly) or ``"invalid"`` (a settings error)."""

    def __init__(self, reason: str, *, kind: str = "skip"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


# ------------------------------------------------------------------------------------------------
# Power spectrum and activation times (official utils.py; aoleg spectral_utils.py)
# ------------------------------------------------------------------------------------------------


def power_spectrum(omega: float, A: float, beta: float) -> float:
    """``P(omega) = A * |omega|^-beta``."""
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """When the frequency with power ``P_omega`` leaves the noise floor (paper Eq. 9)."""
    if delta >= 1.0:
        raise ValueError(f"delta={delta} >= 1, but we assume the error threshold is < 1.")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def equivalent_delta(t_star: float, A: float, beta: float, omega: float) -> float:
    """The ``delta`` whose activation time at ``omega`` is ``t_star`` (aoleg; inverse of Eq. 9)."""
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    P = power_spectrum(omega, A, beta)
    u = ((1.0 - t_star) / t_star) ** 2
    return u * P * (1.0 + P) / (1.0 + u * P)


def reference_coarse_fraction(t_star: float, shift: float) -> float:
    """Share of a uniform-in-t schedule shifted by ``shift`` that lies above ``t_star`` (aoleg)."""
    if shift <= 0.0:
        raise ValueError(f"shift must be positive; got {shift}")
    if not (0.0 < t_star < 1.0):
        raise ValueError(f"t_star must be in (0, 1); got {t_star}")
    T = t_star / (shift * (1.0 - t_star) + t_star)
    return 1.0 - T


def validate_scales(scales: Sequence[float]) -> None:
    """Scales must be in (0, 1], strictly increasing and end at 1.0 (official)."""
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def delta_optimal_transitions(
    scales: Sequence[float], delta: float, A: float, beta: float, H: int, W: int,
) -> list[float]:
    """Transition time per pair of adjacent scales (paper Eq. 10, ``omega_max = min(H, W) / 2``)."""
    validate_scales(scales)
    omega_max = min(H, W) / 2.0
    return [
        activation_time(power_spectrum(scales[i] * omega_max, A, beta), delta)
        for i in range(len(scales) - 1)
    ]


# ------------------------------------------------------------------------------------------------
# Text inputs (``1/2`` fractions and ``;`` accepted, like the official CLI and sam-extra)
# ------------------------------------------------------------------------------------------------


def _number(token: str) -> float:
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        try:
            value = float(numerator) / float(denominator)
        except ZeroDivisionError:   # a ValueError, so the readers report it as a settings error (sam-extra 0.33.0)
            raise ValueError(f"division by zero: {token!r}") from None
    else:
        value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: {token!r}")
    return value


def _numbers(text: Any) -> list[float]:
    """Numbers of a list/tuple, or of a ``,``/``;`` separated string (empty entries skipped)."""
    if isinstance(text, (list, tuple)):
        tokens = [str(item).strip() for item in text]
    else:
        tokens = [part.strip() for part in str("" if text is None else text).replace(";", ",").split(",")]
    return [_number(token) for token in tokens if token]


def parse_scales(text: Any) -> tuple[float, ...]:
    """The resolution ladder, e.g. ``"0.5,1.0"`` or ``"1/4; 1/2; 1"``; it must end at 1.0."""
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"scales: {exc}") from None
    validate_scales(values)
    return tuple(values)


def parse_sigmas(text: Any, *, strictly_decreasing: bool = True) -> tuple[float, ...]:
    """Manual hand-off sigmas, each in (0, 1) (flow-matching sigmas only).

    ``transition`` mode needs them strictly decreasing (official node); respace does not (sorryhyun).
    """
    try:
        values = _numbers(text)
    except ValueError as exc:
        raise ValueError(f"manual sigmas: {exc}") from None
    if not values:
        raise ValueError("manual sigmas: empty")
    if any(not (0.0 < value < 1.0) for value in values):
        raise ValueError(f"every manual sigma must be in (0, 1); got {values}")
    if strictly_decreasing and any(not (a > b) for a, b in zip(values[:-1], values[1:])):
        raise ValueError(f"manual sigmas must be strictly decreasing; got {values}")
    return tuple(values)


def preset_params(name: str, A: float = CUSTOM_DEFAULT_A, beta: float = CUSTOM_DEFAULT_BETA):
    """``(A, beta, ref_latent, ref_shift)`` of a preset; ``custom`` (or an unknown name) uses ``A``/``beta``."""
    preset = PRESETS.get(str(name))
    if preset is None:
        return float(A), float(beta), REFERENCE_LATENT, None
    return preset["A"], preset["beta"], int(preset["ref_latent"]), preset.get("ref_shift")


# ------------------------------------------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPlan:
    values: tuple[float, ...]            # sigma* per transition, after pinning and the divisor
    caps: tuple[int | None, ...]         # adaptive step cap per transition (None: no cap)
    notes: tuple[str, ...] = ()


def _pairs(before: Sequence[float], after: Sequence[float]) -> str:
    return ", ".join(f"{a:.4f}->{b:.4f}" for a, b in zip(before, after))


def _adaptive_note(H, W, ref_latent, actual, pinned, delta, A, beta, first_scale) -> str:
    try:
        omega = first_scale * (min(H, W) / 2.0)
        equivalent = f"{equivalent_delta(pinned[0], A, beta, omega):.6g}"
    except ValueError:
        equivalent = "n/a"
    return (
        f"adaptive delta: latent {H}x{W} vs reference {ref_latent}x{ref_latent}; sigma* "
        + _pairs(actual, pinned)
        + f" (delta {delta:g} -> {equivalent} equivalent at this resolution)"
    )


def _step_cap(index: int, threshold: float, n_steps: int, n_transitions: int, adaptive: bool, ref_shift):
    """aoleg's adaptive limit for one transition (None when it does not apply)."""
    if not adaptive:
        return None
    if ref_shift:
        # the reference split; a threshold the divisor pushed out of (0, 1) has none
        if 0.0 < threshold < 1.0:
            return math.ceil(n_steps * reference_coarse_fraction(threshold, ref_shift))
        return None
    # no ref_shift (measured presets, custom): the flat backstop shared over the ladder
    return int(n_steps * MAX_COARSE_FRACTION * (index + 1) / n_transitions)


def delta_threshold_plan(
    n_steps: int,
    scales: Sequence[float],
    delta: float,
    A: float,
    beta: float,
    H: int,
    W: int,
    *,
    adaptive: bool = False,
    ref_latent: int = REFERENCE_LATENT,
    ref_shift: float | None = None,
    sigma_divisor: float = 1.0,
) -> ThresholdPlan:
    """sigma* per transition and its adaptive cap (aoleg ``_resolve_transitions`` before its step lookup)."""
    if len(scales) < 2:
        return ThresholdPlan((), (), ())
    notes: list[str] = []
    thresholds = delta_optimal_transitions(scales, delta, A, beta, H, W)
    if adaptive:
        reference = delta_optimal_transitions(scales, delta, A, beta, ref_latent, ref_latent)
        pinned = [max(own, ref) for own, ref in zip(thresholds, reference)]
        if any(abs(a - b) > 1e-9 for a, b in zip(thresholds, pinned)):
            notes.append(_adaptive_note(H, W, ref_latent, thresholds, pinned, delta, A, beta, scales[0]))
        thresholds = pinned
    if sigma_divisor != 1.0:
        divided = [value / sigma_divisor for value in thresholds]
        notes.append(f"neo_shift: dividing sigma* by {sigma_divisor:g} (" + _pairs(thresholds, divided) + ")")
        thresholds = divided
    n_transitions = len(scales) - 1
    caps = [
        _step_cap(index, value, n_steps, n_transitions, adaptive, ref_shift)
        for index, value in enumerate(thresholds)
    ]
    return ThresholdPlan(tuple(float(value) for value in thresholds), tuple(caps), tuple(notes))


# ------------------------------------------------------------------------------------------------
# The two step lookups
# ------------------------------------------------------------------------------------------------


def _first_step_at_or_below(sigmas, threshold: float, n_steps: int) -> int:
    """Index of the first sigma <= ``threshold`` (never the final sigma); ``n_steps`` when none."""
    for index in range(n_steps):
        if float(sigmas[index]) <= threshold:
            return index
    return n_steps


def static_transitions(sigmas, scales: Sequence[float], values: Sequence[float], caps=None) -> list[tuple]:
    """``(step, s_from, s_to)`` per transition on the unpatched schedule (official/aoleg lookup).

    Each step is held at its adaptive cap; the lookup stops at the first transition that does not
    happen before the final sigma.
    """
    n_steps = len(sigmas) - 1
    limits = list(caps) if caps is not None else [None] * len(values)
    found: list[tuple] = []
    for index, (s_from, s_to, threshold) in enumerate(zip(scales[:-1], scales[1:], values)):
        step = _first_step_at_or_below(sigmas, threshold, n_steps)
        limit = limits[index] if index < len(limits) else None
        if limit is not None and step > limit:
            step = limit
        if step >= n_steps:
            break
        found.append((step, s_from, s_to))
    return found


def _hand_off_due(schedule, index: int, stage: int, values, caps) -> bool:
    limit = caps[stage]
    return float(schedule[index]) <= values[stage] or (limit is not None and index >= limit)


def respace_transitions(sigmas, scales: Sequence[float], values: Sequence[float], caps=None):
    """sorryhyun's respace loop replayed on the schedule alone.

    Returns ``(dropped, events, schedule)``: the number of leading stages whose hand-off is already
    due at step 0 (dropped, not expanded), ``(step, k, sigma, aligned)`` per hand-off (``aligned`` as
    the float32 schedule stores it) and the re-spaced float32 schedule after the last hand-off.
    """
    torch = require_torch()
    limits = list(caps) if caps is not None else [None] * len(values)
    if torch.is_tensor(sigmas):
        schedule = sigmas.detach().clone().float()
    else:
        schedule = torch.tensor(list(sigmas), dtype=torch.float32)
    n_steps = len(schedule) - 1
    stage = 0
    while stage < len(values) and n_steps >= 1 and _hand_off_due(schedule, 0, stage, values, limits):
        stage += 1
    dropped = stage
    events = []
    for index in range(n_steps):
        while stage < len(values) and _hand_off_due(schedule, index, stage, values, limits):
            sigma = float(schedule[index])
            aligned, _factor = respace_alignment(sigma, scales[stage + 1] / scales[stage])
            if sigma > 0.0 and aligned != sigma:
                schedule[index + 1:] = aligned * (schedule[index + 1:] / sigma)
            schedule[index] = aligned
            events.append((index, stage, sigma, float(schedule[index])))
            stage += 1
    return dropped, events, schedule


def _transition_hand_offs(sigmas, scales: Sequence[float], steps) -> list[tuple[float, float]]:
    """``(sigma, aligned)`` per transition-mode hand-off, patched on a copy as the run patches it
    (only the hand-off step), so two hand-offs at one step see the first one's aligned sigma."""
    torch = require_torch()
    if torch.is_tensor(sigmas):
        schedule = sigmas.detach().clone()
    else:
        schedule = torch.tensor(list(sigmas), dtype=torch.float32)
    out = []
    for step, k in steps:
        sigma = float(schedule[step])
        schedule[step] = float(align_timestep(sigma, float(scales[k + 1]) / float(scales[k])))
        out.append((sigma, float(schedule[step])))
    return out


# ------------------------------------------------------------------------------------------------
# The plan
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One expansion at schedule index ``step``: ladder entry ``k`` -> ``k + 1``.

    ``sigma`` is the schedule value the expansion runs at and ``aligned`` the value the next segment
    starts from, as the run will see them (respace: after the earlier re-spacings)."""

    step: int
    k: int
    scale_from: float
    scale_to: float
    grid_from: tuple[int, int]
    grid_to: tuple[int, int]
    sigma: float = math.nan
    aligned: float = math.nan


@dataclass
class Plan:
    mode: str
    transform: str
    full_grid: tuple[int, int]
    first_scale: float
    first_grid: tuple[int, int]
    transitions: list[Transition]
    thresholds: tuple[float, ...]
    n_steps: int
    notes: list[str] = field(default_factory=list)

    @property
    def coarse_steps(self) -> int:
        """Sampler steps that run below the full grid."""
        return self.transitions[-1].step if self.transitions else 0


def stage_grids(mode: str, scales: Sequence[float], H: int, W: int) -> list[tuple[int, int]]:
    """Latent grid of every stage of ``scales`` (the stages that run, first to full)."""
    if mode != "respace":
        return [(int(round(s * H)), int(round(s * W))) for s in scales]
    grids: list[tuple[int, int]] = []
    for position, s in enumerate(scales):
        if abs(s - 1.0) <= 1e-6:
            # the tolerance validate_scales accepts for the last scale: that stage is the exact grid
            grids.append((H, W))
            continue
        h, w = snap(H * s, RESPACE_PATCH), snap(W * s, RESPACE_PATCH)
        if position == 0:
            grids.append((min(h, H), min(w, W)))
        else:
            grids.append((max(h, grids[-1][0]), max(w, grids[-1][1])))
    return grids


def _check_choices(mode, transform, threshold, scales) -> None:
    if mode not in MODES:
        raise PlanError(f"unknown mode {mode!r}", kind="invalid")
    if transform not in TRANSFORMS:
        raise PlanError(f"unknown transform {transform!r}", kind="invalid")
    if threshold not in THRESHOLDS:
        raise PlanError(f"unknown threshold source {threshold!r}", kind="invalid")
    try:
        validate_scales(scales)
    except ValueError as exc:
        raise PlanError(str(exc), kind="invalid") from None


def _manual_thresholds(mode: str, manual_sigmas, n_transitions: int):
    values = [float(v) for v in manual_sigmas]
    if mode == "respace" and len(values) == 1 and n_transitions > 1:
        values = values * n_transitions      # sorryhyun: one sigma serves every hand-off
    if len(values) != n_transitions:
        raise PlanError(
            f"manual sigmas has length {len(values)}, expected {n_transitions} "
            "(one threshold per transition in scales)", kind="invalid",
        )
    if any(not (0.0 < v < 1.0) for v in values):
        raise PlanError(f"every manual sigma must be in (0, 1); got {values}", kind="invalid")
    if mode == "transition" and any(not (a > b) for a, b in zip(values[:-1], values[1:])):
        raise PlanError(f"manual sigmas must be strictly decreasing; got {values}", kind="invalid")
    return values, [None] * n_transitions, []


def _spectrum_thresholds(threshold, n_steps, scales, delta, A, beta, H, W, ref_latent, ref_shift,
                         adaptive, sigma_divisor):
    if not (0.0 < float(delta) < 1.0):
        raise PlanError(f"delta must be in (0, 1); got {delta}", kind="invalid")
    if not (float(sigma_divisor) > 0.0):
        raise PlanError(f"sigma divisor must be positive; got {sigma_divisor}", kind="invalid")
    # sam-extra 0.32.1: A <= 0 has no activation time and a non-finite A or beta would hand the cap a
    # NaN sigma* — both are settings errors, not crashes
    if not (math.isfinite(float(A)) and float(A) > 0.0):
        raise PlanError(f"spectrum A must be a positive number; got {A}", kind="invalid")
    if not math.isfinite(float(beta)):
        raise PlanError(f"spectrum beta must be a finite number; got {beta}", kind="invalid")
    divisor = float(sigma_divisor) if threshold == "neo_shift" else 1.0
    try:
        planned = delta_threshold_plan(
            n_steps, scales, float(delta), float(A), float(beta), H, W,
            adaptive=bool(adaptive), ref_latent=int(ref_latent), ref_shift=ref_shift, sigma_divisor=divisor,
        )
    except (ArithmeticError, ValueError) as exc:   # e.g. P(omega) under- or overflowing for a huge beta
        raise PlanError(
            f"spectrum A {float(A):g} / beta {float(beta):g} gives no transition time ({exc})", kind="invalid",
        ) from None
    return list(planned.values), list(planned.caps), list(planned.notes)


def _transition_mode_steps(sigmas, scales, values, caps, n_transitions):
    found = static_transitions(sigmas, scales, values, caps)
    dropped = 0
    while dropped < len(found) and found[dropped][0] == 0:
        dropped += 1
    if len(found) < n_transitions:
        missing = len(found)
        raise PlanError(
            f"the {scales[missing]:g}->{scales[missing + 1]:g} transition (sigma* {values[missing]:.4f}) "
            "is never reached in this schedule — the output would stay below full size"
        )
    steps = [(found[k][0], k) for k in range(dropped, n_transitions)]
    return dropped, steps, _transition_hand_offs(sigmas, scales, steps)


def _respace_mode_steps(sigmas, scales, values, caps, n_transitions):
    dropped, events, _schedule = respace_transitions(sigmas, scales, values, caps)
    if dropped + len(events) < n_transitions:
        missing = dropped + len(events)
        raise PlanError(
            f"the {scales[missing]:g}->{scales[missing + 1]:g} hand-off (sigma {values[missing]:.4f}) "
            "is never reached in this schedule — the output would stay below full size"
        )
    steps = [(step, k) for step, k, _sigma, _aligned in events]
    return dropped, steps, [(sigma, aligned) for _step, _k, sigma, aligned in events]


def _check_dwt_ratios(scales, steps) -> None:
    # the one-level Haar synthesis is an exact 2x step: every hand-off that runs needs r = 2
    for _step, k in steps:
        r = float(scales[k + 1]) / float(scales[k])
        if abs(r - 2.0) > 1e-6:
            raise PlanError(
                f"dwt needs r = 2 between consecutive scales; got {scales[k]:g}->{scales[k + 1]:g} "
                f"(r = {r:.4f}) - use dct or fft for non-dyadic scales", kind="invalid",
            )


def _check_grids(transform, grids, scales, H, W) -> None:
    if any(h < 1 or w < 1 for h, w in grids):
        raise PlanError(f"a coarse grid is empty for scales {list(scales)} at latent {H}x{W}", kind="invalid")
    if grids[-1] != (H, W):
        raise PlanError("internal: the last stage is not the full grid", kind="invalid")
    for (h0, w0), (h1, w1) in zip(grids[:-1], grids[1:]):
        if transform == "dwt" and (h1 != 2 * h0 or w1 != 2 * w0):
            raise PlanError(
                f"dwt needs every stage to double the grid exactly; got {(h0, w0)} -> {(h1, w1)} "
                "(use dct or fft for this ladder or latent size)", kind="invalid",
            )
        if h1 < h0 or w1 < w0:
            raise PlanError(f"grid shrinks between stages: {(h0, w0)} -> {(h1, w1)}", kind="invalid")


def build_plan(
    *,
    mode: str,
    transform: str,
    sigmas,
    full_grid: tuple[int, int],
    scales: Sequence[float],
    threshold: str,
    delta: float = 0.01,
    A: float = CUSTOM_DEFAULT_A,
    beta: float = CUSTOM_DEFAULT_BETA,
    ref_latent: int = REFERENCE_LATENT,
    ref_shift: float | None = None,
    adaptive: bool = True,
    sigma_divisor: float = 1.0,
    manual_sigmas: Sequence[float] = (),
) -> Plan:
    """The plan of one sampling pass, or :class:`PlanError` when it cannot or need not run coarse."""
    _check_choices(mode, transform, threshold, scales)
    H, W = int(full_grid[0]), int(full_grid[1])
    n_steps = len(sigmas) - 1
    if n_steps < 1:
        raise PlanError("empty sigma schedule")
    n_transitions = len(scales) - 1
    if n_transitions == 0:
        raise PlanError("scales has no coarse stage (only 1.0)")

    if threshold == "manual":
        values, caps, notes = _manual_thresholds(mode, manual_sigmas, n_transitions)
    else:
        values, caps, notes = _spectrum_thresholds(
            threshold, n_steps, scales, delta, A, beta, H, W, ref_latent, ref_shift, adaptive, sigma_divisor,
        )

    if mode == "transition":
        dropped, steps, hand_offs = _transition_mode_steps(sigmas, scales, values, caps, n_transitions)
    else:
        dropped, steps, hand_offs = _respace_mode_steps(sigmas, scales, values, caps, n_transitions)
    if not steps:
        first = values[dropped - 1] if dropped else values[0]
        raise PlanError(
            f"no coarse steps: the pass starts at sigma {float(sigmas[0]):.4f}, at or below the first "
            f"transition sigma {first:.4f}"
        )
    if transform == "dwt":
        _check_dwt_ratios(scales, steps)
    used = list(scales[dropped:])
    grids = stage_grids(mode, used, H, W)
    _check_grids(transform, grids, scales, H, W)
    if dropped:
        notes.append(
            f"{dropped} leading stage(s) skipped: their transition fires at step 0 "
            f"(pass starts at sigma {float(sigmas[0]):.4f})"
        )
    transitions = [
        Transition(
            step=int(step), k=int(k), scale_from=float(scales[k]), scale_to=float(scales[k + 1]),
            grid_from=grids[position], grid_to=grids[position + 1],
            sigma=float(hand_offs[position][0]), aligned=float(hand_offs[position][1]),
        )
        for position, (step, k) in enumerate(steps)
    ]
    return Plan(
        mode=mode, transform=transform, full_grid=(H, W), first_scale=float(used[0]), first_grid=grids[0],
        transitions=transitions, thresholds=tuple(values), n_steps=n_steps, notes=notes,
    )
