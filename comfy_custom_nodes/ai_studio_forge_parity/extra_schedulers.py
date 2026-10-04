"""Extra Schedulers for ComfyUI (pack 1.7.0): sam-extra's nine schedulers in KSampler's list.

sam-extra adds ``cosine``, ``cosine_exponential``, ``phi``, ``laplace``, ``karras_dynamic``,
``custom``, ``react_cosinusoidal_dynsf``, ``flow_cosmos_rho7`` and ``flow_cosmos_dynamic`` to Forge's
"Schedule type" list. This
module gives ComfyUI the same schedules under the same names, so every pass the app compiles (``ForgeNeoKSamplerCNS``, ``ForgeNeoHiresFix``, ADetailer
through Impact's FaceDetailer, SAM3, a custom workflow's stock ``KSampler``) computes the sigmas
Forge computes for the same steps and sigma range:

* **schedules** (this part) — Forge's scheduler signature ``fn(n, sigma_min, sigma_max, ...)``
  returning ``n`` sigmas and the final 0 as a float32 CPU tensor, with sam-extra's guards: no step
  may keep sigma where it was (``ExtraSchedulerError``), Karras Dynamic must fall
  (``KarrasDynamicError``), a custom schedule must fit float32 (``CustomSchedulerError``). The
  closed forms, the expression and the sigma list are in the pure ``extra_schedulers_math``.
  Laplace calls the host's own ``comfy.k_diffusion.sampling.get_sigmas_laplace`` and, where its
  clamp would repeat a sigma (a flow model's sigma_max 1 with the default mu 0), samples the same
  curve only over the part inside [sigma_min, sigma_max]; the sigma list interpolates with the host's
  ``comfy_extras.nodes_align_your_steps.loglinear_interp`` (Forge's ``_loglinear_interp``, the same
  function). Flow Cosmos rho7 gives a flow model (ComfyUI's ``CONST``, the rule the pack's Dy and Restart
  entries use: ``extra_samplers_dy.is_const``) the flow times of the ρ ramp over the VE-equivalent range
  σ̃ min … σ̃ max (the run's ``flow_cosmos_*`` values; defaults Cosmos-Predict2's ρ 7 on 0.002 … 80, the
  fixed list of before) and any other model the host's own ``comfy.k_diffusion.sampling.get_sigmas_karras``
  (ComfyUI's "karras" scheduler; Forge ships the same function) with the run's ρ on the MODEL's sigma range
  (σ̃ unused there), logging that once per process. Flow Cosmos Dynamic (sam-extra's own composition of Karras
  Dynamic and Flow Cosmos rho7, same three values) gives a flow model Karras Dynamic's per-step-ρ ramp over that
  σ̃ range in flow time, and any other model the pack's Karras Dynamic with the run's ρ on the MODEL's range
  (its own once-per-process line). Nothing is copied from ComfyUI.
* **handlers** — ``SchedulerHandler(fn, use_ms=True)`` functions ``fn(model_sampling, steps)``: the
  sigma range exactly as ``comfy.samplers.calculate_sigmas`` takes it (``float(sigma_min/max)``) and
  the run's values from the MODEL's ``model_sampling`` (``CARRIER_ATTR``; absent = the defaults);
  the two Flow Cosmos schedulers also read which kind of model that ``model_sampling`` belongs to.
  A handler sees nothing else — KSampler computes sigmas before it stores ``model_options``, and
  Impact calls the module-level ``calculate_sigmas`` — which is why the values ride on
  ``model_sampling``.
* **carrier, node, registration** — ``ForgeNeoExtraSchedulers`` (MODEL -> MODEL) puts the coerced
  values on a private copy of the MODEL's ``model_sampling`` (``carry``); the pack's flow shift keeps
  them (``hires_carry.register_sampling_attr``) and a Hires checkpoint override gets them too
  (``hires_carry.register``). ``install_extra_schedulers`` adds the nine names to ComfyUI's scheduler
  table without ever replacing another provider's entry; the node's optional ``owned_names`` COMBO
  reports which of them are the pack's (the app reads it from ``/object_info``).

Behaviour mirrored from sam-extra v0.33.0 ``sam3ext/extra_schedulers/schedulers.py`` (same float
operations, the same error classes and texts); the code is this pack's own — sam-extra (GPL-3.0) is
not imported or copied. ``tests/test_comfy_extra_schedulers_parity.py`` compares the float32 results
bit for bit with the installed extension.

No torch or ComfyUI import at module level (the app imports the pack in its own tests).
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
import os
from typing import Any, Callable, Mapping

from . import extra_schedulers_math as xmath
from . import hires_carry
from .compat import (
    NO_OWNED_NAMES,
    OWNED_NAMES_INPUT,
    NameRegistration,
    clone_model,
    entry_owner,
    mark_owned,
    own_attribute,
    owned_names_input,
    register_names,
    require_torch,
)
from .extra_schedulers_math import (
    CustomSchedulerError,
    ExpressionError,
    ExtraSchedulerError,
    KarrasDynamicError,
    Settings,
    SigmaListError,
)


LOGGER = logging.getLogger("ai_studio_forge_parity")

# Comfy names = Forge names (presets, history and XYZ values carry across backends unchanged).
SCHEDULER_NAMES = (
    "cosine", "cosine_exponential", "phi", "laplace", "karras_dynamic", "custom", "react_cosinusoidal_dynsf",
    "flow_cosmos_rho7", "flow_cosmos_dynamic",
)
# The Forge labels; error texts name a scheduler by its label, like sam-extra.
LABELS = {
    "cosine": "Cosine",
    "cosine_exponential": "CosineExponential blend",
    "phi": "Phi",
    "laplace": "Laplace",
    "karras_dynamic": "Karras Dynamic",
    "custom": "custom",
    "react_cosinusoidal_dynsf": "React Cosinusoidal DynSF",
    "flow_cosmos_rho7": "Flow Cosmos rho7",
    "flow_cosmos_dynamic": "Flow Cosmos Dynamic",
}
# The schedulers that read the block's values (the others depend on n and the sigma range only). Flow Cosmos rho7
# reads its ρ / σ̃ max / σ̃ min since sam-extra v0.33.0 (D19), Flow Cosmos Dynamic the same three (D20); without a
# carrier both run on their defaults.
VALUE_SCHEDULERS = ("laplace", "custom", "flow_cosmos_rho7", "flow_cosmos_dynamic")

# The run's settings on a MODEL's model_sampling (a ``Settings``); see the module docstring.
CARRIER_ATTR = "_ai_studio_extra_schedulers"
# The same values in the MODEL's model_options (informational: nothing samples from it).
MODEL_OPTIONS_KEY = "ai_studio_extra_schedulers"
# compat.OWNER_ATTR tag on the registered handlers, and the Hires carry key.
OWNER = "extra_schedulers"
CATEGORY = "AI Studio/Forge Neo parity/Generation"
# AISTUDIO_XSCHED_DEBUG=1: log every schedule a handler builds (the GPU check compares it with Forge).
DEBUG_ENV = "AISTUDIO_XSCHED_DEBUG"

# ``float(torch.finfo(torch.float32).max / .tiny)``: a custom sigma outside them becomes inf, or 0 /
# a subnormal, in the float32 schedule (checked against torch in the tests).
FLOAT32_MAX = 3.4028234663852886e38
FLOAT32_TINY = 1.1754943508222875e-38


# ── the host's functions (resolved when a schedule needs them) ──────────────────────────────────

def host_get_sigmas_laplace() -> Callable[..., Any]:
    """ComfyUI's ``comfy.k_diffusion.sampling.get_sigmas_laplace``."""

    return importlib.import_module("comfy.k_diffusion.sampling").get_sigmas_laplace


def host_get_sigmas_karras() -> Callable[..., Any]:
    """ComfyUI's ``comfy.k_diffusion.sampling.get_sigmas_karras`` — the function of its built-in "karras"
    scheduler (Forge's ``k_diffusion.sampling.get_sigmas_karras`` is the same function), looked up when Flow
    Cosmos rho7 runs on a model that is not a flow model; ``ExtraSchedulerError`` when it is not there."""

    try:
        function = importlib.import_module("comfy.k_diffusion.sampling").get_sigmas_karras
    except (ImportError, AttributeError) as exc:
        raise ExtraSchedulerError(
            "Extra Schedulers: Flow Cosmos rho7 on an eps/v model uses ComfyUI's Karras "
            f"(comfy.k_diffusion.sampling.get_sigmas_karras), which is not available: {type(exc).__name__}"
        ) from None
    if not callable(function):
        raise ExtraSchedulerError(
            "Extra Schedulers: ComfyUI's comfy.k_diffusion.sampling.get_sigmas_karras is not callable"
        )
    return function


def host_loglinear_interp(values: list, count: int):
    """ComfyUI's ``comfy_extras.nodes_align_your_steps.loglinear_interp`` (Forge's
    ``sd_schedulers._loglinear_interp``); looked up only when a sigma list is interpolated."""

    try:
        interpolate = importlib.import_module("comfy_extras.nodes_align_your_steps").loglinear_interp
    except (ImportError, AttributeError) as exc:
        raise SigmaListError(
            "ComfyUI's log-linear interpolation (comfy_extras.nodes_align_your_steps.loglinear_interp) is not "
            f"available: {type(exc).__name__}"
        ) from None
    return interpolate(values, count)


# ── float32 schedules ─────────────────────────────────────────────────────────────────────────

def with_final_zero(values, device: Any = "cpu"):
    """``values`` and the final 0 as a float32 tensor (each value through Python ``float`` first)."""

    torch = require_torch()
    return torch.tensor([*(float(value) for value in values), 0.0], dtype=torch.float32, device=device)


def first_flat_step(sigmas) -> int | None:
    """The first step whose sigma equals the next one's, among the sigmas before the final 0."""

    torch = require_torch()
    head = sigmas[:-1]
    repeats = torch.nonzero(head[1:] == head[:-1])
    return int(repeats[0]) if repeats.numel() else None


def _refuse_flat_step(label: str, sigmas, sigma_min: float, sigma_max: float, *, source: str | None = None) -> None:
    """``source`` replaces the note on where the range comes from, for a list that is not laid on it (Flow Cosmos
    rho7 on a flow model)."""

    step = first_flat_step(sigmas)
    if step is None:
        return
    if source is None:
        source = (f"sigma_min {sigma_min:g}, sigma_max {sigma_max:g} — Settings → Sampler Parameters → sigma min / "
                  f"sigma max")
    raise ExtraSchedulerError(
        f"Extra Schedulers: {label} gives sigma {float(sigmas[step]):.6g} at steps {step} and {step + 1} "
        f"({source}); {xmath.ZERO_LENGTH_STEP}"
    )


def _closed_form(label: str, values, sigma_min: float, sigma_max: float, device: Any):
    sigmas = with_final_zero(values, device)
    _refuse_flat_step(label, sigmas, sigma_min, sigma_max)
    return sigmas


def cosine(n, sigma_min, sigma_max, device: Any = "cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    return _closed_form(LABELS["cosine"], xmath.cosine_values(int(n), sigma_min, sigma_max),
                        sigma_min, sigma_max, device)


def cosine_exponential(n, sigma_min, sigma_max, device: Any = "cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    return _closed_form(LABELS["cosine_exponential"], xmath.cosine_exponential_values(int(n), sigma_min, sigma_max),
                        sigma_min, sigma_max, device)


def phi(n, sigma_min, sigma_max, device: Any = "cpu"):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    return _closed_form(LABELS["phi"], xmath.phi_values(int(n), sigma_min, sigma_max),
                        sigma_min, sigma_max, device)


def _karras_dynamic(n: int, sigma_min: float, sigma_max: float, rho: float, device: Any, *, label: str,
                    rho_source: str):
    """Karras Dynamic on the sigma range passed, as float32 with the final 0, which must fall at every step;
    ``label`` and ``rho_source`` (where the rho comes from) go into the texts."""

    sigmas = with_final_zero(xmath.karras_dynamic_values(n, sigma_min, sigma_max, rho, name=label,
                                                         rho_source=rho_source), device)
    _refuse_karras_dynamic_rise(sigmas, n, rho, sigma_min, sigma_max, label=label, rho_source=rho_source)
    return sigmas


def karras_dynamic(n, sigma_min, sigma_max, rho: float = xmath.KARRAS_DYNAMIC_RHO, device: Any = "cpu"):
    """Karras Dynamic; the float32 schedule must fall at every step (``KarrasDynamicError``)."""

    sigma_min, sigma_max, rho, n = float(sigma_min), float(sigma_max), float(rho), int(n)
    return _karras_dynamic(n, sigma_min, sigma_max, rho, device, label=LABELS["karras_dynamic"],
                           rho_source=xmath.RHO_SOURCE)


def react_cosinusoidal_dynsf(n, sigma_min, sigma_max, device: Any = "cpu"):
    """React Cosinusoidal DynSF (``extra_schedulers_math.react_cosinusoidal_dynsf_values``, factor 2.15)
    as float32 with the final 0. Near sigma_max neighbouring float32 steps meet from about 448 steps on
    an Anima range — refused like any flat step."""

    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    return _closed_form(LABELS["react_cosinusoidal_dynsf"],
                        xmath.react_cosinusoidal_dynsf_values(int(n), sigma_min, sigma_max),
                        sigma_min, sigma_max, device)


def is_flow_sampling(model_sampling: Any) -> bool | None:
    """True for a flow model's ``model_sampling`` (rectified flow: ComfyUI's ``CONST``, or a Forge-style
    ``prediction_type == "const"``), False for any other, None without one — ``extra_samplers_dy.is_const``,
    the rule the pack's Dy and Restart entries use for sam-extra's ``prediction_type == "const"``. The
    carrier's private copy and the pack's flow shift keep the class, so they count as flow too."""

    if model_sampling is None:
        return None
    from .extra_samplers_dy import is_const

    return bool(is_const(model_sampling))


# Set the first time Flow Cosmos rho7 runs on a model that is not a flow model (one line per process).
_flow_cosmos_not_flow_logged = False


def _log_not_flow_once(rho: float) -> None:
    global _flow_cosmos_not_flow_logged
    if not _flow_cosmos_not_flow_logged:
        _flow_cosmos_not_flow_logged = True
        LOGGER.info(xmath.FLOW_COSMOS_NOT_FLOW_LOG.format(rho=rho))


def _refuse_unusable_flow_cosmos(sigmas, rho: float, where: str, label: str = "Flow Cosmos rho7") -> None:
    """Every sigma before the final 0 must be a finite float32 of at least ``FLOAT32_TINY`` (the custom
    scheduler's rule; samplers divide by sigma). Values past the node's limits get here — a tiny ρ or σ̃ min
    cancels the ramp's end to 0, which the flat-step rule does not see when only the last sigma is 0, and a
    tiny ρ can make the float32 Karras NaN — and, within them, an eps/v MODEL whose own sigma_min is so small
    that the float32 Karras ends on 0 or a subnormal. Flow Cosmos Dynamic (``label``) applies it on a flow model
    only: on an eps/v one it is Karras Dynamic, which has no such rule."""

    torch = require_torch()
    head = sigmas[:-1]
    bad = torch.nonzero(~(torch.isfinite(head) & (head >= FLOAT32_TINY)))
    if bad.numel():
        step = int(bad[0])
        raise ExtraSchedulerError(
            f"Extra Schedulers: {label} gives sigma {float(head[step]):.6g} at step {step} ({where}; rho "
            f"{rho:g} — {xmath.FLOW_COSMOS_VALUES_SOURCE}), not usable as one of Forge's float32 sigmas (the "
            f"smallest usable value is {FLOAT32_TINY:.4g})"
        )


def flow_cosmos_rho7(n, sigma_min, sigma_max, model_sampling: Any = None, settings: Settings | None = None,
                     device: Any = "cpu", *, karras: Callable[..., Any] | None = None):
    """Flow Cosmos rho7 with the run's ρ / σ̃ max / σ̃ min (``settings``; None = the defaults, Cosmos-Predict2's
    ρ 7 on 0.002 … 80).

    On a flow model (``is_flow_sampling``): the flow times of the ρ ramp over σ̃ min … σ̃ max
    (``extra_schedulers_math.flow_cosmos_rho7_times``) as float32 with the final 0 — the model's own sigma range is
    not used. On any other model: the host's Karras (``host_get_sigmas_karras``; ``karras`` stands in for it in
    tests outside ComfyUI) with the run's ρ on the sigma range passed — the MODEL's — which with ρ 7 is ComfyUI's
    "karras" schedule for that MODEL; σ̃ is not used there; the first such call of the process logs
    ``FLOW_COSMOS_NOT_FLOW_LOG`` with that ρ. sam-extra's checks and texts, in its order: both ends of the range
    above 0; a ``model_sampling`` to tell the two apart; ρ finite and above 0; on a flow model 0 < σ̃ min < σ̃ max,
    both finite; an overflow of either branch's maths; every sigma before the final 0 a usable float32; no flat
    step (on a flow model the message names ρ and σ̃ instead of the sigma range)."""

    current = settings if settings is not None else xmath.DEFAULTS
    sigma_min, sigma_max, n = float(sigma_min), float(sigma_max), int(n)
    rho = float(current.flow_cosmos_rho)
    label = LABELS["flow_cosmos_rho7"]
    xmath.require_positive_range(label, sigma_min, sigma_max)
    flow = is_flow_sampling(model_sampling)
    if flow is None:
        raise ExtraSchedulerError(
            f"Extra Schedulers: {label} needs the MODEL's model_sampling to tell a flow model from an eps/v one; "
            "none was given"
        )
    xmath.require_flow_cosmos_rho(rho)
    if flow:
        ve_sigma_max, ve_sigma_min = float(current.flow_cosmos_sigma_max), float(current.flow_cosmos_sigma_min)
        xmath.require_flow_cosmos_range(ve_sigma_min, ve_sigma_max)
        try:
            times = xmath.flow_cosmos_rho7_times(n, rho, ve_sigma_min, ve_sigma_max)
        except OverflowError:
            raise ExtraSchedulerError(
                f"Extra Schedulers: {label} with rho {rho:g} on sigma~ {ve_sigma_min:g} … {ve_sigma_max:g} "
                f"overflows ({xmath.FLOW_COSMOS_VALUES_SOURCE})"
            ) from None
        sigmas = with_final_zero(times, device)
        _refuse_unusable_flow_cosmos(sigmas, rho, f"sigma~ {ve_sigma_min:g} … {ve_sigma_max:g}")
        _refuse_flat_step(label, sigmas, sigma_min, sigma_max, source=(
            f"rho {rho:g}, sigma~ {ve_sigma_min:g} … {ve_sigma_max:g} — {xmath.FLOW_COSMOS_VALUES_SOURCE}; "
            "the model's sigma range is not used on a flow model"))
    else:
        function = karras if karras is not None else host_get_sigmas_karras()
        _log_not_flow_once(rho)
        try:
            sigmas = function(n, sigma_min, sigma_max, rho=rho, device=device)
        except OverflowError:   # sigma ** (1 / rho) for a ρ past the node's limits, e.g. 14.6 ** 1000
            raise ExtraSchedulerError(
                f"Extra Schedulers: {label} with rho {rho:g} on sigma {sigma_min:g} … {sigma_max:g} overflows "
                f"in Forge's Karras ({xmath.FLOW_COSMOS_VALUES_SOURCE})"
            ) from None
        _refuse_unusable_flow_cosmos(sigmas, rho, f"Forge's Karras on sigma {sigma_min:g} … {sigma_max:g} — the "
                                                  "model's range or Settings → Sampler Parameters → sigma min / "
                                                  "sigma max")
        _refuse_flat_step(label, sigmas, sigma_min, sigma_max)
    return sigmas


# Set the first time Flow Cosmos Dynamic runs on a model that is not a flow model (its own line per process).
_flow_cosmos_dynamic_not_flow_logged = False


def _log_dynamic_not_flow_once(rho: float) -> None:
    global _flow_cosmos_dynamic_not_flow_logged
    if not _flow_cosmos_dynamic_not_flow_logged:
        _flow_cosmos_dynamic_not_flow_logged = True
        LOGGER.info(xmath.FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG.format(rho=rho))


def flow_cosmos_dynamic(n, sigma_min, sigma_max, model_sampling: Any = None, settings: Settings | None = None,
                        device: Any = "cpu"):
    """Flow Cosmos Dynamic with Flow Cosmos rho7's three values (``settings``; None = the defaults, ρ 7 on
    σ̃ 0.002 … 80) — sam-extra v0.33.0's composition of Karras Dynamic and Flow Cosmos rho7.

    On a flow model (``is_flow_sampling``): Karras Dynamic's per-step-ρ ramp over σ̃ min … σ̃ max in flow time
    (``extra_schedulers_math.flow_cosmos_dynamic_times``) as float32 with the final 0 — the model's own sigma range
    is not used. On any other model: Karras Dynamic itself (``karras_dynamic``'s values and checks, bit for bit)
    with the run's ρ on the sigma range passed — the MODEL's — under this scheduler's name, the node's values named
    as the rho's origin; σ̃ is not used there; the first such call of the process logs
    ``FLOW_COSMOS_DYNAMIC_NOT_FLOW_LOG`` with that ρ. sam-extra's checks and texts, in its order: both ends of the
    range above 0; a ``model_sampling`` to tell the two apart; ρ above 2 (``KarrasDynamicError``); on a flow model
    0 < σ̃ min < σ̃ max, both finite; every σ̃ finite (``KarrasDynamicError``); every sigma before the final 0 a
    usable float32; no flat step (the message names ρ and σ̃); every step falling (``KarrasDynamicError``)."""

    current = settings if settings is not None else xmath.DEFAULTS
    sigma_min, sigma_max, n = float(sigma_min), float(sigma_max), int(n)
    rho = float(current.flow_cosmos_rho)
    label = LABELS["flow_cosmos_dynamic"]
    source = xmath.FLOW_COSMOS_VALUES_SOURCE
    xmath.require_positive_range(label, sigma_min, sigma_max)
    flow = is_flow_sampling(model_sampling)
    if flow is None:
        raise ExtraSchedulerError(
            f"Extra Schedulers: {label} needs the MODEL's model_sampling to tell a flow model from an eps/v one; "
            "none was given"
        )
    xmath.require_karras_dynamic_rho(rho, label, source)
    if not flow:
        _log_dynamic_not_flow_once(rho)
        return _karras_dynamic(n, sigma_min, sigma_max, rho, device, label=label, rho_source=source)
    ve_sigma_max, ve_sigma_min = float(current.flow_cosmos_sigma_max), float(current.flow_cosmos_sigma_min)
    xmath.require_flow_cosmos_range(ve_sigma_min, ve_sigma_max, label)
    times, bad = xmath.flow_cosmos_dynamic_times(n, rho, ve_sigma_min, ve_sigma_max)
    if bad is not None:
        raise KarrasDynamicError(
            f"Extra Schedulers: {label} gave no finite sigma~ at step {bad} (rho {rho:g}, sigma~ "
            f"{ve_sigma_min:g} … {ve_sigma_max:g} — {source})"
        )
    sigmas = with_final_zero(times, device)
    ve_range = f"sigma~ {ve_sigma_min:g} … {ve_sigma_max:g}"
    _refuse_unusable_flow_cosmos(sigmas, rho, ve_range, label)
    _refuse_flat_step(label, sigmas, sigma_min, sigma_max, source=(
        f"rho {rho:g}, {ve_range} — {source}; the model's sigma range is not used on a flow model"))
    _refuse_karras_dynamic_rise(sigmas, n, rho, ve_sigma_min, ve_sigma_max, label=label, rho_source=source)
    return sigmas


def _refuse_karras_dynamic_rise(sigmas, n: int, rho: float, sigma_min: float, sigma_max: float, *,
                                label: str = "Karras Dynamic", rho_source: str = xmath.RHO_SOURCE) -> None:
    """A small per-step exponent lifts its step towards sigma_max: below rho ≈ 4 the schedule rises
    mid-run for most step counts. Checked on the float32 values the sampler gets. ``label`` / ``rho_source``: the
    scheduler's name and where its rho comes from (Flow Cosmos Dynamic: the node's values; on a flow model the
    range passed is its σ̃ range)."""

    torch = require_torch()
    head = sigmas[:-1]
    rising = torch.nonzero(head[1:] >= head[:-1])
    if rising.numel() == 0:
        return
    step = int(rising[0]) + 1
    before, after = float(head[step - 1]), float(head[step])
    if sigma_min >= sigma_max:
        relation = "a larger" if sigma_min > sigma_max else "the same"
        raise KarrasDynamicError(
            f"Extra Schedulers: {label} cannot fall from sigma_max {sigma_max:g} to {relation} sigma_min "
            f"{sigma_min:g} (Settings → Sampler Parameters → sigma min / sigma max)"
        )
    if after == before:
        change = f"sigma stays at {before:.6g} at step {step} ({xmath.ZERO_LENGTH_STEP})"
    else:
        change = f"sigma rises from {before:.4g} to {after:.4g} at step {step}"
    raise KarrasDynamicError(
        f"Extra Schedulers: {label} with rho {rho:g} does not fall over {n} steps — {change}. "
        f"Raise rho to about 4 or more: {rho_source}."
    )


def _laplace_over_range(n: int, sigma_min: float, sigma_max: float, mu: float, beta: float, device: Any):
    """The Laplace curve (Hang et al., arXiv:2407.03297, as ``get_sigmas_laplace`` writes it) at
    ``n`` evenly spaced positions between where it passes sigma_max and where it passes sigma_min:
    there every position has its own sigma."""

    torch = require_torch()
    start = xmath.laplace_position(sigma_max, mu, beta)
    end = xmath.laplace_position(sigma_min, mu, beta)
    x = torch.linspace(start, end, n, device=device)
    centred = 0.5 - x
    log_sigma = mu - beta * torch.sign(centred) * torch.log(1 - 2 * torch.abs(centred) + xmath.LAPLACE_EPSILON)
    return torch.clamp(torch.exp(log_sigma), min=sigma_min, max=sigma_max)


def laplace(n, sigma_min, sigma_max, settings: Settings | None = None, device: Any = "cpu", *,
            plain: Callable[..., Any] | None = None):
    """The host's ``get_sigmas_laplace`` with the run's mu/beta plus the final 0; where its clamp
    repeats a sigma and beta > 0, the curve is spread over the range instead; still flat -> error.

    ``plain`` stands in for the host function (tests outside ComfyUI).
    """

    torch = require_torch()
    current = settings if settings is not None else xmath.DEFAULTS
    n, sigma_min, sigma_max = int(n), float(sigma_min), float(sigma_max)
    mu, beta = float(current.laplace_mu), float(current.laplace_beta)
    plain_schedule = plain if plain is not None else host_get_sigmas_laplace()
    curve = plain_schedule(n, sigma_min, sigma_max, mu=mu, beta=beta, device=device)
    sigmas = torch.cat([curve, curve.new_zeros([1])])
    if beta > 0.0 and first_flat_step(sigmas) is not None:
        spread = _laplace_over_range(n, sigma_min, sigma_max, mu, beta, device)
        sigmas = torch.cat([spread, spread.new_zeros([1])])
    step = first_flat_step(sigmas)
    if step is not None:
        # beta 0 (one sigma, e^mu), a curve wholly outside the range, or more steps than float32 can
        # tell apart on the part inside it
        top, bottom = xmath.laplace_curve_ends(mu, beta)
        raise ExtraSchedulerError(
            f"Extra Schedulers: Laplace with mu {mu:g} and beta {beta:g} gives sigma {float(sigmas[step]):.6g} at "
            f"steps {step} and {step + 1} — its curve runs from {top:.4g} down to {bottom:.4g} (e^mu = "
            f"{math.exp(mu):.4g}) and too little of it lies between sigma_min {sigma_min:g} and sigma_max "
            f"{sigma_max:g} for {n} steps; {xmath.ZERO_LENGTH_STEP}. Raise beta, or move mu so that e^mu lies in "
            "that range."
        )
    return sigmas


def custom(n, sigma_min, sigma_max, settings: Settings | None = None, device: Any = "cpu", *,
           interp: xmath.Interpolation | None = None):
    """The run's expression or sigma list as a float32 schedule (``CustomSchedulerError`` when it
    cannot be used). ``interp`` stands in for the host interpolation (tests outside ComfyUI)."""

    current = settings if settings is not None else xmath.DEFAULTS
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    uses_list = current.custom_mode == xmath.MODE_SIGMAS
    what = "sigma list" if uses_list else "expression"
    try:
        if uses_list:
            values = xmath.sigmas_from_list(current.custom_sigmas, int(n), sigma_min, sigma_max,
                                            interp=interp if interp is not None else host_loglinear_interp)
        else:
            values = xmath.sigmas_from_expression(current.custom_expression, int(n), sigma_min, sigma_max)
    except (ExpressionError, SigmaListError) as exc:
        raise CustomSchedulerError(f"Extra Schedulers: the custom scheduler's {what} cannot be used: {exc}") from None
    for step, value in enumerate(values):
        if value > FLOAT32_MAX:
            raise CustomSchedulerError(
                f"Extra Schedulers: the custom scheduler's {what} gives sigma {value:g} at step {step}, "
                "too large for Forge's float32 sigmas"
            )
        if value < FLOAT32_TINY:
            raise CustomSchedulerError(
                f"Extra Schedulers: the custom scheduler's {what} gives sigma {value:g} at step {step}, "
                f"too small for Forge's float32 sigmas (the smallest usable value is {FLOAT32_TINY:.4g})"
            )
    sigmas = with_final_zero(values, device)
    step = first_flat_step(sigmas)   # on the float32 values: two close float64 values can become one
    if step is not None:
        raise CustomSchedulerError(
            f"Extra Schedulers: the custom scheduler's {what} gives sigma {float(sigmas[step]):.6g} at steps "
            f"{step} and {step + 1}; {xmath.ZERO_LENGTH_STEP}"
        )
    return sigmas


def schedule(name: str, n, sigma_min, sigma_max, settings: Settings | None = None, device: Any = "cpu", *,
             model_sampling: Any = None):
    """The float32 schedule of scheduler ``name`` (one of ``SCHEDULER_NAMES``), as Forge builds it
    for ``n`` steps; Karras Dynamic with rho 7 (Forge's Settings rho does not exist in ComfyUI). Flow
    Cosmos rho7 and Flow Cosmos Dynamic read the kind of model from ``model_sampling`` and their ρ / σ̃ from
    ``settings``."""

    if name == "laplace":
        return laplace(n, sigma_min, sigma_max, settings, device)
    if name == "custom":
        return custom(n, sigma_min, sigma_max, settings, device)
    if name == "flow_cosmos_rho7":
        return flow_cosmos_rho7(n, sigma_min, sigma_max, model_sampling, settings, device)
    if name == "flow_cosmos_dynamic":
        return flow_cosmos_dynamic(n, sigma_min, sigma_max, model_sampling, settings, device)
    function = _SETTINGS_FREE.get(name)
    if function is None:
        raise KeyError(f"Extra Schedulers has no scheduler {name!r}")
    return function(n, sigma_min, sigma_max, device=device)


_SETTINGS_FREE: dict[str, Callable[..., Any]] = {
    "cosine": cosine,
    "cosine_exponential": cosine_exponential,
    "phi": phi,
    "karras_dynamic": karras_dynamic,
    "react_cosinusoidal_dynsf": react_cosinusoidal_dynsf,
}


# ── ComfyUI scheduler handlers ────────────────────────────────────────────────────────────────

def as_settings(value: Any) -> Settings:
    """The carried value as ``Settings`` (a ``Settings``, a dict of the block, or nothing = defaults)."""

    if isinstance(value, Settings):
        return value
    if isinstance(value, Mapping):
        return xmath.coerce(value)
    return xmath.DEFAULTS


def carried_settings(model_sampling: Any) -> Settings:
    """The run's values on ``model_sampling`` (its own attribute ``CARRIER_ATTR``), else the defaults."""

    return as_settings(own_attribute(model_sampling, CARRIER_ATTR, None))


def sigma_range(model_sampling: Any) -> tuple[float, float]:
    """``(sigma_min, sigma_max)`` exactly as ``comfy.samplers.calculate_sigmas`` passes them."""

    return float(model_sampling.sigma_min), float(model_sampling.sigma_max)


def handler_function(name: str) -> Callable[[Any, int], Any]:
    """The ``SchedulerHandler(..., use_ms=True)`` function for scheduler ``name``."""

    if name not in SCHEDULER_NAMES:
        raise KeyError(f"Extra Schedulers has no scheduler {name!r}")

    def extra_scheduler(model_sampling, steps):
        sigma_min, sigma_max = sigma_range(model_sampling)
        settings = carried_settings(model_sampling) if name in VALUE_SCHEDULERS else xmath.DEFAULTS
        sigmas = schedule(name, steps, sigma_min, sigma_max, settings, model_sampling=model_sampling)
        if os.environ.get(DEBUG_ENV, "").strip() == "1":
            LOGGER.info(
                "Extra Schedulers %s: steps=%s sigma_min=%r sigma_max=%r settings=%r sigmas=%s",
                name, steps, sigma_min, sigma_max, tuple(settings) if name in VALUE_SCHEDULERS else (),
                [float(value) for value in sigmas],
            )
        return sigmas

    extra_scheduler.__name__ = extra_scheduler.__qualname__ = f"extra_scheduler_{name}"
    return extra_scheduler


# ── the carrier: the run's values on a private copy of the MODEL's model_sampling ───────────────

def _private_sampling_copy(sampling: Any) -> Any:
    """A shallow copy of ``sampling`` of the same class (``isinstance(ms, CONST)`` checks hold) that
    owns its ``_buffers``/``_parameters``/``_modules`` dicts: the tensors are shared, but moving the
    copy to another device replaces the entries of its own dicts and never touches the original."""

    duplicate = copy.copy(sampling)
    namespace = getattr(duplicate, "__dict__", None)
    if isinstance(namespace, dict):
        for name in ("_buffers", "_parameters", "_modules"):
            table = namespace.get(name)
            if isinstance(table, dict):
                namespace[name] = type(table)(table)
    return duplicate


def carry(model: Any, settings: Settings | Mapping[str, Any]) -> Any:
    """A clone of ``model`` whose ``model_sampling`` (a private copy) carries ``settings``.

    ``ModelPatcher.get_model_object`` honours object patches and ``clone`` keeps them, so every
    sampler downstream — the pack's nodes, stock ``KSampler``/``KSamplerAdvanced``, Impact's
    module-level ``calculate_sigmas``, ``BasicScheduler`` — schedules with these values, like Forge's
    one state per generation. A later node that installs a new ``model_sampling`` drops them, except
    the pack's own flow shift (it copies ``CARRIER_ATTR``, registered below).
    """

    values = as_settings(settings)
    clone = clone_model(model, "Extra Schedulers")
    sampling = clone.get_model_object("model_sampling")
    duplicate = _private_sampling_copy(sampling)
    setattr(duplicate, CARRIER_ATTR, values)
    clone.add_object_patch("model_sampling", duplicate)
    options = getattr(clone, "model_options", None)
    if isinstance(options, dict):
        options[MODEL_OPTIONS_KEY] = dict(values._asdict())   # informational (debugging, graph dumps)
    return clone


def model_settings(model: Any) -> Settings | None:
    """The values carried by ``model``'s ``model_sampling``, or None when it carries none."""

    get_model_object = getattr(model, "get_model_object", None)
    if not callable(get_model_object):
        return None
    try:
        sampling = get_model_object("model_sampling")
    except AttributeError:
        return None
    value = own_attribute(sampling, CARRIER_ATTR, None)
    return None if value is None else as_settings(value)


def _carry_to_override(source_model: Any, target_model: Any, target_vae: Any) -> Any:
    """Hires checkpoint/UNET override: the pass keeps the generation's values (Forge keeps one
    state for the whole generation, whatever checkpoint the Hires pass uses)."""

    settings = model_settings(source_model)
    if settings is None:
        return None
    return carry(target_model, settings)


# ── registration in ComfyUI's scheduler table ────────────────────────────────────────────────

def _registry() -> Any:
    try:
        return importlib.import_module("comfy.samplers")
    except ModuleNotFoundError:
        return None


def _handler_entry_factory(scheduler_handler: Callable[..., Any], name: str) -> Callable[[], Any]:
    def build():
        return scheduler_handler(mark_owned(handler_function(name), OWNER), use_ms=True)

    return build


def install_extra_schedulers(*, required: bool = False) -> NameRegistration:
    """Add the nine schedulers to ComfyUI's ``SCHEDULER_HANDLERS`` and its published choice lists
    (``SCHEDULER_NAMES``, ``KSampler.SCHEDULERS``), never replacing a name another provider holds
    (``compat.register_names``: skipped, logged once, not published — the app then refuses a graph
    that needs that name). Called once when the pack is imported; again is harmless. Outside ComfyUI
    (the app's tests) it does nothing. Without ``required`` it never raises: a host it cannot
    register in is logged and nothing is owned (the app's owner check then names the problem)."""

    try:
        return _install(required)
    except Exception:
        if required:
            raise
        LOGGER.exception("AI Studio pack: Extra Schedulers were not registered in ComfyUI.")
        return NameRegistration()


def _install(required: bool) -> NameRegistration:
    comfy_samplers = _registry()
    if comfy_samplers is None:
        if required:
            raise RuntimeError("Extra Schedulers registration requires a ComfyUI host.")
        return NameRegistration()
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    scheduler_handler = getattr(comfy_samplers, "SchedulerHandler", None)
    if not isinstance(handlers, dict) or not callable(scheduler_handler):
        if required:
            raise RuntimeError("This ComfyUI version has no extensible scheduler registry (SCHEDULER_HANDLERS).")
        return NameRegistration()
    ksampler = getattr(comfy_samplers, "KSampler", None)
    return register_names(
        handlers,
        {name: _handler_entry_factory(scheduler_handler, name) for name in SCHEDULER_NAMES},
        owner=OWNER,
        name_lists=(getattr(comfy_samplers, "SCHEDULER_NAMES", None), getattr(ksampler, "SCHEDULERS", None)),
        label="scheduler",
        required=required,
    )


def owned_names() -> tuple[str, ...]:
    """The nine names whose ComfyUI entry is this pack's and that KSampler offers — read live, so a
    provider that replaced an entry after the pack's import is not reported as the pack's."""

    comfy_samplers = _registry()
    handlers = getattr(comfy_samplers, "SCHEDULER_HANDLERS", None)
    offered = getattr(getattr(comfy_samplers, "KSampler", None), "SCHEDULERS", None)
    if not isinstance(handlers, dict) or not isinstance(offered, list):
        return ()
    return tuple(name for name in SCHEDULER_NAMES
                 if entry_owner(handlers.get(name)) == OWNER and name in offered)


# ── the node ─────────────────────────────────────────────────────────────────────────────────

class ForgeNeoExtraSchedulers:
    """MODEL -> MODEL: the Extra Schedulers values every later sampler of this MODEL uses."""

    CATEGORY = CATEGORY
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = (
        "Extra Schedulers (sam-extra parity): the values the 'laplace', 'custom', 'flow_cosmos_rho7' and "
        "'flow_cosmos_dynamic' schedulers use, carried on the MODEL so every later sampler (KSampler, Hires, "
        "ADetailer, SAM3) builds Forge's schedule. Place it after any ModelSampling node. The nine schedulers "
        "(cosine, cosine_exponential, phi, laplace, karras_dynamic, custom, react_cosinusoidal_dynsf, "
        "flow_cosmos_rho7, flow_cosmos_dynamic) are in every KSampler's scheduler list; the other five need no "
        "values, and without this node the two Flow Cosmos schedulers use their defaults (rho 7 on sigma~ "
        "0.002-80). A custom expression or sigma list is checked when a 'custom' schedule is built, the Flow "
        "Cosmos values when a 'flow_cosmos_rho7' or 'flow_cosmos_dynamic' one is."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "custom_mode": (list(xmath.MODES), {
                    "default": xmath.MODE_EXPRESSION,
                    "tooltip": "custom: 'expression' (a formula of m, M, n, s, x) or 'sigmas' (a sigma list).",
                }),
                "custom_expression": ("STRING", {
                    "default": xmath.DEFAULT_EXPRESSION,
                    "tooltip": "custom expression: m=sigma_min, M=sigma_max, n=steps, s=step from 0, x=s/(n-1); "
                               "constants phi, pi, e; + - * / ** and functions such as exp, log, cos, min, max.",
                }),
                "custom_sigmas": ("STRING", {
                    "default": xmath.DEFAULT_SIGMAS,
                    "tooltip": "custom sigma list; a list from 1.0 to 0.0 is scaled between sigma_max and "
                               "sigma_min, any other list is used as sigmas (log-linear interpolation).",
                }),
                "laplace_mu": ("FLOAT", {
                    "default": xmath.LAPLACE_MU_DEFAULT, "min": xmath.LAPLACE_MU_MIN,
                    "max": xmath.LAPLACE_MU_MAX, "step": 0.1,
                }),
                "laplace_beta": ("FLOAT", {
                    "default": xmath.LAPLACE_BETA_DEFAULT, "min": xmath.LAPLACE_BETA_MIN,
                    "max": xmath.LAPLACE_BETA_MAX, "step": 0.1,
                }),
                "flow_cosmos_rho": ("FLOAT", {
                    "default": xmath.FLOW_COSMOS_RHO_DEFAULT, "min": xmath.FLOW_COSMOS_RHO_MIN,
                    "max": xmath.FLOW_COSMOS_RHO_MAX, "step": xmath.FLOW_COSMOS_RHO_STEP,
                    "tooltip": "flow_cosmos_rho7: rho of the Karras ramp (default 7). Flow models (Anima): laid "
                               "on sigma~ (VE noise level) and mapped to flow time t = sigma~/(1 + sigma~); other "
                               "models: ComfyUI's Karras with this rho on the model's own sigma range. "
                               "flow_cosmos_dynamic: the same with Karras Dynamic's per-step rho + 2cos(2πi/n) "
                               "(needs rho above 2, about 4 or more to fall at every step); other models: Karras "
                               "Dynamic with this rho.",
                }),
                "flow_cosmos_sigma_max": ("FLOAT", {
                    "default": xmath.FLOW_COSMOS_SIGMA_MAX_DEFAULT, "min": xmath.FLOW_COSMOS_SIGMA_MAX_MIN,
                    "max": xmath.FLOW_COSMOS_SIGMA_MAX_MAX, "step": xmath.FLOW_COSMOS_SIGMA_MAX_STEP,
                    "tooltip": "flow_cosmos_rho7 / flow_cosmos_dynamic on a flow model: sigma~ max, the VE noise "
                               "level of the first step (default 80 = Cosmos-Predict2). Scaling both sigma~ by k is "
                               "the flow shift k (Anima's shift 3 = sigma~ 0.006-240). Not used on eps/v models.",
                }),
                "flow_cosmos_sigma_min": ("FLOAT", {
                    "default": xmath.FLOW_COSMOS_SIGMA_MIN_DEFAULT, "min": xmath.FLOW_COSMOS_SIGMA_MIN_MIN,
                    "max": xmath.FLOW_COSMOS_SIGMA_MIN_MAX, "step": xmath.FLOW_COSMOS_SIGMA_MIN_STEP,
                    "tooltip": "flow_cosmos_rho7 / flow_cosmos_dynamic on a flow model: sigma~ min, the VE noise "
                               "level of the last step (default 0.002); must stay below sigma~ max. Not used on "
                               "eps/v models.",
                }),
            },
            "optional": {
                OWNED_NAMES_INPUT: owned_names_input(
                    owned_names(),
                    tooltip=(
                        "Schedulers this node pack registered in ComfyUI (read by AI Studio; leave as it is). "
                        f"{NO_OWNED_NAMES} = another provider holds them or ComfyUI has no scheduler table."
                    ),
                ),
            },
        }

    def patch(self, model, custom_mode=xmath.MODE_EXPRESSION, custom_expression=xmath.DEFAULT_EXPRESSION,
              custom_sigmas=xmath.DEFAULT_SIGMAS, laplace_mu=xmath.LAPLACE_MU_DEFAULT,
              laplace_beta=xmath.LAPLACE_BETA_DEFAULT, flow_cosmos_rho=xmath.FLOW_COSMOS_RHO_DEFAULT,
              flow_cosmos_sigma_max=xmath.FLOW_COSMOS_SIGMA_MAX_DEFAULT,
              flow_cosmos_sigma_min=xmath.FLOW_COSMOS_SIGMA_MIN_DEFAULT, **_informational):
        settings = xmath.coerce({
            "custom_mode": custom_mode,
            "custom_expression": custom_expression,
            "custom_sigmas": custom_sigmas,
            "laplace_mu": laplace_mu,
            "laplace_beta": laplace_beta,
            "flow_cosmos_rho": flow_cosmos_rho,
            "flow_cosmos_sigma_max": flow_cosmos_sigma_max,
            "flow_cosmos_sigma_min": flow_cosmos_sigma_min,
        })
        return (carry(model, settings),)


NODE_CLASS_MAPPINGS = {"ForgeNeoExtraSchedulers": ForgeNeoExtraSchedulers}
NODE_DISPLAY_NAME_MAPPINGS = {"ForgeNeoExtraSchedulers": "Forge Neo Extra Schedulers (sam-extra)"}


# The pack's flow shift keeps the carrier on every pass; a Hires override model gets the values too.
hires_carry.register_sampling_attr(CARRIER_ATTR)
hires_carry.register(_carry_to_override, key=OWNER)
