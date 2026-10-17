"""Extra Samplers for ComfyUI (pack 1.7.0): when Euler (SMEA) Dy (CFG++) runs without its sub-steps.

The Dy/SMEA sub-steps evaluate the model at another resolution (half, ×1.25). ``extra_samplers_dy``'s
``SubstepInputs`` resizes what ComfyUI's sampler itself holds at the latent's size (the denoise mask, the
inpainting latent and noise). Some full-resolution machinery cannot follow a sub-step, so — like
sam-extra on Forge, which skips the sub-steps for Spectrum Integrated, Wan I2V ``concat_latent``, PiD and
``extra_concat_condition`` — the sampler then runs its plain (CFG++) Euler steps instead of failing:

* ``REASON_FLAG`` — the MODEL carries ``transformer_options[DY_SUBSTEPS_OFF_KEY]``:
  ``ForgeNeoKSamplerCNS`` sets it for ComfyUI Spectrum (``spectrum_enabled``) and the legacy SPEED
  provider (``speed_enabled``), which forecast whole steps; the value names the feature. (The pack's
  ``ForgeNeoAnimaSpeed`` never sets it: sam-extra keeps the sub-steps under SPEED and places them by
  ``sam_extra_step_offset``.)
* ``REASON_SPECTRUM`` — a ``model_function_wrapper`` from a Spectrum provider (its name or module says
  ``spectrum`` — sam-extra's detection rule), e.g. a user workflow's own Spectrum patch, or the pack's MOSS
  Spectrum (pack 1.9.0), found by its own APPLY_MODEL / OUTER_SAMPLE wrappers
  (``guidance_common.spectrum_wrapper_registered``; other patcher-extension wrappers that merely say
  ``spectrum``, such as upstream Spectrum-KSampler's modulation guidance, do not count).
* ``REASON_CONCAT`` — concat conditioning (``c_concat`` / ``concat_latent_image``: inpainting
  checkpoints, Flux Fill, Wan I2V): the model concatenates it with the latent on the channel axis at
  full size (``model_base._apply_model``), a size mismatch at the sub-step's resolution.
* ``REASON_REGIONAL`` — a cond ``mask`` (regional masks are resized once when sampling starts) or
  ``area`` (full-resolution coordinates).

Each reason is logged once per ComfyUI session. No third-party code; the rules follow sam-extra's
``substep_guard`` (sam-extra's code, licensed to this pack under MIT by its author (THIRD_PARTY_NOTICES.md)) for
what Forge has and ComfyUI's sampler structure for the rest. No torch or ComfyUI import at module level.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from .extra_samplers_dy import SubstepInputs
from .guidance_common import DY_SUBSTEPS_OFF_KEY, spectrum_wrapper_registered

LOGGER = logging.getLogger("ai_studio_forge_parity")

REASON_FLAG = "flag"
REASON_SPECTRUM = "spectrum"
REASON_CONCAT = "concat"
REASON_REGIONAL = "regional"

_EXPLANATIONS = {
    REASON_FLAG: "{flag} is on for this MODEL (it forecasts whole steps)",
    REASON_SPECTRUM: "a Spectrum model wrapper forecasts whole steps",
    REASON_CONCAT: "concat conditioning (c_concat: inpainting / fill models, Wan I2V) is joined to the latent at "
                   "full size",
    REASON_REGIONAL: "area / mask conditioning is tied to the full latent grid",
}
_CONCAT_KEYS = ("c_concat", "concat_latent_image")
_logged: set[str] = set()


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _transformer_options(extra_args: Any) -> Mapping:
    return _mapping(_mapping(_mapping(extra_args).get("model_options")).get("transformer_options"))


def _model_function_wrapper(extra_args: Any) -> Any:
    return _mapping(_mapping(extra_args).get("model_options")).get("model_function_wrapper")


def processed_conds(model: Any) -> list[Mapping]:
    """Every cond dict of the run: the sampler's ``model`` is ComfyUI's ``KSamplerX0Inpaint`` whose
    ``inner_model`` is the ``CFGGuider`` holding the processed ``conds``."""
    conds = getattr(getattr(model, "inner_model", None), "conds", None)
    found: list[Mapping] = []
    for value in _mapping(conds).values():
        if isinstance(value, (list, tuple)):
            found.extend(item for item in value if isinstance(item, Mapping))
    return found


def is_spectrum_wrapper(function: Any) -> bool:
    if function is None:
        return False
    name = getattr(function, "__qualname__", None) or getattr(function, "__name__", None) or type(function).__qualname__
    module = getattr(function, "__module__", None) or type(function).__module__
    return "spectrum" in f"{module}.{name}".lower()


def blocking_reasons(model: Any, extra_args: Any) -> list[str]:
    """Why the sub-steps cannot run on this sampling call (in a fixed order), or ``[]``."""
    reasons: list[str] = []
    if _transformer_options(extra_args).get(DY_SUBSTEPS_OFF_KEY):
        reasons.append(REASON_FLAG)
    if is_spectrum_wrapper(_model_function_wrapper(extra_args)) or spectrum_wrapper_registered(
            _mapping(extra_args).get("model_options")):
        reasons.append(REASON_SPECTRUM)
    conds = processed_conds(model)
    if any(key in cond or key in _mapping(cond.get("model_conds")) for cond in conds for key in _CONCAT_KEYS):
        reasons.append(REASON_CONCAT)
    if any(cond.get("mask") is not None or cond.get("area") is not None for cond in conds):
        reasons.append(REASON_REGIONAL)
    return reasons


def explain(reason: str, extra_args: Any = None) -> str:
    flag = _transformer_options(extra_args).get(DY_SUBSTEPS_OFF_KEY)
    return _EXPLANATIONS[reason].format(flag=flag if isinstance(flag, str) and flag else "a whole-step forecaster")


def _log_once(name: str, reason: str, extra_args: Any, cfg_pp: bool) -> None:
    key = f"{name}:{reason}"
    if key in _logged:
        return
    _logged.add(key)
    steps = "plain CFG++ Euler steps" if cfg_pp else "plain Euler steps"
    LOGGER.warning("AI Studio pack: %s: sub-steps skipped - %s; it runs %s instead.", name,
                   explain(reason, extra_args), steps)


def guarded(function: Callable[..., Any], name: str, *, cfg_pp: bool) -> Callable[..., Any]:
    """The ComfyUI sampler function of one Dy entry: ``function`` with ComfyUI's ``SubstepInputs`` and
    ``substeps=False`` when ``blocking_reasons`` finds any (each logged once)."""

    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0,
                s_tmax=float("inf"), s_noise=1.0, noise_sampler=None, substeps=True, sam_extra_step_offset=0):
        reasons = blocking_reasons(model, extra_args) if substeps else []
        for reason in reasons:
            _log_once(name, reason, extra_args, cfg_pp)
        return function(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_churn=s_churn,
            s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, noise_sampler=noise_sampler,
            substeps=bool(substeps) and not reasons, sam_extra_step_offset=sam_extra_step_offset,
            substep_inputs=SubstepInputs,
        )

    sampler.__name__ = sampler.__qualname__ = getattr(function, "__name__", "sample_" + name)
    sampler.__doc__ = getattr(function, "__doc__", None)
    return sampler
