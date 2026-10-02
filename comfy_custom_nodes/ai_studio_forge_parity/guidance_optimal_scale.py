"""Anima Optimal Scale (``ForgeNeoAnimaOptimalScale``) -- the optimized-scale part of CFG-Zero*.

Mirrors sam-extra v0.30.0's experimental ``scripts/anima_cfg_optimal_scale.py`` (default off): the
optimized-scale equation of CFG-Zero* (arXiv 2503.18886) only, **without zero-init**. With Anima's
``x0 = x - sigma*v`` the optimized-scale CFG ``v = s*v_u + w(v_c - s*v_u)``, ``s = <v_c, v_u>/|v_u|^2``
differs from plain CFG by ``(w - 1)(s - 1)*sigma*v_u``, i.e. in x0 space by ``(w - 1)(s - 1)*r_u`` with
the residuals ``r = x - x0`` (sigma cancels in ``s``; the epsilon is scaled by ``sigma**2`` so it
matches the v-space one). That correction times ``blend`` is added to the incoming result.

The extension's safety rules, read the same way here (each skip leaves the result unchanged):

- another CFG function is set (``sampler_cfg_function`` -- here the guidance suite's SMC/APG/CWM stage;
  in Forge that stage recomputes the base from cond/uncond, so the correction is not kept there either);
- Skimmed CFG is on (the pack's vendored pre-CFG node);
- CFG <= 1, no uncond pass, identical predictions, a non-flow (non-CONST) model, a sigma batch mismatch;
- the sigma is outside ``[percent_to_sigma(end), percent_to_sigma(start)]``;
- the incoming result is not the linear CFG of the predictions (an earlier post-CFG correction).

The app compiler puts this node before the guidance suite, so its post-CFG function runs first, like
the extension's (its callback attaches before Safe PAG's). The node only patches Anima MODELs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .compat import clone_model, require_torch
from .guidance_common import CATEGORY
from .guidance_detail import append_post_cfg

LOGGER = logging.getLogger("ai_studio_forge_parity")

DEFAULT_BLEND = 0.25
_SKIMMED_MODULE = "skimmed_cfg"   # vendor/skimmed_cfg (ForgeNeoSkimmedCFG's pre-CFG function)


def optimal_scale_residual(incoming: Any, x: Any, cond_x0: Any, uncond_x0: Any, scale: float, blend: float,
                           sigma: Any) -> Any:
    """``incoming`` plus the optimized-scale residual (x0 space)."""
    if blend <= 0.0:
        return incoming
    cond_residual = x.float() - cond_x0.float()
    uncond_residual = x.float() - uncond_x0.float()
    dims = tuple(range(1, x.ndim))
    product = (cond_residual * uncond_residual).sum(dim=dims, keepdim=True)
    norm = (uncond_residual * uncond_residual).sum(dim=dims, keepdim=True)
    sigma = sigma.reshape(-1, *([1] * (x.ndim - 1))).float()
    alpha = product / (norm + 1e-8 * sigma * sigma)
    correction = (scale - 1.0) * (alpha - 1.0) * uncond_residual * blend
    return incoming + correction.to(dtype=incoming.dtype)


def _is_flow(model: Any) -> bool:
    sampling = getattr(model, "model_sampling", None)
    return sampling is not None and "CONST" in {cls.__name__ for cls in type(sampling).__mro__}


def _skimmed_active(options: dict[str, Any]) -> bool:
    return any(_SKIMMED_MODULE in str(getattr(fn, "__module__", "") or "")
               for fn in (options.get("sampler_pre_cfg_function") or []))


def _edit_strength(conditions: Any) -> float:
    total = 0.0
    for item in conditions or []:
        total += float(item.get("strength", 1.0)) if isinstance(item, dict) else 1.0
    return total


def make_callback(blend: float, start: float, end: float, counts: dict[str, Any]):
    """One post-CFG function; ``counts`` collects applied/skipped and the last skip reason."""
    torch = require_torch()

    def skip(incoming: Any, reason: str):
        counts["skipped"] += 1
        counts["last_skip"] = reason
        return incoming

    def optimal_scale(args: dict[str, Any]):
        incoming = args["denoised"]
        try:
            options = args.get("model_options") or {}
            if options.get("sampler_cfg_function") is not None:
                return skip(incoming, "custom CFG function")
            if _skimmed_active(options):
                return skip(incoming, "Skimmed CFG modifies the prediction pair")
            cfg = float(args["cond_scale"])
            if not math.isfinite(cfg) or cfg <= 1.0 + 1e-6:
                return skip(incoming, "CFG <= 1 or skipped negative step")
            x, cond, uncond = args.get("input"), args.get("cond_denoised"), args.get("uncond_denoised")
            if not all(torch.is_tensor(t) for t in (incoming, x, cond, uncond)):
                return skip(incoming, "missing predictions")
            if x.ndim < 2 or not (x.shape == cond.shape == uncond.shape == incoming.shape):
                return skip(incoming, "prediction shape mismatch")
            if not args.get("uncond") or torch.equal(cond, uncond):
                return skip(incoming, "negative not evaluated or identical predictions")
            model = args.get("model")
            if not _is_flow(model):
                return skip(incoming, "non-const prediction")
            sigmas = args["sigma"].reshape(-1)
            sigma = float(sigmas[0])
            if sigmas.shape[0] not in (1, x.shape[0]):
                return skip(incoming, "sigma batch mismatch")
            sampling = model.model_sampling
            sigma_start = float(sampling.percent_to_sigma(start))
            sigma_end = float(sampling.percent_to_sigma(end))
            if not math.isfinite(sigma) or sigma <= 0.0:
                return skip(incoming, "invalid sigma")
            if not sigma_end <= sigma <= sigma_start:
                return skip(incoming, "outside sigma window")
            strength = _edit_strength(args.get("cond"))
            if not math.isfinite(strength) or strength <= 0.0:
                return skip(incoming, "invalid edit strength")
            linear = uncond + (cond - uncond) * (cfg * strength)
            if not torch.allclose(incoming, linear, rtol=1e-5, atol=1e-6):
                return skip(incoming, "prior post-CFG correction (APG/PAG/DCW/etc.)")
            result = optimal_scale_residual(incoming, x, cond, uncond, cfg * strength, blend, args["sigma"])
            if not bool(torch.isfinite(result).all()):
                return skip(incoming, "non-finite result")
            counts["applied"] += 1
            return result
        except Exception as exc:   # noqa: BLE001 - keep the incoming result on an unexpected contract
            return skip(incoming, type(exc).__name__)

    return optimal_scale


class ForgeNeoAnimaOptimalScale:
    """sam-extra's 'Anima Optimal Scale' (experimental): inputs, defaults and ranges are the extension's."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False}),
            "blend": ("FLOAT", {"default": DEFAULT_BLEND, "min": 0.0, "max": 1.0, "step": 0.05,
                                "tooltip": "1 = the optimized-scale equation as is (with plain CFG)."}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, blend=DEFAULT_BLEND, start_percent=0.0, end_percent=1.0):
        if not enabled:
            return (model,)
        try:
            blend = min(1.0, max(0.0, float(blend)))
            start, end = float(start_percent), float(end_percent)
        except (TypeError, ValueError):
            return (model,)
        if not (math.isfinite(blend) and blend > 0.0):
            return (model,)   # the extension's no-op contract for blend 0
        if not (math.isfinite(start) and math.isfinite(end) and 0.0 <= start < end <= 1.0):
            return (model,)
        if type(getattr(model, "model", None)).__name__ != "Anima":
            LOGGER.info("Anima Optimal Scale: not an Anima MODEL; left unchanged.")
            return (model,)
        patched = clone_model(model, "Anima Optimal Scale")
        counts: dict[str, Any] = {"applied": 0, "skipped": 0, "last_skip": ""}
        append_post_cfg(patched, make_callback(blend, start, end, counts))
        return (patched,)
