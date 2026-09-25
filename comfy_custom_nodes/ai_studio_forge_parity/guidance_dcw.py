"""DCW/RDC, CWM, SMC and the pack's APG (``ForgeNeoDCWCWMSMC``).

The behaviour follows namemechan/ComfyUI-DCW@66aaf9dd (``DCWModelPatch``,
shown as "DCW(+a)"). That project is GPL-3.0, so nothing here is copied from
it: this is a separate implementation of the same behaviour, checked against
numbers produced by running the original
(``tests/fixtures/dcw_origin_golden.json``, ``tests/test_dcw_origin_parity.py``).

Behaviour that matches the original node:

- CWM and SMC replace the CFG combination through ``sampler_cfg_function``.
  They work on Comfy's noise-space ``cond``/``uncond`` arguments and set
  ``disable_cfg1_optimization``, so they also run at CFG 1.
- DCW and RDC are appended to ``sampler_post_cfg_function``. They run after
  whatever post-CFG hooks earlier nodes added.
- Cross-step state lives in the sampling run's ``model_options``, under
  ``_dcw_smc_state`` and ``_dcw_rdc_state``. Comfy copies ``model_options`` for
  every sampling run, so each run (and each hires pass) starts fresh. There is
  no sigma-rise reset inside a run.
- If an earlier node already set ``sampler_cfg_function``, CWM/SMC are skipped
  with a warning and DCW is still added.
- An error in one step falls back for that step only: plain CFG for the CFG
  hook, the unchanged prediction for DCW.
- The math runs in the tensor's own dtype (fp8 is lifted to bf16). SMC runs in
  fp32.

APG is this pack's addition for Forge-extension parity. It rides in the same
CFG hook (after SMC, before CWM), keeps its state under its own key and is off
unless the guidance suite asks for it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from .compat import clone_model, require_torch
from .guidance_common import CATEGORY, _haar_dwt, _haar_idwt, _scalar_sigma


LOGGER = logging.getLogger("ai_studio_forge_parity")

# Keys in the per-run ``model_options`` that carry state between steps. The
# first two are the original node's key names, so state sharing behaves the
# same way as with DCW(+a).
SMC_STATE_KEY = "_dcw_smc_state"
RDC_STATE_KEY = "_dcw_rdc_state"
APG_STATE_KEY = "_forge_neo_apg_state"

_BANDS = ("LL", "LH", "HL", "HH")
_ENERGY_WEIGHT_LIMITS = (0.25, 4.0)
_NORM_FLOOR = 1e-8
_TAU_FLOOR = 1e-6
_FP8_NAMES = ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")


# ---------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------

def _work_dtype(dtype: Any) -> Any:
    """Arithmetic dtype for a tensor dtype: fp8 is lifted to bf16."""

    torch = require_torch()
    fp8 = {getattr(torch, name) for name in _FP8_NAMES if hasattr(torch, name)}
    return torch.bfloat16 if dtype in fp8 else dtype


def _finite(value: Any) -> Any:
    """Replace NaN and +/-inf with zero."""

    torch = require_torch()
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _noise_level(sigma: Any, like: Any, dtype: Any) -> Any:
    """``sigma / (sigma + 1)``.

    A tensor sigma is computed in fp32, shaped to broadcast over the batch of
    ``like`` and returned in ``dtype``. A plain number stays a Python float.
    """

    torch = require_torch()
    if not torch.is_tensor(sigma):
        value = float(sigma)
        return value / (value + 1.0)
    level = sigma.float() / (sigma.float() + 1.0)
    if level.ndim == 1:
        level = level.view(-1, *([1] * (like.ndim - 1)))
    return level.to(dtype=dtype, device=like.device)


def _reflect_even(value: Any):
    """Reflect-pad the last two dims to even sizes for the Haar transform."""

    torch = require_torch()
    height, width = value.shape[-2], value.shape[-1]
    if height % 2 or width % 2:
        padding = (0, width % 2, 0, height % 2) + (0, 0) * max(0, value.ndim - 4)
        value = torch.nn.functional.pad(value, padding, mode="reflect")
    return value, (height, width)


def _energy_weight(band: Any) -> Any:
    """Per-channel weight: channel energy over the mean channel energy.

    Clamped to [0.25, 4] and returned in the band's dtype.
    """

    dims = tuple(range(2, band.ndim))
    energy = band.float().pow(2).mean(dim=dims, keepdim=True)
    average = energy.mean(dim=1, keepdim=True).clamp(min=_NORM_FLOOR)
    low, high = _ENERGY_WEIGHT_LIMITS
    return (energy / average).clamp(low, high).to(band.dtype)


# ---------------------------------------------------------------------------
# DCW + RDC (post-CFG)
# ---------------------------------------------------------------------------

def apply_dcw(
    denoised: Any,
    live: Any,
    sigma: Any,
    lambda_low: float,
    lambda_high: float,
    *,
    rdc_tau: float = 0.0,
    rdc_alpha_low: float = 0.03,
    rdc_alpha_high: float = 0.0,
    state: dict[str, Any] | None = None,
):
    """Correct ``denoised`` toward the live latent in the Haar domain.

    For each band the correction is ``d + gain * w * (x - d)``, where ``w`` is
    the band's energy weight from the live latent. The gains are
    ``lambda_low * s`` for LL, ``lambda_high * (1 - s)`` for HH and their mean
    for LH/HL, with ``s = sigma / (sigma + 1)``.

    With ``rdc_tau > 0`` and a ``state`` dict, RDC then pulls each band toward
    its running average: ``b - alpha * (b - ema)``. The average moves by
    ``beta = 1 - exp(-|delta s| / tau)`` per step. The first step, or a change
    in shape or device, only records the band.

    With both lambdas at 0 and RDC off, ``denoised`` itself is returned.
    """

    tau = float(rdc_tau)
    recurrent = tau > 0.0
    low, high = float(lambda_low), float(lambda_high)
    if low == 0.0 and high == 0.0 and not recurrent:
        return denoised

    torch = require_torch()
    source_dtype = denoised.dtype
    dtype = _work_dtype(source_dtype)
    clean = denoised if dtype == source_dtype else denoised.to(dtype=dtype)
    level = _noise_level(sigma, clean, dtype)
    gain_low = low * level
    gain_high = high * (1.0 - level)
    gain_mid = (gain_low + gain_high) * 0.5

    clean_padded, (height, width) = _reflect_even(clean)
    live_padded, _ = _reflect_even(live.to(dtype=dtype, device=clean.device))
    bands = [
        clean_band + gain * _energy_weight(live_band) * (live_band - clean_band)
        for clean_band, live_band, gain in zip(
            _haar_dwt(clean_padded),
            _haar_dwt(live_padded),
            (gain_low, gain_mid, gain_mid, gain_high),
        )
    ]

    if recurrent and state is not None:
        level_now = float(level.mean()) if torch.is_tensor(level) else float(level)
        level_before = state.get("_s_prev", level_now)
        beta = 1.0 - math.exp(-abs(level_before - level_now) / max(tau, _TAU_FLOOR))
        state["_s_prev"] = level_now
        pull_low, pull_high = float(rdc_alpha_low), float(rdc_alpha_high)
        pull_mid = (pull_low + pull_high) * 0.5
        for index, (name, pull) in enumerate(
            zip(_BANDS, (pull_low, pull_mid, pull_mid, pull_high))
        ):
            if pull == 0.0:
                continue
            band = bands[index]
            frozen = band.detach()
            average = state.get(name)
            if (
                average is None
                or average.shape != frozen.shape
                or average.device != frozen.device
            ):
                state[name] = frozen.to(dtype=dtype).clone()
                continue
            average = (1.0 - beta) * average + beta * frozen
            state[name] = average.to(dtype=dtype)
            bands[index] = band - pull * (band - average)

    corrected = _haar_idwt(*bands)[..., :height, :width]
    return corrected if dtype == source_dtype else corrected.to(dtype=source_dtype)


# ---------------------------------------------------------------------------
# SMC, CWM and APG (CFG combination)
# ---------------------------------------------------------------------------

def _smc_error(error: Any, previous: Any, strength: float, k: float):
    """One sliding-mode correction of the guidance error (in fp32).

    ``surface = (e - e_prev) + strength * e_prev``. The step is
    ``-k * surface / ||surface||`` (L2 norm per sample), clamped to half the
    sample's mean ``|e|``. Returns ``(corrected, next_previous)``. A missing
    ``previous``, or one whose shape changed, starts from ``e`` itself.
    """

    torch = require_torch()
    current = error.to(dtype=torch.float32)
    if previous is None or previous.shape != current.shape:
        if previous is not None:
            LOGGER.info(
                "DCW(+a) SMC: latent shape changed %s -> %s; SMC state restarted.",
                list(previous.shape), list(current.shape),
            )
        reference = current.detach().clone()
    else:
        reference = _finite(previous.to(device=current.device, dtype=torch.float32))
    surface = _finite((current - reference) + strength * reference)
    dims = tuple(range(1, surface.ndim))
    length = torch.linalg.vector_norm(surface, dim=dims, keepdim=True).clamp(
        min=_NORM_FLOOR
    )
    step = -k * (surface / length)
    limit = (0.5 * current.abs().mean(dim=dims, keepdim=True)).clamp(min=_NORM_FLOOR)
    corrected = _finite(current + step.clamp(-limit, limit))
    return corrected, corrected.detach().clone()


def _cwm_error(error: Any, sigma: Any, scale: float, low: float, high: float):
    """Scale the guidance error per Haar band (CFG wavelet mixing).

    LL gets ``w * (1 + low * s)``, HH gets ``w * (1 + high * (1 - s))`` and
    LH/HL get their geometric mean, with ``w`` the CFG scale.
    """

    dtype = error.dtype
    level = _noise_level(sigma, error, dtype)
    weight = float(scale)
    weight_low = weight * (1.0 + float(low) * level)
    weight_high = weight * (1.0 + float(high) * (1.0 - level))
    weight_mid = (weight_low * weight_high) ** 0.5
    padded, (height, width) = _reflect_even(error)
    ll, lh, hl, hh = _haar_dwt(padded)
    return _haar_idwt(
        weight_low * ll, weight_mid * lh, weight_mid * hl, weight_high * hh
    )[..., :height, :width]


def _project_guidance(guidance: Any, direction: Any):
    torch = require_torch()
    dims = tuple(range(1, direction.ndim))
    normalized = torch.nn.functional.normalize(direction, dim=dims)
    parallel = (guidance * normalized).sum(dim=dims, keepdim=True) * normalized
    return parallel, guidance - parallel


def _apply_apg_error(
    guidance: Any,
    cond: Any,
    *,
    eta: float,
    norm_threshold: float,
    momentum: float,
    sigma: Any,
    state: dict[str, Any],
):
    torch = require_torch()
    work = torch.nan_to_num(guidance.float())
    current_sigma = _scalar_sigma(sigma)
    average = state.get("average")
    last_sigma = state.get("sigma")
    if float(momentum) != 0.0:
        if (
            average is None
            or average.shape != work.shape
            or (last_sigma is not None and current_sigma > float(last_sigma) + 1e-6)
        ):
            average = torch.zeros_like(work)
        average = float(momentum) * average.to(work) + work
        state["average"] = average.detach()
        work = average
    state["sigma"] = current_sigma
    threshold = float(norm_threshold)
    if threshold > 0.0:
        dims = tuple(range(1, work.ndim))
        norm = torch.linalg.vector_norm(work, dim=dims, keepdim=True).clamp_min(1e-8)
        work = work * (threshold / norm).clamp(max=1.0)
    parallel, orthogonal = _project_guidance(work, cond.float())
    return orthogonal + float(eta) * parallel


def _plain_cfg(cond: Any, uncond: Any, scale: float) -> Any:
    return uncond + scale * (cond - uncond)


def _guided_noise(
    cond: Any,
    uncond: Any,
    sigma: Any,
    scale: float,
    *,
    alpha_low: float,
    alpha_high: float,
    smc_lambda: float,
    smc_k: float,
    smc_state: dict[str, Any],
    apg: Callable[[Any], Any] | None = None,
) -> Any:
    """Combine noise-space ``cond``/``uncond`` the way the DCW(+a) hook does.

    Order: guidance error, SMC, (APG), then CWM or plain scaling. With SMC,
    CWM and APG all inactive this is exactly plain CFG.
    """

    smc_on = smc_lambda != 0.0 and smc_k != 0.0
    cwm_on = alpha_low != 0.0 or alpha_high != 0.0
    if not (smc_on or cwm_on or apg is not None):
        return _plain_cfg(cond, uncond, scale)

    source_dtype = cond.dtype
    dtype = _work_dtype(source_dtype)
    if dtype != source_dtype:
        cond, uncond = cond.to(dtype=dtype), uncond.to(dtype=dtype)
    error = _finite(cond - uncond)
    if smc_on:
        corrected, smc_state["e_prev"] = _smc_error(
            error, smc_state.get("e_prev"), smc_lambda, smc_k
        )
        error = corrected.to(dtype=dtype)
    if apg is not None:
        error = apg(error).to(dtype=dtype)
    if cwm_on:
        guided = _cwm_error(error, sigma, scale, alpha_low, alpha_high)
    else:
        guided = float(scale) * error
    combined = _finite(uncond + guided)
    return combined if dtype == source_dtype else combined.to(dtype=source_dtype)


# ---------------------------------------------------------------------------
# SMC presets and model-family detection
# ---------------------------------------------------------------------------

# (lambda, k) per model family, the CFG-Ctrl paper values offered by DCW(+a).
_SMC_PRESETS = {
    "SD1.5 / SD2": (5.0, 0.10),
    "SDXL": (5.0, 0.10),
    "SD3 / SD3.5": (6.0, 0.10),
    "Flux": (6.0, 0.70),
    "Qwen-Image": (6.0, 0.10),
    "Cosmos / Wan": (6.0, 0.20),
    "Custom": (6.0, 0.10),
}
SMC_PRESET_CHOICES = ["Off", "Auto", *_SMC_PRESETS]
_SMC_FALLBACK = "SD1.5 / SD2"

# Auto detection looks at three names in turn: the BaseModel class name, then
# its ``model_type`` and then the diffusion_model class name. Within each name
# the first matching rule wins. Nothing matched, or an error: SD1.5 / SD2.
_BASE_CLASS_RULES = (
    (("flux",), "Flux"),
    (("cosmos", "predict2", "wan", "anima"), "Cosmos / Wan"),
    (("sd3", "mmdit"), "SD3 / SD3.5"),
    (("sdxl",), "SDXL"),
    (("qwen",), "Qwen-Image"),
)
_MODEL_TYPE_RULES = (
    (("flow",), "SD3 / SD3.5"),
    (("v_pred",), "SD1.5 / SD2"),
)
_DIFFUSION_CLASS_RULES = (
    (("flux",), "Flux"),
    (("cosmos", "wan"), "Cosmos / Wan"),
    (("mmdit", "sd3"), "SD3 / SD3.5"),
    (("sdxl",), "SDXL"),
)


def _first_rule(text: str, rules) -> str | None:
    text = text.lower()
    for tokens, preset in rules:
        if any(token in text for token in tokens):
            return preset
    return None


def _detect_smc(model: Any) -> str:
    try:
        inner = model.model
        found = _first_rule(type(inner).__name__, _BASE_CLASS_RULES) or _first_rule(
            str(getattr(inner, "model_type", "")), _MODEL_TYPE_RULES
        )
        if found is None:
            diffusion = getattr(inner, "diffusion_model", None)
            if diffusion is not None:
                found = _first_rule(type(diffusion).__name__, _DIFFUSION_CLASS_RULES)
        return found or _SMC_FALLBACK
    except Exception:
        return _SMC_FALLBACK


def _canonical_preset(name: Any) -> str:
    text = str(name).strip()
    for choice in SMC_PRESET_CHOICES:
        if choice.casefold() == text.casefold():
            return choice
    raise ValueError(f"Unknown SMC preset {name!r}; expected one of {SMC_PRESET_CHOICES}.")


def resolve_smc(model: Any, preset: Any, smc_lambda: float, smc_k: float):
    """``(lambda, k)`` for a preset choice, or ``None`` when the preset is Off."""

    choice = _canonical_preset(preset)
    if choice == "Off":
        return None
    if choice == "Auto":
        family = _detect_smc(model)
        values = _SMC_PRESETS[family]
        LOGGER.info("DCW(+a) SMC: Auto picked %r (lambda=%s, k=%s).", family, *values)
        return values
    if choice == "Custom":
        return float(smc_lambda), float(smc_k)
    return _SMC_PRESETS[choice]


# ---------------------------------------------------------------------------
# Hooks and model patching
# ---------------------------------------------------------------------------

def _cfg_hook(
    fallback_options: dict[str, Any],
    *,
    alpha_low: float,
    alpha_high: float,
    smc_lambda: float,
    smc_k: float,
    apg_settings: dict[str, float] | None,
):
    def dcw_cfg_hook(args: dict[str, Any]):
        cond, uncond = args.get("cond"), args.get("uncond")
        scale = float(args.get("cond_scale", 7.0))
        if cond is None or uncond is None:
            return _plain_cfg(cond, uncond, scale)
        options = args.get("model_options", fallback_options)
        smc_state = options.setdefault(SMC_STATE_KEY, {})
        sigma = args.get("sigma")
        apg = None
        if apg_settings is not None:
            apg_state = options.setdefault(APG_STATE_KEY, {})
            direction = args.get("cond_denoised")
            if direction is None:
                direction = args["input"] - cond

            def apg(error):
                return _apply_apg_error(
                    error, direction, sigma=sigma, state=apg_state, **apg_settings
                )

        try:
            return _guided_noise(
                cond, uncond, sigma, scale,
                alpha_low=alpha_low, alpha_high=alpha_high,
                smc_lambda=smc_lambda, smc_k=smc_k, smc_state=smc_state, apg=apg,
            )
        except Exception as exc:
            LOGGER.warning("DCW(+a) CWM/SMC skipped for this step: %s", exc)
            return _plain_cfg(cond, uncond, scale)

    return dcw_cfg_hook


def _dcw_hook(
    fallback_options: dict[str, Any],
    *,
    lambda_low: float,
    lambda_high: float,
    rdc_tau: float,
    rdc_alpha_low: float,
    rdc_alpha_high: float,
):
    recurrent = rdc_tau > 0.0

    def dcw_post_cfg(args: dict[str, Any]):
        denoised, live, sigma = args.get("denoised"), args.get("input"), args.get("sigma")
        if denoised is None or live is None or sigma is None:
            return denoised
        try:
            state = None
            if recurrent:
                options = args.get("model_options", fallback_options)
                state = options.setdefault(RDC_STATE_KEY, {})
            return apply_dcw(
                denoised, live, sigma, lambda_low, lambda_high,
                rdc_tau=rdc_tau, rdc_alpha_low=rdc_alpha_low,
                rdc_alpha_high=rdc_alpha_high, state=state,
            )
        except Exception as exc:
            LOGGER.warning("DCW(+a) correction skipped for this step: %s", exc)
            return denoised

    return dcw_post_cfg


def patch_dcw(
    model: Any,
    *,
    dcw_enabled: bool,
    lambda_l: float,
    lambda_h: float,
    cwm_enabled: bool,
    alpha_l: float,
    alpha_h: float,
    smc_preset: Any,
    smc_lambda: float,
    smc_k: float,
    rdc_tau: float,
    rdc_alpha_ll: float,
    rdc_alpha_hh: float,
    apg_settings: dict[str, float] | None = None,
):
    """Apply DCW(+a) with the original node's switches and gating.

    - SMC is on when ``smc_preset`` is not Off.
    - RDC is on when ``rdc_tau > 0``. It only runs inside DCW, so it also
      needs ``dcw_enabled``.
    - DCW is added when ``dcw_enabled`` and a lambda is non-zero or RDC is on.
    - CWM needs ``cwm_enabled`` and a non-zero alpha.

    ``apg_settings`` (``eta``, ``norm_threshold``, ``momentum``) adds the pack's
    APG to the CFG hook. Returns ``model`` itself when nothing is active.
    """

    lambda_l, lambda_h = float(lambda_l), float(lambda_h)
    alpha_l, alpha_h = float(alpha_l), float(alpha_h)
    rdc_tau = float(rdc_tau)
    smc = resolve_smc(model, smc_preset, smc_lambda, smc_k)
    rdc_on = rdc_tau > 0.0
    dcw_on = bool(dcw_enabled) and (lambda_l != 0.0 or lambda_h != 0.0 or rdc_on)
    cwm_on = bool(cwm_enabled) and (alpha_l != 0.0 or alpha_h != 0.0)
    cfg_hook_on = cwm_on or smc is not None or apg_settings is not None
    if not dcw_on and not cfg_hook_on:
        return model

    patched = clone_model(model, "DCW(+a)")
    options = dict(getattr(patched, "model_options", {}) or {})
    patched.model_options = options

    if cfg_hook_on:
        if "sampler_cfg_function" in options:
            LOGGER.warning(
                "DCW(+a): another node already set sampler_cfg_function, so "
                "CWM/SMC%s are skipped; DCW still applies. Put this node before "
                "that node or turn off that node's CFG hook.",
                "/APG" if apg_settings is not None else "",
            )
        else:
            smc_lambda_value, smc_k_value = smc if smc is not None else (0.0, 0.0)
            options["sampler_cfg_function"] = _cfg_hook(
                options,
                alpha_low=alpha_l if cwm_on else 0.0,
                alpha_high=alpha_h if cwm_on else 0.0,
                smc_lambda=smc_lambda_value,
                smc_k=smc_k_value,
                apg_settings=apg_settings,
            )
            options["disable_cfg1_optimization"] = True

    if dcw_on:
        options["sampler_post_cfg_function"] = [
            *options.get("sampler_post_cfg_function", []),
            _dcw_hook(
                options,
                lambda_low=lambda_l,
                lambda_high=lambda_h,
                rdc_tau=rdc_tau if rdc_on else 0.0,
                rdc_alpha_low=float(rdc_alpha_ll) if rdc_on else 0.0,
                rdc_alpha_high=float(rdc_alpha_hh) if rdc_on else 0.0,
            ),
        ]
        if rdc_on:
            LOGGER.info(
                "DCW(+a) RDC on: alpha_LL=%s alpha_HH=%s tau=%s",
                rdc_alpha_ll, rdc_alpha_hh, rdc_tau,
            )
    return patched


def _float_input(default, low, high, step, tip, *, round_to=None):
    options = {"default": default, "min": low, "max": high, "step": step}
    if round_to is not None:
        options["round"] = round_to
    options["tooltip"] = tip
    return ("FLOAT", options)


class ForgeNeoDCWCWMSMC:
    """DCW(+a) as a pack node.

    The inputs, defaults, ranges and steps are the original ``DCWModelPatch``
    ones. The Python defaults of the switches are off, so a bare
    ``patch(model)`` from Python returns ``model`` unchanged. ComfyUI always
    passes every input, so the node itself uses the INPUT_TYPES defaults.
    The keyword-only ``lambda_low``... names are the older pack spellings the
    guidance suite still passes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lambda_l": _float_input(
                0.05, -0.5, 0.5, 0.005,
                "DCW low-band (LL) strength, strongest early. Positive moves "
                "the prediction toward x_t. 0 turns the low band off.",
                round_to=0.001,
            ),
            "lambda_h": _float_input(
                0.01, -0.3, 0.3, 0.001,
                "DCW high-band (HH) strength, strongest late. Positive moves "
                "the prediction toward x_t. 0 turns the high band off.",
                round_to=0.001,
            ),
            "dcw_enabled": ("BOOLEAN", {
                "default": True, "tooltip": "Turn DCW (and with it RDC) on or off.",
            }),
            "alpha_l": _float_input(
                0.0, -1.0, 2.0, 0.01,
                "CWM: extra CFG for the low band early in sampling. 0 = plain CFG.",
                round_to=0.001,
            ),
            "alpha_h": _float_input(
                0.0, -1.0, 2.0, 0.01,
                "CWM: extra CFG for the high band late in sampling. 0 = plain CFG.",
                round_to=0.001,
            ),
            "cwm_enabled": ("BOOLEAN", {
                "default": True, "tooltip": "Turn CFG wavelet mixing on or off.",
            }),
            "smc_preset": (SMC_PRESET_CHOICES, {
                "default": "Off",
                "tooltip": "SMC (sliding-mode CFG). Off, Auto (from the model "
                "family), a family preset, or Custom (the two values below).",
            }),
            "smc_lambda": _float_input(
                6.0, 0.5, 30.0, 0.1, "SMC surface shape lambda (Custom only).",
            ),
            "smc_k": _float_input(
                0.1, 0.0, 5.0, 0.01, "SMC switching gain k (Custom only).",
            ),
            "rdc_tau": _float_input(
                0.0, 0.0, 0.5, 0.01,
                "RDC memory in sigma/(sigma+1) units. 0 = RDC off. Needs DCW on.",
                round_to=0.001,
            ),
            "rdc_alpha_ll": _float_input(
                0.03, 0.0, 0.3, 0.005,
                "RDC pull of the low band toward its running average.",
                round_to=0.001,
            ),
            "rdc_alpha_hh": _float_input(
                0.0, 0.0, 0.1, 0.001,
                "RDC pull of the high band toward its running average.",
                round_to=0.001,
            ),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(
        self,
        model,
        lambda_l=0.05,
        lambda_h=0.01,
        dcw_enabled=False,
        alpha_l=0.0,
        alpha_h=0.0,
        cwm_enabled=False,
        smc_preset="Off",
        smc_lambda=6.0,
        smc_k=0.1,
        rdc_tau=0.0,
        rdc_alpha_ll=0.03,
        rdc_alpha_hh=0.0,
        *,
        lambda_low=None,
        lambda_high=None,
        alpha_low=None,
        alpha_high=None,
        smc_enabled=None,
        rdc_enabled=None,
        rdc_alpha_low=None,
        rdc_alpha_high=None,
        apg_enabled=False,
        apg_eta=0.0,
        apg_norm=15.0,
        apg_momentum=0.0,
    ):
        # Older pack spellings, still sent by the guidance suite.
        lambda_l = lambda_l if lambda_low is None else lambda_low
        lambda_h = lambda_h if lambda_high is None else lambda_high
        alpha_l = alpha_l if alpha_low is None else alpha_low
        alpha_h = alpha_h if alpha_high is None else alpha_high
        rdc_alpha_ll = rdc_alpha_ll if rdc_alpha_low is None else rdc_alpha_low
        rdc_alpha_hh = rdc_alpha_hh if rdc_alpha_high is None else rdc_alpha_high
        if smc_enabled is not None and not smc_enabled:
            smc_preset = "Off"
        if rdc_enabled is not None and not rdc_enabled:
            rdc_tau = 0.0
        apg_settings = None
        if apg_enabled:
            apg_settings = {
                "eta": float(apg_eta),
                "norm_threshold": float(apg_norm),
                "momentum": float(apg_momentum),
            }
        return (patch_dcw(
            model,
            dcw_enabled=dcw_enabled, lambda_l=lambda_l, lambda_h=lambda_h,
            cwm_enabled=cwm_enabled, alpha_l=alpha_l, alpha_h=alpha_h,
            smc_preset=smc_preset, smc_lambda=smc_lambda, smc_k=smc_k,
            rdc_tau=rdc_tau, rdc_alpha_ll=rdc_alpha_ll, rdc_alpha_hh=rdc_alpha_hh,
            apg_settings=apg_settings,
        ),)
