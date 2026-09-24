"""Forge Neo guidance features expressed as composable ComfyUI MODEL nodes.

The tensor math mirrors the user's ``forge_sam3_extension`` implementation.
Imports are deliberately lazy so the app can validate contracts without
loading ComfyUI or torch.
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from typing import Any

from .compat import (
    clone_model,
    filename_choices,
    invoke_provider,
    json_object,
    node_result,
    provider,
    require_torch,
)


CATEGORY = "AI Studio/Forge Neo parity/Guidance"


def parse_indices(spec: str, count: int, *, default: str = "") -> set[int]:
    """Parse ``8-18,22`` and clamp it to the available block count."""

    text = str(spec or default).replace(" ", "")
    result: set[int] = set()
    for part in text.split(","):
        if not part:
            continue
        if "-" in part:
            left, _, right = part.partition("-")
            try:
                start, end = int(left), int(right)
            except ValueError:
                continue
            if start <= end:
                result.update(range(start, end + 1))
        else:
            try:
                result.add(int(part))
            except ValueError:
                continue
    return {index for index in result if 0 <= index < max(0, int(count))}


def _scalar_sigma(sigma: Any) -> float:
    try:
        return float(sigma.flatten()[0].item())
    except Exception:
        try:
            return float(sigma)
        except Exception:
            return 1.0


def _sampling_percent(sigma: Any) -> float:
    # Anima is a flow model whose public sampler sigma falls from ~1 to 0.
    return min(1.0, max(0.0, 1.0 - _scalar_sigma(sigma)))


def _sampling_percent_for_model(model: Any, sigma: Any) -> float:
    """Invert Comfy's model-specific percent_to_sigma when it is available."""

    inner = getattr(model, "model", None)
    sampling = getattr(inner, "model_sampling", None)
    convert = getattr(sampling, "percent_to_sigma", None)
    if not callable(convert):
        return _sampling_percent(sigma)
    target = _scalar_sigma(sigma)
    try:
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = (low + high) * 0.5
            candidate = float(convert(middle))
            if candidate > target:
                low = middle
            else:
                high = middle
        return min(1.0, max(0.0, (low + high) * 0.5))
    except Exception:
        return _sampling_percent(sigma)


# settings_json parsing is shared by every node in this pack (compat.json_object).
_json_settings = json_object


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "enabled"}


def _setting(settings: dict[str, Any], key: str, default: Any) -> Any:
    value = settings.get(key, default)
    return default if value is None else value


def detail_schedule_value(
    progress: float,
    *,
    start: float = 0.2,
    end: float = 0.8,
    bias: float = 0.5,
    amount: float = 0.1,
    exponent: float = 1.0,
    start_offset: float = 0.0,
    end_offset: float = 0.0,
    fade: float = 0.0,
    smooth: bool = True,
) -> float:
    """Continuous form of Forge Detail Daemon's two-sided schedule."""

    progress = min(1.0, max(0.0, float(progress)))
    start, end = sorted((
        min(1.0, max(0.0, float(start))),
        min(1.0, max(0.0, float(end))),
    ))
    bias = min(1.0, max(0.0, float(bias)))
    exponent = max(0.0, float(exponent))
    midpoint = start + bias * (end - start)
    if progress < start:
        value = float(start_offset)
    elif progress > end:
        value = float(end_offset)
    elif progress <= midpoint:
        span = midpoint - start
        phase = 1.0 if span <= 1e-12 else (progress - start) / span
        if smooth:
            phase = 0.5 * (1.0 - math.cos(phase * math.pi))
        phase = phase**exponent if exponent != 0.0 else 1.0
        value = float(start_offset) + (float(amount) - float(start_offset)) * phase
    else:
        span = end - midpoint
        phase = 1.0 if span <= 1e-12 else (end - progress) / span
        if smooth:
            phase = 0.5 * (1.0 - math.cos(phase * math.pi))
        phase = phase**exponent if exponent != 0.0 else 1.0
        value = float(end_offset) + (float(amount) - float(end_offset)) * phase
    return value * (1.0 - min(1.0, max(0.0, float(fade))))


def apply_dave(output: Any, attenuation: float):
    """Remove a fraction of each block's spatial/token DC component."""

    if float(attenuation) == 0.0:
        return output
    if getattr(output, "ndim", 0) < 3:
        raise ValueError("DAVE expects token or spatial feature tensors.")
    dims = tuple(range(1, output.ndim - 1)) or (1,)
    mean = output.float().mean(dim=dims, keepdim=True)
    return (output.float() - float(attenuation) * mean).to(output.dtype)


def _pad_even(value: Any):
    torch = require_torch()
    height, width = int(value.shape[-2]), int(value.shape[-1])
    pad_h, pad_w = height % 2, width % 2
    if not (pad_h or pad_w):
        return value, (height, width)
    padding = (0, pad_w, 0, pad_h) + (0, 0) * max(0, value.ndim - 4)
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    try:
        padded = torch.nn.functional.pad(value, padding, mode=mode)
    except RuntimeError:
        padded = torch.nn.functional.pad(value, padding, mode="constant", value=0)
    return padded, (height, width)


def _haar_dwt(value: Any):
    low_h = (value[..., 0::2, :] + value[..., 1::2, :]) * 0.5
    high_h = (value[..., 0::2, :] - value[..., 1::2, :]) * 0.5
    return (
        (low_h[..., 0::2] + low_h[..., 1::2]) * 0.5,
        (low_h[..., 0::2] - low_h[..., 1::2]) * 0.5,
        (high_h[..., 0::2] + high_h[..., 1::2]) * 0.5,
        (high_h[..., 0::2] - high_h[..., 1::2]) * 0.5,
    )


def _haar_idwt(ll: Any, lh: Any, hl: Any, hh: Any):
    torch = require_torch()
    *leading, height, width = ll.shape
    low_h = torch.empty(
        *leading, height, width * 2, device=ll.device, dtype=ll.dtype
    )
    high_h = torch.empty_like(low_h)
    low_h[..., 0::2], low_h[..., 1::2] = ll + lh, ll - lh
    high_h[..., 0::2], high_h[..., 1::2] = hl + hh, hl - hh
    output = torch.empty(
        *leading, height * 2, width * 2, device=ll.device, dtype=ll.dtype
    )
    output[..., 0::2, :] = low_h + high_h
    output[..., 1::2, :] = low_h - high_h
    return output


def _sigma_norm(sigma: Any, like: Any):
    torch = require_torch()
    if torch.is_tensor(sigma):
        value = sigma.float() / (sigma.float() + 1.0)
        if value.ndim == 1:
            value = value.view(-1, *([1] * (like.ndim - 1)))
        return value.to(device=like.device, dtype=like.dtype)
    value = float(sigma)
    return torch.as_tensor(
        value / (value + 1.0), device=like.device, dtype=like.dtype
    )


def color_noise_wavelet(
    noise: Any,
    live: Any,
    strength: float = 1.0,
    gamma_power: float = 0.5,
    gamma_scale: float = 3.0,
):
    """CNS live-wavelet recoloring, preserving seeded noise variance."""

    torch = require_torch()
    if float(strength) == 0.0:
        return noise
    if noise.shape != live.shape or noise.ndim not in (4, 5):
        raise ValueError("CNS needs matching 4-D or 5-D noise/live latents.")
    if float(gamma_scale) <= 0:
        raise ValueError("CNS gamma_scale must be positive.")
    dtype = noise.dtype
    white, current = noise.float(), live.to(noise.device, dtype=torch.float32)
    current, (height, width) = _pad_even(current)
    bands = _haar_dwt(current)
    dims = tuple(range(1, current.ndim))
    energy = tuple(
        band.float().square().mean(dim=dims, keepdim=True).clamp_min(1e-8)
        for band in bands
    )
    total = sum(energy)
    weights = tuple(
        (1.0 - (item / total / float(gamma_scale)).clamp(0, 1))
        .clamp_min(1e-8)
        .pow(float(gamma_power))
        for item in energy
    )
    rms = (sum(item.square() for item in weights) / len(weights)).sqrt().clamp_min(1e-8)
    weights = tuple(item / rms for item in weights)
    white_padded, _ = _pad_even(white)
    colored = _haar_idwt(
        *(band * weight for band, weight in zip(_haar_dwt(white_padded), weights))
    )[..., :height, :width]
    colored *= white.std().clamp_min(1e-8) / colored.std().clamp_min(1e-8)
    amount = min(1.0, max(0.0, float(strength)))
    mixed = torch.lerp(white, colored, amount)
    mixed *= white.std().clamp_min(1e-8) / mixed.std().clamp_min(1e-8)
    return mixed.to(dtype)


def _energy_weight(band: Any):
    dims = tuple(range(2, band.ndim))
    energy = band.float().square().mean(dim=dims, keepdim=True)
    relative = energy / energy.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return relative.clamp(0.25, 4.0).to(band.dtype)


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
    """Forge extension's DCW transform followed by optional recurrent DCW."""

    torch = require_torch()
    if denoised.shape != live.shape or denoised.ndim not in (4, 5):
        raise ValueError("DCW needs matching 4-D or 5-D denoised/live latents.")
    dtype = denoised.dtype
    clean, current = denoised.float(), live.to(denoised.device, dtype=torch.float32)
    schedule = _sigma_norm(sigma, clean)
    low_gain = float(lambda_low) * schedule
    high_gain = float(lambda_high) * (1.0 - schedule)
    middle_gain = (low_gain + high_gain) * 0.5
    clean, (height, width) = _pad_even(clean)
    current, _ = _pad_even(current)
    corrected = [
        c + gain * _energy_weight(x) * (x - c)
        for c, x, gain in zip(
            _haar_dwt(clean), _haar_dwt(current),
            (low_gain, middle_gain, middle_gain, high_gain),
        )
    ]
    if float(rdc_tau) > 0 and state is not None:
        schedule_value = float(schedule.float().mean().item())
        previous_schedule = float(state.get("_sigma", schedule_value))
        beta = 1.0 - math.exp(
            -abs(previous_schedule - schedule_value) / max(float(rdc_tau), 1e-6)
        )
        state["_sigma"] = schedule_value
        alphas = (
            float(rdc_alpha_low),
            (float(rdc_alpha_low) + float(rdc_alpha_high)) * 0.5,
            (float(rdc_alpha_low) + float(rdc_alpha_high)) * 0.5,
            float(rdc_alpha_high),
        )
        for index, (key, band, alpha) in enumerate(
            zip(("LL", "LH", "HL", "HH"), corrected, alphas)
        ):
            if alpha == 0:
                continue
            old = state.get(key)
            if not torch.is_tensor(old) or old.shape != band.shape:
                state[key] = band.detach().clone()
                continue
            ema = (1.0 - beta) * old.to(band) + beta * band.detach()
            state[key] = ema.detach().clone()
            corrected[index] = band - alpha * (band - ema)
    return _haar_idwt(*corrected)[..., :height, :width].to(dtype)


def _smc_error(error: Any, previous: Any, strength: float, k: float):
    torch = require_torch()
    if not strength or not k:
        return error, error.detach()
    work = torch.nan_to_num(error.float())
    old = work.detach() if previous is None or previous.shape != work.shape else previous.to(work)
    surface = torch.nan_to_num((work - old) + float(strength) * old)
    dims = tuple(range(1, work.ndim))
    norm = torch.linalg.vector_norm(surface, dim=dims, keepdim=True).clamp_min(1e-8)
    delta = -float(k) * surface / norm
    limit = (0.5 * work.abs().mean(dim=dims, keepdim=True)).clamp_min(1e-8)
    corrected = torch.nan_to_num(work + delta.clamp(-limit, limit))
    return corrected.to(error.dtype), corrected.detach()


def _cwm_error(error: Any, sigma: Any, scale: float, low: float, high: float):
    torch = require_torch()
    if float(low) == 0 and float(high) == 0:
        return error * float(scale)
    dtype = error.dtype
    work = torch.nan_to_num(error.float())
    schedule = _sigma_norm(sigma, work)
    low_scale = float(scale) * (1.0 + float(low) * schedule)
    high_scale = float(scale) * (1.0 + float(high) * (1.0 - schedule))
    product = low_scale * high_scale
    middle = torch.where(
        product >= 0,
        product.clamp_min(0).sqrt(),
        (low_scale + high_scale) * 0.5,
    )
    padded, (height, width) = _pad_even(work)
    ll, lh, hl, hh = _haar_dwt(padded)
    return _haar_idwt(
        ll * low_scale, lh * middle, hl * middle, hh * high_scale
    )[..., :height, :width].to(dtype)


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


def skim_predictions(
    latent: Any,
    target: Any,
    reference: Any,
    scale: float,
    skimming_scale: float,
    disable_flipping_filter: bool,
):
    if abs(float(scale)) < 1e-6:
        return target
    denoised = reference + float(scale) * (target - reference)
    outer = ((target - reference).sign() == target.sign()) & (
        target.sign() == denoised.sign()
    )
    if not disable_flipping_filter:
        outer &= denoised.sign() == (denoised - latent).sign()
    low = reference + float(skimming_scale) * (target - reference)
    correction = (denoised - low) / float(scale)
    return target - correction * outer.to(correction.dtype)


class ForgeNeoNegPip:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, enabled=True):
        if not enabled:
            return model, clip
        return invoke_provider(
            "CLIPNegPip",
            method="execute",
            feature="NegPiP",
            kwargs={"model": model, "clip": clip},
        )[:2]


def _model_blocks(model: Any) -> list[Any]:
    current = getattr(model, "model", None)
    diffusion = getattr(current, "diffusion_model", None)
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None:
        return []
    return list(blocks)


def _transformer_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    # Cosmos/Predict2 Block.forward has transformer_options as positional #7.
    if len(args) > 6 and isinstance(args[6], dict):
        return args[6]
    return {}


def _patch_anima_blocks(
    model: Any,
    *,
    dave_enabled: bool,
    dave_blocks: str,
    dave_strength: float,
    dave_tau: float,
    slg_enabled: bool,
    slg_blocks: str,
):
    if not dave_enabled and not slg_enabled:
        return model
    patched = clone_model(model, "Anima DAVE/SLG")
    blocks = _model_blocks(patched)
    if not blocks:
        raise RuntimeError(
            "DAVE/SLG requires an Anima/Cosmos MODEL exposing diffusion_model.blocks."
        )
    dave_targets = (
        parse_indices(dave_blocks, len(blocks), default="8-18")
        if dave_enabled else set()
    )
    slg_targets = (
        parse_indices(slg_blocks, len(blocks), default="18")
        if slg_enabled else set()
    )
    if dave_enabled and not dave_targets:
        raise RuntimeError("DAVE has no valid target blocks for this model.")
    if slg_enabled and not slg_targets:
        raise RuntimeError("SLG has no valid target blocks for this model.")
    for index in sorted(dave_targets | slg_targets):
        original = blocks[index].forward

        def combined_forward(*args, _index=index, _original=original, **kwargs):
            options = _transformer_options(args, kwargs)
            if _index in slg_targets and options.get("forge_neo_slg_active"):
                if args:
                    return args[0]
                value = kwargs.get("x_B_T_H_W_D", kwargs.get("x"))
                if value is None:
                    raise RuntimeError("SLG block wrapper did not receive its input tensor.")
                return value
            output = _original(*args, **kwargs)
            if _index in dave_targets:
                progress = _sampling_percent(options.get("sigmas", 1.0))
                if float(dave_tau) <= 0.0 or progress < float(dave_tau):
                    output = apply_dave(output, float(dave_strength))
            return output

        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.forward", combined_forward
        )
    return patched


def _gaussian_blur_query(query: Any, sigma: float):
    """Blur Anima's pre-projection query over its actual H/W dimensions."""

    torch = require_torch()
    if query.ndim != 5:
        raise RuntimeError(
            "SEG requires Anima query tensors shaped [B,T,H,W,D]; this model exposes a different attention layout."
        )
    height, width = int(query.shape[2]), int(query.shape[3])
    limit = min(height, width)
    if limit < 2 or float(sigma) <= 0.0:
        return query
    requested = int(math.ceil(6.0 * float(sigma)))
    kernel_size = requested + 1 - requested % 2
    max_odd = limit if limit % 2 else limit - 1
    kernel_size = max(1, min(kernel_size, max_odd))
    if kernel_size <= 1:
        return query
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=query.device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coords / float(sigma)).square())
    kernel = (kernel / kernel.sum()).to(dtype=query.dtype)
    batch, frames, _, _, channels = query.shape
    image = query.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    mode = "reflect" if radius < height and radius < width else "replicate"
    image = torch.nn.functional.pad(image, (radius, radius, 0, 0), mode=mode)
    image = torch.nn.functional.conv2d(image, horizontal, groups=channels)
    image = torch.nn.functional.pad(image, (0, 0, radius, radius), mode=mode)
    image = torch.nn.functional.conv2d(image, vertical, groups=channels)
    return image.reshape(batch, frames, channels, height, width).permute(0, 1, 3, 4, 2)


def _patch_pag_attention(model: Any, block_indices: str, strength: float):
    """Blend Cosmos self-attention toward V, after Q/K normalization.

    The Comfy pre-projection hook cannot implement partial PAG: its query
    scaling is cancelled by q_norm. Patch only each selected MODEL object's
    attention operation, and activate it solely in this node's weak forward.
    This preserves the original normal/CFG/SLG passes and other node clones.
    """

    blocks = _model_blocks(model)
    targets = parse_indices(block_indices, len(blocks), default="18")
    if not targets:
        raise RuntimeError("PAG requires valid Anima/Cosmos self-attention blocks.")
    amount = min(1.0, max(0.0, float(strength)))
    token = uuid.uuid4().hex
    for index in sorted(targets):
        path = f"diffusion_model.blocks.{index}.self_attn.attn_op"
        getter = getattr(model, "get_model_object", None)
        original = (
            getter(path) if callable(getter)
            else getattr(getattr(blocks[index], "self_attn", None), "attn_op", None)
        )
        if not callable(original):
            raise RuntimeError(
                f"PAG requires Anima/Cosmos block {index} self_attn.attn_op."
            )

        def attention_op(q, k, v, *args, _original=original, **kwargs):
            output = _original(q, k, v, *args, **kwargs)
            options = kwargs.get("transformer_options")
            if options is None and args:
                options = args[0]
            if not isinstance(options, dict) or options.get("forge_neo_pag_active") != token:
                return output
            # Cosmos attention takes [B,...,heads,head_dim], and merges both
            # the spatial/token axes and heads before output_proj. Forge's
            # identity-attention endpoint keeps each token's own V, not mean(V).
            target = v.reshape(v.shape[0], -1, v.shape[-2] * v.shape[-1])
            if target.shape != output.shape:
                raise RuntimeError(
                    f"PAG attention/value layout mismatch: {tuple(output.shape)} vs {tuple(target.shape)}"
                )
            return require_torch().lerp(output, target.to(output), amount)

        model.add_object_patch(path, attention_op)
    return token


def _patch_perturbation_guidance(
    model: Any,
    *,
    attention_method: str | None,
    attention_scale: float,
    attention_blocks: str,
    attention_strength: float,
    seg_sigma: float,
    slg_enabled: bool,
    slg_scale: float,
    start_percent: float,
    end_percent: float,
    rescale: float,
    rescale_mode: str,
):
    attention_enabled = (
        attention_method in {"pag", "seg"}
        and float(attention_scale) != 0.0 and float(attention_strength) > 0.0
    )
    slg_enabled = bool(slg_enabled) and float(slg_scale) != 0.0
    if not attention_enabled and not slg_enabled:
        return model
    targets = parse_indices(attention_blocks, 4096, default="18")
    if attention_enabled and not targets:
        raise RuntimeError("PAG/SEG has no valid attention block indices.")
    patched = clone_model(model, "Anima PAG/SEG/SLG")
    pag_token = (
        _patch_pag_attention(patched, attention_blocks, attention_strength)
        if attention_enabled and attention_method == "pag" else None
    )

    def perturb(q, k, v, **kwargs):
        extra = kwargs.get("extra_options") or {}
        if int(extra.get("block_index", -1)) not in targets:
            return {"q": q, "k": k, "v": v}
        weak_q = _gaussian_blur_query(q, float(seg_sigma))
        strength = min(1.0, max(0.0, float(attention_strength)))
        return {"q": q + (weak_q - q) * strength, "k": k, "v": v}

    def weak_prediction(args: dict[str, Any], *, attention: bool):
        import comfy.samplers  # lazy: only reachable inside ComfyUI

        options = dict(args["model_options"])
        transformer = dict(options.get("transformer_options", {}) or {})
        if attention:
            if pag_token is not None:
                transformer["forge_neo_pag_active"] = pag_token
            else:
                patches = dict(transformer.get("patches", {}) or {})
                patches["attn1_patch"] = [*(patches.get("attn1_patch", []) or []), perturb]
                transformer["patches"] = patches
        else:
            transformer["forge_neo_slg_active"] = True
        options["transformer_options"] = transformer
        (weak,) = comfy.samplers.calc_cond_batch(
            args["model"], [args["cond"]], args["input"], args["sigma"], options
        )
        return weak

    def post_cfg(args):
        original = args["denoised"]
        progress = _sampling_percent(args.get("sigma", 1.0))
        if not float(start_percent) <= progress <= float(end_percent):
            return original
        cond = args.get("cond_denoised")
        if cond is None:
            raise RuntimeError("PAG/SEG/SLG requires a conditional prediction.")
        cond_work = cond.float()
        guidance = None
        if attention_enabled:
            weak = weak_prediction(args, attention=True)
            guidance = (cond_work - weak.float()) * float(attention_scale)
        if slg_enabled:
            weak = weak_prediction(args, attention=False)
            term = (cond_work - weak.float()) * float(slg_scale)
            guidance = term if guidance is None else guidance + term
        if guidance is None:
            return original
        amount = min(1.0, max(0.0, float(rescale)))
        if amount > 0.0:
            source = (
                cond_work + guidance
                if str(rescale_mode).casefold() == "partial"
                else original.float() + guidance
            )
            dims = tuple(range(1, source.ndim))
            ratio = cond_work.std(dim=dims, keepdim=True).clamp_min(1e-6) / source.std(
                dim=dims, keepdim=True
            ).clamp_min(1e-6)
            guidance = guidance * (amount * ratio + (1.0 - amount))
        return (original.float() + guidance).to(original.dtype)

    patched.set_model_sampler_post_cfg_function(
        post_cfg, disable_cfg1_optimization=True
    )
    return patched


class ForgeNeoAnimaDAVE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "mask": (["dave_alpha.npz", "blocks:8-18"],),
            "strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "tau": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, mask="dave_alpha.npz", strength=0.3, tau=0.1):
        if not enabled or float(strength) == 0:
            return (model,)
        # 'dave_alpha.npz' stays a COMBO choice so saved workflows validate,
        # but Comfy has no per-block alpha file: it means Forge's 8-18 range.
        spec = str(mask or "")
        if spec.casefold().startswith("blocks:"):
            spec = spec.split(":", 1)[1]
        elif not any(ch.isdigit() for ch in spec):
            spec = "8-18"
        # One block wrapper for the standalone node and the Suite keeps the
        # tau cutoff and positional transformer_options fallback identical.
        return (_patch_anima_blocks(
            model,
            dave_enabled=True,
            dave_blocks=spec,
            dave_strength=float(strength),
            dave_tau=float(tau),
            slg_enabled=False,
            slg_blocks="",
        ),)


class ForgeNeoAnimaModGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "quality_tags": ("STRING", {"default": "highres, best quality, score_7", "multiline": True}),
            "quality_negative": ("STRING", {"default": "score_1, score_2, score_3, worst quality, lowres", "multiline": True}),
            "weight": ("FLOAT", {"default": 3.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            "start_layer": ("INT", {"default": 8, "min": 0, "max": 255}),
            "end_layer": ("INT", {"default": 27, "min": 0, "max": 255}),
            "taper_layers": ("INT", {"default": 0, "min": 0, "max": 255}),
            "taper_scale": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0}),
            "final_layer_weight": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0}),
            "adapter_name": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, positive, negative, enabled=False, quality_tags="", quality_negative="", weight=3.0, start_layer=8, end_layer=27, taper_layers=0, taper_scale=0.25, final_layer_weight=0.0, adapter_name=""):
        if not enabled or float(weight) == 0:
            return (model,)
        instance = provider("AnimaModGuidance", feature="Anima modulation guidance")
        public = getattr(instance, "patch", None)
        globals_dict = getattr(public, "__globals__", {})
        exact = globals_dict.get("_apply_mod_guidance")
        if not callable(exact):
            raise RuntimeError(
                "Installed Spectrum provider lacks the exact layer-range modulation API; update comfyui-spectrum-ksampler."
            )
        return (exact(
            model, clip, positive, negative, adapter_name or None, quality_tags,
            quality_neg=quality_negative, w=float(weight),
            start_layer=int(start_layer), end_layer=int(end_layer),
            taper=int(taper_layers), taper_scale=float(taper_scale),
            final_w=float(final_layer_weight),
        ),)


class ForgeNeoSkimmedCFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False}),
            "skimming_cfg": ("FLOAT", {"default": 7.0, "min": -1.0, "max": 100.0}),
            "full_skim_negative": ("BOOLEAN", {"default": False}),
            "disable_flipping_filter": ("BOOLEAN", {"default": False}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "flip_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, skimming_cfg=7.0, full_skim_negative=False, disable_flipping_filter=False, start_percent=0.0, end_percent=1.0, flip_percent=0.0):
        if not enabled:
            return (model,)
        patched = clone_model(model, "Skimmed CFG")

        def post_cfg(args):
            original = args["denoised"]
            progress = _sampling_percent(args.get("sigma", 1.0))
            if not float(start_percent) <= progress <= float(end_percent):
                return original
            cond, uncond = args.get("cond_denoised"), args.get("uncond_denoised")
            scale, latent = float(args.get("cond_scale", 1.0)), args.get("input")
            if cond is None or uncond is None or latent is None or abs(scale - 1.0) < 1e-6:
                return original
            practical = scale if float(skimming_cfg) < 0 else float(skimming_cfg)
            flip_filter = bool(disable_flipping_filter)
            if float(flip_percent) > 0 and progress < float(flip_percent):
                flip_filter = not flip_filter
            u = skim_predictions(
                latent.float(), uncond.float(), cond.float(), scale,
                0.0 if full_skim_negative else practical, flip_filter,
            )
            c = skim_predictions(latent.float(), cond.float(), u, scale - 1.0, practical, flip_filter)
            return (u + scale * (c - u)).to(original.dtype)

        options = dict(getattr(patched, "model_options", {}) or {})
        callbacks = list(options.get("sampler_post_cfg_function", []) or [])
        options["sampler_post_cfg_function"] = [post_cfg, *callbacks]
        options["disable_cfg1_optimization"] = True
        patched.model_options = options
        return (patched,)


class ForgeNeoAnimaSafePAG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False}),
            "scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
            "block_indices": ("STRING", {"default": "18"}),
            "perturbation_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0}),
            "head_indices": ("STRING", {"default": ""}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "end_percent": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            "rescale": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
            "rescale_mode": (["full", "partial"],),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, scale=4.0, block_indices="18", perturbation_strength=0.75, head_indices="", start_percent=0.0, end_percent=0.7, rescale=0.2, rescale_mode="full"):
        if not enabled or float(scale) == 0 or float(perturbation_strength) == 0:
            return (model,)
        if str(head_indices).strip():
            raise RuntimeError(
                "Head-selective PAG is unavailable through ComfyUI's Anima pre-projection hook; leave head_indices empty."
            )
        return (_patch_perturbation_guidance(
            model, attention_method="pag", attention_scale=scale,
            attention_blocks=block_indices, attention_strength=perturbation_strength,
            seg_sigma=0.0, slg_enabled=False, slg_scale=0.0,
            start_percent=start_percent, end_percent=end_percent,
            rescale=rescale, rescale_mode=rescale_mode,
        ),)


_SMC_PRESETS = {
    "SD1.5 / SD2": (5.0, 0.10), "SDXL": (5.0, 0.10),
    "SD3 / SD3.5": (6.0, 0.10), "Flux": (6.0, 0.70),
    "Qwen-Image": (6.0, 0.10), "Cosmos / Wan": (6.0, 0.20),
    "Custom": (6.0, 0.10),
}


def _detect_smc(model: Any) -> str:
    candidates = [model, getattr(model, "model", None)]
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    candidates.append(diffusion)
    names = " ".join(type(item).__name__.casefold() for item in candidates if item is not None)
    if "flux" in names:
        return "Flux"
    if any(token in names for token in ("anima", "cosmos", "wan", "predict2")):
        return "Cosmos / Wan"
    if "sdxl" in names:
        return "SDXL"
    if any(token in names for token in ("sd3", "mmdit")):
        return "SD3 / SD3.5"
    if "qwen" in names:
        return "Qwen-Image"
    return "SD1.5 / SD2"


class ForgeNeoDCWCWMSMC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "dcw_enabled": ("BOOLEAN", {"default": False}),
            "lambda_low": ("FLOAT", {"default": 0.05, "min": -1.0, "max": 1.0}),
            "lambda_high": ("FLOAT", {"default": 0.01, "min": -1.0, "max": 1.0}),
            "cwm_enabled": ("BOOLEAN", {"default": False}),
            "alpha_low": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0}),
            "alpha_high": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0}),
            "smc_enabled": ("BOOLEAN", {"default": False}),
            "smc_preset": (["Off", "Auto", *_SMC_PRESETS.keys()],),
            "smc_lambda": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 100.0}),
            "smc_k": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0}),
            "rdc_enabled": ("BOOLEAN", {"default": False}),
            "rdc_tau": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0}),
            "rdc_alpha_low": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 2.0}),
            "rdc_alpha_high": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, dcw_enabled=False, lambda_low=0.05, lambda_high=0.01, cwm_enabled=False, alpha_low=0.0, alpha_high=0.0, smc_enabled=False, smc_preset="Auto", smc_lambda=6.0, smc_k=0.1, rdc_enabled=False, rdc_tau=0.15, rdc_alpha_low=0.03, rdc_alpha_high=0.0, *, apg_enabled=False, apg_eta=0.0, apg_norm=15.0, apg_momentum=0.0):
        if not any((dcw_enabled, cwm_enabled, smc_enabled, rdc_enabled, apg_enabled)):
            return (model,)
        patched = clone_model(model, "DCW/CWM/SMC/RDC")
        state: dict[str, Any] = {
            "smc": None, "rdc": {}, "apg": {}, "sigma": None,
        }
        if smc_enabled:
            preset = _detect_smc(patched) if smc_preset == "Auto" else str(smc_preset)
            if preset == "Off":
                raise RuntimeError("SMC is enabled but its preset is Off.")
            resolved_lambda, resolved_k = (
                (float(smc_lambda), float(smc_k))
                if preset == "Custom" else _SMC_PRESETS.get(preset, (float(smc_lambda), float(smc_k)))
            )
        else:
            resolved_lambda = resolved_k = 0.0

        if cwm_enabled or smc_enabled or apg_enabled:
            existing = (getattr(patched, "model_options", {}) or {}).get("sampler_cfg_function")
            if existing is not None:
                raise RuntimeError(
                    "DCW/CWM/SMC cannot replace an existing sampler_cfg_function. Put this node before other CFG replacers or disable CWM/SMC."
                )

            def cfg_function(args):
                cond = args.get("cond_denoised")
                uncond = args.get("uncond_denoised")
                live = args.get("input")
                if cond is None or uncond is None or live is None:
                    raise RuntimeError(
                        "DCW/CWM/SMC/APG requires ComfyUI denoised CFG tensors."
                    )
                scale, sigma = float(args["cond_scale"]), args.get("sigma", 1.0)
                current_sigma = _scalar_sigma(sigma)
                if state["sigma"] is not None and current_sigma > state["sigma"] + 1e-6:
                    state["smc"], state["rdc"], state["apg"] = None, {}, {}
                state["sigma"] = current_sigma
                error = cond.float() - uncond.float()
                if smc_enabled:
                    error, state["smc"] = _smc_error(
                        error, state["smc"], resolved_lambda, resolved_k
                    )
                if apg_enabled:
                    error = _apply_apg_error(
                        error, cond, eta=float(apg_eta),
                        norm_threshold=float(apg_norm),
                        momentum=float(apg_momentum), sigma=sigma,
                        state=state["apg"],
                    )
                guided = (
                    _cwm_error(error, sigma, scale, alpha_low, alpha_high)
                    if cwm_enabled else error * scale
                )
                denoised = uncond.float() + guided
                # Comfy's sampler_cfg_function contract expects a noise
                # prediction, while Forge's implementation composes x0.
                return (live.float() - denoised).to(cond.dtype)

            patched.set_model_sampler_cfg_function(cfg_function, disable_cfg1_optimization=True)

        if dcw_enabled or rdc_enabled:
            def post_cfg(args):
                original, live = args["denoised"], args.get("input")
                if live is None:
                    raise RuntimeError("DCW/RDC did not receive the live sampler latent.")
                return apply_dcw(
                    original, live, args.get("sigma", 1.0),
                    float(lambda_low) if dcw_enabled else 0.0,
                    float(lambda_high) if dcw_enabled else 0.0,
                    rdc_tau=float(rdc_tau) if rdc_enabled else 0.0,
                    rdc_alpha_low=float(rdc_alpha_low),
                    rdc_alpha_high=float(rdc_alpha_high), state=state["rdc"],
                )

            patched.set_model_sampler_post_cfg_function(post_cfg)
        return (patched,)


def _patch_adaptive_guidance(
    model: Any, *, start_percent: float, interval: int
):
    patched = clone_model(model, "Anima Adaptive Guidance")
    options = dict(getattr(patched, "model_options", {}) or {})
    if options.get("sampler_calc_cond_batch_function") is not None:
        raise RuntimeError(
            "Adaptive Guidance cannot replace an existing sampler_calc_cond_batch_function."
        )
    state: dict[str, Any] = {"sigma": None, "step": 0}

    def calculate(args):
        import comfy.samplers  # lazy: only reachable inside ComfyUI

        sigma = args.get("sigma", 1.0)
        current = _scalar_sigma(sigma)
        last = state["sigma"]
        if last is None or current > float(last) + 1e-6:
            state["step"] = 0
        elif abs(current - float(last)) > 1e-8:
            state["step"] += 1
        state["sigma"] = current
        conds = list(args.get("conds") or [])
        if not conds:
            raise RuntimeError("Adaptive Guidance received no sampler conditioning.")
        progress = _sampling_percent_for_model(patched, sigma)
        keep_interval = max(0, int(interval))
        skip_uncond = progress >= float(start_percent) and (
            keep_interval == 0 or state["step"] % keep_interval != 0
        )
        if skip_uncond and len(conds) > 1 and conds[1] is not None:
            (conditional,) = comfy.samplers.calc_cond_batch(
                args["model"], [conds[0]], args["input"], sigma,
                args["model_options"],
            )
            return [conditional, conditional]
        return comfy.samplers.calc_cond_batch(
            args["model"], conds, args["input"], sigma, args["model_options"]
        )

    options["sampler_calc_cond_batch_function"] = calculate
    options["disable_cfg1_optimization"] = True
    patched.model_options = options
    return patched


class ForgeNeoAnimaGuidanceSuite:
    """Compose the extension's 62-field Anima guidance payload on one MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "settings_json": ("STRING", {"default": "{}", "multiline": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, clip, positive, negative, enabled=False, settings_json="{}"):
        if not enabled:
            return (model,)
        settings = _json_settings(settings_json, "Anima guidance suite")

        attention_requested = _as_bool(_setting(settings, "guid_enabled", False))
        method_text = str(_setting(settings, "guid_attn_method", "PAG")).strip().casefold()
        method_map = {"pag": "pag", "seg": "seg", "none": None, "off": None, "": None}
        if method_text not in method_map:
            raise RuntimeError(f"Unsupported Anima attention guidance method: {method_text!r}.")
        attention_method = method_map[method_text] if attention_requested else None
        if attention_method and _as_bool(_setting(settings, "guid_legacy_attn", False)):
            raise RuntimeError(
                "Legacy Anima attention mode is enabled, but ComfyUI only exposes the current pre-projection PAG/SEG hook."
            )
        if attention_method and str(_setting(settings, "guid_head_indices", "")).strip():
            raise RuntimeError(
                "Head-selective PAG/SEG is unavailable through ComfyUI's Anima pre-projection hook; leave guid_head_indices empty."
            )

        slg_enabled = _as_bool(_setting(settings, "guid_slg_on", False))
        dave_enabled = _as_bool(_setting(settings, "guid_dave_enabled", False))
        current = _patch_anima_blocks(
            model,
            dave_enabled=dave_enabled,
            dave_blocks=str(_setting(settings, "guid_dave_blocks", "8-18")),
            dave_strength=float(_setting(settings, "guid_dave_strength", 0.3)),
            dave_tau=float(_setting(settings, "guid_dave_tau", 0.1)),
            slg_enabled=slg_enabled,
            slg_blocks=str(_setting(settings, "guid_slg_blocks", "18")),
        )

        mode_text = str(_setting(settings, "guid_cfg_mode", "Preserve incoming")).strip().casefold()
        mode_map = {
            "preserve incoming": "preserve", "preserve": "preserve",
            "apg": "apg", "cwm": "cwm", "smc": "smc",
            "smc + cwm": "smc+cwm", "smc+cwm": "smc+cwm",
        }
        if mode_text not in mode_map:
            raise RuntimeError(f"Unsupported Anima CFG mode: {mode_text!r}.")
        mode = mode_map[mode_text]
        stack = _as_bool(_setting(settings, "guid_experimental_stack", False))
        apg_enabled = (
            _as_bool(_setting(settings, "guid_apg_enabled", False))
            or mode == "apg" or stack
        )
        smc_enabled = (
            _as_bool(_setting(settings, "guid_smc_enabled", False))
            or _as_bool(_setting(settings, "guid_smc_master_enabled", False))
            or mode in {"smc", "smc+cwm"} or stack
        )
        cwm_enabled = (
            _as_bool(_setting(settings, "guid_cwm_enabled", False))
            or mode in {"cwm", "smc+cwm"} or stack
        )
        dcw_enabled = _as_bool(_setting(settings, "guid_dcw_enabled", False))
        rdc_enabled = _as_bool(_setting(settings, "guid_rdc_enabled", False))
        if any((apg_enabled, smc_enabled, cwm_enabled, dcw_enabled, rdc_enabled)):
            current = ForgeNeoDCWCWMSMC().patch(
                current,
                dcw_enabled=dcw_enabled,
                lambda_low=float(_setting(settings, "guid_dcw_lambda_low", 0.10)),
                lambda_high=float(_setting(settings, "guid_dcw_lambda_high", 0.02)),
                cwm_enabled=cwm_enabled,
                alpha_low=float(_setting(settings, "guid_cwm_alpha_low", 0.30)),
                alpha_high=float(_setting(settings, "guid_cwm_alpha_high", 0.15)),
                smc_enabled=smc_enabled,
                smc_preset=str(_setting(settings, "guid_smc_preset", "Auto")),
                smc_lambda=float(_setting(settings, "guid_smc_lambda", 6.0)),
                smc_k=float(_setting(settings, "guid_smc_k", 0.10)),
                rdc_enabled=rdc_enabled,
                rdc_tau=float(_setting(settings, "guid_rdc_tau", 0.15)),
                rdc_alpha_low=float(_setting(settings, "guid_rdc_alpha_ll", 0.03)),
                rdc_alpha_high=float(_setting(settings, "guid_rdc_alpha_hh", 0.0)),
                apg_enabled=apg_enabled,
                apg_eta=float(_setting(settings, "guid_apg_eta", 0.0)),
                apg_norm=float(_setting(settings, "guid_apg_norm", 15.0)),
                apg_momentum=float(_setting(settings, "guid_apg_momentum", 0.0)),
            )[0]

        if attention_method or slg_enabled:
            rescale = float(_setting(settings, "guid_rescale", 0.20))
            if apg_enabled and _as_bool(_setting(settings, "guid_apg_autooff", True), True):
                rescale = 0.0
            current = _patch_perturbation_guidance(
                current,
                attention_method=attention_method,
                attention_scale=float(_setting(settings, "guid_scale", 4.0)),
                attention_blocks=str(_setting(settings, "guid_block_indices", "18")),
                attention_strength=float(
                    _setting(settings, "guid_official_strength", _setting(settings, "guid_legacy_strength", 0.75))
                ),
                seg_sigma=float(_setting(settings, "guid_seg_sigma", 100.0)),
                slg_enabled=slg_enabled,
                slg_scale=float(_setting(settings, "guid_slg_scale", 3.0)),
                start_percent=float(_setting(settings, "guid_start_percent", 0.0)),
                end_percent=float(_setting(settings, "guid_end_percent", 0.7)),
                rescale=rescale,
                rescale_mode=str(_setting(settings, "guid_rescale_mode", "full")),
            )

        if _as_bool(_setting(settings, "guid_adg_enabled", False)):
            current = _patch_adaptive_guidance(
                current,
                start_percent=float(_setting(settings, "guid_adg_start", 0.5)),
                interval=int(_setting(settings, "guid_adg_interval", 0)),
            )

        if _as_bool(_setting(settings, "guid_mod_enabled", False)):
            mod_clip = clip
            clip_name = str(_setting(settings, "guid_mod_clip_model", "")).strip()
            if clip_name:
                mod_clip = invoke_provider(
                    "CLIPLoader", method="load_clip", feature="Anima modulation CLIP-L",
                    args=(clip_name, "stable_diffusion", "default"),
                )[0]
            mod_positive, mod_negative = positive, negative
            if str(_setting(settings, "guid_mod_base_source", "Main positive")).strip().casefold() == "custom":
                mod_positive = invoke_provider(
                    "CLIPTextEncode", method="encode", feature="Anima modulation base prompt",
                    args=(mod_clip, str(_setting(settings, "guid_mod_base_prompt", ""))),
                )[0]
            if str(_setting(settings, "guid_mod_negative_source", "Main negative")).strip().casefold() == "custom":
                mod_negative = invoke_provider(
                    "CLIPTextEncode", method="encode", feature="Anima modulation negative prompt",
                    args=(mod_clip, str(_setting(settings, "guid_mod_negative_prompt", ""))),
                )[0]
            adapter_mode = str(_setting(settings, "guid_mod_adapter_mode", "Auto-download official")).strip().casefold()
            adapter = ""
            if adapter_mode == "local file":
                adapter = str(_setting(settings, "guid_mod_adapter_path", "")).strip()
                if not adapter:
                    raise RuntimeError(
                        "Anima modulation is set to Local file but guid_mod_adapter_path is empty."
                    )
            elif adapter_mode != "auto-download official":
                raise RuntimeError(f"Unsupported modulation adapter mode: {adapter_mode!r}.")
            blocks = _model_blocks(current)
            end_layer = int(_setting(settings, "guid_mod_end_layer", -1))
            if end_layer < 0 and blocks:
                end_layer = len(blocks) - 1
            current = ForgeNeoAnimaModGuidance().patch(
                current, mod_clip, mod_positive, mod_negative, True,
                str(_setting(settings, "guid_mod_positive_prompt", "masterpiece, best quality, highres")),
                "",
                float(_setting(settings, "guid_mod_weight", 3.0)),
                int(_setting(settings, "guid_mod_start_layer", 0)),
                end_layer,
                0, 0.25, 0.0, adapter,
            )[0]

        if _as_bool(_setting(settings, "guid_cns_enabled", False)):
            current = clone_model(current, "Anima CNS sampler handoff")
            options = dict(getattr(current, "model_options", {}) or {})
            options["forge_neo_cns"] = {
                "strength": float(_setting(settings, "guid_cns_strength", 1.0)),
                "gamma_power": float(_setting(settings, "guid_cns_gamma_power", 0.5)),
                "gamma_scale": float(_setting(settings, "guid_cns_gamma_scale", 3.0)),
            }
            current.model_options = options
        return (current,)


class ForgeNeoAnimaDetailDaemon:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "settings_json": ("STRING", {"default": "{}", "multiline": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, settings_json="{}"):
        if not enabled:
            return (model,)
        settings = _json_settings(settings_json, "Anima Detail Daemon")
        if not _as_bool(_setting(settings, "dd_enabled", True), True):
            return (model,)
        preset = str(_setting(settings, "dd_preset", "Medium")).strip()
        preset_amounts = {"Subtle": 0.05, "Medium": 0.10, "Strong": 0.25}
        amount = preset_amounts.get(
            preset, float(_setting(settings, "dd_amount", 0.10))
        )
        multiplier = float(_setting(settings, "dd_multiplier", 1.0))
        if amount == 0.0 or multiplier == 0.0:
            return (model,)
        patched = clone_model(model, "Anima Detail Daemon")
        existing = (getattr(patched, "model_options", {}) or {}).get("model_function_wrapper")

        def wrapper(apply_model, args):
            sigma = args["timestep"]
            schedule = detail_schedule_value(
                _sampling_percent_for_model(patched, sigma),
                start=float(_setting(settings, "dd_start", 0.2)),
                end=float(_setting(settings, "dd_end", 0.8)),
                bias=float(_setting(settings, "dd_bias", 0.5)),
                amount=float(amount),
                exponent=float(_setting(settings, "dd_exponent", 1.0)),
                start_offset=float(_setting(settings, "dd_start_offset", 0.0)),
                end_offset=float(_setting(settings, "dd_end_offset", 0.0)),
                fade=float(_setting(settings, "dd_fade", 0.0)),
                smooth=_as_bool(_setting(settings, "dd_smooth", True), True),
            )
            cfg = (
                float(_setting(settings, "cfg_scale", 1.0))
                if _as_bool(_setting(settings, "dd_cfg_couple", True), True)
                else 1.0
            )
            factor = min(3.0, max(0.05, 1.0 - schedule * multiplier * cfg))
            changed = dict(args)
            changed["timestep"] = sigma * factor
            if existing is not None:
                return existing(apply_model, changed)
            return apply_model(changed["input"], changed["timestep"], **dict(changed.get("c") or {}))

        patched.set_model_unet_function_wrapper(wrapper)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "ForgeNeoNegPip": ForgeNeoNegPip,
    "ForgeNeoAnimaDAVE": ForgeNeoAnimaDAVE,
    "ForgeNeoAnimaModGuidance": ForgeNeoAnimaModGuidance,
    "ForgeNeoSkimmedCFG": ForgeNeoSkimmedCFG,
    "ForgeNeoAnimaSafePAG": ForgeNeoAnimaSafePAG,
    "ForgeNeoDCWCWMSMC": ForgeNeoDCWCWMSMC,
    "ForgeNeoAnimaGuidanceSuite": ForgeNeoAnimaGuidanceSuite,
    "ForgeNeoAnimaDetailDaemon": ForgeNeoAnimaDetailDaemon,
}
