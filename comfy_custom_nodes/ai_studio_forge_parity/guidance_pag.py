"""Anima perturbation guidance: PAG, SEG and SLG (``ForgeNeoAnimaSafePAG``).

PAG is the original node, not a port. ``vendor/comfyui_anima_safe_pag`` is
``iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py`` byte for byte (MIT),
and both ``ForgeNeoAnimaSafePAG`` and the guidance suite call its
``AnimaSafePAG().patch``. That brings the original's behaviour as is:

- the sigma window from ``model_sampling.percent_to_sigma`` (percents clamped
  to 0-1 and swapped when reversed, both ends inclusive; :15-34),
- block and head parsing that swaps reversed ranges and raises on an empty or
  invalid list (:37-64),
- one extra batch row appended in ``sampler_calc_cond_batch_function`` and
  perturbed only in the selected blocks and heads (:119-178, :271-333), so the
  model is not run a second time and ``disable_cfg1_optimization`` stays off,
- the post-CFG guidance and rescale (:181-192, :335-364).

SEG and SLG are not part of that original. They keep this pack's own weak pass
(a second ``calc_cond_batch`` in post-CFG); the SLG block skip itself lives in
``guidance_dave._patch_anima_blocks``. They are gated by the original's sigma
window (the vendored ``_percent_range_to_sigmas``/``_sigma_active``), so PAG,
SEG and SLG switch on and off at the same steps, as sam-extra's
``_pert_in_range`` does. ``_patch_perturbation_guidance`` is the entry point
the guidance suite uses for all three.

Host difference with PAG and SEG/SLG both on: sam-extra adds the terms and
rescales their sum once (``scripts/anima_safe_pag.py`` ``_apply_perturbation``).
Here the vendored PAG post-CFG rescales its own term, and the SEG/SLG post-CFG
rescales its term against the result it receives. With ``rescale`` 0 the two
are the same sum.
"""

from __future__ import annotations

import math
from typing import Any

from .compat import clone_model, require_torch
from .guidance_common import CATEGORY, parse_indices


def _original_pag_node():
    """The vendored original node class.

    Imported on first use: the upstream module imports ``torch`` and
    ``comfy.samplers`` at import time, and the app imports this pack without
    either.
    """

    from .vendor.comfyui_anima_safe_pag import AnimaSafePAG

    return AnimaSafePAG


def normalize_rescale_mode(value: Any) -> str:
    """``partial`` or ``full``, read the way sam-extra reads it.

    sam-extra lower-cases and strips the value, and anything but ``partial`` is
    ``full`` (``scripts/anima_safe_pag.py`` :3632-3635, :3736-3738). The
    original node compares ``mode == "full"`` as is (origin :186), so the suite
    normalizes before calling it. The standalone node passes its combo value
    through untouched, like the original.
    """

    return "partial" if str(value).strip().casefold() == "partial" else "full"


def reject_unsupported_seg(
    attention_method: str | None, *, legacy_attn: bool, head_indices: str
) -> None:
    """Raise for the SEG modes this pack cannot express.

    PAG takes legacy (as ``legacy_strength``) and head indices through the
    original node. SEG has no original here, and this pack's SEG has neither a
    legacy path nor head selection. The guidance suite calls this before it
    touches the model.
    """

    if attention_method != "seg":
        return
    if legacy_attn:
        raise RuntimeError(
            "Legacy SEG has no ComfyUI implementation; turn legacy attention off for SEG."
        )
    if str(head_indices or "").strip():
        raise RuntimeError(
            "Head-selective SEG has no ComfyUI implementation; leave head indices empty for SEG."
        )


def apply_anima_safe_pag(
    model: Any,
    *,
    scale: float,
    block_indices: str,
    perturbation_strength: float,
    head_indices: str,
    start_percent: float,
    end_percent: float,
    rescale: float,
    rescale_mode: str,
):
    """Return ``model`` patched by the original ``AnimaSafePAG().patch``."""

    (patched,) = _original_pag_node()().patch(
        model,
        scale,
        block_indices,
        perturbation_strength,
        head_indices,
        start_percent,
        end_percent,
        rescale,
        rescale_mode,
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


def _patch_seg_slg_guidance(
    model: Any,
    *,
    seg_enabled: bool,
    seg_scale: float,
    seg_blocks: str,
    seg_strength: float,
    seg_sigma: float,
    slg_enabled: bool,
    slg_scale: float,
    start_percent: float,
    end_percent: float,
    rescale: float,
    rescale_mode: str,
):
    """SEG and SLG through a second, weak ``calc_cond_batch`` in post-CFG.

    The window is the original PAG node's: the percents become sigmas once,
    through the model's ``percent_to_sigma`` (clamped, swapped when reversed),
    and each call's sigma is tested with both ends inclusive (origin:
    iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:15-34, :229-233,
    :337). sam-extra gates PAG, SEG and SLG with this one window
    (``_pert_in_range``), so SEG/SLG here run at exactly the PAG steps.
    """

    if not seg_enabled and not slg_enabled:
        return model
    targets = parse_indices(seg_blocks, 4096, default="18")
    if seg_enabled and not targets:
        raise RuntimeError("PAG/SEG has no valid attention block indices.")
    patched = clone_model(model, "Anima PAG/SEG/SLG")
    from .vendor.comfyui_anima_safe_pag import _percent_range_to_sigmas, _sigma_active

    sigma_start, sigma_end, _start, _end = _percent_range_to_sigmas(
        patched, start_percent, end_percent
    )

    def perturb(q, k, v, **kwargs):
        extra = kwargs.get("extra_options") or {}
        if int(extra.get("block_index", -1)) not in targets:
            return {"q": q, "k": k, "v": v}
        weak_q = _gaussian_blur_query(q, float(seg_sigma))
        strength = min(1.0, max(0.0, float(seg_strength)))
        return {"q": q + (weak_q - q) * strength, "k": k, "v": v}

    def weak_prediction(args: dict[str, Any], *, attention: bool):
        import comfy.samplers  # lazy: only reachable inside ComfyUI

        options = dict(args["model_options"])
        transformer = dict(options.get("transformer_options", {}) or {})
        if attention:
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
        if not _sigma_active(args["sigma"], sigma_start, sigma_end):
            return original
        cond = args.get("cond_denoised")
        if cond is None:
            raise RuntimeError("PAG/SEG/SLG requires a conditional prediction.")
        cond_work = cond.float()
        guidance = None
        if seg_enabled:
            weak = weak_prediction(args, attention=True)
            guidance = (cond_work - weak.float()) * float(seg_scale)
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
                if normalize_rescale_mode(rescale_mode) == "partial"
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
    head_indices: str = "",
    legacy_attn: bool = False,
    legacy_strength: float | None = None,
):
    """Apply PAG (original node), SEG and/or SLG in that order.

    ``legacy_attn`` with PAG reads ``legacy_strength`` instead of
    ``attention_strength``, as sam-extra does (``anima_safe_pag.py:3624``);
    both modes share one PAG formula there. Legacy SEG and head-selective SEG
    have no ComfyUI implementation and raise before the model is touched.
    ``rescale_mode`` is normalized once (``normalize_rescale_mode``), so PAG and
    SEG/SLG read it the same way.
    """

    reject_unsupported_seg(
        attention_method, legacy_attn=legacy_attn, head_indices=head_indices
    )
    rescale_mode = normalize_rescale_mode(rescale_mode)
    current = model
    if attention_method == "pag" and float(attention_scale) != 0.0:
        strength = (
            legacy_strength
            if legacy_attn and legacy_strength is not None
            else attention_strength
        )
        current = apply_anima_safe_pag(
            current,
            scale=attention_scale,
            block_indices=attention_blocks,
            perturbation_strength=strength,
            head_indices=head_indices,
            start_percent=start_percent,
            end_percent=end_percent,
            rescale=rescale,
            rescale_mode=rescale_mode,
        )
    return _patch_seg_slg_guidance(
        current,
        seg_enabled=(
            attention_method == "seg"
            and float(attention_scale) != 0.0 and float(attention_strength) > 0.0
        ),
        seg_scale=attention_scale,
        seg_blocks=attention_blocks,
        seg_strength=attention_strength,
        seg_sigma=seg_sigma,
        slg_enabled=bool(slg_enabled) and float(slg_scale) != 0.0,
        slg_scale=slg_scale,
        start_percent=start_percent,
        end_percent=end_percent,
        rescale=rescale,
        rescale_mode=rescale_mode,
    )


class ForgeNeoAnimaSafePAG:
    """The original Anima Safe PAG node behind this pack's ``enabled`` toggle.

    The inputs after ``enabled`` are the original's, with the same defaults,
    ranges and steps (origin :197-210; ``tests/test_forge_parity_pag_origin.py``
    compares them with the vendored file).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False}),
            "scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "block_indices": ("STRING", {"default": "18", "multiline": False}),
            "perturbation_strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001}),
            "head_indices": ("STRING", {"default": "", "multiline": False}),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            "end_percent": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.001}),
            "rescale": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "rescale_mode": (["full", "partial"], {"default": "full"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, scale=4.0, block_indices="18", perturbation_strength=0.75, head_indices="", start_percent=0.0, end_percent=0.7, rescale=0.2, rescale_mode="full"):
        if not enabled:
            return (model,)
        return (apply_anima_safe_pag(
            model, scale=scale, block_indices=block_indices,
            perturbation_strength=perturbation_strength, head_indices=head_indices,
            start_percent=start_percent, end_percent=end_percent,
            rescale=rescale, rescale_mode=rescale_mode,
        ),)
