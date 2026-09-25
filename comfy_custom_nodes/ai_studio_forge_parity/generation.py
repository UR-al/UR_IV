"""Generation, input, detailer, reference, and output Forge-parity nodes."""

from __future__ import annotations

import json
import logging
import math
import secrets
from typing import Any

from .compat import (
    filename_choices,
    folder_paths_module,
    invoke_provider,
    is_disabled_choice,
    json_object as _json_object,
    node_result,
    provider,
    require_torch,
    sampler_names,
    scheduler_names,
)
from .guidance_cns import (
    CNS_DEFAULTS,
    ForgeNeoCNSSamplerPatch,
    apply_cns,
    cns_float_input,
    model_cns_settings,
)
from .anima_lora_nodes import (
    AnimaLoraStateCache,
    load_lora_block_weight,
    load_lora_model_only,
)


CATEGORY = "AI Studio/Forge Neo parity/Generation"
LOGGER = logging.getLogger("ai_studio_forge_parity")


def _image_tensor(value: Any, name: str = "image"):
    torch = require_torch()
    if not torch.is_tensor(value) or value.ndim != 4 or value.shape[-1] < 1:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor [B,H,W,C].")
    return value


def resize_image(image: Any, width: int, height: int, fit: str = "crop", background: float = 0.0, *, interpolation: str = "bicubic"):
    """Resize a BHWC Comfy image without changing its batch/channel layout.

    Bicubic resampling overshoots hard edges (white line art peaks near 1.2);
    the result is clamped back into Comfy's ``[0,1]`` IMAGE range so no
    downstream node mistakes it for 0..255 data.
    """

    torch = require_torch()
    image = _image_tensor(image)
    width, height = max(1, int(width)), max(1, int(height))
    source_h, source_w = int(image.shape[1]), int(image.shape[2])
    mode = str(fit or "crop").casefold()
    nchw = image.movedim(-1, 1)
    if mode in {"stretch", "resize", "just resize"}:
        result = torch.nn.functional.interpolate(
            nchw, size=(height, width), mode=interpolation, align_corners=False, antialias=True
        ).clamp(0.0, 1.0)
        return result.movedim(1, -1).contiguous()

    scale = max(width / source_w, height / source_h) if mode in {"crop", "cover", "crop and resize"} else min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = torch.nn.functional.interpolate(
        nchw, size=(resized_h, resized_w), mode=interpolation, align_corners=False, antialias=True
    ).clamp(0.0, 1.0)
    if mode in {"crop", "cover", "crop and resize"}:
        left, top = max(0, (resized_w - width) // 2), max(0, (resized_h - height) // 2)
        return resized[:, :, top : top + height, left : left + width].movedim(1, -1).contiguous()
    canvas = torch.full(
        (resized.shape[0], resized.shape[1], height, width),
        float(background), dtype=resized.dtype, device=resized.device,
    )
    left, top = (width - resized_w) // 2, (height - resized_h) // 2
    canvas[:, :, top : top + resized_h, left : left + resized_w] = resized
    return canvas.movedim(1, -1).contiguous()


def _resize_mask(mask: Any, width: int, height: int, *, blur: bool = False):
    torch = require_torch()
    if not torch.is_tensor(mask):
        raise ValueError("mask must be a ComfyUI MASK tensor.")
    while mask.ndim > 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError("mask must have shape [B,H,W].")
    interpolate_options = {
        "size": (int(height), int(width)),
        "mode": "bilinear" if blur else "nearest",
    }
    if blur:
        interpolate_options["align_corners"] = False
    resized = torch.nn.functional.interpolate(
        mask.unsqueeze(1).float(), **interpolate_options
    )
    return resized[:, 0].clamp(0, 1).to(mask.dtype)


def _mask_from_image(image: Any):
    image = _image_tensor(image, "mask image")
    if image.shape[-1] >= 4:
        return image[..., 3]
    return image[..., :3].float().mean(dim=-1).to(image.dtype)


def _blur_and_grow_mask(mask: Any, blur: int, grow: int):
    torch = require_torch()
    work = mask.unsqueeze(1).float()
    if int(grow) != 0:
        radius = abs(int(grow))
        kernel = radius * 2 + 1
        if grow > 0:
            work = torch.nn.functional.max_pool2d(work, kernel, stride=1, padding=radius)
        else:
            work = -torch.nn.functional.max_pool2d(-work, kernel, stride=1, padding=radius)
    if int(blur) > 0:
        radius = int(blur)
        kernel = radius * 2 + 1
        work = torch.nn.functional.avg_pool2d(work, kernel, stride=1, padding=radius)
    return work[:, 0].clamp(0, 1).to(mask.dtype)


def reference_target_box(width: int, height: int, layout: str, fraction: float, gap: int) -> tuple[int, int, int, int]:
    """Return the generated panel box for the shared reference canvas."""

    width, height = int(width), int(height)
    fraction = min(0.9, max(0.1, float(fraction)))
    gap = max(0, int(gap))
    layout = str(layout or "reference_left").casefold()
    if layout in {"reference_top", "top"}:
        edge = min(height, int(round(height * fraction)) + gap)
        return 0, edge, width, height
    if layout in {"reference_bottom", "bottom"}:
        edge = max(0, int(round(height * (1.0 - fraction))) - gap)
        return 0, 0, width, edge
    if layout in {"reference_right", "right"}:
        edge = max(0, int(round(width * (1.0 - fraction))) - gap)
        return 0, 0, edge, height
    edge = min(width, int(round(width * fraction)) + gap)
    return edge, 0, width, height


def compose_reference_canvas(image: Any, reference: Any, width: int, height: int, layout: str, fraction: float, gap: int, background: float, fit: str):
    torch = require_torch()
    target = resize_image(image, width, height, "stretch")
    result = torch.full_like(target, float(background))
    left, top, right, bottom = reference_target_box(width, height, layout, fraction, gap)
    # Reference box is the complement of the generated target box.
    layout_key = str(layout or "reference_left").casefold()
    if layout_key in {"reference_top", "top"}:
        box = (0, 0, width, max(1, top - max(0, int(gap))))
    elif layout_key in {"reference_bottom", "bottom"}:
        box = (0, min(height - 1, bottom + max(0, int(gap))), width, height)
    elif layout_key in {"reference_right", "right"}:
        box = (min(width - 1, right + max(0, int(gap))), 0, width, height)
    else:
        box = (0, 0, max(1, left - max(0, int(gap))), height)
    x0, y0, x1, y1 = box
    ref = resize_image(reference, x1 - x0, y1 - y0, fit, background)
    if ref.shape[0] == 1 and result.shape[0] > 1:
        ref = ref.repeat(result.shape[0], 1, 1, 1)
    result[:, y0:y1, x0:x1] = ref[: result.shape[0]]
    # The target side remains the supplied image (usually a neutral blank).
    result[:, top:bottom, left:right] = target[:, top:bottom, left:right]
    mask = torch.zeros((result.shape[0], height, width), dtype=result.dtype, device=result.device)
    mask[:, top:bottom, left:right] = 1
    return result, mask


def _encode(vae: Any, image: Any):
    encode = getattr(vae, "encode", None)
    if not callable(encode):
        raise RuntimeError("A valid ComfyUI VAE is required for image encoding.")
    return {"samples": encode(image[..., :3])}


class ForgeNeoLatentInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",), "mode": (["txt2img", "img2img", "inpaint"],),
            "width": ("INT", {"default": 1536, "min": 16, "max": 16384, "step": 8}),
            "height": ("INT", {"default": 1536, "min": 16, "max": 16384, "step": 8}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            "fit": (["crop", "contain", "stretch"],),
            "mask_invert": ("BOOLEAN", {"default": False}),
            "mask_blur": ("INT", {"default": 0, "min": 0, "max": 256}),
            "grow_mask_by": ("INT", {"default": 6, "min": -256, "max": 256}),
            "reference_enabled": ("BOOLEAN", {"default": False}),
            "reference_layout": (["reference_left", "reference_right", "reference_top", "reference_bottom"],),
            "reference_fraction": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9}),
            "reference_gap": ("INT", {"default": 0, "min": 0, "max": 1024}),
            "reference_background": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "reference_fit": (["contain", "crop", "stretch"],),
        }, "optional": {
            "img2img_image": ("IMAGE",), "inpaint_image": ("IMAGE",),
            "inpaint_mask_image": ("IMAGE",), "inpaint_mask": ("MASK",),
            "reference_image": ("IMAGE",),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "make"
    CATEGORY = CATEGORY

    def make(self, vae, mode="txt2img", width=1536, height=1536, batch_size=1, fit="crop", mask_invert=False, mask_blur=0, grow_mask_by=6, reference_enabled=False, reference_layout="reference_left", reference_fraction=0.5, reference_gap=0, reference_background=0.0, reference_fit="contain", img2img_image=None, inpaint_image=None, inpaint_mask_image=None, inpaint_mask=None, reference_image=None):
        torch = require_torch()
        width, height, batch_size = int(width), int(height), int(batch_size)
        if mode == "txt2img" and not reference_enabled:
            return invoke_provider(
                "EmptyLatentImage", method="generate", feature="txt2img latent",
                args=(width, height, batch_size),
            )[:1]
        source = inpaint_image if mode == "inpaint" else img2img_image
        if source is None:
            # Reference txt2img needs a target panel; use a neutral image.
            if mode == "txt2img" and reference_enabled:
                source = torch.full((batch_size, height, width, 3), float(reference_background))
            else:
                raise RuntimeError(f"{mode} mode requires its source IMAGE input.")
        source_height, source_width = source.shape[1:3]
        source = resize_image(source, width, height, fit, reference_background)
        mask = None
        if reference_enabled:
            if reference_image is None:
                raise RuntimeError("Character reference is enabled but reference_image is not connected.")
            source, reference_mask = compose_reference_canvas(
                source, reference_image, width, height, reference_layout,
                reference_fraction, reference_gap, reference_background, reference_fit,
            )
            mask = reference_mask
            mode = "inpaint"
        if mode == "img2img":
            latent = _encode(vae, source)
        elif mode == "inpaint":
            if mask is None:
                if inpaint_mask is not None:
                    mask = inpaint_mask
                elif inpaint_mask_image is not None:
                    mask = _mask_from_image(inpaint_mask_image)
                else:
                    raise RuntimeError("inpaint mode requires a MASK or mask IMAGE input.")
                # A mask is drawn in source-image coordinates. Use the same
                # scale, center crop and letterbox as its image, with zero
                # outside the image; stretching it directly shifts the edit.
                mask = _resize_mask(mask, source_width, source_height, blur=True)
                mask = resize_image(
                    mask.unsqueeze(-1), width, height, fit, 0.0,
                    interpolation="bilinear",
                )[..., 0].clamp(0, 1)
            if mask_invert:
                mask = 1.0 - mask
            mask = _blur_and_grow_mask(mask, int(mask_blur), int(grow_mask_by))
            latent = invoke_provider(
                "VAEEncodeForInpaint", method="encode", feature="inpaint",
                args=(vae, source[..., :3], mask, 0),
            )[0]
        else:
            latent = _encode(vae, source)
        samples = latent["samples"]
        if samples.shape[0] == 1 and batch_size > 1:
            latent = dict(latent)
            latent["samples"] = samples.repeat(batch_size, *([1] * (samples.ndim - 1)))
            if "noise_mask" in latent:
                latent["noise_mask"] = latent["noise_mask"].repeat(batch_size, *([1] * (latent["noise_mask"].ndim - 1)))
        return (latent,)


def _common_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise):
    require_torch()
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image, latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    batch_inds = latent.get("batch_index")
    # The initial noise is never coloured: CNS colours each step's ancestral
    # noise through the MODEL's SAMPLER_SAMPLE wrapper (guidance_cns.apply_cns).
    noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_inds)
    callback = latent_preview.prepare_callback(model, int(steps))
    samples = comfy.sample.sample(
        model, noise, int(steps), float(cfg), sampler_name, scheduler,
        positive, negative, latent_image, denoise=float(denoise),
        noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed),
    )
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    return (output,)


def _with_cns(model, enabled, strength, gamma_power, gamma_scale):
    """MODEL whose sampler runs colour every step's noise (``guidance_cns``).

    Explicit node inputs replace a CNS wrapper already on the MODEL.  A MODEL
    that only carries the suite's old ``forge_neo_cns`` hand-off dict gets the
    wrapper from it; one that already has the wrapper is left as it is.
    """

    if enabled:
        return apply_cns(model, float(strength), float(gamma_power), float(gamma_scale))
    inherited = (getattr(model, "model_options", {}) or {}).get("forge_neo_cns")
    if isinstance(inherited, dict) and model_cns_settings(model) is None:
        return apply_cns(model, *(
            float(inherited.get(name, CNS_DEFAULTS[name]))
            for name in ("strength", "gamma_power", "gamma_scale")
        ))
    return model


class ForgeNeoKSamplerCNS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "latent_image": ("LATENT",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (sampler_names(),), "scheduler": (scheduler_names(),),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "cns_enabled": ("BOOLEAN", {"default": False}), "cns_strength": cns_float_input("strength"),
            "cns_gamma_power": cns_float_input("gamma_power"),
            "cns_gamma_scale": cns_float_input("gamma_scale"),
            "spectrum_enabled": ("BOOLEAN", {"default": False}), "spectrum_window_size": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0}),
            "spectrum_flex_window": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0}),
            "spectrum_warmup_steps": ("INT", {"default": 6, "min": 0, "max": 10000}),
            "spectrum_tail_actual_steps": ("INT", {"default": 3, "min": 0, "max": 10000}),
            "spectrum_blend_w": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            "spectrum_cheby_degree": ("INT", {"default": 3, "min": 1, "max": 10}),
            "spectrum_ridge_lambda": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 10.0}),
            "spectrum_history_size": ("INT", {"default": 100, "min": 5, "max": 10000}),
            "spectrum_one_sampler_only": ("BOOLEAN", {"default": False}), "spectrum_verbose": ("BOOLEAN", {"default": False}),
            "speed_enabled": ("BOOLEAN", {"default": False}), "speed_split_mode": (["single", "multi"],),
            "speed_spd_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
            "speed_spd_sigma": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            "speed_adaptive_smc_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("output",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise=1.0, cns_enabled=False, cns_strength=1.0, cns_gamma_power=0.5, cns_gamma_scale=2.0, spectrum_enabled=False, spectrum_window_size=2.0, spectrum_flex_window=0.25, spectrum_warmup_steps=6, spectrum_tail_actual_steps=3, spectrum_blend_w=0.3, spectrum_cheby_degree=3, spectrum_ridge_lambda=0.1, spectrum_history_size=100, spectrum_one_sampler_only=False, spectrum_verbose=False, speed_enabled=False, speed_split_mode="single", speed_spd_scale=0.5, speed_spd_sigma=0.7, speed_adaptive_smc_alpha=0.0):
        model = _with_cns(model, cns_enabled, cns_strength, cns_gamma_power, cns_gamma_scale)
        if speed_enabled:
            # CNS rides on the MODEL's sampler wrapper, so it also colours the
            # SPEED sampler's steps (Path B when SPEED wraps sample_*).
            if spectrum_enabled and (
                float(spectrum_window_size), float(spectrum_flex_window), int(spectrum_warmup_steps), int(spectrum_tail_actual_steps),
                float(spectrum_blend_w), int(spectrum_cheby_degree), float(spectrum_ridge_lambda), int(spectrum_history_size),
            ) != (2.0, 0.25, 6, 3, 0.3, 3, 0.1, 100):
                raise RuntimeError("Custom Spectrum tuning cannot be combined with the installed SPEED provider; use Spectrum defaults or disable SPEED.")
            return invoke_provider(
                "SpectrumSPDKSampler", method="sample", feature="SPEED sampler",
                args=(model, int(seed), int(steps), float(cfg), sampler_name, scheduler, positive, negative, latent_image, speed_split_mode, float(speed_spd_scale), float(speed_spd_sigma), float(denoise), float(speed_adaptive_smc_alpha)),
            )[:1]
        if spectrum_enabled:
            # A new provider/model-options state for each sampling invocation.
            # ModelPatcher.clone() already copies model_options dict/list
            # containers (comfy.utils.deepcopy_list_dict) while sharing tensors
            # and GPU handles, so the upstream cached MODEL is never modified.
            model = model.clone()
            model = invoke_provider(
                "DiTSpectrumPatch", method="patch", feature="Spectrum sampler",
                args=(model, int(steps), float(spectrum_window_size), float(spectrum_flex_window), int(spectrum_warmup_steps), int(spectrum_tail_actual_steps), float(spectrum_blend_w), int(spectrum_cheby_degree), float(spectrum_ridge_lambda), int(spectrum_history_size), True, bool(spectrum_one_sampler_only), bool(spectrum_verbose)),
            )[0]
        return _common_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)


def _patch_flow_shift(model: Any, shift: float):
    """Patch a discrete-flow shift without changing its timestep unit.

    Comfy's ``ModelSamplingSD3`` defaults to a 1000-unit timestep multiplier,
    while Anima is trained with multiplier 1.0.  Forge changes only ``shift``
    on the existing predictor, so forwarding the model's current multiplier is
    required for parity (and prevents Anima generations from remaining noise).
    """

    multiplier = 1000.0
    get_model_object = getattr(model, "get_model_object", None)
    if callable(get_model_object):
        original = get_model_object("model_sampling")
        current = getattr(original, "multiplier", None)
        if current is not None:
            multiplier = float(current)
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise RuntimeError(
            f"Invalid discrete-flow timestep multiplier: {multiplier!r}"
        )
    return invoke_provider(
        "ModelSamplingSD3",
        method="patch",
        feature="Forge flow shift",
        args=(model, float(shift), multiplier),
    )[0]


class ForgeNeoModelSamplingShift:
    """Forge-compatible flow shift that preserves the model timestep scale."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "shift": (
                "FLOAT",
                {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01},
            ),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, shift):
        return (_patch_flow_shift(model, shift),)


class ForgeNeoHiresFix:
    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_choices = list(dict.fromkeys([
            "Use same checkpoint",
            *filename_choices("checkpoints"),
            *filename_choices("diffusion_models"),
        ]))
        upscale_choices = list(dict.fromkeys([
            "latent:bislerp", "latent:bicubic", "latent:bilinear",
            "latent:nearest-exact", "latent:area",
            *filename_choices("upscale_models"),
        ]))
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "samples": ("LATENT",),
            "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}), "enabled": ("BOOLEAN", {"default": False}),
            "scale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0}),
            "upscale_method": (upscale_choices,),
            "steps": ("INT", {"default": 0, "min": 0, "max": 10000}), "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0}),
            "sampler_name": (sampler_names(),), "scheduler": (scheduler_names(),),
            "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}), "seed_delta": ("INT", {"default": 0, "min": -0xFFFFFFFF, "max": 0xFFFFFFFF}),
            "cns_enabled": ("BOOLEAN", {"default": False}), "cns_strength": cns_float_input("strength"),
            "cns_gamma_power": cns_float_input("gamma_power"), "cns_gamma_scale": cns_float_input("gamma_scale"),
            "base_vae": ("VAE",), "base_clip": ("CLIP",), "base_steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
            "base_sampler_name": (sampler_names(),), "base_scheduler": (scheduler_names(),),
            "resize_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}), "resize_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}), "checkpoint_name": (checkpoint_choices,),
            "checkpoint_weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
            "text_encoder_name": (["Use same choices", *filename_choices("text_encoders")],), "vae_name": (["Use same choices", *filename_choices("vae")],),
            "positive_text": ("STRING", {"default": "", "multiline": True}), "negative_text": ("STRING", {"default": "", "multiline": True}),
            "base_positive_text": ("STRING", {"default": "", "multiline": True}), "base_negative_text": ("STRING", {"default": "", "multiline": True}),
        }, "optional": {
            "clip_type": (["stable_diffusion", "cosmos", "sd3", "qwen_image", "wan", "krea2"],),
        }}

    RETURN_TYPES = ("LATENT", "VAE")
    RETURN_NAMES = ("output", "decode_vae")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, positive, negative, samples, seed=-1, enabled=False, scale_by=1.5, upscale_method="latent:bislerp", steps=0, cfg=5.0, sampler_name="euler", scheduler="simple", denoise=0.35, seed_delta=0, cns_enabled=False, cns_strength=1.0, cns_gamma_power=0.5, cns_gamma_scale=2.0, base_vae=None, base_clip=None, base_steps=30, base_sampler_name="euler", base_scheduler="simple", resize_width=0, resize_height=0, shift=3.0, checkpoint_name="Use same checkpoint", checkpoint_weight_dtype="default", text_encoder_name="Use same choices", vae_name="Use same choices", positive_text="", negative_text="", base_positive_text="", base_negative_text="", clip_type="stable_diffusion"):
        if not enabled:
            return samples, base_vae
        import comfy.utils

        active_model, active_clip, active_vae = model, base_clip, base_vae
        if not is_disabled_choice(checkpoint_name):
            paths = folder_paths_module()
            full_checkpoint = paths.get_full_path("checkpoints", checkpoint_name)
            diffusion_model = paths.get_full_path("diffusion_models", checkpoint_name)
            if full_checkpoint is not None:
                if checkpoint_weight_dtype != "default":
                    raise RuntimeError(
                        "Hi-res full-checkpoint override cannot honor a non-default weight dtype in ComfyUI; select the UNET/diffusion-model file or use default dtype."
                    )
                loaded = invoke_provider("CheckpointLoaderSimple", method="load_checkpoint", feature="hires checkpoint override", args=(checkpoint_name,))
                active_model = loaded[0]
                if len(loaded) > 1:
                    active_clip = loaded[1]
                if len(loaded) > 2:
                    active_vae = loaded[2]
            elif diffusion_model is not None:
                active_model = invoke_provider(
                    "UNETLoader", method="load_unet", feature="hires UNET override",
                    args=(checkpoint_name, checkpoint_weight_dtype),
                )[0]
            else:
                raise RuntimeError(
                    f"Hi-res model override {checkpoint_name!r} is neither a Comfy checkpoint nor a diffusion-model resource."
                )
        if not is_disabled_choice(text_encoder_name):
            active_clip = invoke_provider("CLIPLoader", method="load_clip", feature="hires text encoder override", args=(text_encoder_name, clip_type, "default"))[0]
        if not is_disabled_choice(vae_name):
            active_vae = invoke_provider("VAELoader", method="load_vae", feature="hires VAE override", args=(vae_name,))[0]
        clip_changed = active_clip is not base_clip
        if positive_text.strip() or negative_text.strip() or clip_changed:
            if active_clip is None:
                raise RuntimeError("Hi-res prompt override requires a CLIP input or text-encoder override.")
            positive_source = positive_text if positive_text.strip() else base_positive_text
            negative_source = negative_text if negative_text.strip() else base_negative_text
            if positive_text.strip() or clip_changed:
                positive = invoke_provider("CLIPTextEncode", method="encode", feature="hires positive prompt", args=(active_clip, positive_source))[0]
            if negative_text.strip() or clip_changed:
                negative = invoke_provider("CLIPTextEncode", method="encode", feature="hires negative prompt", args=(active_clip, negative_source))[0]
        if float(shift) > 0:
            active_model = _patch_flow_shift(active_model, float(shift))
        tensor = samples["samples"]
        ratio_method = getattr(active_vae, "spacial_compression_decode", None)
        try:
            spatial_ratio = max(1, int(ratio_method())) if callable(ratio_method) else 8
        except Exception:
            spatial_ratio = 8
        target_w = int(resize_width) // spatial_ratio if int(resize_width) > 0 else max(1, int(round(tensor.shape[-1] * float(scale_by))))
        target_h = int(resize_height) // spatial_ratio if int(resize_height) > 0 else max(1, int(round(tensor.shape[-2] * float(scale_by))))
        if str(upscale_method).startswith("latent:"):
            method = str(upscale_method).split(":", 1)[1]
            resized = dict(samples)
            resized["samples"] = comfy.utils.common_upscale(
                tensor, target_w, target_h, method, "disabled"
            )
        else:
            if base_vae is None or active_vae is None:
                raise RuntimeError("Pixel-space Hi-res upscaling requires both base_vae and active VAE inputs.")
            decoded = invoke_provider(
                "VAEDecode", method="decode", feature="Hi-res pixel decode",
                args=(base_vae, samples),
            )[0]
            upscale_model = invoke_provider(
                "UpscaleModelLoader", method="load_model", feature="Hi-res pixel upscaler",
                args=(upscale_method,),
            )[0]
            upscaled = invoke_provider(
                "ImageUpscaleWithModel", method="upscale", feature="Hi-res pixel upscaler",
                args=(upscale_model, decoded),
            )[0]
            pixel_width = int(resize_width) if int(resize_width) > 0 else target_w * spatial_ratio
            pixel_height = int(resize_height) if int(resize_height) > 0 else target_h * spatial_ratio
            upscaled = resize_image(upscaled, pixel_width, pixel_height, "stretch")
            resized = _encode(active_vae, upscaled)
        actual_seed = secrets.randbits(64) if int(seed) < 0 else (int(seed) + int(seed_delta)) & 0xFFFFFFFFFFFFFFFF
        actual_steps = int(steps) if int(steps) > 0 else int(base_steps)
        active_model = _with_cns(active_model, cns_enabled, cns_strength, cns_gamma_power, cns_gamma_scale)
        output = _common_sample(active_model, actual_seed, actual_steps, cfg, sampler_name or base_sampler_name, scheduler or base_scheduler, positive, negative, resized, denoise)[0]
        return output, active_vae


class ForgeNeoMaskSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"enabled": ("BOOLEAN", {"default": False})}, "optional": {"manual_mask": ("MASK",), "generated_mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select"
    CATEGORY = CATEGORY

    def select(self, enabled=False, manual_mask=None, generated_mask=None):
        selected = generated_mask if enabled else manual_mask
        if selected is None:
            selected = manual_mask if generated_mask is None else generated_mask
        if selected is None:
            raise RuntimeError("Mask selector has no connected mask.")
        return (selected,)


class ForgeNeoLoraBlockWeight:
    def __init__(self):
        self._anima_lora_cache = AnimaLoraStateCache()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "clip": ("CLIP",), "enabled": ("BOOLEAN", {"default": False}),
            "lora_name": (filename_choices("loras"),), "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}), "inverse": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}), "A": ("FLOAT", {"default": 4.0, "min": -10.0, "max": 10.0}),
            "B": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}), "preset": (["Preset"],),
            "block_vector": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": (
                    "ANIMA requires exactly 28, 40, or 52 block values for the "
                    "active model; an optional leading value controls non-block keys. "
                    "Leave empty to use weight 1 for the base and every block."
                ),
            }),
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "populated_vector")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Uses native per-block weighting for ANIMA 28/40/52 models and Inspire "
        "Pack weighting for other architectures."
    )

    def load(self, model, clip, enabled=False, lora_name="None", strength_model=1.0, strength_clip=1.0, inverse=False, seed=0, A=4.0, B=1.0, preset="Preset", block_vector=""):
        if not enabled:
            return model, clip, str(block_vector)
        if is_disabled_choice(lora_name):
            raise RuntimeError("LoRA block weighting is enabled but no LoRA is selected.")
        return load_lora_block_weight(
            model,
            clip,
            lora_name,
            strength_model,
            strength_clip,
            inverse,
            seed,
            A,
            B,
            block_vector,
            cache=self._anima_lora_cache,
        )


def compose_reference_prompt(text: str, enabled: bool, prefix: str) -> str:
    base, leading = str(text or "").strip(), str(prefix or "").strip()
    if not enabled or not leading:
        return base
    return ", ".join(part for part in (leading, base) if part)


class ForgeNeoReferencePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"default": "", "multiline": True}), "enabled": ("BOOLEAN", {"default": False}), "prefix": ("STRING", {"default": "(split screen, multiple view:1.2)"})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    def compose(self, text="", enabled=False, prefix=""):
        return (compose_reference_prompt(text, enabled, prefix),)


class ForgeNeoCharacterReference:
    def __init__(self):
        self._anima_lora_cache = AnimaLoraStateCache()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",), "vae": ("VAE",),
            "enabled": ("BOOLEAN", {"default": False}), "lora_name": (["None", *filename_choices("loras")],),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0}),
            "reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0}),
            "reference_method": (["temporal_mask", "temporal_concat", "split_screen"],),
            "reference_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
            "reference_height": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 8}),
            "reference_fit": (["contain", "crop", "stretch"],), "reference_image": ("IMAGE",),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, positive, negative, vae, enabled=False, lora_name="None", lora_strength=1.0, reference_strength=1.0, reference_method="temporal_mask", reference_width=1024, reference_height=1536, reference_fit="contain", reference_image=None):
        if not enabled:
            return model, positive, negative
        if reference_image is None:
            raise RuntimeError("Character Reference is enabled but reference_image is missing.")
        if reference_method == "temporal_mask":
            raise RuntimeError(
                "Character Reference temporal_mask is not exposed by ComfyUI's current "
                "Anima API. Use temporal_concat or split_screen; temporal_mask is not "
                "silently downgraded to temporal_concat."
            )
        patched = model
        if not is_disabled_choice(lora_name):
            patched = load_lora_model_only(
                patched,
                lora_name,
                lora_strength,
                cache=self._anima_lora_cache,
            )
        if reference_method == "split_screen":
            # The actual reference pixels are carried by ForgeNeoLatentInput's
            # split canvas. This node still validates and loads the edit LoRA.
            return patched, positive, negative
        reference = resize_image(reference_image, int(reference_width), int(reference_height), reference_fit)
        ref_latent = _encode(vae, reference)["samples"] * float(reference_strength)
        options = getattr(patched, "model_options", {}) or {}
        if options.get("model_function_wrapper") is not None:
            raise RuntimeError("Character Reference temporal injection cannot overwrite another model_function_wrapper; place it before that patch or use split_screen.")
        patched = patched.clone()

        def reference_wrapper(apply_model, args):
            torch = require_torch()
            x, timestep, cond = args["input"], args["timestep"], args.get("c") or {}
            original_ndim = x.ndim
            work = x.unsqueeze(2) if original_ndim == 4 else x
            ref = ref_latent.to(device=work.device, dtype=work.dtype)
            if ref.ndim == 4:
                ref = ref.unsqueeze(2)
            if ref.shape[-2:] != work.shape[-2:]:
                flat = ref.movedim(2, 1).reshape(-1, ref.shape[1], *ref.shape[-2:])
                flat = torch.nn.functional.interpolate(flat, size=work.shape[-2:], mode="bilinear", align_corners=False)
                ref = flat.reshape(ref.shape[0], -1, ref.shape[1], *work.shape[-2:]).movedim(1, 2)
            if ref.shape[0] != work.shape[0]:
                ref = ref[:1].repeat(work.shape[0], 1, 1, 1, 1)
            target_frames = work.shape[2]
            merged = torch.cat((work, ref), dim=2)
            output = apply_model(merged, timestep, **cond)[:, :, :target_frames]
            return output.squeeze(2) if original_ndim == 4 else output

        patched.set_model_unet_function_wrapper(reference_wrapper)
        return patched, positive, negative


class ForgeNeoReferenceOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "enabled": ("BOOLEAN", {"default": False}),
            "layout": (["reference_left", "reference_right", "reference_top", "reference_bottom"],),
            "fraction": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9}), "gap": ("INT", {"default": 0, "min": 0, "max": 1024}),
            "output_width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
            "output_height": ("INT", {"default": 1536, "min": 16, "max": 16384, "step": 8}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"
    CATEGORY = CATEGORY

    def crop(self, image, enabled=False, layout="reference_left", fraction=0.5, gap=0, output_width=1024, output_height=1536):
        if not enabled:
            return (image,)
        image = _image_tensor(image)
        left, top, right, bottom = reference_target_box(image.shape[2], image.shape[1], layout, fraction, gap)
        if right <= left or bottom <= top:
            raise ValueError("Reference output crop has an empty target region.")
        cropped = image[:, top:bottom, left:right]
        return (resize_image(cropped, int(output_width), int(output_height), "stretch"),)


def _vae2x_transform(image: Any, *, refine_1x: bool, blur_sigma: float):
    torch = require_torch()
    if image.ndim == 5:
        image = image[:, 0]
    image = _image_tensor(image, "12-channel VAE output")
    if int(image.shape[-1]) != 12:
        raise RuntimeError(
            f"Anima VAE 2x expected a 12-channel decoder output, got {int(image.shape[-1])}."
        )
    output = torch.nn.functional.pixel_shuffle(image.movedim(-1, 1), 2)
    if refine_1x:
        output = torch.nn.functional.interpolate(
            output, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        sigma = float(blur_sigma)
        if sigma > 0.0:
            radius = max(1, int(round(sigma * 2.0)))
            coords = torch.arange(-radius, radius + 1, device=output.device, dtype=torch.float32)
            kernel = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
            kernel = (kernel / kernel.sum()).to(output.dtype)
            channels = int(output.shape[1])
            vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
            horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
            output = torch.nn.functional.conv2d(
                output, vertical, padding=(radius, 0), groups=channels
            )
            output = torch.nn.functional.conv2d(
                output, horizontal, padding=(0, radius), groups=channels
            )
    return output.movedim(1, -1).clamp(0, 1).contiguous()


class _ComfyVAE2xWrapper:
    def __init__(self, base_vae, decoder_vae, refine_1x: bool, blur_sigma: float, renorm: bool):
        object.__setattr__(self, "_base_vae", base_vae)
        object.__setattr__(self, "_decoder_vae", decoder_vae)
        object.__setattr__(self, "_refine_1x", bool(refine_1x))
        object.__setattr__(self, "_blur_sigma", float(blur_sigma))
        object.__setattr__(self, "_renorm", bool(renorm))

    def _latent(self, samples):
        if not self._renorm:
            return samples
        dims = tuple(range(1, samples.ndim))
        mean = samples.float().mean(dim=dims, keepdim=True)
        std = samples.float().std(dim=dims, keepdim=True).clamp_min(1e-6)
        return ((samples.float() - mean) / std).to(samples.dtype)

    def decode(self, samples, *args, **kwargs):
        pixels = self._decoder_vae.decode(self._latent(samples), *args, **kwargs)
        return _vae2x_transform(
            pixels, refine_1x=self._refine_1x, blur_sigma=self._blur_sigma
        )

    def decode_tiled(self, samples, *args, **kwargs):
        pixels = self._decoder_vae.decode_tiled(
            self._latent(samples), *args, **kwargs
        )
        return _vae2x_transform(
            pixels, refine_1x=self._refine_1x, blur_sigma=self._blur_sigma
        )

    def clone(self):
        return _ComfyVAE2xWrapper(
            self._base_vae, self._decoder_vae, self._refine_1x,
            self._blur_sigma, self._renorm,
        )

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base_vae"), name)


class ForgeNeoAnimaVAE2x:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",), "enabled": ("BOOLEAN", {"default": False}),
            "vae_name": (["None", *filename_choices("vae")],),
            "output_mode": (["1x refined (downsample)", "2x upscaled"],),
            "blur_sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            "latent_renorm": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, vae, enabled=False, vae_name="None", output_mode="1x refined (downsample)", blur_sigma=0.5, latent_renorm=False):
        if not enabled:
            return (vae,)
        if is_disabled_choice(vae_name):
            raise RuntimeError("Anima VAE 2x is enabled but no 12-channel VAE is selected.")
        decoder = invoke_provider(
            "VAELoader", method="load_vae", feature="Anima VAE 2x decoder",
            args=(vae_name,),
        )[0]
        output_channels = getattr(decoder, "conv_out_channels", None)
        if output_channels is None:
            head = getattr(
                getattr(getattr(decoder, "first_stage_model", None), "decoder", None),
                "head", None,
            )
            try:
                output_channels = int(head[2].weight.shape[0])
            except Exception:
                output_channels = None
        if int(output_channels or 0) != 12:
            raise RuntimeError(
                f"{vae_name!r} is not a spacepxl 2x VAE: decoder.head.2 must output 12 channels."
            )
        # Comfy's current Wan loader records the encoder image channels (3)
        # as output_channels. Its tiled compositor must instead allocate all
        # twelve decoder channels before the pixel-shuffle wrapper runs.
        decoder.output_channels = 12
        return (_ComfyVAE2xWrapper(
            vae, decoder, str(output_mode).startswith("1x"),
            float(blur_sigma), bool(latent_renorm),
        ),)


class ForgeNeoAnimaPiD:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "latent": ("LATENT",), "enabled": ("BOOLEAN", {"default": False}),
            "ckpt_name": (["(auto-download)", *filename_choices("checkpoints")],),
            "dtype": (["bf16", "fp32"],), "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
            "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}), "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
            "tile_latent": ("INT", {"default": 64, "min": 0, "max": 2048}), "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 1024}),
            "compile": ("BOOLEAN", {"default": False}), "use_calib": ("BOOLEAN", {"default": False}),
            "attention": (["auto", "sdpa", "sage"],),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, image, latent, enabled=False, ckpt_name="(auto-download)", dtype="bf16", steps=4, sigma=0.0, seed=-1, tile_latent=64, tile_overlap=16, compile=False, use_calib=False, attention="auto"):
        if not enabled:
            return (image,)
        if ckpt_name == "(auto-download)":
            raise RuntimeError("PiD is enabled but no local PiD checkpoint is selected. ComfyUI has PiD model support but does not provide Forge's auto-downloader.")
        if compile or use_calib or attention != "auto":
            raise RuntimeError("The selected PiD compile/calibration/attention override is not exposed by ComfyUI's current PiD API.")
        if str(dtype).casefold() != "bf16":
            raise RuntimeError(
                "PiD dtype override is not exposed by ComfyUI's current PiD loader; "
                "only its native bf16 path is available."
            )
        if (int(tile_latent), int(tile_overlap)) != (64, 16):
            raise RuntimeError(
                "PiD tile_latent/tile_overlap overrides are not exposed by ComfyUI's "
                "current PiDConditioning API."
            )
        loaded = invoke_provider("CheckpointLoaderSimple", method="load_checkpoint", feature="PiD decoder", args=(ckpt_name,))
        if len(loaded) < 2:
            raise RuntimeError("The PiD checkpoint did not provide both MODEL and CLIP.")
        pid_model, pid_clip = loaded[0], loaded[1]
        positive = invoke_provider("CLIPTextEncode", method="encode", feature="PiD decoder", args=(pid_clip, ""))[0]
        conditioned = invoke_provider("PiDConditioning", method="execute", feature="PiD conditioning", args=(positive, latent, "qwenimage", float(sigma)))[0]
        negative = invoke_provider("ConditioningZeroOut", method="zero_out", feature="PiD decoder", args=(conditioned,))[0]
        torch = require_torch()
        source = _image_tensor(image)
        target_h, target_w = int(source.shape[1]) * 4, int(source.shape[2]) * 4
        empty = {"samples": torch.zeros((source.shape[0], 3, target_h, target_w), dtype=source.dtype, device=source.device)}
        actual_seed = secrets.randbits(64) if int(seed) < 0 else int(seed)
        result = _common_sample(pid_model, actual_seed, int(steps), 1.0, "euler", "simple", conditioned, negative, empty, 1.0)[0]["samples"]
        latent_format = getattr(getattr(pid_model, "model", None), "latent_format", None)
        if latent_format is not None and callable(getattr(latent_format, "process_out", None)):
            result = latent_format.process_out(result)
        return (result.movedim(1, -1).clamp(0, 1).contiguous(),)


# Forge/A1111 sampler labels -> ComfyUI names.  This pack runs inside ComfyUI
# and cannot import the app, so these tables mirror the app compiler's
# (core/comfy_workflow_compiler._FORGE_SAMPLER_ALIASES & co.);
# tests/test_comfy_sampler_aliases.py pins both to the same results.
_AD_SAMPLER_ALIASES = {
    "euler": "euler", "euler a": "euler_ancestral", "euler ancestral": "euler_ancestral",
    "lms": "lms", "heun": "heun", "dpm2": "dpm_2", "dpm2 a": "dpm_2_ancestral",
    "dpm++ 2s a": "dpmpp_2s_ancestral", "dpm++ 2m": "dpmpp_2m",
    "dpm++ sde": "dpmpp_sde", "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 2m sde heun": "dpmpp_2m_sde_heun", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm fast": "dpm_fast", "dpm adaptive": "dpm_adaptive",
    "unipc": "uni_pc", "uni pc": "uni_pc", "uni_pc": "uni_pc",
    "lcm": "lcm", "ddim": "ddim", "er sde": "er_sde", "er-sde": "er_sde",
}

_AD_SAMPLER_SCHEDULER_SUFFIXES = (
    (" karras", "karras"), (" exponential", "exponential"),
    (" sgm uniform", "sgm_uniform"), (" beta", "beta"),
)

_AD_SCHEDULER_ALIASES = {
    "karras": "karras", "exponential": "exponential",
    "sgm uniform": "sgm_uniform", "sgm_uniform": "sgm_uniform",
    "simple": "simple", "normal": "normal",
    "ddim uniform": "ddim_uniform", "ddim_uniform": "ddim_uniform",
    "beta": "beta",
    "beta57": "beta57", "beta 57": "beta57",
    "beta57 (res4lyf)": "beta57", "beta 57 (res4lyf)": "beta57",
}

_AD_SAME_SCHEDULER = frozenset({"use same scheduler", "same", "automatic", "auto", ""})


def _normalize_ad_sampler(sampler: Any, scheduler: Any) -> tuple[str, str]:
    sampler_text = str(sampler or "euler").strip()
    scheduler_text = str(scheduler or "normal").strip()
    lowered = " ".join(sampler_text.split()).casefold()
    inferred_scheduler = None
    for suffix, resolved in _AD_SAMPLER_SCHEDULER_SUFFIXES:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)].strip()
            inferred_scheduler = resolved
            break
    if lowered == "plms":
        raise RuntimeError("ADetailer sampler PLMS has no equivalent in ComfyUI's KSampler.")
    normalized_sampler = _AD_SAMPLER_ALIASES.get(lowered, lowered.replace(" ", "_"))
    scheduler_lower = " ".join(scheduler_text.split()).casefold()
    if scheduler_lower in _AD_SAME_SCHEDULER:
        normalized_scheduler = inferred_scheduler or "normal"
    else:
        normalized_scheduler = _AD_SCHEDULER_ALIASES.get(
            scheduler_lower, scheduler_lower.replace(" ", "_")
        )
    return normalized_sampler, inferred_scheduler or normalized_scheduler


def normalize_adetailer_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate an ADetailer-NeoForge slot into Impact FaceDetailer inputs."""

    unsupported: list[str] = []
    if bool(settings.get("ad_hires_fix_only", False)):
        unsupported.append("ad_hires_fix_only")
    if str(settings.get("ad_model_classes") or "").strip():
        unsupported.append("ad_model_classes")
    if bool(settings.get("ad_use_autotag", False)):
        unsupported.append("ad_use_autotag")
    if bool(settings.get("ad_copy_main_lora_triggers", False)) or bool(
        settings.get("ad_copy_main_lora_triggers_only", False)
    ):
        unsupported.append("LoRA trigger copying")
    if str(settings.get("ad_mask_filter_method", "Area")) != "Area":
        unsupported.append("ad_mask_filter_method")
    if int(settings.get("ad_mask_k", 0) or 0) != 0:
        unsupported.append("ad_mask_k")
    if float(settings.get("ad_mask_min_ratio", 0.0) or 0.0) != 0.0:
        unsupported.append("ad_mask_min_ratio")
    if float(settings.get("ad_mask_max_ratio", 1.0) or 1.0) != 1.0:
        unsupported.append("ad_mask_max_ratio")
    if int(settings.get("ad_x_offset", 0) or 0) != 0 or int(
        settings.get("ad_y_offset", 0) or 0
    ) != 0:
        unsupported.append("ad_x_offset/ad_y_offset")
    if str(settings.get("ad_mask_merge_invert", "None")) != "None":
        unsupported.append("ad_mask_merge_invert")
    if float(settings.get("ad_inpaint_scale", 1.0) or 1.0) != 1.0:
        unsupported.append("ad_inpaint_scale")
    for toggle, label in (
        ("ad_use_checkpoint", "separate checkpoint"),
        ("ad_use_vae", "separate VAE"),
        ("ad_use_noise_multiplier", "noise multiplier"),
        ("ad_use_clip_skip", "CLIP skip"),
        ("ad_restore_face", "restore face"),
    ):
        if bool(settings.get(toggle, False)):
            unsupported.append(label)
    control_model = str(settings.get("ad_controlnet_model", "None") or "None").strip()
    control_module = str(settings.get("ad_controlnet_module", "None") or "None").strip()
    if control_model.casefold() not in {"none", ""} or control_module.casefold() not in {"none", ""}:
        unsupported.append("ADetailer ControlNet")
    if unsupported:
        raise RuntimeError(
            "ADetailer settings are enabled but not representable by Impact FaceDetailer: "
            + ", ".join(unsupported)
            + ". Disable those options or use a native Comfy workflow for them."
        )

    if "sampler_name" in settings:
        raw_sampler = settings.get("sampler_name")
        raw_scheduler = settings.get("scheduler", "normal")
    elif bool(settings.get("ad_use_sampler", False)):
        raw_sampler = settings.get("ad_sampler", "euler")
        raw_scheduler = settings.get("ad_scheduler", "normal")
    else:
        raw_sampler, raw_scheduler = "euler", "normal"
    sampler, scheduler = _normalize_ad_sampler(raw_sampler, raw_scheduler)
    use_dimensions = bool(settings.get("ad_use_inpaint_width_height", False))
    if use_dimensions:
        guide_size = max(
            64, int(settings.get("ad_inpaint_width", 512) or 512),
            int(settings.get("ad_inpaint_height", 512) or 512),
        )
        max_size = guide_size
    else:
        guide_size = float(settings.get("guide_size", 512) or 512)
        max_size = max(guide_size, float(settings.get("max_size", 1024) or 1024))
    mask_blur = int(settings.get("ad_mask_blur", settings.get("feather", 4)) or 0)
    return {
        "guide_size": guide_size,
        "guide_size_for": bool(settings.get("guide_size_for_bbox", True)),
        "max_size": max_size,
        "seed": int(settings.get("seed", settings.get("ad_seed", 0)) or 0),
        "steps": int(settings.get("steps", settings.get("ad_steps", 20)) or 20),
        "cfg": float(settings.get("cfg", settings.get("ad_cfg_scale", 7.0)) or 0.0),
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": float(settings.get("denoise", settings.get("ad_denoising_strength", 0.4)) or 0.0),
        "feather": mask_blur,
        "noise_mask": bool(settings.get("noise_mask", True)),
        "force_inpaint": bool(settings.get("force_inpaint", settings.get("ad_inpaint_only_masked", True))),
        "bbox_threshold": float(settings.get("bbox_threshold", settings.get("ad_confidence", 0.3)) or 0.0),
        "bbox_dilation": int(settings.get("bbox_dilation", settings.get("ad_dilate_erode", 4)) or 0),
        "bbox_crop_factor": float(settings.get("bbox_crop_factor", 3.0) or 3.0),
        "noise_mask_feather": int(settings.get("noise_mask_feather", mask_blur) or 0),
        "wildcard": str(settings.get("prompt", settings.get("ad_prompt", "")) or ""),
        "cycle": int(settings.get("cycle", 1) or 1),
        "inpaint_model": bool(settings.get("inpaint_model", False)),
        "tiled_encode": bool(settings.get("tiled_encode", False)),
        "tiled_decode": bool(settings.get("tiled_decode", False)),
    }


class ForgeNeoADetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "model": ("MODEL",), "clip": ("CLIP",), "vae": ("VAE",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "enabled": ("BOOLEAN", {"default": False}),
            "settings_json": ("STRING", {"default": "{}", "multiline": True}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "report")
    FUNCTION = "detail"
    CATEGORY = CATEGORY

    def detail(self, image, model, clip, vae, positive, negative, enabled=False, settings_json="{}"):
        torch = require_torch()
        image = _image_tensor(image)
        if not enabled:
            return image, torch.zeros(image.shape[:3], dtype=image.dtype, device=image.device), json.dumps({"enabled": False})
        settings = _json_object(settings_json, "ADetailer")
        model_name = str(settings.get("model_name") or settings.get("ad_model") or "").strip()
        if not model_name or model_name.casefold() in {"none", "disabled"}:
            raise RuntimeError("ADetailer is enabled but settings_json has no model_name/ad_model.")
        if not model_name.startswith(("bbox/", "segm/")):
            model_name = "bbox/" + model_name
        resolved = normalize_adetailer_settings(settings)
        detector_result = invoke_provider("UltralyticsDetectorProvider", method="doit", feature="ADetailer detector", args=(model_name,))
        bbox = detector_result[0]
        segm_candidate = detector_result[1] if len(detector_result) > 1 else None
        # Impact Subpack returns a truthy NO_SEGM_DETECTOR sentinel for bbox
        # models. FaceDetailer treats every non-None value as a real detector
        # and calls ``detect`` on it, so normalize that sentinel to None.
        segm = (
            segm_candidate
            if callable(getattr(segm_candidate, "detect", None))
            else None
        )
        result = invoke_provider(
            "FaceDetailer", method="doit", feature="ADetailer",
            kwargs={
                "image": image, "model": model, "clip": clip, "vae": vae,
                "guide_size": resolved["guide_size"],
                "guide_size_for": resolved["guide_size_for"],
                "max_size": resolved["max_size"],
                "seed": resolved["seed"], "steps": resolved["steps"],
                "cfg": resolved["cfg"], "sampler_name": resolved["sampler_name"],
                "scheduler": resolved["scheduler"], "positive": positive, "negative": negative,
                "denoise": resolved["denoise"], "feather": resolved["feather"],
                "noise_mask": resolved["noise_mask"], "force_inpaint": resolved["force_inpaint"],
                "bbox_threshold": resolved["bbox_threshold"],
                "bbox_dilation": resolved["bbox_dilation"],
                "bbox_crop_factor": resolved["bbox_crop_factor"],
                "sam_detection_hint": str(settings.get("sam_detection_hint", "none")),
                "sam_dilation": int(settings.get("sam_dilation", 0)), "sam_threshold": float(settings.get("sam_threshold", 0.93)),
                "sam_bbox_expansion": int(settings.get("sam_bbox_expansion", 0)), "sam_mask_hint_threshold": float(settings.get("sam_mask_hint_threshold", 0.7)),
                "sam_mask_hint_use_negative": str(settings.get("sam_mask_hint_use_negative", "False")),
                "drop_size": int(settings.get("drop_size", 10)), "bbox_detector": bbox,
                "wildcard": resolved["wildcard"],
                "cycle": resolved["cycle"], "segm_detector_opt": segm,
                "inpaint_model": resolved["inpaint_model"],
                "noise_mask_feather": resolved["noise_mask_feather"],
                "tiled_encode": resolved["tiled_encode"], "tiled_decode": resolved["tiled_decode"],
            },
        )
        report = {
            "enabled": True, "model": model_name,
            "sampler": resolved["sampler_name"], "scheduler": resolved["scheduler"],
            "detections": int((result[3] > 0).any(dim=(-2, -1)).sum().item()) if len(result) > 3 else None,
        }
        return result[0], result[3], json.dumps(report, ensure_ascii=False)


class ForgeNeoSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "filename_prefix": ("STRING", {"default": "ForgeNeo/%date:yyyy-MM-dd%/%date:hhmmss%"}),
            "file_format": (["PNG", "JPEG", "WebP"],), "metadata_mode": (["ComfyUI (prompt + workflow)", "None"],),
            "quality": ("INT", {"default": 95, "min": 1, "max": 100}), "webp_lossless": ("BOOLEAN", {"default": False}),
            "collision_mode": (["Number suffix", "Overwrite", "Error"],),
            "save_before_hires": ("BOOLEAN", {"default": False}), "save_img2img_source": ("BOOLEAN", {"default": False}),
            "save_inpaint_mask": ("BOOLEAN", {"default": False}), "save_inpaint_composite": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "before_hires_images": ("IMAGE",), "img2img_source": ("IMAGE",), "inpaint_source": ("IMAGE",), "inpaint_mask": ("MASK",),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, images, filename_prefix="ForgeNeo", file_format="PNG", metadata_mode="ComfyUI (prompt + workflow)", quality=95, webp_lossless=False, collision_mode="Number suffix", save_before_hires=False, save_img2img_source=False, save_inpaint_mask=False, save_inpaint_composite=False, before_hires_images=None, img2img_source=None, inpaint_source=None, inpaint_mask=None, prompt=None, extra_pnginfo=None):
        images = _image_tensor(images)
        if file_format == "PNG" and collision_mode == "Number suffix" and not any((save_before_hires, save_img2img_source, save_inpaint_mask, save_inpaint_composite)):
            return provider("SaveImage", feature="image save").save_images(
                images, filename_prefix, prompt if metadata_mode != "None" else None,
                extra_pnginfo if metadata_mode != "None" else None,
            )
        return self._save_all(
            images, filename_prefix, file_format, metadata_mode, quality,
            webp_lossless, collision_mode, prompt, extra_pnginfo,
            save_before_hires=save_before_hires, before_hires_images=before_hires_images,
            save_img2img_source=save_img2img_source, img2img_source=img2img_source,
            save_inpaint_mask=save_inpaint_mask, inpaint_mask=inpaint_mask,
            save_inpaint_composite=save_inpaint_composite, inpaint_source=inpaint_source,
        )

    def _save_all(self, images, prefix, file_format, metadata_mode, quality, lossless, collision_mode, prompt, extra_pnginfo, **artifacts):
        import os
        import numpy as np
        from PIL import Image, PngImagePlugin

        folder_paths = folder_paths_module()
        extension = {"PNG": "png", "JPEG": "jpg", "WebP": "webp"}[file_format]
        results = []

        def save_batch(batch, suffix=""):
            nonlocal results
            if batch is None:
                raise RuntimeError(f"Saving {suffix or 'requested artifact'} is enabled but its input is not connected.")
            batch = _image_tensor(batch)
            full, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix + suffix, folder_paths.get_output_directory(), batch[0].shape[1], batch[0].shape[0]
            )
            batch_count = len(batch)
            for batch_index, tensor in enumerate(batch):
                image = Image.fromarray(np.clip(tensor.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
                resolved_stem = stem.replace("%batch_num%", str(batch_index))
                if "%batch_num%" not in stem and batch_count > 1:
                    resolved_stem += f"_{batch_index:05}"
                if collision_mode == "Number suffix":
                    filename = f"{resolved_stem}_{counter:05}_.{extension}"
                else:
                    # Overwrite and Error deliberately address the same stable
                    # filename. Error checks it below; Number suffix is the only
                    # mode allowed to silently choose a new numbered name.
                    filename = f"{resolved_stem}.{extension}"
                path = os.path.join(full, filename)
                if collision_mode == "Error" and os.path.exists(path):
                    raise FileExistsError(f"Output already exists: {path}")
                save_kwargs = {}
                if file_format == "PNG":
                    if metadata_mode != "None":
                        pnginfo = PngImagePlugin.PngInfo()
                        if prompt is not None:
                            pnginfo.add_text("prompt", json.dumps(prompt))
                        for key, value in (extra_pnginfo or {}).items():
                            pnginfo.add_text(str(key), json.dumps(value))
                        save_kwargs["pnginfo"] = pnginfo
                elif file_format == "JPEG":
                    save_kwargs["quality"] = int(quality)
                    image = image.convert("RGB")
                else:
                    save_kwargs.update(quality=int(quality), lossless=bool(lossless))
                image.save(path, **save_kwargs)
                if metadata_mode != "None" and file_format != "PNG":
                    with open(path + ".json", "w", encoding="utf-8") as handle:
                        json.dump({"prompt": prompt, "extra_pnginfo": extra_pnginfo}, handle, ensure_ascii=False, indent=2)
                results.append({"filename": filename, "subfolder": subfolder, "type": "output"})
                counter += 1

        save_batch(images)
        if artifacts["save_before_hires"]:
            save_batch(artifacts["before_hires_images"], "-before-hires")
        if artifacts["save_img2img_source"]:
            save_batch(artifacts["img2img_source"], "-img2img-source")
        if artifacts["save_inpaint_mask"]:
            mask = artifacts["inpaint_mask"]
            if mask is None:
                raise RuntimeError("save_inpaint_mask is enabled but inpaint_mask is not connected.")
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)
            save_batch(mask, "-inpaint-mask")
        if artifacts["save_inpaint_composite"]:
            source, mask = artifacts["inpaint_source"], artifacts["inpaint_mask"]
            if source is None or mask is None:
                raise RuntimeError("save_inpaint_composite requires inpaint_source and inpaint_mask.")
            source = resize_image(source, images.shape[2], images.shape[1], "stretch")
            mask = _resize_mask(mask, images.shape[2], images.shape[1], blur=True).unsqueeze(-1)
            save_batch(source * (1 - mask) + images * mask, "-inpaint-composite")
        return {"ui": {"images": results}, "result": (images,)}


NODE_CLASS_MAPPINGS = {
    "ForgeNeoKSamplerCNS": ForgeNeoKSamplerCNS,
    "ForgeNeoCNSSamplerPatch": ForgeNeoCNSSamplerPatch,
    "ForgeNeoModelSamplingShift": ForgeNeoModelSamplingShift,
    "ForgeNeoLatentInput": ForgeNeoLatentInput,
    "ForgeNeoHiresFix": ForgeNeoHiresFix,
    "ForgeNeoMaskSelector": ForgeNeoMaskSelector,
    "ForgeNeoLoraBlockWeight": ForgeNeoLoraBlockWeight,
    "ForgeNeoCharacterReference": ForgeNeoCharacterReference,
    "ForgeNeoReferencePrompt": ForgeNeoReferencePrompt,
    "ForgeNeoReferenceOutput": ForgeNeoReferenceOutput,
    "ForgeNeoAnimaPiD": ForgeNeoAnimaPiD,
    "ForgeNeoAnimaVAE2x": ForgeNeoAnimaVAE2x,
    "ForgeNeoADetailer": ForgeNeoADetailer,
    "ForgeNeoSaveImage": ForgeNeoSaveImage,
}
