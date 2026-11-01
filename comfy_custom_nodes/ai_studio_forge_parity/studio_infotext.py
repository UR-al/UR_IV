"""Forge-style ``parameters`` text (A1111 infotext) for images the Anima studio saves.

Pure Python (no ComfyUI / torch).  The key names and their order follow what Forge Neo writes for txt2img
(``Steps, Sampler, Schedule type, CFG scale, Shift, Seed, Size, Model, ... Denoising strength, Hires ...``), so the
app's PNG Info / Gallery reader and Forge's own "send to" buttons read these files like Forge's.  The label tables
are the reverse of the app compiler's ``_FORGE_SAMPLER_ALIASES`` / ``_FORGE_SCHEDULER_ALIASES`` (the pack cannot
import ``core``); ``tests/test_studio_infotext.py`` pins them against it.
"""
from __future__ import annotations

import json
import re

from .studio_state import effective_prompt, generation_size, size_grid
from .studio_upscale import hires_size

# ComfyUI sampler name -> Forge label.  Unknown names are written as they are.
SAMPLER_LABELS = {
    "euler": "Euler", "euler_ancestral": "Euler a", "lms": "LMS", "heun": "Heun", "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a", "dpmpp_2s_ancestral": "DPM++ 2S a", "dpmpp_2m": "DPM++ 2M",
    "dpmpp_sde": "DPM++ SDE", "dpmpp_2m_sde": "DPM++ 2M SDE", "dpmpp_2m_sde_heun": "DPM++ 2M SDE Heun",
    "dpmpp_3m_sde": "DPM++ 3M SDE", "dpm_fast": "DPM fast", "dpm_adaptive": "DPM adaptive", "uni_pc": "UniPC",
    "lcm": "LCM", "ddim": "DDIM", "er_sde": "ER SDE",
    # sam-extra Extra Samplers (core/extra_samplers.py LABEL_*).
    "er_sde_reverse_time": "ER SDE (Reverse-time)", "er_sde_ode": "ER SDE (ODE)",
    "er_sde_tunable": "ER SDE (Tunable)", "dpmpp_4m_sde": "DPM++ 4M SDE",
    "euler_dy_cfg_pp": "Euler Dy CFG++", "euler_smea_dy_cfg_pp": "Euler SMEA Dy CFG++",
    "euler_dy": "Euler Dy", "euler_smea_dy": "Euler SMEA Dy",
    "dpmpp_2m_flow_ode": "DPM++ 2M (flow ODE)", "dpmpp_2m_heun_flow_ode": "DPM++ 2M Heun (flow ODE)",
    "dpmpp_3m_flow_ode": "DPM++ 3M (flow ODE)", "uni_pc_bh2": "UniPC bh2", "cfgpp_ud10_ab": "CFG++ UD10 AB",
    "ipndm": "IPNDM", "ipndm_v_fixed": "IPNDM_V", "deis": "DEIS", "restart_flow": "Restart (flow)",
    # ComfyUI core sampler that Forge Neo's extra-samplers extension lists under this label.
    "exp_heun_2_x0_sde": "EXP Heun 2 x0 SDE",
}
# ComfyUI scheduler name -> Forge label.
SCHEDULER_LABELS = {
    "karras": "Karras", "exponential": "Exponential", "sgm_uniform": "SGM Uniform", "simple": "Simple",
    "normal": "Normal", "ddim_uniform": "DDIM Uniform", "beta": "Beta", "beta57": "Beta57",
    # sam-extra Extra Schedulers (core/extra_schedulers.py SCHEDULER_LABELS).
    "cosine": "Cosine", "cosine_exponential": "CosineExponential blend", "phi": "Phi", "laplace": "Laplace",
    "karras_dynamic": "Karras Dynamic", "custom": "custom", "react_cosinusoidal_dynsf": "React Cosinusoidal DynSF",
    "flow_cosmos_rho7": "Flow Cosmos rho7", "flow_cosmos_dynamic": "Flow Cosmos Dynamic",
}


def forge_sampler_label(name: str) -> str:
    return SAMPLER_LABELS.get(name, name)


def forge_scheduler_label(name: str) -> str:
    return SCHEDULER_LABELS.get(name, name)


def _quote(text: str) -> str:
    """Forge's ``infotext_utils.quote``: a value with a comma, colon or newline is JSON-quoted."""
    if "," not in text and "\n" not in text and ":" not in text:
        return text
    return json.dumps(text, ensure_ascii=False)


def _stem(file_name: str) -> str:
    """``dir\\name.safetensors`` -> ``name`` (Forge writes a model/module without folder or extension)."""
    base = re.split(r"[\\/]", file_name)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


# Forge's latent upscale modes; latent:bislerp and latent:area have no Forge name and stay as they are.
LATENT_LABELS = {
    "latent:bilinear": "Latent", "latent:bilinear antialiased": "Latent (antialiased)", "latent:bicubic": "Latent (bicubic)",
    "latent:bicubic antialiased": "Latent (bicubic antialiased)", "latent:nearest": "Latent (nearest)",
    "latent:nearest-exact": "Latent (nearest-exact)",
}


def _upscaler_label(name: str) -> str:
    # Forge lists pixel upscalers by name without the file extension.
    return LATENT_LABELS.get(name, name) if name.startswith("latent:") else _stem(name)


def _number(value) -> str:
    return f"{value:g}"


# (state key, Forge's name, the value Forge leaves out; None = always written), in the order of sam-extra's ALL_ARGS.
_SAM3_PARAMS = (
    ("mode", "SAM3 Mode", None), ("mask_mode", "SAM3 Mask Mode", None), ("prompt", "SAM3 Prompt", None),
    ("exclude_prompt", "SAM3 Exclude Prompt", ""), ("inpaint_prompt", "SAM3 Inpaint Prompt", ""),
    ("negative_prompt", "SAM3 Negative Prompt", ""), ("threshold", "SAM3 Threshold", None),
    ("mask_dilation", "SAM3 Mask Dilation", 0), ("convex_hull", "SAM3 Mask Hull", False),
    ("mask_outline_px", "SAM3 Mask Outline Px", 0), ("checkpoint", "SAM3 Checkpoint", ""), ("device", "SAM3 Device", None),
    ("mask_blur", "SAM3 Mask Blur", 4), ("denoise", "SAM3 Denoising Strength", 0.4),
    ("fill_mode", "SAM3 Inpainting Fill", "original"), ("only_masked", "SAM3 Inpaint Only Masked", True),
    ("mask_padding", "SAM3 Inpaint Padding", 32),
)
# (state key, the value that means "the main one", Forge's switch, Forge's value), so that Forge's send-to turns them on.
_SAM3_OWN = (
    ("steps", 0, "SAM3 Use Separate Steps", "SAM3 Steps"), ("cfg", 0.0, "SAM3 Use Separate CFG Scale", "SAM3 CFG Scale"),
    ("sampler", "", "SAM3 Use Separate Sampler", "SAM3 Sampler"), ("scheduler", "", "SAM3 Use Separate Scheduler", "SAM3 Scheduler"),
    ("seed", -1, "SAM3 Use Seed", "SAM3 Seed"), ("noise_multiplier", 1.0, "SAM3 Use Noise Multiplier", "SAM3 Noise Multiplier"),
)
_SAM3_CONTROLNET = (
    ("cn_module", "SAM3 CN Module"), ("cn_strength", "SAM3 CN Weight"), ("cn_start", "SAM3 CN Guidance Start"),
    ("cn_end", "SAM3 CN Guidance End"), ("cn_pixel_perfect", "SAM3 CN Pixel Perfect"),
    ("cn_control_mode", "SAM3 CN Control Mode"), ("cn_processor_res", "SAM3 CN Processor Res"),
    ("cn_threshold_a", "SAM3 CN Threshold A"), ("cn_threshold_b", "SAM3 CN Threshold B"),
)


def _file_name(path: str) -> str:
    """No folder in a saved file: the picks are absolute paths of this PC."""
    return re.split(r"[\\/]", path)[-1]


def _text(value) -> str:
    return _number(value) if isinstance(value, float) else str(value)


def _sam3_own_value(sam3: dict, key: str) -> str:
    if key == "sampler":
        return forge_sampler_label(sam3[key])
    return forge_scheduler_label(sam3[key]) if key == "scheduler" else _text(sam3[key])


def _sam3_params(sam3: dict) -> dict[str, str]:
    """The keys of sam-extra's ``Sam3Args.extra_params``, with the switch Forge's send-to reads beside each value."""
    if not sam3.get("on"):
        return {}
    params = {"SAM3 Enable": "True"}
    params.update({name: _text(sam3[key]) for key, name, skip in _SAM3_PARAMS if key in sam3 and sam3[key] != skip})
    if params.get("SAM3 Checkpoint"):
        params["SAM3 Checkpoint"] = _file_name(params["SAM3 Checkpoint"])
    if sam3.get("use_custom_size"):
        params.update({"SAM3 Use Inpaint Width Height": "True", "SAM3 Inpaint Width": str(sam3["custom_width"]),
                       "SAM3 Inpaint Height": str(sam3["custom_height"])})
    for key, main, switch, name in _SAM3_OWN:
        if key in sam3 and sam3[key] != main:
            params.update({switch: "True", name: _sam3_own_value(sam3, key)})
    if sam3.get("overlay"):
        params["SAM3 Preview Overlay"] = "True"
    if sam3.get("cn_enable"):
        params.update({"SAM3 CN Enable": "True", "SAM3 CN Model": _stem(sam3["cn_model"])})
        params.update({name: _text(sam3[key]) for key, name in _SAM3_CONTROLNET})
    if sam3.get("face_crop"):
        params.update({"SAM3 Face Crop": "True", "SAM3 Face Crop Model": _file_name(sam3["face_crop_model"]),
                       "SAM3 Face Crop Confidence": _text(sam3["face_crop_confidence"])})
    return params


def _upscale_params(upscale: dict) -> dict[str, str]:
    """Forge's Extras keys; a saved picture went through them when its state has the upscale on."""
    if not upscale.get("on") or not upscale.get("model"):
        return {}
    if upscale.get("mode") == "Scale to":
        size = f"{upscale['width']}x{upscale['height']}"
        params = {"Postprocess upscale to": size, **({"Postprocess crop to": size} if upscale.get("crop") else {})}
    else:
        params = {"Postprocess upscale by": _number(upscale["scale"])}
        if upscale.get("max_side"):
            params["Max side length"] = str(upscale["max_side"])
    params["Postprocess upscaler"] = _upscaler_label(upscale["model"])
    if upscale.get("model_2") and upscale.get("visibility"):
        params["Postprocess upscaler 2"] = _upscaler_label(upscale["model_2"])
    return params


def _hires_params(state: dict, hires: dict) -> dict[str, str | None]:
    """The first-pass size comes from ``generation_size``, the size boxes win over the scale as in Forge."""
    params: dict[str, str | None] = {}
    if hires.get("resize_width") or hires.get("resize_height"):
        width, height = generation_size(state)
        target = hires_size(width, height, hires["scale"], hires["resize_width"], hires["resize_height"], hires["size_step"], size_grid(state))
        params["Hires resize"] = "x".join(map(str, target))
    else:
        params["Hires upscale"] = _number(hires["scale"])
    if hires.get("cfg"):
        params["Hires CFG Scale"] = _number(hires["cfg"])
    if hires.get("shift"):
        params["Hires Shift"] = _number(hires["shift"])
    if hires.get("sampler") and hires["sampler"] != state["sampler"]:
        params["Hires sampler"] = forge_sampler_label(hires["sampler"])
    if hires.get("scheduler") and hires["scheduler"] != state["scheduler"]:
        params["Hires schedule type"] = forge_scheduler_label(hires["scheduler"])
    if hires.get("steps"):
        params["Hires steps"] = str(hires["steps"])
    params["Hires upscaler"] = _upscaler_label(hires["upscaler"])
    if hires.get("vae"):
        params["Hires Module 1"] = _stem(hires["vae"])
    if hires.get("prompt"):
        params["Hires prompt"] = hires["prompt"]
    if hires.get("negative"):
        params["Hires negative prompt"] = hires["negative"]
    return params


def build_infotext(state: dict, *, seed: int, extra: dict[str, str] | None = None) -> str:
    """``prompt`` / ``Negative prompt: ...`` / ``Steps: ..., Sampler: ...`` for a normalized studio state."""
    prompt = effective_prompt(state)
    negative = state.get("negative") or ""
    hires = state.get("hires") or {}
    i2i = state.get("i2i") or {}
    is_i2i = state.get("tab") == "i2i"
    hires_on = bool(hires.get("on")) and not is_i2i

    sampler = state["sampler"]
    params: dict[str, str | None] = {
        "Steps": str(state["steps"]),
        "Sampler": forge_sampler_label(sampler),
        "Schedule type": forge_scheduler_label(state["scheduler"]),
        "CFG scale": _number(state["cfg"]),
        "Shift": _number(state["shift"]),
        "Seed": str(int(seed)),
        "Size": "x".join(map(str, generation_size(state))),
        "Model": _stem(state["unet"]) if state.get("unet") else None,
        "VAE": _stem(state["vae"]) if state.get("vae") else None,
    }
    if is_i2i:
        params["Denoising strength"] = _number(i2i.get("denoise", 0.0))
    elif hires_on:
        params["Denoising strength"] = _number(hires["denoise"])
    if hires_on:
        params.update(_hires_params(state, hires))
    params.update(_sam3_params(state.get("sam3") or {}))
    params.update(_upscale_params(state.get("upscale") or {}))
    for key, value in (extra or {}).items():
        params[str(key)] = str(value)

    line = ", ".join(f"{key}: {_quote(value)}" for key, value in params.items() if value is not None)
    negative_line = f"\nNegative prompt: {negative}" if negative else ""
    return f"{prompt}{negative_line}\n{line}".strip()
