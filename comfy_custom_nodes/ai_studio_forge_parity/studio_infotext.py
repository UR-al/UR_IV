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

from .studio_state import effective_prompt

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


def _upscaler_label(name: str) -> str:
    # Forge lists pixel upscalers by name without the file extension; ``latent:<mode>`` entries stay as they are.
    return name if name.startswith("latent:") else _stem(name)


def _number(value) -> str:
    return f"{value:g}"


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
        "Size": f"{state['width']}x{state['height']}",
        "Model": _stem(state["unet"]) if state.get("unet") else None,
        "VAE": _stem(state["vae"]) if state.get("vae") else None,
    }
    if is_i2i:
        params["Denoising strength"] = _number(i2i.get("denoise", 0.0))
    elif hires_on:
        params["Denoising strength"] = _number(hires["denoise"])
    if hires_on:
        params["Hires upscale"] = _number(hires["scale"])
        if hires.get("sampler") and hires["sampler"] != sampler:
            params["Hires sampler"] = forge_sampler_label(hires["sampler"])
        params["Hires steps"] = str(hires["steps"])
        params["Hires upscaler"] = _upscaler_label(hires["upscaler"])
    for key, value in (extra or {}).items():
        params[str(key)] = str(value)

    line = ", ".join(f"{key}: {_quote(value)}" for key, value in params.items() if value is not None)
    negative_line = f"\nNegative prompt: {negative}" if negative else ""
    return f"{prompt}{negative_line}\n{line}".strip()
