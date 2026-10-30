"""Control-panel state schema for the Anima studio: defaults, normalization, effective prompt.

Pure Python (no ComfyUI / torch).  The studio's JS mirrors this shape.
"""
from __future__ import annotations

import math
import re

DORA_MODES = (
    "LyCORIS fp32 (as trained)",
    "Forge/Comfy fp32",
    "Forge/Comfy (stock)",
    "DoRA off (no magnitude)",
)

DEFAULT_STATE: dict = {
    "version": 1,
    "tab": "t2i",
    "i2i_mode": "img2img",
    "prompt": "",
    "negative": "",
    "unet": "",
    "te": "",
    "vae": "",
    "sampler": "exp_heun_2_x0_sde",
    "scheduler": "flow_cosmos_dynamic",
    "steps": 32,
    "cfg": 3.0,
    "shift": 3.0,
    "seed": -1,
    "width": 832,
    "height": 1216,
    "batch": 1,
    "loras": [],
    "negpip": True,
    "dora_mode": DORA_MODES[0],
    "forge_like": {"master": True, "noise": True, "prompt_syntax": True, "dora": True, "attention": False},
    "hires": {"on": False, "upscaler": "4x-UltraSharp.pth", "sampler": "", "scale": 1.5, "steps": 16, "denoise": 0.4},
    "upscale": {"on": False, "model": "", "scale": 2.0},
    "sam3": {"on": False, "checkpoint": "", "prompt": "face", "denoise": 0.4, "steps": 0, "cfg": 0.0},
    # inpaint_area: no node input takes it yet (ForgeNeoLatentInput inpaints the whole picture); the panel shows
    # "전체" greyed out (studio_forge_parity.md), the key keeps the saved shape.
    "i2i": {"image": "", "denoise": 0.55, "mask_blur": 4, "inpaint_area": "whole"},
}

_ENUMS = {
    "tab": ("t2i", "i2i"),
    "i2i_mode": ("img2img", "inpaint"),
    "dora_mode": DORA_MODES,
    "inpaint_area": ("masked", "whole"),
}
_LORA_TAG = re.compile(r"<lora:([^:>]*)[:>]")


def _int(value, default, lo, hi, step=1):
    if isinstance(value, bool):
        return default
    try:
        number = int(float(value)) if isinstance(value, str) else int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    number = max(lo, min(hi, number))
    return max(lo, min(hi, (number // step) * step)) if step > 1 else number


def _float(value, default, lo, hi):
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(lo, min(hi, number))


def _str(value, default, allow_empty=True):
    if not isinstance(value, str) or (not allow_empty and not value):
        return default
    return value


def _bool(value, default):
    return value if isinstance(value, bool) else default


def _enum(value, default, key):
    return value if value in _ENUMS[key] else default


def _section(raw, default: dict, fields: dict) -> dict:
    """Normalize one nested dict; ``fields`` maps key -> callable(value, default)."""
    raw = raw if isinstance(raw, dict) else {}
    return {key: fn(raw.get(key), default[key]) for key, fn in fields.items()}


def _lora(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    return {
        "on": _bool(item.get("on"), True),
        "name": _str(item.get("name"), ""),
        "strength": _float(item.get("strength"), 1.0, -10.0, 10.0),
    }


def normalize_state(raw: object) -> dict:
    """Return a complete, clamped copy of the state; unknown keys are dropped."""
    raw = raw if isinstance(raw, dict) else {}
    d = DEFAULT_STATE
    loras = raw.get("loras")
    out = {
        "version": 1,
        "tab": _enum(raw.get("tab"), d["tab"], "tab"),
        "i2i_mode": _enum(raw.get("i2i_mode"), d["i2i_mode"], "i2i_mode"),
        "prompt": _str(raw.get("prompt"), d["prompt"]),
        "negative": _str(raw.get("negative"), d["negative"]),
        "unet": _str(raw.get("unet"), d["unet"]),
        "te": _str(raw.get("te"), d["te"]),
        "vae": _str(raw.get("vae"), d["vae"]),
        "sampler": _str(raw.get("sampler"), d["sampler"], allow_empty=False),
        "scheduler": _str(raw.get("scheduler"), d["scheduler"], allow_empty=False),
        "steps": _int(raw.get("steps"), d["steps"], 1, 150),
        "cfg": _float(raw.get("cfg"), d["cfg"], 0.0, 30.0),
        "shift": _float(raw.get("shift"), d["shift"], 0.0, 100.0),
        "seed": max(-1, _int(raw.get("seed"), d["seed"], -1, 2**63 - 1)),
        "width": _int(raw.get("width"), d["width"], 64, 4096, 8),
        "height": _int(raw.get("height"), d["height"], 64, 4096, 8),
        "batch": _int(raw.get("batch"), d["batch"], 1, 16),
        "loras": [x for x in map(_lora, loras) if x] if isinstance(loras, list) else [],
        "negpip": _bool(raw.get("negpip"), d["negpip"]),
        "dora_mode": _enum(raw.get("dora_mode"), d["dora_mode"], "dora_mode"),
        "forge_like": _section(raw.get("forge_like"), d["forge_like"],
                               {k: _bool for k in d["forge_like"]}),
        "hires": _section(raw.get("hires"), d["hires"], {
            "on": _bool,
            "upscaler": lambda v, df: _str(v, df, allow_empty=False),
            "sampler": _str,
            "scale": lambda v, df: _float(v, df, 1.0, 8.0),
            "steps": lambda v, df: _int(v, df, 1, 150),
            "denoise": lambda v, df: _float(v, df, 0.0, 1.0),
        }),
        "upscale": _section(raw.get("upscale"), d["upscale"], {
            "on": _bool,
            "model": _str,
            "scale": lambda v, df: _float(v, df, 1.0, 8.0),
        }),
        "sam3": _section(raw.get("sam3"), d["sam3"], {
            "on": _bool,
            "checkpoint": _str,
            "prompt": lambda v, df: _str(v, df, allow_empty=False),
            "denoise": lambda v, df: _float(v, df, 0.0, 1.0),
            "steps": lambda v, df: _int(v, df, 0, 150),
            "cfg": lambda v, df: _float(v, df, 0.0, 30.0),
        }),
        "i2i": _section(raw.get("i2i"), d["i2i"], {
            "image": _str,
            "denoise": lambda v, df: _float(v, df, 0.0, 1.0),
            "mask_blur": lambda v, df: _int(v, df, 0, 64),
            "inpaint_area": lambda v, df: _enum(v, df, "inpaint_area"),
        }),
    }
    return out


def effective_prompt(state: dict) -> str:
    """Prompt plus ``<lora:name:strength>`` tags for enabled list LoRAs not already tagged in the prompt."""
    prompt = state.get("prompt") or ""
    tagged = set(_LORA_TAG.findall(prompt))
    tags = [
        f"<lora:{item['name']}:{item['strength']:g}>"
        for item in state.get("loras") or []
        if item.get("on") and item.get("name") and item["name"] not in tagged
    ]
    return " ".join(([prompt] if prompt else []) + tags)
