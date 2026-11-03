"""Reads ``parameters`` text back into part of a studio state.

What the EXIF panel's 설정 가져오기 takes from an image without a studio panel in it (a Forge image): the reverse of
``studio_infotext.build_infotext``, with Forge's own keys beside it.  File names stay stems here, the panel matches them
to its lists (web/ai_studio/studio_settings.js).  The seed is no setting, and what the text does not name stays as the
panel has it.  Pure Python (no ComfyUI / torch).
"""
from __future__ import annotations

import json
import re

from .studio_infotext import LATENT_LABELS, SAMPLER_LABELS, SCHEDULER_LABELS
from .studio_state import ATTENTION_MODES, INPAINT_FILLS, PROMPT_BOXES

_PARAM = re.compile(r'\s*([^,:"]+?):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)')
_SIZE = re.compile(r"(\d+)x(\d+)")
_ORIGIN = re.compile(r"\s*\(.*\)$")  # Forge's "Beta57 (RES4LYF)": where a scheduler comes from
_SAMPLERS = {label: name for name, label in SAMPLER_LABELS.items()}
_SCHEDULERS = {label: name for name, label in SCHEDULER_LABELS.items()}
_LATENTS = {label: name for name, label in LATENT_LABELS.items()}
_FILE_KEYS = ("VAE", "Module 1", "Module 2", "Module 3")
_NEGATIVE = "Negative prompt:"
# What a hires pass the text does not mention falls back to: the panel's old value would change the picture.
_HIRES_BLANK = {"cfg": 0, "shift": 0, "sampler": "", "scheduler": "", "vae": "", "prompt": "", "negative": "", "resize_width": 0, "resize_height": 0}


def _params(line: str) -> dict[str, str]:
    params = {}
    for key, value in _PARAM.findall(line):
        value = value.strip()
        if value.startswith('"'):
            try:
                value = json.loads(value)
            except ValueError:
                pass  # keep the raw text
        params[key.strip()] = value
    return params


def _number(kind, text):
    try:
        return kind(text)
    except (TypeError, ValueError):
        return None


def _put(out: dict, key: str, value) -> None:
    if value is not None:
        out[key] = value


def _size(text) -> tuple[int, int] | None:
    match = _SIZE.fullmatch(text or "")
    return (int(match[1]), int(match[2])) if match else None


def _scheduler(label: str) -> str:
    return _SCHEDULERS.get(label) or _SCHEDULERS.get(_ORIGIN.sub("", label), label)


def _prompts(lines: list[str]) -> tuple[str, str]:
    at = next((index for index, line in enumerate(lines) if line.startswith(_NEGATIVE)), len(lines))
    negative = [lines[at][len(_NEGATIVE):], *lines[at + 1:]] if at < len(lines) else []
    return "\n".join(lines[:at]).strip(), "\n".join(negative).strip()


def _hires(params: dict) -> dict:
    out = {"on": True, **_HIRES_BLANK}
    size = _size(params.get("Hires resize"))
    if size:
        out["resize_width"], out["resize_height"] = size
    else:
        _put(out, "scale", _number(float, params.get("Hires upscale")))
    _put(out, "denoise", _number(float, params.get("Denoising strength")))
    _put(out, "cfg", _number(float, params.get("Hires CFG Scale")))
    _put(out, "shift", _number(float, params.get("Hires Shift")))
    _put(out, "steps", _number(int, params.get("Hires steps")))
    if "Hires sampler" in params:
        out["sampler"] = _SAMPLERS.get(params["Hires sampler"], params["Hires sampler"])
    if "Hires schedule type" in params:
        out["scheduler"] = _scheduler(params["Hires schedule type"])
    if "Hires upscaler" in params:
        out["upscaler"] = _LATENTS.get(params["Hires upscaler"], params["Hires upscaler"])
    out["vae"] = params.get("Hires Module 1", "")
    out["prompt"] = params.get("Hires prompt", "")
    out["negative"] = params.get("Hires negative prompt", "")
    return out


def _i2i(params: dict) -> dict:
    out: dict = {}
    _put(out, "denoise", _number(float, params.get("Denoising strength")))
    _put(out, "mask_blur", _number(int, params.get("Mask blur")))
    if "Mask blur" in params:  # the keys of an inpaint: Forge leaves Inpaint area out for the whole picture
        out["inpaint_area"] = "masked" if params.get("Inpaint area") == "Only masked" else "whole"
    _put(out, "inpaint_padding", _number(int, params.get("Masked area padding")))
    if params.get("Masked content") in INPAINT_FILLS:
        out["inpaint_fill"] = params["Masked content"]
    return out


def parse_infotext(text) -> dict | None:
    """``{"settings": part of a state, "model": stem or "", "modules": [stems]}``; None when ``text`` has no ``Steps:`` line."""
    if not isinstance(text, str):
        return None
    lines = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    at = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].startswith("Steps: ")), None)
    if at is None:
        return None
    params = _params(lines[at])
    steps = _number(int, params.get("Steps"))
    if steps is None:
        return None

    prompt, negative = _prompts(lines[:at])
    # The prompt holds the LoRAs as tags and every box in one text.
    settings: dict = {**dict.fromkeys(PROMPT_BOXES, ""), "loras": [], "steps": steps}
    settings["prompt"] = params.get("Template") or prompt
    settings["negative"] = params.get("Negative Template") or negative
    if "Sampler" in params:
        settings["sampler"] = _SAMPLERS.get(params["Sampler"], params["Sampler"])
    if "Schedule type" in params:
        settings["scheduler"] = _scheduler(params["Schedule type"])
    _put(settings, "cfg", _number(float, params.get("CFG scale")))
    _put(settings, "shift", _number(float, params.get("Shift")))
    if params.get("Attention") in ATTENTION_MODES:
        settings["attention"] = params["Attention"]

    hires = any(key.startswith("Hires ") for key in params)
    settings["hires"] = _hires(params) if hires else {"on": False}
    i2i = {} if hires else _i2i(params)
    if i2i:
        settings["i2i"] = i2i
    size = _size(params.get("Size"))
    if size:
        (i2i or settings).update(width=size[0], height=size[1])
    return {"settings": settings, "model": params.get("Model", ""), "modules": [params[key] for key in _FILE_KEYS if key in params]}
