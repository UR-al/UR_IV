"""MOSS-Image LoRA files onto the loaded MOSS-Image MODEL.

ComfyUI maps a LoRA with split q / k / v keys onto a fused model, not the reverse, and only knows some key prefixes, so a
LoRA from another trainer can patch nothing without any error. ComfyUI-MOSS-Toolkit's ``lora_compat`` converts those files;
it stays a separate install. Without it the file goes in unchanged, and a LoRA that patches nothing stops the run here.
"""
from __future__ import annotations

import importlib
from typing import Any

from .compat import provider_class


def is_moss_model(model: Any) -> bool:
    return type(getattr(model, "model", None)).__name__ == "MOSSImage"


def _toolkit_layout():
    try:
        owner = provider_class("MOSSStreamFP32").__module__
        return importlib.import_module(owner.rsplit(".", 1)[0] + ".lora_compat")
    except Exception:
        return None


def prepare_moss_lora(model: Any, clip: Any, state_dict: dict, name: str) -> dict:
    """``state_dict`` in this model's layout; raises when none of it patches this model."""
    layout = _toolkit_layout()
    if layout is not None:
        state_dict = layout.to_model_layout(state_dict, layout.model_is_fused(model))
    lora = importlib.import_module("comfy.lora")
    convert = importlib.import_module("comfy.lora_convert")
    key_map = lora.model_lora_keys_unet(model.model, {})
    if clip is not None:
        key_map = lora.model_lora_keys_clip(clip.cond_stage_model, key_map)  # a text-encoder-only file still patches
    if state_dict and not lora.load_lora(convert.convert_lora(dict(state_dict)), key_map, log_missing=False):
        raise ValueError(f"MOSS LoRA {name}: 이 모델에 맞는 키가 하나도 없음 (다른 모델용 LoRA 이거나 형식이 다름)")
    return state_dict
