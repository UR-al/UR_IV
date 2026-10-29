"""Anima studio nodes (the studio workflow's own helper nodes).

ComfyUI-dependent imports stay inside functions so this module imports in the app's test process.
"""
from __future__ import annotations

import logging

from .studio_lora_tags import collapse_duplicates, parse_lora_tags, resolve_lora_name

CATEGORY = "AI Studio/Studio"
LOGGER = logging.getLogger("ai_studio_forge_parity")


def _available_loras() -> list[str]:
    from .compat import folder_paths_module

    try:
        return list(folder_paths_module().get_filename_list("loras"))
    except Exception:
        return []


class AIStudioLoraStack:
    """Strips Forge ``<lora:...>`` tags from both prompts and applies them to MODEL and CLIP in order."""

    def __init__(self) -> None:
        self._loader = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Applies <lora:name:strength> tags from the prompts (Forge rules) and returns the cleaned prompts."

    def _get_loader(self):
        # One loader per node object: its state cache survives between runs of the same queue.
        if self._loader is None:
            from .anima_lora_nodes import ForgeNeoAnimaLoraLoader

            self._loader = ForgeNeoAnimaLoraLoader()
        return self._loader

    def apply(self, model, clip, positive, negative):
        tags, (positive_text, negative_text) = parse_lora_tags(positive, negative)
        if tags:
            available = _available_loras()
            for tag in collapse_duplicates(tags):
                file_name = resolve_lora_name(tag.name, available)
                if file_name is None:
                    LOGGER.warning("LoRA not found, skipped: %s", tag.name)
                    continue
                model, clip = self._get_loader().load_lora(
                    model, clip, file_name, tag.strength_model, tag.strength_clip
                )
        return (model, clip, positive_text, negative_text)


NODE_CLASS_MAPPINGS = {
    "AIStudioLoraStack": AIStudioLoraStack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioLoraStack": "AI Studio LoRA Stack",
}

__all__ = ["AIStudioLoraStack", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
