"""Anima studio nodes (the studio workflow's own helper nodes).

ComfyUI-dependent imports stay inside functions so this module imports in the app's test process.
"""
from __future__ import annotations

import json
import logging

from . import studio_upscale
from .compat import filename_choices, invoke_provider
from .studio_infotext import build_infotext
from .studio_lora_tags import collapse_duplicates, parse_lora_tags, resolve_lora_name
from .studio_prompt_schedule import schedule_prompt
from .studio_state import normalize_state

CATEGORY = "AI Studio/Studio"
LOGGER = logging.getLogger("ai_studio_forge_parity")


def _available_loras() -> list[str]:
    """ComfyUI's LoRA file list.  A folder error is raised, not turned into "no LoRA found" for every tag."""
    from .compat import folder_paths_module

    return list(folder_paths_module().get_filename_list("loras"))


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


def _parse_state(text: str) -> dict:
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"studio_state is not valid JSON: {exc}") from exc
    return normalize_state(raw)


def _uint8_image(image):
    """One ``[H, W, C]`` float image in 0..1 (torch tensor or array) -> PIL image, as ComfyUI's SaveImage does."""
    import numpy as np
    from PIL import Image

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    return Image.fromarray(np.clip(255.0 * array, 0, 255).astype(np.uint8))


class AIStudioSaveForge:
    """Saves PNGs with a Forge-style ``parameters`` chunk next to ComfyUI's ``prompt`` / ``workflow`` chunks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "studio_state": ("STRING", {"multiline": True, "default": "{}"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**63 - 1}),
                "filename_prefix": ("STRING", {"default": "AIStudio/anima"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Saves images as PNG with Forge-style generation info (parameters) plus the ComfyUI workflow."

    def save(self, images, studio_state, seed, filename_prefix, prompt=None, extra_pnginfo=None):
        from PIL import PngImagePlugin

        from .compat import folder_paths_module

        state = _parse_state(studio_state)
        # Forge-like noise drew image i of a batch from seed + i, and Forge writes each image's own seed.
        per_image = state["forge_like"]["master"] and state["forge_like"]["noise"]
        folder_paths = folder_paths_module()
        full_folder, stem, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        results = []
        for batch_index, image in enumerate(images):
            pnginfo = PngImagePlugin.PngInfo()
            image_seed = int(seed) + batch_index if per_image else int(seed)
            pnginfo.add_text("parameters", build_infotext(state, seed=image_seed))
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))
            for key, value in (extra_pnginfo or {}).items():
                pnginfo.add_text(str(key), json.dumps(value))
            file_name = f"{stem.replace('%batch_num%', str(batch_index))}_{counter:05}_.png"
            _uint8_image(image).save(f"{full_folder}/{file_name}", pnginfo=pnginfo, compress_level=4)
            results.append({"filename": file_name, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


class AIStudioUpscaleImage:
    """Forge's Extras upscale (scripts/postprocessing_upscale.py): one or two upscalers, by a factor or to a size."""

    @classmethod
    def INPUT_TYPES(cls):
        files = studio_upscale.model_files()
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": False}),
                "model_name": ([*files, *studio_upscale.PIXEL_RESIZERS],),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
            },
            "optional": {
                "resize_mode": (["Scale by", studio_upscale.SCALE_TO],),
                "max_side_length": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "crop": ("BOOLEAN", {"default": True}),
                "model_name_2": ([studio_upscale.NO_SECOND_MODEL, *files, *studio_upscale.PIXEL_RESIZERS],),
                "upscaler_2_visibility": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_correction": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": studio_upscale.DEFAULT_TILE, "min": 0, "max": 4096, "step": 16}),
                "tile_overlap": ("INT", {"default": studio_upscale.DEFAULT_OVERLAP, "min": 0, "max": 512, "step": 4}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Forge's Extras upscale: by a factor or to a size, a second upscaler, colour correction (off = unchanged)."

    def upscale(self, image, enabled, model_name, scale_by, resize_mode="Scale by", max_side_length=0, width=1024, height=1024,
                crop=True, model_name_2=studio_upscale.NO_SECOND_MODEL, upscaler_2_visibility=0.0, color_correction=False,
                tile_size=studio_upscale.DEFAULT_TILE, tile_overlap=studio_upscale.DEFAULT_OVERLAP):
        if not enabled:
            return (image,)
        if not model_name:
            raise ValueError("Upscale is on but no upscale model is chosen.")
        if color_correction:
            studio_upscale.require_opencv()
        plan = studio_upscale.extras_plan(
            int(image.shape[2]), int(image.shape[1]), resize_mode, float(scale_by), int(max_side_length),
            int(width), int(height), bool(crop),
        )

        def upscaled(name):
            result = studio_upscale.upscale(image, name, plan.scale, int(tile_size), int(tile_overlap))
            return studio_upscale.paste_centered(result, *plan.crop_to) if plan.crop_to else result

        result = upscaled(model_name)
        if model_name_2 not in ("", studio_upscale.NO_SECOND_MODEL) and float(upscaler_2_visibility) > 0:
            result = studio_upscale.blend(result, upscaled(model_name_2), float(upscaler_2_visibility))
        return (studio_upscale.match_colors(result, image) if color_correction else result,)


def _plain_encode(clip, text: str):
    return invoke_provider("CLIPTextEncode", method="encode", feature="Studio prompt", args=(clip, text))[0]


def scheduled_encode(encode_one, text, steps, forge_syntax=True):
    """``encode_one(text)`` as Forge reads the prompt: each ``[a:b:w]`` / ``[a|b]`` segment on its own timestep range."""
    if not forge_syntax:
        return encode_one(text)
    steps = int(steps)
    segments = schedule_prompt(text, steps)
    if len(segments) == 1:
        return encode_one(segments[0][1])
    encoded: dict = {}
    combined = []
    start = 0
    for end, segment in segments:
        if segment not in encoded:
            encoded[segment] = encode_one(segment)
        combined += invoke_provider(
            "ConditioningSetTimestepRange", method="set_range", feature="Studio prompt editing",
            args=(encoded[segment], start / steps, end / steps),
        )[0]
        start = end
    return combined


class AIStudioPromptEncode:
    """CLIPTextEncode with Forge prompt editing: each ``[a:b:w]`` / ``[a|b]`` segment becomes a timestep range."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                # dynamicPrompts off: the frontend must not rewrite {a|b} before Forge's [a|b] rules run.
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
                "steps": ("INT", {"default": 32, "min": 1, "max": 10000}),
                "forge_syntax": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Encodes a prompt; with forge_syntax on, Forge's [a:b:step] / [a|b] prompt editing becomes timestep "
                   "ranges over the given steps (off = the same as CLIP Text Encode).")

    def encode(self, clip, text, steps, forge_syntax=True):
        return (scheduled_encode(lambda segment: _plain_encode(clip, segment), text, steps, forge_syntax),)


def _ui_node(class_name: str, widget: str, description: str) -> type:
    """A node the studio JS draws as a DOM widget: one hidden, socketless STRING (its saved state), no outputs.

    Not an output node and nothing reads it, so ComfyUI never executes or validates it.
    """

    def input_types(cls):
        return {"required": {widget: ("STRING", {"default": "{}", "hidden": True, "socketless": True})}}

    return type(class_name, (), {
        "INPUT_TYPES": classmethod(input_types),
        "RETURN_TYPES": (),
        "FUNCTION": "noop",
        "OUTPUT_NODE": False,
        "CATEGORY": CATEGORY,
        "DESCRIPTION": description,
        "noop": lambda self, **_: (),
    })


AIStudioPanel = _ui_node("AIStudioPanel", "state", "Anima studio control panel (drawn by the studio JS; never runs).")
AIStudioResultView = _ui_node("AIStudioResultView", "view", "Anima studio result / gallery view (drawn by the studio JS).")
AIStudioXYZView = _ui_node("AIStudioXYZView", "xyz", "Anima studio X/Y/Z plot panel (drawn by the studio JS).")
AIStudioExifView = _ui_node("AIStudioExifView", "exif", "Anima studio EXIF panel (drawn by the studio JS; never runs).")


class AIStudioSwitch:
    """Passes ``a`` (select 0) or ``b`` (select 1); only the selected input is evaluated (lazy), so the other model
    family's loaders never run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"select": ("INT", {"default": 0, "min": 0, "max": 1})},
            "optional": {"a": ("*", {"lazy": True}), "b": ("*", {"lazy": True})},
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "switch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Outputs input a (select 0) or b (select 1); the other input is not evaluated."

    def check_lazy_status(self, select, a=None, b=None):
        return ["b"] if int(select) == 1 else ["a"]

    def switch(self, select, a=None, b=None):
        return (b if int(select) == 1 else a,)


NODE_CLASS_MAPPINGS = {
    "AIStudioLoraStack": AIStudioLoraStack,
    "AIStudioSaveForge": AIStudioSaveForge,
    "AIStudioUpscaleImage": AIStudioUpscaleImage,
    "AIStudioPromptEncode": AIStudioPromptEncode,
    "AIStudioPanel": AIStudioPanel,
    "AIStudioResultView": AIStudioResultView,
    "AIStudioXYZView": AIStudioXYZView,
    "AIStudioExifView": AIStudioExifView,
    "AIStudioSwitch": AIStudioSwitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioLoraStack": "AI Studio LoRA Stack",
    "AIStudioSaveForge": "AI Studio Save (Forge info)",
    "AIStudioUpscaleImage": "AI Studio Upscale Image",
    "AIStudioPromptEncode": "AI Studio Prompt Encode (Forge syntax)",
    "AIStudioPanel": "AI Studio Panel",
    "AIStudioResultView": "AI Studio Result View",
    "AIStudioXYZView": "AI Studio XYZ View",
    "AIStudioExifView": "AI Studio EXIF View",
    "AIStudioSwitch": "AI Studio Switch",
}

__all__ = [
    "AIStudioExifView", "AIStudioLoraStack", "AIStudioPanel", "AIStudioPromptEncode", "AIStudioResultView", "AIStudioSaveForge",
    "AIStudioSwitch", "AIStudioUpscaleImage", "AIStudioXYZView",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
]
