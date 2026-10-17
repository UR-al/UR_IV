"""DoRA inference mode for ComfyUI (pack 1.9.0): ``ForgeNeoDoraInferenceMode`` (MODEL -> MODEL).

The sam-extra Forge extension's "DoRA 추론 방식" (DoRA Inference Mode) as a node: the DoRA patches already on the
MODEL (LoRA/LoCon, LoHa, LoKr, GLoRA files that carry ``dora_scale``) merge in the chosen mode — LyCORIS' training
formula in fp32 (default), ComfyUI's formula in fp32, ComfyUI's own merge, or without the magnitude. Put it after the
LoRA loaders: LoRAs loaded after it, the CLIP's patches and other MODELs merge the stock way. The work is in
``dora_infer_mode``; no torch or ComfyUI import at module level.
"""

from __future__ import annotations

from typing import Any

from .dora_infer_mode import DEFAULT_MODE, MODE_CHOICES, MODE_LABELS, apply_dora_mode, mode_key

CATEGORY = "AI Studio/Forge Neo parity/Loaders"
NODE_ID = "ForgeNeoDoraInferenceMode"
DISPLAY_NAME = "Forge Neo DoRA Inference Mode (sam-extra)"


class ForgeNeoDoraInferenceMode:
    """MODEL -> MODEL: how the MODEL's DoRA patches merge."""

    CATEGORY = CATEGORY
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "sam-extra's DoRA Inference Mode for the DoRA patches already on the MODEL (LoRA, LoHa, LoKr, GLoRA files "
        "with dora_scale): LyCORIS fp32 = the LyCORIS training merge, computed in fp32 from the W0 ComfyUI hands the "
        "patch (its fp16 copy on the GPU full-load path: exact for fp16/fp8 weights and practically for bf16, not for "
        "fp32-stored weights); ComfyUI divides the output axis by ||W0|| and merges in fp16/bf16. Use it after the "
        "LoRA loaders; LoRAs added after it, the CLIP and models without DoRA patches are unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The MODEL after its LoRA loaders."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = the MODEL is returned unchanged."}),
                "mode": (list(MODE_CHOICES), {
                    "default": MODE_LABELS[DEFAULT_MODE],
                    "tooltip": (
                        "LyCORIS fp32: (W0+dW)*m/||W0+dW|| in fp32 with LyCORIS' operations, as trained. "
                        "Forge/Comfy fp32: ComfyUI's formula "
                        "(||W0|| on the output axis) in fp32. Forge/Comfy (stock): ComfyUI's own merge. DoRA off: "
                        "dora_scale ignored (W0 + strength*dW). Strength blends W0 + s*(W_dora - W0)."
                    ),
                }),
            },
        }

    def patch(self, model: Any, enabled: bool = True, mode: str = MODE_LABELS[DEFAULT_MODE]):
        if not enabled:
            return (model,)
        return (apply_dora_mode(model, mode_key(mode)),)


NODE_CLASS_MAPPINGS = {NODE_ID: ForgeNeoDoraInferenceMode}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: DISPLAY_NAME}

__all__ = ["CATEGORY", "DISPLAY_NAME", "ForgeNeoDoraInferenceMode", "NODE_CLASS_MAPPINGS",
           "NODE_DISPLAY_NAME_MAPPINGS", "NODE_ID"]
