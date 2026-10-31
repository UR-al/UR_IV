"""Studio nodes over optional third-party packs: MOSS-Image's fp32 stream, Anima's two references, Krea2's reference.

The packs stay separate installs with their own licences and updates, and their weights are never bundled. A node here
only dispatches: a missing pack stops the run that needed it with ``compat.MissingComfyProvider``'s message, and the
panel greys the switch (``web/ai_studio/reference_view.js``). Off, each of these is bypassed in the graph (R7).
"""
from __future__ import annotations

from .compat import invoke_provider, require_torch
from .studio_state import KREA2_POSITIONS, REFERENCE_CORRECTIONS, REFERENCE_METHODS, REFERENCE_MODES

CATEGORY = "AI Studio/모델"
KREA2_LATENT_CHANNELS = 16
KREA2_LATENT_SCALE = 8
KREA2_NEGATIVE_GROUNDING = 768  # both of the packs' example workflows ground the empty negative at 768


class AIStudioMossStream:
    """MOSS-Image's fp32 residual stream (ComfyUI-MOSS-Toolkit ``MOSSStreamFP32``), for a core-loaded MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Runs MOSS-Image's stream in fp32 around its block GEMMs (ComfyUI-MOSS-Toolkit)."

    def patch(self, model):
        return invoke_provider("MOSSStreamFP32", method="patch", feature="MOSS fp32 stream", args=(model,))[:1]


class AIStudioAnimaReference:
    """One Anima reference: SnapRef or ElectronGirl's Anima Ref Adapter, picked by ``method``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "method": (list(REFERENCE_METHODS),),
            "file": ("STRING", {"default": "", "tooltip": "the adapter's file in the LoRA folders"}),
            "mode": (list(REFERENCE_MODES), {"tooltip": "SnapRef: cs = character and style, char = character only"}),
            "strength": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05,
                                   "tooltip": "SnapRef: log scale, 0 = as trained. Ref Adapter: LoRA strength"}),
            "guidance_on": ("BOOLEAN", {"default": True, "tooltip": "SnapRef only; Ref Adapter always uses guidance"}),
            "guidance": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "correction": (list(REFERENCE_CORRECTIONS),),
            "ref_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.0, "step": 0.05}),
            "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "SnapRef: the reference's appearance tags"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Character / style reference for Anima through SnapRef or Anima Ref Adapter."

    def apply(self, model, clip, vae, image, method, file, mode, strength, guidance_on, guidance, correction, ref_end,
              megapixels, caption):
        if not file:
            raise ValueError("레퍼런스 파일이 비어 있음")
        if method == "snapref":
            return invoke_provider("SnapRef", method="apply", feature="SnapRef", kwargs={
                "model": model, "clip": clip, "vae": vae, "image": image, "adapter": file, "mode": mode,
                "ref_in_uncond": False, "ref_megapixels": megapixels, "strength": strength, "ref_caption": caption,
                "ref_guidance_on": guidance_on, "ref_guidance": guidance, "correction": correction, "ref_end": ref_end,
            })[:1]
        return invoke_provider("AnimaRefAdapter", method="apply", feature="Anima Ref Adapter", kwargs={
            "model": model, "vae": vae, "image": image, "lora_name": file, "ref_guidance": guidance,
            "correction": correction, "strength": strength, "ref_end": ref_end, "ref_megapixels": megapixels,
        })[:1]


class AIStudioKrea2Reference:
    """Krea2 reference: an edit LoRA (identity, general, QuadView, ...), then comfyui-krea2edit's patch and prompts.

    Sockets start ``model, positive, negative`` so a bypassed node hands each output its own input. Running, it writes
    both prompts itself, so those two inputs stay lazy and are never evaluated.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING", {"lazy": True}),
            "negative": ("CONDITIONING", {"lazy": True}),
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "image": ("IMAGE",),
            "text": ("STRING", {"forceInput": True}),
            "negative_text": ("STRING", {"forceInput": True}),
            "file": ("STRING", {"default": "", "tooltip": "the edit LoRA's file in the LoRA folders"}),
            "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 64,
                                     "tooltip": "the reference's long side in the text encoder"}),
            "boost": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "position": (list(KREA2_POSITIONS),),
            "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
            "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Reference image for Krea2 through comfyui-krea2edit and an edit LoRA."

    def check_lazy_status(self, **_inputs):
        return []

    def apply(self, model, clip, vae, image, text, negative_text, file, strength, grounding_px, boost, position, width,
              height, positive=None, negative=None):
        if not file:
            raise ValueError("레퍼런스 파일이 비어 있음")
        torch = require_torch()
        feature = "Krea2 reference"
        patched = invoke_provider(
            "LoraLoaderModelOnly", method="load_lora_model_only", feature=feature, args=(model, file, strength))[0]
        # Only the shape counts: with vae + source_image the patch encodes the image itself at this size, before
        # sampling, and never reads source_latent.
        target = {"samples": torch.zeros(
            (1, KREA2_LATENT_CHANNELS, int(height) // KREA2_LATENT_SCALE, int(width) // KREA2_LATENT_SCALE))}
        patched = invoke_provider("Krea2EditModelPatch", method="patch", feature=feature, kwargs={
            "model": patched, "source_latent": target, "vae": vae, "source_image": image, "fit_mode": "fit",
            "target_latent": target, "ref_boost": boost, "ref_position": position,
        })[0]

        def encode(prompt, pixels):
            return invoke_provider("Krea2EditGroundedEncode", method="encode", feature=feature, kwargs={
                "clip": clip, "prompt": prompt, "image": image, "grounding_px": pixels})[0]

        return patched, encode(text, grounding_px), encode(negative_text, KREA2_NEGATIVE_GROUNDING)


NODE_CLASS_MAPPINGS = {
    "AIStudioMossStream": AIStudioMossStream,
    "AIStudioAnimaReference": AIStudioAnimaReference,
    "AIStudioKrea2Reference": AIStudioKrea2Reference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioMossStream": "AI Studio MOSS fp32 Stream",
    "AIStudioAnimaReference": "AI Studio Anima Reference",
    "AIStudioKrea2Reference": "AI Studio Krea2 Reference",
}
