"""Studio nodes over optional third-party packs: MOSS-Image's fp32 stream, Anima's two references, Krea2's reference.

The packs stay separate installs with their own licences and updates, and their weights are never bundled. A node here
only dispatches: a missing pack stops the run that needed it with ``compat.MissingComfyProvider``'s message, and the
panel greys the switch (``web/ai_studio/reference_view.js``). Off, each of these is bypassed in the graph (R7).
"""
from __future__ import annotations

import importlib
import json

from .compat import filename_choices, folder_paths_module, invoke_provider, require_torch
from .guidance_common import ASYM_REFERENCE_KEY
from .reference_layout import adapter_for_model
from .studio_nodes import scheduled_encode
from .studio_state import ATTENTION_MODES, KREA2_POSITIONS, REFERENCE_CORRECTIONS, REFERENCE_METHODS, REFERENCE_MODES

CATEGORY = "AI Studio/모델"
TE_TYPES = ("stable_diffusion", "moss_image", "krea2")
SPLIT_WRAPPER_KEY = "ai_studio_snapref_split"
DIFFUSION_MODEL_WRAPPERS = "diffusion_model"  # comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
KREA2_LATENT_CHANNELS = 16
KREA2_LATENT_SCALE = 8
KREA2_NEGATIVE_GROUNDING = 768  # both of the packs' example workflows ground the empty negative at 768


def _read_moss_te(path):
    """The Qwen3.5 weights under the prefixes core's detection reads; the moss-image node's single file has bare ones."""
    utils = importlib.import_module("comfy.utils")
    state, metadata = utils.load_torch_file(path, safe_load=True, return_metadata=True)
    state, _ = utils.convert_old_quants(state, model_prefix="", metadata=metadata)
    return utils.state_dict_prefix_replace(state, {"language_model.": "model.language_model.", "visual.": "model.visual."})


def load_moss_te(path, disable_dynamic=False):
    sd = importlib.import_module("comfy.sd")
    clip = sd.load_text_encoder_state_dicts(
        [_read_moss_te(path)], embedding_directory=folder_paths_module().get_folder_paths("embeddings"),
        clip_type=sd.CLIPType.MOSS_IMAGE, disable_dynamic=disable_dynamic)
    clip.patcher.cached_patcher_init = (load_moss_te_patcher, (path,))  # how dynamic VRAM rebuilds it after eviction
    return clip


def load_moss_te_patcher(path, disable_dynamic=False):
    return load_moss_te(path, disable_dynamic).patcher


class AIStudioTextEncoderLoader:
    """``CLIPLoader`` of the studio's families, which also reads a MOSS text encoder saved with bare key prefixes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip_name": (filename_choices("text_encoders"),), "type": (list(TE_TYPES),)}}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads a text encoder for Anima, MOSS-Image or Krea2."

    def load(self, clip_name, type="stable_diffusion"):
        if type != "moss_image":
            return invoke_provider("CLIPLoader", method="load_clip", feature="Text encoder", args=(clip_name, type))[:1]
        return (load_moss_te(folder_paths_module().get_full_path_or_raise("text_encoders", clip_name)),)


class AIStudioAttention:
    """The attention function of this MODEL, for every pass after it; ``launch`` keeps what ComfyUI was started with."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mode": (list(ATTENTION_MODES),)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = "Picks the attention backend without restarting ComfyUI (launch = ComfyUI's own)."

    def patch(self, model, mode):
        if mode == "launch":
            return (model,)
        function = importlib.import_module("comfy.ldm.modules.attention").get_attention_function(mode, None)
        if function is None:
            raise ValueError(f"Attention backend {mode} is not available in this ComfyUI.")
        patched = model.clone()
        patched.set_model_optimized_attention(function)
        return (patched,)


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


def _split_mixed_batch(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    """One call per cond / uncond chunk, kwargs included. SnapRef splits a mixed batch itself but leaves per-batch kwargs
    whole, so NegPiP's mask for both rows would meet one row of context."""
    torch = require_torch()
    options = kwargs.get("transformer_options") or {}
    flags = options.get("cond_or_uncond") or [0]
    if len(set(flags)) < 2 or x.shape[0] % len(flags):
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
    rows = x.shape[0] // len(flags)
    outs = []
    for index, flag in enumerate(flags):
        part = slice(index * rows, (index + 1) * rows)
        chunk = {key: value[part] if torch.is_tensor(value) and value.shape[:1] == x.shape[:1] else value
                 for key, value in kwargs.items()}
        chunk["transformer_options"] = {**options, "cond_or_uncond": [flag]}
        if len(options.get("uuids") or ()) == len(flags):
            chunk["transformer_options"]["uuids"] = [options["uuids"][index]]
        outs.append(executor(x[part], timesteps[part], context[part], fps, padding_mask, **chunk))
    return torch.cat(outs, 0)


def _outermost_wrapper(model, key, wrapper):
    """A clone with ``wrapper`` around every other diffusion-model wrapper (the first key runs first)."""
    patched = model.clone()
    patched.add_wrapper_with_key(DIFFUSION_MODEL_WRAPPERS, key, wrapper)
    wrappers = patched.wrappers[DIFFUSION_MODEL_WRAPPERS]
    patched.wrappers[DIFFUSION_MODEL_WRAPPERS] = {key: wrappers.pop(key), **wrappers}
    return patched


def _adapter_is_asym(path) -> bool:
    """Whether a Ref Adapter file runs the blocks itself (``arch`` in its metadata, as the pack reads it)."""
    try:
        safetensors = importlib.import_module("safetensors")
        with safetensors.safe_open(path, "pt") as handle:
            return json.loads((handle.metadata() or {}).get("anima_ref_adapter", "{}")).get("arch") == "asym"
    except Exception:
        return False


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
        }, "optional": {
            "image_2": ("IMAGE", {"tooltip": "SnapRef: another picture of the same character"}),
            "weight_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "SnapRef: how much image counts"}),
            "weight_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "SnapRef: how much image_2 counts"}),
            "fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "SnapRef: 1 keeps the picture's details, lower follows the prompt; off with guidance_on"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Character / style reference for Anima through SnapRef or Anima Ref Adapter."

    def apply(self, model, clip, vae, image, method, file, mode, strength, guidance_on, guidance, correction, ref_end,
              megapixels, caption, image_2=None, weight_1=1.0, weight_2=1.0, fidelity=1.0):
        if not file:
            raise ValueError("레퍼런스 파일이 비어 있음")
        file = adapter_for_model(model, file)
        if method == "snapref":
            # defaults stay out of the call, so an older SnapRef node keeps working
            newer = {} if image_2 is None else {"image_2": image_2, "weight_1": weight_1, "weight_2": weight_2}
            if fidelity < 1.0:
                newer["fidelity"] = fidelity
            try:
                patched = invoke_provider("SnapRef", method="apply", feature="SnapRef", kwargs={
                    "model": model, "clip": clip, "vae": vae, "image": image, "adapter": file, "mode": mode,
                    "ref_in_uncond": False, "ref_megapixels": megapixels, "strength": strength, "ref_caption": caption,
                    "ref_guidance_on": guidance_on, "ref_guidance": guidance, "correction": correction, "ref_end": ref_end,
                    **newer,
                })[0]
            except TypeError as error:
                if newer and "unexpected keyword" in str(error):
                    raise RuntimeError("SnapRef 노드가 옛 버전이라 충실도·두 번째 이미지를 못 씀: 노드를 새로 받으세요") from error
                raise
            # with its guidance on SnapRef keeps the batch whole; off, only the positive row reads the reference
            return (patched if guidance_on else _outermost_wrapper(patched, SPLIT_WRAPPER_KEY, _split_mixed_batch),)
        patched = invoke_provider("AnimaRefAdapter", method="apply", feature="Anima Ref Adapter", kwargs={
            "model": model, "vae": vae, "image": image, "lora_name": file, "ref_guidance": guidance,
            "correction": correction, "strength": strength, "ref_end": ref_end, "ref_megapixels": megapixels,
        })[0]
        if _adapter_is_asym(folder_paths_module().get_full_path_or_raise("loras", file)):
            patched = patched.clone()
            patched.model_options["transformer_options"][ASYM_REFERENCE_KEY] = True  # read by the guidance suite
        return (patched,)


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
            "steps": ("INT", {"default": 32, "min": 1, "max": 10000}),
            "forge_syntax": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Reference image for Krea2 through comfyui-krea2edit and an edit LoRA."

    def check_lazy_status(self, **_inputs):
        return []

    def apply(self, model, clip, vae, image, text, negative_text, file, strength, grounding_px, boost, position, width,
              height, steps=32, forge_syntax=True, positive=None, negative=None):
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
            return scheduled_encode(
                lambda segment: invoke_provider("Krea2EditGroundedEncode", method="encode", feature=feature, kwargs={
                    "clip": clip, "prompt": segment, "image": image, "grounding_px": pixels})[0],
                prompt, steps, forge_syntax)

        return patched, encode(text, grounding_px), encode(negative_text, KREA2_NEGATIVE_GROUNDING)


NODE_CLASS_MAPPINGS = {
    "AIStudioTextEncoderLoader": AIStudioTextEncoderLoader,
    "AIStudioAttention": AIStudioAttention,
    "AIStudioMossStream": AIStudioMossStream,
    "AIStudioAnimaReference": AIStudioAnimaReference,
    "AIStudioKrea2Reference": AIStudioKrea2Reference,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIStudioTextEncoderLoader": "AI Studio Text Encoder Loader",
    "AIStudioAttention": "AI Studio Attention",
    "AIStudioMossStream": "AI Studio MOSS fp32 Stream",
    "AIStudioAnimaReference": "AI Studio Anima Reference",
    "AIStudioKrea2Reference": "AI Studio Krea2 Reference",
}
