"""ComfyUI node-class vocabularies shared by the workflow inspector and compiler.

The workflow picker (``backends/comfyui_workflow_inspector.py``), the
generation compiler (``core/comfy_workflow_compiler.py``) and the metadata
reader must agree on which nodes are samplers, text encoders and model
loaders; a class known to one and not the other is how a workflow ends up
"valid" in the picker yet rejected at generation time.  Keep the sets here.

Output/save vocabularies differ on purpose (the metadata reader also reads
video/animated outputs, the compiler only rewrites image outputs) and are not
merged here.
"""
from __future__ import annotations

from types import MappingProxyType

SAMPLER_NODES: frozenset[str] = frozenset({
    "KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced",
    "ForgeNeoKSamplerCNS",
})
# Samplers whose steps/CFG/denoise live in separate SIGMAS/GUIDER nodes; the
# app cannot map a Forge payload onto them.
CUSTOM_SAMPLER_NODES: frozenset[str] = frozenset({
    "SamplerCustom", "SamplerCustomAdvanced",
})

SEMANTIC_ENCODER_NODES: frozenset[str] = frozenset({
    "ForgeNeoAnimaQwen35Prompt", "ForgeNeoAnima38V2Prompt",
})
TEXT_ENCODER_NODES: frozenset[str] = frozenset(
    {"CLIPTextEncode", "CLIPTextEncodeSDXL"} | SEMANTIC_ENCODER_NODES
)

# Image outputs a custom workflow can end in (compiler rewrites, picker counts).
IMAGE_SAVE_NODES: frozenset[str] = frozenset({
    "SaveImage", "PreviewImage", "ForgeNeoSaveImage",
})

# Model loaders whose file input the app may replace with the selected model:
# class -> input name.  Anything else upstream of the sampler is a custom
# ("locked") loader that owns its model.
MODEL_LOADER_INPUTS = MappingProxyType({
    "CheckpointLoaderSimple": "ckpt_name",
    "CheckpointLoader": "ckpt_name",
    "UNETLoader": "unet_name",
    "DiffusionModelLoaderKJ": "model_name",
    "ForgeNeoAnima38V2Loader": "model_name",
})
CHECKPOINT_LOADER_NODES: frozenset[str] = frozenset({
    "CheckpointLoaderSimple", "CheckpointLoader",
})
UNET_LOADER_NODES: frozenset[str] = frozenset({
    "UNETLoader", "DiffusionModelLoaderKJ", "ForgeNeoAnima38V2Loader",
})


__all__ = [
    "CHECKPOINT_LOADER_NODES", "CUSTOM_SAMPLER_NODES", "IMAGE_SAVE_NODES",
    "MODEL_LOADER_INPUTS", "SAMPLER_NODES", "SEMANTIC_ENCODER_NODES",
    "TEXT_ENCODER_NODES", "UNET_LOADER_NODES",
]
