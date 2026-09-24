"""MiniMax H3 / Krea 2 ComfyUI workflow builder.

This module deliberately has one public interface: :func:`build`.  It performs
all input normalization and model-specific graph construction without importing
PyQt, Vue, ComfyUI, or application state.  The returned value is JSON-safe and
contains both the prompt graph and the node capability contract needed to check
``/object_info`` before a graph is queued.

Supported modes
---------------
``h3_t2v``
    Text-to-video.
``h3_i2v``
    Image-to-video.  Requires ``input_image``.
``h3_v2v``
    Reference-video generation.  Requires ``input_video`` and optionally uses
    ``input_image`` as an identity reference.
``krea2_edit``
    Krea 2 Identity Edit.  Requires ``input_image``.
``krea2_t2i``
    Krea 2 Turbo text-to-image.
``krea2_hires``
    Krea 2 latent hires pass.  Requires ``input_image``.

``params["model_preset"]`` may be a partial mapping over
:data:`DEFAULT_MODEL_PRESET`; filenames are validated before being placed in a
graph.  No filesystem access is performed here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any


class CreatorWorkflowError(ValueError):
    """Raised when a creator workflow cannot be built from the supplied input."""


class CreatorWorkflow(dict[str, Any]):
    """JSON-safe result mapping with adapter-friendly attribute access."""

    @property
    def workflow(self) -> dict[str, dict[str, Any]]:
        return self["workflow"]

    @property
    def required_node_types(self) -> list[str]:
        return self["required_node_types"]

    @property
    def custom_node_types(self) -> list[str]:
        return self["custom_node_types"]


SUPPORTED_MODES: tuple[str, ...] = (
    "h3_t2v",
    "h3_i2v",
    "h3_v2v",
    "krea2_t2i",
    "krea2_edit",
    "krea2_hires",
)

_MODE_ALIASES: dict[str, str] = {
    "krea2": "krea2_edit",
    "krea_edit": "krea2_edit",
    "krea_hires": "krea2_hires",
}


DEFAULT_MODEL_PRESET: dict[str, str] = {
    "h3_unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "h3_ref_unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "h3_clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "h3_video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "h3_audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "h3_turbo_lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
    "krea2_unet": "krea2_turbo_int8_convrot.safetensors",
    "krea2_clip": "qwen3vl_4b_fp8_scaled.safetensors",
    "krea2_vae": "qwen_image_vae.safetensors",
    "krea2_identity_lora": "Krea2/krea2_identity_edit_v1_2.safetensors",
    "krea2_textfusion_lora": "Krea2/Krea2_TextFusion_Refusal_Reduction.safetensors",
    "krea2_upscaler": "RealESRGAN_x4plus.safetensors",
}


_MODEL_EXTENSIONS: dict[str, frozenset[str]] = {
    **{
        key: frozenset({".safetensors"})
        for key in DEFAULT_MODEL_PRESET
        if key != "krea2_upscaler"
    },
    "krea2_upscaler": frozenset({".pth", ".pt", ".safetensors"}),
}

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"})
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')
_SAMPLERS = frozenset({"euler", "heun", "dpmpp_sde", "dpmpp_2m_sde"})

_CUSTOM_NODE_TYPES = frozenset(
    {
        "GemmaVideoReferencePreprocessor",
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3TurboSampler",
        "MiniMaxH3BlockCacheT8",
        "Krea2EditGroundedEncode",
        "Krea2EditModelPatch",
        "ForgeNeoH3ConditioningCachePrepare",
        "ForgeNeoH3ConditioningCacheLoad",
    }
)

_CAPABILITIES: dict[str, dict[str, Any]] = {
    "h3_t2v": {
        "family": "minimax_h3",
        "description": "MiniMax H3 text-to-video with MP4 and animated WebP outputs.",
        "inputs": ["prompt"],
        "outputs": ["video/mp4", "image/webp"],
    },
    "h3_i2v": {
        "family": "minimax_h3",
        "description": "MiniMax H3 image-to-video with a first-frame image.",
        "inputs": ["prompt", "input_image"],
        "outputs": ["video/mp4", "image/webp"],
    },
    "h3_v2v": {
        "family": "minimax_h3",
        "description": "MiniMax H3 reference-video generation with an optional identity image.",
        "inputs": ["prompt", "input_video", "input_image?"],
        "outputs": ["video/mp4", "image/webp"],
    },
    "krea2_t2i": {
        "family": "krea2",
        "description": "Krea 2 Turbo text-to-image generation.",
        "inputs": ["prompt"],
        "outputs": ["image"],
    },
    "krea2_edit": {
        "family": "krea2",
        "description": "Krea 2 identity-preserving image edit.",
        "inputs": ["prompt", "input_image", "reference_image?"],
        "outputs": ["image"],
    },
    "krea2_hires": {
        "family": "krea2",
        "description": "Krea 2 model-guided hires pass after pixel upscaling.",
        "inputs": ["input_image", "prompt?"],
        "outputs": ["image"],
    },
}


def build(mode: str, params: Mapping[str, Any] | None = None) -> CreatorWorkflow:
    """Build one validated, JSON-safe ComfyUI workflow description.

    The return mapping has these stable keys:

    ``workflow``
        ComfyUI API-format prompt graph.
    ``required_node_types``
        Every ``class_type`` the target ComfyUI must expose in ``/object_info``.
    ``custom_node_types``
        The subset most likely to require optional/custom node packs.
    ``capability``
        Human-readable input/output contract and a clear missing-node error.
    ``output_node_ids``
        Output nodes whose history entries contain generated artifacts.
    ``metadata``
        Canonical mode, effective dimensions/frame count, seed, and model preset.
    """

    canonical_mode = _canonical_mode(mode)
    if params is None:
        values: Mapping[str, Any] = {}
    elif isinstance(params, Mapping):
        values = _normalize_params(canonical_mode, params)
    else:
        raise TypeError("params must be a mapping")

    models = _resolve_model_preset(values.get("model_preset"))
    if canonical_mode.startswith("h3_"):
        graph, output_node_ids, metadata = _build_h3(canonical_mode, values, models)
    elif canonical_mode == "krea2_t2i":
        graph, output_node_ids, metadata = _build_krea2_t2i(values, models)
    elif canonical_mode == "krea2_edit":
        graph, output_node_ids, metadata = _build_krea2_edit(values, models)
    else:
        graph, output_node_ids, metadata = _build_krea2_hires(values, models)

    stages = []
    if canonical_mode.startswith("h3_") and _boolean(values.get("conditioning_cache", False), "conditioning_cache"):
        from core.h3_conditioning_cache import DEFAULT_MAX_BYTES, DEFAULT_MAX_ENTRIES, split_workflow
        stages = split_workflow(graph, {"mode": canonical_mode, **metadata},
                                max_bytes=values.get("conditioning_cache_max_bytes", DEFAULT_MAX_BYTES),
                                max_entries=values.get("conditioning_cache_max_entries", DEFAULT_MAX_ENTRIES))
        graph = stages[-1]["workflow"]
    required = sorted(
        {
            str(node.get("class_type", ""))
            for stage_graph in ([stage["workflow"] for stage in stages] or [graph])
            for node in stage_graph.values()
            if isinstance(node, Mapping) and node.get("class_type")
        }
    )
    custom = [node_type for node_type in required if node_type in _CUSTOM_NODE_TYPES]
    capability = dict(_CAPABILITIES[canonical_mode])
    capability.update(
        {
            "id": canonical_mode,
            "required_node_types": list(required),
            "missing_node_error": (
                f"{canonical_mode} cannot run because ComfyUI is missing required node types. "
                "Compare required_node_types with /object_info before queueing the workflow."
            ),
        }
    )
    metadata = {
        "mode": canonical_mode,
        "models": dict(models),
        **metadata,
    }
    result = CreatorWorkflow({
        "workflow": graph,
        "required_node_types": required,
        "custom_node_types": custom,
        "capability": capability,
        "output_node_ids": output_node_ids,
        "metadata": metadata,
    })
    if stages:
        result["stages"] = stages
        result["metadata"]["conditioning_cache"] = True
    return result


def _normalize_params(mode: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Accept transport-friendly aliases while keeping one canonical graph contract."""

    values = dict(params)

    def alias(canonical: str, *alternatives: str) -> None:
        if values.get(canonical) is not None:
            return
        for name in alternatives:
            if values.get(name) is not None:
                values[canonical] = values[name]
                return

    alias("model_preset", "modelPreset")
    alias("output_prefix", "outputPrefix")
    alias("save_webp", "saveWebp", "saveAnimatedWebp")
    alias("generate_audio", "include_audio", "includeAudio", "generateAudio")
    alias("include_reference_audio", "includeReferenceAudio")
    alias("reference_megapixels", "referenceMegapixels")
    alias("ref_image_size", "refImageSize")
    alias("block_cache", "blockCache")
    alias("conditioning_cache", "h3ConditioningCache", "h3ConditioningCacheEnabled")
    alias("low_vram", "lowVram")
    alias("webp_lossless", "webpLossless")
    alias("webp_quality", "webpQuality")
    alias("webp_method", "webpMethod")
    alias("use_textfusion", "useTextFusion", "useTextFusionLora")
    alias("textfusion_strength", "textFusionStrength")
    alias("source_size", "sourceSize")
    alias("scale", "hiresScale")
    alias("denoise", "hiresDenoise")
    alias("reference_image", "referenceImage")

    # The action adapter historically names uploaded media source_image/source_video.
    # In V2V, a generic source_image field may actually contain an uploaded video;
    # detect that by extension so the pure builder remains transport-independent.
    if mode == "h3_v2v":
        alias("input_video", "source_video", "sourceVideo")
        source = values.get("source_image", values.get("sourceImage"))
        if values.get("input_video") is None and _has_extension(source, _VIDEO_EXTENSIONS):
            values["input_video"] = source
        elif values.get("input_image") is None and source is not None:
            values["input_image"] = source
        if "include_reference_audio" not in values and "includeAudio" in values:
            values["include_reference_audio"] = values["includeAudio"]
    else:
        alias("input_image", "source_image", "sourceImage")
        alias("input_video", "source_video", "sourceVideo")
    return values


def _has_extension(raw: Any, extensions: frozenset[str]) -> bool:
    if not isinstance(raw, str):
        return False
    return PurePosixPath(raw.replace("\\", "/")).suffix.lower() in extensions


def canonical_mode(mode: Any) -> str:
    """요청 mode 문자열 → build() 가 쓰는 정규 mode (별칭·대소문자·'-' 처리).

    호출부(Creator 액션 등)가 별칭 표를 따로 들고 있다 갈라지지 않게 이 함수만 쓴다.
    지원하지 않는 mode 는 CreatorWorkflowError.
    """
    return _canonical_mode(mode)


def media_inputs(mode: Any) -> frozenset[str]:
    """mode 의 그래프가 읽는 업로드 입력 키(``input_image``/``input_video``/``reference_image``).

    ``_CAPABILITIES`` 의 ``inputs`` 에서 ``prompt`` 를 뺀 것(선택 입력의 ``?`` 제거). Creator
    액션은 이 집합 밖의 미디어(t2v 의 원본, i2v 에 남은 아이덴티티 등)를 올리지 않는다.
    지원하지 않는 mode 는 CreatorWorkflowError.
    """
    names = (str(name).rstrip("?") for name in _CAPABILITIES[_canonical_mode(mode)]["inputs"])
    return frozenset(name for name in names if name != "prompt")


def _canonical_mode(mode: Any) -> str:
    if not isinstance(mode, str) or not mode.strip():
        raise CreatorWorkflowError("mode must be a non-empty string")
    canonical = mode.strip().lower().replace("-", "_")
    canonical = _MODE_ALIASES.get(canonical, canonical)
    if canonical not in SUPPORTED_MODES:
        choices = ", ".join(SUPPORTED_MODES)
        raise CreatorWorkflowError(f"unsupported creator mode '{mode}'; expected one of: {choices}")
    return canonical


def _resolve_model_preset(raw_preset: Any) -> dict[str, str]:
    models = dict(DEFAULT_MODEL_PRESET)
    if raw_preset is None:
        return models
    if not isinstance(raw_preset, Mapping):
        raise CreatorWorkflowError("model_preset must be a mapping of model key to relative filename")
    unknown = sorted(set(raw_preset) - set(DEFAULT_MODEL_PRESET))
    if unknown:
        raise CreatorWorkflowError(f"unknown model_preset keys: {', '.join(map(str, unknown))}")
    for key, value in raw_preset.items():
        models[str(key)] = _relative_filename(
            value,
            f"model_preset.{key}",
            extensions=_MODEL_EXTENSIONS[str(key)],
        )
    return models


def _build_h3(
    mode: str,
    params: Mapping[str, Any],
    models: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    prompt = _text(params.get("prompt"), "prompt", required=True)
    width, height = _size(params, default=(608, 352), minimum=64, maximum=4096)
    if width % 32 or height % 32:
        raise CreatorWorkflowError("MiniMax H3 width and height must be multiples of 32")

    fps = _number(params.get("fps", 24), "fps", minimum=24, maximum=24)
    requested_frames = _requested_frames(params, fps)
    frames = _align_h3_frames(requested_frames)
    if frames > 3600:
        raise CreatorWorkflowError("frames align past MiniMax H3's maximum of 3600")
    if requested_frames / fps > 149:
        raise CreatorWorkflowError("MiniMax H3 duration derived from frames/fps must not exceed 149 seconds")

    seed = _integer(params.get("seed", 1), "seed", minimum=0, maximum=(1 << 64) - 1)
    quality = str(params.get("quality", "quality" if mode == "h3_v2v" else "turbo")).strip().lower()
    if quality not in {"turbo", "quality"}:
        raise CreatorWorkflowError("quality must be 'turbo' or 'quality'")
    if mode == "h3_v2v" and quality == "turbo":
        raise CreatorWorkflowError(
            "h3_v2v requires quality mode because the Turbo LoRA is validated only for t2v/i2v"
        )
    output_prefix = _relative_filename(
        params.get("output_prefix", "Creator/MiniMaxH3"),
        "output_prefix",
    )
    save_webp = _boolean(params.get("save_webp", True), "save_webp")
    include_reference_audio = _boolean(
        params.get("include_reference_audio", False),
        "include_reference_audio",
    )
    generate_audio = _boolean(
        params.get("generate_audio", include_reference_audio),
        "generate_audio",
    )
    if mode != "h3_v2v" and include_reference_audio:
        raise CreatorWorkflowError("include_reference_audio is only valid for h3_v2v")
    with_audio = generate_audio or include_reference_audio

    input_image = ""
    if mode in {"h3_i2v", "h3_v2v"} and params.get("input_image"):
        input_image = _relative_filename(
            params.get("input_image"),
            "input_image",
            extensions=_IMAGE_EXTENSIONS,
        )
    if mode == "h3_i2v" and not input_image:
        raise CreatorWorkflowError("h3_i2v requires input_image")

    input_video = ""
    if mode == "h3_v2v":
        input_video = _relative_filename(
            params.get("input_video"),
            "input_video",
            extensions=_VIDEO_EXTENSIONS,
        )

    graph: dict[str, dict[str, Any]] = {
        "1": _node(
            "UNETLoader",
            unet_name=models["h3_ref_unet" if mode == "h3_v2v" else "h3_unet"],
            weight_dtype="default",
        ),
        "2": _node("CLIPLoader", clip_name=models["h3_clip"], type="minimax", device="default"),
        "3": _node("VAELoader", vae_name=models["h3_video_vae"]),
    }
    if mode == "h3_v2v" or with_audio:
        graph["4"] = _node("VAELoader", vae_name=models["h3_audio_vae"])
    if input_image:
        graph["5"] = _node("LoadImage", image=input_image)

    if mode == "h3_v2v":
        reference_megapixels = _number(
            params.get("reference_megapixels", min(0.2, width * height / (1024 * 1024))),
            "reference_megapixels",
            minimum=0.01,
            maximum=1,
        )
        graph["18"] = _node(
            "GemmaVideoReferencePreprocessor",
            file=input_video,
            megapixels=reference_megapixels,
            fps=fps,
            duration=min(15.0, requested_frames / fps),
            include_audio=include_reference_audio,
        )
        conditioning_inputs: dict[str, Any] = {
            "clip": _ref(2),
            "vae": _ref(3),
            "audio_vae": _ref(4),
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": frames,
            "ref_image_size": _choice(
                params.get("ref_image_size", "max"),
                "ref_image_size",
                {"match", "max"},
            ),
            "ref_videos.ref_video_0": _ref(18),
        }
        if input_image:
            conditioning_inputs["ref_images.ref_image_0"] = _ref(5)
        if include_reference_audio:
            conditioning_inputs["ref_video_audios.ref_video_audio_0"] = _ref(18, 1)
        graph["6"] = _node("MiniMaxH3ReferenceToVideo", **conditioning_inputs)
    else:
        conditioning_inputs = {
            "clip": _ref(2),
            "vae": _ref(3),
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": frames,
        }
        if input_image:
            conditioning_inputs["first_frame"] = _ref(5)
        graph["6"] = _node("MiniMaxH3ImageToVideo", **conditioning_inputs)

    block_cache = _boolean(params.get("block_cache", False), "block_cache")
    if quality != "turbo" and block_cache:
        raise CreatorWorkflowError("block_cache is only valid with quality='turbo'")
    if quality == "turbo":
        graph["16"] = _node(
            "MiniMaxH3TurboLoRA",
            model=_ref(1),
            lora_name=models["h3_turbo_lora"],
            strength=_number(params.get("turbo_lora_strength", 1), "turbo_lora_strength", minimum=0, maximum=2),
            low_vram=_boolean(params.get("low_vram", mode == "h3_v2v"), "low_vram"),
        )
        model_ref = _ref(16)
        if block_cache:
            graph["19"] = _node(
                "MiniMaxH3BlockCacheT8",
                model=model_ref,
                residual_diff_threshold=_number(
                    params.get("cache_threshold", 0.12),
                    "cache_threshold",
                    minimum=0,
                    maximum=1,
                ),
                start_percent=0.08,
                end_percent=0.95,
                max_consecutive_hits=2,
                cache_device="cpu",
                metric_stride=8,
                verbose=False,
            )
            model_ref = _ref(19)
    else:
        model_ref = _ref(1)

    graph.update(
        {
            "7": _node("RandomNoise", noise_seed=seed),
            "8": _node("BasicGuider", model=model_ref, conditioning=_ref(6)),
            "9": (
                _node("MiniMaxH3TurboSampler")
                if quality == "turbo"
                else _node("KSamplerSelect", sampler_name="res_multistep")
            ),
            "10": _node(
                "BasicScheduler",
                model=model_ref,
                scheduler="simple",
                steps=8 if quality == "turbo" else 20,
                denoise=1,
            ),
            "11": _node(
                "SamplerCustomAdvanced",
                noise=_ref(7),
                guider=_ref(8),
                sampler=_ref(9),
                sigmas=_ref(10),
                latent_image=_ref(6, 1),
            ),
            "12": _node("VAEDecode", samples=_ref(11), vae=_ref(3)),
        }
    )
    if with_audio:
        graph["13"] = _node("VAEDecodeAudio", samples=_ref(11), vae=_ref(4))
    video_inputs: dict[str, Any] = {"images": _ref(12), "fps": fps, "bit_depth": 8}
    if with_audio:
        video_inputs["audio"] = _ref(13)
    graph["14"] = _node("CreateVideo", **video_inputs)
    graph["15"] = _node(
        "SaveVideo",
        video=_ref(14),
        filename_prefix=output_prefix,
        format="mp4",
        codec={"codec": "auto"},
    )
    output_node_ids = ["15"]
    if save_webp:
        graph["20"] = _node(
            "SaveAnimatedWEBP",
            images=_ref(12),
            filename_prefix=f"{output_prefix}_preview",
            fps=fps,
            lossless=_boolean(params.get("webp_lossless", False), "webp_lossless"),
            quality=_integer(params.get("webp_quality", 82), "webp_quality", minimum=0, maximum=100),
            method=_choice(params.get("webp_method", "default"), "webp_method", {"default", "fastest", "slowest"}),
        )
        output_node_ids.append("20")

    return graph, output_node_ids, {
        "width": width,
        "height": height,
        "fps": fps,
        "requested_frames": requested_frames,
        "frames": frames,
        "duration": requested_frames / fps,
        "seed": seed,
        "quality": quality,
        "audio": with_audio,
    }


def _build_krea2_t2i(
    params: Mapping[str, Any],
    models: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    """Build the native ComfyUI Krea 2 Turbo text-to-image graph.

    This mirrors Comfy's official Turbo template: eight distilled Euler/simple
    steps at CFG 1, with the positive conditioning zeroed for the unconditional
    branch.  The optional TextFusion LoRA is deliberately off by default so the
    three official Krea model files are sufficient for T2I.
    """

    prompt = _text(params.get("prompt"), "prompt", required=True)
    width, height = _size(params, default=(1024, 1024), minimum=256, maximum=4096)
    internal_width = _align(width, 8)
    internal_height = _align(height, 8)
    seed = _integer(params.get("seed", 1), "seed", minimum=0, maximum=(1 << 64) - 1)
    steps = _integer(params.get("steps", 8), "steps", minimum=1, maximum=80)
    cfg = _number(params.get("cfg", 1), "cfg", minimum=1, maximum=10)
    sampler = _choice(params.get("sampler", "euler"), "sampler", _SAMPLERS)
    output_prefix = _relative_filename(
        params.get("output_prefix", "Krea2/T2I"),
        "output_prefix",
    )
    use_textfusion = _boolean(params.get("use_textfusion", False), "use_textfusion")
    textfusion_strength = _number(
        params.get("textfusion_strength", 0.35),
        "textfusion_strength",
        minimum=0,
        maximum=1.5,
    )

    graph: dict[str, dict[str, Any]] = {
        "1": _node("UNETLoader", unet_name=models["krea2_unet"], weight_dtype="default"),
        "2": _node("CLIPLoader", clip_name=models["krea2_clip"], type="krea2", device="default"),
        "3": _node("VAELoader", vae_name=models["krea2_vae"]),
        "4": _node("CLIPTextEncode", text=prompt, clip=_ref(2)),
        "5": _node("ConditioningZeroOut", conditioning=_ref(4)),
        "6": _node(
            "EmptyLatentImage",
            width=internal_width,
            height=internal_height,
            batch_size=1,
        ),
    }
    model_ref = _ref(1)
    if use_textfusion:
        graph["7"] = _node(
            "LoraLoaderModelOnly",
            model=model_ref,
            lora_name=models["krea2_textfusion_lora"],
            strength_model=textfusion_strength,
        )
        model_ref = _ref(7)
    graph["8"] = _node(
        "KSampler",
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler="simple",
        denoise=1,
        model=model_ref,
        positive=_ref(4),
        negative=_ref(5),
        latent_image=_ref(6),
    )
    graph["9"] = _node("VAEDecode", samples=_ref(8), vae=_ref(3))
    image_ref = _ref(9)
    if (internal_width, internal_height) != (width, height):
        graph["10"] = _node(
            "ImageCrop",
            width=width,
            height=height,
            x=(internal_width - width) // 2,
            y=(internal_height - height) // 2,
            image=image_ref,
        )
        image_ref = _ref(10)
    graph["11"] = _node("SaveImage", filename_prefix=output_prefix, images=image_ref)
    return graph, ["11"], {
        "width": width,
        "height": height,
        "internal_width": internal_width,
        "internal_height": internal_height,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": "simple",
    }


def _build_krea2_edit(
    params: Mapping[str, Any],
    models: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    prompt = _text(params.get("prompt"), "prompt", required=True)
    input_image = _relative_filename(
        params.get("input_image"),
        "input_image",
        extensions=_IMAGE_EXTENSIONS,
    )
    reference_image = ""
    if params.get("reference_image"):
        reference_image = _relative_filename(
            params.get("reference_image"),
            "reference_image",
            extensions=_IMAGE_EXTENSIONS,
        )
    width, height = _size(params, default=(1024, 1024), minimum=256, maximum=4096)
    internal_width = _align(width, 16)
    internal_height = _align(height, 16)
    seed = _integer(params.get("seed", 1), "seed", minimum=0, maximum=(1 << 64) - 1)
    fidelity = _number(params.get("fidelity", 4), "fidelity", minimum=0.5, maximum=12)
    steps = _integer(params.get("steps", 15), "steps", minimum=1, maximum=80)
    cfg = _number(params.get("cfg", 1), "cfg", minimum=1, maximum=10)
    sampler = _choice(params.get("sampler", "euler"), "sampler", _SAMPLERS)
    grounding_px = _integer(params.get("grounding_px", 768), "grounding_px", minimum=256, maximum=2048)
    output_prefix = _relative_filename(
        params.get("output_prefix", "Creator/Krea2Edit"),
        "output_prefix",
    )
    use_textfusion = _boolean(params.get("use_textfusion", True), "use_textfusion")
    textfusion_strength = _number(
        params.get("textfusion_strength", 0.35),
        "textfusion_strength",
        minimum=0,
        maximum=1.5,
    )

    graph: dict[str, dict[str, Any]] = {
        "1": _node("UNETLoader", unet_name=models["krea2_unet"], weight_dtype="default"),
        "2": _node("CLIPLoader", clip_name=models["krea2_clip"], type="krea2", device="default"),
        "3": _node("VAELoader", vae_name=models["krea2_vae"]),
        "5": _node("LoadImage", image=input_image),
        "6": _node("VAEEncode", pixels=_ref(5), vae=_ref(3)),
        "7": _node(
            "EmptySD3LatentImage",
            width=internal_width,
            height=internal_height,
            batch_size=1,
        ),
        "8": _node(
            "Krea2EditGroundedEncode",
            clip=_ref(2),
            prompt=prompt,
            image=_ref(5),
            grounding_px=grounding_px,
            system_prompt="",
        ),
        "9": _node(
            "Krea2EditGroundedEncode",
            clip=_ref(2),
            prompt="",
            image=_ref(5),
            grounding_px=grounding_px,
            system_prompt="",
        ),
    }
    if reference_image:
        graph["16"] = _node("LoadImage", image=reference_image)
        graph["17"] = _node("VAEEncode", pixels=_ref(16), vae=_ref(3))
        graph["8"]["inputs"]["image_b"] = _ref(16)
        graph["9"]["inputs"]["image_b"] = _ref(16)
    base_model_ref = _ref(1)
    if use_textfusion:
        graph["15"] = _node(
            "LoraLoaderModelOnly",
            model=base_model_ref,
            lora_name=models["krea2_textfusion_lora"],
            strength_model=textfusion_strength,
        )
        base_model_ref = _ref(15)
    graph["4"] = _node(
        "LoraLoaderModelOnly",
        model=base_model_ref,
        lora_name=models["krea2_identity_lora"],
        strength_model=1,
    )
    patch_inputs: dict[str, Any] = {
        "model": _ref(4),
        "source_latent": _ref(6),
        "ref_boost": fidelity,
        "ref_boost_a": 1,
        "fit_mode": "fit",
        "vae": _ref(3),
        "source_image": _ref(5),
        "target_latent": _ref(7),
    }
    if reference_image:
        patch_inputs["source_latent_b"] = _ref(17)
        patch_inputs["source_image_b"] = _ref(16)
    graph["10"] = _node(
        "Krea2EditModelPatch",
        **patch_inputs,
    )
    graph["11"] = _node(
        "KSampler",
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler="simple",
        denoise=1,
        model=_ref(10),
        positive=_ref(8),
        negative=_ref(9),
        latent_image=_ref(7),
    )
    graph["12"] = _node("VAEDecode", samples=_ref(11), vae=_ref(3))
    image_ref = _ref(12)
    if (internal_width, internal_height) != (width, height):
        graph["13"] = _node(
            "ImageCrop",
            width=width,
            height=height,
            x=(internal_width - width) // 2,
            y=(internal_height - height) // 2,
            image=image_ref,
        )
        image_ref = _ref(13)
    graph["14"] = _node("SaveImage", filename_prefix=output_prefix, images=image_ref)
    return graph, ["14"], {
        "width": width,
        "height": height,
        "internal_width": internal_width,
        "internal_height": internal_height,
        "seed": seed,
        "fidelity": fidelity,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "scheduler": "simple",
        "uses_reference_image": bool(reference_image),
    }


def _build_krea2_hires(
    params: Mapping[str, Any],
    models: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    prompt = _text(params.get("prompt", ""), "prompt", required=False)
    input_image = _relative_filename(
        params.get("input_image"),
        "input_image",
        extensions=_IMAGE_EXTENSIONS,
    )
    width, height, scale = _hires_size(params)
    internal_width = _align(width, 16)
    internal_height = _align(height, 16)
    seed = _integer(params.get("seed", 1), "seed", minimum=0, maximum=(1 << 64) - 1)
    steps = _integer(params.get("steps", 6), "steps", minimum=1, maximum=80)
    cfg = _number(params.get("cfg", 2), "cfg", minimum=1, maximum=10)
    denoise = _number(params.get("denoise", 0.45), "denoise", minimum=0.05, maximum=1)
    sampler = _choice(params.get("sampler", "euler"), "sampler", _SAMPLERS)
    output_prefix = _relative_filename(
        params.get("output_prefix", "Creator/Krea2Hires"),
        "output_prefix",
    )
    use_textfusion = _boolean(params.get("use_textfusion", True), "use_textfusion")
    textfusion_strength = _number(
        params.get("textfusion_strength", 0.35),
        "textfusion_strength",
        minimum=0,
        maximum=1.5,
    )

    graph: dict[str, dict[str, Any]] = {
        "1": _node("UNETLoader", unet_name=models["krea2_unet"], weight_dtype="default"),
        "2": _node("CLIPLoader", clip_name=models["krea2_clip"], type="krea2", device="default"),
        "3": _node("VAELoader", vae_name=models["krea2_vae"]),
        "4": _node("CLIPTextEncode", text=prompt, clip=_ref(2)),
        "5": _node("ConditioningZeroOut", conditioning=_ref(4)),
        "6": _node("LoadImage", image=input_image),
        "7": _node("UpscaleModelLoader", model_name=models["krea2_upscaler"]),
        "8": _node("ImageUpscaleWithModel", upscale_model=_ref(7), image=_ref(6)),
        "9": _node(
            "ImageScale",
            image=_ref(8),
            upscale_method="lanczos",
            width=internal_width,
            height=internal_height,
            crop="disabled",
        ),
        "10": _node("VAEEncode", pixels=_ref(9), vae=_ref(3)),
    }
    model_ref = _ref(1)
    if use_textfusion:
        graph["11"] = _node(
            "LoraLoaderModelOnly",
            model=model_ref,
            lora_name=models["krea2_textfusion_lora"],
            strength_model=textfusion_strength,
        )
        model_ref = _ref(11)
    graph["12"] = _node(
        "KSampler",
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler="simple",
        denoise=denoise,
        model=model_ref,
        positive=_ref(4),
        negative=_ref(5),
        latent_image=_ref(10),
    )
    graph["13"] = _node("VAEDecode", samples=_ref(12), vae=_ref(3))
    image_ref = _ref(13)
    if (internal_width, internal_height) != (width, height):
        graph["14"] = _node(
            "ImageCrop",
            width=width,
            height=height,
            x=(internal_width - width) // 2,
            y=(internal_height - height) // 2,
            image=image_ref,
        )
        image_ref = _ref(14)
    graph["15"] = _node("SaveImage", filename_prefix=output_prefix, images=image_ref)
    return graph, ["15"], {
        "width": width,
        "height": height,
        "internal_width": internal_width,
        "internal_height": internal_height,
        "scale": scale,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "denoise": denoise,
    }


def _hires_size(params: Mapping[str, Any]) -> tuple[int, int, float | None]:
    if params.get("size") is not None or (params.get("width") is not None and params.get("height") is not None):
        width, height = _size(params, default=(2048, 2048), minimum=256, maximum=8192)
        return width, height, None
    source_raw = params.get("source_size")
    if source_raw is None:
        return 2048, 2048, None
    source_width, source_height = _parse_size(source_raw, "source_size", minimum=64, maximum=4096)
    scale = _number(params.get("scale", 2), "scale", minimum=1, maximum=4)
    width = max(256, int(round(source_width * scale)))
    height = max(256, int(round(source_height * scale)))
    if width > 8192 or height > 8192:
        raise CreatorWorkflowError("krea2_hires target size must not exceed 8192 on either side")
    return width, height, scale


def _requested_frames(params: Mapping[str, Any], fps: float) -> int:
    has_frames = params.get("frames") is not None
    has_duration = params.get("duration") is not None
    if has_frames and has_duration:
        raise CreatorWorkflowError("provide frames or duration, not both")
    if has_duration:
        duration = _number(params.get("duration"), "duration", minimum=5 / fps, maximum=149)
        return max(5, int(round(duration * fps)))
    return _integer(params.get("frames", 124), "frames", minimum=5, maximum=3600)


def _align_h3_frames(frames: int) -> int:
    """Snap up to MiniMax H3's required ``17k + 5`` frame grid."""

    return frames + ((5 - (frames % 17)) + 17) % 17


def _size(
    params: Mapping[str, Any],
    *,
    default: tuple[int, int],
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    raw = params.get("size")
    if raw is None and (params.get("width") is not None or params.get("height") is not None):
        if params.get("width") is None or params.get("height") is None:
            raise CreatorWorkflowError("width and height must be provided together")
        raw = (params.get("width"), params.get("height"))
    if raw is None:
        return default
    return _parse_size(raw, "size", minimum=minimum, maximum=maximum)


def _parse_size(raw: Any, name: str, *, minimum: int, maximum: int) -> tuple[int, int]:
    if isinstance(raw, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", raw)
        if not match:
            raise CreatorWorkflowError(f"{name} must look like WIDTHxHEIGHT")
        width = int(match.group(1))
        height = int(match.group(2))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) == 2:
        width = _integer(raw[0], f"{name}.width", minimum=minimum, maximum=maximum)
        height = _integer(raw[1], f"{name}.height", minimum=minimum, maximum=maximum)
    else:
        raise CreatorWorkflowError(f"{name} must be WIDTHxHEIGHT or a two-item sequence")
    if not minimum <= width <= maximum or not minimum <= height <= maximum:
        raise CreatorWorkflowError(f"{name} sides must be between {minimum} and {maximum}")
    return width, height


def _relative_filename(
    raw: Any,
    name: str,
    *,
    extensions: frozenset[str] | None = None,
) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CreatorWorkflowError(f"{name} must be a non-empty relative filename")
    value = raw.strip().replace("\\", "/")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value) or "://" in value:
        raise CreatorWorkflowError(f"{name} must be relative to the ComfyUI model/input/output directory")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise CreatorWorkflowError(f"{name} must not contain empty or traversal path segments")
    for part in parts:
        if _INVALID_FILENAME_CHARS.search(part):
            raise CreatorWorkflowError(f"{name} contains characters that are invalid on Windows")
        if part.endswith((" ", ".")):
            raise CreatorWorkflowError(f"{name} path segments must not end in a space or dot")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED_NAMES:
            raise CreatorWorkflowError(f"{name} contains reserved Windows filename '{part}'")
    if extensions is not None:
        suffix = PurePosixPath(value).suffix.lower()
        if suffix not in extensions:
            choices = ", ".join(sorted(extensions))
            raise CreatorWorkflowError(f"{name} must use one of these extensions: {choices}")
    return value


def _text(raw: Any, name: str, *, required: bool) -> str:
    if not isinstance(raw, str):
        if raw is None:
            raw = ""
        else:
            raise CreatorWorkflowError(f"{name} must be a string")
    value = raw.strip()
    if required and not value:
        raise CreatorWorkflowError(f"{name} must not be empty")
    return value


def _boolean(raw: Any, name: str) -> bool:
    if not isinstance(raw, bool):
        raise CreatorWorkflowError(f"{name} must be a boolean")
    return raw


def _integer(raw: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise CreatorWorkflowError(f"{name} must be an integer")
    if not minimum <= raw <= maximum:
        raise CreatorWorkflowError(f"{name} must be between {minimum} and {maximum}")
    return raw


def _number(raw: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise CreatorWorkflowError(f"{name} must be a number")
    value = float(raw)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise CreatorWorkflowError(f"{name} must be between {minimum} and {maximum}")
    if isinstance(raw, int):
        return raw
    return value


def _choice(raw: Any, name: str, choices: set[str] | frozenset[str]) -> str:
    if not isinstance(raw, str):
        raise CreatorWorkflowError(f"{name} must be one of: {', '.join(sorted(choices))}")
    value = raw.strip().lower()
    if value not in choices:
        raise CreatorWorkflowError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value


def _align(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _node(class_type: str, **inputs: Any) -> dict[str, Any]:
    return {"class_type": class_type, "inputs": inputs}


def _ref(node_id: int | str, slot: int = 0) -> list[Any]:
    return [str(node_id), slot]
