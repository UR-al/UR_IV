"""ComfyUI nodes implementing Forge Neo SAM3 mask/detailer semantics.

The public node classes are intentionally thin.  Provider discovery, mask
normalization, prompt grouping, morphology, artifact persistence and inpaint
sampling live behind private seams so the module remains importable and
testable without importing ComfyUI at discovery time.
"""

from __future__ import annotations

import functools
import gc
import inspect
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from .mask_ops import (
    empty_mask_like,
    ensure_image,
    ensure_mask,
    expand_crop_region,
    finish_masks,
    fit_sample_size,
    make_overlay,
    mask_bounds,
    refine_generated_mask,
    save_mask_artifacts,
    select_mask_groups,
    split_prompt_groups,
    subtract_exclusion,
    union_masks,
)


CATEGORY = "AI Studio/Forge parity/SAM3"
_EASY_SEGMENTATION_KEYS = (
    "easy sam3ImageSegmentation",
    "EasySam3ImageSegmentation",
    "Sam3ImageSegmentation",
)
_EASY_LOADER_KEYS = (
    "easy sam3ModelLoader",
    "EasySam3ModelLoader",
    "LoadSam3Model",
)

_CONTROL_PREPROCESSORS = {
    "tile_resample": "TilePreprocessor",
    "depth_midas": "MiDaS-DepthMapPreprocessor",
    "depth_zoe": "Zoe-DepthMapPreprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "openpose": "OpenposePreprocessor",
    "openpose_full": "OpenposePreprocessor",
    "openpose_hand": "OpenposePreprocessor",
    "lineart_realistic": "LineArtPreprocessor",
    "lineart_anime": "AnimeLineArtPreprocessor",
    "lineart_coarse": "LineartStandardPreprocessor",
    "canny": "CannyEdgePreprocessor",
    "softedge_hed": "HEDPreprocessor",
    "scribble_pidinet": "Scribble_PiDiNet_Preprocessor",
}

_UNSUPPORTED_CONTROL_MODULES = {
    "inpaint_only+lama": (
        "Forge's LaMa-assisted inpaint preprocessor has no equivalent provider "
        "in this node pack"
    ),
    "inpaint_global_harmonious": (
        "Forge's global-harmonious inpaint preprocessor cannot be represented "
        "by ComfyUI's plain ControlNet hint"
    ),
    "tile_colorfix": (
        "Forge tile_colorfix is not equivalent to Comfy's TilePreprocessor"
    ),
    "tile_colorfix+sharp": (
        "Forge tile_colorfix+sharp is not equivalent to TTPlanet Tile Simple"
    ),
}

_CONTROLNET_EXTRA_DEFAULTS = {
    "pixel_perfect": True,
    "override_external": False,
    "control_mode": "Balanced",
    "resize_mode": "Crop and Resize",
    "threshold_a": -1.0,
    "threshold_b": -1.0,
}

_RESTORE_FACE_DEFAULTS = {
    "detector_model": "bbox/face_yolov8m.pt",
    "guide_size": 512,
    "guide_size_for_bbox": True,
    "max_size": 1024,
    "bbox_threshold": 0.5,
    "bbox_dilation": 10,
    "bbox_crop_factor": 3.0,
    "denoise": 0.4,
    "feather": 5,
    "noise_mask": True,
    "force_inpaint": True,
    "drop_size": 10,
    "cycle": 1,
}

_TILE_REPAIR_DEFAULTS = {
    "mask_mode": "Individual",
    "tile_width": 512,
    "tile_height": 512,
    "tile_padding": 64,
    "steps": 28,
    "cfg": 7.0,
    "sampler_name": "euler",
    "scheduler": "normal",
    "denoise": 0.4,
    "noise_multiplier": 1.0,
    "fill_mode": "original",
    "grow_mask_by": 6,
    "seed": 0,
    "inpaint_prompt": "",
    "negative_prompt": "",
    "controlnet_enable": False,
    "controlnet_model_name": "None",
    "controlnet_module": "tile_resample",
    "controlnet_override_external": False,
    "controlnet_strength": 1.0,
    "controlnet_start": 0.0,
    "controlnet_end": 1.0,
    "controlnet_processor_resolution": 512,
    "controlnet_settings": _CONTROLNET_EXTRA_DEFAULTS,
    "restore_face": False,
    "restore_face_settings": _RESTORE_FACE_DEFAULTS,
}

# These are the controls exposed by Forge's *actual* Anima Tile-Repair/PiD
# panel.  The implementation there runs the vendored Qwen/Anima DiT stack and
# ControlNet-LLLite (or PiD), not an ordinary Comfy VAE inpaint.  Accepting any
# of these keys in the compatibility node would silently promise a pipeline we
# cannot execute with MODEL/CLIP/VAE alone, so reject them with a precise error.
_FORGE_TILE_ONLY_KEYS = {
    "lllite_model",
    "dit_override",
    "te_override",
    "vae_override",
    "lora_slots",
    "positive",
    "negative",
    "flow_shift",
    "width",
    "height",
    "lllite_strength",
    "lllite_start",
    "lllite_end",
    "lllite_multiplier",
    "unload_forge_before",
    "insert_mode",
    "restore_mode",
    "pid_checkpoint",
    "pid_scale",
    "pid_steps",
    "pid_degrade",
}


# Lazy torch import shared by the whole pack (compat.require_torch).
from .compat import require_torch as _torch  # noqa: E402


def _fallback_samplers() -> list[str]:
    return ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"]


def _fallback_schedulers() -> list[str]:
    return ["normal", "karras", "exponential", "sgm_uniform", "simple"]


def _sampler_choices() -> tuple[list[str], list[str]]:
    try:
        import comfy.samplers

        return list(comfy.samplers.KSampler.SAMPLERS), list(comfy.samplers.KSampler.SCHEDULERS)
    except Exception:
        return _fallback_samplers(), _fallback_schedulers()


def _node_mappings() -> dict[str, Any]:
    """The Easy SAM3 adapter seam.  Discovery happens only at execution."""

    try:
        import nodes
    except ImportError as exc:
        raise RuntimeError(
            "ComfyUI's node registry is unavailable. Run this node inside ComfyUI."
        ) from exc
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise RuntimeError("ComfyUI NODE_CLASS_MAPPINGS is unavailable or invalid.")
    return mappings


def _resolve_easy_node(candidates: Sequence[str], purpose: str):
    mappings = _node_mappings()
    for key in candidates:
        if key in mappings:
            return mappings[key]
    folded = {str(key).casefold(): value for key, value in mappings.items()}
    for key in candidates:
        if key.casefold() in folded:
            return folded[key.casefold()]
    raise RuntimeError(
        f"Easy SAM3 {purpose} provider is not installed/loaded. Install and enable "
        "ComfyUI-Easy-SAM3 (expected node ids: " + ", ".join(candidates) + ")."
    )


def _invoke_node(node_type: Any, function_name: str, **kwargs):
    function = getattr(node_type, function_name, None)
    if function is None:
        instance = node_type()
        function = getattr(instance, function_name)
    else:
        parameters = list(inspect.signature(function).parameters.values())
        if parameters and parameters[0].name in {"self", "s"}:
            function = getattr(node_type(), function_name)
    return function(**kwargs)


def _result_tuple(value: Any) -> tuple[Any, ...]:
    if hasattr(value, "result"):
        value = value.result
    elif isinstance(value, dict) and "result" in value:
        value = value["result"]
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _resolve_sam3_request(checkpoint: str, device: str, precision: str):
    resolved_device = str(device or "cuda").lower()
    if resolved_device == "auto":
        torch = _torch()
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device not in {"cuda", "cpu", "mps"}:
        raise ValueError("device must be auto, cuda, cpu, or mps")
    resolved_precision = str(precision or "fp32").lower()
    if resolved_precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if resolved_device == "cpu" and resolved_precision != "fp32":
        raise ValueError("Easy SAM3 does not support fp16/bf16 on CPU")
    checkpoint_value = str(checkpoint or "sam3.pt").strip()
    checkpoint_path = Path(checkpoint_value).expanduser()
    if checkpoint_path.is_absolute():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
        model_name = checkpoint_path.name
    else:
        model_name = checkpoint_value.replace("\\", "/")
    return resolved_device, resolved_precision, checkpoint_path, model_name


# One CPU/GPU-resident Easy SAM3 bundle (~3.3-3.5 GB), reused across nodes
# and prompts like the Forge extension's _BUNDLE.  Keyed by the checkpoint
# file identity so a replaced sam3.pt is never served from memory.
_SAM3_CACHE_LOCK = threading.Lock()
_SAM3_CACHE: dict[str, Any] = {}


def _sam3_checkpoint_file(checkpoint_path: Path, model_name: str) -> Path | None:
    if checkpoint_path.is_absolute():
        return checkpoint_path
    try:
        import folder_paths

        full_path = folder_paths.get_full_path("sam3", model_name)
    except Exception:
        return None
    return Path(full_path) if full_path else None


def _sam3_cache_key(checkpoint_file: Path | None, device: str, precision: str):
    """(resolved path, mtime, size, device, precision), or None if unknowable."""

    if checkpoint_file is None:
        return None
    try:
        resolved = checkpoint_file.resolve()
        stat = resolved.stat()
    except OSError:
        return None
    return (
        os.path.normcase(str(resolved)), int(stat.st_mtime_ns), int(stat.st_size),
        str(device), str(precision),
    )


def _sam3_home_device(bundle: dict[str, Any], fallback: str):
    model = bundle.get("model")
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(iter(parameters())).device
        except (StopIteration, TypeError, RuntimeError, AttributeError):
            pass
    return fallback


def _replace_sam3_cache(entry: dict[str, Any] | None, *, in_use: Any = None) -> bool:
    """Swap the cache slot; offload the evicted bundle unless it is in use.

    A provider may hand back the very same bundle/model object again (its
    own cache), so the evicted one is only moved off the device when it is
    not the bundle the current node is about to use.  Returns whether a
    different bundle was evicted (its memory is now releasable).
    """

    with _SAM3_CACHE_LOCK:
        previous = _SAM3_CACHE.get("bundle")
        _SAM3_CACHE.clear()
        if entry:
            _SAM3_CACHE.update(entry)
    if previous is None or previous is in_use:
        return False
    if isinstance(in_use, dict) and previous.get("model") is in_use.get("model"):
        return False
    try:
        # Moving it off the GPU first frees its VRAM even if something else
        # (e.g. an exception traceback) still references the bundle.
        _offload_sam3(previous)
    except Exception:
        pass
    return True


def clear_sam3_cache() -> bool:
    """Drop the cached bundle (moving it off the GPU first when it is there).

    Returns whether a bundle was released.  Called on ComfyUI "unload all
    models" (see :func:`install_unload_release_hook`), when ``cache_model``
    is off, and when a cache hit fails to move the bundle back.
    """

    released = _replace_sam3_cache(None)
    if released:
        # Return the ~3.4 GB CPU copy now rather than at some later GC pass.
        gc.collect()
    return released


_UNLOAD_HOOK_MARK = "_aistudio_releases_sam3_cache"


def install_unload_release_hook(management: Any = None) -> bool:
    """Make ComfyUI's ``unload_all_models`` also release the SAM3 cache.

    The cached bundle is not a Comfy-managed model, so ``/free`` (the app's
    "unload after generation"), OOM recovery and ``--disable-smart-memory``
    would otherwise leave ~3.4 GB in RAM (or VRAM with ``unload_after=False``)
    for the process lifetime.  Those callers all run on the prompt worker
    thread, never while a SAM3 node is using the bundle.  Idempotent; returns
    whether the hook was installed by this call.
    """

    if management is None:
        try:
            import comfy.model_management as management
        except Exception:
            return False  # standalone import (tests, the app's relight action)
    original = getattr(management, "unload_all_models", None)
    if not callable(original) or getattr(original, _UNLOAD_HOOK_MARK, False):
        return False

    @functools.wraps(original)
    def unload_all_models(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            try:
                clear_sam3_cache()
            except Exception:
                logging.getLogger(__name__).warning(
                    "AI Studio SAM3 cache release failed", exc_info=True,
                )

    setattr(unload_all_models, _UNLOAD_HOOK_MARK, True)
    management.unload_all_models = unload_all_models
    return True


def _acquire_sam3(checkpoint: str, device: str, precision: str, *, use_cache: bool = True):
    """Return ``(bundle, device, model_name, cache_state)`` for one Mask node.

    ``cache_state`` is ``hit`` (reused, moved back to its device), ``miss``
    (loaded and now cached), or ``disabled`` (loaded for this node only; any
    cached bundle is released so ``cache_model=False`` frees its RAM).
    """

    resolved_device, resolved_precision, checkpoint_path, model_name = _resolve_sam3_request(
        checkpoint, device, precision,
    )
    key = _sam3_cache_key(
        _sam3_checkpoint_file(checkpoint_path, model_name),
        resolved_device, resolved_precision,
    ) if use_cache else None
    with _SAM3_CACHE_LOCK:
        cached = dict(_SAM3_CACHE) if key is not None and _SAM3_CACHE.get("key") == key else None
    if cached is not None:
        bundle = cached["bundle"]
        model = bundle.get("model")
        if model is not None and hasattr(model, "to"):
            try:
                model.to(cached["home"])
            except BaseException:
                # A half-moved bundle (e.g. CUDA OOM midway) cannot be trusted:
                # drop it like Forge's release_sam3 so the next run reloads
                # cleanly instead of reusing a split CPU/GPU model.
                clear_sam3_cache()
                raise
        return bundle, resolved_device, model_name, "hit"
    bundle = _invoke_sam3_loader(
        checkpoint_path, model_name, resolved_device, resolved_precision,
    )
    if key is None:
        _replace_sam3_cache(None, in_use=bundle)
        return bundle, resolved_device, model_name, "disabled"
    _replace_sam3_cache({
        "key": key,
        "bundle": bundle,
        "home": _sam3_home_device(bundle, resolved_device),
    }, in_use=bundle)
    return bundle, resolved_device, model_name, "miss"


def _invoke_sam3_loader(checkpoint_path: Path, model_name: str,
                        resolved_device: str, resolved_precision: str):
    loader = _resolve_easy_node(_EASY_LOADER_KEYS, "model loader")
    restore_registry = None
    if checkpoint_path.is_absolute():
        try:
            import folder_paths

            previous = folder_paths.folder_names_and_paths.get("sam3")
            if previous is None:
                raise RuntimeError("Easy SAM3 did not register its sam3 model folder.")
            paths, extensions = previous
            restore_registry = previous
            folder_paths.folder_names_and_paths["sam3"] = (
                [str(checkpoint_path.parent), *[str(path) for path in paths]],
                extensions,
            )
        except ImportError as exc:
            raise RuntimeError("Absolute SAM3 checkpoints require ComfyUI folder_paths.") from exc
    try:
        output = _invoke_node(
            loader,
            "execute",
            model=model_name,
            segmentor="image",
            device=resolved_device,
            precision=resolved_precision,
        )
    finally:
        if restore_registry is not None:
            import folder_paths

            folder_paths.folder_names_and_paths["sam3"] = restore_registry
    result = _result_tuple(output)
    if not result or not isinstance(result[0], dict):
        raise RuntimeError("Easy SAM3 loader returned an invalid model bundle.")
    return result[0]


def _offload_sam3(bundle: Any) -> None:
    if not isinstance(bundle, dict):
        return
    model = bundle.get("model")
    if model is None or not hasattr(model, "to"):
        return
    try:
        import comfy.model_management as management

        model.to(management.unet_offload_device())
        management.soft_empty_cache()
    except Exception:
        # A standalone/fake provider has no Comfy model manager. CPU remains a
        # deterministic, explicit offload target rather than a silent no-op.
        model.to("cpu")


def _as_python(value: Any) -> Any:
    torch = _torch()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _flatten_boxes(value: Any) -> list[list[float]]:
    value = _as_python(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [[float(item) for item in value]]
        result: list[list[float]] = []
        for item in value:
            result.extend(_flatten_boxes(item))
        return result
    return []


def _flatten_scores(value: Any) -> list[float]:
    value = _as_python(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten_scores(item))
        return result
    return [float(value)] if isinstance(value, (int, float)) else []


def _detect_token(provider: Any, model: Any, image: Any, token: str,
                  threshold: float, detection_limit: int):
    image_value = ensure_image(image)
    masks = []
    boxes: list[list[float]] = []
    scores: list[float] = []

    # Easy-SAM3's current image node stacks per-frame raw masks/boxes.  That
    # raises when two images contain different object counts, which is the
    # normal case for an IMAGE batch.  Invoke it with B=1 and concatenate only
    # its stable combined-mask output ourselves.
    function_name = getattr(provider, "FUNCTION", "execute")
    # New-style Comfy nodes advertise EXECUTE_NORMALIZED but keep their public
    # classmethod ``execute``. Prefer the latter because it accepts ordinary
    # Python values and returns io.NodeOutput directly.
    if hasattr(provider, "execute"):
        function_name = "execute"
    for batch_index in range(image_value.shape[0]):
        output = _invoke_node(
            provider,
            function_name,
            sam3_model=model,
            images=image_value[batch_index:batch_index + 1],
            prompt=token,
            threshold=float(threshold),
            keep_model_loaded=True,
            add_background="none",
            detection_limit=int(detection_limit),
            coordinates_positive=None,
            coordinates_negative=None,
            bboxes=None,
            mask=None,
        )
        values = _result_tuple(output)
        if len(values) < 1:
            raise RuntimeError(f"Easy SAM3 returned no mask output for prompt {token!r}.")
        detected = ensure_mask(
            values[0], height=image_value.shape[1], width=image_value.shape[2], batch=1,
        ).to(image_value.device)
        masks.append(detected)
        # Easy-SAM3 represents no detection as a zero mask plus a synthetic
        # [0,0,0,0] box/0 score. Do not expose or draw that sentinel.
        if bool((detected > 0.0).any()):
            boxes.extend(_flatten_boxes(values[3] if len(values) > 3 else None))
            scores.extend(_flatten_scores(values[4] if len(values) > 4 else None))
    return _torch().cat(masks, dim=0), boxes, scores


def _default_artifact_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / "sam3"
    except Exception:
        return Path.cwd() / "output" / "sam3"


class ForgeNeoSAM3Mask:
    """Deep mask module matching Forge's grouped SAM3 detection semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "face", "multiline": True}),
                "exclude_prompt": ("STRING", {"default": "", "multiline": True}),
                "mask_mode": (["Combined", "Individual"], {"default": "Individual"}),
                "mask_source": (
                    ["generated", "manual", "intersection", "union"],
                    {"default": "generated"},
                ),
                "threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detection_limit": ("INT", {"default": -1, "min": -1, "max": 1000}),
                "convex_hull": ("BOOLEAN", {"default": False}),
                "mask_dilation": ("INT", {"default": 0, "min": 0, "max": 512}),
                "mask_outline_px": ("INT", {"default": 0, "min": 0, "max": 512}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 256}),
                "invert": ("BOOLEAN", {"default": False}),
                "checkpoint": ("STRING", {"default": "sam3.pt"}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "cuda"}),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
                "unload_after": ("BOOLEAN", {"default": True}),
                "save_artifacts": ("BOOLEAN", {"default": True}),
                "artifact_directory": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xFFFFFFFFFFFFFFFF}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sam3_model": ("EASY_SAM3_MODEL",),
                "manual_mask": ("MASK",),
                # Keep the node-loaded bundle between runs (CPU RAM when
                # unload_after, on the device otherwise). False frees it.
                # The app compiler sets it from its "keep SAM3 in RAM"
                # setting; ComfyUI's unload-all-models (/free) also frees it.
                "cache_model": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE", "BBOX", "FLOAT", "STRING")
    RETURN_NAMES = (
        "selected_mask", "combined_mask", "individual_masks", "overlay",
        "boxes", "scores", "artifacts_json",
    )
    OUTPUT_IS_LIST = (False, False, False, False, False, True, False)
    FUNCTION = "segment"
    CATEGORY = CATEGORY

    def segment(
        self,
        image,
        prompt="face",
        exclude_prompt="",
        mask_mode="Individual",
        mask_source="generated",
        threshold=0.4,
        detection_limit=-1,
        convex_hull=False,
        mask_dilation=0,
        mask_outline_px=0,
        mask_blur=4,
        invert=False,
        checkpoint="sam3.pt",
        device="cuda",
        precision="fp32",
        unload_after=True,
        save_artifacts=True,
        artifact_directory="",
        seed=-1,
        enabled=True,
        sam3_model=None,
        manual_mask=None,
        cache_model=True,
    ):
        image_value = ensure_image(image)
        mode = str(mask_mode).strip().casefold()
        if mode not in {"combined", "individual"}:
            raise ValueError("mask_mode must be Combined or Individual")

        if not enabled:
            empty = empty_mask_like(image_value)
            overlay = make_overlay(image_value, empty)
            report = json.dumps({"enabled": False, "status": "disabled"})
            return empty, empty, empty, overlay, [], [], report

        source = str(mask_source or "generated").strip().casefold()
        if source not in {"generated", "manual", "intersection", "union"}:
            raise ValueError(
                "mask_source must be generated, manual, intersection, or union"
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if int(detection_limit) < -1:
            raise ValueError("detection_limit must be -1 or greater")
        needs_main_detection = source != "manual"
        groups_spec = split_prompt_groups(prompt)
        exclusion_groups = split_prompt_groups(exclude_prompt)
        needs_provider = needs_main_detection or bool(exclusion_groups)
        if needs_main_detection and not groups_spec:
            raise ValueError("A non-empty SAM3 prompt is required for generated masks.")

        loaded_here = False
        cache_state = "not_loaded"
        resolved_device = str(device)
        resolved_checkpoint = str(checkpoint)
        model_bundle = sam3_model
        if needs_provider and model_bundle is None:
            model_bundle, resolved_device, resolved_checkpoint, cache_state = _acquire_sam3(
                checkpoint, device, precision, use_cache=bool(cache_model),
            )
            loaded_here = True
        elif isinstance(model_bundle, dict):
            resolved_device = str(model_bundle.get("device", "external"))
            resolved_checkpoint = "externally supplied EASY_SAM3_MODEL"
            cache_state = "external"

        provider = None
        generated: list[Any] = []
        all_boxes: list[Any] = []
        all_scores: list[Any] = []
        try:
            if needs_provider:
                provider = _resolve_easy_node(_EASY_SEGMENTATION_KEYS, "image segmentation")
            if needs_main_detection:
                for tokens in groups_spec:
                    token_masks = []
                    for token in tokens:
                        detected, boxes, scores = _detect_token(
                            provider, model_bundle, image_value, token,
                            float(threshold), int(detection_limit),
                        )
                        token_masks.append(detected)
                        all_boxes.extend(boxes)
                        all_scores.extend(scores)
                    group = union_masks(token_masks, reference=image_value)
                    if bool((group > 0.0).any()):
                        generated.append(refine_generated_mask(
                            group,
                            image_value,
                            use_convex_hull=bool(convex_hull),
                            outline_pixels=int(mask_outline_px),
                            dilation_pixels=int(mask_dilation),
                        ))

            selected_groups = select_mask_groups(
                generated, manual_mask, source, reference=image_value
            )
            # Forge applies the hand-drawn intersection before subtracting
            # protected objects. This prevents the safe manual fallback from
            # accidentally restoring an excluded face/hand region.
            if exclusion_groups and selected_groups:
                exclusion_masks = []
                for tokens in exclusion_groups:
                    for token in tokens:
                        detected, _, _ = _detect_token(
                            provider, model_bundle, image_value, token,
                            float(threshold), int(detection_limit),
                        )
                        exclusion_masks.append(detected)
                exclusion = union_masks(exclusion_masks, reference=image_value)
                selected_groups = subtract_exclusion(selected_groups, exclusion)
            combined, individuals = finish_masks(
                selected_groups, image_value,
                blur_pixels=int(mask_blur), invert=bool(invert),
            )
        finally:
            # Only a bundle this node loaded is offloaded. An externally
            # connected EASY_SAM3_MODEL is owned (and cached) by its loader
            # node; moving it to CPU in place would hand the next run a CPU
            # model still labelled device='cuda'.
            if needs_provider and loaded_here and unload_after and model_bundle is not None:
                _offload_sam3(model_bundle)

        selected = combined if mode == "combined" else individuals
        overlay = make_overlay(image_value, combined, all_boxes, all_scores)
        report: dict[str, Any] = {
            "enabled": True,
            "status": "ok" if bool((combined > 0.0).any()) else "no_detection",
            "mask_mode": "Combined" if mode == "combined" else "Individual",
            "mask_source": source,
            "prompt_groups": groups_spec,
            "exclude_groups": exclusion_groups,
            "mask_count": len(selected_groups),
            "checkpoint": resolved_checkpoint,
            "device": resolved_device,
            "model_loaded_by_node": loaded_here,
            "model_cache": cache_state,
            "unloaded": bool(unload_after and loaded_here),
            "boxes": all_boxes,
            "scores": all_scores,
        }
        if save_artifacts:
            target = Path(artifact_directory) if str(artifact_directory).strip() else _default_artifact_directory()
            report["artifacts"] = save_mask_artifacts(
                target,
                combined=combined,
                individuals=individuals,
                overlay=overlay,
                prompt=str(prompt),
                seed=int(seed),
                metadata=report,
            )
        return selected, combined, individuals, overlay, all_boxes, all_scores, json.dumps(
            report, ensure_ascii=False
        )


def _encode_prompt(clip: Any, text: str):
    mappings = _node_mappings()
    encoder = mappings.get("CLIPTextEncode")
    if encoder is None:
        raise RuntimeError("ComfyUI CLIPTextEncode is unavailable for prompt override.")
    output = _invoke_node(encoder, "encode", clip=clip, text=text)
    values = _result_tuple(output)
    if not values:
        raise RuntimeError("CLIPTextEncode returned no conditioning.")
    return values[0]


def _forge_fill_pixels(pixels: Any, mask: Any):
    """Port Forge's blur-fill preprocessing for masked content.

    Forge applies this fill before VAE encoding for every mode except
    ``original``.  Keeping it in pixel space also makes ``fill`` observably
    distinct without relying on Comfy's VAEEncodeForInpaint, which always
    replaces the masked pixels with neutral gray.
    """

    try:
        import numpy as np
        from PIL import Image, ImageFilter, ImageOps
    except ImportError as exc:  # pragma: no cover - declared dependencies
        raise RuntimeError("Forge-compatible inpaint fill requires Pillow and NumPy.") from exc

    torch = _torch()
    image_value = ensure_image(pixels)[..., :3]
    mask_value = ensure_mask(
        mask,
        height=image_value.shape[1],
        width=image_value.shape[2],
        batch=image_value.shape[0],
    )
    output = []
    for image_item, mask_item in zip(image_value, mask_value):
        rgb = (image_item.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        alpha = (mask_item.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        pil_image = Image.fromarray(rgb, mode="RGB")
        pil_mask = Image.fromarray(alpha, mode="L")
        image_mod = Image.new("RGBA", pil_image.size)
        image_masked = Image.new("RGBa", pil_image.size)
        image_masked.paste(
            pil_image.convert("RGBA").convert("RGBa"),
            mask=ImageOps.invert(pil_mask),
        )
        image_masked = image_masked.convert("RGBa")
        for radius, repeats in ((256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)):
            blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert("RGBA")
            for _ in range(repeats):
                image_mod.alpha_composite(blurred)
        filled = np.asarray(image_mod.convert("RGB"), dtype=np.float32) / 255.0
        output.append(torch.from_numpy(filled))
    return torch.stack(output, dim=0).to(
        device=image_value.device, dtype=image_value.dtype
    )


def _grown_noise_mask(mask: Any, grow_mask_by: int):
    """Device-safe equivalent of Comfy's VAEEncodeForInpaint mask growth."""

    torch = _torch()
    import torch.nn.functional as functional

    value = ensure_mask(mask).round().unsqueeze(1)
    grow = int(grow_mask_by)
    if grow < 0:
        raise ValueError("grow_mask_by cannot be negative")
    if grow == 0:
        return value
    kernel = torch.ones((1, 1, grow, grow), device=value.device, dtype=value.dtype)
    padding = (grow - 1 + 1) // 2
    grown = functional.conv2d(value, kernel, padding=padding).clamp(0.0, 1.0)
    # Even-sized kernels produce one extra row/column with symmetric padding,
    # matching Comfy's implementation before its final spatial slice.
    return grown[..., : value.shape[-2], : value.shape[-1]]


def _vae_encode_for_inpaint(
    vae: Any,
    pixels: Any,
    mask: Any,
    grow_mask_by: int,
    *,
    fill_mode: str = "original",
    seed: int = 0,
):
    """Encode pixels while preserving Forge's four masked-content modes."""

    torch = _torch()
    image_value = ensure_image(pixels)[..., :3]
    mask_value = ensure_mask(
        mask,
        height=image_value.shape[1],
        width=image_value.shape[2],
        batch=image_value.shape[0],
    ).to(image_value.device)
    mode = str(fill_mode or "original").strip().casefold()
    if mode not in {"fill", "original", "latent noise", "latent nothing"}:
        raise ValueError("fill_mode must be fill, original, latent noise, or latent nothing")

    encode_pixels = image_value if mode == "original" else _forge_fill_pixels(
        image_value, mask_value
    )
    if vae is None or not hasattr(vae, "encode"):
        raise RuntimeError("A Comfy VAE with an encode() method is required for SAM3 inpaint.")
    samples = vae.encode(encode_pixels)
    if not torch.is_tensor(samples):
        raise RuntimeError("Comfy VAE encode() returned an invalid latent tensor.")

    noise_mask = _grown_noise_mask(mask_value, int(grow_mask_by))
    latent_mask = torch.nn.functional.interpolate(
        mask_value.unsqueeze(1),
        size=samples.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).round().to(device=samples.device, dtype=samples.dtype)
    if mode == "latent nothing":
        samples = samples * (1.0 - latent_mask)
    elif mode == "latent noise":
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) & 0xFFFFFFFFFFFFFFFF
        )
        noise = torch.randn(samples.shape, generator=generator, dtype=torch.float32).to(
            device=samples.device, dtype=samples.dtype
        )
        samples = samples * (1.0 - latent_mask) + noise * latent_mask
    return {"samples": samples, "noise_mask": noise_mask}


def _sample_latent(model: Any, seed: int, steps: int, cfg: float,
                   sampler_name: str, scheduler: str, positive: Any,
                   negative: Any, latent: dict[str, Any], denoise: float,
                   noise_multiplier: float):
    try:
        import nodes
    except ImportError as exc:
        raise RuntimeError("ComfyUI sampler implementation is unavailable.") from exc
    multiplier = float(noise_multiplier)
    if multiplier < 0.0:
        raise ValueError("noise_multiplier cannot be negative")
    if abs(multiplier - 1.0) < 1e-9:
        return nodes.common_ksampler(
            model, int(seed), int(steps), float(cfg), sampler_name, scheduler,
            positive, negative, latent, denoise=float(denoise),
        )[0]

    # Comfy's common_ksampler fixes model-specific latent channels and creates
    # deterministic noise.  Scaling that exact noise is the Forge multiplier
    # semantic; duplicating this short seam avoids a global sampler patch.
    try:
        import comfy.sample
        import comfy.utils
        import latent_preview
    except ImportError as exc:
        raise RuntimeError("Custom noise multiplier requires ComfyUI sampling modules.") from exc
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    batch_indices = latent.get("batch_index")
    noise = comfy.sample.prepare_noise(latent_image, int(seed), batch_indices) * multiplier
    callback = latent_preview.prepare_callback(model, int(steps))
    sampled = comfy.sample.sample(
        model, noise, int(steps), float(cfg), sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=float(denoise), disable_noise=False, start_step=None,
        last_step=None, force_full_denoise=False,
        noise_mask=latent.get("noise_mask"), callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed),
    )
    output = dict(latent)
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = sampled
    return output


def _vae_decode(vae: Any, latent: Any):
    mappings = _node_mappings()
    decoder = mappings.get("VAEDecode")
    if decoder is None:
        try:
            import nodes

            decoder = nodes.VAEDecode
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("ComfyUI VAEDecode is unavailable.") from exc
    output = _invoke_node(decoder, "decode", vae=vae, samples=latent)
    values = _result_tuple(output)
    if not values:
        raise RuntimeError("VAEDecode returned no image.")
    return ensure_image(values[0])


def _settings_object(raw: str, defaults: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must contain a JSON object")
    unknown = sorted(set(parsed) - set(defaults))
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(unknown)}")
    result = dict(defaults)
    result.update(parsed)
    return result


def _load_controlnet(control_net: Any, model_name: str):
    if control_net is not None:
        return control_net, "connected CONTROL_NET"
    requested = str(model_name or "").strip()
    if not requested or requested.casefold() in {"none", "null"}:
        raise ValueError(
            "ControlNet is enabled but neither control_net nor controlnet_model_name was provided."
        )
    mappings = _node_mappings()
    loader = mappings.get("ControlNetLoader")
    if loader is None:
        raise RuntimeError("ComfyUI ControlNetLoader is unavailable.")

    path = Path(requested).expanduser()
    restore_registry = None
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(f"ControlNet model not found: {path}")
        try:
            import folder_paths
        except ImportError as exc:
            raise RuntimeError("Absolute ControlNet paths require ComfyUI folder_paths.") from exc
        previous = folder_paths.folder_names_and_paths.get("controlnet")
        if previous is None:
            raise RuntimeError("ComfyUI did not register its ControlNet model folder.")
        paths, extensions = previous
        restore_registry = previous
        folder_paths.folder_names_and_paths["controlnet"] = (
            [str(path.parent), *[str(item) for item in paths]], extensions
        )
        requested = path.name
    try:
        output = _invoke_node(loader, "load_controlnet", control_net_name=requested)
    finally:
        if restore_registry is not None:
            import folder_paths

            folder_paths.folder_names_and_paths["controlnet"] = restore_registry
    values = _result_tuple(output)
    if not values or values[0] is None:
        raise RuntimeError(f"ControlNetLoader returned no model for {model_name!r}.")
    return values[0], requested


_CONTROL_RESIZE_MODES = frozenset({"Just Resize", "Crop and Resize", "Resize and Fill"})


def _effective_control_resize_mode(resize_mode: Any, *, explicit_control_image: bool) -> str:
    """ControlNet 힌트를 sample 크기에 맞출 때 실제로 쓸 resize mode.

    힌트를 입력(샘플링할) 이미지에서 뽑으면 잠재 입력과 같은 기하여야 한다. 잠재 입력은
    region 을 sample 크기로 '늘려'(Just Resize) 만들므로 — 전체 이미지 + 다른 종횡비 요청이면
    실제로 찌그러진다 — 힌트도 Just Resize 로 맞춘다. 설정값(기본 Crop and Resize)을 쓰면
    힌트만 가운데가 잘려 잠재 입력과 어긋났다. Forge ControlNet(get_input_data)도 유닛
    이미지가 없으면 unit.resize_mode 대신 p.resize_mode(ADetailer/SAM3 i2i 는 0 = Just Resize)를
    쓴다. 따로 연결한 control_image 는 Forge 유닛 이미지처럼 설정값을 따르되, 입력 크기로
    미리 늘리지 않는다: 전체 이미지 패스는 원본 control 을 이 모드로 sample 크기에 맞추고
    (controlnet.py crop_and_resize_image), only-masked 패스는 이 모드로 입력 이미지 크기에
    맞춘 control 에서 크롭을 잘라 잠재 입력과 같이 늘린다(try_crop_image_with_a1111_mask).
    설정값은 두 경우 모두 검증한다(입력 힌트에서 무시되더라도 잘못된 값을 조용히 넘기지 않는다).
    """

    mode = str(resize_mode)
    if mode not in _CONTROL_RESIZE_MODES:
        raise ValueError(
            f"controlnet resize_mode must be one of {sorted(_CONTROL_RESIZE_MODES)}, got {mode!r}"
        )
    return mode if explicit_control_image else "Just Resize"


# Resize and Fill 의 띠(맞춘 control 이 덮지 않는 칸)를 채우는 방식 — 흉내 내는 Forge 함수마다 다르다.
_CONTROL_FILL_MODES = frozenset({"median", "edge"})


def _resize_and_fill_box(source_height: int, source_width: int,
                         target_height: int, target_width: int) -> tuple[int, int, int, int]:
    """Resize and Fill 로 맞춘 원본이 캔버스에서 차지하는 칸 ``(top, left, height, width)``.

    비율을 지켜 캔버스 안에 들어가게 줄이고(반올림) 가운데 둔다(Forge ControlNet
    crop_and_resize_image 의 OUTER_FIT 과 같은 기하). 나머지 칸이 띠다.
    """
    scale = min(target_height / source_height, target_width / source_width)
    height = max(1, min(int(target_height), int(round(source_height * scale))))
    width = max(1, min(int(target_width), int(round(source_width * scale))))
    return (int(target_height) - height) // 2, (int(target_width) - width) // 2, height, width


def _border_median(image: Any):
    """Forge crop_and_resize_image 의 띠 색: 맨 위·아래 행과 맨 왼·오른 열 픽셀의 채널별 중앙값.

    ``np.median`` 처럼 짝수 개면 가운데 두 값의 평균이다(``torch.median`` 은 아래 값을 준다).
    반환 모양은 ``(batch, channels)``.
    """
    torch = _torch()
    value = ensure_image(image)
    borders = torch.cat(
        (value[:, 0, :, :], value[:, -1, :, :], value[:, :, 0, :], value[:, :, -1, :]),
        dim=1,
    )
    return torch.quantile(borders.float(), 0.5, dim=1).to(dtype=value.dtype)


def _resize_and_fill_band_mask(source: Any, target_height: int, target_width: int,
                               padding: tuple[int, int, int, int] = (0, 0, 0, 0)):
    """Resize and Fill 로 맞춘 ``source`` 가 덮지 않는 띠 = 1 인 MASK ``(B, H, W)``.

    Forge inpaint 전처리기(expand_mask_when_resize_and_fill ·
    fill_mask_with_one_when_resize_and_fill)는 유닛 마스크를 띠를 255 로 채워 맞춘다 — 띠는
    새로 그릴 칸(inpaint 힌트 -1)이 된다. ``padding`` 은 힌트와 같은 VAE 패딩(가장자리 복제).
    """
    import torch.nn.functional as functional

    torch = _torch()
    value = ensure_image(source)
    top, left, height, width = _resize_and_fill_box(
        value.shape[1], value.shape[2], target_height, target_width,
    )
    mask = torch.ones(
        (value.shape[0], int(target_height), int(target_width)),
        device=value.device, dtype=value.dtype,
    )
    mask[:, top:top + height, left:left + width] = 0.0
    if any(padding):
        mask = functional.pad(mask.unsqueeze(1), padding, mode="replicate").squeeze(1)
    return mask


def _fit_control_image(image: Any, target_height: int, target_width: int,
                       resize_mode: str, *, fill: str = "median"):
    """control 이미지를 ``resize_mode`` 로 target 크기에 맞춘다.

    Resize and Fill 의 띠는 흉내 내는 Forge 함수대로 채운다 — 검정으로 채우는 경로는 Forge 에 없다.

    * ``fill="median"`` — ControlNet crop_and_resize_image(OUTER_FIT): 원본 테두리 픽셀의
      채널별 중앙값. 힌트를 sample(p.width/p.height) 크기로 맞출 때 쓴다.
    * ``fill="edge"`` — A1111 images.resize_image(2): 맞춘 이미지의 가장자리 행/열을 띠 끝까지
      늘린다. only-masked 의 try_crop_image_with_a1111_mask 가 유닛 이미지를 입력 이미지 크기로
      맞출 때 쓴다.
    """
    import torch.nn.functional as functional

    mode = str(resize_mode)
    valid = _CONTROL_RESIZE_MODES
    if mode not in valid:
        raise ValueError(f"controlnet resize_mode must be one of {sorted(valid)}, got {mode!r}")
    if fill not in _CONTROL_FILL_MODES:
        raise ValueError(
            f"controlnet fill must be one of {sorted(_CONTROL_FILL_MODES)}, got {fill!r}"
        )
    source = ensure_image(image)[..., :3]
    if mode == "Just Resize" or tuple(source.shape[1:3]) == (target_height, target_width):
        return _resize_image(source, target_height, target_width)
    source_height, source_width = source.shape[1:3]
    if mode == "Crop and Resize":
        scale = max(target_height / source_height, target_width / source_width)
        scaled_height = max(1, int(round(source_height * scale)))
        scaled_width = max(1, int(round(source_width * scale)))
        resized = _resize_image(source, scaled_height, scaled_width)
        y = max(0, (scaled_height - target_height) // 2)
        x = max(0, (scaled_width - target_width) // 2)
        return resized[:, y:y + target_height, x:x + target_width, :]
    top, left, height, width = _resize_and_fill_box(
        source_height, source_width, target_height, target_width,
    )
    resized = _resize_image(source, height, width)
    bottom = int(target_height) - height - top
    right = int(target_width) - width - left
    if fill == "edge":
        return functional.pad(
            resized.permute(0, 3, 1, 2), (left, right, top, bottom), mode="replicate",
        ).permute(0, 2, 3, 1)
    color = _border_median(source).to(device=resized.device, dtype=resized.dtype)
    canvas = color[:, None, None, :].expand(
        -1, int(target_height), int(target_width), -1,
    ).clone()
    canvas[:, top:top + height, left:left + width, :] = resized
    return canvas


def _pixel_perfect_resolution(source: Any, target_height: int, target_width: int,
                              resize_mode: str) -> int:
    image = ensure_image(source)
    raw_height, raw_width = image.shape[1:3]
    height_scale = float(target_height) / float(raw_height)
    width_scale = float(target_width) / float(raw_width)
    scale = min(height_scale, width_scale) if resize_mode == "Resize and Fill" else max(height_scale, width_scale)
    return max(64, int(round(scale * min(raw_height, raw_width))))


def _inpaint_control_hint(image: Any, mask: Any):
    import torch.nn.functional as functional

    value = ensure_image(image)[..., :3].clone()
    mask_value = ensure_mask(mask)
    mask_value = functional.interpolate(
        mask_value.unsqueeze(1), size=value.shape[1:3], mode="bilinear",
        align_corners=False,
    ).movedim(1, -1).expand(-1, -1, -1, 3).to(value.device)
    value[mask_value > 0.5] = -1.0
    return value


def _preprocessor_kwargs(node_type: Any, image: Any, mask: Any, resolution: int,
                         threshold_a: float, threshold_b: float):
    schema = node_type.INPUT_TYPES()
    required = dict(schema.get("required", {}))
    optional = dict(schema.get("optional", {}))
    parameters: dict[str, Any] = {}
    consumed_a = consumed_b = False
    low_names = {"low_threshold", "threshold_a", "thr_a", "value_threshold"}
    high_names = {"high_threshold", "threshold_b", "thr_b", "distance_threshold"}
    for name, descriptor in {**required, **optional}.items():
        if name in {"image", "images"}:
            parameters[name] = image
            continue
        if name == "mask":
            parameters[name] = mask
            continue
        if name in {"resolution", "detect_resolution", "processor_res"}:
            parameters[name] = int(resolution)
            continue
        if name in low_names and float(threshold_a) >= 0:
            parameters[name] = int(threshold_a) if descriptor[0] == "INT" else float(threshold_a)
            consumed_a = True
            continue
        if name in high_names and float(threshold_b) >= 0:
            parameters[name] = int(threshold_b) if descriptor[0] == "INT" else float(threshold_b)
            consumed_b = True
            continue
        options = descriptor[0] if isinstance(descriptor, (tuple, list)) and descriptor else descriptor
        config = descriptor[1] if isinstance(descriptor, (tuple, list)) and len(descriptor) > 1 and isinstance(descriptor[1], dict) else {}
        if "default" in config:
            parameters[name] = config["default"]
        elif isinstance(options, list) and options:
            parameters[name] = options[0]
        elif options == "INT":
            parameters[name] = 0
        elif options == "FLOAT":
            parameters[name] = 0.0
        elif options == "BOOLEAN":
            parameters[name] = False
        elif name in required:
            raise RuntimeError(
                f"ControlNet preprocessor requires unsupported input {name!r}; connect/use it explicitly."
            )
    if float(threshold_a) >= 0 and not consumed_a:
        raise ValueError("controlnet threshold_a is set but the selected preprocessor has no compatible threshold input")
    if float(threshold_b) >= 0 and not consumed_b:
        raise ValueError("controlnet threshold_b is set but the selected preprocessor has no compatible threshold input")
    return parameters


def _control_module_name(module: Any) -> str:
    """정규화한 ControlNet 전처리기 이름(빈 값 = 기본 inpaint_only)."""

    return str(module or "inpaint_only").strip().casefold()


def _prepare_control_hint(image: Any, mask: Any, module: str, resolution: int,
                          threshold_a: float, threshold_b: float):
    name = _control_module_name(module)
    if name == "inpaint_only":
        return _inpaint_control_hint(image, mask), name
    if name in _UNSUPPORTED_CONTROL_MODULES:
        raise RuntimeError(
            f"ControlNet module {module!r} is unsupported: "
            f"{_UNSUPPORTED_CONTROL_MODULES[name]}. Select inpaint_only or an "
            "installed, semantically matching Comfy preprocessor."
        )
    if name in {"none", "raw"}:
        if float(threshold_a) >= 0 or float(threshold_b) >= 0:
            raise ValueError("ControlNet thresholds cannot be applied when controlnet_module is none")
        return ensure_image(image)[..., :3], "none"
    provider_id = _CONTROL_PREPROCESSORS.get(name)
    if provider_id is None:
        raise ValueError(
            f"Unknown ControlNet module {module!r}; refusing to substitute a different preprocessor"
        )
    mappings = _node_mappings()
    provider = mappings.get(provider_id)
    if provider is None:
        raise RuntimeError(
            f"ControlNet preprocessor {provider_id} is unavailable for module "
            f"{module!r}. Install/enable comfyui_controlnet_aux; refusing to "
            "substitute a different hint."
        )
    parameters = _preprocessor_kwargs(
        provider, image, mask, int(resolution), float(threshold_a), float(threshold_b)
    )
    if name == "openpose":
        parameters.update(detect_hand="disable", detect_body="enable", detect_face="disable")
    elif name == "openpose_full":
        parameters.update(detect_hand="enable", detect_body="enable", detect_face="enable")
    elif name == "openpose_hand":
        parameters.update(detect_hand="enable", detect_body="disable", detect_face="disable")
    function_name = getattr(provider, "FUNCTION", "execute")
    try:
        output = _invoke_node(provider, function_name, **parameters)
    except Exception as exc:
        raise RuntimeError(f"ControlNet preprocessor {provider_id} failed: {exc}") from exc
    values = _result_tuple(output)
    if not values:
        raise RuntimeError(f"ControlNet preprocessor {provider_id} returned no hint image.")
    return ensure_image(values[0])[..., :3], provider_id


def _without_existing_control(conditioning: Any):
    """Copy CONDITIONING while removing an upstream ControlNet chain."""

    if not isinstance(conditioning, (list, tuple)):
        raise ValueError("CONDITIONING must be a list when overriding external ControlNet")
    cleaned = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            raise ValueError("Invalid Comfy CONDITIONING entry while overriding external ControlNet")
        metadata = dict(entry[1])
        metadata.pop("control", None)
        metadata.pop("control_apply_to_uncond", None)
        cleaned.append([entry[0], metadata])
    return cleaned


def _apply_controlnet(positive: Any, negative: Any, control_net: Any, hint: Any,
                      strength: float, start: float, end: float, vae: Any,
                      control_mode: str):
    mode = str(control_mode)
    valid = {
        "Balanced", "My prompt is more important", "ControlNet is more important"
    }
    if mode not in valid:
        raise ValueError(f"Unknown ControlNet control_mode {mode!r}")
    requested_strength = float(strength)
    if not 0.0 <= requested_strength <= 10.0:
        raise ValueError("ControlNet strength must be between zero and 10")
    if not 0.0 <= float(start) < float(end) <= 1.0:
        raise ValueError("ControlNet guidance must satisfy 0 <= start < end <= 1")
    if requested_strength == 0.0:
        return positive, negative, {
            "mode": mode,
            "translation": "disabled_zero_strength",
            "requested_strength": 0.0,
            "effective_strength": 0.0,
        }

    mappings = _node_mappings()
    apply_node = mappings.get("ControlNetApplyAdvanced")
    if apply_node is None:
        try:
            import nodes

            apply_node = nodes.ControlNetApplyAdvanced
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("ComfyUI ControlNetApplyAdvanced is unavailable.") from exc

    effective_strength = requested_strength
    apply_negative = True
    translation = "balanced"
    if mode == "My prompt is more important":
        effective_strength *= 0.825
        translation = "forge_soft_weighting_as_global_0.825"
    elif mode == "ControlNet is more important":
        effective_strength *= 0.825
        apply_negative = False
        translation = "forge_positive_soft_negative_zero"
    input_negative = negative if apply_negative else []
    output = _invoke_node(
        apply_node,
        "apply_controlnet",
        positive=positive,
        negative=input_negative,
        control_net=control_net,
        image=hint,
        strength=effective_strength,
        start_percent=float(start),
        end_percent=float(end),
        vae=vae,
    )
    values = _result_tuple(output)
    if len(values) < 2:
        raise RuntimeError("ControlNetApplyAdvanced returned invalid conditioning outputs.")
    return values[0], (values[1] if apply_negative else negative), {
        "mode": mode,
        "translation": translation,
        "requested_strength": requested_strength,
        "effective_strength": effective_strength,
    }


def _restore_faces(image: Any, *, model: Any, clip: Any, vae: Any,
                   positive: Any, negative: Any, sampler_name: str,
                   scheduler: str, seed: int, steps: int, cfg: float,
                   settings_json: str, face_detector: Any = None):
    settings = _settings_object(
        settings_json, _RESTORE_FACE_DEFAULTS, "restore_face_settings_json"
    )
    mappings = _node_mappings()
    detector = face_detector
    detector_source = "connected BBOX_DETECTOR"
    if detector is None:
        provider = mappings.get("UltralyticsDetectorProvider")
        if provider is None:
            raise RuntimeError(
                "restore_face is enabled but Impact Subpack UltralyticsDetectorProvider is unavailable."
            )
        output = _invoke_node(
            provider, getattr(provider, "FUNCTION", "doit"),
            model_name=str(settings["detector_model"]),
        )
        values = _result_tuple(output)
        if not values or values[0] is None:
            raise RuntimeError("UltralyticsDetectorProvider returned no BBOX_DETECTOR.")
        detector = values[0]
        detector_source = str(settings["detector_model"])
    detailer = mappings.get("FaceDetailer")
    if detailer is None:
        raise RuntimeError("restore_face is enabled but Impact Pack FaceDetailer is unavailable.")
    output = _invoke_node(
        detailer,
        getattr(detailer, "FUNCTION", "doit"),
        image=image,
        model=model,
        clip=clip,
        vae=vae,
        guide_size=float(settings["guide_size"]),
        guide_size_for=bool(settings["guide_size_for_bbox"]),
        max_size=float(settings["max_size"]),
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        sampler_name=str(sampler_name),
        scheduler=str(scheduler),
        positive=positive,
        negative=negative,
        denoise=float(settings["denoise"]),
        feather=int(settings["feather"]),
        noise_mask=bool(settings["noise_mask"]),
        force_inpaint=bool(settings["force_inpaint"]),
        bbox_threshold=float(settings["bbox_threshold"]),
        bbox_dilation=int(settings["bbox_dilation"]),
        bbox_crop_factor=float(settings["bbox_crop_factor"]),
        sam_detection_hint="none",
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative="False",
        drop_size=int(settings["drop_size"]),
        bbox_detector=detector,
        wildcard="",
        cycle=int(settings["cycle"]),
    )
    values = _result_tuple(output)
    if not values:
        raise RuntimeError("Impact FaceDetailer returned no image.")
    restored = ensure_image(values[0])[..., :3]
    face_mask = values[3] if len(values) > 3 else None
    return restored, face_mask, {
        "provider": "Impact FaceDetailer",
        "detector": detector_source,
        "settings": settings,
    }


def _resize_image(image: Any, height: int, width: int):
    import torch.nn.functional as functional

    value = ensure_image(image)
    return functional.interpolate(
        value.permute(0, 3, 1, 2), size=(height, width),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1)


def _resample_image(image: Any, height: int, width: int):
    """Scale a crop to/from its sampling size like Forge's LANCZOS resize.

    Bicubic with antialiasing is the closest torch equivalent for both the
    enlarge-to-processing-size and the shrink-back-to-crop step; the bicubic
    overshoot is clamped so the result stays a valid ``[0,1]`` IMAGE.
    """

    import torch.nn.functional as functional

    value = ensure_image(image)
    if tuple(value.shape[1:3]) == (int(height), int(width)):
        return value
    return functional.interpolate(
        value.permute(0, 3, 1, 2), size=(int(height), int(width)),
        mode="bicubic", align_corners=False, antialias=True,
    ).clamp(0.0, 1.0).permute(0, 2, 3, 1)


def _resize_mask(mask: Any, height: int, width: int):
    import torch.nn.functional as functional

    value = ensure_mask(mask)
    return functional.interpolate(
        value.unsqueeze(1), size=(height, width), mode="bilinear",
        align_corners=False, antialias=True,
    ).squeeze(1).clamp(0.0, 1.0)


def _processing_size(use_custom_size: bool, custom_width: int, custom_height: int,
                     target_width: int, target_height: int) -> tuple[int, int] | None:
    """Forge's img2img ``p.width/p.height`` for an inpaint pass, if any.

    ``use_custom_size`` mirrors ``sam3_use_inpaint_width_height``; otherwise
    the caller-supplied target (the generation resolution before hires) is
    used.  ``None`` keeps the legacy behaviour of sampling at crop size.
    """

    if use_custom_size:
        return int(custom_width), int(custom_height)
    width, height = int(target_width or 0), int(target_height or 0)
    if width == 0 and height == 0:
        return None
    return width, height


def _vae_spatial_ratio(vae: Any) -> int:
    getter = getattr(vae, "spacial_compression_encode", None)
    if not callable(getter):
        return 8
    try:
        ratio = int(getter())
    except Exception as exc:
        raise RuntimeError("Could not read the VAE spatial compression ratio.") from exc
    if ratio < 1:
        raise RuntimeError(f"Invalid VAE spatial compression ratio: {ratio}")
    return ratio


def _pad_for_vae(image: Any, mask: Any, vae: Any):
    """Pad, never stretch, a crop to the VAE's exact spatial multiple."""

    import torch.nn.functional as functional

    image_value = ensure_image(image)
    mask_value = ensure_mask(
        mask,
        height=image_value.shape[1],
        width=image_value.shape[2],
        batch=image_value.shape[0],
    ).to(image_value.device)
    ratio = _vae_spatial_ratio(vae)
    height, width = image_value.shape[1:3]
    target_height = max(ratio, ((height + ratio - 1) // ratio) * ratio)
    target_width = max(ratio, ((width + ratio - 1) // ratio) * ratio)
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left
    padding = (left, right, top, bottom)
    if not any(padding):
        return image_value, mask_value, padding
    padded_image = functional.pad(
        image_value.permute(0, 3, 1, 2),
        (left, right, top, bottom),
        mode="replicate",
    ).permute(0, 2, 3, 1)
    padded_mask = functional.pad(
        mask_value.unsqueeze(1),
        (left, right, top, bottom),
        mode="constant",
        value=0.0,
    ).squeeze(1)
    return padded_image, padded_mask, padding


def _pad_image(image: Any, padding: tuple[int, int, int, int]):
    import torch.nn.functional as functional

    value = ensure_image(image)
    if not any(padding):
        return value
    return functional.pad(
        value.permute(0, 3, 1, 2), padding, mode="replicate"
    ).permute(0, 2, 3, 1)


def _unpad_image(image: Any, padding: tuple[int, int, int, int]):
    value = ensure_image(image)
    left, right, top, bottom = padding
    y2 = value.shape[1] - bottom if bottom else value.shape[1]
    x2 = value.shape[2] - right if right else value.shape[2]
    if top >= y2 or left >= x2:
        raise RuntimeError(f"Invalid VAE padding {padding} for decoded image {tuple(value.shape)}")
    return value[:, top:y2, left:x2, :]


def _mask_passes(mask: Any, image_batch: int, height: int, width: int,
                 mode: str) -> list[Any]:
    value = ensure_mask(mask, height=height, width=width)
    if value.shape[0] == 1 and image_batch > 1:
        value = value.expand(image_batch, -1, -1).clone()
    if value.shape[0] % image_batch:
        raise ValueError(
            f"MASK batch {value.shape[0]} must be 1, IMAGE batch {image_batch}, "
            "or an integer number of IMAGE-sized individual groups"
        )
    groups = list(value.reshape(-1, image_batch, height, width).unbind(0))
    if str(mode).casefold() == "combined":
        return [_torch().stack(groups, dim=0).amax(dim=0)]
    if str(mode).casefold() != "individual":
        raise ValueError("mask_mode must be Combined or Individual")
    return groups


class ForgeNeoSAM3Detailer:
    """Sequential VAE inpaint over Combined or slash-derived Individual masks."""

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _sampler_choices()
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "inpaint_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "mask_mode": (["Combined", "Individual"], {"default": "Individual"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.05}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "fill_mode": (["fill", "original", "latent noise", "latent nothing"], {"default": "original"}),
                "only_masked": ("BOOLEAN", {"default": True}),
                "mask_padding": ("INT", {"default": 32, "min": 0, "max": 2048}),
                "use_custom_size": ("BOOLEAN", {"default": False}),
                "custom_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "custom_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64}),
                "controlnet_enable": ("BOOLEAN", {"default": False}),
                "controlnet_model_name": ("STRING", {"default": "None"}),
                "controlnet_module": ("STRING", {"default": "inpaint_only"}),
                "controlnet_override_external": ("BOOLEAN", {"default": False}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "controlnet_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "controlnet_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "controlnet_processor_resolution": ("INT", {"default": 512, "min": 0, "max": 4096, "step": 8}),
                "controlnet_settings_json": (
                    "STRING",
                    {"default": json.dumps(_CONTROLNET_EXTRA_DEFAULTS), "multiline": True},
                ),
                "restore_face": ("BOOLEAN", {"default": False}),
                "restore_face_settings_json": (
                    "STRING",
                    {"default": json.dumps(_RESTORE_FACE_DEFAULTS), "multiline": True},
                ),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "control_image": ("IMAGE",),
                "face_detector": ("BBOX_DETECTOR",),
                # Forge p.width/p.height for "only masked": the crop is
                # widened to this aspect and sampled at this size. 0 keeps the
                # crop-size sampling of saved workflows. Custom size wins.
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "applied_mask", "report_json")
    FUNCTION = "detail"
    CATEGORY = CATEGORY

    def detail(
        self,
        image,
        mask,
        model,
        clip,
        vae,
        positive,
        negative,
        inpaint_prompt="",
        negative_prompt="",
        mask_mode="Individual",
        seed=0,
        steps=28,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.4,
        noise_multiplier=1.0,
        fill_mode="original",
        only_masked=True,
        mask_padding=32,
        use_custom_size=False,
        custom_width=512,
        custom_height=512,
        grow_mask_by=6,
        controlnet_enable=False,
        controlnet_model_name="None",
        controlnet_module="inpaint_only",
        controlnet_override_external=False,
        controlnet_strength=1.0,
        controlnet_start=0.0,
        controlnet_end=1.0,
        controlnet_processor_resolution=512,
        controlnet_settings_json="{}",
        restore_face=False,
        restore_face_settings_json="{}",
        enabled=True,
        control_net=None,
        control_image=None,
        face_detector=None,
        target_width=0,
        target_height=0,
    ):
        image_value = ensure_image(image)[..., :3]
        mask_value = ensure_mask(mask, height=image_value.shape[1], width=image_value.shape[2])
        mask_value = mask_value.to(image_value.device)
        if not enabled:
            return image_value, mask_value, json.dumps({"enabled": False, "status": "disabled"})
        if not bool((mask_value > 0.0).any()):
            return image_value, mask_value, json.dumps({
                "enabled": True,
                "status": "no_detection",
                "passes": [],
                "controlnet_requested": bool(controlnet_enable),
                "controlnet_enabled": False,
                "restore_face": None,
            })
        denoise_value = float(denoise)
        if not 0.0 <= denoise_value <= 1.0:
            raise ValueError("SAM3 detailer denoise must be between zero and one.")
        if int(steps) < 1:
            raise ValueError("SAM3 detailer steps must be at least one.")
        if float(cfg) < 0.0:
            raise ValueError("SAM3 detailer cfg cannot be negative.")
        if not 0.0 <= float(noise_multiplier) <= 2.0:
            raise ValueError("SAM3 detailer noise_multiplier must be between zero and two.")
        if int(mask_padding) < 0:
            raise ValueError("SAM3 detailer mask_padding cannot be negative.")
        if use_custom_size and not (
            64 <= int(custom_width) <= 8192 and 64 <= int(custom_height) <= 8192
        ):
            raise ValueError("SAM3 detailer custom size must be between 64 and 8192.")
        target_width_value = int(target_width or 0)
        target_height_value = int(target_height or 0)
        if (target_width_value, target_height_value) != (0, 0) and not (
            64 <= target_width_value <= 8192 and 64 <= target_height_value <= 8192
        ):
            raise ValueError(
                "SAM3 detailer target size must be 0x0 (crop size) or between 64 and 8192."
            )
        processing_size = _processing_size(
            bool(use_custom_size), int(custom_width), int(custom_height),
            target_width_value, target_height_value,
        )
        controlnet_strength_value = float(controlnet_strength)
        if not 0.0 <= controlnet_strength_value <= 10.0:
            raise ValueError("ControlNet strength must be between zero and 10")

        needs_conditioning = denoise_value > 0.0 or bool(restore_face)
        positive_value = (
            _encode_prompt(clip, str(inpaint_prompt))
            if needs_conditioning and str(inpaint_prompt).strip()
            else positive
        )
        negative_value = (
            _encode_prompt(clip, str(negative_prompt))
            if needs_conditioning and str(negative_prompt).strip()
            else negative
        )
        control_settings = None
        control_model = None
        control_model_source = None
        control_source = None
        control_image_mode = None
        controlnet_requested = bool(controlnet_enable)
        controlnet_active = (
            controlnet_requested
            and controlnet_strength_value > 0.0
            and denoise_value > 0.0
        )
        controlnet_disabled_reason = None
        if controlnet_requested and not controlnet_active:
            controlnet_disabled_reason = (
                "zero_strength" if controlnet_strength_value == 0.0 else "zero_denoise"
            )
        if controlnet_active:
            control_settings = _settings_object(
                controlnet_settings_json,
                _CONTROLNET_EXTRA_DEFAULTS,
                "controlnet_settings_json",
            )
            controlnet_override_external = bool(
                controlnet_override_external or control_settings["override_external"]
            )
            control_model, control_model_source = _load_controlnet(
                control_net, str(controlnet_model_name)
            )
            if control_image is not None:
                control_source = ensure_image(control_image)[..., :3]
                if control_source.shape[0] == 1 and image_value.shape[0] > 1:
                    control_source = control_source.expand(
                        image_value.shape[0], -1, -1, -1
                    ).clone()
                elif control_source.shape[0] != image_value.shape[0]:
                    raise ValueError(
                        f"control_image batch {control_source.shape[0]} does not match image batch {image_value.shape[0]}"
                    )
            # 설정한 resize_mode 는 여기서 한 번 검증한다(입력 힌트에선 쓰지 않더라도).
            control_image_mode = _effective_control_resize_mode(
                control_settings["resize_mode"], explicit_control_image=True,
            )
            if control_source is not None and only_masked:
                # Forge only-masked(try_crop_image_with_a1111_mask): 유닛 이미지를 먼저
                # unit.resize_mode 로 init 이미지 크기에 맞춘 뒤 마스크 크롭을 자른다. 예전엔
                # 모드와 상관없이 입력 크기로 늘려(Just Resize) 종횡비가 다른 control 이 찌그러졌다.
                # 이 맞춤은 A1111 images.resize_image 라 Resize and Fill 의 띠는 가장자리 행/열을
                # 늘려 채운다(검정 띠면 띠에 걸린 크롭 힌트가 Forge 와 달라진다).
                # 전체 이미지 패스는 원본 control 을 그대로 sample 크기에 맞춘다(아래 루프).
                control_source = _fit_control_image(
                    control_source, image_value.shape[1], image_value.shape[2],
                    control_image_mode, fill="edge",
                )
        passes = _mask_passes(
            mask_value, image_value.shape[0], image_value.shape[1], image_value.shape[2],
            mask_mode,
        )
        current = image_value.clone()
        pass_reports = []
        applied = union_masks(passes, reference=image_value)
        image_height, image_width = image_value.shape[1:3]
        for pass_index, pass_mask in enumerate(passes):
            mask_crop = mask_bounds(pass_mask, int(mask_padding)) if only_masked else (
                0, 0, image_width, image_height
            )
            if mask_crop is None:
                pass_reports.append({"pass": pass_index + 1, "status": "empty_skipped"})
                continue
            bounds = mask_crop
            if only_masked and processing_size is not None:
                # Forge inpaint_full_res: widen the padded mask crop to the
                # processing aspect ratio before scaling it to that size.
                bounds = expand_crop_region(
                    mask_crop, processing_size[0], processing_size[1],
                    image_width, image_height,
                )
            x1, y1, x2, y2 = bounds
            region = current[:, y1:y2, x1:x2, :]
            region_mask = pass_mask[:, y1:y2, x1:x2]
            original_height, original_width = region.shape[1:3]
            if processing_size is None:
                sample_width, sample_height = original_width, original_height
            elif only_masked:
                sample_width, sample_height = fit_sample_size(
                    original_width, original_height,
                    processing_size[0], processing_size[1],
                )
            else:
                # Forge whole picture (inpaint_full_res=False): img2img
                # resize_mode 0 "Just Resize" samples the whole image at exactly
                # p.width/p.height.  The aspect-fitting above is only for the
                # only-masked crop; here it would sample below the requested
                # size (1024x512 image, 512x512 → 512x256).  The result is still
                # resampled back to the original size below.
                sample_width, sample_height = int(processing_size[0]), int(processing_size[1])
            if (sample_height, sample_width) != (original_height, original_width):
                work_image = _resample_image(region, sample_height, sample_width)
                work_mask = _resize_mask(region_mask, sample_height, sample_width)
            else:
                work_image, work_mask = region, region_mask

            # Forge builds every individual pass with the selected seed; it
            # does not silently advance the seed by mask index.
            pass_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
            if denoise_value == 0.0:
                pass_reports.append({
                    "pass": pass_index + 1,
                    "status": "no_op_zero_denoise",
                    "seed": pass_seed,
                    "crop": [x1, y1, x2, y2],
                    "mask_crop": list(mask_crop),
                    "sample_size": [sample_width, sample_height],
                    "controlnet": None,
                })
                continue

            requested_height, requested_width = work_image.shape[1:3]
            work_image, work_mask, vae_padding = _pad_for_vae(
                work_image, work_mask, vae
            )
            vae_height, vae_width = work_image.shape[1:3]
            pass_positive, pass_negative = positive_value, negative_value
            control_report = None
            if controlnet_active:
                # An explicitly connected control image stays fixed. Without
                # one, Forge feeds the image being processed, so later
                # sequential passes must see the preceding pass's result.
                whole_image_control = control_source is not None and not only_masked
                if whole_image_control:
                    # Forge 전체 이미지: 유닛 이미지를 그대로 unit.resize_mode 로 p.width/p.height
                    # (= sample 크기)에 맞춘다 — 입력 크기로 먼저 늘리지 않는다.
                    control_region = control_source
                    control_resize_mode = control_image_mode
                else:
                    # 입력 이미지(또는 입력 크기에 맞춘 control)의 같은 크롭은 잠재 입력(region 을
                    # sample 크기로 늘린 것)과 같은 기하여야 한다 — Forge 도 유닛 이미지가 없으면
                    # p.resize_mode(0)로, only-masked 유닛 이미지는 크롭 뒤 잠재 입력과 같은
                    # 맞춤(OUTER_FIT, 여기선 비율 유지 sample 크기라 그대로 늘리기)으로 맞춘다.
                    pass_control_source = control_source if control_source is not None else current
                    control_region = pass_control_source[:, y1:y2, x1:x2, :]
                    control_resize_mode = _effective_control_resize_mode(
                        control_settings["resize_mode"], explicit_control_image=False,
                    )
                processor_resolution = int(controlnet_processor_resolution) or 512
                if bool(control_settings["pixel_perfect"]):
                    processor_resolution = _pixel_perfect_resolution(
                        control_region,
                        requested_height,
                        requested_width,
                        control_resize_mode,
                    )
                mask_driven_module = _control_module_name(controlnet_module) == "inpaint_only"
                if whole_image_control and not mask_driven_module:
                    # Forge 전체 이미지 + 마스크를 안 쓰는 전처리기: preprocessor(유닛 이미지) 다음
                    # crop_and_resize_image(출력, unit.resize_mode, h, w). Resize and Fill 띠는
                    # 전처리 '출력' 테두리의 중앙값이다 — 입력에 띠를 먼저 칠하고 전처리하면 띠
                    # 경계에 캐니 선·가짜 깊이면 같은 힌트가 생긴다. 매핑된 전처리기는 마스크를
                    # 받지 않지만, 받는다면 control 크기로 늘린 패스 마스크를 준다.
                    raw_hint, preprocessor_used = _prepare_control_hint(
                        control_region,
                        _resize_mask(region_mask, control_region.shape[1], control_region.shape[2]),
                        str(controlnet_module),
                        processor_resolution,
                        float(control_settings["threshold_a"]),
                        float(control_settings["threshold_b"]),
                    )
                    hint = _pad_image(
                        _fit_control_image(
                            raw_hint, requested_height, requested_width, control_resize_mode,
                        ),
                        vae_padding,
                    )
                else:
                    resized_control = _fit_control_image(
                        control_region,
                        requested_height,
                        requested_width,
                        control_resize_mode,
                    )
                    resized_control = _pad_image(resized_control, vae_padding)
                    hint_mask = work_mask
                    if whole_image_control and control_resize_mode == "Resize and Fill":
                        # Forge inpaint 전처리기는 Resize and Fill 에서 유닛 마스크의 띠를 255 로
                        # 채운다(expand_mask_when_resize_and_fill) — 띠는 힌트 -1(새로 그릴 칸).
                        band = _resize_and_fill_band_mask(
                            control_region, requested_height, requested_width, vae_padding,
                        )
                        hint_mask = work_mask.maximum(
                            band.to(device=work_mask.device, dtype=work_mask.dtype)
                        )
                    hint, preprocessor_used = _prepare_control_hint(
                        resized_control,
                        hint_mask,
                        str(controlnet_module),
                        processor_resolution,
                        float(control_settings["threshold_a"]),
                        float(control_settings["threshold_b"]),
                    )
                control_positive = (
                    _without_existing_control(positive_value)
                    if controlnet_override_external else positive_value
                )
                control_negative = (
                    _without_existing_control(negative_value)
                    if controlnet_override_external else negative_value
                )
                pass_positive, pass_negative, mode_report = _apply_controlnet(
                    control_positive,
                    control_negative,
                    control_model,
                    hint,
                    controlnet_strength_value,
                    float(controlnet_start),
                    float(controlnet_end),
                    vae,
                    str(control_settings["control_mode"]),
                )
                control_report = {
                    "model": control_model_source,
                    "module": str(controlnet_module),
                    "preprocessor": preprocessor_used,
                    "processor_resolution": processor_resolution,
                    "pixel_perfect": bool(control_settings["pixel_perfect"]),
                    "resize_mode": control_resize_mode,
                    "resize_mode_setting": str(control_settings["resize_mode"]),
                    "hint_source": "control_image" if control_source is not None else "input",
                    # control_image 를 맞춘 모드(only-masked 는 입력 크기로, 전체 이미지는 sample 크기로).
                    "control_image_resize_mode": (
                        control_image_mode if control_source is not None else None
                    ),
                    "threshold_a": float(control_settings["threshold_a"]),
                    "threshold_b": float(control_settings["threshold_b"]),
                    "override_external": bool(controlnet_override_external),
                    **mode_report,
                }
            latent = _vae_encode_for_inpaint(
                vae,
                work_image,
                work_mask,
                int(grow_mask_by),
                fill_mode=str(fill_mode),
                seed=pass_seed,
            )
            sampled = _sample_latent(
                model, pass_seed, int(steps), float(cfg), str(sampler_name),
                str(scheduler), pass_positive, pass_negative, latent,
                denoise_value, float(noise_multiplier),
            )
            generated = ensure_image(_vae_decode(vae, sampled))[..., :3]
            if generated.shape[0] != region.shape[0]:
                raise RuntimeError(
                    f"VAE decoded batch {generated.shape[0]} does not match image batch {region.shape[0]}"
                )
            if tuple(generated.shape[1:3]) != (vae_height, vae_width):
                generated = _resize_image(generated, vae_height, vae_width)
            generated = _unpad_image(generated, vae_padding)
            if tuple(generated.shape[1:3]) != (original_height, original_width):
                # Back from the processing size into the (expanded) crop.
                generated = _resample_image(generated, original_height, original_width)
            alpha = region_mask.unsqueeze(-1).to(device=current.device, dtype=current.dtype)
            generated = generated.to(device=current.device, dtype=current.dtype)[..., : current.shape[-1]]
            original_region = current[:, y1:y2, x1:x2, :]
            current[:, y1:y2, x1:x2, :] = original_region * (1.0 - alpha) + generated * alpha
            pass_reports.append({
                "pass": pass_index + 1,
                "status": "sampled",
                "seed": pass_seed,
                "crop": [x1, y1, x2, y2],
                "mask_crop": list(mask_crop),
                "sample_size": [sample_width, sample_height],
                "vae_sample_size": [vae_width, vae_height],
                "vae_padding": list(vae_padding),
                "controlnet": control_report,
            })

        restore_report = None
        if restore_face:
            current, face_mask, restore_report = _restore_faces(
                current,
                model=model,
                clip=clip,
                vae=vae,
                positive=positive_value,
                negative=negative_value,
                sampler_name=str(sampler_name),
                scheduler=str(scheduler),
                seed=int(seed) & 0xFFFFFFFFFFFFFFFF,
                steps=int(steps),
                cfg=float(cfg),
                settings_json=str(restore_face_settings_json),
                face_detector=face_detector,
            )
            if tuple(current.shape[1:3]) != tuple(image_value.shape[1:3]):
                current = _resize_image(current, image_value.shape[1], image_value.shape[2])
            if face_mask is not None:
                face_mask_value = ensure_mask(
                    face_mask,
                    height=image_value.shape[1],
                    width=image_value.shape[2],
                )
                restore_report["mask_pixels"] = int((face_mask_value > 0.0).sum().item())

        report = {
            "enabled": True,
            "status": (
                "no_op_zero_denoise"
                if denoise_value == 0.0 and not bool(restore_face)
                else "ok"
            ),
            "mask_mode": str(mask_mode),
            "only_masked": bool(only_masked),
            "padding": int(mask_padding),
            "custom_size": [int(custom_width), int(custom_height)] if use_custom_size else None,
            "processing_size": list(processing_size) if processing_size is not None else None,
            "controlnet_requested": controlnet_requested,
            "controlnet_enabled": controlnet_active,
            "controlnet_disabled_reason": controlnet_disabled_reason,
            "restore_face": restore_report,
            "passes": pass_reports,
        }
        return current.clamp(0.0, 1.0), applied, json.dumps(report, ensure_ascii=False)


class ForgeNeoSAM3Refine(ForgeNeoSAM3Detailer):
    """Forge-style independent results: every mask starts from the source."""

    def detail(
        self,
        image,
        mask,
        model,
        clip,
        vae,
        positive,
        negative,
        inpaint_prompt="",
        negative_prompt="",
        mask_mode="Individual",
        seed=0,
        steps=28,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        denoise=0.4,
        noise_multiplier=1.0,
        fill_mode="original",
        only_masked=True,
        mask_padding=32,
        use_custom_size=False,
        custom_width=512,
        custom_height=512,
        grow_mask_by=6,
        controlnet_enable=False,
        controlnet_model_name="None",
        controlnet_module="inpaint_only",
        controlnet_override_external=False,
        controlnet_strength=1.0,
        controlnet_start=0.0,
        controlnet_end=1.0,
        controlnet_processor_resolution=512,
        controlnet_settings_json="{}",
        restore_face=False,
        restore_face_settings_json="{}",
        enabled=True,
        control_net=None,
        control_image=None,
        face_detector=None,
        target_width=0,
        target_height=0,
    ):
        image_value = ensure_image(image)[..., :3]
        shared_control_net = control_net

        def run_one(pass_mask, pass_mode):
            return ForgeNeoSAM3Detailer.detail(
                self,
                image=image_value,
                mask=pass_mask,
                model=model,
                clip=clip,
                vae=vae,
                positive=positive,
                negative=negative,
                inpaint_prompt=inpaint_prompt,
                negative_prompt=negative_prompt,
                mask_mode=pass_mode,
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                noise_multiplier=noise_multiplier,
                fill_mode=fill_mode,
                only_masked=only_masked,
                mask_padding=mask_padding,
                use_custom_size=use_custom_size,
                custom_width=custom_width,
                custom_height=custom_height,
                grow_mask_by=grow_mask_by,
                controlnet_enable=controlnet_enable,
                controlnet_model_name=controlnet_model_name,
                controlnet_module=controlnet_module,
                controlnet_override_external=controlnet_override_external,
                controlnet_strength=controlnet_strength,
                controlnet_start=controlnet_start,
                controlnet_end=controlnet_end,
                controlnet_processor_resolution=controlnet_processor_resolution,
                controlnet_settings_json=controlnet_settings_json,
                restore_face=restore_face,
                restore_face_settings_json=restore_face_settings_json,
                enabled=enabled,
                control_net=shared_control_net,
                control_image=control_image,
                face_detector=face_detector,
                target_width=target_width,
                target_height=target_height,
            )

        if not enabled or str(mask_mode).casefold() == "combined":
            return run_one(mask, mask_mode)

        passes = _mask_passes(
            mask,
            image_value.shape[0],
            image_value.shape[1],
            image_value.shape[2],
            mask_mode,
        )
        nonempty = [item for item in passes if bool((item > 0.0).any())]
        applied = union_masks(nonempty, reference=image_value)
        if not nonempty:
            return image_value, applied, json.dumps({
                "enabled": True,
                "status": "no_detection",
                "mask_mode": "Individual",
                "processing": "independent_from_original",
                "results": [],
            })

        # Every independent result would otherwise run ControlNetLoader again
        # (Comfy's loader has no cache). Load a named model once and hand the
        # same CONTROL_NET to each pass; a connected one is reused as-is.
        preloaded_source = None
        if (
            shared_control_net is None
            and len(nonempty) > 1
            and bool(controlnet_enable)
            and 0.0 < float(controlnet_strength) <= 10.0
            and 0.0 < float(denoise) <= 1.0
        ):
            shared_control_net, preloaded_source = _load_controlnet(
                None, str(controlnet_model_name)
            )

        outputs = []
        reports = []
        for mask_index, pass_mask in enumerate(nonempty, start=1):
            refined, _, report_json = run_one(pass_mask, "Combined")
            outputs.append(ensure_image(refined)[..., :3])
            child_report = json.loads(report_json)
            child_report["mask_index"] = mask_index
            if preloaded_source is not None:
                for pass_report in child_report.get("passes", []):
                    if isinstance(pass_report.get("controlnet"), dict):
                        pass_report["controlnet"]["model"] = preloaded_source
            reports.append(child_report)

        statuses = {str(item.get("status")) for item in reports}
        status = (
            "no_op_zero_denoise"
            if statuses == {"no_op_zero_denoise"}
            else "ok"
        )
        report = {
            "enabled": True,
            "status": status,
            "mask_mode": "Individual",
            "processing": "independent_from_original",
            "result_count": len(outputs),
            "controlnet_requested": bool(controlnet_enable),
            "controlnet_enabled": any(
                bool(item.get("controlnet_enabled")) for item in reports
            ),
            "results": reports,
        }
        return (
            _torch().cat(outputs, dim=0).clamp(0.0, 1.0),
            applied,
            json.dumps(report, ensure_ascii=False),
        )


class ForgeNeoSAM3TileRepair(ForgeNeoSAM3Detailer):
    """Comfy masked-region repair; not Forge's vendored Anima/PiD pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": True}),
                "settings_json": (
                    "STRING",
                    {
                        "default": json.dumps(_TILE_REPAIR_DEFAULTS),
                        "multiline": True,
                    },
                ),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "control_image": ("IMAGE",),
                "face_detector": ("BBOX_DETECTOR",),
            },
        }

    FUNCTION = "tile_repair"

    def tile_repair(self, image, mask, model, clip, vae, positive, negative,
                    enabled=True, settings_json="{}", control_net=None,
                    control_image=None, face_detector=None):
        try:
            requested_settings = json.loads(settings_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid SAM3 tile settings_json: {exc}") from exc
        if not isinstance(requested_settings, dict):
            raise ValueError("SAM3 tile settings_json must contain a JSON object")
        forge_only = sorted(set(requested_settings) & _FORGE_TILE_ONLY_KEYS)
        if forge_only:
            raise RuntimeError(
                "Forge Anima Tile-Repair/PiD settings are unavailable in this "
                "Comfy masked-region repair node: " + ", ".join(forge_only) +
                ". The Forge path requires its vendored Anima/Qwen DiT, Qwen3 "
                "text encoder, Qwen-Image VAE, ControlNet-LLLite or PiD runtime; "
                "use a dedicated Comfy provider for that pipeline."
            )
        settings = _settings_object(
            json.dumps(requested_settings),
            _TILE_REPAIR_DEFAULTS,
            "SAM3 tile settings_json",
        )
        samplers, schedulers = _sampler_choices()
        sampler = str(settings.get("sampler_name", "euler"))
        scheduler = str(settings.get("scheduler", "normal"))
        if sampler not in samplers:
            raise ValueError(f"Unknown Comfy sampler_name {sampler!r}")
        if scheduler not in schedulers:
            raise ValueError(f"Unknown Comfy scheduler {scheduler!r}")
        output, applied, report_json = self.detail(
            image=image,
            mask=mask,
            model=model,
            clip=clip,
            vae=vae,
            positive=positive,
            negative=negative,
            inpaint_prompt=str(settings.get("inpaint_prompt", "")),
            negative_prompt=str(settings.get("negative_prompt", "")),
            mask_mode=str(settings.get("mask_mode", "Individual")),
            seed=int(settings.get("seed", 0)),
            steps=int(settings.get("steps", 28)),
            cfg=float(settings.get("cfg", 7.0)),
            sampler_name=sampler,
            scheduler=scheduler,
            denoise=float(settings.get("denoise", 0.4)),
            noise_multiplier=float(settings.get("noise_multiplier", 1.0)),
            fill_mode=str(settings.get("fill_mode", "original")),
            only_masked=True,
            mask_padding=int(settings.get("tile_padding", 64)),
            use_custom_size=True,
            custom_width=int(settings.get("tile_width", 512)),
            custom_height=int(settings.get("tile_height", 512)),
            grow_mask_by=int(settings.get("grow_mask_by", 6)),
            controlnet_enable=bool(settings.get("controlnet_enable", False)),
            controlnet_model_name=str(settings.get("controlnet_model_name", "None")),
            controlnet_module=str(settings.get("controlnet_module", "tile_resample")),
            controlnet_override_external=bool(settings.get("controlnet_override_external", False)),
            controlnet_strength=float(settings.get("controlnet_strength", 1.0)),
            controlnet_start=float(settings.get("controlnet_start", 0.0)),
            controlnet_end=float(settings.get("controlnet_end", 1.0)),
            controlnet_processor_resolution=int(settings.get("controlnet_processor_resolution", 512)),
            controlnet_settings_json=json.dumps(settings.get("controlnet_settings", _CONTROLNET_EXTRA_DEFAULTS)),
            restore_face=bool(settings.get("restore_face", False)),
            restore_face_settings_json=json.dumps(settings.get("restore_face_settings", _RESTORE_FACE_DEFAULTS)),
            enabled=bool(enabled),
            control_net=control_net,
            control_image=control_image,
            face_detector=face_detector,
        )
        report = json.loads(report_json)
        report.update({
            "implementation": "comfy_masked_region_repair",
            "forge_anima_tile_repair": False,
        })
        return output, applied, json.dumps(report, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {
    "ForgeNeoSAM3Mask": ForgeNeoSAM3Mask,
    "ForgeNeoSAM3Detailer": ForgeNeoSAM3Detailer,
    "ForgeNeoSAM3Refine": ForgeNeoSAM3Refine,
    "ForgeNeoSAM3TileRepair": ForgeNeoSAM3TileRepair,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForgeNeoSAM3Mask": "Forge Neo SAM3 Mask",
    "ForgeNeoSAM3Detailer": "Forge Neo SAM3 Detailer",
    "ForgeNeoSAM3Refine": "Forge Neo SAM3 Refine",
    "ForgeNeoSAM3TileRepair": "SAM3 Region Repair (Comfy)",
}
