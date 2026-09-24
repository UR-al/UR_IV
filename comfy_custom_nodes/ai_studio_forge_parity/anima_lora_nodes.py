"""ComfyUI node adapters for ANIMA 28/40/52-block LoRA compatibility.

The block remapper in :mod:`.anima_lora` is deliberately pure.  This module
owns the narrow ComfyUI boundary: resolving a selected file, caching the raw
state dict without mutating it, preserving safetensors metadata, and finally
delegating patch creation to ComfyUI.
"""

from __future__ import annotations

import importlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .anima_lora import PreparedAnimaLora, prepare_anima_lora_for_model
from .compat import filename_choices, folder_paths_module, node_result, provider_class


CATEGORY = "AI Studio/Forge Neo parity/Loaders"
LOGGER = logging.getLogger("ai_studio_forge_parity")


@dataclass
class _CachedLora:
    path: str
    state_dict: Any
    metadata: Any


@dataclass(frozen=True)
class _AnimaBlockVector:
    base: float
    blocks: tuple[float, ...]
    populated: str


_ANIMA_PATCH_BLOCK_RE = re.compile(r"^diffusion_model\.blocks\.(\d+)\.")
_NUMERIC_VECTOR_TOKEN_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_LEGACY_INSPIRE_BLOCK_VECTOR = "1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1"


class AnimaLoraStateCache:
    """One-node cache matching ComfyUI's native ``LoraLoader`` semantics."""

    def __init__(self) -> None:
        self._entry: _CachedLora | None = None

    def load(self, lora_name: str) -> tuple[Any, Any]:
        folder_paths = folder_paths_module()
        resolve = getattr(folder_paths, "get_full_path_or_raise", None)
        if callable(resolve):
            path = str(resolve("loras", lora_name))
        else:  # compatibility with older ComfyUI builds
            path = str(folder_paths.get_full_path("loras", lora_name) or "")
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"LoRA not found: {lora_name}")

        if self._entry is not None and self._entry.path == path:
            return self._entry.state_dict, self._entry.metadata

        comfy_utils = importlib.import_module("comfy.utils")
        state_dict, metadata = comfy_utils.load_torch_file(
            path,
            safe_load=True,
            return_metadata=True,
        )
        self._entry = _CachedLora(path, state_dict, metadata)
        return state_dict, metadata


def _format_ratio(value: float) -> str:
    value = 0.0 if value == 0.0 else value
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _parse_anima_block_vector(
    block_vector: str,
    block_count: int,
    *,
    inverse: bool,
    seed: int,
    a_value: float,
    b_value: float,
) -> _AnimaBlockVector:
    """Parse a canonical ANIMA vector: N blocks, optionally preceded by base."""

    text = str(block_vector or "").strip()
    if not text:
        default_value = 0.0 if inverse else 1.0
        blocks = (default_value,) * block_count
        return _AnimaBlockVector(
            default_value,
            blocks,
            ",".join(_format_ratio(value) for value in (default_value, *blocks)),
        )
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    tokens = [token.strip() for token in text.split(",")]
    if len(tokens) == block_count:
        base_token = "1"
        block_tokens = tokens
    elif len(tokens) == block_count + 1:
        base_token = tokens[0]
        block_tokens = tokens[1:]
    else:
        raise RuntimeError(
            "ANIMA LoRA block vector must contain exactly "
            f"{block_count} block values, or {block_count + 1} values with "
            f"a leading base value; received {len(tokens)}"
        )

    numpy = None
    rng = None

    def resolve(token: str) -> float:
        nonlocal numpy, rng
        if token == "A":
            value = float(a_value)
        elif token == "a":
            value = float(a_value) / 2.0
        elif token == "B":
            value = float(b_value)
        elif token == "b":
            value = float(b_value) / 2.0
        elif token in {"R", "r", "U", "u"}:
            if rng is None:
                numpy = importlib.import_module("numpy")
                rng = numpy.random.RandomState(int(seed) % (2**31))
            bounds = (-1.5, 1.5) if token in {"U", "u"} else (0.0, 3.0)
            value = round(float(rng.uniform(*bounds)), 2)
        elif _NUMERIC_VECTOR_TOKEN_RE.fullmatch(token):
            value = float(token)
        else:
            raise RuntimeError(
                f"Invalid ANIMA LoRA block-vector value {token!r}; use a number, "
                "A/a, B/b, R/r, or U/u"
            )
        if not math.isfinite(value):
            raise RuntimeError("ANIMA LoRA block-vector values must be finite")
        return 1.0 - value if inverse else value

    base = resolve(base_token)
    blocks = tuple(resolve(token) for token in block_tokens)
    populated = ",".join(_format_ratio(value) for value in (base, *blocks))
    return _AnimaBlockVector(base, blocks, populated)


def _require_lora_metadata_support(model: Any, clip: Any, metadata: Any) -> None:
    if not metadata:
        return
    set_model = getattr(model, "set_attachments", None) if model is not None else None
    clip_patcher = getattr(clip, "patcher", None) if clip is not None else None
    set_clip = (
        getattr(clip_patcher, "set_attachments", None)
        if clip_patcher is not None
        else None
    )
    missing: list[str] = []
    if model is not None and not callable(set_model):
        missing.append("MODEL.set_attachments")
    if clip is not None and not callable(set_clip):
        missing.append("CLIP.patcher.set_attachments")
    if missing:
        raise RuntimeError(
            "LoRA block weighting cannot preserve safetensors metadata because "
            "the installed runtime lacks " + ", ".join(missing)
        )


def _attach_lora_metadata(model: Any, clip: Any, metadata: Any) -> None:
    if not metadata:
        return
    _require_lora_metadata_support(model, clip, metadata)
    set_model = getattr(model, "set_attachments", None) if model is not None else None
    clip_patcher = getattr(clip, "patcher", None) if clip is not None else None
    set_clip = (
        getattr(clip_patcher, "set_attachments", None)
        if clip_patcher is not None
        else None
    )
    if model is not None:
        set_model("lora_metadata", metadata)
    if clip is not None:
        set_clip("lora_metadata", metadata)


_ALIAS_BOUNDARIES = frozenset("._")


def _payload_alias_prefixes(state_keys: Iterable[str]) -> frozenset[str]:
    """Every string an alias may equal to own a payload key.

    An alias owns ``key`` exactly when ``key == alias`` or ``key`` starts with
    ``alias + "."`` / ``alias + "_"``.  The latter holds iff ``alias`` is the
    prefix of ``key`` that ends right before one of its ``.``/``_``
    characters, so collecting the key itself plus every such boundary prefix
    turns the per-alias linear scan into one set lookup with the same result.
    Multi-dot aliases (``diffusion_model.blocks.0.attn``) are covered because
    *every* boundary is recorded, not only the first one.
    """

    prefixes: set[str] = set()
    for key in state_keys:
        prefixes.add(key)
        for index, character in enumerate(key):
            if character in _ALIAS_BOUNDARIES:
                prefixes.add(key[:index])
    return frozenset(prefixes)


def _reject_comfy_alias_collisions(
    state_dict: dict[Any, Any],
    key_map: dict[Any, Any],
) -> None:
    """Fail before Comfy silently lets a later alias overwrite the first patch."""

    payload_prefixes = _payload_alias_prefixes(
        key for key in state_dict if isinstance(key, str)
    )
    aliases_by_target: dict[str, list[str]] = {}
    for alias, target in key_map.items():
        if not isinstance(alias, str) or alias not in payload_prefixes:
            continue
        target_name = str(target[0] if isinstance(target, tuple) else target)
        aliases_by_target.setdefault(target_name, []).append(alias)
    collisions = {
        target: tuple(dict.fromkeys(aliases))
        for target, aliases in aliases_by_target.items()
        if len(set(aliases)) > 1
    }
    if collisions:
        target, aliases = next(iter(collisions.items()))
        raise RuntimeError(
            "ANIMA LoRA aliases normalize to the same Comfy destination "
            f"{target!r}: {', '.join(aliases)}"
        )


def _prepare_comfy_lora_payload(
    model: Any,
    clip: Any,
    state_dict: dict[Any, Any],
) -> tuple[Any, dict[Any, Any], dict[Any, Any], dict[Any, Any]]:
    comfy_lora = importlib.import_module("comfy.lora")
    comfy_lora_convert = importlib.import_module("comfy.lora_convert")
    model_key_map = comfy_lora.model_lora_keys_unet(model.model, {})
    clip_key_map: dict[Any, Any] = {}
    if clip is not None:
        clip_key_map = comfy_lora.model_lora_keys_clip(clip.cond_stage_model, {})
    key_map = dict(model_key_map)
    key_map.update(clip_key_map)
    converted = comfy_lora_convert.convert_lora(state_dict)
    _reject_comfy_alias_collisions(converted, key_map)
    return comfy_lora, converted, model_key_map, clip_key_map


def _load_anima_lora_block_weight(
    model: Any,
    clip: Any,
    prepared: PreparedAnimaLora,
    metadata: Any,
    strength_model: float,
    strength_clip: float,
    inverse: bool,
    seed: int,
    a_value: float,
    b_value: float,
    block_vector: str,
) -> tuple[Any, Any, str]:
    block_count = prepared.report.target_blocks
    if block_count is None:
        raise RuntimeError("ANIMA LoRA block weighting could not determine model depth")
    vector = _parse_anima_block_vector(
        block_vector,
        block_count,
        inverse=bool(inverse),
        seed=int(seed),
        a_value=float(a_value),
        b_value=float(b_value),
    )

    comfy_lora, converted, model_key_map, clip_key_map = (
        _prepare_comfy_lora_payload(model, clip, prepared.state_dict)
    )
    key_map = dict(model_key_map)
    key_map.update(clip_key_map)
    loaded = comfy_lora.load_lora(converted, key_map)
    if not loaded:
        raise RuntimeError(
            "ANIMA LoRA did not produce any Comfy patches for the active MODEL/CLIP"
        )

    model_targets = {
        str(target[0] if isinstance(target, tuple) else target)
        for target in model_key_map.values()
    }
    clip_targets = {
        str(target[0] if isinstance(target, tuple) else target)
        for target in clip_key_map.values()
    }
    def owner_key(value: str, targets: set[str]) -> str | None:
        if value in targets:
            return value
        if value.endswith(".bias"):
            weight_key = value[:-len(".bias")] + ".weight"
            if weight_key in targets:
                return weight_key
        return None

    ambiguous_targets = model_targets.intersection(clip_targets)
    if any(
        owner_key(
            str(key[0] if isinstance(key, tuple) else key),
            ambiguous_targets,
        )
        is not None
        for key in loaded
    ):
        raise RuntimeError(
            "ANIMA LoRA contains a patch destination shared by MODEL and CLIP"
        )

    _require_lora_metadata_support(model, clip, metadata)
    new_model = model.clone()
    new_clip = clip.clone() if clip is not None else None

    def add_verified(target: Any, key: Any, weights: Any, strength: float) -> None:
        accepted = target.add_patches({key: weights}, strength)
        try:
            was_accepted = any(candidate == key for candidate in accepted)
        except TypeError:
            was_accepted = False
        if not was_accepted:
            key_name = str(key[0] if isinstance(key, tuple) else key)
            raise RuntimeError(
                "ANIMA LoRA patch was rejected by the active Comfy model: "
                f"{key_name}"
            )

    for key, weights in loaded.items():
        key_name = str(key[0] if isinstance(key, tuple) else key)
        match = _ANIMA_PATCH_BLOCK_RE.match(key_name)
        if match:
            block_index = int(match.group(1))
            if block_index >= len(vector.blocks):
                raise RuntimeError(
                    f"ANIMA LoRA patch targets out-of-range block {block_index}"
                )
            ratio = vector.blocks[block_index]
        else:
            ratio = vector.base
        if ratio == 0.0:
            continue
        if owner_key(key_name, model_targets) is not None:
            effective_strength = float(strength_model) * ratio
            if effective_strength != 0.0:
                add_verified(new_model, key, weights, effective_strength)
        elif owner_key(key_name, clip_targets) is not None and new_clip is not None:
            effective_strength = float(strength_clip) * ratio
            if effective_strength != 0.0:
                add_verified(new_clip, key, weights, effective_strength)
        else:
            raise RuntimeError(
                f"ANIMA LoRA produced an unknown Comfy patch destination: {key_name}"
            )
    _attach_lora_metadata(new_model, new_clip, metadata)
    return new_model, new_clip, vector.populated


def prepare_selected_lora(
    model: Any,
    lora_name: str,
    *,
    cache: AnimaLoraStateCache | None = None,
) -> tuple[PreparedAnimaLora, Any]:
    """Load and prepare one selected LoRA while keeping the cached raw dict pristine."""

    loader = cache if cache is not None else AnimaLoraStateCache()
    raw_state, metadata = loader.load(str(lora_name))
    prepared = prepare_anima_lora_for_model(raw_state, model)
    if prepared.report.changed:
        LOGGER.info(
            "ANIMA LoRA compatibility: %s %s->%s (%s duplicated keys, %s dropped keys)",
            prepared.report.action,
            prepared.report.source_blocks,
            prepared.report.target_blocks,
            prepared.report.duplicated_tensor_keys,
            prepared.report.dropped_block_keys + prepared.report.dropped_connector_keys,
        )
    return prepared, metadata


def load_lora_for_models(
    model: Any,
    clip: Any,
    lora_name: str,
    strength_model: float,
    strength_clip: float,
    *,
    cache: AnimaLoraStateCache | None = None,
) -> tuple[Any, Any]:
    """Apply a selected LoRA through ComfyUI after ANIMA layout preparation."""

    if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
        return model, clip
    prepared, metadata = prepare_selected_lora(
        model,
        lora_name,
        cache=cache,
    )
    state_dict = prepared.state_dict
    if prepared.report.model_is_anima:
        _comfy_lora, state_dict, _model_map, _clip_map = (
            _prepare_comfy_lora_payload(model, clip, state_dict)
        )
    comfy_sd = importlib.import_module("comfy.sd")
    result = comfy_sd.load_lora_for_models(
        model,
        clip,
        state_dict,
        float(strength_model),
        float(strength_clip),
        lora_metadata=metadata,
    )
    values = node_result(result)
    if len(values) < 2:
        raise RuntimeError("ComfyUI LoRA loader returned fewer than two outputs.")
    return values[0], values[1]


def load_lora_model_only(
    model: Any,
    lora_name: str,
    strength_model: float,
    *,
    cache: AnimaLoraStateCache | None = None,
) -> Any:
    return load_lora_for_models(
        model,
        None,
        lora_name,
        strength_model,
        0.0,
        cache=cache,
    )[0]


def load_lora_block_weight(
    model: Any,
    clip: Any,
    lora_name: str,
    strength_model: float,
    strength_clip: float,
    inverse: bool,
    seed: int,
    a_value: float,
    b_value: float,
    block_vector: str,
    *,
    cache: AnimaLoraStateCache | None = None,
) -> tuple[Any, Any, str]:
    """Apply true per-block ANIMA weights, or preserve Inspire for other models."""

    if float(strength_model) == 0.0 and float(strength_clip) == 0.0:
        return model, clip, ""
    prepared, metadata = prepare_selected_lora(model, lora_name, cache=cache)
    if prepared.report.model_is_anima:
        return _load_anima_lora_block_weight(
            model,
            clip,
            prepared,
            metadata,
            strength_model,
            strength_clip,
            inverse,
            seed,
            a_value,
            b_value,
            block_vector,
        )

    _require_lora_metadata_support(model, clip, metadata)
    provider = provider_class(
        "LoraLoaderBlockWeight //Inspire",
        feature="LoRA block weighting",
    )
    apply_weighted = getattr(provider, "load_lora_for_models", None)
    if not callable(apply_weighted):
        raise RuntimeError(
            "Installed Inspire Pack does not expose raw-state LoRA block weighting. "
            "Update ComfyUI-Inspire-Pack and restart ComfyUI."
        )
    provider_vector = str(block_vector).strip() or _LEGACY_INSPIRE_BLOCK_VECTOR
    values = node_result(
        apply_weighted(
            model,
            clip,
            prepared.state_dict,
            float(strength_model),
            float(strength_clip),
            bool(inverse),
            int(seed),
            float(a_value),
            float(b_value),
            provider_vector,
        )
    )
    if len(values) < 3:
        raise RuntimeError("Inspire LoRA block weighting returned fewer than three outputs.")
    output_model, output_clip = values[0], values[1]
    _attach_lora_metadata(output_model, output_clip, metadata)
    return output_model, output_clip, str(values[2])


class ForgeNeoAnimaLoraLoader:
    def __init__(self) -> None:
        self._cache = AnimaLoraStateCache()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (filename_choices("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads a LoRA and safely adapts ANIMA 28/40/52-block layouts."

    def load_lora(
        self,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
    ):
        return load_lora_for_models(
            model,
            clip,
            lora_name,
            strength_model,
            strength_clip,
            cache=self._cache,
        )


class ForgeNeoAnimaLoraLoaderModelOnly:
    def __init__(self) -> None:
        self._cache = AnimaLoraStateCache()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (filename_choices("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora_model_only"
    CATEGORY = CATEGORY
    DESCRIPTION = "Model-only ANIMA-compatible LoRA loader."

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (
            load_lora_model_only(
                model,
                lora_name,
                strength_model,
                cache=self._cache,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "ForgeNeoAnimaLoraLoader": ForgeNeoAnimaLoraLoader,
    "ForgeNeoAnimaLoraLoaderModelOnly": ForgeNeoAnimaLoraLoaderModelOnly,
}


__all__ = [
    "AnimaLoraStateCache",
    "ForgeNeoAnimaLoraLoader",
    "ForgeNeoAnimaLoraLoaderModelOnly",
    "NODE_CLASS_MAPPINGS",
    "load_lora_block_weight",
    "load_lora_for_models",
    "load_lora_model_only",
    "prepare_selected_lora",
]
