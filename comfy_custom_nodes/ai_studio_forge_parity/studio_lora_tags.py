"""Forge's ``<lora:...>`` prompt-tag rules for the Anima studio (pure: no ComfyUI, no torch).

The app's ``core/comfy_workflow_compiler.parse_lora_tags`` cannot be imported inside ComfyUI, so the same
rules live here; ``tests/test_studio_lora_tags.py`` pins the two to identical results.

Rules: every extra-network tag ``<word:args>`` is removed from the positive prompt, only ``<lora:...>``
is loaded.  ``<lora:n:a:b>`` is text encoder (clip) ``a`` / UNet (model) ``b``; ``<lora:n:a>`` and ``<lora:n>``
give both the same value (``a`` / 1.0).  A tag swallowed by an earlier unclosed ``<x:`` is removed and not
loaded, and ``<lora :x>`` (space before the colon) is plain text.  Negatives only lose ``<lora:...>`` tags
(case-insensitive); the ones named exactly ``lora`` are loaded after the positive ones.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional

_EXTRA_NET_RE = re.compile(r"<(\w+):([^>]+)>")
_NEGATIVE_LORA_RE = re.compile(r"<(lora):([^>]+)>", re.IGNORECASE)
_LORA_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")


@dataclass(frozen=True)
class LoraTag:
    name: str
    strength_model: float
    strength_clip: float


def _finite_float(value: str, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _lora_tag(args: str) -> Optional[LoraTag]:
    positional: list[str] = []
    named: dict[str, str] = {}
    for item in args.split(":"):
        parts = item.split("=", 2)
        if len(parts) == 2:
            named[parts[0]] = parts[1]
        else:
            positional.append(item)
    name = positional[0].strip() if positional else ""
    if not name:
        return None
    text_encoder = 1.0
    if len(positional) > 1:
        text_encoder = 0.0 if "@" in positional[1] else _finite_float(positional[1], 1.0)
    if "te" in named:
        text_encoder = _finite_float(named["te"], text_encoder)
    unet = _finite_float(positional[2], text_encoder) if len(positional) > 2 else text_encoder
    if "unet" in named:
        unet = _finite_float(named["unet"], unet)
    return LoraTag(name, strength_model=unet, strength_clip=text_encoder)


def _tidy(text: str) -> str:
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    return text.strip(" \t\r\n,")


def parse_lora_tags(prompt: str, *negatives: str) -> tuple[list[LoraTag], list[str]]:
    """Ordered LoRA requests and the cleaned prompts (positive first, then each negative)."""
    tags: list[LoraTag] = []

    def found(match: re.Match) -> str:
        if match.group(1) == "lora":
            tag = _lora_tag(match.group(2))
            if tag is not None:
                tags.append(tag)
        return ""

    cleaned = [_tidy(_EXTRA_NET_RE.sub(found, str(prompt or "")))]
    for negative in negatives:
        text = str(negative or "")
        for match in _NEGATIVE_LORA_RE.finditer(text):
            tag = _lora_tag(match.group(2)) if match.group(1) == "lora" else None
            if tag is not None:
                tags.append(tag)
        cleaned.append(_tidy(_NEGATIVE_LORA_RE.sub("", text)))
    return tags, cleaned


def collapse_duplicates(tags: Iterable[LoraTag]) -> list[LoraTag]:
    """One entry per name: the last strengths, at the position of the name's first appearance."""
    merged: dict[str, LoraTag] = {}
    for tag in tags:
        merged[tag.name] = tag
    return list(merged.values())


def _strip_extension(path: str) -> str:
    lowered = path.lower()
    for extension in _LORA_EXTENSIONS:
        if lowered.endswith(extension):
            return path[: -len(extension)]
    return path


def resolve_lora_name(name: str, available: Iterable[str]) -> Optional[str]:
    """The file-list entry a tag name means, or None.

    Like Forge: the name as written, then with a LoRA extension added, then the file stem anywhere
    (also inside subfolders) — all case-insensitive.  Only whole stems match, never part of one.
    """
    wanted = _strip_extension(str(name or "").strip().replace("\\", "/")).casefold()
    if not wanted:
        return None
    entries = [str(entry) for entry in available]
    stems = [(_strip_extension(entry.replace("\\", "/")).casefold(), entry) for entry in entries]
    for stem, entry in stems:                      # exact name (with or without the extension)
        if stem == wanted:
            return entry
    for stem, entry in stems:                      # file stem inside a subfolder; "sub/name" keeps its folder
        if stem.endswith("/" + wanted):
            return entry
    return None


__all__ = ["LoraTag", "parse_lora_tags", "collapse_duplicates", "resolve_lora_name"]
