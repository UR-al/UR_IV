"""Prompt adherence controls of the Anima studio (pack 1.17.0): the pure parts.

Four controls, all aimed at keeping what the prompt asks for when a LoRA is on:

- **LoRA Base 먼저** (``base_steps``): the first N % of the main sampler's steps run the MODEL without the LoRAs, the
  rest with them. Composition is settled in the early, high-noise steps; the LoRA then paints style onto a layout the
  base model drew from the prompt. ``ForgeNeoKSamplerCNS`` samples it (``studio_adherence_nodes.split_lora_sample``).
- **LoRA 교차 어텐션 제외** (``drop_cross_attn``): the LoRAs leave Anima's text path (its cross-attention and the LLM
  adapter; in the third mode also the cross-attention's adaLN modulation) as the base model has it.
- **CFG 스케줄** (``scheduled_cfg``): a higher CFG in the early steps that falls to the panel's CFG by ``end`` %.
- **단어 주목** (``find_positions``): the cross-attention gives the listed words' T5 tokens ``multiplier`` times their
  attention weight (``log(multiplier)`` added to their logits) between ``start`` and ``end`` %. Prompt weights
  ``(word:1.2)`` scale the token's value (Anima normalizes keys, so its attention share stays); this changes the share.

Progress is measured in steps of the run's own schedule (``step_progress``), so "30 %" means the same steps here as
in the LoRA split (whose two runs both see the whole schedule).

No ComfyUI or torch import: the app and its tests import this module.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence

# Values of the LoRA card's cross-attention pick, in panel order (state ``lora_cross_attn``, node input ``cross_attn``).
CROSS_ATTN_OFF = "끔"
CROSS_ATTN_ONLY = "교차 어텐션"
CROSS_ATTN_WITH_MOD = "교차 어텐션+변조"
CROSS_ATTN_MODES = (CROSS_ATTN_OFF, CROSS_ATTN_ONLY, CROSS_ATTN_WITH_MOD)
LORA_BASE_MAX = 90  # the panel's ceiling: at least one step in ten keeps the LoRA

# Anima/Cosmos key parts (ComfyUI's model keys, e.g. ``diffusion_model.blocks.3.cross_attn.k_proj.weight``). The LLM
# adapter is the text path as a whole: all of it or none of it, never a half-LoRA mix it was not trained as.
_CROSS_ATTN_PART = ".cross_attn."
_CROSS_MOD_PART = ".adaln_modulation_cross_attn."
_TEXT_ADAPTER_PART = ".llm_adapter."

# model_options key the LoRA control node writes for the main sampler: {"base_percent": int, "spans": {key: (a, b)}}.
LORA_SCHEDULE_KEY = "ai_studio_lora_schedule"

CFG_CURVES = ("코사인", "선형", "계단")


# ---------------------------------------------------------------------------------------------- LoRA control

def cross_attn_key(key: str, mode: str) -> bool:
    """Is model key ``key`` part of what ``mode`` keeps free of LoRAs?"""
    if mode == CROSS_ATTN_OFF:
        return False
    text = f".{key}"
    if _CROSS_ATTN_PART in text or _TEXT_ADAPTER_PART in text:
        return True
    return mode == CROSS_ATTN_WITH_MOD and _CROSS_MOD_PART in text


def lora_spans(before: Mapping[str, Sequence], after: Mapping[str, Sequence]) -> dict[str, tuple[int, int]]:
    """Per model key, the patch entries ``after`` has beyond ``before``: ``(start, end)`` list positions.

    ``before`` is the MODEL before the LoRA nodes, ``after`` the one after them. The nodes in between only append to a
    key's list (DoRA mode rewrites entries in place), so the extra entries are the tail.
    """
    spans = {}
    for key, items in after.items():
        start, end = len(before.get(key) or ()), len(items or ())
        if end > start:
            spans[key] = (start, end)
    return spans


def without_spans(patches: Mapping[str, Sequence], spans: Mapping[str, tuple[int, int]]) -> dict[str, list]:
    """``patches`` with the ``spans`` entries removed; a key left with nothing is dropped."""
    out = {}
    for key, items in patches.items():
        items = list(items)
        if key in spans:
            start, end = spans[key]
            items = items[:start] + items[end:]
        if items:
            out[key] = items
    return out


def drop_cross_attn(patches: Mapping[str, Sequence], spans: Mapping[str, tuple[int, int]], mode: str):
    """(patches, spans, keys) with the LoRA entries of the keys ``mode`` keeps free removed.

    Only the LoRA entries go (the span); entries that were there before the LoRA nodes stay. ``keys`` is how many model
    keys lost their LoRA entries.
    """
    dropped = {key: span for key, span in spans.items() if cross_attn_key(key, mode)}
    if not dropped:
        return {key: list(items) for key, items in patches.items()}, dict(spans), 0
    out = without_spans(patches, dropped)
    return out, {key: span for key, span in spans.items() if key not in dropped}, len(dropped)


def base_steps(percent, steps) -> int:
    """How many of ``steps`` the base model runs for ``percent`` % (rounded; 0 = no split). Below 100 % the last step
    always keeps the LoRAs; 100 % is every step without them."""
    steps = max(0, int(steps))
    percent = min(100.0, max(0.0, float(percent)))
    split = min(steps, int(math.floor(steps * percent / 100.0 + 0.5)))
    return min(split, steps - 1) if percent < 100.0 and steps > 0 else split


# ---------------------------------------------------------------------------------------------- progress

def _float(value) -> float:
    try:
        return float(value.flatten()[0].item())
    except AttributeError:
        return float(value)


def step_progress(sigma, sigmas) -> float:
    """Where ``sigma`` sits in the run, 0 at the first step: ``step / steps``.

    ``step`` is the index of the last schedule entry at or above ``sigma`` (a second-order sampler's end-of-step call
    at ``sigmas[i + 1]`` counts as step ``i + 1``), at most the last step. No schedule: 0.
    """
    try:
        values = [_float(item) for item in sigmas]
    except TypeError:
        return 0.0
    count = len(values) - 1
    if count <= 0:
        return 0.0
    current = _float(sigma)
    index = 0
    for position in range(count):
        if values[position] >= current - 1e-6 * max(1.0, abs(current)):
            index = position
        else:
            break
    return min(1.0, max(0.0, index / count))


def scheduled_cfg(progress: float, base_cfg: float, start_cfg: float, end_percent: float, curve: str) -> float:
    """The CFG at ``progress`` (0..1): ``start_cfg`` at the start, ``base_cfg`` from ``end_percent`` % on.

    ``코사인`` eases from one to the other, ``선형`` goes straight, ``계단`` holds ``start_cfg`` until the end. An end
    of 0 % is the panel's CFG throughout.
    """
    end = min(1.0, max(0.0, float(end_percent) / 100.0))
    if end <= 0.0 or progress >= end:
        return float(base_cfg)
    if curve == "계단":
        return float(start_cfg)
    t = min(1.0, max(0.0, float(progress) / end))
    weight = (1.0 - t) if curve == "선형" else 0.5 * (1.0 + math.cos(math.pi * t))
    return float(base_cfg) + (float(start_cfg) - float(base_cfg)) * weight


def in_window(progress: float, start_percent: float, end_percent: float) -> bool:
    """``start`` % <= progress < ``end`` % (an end of 100 % includes the last step)."""
    start, end = float(start_percent) / 100.0, float(end_percent) / 100.0
    return start <= progress < end or (end >= 1.0 and progress >= start)


# ---------------------------------------------------------------------------------------------- word emphasis

_SPLIT = re.compile(r"[,\n]")


def emphasis_phrases(text: str) -> list[str]:
    """The comma or line separated phrases of the 단어 field, blanks and repeats left out."""
    out = []
    for part in _SPLIT.split(text or ""):
        phrase = " ".join(part.split())
        if phrase and phrase not in out:
            out.append(phrase)
    return out


def find_positions(ids: Iterable[int], sequences: Iterable[Sequence[int]]) -> list[int]:
    """Sorted positions in ``ids`` covered by any occurrence of any token sequence of ``sequences``."""
    tokens = [int(token) for token in ids]
    found: set[int] = set()
    for sequence in sequences:
        sequence = [int(token) for token in sequence]
        size = len(sequence)
        if not size:
            continue
        for start in range(len(tokens) - size + 1):
            if tokens[start:start + size] == sequence:
                found.update(range(start, start + size))
    return sorted(found)


def emphasis_logit(multiplier: float) -> float:
    """The logit offset that multiplies a token's attention weight by ``multiplier`` before renormalizing."""
    return math.log(max(1e-6, float(multiplier)))
