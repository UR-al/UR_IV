"""Resolve Forge's random seed (-1) once per ComfyUI job.

Forge picks one concrete seed for ``seed=-1`` and every pass of the job
(base sampler, Hires.fix, ADetailer, SAM3) derives from it; the value is
reported back in the generation info.  The ComfyUI path must do the same:
the compiler writes one concrete seed into every node, the backend reports it
as ``result.info['seed']``, and the saved graph replays exactly.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Optional

from core.comfy_node_classes import SAMPLER_NODES
from core.lenient_numbers import lenient_int

# Forge draws random seeds from the 32-bit range (the app's replay range);
# explicit seeds up to ComfyUI's uint64 limit are passed through unchanged.
RANDOM_SEED_MAX = 2**32 - 1
COMFY_SEED_MAX = 0xFFFFFFFFFFFFFFFF


def concrete_seed(value: Any, *, randint: Optional[Callable[[int, int], int]] = None) -> int:
    """Return ``value`` as a usable seed, drawing a random one for ``< 0``."""
    seed = lenient_int(value, -1)
    if seed < 0:
        draw = randint or random.randint
        return int(draw(0, RANDOM_SEED_MAX))
    return min(seed, COMFY_SEED_MAX)


def with_concrete_seed(payload: Mapping[str, Any]) -> dict:
    """Shallow copy of ``payload`` whose ``seed`` is concrete (caller untouched)."""
    resolved = dict(payload)
    resolved["seed"] = concrete_seed(payload.get("seed", -1))
    return resolved


def graph_main_seed(graph: Mapping[str, Any]) -> Optional[int]:
    """The seed actually queued on the graph's single main sampler.

    ``None`` when the graph has no sampler (post-processing only), several
    samplers (ambiguous) or a linked seed.  ``KSamplerAdvanced`` stores it
    as ``noise_seed``.
    """
    samplers = [
        node for node in graph.values()
        if isinstance(node, Mapping) and node.get("class_type") in SAMPLER_NODES
    ]
    if len(samplers) != 1:
        return None
    inputs = samplers[0].get("inputs", {})
    if not isinstance(inputs, Mapping):
        return None
    for key in ("seed", "noise_seed"):
        value = inputs.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


__all__ = [
    "COMFY_SEED_MAX", "RANDOM_SEED_MAX",
    "concrete_seed", "graph_main_seed", "with_concrete_seed",
]
