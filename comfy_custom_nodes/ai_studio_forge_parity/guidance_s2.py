"""S²-Guidance for the suite's SLG: a fresh random set of skipped blocks for every model evaluation.

Paper: "S²-Guidance: Stochastic Self Guidance for Training-Free Enhancement of Diffusion Models",
arXiv 2508.12880 -- ``CFG + omega*(D_cond - D_drop)``, ``D_drop`` the conditional prediction with a
random subset of transformer blocks skipped (the SLG weak row), drawn anew at every evaluation. No
official code was released; this mirrors sam-extra v0.30.0 (``sam3ext/guidance/s2.py`` and the
S² branch of ``scripts/anima_safe_pag.py``), so the same settings draw the same blocks:

- the draw is ``random.Random("s2:<seed>:<pass>:<draw>").sample(eligible, count)`` -- Python seeds a
  ``str`` through SHA-512, so the sequence is the same on every machine. ``seed`` is the generation
  seed (the app compiler passes it as ``guid_s2_seed``), ``pass`` is ``base``/``hires`` and ``draw``
  counts every evaluation of the run from 0 (inside and outside the S² window, Adaptive Guidance's
  cond-only steps included -- the extension draws when the evaluation opens);
- ``count = round(ratio*n)``, at least 1 while ``ratio > 0``; the default eligible set is blocks
  1..N-1 (the paper drops block 0 never);
- the window is the step fraction the extension reads from Forge's step counter, which the sampler
  callback sets after each step: ``max(step - 1, 0)/(steps - 1)`` (``forge_step_fraction``).

Host difference: a second-order sampler's corrector evaluation is placed by its sigma here (Forge's
counter still points at the step it belongs to), so its window edge can differ by one evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .guidance_common import S2_DROP_KEY, _as_bool, _setting, parse_indices
from .guidance_detail import (
    PASS_HIRES,
    _finite_clamp,
    _schedule_values,
    pass_tag,
    sampler_sigma,
    sampling_schedule,
    schedule_index,
)

# The block set one evaluation drops, handed from the suite's SLG weak pass to the block wrappers
# (guidance_dave._patch_anima_blocks) in transformer_options.
DROP_KEY = S2_DROP_KEY

SLG_MODE_FIXED = "Fixed"
SLG_MODE_S2 = "Stochastic (S²)"
DEFAULT_SCALE = 0.25
DEFAULT_RATIO = 0.05
DEFAULT_START = 0.10
DEFAULT_END = 0.90


def is_s2_mode(value: Any) -> bool:
    """The extension's reading of the SLG mode radio: 'stoch…', 's2' or 's²' (case-insensitive)."""
    text = str(value or SLG_MODE_FIXED).strip().lower()
    return text.startswith("stoch") or text in {"s2", "s²"}


def default_eligible(blocks: int) -> set[int]:
    return set(range(1, int(blocks)))


def count_for(ratio: float, eligible: int) -> int:
    eligible = int(eligible)
    if eligible <= 0 or not ratio or ratio <= 0.0:
        return 0
    return max(1, min(eligible, int(round(float(ratio) * eligible))))


def draw_blocks(eligible, ratio: float, seed: int, tag: str, draw: int) -> set[int]:
    pool = sorted({int(index) for index in eligible})
    count = count_for(ratio, len(pool))
    if count <= 0:
        return set()
    rng = random.Random(f"s2:{int(seed)}:{tag}:{int(draw)}")
    return set(rng.sample(pool, count))


def forge_step_fraction(args: dict[str, Any]) -> float:
    """The extension's ``_pct_now`` -- Forge's step counter over ``steps - 1`` -- from the run's schedule.

    The counter is the index of the last finished step (0 before the first callback), so step ``k``'s
    evaluation reads ``max(k - 1, 0)``. ``k`` is the schedule index of the sampler's sigma, or the step
    whose interval contains it."""
    values = _schedule_values(sampling_schedule(args))
    sigma = sampler_sigma(args)
    if not values or len(values) < 2 or sigma is None:
        return 0.0
    steps = len(values) - 1
    k = schedule_index(values, sigma)
    if k is None:
        k = next((i for i in range(steps) if values[i] >= sigma >= values[i + 1]), 0)
    return min(1.0, max(0.0, max(k - 1, 0) / max(steps - 1, 1)))


@dataclass(frozen=True)
class S2Settings:
    scale: float
    ratio: float
    blocks: str
    start: float
    end: float
    seed: int


def s2_settings(settings: dict[str, Any]) -> S2Settings | None:
    """S² settings when the suite's SLG mode is Stochastic, else None (Fixed SLG)."""
    if not is_s2_mode(_setting(settings, "guid_slg_mode", SLG_MODE_FIXED)):
        return None
    start = _finite_clamp(_setting(settings, "guid_s2_start", DEFAULT_START), 0.0, 1.0, DEFAULT_START)
    end = _finite_clamp(_setting(settings, "guid_s2_end", DEFAULT_END), 0.0, 1.0, DEFAULT_END)
    try:
        seed = int(float(_setting(settings, "guid_s2_seed", 0)))
    except (TypeError, ValueError):
        seed = 0
    return S2Settings(
        scale=_finite_clamp(_setting(settings, "guid_s2_scale", DEFAULT_SCALE), 0.0, 5.0, DEFAULT_SCALE),
        ratio=_finite_clamp(_setting(settings, "guid_s2_ratio", DEFAULT_RATIO), 0.0, 0.5, DEFAULT_RATIO),
        blocks=str(_setting(settings, "guid_s2_blocks", "") or ""),
        start=min(start, end), end=max(start, end), seed=seed,
    )


def eligible_blocks(settings: S2Settings, block_count: int) -> set[int]:
    """The pool S² draws from: the block field, or 1..N-1 when it is blank."""
    if settings.blocks.strip():
        return parse_indices(settings.blocks, block_count)
    return default_eligible(block_count)


def s2_active(settings: S2Settings, eligible: set[int]) -> bool:
    """The extension turns SLG on in S² mode only with omega > 0 and at least one block to drop."""
    return settings.scale > 0.0 and count_for(settings.ratio, len(eligible)) > 0


class S2Run:
    """Draw index of the current run, shared by every call of one MODEL; restarts with each run."""

    def __init__(self, settings: S2Settings, eligible: set[int]):
        self.settings = settings
        self.eligible = frozenset(eligible)
        self.draws = 0
        self.tag: str | None = None
        self.last_sigma: float | None = None
        self.last: tuple[int, ...] = ()

    def draw(self, args: dict[str, Any]) -> frozenset:
        """This evaluation's dropped blocks; advances the draw index (call once per evaluation)."""
        sigma = sampler_sigma(args)
        tag = PASS_HIRES if pass_tag(args) == PASS_HIRES else "base"
        if tag != self.tag or sigma is None or self.last_sigma is None or sigma > self.last_sigma + 1e-6:
            self.draws = 0   # a new sampling run (the extension resets per pass)
        self.tag = tag
        if sigma is not None:
            self.last_sigma = sigma
        blocks = draw_blocks(self.eligible, self.settings.ratio, self.settings.seed, tag, self.draws)
        self.draws += 1
        self.last = tuple(sorted(blocks))
        return frozenset(blocks)

    def in_window(self, args: dict[str, Any]) -> bool:
        return self.settings.start <= forge_step_fraction(args) <= self.settings.end


def slg_mode_requested(settings: dict[str, Any]) -> bool:
    return _as_bool(_setting(settings, "guid_slg_on", False)) and is_s2_mode(
        _setting(settings, "guid_slg_mode", SLG_MODE_FIXED))
