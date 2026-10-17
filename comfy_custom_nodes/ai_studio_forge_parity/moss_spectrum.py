"""Spectrum feature forecasting for MOSS-Image (pack 1.9.0): schedule, forecaster, MOSS head, per-run state, wrappers.

A port of the "KSampler (Spectrum for MOSS)" patch of sorryhyun/ComfyUI-Spectrum-KSampler (MIT,
Copyright (c) 2026 sorryhyun; ``LICENSES/ComfyUI-Spectrum-KSampler-MIT.txt``):

origin: sorryhyun/ComfyUI-Spectrum-KSampler@ff25e70203cfd9b83c20307474ceff29dfb2139c:moss.py:57-103 (``_Forecaster``,
the linear form: one weight per stored observation, the prediction their weighted sum in fp32), :112-114 (readiness),
:144-161 (``_moss_head``), :164-177 (``_update_forecasters``), :313-374 (the wrapper's step logic);
spectrum.py:139-154 (``_spectrum_batch_keys``), :488-506 (``SpectrumState.should_cache``), :1036-1049
(``_normalize_cond_or_uncond``), :1052-1092 (``_advance_spectrum_state``, the window bookkeeping). The algorithm is
upstream's: warmup 6, the last 3 steps actual, a window from 2.0 growing by ``flex_window`` after every actual step
past warmup, an actual step every ``floor(window)`` steps, Chebyshev degree 3 ridge 0.1 over the step index blended
with a first-order Taylor step (w 0.3), the last 100 observations; at 50 steps ``flex_window`` 0.25 / 0.5 / 1.0 run
22 / 19 / 16 actual steps (``planned_actual_steps``).

The cached step runs MOSS-Image's own head on the forecast features: the head of ``MOSSImageTransformer.forward`` of
the ``moss-image`` ComfyUI node (OpenMOSS, Apache-2.0, ``LICENSES/moss-image-Apache-2.0.txt``), moss-image 0.1.1
``model.py`` lines 198-209 (time embedding, ``video_head_modulation``) and 242-245 (``final_norm_video``, modulate,
``final_linear_video``, reshape), restated op for op in :func:`moss_head` (batched, as upstream's ``_moss_head``).

What is the pack's own (local integration; upstream patches ``model_function_wrapper`` and keeps one state on the
patched MODEL):

* Hooks that chain: an ``APPLY_MODEL`` wrapper (:func:`spectrum_apply_model_wrapper`) and an ``OUTER_SAMPLE`` wrapper
  (:func:`spectrum_outer_sample_wrapper`), registered under :data:`WRAPPER_KEY` on a MODEL clone. The outer wrapper
  opens a fresh :class:`SpectrumRun` for every sampling run (base, hires, a SAM3/ADetailer detailer pass on the same
  MODEL each start at step 0) under ``transformer_options[RUN_KEY]`` of that run's own options, and in ``finally``
  releases every forecaster tensor. The capture hook on ``diffusion_model.final_norm_video`` exists only during an
  actual forward of a run (registered before the forward, removed in ``finally``).
* Step bookkeeping from the run itself, no steps widget: ``num_steps = len(sample_sigmas) - 1``; the step index is
  the sampler's sigma looked up on ``sample_sigmas`` — the sigma Detail Daemon noted before scaling it
  (``guidance_common.PRE_DD_SIGMAS_KEY``) when present, else the model's sigma. An evaluation whose sigma is not on
  the grid (midpoints of multi-evaluation samplers), or lies behind the current step (a restart sampler going back
  up), runs the real model and neither advances the schedule nor touches a forecaster; so do Euler (SMEA) Dy's extra
  sub-steps (``guidance_common.SUBSTEP_MARKER``), which the Dy guard switches off under Spectrum anyway. One
  exception, for upstream's count: the run's first call may come just below ``sample_sigmas[0]`` — ComfyUI's
  ``offset_first_sigma_for_snr`` starts its SDE and exponential-integrator samplers (dpmpp_2m_sde, dpmpp_3m_sde,
  er_sde, sa_solver, seeds, ...; the pack's dpmpp_4m_sde and flow ODE entries) on a flow model at full denoise at
  ``percent_to_sigma(1e-4)`` instead of 1.0. Upstream counts every new sigma, so that sigma is step 0
  (:func:`offset_start`), and later calls at it belong to step 0 while it lasts.
* Per-branch forecasters keyed like upstream (ComfyUI ``uuids`` with ``cond_or_uncond``, else ``cond_or_uncond``),
  so SkipLayerGuidanceDiTSimple's separate cond and uncond calls each have their own; batch > 1 in batch order.
  A branch is forecast only once it holds ``ready_after`` observations, upstream's readiness rule (upstream's
  cached path checks only that the branch has a forecaster, so a branch that first appears mid-run — a negative
  under ConditioningSetTimestepRange, prompt editing — would be forecast from one): until then it runs the model,
  which makes the step an actual one, as upstream's fallback does. A branch records one observation per step
  (``SpectrumRun.update``): Heun-type correctors call the next step's sigma before that step's own call, and a
  second observation at the same step index would give the Taylor step a zero width (upstream clamps it to 1e-8
  and the forecast explodes).
* A cached step also needs the latent shape the forecasters were fitted on (upstream's conservative-policy check;
  equal in every run without sub-steps). A forecast that raises logs one warning and the rest of the run samples
  without Spectrum.
* :func:`is_moss_model`: a MODEL without the MOSS head (upstream's ``_require_moss`` names) is left unchanged.

torch and ComfyUI are imported when a run starts, never at import (the app imports this module without them).
"""

from __future__ import annotations

import importlib
import logging
import math
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Sequence

from .compat import clone_model, require_torch
from .guidance_common import (
    MOSS_SPECTRUM_ATTR,
    MOSS_SPECTRUM_WRAPPER_KEY,
    MOSS_SPECTRUM_WRAPPER_TYPES,
    PRE_DD_SIGMAS_KEY,
    SUBSTEP_MARKER,
)

__all__ = [
    "ACTUAL",
    "CACHED",
    "DEFAULTS",
    "LOG_PREFIX",
    "MOSS_HEAD",
    "RUN_KEY",
    "WRAPPER_KEY",
    "Forecaster",
    "RunReport",
    "SpectrumPatch",
    "SpectrumRun",
    "SpectrumSettings",
    "WindowSchedule",
    "apply_moss_spectrum",
    "batch_keys",
    "is_moss_model",
    "locate_step",
    "model_spectrum_patch",
    "moss_head",
    "normalize_cond_or_uncond",
    "offset_start",
    "planned_actual_steps",
    "spectrum_apply_model_wrapper",
    "spectrum_outer_sample_wrapper",
]

LOGGER = logging.getLogger("ai_studio_forge_parity")
LOG_PREFIX = "[MOSS Spectrum]"

# ModelPatcher.add_wrapper_with_key key of the node's APPLY_MODEL and OUTER_SAMPLE wrappers (one patch per MODEL);
# both callables carry their SpectrumPatch as MOSS_SPECTRUM_ATTR. The SPEED and Euler Dy guards find the patch by
# exactly these (guidance_common.spectrum_wrapper_registered), never by the word "spectrum".
WRAPPER_KEY = MOSS_SPECTRUM_WRAPPER_KEY
# transformer_options key of one sampling run's SpectrumRun (only in that run's own options).
RUN_KEY = "ai_studio_moss_spectrum_run"
# comfy.patcher_extension.WrappersMP values (literal fallbacks outside ComfyUI).
_APPLY_MODEL_FALLBACK, _OUTER_SAMPLE_FALLBACK = MOSS_SPECTRUM_WRAPPER_TYPES

# origin: moss.py:47-54 — the diffusion_model attributes the cached step uses.
MOSS_HEAD = ("layers", "time_embedding", "video_head_modulation", "final_norm_video", "final_linear_video", "freq_dim")

ACTUAL = "actual"
CACHED = "cached"

# origin: moss.py:255-265 (apply_moss_spectrum_patch defaults) and :425-432 (the node's flex_window).
DEFAULTS: dict[str, Any] = {
    "window_size": 2.0,
    "flex_window": 0.5,
    "warmup_steps": 6,
    "tail_actual_steps": 3,
    "blend_w": 0.3,
    "cheby_degree": 3,
    "ridge_lambda": 0.1,
    "history_size": 100,
}

# A sampler's sigma equals a grid sigma it was read from; the tolerance only absorbs float32 round trips.
SIGMA_TOLERANCE = 1e-6

torch: Any = None


def _torch() -> Any:
    global torch
    if torch is None:
        torch = require_torch()
    return torch


# ------------------------------------------------------------------------------------------------
# Settings and the window schedule (no torch)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpectrumSettings:
    window_size: float = DEFAULTS["window_size"]
    flex_window: float = DEFAULTS["flex_window"]
    warmup_steps: int = DEFAULTS["warmup_steps"]
    tail_actual_steps: int = DEFAULTS["tail_actual_steps"]
    blend_w: float = DEFAULTS["blend_w"]
    cheby_degree: int = DEFAULTS["cheby_degree"]
    ridge_lambda: float = DEFAULTS["ridge_lambda"]
    history_size: int = DEFAULTS["history_size"]

    def __post_init__(self):
        # origin: moss.py:275-279
        if int(self.history_size) < int(self.cheby_degree) + 2:
            raise ValueError(
                "MOSS Spectrum requires history_size >= cheby_degree + 2 "
                f"(got history_size={self.history_size}, cheby_degree={self.cheby_degree})."
            )

    @property
    def ready_after(self) -> int:
        """Observations a branch needs before it may be forecast (origin: moss.py:112-114)."""
        return max(2, int(self.cheby_degree) + 2)

    def summary(self) -> str:
        return (f"window {self.window_size:g}+{self.flex_window:g}, warmup {self.warmup_steps}, "
                f"tail {self.tail_actual_steps}, w {self.blend_w:g}, m {self.cheby_degree}, "
                f"lambda {self.ridge_lambda:g}")


class WindowSchedule:
    """Upstream ``SpectrumState``'s window bookkeeping over one run's step indices.

    origin: spectrum.py:1080-1091 (``_advance_spectrum_state``: the per-step update of the window, run when a new
    step begins) and :488-506 (``should_cache``, the window branch). The sigma comparisons of
    ``_advance_spectrum_state`` are replaced by the grid lookup of :func:`locate_step`; the index jumps to the
    looked-up step (upstream counts one per new sigma, the same in every run that evaluates each step in order).
    """

    def __init__(self, settings: SpectrumSettings, num_steps: int):
        self.settings = settings
        self.num_steps = int(num_steps)
        self.step_idx = -1
        self.mode = ACTUAL
        self.curr_ws = float(settings.window_size)
        self.consec_cached = 0
        self.fwd_count = 0

    def advance(self, index: int) -> None:
        if self.step_idx >= 0:
            if self.mode == ACTUAL:
                self.fwd_count += 1
                if self.step_idx >= self.settings.warmup_steps:
                    self.curr_ws = round(self.curr_ws + self.settings.flex_window, 3)
                self.consec_cached = 0
            else:
                self.consec_cached += 1
        self.step_idx = int(index)

    def should_cache(self, ready: bool) -> bool:
        if self.step_idx < self.settings.warmup_steps:
            return False
        if self.step_idx >= self.num_steps - self.settings.tail_actual_steps:
            return False
        if not ready:
            return False
        return (self.consec_cached + 1) % max(1, math.floor(self.curr_ws)) != 0


def planned_actual_steps(num_steps: int, settings: SpectrumSettings | None = None) -> list[int]:
    """The steps a run of ``num_steps`` runs the model on when every step is evaluated once in order and each actual
    step gives every branch one observation (what a plain KSampler run does)."""
    settings = settings or SpectrumSettings()
    schedule = WindowSchedule(settings, num_steps)
    observations = 0
    actual: list[int] = []
    for index in range(int(num_steps)):
        schedule.advance(index)
        cached = schedule.should_cache(observations >= settings.ready_after)
        schedule.mode = CACHED if cached else ACTUAL
        if not cached:
            observations += 1
            actual.append(index)
    return actual


def locate_step(sigma: float, grid: Sequence[float], current: int) -> int | None:
    """The step index of ``sigma`` on the run's sigma list ``grid`` (``len(grid) - 1`` steps), or None.

    None when no step sigma equals it (a midpoint, a sigma Detail Daemon scaled without a noted one, the final
    sigma) and when every equal step lies behind ``current`` (a restart sampler back above a finished step). Among
    equal step sigmas the current step wins, then the first one after it.
    """
    later = None
    for index in range(len(grid) - 1):
        if not same_sigma(sigma, grid[index]):
            continue
        if index == current:
            return index
        if index > current and later is None:
            later = index
    return later


def same_sigma(sigma: float, value: float) -> bool:
    return abs(sigma - value) <= SIGMA_TOLERANCE * max(1.0, abs(value))


def offset_start(sigma: float, grid: Sequence[float], current: int) -> bool:
    """Is ``sigma`` the run's first sigma, moved by the sampler just below ``grid[0]``?

    ComfyUI's ``offset_first_sigma_for_snr`` (``comfy/k_diffusion/sampling.py``) starts its SDE and
    exponential-integrator samplers on a flow model at full denoise at ``percent_to_sigma(1e-4)`` (about 0.9999)
    instead of 1.0. Upstream counts each new sigma as a step, so its step 0 is that call: before any step began
    (``current`` -1) a sigma strictly between ``grid[1]`` and ``grid[0]`` is step 0 here too. (``grid[0]`` and
    ``grid[1]`` themselves are :func:`locate_step`'s.)
    """
    return current < 0 and len(grid) >= 2 and grid[1] < sigma < grid[0]


def normalize_cond_or_uncond(raw: Any, batch_size: int) -> tuple[list[int], bool]:
    """origin: spectrum.py:1036-1049 — the branch list of a model call and whether it splits the batch evenly."""
    if raw is None:
        raw = [0]
    if hasattr(raw, "detach") and hasattr(raw, "tolist"):
        raw = raw.detach().cpu().tolist()
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    branches = [int(value) for value in raw] or [0]
    return branches, batch_size % len(branches) == 0


def batch_keys(uuids: Any, cond_or_uncond: Sequence[int]) -> list[Hashable]:
    """origin: spectrum.py:139-154 — branch-stable forecaster keys: ``(cond_or_uncond, uuid)`` per chunk, else the
    ``cond_or_uncond`` ints."""
    fallback = [int(value) for value in cond_or_uncond]
    if uuids is None:
        return fallback
    if hasattr(uuids, "detach") and hasattr(uuids, "tolist"):
        uuids = uuids.detach().cpu().tolist()
    try:
        listed = list(uuids)
    except TypeError:
        return fallback
    if len(listed) != len(fallback):
        return fallback
    return [(branch, str(uid)) for branch, uid in zip(fallback, listed)]


# ------------------------------------------------------------------------------------------------
# Forecaster and the MOSS head (torch)
# ------------------------------------------------------------------------------------------------


class Forecaster:
    """Chebyshev (degree ``m``) ridge regression over the step index, blended with a first-order Taylor step by
    ``w``; the last ``k`` observations kept. origin: moss.py:57-103 (``_Forecaster``), unchanged but for ``clear``."""

    def __init__(self, m: int, lam: float, w: float, total_steps: int, k: int):
        self.m, self.lam, self.w, self.total, self.k = int(m), float(lam), float(w), int(total_steps), int(k)
        self.t: list[float] = []
        self.h: list[Any] = []

    @property
    def shape(self):
        return self.h[0].shape if self.h else None

    def update(self, t: float, h: Any) -> None:
        self.t.append(float(t))
        self.h.append(h.detach())
        if len(self.t) > self.k:
            del self.t[0], self.h[0]

    def _design(self, t: Any) -> Any:
        tau = (2.0 * t / self.total - 1.0).reshape(-1, 1)
        cols = [_torch().ones_like(tau), tau]
        for _ in range(2, self.m + 1):
            cols.append(2 * tau * cols[-1] - cols[-2])
        return _torch().cat(cols[: self.m + 1], 1)

    def weights(self, t_star: float) -> Any:
        torch = _torch()
        t = torch.tensor(self.t, dtype=torch.float64)
        x = self._design(t)
        a = x.T @ x + self.lam * torch.eye(x.shape[1], dtype=torch.float64)
        x_star = self._design(torch.tensor([t_star], dtype=torch.float64))[0]
        c = x @ torch.linalg.solve(a, x_star)
        if self.w < 1.0 and len(t) >= 2:
            dt = max(self.t[-1] - self.t[-2], 1e-8)
            kk = (t_star - self.t[-1]) / dt
            taylor = torch.zeros_like(c)
            taylor[-1], taylor[-2] = 1 + kk, -kk
            c = (1 - self.w) * taylor + self.w * c
        return c

    def predict(self, t_star: float) -> Any:
        torch = _torch()
        with torch.no_grad():
            c = self.weights(t_star).tolist()
            out = torch.zeros(self.h[0].shape, dtype=torch.float32, device=self.h[0].device)
            for ck, hk in zip(c, self.h):
                out.add_(hk, alpha=ck)
            return out.to(self.h[0].dtype)

    def clear(self) -> None:
        self.t.clear()
        self.h.clear()


def moss_head(dit: Any, timestep: Any, feat: Any, height: int, width: int) -> Any:
    """MOSS-Image's head on the image-token features ``feat`` (B, H*W, dim) -> (B, C, H, W).

    The tail of ``MOSSImageTransformer.forward`` (moss-image ``model.py`` 198-209 and 242-245, Apache-2.0) with the
    batch done at once (the head is per token), as upstream's ``_moss_head`` (moss.py:144-161). ``timestep`` is the
    model's internal timestep (``model_sampling.timestep(sigma)``).
    """
    torch = _torch()
    batch = feat.shape[0]
    times = timestep.to(feat.device, torch.float32).reshape(-1).expand(batch)
    half = dit.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = dit.time_embedding(embedding.to(dit.time_embedding[0].weight.dtype)).float()
    head = dit.video_head_modulation(time.to(dit.video_head_modulation.weight.dtype)).view(batch, 2, -1)
    x = dit.final_norm_video(feat.to(dit.final_norm_video.weight.dtype))
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    x = dit.final_linear_video(x.to(dit.final_linear_video.weight.dtype))
    return x.transpose(1, 2).reshape(batch, -1, height, width)


def _is_moss_base(base: Any) -> bool:
    dit = getattr(base, "diffusion_model", None)
    if dit is None or getattr(base, "model_sampling", None) is None:
        return False
    return all(hasattr(dit, name) for name in MOSS_HEAD)


def is_moss_model(model: Any) -> bool:
    """Does ``model`` (a ComfyUI MODEL) carry MOSS-Image's head and a ``model_sampling``? (origin: moss.py:131-141)"""
    return _is_moss_base(getattr(model, "model", None))


# ------------------------------------------------------------------------------------------------
# One sampling run
# ------------------------------------------------------------------------------------------------


@dataclass
class RunReport:
    """What one sampling run did (console line and tests; ints only, no tensors)."""

    steps: int = 0                  # len(sample_sigmas) - 1 of the run
    evaluations: int = 0            # model calls seen
    forecast: int = 0               # answered by the forecast + MOSS head
    model: int = 0                  # on-schedule calls that ran the model
    off_grid: int = 0               # sigma not on the grid, or behind the current step: the model, no bookkeeping
    passed: int = 0                 # no run context to use (no sigma list, a Dy sub-step, not a 4-D latent, failed)
    branches: int = 0               # forecasters the run built
    step_modes: dict = field(default_factory=dict)   # step index -> the mode the step ended with
    error: str = ""

    def actual_steps(self) -> list[int]:
        return sorted(index for index, mode in self.step_modes.items() if mode == ACTUAL)

    def cached_steps(self) -> list[int]:
        return sorted(index for index, mode in self.step_modes.items() if mode == CACHED)

    def summary(self) -> str:
        actual = len(self.actual_steps())
        parts = [f"{actual}/{self.steps} steps ran the model, {len(self.cached_steps())} forecast "
                 f"({self.forecast} of {self.evaluations} model calls)"]
        if self.off_grid:
            parts.append(f"{self.off_grid} off-grid calls ran the model")
        if self.passed:
            parts.append(f"{self.passed} calls passed through")
        if self.error:
            parts.append(f"stopped after {self.error}")
        return "; ".join(parts)


def _scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "flatten"):
            return float(value.flatten()[0].item())
        return float(value)
    except Exception:
        return None


class SpectrumRun:
    """The state of one sampling run: its schedule, per-branch forecasters and report."""

    def __init__(self, settings: SpectrumSettings):
        self.settings = settings
        self.schedule: WindowSchedule | None = None
        self.forecasters: dict[Hashable, Forecaster] = {}
        self.input_shape: tuple | None = None
        self.report = RunReport()
        self.closed = False
        self.failed = False
        self._grid_source: Any = None
        self._grid: list[float] = []
        self._start: float | None = None     # the run's first sigma when the sampler moved it below grid[0]
        self._observed: set = set()          # branches that gave the current step its observation

    # -- the run's sigma list ---------------------------------------------------------------------

    def _grid_for(self, sample_sigmas: Any) -> list[float] | None:
        if sample_sigmas is None or getattr(sample_sigmas, "ndim", None) != 1 or len(sample_sigmas) < 2:
            return None
        if sample_sigmas is not self._grid_source:
            self._grid = [float(value) for value in sample_sigmas.detach().cpu().tolist()]
            self._grid_source = sample_sigmas
        if self.schedule is None:
            self.schedule = WindowSchedule(self.settings, len(self._grid) - 1)
            self.report.steps = self.schedule.num_steps
        elif len(self._grid) - 1 != self.schedule.num_steps:
            return None
        return self._grid

    def _locate(self, sigma: float, grid: Sequence[float]) -> int | None:
        """:func:`locate_step`, with the run's first sigma standing for step 0 when the sampler moved it below
        ``grid[0]`` (:func:`offset_start`) — for that call and every later call of step 0 at that sigma."""
        current = self.schedule.step_idx
        index = locate_step(sigma, grid, current)
        if index is not None:
            return index
        if current == 0 and self._start is not None and same_sigma(sigma, self._start):
            return 0
        if offset_start(sigma, grid, current):
            self._start = sigma
            return 0
        return None

    # -- forecasters (origin: moss.py:112-114, :164-177) ------------------------------------------

    def ready(self, keys: Sequence[Hashable]) -> bool:
        return all(key in self.forecasters and len(self.forecasters[key].t) >= self.settings.ready_after
                   for key in keys)

    def update(self, feat: Any, input_x: Any, keys: Sequence[Hashable]) -> None:
        """Record each branch's features at the current step (origin: moss.py:164-177). A branch keeps its first
        observation of a step: another call of the same branch and step (the corrector of Heun-type samplers at the
        next step's sigma, then that step's own call) runs the model without a second one, which would leave the
        Taylor step a zero step width (upstream clamps it to 1e-8 and the extrapolation explodes)."""
        chunks = feat.chunk(len(keys), dim=0)
        if any(forecaster.shape != chunks[0].shape for forecaster in self.forecasters.values()):
            self._clear_forecasters()
        settings = self.settings
        for key, h in zip(keys, chunks):
            if key in self._observed:
                continue
            self._observed.add(key)
            forecaster = self.forecasters.get(key)
            if forecaster is None:
                forecaster = self.forecasters[key] = Forecaster(
                    settings.cheby_degree, settings.ridge_lambda, settings.blend_w, self.schedule.num_steps,
                    settings.history_size,
                )
            forecaster.update(float(self.schedule.step_idx), h)
        self.input_shape = tuple(input_x.shape[1:])
        self.report.branches = max(self.report.branches, len(self.forecasters))

    def _clear_forecasters(self) -> None:
        for forecaster in self.forecasters.values():
            forecaster.clear()
        self.forecasters = {}
        self.input_shape = None
        self._observed = set()

    # -- one model call ----------------------------------------------------------------------------

    def evaluate(self, executor: Any, args: tuple, kwargs: dict, x: Any, t: Any, options: Mapping) -> Any:
        report = self.report
        report.evaluations += 1
        if (self.closed or self.failed or options.get(SUBSTEP_MARKER) or getattr(x, "ndim", None) != 4
                or not _is_moss_base(getattr(executor, "class_obj", None))):
            report.passed += 1
            return executor(*args, **kwargs)
        grid = self._grid_for(options.get("sample_sigmas"))
        noted = options.get(PRE_DD_SIGMAS_KEY)
        sigma = _scalar(noted if noted is not None else t)
        if grid is None or sigma is None:
            report.passed += 1
            return executor(*args, **kwargs)
        schedule = self.schedule
        index = self._locate(sigma, grid)
        if index is None:
            report.off_grid += 1
            return executor(*args, **kwargs)

        cond_or_uncond, valid = normalize_cond_or_uncond(options.get("cond_or_uncond"), int(x.shape[0]))
        keys = batch_keys(options.get("uuids"), cond_or_uncond)
        if index > schedule.step_idx:
            # origin: moss.py:327, :336-337 — a new step decides its mode once.
            schedule.advance(index)
            self._observed = set()
            schedule.mode = CACHED if valid and schedule.should_cache(self.ready(keys)) else ACTUAL
        # origin: moss.py:347-349 (_can_use_cached_prediction), except that every branch must be ready
        # (moss.py:112-114), not merely have a forecaster: a branch new to the run runs the model until it holds
        # ready_after observations.
        if (schedule.mode == CACHED and valid and self.ready(keys)
                and self.input_shape == tuple(x.shape[1:])):
            try:
                result = self._forecast(executor, x, t, keys)
            except Exception as exc:  # the forecast failed: this and every later call run the model
                self._fail(exc)
                report.passed += 1
                return executor(*args, **kwargs)
            report.forecast += 1
            report.step_modes[schedule.step_idx] = CACHED
            return result
        # origin: moss.py:356-374 — an actual forward (also a cached step's call that cannot be forecast).
        schedule.mode = ACTUAL
        report.step_modes[schedule.step_idx] = ACTUAL
        report.model += 1
        return self._actual(executor, args, kwargs, x, keys, valid)

    def _forecast(self, executor: Any, x: Any, t: Any, keys: Sequence[Hashable]) -> Any:
        """origin: moss.py:350-354 — the forecast features through the MOSS head, then ``calculate_denoised``."""
        torch = _torch()
        base = executor.class_obj
        dit, model_sampling = base.diffusion_model, base.model_sampling
        step = float(self.schedule.step_idx)
        feat = torch.cat([self.forecasters[key].predict(step) for key in keys])
        t_internal = model_sampling.timestep(t).float()
        out = moss_head(dit, t_internal, feat, x.shape[-2], x.shape[-1])
        return model_sampling.calculate_denoised(t, out.float(), x)

    def _actual(self, executor: Any, args: tuple, kwargs: dict, x: Any, keys: Sequence[Hashable], valid: bool) -> Any:
        norm = executor.class_obj.diffusion_model.final_norm_video
        feats: list = []

        def capture_spectrum_feature(module, inputs):   # origin: moss.py:117-120
            feats.append(inputs[0].detach().clone())

        handle = norm.register_forward_pre_hook(capture_spectrum_feature)
        try:
            result = executor(*args, **kwargs)
        finally:
            handle.remove()
        try:
            if valid and len(feats) == int(x.shape[0]):
                self.update(_torch().cat(feats), x, keys)
        except Exception as exc:
            self._fail(exc)
        finally:
            feats.clear()
        return result

    def _fail(self, exc: Exception) -> None:
        self.failed = True
        self.report.error = f"{type(exc).__name__}: {exc}"
        self._clear_forecasters()
        LOGGER.warning("%s forecasting failed, the rest of this run samples without it:\n%s", LOG_PREFIX,
                       traceback.format_exc())

    def close(self) -> RunReport:
        """Release every forecaster tensor and the grid copy; the report stays."""
        self.closed = True
        self._clear_forecasters()
        self._grid_source = None
        self._grid = []
        self._start = None
        return self.report


# ------------------------------------------------------------------------------------------------
# Wrappers, the MODEL patch
# ------------------------------------------------------------------------------------------------


class SpectrumPatch:
    """What the node put on a MODEL: the settings, and the last finished run's report (ints only)."""

    def __init__(self, settings: SpectrumSettings):
        self.settings = settings
        self.last_report: RunReport | None = None


def _wrapper_types() -> tuple[str, str]:
    try:
        wrappers = importlib.import_module("comfy.patcher_extension").WrappersMP
        return str(wrappers.APPLY_MODEL), str(wrappers.OUTER_SAMPLE)
    except Exception:
        return _APPLY_MODEL_FALLBACK, _OUTER_SAMPLE_FALLBACK


def _call_parts(args: tuple, kwargs: dict) -> tuple[Any, Any, Mapping]:
    """``(x, t, transformer_options)`` of ``BaseModel.apply_model``'s executor call
    ``(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)``."""
    x = args[0] if args else kwargs.get("x")
    t = args[1] if len(args) > 1 else kwargs.get("t")
    options = args[5] if len(args) > 5 else kwargs.get("transformer_options")
    return x, t, options if isinstance(options, Mapping) else {}


def spectrum_apply_model_wrapper(patch: SpectrumPatch) -> Callable:
    """``WrappersMP.APPLY_MODEL`` wrapper: every model call of a run this patch opened goes through its
    :class:`SpectrumRun`; any other call (no run under ``RUN_KEY``) runs unchanged."""

    def moss_spectrum_apply_model(executor, *args, **kwargs):
        x, t, options = _call_parts(args, kwargs)
        run = options.get(RUN_KEY)
        if not isinstance(run, SpectrumRun) or run.settings is not patch.settings:
            return executor(*args, **kwargs)
        return run.evaluate(executor, args, kwargs, x, t, options)

    setattr(moss_spectrum_apply_model, MOSS_SPECTRUM_ATTR, patch)
    return moss_spectrum_apply_model


def _options_clone(options: Any) -> dict:
    try:
        return importlib.import_module("comfy.model_patcher").create_model_options_clone(options)
    except Exception:
        def copy(value):
            if isinstance(value, dict):
                return {key: copy(item) for key, item in value.items()}
            if isinstance(value, list):
                return list(value)
            return value

        return copy(dict(options or {}))


def spectrum_outer_sample_wrapper(patch: SpectrumPatch) -> Callable:
    """``WrappersMP.OUTER_SAMPLE`` wrapper: a fresh :class:`SpectrumRun` for this sampling run, put under
    ``RUN_KEY`` of a copy of the guider's model options (the MODEL's own options never see it), released in
    ``finally`` with one console line."""

    def moss_spectrum_outer_sample(executor, *args, **kwargs):
        guider = getattr(executor, "class_obj", None)
        original = getattr(guider, "model_options", None)
        if not isinstance(original, dict):
            return executor(*args, **kwargs)
        run = SpectrumRun(patch.settings)
        options = _options_clone(original)
        options.setdefault("transformer_options", {})[RUN_KEY] = run
        guider.model_options = options
        try:
            return executor(*args, **kwargs)
        finally:
            guider.model_options = original
            report = run.close()
            patch.last_report = report
            if report.evaluations:
                LOGGER.info("%s %s", LOG_PREFIX, report.summary())

    setattr(moss_spectrum_outer_sample, MOSS_SPECTRUM_ATTR, patch)
    return moss_spectrum_outer_sample


def model_spectrum_patch(model: Any) -> SpectrumPatch | None:
    """The :class:`SpectrumPatch` :func:`apply_moss_spectrum` put on ``model`` (else None)."""
    wrappers = getattr(model, "wrappers", None)
    if not isinstance(wrappers, dict):
        return None
    for wrapper_type in _wrapper_types():
        registered = wrappers.get(wrapper_type)
        for wrapper in reversed(list((registered or {}).get(WRAPPER_KEY) or ())):
            patch = getattr(wrapper, MOSS_SPECTRUM_ATTR, None)
            if isinstance(patch, SpectrumPatch):
                return patch
    return None


def apply_moss_spectrum(model: Any, settings: SpectrumSettings) -> Any:
    """Clone ``model`` with MOSS Spectrum on its sampling runs (replaces an earlier patch of this node)."""
    patched = clone_model(model, "MOSS Spectrum")
    add = getattr(patched, "add_wrapper_with_key", None)
    remove = getattr(patched, "remove_wrappers_with_key", None)
    if not callable(add) or not callable(remove):
        raise RuntimeError("MOSS Spectrum needs a ComfyUI MODEL that supports model wrappers; update ComfyUI.")
    patch = SpectrumPatch(settings)
    apply_model, outer_sample = _wrapper_types()
    for wrapper_type in (apply_model, outer_sample):
        remove(wrapper_type, WRAPPER_KEY)
    add(outer_sample, WRAPPER_KEY, spectrum_outer_sample_wrapper(patch))
    add(apply_model, WRAPPER_KEY, spectrum_apply_model_wrapper(patch))
    return patched
