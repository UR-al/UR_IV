"""The SPEED sampling run for the pack's ``ForgeNeoAnimaSpeed`` (pack 1.7.0).

``run_speed`` wraps a k-diffusion style ``sample_*(model, x, sigmas, extra_args, callback, disable,
**kw)`` and calls it once per segment of a :class:`speed_schedule.Plan`: the start latent is
DCT-truncated to the first grid, each coarse segment runs, the latent is spectrally expanded at the
hand-off (``speed_spectral``), the schedule is patched (``transition``: only the hand-off sigma;
``respace``: the rest is re-spaced too) and published, and the full-size tail finishes the pass.
Host specifics (seeds, noise samplers, where the patched schedule is published) come through
:class:`SpeedHost`, so the run itself needs no ComfyUI and is checked on the CPU.

The behaviour is sam-extra's (``forge_sam3_extension`` ``sam3ext/speed/runner.py``): every random
draw is per image seed (a batch image equals the same seed generated alone) — ``transition`` draws
the expansion noise with numpy ``default_rng(seed + (k + 1) * 10000)`` per image in upstream's
per-channel order, ``respace`` keeps one ``torch.Generator(seed + 10000)`` per image on the latent's
device; coarse segments get coarse noise samplers from the host in place of the full-grid one; an
img2img/hires start (``init_latent``, sigma0 < 1) is rescaled to the flow form ``(1 - s0) y + s0 eps``
after truncation (``img2img_rescale``, sam-extra's option, default on) with per-seed top-up noise
from a CPU generator at ``seed + 7919``; the patched schedule is published after every hand-off and
restored after the run; a sampler that names ``STEP_OFFSET_KWARG`` learns where each segment starts
in the pass; the run reports what it did for the status line. The code is the pack's own, written
from the MIT upstreams (full notices in THIRD_PARTY_NOTICES.md, texts in LICENSES/):

* howardhx/speed@ca7801c9bdffe681742e9592345bcf4885959be5 ``comfyui/speed_sampler.py`` (the segmented
  run: segment callback, expansion seed ``seed + (k + 1) * 10000``, only the hand-off sigma patched)
  — Copyright (c) 2026 Howard Xiao;
* aoleg/ComfyUI-SPEED@a8873591a27f2c1e086a2caf546f9b6aeec62b81 ``speed_core.py`` ``sample_speed_core``
  (sampler kwargs to every segment, the full-grid noise sampler only to the full-size one, 5-D
  latents) — Copyright (c) 2026 A. Izzuddin Al Faruq (fork work by Oleg Afonin);
* sorryhyun/ComfyUI-Spectrum-KSampler@b46a364aec3b161b889c9cc26cd976a49eb537ae ``spd.py``
  ``make_speed_sampler`` (the respace loop, the expansion generator ``seed + 10_000``) — Copyright
  (c) 2026 sorryhyun.

The one ComfyUI-specific rule: whether the sampler names ``STEP_OFFSET_KWARG`` is read through
``guidance_common.declares_keyword``, which follows the pack's sampler wrappers (CNS sets
``_forge_neo_base``) to the real ``sample_*``. ``tests/test_comfy_speed_parity.py`` runs this module
and the installed extension's runner on the same toy model and compares every output bit, report,
published schedule and model call. numpy and torch are imported when a run starts, never at import.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from . import speed_spectral
from .compat import require_torch
from .guidance_common import STEP_OFFSET_KWARG, declares_keyword
from .speed_schedule import Plan, Transition

__all__ = [
    "EXPANSION_SEED_STRIDE",
    "INIT_NOISE_SEED_OFFSET",
    "RESPACE_SEED_OFFSET",
    "STEP_OFFSET_KWARG",
    "RunReport",
    "SpeedHost",
    "SpeedRun",
    "run_speed",
]

# official comfyui/speed_sampler.py / aoleg speed_core.py: transition k draws at seed + (k + 1) * 10000.
EXPANSION_SEED_STRIDE = 10000
# sorryhyun spd.py: the respace expansion generator is seeded at seed + 10_000 (one per image here).
RESPACE_SEED_OFFSET = 10_000
# sam-extra: the img2img top-up noise comes from a CPU generator at seed + 7919 per image.
INIT_NOISE_SEED_OFFSET = 7919


def _require_numpy():
    import importlib

    try:
        return importlib.import_module("numpy")
    except Exception as exc:  # pragma: no cover - ComfyUI always ships numpy
        raise RuntimeError("Anima SPEED needs numpy and must run inside ComfyUI.") from exc


class SpeedHost:
    """What a run needs from its sampling host. These defaults need no host (tests, scripts)."""

    def seeds(self, batch: int) -> list[int]:
        return list(range(batch))

    def noise_scope(self, full_shape: tuple[int, ...]):
        """Context the whole run executes in (sam-extra's Forge host installs its noise hijack here)."""
        return contextlib.nullcontext()

    def coarse_noise_sampler(self, template, x, stage: int, sigmas):
        """A noise sampler for coarse segment ``stage`` in place of ``template`` (made for the full
        grid); None lets the sampler build its default one."""
        return None

    def publish_sigmas(self, sigmas) -> None:
        """The pass schedule as it is now (after a hand-off patched it)."""

    def restore_sigmas(self) -> None:
        """Undo :meth:`publish_sigmas` once the run is over (also after a failure)."""

    def log(self, message: str) -> None:
        pass


@dataclass
class SpeedRun:
    """Per-pass inputs that are not part of the plan."""

    plan: Plan
    seeds: list[int]
    init_latent: bool = False          # img2img / hires: the start state holds an image
    img2img_rescale: bool = True       # sam-extra's init correction (its option, default on)


@dataclass
class RunReport:
    applied: bool = False
    coarse_steps: int = 0
    total_steps: int = 0
    events: list = field(default_factory=list)   # one dict per hand-off
    init: str = "txt2img"

    def summary(self, plan: Plan) -> str:
        """The status text: ``5/28 steps coarse, 52x76 -> 104x152, step 5 sigma 0.9600->0.9796``."""
        grids = [plan.first_grid] + [t.grid_to for t in plan.transitions]
        parts = [f"{self.coarse_steps}/{self.total_steps} steps coarse", " -> ".join(f"{h}x{w}" for h, w in grids)]
        parts.extend(f"step {e['step']} sigma {e['sigma']:.4f}->{e['aligned']:.4f}" for e in self.events)
        if self.init != "txt2img":
            parts.append(f"init={self.init}")
        return ", ".join(parts)


# ------------------------------------------------------------------------------------------------
# Expansion noise, per image seed
# ------------------------------------------------------------------------------------------------


def _item_layout(x) -> tuple[int, ...]:
    """Axes of one image between the batch and the grid: ``(C,)`` for 4-D, ``(C, T)`` for 5-D."""
    return tuple(int(v) for v in x.shape[1:-2])


def _numpy_draws(seed: int, lead: tuple[int, ...], shapes: list[tuple[int, int]]) -> list:
    """Transition-mode noise for one image, in the upstream draw order.

    One ``np.random.default_rng(seed)`` per image (numpy takes no negative entropy: a negative seed
    is used modulo 2**64); for each channel — visited as the official code visits its 4-D view, so a
    5-D ``(C, T)`` image goes ``(T, C)`` — one ``standard_normal`` per entry of ``shapes``, as float32.
    """
    np = _require_numpy()
    torch = require_torch()
    seed = int(seed)
    rng = np.random.default_rng(seed % 2 ** 64 if seed < 0 else seed)
    order = (lead[1], lead[0]) if len(lead) == 2 else tuple(lead)
    arrays = [np.empty(order + tuple(shape), dtype=np.float32) for shape in shapes]
    for index in np.ndindex(*order):
        for array, shape in zip(arrays, shapes):
            array[index] = rng.standard_normal(tuple(shape)).astype(np.float32)
    draws = [torch.from_numpy(array) for array in arrays]
    if len(lead) == 2:
        draws = [draw.permute(1, 0, 2, 3) for draw in draws]
    return draws


def _torch_draws(generator, lead: tuple[int, ...], shapes: list[tuple[int, int]], device) -> list:
    """Respace-mode noise for one image: sorryhyun's ``torch.randn((1, C, H, W), generator)``."""
    torch = require_torch()
    return [
        torch.randn((1, *lead, *shape), generator=generator, device=device, dtype=torch.float32)[0]
        for shape in shapes
    ]


def _draw_shapes(transform: str, grid_from, grid_to) -> list[tuple[int, int]]:
    if transform == "dwt":
        return [tuple(grid_from)] * 3            # LH, HL, HH on the source grid (official order)
    if transform == "fft":
        return [tuple(grid_to)] * 2              # real and imaginary parts
    return [tuple(grid_to)]


def _expand(transform: str, x, grid_to, sigma: float, draws: list):
    if transform == "dct":
        return speed_spectral.expand_dct(x, grid_to, sigma, draws[0])
    if transform == "fft":
        return speed_spectral.expand_fft(x, grid_to, sigma, draws[0], draws[1])
    return speed_spectral.expand_haar(x, sigma, draws[0], draws[1], draws[2])


def _batch(per_image: list[list], device) -> list:
    """Per-image draw lists -> one batched tensor per draw."""
    torch = require_torch()
    return [torch.stack([draws[k] for draws in per_image]).to(device=device) for k in range(len(per_image[0]))]


# ------------------------------------------------------------------------------------------------
# The run
# ------------------------------------------------------------------------------------------------


def _segment_callback(outer: Callable | None, start: int):
    """Report a segment's steps as steps of the whole pass (official ``_segment_callback``)."""
    if outer is None:
        return None

    def rebased(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + start
        return outer(d)

    return rebased


def _init_coarse(x, run: SpeedRun, sigma0: float, report: RunReport):
    """The start latent on the first grid (DCT truncation), flow-rescaled for an img2img start."""
    torch = require_torch()
    plan = run.plan
    grid = plan.first_grid
    if tuple(x.shape[-2:]) == tuple(grid):
        return x
    work = speed_spectral.dct_downscale(x, grid)
    if run.init_latent and sigma0 < 1.0 - 1e-6:
        if run.img2img_rescale:
            # truncation leaves the image part sqrt(HW / hw) times too strong for the coarse grid;
            # bring it back to (1 - s0) y and top the noise up to s0 with fresh per-seed noise
            full_h, full_w = plan.full_grid
            r_eff = math.sqrt((full_h * full_w) / (grid[0] * grid[1]))
            top_up = sigma0 * math.sqrt(max(0.0, 1.0 - 1.0 / (r_eff * r_eff)))
            noise = []
            for seed in run.seeds:
                generator = torch.Generator(device="cpu").manual_seed(int(seed) + INIT_NOISE_SEED_OFFSET)
                noise.append(torch.randn(tuple(work.shape[1:]), generator=generator, dtype=torch.float32))
            noise = torch.stack(noise).to(device=work.device, dtype=work.dtype)
            work = work / r_eff + top_up * noise
            report.init = f"rescaled 1/{r_eff:.3f}"
        else:
            report.init = "upstream (not rescaled)"
    return work.to(dtype=x.dtype)


def _apply_transition(x_cur, transition: Transition, plan: Plan, sig, seeds: list[int], generators: list,
                      lead: tuple[int, ...], device, dtype):
    """Expand ``x_cur`` at ``transition`` and patch the working schedule ``sig`` in place."""
    step = transition.step
    r = transition.scale_to / transition.scale_from
    shapes = _draw_shapes(plan.transform, transition.grid_from, transition.grid_to)
    if tuple(x_cur.shape[-2:]) != tuple(transition.grid_from):
        raise RuntimeError(f"SPEED: latent {tuple(x_cur.shape[-2:])} is not the planned {transition.grid_from}")
    if plan.mode == "respace":
        sigma = float(sig[step])
        aligned, factor = speed_spectral.respace_alignment(sigma, r)
        draws = _batch([_torch_draws(g, lead, shapes, device) for g in generators], device)
        expanded = _expand(plan.transform, x_cur, transition.grid_to, sigma, draws)
        x_new = (expanded * factor).to(dtype=dtype)
        if sigma > 0.0 and aligned != sigma:
            sig[step + 1:] = aligned * (sig[step + 1:] / sigma)
        sig[step] = aligned
    else:
        sigma = float(sig[step])
        offset = (transition.k + 1) * EXPANSION_SEED_STRIDE
        draws = _batch([_numpy_draws(seed + offset, lead, shapes) for seed in seeds], device)
        expanded = _expand(plan.transform, x_cur, transition.grid_to, sigma, draws)
        x_new = (speed_spectral.kappa(sigma, r) * expanded).to(dtype=dtype)
        sig[step] = float(speed_spectral.align_timestep(sigma, r))
    event = {"step": step, "k": transition.k, "sigma": sigma, "aligned": float(sig[step]),
             "grid": tuple(transition.grid_to)}
    return x_new, event


def run_speed(
    base_fn: Callable,
    model,
    x,
    sigmas,
    *,
    run: SpeedRun,
    host: SpeedHost | None = None,
    extra_args: dict | None = None,
    callback: Callable | None = None,
    disable=None,
    sampler_kwargs: dict | None = None,
):
    """Run ``base_fn`` segment by segment along ``run.plan``; returns ``(x, report)``.

    ``sampler_kwargs`` reach every segment, except ``noise_sampler``: the one given is made for the
    full grid, so only the full-size segment gets it and each coarse stage asks the host for its own.
    """
    torch = require_torch()
    host = host or SpeedHost()
    plan = run.plan
    extra_args = {} if extra_args is None else extra_args
    kwargs = dict(sampler_kwargs or {})
    full_grid_noise = kwargs.pop("noise_sampler", None)
    report = RunReport(total_steps=len(sigmas) - 1)
    if x.ndim not in (4, 5):
        raise ValueError(f"SPEED expects a 4-D or 5-D latent, got {tuple(x.shape)}")
    if tuple(x.shape[-2:]) != tuple(plan.full_grid):
        raise ValueError(f"plan grid {plan.full_grid} does not match the latent {tuple(x.shape[-2:])}")
    batch = int(x.shape[0])
    seeds = [int(s) for s in run.seeds]
    if len(seeds) != batch:
        raise ValueError(f"{len(seeds)} seeds for a batch of {batch}")
    lead = _item_layout(x)
    full_shape = tuple(x.shape)
    device = x.device

    if plan.mode == "respace":
        sig = sigmas.detach().clone().float()           # sorryhyun's working copy
        generators = [torch.Generator(device=device).manual_seed(seed + RESPACE_SEED_OFFSET) for seed in seeds]
    else:
        sig = sigmas.clone()                            # official: the schedule's own dtype
        generators = []

    sigma0 = float(sigmas[0])
    coarse_noise: dict[int, Any] = {}
    with_offset = declares_keyword(base_fn, STEP_OFFSET_KWARG)

    def segment(x_cur, start: int, end: int, stage: int, full: bool):
        seg = sig[start:end + 1]
        if len(seg) < 2:
            return x_cur
        seg_kwargs = dict(kwargs)
        if with_offset:
            seg_kwargs[STEP_OFFSET_KWARG] = start
        if full_grid_noise is not None:
            if full:
                seg_kwargs["noise_sampler"] = full_grid_noise
            else:
                if stage not in coarse_noise:
                    coarse_noise[stage] = host.coarse_noise_sampler(full_grid_noise, x_cur, stage, sigmas)
                if coarse_noise[stage] is not None:
                    seg_kwargs["noise_sampler"] = coarse_noise[stage]
        return base_fn(
            model, x_cur, seg, extra_args=extra_args,
            callback=_segment_callback(callback, start), disable=disable, **seg_kwargs,
        )

    with host.noise_scope(full_shape):
        try:
            x_cur = _init_coarse(x, run, sigma0, report)
            start = 0
            for stage, transition in enumerate(plan.transitions):
                x_cur = segment(x_cur, start, transition.step, stage, full=False)
                x_cur, event = _apply_transition(x_cur, transition, plan, sig, seeds, generators, lead, device, x.dtype)
                report.events.append(event)
                host.publish_sigmas(sig)
                start = transition.step
            x_cur = segment(x_cur, start, len(sig) - 1, len(plan.transitions), full=True)
        finally:
            host.restore_sigmas()
    if tuple(x_cur.shape) != full_shape:
        raise RuntimeError(f"SPEED ended on {tuple(x_cur.shape)}, expected {full_shape}")
    report.applied = True
    report.coarse_steps = plan.coarse_steps
    return x_cur, report
