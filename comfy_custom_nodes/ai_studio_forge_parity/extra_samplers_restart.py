"""Extra Samplers for ComfyUI (pack 1.7.0): ``restart_flow`` — Restart sampling that keeps a flow model on its path.

Restart sampling (Y. Xu, M. Deng, X. Cheng, Y. Tian, Z. Liu, T. Jaakkola, "Restart Sampling for Improving
Generative Processes", NeurIPS 2023, arXiv:2306.14878, Algorithm 1) alternates a deterministic ODE solver with a
few large forward jumps that add back noise: after the ODE reaches the low end of a restart interval it jumps to
the high end by the forward process and solves the interval again, ``K`` times. sam-extra v0.33.0's "Restart
(flow)" applies it to the schedule the sampler is given, in the ε-equivalent noise level ``s = σ/α`` (``α = 1 −
σ`` on a rectified-flow model, ``α = 1`` otherwise), so on Anima the jump is the flow model's own forward
transition instead of the VE one Forge's built-in Restart uses (which leaves a flow latent off its path). This
module is sam-extra's implementation of the paper and the rule below (sam-extra's code, licensed to this pack
under MIT by its author (THIRD_PARTY_NOTICES.md); the installed extension is also the test oracle); the paper's
reference code (unlicensed) was not used.

* **Plan** (``restart_plan``): ``n`` steps; fewer than ``RESTART_MIN_STEPS`` → no restart. ``lo`` = the grid index in
  ``1..n−1`` whose ``s`` is closest to ``RESTART_S_MIN``, ``hi`` = the index in ``0..lo−1`` whose ``s`` is closest
  to ``RESTART_S_MAX`` (σ = 1 on a flow model is ``s = ∞`` and never a restart point); the interval must exist
  (``s_lo < s_hi < ∞``). ``K`` = ``RESTART_TIMES``, ``RESTART_TIMES_LONG`` from ``RESTART_LONG_FROM`` steps (the
  paper's [0.1, 2] window in VE σ, and the step rules of the A1111 Restart sampler that sam-extra keeps).
* **Steps**: every step — main path and re-walk — is a Heun step of the host's ``sample_heun`` at ``s_churn = 0``
  (the paper's second-order ODE solver; Euler for the step that reaches σ = 0). Without a plan the sampler *is*
  ``sample_heun``.
* **Restart**: once the main path has reached grid index ``lo``, ``K`` times: the forward jump
  ``x ← (α_hi/α_lo)·x + α_hi·√(s_hi² − s_lo²)·ε·s_noise`` — the exact transition from σ_lo to σ_hi of
  ``x = α·x₀ + σ·n`` (on ε models ``x ← x + √(σ_hi² − σ_lo²)·ε·s_noise``) — then the grid steps ``hi … lo−1`` again.
  The scheduler's own σ are re-walked (no new sub-grid), so the shift of a flow schedule is kept. ``ε`` is one draw
  of the run's noise sampler (ComfyUI's ``default_noise_sampler(x, seed)``), called with ``(σ_lo, σ_hi)``.
* **Callback**: ``i`` is the grid index of the step being taken (it steps back during a restart).
* **Host difference**: Forge rewinds its denoiser's step counter during a re-walk (prompt editing, Skip Early CFG);
  ComfyUI schedules prompts by σ and keeps no such counter, so there is nothing to rewind — same latents.

``tests/test_comfy_extra_samplers_parity.py`` runs this and sam-extra's ``restart_flow`` on Forge's k-diffusion module
and compares the latents bit for bit; ``kd`` (default ``comfy.k_diffusion.sampling``) is the k-diffusion module. No
torch or ComfyUI import at module level.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, NamedTuple

from .extra_samplers_dy import is_const
from .extra_samplers_math import default_noise_sampler, k_diffusion, model_sampling

# The restart window in s = σ/α (the paper's VE σ window [0.1, 2]); sam-extra restart_flow.py, pinned by the tests.
RESTART_S_MIN = 0.1
RESTART_S_MAX = 2.0
RESTART_MIN_STEPS = 20       # fewer steps: no restart
RESTART_TIMES = 1            # K
RESTART_TIMES_LONG = 2       # K from RESTART_LONG_FROM steps
RESTART_LONG_FROM = 36


class RestartPlan(NamedTuple):
    """After the main step that reaches grid index ``lo``: jump to index ``hi`` and walk back to ``lo``, ``times`` times."""

    lo: int
    hi: int
    times: int


def noise_level(sigma: float, flow: bool) -> float:
    """``s = σ/α`` with ``α = 1 − σ`` on a flow model and 1 otherwise; ``∞`` where ``α ≤ 0``."""
    alpha = 1.0 - sigma if flow else 1.0
    if alpha <= 0.0:
        return math.inf
    return sigma / alpha


def _closest(levels: list[float], indices: range, target: float) -> int:
    best = indices[0]
    for index in indices:
        if abs(levels[index] - target) < abs(levels[best] - target):
            best = index
    return best


def restart_plan(sigmas: Any, flow: bool) -> RestartPlan | None:
    """The restart interval of ``sigmas`` (module docstring), or None."""
    values = [float(value) for value in sigmas]
    steps = len(values) - 1
    if steps < RESTART_MIN_STEPS:
        return None
    levels = [noise_level(value, flow) for value in values]
    lo = _closest(levels, range(1, steps), RESTART_S_MIN)
    hi = _closest(levels, range(0, lo), RESTART_S_MAX)
    if not (levels[lo] < levels[hi] < math.inf):
        return None
    return RestartPlan(lo=lo, hi=hi, times=RESTART_TIMES_LONG if steps >= RESTART_LONG_FROM else RESTART_TIMES)


def restart_kernel(sigma_lo: Any, sigma_hi: Any, flow: bool) -> tuple[Any, Any]:
    """``(scale, noise_std)`` of the forward jump σ_lo → σ_hi: ``x ← scale·x + noise_std·ε`` (before ``s_noise``).

    Computed in the arithmetic of the values given (the sampler passes the schedule's 0-dim tensors, so the
    coefficients are rounded like the schedule); on an ε model ``scale`` is exactly 1."""
    if not flow:
        return 1.0, (sigma_hi ** 2 - sigma_lo ** 2) ** 0.5
    alpha_lo, alpha_hi = 1.0 - sigma_lo, 1.0 - sigma_hi
    s_lo, s_hi = sigma_lo / alpha_lo, sigma_hi / alpha_hi
    return alpha_hi / alpha_lo, alpha_hi * (s_hi ** 2 - s_lo ** 2) ** 0.5


def _heun_step(kd: Any, model: Any, x: Any, sigmas: Any, j: int, s_in: Any, extra_args: dict, callback: Any) -> Any:
    """Grid step ``j`` of the host's ``sample_heun`` at ``s_churn = 0`` (Euler when it reaches σ = 0)."""
    sigma_hat = sigmas[j]
    denoised = model(x, sigma_hat * s_in, **extra_args)
    d = kd.to_d(x, sigma_hat, denoised)
    if callback is not None:
        callback({"x": x, "i": j, "sigma": sigmas[j], "sigma_hat": sigma_hat, "denoised": denoised})
    dt = sigmas[j + 1] - sigma_hat
    if sigmas[j + 1] == 0:
        return x + d * dt
    x_2 = x + d * dt
    denoised_2 = model(x_2, sigmas[j + 1] * s_in, **extra_args)
    d_2 = kd.to_d(x_2, sigmas[j + 1], denoised_2)
    d_prime = (d + d_2) / 2
    return x + d_prime * dt


def sample_restart_flow(model, x, sigmas, extra_args=None, callback=None, disable=None, s_noise=1.0,
                        noise_sampler=None, kd=None):
    """Heun with Restart in the ε-equivalent noise level (module docstring)."""
    with importlib.import_module("torch").no_grad():
        kd = k_diffusion(kd)
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        flow = is_const(model_sampling(model))
        plan = restart_plan(sigmas, flow)
        if plan is not None and noise_sampler is None:
            noise_sampler = default_noise_sampler(kd, x, extra_args)
        for i in kd.trange(len(sigmas) - 1, disable=disable):
            x = _heun_step(kd, model, x, sigmas, i, s_in, extra_args, callback)
            if plan is None or i + 1 != plan.lo:
                continue
            scale, noise_std = restart_kernel(sigmas[plan.lo], sigmas[plan.hi], flow)
            for _ in range(plan.times):
                eps = noise_sampler(sigmas[plan.lo], sigmas[plan.hi])
                if flow:
                    x = scale * x + noise_std * eps * s_noise
                else:   # the ε form x + ε·s_noise·√(σ_hi² − σ_lo²)
                    x = x + eps * s_noise * noise_std
                for j in range(plan.hi, plan.lo):
                    x = _heun_step(kd, model, x, sigmas, j, s_in, extra_args, callback)
        return x
