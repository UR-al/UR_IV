"""Extra Samplers for ComfyUI (pack 1.7.0): the three "flow ODE" DPM++ entries.

sam-extra v0.33.0 registers ``DPM++ 2M (flow ODE)``, ``DPM++ 2M Heun (flow ODE)`` and ``DPM++ 3M (flow ODE)``
as Forge's own DPM-Solver++ functions run at η = 0. The pack does the same with **ComfyUI's own** functions,
``comfy.k_diffusion.sampling.sample_dpmpp_2m_sde`` (``solver_type`` ``"midpoint"`` / ``"heun"`` — k-diffusion's
Heun/φ₂ correction) and ``sample_dpmpp_3m_sde`` — the same function bodies as Forge Neo's copy of ComfyUI
v0.3.75 — so the result is the same by construction. Called with ``eta = 0``, ``s_noise = 0`` and a noise
sampler that raises if it is ever called (``no_noise``): at η = 0 the functions never call it, and passing one
keeps them from building a Brownian tree. They step in the model's half-log-SNR λ (``log((1 − σ)/σ)`` on a
rectified-flow model such as Anima, ``−log σ`` otherwise) with ``α_t = σ_t·e^{λ_t}``, i.e. DPM-Solver++ (Lu et
al., arXiv:2211.01095) for the model's probability-flow ODE.

* Like sam-extra's wrappers they take no ``eta`` and no ``noise_sampler`` — nothing to set and nothing drawn
  (ComfyUI's KSampler passes neither; CNS colours no noise here).
* ``dpmpp_3m_flow_ode`` is in ``KSampler.DISCARD_PENULTIMATE_SIGMA_SAMPLERS`` (``extra_samplers_register``):
  Forge's "DPM++ 3M SDE" option ``discard_next_to_last_sigma``, which "DPM++ 3M (flow ODE)" keeps, so it is
  Forge's DPM++ 3M SDE at Eta 0 (ComfyUI's own ``dpmpp_3m_sde`` does not discard). The two 2M entries do not
  discard (like Forge's 2M SDE).
* sam-extra's fourth wrapper of the same family, "DPM++ 2M SDE Heun", is ComfyUI core's ``dpmpp_2m_sde_heun``
  (same function, ``solver_type="heun"``) and is not registered by the pack.

The pack's own three calls (nothing copied from sam-extra, GPL-3.0). ``tests/test_comfy_extra_samplers_parity.py``
runs them and sam-extra's ``dpmpp_flow`` on Forge's k-diffusion module (bit for bit) and
``tests/test_comfy_extra_samplers_runtime.py`` on ComfyUI's (== ``dpmpp_2m_sde``/``dpmpp_3m_sde`` at η 0).
``kd`` (default ``comfy.k_diffusion.sampling``) is the k-diffusion module. No torch or ComfyUI import at module
level.
"""

from __future__ import annotations

from typing import Any

from .extra_samplers_math import k_diffusion

# sam-extra dpmpp_flow._no_noise's message (pinned by the parity test).
NOISE_DRAWN = "a flow ODE sampler drew noise"


def no_noise(sigma: Any, sigma_next: Any) -> Any:
    """The flow ODE entries' noise sampler: at η = 0 the host's functions never call it."""
    raise RuntimeError(NOISE_DRAWN)


def sample_dpmpp_2m_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, kd=None):
    """DPM-Solver++(2M) for the model's probability-flow ODE (midpoint correction)."""
    return k_diffusion(kd).sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=no_noise, solver_type="midpoint",
    )


def sample_dpmpp_2m_heun_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, kd=None):
    """DPM-Solver++(2M) for the model's probability-flow ODE (Heun / φ₂ correction)."""
    return k_diffusion(kd).sample_dpmpp_2m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=no_noise, solver_type="heun",
    )


def sample_dpmpp_3m_flow_ode(model, x, sigmas, extra_args=None, callback=None, disable=None, kd=None):
    """DPM-Solver++(3M) for the model's probability-flow ODE."""
    return k_diffusion(kd).sample_dpmpp_3m_sde(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
        eta=0.0, s_noise=0.0, noise_sampler=no_noise,
    )
