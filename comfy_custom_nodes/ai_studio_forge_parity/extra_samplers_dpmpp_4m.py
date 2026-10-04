# BSD 3-Clause License
#
# Copyright (c) 2024, Clybius
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Extra Samplers for ComfyUI (pack 1.7.0): ``dpmpp_4m_sde`` — Clybius' DPM-Solver++(4M) SDE.

Origin: Clybius/ComfyUI-Extra-Samplers@52eac1b7c847d2727e0ca93ca26d9ffd77029daa ``extra_samplers.py``
``sample_clyb_4m_sde_momentumized`` (registered there as ``clyb_4m_sde_momentumized`` through
``sample_clyb_4m_sde``: momentum 0, a Brownian-tree noise sampler, the penultimate sigma discarded).
BSD-3-Clause; the notice above, the full text in ``LICENSES/ComfyUI-Extra-Samplers-BSD-3-Clause.txt``.
Background: DPM-Solver++ (Lu et al., arXiv:2211.01095) and k-diffusion's DPM++ 2M/3M SDE, which this
extends by one more history term.

Changes for ComfyUI and Forge parity (the same host changes sam-extra lists for its Forge port, so
the two give the same latent on the same inputs — ``tests/test_comfy_extra_samplers_parity.py``):

* **Half-log-SNR coordinates** λ = log(α/σ) through the host's ``sigma_to_half_log_snr`` /
  ``offset_first_sigma_for_snr`` and ``α_t = σ_t·e^{λ_t}``: the step is
  ``x ← (σ_t/σ_s)·e^{−hη}·x + α_t·(1 − e^{−h(1+η)})·x0`` and each history correction is scaled by
  ``α_t``. On an epsilon/v model λ = −log σ and α_t = 1 (upstream's update); on a rectified-flow
  model the same update in epsilon-equivalent coordinates. The 1st–3rd order steps equal ComfyUI's
  ``sample_dpmpp_3m_sde`` step for step.
* **Momentum** is not ported: at upstream's registered 0.0 its ``momentum_func`` returns ``diff``.
* **Noise** only when ``eta > 0 and s_noise > 0`` (upstream: ``if eta:``); without a given
  ``noise_sampler`` a CPU ``BrownianTreeNoiseSampler`` seeded from ``extra_args["seed"]`` (ComfyUI's
  3M SDE convention); ``s_noise`` times the model sampling's ``noise_scale`` (ComfyUI convention, 1.0
  for every model the app runs).
* The host's ``trange``; the momentum term's time bookkeeping is dropped. Order selection, ratios,
  ``d1``/``d2`` and the φ expressions are upstream's.

No torch or ComfyUI import at module level; ``kd`` (default ``comfy.k_diffusion.sampling``) lets the
parity tests run it on Forge's k-diffusion module.
"""

from __future__ import annotations

import importlib
from functools import partial

from .extra_samplers_math import k_diffusion, model_sampling


def sample_dpmpp_4m_sde(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1.0,
                        noise_sampler=None, kd=None):
    """DPM-Solver++(4M) SDE (Clybius, momentum 0) for epsilon, v and rectified-flow models."""
    with importlib.import_module("torch").no_grad():
        return _sample(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler,
                       k_diffusion(kd))


def _sample(model, x, sigmas, extra_args, callback, disable, eta, s_noise, noise_sampler, kd):
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = kd.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=True)
    s_in = x.new_ones([x.shape[0]])

    sampling = model_sampling(model)
    lambda_fn = partial(kd.sigma_to_half_log_snr, model_sampling=sampling)
    sigmas = kd.offset_first_sigma_for_snr(sigmas, sampling)
    s_noise = s_noise * getattr(sampling, "noise_scale", 1.0)

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    for i in kd.trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
            continue
        lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        alpha_t = sigmas[i + 1] * lambda_t.exp()

        x = sigmas[i + 1] / sigmas[i] * (-h * eta).exp() * x + alpha_t * (-h_eta).expm1().neg() * denoised

        if h_3 is not None:
            # fourth order: upstream's extrapolation over three history differences
            r0 = h_3 / h_2
            r1 = h_2 / h
            r2 = h / h_1
            d1_0 = (denoised - denoised_1) / r2
            d1_1 = (denoised_1 - denoised_2) / r1
            d1_2 = (denoised_2 - denoised_3) / r0
            d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
            d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
            phi_3 = h_eta.neg().expm1() / h_eta + 1
            phi_4 = phi_3 / h_eta - 0.5
            x = x + (alpha_t * phi_3) * d1 - (alpha_t * phi_4) * d2
        elif h_2 is not None:
            # third order (DPM-Solver++(3M) SDE)
            r0 = h_1 / h
            r1 = h_2 / h
            d1_0 = (denoised - denoised_1) / r0
            d1_1 = (denoised_1 - denoised_2) / r1
            d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
            d2 = (d1_0 - d1_1) / (r0 + r1)
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            phi_3 = phi_2 / h_eta - 0.5
            x = x + (alpha_t * phi_2) * d1 - (alpha_t * phi_3) * d2
        elif h_1 is not None:
            # second order (DPM-Solver++(2M) SDE)
            r = h_1 / h
            d = (denoised - denoised_1) / r
            phi_2 = h_eta.neg().expm1() / h_eta + 1
            x = x + (alpha_t * phi_2) * d

        if eta > 0 and s_noise > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
        h_1, h_2, h_3 = h, h_1, h_2
    return x
