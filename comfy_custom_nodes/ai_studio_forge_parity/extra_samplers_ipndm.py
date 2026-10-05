# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The step weights in ``weights`` are those of zju-pi/diff-sampler (https://github.com/zju-pi/diff-sampler, commit
# 68d5ce427f261962b89ce3b0ee8f6b29f0577328, ``diff-solvers-main/solvers.py`` ``ipndm_v_sampler``; Apache License
# 2.0), written as ComfyUI's ``sample_ipndm_v`` writes them (comfyanonymous/ComfyUI@387f98aa, marked there "From
# https://github.com/zju-pi/diff-sampler/... under Apache 2 license"). That repository's LICENSE file was not read
# here, so no copyright line of it is reproduced; the repository and the commit are named, as ComfyUI does. The full
# licence text is in ``LICENSES/diff-sampler-Apache-2.0.txt``.
# MODIFIED by the AI Studio node pack, 2026-10-05 (Apache-2.0 section 4(b)) — every change is listed in the
# docstring below.
"""Extra Samplers for ComfyUI (pack 1.7.0): ``ipndm_v_fixed`` — IPNDM_V with its fourth-order weight fixed.

IPNDM_V (the variable-step iPNDM of Zhang & Chen, arXiv:2204.13902, in zju-pi/diff-sampler's ``ipndm_v_sampler``;
ComfyUI's ``ipndm_v``, sam-extra's Forge entry "IPNDM_V") integrates the probability-flow ODE ``dx/dσ = d``,
``d = (x − D)/σ``, with one model call per step: ``x ← x + h_n·Σ_k w_k·d_{n−k}`` over the last ``max_order`` (≤ 4)
derivatives, the weights ``w_k`` coming from the step sizes ``h_n = σ_{n+1} − σ_n``, ``h_{n−1}``, ``h_{n−2}``,
``h_{n−3}``. A step to σ = 0 returns the model's x0.

**The fix.** Upstream's order-4 weight of ``d_{n−3}`` is ``−temp2·q·h_{n−1}/h_{n−2}`` with
``q = h_{n−1}(h_{n−1} + h_{n−2}) / (h_{n−2}(h_{n−2} + h_{n−3}))``. ``temp2`` multiplies the third divided difference,
whose last entry is ``−q·h_{n−2}/h_{n−3}`` (the weight of ``d_{n−2}`` already carries the matching
``q·(1 + h_{n−2}/h_{n−3})``). With the typo the four weights sum to ``1 + q·temp2·(h_{n−2}/h_{n−3} − h_{n−1}/h_{n−2})``
instead of 1: on Anima's Linear Quadratic 28-step list the sum is about −174 at step 15 (0-based; σ 0.968 → 0.952)
and the image turns to green noise, in ComfyUI's ``ipndm_v`` and in sam-extra up to 0.33.0 alike. sam-extra 0.33.1 changes
that last factor to ``h_{n−2}/h_{n−3}``; this sampler is that version. The fix makes every step consistent; it does not
make IPNDM_V suit Linear Quadratic: on that list the image is burnt instead of green noise (2026-10-05 GPU check, Forge
and ComfyUI alike) — after 14 tiny steps the step size jumps ×3.71, ×2.46, ×1.59 and any step-size-aware fourth-order
extrapolation weights the old derivatives up to about 155 there (a sampler × scheduler combination). ComfyUI core keeps ``ipndm_v`` as it is and the
pack never replaces a core name, so the fixed sampler is registered under a name of its own (``NAME``); the app sends
Forge's "IPNDM_V" to it.

**Kept as upstream** (as sam-extra 0.33.1 keeps them): ``temp1``'s parenthesisation (its factor
``h_n(h_n + h_{n−1}) / (h_{n−1}(h_{n−1} + h_{n−2}))`` sits inside ``1 − h_n/(3(h_n + h_{n−1}))·…`` instead of
multiplying ``(1 − h_n/(3(h_n + h_{n−1})))/2``) and the sign of ``temp2``'s second term (``+`` where the Adams–Bashforth
expansion has ``−``). Both are known upstream inaccuracies: the weights still sum to 1 on every list (the step stays
consistent), only orders 3 and 4 are less accurate than the exact variable-step Adams–Bashforth weights. Keeping them
keeps this sampler equal to sam-extra 0.33.1's "IPNDM_V" bit for bit on every list, and its weights equal to
ComfyUI's ``ipndm_v`` ones in exact arithmetic wherever ``h_{n−1}/h_{n−2} = h_{n−2}/h_{n−3}`` (equal steps, geometric
lists). The latents are ComfyUI's bit for bit there only when the step sizes are exact in binary (equal, or
power-of-two multiples of one another: power-of-two equal steps, halving or doubling lists). The last weight is
computed as ``(−temp2·q·h_{n−2})/h_{n−3}``, not from the ratio, so it can round differently even where the two ratios
are the same floating-point number, and on float32/float64 equal-step (linspace) or geometric schedules the result
differs from ComfyUI's ``ipndm_v`` at rounding level.

Changes (Apache-2.0 4(b)) against upstream as ComfyUI adapts it, the first three and the fifth as sam-extra makes them:

1. The fourth-order weight above (it changes the result where the step ratio varies, and at rounding level on
   constant-ratio lists whose steps are not exact in binary).
2. ``max_order`` is clamped to 1..4 and order 1 keeps no history (upstream raises ``IndexError`` there), so order 1
   is plain Euler with the final denoising step.
3. The callback gets the current latent (ComfyUI passes the first one every step — a preview bug).
4. The loop and the history are this pack's own code (a newest-first list instead of upstream's shifted buffer, the
   weights from ``weights``); the weights keep upstream's floating-point operations, in the schedule's own 0-dim
   tensors, because the tests compare bit for bit (THIRD_PARTY_NOTICES.md, "Formulas shared with GPL-3.0 hosts").
5. The sampler runs under ``torch.no_grad()`` (ComfyUI's ``sample_ipndm_v`` has no such guard; sam-extra's carries
   ``@torch.no_grad()``): no autograd graph is kept, the values are the same.

``tests/test_comfy_extra_samplers_parity.py`` runs this and sam-extra's IPNDM_V on Forge's k-diffusion module and
compares the latents bit for bit; ``tests/test_comfy_extra_samplers_runtime.py`` registers it in ComfyUI's real tables
(CPU). ``kd`` (default ``comfy.k_diffusion.sampling``) is the k-diffusion module. No torch or ComfyUI import at module
level.
"""

from __future__ import annotations

import importlib
from typing import Any

from .extra_samplers_math import k_diffusion

NAME = "ipndm_v_fixed"
IPNDM_MAX_ORDER = 4      # sam-extra ipndm_deis.py IPNDM_MAX_ORDER = ComfyUI's KSampler default
_HIGHEST_ORDER = 4       # the weights exist for orders 1..4


def clamp_order(max_order: Any) -> int:
    """``max_order`` as an order 1..4 (sam-extra's rule)."""
    return max(1, min(_HIGHEST_ORDER, int(max_order)))


def step_sizes(sigmas: Any, i: int, order: int) -> tuple[Any, ...]:
    """``(h_n, h_{n−1}, …)``, ``order`` of them, for the step ``σ_i → σ_{i+1}`` — differences of the schedule's
    entries, so the weights are computed in its dtype like upstream's."""
    return (sigmas[i + 1] - sigmas[i], *(sigmas[i - k + 1] - sigmas[i - k] for k in range(1, order)))


def weights(h: tuple[Any, ...]) -> tuple[Any, ...]:
    """IPNDM_V's weights of ``(d_n, d_{n−1}, …)`` for the step sizes ``h = (h_n, h_{n−1}, …)`` of orders 2..4 — the
    upstream expressions with the fourth-order fix (module docstring)."""
    if len(h) == 2:
        h_n, h_n_1 = h
        return (2 + (h_n / h_n_1)) / 2, -(h_n / h_n_1) / 2
    if len(h) == 3:
        h_n, h_n_1, h_n_2 = h
        temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
        return ((2 + (h_n / h_n_1)) / 2 + temp, -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp,
                temp * h_n_1 / h_n_2)
    h_n, h_n_1, h_n_2, h_n_3 = h
    temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
    temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
        * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
    q = h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))
    return ((2 + (h_n / h_n_1)) / 2 + temp1 + temp2,
            -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + q) * temp2,
            temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + q * (1 + h_n_2 / h_n_3)) * temp2,
            -temp2 * q * h_n_2 / h_n_3)          # upstream ends in "* h_n_1 / h_n_2" (the fix)


def combine(step_weights: tuple[Any, ...], derivatives: tuple[Any, ...]) -> Any:
    """``Σ w_k·d_k``, added left to right from the newest derivative (upstream's order of operations)."""
    total = step_weights[0] * derivatives[0]
    for weight, derivative in zip(step_weights[1:], derivatives[1:]):
        total = total + weight * derivative
    return total


def sample_ipndm_v_fixed(model, x, sigmas, extra_args=None, callback=None, disable=None, max_order=IPNDM_MAX_ORDER,
                         kd=None):
    """IPNDM_V with the fourth-order weight fixed (module docstring) — sam-extra 0.33.1's "IPNDM_V"."""
    with importlib.import_module("torch").no_grad():
        kd = k_diffusion(kd)
        extra_args = {} if extra_args is None else extra_args
        max_order = clamp_order(max_order)
        s_in = x.new_ones([x.shape[0]])
        history: list = []        # the earlier derivatives, newest first (at most max_order − 1)
        for i in kd.trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
            d = (x - denoised) / sigmas[i]
            order = min(max_order, i + 1)
            if sigmas[i + 1] == 0:      # the final denoising step
                x = denoised
            elif order == 1:            # Euler
                x = x + (sigmas[i + 1] - sigmas[i]) * d
            else:
                h = step_sizes(sigmas, i, order)
                x = x + h[0] * combine(weights(h), (d, *history[:order - 1]))
            if max_order > 1:
                history = [d, *history][:max_order - 1]
        return x
