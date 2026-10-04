# Copyright 2024 KBlueLeaf
#
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
# This file is derived from Koishi-Star/Euler-Smea-Dyn-Sampler
# (https://github.com/Koishi-Star/Euler-Smea-Dyn-Sampler, commit
# d98a504c8419be5274068ad91ca6bbf2e13635a8, ``smea_sampling.py``); the copyright line above is the
# one in that repository's LICENSE (Apache License 2.0, no NOTICE file). The full licence text is in
# ``LICENSES/Euler-Smea-Dyn-Sampler-Apache-2.0.txt``.
# MODIFIED by the AI Studio node pack, 2026-10-04 (Apache-2.0 section 4(b)) — every change is listed
# in the docstring below.
"""Extra Samplers for ComfyUI (pack 1.7.0): Euler Dy / Euler SMEA Dy and their CFG++ entries.

Four registered samplers share one core (``euler_dy``): ``euler_dy`` / ``euler_smea_dy`` (plain
Euler steps) and ``euler_dy_cfg_pp`` / ``euler_smea_dy_cfg_pp`` (the CFG++ update, change 1).

Upstream (``smea_sampling.py``):

* ``sample_euler_dy`` — Euler with Karras churn; at steps 2 and 3 (``i // 2 == 1``) it first runs
  ``dy_sampling_step``: the model sees the (odd row, odd column) pixel of every 2×2 block — a
  half-resolution latent —, one Euler step moves those pixels, they are written back, and the step's
  own Euler increment (from the full-resolution evaluation) is added on top.
* ``sample_euler_smea_dy`` — after the step's Euler update, ``smea_sampling_step`` (``nearest-exact``
  ×1.25 upscale, one Euler step, back to the original size) at step 0 and ``dy_sampling_step`` at
  step 1 (upstream writes ``i + 1 // 2 == 0`` / ``== 1``, i.e. ``i == 0`` / ``i == 1``). The sub-steps
  reuse the step's σ̂ and ``dt`` on the updated latent.
* ``_Rescaler`` — during a sub-step the resolution-bound sampler inputs are resized
  (``nearest-exact``) to the sub-step size and restored afterwards.

Changes (the same host changes sam-extra makes for Forge, so the pack and sam-extra give the same
latent for the same inputs; ``tests/test_comfy_extra_samplers_parity.py``):

1. **CFG++** (the CFG++ entries): every Euler update — the step's and both sub-steps' — is the CFG++
   update of the host's ``sample_euler_cfg_pp`` (``sample_euler_ancestral_cfg_pp`` at η = 0):
   ``x' = α_t·x0 + σ_t·(x − α_s·x0_uncond)/σ_s`` with ``α = σ·e^{λ(σ)}`` from the host's half-log-SNR
   helper and the unconditional x0 kept by a post-CFG function appended with
   ``disable_cfg1_optimization=True``. With CFG++ the Dy step's result is
   ``CFG++(x̂) + (x_dy − x̂)``; the step that reaches σ = 0 returns the guided x0. Without an
   unconditional prediction the update falls back to the plain Euler one.
2. **Churn rule** ``min`` (k-diffusion's ``sample_euler``, Karras et al. 2022 Algorithm 2) instead of
   upstream's ``max(s_churn / N, √2 − 1)``, so the default ``s_churn = 0`` does not churn;
   ``churn_rule="max"`` reproduces upstream. ``eps`` is drawn every step as upstream does, from the
   host's ``default_noise_sampler`` (seeded from ``extra_args["seed"]`` on ComfyUI).
3. **Flow models** (rectified flow, ``x = (1 − σ)·x0 + σ·n``): churn on ``s = σ/(1 − σ)``:
   ``ŝ = s·(1 + γ)``, ``x̂ = α̂·(x/α − eps·sqrt(ŝ² − s²))``, ``σ̂ = ŝ/(1 + ŝ)``, ``α̂ = 1 − σ̂``; no churn at
   σ ≥ 1. ``s_tmin``/``s_tmax`` compare the model's own σ.
4. **3-D/4-D/5-D tensors** (Anima, Wan: ``[B, C, T, H, W]``): the sub-steps work on the last two
   dimensions; the ×1.25 resize interpolates each frame on its own. Upstream's ``unfold`` gather is
   strided indexing — the same pixels, the same odd-size handling incl. its bottom-right corner (when
   both sizes are odd that pixel takes the value above it).
5. **Sub-step inputs** (``SubstepInputs``, upstream's ``_Rescaler`` ComfyUI branch): the evaluation
   carries ``transformer_options["sam_extra_substep"] = "dy" | "smea"`` (sam-extra's marker, so guidance
   and Colorcraft keep their step state); ``extra_args["denoise_mask"]`` and the ``KSamplerX0Inpaint``'s
   ``latent_image``/``noise`` are resized to the sub-step size and restored — ``noise`` from the noise
   (upstream resized ``latent_image`` into it), 5-D frame by frame. What a sub-step cannot follow (concat
   conditioning, regional masks/areas, Spectrum) turns the sub-steps off: ``extra_samplers_dy_host``.
6. Upstream's preview slot handling → ``after_substep(denoised)`` when the host passes one (ComfyUI
   previews come from the step callback; the pack never passes it).
7. **No sub-steps**: ``substeps=False`` runs the plain steps only (the CFG++ entries are then the
   host's Euler CFG++ with the churn above, the plain ones Euler with it).
8. A latent smaller than 2×2 skips the Dy sub-step (upstream would call the model on an empty tensor).
9. **Segmented runs**: ``sam_extra_step_offset`` is the pass index of ``sigmas[0]`` when a pass is
   sampled in several calls (Anima SPEED); the sub-steps are placed by the pass index.

No torch or ComfyUI import at module level; ``kd`` (default ``comfy.k_diffusion.sampling``) lets the
parity tests run it on Forge's k-diffusion module.
"""

from __future__ import annotations

import importlib
from functools import partial
from typing import Any

from .extra_samplers_math import default_noise_sampler, k_diffusion, model_sampling
from .guidance_common import SUBSTEP_MARKER

CHURN_MIN = "min"   # k-diffusion sample_euler (Karras et al. 2022, Algorithm 2)
CHURN_MAX = "max"   # Koishi-Star sample_euler_dy / sample_euler_smea_dy
CHURN_LIMIT = 2 ** 0.5 - 1

SMEA_SCALE = (1.25, 1.25)        # smea_sampling_step
RESIZE_MODE = "nearest-exact"    # smea_sampling_step and _Rescaler
SUBSTEP_DY = "dy"
SUBSTEP_SMEA = "smea"


def _torch():
    return importlib.import_module("torch")


def churn_gamma(rule: str, s_churn: float, steps: int, sigma, s_tmin: float, s_tmax: float) -> float:
    """γ of one step (0 outside ``[s_tmin, s_tmax]``)."""
    if rule == CHURN_MAX:
        return max(s_churn / steps, CHURN_LIMIT) if s_tmin <= sigma <= s_tmax else 0.
    if rule == CHURN_MIN:
        return min(s_churn / steps, CHURN_LIMIT) if s_tmin <= sigma <= s_tmax else 0.
    raise ValueError(f"unknown churn rule: {rule!r}")


def spatial_interpolate(t: Any, *, size=None, scale_factor=None, mode: str = RESIZE_MODE) -> Any:
    """``F.interpolate`` over the last two dimensions of a 3-D, 4-D or 5-D tensor (frames folded into
    the batch, so each frame is interpolated on its own)."""
    interpolate = importlib.import_module("torch.nn.functional").interpolate
    if t.ndim == 4:
        return interpolate(t, size=size, scale_factor=scale_factor, mode=mode)
    if t.ndim == 3:
        return interpolate(t.unsqueeze(0), size=size, scale_factor=scale_factor, mode=mode).squeeze(0)
    if t.ndim == 5:
        batch, channels, frames, height, width = t.shape
        flat = t.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        out = interpolate(flat, size=size, scale_factor=scale_factor, mode=mode)
        return out.reshape(batch, frames, channels, out.shape[-2], out.shape[-1]).permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(f"expected a 3-D, 4-D or 5-D tensor, got shape {tuple(t.shape)}")


def is_const(sampling: Any) -> bool:
    """Rectified-flow parameterisation: ComfyUI's ``CONST`` model sampling (Forge's predictors, which
    the parity tests use, say so in ``prediction_type``)."""
    if hasattr(sampling, "prediction_type"):
        return getattr(sampling, "prediction_type", None) == "const"
    try:
        const = importlib.import_module("comfy.model_sampling").CONST
    except Exception:
        return False
    return isinstance(sampling, const)


class UncondCapture:
    """The CFG++ post-CFG function: keep the unconditional x0, return the guided one unchanged."""

    __slots__ = ("value",)

    def __init__(self):
        self.value = None

    def __call__(self, args):
        self.value = args["uncond_denoised"]
        return args["denoised"]


def post_cfg_setter(kd: Any):
    """The host's ``set_model_options_post_cfg_function``: Forge's k-diffusion module imports it by name,
    ComfyUI's calls ``comfy.model_patcher.set_model_options_post_cfg_function`` (no module attribute)."""
    setter = getattr(kd, "set_model_options_post_cfg_function", None)
    if setter is None:
        setter = importlib.import_module("comfy.model_patcher").set_model_options_post_cfg_function
    return setter


def install_uncond_capture(extra_args: dict, kd: Any) -> UncondCapture:
    """Append an ``UncondCapture`` to ``extra_args["model_options"]`` as the host's CFG++ samplers do."""
    capture = UncondCapture()
    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = post_cfg_setter(kd)(model_options, capture, disable_cfg1_optimization=True)
    return capture


def with_substep_marker(extra_args: dict, kind: str) -> dict:
    """A copy of ``extra_args`` whose ``model_options["transformer_options"]`` carries ``SUBSTEP_MARKER``
    (only the dictionaries on that path are copied; the post-CFG list is shared with the step's own)."""
    args = dict(extra_args)
    options = dict(args.get("model_options") or {})
    transformer_options = dict(options.get("transformer_options") or {})
    transformer_options[SUBSTEP_MARKER] = kind
    options["transformer_options"] = transformer_options
    args["model_options"] = options
    return args


# ComfyUI's resolution-bound sampler inputs (upstream ``_Rescaler``, ComfyUI branch): the
# ``KSamplerX0Inpaint`` the sampler gets keeps the inpainting latent and noise as attributes, the
# denoise mask travels in ``extra_args``. All three have the latent's shape (comfy.utils.reshape_mask).
MODEL_INPUTS = ("latent_image", "noise")
ARG_INPUTS = ("denoise_mask",)


class SubstepInputs:
    """``_Rescaler``: while a sub-step is evaluated, the resolution-bound inputs follow it.

    Upstream's ComfyUI branch, with these changes: ``noise`` is resized from the noise (upstream resized
    ``latent_image`` into it); only tensors of 3 or more dimensions whose last two match the full latent
    are resized (each frame of a 5-D one on its own, ``spatial_interpolate``); the attributes come back
    by plain assignment, also when the evaluation raises; the denoise mask is resized in the sub-step's
    own copy of ``extra_args``, which also carries ``SUBSTEP_MARKER``."""

    def __init__(self, model: Any, extra_args: dict, full_shape: tuple, size: tuple, kind: str):
        self.model = model
        self.extra_args = extra_args
        self.full_hw = tuple(int(value) for value in tuple(full_shape)[-2:])
        self.size = tuple(int(value) for value in size)
        self.kind = kind
        self._saved: dict = {}

    def _follows(self, value: Any) -> bool:
        return (_torch().is_tensor(value) and value.ndim >= 3
                and tuple(int(v) for v in value.shape[-2:]) == self.full_hw)

    def _resized(self, value: Any) -> Any:
        return spatial_interpolate(value, size=self.size, mode=RESIZE_MODE)

    def __enter__(self) -> dict:
        args = with_substep_marker(self.extra_args, self.kind)
        for name in ARG_INPUTS:
            if self._follows(args.get(name)):
                args[name] = self._resized(args[name])
        self._saved = {}
        for name in MODEL_INPUTS:
            value = getattr(self.model, name, None)
            if self._follows(value):
                resized = self._resized(value)
                self._saved[name] = value
                setattr(self.model, name, resized)
        return args

    def __exit__(self, exc_type, exc, tb) -> bool:
        for name, value in self._saved.items():
            setattr(self.model, name, value)
        self._saved = {}
        return False


def evaluate(model: Any, x: Any, sigma: Any, extra_args: dict, capture: UncondCapture | None):
    """One model call → ``(guided x0, unconditional x0 or None)``."""
    if capture is not None:
        capture.value = None
    denoised = model(x, sigma * x.new_ones([x.shape[0]]), **extra_args)
    if capture is None:
        return denoised, None
    uncond = capture.value
    if not _torch().is_tensor(uncond) or uncond.shape != denoised.shape:
        uncond = denoised   # no unconditional prediction: the update is plain Euler
    return denoised, uncond


def cfg_pp_target(kd: Any, lambda_fn, x, denoised, uncond, sigma, sigma_next):
    """The host's CFG++ Euler step from ``sigma`` to ``sigma_next`` (> 0), η = 0."""
    alpha_s = sigma * lambda_fn(sigma).exp()
    alpha_t = sigma_next * lambda_fn(sigma_next).exp()
    d = kd.to_d(x, sigma, alpha_s * uncond)
    sigma_down, _sigma_up = kd.get_ancestral_step(sigma / alpha_s, sigma_next / alpha_t, eta=0.0)
    sigma_down = alpha_t * sigma_down
    return alpha_t * denoised + sigma_down * d


def churn(x, sigma, gamma: float, eps, const: bool):
    """Raise the noise level of ``x`` by γ → ``(x̂, σ̂)`` (change 3 for flow models)."""
    if not const:
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            x = x - eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
        return x, sigma_hat
    if not gamma > 0 or not (0 < sigma < 1):
        return x, sigma
    alpha = 1 - sigma
    s = sigma / alpha
    s_hat = s * (gamma + 1)
    sigma_hat = s_hat / (1 + s_hat)
    alpha_hat = 1 - sigma_hat
    x = alpha_hat * (x / alpha - eps * (s_hat ** 2 - s ** 2) ** 0.5)
    return x, sigma_hat


def step_target(kd: Any, lambda_fn, cfg_pp: bool, x, denoised, uncond, sigma_hat, sigma_next, dt):
    """``x`` after one Euler (or CFG++) step from ``sigma_hat`` to ``sigma_next``."""
    if not cfg_pp:
        return x + kd.to_d(x, sigma_hat, denoised) * dt
    if sigma_next == 0:
        return denoised
    return cfg_pp_target(kd, lambda_fn, x, denoised, uncond, sigma_hat, sigma_next)


class _Run:
    """What the sub-steps share with the step loop."""

    __slots__ = ("kd", "lambda_fn", "model", "extra_args", "capture", "cfg_pp", "inputs")

    def __init__(self, kd, lambda_fn, model, extra_args, capture, cfg_pp, inputs):
        self.kd, self.lambda_fn, self.model, self.extra_args = kd, lambda_fn, model, extra_args
        self.capture, self.cfg_pp, self.inputs = capture, cfg_pp, inputs


def dy_substep(run: _Run, x, sigma_hat, sigma_next, dt):
    """``dy_sampling_step`` on the last two dimensions (any leading shape)."""
    height, width = int(x.shape[-2]), int(x.shape[-1])
    m, n = height // 2, width // 2
    if m == 0 or n == 0:
        return x
    c = x[..., 1:2 * m:2, 1:2 * n:2].contiguous()
    with run.inputs(run.model, run.extra_args, tuple(x.shape), (m, n), SUBSTEP_DY) as args:
        den_c, unc_c = evaluate(run.model, c, sigma_hat, args, run.capture)
    c = step_target(run.kd, run.lambda_fn, run.cfg_pp, c, den_c, unc_c, sigma_hat, sigma_next, dt)
    out = x.clone()
    out[..., 1:2 * m:2, 1:2 * n:2] = c
    if height % 2 == 1 and width % 2 == 1:
        # upstream rebuilds the odd row and column, then sets the corner from its column copy taken
        # after the row was cropped: the bottom-right pixel gets the one above it
        out[..., -1:, -1:] = x[..., 2 * m - 1:2 * m, -1:]
    return out


def smea_substep(run: _Run, x, sigma_hat, sigma_next, dt):
    """``smea_sampling_step``: ×1.25 ``nearest-exact``, one step, back to the input size."""
    height, width = int(x.shape[-2]), int(x.shape[-1])
    x_up = spatial_interpolate(x, scale_factor=SMEA_SCALE, mode=RESIZE_MODE)
    with run.inputs(run.model, run.extra_args, tuple(x.shape), tuple(x_up.shape[-2:]), SUBSTEP_SMEA) as args:
        den, unc = evaluate(run.model, x_up, sigma_hat, args, run.capture)
    x_up = step_target(run.kd, run.lambda_fn, run.cfg_pp, x_up, den, unc, sigma_hat, sigma_next, dt)
    return spatial_interpolate(x_up, size=(height, width), mode=RESIZE_MODE)


def euler_dy(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0,
             s_tmax=float("inf"), s_noise=1.0, noise_sampler=None, after_substep=None, *, smea: bool = False,
             cfg_pp: bool = True, churn_rule: str = CHURN_MIN, substeps: bool = True, sam_extra_step_offset: int = 0,
             kd: Any = None, substep_inputs: Any = None):
    """Euler Dy (``smea=False``) or Euler SMEA Dy (``smea=True``); see the module docstring.

    ``cfg_pp=False, churn_rule="max"`` is upstream's sampler. ``substep_inputs`` builds each sub-step's
    evaluation arguments (``SubstepInputs`` by default)."""
    with _torch().no_grad():
        kd = k_diffusion(kd)
        extra_args = {} if extra_args is None else extra_args
        if noise_sampler is None:
            noise_sampler = default_noise_sampler(kd, x, extra_args)
        sampling = model_sampling(model)
        const = is_const(sampling)
        lambda_fn = partial(kd.sigma_to_half_log_snr, model_sampling=sampling)
        capture = install_uncond_capture(extra_args, kd) if cfg_pp else None
        run = _Run(kd, lambda_fn, model, extra_args, capture, cfg_pp, substep_inputs or SubstepInputs)
        steps = len(sigmas) - 1
        offset = int(sam_extra_step_offset or 0)

        for i in kd.trange(steps, disable=disable):
            step = i + offset   # the step's index in its pass (change 9)
            sigma, sigma_next = sigmas[i], sigmas[i + 1]
            gamma = churn_gamma(churn_rule, s_churn, steps, sigma, s_tmin, s_tmax)
            eps = noise_sampler(sigma, sigma_next) * s_noise
            x, sigma_hat = churn(x, sigma, gamma, eps, const)
            dt = sigma_next - sigma_hat
            denoised, uncond = evaluate(model, x, sigma_hat, extra_args, capture)

            if not smea:
                x_dy = x
                if substeps and sigma_next > 0 and step // 2 == 1:
                    x_dy = dy_substep(run, x, sigma_hat, sigma_next, dt)
                    if after_substep is not None:
                        after_substep(denoised)
                if callback is not None:
                    callback({"x": x_dy, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})
                if not cfg_pp:
                    # upstream: the step's own derivative added to the Dy result
                    x = x_dy + kd.to_d(x, sigma_hat, denoised) * dt
                else:
                    stepped = step_target(kd, lambda_fn, True, x, denoised, uncond, sigma_hat, sigma_next, dt)
                    x = stepped if x_dy is x else stepped + (x_dy - x)
                continue

            x = step_target(kd, lambda_fn, cfg_pp, x, denoised, uncond, sigma_hat, sigma_next, dt)
            if substeps and sigma_next > 0:
                ran = False
                if step == 1:   # upstream: ``if i + 1 // 2 == 1``
                    x = dy_substep(run, x, sigma_hat, sigma_next, dt)
                    ran = True
                if step == 0:   # upstream: ``if i + 1 // 2 == 0``
                    x = smea_substep(run, x, sigma_hat, sigma_next, dt)
                    ran = True
                if ran and after_substep is not None:
                    after_substep(denoised)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma_hat, "denoised": denoised})
        return x


def _entry(name: str, *, smea: bool, cfg_pp: bool, doc: str):
    def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0.0, s_tmin=0.0,
                s_tmax=float("inf"), s_noise=1.0, noise_sampler=None, after_substep=None, substeps=True,
                sam_extra_step_offset=0, kd=None, substep_inputs=None):
        return euler_dy(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, s_churn=s_churn,
            s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise, noise_sampler=noise_sampler,
            after_substep=after_substep, smea=smea, cfg_pp=cfg_pp, churn_rule=CHURN_MIN, substeps=bool(substeps),
            sam_extra_step_offset=sam_extra_step_offset, kd=kd, substep_inputs=substep_inputs,
        )

    sampler.__name__ = sampler.__qualname__ = "sample_" + name
    sampler.__doc__ = doc
    return sampler


sample_euler_dy = _entry(
    "euler_dy", smea=False, cfg_pp=False,
    doc="Euler Dy: plain Euler steps, half-resolution sub-step at steps 2 and 3, ``min`` churn.")
sample_euler_smea_dy = _entry(
    "euler_smea_dy", smea=True, cfg_pp=False,
    doc="Euler SMEA Dy: plain Euler steps, ×1.25 sub-step at step 0 and half-resolution at step 1, ``min`` churn.")
sample_euler_dy_cfg_pp = _entry(
    "euler_dy_cfg_pp", smea=False, cfg_pp=True,
    doc="Euler Dy CFG++: the CFG++ update, half-resolution sub-step at steps 2 and 3.")
sample_euler_smea_dy_cfg_pp = _entry(
    "euler_smea_dy_cfg_pp", smea=True, cfg_pp=True,
    doc="Euler SMEA Dy CFG++: the CFG++ update, ×1.25 sub-step at step 0 and half-resolution at step 1.")
