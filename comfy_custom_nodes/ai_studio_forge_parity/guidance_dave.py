"""DAVE for Anima/Cosmos blocks (``ForgeNeoAnimaDAVE``).

Behaviour is the original node's, sorryhyun/ComfyUI-Anima-DAVE@83143e8d
``nodes.py`` (MIT, notice below):

- the block edit is ``out - atten * mu`` with ``mu`` the per-channel mean over
  every dim but batch and channel (:75-88, ``apply_dave``);
- ``atten = clip(strength * w(block), 0, 1)``; blocks with ``atten <= 1e-3``
  are not patched, and a DAVE with no block left returns the MODEL unchanged
  (:163-172). The shipped ``dave_alpha.npz`` mask is ``w = 1`` on blocks 8-18
  and 0 elsewhere, so a block list stands in for it and an empty list is 8-18;
- only target blocks the model has are hooked, and a model without
  ``diffusion_model.blocks`` (or with none of the targets) runs unchanged
  (:216-225): DAVE never fails a graph, it logs and passes the MODEL through;
- the tau gate (:91-106, :199-208, ``dave_gate_active``): the current sigma is
  looked up in ``transformer_options['sample_sigmas']`` (first
  ``isclose(rtol=1e-4, atol=1e-6)`` index, 0 when none matches, so an
  off-schedule sigma such as a second-order sampler's mid-point counts as
  step 0 and is always on), ``n = len - 1``, ``k = max(1, min(n,
  round(tau * n)))`` and DAVE runs while ``step < k``. No published schedule,
  or ``tau <= 0``, means every step.

Host difference: the original hooks the blocks from an APPLY_MODEL wrapper and
reads the sigma as ``transformer_options.get('sigmas', t)``. Here every target
block's ``forward`` is an object patch that reads the same
``transformer_options`` (Comfy's ``calc_cond_batch`` always publishes
``'sigmas'``; the block never sees ``t``, so a missing ``'sigmas'`` is an
unmatched sigma, i.e. step 0).

``_patch_anima_blocks`` is the one block wrapper for the standalone DAVE node
and the suite, so the DAVE gate and the positional ``transformer_options``
fallback are identical in both. The same wrapper also carries the suite's SLG
block skip, which is not gated by tau (it follows ``forge_neo_slg_active``).
"""

# The DC edit, the attenuation rule and the tau gate are adapted from
# ComfyUI-Anima-DAVE (https://github.com/sorryhyun/ComfyUI-Anima-DAVE,
# commit 83143e8d84768e25f72755ec00ea00ded07ee06e, nodes.py), used under:
#
# MIT License
#
# Copyright (c) 2026 Seunghyun Ji
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import logging
import sys
from typing import Any

from .compat import clone_model
from .guidance_common import (
    CATEGORY,
    PRE_DD_SIGMAS_KEY,
    _model_blocks,
    _transformer_options,
    parse_indices,
)

LOGGER = logging.getLogger("ai_studio_forge_parity")

# dave_alpha.npz 'weight': 1.0 on flat blocks 8..18, 0.0 on the other 17 of 28.
DAVE_DEFAULT_BLOCKS = "8-18"
# nodes.py:166 — a block whose attenuation is at most this is not hooked.
DAVE_MIN_ATTENUATION = 1e-3
# nodes.py:103 — torch.isclose(sample_sigmas, sigma_now, rtol=1e-4, atol=1e-6).
_SIGMA_RTOL = 1e-4
_SIGMA_ATOL = 1e-6


def apply_dave(output: Any, attenuation: float):
    """Remove ``attenuation`` of each block output's DC component (nodes.py:83-86).

    The mean keeps batch (dim 0) and channel (last dim); for the Cosmos/predict2
    block output ``(B, T, H, W, D)`` that is dims ``(1, 2, 3)``.
    """

    dims = tuple(range(1, output.ndim - 1)) or (1,)
    mean = output.float().mean(dim=dims, keepdim=True)
    return (output.float() - attenuation * mean).to(output.dtype)


def dave_attenuation(strength: Any) -> float:
    """``clip(strength * w, 0, 1)`` for a target block, whose ``w`` is 1 (nodes.py:165)."""

    return min(1.0, max(0.0, float(strength)))


def _torch_tensor(value: Any) -> bool:
    # A tensor can only exist once torch is imported; never import it here.
    torch = sys.modules.get("torch")
    return torch is not None and torch.is_tensor(value)


def _first_scalar(value: Any) -> float | None:
    try:
        if hasattr(value, "flatten"):
            return float(value.flatten()[0].item())
        return float(value)
    except Exception:
        return None


def dave_step(options: dict[str, Any], pre_dd: bool = False) -> tuple[int | None, int | None]:
    """``(step, n_steps)`` of this forward against the schedule (nodes.py:91-106).

    ``(None, None)`` when the sampler publishes no ``sample_sigmas`` (or fewer
    than two). The step is the FIRST schedule index whose sigma is
    ``isclose(rtol=1e-4, atol=1e-6)`` to the current one, and 0 when none is.
    """

    schedule = options.get("sample_sigmas")
    if schedule is None or len(schedule) < 2:
        return None, None
    n_steps = len(schedule) - 1
    current = options.get("sigmas")
    if pre_dd and options.get(PRE_DD_SIGMAS_KEY) is not None:
        # Detail Daemon's note: the sampler's own sigma before it scaled this call (guidance_dd).
        current = options[PRE_DD_SIGMAS_KEY]
    if current is None:
        return 0, n_steps
    if _torch_tensor(schedule):
        torch = sys.modules["torch"]
        if _torch_tensor(current):
            current0 = current.flatten()[0].to(schedule.device)
        else:
            current0 = torch.tensor(
                float(current), dtype=schedule.dtype, device=schedule.device
            )
        matches = torch.isclose(schedule, current0, rtol=_SIGMA_RTOL, atol=_SIGMA_ATOL)
        found = torch.nonzero(matches).flatten()
        return (int(found[0].item()) if found.numel() else 0), n_steps
    # Plain sequences (no torch): torch.isclose's rule, |a - b| <= atol + rtol*|b|
    # with b the current sigma, so the tolerance is relative to the current sigma.
    current_value = _first_scalar(current)
    if current_value is None:
        return 0, n_steps
    values = schedule.flatten().tolist() if hasattr(schedule, "flatten") else schedule
    for index, value in enumerate(values):
        value = float(value)
        if value == current_value or (
            abs(value - current_value) <= _SIGMA_ATOL + _SIGMA_RTOL * abs(current_value)
        ):
            return index, n_steps
    return 0, n_steps


def dave_gate_active(options: dict[str, Any], tau: Any, pre_dd: bool = False) -> bool:
    """Whether DAVE runs on this forward (nodes.py:199-208). ``tau <= 0``: every step.

    ``pre_dd``: look up the sigma Detail Daemon noted before scaling it (the pre-DD option).
    """

    tau_value = float(tau)
    if tau_value > 0.0:
        step, n_steps = dave_step(options, pre_dd)
        if step is None:  # no schedule published: every step (the original's safe default)
            return True
        cutoff = max(1, min(n_steps, round(tau_value * n_steps)))
        return step < cutoff
    return True


def _patch_anima_blocks(
    model: Any,
    *,
    dave_enabled: bool,
    dave_blocks: str,
    dave_strength: float,
    dave_tau: float,
    slg_enabled: bool,
    slg_blocks: str,
    dave_pre_dd: bool = True,
):
    attenuation = dave_attenuation(dave_strength) if dave_enabled else 0.0
    dave_on = attenuation > DAVE_MIN_ATTENUATION
    if not dave_on and not slg_enabled:
        return model
    patched = clone_model(model, "Anima DAVE/SLG")
    blocks = _model_blocks(patched)
    dave_targets = (
        parse_indices(str(dave_blocks or "").strip() or DAVE_DEFAULT_BLOCKS, len(blocks))
        if dave_on else set()
    )
    if dave_on and not dave_targets:
        # nodes.py:216-225: a model without ``diffusion_model.blocks`` runs unchanged and
        # only the mask's blocks that exist (``i < len(blocks)``) are hooked, so DAVE
        # never fails a graph; it is a no-op here (the Forge extension skips it too).
        LOGGER.warning(
            "Anima DAVE: no target block among %s on this MODEL (%d blocks); DAVE skipped.",
            str(dave_blocks or "").strip() or DAVE_DEFAULT_BLOCKS, len(blocks),
        )
        dave_on = False
        if not slg_enabled:
            return model
    if slg_enabled and not blocks:
        raise RuntimeError(
            "SLG requires an Anima/Cosmos MODEL exposing diffusion_model.blocks."
        )
    slg_targets = (
        parse_indices(slg_blocks, len(blocks), default="18")
        if slg_enabled else set()
    )
    if slg_enabled and not slg_targets:
        raise RuntimeError("SLG has no valid target blocks for this model.")
    tau = float(dave_tau) if dave_on else 0.0
    for index in sorted(dave_targets | slg_targets):
        original = blocks[index].forward

        def combined_forward(*args, _index=index, _original=original, **kwargs):
            options = _transformer_options(args, kwargs)
            if _index in slg_targets and options.get("forge_neo_slg_active"):
                if args:
                    return args[0]
                value = kwargs.get("x_B_T_H_W_D", kwargs.get("x"))
                if value is None:
                    raise RuntimeError("SLG block wrapper did not receive its input tensor.")
                return value
            output = _original(*args, **kwargs)
            if _index in dave_targets and dave_gate_active(options, tau, dave_pre_dd):
                output = apply_dave(output, attenuation)
            return output

        patched.add_object_patch(
            f"diffusion_model.blocks.{index}.forward", combined_forward
        )
    return patched


class ForgeNeoAnimaDAVE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": False}),
            "mask": (["dave_alpha.npz", "blocks:8-18"], {
                "tooltip": "Per-block pool mask (w(ℓ)). Shipped: flat blocks 8–18.",
            }),
            "strength": ("FLOAT", {
                "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": (
                    "DC removal dose s = (1−α). 0.30 is the conservative default; "
                    "up to ~0.80 at tau=0.10 for max diversity. 0 disables."
                ),
            }),
            "tau": ("FLOAT", {
                "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": (
                    "Fraction of the early (high-σ) steps DAVE is active. "
                    "KEEP ≤ 0.10 — wider windows garble text/hands far more than "
                    "extra dose does. Recommended 0.10. Set 0 to run on every step."
                ),
            }),
        }, "optional": {
            "pre_dd_sigma": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "With Anima Detail Daemon on the same model: judge DAVE's steps by the sigma "
                    "before Detail Daemon scaled it. Off = the original nodes chained, where the "
                    "scaled sigma is on no schedule and DAVE runs on every step."
                ),
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, mask="dave_alpha.npz", strength=0.3, tau=0.1, pre_dd_sigma=True):
        if not enabled:
            return (model,)
        # 'dave_alpha.npz' stays a COMBO choice so saved workflows validate; its
        # weight is 1 on blocks 8-18 and 0 elsewhere, which is the block list 8-18.
        spec = str(mask or "")
        if spec.casefold().startswith("blocks:"):
            spec = spec.split(":", 1)[1]
        elif not any(ch.isdigit() for ch in spec):
            spec = DAVE_DEFAULT_BLOCKS
        # One block wrapper for the standalone node and the Suite keeps the
        # gate and the positional transformer_options fallback identical.
        # A strength whose clip(s, 0, 1) is <= 1e-3 returns the model unchanged.
        return (_patch_anima_blocks(
            model,
            dave_enabled=True,
            dave_blocks=spec,
            dave_strength=float(strength),
            dave_tau=float(tau),
            slg_enabled=False,
            slg_blocks="",
            dave_pre_dd=bool(pre_dd_sigma),
        ),)
