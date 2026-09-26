"""Detail Daemon as a ComfyUI MODEL patch (``ForgeNeoAnimaDetailDaemon``).

Behaves exactly like the original ComfyUI node, the "Detail Daemon Sampler" of
Jonseed/ComfyUI-Detail-Daemon (``DetailDaemonSamplerNode``), pinned at
origin: Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b:detail_daemon_node.py

* The node's schedule, sigma lookup and sampler wrapper are copied below unchanged (MIT, notice
  kept): ``make_detail_daemon_schedule`` (:25-67), ``get_dd_schedule`` (:226-262) and
  ``detail_daemon_sampler`` (:265-310). The sigma each model call gets is looked up in the
  sampler's own sigma list (interpolated between neighbours, e.g. second-order midpoints) and
  scaled by ``max(1e-06, 1 - schedule * 0.1 * cfg)``: no other clamp, no presets.
* The original is a SAMPLER node. This pack node is a MODEL patch so the app's graphs keep their
  model chain: it registers a ``SAMPLER_SAMPLE`` wrapper that swaps the sampler of every sampling
  run of the patched model for ``KSAMPLER(detail_daemon_sampler, ...)``, built exactly as
  ``DetailDaemonSamplerNode.go`` (:371-410) builds it. The adjusted sigma is therefore what the
  whole CFG step sees (cond/uncond batches, cfg functions, post-CFG hooks, inpaint latent
  blending), as with the original node.
* Inputs: ``settings_json`` carries the node's values under the app's keys (``SETTING_KEYS``).
  Missing keys take the node's defaults; values outside the node's min/max are refused, as
  ComfyUI's prompt validation refuses them for the original widgets. ``cfg_scale_override`` is
  the node's own input (0 = the sampler's CFG, i.e. the guider's ``cfg``). Keys the original has
  no input for (the app's hidden ``dd_preset``/``dd_multiplier``/``dd_cfg_couple`` slots, the
  Forge-only ``dd_hires``) are not read.
* Which sampling passes get it is decided by the graph, following what Forge does with
  muerrilla/sd-webui-detail-daemon and with the sam-extra extension: the app compiler
  (``core/comfy_workflow_compiler.py`` ``_add_detail_daemon``) hands this model to the main pass
  the user chose (base, or hires with Hires Pass), to ADetailer when that pass was the last main
  pass (the original's callback stays registered until ``postprocess``, after ADetailer ran), and
  to SAM3 detailer passes without Hires Pass (the extension re-runs the script on its img2img
  pass, with that pass's CFG and sampler). Standalone post-processing gets none. Each detailer
  sampling run builds its schedule from its own sigma list, as the extension does.

numpy and torch are imported lazily (the app imports this module without them); the copied
functions use the module globals ``np`` and ``torch`` that ``_bind_numpy``/``_bind_torch`` fill.
"""

from __future__ import annotations

import importlib
from typing import Any

from .compat import clone_model, require_torch
from .guidance_common import CATEGORY, PRE_DD_SIGMAS_KEY, _as_bool, _json_settings, _setting

ORIGIN = "Jonseed/ComfyUI-Detail-Daemon@3394e44afea04ed0188fb37b21f0d9952469766b:detail_daemon_node.py"

# The node multiplies the schedule by 0.1 before the cfg scale (:169, :294 below).
# core/sam_extra_contract.py SEMANTIC_PINS (dd_sigma_scale_comfy) pins it against the extension.
DD_SIGMA_SCALE = 0.1

# settings_json key -> the original node's input name (DetailDaemonSamplerNode.INPUT_TYPES).
SETTING_KEYS = (
    ("dd_amount", "detail_amount"),
    ("dd_start", "start"),
    ("dd_end", "end"),
    ("dd_bias", "bias"),
    ("dd_exponent", "exponent"),
    ("dd_start_offset", "start_offset"),
    ("dd_end_offset", "end_offset"),
    ("dd_fade", "fade"),
    ("dd_smooth", "smooth"),
)

# ModelPatcher.add_wrapper_with_key key of the SAMPLER_SAMPLE wrapper this node registers.
DD_WRAPPER_KEY = "ai_studio_forge_parity.detail_daemon"

np: Any = None
torch: Any = None


def _bind_numpy() -> None:
    global np
    if np is None:
        np = importlib.import_module("numpy")


def _bind_torch() -> None:
    global torch
    if torch is None:
        torch = require_torch()


# ── Copied from the original node (MIT) ─────────────────────────────────────────────────────
# ComfyUI-Detail-Daemon — Copyright (c) 2024 Jonseed — MIT License
# (full text: LICENSES/ComfyUI-Detail-Daemon-MIT.txt)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The three functions and the INPUT_TYPES body are byte-for-byte the origin's lines (only the
# CRLF line ends differ); the node classes are not copied (they import matplotlib/folder_paths).
#
# make_detail_daemon_schedule is Jonseed's port of make_schedule from
# muerrilla/sd-webui-detail-daemon@19479998340831d7804fca8efd3f262b54b6373f:scripts/detail_daemon.py:309-338
# sd-webui-detail-daemon — Copyright (c) 2024 Sahand Ahmadian — MIT License
# (full text: LICENSES/sd-webui-detail-daemon-MIT.txt; same permission notice as above)


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:24-67
# Schedule creation function from https://github.com/muerrilla/sd-webui-detail-daemon
def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:226-262
def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item()
    # If we're here, that means the sigma is in between two sigmas in the
    # list.
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        # Shouldn't be possible, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow]
    # Ratio of how close we are to the high neighbor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    # Mix the DD schedule high/low items according to the ratio.
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


# origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:265-310
def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
) -> torch.Tensor:
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


def origin_input_types() -> dict:
    # origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:320-369
    # (DetailDaemonSamplerNode.INPUT_TYPES body)
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": (
                    "FLOAT",
                    {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01},
                ),
                "start": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "bias": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05},
                ),
                "start_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "end_offset": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "fade": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "round": 0.01,
                        "tooltip": "If set to 0, the sampler will automatically determine the CFG scale (if possible). Set to some other value to override.",
                    },
                ),
            },
        }


# ── End of the copied code ──────────────────────────────────────────────────────────────────


def _node_input(name: str, raw: Any) -> Any:
    """One value as ComfyUI's prompt validation hands it to the original node's input.

    FLOAT is converted with ``float()`` and refused below ``min``/above ``max``, BOOLEAN keeps the
    pack's lenient string parsing (``"false"``). Same checks as ComfyUI ``execution.validate_inputs``.
    """
    kind, options = origin_input_types()["required"][name]
    if kind == "BOOLEAN":
        return _as_bool(raw, bool(options["default"]))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Detail Daemon {name}: failed to convert {raw!r} to a FLOAT value") from exc
    if "min" in options and value < options["min"]:
        raise ValueError(
            f"Detail Daemon {name}: value {value} smaller than min of {options['min']} "
            "(the original node's input range)"
        )
    if "max" in options and value > options["max"]:
        raise ValueError(
            f"Detail Daemon {name}: value {value} bigger than max of {options['max']} "
            "(the original node's input range)"
        )
    return value


def detail_daemon_node_values(settings: dict[str, Any], cfg_scale_override: Any = 0.0) -> dict[str, Any]:
    """The original node's inputs from a settings dict (missing keys = the node's defaults)."""
    required = origin_input_types()["required"]
    values = {
        name: _node_input(name, _setting(settings, key, required[name][1]["default"]))
        for key, name in SETTING_KEYS
    }
    values["cfg_scale_override"] = _node_input(
        "cfg_scale_override",
        required["cfg_scale_override"][1]["default"] if cfg_scale_override is None else cfg_scale_override,
    )
    return values


def detail_daemon_schedule(steps: int, values: dict[str, Any]):
    """``dds_make_schedule(steps)`` of ``DetailDaemonSamplerNode.go`` (origin :387-399)."""
    _bind_numpy()
    return make_detail_daemon_schedule(
        steps,
        values["start"],
        values["end"],
        values["bias"],
        values["detail_amount"],
        values["exponent"],
        values["start_offset"],
        values["end_offset"],
        values["fade"],
        values["smooth"],
    )


class _PreDDSigmaSampler:
    """``sampler`` whose model calls first note the sampler's own sigma for the DAVE gate.

    The vendored ``detail_daemon_sampler`` hands its ``model_wrapper`` (which scales the sigma) to
    ``dds_wrapped_sampler.sampler_function``; this stands in for that sampler, so the sampler calls
    the recorder with its own sigma, and the recorder calls the unchanged ``model_wrapper``. The
    note goes into this sampling run's ``model_options["transformer_options"]`` (a per-run clone),
    which every model call copies into its forward's ``transformer_options``.
    """

    def __init__(self, sampler: Any):
        self._sampler = sampler
        self.extra_options = sampler.extra_options

    def sampler_function(self, model_wrapper, x, sigmas, **kwargs):
        def noting_wrapper(x, sigma, **extra_args):
            model_options = extra_args.get("model_options")
            if isinstance(model_options, dict):
                model_options.setdefault("transformer_options", {})[PRE_DD_SIGMAS_KEY] = sigma
            return model_wrapper(x, sigma, **extra_args)

        for key in ("inner_model", "sigmas"):
            if hasattr(model_wrapper, key):
                setattr(noting_wrapper, key, getattr(model_wrapper, key))
        return self._sampler.sampler_function(noting_wrapper, x, sigmas, **kwargs)


def detail_daemon_ksampler(sampler: Any, values: dict[str, Any]) -> Any:
    """The ``KSAMPLER`` ``DetailDaemonSamplerNode.go`` returns for ``sampler`` (origin :401-410).

    Like the original, the wrapper sampler has the default (empty) inpaint options.
    """
    samplers = importlib.import_module("comfy.samplers")
    _bind_numpy()
    _bind_torch()

    def dds_make_schedule(steps):
        return detail_daemon_schedule(steps, values)

    return samplers.KSAMPLER(
        detail_daemon_sampler,
        extra_options={
            "dds_wrapped_sampler": _PreDDSigmaSampler(sampler),
            "dds_make_schedule": dds_make_schedule,
            "dds_cfg_scale_override": values["cfg_scale_override"],
        },
    )


def detail_daemon_sampler_sample_wrapper(values: dict[str, Any]):
    """SAMPLER_SAMPLE wrapper: run the rest of the chain with the node's KSAMPLER in place.

    ``executor.class_obj`` is the sampler this sampling run picked (``CFGGuider.inner_sample``).
    The remaining wrappers and the final ``sample`` call see the Detail Daemon KSAMPLER, as they
    would if the graph had routed that sampler through the original SAMPLER node.
    """

    def detail_daemon_sampler_sample(executor, *args, **kwargs):
        sampler = executor.class_obj
        if not hasattr(sampler, "sampler_function") or not hasattr(sampler, "extra_options"):
            raise RuntimeError(
                "Anima Detail Daemon wraps KSAMPLER samplers only, like the original node; "
                f"this sampling run uses {type(sampler).__name__}."
            )
        dd_sampler = detail_daemon_ksampler(sampler, values)
        return type(executor).new_class_executor(
            dd_sampler.sample, dd_sampler, executor.wrappers, idx=executor.idx + 1,
        ).execute(*args, **kwargs)

    return detail_daemon_sampler_sample


def _attach_detail_daemon(patched: Any, values: dict[str, Any]) -> None:
    try:
        wrappers = importlib.import_module("comfy.patcher_extension").WrappersMP
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Anima Detail Daemon needs ComfyUI's sampler wrappers (comfy.patcher_extension, "
            "ComfyUI 0.3.10 or newer). Update ComfyUI."
        ) from exc
    add_wrapper = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add_wrapper):
        raise RuntimeError(
            "Anima Detail Daemon needs a ComfyUI MODEL that accepts sampler wrappers "
            "(ModelPatcher.add_wrapper_with_key). Update ComfyUI."
        )
    add_wrapper(wrappers.SAMPLER_SAMPLE, DD_WRAPPER_KEY, detail_daemon_sampler_sample_wrapper(values))


class ForgeNeoAnimaDetailDaemon:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": False}),
                "settings_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            # The original node's own input. Also marks this pack version: a stale pack's
            # /object_info lacks it, so the app compiler refuses the graph before queueing.
            "optional": {
                "cfg_scale_override": origin_input_types()["required"]["cfg_scale_override"],
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, settings_json="{}", cfg_scale_override=0.0):
        """Wrap every sampling run of ``model`` in the original Detail Daemon sampler."""
        if not enabled:
            return (model,)
        settings = _json_settings(settings_json, "Anima Detail Daemon")
        if not _as_bool(_setting(settings, "dd_enabled", True), True):
            return (model,)
        values = detail_daemon_node_values(settings, cfg_scale_override)
        patched = clone_model(model, "Anima Detail Daemon")
        _attach_detail_daemon(patched, values)
        return (patched,)
