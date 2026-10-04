"""Extra Samplers for ComfyUI (pack 1.7.0): ``ForgeNeoExtraSamplerSettings`` (MODEL -> MODEL).

sam-extra's "Extra Samplers" block holds five values the ER SDE samplers read — ``max_stage``, ``eta``,
``noise_window``, ``noise_start``, ``noise_end`` (``extra_samplers_math.ARG_NAMES``). The node puts them,
coerced like sam-extra, into the MODEL's ``model_options[MODEL_OPTIONS_KEY]``; ``CFGGuider`` hands the
model options to every sampler function, so every later pass of this MODEL (KSampler, Hires, ADetailer,
SAM3, a custom workflow's stock nodes) runs with them — Forge keeps one set of values per generation the
same way. Without the node the samplers run the defaults (3, 1.0, off, 0.2, 0.8), as Forge does for a
request without the block. Its position on the MODEL chain does not matter (it only adds an option).

A Hires checkpoint/UNET override keeps the values (``hires_carry`` key ``extra_samplers``). The optional
``owned_names`` COMBO lists the sampler names this pack registered in ComfyUI (``extra_samplers_register``);
the app reads it from ``/object_info`` and refuses a graph whose sampler another provider owns. This node
class is also the app's stale-pack marker for pack 1.7.0's samplers. No torch or ComfyUI import at module
level.
"""

from __future__ import annotations

from typing import Any, Mapping

from . import extra_samplers_math as xs
from . import hires_carry
from .compat import NO_OWNED_NAMES, OWNED_NAMES_INPUT, clone_model, owned_names_input
from .extra_samplers_register import owned_names

CATEGORY = "AI Studio/Forge Neo parity/Generation"
NODE_NAME = "ForgeNeoExtraSamplerSettings"
HIRES_CARRY_KEY = "extra_samplers"


def apply_settings(model: Any, settings: xs.Settings | Mapping[str, Any]) -> Any:
    """A clone of ``model`` whose ``model_options[MODEL_OPTIONS_KEY]`` holds the coerced ``settings``."""
    values = xs.coerce(settings)
    clone = clone_model(model, "Extra Samplers")
    options = getattr(clone, "model_options", None)
    if not isinstance(options, dict):
        options = {}
        clone.model_options = options
    options[xs.MODEL_OPTIONS_KEY] = values.as_dict()
    return clone


def model_settings(model: Any) -> xs.Settings | None:
    """The values ``model`` carries, or None when it carries none."""
    options = getattr(model, "model_options", None)
    if not isinstance(options, Mapping) or xs.MODEL_OPTIONS_KEY not in options:
        return None
    return xs.coerce(options.get(xs.MODEL_OPTIONS_KEY))


def _carry_to_override(source_model: Any, target_model: Any, target_vae: Any) -> Any:
    """Hires checkpoint/UNET override: the pass keeps the generation's values."""
    settings = model_settings(source_model)
    if settings is None or model_settings(target_model) == settings:
        return None
    return apply_settings(target_model, settings)


hires_carry.register(_carry_to_override, key=HIRES_CARRY_KEY)


class ForgeNeoExtraSamplerSettings:
    """MODEL -> MODEL: the Extra Samplers (ER SDE) values every later sampler of this MODEL uses."""

    CATEGORY = CATEGORY
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    DESCRIPTION = (
        "Extra Samplers (sam-extra parity): the ER SDE values the pack's samplers er_sde_reverse_time, "
        "er_sde_ode and er_sde_tunable read - solver stage, eta and the optional noise window - carried "
        "on the MODEL so every later sampler (KSampler, Hires, ADetailer, SAM3) uses them. ComfyUI's own "
        "er_sde ignores them."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "max_stage": ("INT", {
                    "default": xs.DEFAULT_MAX_STAGE, "min": xs.MAX_STAGE_MIN, "max": xs.MAX_STAGE_MAX, "step": 1,
                    "tooltip": "ER SDE solver stage: 1 = exponential Euler, 2 and 3 add the multistep terms.",
                }),
                "eta": ("FLOAT", {
                    "default": xs.DEFAULT_ETA, "min": xs.ETA_MIN, "max": xs.ETA_MAX, "step": xs.ETA_STEP,
                    "tooltip": "ER SDE eta (Reverse-time and Tunable): the noise scaler's exponent; 0 = the ODE.",
                }),
                "noise_window": ("BOOLEAN", {
                    "default": xs.DEFAULT_NOISE_WINDOW,
                    "tooltip": "Reverse-time and Tunable add SDE noise only on the steps that land inside "
                               "noise_start..noise_end (sampling progress 0-1); the other steps are ODE steps.",
                }),
                "noise_start": ("FLOAT", {
                    "default": xs.DEFAULT_NOISE_START, "min": xs.NOISE_PERCENT_MIN, "max": xs.NOISE_PERCENT_MAX,
                    "step": xs.NOISE_PERCENT_STEP,
                    "tooltip": "Noise window start (sampling progress). start >= end: no step gets noise.",
                }),
                "noise_end": ("FLOAT", {
                    "default": xs.DEFAULT_NOISE_END, "min": xs.NOISE_PERCENT_MIN, "max": xs.NOISE_PERCENT_MAX,
                    "step": xs.NOISE_PERCENT_STEP,
                    "tooltip": "Noise window end (sampling progress).",
                }),
            },
            "optional": {
                OWNED_NAMES_INPUT: owned_names_input(
                    owned_names(),
                    tooltip=(
                        "Samplers this node pack registered in ComfyUI (read by AI Studio; leave as it is). "
                        f"{NO_OWNED_NAMES} = another provider holds them or ComfyUI has no sampler table."
                    ),
                ),
            },
        }

    def patch(self, model, max_stage=xs.DEFAULT_MAX_STAGE, eta=xs.DEFAULT_ETA, noise_window=xs.DEFAULT_NOISE_WINDOW,
              noise_start=xs.DEFAULT_NOISE_START, noise_end=xs.DEFAULT_NOISE_END, **_optional):
        values = {"max_stage": max_stage, "eta": eta, "noise_window": noise_window, "noise_start": noise_start,
                  "noise_end": noise_end}
        return (apply_settings(model, values),)


NODE_CLASS_MAPPINGS = {NODE_NAME: ForgeNeoExtraSamplerSettings}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_NAME: "Forge Neo Extra Sampler Settings (sam-extra)"}
