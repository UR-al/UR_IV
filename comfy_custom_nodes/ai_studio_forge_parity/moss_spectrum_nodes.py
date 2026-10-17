"""MOSS Spectrum for ComfyUI (pack 1.9.0): ``ForgeNeoMossSpectrum`` (MODEL -> MODEL).

The Spectrum feature forecasting of sorryhyun/ComfyUI-Spectrum-KSampler's "KSampler (Spectrum for MOSS)"
(``moss_spectrum``) as a MODEL patch: any sampler node that samples the returned MODEL (KSampler, KSamplerAdvanced,
SamplerCustom, the pack's Hires and detailers) gets it, each sampling run starting fresh. The step count, the step
index and the branches come from the run (no steps input). Upstream's node also sets the flow shift and the uncond
block-22 skip and offers ``torch.compile``; here those stay separate nodes (``ModelSamplingSD3``,
``SkipLayerGuidanceDiTSimple``), and nothing is compiled.

A MODEL that is not MOSS-Image is returned unchanged with one warning. No torch or ComfyUI import at module level.
"""

from __future__ import annotations

from typing import Any

from .moss_spectrum import DEFAULTS, LOG_PREFIX, LOGGER, MOSS_HEAD, SpectrumSettings, apply_moss_spectrum, is_moss_model

CATEGORY = "AI Studio/Forge Neo parity/Generation"
NODE_ID = "ForgeNeoMossSpectrum"
DISPLAY_NAME = "Forge Neo MOSS Spectrum (Spectrum-KSampler)"
# The optional inputs, upstream's defaults (moss.py:255-265); shown as advanced widgets.
ADVANCED_INPUTS = ("window_size", "blend_w", "cheby_degree", "ridge_lambda")


def _float(default: float, low: float, high: float, step: float, tooltip: str, *, advanced: bool = False):
    options = {"default": default, "min": low, "max": high, "step": step, "tooltip": tooltip}
    if advanced:
        options["advanced"] = True
    return ("FLOAT", options)


def _int(default: int, low: int, high: int, tooltip: str, *, advanced: bool = False):
    options = {"default": default, "min": low, "max": high, "step": 1, "tooltip": tooltip}
    if advanced:
        options["advanced"] = True
    return ("INT", options)


class ForgeNeoMossSpectrum:
    """MODEL -> MODEL: Spectrum forecasting on every sampling run of a MOSS-Image MODEL."""

    CATEGORY = CATEGORY
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = (
        "Spectrum for MOSS-Image (ComfyUI-Spectrum-KSampler's MOSS patch): after a warmup, most steps skip the "
        "DiT's blocks - the image tokens before the head are forecast from the steps that ran (Chebyshev ridge + "
        "Taylor) and only MOSS-Image's head runs. The step count and index come from the sampler's sigma list. "
        "Other models are returned unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MOSS-Image MODEL (moss-image loader)."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = the MODEL is returned unchanged."}),
                "flex_window": _float(
                    DEFAULTS["flex_window"], 0.0, 2.0, 0.05,
                    "Window growth after each actual step. At 50 steps 0.25 / 0.5 / 1.0 run 22 / 19 / 16 "
                    "actual steps.",
                ),
                "warmup_steps": _int(DEFAULTS["warmup_steps"], 0, 10000, "Steps that always run the model first."),
                "tail_actual_steps": _int(
                    DEFAULTS["tail_actual_steps"], 0, 10000, "Last steps that always run the model.",
                ),
            },
            "optional": {
                "window_size": _float(
                    DEFAULTS["window_size"], 1.0, 10.0, 0.05,
                    "Starting window: after warmup an actual step runs every floor(window) steps.", advanced=True,
                ),
                "blend_w": _float(
                    DEFAULTS["blend_w"], 0.0, 1.0, 0.01,
                    "Weight of the Chebyshev forecast against the first-order Taylor step (1 = Chebyshev only).",
                    advanced=True,
                ),
                "cheby_degree": _int(
                    DEFAULTS["cheby_degree"], 1, 10, "Chebyshev polynomial degree of the forecast.", advanced=True,
                ),
                "ridge_lambda": _float(
                    DEFAULTS["ridge_lambda"], 0.001, 10.0, 0.001, "Ridge regularisation of the Chebyshev fit.",
                    advanced=True,
                ),
            },
        }

    def patch(self, model: Any, enabled: bool = True, flex_window: float = DEFAULTS["flex_window"],
              warmup_steps: int = DEFAULTS["warmup_steps"], tail_actual_steps: int = DEFAULTS["tail_actual_steps"],
              window_size: float = DEFAULTS["window_size"], blend_w: float = DEFAULTS["blend_w"],
              cheby_degree: int = DEFAULTS["cheby_degree"], ridge_lambda: float = DEFAULTS["ridge_lambda"]):
        if not enabled:
            return (model,)
        settings = SpectrumSettings(
            window_size=float(window_size), flex_window=float(flex_window), warmup_steps=int(warmup_steps),
            tail_actual_steps=int(tail_actual_steps), blend_w=float(blend_w), cheby_degree=int(cheby_degree),
            ridge_lambda=float(ridge_lambda),
        )
        if not is_moss_model(model):
            LOGGER.warning(
                "%s not applied - the MODEL is not MOSS-Image (its diffusion_model lacks %s); it samples unchanged.",
                LOG_PREFIX, ", ".join(MOSS_HEAD),
            )
            return (model,)
        return (apply_moss_spectrum(model, settings),)


NODE_CLASS_MAPPINGS = {NODE_ID: ForgeNeoMossSpectrum}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: DISPLAY_NAME}

__all__ = ["ADVANCED_INPUTS", "CATEGORY", "DISPLAY_NAME", "ForgeNeoMossSpectrum", "NODE_CLASS_MAPPINGS",
           "NODE_DISPLAY_NAME_MAPPINGS", "NODE_ID"]
