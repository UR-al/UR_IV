"""Skimmed CFG (``ForgeNeoSkimmedCFG``) — the original node, run unchanged.

origin: Extraltodeus/Skimmed_CFG@d83005832ac42783adfd6f4ae96f6ef6406d1a74:skimmed_CFG.py:83-201
(``CFG_Skimming_Single_Scale_Pre_CFG``, Apache-2.0). The upstream file is vendored
unchanged under ``vendor/skimmed_cfg/`` (pinned, see its ``UPSTREAM.md``, the
license in ``LICENSES/Skimmed_CFG-Apache-2.0.txt`` and ``THIRD_PARTY_NOTICES.md``).

This node only forwards its inputs to the upstream ``execute``, which does
everything the original node does:

- reads ``model_sampling`` when the node runs and turns start/end/flip into
  sigmas with ``percent_to_sigma`` (the flow shift node sits before this node,
  so the shift is already applied);
- clones the model and registers ``pre_cfg_patch`` with
  ``set_model_sampler_pre_cfg_function`` — it rewrites ``conds_out`` in place
  between ``calc_cond_batch`` and ``cfg_function``, so a
  ``sampler_cfg_function`` (the suite's CWM/SMC/APG) and every post-CFG hook
  (PAG, DCW, ...) see the skimmed predictions;
- gates on the model-call sigma with the upstream strict comparisons (Anima's
  first step, sigma 1.0 = ``percent_to_sigma(0)``, is not skimmed) and flips
  the filter while sigma is above the flip sigma;
- does not set the cfg1 flag, so at CFG 1 Comfy normally skips the negative
  pass and the patch leaves the all-zero uncond alone. That holds only while
  no other node sets ``disable_cfg1_optimization``. This pack's SEG/SLG,
  Adaptive Guidance and CWM/SMC/APG set it, and third-party nodes can too.
  Comfy then computes a real uncond at CFG exactly 1. Upstream skims the
  positive at ``cond_scale - 1`` = 0 and divides by that (skimmed_CFG.py:189-196,
  :51-53), so every element in the skim mask of the positive prediction
  becomes inf or NaN, and the linear CFG combine carries it into the denoised
  output. The pack keeps this original behaviour. The Forge extension's CFG-1
  guard, which skips this case, is a Forge-only host difference.

Local differences, all at the node boundary: the ``enabled`` switch, the input
names ``start_percent``/``end_percent``/``flip_percent`` (the app compiler's
contract for upstream ``start_at_percentage``/``end_at_percentage``/
``flip_at_percentage``) and ``skimming_cfg`` min -1 — upstream's widget min is
0 but its tooltip, and its Timed flip / Clean Skim presets, use -1 for "the
current CFG scale".
"""

from __future__ import annotations

import importlib

from .guidance_common import CATEGORY

# origin: skimmed_CFG.py:5-6 (MAX_SCALE 10, STEP_STEP 2 -> step 0.5), :93-134 (inputs).
# Literal here so INPUT_TYPES never imports torch (the app reads it without ComfyUI).
_MAX_SCALE = 10.0
_SCALE_STEP = 0.5
_PERCENT_STEP = 0.01
_UPSTREAM_MODULE = ".vendor.skimmed_cfg.skimmed_CFG"


def _upstream():
    """Import the vendored original lazily — it needs torch and ComfyUI's ``comfy_api.latest``."""
    try:
        return importlib.import_module(_UPSTREAM_MODULE, __package__)
    except ImportError as exc:
        raise RuntimeError(
            "Skimmed CFG is enabled, but the vendored original node could not be "
            f"imported ({exc}). It needs ComfyUI's comfy_api.latest; update ComfyUI."
        ) from exc


class ForgeNeoSkimmedCFG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "enabled": ("BOOLEAN", {"default": False}),
            "skimming_cfg": ("FLOAT", {
                "default": 7.0, "min": -1.0, "max": _MAX_SCALE, "step": _SCALE_STEP,
                "tooltip": "The fallback scale for the 'bad' values. Set to -1 to use the current CFG scale.",
            }),
            "full_skim_negative": ("BOOLEAN", {
                "default": False,
                "tooltip": "If enabled, fully skim negative conditioning (set to 0).",
            }),
            "disable_flipping_filter": ("BOOLEAN", {
                "default": False,
                "tooltip": "Disable the flipping filter for skimming detection.",
            }),
            "start_percent": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": _PERCENT_STEP,
                "tooltip": "Start applying skimming at this percentage of the denoising process (0 = start, 1 = end).",
            }),
            "end_percent": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": _PERCENT_STEP,
                "tooltip": "Stop applying skimming at this percentage of the denoising process (0 = start, 1 = end).",
            }),
            "flip_percent": ("FLOAT", {
                "default": 0.0, "min": 0.0, "max": 1.0, "step": _PERCENT_STEP,
                "tooltip": "Flip the flipping filter at this percentage. Set to 0 to disable.",
            }),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, enabled=False, skimming_cfg=7.0, full_skim_negative=False, disable_flipping_filter=False, start_percent=0.0, end_percent=1.0, flip_percent=0.0):
        if not enabled:
            return (model,)
        if not callable(getattr(model, "get_model_object", None)):
            raise RuntimeError("Skimmed CFG requires a ComfyUI MODEL input.")
        output = _upstream().CFG_Skimming_Single_Scale_Pre_CFG.execute(
            model=model,
            skimming_cfg=float(skimming_cfg),
            full_skim_negative=bool(full_skim_negative),
            disable_flipping_filter=bool(disable_flipping_filter),
            start_at_percentage=float(start_percent),
            end_at_percentage=float(end_percent),
            flip_at_percentage=float(flip_percent),
        )
        return (output.args[0],)
