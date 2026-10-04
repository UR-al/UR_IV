"""``ForgeNeoColorcraft``: sam-extra v0.32.1's Colorcraft (sampling-time colour grading) as a ComfyUI MODEL patch.

Pack 1.7.0. The app compiler adds one node per pass that grades something — the base sampler's MODEL and the
MODEL handed to ``ForgeNeoHiresFix`` — as the last patch before the sampler, so its post-CFG function runs after
Optimal Scale, the guidance suite, Skimmed CFG and Detail Daemon (Forge's order). Detailer passes never get it
(Forge skips the SAM3/ADetailer inner passes).

Inputs: ``model``; ``vae`` (the pass's VAE: the colour anchors are encoded with it when the node runs, before
sampling, as Forge encodes them before its pass); ``settings_json`` (``colorcraft_host.parse_settings``: the
pass's chain entries the app built, the infotext, optional tab names); ``contract`` (the settings format — a
later format is a new choice, so an older pack refuses it at ComfyUI's input validation, before ``/prompt`` runs);
``pre_dd_sigma`` (look the schedule up by the sampler's own sigma when Detail Daemon scaled it; sam-extra's
``sam3_colorcraft_pre_dd_sigma`` default, on).

No chain entries -> the input MODEL itself (no clone). A Hires checkpoint/UNET override keeps the hires pass's
Colorcraft: ``hires_carry`` prepares it again on the override MODEL with the override VAE (Forge re-attaches it
to the new UNet and encodes with that model's VAE).

No torch or ComfyUI import at module import (``colorcraft_host`` imports torch when a node runs).
"""

from __future__ import annotations

from . import colorcraft_host, hires_carry
from .guidance_common import CATEGORY

NODE_ID = "ForgeNeoColorcraft"
DISPLAY_NAME = "Colorcraft (Forge parity)"
# settings_json formats this pack reads (the app compiler sends core/colorcraft.COMFY_CONTRACT).
CONTRACTS = ("v1",)
# hires_carry key: a Hires checkpoint override keeps the hires pass's Colorcraft.
CARRY_KEY = "colorcraft"


def _attach_line(prepared) -> str:
    kinds = ", ".join(f"{tag} {entry['kind']}" for tag, entry in zip(prepared.tags, prepared.entries))
    return f"{prepared.pass_name} pass: {prepared.family_label}; {kinds}"


def colorcraft_model(model, vae, settings: dict, pre_dd: bool = True):
    """``model`` graded by the parsed ``settings`` (``model`` itself when nothing applies on it)."""
    if not settings["entries"]:
        return model
    prepared = colorcraft_host.prepare(model, vae, settings, pre_dd)
    if not prepared.entries:
        colorcraft_host.log_line(f"{prepared.pass_name} pass: {prepared.family_label}; "
                                 + "; ".join(prepared.notes))
        return model
    patched = colorcraft_host.apply_colorcraft(model, colorcraft_host.ColorcraftPatch(prepared, settings))
    colorcraft_host.log_line(_attach_line(prepared))
    return patched


class ForgeNeoColorcraft:
    """sam-extra's Colorcraft for ComfyUI: modifiers graded on the CFG-combined prediction at every step."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    DESCRIPTION = (
        "Colorcraft (sam-extra v0.32.1, muerrilla/ComfyUI-Colorcraft): exposure, contrast, colour and detail "
        "modifiers applied to the denoised prediction during sampling, scheduled over the steps and optionally "
        "masked. Settings come from the AI Studio app."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE", {"tooltip": "The pass's VAE: colour anchors are encoded with it before sampling"}),
                "settings_json": ("STRING", {"default": "", "multiline": True,
                                             "tooltip": "The pass's Colorcraft chain (written by the AI Studio app)"}),
                "contract": (list(CONTRACTS), {"default": CONTRACTS[0]}),
                "pre_dd_sigma": ("BOOLEAN", {"default": True,
                                             "tooltip": "With Detail Daemon: follow the sampler's own sigma"}),
            }
        }

    def patch(self, model, vae, settings_json, contract=CONTRACTS[0], pre_dd_sigma=True):
        if contract not in CONTRACTS:
            raise ValueError(f"Colorcraft settings format {contract!r} is not supported by this node pack "
                             f"(supported: {', '.join(CONTRACTS)}); update the bundled node pack.")
        settings = colorcraft_host.parse_settings(settings_json)
        return (colorcraft_model(model, vae, settings, bool(pre_dd_sigma)),)


def _carry_to_override(source_model, target_model, target_vae):
    """``hires_carry``: the override MODEL gets the hires pass's Colorcraft, prepared again with its VAE."""
    patch = colorcraft_host.model_colorcraft_patch(source_model)
    if patch is None or patch.settings is None:
        return None
    carried = colorcraft_model(target_model, target_vae, patch.settings, patch.prepared.pre_dd)
    return None if carried is target_model else carried


NODE_CLASS_MAPPINGS = {NODE_ID: ForgeNeoColorcraft}
NODE_DISPLAY_NAME_MAPPINGS = {NODE_ID: DISPLAY_NAME}

hires_carry.register(_carry_to_override, key=CARRY_KEY)
