"""X/Y/Z axis list for the Anima studio (pure data; labels and order follow Forge Neo's plot).

``field`` is a dotted path into the studio state (``studio_state.DEFAULT_STATE``); several paths are
comma separated (``Size`` sets width and height), ``Nothing`` has none.  ``choices`` names the studio
route that lists the allowed values.  ``fmt`` says how a cell label is built: ``label_value`` is
"Label: value", ``value`` the value alone, ``basename`` the file name of a path, ``joined`` the tokens
joined with ", " and ``empty`` no label.  ``cost`` orders the loops: the dearest axis goes outermost so
heavy model reloads happen least often.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Fmt = Literal["label_value", "value", "basename", "joined", "empty"]


@dataclass(frozen=True)
class AxisSpec:
    id: str
    label: str
    kind: str
    field: str
    cost: float
    fmt: Fmt
    choices: Optional[str] = None


def _axis(id, label, kind, field, cost=0.0, fmt="label_value", choices=None) -> AxisSpec:
    return AxisSpec(id, label, kind, field, cost, fmt, choices)


AXES: tuple[AxisSpec, ...] = (
    _axis("nothing", "Nothing", "nothing", "", fmt="empty"),
    _axis("seed", "Seed", "int", "seed"),
    _axis("steps", "Steps", "int", "steps"),
    _axis("hires_steps", "Hires steps", "int", "hires.steps"),
    _axis("size", "Size", "size", "width,height"),
    _axis("cfg", "CFG Scale", "float", "cfg"),
    _axis("shift", "Shift", "float", "shift"),
    _axis("prompt_sr", "Prompt S/R", "str", "prompt", 0.1, "value"),
    _axis("prompt_order", "Prompt order", "permutations", "prompt", 0.1, "joined"),
    _axis("sampler", "Sampler", "choice", "sampler", fmt="value", choices="samplers"),
    _axis("hires_sampler", "Hires sampler", "choice", "hires.sampler", choices="samplers"),
    _axis("scheduler", "Schedule type", "choice", "scheduler", choices="schedulers"),
    _axis("checkpoint", "Checkpoint name", "choice", "unet", 2.0, "basename", "unets"),
    _axis("vae", "VAE", "choice", "vae", 0.7, choices="vaes"),
    _axis("denoising", "Denoising", "float", "i2i.denoise"),
    _axis("hires_upscaler", "Hires upscaler", "choice", "hires.upscaler", choices="upscalers"),
    _axis("text_encoder", "[Modules] Text Encoder", "choice", "te", 0.9, choices="tes"),
    _axis("dora_mode", "[DoRA] Inference mode", "choice", "dora_mode", 0.8, choices="dora_modes"),
    _axis("sam3_enable", "[SAM3] Enable", "bool", "sam3.on"),
    _axis("sam3_checkpoint", "[SAM3] Checkpoint", "choice", "sam3.checkpoint", 0.9, "basename", "sam3"),
    _axis("sam3_prompt", "[SAM3] Detect Prompt", "str", "sam3.prompt"),
    _axis("sam3_denoise", "[SAM3] Denoising Strength", "float", "sam3.denoise"),
    _axis("sam3_steps", "[SAM3] Steps", "int", "sam3.steps"),
    _axis("sam3_cfg", "[SAM3] CFG Scale", "float", "sam3.cfg"),
)

_BY_ID = {axis.id: axis for axis in AXES}


def axis_by_id(axis_id: str) -> AxisSpec:
    """The axis with this id; ``KeyError`` for an unknown id."""
    return _BY_ID[axis_id]
