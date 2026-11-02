"""X/Y/Z axis list for the Anima studio (pure data; labels and order follow Forge Neo's plot).

``field`` is a dotted path into the studio state (``studio_state.DEFAULT_STATE``); several paths are
comma separated (``Size`` sets width and height), ``Nothing`` has none.  ``choices`` names the studio
route that lists the allowed values.  ``fmt`` says how a cell label is built: ``label_value`` is
"Label: value", ``value`` the value alone, ``basename`` the file name of a path, ``joined`` the tokens
joined with ", " and ``empty`` no label.  ``cost`` orders the loops: the dearest axis goes outermost so
heavy model reloads happen least often.

``EXT_AXES`` are not Forge's: one axis per setting of the extension table (``web/ai_studio/ext_schema.json``), which
``studio_ext.js extAxes`` lists the same way.  ``values`` are a choice axis' fixed values.  ``switches`` are the state
keys a value turns on as well, so a column never shows a setting whose switch is off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from .studio_state import EXT_SCHEMA

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
    values: tuple = ()
    switches: tuple = ()  # ((dotted state path, value), ...)
    group: str = ""


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
    _axis("lora", "LoRA", "lora", "prompt", 0.5, "value"),
    _axis("sampler", "Sampler", "choice", "sampler", fmt="value", choices="samplers"),
    _axis("hires_sampler", "Hires sampler", "choice", "hires.sampler", choices="samplers"),
    _axis("scheduler", "Schedule type", "choice", "scheduler", choices="schedulers"),
    _axis("checkpoint", "Checkpoint name", "choice", "unet", 2.0, "basename", "unets"),
    _axis("vae", "VAE", "choice", "vae", 0.7, choices="vaes"),
    _axis("denoising", "Denoising", "float", "i2i.denoise"),
    _axis("hires_upscaler", "Hires upscaler", "choice", "hires.upscaler", choices="upscalers"),
    _axis("text_encoder", "[Modules] Text Encoder", "choice", "te", 0.9, choices="tes"),
    _axis("dora_mode", "[DoRA] Inference mode", "choice", "dora_mode", 0.8, choices="dora_modes"),
    _axis("attention", "[Attention] Backend", "choice", "attention", 0.3, choices="attentions"),
    _axis("sam3_enable", "[SAM3] Enable", "bool", "sam3.on"),
    _axis("sam3_checkpoint", "[SAM3] Checkpoint", "choice", "sam3.checkpoint", 0.9, "basename", "sam3"),
    _axis("sam3_prompt", "[SAM3] Detect Prompt", "str", "sam3.prompt"),
    _axis("sam3_denoise", "[SAM3] Denoising Strength", "float", "sam3.denoise"),
    _axis("sam3_steps", "[SAM3] Steps", "int", "sam3.steps"),
    _axis("sam3_cfg", "[SAM3] CFG Scale", "float", "sam3.cfg"),
)

EXT_COST = 0.2  # a model patch to redo, no file to load
_S2_MODE = "Stochastic (S²)"
_LEFT_OUT_TEXT = re.compile(r"prompt|path|clip_model")  # text boxes that hold prompts or file paths
# The guidance suite's table has one switch per section, some features share a section.
_SECTION_SWITCHES = {
    "PAG·SEG": {"guid_enabled": True}, "스케줄": {"guid_enabled": True}, "SLG·S²": {"guid_slg_on": True},
    "APG": {"guid_apg_enabled": True}, "ADG": {"guid_adg_enabled": True}, "SMC": {"guid_smc_master_enabled": True},
    "CWM": {"guid_cwm_enabled": True}, "DCW·RDC": {"guid_dcw_enabled": True}, "DAVE": {"guid_dave_enabled": True},
    "CNS": {"guid_cns_enabled": True}, "Mod": {"guid_mod_enabled": True}, "TSR": {"guid_tsr_enabled": True},
    "MG": {"guid_mg_enabled": True}, "HiGS": {"guid_higs_enabled": True}, "HiFlow": {"guid_hiflow_enabled": True},
}
_FIELD_SWITCHES = (  # (key prefix, what those settings need beyond the section's)
    ("guid_s2_", {"guid_slg_mode": _S2_MODE}),
    ("guid_rdc_", {"guid_rdc_enabled": True, "guid_dcw_enabled": True}),
)
_SWITCH_ENDS = ("_enabled", "_on")


def _field_switches(key: str, field: dict) -> tuple:
    own = field["key"]
    wanted = {"on": True}
    if not own.endswith(_SWITCH_ENDS):  # an axis of a switch compares it: its section's other switch must not stay on
        wanted.update(_SECTION_SWITCHES.get(field.get("section"), {}))
    for prefix, more in _FIELD_SWITCHES:
        if own.startswith(prefix):
            wanted.update(more)
    wanted.pop(own, None)
    return tuple((f"ext.{key}.{name}", value) for name, value in wanted.items())


def _ext_field_axis(item: dict, field: dict) -> Optional[AxisSpec]:
    kind = field["kind"]
    if kind == "choice" and not field["choices"]:  # a file list of the user's ComfyUI
        return None
    if kind == "text" and _LEFT_OUT_TEXT.search(field["key"]):
        return None
    section = f" {field['section']}" if field.get("section") else ""
    return AxisSpec(f"ext.{item['key']}.{field['key']}", f"[{item['label']}{section}] {field['label']}",
                    "lines" if kind == "text" else kind, f"ext.{item['key']}.{field['key']}", EXT_COST, "label_value",
                    values=tuple(field.get("choices") or ()), switches=_field_switches(item["key"], field), group=item["label"])


def _ext_axes() -> tuple[AxisSpec, ...]:
    axes = []
    for item in EXT_SCHEMA["extensions"]:
        axes.append(AxisSpec(f"ext.{item['key']}.on", f"[{item['label']}] 켜기", "bool", f"ext.{item['key']}.on", EXT_COST,
                             "label_value", group=item["label"]))
        axes += filter(None, (_ext_field_axis(item, field) for field in item["fields"]))
    return tuple(axes)


EXT_AXES = _ext_axes()
_BY_ID = {axis.id: axis for axis in (*AXES, *EXT_AXES)}


def ext_key(axis: AxisSpec) -> Optional[str]:
    """The extension an axis belongs to, None for the other axes."""
    return axis.id.split(".")[1] if axis.id.startswith("ext.") else None


def axis_by_id(axis_id: str) -> AxisSpec:
    """The axis with this id; ``KeyError`` for an unknown id."""
    return _BY_ID[axis_id]
