"""Colorcraft on ComfyUI: the host rules of sam-extra v0.32.1's Forge hook, in ComfyUI terms.

Pack 1.7.0 (``ForgeNeoColorcraft``). The per-evaluation maths is ``colorcraft_engine`` (the MIT upstream's
modifier loop); this module decides what one sampling pass grades and with what. The rules and much of the code
are sam-extra's (``sam3ext/colorcraft/hook.py``; sam-extra is GPL-3.0-only, and this code is licensed to this
pack under MIT by its author (THIRD_PARTY_NOTICES.md)), and ``tests/test_comfy_colorcraft_parity.py`` checks
them against the installed extension on CPU.

Part 1 — before sampling (the node runs before its sampler, as Forge's ``process_before_every_sampling``):

* ``parse_settings`` reads the node's ``settings_json`` — ``{"v": 1, "pass": "base"|"hires", "entries": [...],
  "infotext": "...", "tags": [...]}`` with ``entries`` the chain entries the app's ``core/colorcraft.build_chain``
  builds (``{"kind", "params", "schedule", "mask"}``, upstream's node format) and the optional ``tags`` the tab
  names for the console line. Anything else is refused with a ``ValueError`` (the app never sends it).
* ``prepare`` resolves the basis family from the MODEL's latent format, keeps only the entries that can do
  something on that model (no family = SD 1.5/SDXL & co.: Contrast and Color Shift only), and encodes the
  colour anchors those entries need with the pass's VAE now — a flat 512x512 image per colour
  (upstream ``build_color_latent``), lifted to model space with the rank the encode really returned and
  kept in float32. Encoded anchors are cached across prompts per VAE (weakly referenced encoder, its dtype,
  device and weight patches, the latent format, the rounded colour; 32 entries, least recently used out).
  A failed encode drops Contrast/Color Shift for the pass (the other edits still run).

Part 2 — during sampling: the comment above ``RUN_KEY``.

torch is imported only when anchors are encoded or a run is graded: the module loads with the pack.
"""

from __future__ import annotations

import json
import math
import os
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

from . import colorcraft_basis
from .colorcraft_engine import KINDS, needed_colors, needs_basis
from .compat import LOGGER, clone_model, require_torch, sampler_sample_wrapper_type
from .guidance_common import PRE_DD_SIGMAS_KEY, is_extra_sampler_substep, republished_from

LOG_PREFIX = "[AI Studio Colorcraft]"
SETTINGS_VERSION = 1
PASSES = ("base", "hires")

# Each upstream node's chain-entry params (``make`` of ColorcraftBasic ... ColorcraftShift: its widgets minus
# the schedule ones; Advanced's dev overrides included). Value kinds: bool flags, choices, numbers.
_SHIFT = ("color_shift_amount", "mode", "red", "green", "blue", "brightness")
_CHROMA_PLUS = ("temp_plus_tint", "temp_minus_tint", "lab_a", "lab_b", "lab_a_plus_b", "lab_a_minus_b")
KIND_PARAMS: dict[str, tuple[str, ...]] = {
    "advanced": ("exposure", "tone_compression", "contrast", "clarity", "sharpness", "temperature", "tint",
                 "vibrance", "saturation", "chroma_contrast", "chroma_center", "more_colors") + _CHROMA_PLUS
                + ("color_shift",) + _SHIFT + ("recenter_override", "recenter", "max_chroma_override",
                                               "max_chroma", "chroma_plane_override", "chroma_plane"),
    "basic": ("contrast",) + _SHIFT,
    "luma": ("exposure", "tone_compression"),
    "chroma": ("temperature", "tint", "vibrance", "saturation", "chroma_contrast", "chroma_center"),
    "chroma_plus": _CHROMA_PLUS,
    "punch": ("contrast", "clarity", "sharpness"),
    "shift": _SHIFT,
}
BOOL_PARAMS = frozenset({"more_colors", "color_shift", "recenter_override", "max_chroma_override",
                         "chroma_plane_override"})
CHOICE_PARAMS = {"mode": ("default", "legacy"), "chroma_plane": ("temp_tint", "lab")}
SCHEDULE_KEYS = ("strength", "start", "end", "bias", "exponent", "start_off", "end_off", "smooth")
# Basic has no masking input (upstream ColorcraftBasic).
TAKES_MASK = {kind: kind != "basic" for kind in KINDS}
# Mask specs (upstream's COLORCRAFT_MASK): a leaf, a blur/shaping node over one spec, or a combine node.
MASK_AXES = ("exposure", "hue", "saturation", "temperature", "tint", "temp+tint", "temp-tint", "lab-a", "lab-b",
             "lab-a+b", "lab-a-b", "clarity", "sharpness")
MASK_MODES = ("highs", "lows", "split", "range", "protect range")
COMBINE_OPERATIONS = ("and", "or", "subtract", "xor")
LEAF_KEYS = ("mask_axis", "mask_mode", "mask_center", "mask_hardness", "mask_width", "mask_strength")
SHAPING_KEYS = ("blur", "spread", "contrast", "normalize", "a")
COMBINE_KEYS = ("operation", "a", "b")
_MAX_MASK_DEPTH = 32
# The edits each kind makes (the amounts its loop branch multiplies by ``s``); with all of them at 0 the
# entry hands its input back. Advanced's Chroma Plus and Color Shift amounts count only with their gate on.
_EDIT_AMOUNTS = {
    "basic": ("contrast", "color_shift_amount"),
    "advanced": ("exposure", "tone_compression", "contrast", "clarity", "sharpness", "temperature", "tint",
                 "vibrance", "saturation", "chroma_contrast"),
    "luma": ("exposure", "tone_compression"),
    "chroma": ("temperature", "tint", "vibrance", "saturation", "chroma_contrast"),
    "chroma_plus": _CHROMA_PLUS,
    "punch": ("contrast", "clarity", "sharpness"),
    "shift": ("color_shift_amount",),
}

_warned: set[str] = set()


def log_line(message: str) -> None:
    """One ``[AI Studio Colorcraft]`` console line; never raises."""
    try:
        LOGGER.info("%s %s", LOG_PREFIX, message)
    except Exception:  # pragma: no cover - a broken handler must not fail the node
        pass


def _log_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log_line(message)


# ---------------------------------------------------------------------------------------------------------
# settings_json
# ---------------------------------------------------------------------------------------------------------

def _refuse(where: str, problem: str):
    raise ValueError(f"Colorcraft 설정을 읽지 못했습니다 ({where}): {problem}")


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(where, f"숫자가 아닙니다: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        _refuse(where, f"유한한 숫자가 아닙니다: {value!r}")
    return number


def _flag(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        _refuse(where, f"true/false 가 아닙니다: {value!r}")
    return value


def _choice(value: Any, choices: tuple[str, ...], where: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _refuse(where, f"{value!r} 는 {', '.join(choices)} 중 하나가 아닙니다")
    return value


def _exact_keys(mapping: Any, keys: tuple[str, ...], where: str) -> dict:
    if not isinstance(mapping, dict):
        _refuse(where, "객체가 아닙니다")
    missing = [key for key in keys if key not in mapping]
    extra = sorted(str(key) for key in mapping if key not in keys)
    if missing or extra:
        _refuse(where, f"키가 다릅니다 (빠짐 {missing}, 남음 {extra})")
    return mapping


def _parse_mask(spec: Any, where: str, depth: int = 0) -> dict:
    if depth > _MAX_MASK_DEPTH:
        _refuse(where, "마스크가 너무 깊게 겹쳐 있습니다")
    if not isinstance(spec, dict):
        _refuse(where, "마스크가 객체가 아닙니다")
    if "operation" in spec:
        _exact_keys(spec, COMBINE_KEYS, where)
        return {"operation": _choice(spec["operation"], COMBINE_OPERATIONS, f"{where}.operation"),
                "a": _parse_mask(spec["a"], f"{where}.a", depth + 1),
                "b": _parse_mask(spec["b"], f"{where}.b", depth + 1)}
    if "blur" in spec:
        _exact_keys(spec, SHAPING_KEYS, where)
        return {"blur": _number(spec["blur"], f"{where}.blur"),
                "spread": _number(spec["spread"], f"{where}.spread"),
                "contrast": _number(spec["contrast"], f"{where}.contrast"),
                "normalize": _flag(spec["normalize"], f"{where}.normalize"),
                "a": _parse_mask(spec["a"], f"{where}.a", depth + 1)}
    _exact_keys(spec, LEAF_KEYS, where)
    leaf = {"mask_axis": _choice(spec["mask_axis"], MASK_AXES, f"{where}.mask_axis"),
            "mask_mode": _choice(spec["mask_mode"], MASK_MODES, f"{where}.mask_mode")}
    for key in LEAF_KEYS[2:]:
        leaf[key] = _number(spec[key], f"{where}.{key}")
    return {key: leaf[key] for key in LEAF_KEYS}


def _parse_entry(entry: Any, where: str) -> dict:
    _exact_keys(entry, ("kind", "params", "schedule", "mask"), where)
    kind = _choice(entry["kind"], KINDS, f"{where}.kind")
    raw = _exact_keys(entry["params"], KIND_PARAMS[kind], f"{where}.params")
    params = {}
    for name in KIND_PARAMS[kind]:
        here = f"{where}.params.{name}"
        if name in BOOL_PARAMS:
            params[name] = _flag(raw[name], here)
        elif name in CHOICE_PARAMS:
            params[name] = _choice(raw[name], CHOICE_PARAMS[name], here)
        else:
            params[name] = _number(raw[name], here)
    raw_schedule = _exact_keys(entry["schedule"], SCHEDULE_KEYS, f"{where}.schedule")
    schedule = {name: (_flag(raw_schedule[name], f"{where}.schedule.{name}") if name == "smooth"
                       else _number(raw_schedule[name], f"{where}.schedule.{name}")) for name in SCHEDULE_KEYS}
    mask = entry["mask"]
    if mask is not None:
        if not TAKES_MASK[kind]:
            _refuse(f"{where}.mask", f"{kind} 는 마스크를 받지 않습니다")
        mask = _parse_mask(mask, f"{where}.mask")
    return {"kind": kind, "params": params, "schedule": schedule, "mask": mask}


def parse_settings(text: Any) -> dict:
    """The node's ``settings_json`` as ``{"v", "pass", "entries", "infotext", "tags"}``, every value checked.

    Numbers become floats (as the extension's coerced panel values are); out-of-range values are kept (the
    extension does not clamp typed values either), non-finite ones refused."""
    if isinstance(text, dict):
        data = text
    else:
        try:
            data = json.loads(text if isinstance(text, str) else "")
        except (TypeError, ValueError) as exc:
            _refuse("settings_json", f"JSON 이 아닙니다 ({exc})")
    if not isinstance(data, dict):
        _refuse("settings_json", "객체가 아닙니다")
    allowed = {"v", "pass", "entries", "infotext", "tags"}
    extra = sorted(str(key) for key in data if key not in allowed)
    if extra:
        _refuse("settings_json", f"모르는 키 {extra}")
    if isinstance(data.get("v"), bool) or data.get("v") != SETTINGS_VERSION:
        _refuse("v", f"지원하지 않는 형식 {data.get('v')!r} (이 팩은 {SETTINGS_VERSION})")
    pass_name = _choice(data.get("pass"), PASSES, "pass")
    raw_entries = data.get("entries")
    if not isinstance(raw_entries, list):
        _refuse("entries", "목록이 아닙니다")
    entries = [_parse_entry(entry, f"entries[{index}]") for index, entry in enumerate(raw_entries)]
    infotext = data.get("infotext", "")
    if not isinstance(infotext, str):
        _refuse("infotext", "문자열이 아닙니다")
    tags = data.get("tags")
    if tags is None:
        tags = [f"#{index + 1}" for index in range(len(entries))]
    elif (not isinstance(tags, list) or len(tags) != len(entries)
          or not all(isinstance(tag, str) and tag for tag in tags)):
        _refuse("tags", "항목 수만큼의 이름 목록이 아닙니다")
    return {"v": SETTINGS_VERSION, "pass": pass_name, "entries": entries, "infotext": infotext,
            "tags": list(tags)}


# ---------------------------------------------------------------------------------------------------------
# Which entries can run on this model
# ---------------------------------------------------------------------------------------------------------

def params_are_noop(kind: str, params: dict) -> bool:
    """True when every edit of the entry is 0 — it would hand its input back unchanged."""
    amounts = list(_EDIT_AMOUNTS[kind])
    if kind == "advanced":
        if params["more_colors"]:
            amounts += _CHROMA_PLUS
        if params["color_shift"]:
            amounts.append("color_shift_amount")
    return all(params[name] == 0 for name in amounts)


def vector_free(entry: dict) -> bool:
    """Does the entry edit anything without basis vectors? Exactly the edits that use a colour anchor
    (Contrast, Color Shift), so: does it need an anchor."""
    return bool(needed_colors([entry]))


def without_anchor_edits(pairs: list) -> list:
    """``(tag, entry)`` pairs with Contrast and Color Shift set to 0 (no anchors); entries left with
    nothing to do are dropped."""
    kept = []
    for tag, entry in pairs:
        params = dict(entry["params"])
        for name in ("contrast", "color_shift_amount"):
            if name in params:
                params[name] = 0.0
        if not params_are_noop(entry["kind"], params):
            kept.append((tag, dict(entry, params=params)))
    return kept


# ---------------------------------------------------------------------------------------------------------
# Colour anchors
# ---------------------------------------------------------------------------------------------------------

ANCHOR_CACHE_LIMIT = 32


def color_key(color) -> tuple:
    """Upstream's anchor key (``get_color_latent``): the colour rounded to 4 decimals."""
    return tuple(round(float(c), 4) for c in color)


class _RankRecorder:
    """Hands the VAE's ``encode`` through and keeps the latent's spatial rank (2, or 3 for Wan's 5-D)."""

    def __init__(self, vae):
        self._vae = vae
        self.rank = None

    def encode(self, image):
        latent = self._vae.encode(image)
        self.rank = latent.dim() - 2
        return latent


class AnchorCache:
    """Encoded anchors kept across prompts, keyed by what the encode depends on.

    An entry holds a weak reference to the encoder module: once ComfyUI drops that VAE the entry is gone, and
    it never answers for a new module that reuses the old one's ``id``. An encoder that cannot be weakly
    referenced is not cached. Least recently used entries leave first beyond ``limit``."""

    def __init__(self, limit: int = ANCHOR_CACHE_LIMIT):
        self.limit = limit
        self._entries: "OrderedDict[tuple, tuple[weakref.ref, Any]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def get(self, module, key):
        found = self._entries.get(key)
        if found is None:
            return None
        if found[0]() is not module:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return found[1]

    def put(self, module, key, anchor) -> None:
        try:
            ref = weakref.ref(module)
        except TypeError:
            return
        for dead in [k for k, (r, _a) in self._entries.items() if r() is None]:
            del self._entries[dead]
        self._entries[key] = (ref, anchor)
        self._entries.move_to_end(key)
        while len(self._entries) > self.limit:
            self._entries.popitem(last=False)


ANCHORS = AnchorCache()


def _encoder_module(vae):
    """What the anchors are encoded with: ComfyUI's ``VAE.first_stage_model`` (else the VAE object)."""
    module = getattr(vae, "first_stage_model", None)
    return vae if module is None else module


def _anchor_cache_key(vae, module, latent_format, device, key) -> tuple:
    patcher = getattr(vae, "patcher", None)
    return (id(module), str(getattr(vae, "vae_dtype", None)), str(getattr(vae, "device", None)),
            str(getattr(patcher, "patches_uuid", None)),
            "none" if latent_format is None else type(latent_format).__name__, str(device), key)


def anchor_device():
    """The device the flat colour image is made on: ComfyUI's torch device (upstream's choice), else CPU."""
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        return require_torch().device("cpu")


def encode_anchors(vae, latent_format, colors, cache: dict | None = None, device=None) -> dict:
    """``{color_key(colour): float32 model-space anchor}`` for ``colors``.

    ``cache`` (optional, one prompt's) is looked in first, then ``ANCHORS``; only a colour neither holds is
    encoded: upstream's ``build_color_latent`` through ``_RankRecorder``, ``to_model_space`` with the rank the
    encode returned (``vae.latent_dim`` if it could not tell), ``.float()``. Under inference mode."""
    from .vendor.comfyui_colorcraft.lib_colorcraft.color import build_color_latent

    torch = require_torch()
    device = anchor_device() if device is None else device
    cache = {} if cache is None else cache
    module = _encoder_module(vae)
    anchors: dict = {}
    with torch.inference_mode():
        for color in colors:
            key = color_key(color)
            if key in anchors:
                continue
            cache_key = _anchor_cache_key(vae, module, latent_format, device, key)
            anchor = cache.get(cache_key)
            if anchor is None:
                anchor = ANCHORS.get(module, cache_key)
            if anchor is None:
                recorder = _RankRecorder(vae)
                raw = build_color_latent(recorder, device, *color)
                rank = recorder.rank if recorder.rank is not None else getattr(vae, "latent_dim", 2)
                anchor = colorcraft_basis.to_model_space(latent_format, raw, rank).detach().float()
                ANCHORS.put(module, cache_key, anchor)
            cache[cache_key] = anchor
            anchors[key] = anchor
    return anchors


# ---------------------------------------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------------------------------------

@dataclass
class Prepared:
    """What one pass grades, decided before sampling."""

    pass_name: str
    entries: list
    tags: list
    family: str | None
    family_label: str
    downscale: int
    anchors: dict
    pre_dd: bool
    infotext: str = ""
    notes: list = field(default_factory=list)
    chain_tags: list = field(default_factory=list)   # every tab of the pass (the summary's "mods"), before filtering

    @property
    def active(self) -> bool:
        return bool(self.entries)


def model_latent_format(model):
    """The MODEL's latent format object (``get_model_object``, else ``model.model.latent_format``)."""
    getter = getattr(model, "get_model_object", None)
    if callable(getter):
        try:
            found = getter("latent_format")
        except Exception:
            found = None
        if found is not None:
            return found
    return getattr(getattr(model, "model", None), "latent_format", None)


def _family_has_basis(family: str) -> bool:
    return os.path.isfile(os.path.join(colorcraft_basis.vectors_dir(family), f"colorcraft-{family}.safetensors"))


def prepare(model, vae, settings: dict, pre_dd: bool = True, *, device=None,
            encode: Callable[..., dict] | None = None) -> Prepared:
    """Decide the pass from parsed ``settings`` (``parse_settings``), the MODEL and the pass's VAE.

    sam-extra's rules: family from the latent format (a missing vector file switches the vector controls off);
    without a family only the entries with Contrast/Color Shift run; their anchors are encoded now; a failed
    encode drops Contrast/Color Shift. ``Prepared.entries`` empty = nothing to do (the node leaves the MODEL
    alone)."""
    encode = encode_anchors if encode is None else encode
    pairs = list(zip(settings["tags"], settings["entries"]))
    latent_format = model_latent_format(model)
    fmt = "none" if latent_format is None else type(latent_format).__name__
    family = colorcraft_basis.family_for_latent_format(latent_format)
    notes: list[str] = []
    if family is not None and not _family_has_basis(family):
        notes.append(f"colorcraft-{family}.safetensors missing: vector controls off")
        _log_once(f"missing:{family}", f"vectors/colorcraft-{family}.safetensors is missing; vector controls "
                                       "are off.")
        family = None
    family_label = f"{family} ({fmt})" if family else f"no basis ({fmt})"
    if family is None and needs_basis([entry for _tag, entry in pairs]):
        notes.append("vector controls off on this model: only Contrast and Color Shift apply")
        _log_once(f"nobasis:{fmt}", f"no Colorcraft basis for latent format {fmt} (SD 1.5/SDXL and others): "
                                    "exposure, chroma, detail, Chroma Plus and masking are off; Contrast and "
                                    "Color Shift still work.")
    if family is None:
        pairs = [(tag, entry) for tag, entry in pairs if vector_free(entry)]
    anchors: dict = {}
    colors = needed_colors([entry for _tag, entry in pairs])
    if pairs and colors:
        try:
            anchors = encode(vae, latent_format, colors, device=device)
        except Exception as exc:  # the pass still runs its vector edits
            notes.append(f"anchors failed ({type(exc).__name__}): Contrast/Color Shift off")
            log_line(f"colour anchors could not be encoded ({type(exc).__name__}: {exc}); Contrast and Color "
                     "Shift are off for this pass.")
            pairs = without_anchor_edits(pairs) if family is not None else []
    if not pairs:
        notes.append("nothing to apply")
    return Prepared(
        pass_name=settings["pass"], entries=[entry for _tag, entry in pairs], tags=[tag for tag, _e in pairs],
        family=family, family_label=family_label, downscale=colorcraft_basis.vae_downscale_factor(family),
        anchors=anchors, pre_dd=bool(pre_dd), infotext=settings.get("infotext", ""), notes=notes,
        chain_tags=list(settings["tags"]),
    )


# ---------------------------------------------------------------------------------------------------------
# Part 2 — during sampling
# ---------------------------------------------------------------------------------------------------------
#
# ``apply_colorcraft`` puts two things on a MODEL clone: a ``WrappersMP.SAMPLER_SAMPLE`` wrapper that opens a
# ``RunState`` for every sampling run (the run's own sigma list on the CPU — Forge's walked list; ComfyUI hands the
# denoise tail already — its latent grid, and the schedules built on that list) under a private
# ``transformer_options`` key of the run, and a post-CFG function appended **last** to
# ``sampler_post_cfg_function`` (after Optimal Scale, the guidance suite and Skimmed CFG — Forge's order), which
# grades every model evaluation of that run:
#
# * Euler (SMEA) Dy CFG++'s extra sub-step evaluations (``SUBSTEP_MARKER``) pass through: one grade per step;
# * the schedule value is looked up by the evaluation's sigma on the run's list — upstream's ``sigma_to_value``;
#   with Detail Daemon on, the sampler's own sigma it noted (``PRE_DD_SIGMAS_KEY``) when ``pre_dd`` (default on);
# * Anima SPEED's re-published copy of the run's list (``REPUBLISHED_FROM``, same length) is followed; any other
#   list is a different run's and passes through;
# * every value 0 -> the incoming object itself; float16/bfloat16 graded in float32 and cast back; a result that
#   is the input, or not finite, passes the input through; an exception passes it through with one console line;
# * on SPEED's coarse grid the mask blur keeps its image-pixel size (``grid_downscale``).
#
# One console summary per run: ``[AI Studio Colorcraft] base: krea2 (Wan21); mods I; 30 evals, 28 edited``.

RUN_KEY = "ai_studio_colorcraft_runs"            # transformer_options: {patch token: RunState} for this run
PATCH_KEY = "forge_neo_colorcraft"               # model_options: the ColorcraftPatch (Hires carry reads it)
WRAPPER_KEY = "ai_studio_forge_parity.colorcraft"
OWNER = "ai_studio_forge_parity/colorcraft"
OWNER_ATTR = "_ai_studio_colorcraft_owner"
_FOREIGN = object()


@dataclass
class RunReport:
    """What one sampling run did (the console summary; the extension's status line)."""

    name: str
    family: str = ""
    tags: list = field(default_factory=list)
    evals: int = 0
    edited: int = 0
    idle: int = 0
    foreign: int = 0
    substeps: int = 0
    errors: int = 0
    last_error: str = ""
    nonfinite: int = 0
    pre_dd: int = 0
    notes: list = field(default_factory=list)

    def render(self) -> str:
        parts = [self.family] if self.family else []
        if self.tags:
            parts.append("mods " + "+".join(self.tags))
        parts.append(f"{self.evals} evals, {self.edited} edited" if self.evals else "pending model evaluation")
        if self.pre_dd:
            parts.append(f"pre-DD sigma x{self.pre_dd}")
        if self.substeps:
            parts.append(f"Dy/SMEA sub-steps passed through x{self.substeps}")
        if self.foreign:
            parts.append(f"other sampling run passed through x{self.foreign}")
        if self.nonfinite:
            parts.append(f"non-finite result passed through x{self.nonfinite}")
        if self.errors:
            parts.append(f"error {self.last_error} x{self.errors} (input passed through)")
        parts += self.notes
        return f"{self.name}: " + "; ".join(parts)


@dataclass
class RunState:
    """One sampling run of a patched MODEL."""

    prepared: Prepared
    report: RunReport
    run_src: Any = None          # the run's sample_sigmas object
    walked_src: Any = None       # the list ``walked`` was read from (run_src, or SPEED's re-published copy)
    walked: Any = None           # its walked part, on the CPU
    walk_offset: int = 0
    built: list = None
    full_hw: tuple | None = None
    device_anchors: dict = field(default_factory=dict)
    setup_error: Exception | None = None   # the schedules could not be built (e.g. an out-of-range typed Start)


def _transformer_options(args) -> dict:
    options = args.get("model_options") if isinstance(args, dict) else None
    transformer = options.get("transformer_options") if isinstance(options, dict) else None
    return transformer if isinstance(transformer, dict) else {}


def _tail_offset(torch, whole, sigmas) -> int:
    """Where ``sigmas`` (the sampler's list) starts in ``whole`` (the published list), 0 if it does not end it."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or len(whole) <= len(sigmas):
        return 0
    offset = len(whole) - len(sigmas)
    try:
        same = torch.equal(whole[offset:].detach().float().cpu(), sigmas.detach().float().cpu())
    except Exception:
        same = False
    return offset if same else 0


def begin_run(patch: "ColorcraftPatch", sigmas, extra_args, noise) -> RunState:
    """Open the run's state (SAMPLER_SAMPLE time) and register it under the run's ``transformer_options``."""
    from .colorcraft_engine import build_entries

    torch = require_torch()
    prepared = patch.prepared
    report = RunReport(prepared.pass_name, prepared.family_label, list(prepared.chain_tags),
                       notes=list(prepared.notes))
    state = RunState(prepared, report)
    if torch.is_tensor(noise) and noise.dim() >= 4:
        state.full_hw = (int(noise.shape[-2]), int(noise.shape[-1]))
    options = extra_args.get("model_options") if isinstance(extra_args, dict) else None
    if not isinstance(options, dict):
        return state
    transformer = options.get("transformer_options")
    if not isinstance(transformer, dict):
        transformer = options["transformer_options"] = {}
    whole = transformer.get("sample_sigmas")
    if torch.is_tensor(whole) and whole.ndim == 1 and whole.numel() >= 2:
        full = whole.detach().to("cpu", copy=True).flatten()
        offset = _tail_offset(torch, whole, sigmas)
        state.walk_offset = offset if 0 < offset < len(full) - 1 else 0
        state.run_src = state.walked_src = whole
        state.walked = full[state.walk_offset:]
        try:
            state.built = build_entries(prepared.entries, len(state.walked) - 1)
        except Exception as exc:  # every evaluation of the run passes through, counted as an error
            state.setup_error = exc
    runs = transformer.get(RUN_KEY)
    runs = dict(runs) if isinstance(runs, dict) else {}
    runs[patch.token] = state
    transformer[RUN_KEY] = runs
    return state


def lookup_sigma(args, transformer: dict, use_pre_dd: bool):
    """``(sigma, noted)``: the sigma to look the schedule up by — upstream's ``sigma.max().item()`` — and whether
    Detail Daemon's note of the sampler's own sigma replaced a different (scaled) one the model got."""
    model_sigma = args["sigma"]
    noted = transformer.get(PRE_DD_SIGMAS_KEY) if use_pre_dd else None
    sigma = model_sigma if noted is None else noted
    value = float(sigma.max().item()) if hasattr(sigma, "max") else float(sigma)
    if noted is None:
        return value, False
    model_value = float(model_sigma.max().item()) if hasattr(model_sigma, "max") else float(model_sigma)
    return value, model_value != value


def run_values(state: RunState, args, transformer: dict):
    """The schedule values of this evaluation, ``_FOREIGN`` for another run's list, None without a list."""
    from .colorcraft_engine import sigma_to_value

    torch = require_torch()
    source = transformer.get("sample_sigmas")
    if state.run_src is None or not (torch.is_tensor(source) and source.numel() >= 2):
        return None
    if source is not state.walked_src:
        if source is not state.run_src and republished_from(source) is not state.run_src:
            return _FOREIGN
        # SPEED re-published the run's list (the aligned sigma / re-spaced tail) and puts the original back at
        # the end: same step count, looked up on the sigmas the sampler walks now.
        full = source.detach().to("cpu", copy=True).flatten()[state.walk_offset:]
        if len(full) != len(state.walked):
            return _FOREIGN
        state.walked_src = source
        state.walked = full
    if state.setup_error is not None:
        raise state.setup_error.with_traceback(None)
    cur_sigma, noted = lookup_sigma(args, transformer, state.prepared.pre_dd)
    if noted:
        state.report.pre_dd += 1
    return [sigma_to_value(cur_sigma, state.walked, schedule) for schedule, _kind, _p, _mask in state.built]


def grid_downscale(state: RunState, x0) -> float:
    """Image pixels per latent pixel of the grid ``x0`` is on (the unit of the mask blur): the family's factor on
    the run's own grid, scaled by ``full / coarse`` on a smaller one (SPEED's first stages)."""
    full = state.full_hw
    h, w = int(x0.shape[-2]), int(x0.shape[-1])
    if full is None or (h, w) == tuple(full) or h <= 0 or w <= 0:
        return state.prepared.downscale
    return state.prepared.downscale * 0.5 * (full[0] / h + full[1] / w)


_BASIS: dict = {}
_BASIS_ON: dict = {}


def basis_on(family, device, dtype) -> dict:
    """The family's basis vectors on ``device`` in ``dtype`` (cached; loaded once from the vendored file)."""
    key = (family, str(device), dtype)
    found = _BASIS_ON.get(key)
    if found is None:
        source = _BASIS.get(family)
        if source is None:
            loaded = colorcraft_basis.basis_for(family)
            if loaded is None:
                raise FileNotFoundError(f"colorcraft-{family}.safetensors")
            source = _BASIS[family] = {k: v.detach().clone() for k, v in loaded.items()}
        found = _BASIS_ON[key] = {k: v.to(device=device, dtype=dtype) for k, v in source.items()}
    return found


def _same_tensor(a, b) -> bool:
    return (a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()
            and a.dtype == b.dtype)


def evaluate(state: RunState | None, args, orphan: RunReport):
    """One post-CFG call: the graded ``denoised``, or the incoming object itself when nothing applies."""
    from .colorcraft_engine import apply_chain

    x0 = args["denoised"]
    report = state.report if state is not None else orphan
    if is_extra_sampler_substep(args):
        report.substeps += 1
        return x0
    report.evals += 1
    if state is None:
        if "no sampling run: passed through" not in report.notes:
            report.notes.append("no sampling run: passed through")
        return x0
    values = run_values(state, args, _transformer_options(args))
    if values is _FOREIGN:
        report.foreign += 1
        return x0
    if values is None:
        if "no sigma list: passed through" not in report.notes:
            report.notes.append("no sigma list: passed through")
        return x0
    if all(v == 0 for v in values):
        report.idle += 1
        return x0

    torch = require_torch()
    work = x0.float() if x0.dtype in (torch.float16, torch.bfloat16) else x0
    prepared = state.prepared
    cur_basis = basis_on(prepared.family, work.device, work.dtype) if prepared.family else None

    def get_color_latent(color, device, dtype):
        key = (color_key(color), str(device), dtype)
        anchor = state.device_anchors.get(key)
        if anchor is None:
            source = prepared.anchors.get(color_key(color))
            if source is None:
                raise KeyError(f"no anchor encoded for colour {color}")
            anchor = state.device_anchors[key] = source.to(device=device, dtype=dtype)
        return anchor

    out = apply_chain(state.built, work, values, prepared.family, cur_basis, get_color_latent,
                      grid_downscale(state, work))
    if _same_tensor(out, work):
        report.idle += 1
        return x0
    if not bool(torch.isfinite(out).all()):
        report.nonfinite += 1
        return x0
    if out.dtype != x0.dtype:
        out = out.to(x0.dtype)
    report.edited += 1
    return out


class ColorcraftPatch:
    """One pass's Colorcraft on a MODEL: the prepared pass, its SAMPLER_SAMPLE wrapper and post-CFG function."""

    def __init__(self, prepared: Prepared, settings: dict | None = None):
        self.prepared = prepared
        self.settings = settings          # parsed settings_json (the Hires carry prepares again from them)
        self.token = id(self)
        self.orphan = RunReport(prepared.pass_name, prepared.family_label, list(prepared.chain_tags))
        self.last_report: RunReport | None = None      # the latest finished run (console line, tests)
        self.last_state: RunState | None = None
        self.wrapper = self._make_wrapper()
        self.post_cfg = self._make_post_cfg()

    def run_state(self, args) -> RunState | None:
        runs = _transformer_options(args).get(RUN_KEY)
        return runs.get(self.token) if isinstance(runs, dict) else None

    def _make_wrapper(self):
        patch = self

        def colorcraft_sampler_sample(executor, *args, **kwargs):
            sigmas = args[1] if len(args) > 1 else kwargs.get("sigmas")
            extra_args = args[2] if len(args) > 2 else kwargs.get("extra_args")
            noise = args[4] if len(args) > 4 else kwargs.get("noise")
            state = None
            try:
                state = begin_run(patch, sigmas, extra_args, noise)
            except Exception as exc:  # never break sampling: this run is passed through
                log_line(f"{patch.prepared.pass_name} pass: not set up ({type(exc).__name__}: {exc})")
            try:
                return executor(*args, **kwargs)
            finally:
                if state is not None:
                    patch.last_report, patch.last_state = state.report, state
                    log_line(state.report.render())

        setattr(colorcraft_sampler_sample, OWNER_ATTR, OWNER)
        return colorcraft_sampler_sample

    def _make_post_cfg(self):
        patch = self

        def colorcraft_post_cfg(args):
            incoming = args["denoised"]
            state = patch.run_state(args)
            try:
                return evaluate(state, args, patch.orphan)
            except Exception as exc:  # never break a generation: pass the prediction through
                report = state.report if state is not None else patch.orphan
                report.errors += 1
                report.last_error = type(exc).__name__
                if report.errors == 1:
                    log_line(f"{report.name} pass: {type(exc).__name__}: {exc} - input passed through")
                return incoming

        setattr(colorcraft_post_cfg, OWNER_ATTR, OWNER)
        return colorcraft_post_cfg


def is_owned(function) -> bool:
    return getattr(function, OWNER_ATTR, None) == OWNER


def apply_colorcraft(model, patch: ColorcraftPatch):
    """A clone of ``model`` graded by ``patch``: its post-CFG function appended last (an earlier Colorcraft one of
    this pack is replaced, every other kept in order) and its SAMPLER_SAMPLE wrapper registered."""
    patched = clone_model(model, "Colorcraft")
    options = dict(getattr(patched, "model_options", None) or {})
    previous = options.get("sampler_post_cfg_function", [])
    if not isinstance(previous, (list, tuple)):
        raise TypeError("sampler_post_cfg_function is not a callback list")
    options["sampler_post_cfg_function"] = [fn for fn in previous if not is_owned(fn)] + [patch.post_cfg]
    options[PATCH_KEY] = patch
    patched.model_options = options
    add = getattr(patched, "add_wrapper_with_key", None)
    if not callable(add):
        raise RuntimeError("Colorcraft needs a ComfyUI MODEL that supports sampler wrappers; update ComfyUI.")
    wrapper_type = sampler_sample_wrapper_type()
    remove = getattr(patched, "remove_wrappers_with_key", None)
    if callable(remove):
        remove(wrapper_type, WRAPPER_KEY)
    add(wrapper_type, WRAPPER_KEY, patch.wrapper)
    return patched


def without_colorcraft(model):
    """``model`` without this pack's Colorcraft: a clone with its post-CFG function, wrapper and patch entry removed
    (every other post-CFG function and wrapper kept in order); ``model`` itself when it has none."""
    options = getattr(model, "model_options", None) or {}
    functions = options.get("sampler_post_cfg_function", [])
    owned = isinstance(functions, (list, tuple)) and any(is_owned(fn) for fn in functions)
    wrappers = getattr(model, "wrappers", None)
    registered = wrappers.get(sampler_sample_wrapper_type()) if isinstance(wrappers, dict) else None
    wrapped = isinstance(registered, dict) and bool(registered.get(WRAPPER_KEY))
    if PATCH_KEY not in options and not owned and not wrapped:
        return model
    stripped = clone_model(model, "Colorcraft")
    options = dict(getattr(stripped, "model_options", None) or {})
    options.pop(PATCH_KEY, None)
    if owned:
        options["sampler_post_cfg_function"] = [fn for fn in functions if not is_owned(fn)]
    stripped.model_options = options
    remove = getattr(stripped, "remove_wrappers_with_key", None)
    if wrapped:
        if not callable(remove):
            raise RuntimeError("Colorcraft needs a ComfyUI MODEL that supports sampler wrappers; update ComfyUI.")
        remove(sampler_sample_wrapper_type(), WRAPPER_KEY)
    return stripped


def model_colorcraft_patch(model) -> ColorcraftPatch | None:
    """The ``ColorcraftPatch`` ``apply_colorcraft`` put on ``model`` (else None)."""
    found = (getattr(model, "model_options", None) or {}).get(PATCH_KEY)
    return found if isinstance(found, ColorcraftPatch) else None
