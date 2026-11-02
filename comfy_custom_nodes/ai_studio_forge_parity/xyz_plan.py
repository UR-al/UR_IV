"""X/Y/Z planner for the Anima studio: one full studio state per grid cell, the generation order, labels and seeds.

Pure Python (no ComfyUI / torch).  Follows the behaviour of Forge Neo's X/Y/Z plot as described in the recon memo;
the pack carries no Forge code.

- Every cell starts from the normalized state with its prompt boxes and list LoRAs already rendered into the prompt,
  the LoRAs as ``<lora:name:strength>`` tags (``effective_prompt``; the cell's boxes and ``loras`` list are empty),
  so Prompt S/R can rewrite them.  Axes apply X, then Y, then Z.
- Prompt S/R: the first value is the search text and every value (the first included) is one column; plain,
  case-sensitive replace-all in the prompt and the negative prompt.  A search text found in neither aborts the plan.
- Prompt order: the given tokens are cut out of the positive prompt and put back into the same slots in each
  permutation's order.
- LoRA: the axis has a block of settings instead of a value box; ``xyz_lora`` turns it into columns.
- Extension axes also turn on what their setting needs (``AxisSpec.switches``), before any axis writes its value, so a
  switch axis keeps its columns; a family that cannot run the extension aborts the plan.
- Wildcards: each cell's prompts are drawn from the cell's seed, after the axes, so Prompt S/R sees what was written and
  cells of one seed compare the same picks.
- Seeds: unless ``keep_minus1``, a state seed of -1 becomes one random seed shared by every cell, and each -1 of a
  Seed axis becomes its own random seed (listed in ``fixed`` and ``Fixed X Values``).  ``vary_x/y/z`` add the cell
  position on top (``ix``; ``iy * nx``; ``iz * nx * ny``, a factor only for the varied axes).
- Order: the dearest axis (strictly) loops outermost so model reloads happen least often; ties keep Z, Y, X.
- Cells are listed with X fastest (index ``ix + iy*nx + iz*nx*ny``); ``order`` lists those indices in generation
  order.  Each cell state is normalized again, so axis values are clamped like the control panel's.

Validation problems (unparsable values, unknown checkpoint or DoRA mode, 3.8B mixed with 2.9B/base, a grid of
100 MP or more, Prompt S/R or Prompt order text not found, and - when ``loras`` is given - a ``<lora:...>`` tag of
any cell that matches no LoRA file) come back in ``errors`` with no cells; ``plan_xyz`` never raises for them.
The LoRA check exists because the LoRA node only logs a missing file and renders the cell without it: in an
epoch x strength grid a mistyped epoch would read as a bad epoch (Forge tells the user instead).
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field

from .dora_infer_mode import MODE_LABELS, mode_key
from .studio_lora_tags import parse_lora_tags, resolve_lora_name
from .studio_choices import unet_family
from .studio_state import ATTENTION_MODES, EXT_SCHEMA, PROMPT_BOXES, effective_prompt, generation_size, normalize_state
from .studio_wildcards import expand_state, uses_wildcards
from . import xyz_lora
from .xyz_axes import AxisSpec, axis_by_id, ext_key
from .xyz_values import parse_axis_values

LETTERS = ("x", "y", "z")
SEED_RANGE = 4294967294  # random seeds are drawn from [0, SEED_RANGE)
MAX_GRID_MP = 100
MIXED_FAMILIES = "모델 계열이 다르면 한 격자에 섞을 수 없음 (2.9B·base / 3.8B / MOSS / Krea2)"
SCRIPT_NAME = "X/Y/Z plot"


@dataclass
class Cell:
    ix: int
    iy: int
    iz: int
    state: dict


@dataclass
class XyzPlan:
    cells: list = field(default_factory=list)
    order: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    infotext_extra: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    lora_preview: dict = field(default_factory=dict)  # letter -> the LoRA axis' columns, for the panel's list
    wildcards_missing: list = field(default_factory=list)  # wildcard names no file has: they stay as written


class _PlanError(Exception):
    pass


@dataclass
class _Axis:
    spec: AxisSpec
    text: str
    values: list
    block: dict = field(default_factory=dict)  # the LoRA axis' settings


# ---------------------------------------------------------------------------------------------- reading the axes

def _read_axes(axes) -> list[_Axis]:
    entries = list(axes or [])[:3]
    entries += [{}] * (3 - len(entries))
    out, errors = [], []
    for entry in entries:
        entry = entry if isinstance(entry, dict) else {}
        axis_id = entry.get("axis") or "nothing"
        text = entry.get("text") if isinstance(entry.get("text"), str) else ""
        try:
            spec = axis_by_id(axis_id)
        except (KeyError, TypeError):
            errors.append(f"Unknown axis: {axis_id}")
            continue
        try:
            values = parse_axis_values(spec.kind, text)
        except ValueError as exc:
            errors.append(f"{spec.label}: {exc}")
            continue
        if not values:
            errors.append(f"{spec.label}: no values")
            continue
        out.append(_Axis(spec, text, values, entry.get("lora") if isinstance(entry.get("lora"), dict) else {}))
    if errors:
        raise _PlanError(*errors)
    return out


def _lora_file_list(loras):
    def read() -> list:
        if loras is None:
            raise xyz_lora.LoraAxisError("LoRA 축: LoRA 목록을 읽지 못했습니다")
        try:
            return list(loras() if callable(loras) else loras)
        except Exception as exc:  # the LoRA folder could not be listed
            raise xyz_lora.LoraAxisError(f"LoRA 축: LoRA 목록을 읽지 못했습니다: {exc}") from exc

    return read


def _expand_lora_axes(axes: list[_Axis], prompt: str, negative: str, loras, lora_epoch) -> None:
    """A LoRA axis' columns need the prompt and the file list, so they come after the value boxes."""
    lora_axes = [axis for axis in axes if axis.spec.id == "lora"]
    for axis in lora_axes:
        axis.values = xyz_lora.lora_columns(axis.block, prompt, negative, _lora_file_list(loras), lora_epoch)
        axis.text = ", ".join(column.label for column in axis.values)
    names = [xyz_lora.base_lora(axis.block).casefold() for axis in lora_axes]
    if len(set(names)) < len(names):
        raise xyz_lora.LoraAxisError("LoRA 축: 같은 LoRA 를 두 축에서 찾을 수 없습니다")


def _match_checkpoint(text: str, names) -> str:
    """Exact file name, else the shortest name containing the text (Forge's closest-checkpoint rule)."""
    if text in names:
        return text
    hits = sorted((name for name in names if text in name), key=lambda name: (len(name), name))
    if not hits:
        raise _PlanError(f"Unknown Checkpoint: {text}")
    return hits[0]


def _dora_mode(text: str) -> str:
    """A ``[DoRA] Inference mode`` value (sam-extra label, node label or key) as the node's COMBO label."""
    try:
        return MODE_LABELS[mode_key(text)]
    except ValueError:
        pass
    wanted = text.strip().casefold()
    for label in MODE_LABELS.values():
        plain = label.replace("/Comfy", "").casefold()  # sam-extra says "Forge" where the node says "Forge/Comfy"
        if wanted and (plain == wanted or plain.startswith(wanted + " (")):
            return label
    raise _PlanError(f"Unknown DoRA mode: {text}")


def _attention(text: str) -> str:
    wanted = text.strip().casefold()
    if wanted in ATTENTION_MODES:
        return wanted
    raise _PlanError(f"Unknown attention: {text}")


def _model_family(family: str) -> str:
    return family if family in ("moss", "krea2") else "anima"


def _check_ext_family(base: dict, axes: list[_Axis], families: dict) -> None:
    """An extension axis on a family that cannot run it would draw the same picture in every column."""
    unet = base["unet"]
    family = _model_family(families.get(unet) or unet_family(unet)) if unet else "anima"
    runs_on = {item["key"]: item["families"] for item in EXT_SCHEMA["extensions"]}
    for axis in axes:
        key = ext_key(axis.spec)
        if key and family not in runs_on[key]:
            raise _PlanError(f"{axis.spec.label}: 이 모델 계열에서는 못 씀")


def _check_panel_family(base: dict, axes: list[_Axis], resolved: list[list], families: dict) -> None:
    """A checkpoint axis stays in the panel's family: a cell takes only its UNET from the axis, the TE, VAE, sampler and
    extensions stay the panel's."""
    unet = base["unet"]
    if not unet:
        return
    panel = _model_family(families.get(unet) or unet_family(unet))
    for axis, values in zip(axes, resolved):
        if axis.spec.id == "checkpoint" and any(_model_family(families[value]) != panel for value in values):
            raise _PlanError(MIXED_FAMILIES)


def _resolve(axes: list[_Axis], families: dict) -> list[list]:
    """Per axis the values a cell writes (checkpoint names matched, DoRA labels mapped); labels keep the text."""
    resolved, errors, seen = [], [], set()
    for axis in axes:
        values = axis.values
        try:
            if axis.spec.id == "checkpoint":
                values = [_match_checkpoint(v, families) for v in values]
                seen.update(families[v] for v in values)
            elif axis.spec.id == "dora_mode":
                values = [_dora_mode(v) for v in values]
            elif axis.spec.id == "attention":
                values = [_attention(v) for v in values]
            elif axis.spec.values:
                unknown = [v for v in values if v not in axis.spec.values]
                if unknown:
                    raise _PlanError(f"{axis.spec.label}: {', '.join(unknown)} 없음")
        except _PlanError as exc:
            errors.extend(exc.args)
        resolved.append(values)
    if errors:
        raise _PlanError(*errors)
    if len({"anima" if family in ("2.9b", "base") else family for family in seen}) > 1:
        raise _PlanError(MIXED_FAMILIES)
    return resolved


# ---------------------------------------------------------------------------------------------- seeds and order

def _fix_seed_axes(axes: list[_Axis], resolved: list[list], rng: random.Random, keep_minus1: bool) -> dict:
    """Give every -1 of a Seed axis its own random seed (in place: values and labels show it); returns them."""
    fixed = {}
    for letter, axis, values in zip(LETTERS, axes, resolved):
        if axis.spec.id != "seed" or keep_minus1:
            continue
        values[:] = [rng.randrange(SEED_RANGE) if v == -1 else v for v in values]
        axis.values = values
        fixed[letter] = list(values)
    return fixed


def _loop_letters(costs: dict) -> tuple[str, str, str]:
    """Outer, middle, inner axis letters."""
    cx, cy, cz = costs["x"], costs["y"], costs["z"]
    if cx > cy and cx > cz:
        return ("x", "y", "z") if cy > cz else ("x", "z", "y")
    if cy > cx and cy > cz:
        return ("y", "x", "z") if cx > cz else ("y", "z", "x")
    if cz > cx and cz > cy:
        return ("z", "x", "y") if cx > cy else ("z", "y", "x")
    return ("z", "y", "x")


def _order(counts: dict, costs: dict) -> list[int]:
    nx, ny = counts["x"], counts["y"]
    outer, middle, inner = _loop_letters(costs)
    order = []
    for a in range(counts[outer]):
        for b in range(counts[middle]):
            for c in range(counts[inner]):
                pos = {outer: a, middle: b, inner: c}
                order.append(pos["x"] + pos["y"] * nx + pos["z"] * nx * ny)
    return order


def _seed_offset(ix: int, iy: int, iz: int, counts: dict, options: dict) -> int:
    vx, vy, vz = (bool(options.get(f"vary_{letter}")) for letter in LETTERS)
    nx = counts["x"] if vx else 1
    ny = counts["y"] if vy else 1
    return (ix if vx else 0) + (iy * nx if vy else 0) + (iz * nx * ny if vz else 0)


# ---------------------------------------------------------------------------------------------- applying values

def _set_path(state: dict, path: str, value) -> None:
    *parents, last = path.split(".")
    node = state
    for part in parents:
        node = node[part]
    node[last] = value


def _search_replace(state: dict, search: str, value: str) -> None:
    if search not in state["prompt"] and search not in state["negative"]:
        raise _PlanError(f'Prompt S/R did not find "{search}" in either prompt or negative prompt')
    state["prompt"] = state["prompt"].replace(search, value)
    state["negative"] = state["negative"].replace(search, value)


def _reorder(prompt: str, tokens: tuple) -> str:
    """Cut ``tokens`` out of the prompt and put them back into the same slots, in the given order."""
    by_position = []
    for token in tokens:
        at = prompt.find(token)
        if at < 0:
            raise _PlanError(f'Prompt order did not find "{token}" in the prompt')
        by_position.append((at, token))
    by_position.sort(key=lambda item: item[0])
    gaps, rest = [], prompt
    for _at, token in by_position:
        cut = rest.find(token)
        if cut < 0:
            raise _PlanError(f'Prompt order did not find "{token}" in the prompt')
        gaps.append(rest[:cut])
        rest = rest[cut + len(token):]
    return "".join(gap + token for gap, token in zip(gaps, tokens)) + rest


def _apply(state: dict, axis: _Axis, value, search) -> None:
    spec = axis.spec
    if spec.kind == "nothing":
        return
    if spec.id == "prompt_sr":
        _search_replace(state, search, value)
    elif spec.id == "prompt_order":
        state["prompt"] = _reorder(state["prompt"], value)
    elif spec.id == "lora":
        xyz_lora.apply_column(state, axis.block, value)
    elif spec.id == "denoising":
        _set_path(state, "hires.denoise" if state["tab"] == "t2i" else "i2i.denoise", value)
    elif spec.id == "dora_mode":
        state["dora_mode"] = value
        state["forge_like"]["dora"] = False  # otherwise "Forge처럼" would mask the axis value
    elif spec.id == "attention":
        state["attention"] = value
        state["forge_like"]["attention"] = False  # the same
    else:
        paths = spec.field.split(",")
        if spec.kind == "size" and state["tab"] == "i2i":
            paths = ["i2i.width", "i2i.height"]
        parts = value if len(paths) > 1 else (value,)
        for path, part in zip(paths, parts):
            _set_path(state, path, part)


# ---------------------------------------------------------------------------------------------- labels, infotext

def _basename(text: str) -> str:
    return re.split(r"[\\/]", text)[-1]


def _value_text(value) -> str:
    if isinstance(value, float):
        return str(round(value, 8))
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return f"{value[0]}x{value[1]}"
    return str(value)


def _label(spec: AxisSpec, value) -> str:
    if spec.fmt == "empty":
        return ""
    if spec.id == "lora":
        return value.label
    if spec.fmt == "value":
        return _value_text(value)
    if spec.fmt == "basename":
        return _basename(str(value))
    if spec.fmt == "joined":
        return ", ".join(value)
    if spec.id == "text_encoder":  # sam-extra's label: "Text Encoder: <file stem>"
        return f"Text Encoder: {re.sub(r'[.][^.]*$', '', _basename(str(value)))}"
    return f"{spec.label}: {_value_text(value)}"


def _infotext(axes: list[_Axis], fixed: dict) -> dict:
    extra = {"Script": SCRIPT_NAME}
    for letter, axis in zip(LETTERS, axes):
        if axis.spec.kind == "nothing":
            continue
        key = letter.upper()
        extra[f"{key} Type"] = axis.spec.label
        extra[f"{key} Values"] = axis.text
        if letter in fixed:
            extra[f"Fixed {key} Values"] = ", ".join(str(v) for v in fixed[letter])
    return extra


# ---------------------------------------------------------------------------------------------- LoRA files

def missing_loras(states, available) -> list[str]:
    """Tag names (first appearance order) of the ``<lora:...>`` tags the states load that match no file of
    ``available`` (``resolve_lora_name``, the LoRA node's rule).  ``available`` may be a zero-argument callable; it is
    only called when some state has a tag, and its error propagates."""
    names: list[str] = []
    for state in states:
        tags, _ = parse_lora_tags(effective_prompt(state), state.get("negative") or "")
        names += [tag.name for tag in tags if tag.name not in names]
    if not names:
        return []
    files = list(available() if callable(available) else available)
    return [name for name in names if resolve_lora_name(name, files) is None]


def _check_loras(cells: list, loras) -> None:
    try:
        missing = missing_loras([cell.state for cell in cells], loras)
    except Exception as exc:  # the LoRA folder could not be listed: do not report every tag as missing
        raise _PlanError(f"LoRA list could not be read: {exc}") from exc
    if missing:
        raise _PlanError(f"LoRA not found: {', '.join(missing)}")


# ---------------------------------------------------------------------------------------------- the plan

def _refusal(errors, axes) -> list:
    """A LoRA axis' panel lists its columns; without any, the last line says that nothing will run."""
    lora_axis = any(isinstance(entry, dict) and entry.get("axis") == "lora" for entry in list(axes or [])[:3])
    return [*errors, xyz_lora.NOT_MADE] if lora_axis else list(errors)


def _expand_wildcards(cells: list, wildcards, rng: random.Random) -> list:
    missing: list = []
    for cell in cells:
        if not uses_wildcards(cell.state):
            continue
        if cell.state["seed"] < 0:  # keep_minus1: the seed the picks came from must be the one the cell runs with
            cell.state["seed"] = rng.randrange(SEED_RANGE)
        cell.state, names = expand_state(cell.state, cell.state["seed"], wildcards)
        missing += [name for name in names if name not in missing]
    return missing


def plan_xyz(state: dict, axes: list, options: dict, *, rng: random.Random, families: dict, loras=None, lora_epoch=None,
             wildcards=None) -> XyzPlan:
    """Plan an X/Y/Z grid: ``axes`` is ``[{"axis": AxisSpec.id, "text": value box, "lora": LoRA axis settings}]`` for X, Y, Z (missing = Nothing),
    ``families`` maps every UNET file to ``"2.9b" | "base" | "3.8b" | "moss" | "krea2"``, ``loras`` is the LoRA file list (or a callable
    returning it; None skips the LoRA check, and a LoRA axis then cannot plan), ``lora_epoch`` tells a LoRA file's epoch
    number (None if it cannot), ``wildcards`` gives a wildcard's groups by name (None leaves wildcards as written).
    See the module docstring for the rules."""
    options = options if isinstance(options, dict) else {}
    keep_minus1 = bool(options.get("keep_minus1"))
    try:
        parsed = _read_axes(axes)
        base = normalize_state(state)
        prompt = effective_prompt(base)
        _expand_lora_axes(parsed, prompt, base["negative"], loras, lora_epoch)
        counts = {letter: len(axis.values) for letter, axis in zip(LETTERS, parsed)}
        nx, ny, nz = counts["x"], counts["y"], counts["z"]
        resolved = _resolve(parsed, families or {})
        _check_panel_family(base, parsed, resolved, families or {})
        _check_ext_family(base, parsed, families or {})
        width, height = generation_size(base)
        grid_mp = round(nx * ny * nz * width * height / 1e6)
        if grid_mp >= MAX_GRID_MP:
            raise _PlanError(f"Resulting grid would be too large ({grid_mp} MPixels, max {MAX_GRID_MP})")
        if base["seed"] == -1 and not keep_minus1:
            base["seed"] = rng.randrange(SEED_RANGE)
        fixed = _fix_seed_axes(parsed, resolved, rng, keep_minus1)
        base["prompt"] = prompt
        base.update(dict.fromkeys(PROMPT_BOXES, ""))
        base["loras"] = []
        searches = [values[0] if axis.spec.id == "prompt_sr" else None for axis, values in zip(parsed, resolved)]
        find_file = xyz_lora.file_finder(_lora_file_list(loras)()) if any(axis.spec.id == "lora" for axis in parsed) else None
        cells = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cell = copy.deepcopy(base)
                    for axis in parsed:
                        for path, switch in axis.spec.switches:
                            _set_path(cell, path, switch)
                    for axis, values, search, i in zip(parsed, resolved, searches, (ix, iy, iz)):
                        _apply(cell, axis, values[i], search)
                    if find_file:
                        picked = [values[i] for axis, values, i in zip(parsed, resolved, (ix, iy, iz)) if axis.spec.id == "lora"]
                        xyz_lora.check_cell(cell, picked, find_file)
                    cell["seed"] += _seed_offset(ix, iy, iz, counts, options)
                    cells.append(Cell(ix, iy, iz, normalize_state(cell)))
        wildcards_missing = _expand_wildcards(cells, wildcards, rng) if wildcards is not None else []
        if loras is not None:
            _check_loras(cells, loras)
    except (_PlanError, xyz_lora.LoraAxisError) as exc:
        return XyzPlan(errors=_refusal(exc.args, axes))
    return XyzPlan(
        cells=cells,
        order=_order(counts, {letter: axis.spec.cost for letter, axis in zip(LETTERS, parsed)}),
        labels={letter: [_label(axis.spec, v) for v in axis.values] for letter, axis in zip(LETTERS, parsed)},
        fixed=fixed,
        infotext_extra=_infotext(parsed, fixed),
        errors=[],
        lora_preview={letter: [xyz_lora.column_preview(column, axis.block, prompt, base["negative"]) for column in axis.values]
                      for letter, axis in zip(LETTERS, parsed) if axis.spec.id == "lora"},
        wildcards_missing=wildcards_missing,
    )
