"""X/Y/Z planner for the Anima studio: one full studio state per grid cell, the generation order, labels and seeds.

Pure Python (no ComfyUI / torch).  Follows the behaviour of Forge Neo's X/Y/Z plot as described in the recon memo;
the pack carries no Forge code.

- Every cell starts from the normalized state with its list LoRAs already rendered into the prompt as
  ``<lora:name:strength>`` tags (``effective_prompt``; the cell's ``loras`` list is empty), so Prompt S/R can
  rewrite them.  Axes apply X, then Y, then Z.
- Prompt S/R: the first value is the search text and every value (the first included) is one column; plain,
  case-sensitive replace-all in the prompt and the negative prompt.  A search text found in neither aborts the plan.
- Prompt order: the given tokens are cut out of the positive prompt and put back into the same slots in each
  permutation's order.
- Seeds: unless ``keep_minus1``, a state seed of -1 becomes one random seed shared by every cell, and each -1 of a
  Seed axis becomes its own random seed (listed in ``fixed`` and ``Fixed X Values``).  ``vary_x/y/z`` add the cell
  position on top (``ix``; ``iy * nx``; ``iz * nx * ny``, a factor only for the varied axes).
- Order: the dearest axis (strictly) loops outermost so model reloads happen least often; ties keep Z, Y, X.
- Cells are listed with X fastest (index ``ix + iy*nx + iz*nx*ny``); ``order`` lists those indices in generation
  order.  Each cell state is normalized again, so axis values are clamped like the control panel's.

Validation problems (unparsable values, unknown checkpoint or DoRA mode, 3.8B mixed with 2.9B/base, a grid of
100 MP or more, Prompt S/R or Prompt order text not found) come back in ``errors`` with no cells; ``plan_xyz``
never raises for them.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field

from .dora_infer_mode import MODE_LABELS, mode_key
from .studio_state import effective_prompt, normalize_state
from .xyz_axes import AxisSpec, axis_by_id
from .xyz_values import parse_axis_values

LETTERS = ("x", "y", "z")
SEED_RANGE = 4294967294  # random seeds are drawn from [0, SEED_RANGE)
MAX_GRID_MP = 100
MIXED_FAMILIES = "2.9B·base와 3.8B는 한 격자에 섞을 수 없음"
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


class _PlanError(Exception):
    pass


@dataclass
class _Axis:
    spec: AxisSpec
    text: str
    values: list


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
        except KeyError:
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
        out.append(_Axis(spec, text, values))
    if errors:
        raise _PlanError(*errors)
    return out


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
        except _PlanError as exc:
            errors.extend(exc.args)
        resolved.append(values)
    if errors:
        raise _PlanError(*errors)
    if "3.8b" in seen and seen - {"3.8b"}:
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
    elif spec.id == "denoising":
        _set_path(state, "hires.denoise" if state["tab"] == "t2i" else "i2i.denoise", value)
    elif spec.id == "dora_mode":
        state["dora_mode"] = value
        state["forge_like"]["dora"] = False  # otherwise "Forge처럼" would mask the axis value
    else:
        paths = spec.field.split(",")
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


# ---------------------------------------------------------------------------------------------- the plan

def plan_xyz(state: dict, axes: list, options: dict, *, rng: random.Random, families: dict) -> XyzPlan:
    """Plan an X/Y/Z grid: ``axes`` is ``[{"axis": AxisSpec.id, "text": value box}]`` for X, Y, Z (missing = Nothing),
    ``families`` maps every UNET file to ``"2.9b" | "base" | "3.8b"``.  See the module docstring for the rules."""
    options = options if isinstance(options, dict) else {}
    keep_minus1 = bool(options.get("keep_minus1"))
    try:
        parsed = _read_axes(axes)
        base = normalize_state(state)
        counts = {letter: len(axis.values) for letter, axis in zip(LETTERS, parsed)}
        nx, ny, nz = counts["x"], counts["y"], counts["z"]
        resolved = _resolve(parsed, families or {})
        grid_mp = round(nx * ny * nz * base["width"] * base["height"] / 1e6)
        if grid_mp >= MAX_GRID_MP:
            raise _PlanError(f"Resulting grid would be too large ({grid_mp} MPixels, max {MAX_GRID_MP})")
        if base["seed"] == -1 and not keep_minus1:
            base["seed"] = rng.randrange(SEED_RANGE)
        fixed = _fix_seed_axes(parsed, resolved, rng, keep_minus1)
        base["prompt"] = effective_prompt(base)
        base["loras"] = []
        searches = [values[0] if axis.spec.id == "prompt_sr" else None for axis, values in zip(parsed, resolved)]
        cells = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    cell = copy.deepcopy(base)
                    for axis, values, search, i in zip(parsed, resolved, searches, (ix, iy, iz)):
                        _apply(cell, axis, values[i], search)
                    cell["seed"] += _seed_offset(ix, iy, iz, counts, options)
                    cells.append(Cell(ix, iy, iz, normalize_state(cell)))
    except _PlanError as exc:
        return XyzPlan(errors=list(exc.args))
    return XyzPlan(
        cells=cells,
        order=_order(counts, {letter: axis.spec.cost for letter, axis in zip(LETTERS, parsed)}),
        labels={letter: [_label(axis.spec, v) for v in axis.values] for letter, axis in zip(LETTERS, parsed)},
        fixed=fixed,
        infotext_extra=_infotext(parsed, fixed),
        errors=[],
    )
