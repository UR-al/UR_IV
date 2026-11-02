"""X/Y/Z value syntax for the Anima studio: CSV splitting and range expressions (pure Python).

Compatible with Forge's value box: values are comma or newline separated (double quotes protect commas),
and number axes also take ``1-5``, ``1-10 (+3)`` and ``1-10 [5]`` forms.  Written from the behaviour
description only; the pack carries no Forge code.

Deliberate differences from Forge: an unparsable value raises ``ValueError`` (Forge crashes or, for
``Size``, silently keeps the current size) so the whole list is validated before any image is made, and
``1e-3`` is one number, not a range.  One axis expands to at most ``MAX_VALUES`` values (ranges and
permutations are counted before they are built), so a typo like ``0-999999999`` fails fast instead of
filling memory.
"""
from __future__ import annotations

import csv
import io
import itertools
import math
import re
from typing import Literal

Kind = Literal["int", "float", "str", "lines", "choice", "permutations", "size", "nothing", "bool", "lora"]
KINDS: tuple[str, ...] = ("int", "float", "str", "lines", "choice", "permutations", "size", "nothing", "bool", "lora")

_INT_NUM = r"[+-]?\s*\d+"
_FLOAT_NUM = r"[+-]?\s*\d+(?:\.\d*)?"
_INT_RANGE = re.compile(rf"\s*({_INT_NUM})\s*-\s*({_INT_NUM})(?:\s*\(([+-]\d+)\s*\))?\s*")
_INT_COUNT = re.compile(rf"\s*({_INT_NUM})\s*-\s*({_INT_NUM})(?:\s*\[(\d+)\s*])?\s*")
_FLOAT_RANGE = re.compile(rf"\s*({_FLOAT_NUM})\s*-\s*({_FLOAT_NUM})(?:\s*\(([+-]\d+(?:\.\d*)?)\s*\))?\s*")
_FLOAT_COUNT = re.compile(rf"\s*({_FLOAT_NUM})\s*-\s*({_FLOAT_NUM})(?:\s*\[(\d+(?:\.\d*)?)\s*])?\s*")
_SIZE = re.compile(r"\s*(\d+)\s*x\s*(\d+)\s*")
_TRUE = {"true", "1", "on", "yes"}
_FALSE = {"false", "0", "off", "no"}
MAX_VALUES = 10000


def _check_count(count: int) -> None:
    if count > MAX_VALUES:
        raise ValueError(f"too many values ({count}, at most {MAX_VALUES} per axis)")


def split_values(text: str) -> list[str]:
    """Split a value box on commas and newlines (CSV rules, quotes protect commas); every field is stripped."""
    reader = csv.reader(io.StringIO(text or ""), skipinitialspace=True)
    return [field.strip() for row in reader for field in row]


def _num(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _linspace(start: float, end: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [start]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count - 1)] + [end]


def _int_values(field: str) -> list[int]:
    match = _INT_RANGE.fullmatch(field)
    if match:
        start, end = int(_num(match[1])), int(_num(match[2]))
        step = int(match[3]) if match[3] else 1
        if step == 0:
            raise ValueError(f"step must not be 0: {field!r}")
        values = range(start, end + 1, step)
        _check_count(len(values))
        return list(values)  # end + 1 even for a negative step, like Forge
    match = _INT_COUNT.fullmatch(field)
    if match:
        start, end = int(_num(match[1])), int(_num(match[2]))
        count = int(match[3]) if match[3] else 1
        _check_count(count)
        return [int(v) for v in _linspace(start, end, count)]
    try:
        return [int(field)]
    except ValueError:
        raise ValueError(f"not an integer: {field!r}") from None


def _arange(start: float, stop: float, step: float) -> list[float]:
    length = math.ceil((stop - start) / step)
    _check_count(length)
    return [start + i * step for i in range(max(0, length))]


def _float_values(field: str) -> list[float]:
    match = _FLOAT_RANGE.fullmatch(field)
    if match:
        start, end = float(_num(match[1])), float(_num(match[2]))
        step = float(match[3]) if match[3] else 1.0
        if step == 0:
            raise ValueError(f"step must not be 0: {field!r}")
        return _arange(start, end + step, step)  # one extra step so the end is included, like Forge
    match = _FLOAT_COUNT.fullmatch(field)
    if match:
        start, end = float(_num(match[1])), float(_num(match[2]))
        count = int(float(match[3])) if match[3] else 1
        _check_count(count)
        return _linspace(start, end, count)
    try:
        value = float(field)
    except ValueError:
        raise ValueError(f"not a number: {field!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: {field!r}")
    return [value]


def _size_value(field: str) -> tuple[int, int]:
    match = _SIZE.fullmatch(field)
    if not match:
        raise ValueError(f"size must look like 512x768: {field!r}")
    return int(match[1]), int(match[2])


def _bool_value(field: str) -> bool:
    word = field.lower()
    if word in _TRUE:
        return True
    if word in _FALSE:
        return False
    raise ValueError(f"not a true/false value: {field!r}")


def _capped(values) -> list:
    out = []
    for value in values:
        out.append(value)
        _check_count(len(out))
    return out


def parse_axis_values(kind: Kind, text: str) -> list:
    """Parse one axis' value box.

    int/float -> numbers (blank fields skipped, ranges expanded); str/choice -> strings (blanks kept); lines -> one string
    per line, commas kept (a block list is a comma list); permutations -> tuples, every ordering of the tokens;
    size -> ``(width, height)`` tuples;
    bool -> bools; nothing and lora -> one ignored value (a LoRA axis has its block, see ``xyz_lora``).  Raises
    ``ValueError`` for anything unparsable.
    """
    if kind in ("nothing", "lora"):
        return [""]
    if kind == "lines":
        return [line.strip() for line in (text or "").splitlines() if line.strip()]
    fields = split_values(text)
    if kind in ("str", "choice"):
        return fields
    filled = [f for f in fields if f]
    if kind == "int":
        return _capped(v for f in filled for v in _int_values(f))
    if kind == "float":
        return _capped(v for f in filled for v in _float_values(f))
    if kind == "permutations":
        if not filled:
            return []
        _check_count(math.factorial(min(len(filled), 20)))  # 8 tokens already exceed the cap
        return list(itertools.permutations(filled))
    if kind == "size":
        return [_size_value(f) for f in filled]
    if kind == "bool":
        return [_bool_value(f) for f in filled]
    raise ValueError(f"unknown axis kind: {kind!r}")
