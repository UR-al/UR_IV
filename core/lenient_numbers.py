"""Lenient numeric coercion shared by the ComfyUI compiler and backend.

Payload values arrive from widgets, saved settings, PNG info and HTTP clients,
so they can be strings, floats, ``None`` or garbage.  These helpers turn them
into finite numbers or a caller default and never raise: ``"1e400"``/``inf``
and ``nan`` are treated like any other unusable value (``int(float("1e400"))``
raises ``OverflowError``, which a plain ``(TypeError, ValueError)`` guard does
not catch).
"""
from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any, default: float) -> float:
    """``float(value)`` when finite, else ``default``."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def lenient_int(value: Any, default: int) -> int:
    """``int(float(value))`` (truncating) when finite, else ``default``.

    Integers and integer strings are converted exactly, so a uint64 seed such
    as ``18446744073709551615`` never loses precision through ``float``.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


__all__ = ["finite_float", "lenient_int"]
