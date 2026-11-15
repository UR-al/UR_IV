"""Picker order for the studio's sampler and scheduler lists: similar names sit next to each other.

ComfyUI appends what a pack registers to the end of its lists, so ``karras`` and ``karras_dynamic``, or ``beta``,
``beta57`` and ``beta_1_1``, end up far apart.  ``group_similar`` moves names, it never renames, adds or drops one: a saved
panel or workflow names its sampler by value and loads as before.  Pure: no ComfyUI import.
"""
from __future__ import annotations

import re
from typing import Iterable

# A word is a run of letters, or a number with the letters right after it: "beta57" -> beta, 57 and "dpmpp_2m" -> dpmpp, 2m.
_WORD = re.compile(r"[^\W\d_]+|\d+[^\W\d_]*")


def name_words(name: str) -> tuple[str, ...]:
    """The words of a name, whatever its case and separators: ``Karras Dynamic`` and ``karras_dynamic`` are
    ``("karras", "dynamic")``.  A name without a letter or digit is one word."""
    folded = str(name).casefold()
    return tuple(_WORD.findall(folded)) or (folded,)


def _grouped(entries: list, depth: int) -> list:
    if len(entries) < 2:
        return [name for _, name in entries]
    families: dict = {}
    for words, name in entries:
        families.setdefault(words[depth] if depth < len(words) else None, []).append((words, name))
    names = [name for _, name in families.pop(None, ())]  # the names that end here lead the ones that go on
    for members in families.values():
        names += _grouped(members, depth + 1)
    return names


def group_similar(names: Iterable[str]) -> list[str]:
    """``names`` with every family in one run.

    A family is the names that start with the same word (``name_words``).  Families keep the order their first name came
    in, and names keep their order inside a family, with two exceptions.  The family is grouped again by its next word
    (``dpmpp_2m_flow_ode`` goes up to ``dpmpp_2m`` and ``dpmpp_2m_sde``, before ``dpmpp_3m_sde``).  And the names that end
    where the others go on lead them: ``euler_dy`` comes before ``euler_dy_cfg_pp`` although the pack registers it later,
    the way every ComfyUI pair reads (``euler``, ``euler_cfg_pp``).
    """
    return _grouped([(name_words(name), name) for name in names], 0)
