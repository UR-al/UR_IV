"""Forge prompt editing for the Anima studio (pure: no ComfyUI, no torch).

``schedule_prompt(text, steps)`` returns what Forge Neo's prompt parser makes of a prompt: a list of
``(end step, text)`` segments, the text of each segment used up to and including its end step.  The behaviour of
Forge Neo's ``modules/prompt_parser.py`` (AGPL-3.0) is restated here as a small hand-written parser; no Forge code is
carried (see ``studio_forge_parity.md``).  The rules:

- ``[a:b:w]`` uses ``a`` up to step ``w`` and ``b`` after it; ``[a::w]`` drops ``a`` after ``w``; ``[b:w]`` and
  ``[:b:w]`` add ``b`` after ``w``.  ``w`` written with a dot is a share of the steps (``0.5``, ``.25``, ``1.5``),
  without a dot a step count; the step is cut to ``steps`` and a step below 1 means "the second part from the
  start".
- ``[a|b|...]`` switches option every step (an empty option is empty text).
- They nest, and they also work inside ``( )`` / ``[ ]`` emphasis, which is kept as written (the tokenizer reads
  the weights later).
- ``\\x`` keeps ``x`` as text; the backslash stays in the text, as in Forge.
- Brackets and colons that do not form one of the above stay text at the top level; inside a group they make the
  group text.  A ``|`` outside an alternation, or a lone backslash at the very end, makes Forge give up on the
  whole prompt: it is then used as written, with no editing at all.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Union

_SPECIAL = frozenset("\\[]():|")
_TOP_LEVEL_TEXT = frozenset("[]():")
_NUMBER = re.compile(r"[+-]?(?:\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)")
_SPACE = re.compile(r"\s+")


@dataclass
class _Edit:
    """``[before:after:when]``; ``before`` None = the ``[after:when]`` form."""

    before: Optional[list]
    after: list
    raw: str        # the number as written
    step: int = 0   # set by _resolve


@dataclass
class _Alternation:
    options: list


_Item = Union[str, _Edit, _Alternation]


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text

    def char(self, pos: int) -> str:
        return self.text[pos] if pos < len(self.text) else ""

    def plain(self, pos: int) -> int:
        """End of the plain text run at ``pos`` (``pos`` itself when there is none)."""
        text, end = self.text, pos
        while end < len(text):
            ch = text[end]
            if ch == "\\":
                if end + 1 >= len(text) or text[end + 1] == "\n":
                    break
                end += 2
            elif ch in _SPECIAL:
                break
            else:
                end += 1
        return end

    def sequence(self, pos: int) -> tuple[list, int]:
        """Text, groups and edits up to the first character that cannot continue them."""
        items: list = []
        while pos < len(self.text):
            end = self.plain(pos)
            if end > pos:
                items.append(self.text[pos:end])
                pos = end
                continue
            if self.char(pos) not in "[(":
                break
            group = self.group(pos)
            if group is None:
                break
            parts, pos = group
            items.extend(parts)
        return items, pos

    def group(self, pos: int) -> Optional[tuple[list, int]]:
        if self.char(pos) == "(":
            first, pos = self.sequence(pos + 1)
            if self.char(pos) == ")":
                return ["(", *first, ")"], pos + 1
            if self.char(pos) == ":":
                second, pos = self.sequence(pos + 1)
                if self.char(pos) == ")":
                    return ["(", *first, ":", *second, ")"], pos + 1
            return None
        first, pos = self.sequence(pos + 1)
        mark = self.char(pos)
        if mark == "]":
            return ["[", *first, "]"], pos + 1
        if mark == "|":
            options = [first]
            while self.char(pos) == "|":
                option, pos = self.sequence(pos + 1)
                options.append(option)
            if self.char(pos) == "]":
                return [_Alternation(options)], pos + 1
            return None
        if mark == ":":
            closed = self.when(pos + 1)
            if closed is not None:
                return [_Edit(None, first, closed[0])], closed[1]
            second, pos = self.sequence(pos + 1)
            if self.char(pos) == ":":
                closed = self.when(pos + 1)
                if closed is not None:
                    return [_Edit(first, second, closed[0])], closed[1]
        return None

    def when(self, pos: int) -> Optional[tuple[str, int]]:
        """``[spaces] NUMBER [spaces] ]`` at ``pos`` -> (number text, position after ``]``)."""
        space = _SPACE.match(self.text, pos)
        pos = space.end() if space else pos
        number = _NUMBER.match(self.text, pos)
        if number is None:
            return None
        pos = number.end()
        space = _SPACE.match(self.text, pos)
        pos = space.end() if space else pos
        if self.char(pos) != "]":
            return None
        return number.group(), pos + 1

    def prompt(self) -> Optional[list]:
        """The whole prompt, or None when Forge could not parse it."""
        items: list = []
        pos = 0
        while pos < len(self.text):
            parts, pos = self.sequence(pos)
            items.extend(parts)
            if pos < len(self.text):
                ch = self.text[pos]
                if ch not in _TOP_LEVEL_TEXT:
                    return None
                items.append(ch)
                pos += 1
        return items


def _when_step(raw: str, steps: int) -> int:
    value = float(raw)
    if "." in raw:
        value *= steps
    if not math.isfinite(value):
        return steps if value > 0 else 0
    return min(steps, int(value))


def _resolve(items: list, steps: int, ends: set[int]) -> None:
    """Turns each edit's number into its step and collects the steps where the text changes."""
    for item in items:
        if isinstance(item, _Edit):
            item.step = _when_step(item.raw, steps)
            if item.step >= 1:
                ends.add(item.step)
            _resolve(item.before or [], steps, ends)
            _resolve(item.after, steps, ends)
        elif isinstance(item, _Alternation):
            ends.update(range(1, steps + 1))
            for option in item.options:
                _resolve(option, steps, ends)


def _render(items: list, step: int) -> str:
    out = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, _Edit):
            out.append(_render((item.before or []) if step <= item.step else item.after, step))
        else:
            out.append(_render(item.options[(step - 1) % len(item.options)], step))
    return "".join(out)


def schedule_prompt(text: str, steps: int) -> list[tuple[int, str]]:
    """Forge prompt editing: ``[(end step, text of the steps up to it), ...]``, the last end being ``steps``."""
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    text = str(text)
    items = _Parser(text).prompt()
    if items is None:
        return [(steps, text)]
    ends = {steps}
    _resolve(items, steps, ends)
    return [(end, _render(items, end)) for end in sorted(ends)]


__all__ = ["schedule_prompt"]
