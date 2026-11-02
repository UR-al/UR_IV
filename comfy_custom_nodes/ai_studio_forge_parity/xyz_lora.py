"""The LoRA axis of the Anima studio's X/Y/Z plot: LoRAs that take the place of one tag, with their triggers and epochs.

Pure Python (no ComfyUI / torch). The panel keeps one block per axis (``web/ai_studio/xyz_lora_view.js``):

    {"find": text, "after": bool, "rows": [{"lora", "strength", "trigger", "epoch_mode", "epoch"}, ...]}

``rows[0]`` is the base: ``lora`` names a LoRA the prompt loads, exactly, so other LoRAs stay as they are. Tags are read as
``studio_lora_tags`` reads them. Every row gives one column, or one per epoch:

- ``epoch_mode``: ``"off"``; ``"last"``, from the row's file to the run's last file, the one without a number; ``"direct"``,
  from the row's file to epoch ``epoch``. An empty ``epoch`` is ``"off"``; one before the row's file or past the run stops
  the plan. The last file's number is what the file says (``read_epoch``), else one saving interval after the others.
- ``strength``: written into the tag. None takes the base row's, and with none there either the tag keeps its own numbers.
- ``trigger`` (rows after the base): empty keeps the prompt. Otherwise ``find`` says where it goes: it replaces ``find``, or
  with ``after`` goes behind the prompt tag holding ``find``, or without ``find`` to the front of the prompt. ``-`` removes
  ``find``.

``find`` must be in the positive prompt. Like Prompt S/R, replacing also covers the negative prompt, but never the inside of
an extra-network tag: a trigger is usually also the start of its LoRA's file name. Inserting covers the prompt alone: a
trigger is never wanted in the negative. A trigger may not change the prompt's tags, and a finished cell may not load a
column's file more often than the prompt loaded the base (``check_cell``). Problems raise ``LoraAxisError``; ``xyz_plan``
reports them before any image is made.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

from .studio_lora_tags import (
    _EXTRA_NET_RE,
    _NEGATIVE_LORA_RE,
    _lora_tag,
    _strip_extension,
    collapse_duplicates,
    parse_lora_tags,
    resolve_lora_name,
)

REMOVE = "-"  # a row's trigger that removes the find text
LAST_LABEL = "last"
NO_TRIGGER = "없음"
REMOVED = "지움"
NOT_MADE = "생성하지 않습니다."
MAX_STRENGTH = 10.0
_NUMBERED = re.compile(r"^(.*)-(\d{6})$")
_NETWORK_TAG = re.compile(r"(<\w+:[^>]+>)")  # studio_lora_tags' extra-network tag, whole
_NEGATIVE_TAG = re.compile(r"(<lora:[^>]+>)", re.IGNORECASE)  # the only tags of a negative prompt
_MASK = "\0"  # stands in for the characters of an extra-network tag
_WEIGHT = r"(?::\s*-?[\d.]+\s*)?"


class LoraAxisError(Exception):
    """Why the axis cannot run; each argument is one line for the user."""


@dataclass(frozen=True)
class Epoch:
    number: int
    entry: str  # the file-list entry
    last: bool = False  # a file without a number: a run's last file


@dataclass(frozen=True)
class Column:
    name: str  # the LoRA name the tag gets
    strength: Optional[float]  # None: the tag keeps its own numbers
    trigger: str  # the row's new trigger; the base row's is always ""
    label: str = ""
    copies: int = 1  # the base LoRA's tags in the panel's prompts: all of them become this column


@dataclass(frozen=True)
class _Draft:
    name: str
    epoch: Epoch
    strength: Optional[float]
    trigger: str


def _split(entry: str) -> tuple[str, str]:
    """(folder with its slash, file stem) of a file-list entry."""
    path = _strip_extension(str(entry).replace("\\", "/"))
    cut = path.rfind("/") + 1
    return path[:cut], path[cut:]


def _run_of(stem: str) -> tuple[str, Optional[int]]:
    match = _NUMBERED.match(stem)
    return (match.group(1), int(match.group(2))) if match else (stem, None)


def _number_of(name: str) -> Optional[int]:
    return _run_of(_split(name)[1])[1]


def _run_key(entry: str) -> tuple[str, str]:
    folder, stem = _split(entry)
    return folder.casefold(), _run_of(stem)[0].casefold()


def _saving_interval(numbers: list[int]) -> int:
    gaps = [later - earlier for earlier, later in zip(numbers, numbers[1:])]
    return min(gaps) if gaps else numbers[0]


def _last_number(numbers: list[int], final: str, read_epoch: Optional[Callable]) -> int:
    top = numbers[-1] if numbers else 0
    told = read_epoch(final) if read_epoch else None
    if told is not None and told > top:
        return told
    return top + _saving_interval(numbers) if numbers else 1


def epoch_series(entry: str, files, read_epoch: Optional[Callable] = None) -> list[Epoch]:
    """The files of ``entry``'s training run, in epoch order: its folder, its name with ``-NNNNNN``, and the file without a
    number as the last one. ``read_epoch(file-list entry)`` tells that file's number, None if it cannot."""
    key = _run_key(entry)
    numbered, final = {}, None
    for item in files:
        folder, stem = _split(item)
        if (folder.casefold(), stem.casefold()) == key:  # the run's name itself may end like a number
            final = item
        elif _run_key(item) == key:
            numbered[_number_of(item)] = item
    numbers = sorted(numbered)
    series = [Epoch(number, numbered[number]) for number in numbers]
    return series if final is None else [*series, Epoch(_last_number(numbers, final, read_epoch), final, True)]


def _tag_name(entry: str, files) -> str:
    """How a tag names ``entry``: its stem, or with the folder when another file has the same stem."""
    folder, stem = _split(entry)
    return stem if resolve_lora_name(stem, files) == entry else f"{folder}{stem}"


def _epoch_number(value) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(number) if number.is_integer() and number >= 1 else None


def _epoch_files(row: dict, entry: str, files, where: str, read_epoch: Optional[Callable]) -> list[tuple[str, Epoch]]:
    """(tag name, epoch) of each column of a row that compares epochs."""
    direct = row["epoch_mode"] == "direct"
    series = epoch_series(entry, files, read_epoch if direct else None)
    number = _number_of(entry)
    start = next((index for index, item in enumerate(series) if item.entry == entry or item.number == number), 0)
    current, last = series[start], series[-1]
    target = last.number
    if direct:
        target = _epoch_number(row["epoch"])
        if target is None:
            raise LoraAxisError(f"{where}: 에폭 번호를 확인해 주세요")
        if target < current.number:
            raise LoraAxisError(f"{where}: 에폭 {target} 은 현재 에폭 {current.number} 보다 앞입니다. 에폭을 바꿔 주세요")
        if target > last.number:
            raise LoraAxisError(f"{where}: 에폭 {target} 파일이 없습니다. 마지막은 {last.number} 입니다")
    return [(_tag_name(item.entry, files), item) for item in series[start:] if item.number <= target]


def _row_files(row: dict, files, where: str, read_epoch: Optional[Callable]) -> list[tuple[str, Epoch]]:
    """(tag name, epoch) of each column of a row."""
    entry = resolve_lora_name(row["lora"], files)
    if entry is None:
        raise LoraAxisError(f"{where}: {row['lora']} 파일을 찾지 못했습니다")
    number = _number_of(entry)
    picked = [(row["lora"], Epoch(number or 0, entry, number is None))]
    if row["epoch_mode"] != "off":
        picked = _epoch_files(row, entry, files, where, read_epoch)
    for name, _ in picked:
        tag = _lora_tag(name)
        if tag is None or tag.name != name:
            raise LoraAxisError(f"{where}: {name} 은 <lora:...> 태그로 쓸 수 없는 이름입니다")
    return picked


def _strength(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(-MAX_STRENGTH, min(MAX_STRENGTH, number)) if math.isfinite(number) else None


def _clean_row(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    mode = raw.get("epoch_mode")
    epoch = raw.get("epoch")
    if mode == "direct" and epoch in (None, ""):
        mode = "off"
    return {
        "lora": str(raw.get("lora") or "").strip(),
        "strength": _strength(raw.get("strength")),
        "trigger": str(raw.get("trigger") or "").strip(),
        "epoch_mode": mode if mode in ("direct", "last") else "off",
        "epoch": epoch,
    }


def _rows(settings: dict) -> list[dict]:
    rows = settings.get("rows")
    return [_clean_row(raw) for raw in rows] if isinstance(rows, list) else []


def base_lora(settings: dict) -> str:
    rows = _rows(settings)
    return rows[0]["lora"] if rows else ""


def _find_text(settings: dict) -> str:
    return str(settings.get("find") or "").strip()


def _loaded_names(prompt: str, negative: str) -> list[str]:
    return [tag.name.casefold() for tag in parse_lora_tags(prompt, negative)[0]]


def _digits(value: float) -> str:
    return f"{value:g}"


def _masked(text: str) -> str:
    """``text`` with its extra-network tags blanked, at the same length."""
    return _NETWORK_TAG.sub(lambda match: _MASK * len(match.group(0)), text)


def _tags_in(state: dict) -> tuple[list, list]:
    return _NETWORK_TAG.findall(state["prompt"]), _NEGATIVE_TAG.findall(state["negative"])


def _outside_tags(text: str, tag: re.Pattern, change: Callable[[str], str]) -> str:
    parts = tag.split(text)
    parts[::2] = [change(part) for part in parts[::2]]
    return "".join(parts)


def _acts(settings: dict, trigger: str) -> bool:
    """Whether a row's trigger changes the prompt: ``-`` needs a text to remove."""
    return bool(trigger) and not (trigger == REMOVE and (settings.get("after") or not _find_text(settings)))


def _needs_find(settings: dict, trigger: str) -> bool:
    return _acts(settings, trigger) and bool(_find_text(settings))


def _not_found(settings: dict) -> str:
    return f'찾을 말 "{_find_text(settings)}" 이 프롬프트에 없습니다'


def _trigger_text(settings: dict, trigger: str) -> str:
    find = _find_text(settings)
    adds = bool(settings.get("after")) or not find
    if trigger == REMOVE:
        return NO_TRIGGER if adds else REMOVED
    if trigger:
        return f"+ {trigger}" if adds else trigger
    return NO_TRIGGER if adds else find


def _label(draft: _Draft, same_run: bool, numbered: bool, several: bool) -> str:
    stem = _split(draft.name)[1]
    if not same_run:
        text = stem
    elif draft.epoch.last:
        text = LAST_LABEL if numbered else stem
    else:
        text = f"e{draft.epoch.number}"
    return f"{text} ({_digits(draft.strength)})" if several and draft.strength is not None else text


def _told_apart(labels: list[str], drafts: list[_Draft], settings: dict) -> list[str]:
    """Two columns can differ by their folder or their trigger alone; the grid must still tell them apart."""

    def told(label: str, draft: _Draft) -> str:
        twins = [other for other, text in zip(drafts, labels) if text == label]
        folder, stem = _split(draft.epoch.entry)
        if len({other.epoch.entry for other in twins}) > 1:
            label = label.replace(stem, f"{folder}{stem}", 1)
        if len({other.trigger for other in twins}) > 1:
            label = f"{label} / {_trigger_text(settings, draft.trigger)}"
        return label

    return [told(label, draft) for label, draft in zip(labels, drafts)]


def lora_columns(settings: dict, prompt: str, negative: str, read_files: Callable[[], list],
                 read_epoch: Optional[Callable] = None) -> list[Column]:
    """The columns of one axis block for a prompt with its LoRAs as tags. ``read_files`` gives the LoRA file list,
    ``read_epoch`` a file's epoch number (see ``epoch_series``). Raises ``LoraAxisError``."""
    rows = _rows(settings)
    if not rows:
        raise LoraAxisError("LoRA 축: 줄이 없습니다")
    base = rows[0]
    if not base["lora"]:
        raise LoraAxisError("LoRA 축: 기준 LoRA 를 고르세요")
    copies = sum(name == base["lora"].casefold() for name in _loaded_names(prompt, negative))
    problems = [] if copies else [f"LoRA 축: 프롬프트에 {base['lora']} 이 없습니다"]
    problems += [f"LoRA 축: {index + 1}번 줄의 LoRA 를 고르세요" for index, row in enumerate(rows) if index and not row["lora"]]
    if any(_needs_find(settings, row["trigger"]) for row in rows[1:]) and _find_text(settings) not in _masked(prompt):
        problems.append(_not_found(settings))
    if problems:
        raise LoraAxisError(*problems)

    files = read_files()
    drafts = []
    for index, row in enumerate(rows):
        where = "기준" if index == 0 else f"{index + 1}번"
        strength = row["strength"] if row["strength"] is not None else base["strength"]
        drafts += [_Draft(name, epoch, strength, "" if index == 0 else row["trigger"]) for name, epoch in _row_files(row, files, where, read_epoch)]

    same_run = len({_run_key(draft.epoch.entry) for draft in drafts}) == 1
    numbered = any(not draft.epoch.last for draft in drafts)
    several = len({draft.strength for draft in drafts}) > 1
    labels = _told_apart([_label(draft, same_run, numbered, several) for draft in drafts], drafts, settings)
    return [Column(draft.name, draft.strength, draft.trigger, label, copies) for draft, label in zip(drafts, labels)]


def _swap_tags(text: str, pattern: re.Pattern, base: str, column: Column) -> str:
    def swapped(match: re.Match) -> str:
        tag = _lora_tag(match.group(2)) if match.group(1) == "lora" else None
        if tag is None or tag.name.casefold() != base.casefold():
            return match.group(0)
        if column.strength is not None:
            return f"<lora:{column.name}:{_digits(column.strength)}>"
        parts = match.group(2).split(":")
        name_at = next(index for index, part in enumerate(parts) if len(part.split("=", 2)) != 2)
        parts[name_at] = column.name
        return f"<lora:{':'.join(parts)}>"

    return pattern.sub(swapped, text)


def _without(text: str, word: str) -> str:
    """``text`` with ``word`` gone. Where it stands alone, also as ``(word:1.2)``, its brackets and one comma go with it, so
    what stays keeps its groups; inside a longer item only the word goes, so the neighbours stay apart."""
    escaped = re.escape(word)
    wrapped = re.compile(rf"\(\s*{escaped}\s*{_WEIGHT}\)|\[\s*{escaped}\s*{_WEIGHT}\]")
    first = re.compile(rf"(^|[,(\[\n])(\s*){escaped}\s*,\s*")
    last = re.compile(rf"\s*,\s*{escaped}(?=\s*{_WEIGHT}[)\]]|\s*$)")
    while wrapped.search(text):
        text = wrapped.sub(lambda _: word, text)
    while first.search(text):
        text = first.sub(lambda match: match.group(1) + match.group(2), text)
    return last.sub("", text).replace(word, "")


def _after_tag(prompt: str, find: str, trigger: str) -> str:
    """``trigger`` behind the comma item that holds the first ``find``, before any ``<lora:...>`` that ends it."""
    masked = _masked(prompt)
    start = masked.find(find) + len(find)
    item = re.split(r"[,\n]", masked[start:], maxsplit=1)[0]
    end = start + len(item.rstrip(f" \t\r{_MASK}"))
    return f"{prompt[:end]}, {trigger}{prompt[end:]}"


def _place_trigger(state: dict, settings: dict, trigger: str) -> None:
    if not _acts(settings, trigger):
        return
    find = _find_text(settings)
    if not find:
        state["prompt"] = f"{trigger}, {state['prompt']}" if state["prompt"] else trigger
        return
    if find not in _masked(state["prompt"]):
        raise LoraAxisError(_not_found(settings))
    if settings.get("after"):
        state["prompt"] = _after_tag(state["prompt"], find, trigger)
        return
    change = (lambda text: _without(text, find)) if trigger == REMOVE else (lambda text: text.replace(find, trigger))
    state["prompt"] = _outside_tags(state["prompt"], _NETWORK_TAG, change)
    state["negative"] = _outside_tags(state["negative"], _NEGATIVE_TAG, change)


def apply_column(state: dict, settings: dict, column: Column) -> None:
    """Writes one column into a cell's prompt and negative prompt. The cell may differ from the panel's prompt: another axis
    can have taken the tag or the find text away. A trigger with ``<`` or ``:`` can read as a tag, or swallow one."""
    base = base_lora(settings)
    if base.casefold() not in _loaded_names(state["prompt"], state["negative"]):
        raise LoraAxisError(f"LoRA 축: 프롬프트에 {base} 이 없습니다")
    state["prompt"] = _swap_tags(state["prompt"], _EXTRA_NET_RE, base, column)
    state["negative"] = _swap_tags(state["negative"], _NEGATIVE_LORA_RE, base, column)
    tags = _tags_in(state)
    _place_trigger(state, settings, column.trigger)
    if _tags_in(state) != tags:
        raise LoraAxisError(f"LoRA 축: 트리거 {column.trigger} 이 프롬프트의 태그를 바꿉니다")


def file_finder(files) -> Callable[[str], Optional[str]]:
    """``resolve_lora_name`` on one file list, remembered: every cell asks for the same few names."""
    files = list(files)
    known: dict = {}

    def find(name: str) -> Optional[str]:
        if name not in known:
            known[name] = resolve_lora_name(name, files)
        return known[name]

    return find


def check_cell(state: dict, columns: list[Column], find_file: Callable[[str], Optional[str]]) -> None:
    """A finished cell: a tag name loads once with its last strength, and another spelling of one file loads it again, so a
    column's file with more tags than the base had is not what the column's title says, whichever axis brought it in."""
    tags = parse_lora_tags(state["prompt"], state["negative"])[0]
    for column in columns:
        file = find_file(column.name)
        if sum(find_file(tag.name) == file for tag in tags) > column.copies:
            raise LoraAxisError(f"LoRA 축: {column.name} 이 프롬프트에 이미 있습니다")


def _tag_strength(settings: dict, prompt: str, negative: str) -> str:
    wanted = base_lora(settings).casefold()
    tags = collapse_duplicates(parse_lora_tags(prompt, negative)[0])
    return next((_digits(tag.strength_model) for tag in tags if tag.name.casefold() == wanted), "1")


def column_preview(column: Column, settings: dict, prompt: str, negative: str) -> dict:
    """One line of the panel's column list; the panel's prompts give a kept strength its number."""
    return {
        "label": column.label,
        "name": _split(column.name)[1],
        "strength": _digits(column.strength) if column.strength is not None else _tag_strength(settings, prompt, negative),
        "trigger": _trigger_text(settings, column.trigger),
        "changed": _acts(settings, column.trigger),
    }
