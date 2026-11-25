"""Prompt-box presets of the Anima studio: one JSON file in ComfyUI's user folder.

``<user_dir>/default/ai_studio/prompt_presets.json`` (the folder of ``lora_store.store_dir``)::

    {"version": 1,
     "sections": {"<box>": [{"id", "name", "text", "created", "used"}, ...]},   # the user's order
     "prefs": {"close_after_insert": false}}

``<box>`` is one of ``SECTIONS`` (the prompt card's boxes plus ``negative``); ``created`` / ``used`` are Unix seconds
(``used`` 0 = never put into a box).  The file starts empty: nothing is seeded.

Every change is one ``apply(dir, op)``: read the file, change it, write it whole through ``lora_atomic`` (a unique temp
file, then ``os.replace``), all under one lock - two browser tabs never see or leave a partial file, and a change made in
one tab is not lost when the other saves (each change is applied to the file as it is then; for the same preset the last
write wins).  That is why no change carries a whole list: taking a change back is its reverse (``delete`` the added one,
``insert`` the deleted one where it was, ``update`` / ``move`` back).

A file that cannot be read in full (broken JSON, entries that break the rules, a newer ``version``) is read as far as it
goes, and the bytes that were read are copied to ``prompt_presets.json.bak`` (``.bak1`` ... for a different one) before
the first write over it; reading alone never writes.  A UTF-8 BOM in front (Windows PowerShell 5.1, old Notepad) is
skipped.  A file that is there but cannot be opened (a sharing violation, access denied) is *not* an empty one: reading
raises ``OSError`` and nothing is written over it, and neither is a damaged file whose backup cannot be written.

No Qt and no ComfyUI imports; the caller passes the store folder.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from .lora_atomic import atomic_path
from .studio_state import PROMPT_SECTIONS

log = logging.getLogger(__name__)

PRESET_FILE = "prompt_presets.json"
VERSION = 1
SECTIONS = (*PROMPT_SECTIONS, "negative")
EDGE = " \t\r\n,"  # studio_state._PROMPT_EDGE: what a joined prompt box loses at its ends
NAME_MAX = 60
TEXT_MAX = 20000
MAX_PER_SECTION = 200
MAX_BACKUPS = 20
PREF_DEFAULTS = {"close_after_insert": False}
_ID_MAX = 64

_LOCK = threading.Lock()  # one reader or writer of the file at a time (Windows cannot replace a file that is open)


class PresetError(ValueError):
    """A change the store refuses; ``status`` is the HTTP status of the route's answer."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def limits() -> dict:
    return {"name": NAME_MAX, "text": TEXT_MAX, "count": MAX_PER_SECTION}


def empty_doc() -> dict:
    return {"version": VERSION, "sections": {key: [] for key in SECTIONS}, "prefs": dict(PREF_DEFAULTS)}


# ---------------------------------------------------------------------------------------------- rules (JS twin: preset_logic.js)

def clean_name(name) -> str:
    """The name as it is kept: runs of white space (line breaks too) become one space, the ends are cut."""
    if not isinstance(name, str):
        raise PresetError("이름은 글자여야 해요")
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        raise PresetError("이름을 적어 주세요")
    if len(name) > NAME_MAX:
        raise PresetError(f"이름은 {NAME_MAX}자까지예요")
    return name


def clean_text(text) -> str:
    """The text exactly as written; refused when nothing is left without its edges, or when it is too long."""
    if not isinstance(text, str):
        raise PresetError("내용은 글자여야 해요")
    if not text.strip(EDGE):
        raise PresetError("내용이 비어 있어요")
    if len(text) > TEXT_MAX:
        raise PresetError(f"내용은 {TEXT_MAX}자까지예요")
    return text


def match_preset(items, text):
    """The first preset whose text equals ``text`` once both lose their edges (as a joined prompt box does); an empty
    box matches nothing."""
    want = text.strip(EDGE) if isinstance(text, str) else ""
    if not want or not isinstance(items, list):
        return None
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip(EDGE) == want:
            return item
    return None


# ---------------------------------------------------------------------------------------------- reading

def _new_id(taken) -> str:
    while True:
        value = uuid.uuid4().hex[:12]
        if value not in taken:
            return value


def _stamp(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _valid_id(value) -> bool:
    return isinstance(value, str) and 0 < len(value) <= _ID_MAX


def _read_items(raw) -> list:
    """What can be kept of a stored list: entries with a name and a text, each with its own id."""
    items, taken = [], set()
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            name, text = clean_name(entry.get("name")), clean_text(entry.get("text"))
        except PresetError:
            continue
        ident = entry.get("id")
        if not _valid_id(ident) or ident in taken:
            ident = _new_id(taken)
        taken.add(ident)
        items.append({"id": ident, "name": name, "text": text,
                      "created": _stamp(entry.get("created")), "used": _stamp(entry.get("used"))})
        if len(items) >= MAX_PER_SECTION:
            break
    return items


def _read_prefs(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {key: raw[key] if isinstance(raw.get(key), bool) else default for key, default in PREF_DEFAULTS.items()}


def _normalise(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    sections = raw.get("sections") if isinstance(raw.get("sections"), dict) else {}
    return {"version": VERSION, "sections": {key: _read_items(sections.get(key)) for key in SECTIONS},
            "prefs": _read_prefs(raw.get("prefs"))}


def _read(path: Path):
    """``(doc, whole, broken, data)``: ``whole`` is False when a file is there that is not exactly what this module
    writes, ``broken`` when it cannot be parsed at all (``doc`` is then empty), ``data`` its bytes (None: no file).
    Raises ``OSError`` when the file is there but cannot be opened: that is not an empty store."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return empty_doc(), True, False, None
    try:
        raw = json.loads(data.decode("utf-8-sig"))  # -sig: a BOM in front is not part of the document
    except ValueError as exc:  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
        log.warning("prompt presets file is broken (%s): %s", path, exc)
        return empty_doc(), False, True, data
    doc = _normalise(raw)
    return doc, raw == doc, not isinstance(raw, dict), data


def load_state(dir):
    """``(doc, broken)``: the stored presets, and whether a file is there that could not be parsed (it then reads as
    empty and is kept as ``.bak`` by the first change).  A missing file is empty and not broken.  Never writes; raises
    ``OSError`` when the file cannot be opened."""
    with _LOCK:
        doc, _whole, broken, _data = _read(Path(dir) / PRESET_FILE)
    return doc, broken


def load(dir) -> dict:
    """The stored presets (``load_state`` without the flag)."""
    return load_state(dir)[0]


# ---------------------------------------------------------------------------------------------- writing

def _backup(path: Path, data: bytes) -> None:
    """Keep ``data`` - the bytes of a file that was not read whole - next to it (``.bak``, ``.bak1`` ...; the same
    bytes are not kept twice).  Raises ``OSError`` when they cannot be written: the caller then leaves the file alone."""
    target = None
    for number in range(MAX_BACKUPS):
        candidate = path.with_name(f"{path.name}.bak{number or ''}")
        try:
            if candidate.read_bytes() == data:
                return
        except FileNotFoundError:
            target = candidate
            break
        except OSError:
            continue
    target = target or path.with_name(f"{path.name}.bak")  # all taken: the oldest gives way
    with atomic_path(target) as tmp:
        tmp.write_bytes(data)
    log.warning("prompt presets: the unreadable file was kept as %s", target.name)


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(doc, ensure_ascii=False, indent=2)
    with atomic_path(path) as tmp:
        tmp.write_text(data, encoding="utf-8")


def _section(op: dict) -> str:
    key = op.get("section")
    if not isinstance(key, str) or key not in SECTIONS:
        raise PresetError("unknown prompt box")
    return key


def _index(items: list, ident) -> int:
    if not _valid_id(ident):
        raise PresetError("id is required")
    for index, item in enumerate(items):
        if item["id"] == ident:
            return index
    return -1


def _found(items: list, ident) -> int:
    index = _index(items, ident)
    if index < 0:
        raise PresetError("프리셋을 찾지 못했어요 (다른 창에서 지웠을 수 있어요)", 404)
    return index


def _op_add(doc, op, now):
    items = doc["sections"][_section(op)]
    name, text = clean_name(op.get("name")), clean_text(op.get("text"))
    if len(items) >= MAX_PER_SECTION:
        raise PresetError(f"한 칸에 프리셋은 {MAX_PER_SECTION}개까지예요")
    ident = _new_id({item["id"] for item in items})
    items.append({"id": ident, "name": name, "text": text, "created": now, "used": 0})
    return {"id": ident}


def _op_update(doc, op, now):
    items = doc["sections"][_section(op)]
    if "name" not in op and "text" not in op:
        raise PresetError("name or text is required")
    changes = {}
    if "name" in op:
        changes["name"] = clean_name(op["name"])
    if "text" in op:
        changes["text"] = clean_text(op["text"])
    items[_found(items, op.get("id"))].update(changes)
    return {}


def _op_delete(doc, op, now):
    items = doc["sections"][_section(op)]
    index = _index(items, op.get("id"))
    if index >= 0:  # already gone (another tab): nothing to do
        del items[index]
    return {}


def _op_move(doc, op, now):
    items = doc["sections"][_section(op)]
    delta = op.get("delta")
    if isinstance(delta, bool) or not isinstance(delta, int) or delta == 0:
        raise PresetError("delta must be a whole number other than 0")
    index = _found(items, op.get("id"))
    items.insert(max(0, min(len(items) - 1, index + delta)), items.pop(index))
    return {}


def _op_use(doc, op, now):
    items = doc["sections"][_section(op)]
    index = _index(items, op.get("id"))
    if index >= 0:
        items[index]["used"] = now
    return {}


def _op_insert(doc, op, now):
    """A deleted preset back where it was (the window's 되돌리기): a complete, valid preset at ``index`` (kept inside
    the list).  A preset with that id is listed already (another tab put it back): nothing to do."""
    items = doc["sections"][_section(op)]
    entry, index = op.get("item"), op.get("index")
    if not isinstance(entry, dict) or not _valid_id(entry.get("id")):
        raise PresetError("item must be a preset with its id")
    if isinstance(index, bool) or not isinstance(index, int):
        raise PresetError("index must be a whole number")
    item = {"id": entry["id"], "name": clean_name(entry.get("name")), "text": clean_text(entry.get("text")),
            "created": _stamp(entry.get("created")), "used": _stamp(entry.get("used"))}
    if _index(items, item["id"]) >= 0:
        return {}
    if len(items) >= MAX_PER_SECTION:
        raise PresetError(f"한 칸에 프리셋은 {MAX_PER_SECTION}개까지예요")
    items.insert(max(0, min(len(items), index)), item)
    return {}


def _op_prefs(doc, op, now):
    prefs = op.get("prefs")
    if not isinstance(prefs, dict):
        raise PresetError("prefs must be an object")
    for key in PREF_DEFAULTS:
        if isinstance(prefs.get(key), bool):
            doc["prefs"][key] = prefs[key]
    return {}


_OPS = {"add": _op_add, "update": _op_update, "delete": _op_delete, "move": _op_move, "use": _op_use,
        "insert": _op_insert, "prefs": _op_prefs}


def apply(dir, op, *, now=None) -> dict:
    """Apply one change and save: ``{"doc": the new document}`` plus ``"id"`` for ``add``.

    ``op`` (``section`` = a ``SECTIONS`` key):
    ``{"op": "add", section, name, text}`` - a new preset at the end;
    ``{"op": "update", section, id, name?, text?}``; ``{"op": "delete", section, id}`` (a missing id is fine);
    ``{"op": "move", section, id, delta}`` - ``delta`` places up (< 0) or down, stopping at the ends;
    ``{"op": "use", section, id}`` - stamps ``used``;
    ``{"op": "insert", section, item: {id, name, text, created, used}, index}`` - a deleted preset back at ``index``
    (a listed id is left alone); ``{"op": "prefs", prefs: {close_after_insert}}``.
    A refused change raises ``PresetError`` and writes nothing; so does ``OSError`` - the file cannot be opened, or it
    is damaged and its backup cannot be written.
    """
    handler = _OPS.get(op.get("op")) if isinstance(op, dict) else None
    if handler is None:
        raise PresetError("op must be one of: " + ", ".join(_OPS))
    stamp = int(time.time() if now is None else now)
    path = Path(dir) / PRESET_FILE
    with _LOCK:
        doc, whole, _broken, data = _read(path)
        extra = handler(doc, op, stamp)
        if not whole:
            _backup(path, data)
        _write(path, doc)
    return {"doc": doc, **extra}
