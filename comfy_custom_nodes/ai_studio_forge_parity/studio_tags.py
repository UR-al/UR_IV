"""Prompt tag search for the Anima studio's autocomplete (pure: no ComfyUI, no torch).

The list is ``tags/danbooru_tags.csv`` (``name,category,count,aliases,ko``), written by ``tools/build_studio_tag_list.py``
from the app's Danbooru catalog (tags used at least 20 times) and Korean tag table.  It is read once, on the first
search, and then lives in memory.  ``category``: 0 general, 1 artist, 3 copyright, 4 character, 5 meta.  ``aliases`` are
comma-joined; ``ko`` holds Korean search words joined with ``|`` (the first one is the label shown in the popup).

``search_tags(query, limit)`` answers ``[{name, category, count, alias?, ko?}]``.  Case, ``_`` and runs of spaces do not
matter ("Long Hair" = "long_hair").  Order: tags whose name or alias starts with the query, then tags whose name merely
contains it; each group by use count, highest first.  ``alias`` names the alias that matched when the name did not.  A
query with Hangul searches the Korean words the same way (prefix before contains).
"""
from __future__ import annotations

import bisect
import csv
import heapq
import re
import threading
from pathlib import Path
from typing import Iterable, Sequence

TAGS_FILE = Path(__file__).resolve().parent / "tags" / "danbooru_tags.csv"
DEFAULT_LIMIT = 12

_HANGUL = re.compile(r"[ㄱ-ㆎ가-힣]")
_SPACES = re.compile(r"\s+")
_HIGH = "\U0010ffff"  # sorts after any key that starts with the query


def _normalize(text: str) -> str:
    """Lower case, ``_`` as a space, one space per run; leading spaces dropped (a trailing one is kept: "long " != "longing")."""
    return _SPACES.sub(" ", str(text).lstrip().lower().replace("_", " "))


def _whole(row: Sequence[str], index: int) -> str:
    return row[index] if len(row) > index else ""


class TagIndex:
    """All tags of one list, searchable by prefix (bisect) and by substring (scan)."""

    def __init__(self, rows: Iterable[Sequence[str]]):
        self._names: list[str] = []
        self._categories: list[int] = []
        self._counts: list[int] = []
        self._ko: list[str] = []  # the "|" joined Korean words, "" when none
        self._keys: list[str] = []  # normalized names, parallel to _names
        prefix_keys: list[tuple[str, int, str]] = []  # (normalized name or alias, tag, the alias or "")
        korean_keys: list[tuple[str, int]] = []
        for row in rows:
            name = _whole(row, 0).strip()
            try:
                category, count = int(_whole(row, 1)), int(_whole(row, 2))
            except ValueError:
                continue
            if not name:
                continue
            tag = len(self._names)
            ko = _whole(row, 4)
            self._names.append(name)
            self._categories.append(category)
            self._counts.append(count)
            self._ko.append(ko)
            key = _normalize(name)
            self._keys.append(key)
            prefix_keys.append((key, tag, ""))
            for alias in _whole(row, 3).split(","):
                if alias.strip():
                    prefix_keys.append((_normalize(alias), tag, alias.strip()))
            for word in ko.split("|"):
                if word.strip():
                    korean_keys.append((word.strip().lower(), tag))
        prefix_keys.sort()
        korean_keys.sort()
        self._prefix_keys = [item[0] for item in prefix_keys]
        self._prefix_owner = [(item[1], item[2]) for item in prefix_keys]
        self._korean_keys = [item[0] for item in korean_keys]
        self._korean_owner = [item[1] for item in korean_keys]

    @classmethod
    def from_csv(cls, path: Path | str) -> "TagIndex":
        """The list in ``path`` (a header row is skipped); a missing file gives an empty index."""
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                rows = [row for row in csv.reader(handle) if row and row[0] != "name"]
        except OSError:
            rows = []
        return cls(rows)

    def __len__(self) -> int:
        return len(self._names)

    def _rank(self, tags: Iterable[int], limit: int) -> list[int]:
        return heapq.nsmallest(limit, tags, key=lambda tag: (-self._counts[tag], self._names[tag]))

    def _item(self, tag: int, alias: str = "") -> dict:
        item = {"name": self._names[tag], "category": self._categories[tag], "count": self._counts[tag]}
        if alias:
            item["alias"] = alias
        label = self._ko[tag].split("|", 1)[0].strip()
        if label:
            item["ko"] = label
        return item

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
        text = _normalize(query)
        if not text.strip() or limit < 1:
            return []
        if _HANGUL.search(text):
            return self._search_korean(text.strip(), limit)
        return self._search_english(text, limit)

    def _search_english(self, text: str, limit: int) -> list[dict]:
        first = bisect.bisect_left(self._prefix_keys, text)
        last = bisect.bisect_left(self._prefix_keys, text + _HIGH, first)
        matched: dict[int, str] = {}  # tag -> the alias that matched ("" = its name)
        for position in range(first, last):
            tag, alias = self._prefix_owner[position]
            if tag not in matched or not alias:
                matched[tag] = alias
        results = [self._item(tag, matched[tag]) for tag in self._rank(matched, limit)]
        if len(results) < limit:
            contains = (tag for tag, key in enumerate(self._keys) if text in key and tag not in matched)
            results += [self._item(tag) for tag in self._rank(contains, limit - len(results))]
        return results

    def _search_korean(self, text: str, limit: int) -> list[dict]:
        first = bisect.bisect_left(self._korean_keys, text)
        last = bisect.bisect_left(self._korean_keys, text + _HIGH, first)
        prefixed = {self._korean_owner[position] for position in range(first, last)}
        results = [self._item(tag) for tag in self._rank(prefixed, limit)]
        if len(results) < limit:
            contains = {
                self._korean_owner[position]
                for position, key in enumerate(self._korean_keys)
                if text in key and self._korean_owner[position] not in prefixed
            }
            results += [self._item(tag) for tag in self._rank(contains, limit - len(results))]
        return results


_default: TagIndex | None = None
_default_lock = threading.Lock()


def _default_index() -> TagIndex:
    global _default
    index = _default
    if index is None:
        with _default_lock:
            if _default is None:
                _default = TagIndex.from_csv(TAGS_FILE)
            index = _default
    return index


def search_tags(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    """Up to ``limit`` tags for ``query`` from the packed list (read on first use)."""
    return _default_index().search(query, limit)
