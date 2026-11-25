# -*- coding: utf-8 -*-
"""Writes the Anima studio's tag autocomplete list from the app's tag tables.

Output: ``comfy_custom_nodes/ai_studio_forge_parity/tags/danbooru_tags.csv`` (UTF-8, no BOM, header
``name,category,count,aliases,ko``), read by the pack's ``studio_tags.py``.

- The base is ``tags_db/autocomplete/danbooru_autocomplete_catalog.csv`` (``name,category,count,aliases``, no header),
  keeping the tags used at least ``MIN_COUNT`` (20) times, most used first.
- ``ko`` comes from ``tags_db/catalogs/korean_tag_catalog.parquet`` (the table behind the app's Korean search): the
  search words of the tag joined with ``|`` -- plain words first, category markers like ``<헤어스타일>`` last, the same
  order the app uses to pick its label.  Tags the Korean table does not know get an empty ``ko``.

The pack cannot import ``core`` or read parquet, which is why the list is generated.
``tests/test_studio_tags.py`` checks that this script reproduces the file.

Run: ``venv\\Scripts\\python.exe tools\\build_studio_tag_list.py``
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG = ROOT / "tags_db" / "autocomplete" / "danbooru_autocomplete_catalog.csv"
OUTPUT = ROOT / "comfy_custom_nodes" / "ai_studio_forge_parity" / "tags" / "danbooru_tags.csv"
HEADER = ["name", "category", "count", "aliases", "ko"]
MIN_COUNT = 20


def _key(name: object) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def korean_labels(entries: Iterable[tuple[object, object]]) -> dict[str, str]:
    """``{tag: "word|word|..."}`` from ``(tag, keywords)`` pairs of the Korean table (tags without words are left out)."""
    from core.tag_korean_lookup import KoreanTagLookup

    labels: dict[str, str] = {}
    for tag, keywords in entries:
        words = [word.replace("|", " ") for word in KoreanTagLookup._split_keywords(keywords)]
        if _key(tag) and words:
            labels[_key(tag)] = "|".join(words)
    return labels


def read_korean_labels() -> dict[str, str]:
    from core.tag_database import TagAsset, get_tag_database

    frame = get_tag_database().read_parquet(TagAsset.KOREAN_TAG_CATALOG)
    return korean_labels(zip(frame["tag"], frame["keywords"]))


def build_rows(catalog: Iterable[Sequence[str]], korean: Mapping[str, str], min_count: int = MIN_COUNT) -> list[list[str]]:
    """The list's rows: catalog tags with ``count >= min_count`` (most used first, catalog order on ties) plus ``ko``."""
    rows = []
    for row in catalog:
        if len(row) < 3 or not row[0].strip():
            continue
        try:
            count = int(row[2])
        except ValueError:
            continue
        if count < min_count:
            continue
        name = row[0].strip()
        rows.append([name, row[1], str(count), row[3] if len(row) > 3 else "", korean.get(_key(name), "")])
    rows.sort(key=lambda row: -int(row[2]))  # stable: equal counts keep the catalog order
    return rows


def write_tag_list(output: Path = OUTPUT, catalog: Path = CATALOG, korean: Mapping[str, str] | None = None) -> Path:
    if korean is None:
        korean = read_korean_labels()
    with open(catalog, encoding="utf-8", newline="") as handle:
        rows = build_rows(csv.reader(handle), korean)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return output


def main() -> int:
    path = write_tag_list()
    print(f"{path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
