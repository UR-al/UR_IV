"""연관 태그 테스트용 작은 Danbooru 데이터 — parquet 샤드 · manifest · 자동완성 카탈로그 · 가짜 태그 DB.

진짜 데이터(수 GB)는 저장소에 없다. 게시물 묶음(같은 태그 줄을 n 번)을 샤드마다 적어 만든다.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

# (게시물 수, general, character, copyright)
G_POSTS = [
    (100, "1girl solo armor sword holding_sword weapon shoulder_armor", "", ""),
    (100, "1girl solo beach ocean swimsuit bikini smile", "", ""),
    (80, "1girl solo smile long_hair hat", "", ""),
    (40, "1girl solo nurse_cap hat smile", "", ""),
    (50, "1girl solo twintails aqua_hair necktie", "hatsune_miku", "vocaloid"),
    (300, "1girl solo", "", ""),
]
S_POSTS = [
    (100, "1girl solo armor cape sword", "", ""),
    (200, "1girl solo smile", "", ""),
    (400, "1girl solo", "", ""),
]
Q_POSTS = [
    (60, "1girl solo armor bikini_armor", "", ""),
]
# 카탈로그에만 있는 태그 — 작가(1)·메타(5)는 어휘에 들지 않는다.
EXTRA_CATALOG = [("some_artist", 1, 999999), ("highres", 5, 999999)]
CATEGORY = {"hatsune_miku": 4, "vocaloid": 3}
ALIASES = {"longhair": "long_hair", "boobs": "breasts"}
IMPLICATIONS = {"holding_sword": {"sword", "holding_weapon"}, "holding_weapon": {"weapon"},
                "nurse_cap": {"hat"}}
KOREAN = {"twintails": "트윈테일", "aqua_hair": "청록 머리", "sword": "검"}


def _rows(posts):
    rows = []
    for count, general, character, copyright_ in posts:
        rows.extend([{"rating": "g", "general": general, "character": character or None,
                      "copyright": copyright_ or None, "artist": "some_artist", "meta": "highres"}] * count)
    return rows


def write_shard(path: Path, posts, *, row_group_size: int = 128) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _rows(posts)
    columns = ["rating", "general", "character", "copyright", "artist", "meta"]
    table = pa.table({c: pa.array([r[c] for r in rows], type=pa.string()) for c in columns})
    pq.write_table(table, path, row_group_size=row_group_size)
    return len(rows)


def catalog_counts(*shards) -> Counter:
    counts: Counter = Counter()
    for posts in shards:
        for count, general, character, copyright_ in posts:
            for tag in f"{general} {character} {copyright_}".split():
                counts[tag] += count
    return counts


def write_dataset(root: Path, shards: dict | None = None) -> dict:
    """root/danbooru_optimized(샤드·manifest) + root/catalog.csv. 반환: 경로와 manifest."""
    shards = {"g": G_POSTS, "s": S_POSTS, "q": Q_POSTS} if shards is None else shards
    data_dir = root / "danbooru_optimized"
    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for shard, posts in shards.items():
        name = f"danbooru_2026_07_{shard}.parquet"
        rows = write_shard(data_dir / name, posts)
        blob = (data_dir / name).read_bytes()
        artifacts.append({"path": name, "kind": "search", "rating_shard": shard, "rows": rows,
                          "size_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    manifest = {"artifacts": artifacts}
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    counts = catalog_counts(G_POSTS, S_POSTS, Q_POSTS)
    lines = [f"{tag},{CATEGORY.get(tag, 0)},{n},\"\"" for tag, n in counts.most_common()]
    lines += [f"{tag},{cat},{n},\"\"" for tag, cat, n in EXTRA_CATALOG]
    catalog = root / "catalog.csv"
    catalog.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"data_dir": data_dir, "catalog": catalog, "manifest": manifest}


def write_manifest(data_dir: Path, manifest: dict) -> None:
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FakeTagDatabase:
    def __init__(self, catalog: Path, *, fail: bool = False):
        self.catalog = catalog
        self.fail = fail

    def path(self, _asset):
        return self.catalog

    def load_tag_aliases(self):
        if self.fail:
            raise OSError("no aliases")
        return dict(ALIASES)

    def load_active_implications(self):
        if self.fail:
            raise OSError("no implications")
        return {k: set(v) for k, v in IMPLICATIONS.items()}


class _Entry:
    def __init__(self, ko):
        self.ko = ko


class FakeKorean:
    def __init__(self):
        self.asked: list[str] = []

    def label(self, tag):
        self.asked.append(tag)
        return _Entry(KOREAN[tag]) if tag in KOREAN else None
