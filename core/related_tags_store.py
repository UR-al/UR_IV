# core/related_tags_store.py
"""연관 태그 표의 캐시 파일과 상태 — 어디에 있고, 쓸 수 있는가, 만들면 얼마나 걸리나.

캐시: ``cache/related_tags/v<FORMAT_VERSION>/`` (gitignore) — vocab.npz · shard_{g,s,q,e}.npz.
표는 사용자 PC 의 Danbooru 데이터에서 만든 파생물이라 저장소에 넣거나 배포하지 않는다.

샤드 상태:
  ready       — 표가 있고 지금 어휘·형식과 맞고, 만든 parquet 이 manifest 와 같다.
  stale       — 표는 쓸 수 있지만 manifest 의 sha256·크기가 달라졌다(새 릴리스) — 그대로 쓰고 안내만.
  missing     — 표가 없다(또는 어휘·형식이 다르다). parquet 이 있어 만들 수 있다.
  unavailable — 표도 없고 만들 parquet 도 없다(manifest 에 없거나 파일이 없거나 크기가 다르다).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from core.dataset_artifacts import manifest_artifacts, stale_artifacts
from core.related_tags import FORMAT_VERSION, SEC_PER_MILLION_POSTS, SHARDS

READY, STALE, MISSING, UNAVAILABLE = "ready", "stale", "missing", "unavailable"
USABLE_STATES = frozenset({READY, STALE})


@dataclass(frozen=True)
class CachePaths:
    root: Path
    vocab: Path
    shards: Mapping[str, Path]


def cache_paths(storage=None) -> CachePaths:
    """캐시 파일 경로(폴더는 만든다). storage = core.storage_paths.StoragePaths(테스트는 임시 루트)."""
    if storage is None:
        from core.storage_paths import StoragePaths

        storage = StoragePaths()
    folder = f"related_tags/v{FORMAT_VERSION}"
    vocab = Path(storage.cache_file(f"{folder}/vocab.npz"))
    shards = {s: Path(storage.cache_file(f"{folder}/shard_{s}.npz")) for s in SHARDS}
    return CachePaths(root=vocab.parent, vocab=vocab, shards=shards)


def shard_artifact(manifest: dict | None, shard: str) -> tuple[str, dict] | None:
    """manifest 의 검색용 parquet 중 그 등급(rating_shard) 것 — (상대 경로, artifact)."""
    for relpath, artifact in manifest_artifacts(manifest, kind="search"):
        if str(artifact.get("rating_shard", "")).lower() == shard:
            return relpath, artifact
    return None


def artifact_identity(artifact: Mapping[str, object]) -> dict:
    """표가 어떤 parquet 에서 나왔나 — meta 에 적고 stale 판정에 쓴다."""
    return {"sha256": artifact.get("sha256"), "size_bytes": artifact.get("size_bytes"),
            "rows": artifact.get("rows")}


def shard_state(meta: Mapping[str, object] | None, manifest: dict | None, data_dir: Path | str,
                shard: str, vocab_fingerprint: str | None) -> str:
    found = shard_artifact(manifest, shard)
    usable = (isinstance(meta, Mapping) and meta.get("format") == FORMAT_VERSION
              and meta.get("shard") == shard and bool(vocab_fingerprint)
              and meta.get("vocab") == vocab_fingerprint)
    if usable:
        if found is None:
            return STALE
        identity = artifact_identity(found[1])
        same = all(meta.get(key) == identity[key] for key in ("sha256", "size_bytes"))
        return READY if same else STALE
    if found is None or found[0] in stale_artifacts(manifest, data_dir, kind="search"):
        return UNAVAILABLE
    return MISSING


def shard_states(manifest: dict | None, data_dir: Path | str, paths: CachePaths, vocab_fingerprint: str | None,
                 shards: Iterable[str] = SHARDS, *, read_meta=None) -> dict[str, str]:
    if read_meta is None:
        from core.related_tags_build import read_shard_meta as read_meta
    out: dict[str, str] = {}
    for shard in shards:
        path = paths.shards[shard]
        meta = read_meta(path) if vocab_fingerprint and path.is_file() else None
        out[shard] = shard_state(meta, manifest, data_dir, shard, vocab_fingerprint)
    return out


def estimate_seconds(manifest: dict | None, shards: Iterable[str]) -> int:
    """만들 샤드들의 대략 시간(초) — 게시물 100만 개에 SEC_PER_MILLION_POSTS 초(실측 바탕)."""
    rows = 0
    for shard in shards:
        found = shard_artifact(manifest, shard)
        value = found[1].get("rows") if found else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            rows += value
    return int(round(rows / 1_000_000 * SEC_PER_MILLION_POSTS)) if rows else 0


class LoadedTables:
    """한 등급 조합의 표를 합쳐 본다 — occ·게시물 수는 합, 이웃은 쌍 횟수를 합친다."""

    def __init__(self, tables: Iterable[object]):
        self.tables = list(tables)
        if not self.tables:
            raise ValueError("합칠 표가 없습니다.")
        total = None
        posts = 0
        for table in self.tables:
            occ = table.occ
            total = occ.copy() if total is None else total + occ
            posts += int(table.posts)
        self.occ = [int(v) for v in total.tolist()]
        self.total_posts = posts

    def neighbors(self, tag_id: int) -> dict[int, int]:
        merged: dict[int, int] = {}
        for table in self.tables:
            if not (0 <= tag_id < len(table.indptr) - 1):
                continue
            begin, end = int(table.indptr[tag_id]), int(table.indptr[tag_id + 1])
            for g, n in zip(table.nbr[begin:end].tolist(), table.cnt[begin:end].tolist()):
                merged[g] = merged.get(g, 0) + n
        return merged


__all__ = [
    "MISSING", "READY", "STALE", "UNAVAILABLE", "USABLE_STATES",
    "CachePaths", "LoadedTables",
    "artifact_identity", "cache_paths", "estimate_seconds", "shard_artifact", "shard_state", "shard_states",
]
