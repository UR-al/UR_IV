# core/related_tags_build.py
"""연관 태그 표 만들기 — 로컬 Danbooru parquet 에서 태그 동시 출현을 세어 등급(샤드)마다 작은 표로 남긴다.

어휘(vocab): 자동완성 카탈로그(tags_db/autocomplete)에서 일반 태그 N_GENERAL(앞 N_CANDIDATES 개가 추천 후보),
캐릭터 N_CHARACTER, 작품 N_COPYRIGHT 개. 작가·메타 태그는 쓰지 않는다. 별칭(alias → 원래 태그)과
함의(거듭 따라간 결과 중 어휘 안 쌍)도 함께 적어 두어, 질의 때 pandas 없이 쓴다.

표(샤드마다): 게시물 하나의 어휘 안 태그 쌍 중 한쪽이 후보인 쌍을 세고, 태그마다 '더한 값'
P(g|c) − P(g) > 0 이고 횟수 ≥ MIN_PAIR_BUILD 인 후보 이웃을 더한 값 순으로 TOP_K 개만 CSR 로 남긴다.
세는 방법은 이 앱을 위해 새로 짰다(원본 저장소 코드 없음).

파일은 ``*.tmp`` 에 쓴 뒤 ``os.replace`` 로 바꾼다 — 중간에 끊겨도 반쪽 표가 남지 않는다.
numpy·pyarrow 는 함수 안에서만 import 한다(앱 시작 경로가 무거워지지 않게 — tests/test_startup_imports.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from core.related_tags import (
    BATCH_ROWS, FORMAT_VERSION, MAX_TAGS_PER_POST, MIN_PAIR_BUILD, N_CANDIDATES, N_CHARACTER,
    N_COPYRIGHT, N_GENERAL, TOP_K, normalize_tag,
)

TAG_COLUMNS = ("general", "character", "copyright")
CATEGORY_GENERAL, CATEGORY_CHARACTER, CATEGORY_COPYRIGHT = 0, 4, 3
PRUNE_BLOCK = 512
COUNT_SHARE = 0.95       # 진행률 — 세기 95%, 고르기 5%


class BuildCanceled(Exception):
    """사용자가 표 만들기를 취소했다(배치 사이에서 확인)."""


@dataclass(frozen=True)
class Vocab:
    names: tuple[str, ...]
    kind: tuple[int, ...]
    n_candidates: int
    alias_names: tuple[str, ...] = ()
    alias_ids: tuple[int, ...] = ()
    impl_from: tuple[int, ...] = ()
    impl_to: tuple[int, ...] = ()
    fingerprint: str = ""
    source: Mapping[str, object] = field(default_factory=dict)

    def index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.names)}

    def aliases(self) -> dict[str, int]:
        return dict(zip(self.alias_names, self.alias_ids))

    def implications(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for a, b in zip(self.impl_from, self.impl_to):
            out.setdefault(a, []).append(b)
        return out


@dataclass
class ShardTable:
    indptr: object        # int32[V+1]
    nbr: object           # uint16[E] — 후보 id
    cnt: object           # uint32[E] — 쌍 횟수
    occ: object           # int64[V] — 태그마다 게시물 수(건너뛴 게시물 제외)
    posts: int            # 센 게시물 수(건너뛴 것 제외)
    skipped: int
    meta: dict = field(default_factory=dict)


def vocab_params(n_candidates=N_CANDIDATES, n_general=N_GENERAL, n_character=N_CHARACTER,
                 n_copyright=N_COPYRIGHT) -> dict:
    return {"n_candidates": n_candidates, "n_general": n_general, "n_character": n_character,
            "n_copyright": n_copyright}


def read_catalog_rows(path: Path | str) -> Iterator[tuple[str, int, int]]:
    """자동완성 카탈로그(머리글 없음: tag, category, count, aliases) → (tag, category, count)."""
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            try:
                yield row[0], int(row[1]), int(row[2])
            except ValueError:
                continue


def _fingerprint(names: Iterable[str], n_candidates: int) -> str:
    digest = hashlib.sha256()
    digest.update(f"format={FORMAT_VERSION};n_candidates={n_candidates}\n".encode("utf-8"))
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_vocab(catalog_rows: Iterable[tuple[str, int, int]], *, n_candidates: int = N_CANDIDATES,
                 n_general: int = N_GENERAL, n_character: int = N_CHARACTER, n_copyright: int = N_COPYRIGHT,
                 aliases: Mapping[str, str] | None = None,
                 implications: Mapping[str, Iterable[str]] | None = None,
                 source: Mapping[str, object] | None = None) -> Vocab:
    """카탈로그 행 → 어휘. 순서: 후보 일반 → 나머지 기준용 일반 → 캐릭터 → 작품(각각 게시물 수 ↓, 이름 ↑)."""
    groups: dict[int, list[tuple[int, str]]] = {CATEGORY_GENERAL: [], CATEGORY_CHARACTER: [], CATEGORY_COPYRIGHT: []}
    for tag, category, count in catalog_rows:
        key = normalize_tag(tag)
        if key and category in groups:
            groups[category].append((-int(count), key))
    names: list[str] = []
    kinds: list[int] = []
    seen: set[str] = set()
    candidates = 0
    for category, limit in ((CATEGORY_GENERAL, n_general), (CATEGORY_CHARACTER, n_character),
                            (CATEGORY_COPYRIGHT, n_copyright)):
        taken = 0
        for _neg, key in sorted(groups[category]):
            if taken >= limit:
                break
            if key in seen:
                continue
            seen.add(key)
            names.append(key)
            kinds.append(category)
            taken += 1
        if category == CATEGORY_GENERAL:
            candidates = min(n_candidates, taken)
    if candidates > 0xFFFF:
        raise ValueError("후보 태그가 너무 많습니다(최대 65535).")
    index = {name: i for i, name in enumerate(names)}

    alias_pairs: dict[str, int] = {}
    for alias, canonical in (aliases or {}).items():
        a, c = normalize_tag(alias), normalize_tag(canonical)
        if a and a not in index and c in index:
            alias_pairs[a] = index[c]

    graph: dict[str, set[str]] = {}
    for antecedent, consequents in (implications or {}).items():
        a = normalize_tag(antecedent)
        if a:
            graph.setdefault(a, set()).update(normalize_tag(c) for c in consequents if normalize_tag(c))
    impl_from: list[int] = []
    impl_to: list[int] = []
    for name, tag_id in index.items():
        if name not in graph:
            continue
        reached: set[str] = set()
        stack = list(graph[name])
        while stack:
            current = stack.pop()
            if current in reached:
                continue
            reached.add(current)
            stack.extend(graph.get(current, ()))
        for implied in sorted(reached):
            target = index.get(implied)
            if target is not None and target != tag_id:
                impl_from.append(tag_id)
                impl_to.append(target)

    alias_sorted = sorted(alias_pairs.items())
    return Vocab(
        names=tuple(names), kind=tuple(kinds), n_candidates=candidates,
        alias_names=tuple(a for a, _ in alias_sorted), alias_ids=tuple(i for _, i in alias_sorted),
        impl_from=tuple(impl_from), impl_to=tuple(impl_to),
        fingerprint=_fingerprint(names, candidates), source=dict(source or {}),
    )


# ── 파일 ────────────────────────────────────────────────────────────────────

def _atomic_savez(path: Path | str, **arrays) -> None:
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _meta_array(meta: Mapping[str, object]):
    import numpy as np

    return np.array(json.dumps(dict(meta), ensure_ascii=False, sort_keys=True))


def _read_meta(archive) -> dict | None:
    try:
        meta = json.loads(str(archive["meta"][()]))
    except (KeyError, ValueError, TypeError):
        return None
    return meta if isinstance(meta, dict) else None


def save_vocab(path: Path | str, vocab: Vocab) -> None:
    import numpy as np

    meta = {"format": FORMAT_VERSION, "kind": "vocab", "fingerprint": vocab.fingerprint,
            "n_candidates": vocab.n_candidates, "source": dict(vocab.source)}
    _atomic_savez(
        path, meta=_meta_array(meta),
        names=np.array(vocab.names, dtype=str), kind=np.array(vocab.kind, dtype=np.int8),
        alias_names=np.array(vocab.alias_names, dtype=str), alias_ids=np.array(vocab.alias_ids, dtype=np.int32),
        impl_from=np.array(vocab.impl_from, dtype=np.int32), impl_to=np.array(vocab.impl_to, dtype=np.int32),
    )


def load_vocab(path: Path | str) -> Vocab | None:
    """어휘 파일 → Vocab. 없거나 깨졌거나 형식이 다르면 None."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = _read_meta(archive)
            if not meta or meta.get("format") != FORMAT_VERSION or meta.get("kind") != "vocab":
                return None
            names = tuple(str(n) for n in archive["names"].tolist())
            vocab = Vocab(
                names=names, kind=tuple(int(k) for k in archive["kind"].tolist()),
                n_candidates=int(meta.get("n_candidates", 0)),
                alias_names=tuple(str(n) for n in archive["alias_names"].tolist()),
                alias_ids=tuple(int(i) for i in archive["alias_ids"].tolist()),
                impl_from=tuple(int(i) for i in archive["impl_from"].tolist()),
                impl_to=tuple(int(i) for i in archive["impl_to"].tolist()),
                fingerprint=str(meta.get("fingerprint", "")),
                source=meta.get("source") if isinstance(meta.get("source"), dict) else {},
            )
    except (OSError, ValueError, KeyError, EOFError):
        return None
    if vocab.fingerprint != _fingerprint(vocab.names, vocab.n_candidates) or len(vocab.kind) != len(names):
        return None
    return vocab


def save_shard(path: Path | str, table: ShardTable) -> None:
    import numpy as np

    _atomic_savez(
        path, meta=_meta_array(table.meta),
        indptr=np.asarray(table.indptr, dtype=np.int32), nbr=np.asarray(table.nbr, dtype=np.uint16),
        cnt=np.asarray(table.cnt, dtype=np.uint32), occ=np.asarray(table.occ, dtype=np.int64),
    )


def read_shard_meta(path: Path | str) -> dict | None:
    """샤드 표의 meta 만(배열은 읽지 않는다). 없거나 깨졌으면 None."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            return _read_meta(archive)
    except (OSError, ValueError, KeyError, EOFError):
        return None


def load_shard(path: Path | str) -> ShardTable | None:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = _read_meta(archive)
            if not meta:
                return None
            table = ShardTable(
                indptr=archive["indptr"].astype(np.int64), nbr=archive["nbr"], cnt=archive["cnt"],
                occ=archive["occ"].astype(np.int64), posts=int(meta.get("posts", 0)),
                skipped=int(meta.get("skipped", 0)), meta=meta,
            )
    except (OSError, ValueError, KeyError, EOFError):
        return None
    if len(table.indptr) != len(table.occ) + 1 or len(table.nbr) != len(table.cnt):
        return None
    return table


# ── 세기 ────────────────────────────────────────────────────────────────────

def build_shard_table(parquet_path: Path | str, vocab: Vocab, *, batch_rows: int = BATCH_ROWS,
                      max_tags: int = MAX_TAGS_PER_POST, min_pair: int = MIN_PAIR_BUILD, top_k: int = TOP_K,
                      progress: Callable[[float], None] | None = None,
                      should_stop: Callable[[], bool] | None = None) -> ShardTable:
    """parquet 하나 → 샤드 표. should_stop() 이 참이면 배치 사이에서 BuildCanceled."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    names = list(vocab.names)
    V = len(names)
    nc = int(vocab.n_candidates)
    if V == 0 or nc == 0:
        raise ValueError("연관 태그 어휘가 비어 있습니다.")
    value_set = pa.array(names, type=pa.string())
    key_dtype = np.uint32 if V * nc < 2 ** 32 else np.uint64
    acc = np.zeros((V, nc), dtype=np.uint32)     # acc[S, F] = (F < S, F 는 후보) 쌍 횟수
    flat = acc.reshape(-1)
    occ = np.zeros(V, dtype=np.int64)
    triu: dict[int, tuple] = {}

    def stop() -> None:
        if should_stop is not None and should_stop():
            raise BuildCanceled()

    parquet = pq.ParquetFile(parquet_path)
    columns = parquet.schema_arrow.names
    missing = [c for c in TAG_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"Danbooru 데이터에 {', '.join(missing)} 열이 없습니다.")
    total_rows = max(1, parquet.metadata.num_rows)
    rows_seen = 0
    skipped = 0
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(TAG_COLUMNS)):
        stop()
        n = batch.num_rows
        rows_seen += n
        texts = [pc.fill_null(batch.column(c), "") for c in TAG_COLUMNS]
        joined = pc.binary_join_element_wise(*texts, " ")
        split = pc.split_pattern(joined, " ")
        lengths = pc.list_value_length(split).to_numpy(zero_copy_only=False).astype(np.int64)
        ids = pc.fill_null(pc.index_in(pc.list_flatten(split), value_set=value_set), -1)
        ids = ids.to_numpy(zero_copy_only=False).astype(np.int64)
        post = np.repeat(np.arange(n, dtype=np.int64), lengths)
        keep = ids >= 0
        tid = ids[keep].astype(np.int32)                # 게시물 순서대로 이어진 어휘 안 태그
        if progress is not None:
            progress(COUNT_SHARE * rows_seen / total_rows)
        if not tid.size:
            continue
        plen = np.bincount(post[keep], minlength=n)     # 게시물마다 어휘 안 태그 수
        starts = np.zeros(n, dtype=np.int64)
        np.cumsum(plen[:-1], out=starts[1:])
        skipped += int((plen > max_tags).sum())
        chunks = []
        for k in np.unique(plen).tolist():
            if k < 1 or k > max_tags:
                continue
            group_starts = starts[plen == k]
            block = tid[group_starts[:, None] + np.arange(k)]
            block.sort(axis=1)
            if k > 1:
                # 한 게시물에 같은 태그가 두 번이면 뒤의 것을 어휘 밖 표지(V)로 — 세지도 짝짓지도 않는다
                tail = block[:, 1:]
                tail[tail == block[:, :-1]] = V
            occ += np.bincount(block[block < V], minlength=V)
            if k < 2:
                continue
            if k not in triu:
                triu[k] = tuple(a.astype(np.intp) for a in np.triu_indices(k, 1))
            first_idx, second_idx = triu[k]
            first = block[:, first_idx]
            second = block[:, second_idx]
            mask = (first < nc) & (second < V)
            chunks.append(second[mask].astype(key_dtype) * key_dtype(nc) + first[mask].astype(key_dtype))
        if not chunks:
            continue
        pair_keys = np.concatenate(chunks)
        del chunks
        pair_keys.sort()
        breaks = np.flatnonzero(np.r_[True, pair_keys[1:] != pair_keys[:-1]])
        flat[pair_keys[breaks]] += np.diff(np.r_[breaks, pair_keys.size]).astype(np.uint32)
        del pair_keys, breaks

    posts = rows_seen - skipped
    stop()
    indptr, nbr, cnt = _prune(acc, occ, posts, nc, min_pair=min_pair, top_k=top_k,
                              progress=progress, stop=stop)
    del acc, flat
    if progress is not None:
        progress(1.0)
    return ShardTable(indptr=indptr, nbr=nbr, cnt=cnt, occ=occ, posts=int(posts), skipped=int(skipped),
                      meta={"posts": int(posts), "skipped": int(skipped), "entries": int(len(nbr))})


def _prune(acc, occ, posts: int, nc: int, *, min_pair: int, top_k: int, progress, stop):
    """acc[S, F] → 태그마다 후보 이웃(더한 값 > 0, 횟수 ≥ min_pair) 상위 top_k 의 CSR."""
    import numpy as np

    V = acc.shape[0]
    total = float(posts)
    prior = occ[:nc].astype(np.float64) / total if total > 0 else np.zeros(nc)
    sizes = np.zeros(V, dtype=np.int64)
    nbr_parts: list = []
    cnt_parts: list = []
    for begin in range(0, V, PRUNE_BLOCK):
        stop()
        end = min(V, begin + PRUNE_BLOCK)
        rows = acc[begin:end].astype(np.int64)
        if begin < nc:
            cand_end = min(end, nc)
            rows[:cand_end - begin] += acc[:nc, begin:cand_end].T   # (c, g) 쌍이 g 행에 있던 몫
            idx = np.arange(begin, cand_end)
            rows[idx - begin, idx] = 0                                # 자기 자신 제외
        base = occ[begin:end].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            added = rows / base[:, None] - prior[None, :]
        ok = (rows >= min_pair) & (added > 0) & (base[:, None] > 0)
        for r in range(end - begin):
            chosen = np.flatnonzero(ok[r])
            if chosen.size > top_k:
                chosen = chosen[np.argsort(-added[r, chosen], kind="stable")[:top_k]]
            elif chosen.size:
                chosen = chosen[np.argsort(-added[r, chosen], kind="stable")]
            sizes[begin + r] = chosen.size
            nbr_parts.append(chosen.astype(np.uint16))
            cnt_parts.append(rows[r, chosen].astype(np.uint32))
        if progress is not None:
            progress(min(1.0, COUNT_SHARE + (1 - COUNT_SHARE) * end / V))
    indptr = np.zeros(V + 1, dtype=np.int32)
    np.cumsum(sizes, out=indptr[1:])
    nbr = np.concatenate(nbr_parts) if nbr_parts else np.zeros(0, np.uint16)
    cnt = np.concatenate(cnt_parts) if cnt_parts else np.zeros(0, np.uint32)
    return indptr, nbr, cnt


__all__ = [
    "BuildCanceled", "ShardTable", "TAG_COLUMNS", "Vocab",
    "build_shard_table", "load_shard", "load_vocab", "read_catalog_rows", "read_shard_meta",
    "save_shard", "save_vocab", "select_vocab", "vocab_params",
]
