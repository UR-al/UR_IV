#!/usr/bin/env python3
"""Reproducibly refresh the application's Danbooru post datasets.

``posts`` streams a local snapshot into four Search shards and four
self-contained Event graph shards.  An Event shard contains the children of
the selected rating plus every ancestor they reference, even when an ancestor
has a different rating.  Tag catalogs and official tag relations are refreshed
by the companion ``refresh_danbooru_tag_assets.py`` tool.

Every pipeline writes into a same-volume staging directory, validates hashes,
schemas, and row counts, then replaces destination files with ``os.replace``.
The manifest is committed last, so it is the transaction marker consumers can
validate before using a newly published set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


TOOL_VERSION = "1.2.0"  # 1.2.0: 태그 컬럼 소문자 게시 보장
MANIFEST_VERSION = 1
RATINGS = ("g", "s", "q", "e")
ARCHIVE_TAG_CATEGORIES = ("general", "character", "copyright", "artist", "meta")
ARCHIVE_TAG_COLUMNS = ("tag", "legacy_categories", "legacy_releases")
RATING_NAMES = {
    "general": "g",
    "safe": "s",
    "sensitive": "s",
    "questionable": "q",
    "explicit": "e",
}

SEARCH_COLUMNS = (
    "rating",
    "image_width",
    "image_height",
    "general",
    "character",
    "copyright",
    "artist",
    "meta",
    "score",
)
EVENT_COLUMNS = (
    "id",
    "parent_id",
    "has_children",
    "has_visible_children",
    "tag_string_general",
    "tag_string_character",
    "tag_string_copyright",
    "tag_string_artist",
    "tag_string_meta",
    "rating",
    "score",
    "fav_count",
    "image_width",
    "image_height",
)

SOURCE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "id": ("id",),
    "parent_id": ("parent_id",),
    "rating": ("rating",),
    "image_width": ("image_width", "width"),
    "image_height": ("image_height", "height"),
    "score": ("score",),
    "fav_count": ("fav_count", "favorite_count", "favourite_count"),
    "general": ("tag_string_general", "general"),
    "character": ("tag_string_character", "character"),
    "copyright": ("tag_string_copyright", "copyright"),
    "artist": ("tag_string_artist", "artist"),
    "meta": ("tag_string_meta", "meta", "metadata"),
    "has_children": ("has_children",),
    "has_visible_children": ("has_visible_children",),
}


class RefreshError(RuntimeError):
    """A source, validation, or publication error safe to show to the user."""


@dataclass
class GraphScan:
    source_rows: int
    source_rating_rows: dict[str, int]
    child_ids: dict[str, set[int]]
    ancestor_ids: dict[str, set[int]]
    parent_by_child: dict[int, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _log(message: str) -> None:
    print(message, flush=True)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_relative_path(value: str, *, label: str) -> Path:
    portable = PurePosixPath(value.replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts or not portable.parts:
        raise RefreshError(f"{label} must be a relative descendant path: {value!r}")
    if any(part in {"", "."} for part in portable.parts):
        raise RefreshError(f"{label} contains an invalid path component: {value!r}")
    return Path(*portable.parts)


def _validate_dataset_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise RefreshError(
            "dataset label may contain only letters, digits, dot, underscore, and dash"
        )
    return value


def _import_arrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RefreshError("pyarrow is required: pip install pyarrow") from exc
    return pa, pc, pq


def _schema_fields(schema: Any) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _parquet_artifact(path: Path, relative_path: Path) -> dict[str, Any]:
    _pa, _pc, pq = _import_arrow()
    parquet = pq.ParquetFile(path)
    return {
        "path": relative_path.as_posix(),
        "format": "parquet",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": _schema_fields(parquet.schema_arrow),
        "created_by": parquet.metadata.created_by,
    }


def _csv_artifact(path: Path, relative_path: Path, *, columns: Sequence[str]) -> dict[str, Any]:
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, 1):
            if len(row) != len(columns):
                raise RefreshError(
                    f"{path}: row {row_number} has {len(row)} columns; "
                    f"expected {len(columns)}"
                )
            rows += 1
    return {
        "path": relative_path.as_posix(),
        "format": "csv",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": rows,
        "header": False,
        "schema": [
            {"name": name, "type": "string" if name in {"tag", "aliases"} else "int64"}
            for name in columns
        ],
    }


def _manifest_artifact_path(root: Path, artifact: Mapping[str, Any]) -> Path:
    relative = _safe_relative_path(str(artifact.get("path", "")), label="artifact path")
    candidate = (root / relative).resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise RefreshError(f"artifact escapes manifest root: {relative}") from exc
    return candidate


def validate_manifest(manifest_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Validate every artifact in a refresh manifest and return the manifest."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefreshError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise RefreshError(
            f"unsupported manifest version {manifest.get('format_version')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RefreshError("manifest artifacts must be a non-empty list")
    artifact_root = (root or manifest_path.parent).resolve(strict=False)
    _pa, _pc, pq = _import_arrow()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise RefreshError("manifest artifact must be an object")
        path = _manifest_artifact_path(artifact_root, artifact)
        if not path.is_file():
            raise RefreshError(f"manifest artifact is missing: {path}")
        expected_size = int(artifact.get("size_bytes", -1))
        if path.stat().st_size != expected_size:
            raise RefreshError(f"size mismatch: {path}")
        if _sha256(path) != artifact.get("sha256"):
            raise RefreshError(f"SHA-256 mismatch: {path}")
        if artifact.get("format") == "parquet":
            parquet = pq.ParquetFile(path)
            if parquet.metadata.num_rows != int(artifact.get("rows", -1)):
                raise RefreshError(f"row count mismatch: {path}")
            if _schema_fields(parquet.schema_arrow) != artifact.get("schema"):
                raise RefreshError(f"schema mismatch: {path}")
        elif artifact.get("format") == "csv":
            expected_rows = int(artifact.get("rows", -1))
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                actual_rows = sum(1 for _ in csv.reader(handle))
            if actual_rows != expected_rows:
                raise RefreshError(f"row count mismatch: {path}")
        else:
            raise RefreshError(f"unsupported artifact format: {artifact.get('format')!r}")
    return manifest


def _prepare_hardlink_backups(
    output_root: Path, stage_root: Path, relative_paths: Sequence[Path]
) -> dict[Path, Path | None]:
    """Create zero-copy rollback links before replacing any destination file."""

    backup_root = stage_root / ".rollback"
    backups: dict[Path, Path | None] = {}
    try:
        for relative in relative_paths:
            target = output_root / relative
            if not target.exists():
                backups[relative] = None
                continue
            if not target.is_file():
                raise RefreshError(f"destination is not a file: {target}")
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            os.link(target, backup)
            backups[relative] = backup
    except Exception as exc:
        shutil.rmtree(backup_root, ignore_errors=True)
        if isinstance(exc, RefreshError):
            raise
        raise RefreshError(
            "cannot create same-volume rollback links; no files were replaced: "
            f"{exc}"
        ) from exc
    return backups


def _atomic_commit(
    stage_root: Path,
    output_root: Path,
    relative_paths: Sequence[Path],
    *,
    manifest_relative: Path,
) -> None:
    """Atomically replace individual files and roll back the set on failure."""

    ordered = [path for path in relative_paths if path != manifest_relative]
    ordered.append(manifest_relative)  # manifest is the commit marker
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in ordered:
        (output_root / relative).parent.mkdir(parents=True, exist_ok=True)
    backups = _prepare_hardlink_backups(output_root, stage_root, ordered)
    replaced: list[Path] = []
    try:
        for relative in ordered:
            staged = stage_root / relative
            if not staged.is_file():
                raise RefreshError(f"staged output is missing: {staged}")
            os.replace(staged, output_root / relative)
            replaced.append(relative)
    except Exception as exc:
        rollback_errors: list[str] = []
        for relative in reversed(replaced):
            target = output_root / relative
            backup = backups.get(relative)
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except Exception as rollback_exc:  # pragma: no cover - exceptional I/O
                rollback_errors.append(f"{relative}: {rollback_exc}")
        detail = f"; rollback errors: {rollback_errors}" if rollback_errors else ""
        if isinstance(exc, RefreshError):
            raise RefreshError(f"{exc}{detail}") from exc
        raise RefreshError(f"atomic publication failed: {exc}{detail}") from exc


def _resolve_source_columns(names: Iterable[str]) -> dict[str, str]:
    available = set(names)
    resolved: dict[str, str] = {}
    required = {
        "id",
        "parent_id",
        "rating",
        "image_width",
        "image_height",
        "score",
        "fav_count",
        "general",
        "character",
        "copyright",
        "artist",
        "meta",
    }
    for canonical, candidates in SOURCE_COLUMN_CANDIDATES.items():
        match = next((candidate for candidate in candidates if candidate in available), None)
        if match is not None:
            resolved[canonical] = match
        elif canonical in required:
            raise RefreshError(
                f"posts snapshot is missing {canonical!r}; accepted names: "
                f"{', '.join(candidates)}"
            )
    return resolved


def _batch_array(batch: Any, source_name: str) -> Any:
    return batch.column(batch.schema.get_field_index(source_name))


def _cast_int64(array: Any, *, name: str, allow_null: bool = True) -> Any:
    pa, pc, _pq = _import_arrow()
    try:
        result = pc.cast(array, pa.int64(), safe=False)
    except Exception as exc:
        raise RefreshError(f"cannot convert {name} to int64: {exc}") from exc
    if not allow_null and bool(pc.any(pc.is_null(result)).as_py()):
        raise RefreshError(f"{name} contains null values")
    return result


def _normalise_parent_id(array: Any) -> Any:
    pa, pc, _pq = _import_arrow()
    values = _cast_int64(array, name="parent_id")
    positive = pc.fill_null(pc.greater(values, pa.scalar(0, type=pa.int64())), False)
    return pc.if_else(positive, values, pa.scalar(None, type=pa.int64()))


def _normalise_rating(array: Any) -> Any:
    pa, pc, _pq = _import_arrow()
    values = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(array, pa.string())))
    for long_name, short_name in RATING_NAMES.items():
        values = pc.if_else(
            pc.equal(values, pa.scalar(long_name)), pa.scalar(short_name), values
        )
    return values


def _validate_rating_array(ratings: Any) -> None:
    pa, pc, _pq = _import_arrow()
    valid = pc.is_in(ratings, value_set=pa.array(RATINGS, type=pa.string()))
    invalid = pc.invert(pc.fill_null(valid, False))
    if bool(pc.any(invalid).as_py()):
        samples = pc.unique(pc.filter(ratings, invalid)).to_pylist()[:8]
        raise RefreshError(f"snapshot contains unsupported ratings: {samples}")


def _scan_event_graph(parquet: Any, columns: Mapping[str, str], batch_rows: int) -> GraphScan:
    """First streaming pass: collect child→parent edges and rating roots."""

    pa, pc, _pq = _import_arrow()
    child_ids = {rating: set() for rating in RATINGS}
    direct_parents = {rating: set() for rating in RATINGS}
    parent_by_child: dict[int, int] = {}
    rating_rows = {rating: 0 for rating in RATINGS}
    source_rows = 0
    read_columns = list(dict.fromkeys(columns[key] for key in ("id", "parent_id", "rating")))
    for batch_number, batch in enumerate(
        parquet.iter_batches(batch_size=batch_rows, columns=read_columns, use_threads=True),
        1,
    ):
        ids = _cast_int64(_batch_array(batch, columns["id"]), name="id", allow_null=False)
        if bool(pc.any(pc.less_equal(ids, pa.scalar(0, type=pa.int64()))).as_py()):
            raise RefreshError("snapshot contains a non-positive id")
        parents = _normalise_parent_id(_batch_array(batch, columns["parent_id"]))
        ratings = _normalise_rating(_batch_array(batch, columns["rating"]))
        _validate_rating_array(ratings)
        source_rows += batch.num_rows
        for rating in RATINGS:
            rating_mask = pc.fill_null(pc.equal(ratings, pa.scalar(rating)), False)
            rating_rows[rating] += int(pc.sum(pc.cast(rating_mask, pa.int64())).as_py())
            child_mask = pc.and_(rating_mask, pc.invert(pc.is_null(parents)))
            if not bool(pc.any(child_mask).as_py()):
                continue
            selected_ids = pc.filter(ids, child_mask).to_pylist()
            selected_parents = pc.filter(parents, child_mask).to_pylist()
            for child_id, parent_id in zip(selected_ids, selected_parents, strict=True):
                child = int(child_id)
                parent = int(parent_id)
                if child == parent:
                    raise RefreshError(f"self-referential parent edge for post {child}")
                previous = parent_by_child.get(child)
                if previous is not None and previous != parent:
                    raise RefreshError(
                        f"post {child} has conflicting parents {previous} and {parent}"
                    )
                parent_by_child[child] = parent
                child_ids[rating].add(child)
                direct_parents[rating].add(parent)
        if batch_number % 25 == 0:
            _log(f"  graph scan: {source_rows:,} source rows")

    ancestor_ids: dict[str, set[int]] = {}
    for rating in RATINGS:
        ancestors = set(direct_parents[rating])
        frontier = list(ancestors)
        while frontier:
            node = frontier.pop()
            parent = parent_by_child.get(node)
            if parent is not None and parent not in ancestors:
                ancestors.add(parent)
                frontier.append(parent)
        ancestor_ids[rating] = ancestors
    return GraphScan(
        source_rows=source_rows,
        source_rating_rows=rating_rows,
        child_ids=child_ids,
        ancestor_ids=ancestor_ids,
        parent_by_child=parent_by_child,
    )


def _search_schema(metadata: Mapping[bytes, bytes] | None = None) -> Any:
    pa, _pc, _pq = _import_arrow()
    return pa.schema(
        [
            pa.field("rating", pa.string()),
            pa.field("image_width", pa.int64()),
            pa.field("image_height", pa.int64()),
            pa.field("general", pa.string()),
            pa.field("character", pa.string()),
            pa.field("copyright", pa.string()),
            pa.field("artist", pa.string()),
            pa.field("meta", pa.string()),
            pa.field("score", pa.int64()),
        ],
        metadata=metadata,
    )


def _event_schema(metadata: Mapping[bytes, bytes] | None = None) -> Any:
    pa, _pc, _pq = _import_arrow()
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("parent_id", pa.int64()),
            pa.field("has_children", pa.bool_()),
            pa.field("has_visible_children", pa.bool_()),
            pa.field("tag_string_general", pa.string()),
            pa.field("tag_string_character", pa.string()),
            pa.field("tag_string_copyright", pa.string()),
            pa.field("tag_string_artist", pa.string()),
            pa.field("tag_string_meta", pa.string()),
            pa.field("rating", pa.string()),
            pa.field("score", pa.int64()),
            pa.field("fav_count", pa.int64()),
            pa.field("image_width", pa.int64()),
            pa.field("image_height", pa.int64()),
        ],
        metadata=metadata,
    )


def _string_array(batch: Any, source_name: str) -> Any:
    pa, pc, _pq = _import_arrow()
    return pc.cast(_batch_array(batch, source_name), pa.string())


def _tag_string_array(batch: Any, source_name: str) -> Any:
    """Tag columns are published lower-case (null stays null).

    The runtime Search worker matches lower-cased queries against these
    columns directly, without keeping a lower-cased copy of each multi-million
    row column in memory, so the build must guarantee the case contract.
    """
    _pa, pc, _pq = _import_arrow()
    return pc.utf8_lower(_string_array(batch, source_name))


def _bool_array(batch: Any, source_name: str | None, derived: Any) -> Any:
    pa, pc, _pq = _import_arrow()
    if source_name is None:
        return derived
    source = pc.fill_null(pc.cast(_batch_array(batch, source_name), pa.bool_()), False)
    return pc.or_(source, derived)


def _source_metadata(source_sha256: str, dataset_label: str, kind: str) -> dict[bytes, bytes]:
    return {
        b"refresh_tool": f"refresh_danbooru_data.py/{TOOL_VERSION}".encode("ascii"),
        b"source_sha256": source_sha256.encode("ascii"),
        b"dataset_label": dataset_label.encode("utf-8"),
        b"dataset_kind": kind.encode("ascii"),
    }


def _open_writers(
    paths: Mapping[str, Path], schema: Any, *, compression: str, compression_level: int
) -> dict[str, Any]:
    _pa, _pc, pq = _import_arrow()
    writers: dict[str, Any] = {}
    try:
        for rating, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            writers[rating] = pq.ParquetWriter(
                path,
                schema,
                compression=compression,
                compression_level=compression_level,
                use_dictionary=True,
                write_statistics=True,
            )
    except Exception:
        for writer in writers.values():
            writer.close()
        raise
    return writers


def _close_writers(writers: Mapping[str, Any]) -> None:
    errors: list[Exception] = []
    for writer in writers.values():
        try:
            writer.close()
        except Exception as exc:  # pragma: no cover - exceptional I/O
            errors.append(exc)
    if errors:
        raise RefreshError(f"failed closing parquet writers: {errors}")


def build_posts(args: argparse.Namespace) -> dict[str, Any]:
    """Build Search and Event parquets from a local posts snapshot."""

    pa, pc, pq = _import_arrow()
    source = Path(args.source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise RefreshError(f"posts source is not a file: {source}")
    dataset_label = _validate_dataset_label(args.dataset_label)
    output_root = Path(args.output_dir).expanduser().resolve(strict=False)
    event_subdir = _safe_relative_path(args.event_subdir, label="event subdirectory")
    manifest_relative = _safe_relative_path(
        args.manifest_name or f"danbooru_{dataset_label}_manifest.json",
        label="manifest name",
    )
    parquet = pq.ParquetFile(source)
    columns = _resolve_source_columns(parquet.schema_arrow.names)
    plan = {
        "source": str(source),
        "source_rows": parquet.metadata.num_rows,
        "source_schema": _schema_fields(parquet.schema_arrow),
        "resolved_columns": columns,
        "dataset_label": dataset_label,
        "search_outputs": [
            str(output_root / f"danbooru_{dataset_label}_{rating}.parquet")
            for rating in RATINGS
        ],
        "event_outputs": [
            str(output_root / event_subdir / f"danbooru_{rating}.parquet")
            for rating in RATINGS
        ],
        "manifest": str(output_root / manifest_relative),
    }
    if args.dry_run:
        _log(json.dumps(plan, ensure_ascii=False, indent=2))
        return {"dry_run": True, "plan": plan}

    _log(f"Computing source SHA-256: {source}")
    source_sha256 = _sha256(source)
    if args.expected_source_sha256:
        expected = args.expected_source_sha256.lower()
        if source_sha256 != expected:
            raise RefreshError(
                f"source SHA-256 mismatch: expected {expected}, got {source_sha256}"
            )

    _log("Scanning parent-child graph (streaming pass 1/2)...")
    graph = _scan_event_graph(parquet, columns, args.batch_rows)
    if graph.source_rows != parquet.metadata.num_rows:
        raise RefreshError(
            f"source row count changed while reading: metadata={parquet.metadata.num_rows}, "
            f"read={graph.source_rows}"
        )

    stage_parent = output_root.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=".danbooru-posts-stage-", dir=stage_parent))
    search_relatives = {
        rating: Path(f"danbooru_{dataset_label}_{rating}.parquet") for rating in RATINGS
    }
    event_relatives = {
        rating: event_subdir / f"danbooru_{rating}.parquet" for rating in RATINGS
    }
    search_paths = {rating: stage_root / path for rating, path in search_relatives.items()}
    event_paths = {rating: stage_root / path for rating, path in event_relatives.items()}
    common_meta = _source_metadata(source_sha256, dataset_label, "search")
    event_meta = _source_metadata(source_sha256, dataset_label, "event_graph")
    search_schema = _search_schema(common_meta)
    event_schema = _event_schema(event_meta)
    search_writers: dict[str, Any] = {}
    event_writers: dict[str, Any] = {}
    search_rows = {rating: 0 for rating in RATINGS}
    event_rows = {rating: 0 for rating in RATINGS}
    event_seen_ids = {rating: set() for rating in RATINGS}
    all_parent_ids = set(graph.parent_by_child.values())
    all_parent_values = pa.array(sorted(all_parent_ids), type=pa.int64())
    ancestor_values = {
        rating: pa.array(sorted(graph.ancestor_ids[rating]), type=pa.int64())
        for rating in RATINGS
    }
    read_columns = list(dict.fromkeys(columns.values()))
    processed_rows = 0
    try:
        search_writers = _open_writers(
            search_paths,
            search_schema,
            compression=args.compression,
            compression_level=args.compression_level,
        )
        event_writers = _open_writers(
            event_paths,
            event_schema,
            compression=args.compression,
            compression_level=args.compression_level,
        )
        _log("Writing Search and Event shards (streaming pass 2/2)...")
        for batch_number, batch in enumerate(
            parquet.iter_batches(
                batch_size=args.batch_rows, columns=read_columns, use_threads=True
            ),
            1,
        ):
            ids = _cast_int64(
                _batch_array(batch, columns["id"]), name="id", allow_null=False
            )
            parents = _normalise_parent_id(_batch_array(batch, columns["parent_id"]))
            ratings = _normalise_rating(_batch_array(batch, columns["rating"]))
            _validate_rating_array(ratings)
            widths = _cast_int64(
                _batch_array(batch, columns["image_width"]), name="image_width"
            )
            heights = _cast_int64(
                _batch_array(batch, columns["image_height"]), name="image_height"
            )
            scores = _cast_int64(_batch_array(batch, columns["score"]), name="score")
            favourites = _cast_int64(
                _batch_array(batch, columns["fav_count"]), name="fav_count"
            )
            # 태그 5종은 소문자로 게시 — Search 워커가 소문자 사본 없이 원본 컬럼으로 매칭한다.
            general = _tag_string_array(batch, columns["general"])
            character = _tag_string_array(batch, columns["character"])
            copyright_tags = _tag_string_array(batch, columns["copyright"])
            artist = _tag_string_array(batch, columns["artist"])
            meta_tags = _tag_string_array(batch, columns["meta"])
            derived_has_children = pc.is_in(ids, value_set=all_parent_values)
            has_children = _bool_array(
                batch, columns.get("has_children"), derived_has_children
            )
            has_visible_children = _bool_array(
                batch, columns.get("has_visible_children"), derived_has_children
            )

            search_table = pa.Table.from_arrays(
                [
                    ratings,
                    widths,
                    heights,
                    general,
                    character,
                    copyright_tags,
                    artist,
                    meta_tags,
                    scores,
                ],
                schema=search_schema,
            )
            event_table = pa.Table.from_arrays(
                [
                    ids,
                    parents,
                    has_children,
                    has_visible_children,
                    general,
                    character,
                    copyright_tags,
                    artist,
                    meta_tags,
                    ratings,
                    scores,
                    favourites,
                    widths,
                    heights,
                ],
                schema=event_schema,
            )
            has_parent = pc.invert(pc.is_null(parents))
            for rating in RATINGS:
                rating_mask = pc.fill_null(pc.equal(ratings, pa.scalar(rating)), False)
                search_part = search_table.filter(rating_mask)
                if search_part.num_rows:
                    search_writers[rating].write_table(
                        search_part, row_group_size=args.row_group_rows
                    )
                    search_rows[rating] += search_part.num_rows

                child_mask = pc.and_(rating_mask, has_parent)
                ancestor_mask = pc.is_in(ids, value_set=ancestor_values[rating])
                include_mask = pc.fill_null(pc.or_(child_mask, ancestor_mask), False)
                if not bool(pc.any(include_mask).as_py()):
                    continue
                selected_ids = pc.filter(ids, include_mask).to_pylist()
                duplicates = event_seen_ids[rating].intersection(selected_ids)
                if duplicates:
                    sample = sorted(duplicates)[:5]
                    raise RefreshError(
                        f"source contains duplicate Event IDs for rating {rating}: {sample}"
                    )
                event_seen_ids[rating].update(int(value) for value in selected_ids)
                event_part = event_table.filter(include_mask)
                event_writers[rating].write_table(
                    event_part, row_group_size=args.row_group_rows
                )
                event_rows[rating] += event_part.num_rows
            processed_rows += batch.num_rows
            if batch_number % 10 == 0:
                _log(f"  wrote from {processed_rows:,}/{graph.source_rows:,} source rows")
        _close_writers(search_writers)
        search_writers = {}
        _close_writers(event_writers)
        event_writers = {}

        for rating in RATINGS:
            if search_rows[rating] != graph.source_rating_rows[rating]:
                raise RefreshError(
                    f"Search {rating} row mismatch: expected "
                    f"{graph.source_rating_rows[rating]}, wrote {search_rows[rating]}"
                )
            missing = graph.ancestor_ids[rating] - event_seen_ids[rating]
            if missing and not args.allow_missing_parents:
                raise RefreshError(
                    f"Event {rating} is missing {len(missing)} referenced ancestors; "
                    f"sample={sorted(missing)[:8]}. Use --allow-missing-parents only "
                    "for intentionally incomplete snapshots."
                )

        _log("Hashing and validating staged parquet files...")
        artifacts: list[dict[str, Any]] = []
        for rating in RATINGS:
            artifact = _parquet_artifact(search_paths[rating], search_relatives[rating])
            if artifact["rows"] != search_rows[rating]:
                raise RefreshError(f"staged Search {rating} metadata row mismatch")
            artifact.update({"kind": "search", "rating_shard": rating})
            artifacts.append(artifact)
        event_unique_union: set[int] = set()
        for rating in RATINGS:
            artifact = _parquet_artifact(event_paths[rating], event_relatives[rating])
            if artifact["rows"] != event_rows[rating]:
                raise RefreshError(f"staged Event {rating} metadata row mismatch")
            missing = graph.ancestor_ids[rating] - event_seen_ids[rating]
            child_parent_overlap = graph.child_ids[rating] & graph.ancestor_ids[rating]
            artifact.update(
                {
                    "kind": "event_graph",
                    "rating_shard": rating,
                    "child_ids": len(graph.child_ids[rating]),
                    "ancestor_ids": len(graph.ancestor_ids[rating]),
                    "child_ancestor_overlap": len(child_parent_overlap),
                    "missing_ancestor_ids": len(missing),
                    "contains_cross_rating_parents": True,
                }
            )
            artifacts.append(artifact)
            event_unique_union.update(event_seen_ids[rating])
        event_total_rows = sum(event_rows.values())
        manifest = {
            "format_version": MANIFEST_VERSION,
            "generated_by": f"tools/refresh_danbooru_data.py {TOOL_VERSION}",
            "generated_at": _utc_now(),
            "command": "posts",
            "dataset_label": dataset_label,
            "source": {
                "path": source.name,
                "url": args.source_url,
                "revision": args.source_revision,
                "snapshot_at": args.snapshot_at,
                "size_bytes": source.stat().st_size,
                "sha256": source_sha256,
                "rows": graph.source_rows,
                "schema": _schema_fields(parquet.schema_arrow),
                "resolved_columns": columns,
            },
            "build": {
                "batch_rows": args.batch_rows,
                "row_group_rows": args.row_group_rows,
                "compression": args.compression,
                "compression_level": args.compression_level,
                "event_shard_rule": (
                    "children whose own rating matches the shard, plus all recursively "
                    "referenced ancestors regardless of ancestor rating"
                ),
                "cross_rating_parent_duplication": True,
                "event_rows_across_shards": event_total_rows,
                "event_unique_ids_across_shards": len(event_unique_union),
                "event_duplicate_rows_across_shards": (
                    event_total_rows - len(event_unique_union)
                ),
                "allow_missing_parents": bool(args.allow_missing_parents),
            },
            "artifacts": artifacts,
        }
        staged_manifest = stage_root / manifest_relative
        _write_bytes(staged_manifest, _json_bytes(manifest))
        validate_manifest(staged_manifest, stage_root)
        relative_paths = [*search_relatives.values(), *event_relatives.values(), manifest_relative]
        _log("Publishing validated outputs atomically (manifest last)...")
        _atomic_commit(
            stage_root,
            output_root,
            relative_paths,
            manifest_relative=manifest_relative,
        )
        validate_manifest(output_root / manifest_relative, output_root)
        _log(f"Published manifest: {output_root / manifest_relative}")
        return manifest
    finally:
        if search_writers:
            _close_writers(search_writers)
        if event_writers:
            _close_writers(event_writers)
        shutil.rmtree(stage_root, ignore_errors=True)


def _release_parquet_files(input_root: Path, dataset_label: str) -> dict[str, Path]:
    label = _validate_dataset_label(dataset_label)
    files = {
        rating: input_root / f"danbooru_{label}_{rating}.parquet"
        for rating in RATINGS
    }
    missing = [path.name for path in files.values() if not path.is_file()]
    if missing:
        raise RefreshError(
            f"{label} Search 릴리스가 완전하지 않습니다: {', '.join(missing)}"
        )
    return files


def _unique_normalized_tags(values: Any, *, comma_separated: bool) -> list[str]:
    """Return unique canonical tag spellings from one Arrow string chunk."""
    pa, pc, _pq = _import_arrow()
    text = pc.fill_null(pc.cast(values, pa.string()), "")
    if comma_separated:
        pieces = pc.list_flatten(pc.split_pattern(text, pattern=","))
    else:
        pieces = pc.list_flatten(
            pc.split_pattern_regex(pc.utf8_trim_whitespace(text), pattern=r"\s+")
        )
    normalized = pc.replace_substring_regex(
        pc.utf8_trim_whitespace(pieces),
        pattern=r"\s+",
        replacement="_",
    )
    non_empty = pc.not_equal(normalized, "")
    return [
        tag
        for tag in pc.unique(pc.filter(normalized, non_empty)).to_pylist()
        if tag
    ]


def _iter_release_tag_batches(
    input_root: Path,
    dataset_label: str,
) -> Iterable[tuple[str, list[str]]]:
    """Yield category and unique tags while keeping Parquet memory bounded."""
    _pa, _pc, pq = _import_arrow()
    for rating, path in _release_parquet_files(input_root, dataset_label).items():
        parquet = pq.ParquetFile(path)
        schema_names = set(parquet.schema.names)
        column_map: dict[str, str] = {}
        for category in ARCHIVE_TAG_CATEGORIES:
            if category in schema_names:
                column_map[category] = category
            elif category == "meta" and "metadata" in schema_names:
                column_map[category] = "metadata"
            else:
                raise RefreshError(
                    f"{path.name}: 태그 컬럼이 없습니다: {category}"
                )

        comma_separated = "metadata" in schema_names and "meta" not in schema_names
        read_columns = list(dict.fromkeys(column_map.values()))
        _log(
            f"태그 스캔: {path.name} "
            f"({parquet.metadata.num_rows:,}행, {parquet.metadata.num_row_groups}그룹)"
        )
        for row_group in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(row_group, columns=read_columns)
            for category, source_column in column_map.items():
                yield category, _unique_normalized_tags(
                    table[source_column],
                    comma_separated=comma_separated,
                )


def archive_legacy_tags(args: argparse.Namespace) -> dict[str, Any]:
    """Archive tags absent from the latest Search release as one compact CSV."""
    input_root = Path(args.input_dir).expanduser().resolve(strict=True)
    latest_label = _validate_dataset_label(args.latest_label)
    legacy_labels = [
        _validate_dataset_label(label) for label in args.legacy_labels
    ]
    if not legacy_labels:
        raise RefreshError("하나 이상의 구형 릴리스가 필요합니다")
    if latest_label in legacy_labels:
        raise RefreshError("최신 릴리스는 구형 릴리스 목록에 넣을 수 없습니다")
    if len(set(legacy_labels)) != len(legacy_labels):
        raise RefreshError("구형 릴리스 이름이 중복되었습니다")

    output = (
        Path(args.output).expanduser()
        if args.output
        else input_root / f"legacy_search_tags_before_{latest_label}.csv"
    ).resolve(strict=False)
    if output.suffix.lower() != ".csv":
        raise RefreshError("legacy 태그 archive 출력은 .csv 파일이어야 합니다")

    source_files: set[Path] = set()
    for label in (latest_label, *legacy_labels):
        source_files.update(
            path.resolve(strict=True)
            for path in _release_parquet_files(input_root, label).values()
        )
    if output in source_files:
        raise RefreshError("legacy 태그 archive 출력이 입력 parquet와 충돌합니다")
    output.parent.mkdir(parents=True, exist_ok=True)

    latest_tags: set[str] = set()
    for _category, tags in _iter_release_tag_batches(input_root, latest_label):
        latest_tags.update(tags)
    _log(f"최신 고유 태그: {len(latest_tags):,}개")

    category_bits = {
        category: 1 << index
        for index, category in enumerate(ARCHIVE_TAG_CATEGORIES)
    }
    release_bits = {
        label: 1 << index for index, label in enumerate(legacy_labels)
    }
    legacy_only: dict[str, tuple[int, int]] = {}
    for label in legacy_labels:
        for category, tags in _iter_release_tag_batches(input_root, label):
            category_bit = category_bits[category]
            release_bit = release_bits[label]
            for tag in tags:
                if tag in latest_tags:
                    continue
                old_categories, old_releases = legacy_only.get(tag, (0, 0))
                legacy_only[tag] = (
                    old_categories | category_bit,
                    old_releases | release_bit,
                )

    temp_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    try:
        with temp_handle:
            writer = csv.writer(temp_handle, lineterminator="\n")
            writer.writerow(ARCHIVE_TAG_COLUMNS)
            for tag in sorted(legacy_only):
                category_mask, release_mask = legacy_only[tag]
                categories = "|".join(
                    category
                    for category in ARCHIVE_TAG_CATEGORIES
                    if category_mask & category_bits[category]
                )
                releases = "|".join(
                    label
                    for label in legacy_labels
                    if release_mask & release_bits[label]
                )
                writer.writerow((tag, categories, releases))
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)

    result = {
        "output": str(output),
        "rows": len(legacy_only),
        "latest_label": latest_label,
        "latest_unique_tags": len(latest_tags),
        "legacy_labels": legacy_labels,
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
    }
    _log(
        f"구형 전용 태그 CSV 저장: {output} "
        f"({result['rows']:,}행, {result['size_bytes']:,}바이트, "
        f"SHA-256 {result['sha256']})"
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Danbooru Search/Event 데이터와 태그 자산을 검증 후 갱신합니다."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    posts = subparsers.add_parser(
        "posts",
        help="로컬 posts snapshot을 Search/Event parquet로 변환",
    )
    posts.add_argument("--source", required=True, help="원본 posts parquet 경로")
    posts.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "danbooru_optimized"),
        help="출력 danbooru_optimized 경로",
    )
    posts.add_argument("--dataset-label", default="2026_07")
    posts.add_argument("--event-subdir", default="danbooru_sorted")
    posts.add_argument("--manifest-name", default="dataset_manifest.json")
    posts.add_argument("--source-url", default="")
    posts.add_argument("--source-revision", default="")
    posts.add_argument("--snapshot-at", default="")
    posts.add_argument("--expected-source-sha256", default="")
    posts.add_argument("--batch-rows", type=int, default=262_144)
    posts.add_argument("--row-group-rows", type=int, default=131_072)
    posts.add_argument("--compression", default="zstd")
    posts.add_argument("--compression-level", type=int, default=6)
    posts.add_argument("--allow-missing-parents", action="store_true")
    posts.add_argument("--dry-run", action="store_true")
    posts.set_defaults(handler=build_posts)

    archive = subparsers.add_parser(
        "archive-legacy-tags",
        help="구형 Search 릴리스에만 남은 고유 태그를 작은 CSV로 보관",
    )
    archive.add_argument(
        "--input-dir",
        default=str(Path(__file__).resolve().parents[1] / "danbooru_optimized"),
        help="Search parquet가 있는 danbooru_optimized 경로",
    )
    archive.add_argument("--latest-label", default="2026_07")
    archive.add_argument(
        "--legacy-labels",
        nargs="+",
        default=["2025", "2026", "2026_06"],
    )
    archive.add_argument(
        "--output",
        default="",
        help="출력 CSV 경로(기본: input-dir/legacy_search_tags_before_<latest>.csv)",
    )
    archive.set_defaults(handler=archive_legacy_tags)

    validate = subparsers.add_parser(
        "validate",
        help="기존 runtime dataset manifest와 산출물 해시/스키마 검증",
    )
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--root", default="")

    def _validate_handler(args: argparse.Namespace) -> dict[str, Any]:
        manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
        root = Path(args.root).expanduser().resolve(strict=True) if args.root else None
        result = validate_manifest(manifest_path, root)
        _log(f"Validated {len(result.get('artifacts', []))} artifacts: {manifest_path}")
        return result

    validate.set_defaults(handler=_validate_handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "batch_rows", 1) <= 0:
        parser.error("--batch-rows must be positive")
    if getattr(args, "row_group_rows", 1) <= 0:
        parser.error("--row-group-rows must be positive")
    try:
        args.handler(args)
    except (RefreshError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"[danbooru-refresh] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
