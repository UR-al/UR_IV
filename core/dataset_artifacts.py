# core/dataset_artifacts.py
"""danbooru_optimized/dataset_manifest.json 의 산출물(shard) 존재·크기 확인 — 가벼운 순수 로직.

core/fetch_data 가 앱 시작 **전**(config·PyQt·pandas import 전)에 쓰므로 표준 라이브러리만
쓴다. 기동 때마다 수 GB 를 해싱하지 않도록 sha256 은 보지 않고 경로·크기만 비교한다
(Search 는 workers/search_worker 가 sha·행 수·스키마까지 검증한다).

manifest 는 git 이 추적하고 parquet 은 .gitignore 대상이다. 그래서 저장소를 pull 해
manifest 가 새 릴리스를 가리키면, 디스크의 parquet 은 그대로라 여기서 "낡음"으로 잡힌다.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable

MANIFEST_NAME = "dataset_manifest.json"


def read_manifest(data_dir: Path | str) -> dict | None:
    """manifest 를 읽는다. 없거나 깨졌거나 artifacts 목록이 없으면 None."""
    path = Path(data_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), list):
        return None
    return manifest


def artifact_relpath(artifact: object) -> str | None:
    """artifact 의 path → 데이터 폴더 기준 상대 POSIX 경로. 폴더를 벗어나는 값은 None."""
    if not isinstance(artifact, dict):
        return None
    raw = artifact.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("\\", "/")
    if text.startswith("/") or ":" in text:
        return None
    parts = PurePosixPath(text).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return None
    return PurePosixPath(*parts).as_posix()


def _expected_size(artifact: dict) -> int | None:
    size = artifact.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return None
    return size


def manifest_artifacts(manifest: dict | None, *, kind: str | None = None) -> list[tuple[str, dict]]:
    """(상대 경로, artifact) 목록 — 경로가 안전한 것만, kind 로 거를 수 있다."""
    if not isinstance(manifest, dict):
        return []
    out: list[tuple[str, dict]] = []
    for artifact in manifest.get("artifacts") or ():
        relpath = artifact_relpath(artifact)
        if relpath is None:
            continue
        if kind is not None and artifact.get("kind") != kind:
            continue
        out.append((relpath, artifact))
    return out


def stale_artifacts(
    manifest: dict | None, data_dir: Path | str, *, kind: str | None = None
) -> list[str]:
    """없거나 size_bytes 가 다른 artifact 의 상대 경로 (manifest 순서)."""
    root = Path(data_dir)
    stale: list[str] = []
    for relpath, artifact in manifest_artifacts(manifest, kind=kind):
        path = root / relpath
        try:
            actual = path.stat().st_size if path.is_file() else None
        except OSError:
            actual = None
        expected = _expected_size(artifact)
        if actual is None or (expected is not None and actual != expected):
            stale.append(relpath)
    return stale


def manifest_revision(manifest: dict | None) -> str | None:
    """배포 저장소(Hugging Face) 커밋 고정값 — 선택 필드 `hf_revision`."""
    if not isinstance(manifest, dict):
        return None
    revision = manifest.get("hf_revision")
    if isinstance(revision, str) and revision.strip():
        return revision.strip()
    return None


def stale_among(
    manifest: dict | None, data_dir: Path | str, files: Iterable[Path | str], *, kind: str | None = None
) -> list[str]:
    """`files` 중 manifest 와 어긋난 것의 상대 경로 (manifest 에 없는 파일은 판단하지 않음)."""
    root = Path(data_dir).resolve()
    wanted: set[str] = set()
    for file in files:
        try:
            wanted.add(Path(file).resolve().relative_to(root).as_posix())
        except (OSError, ValueError):
            continue
    return [relpath for relpath in stale_artifacts(manifest, root, kind=kind) if relpath in wanted]


__all__ = [
    "MANIFEST_NAME",
    "artifact_relpath",
    "manifest_artifacts",
    "manifest_revision",
    "read_manifest",
    "stale_among",
    "stale_artifacts",
]
