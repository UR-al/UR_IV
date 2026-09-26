"""Installation and capability checks for AI Studio's bundled ComfyUI nodes.

The node sources live with the application so Forge/Comfy feature parity is
versioned together with the workflow compiler.  A linked ComfyUI installation
is never overwritten blindly: only a directory carrying our ownership marker
may be refreshed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACK_ID = "ai_studio_forge_parity"
# Must equal ``comfy_custom_nodes/ai_studio_forge_parity/__init__.__version__``
# (tests/test_comfy_node_pack.py guards the pair).
# 1.4.0: Detail Daemon = the original ComfyUI node (×0.1 always, no presets). Packs before it
# read the same settings 10× stronger; their ForgeNeoAnimaDetailDaemon lacks the
# ``cfg_scale_override`` input, so the compiler's /object_info check refuses them.
# Same (unreleased) 1.4.0: ForgeNeoAnimaTileRepair and the SAM3 detailer's Anima
# ControlNet-LLLite slot (kohya ComfyUI-Anima-LLLite node, vendored). Older packs have
# no ForgeNeoAnimaTileRepair in /object_info and hand an LLLite to ControlNetLoader.
# 1.4.1: DAVE judges its steps by the sigma before Detail Daemon scaled it (optional
# ``pre_dd_sigma`` input / suite ``guid_dave_pre_dd``, default on); inputs a 1.4.0 pack
# accepts are unchanged, so the compiler needs no gate.
PACK_VERSION = "1.4.1"
OWNER_ID = "ai-studio-pro.bundled-comfy-nodes"
OWNER_MARKER = ".aistudio-owned.json"

# Everything copied/verified as part of the pack (tamper detection covers docs
# and licenses too) versus what ComfyUI actually imports or reads at runtime.
_PACK_SUFFIXES = frozenset({".py", ".json", ".md", ".txt"})
_RUNTIME_SUFFIXES = frozenset({".py", ".json"})


class ComfyNodePackError(RuntimeError):
    """Raised when the bundled node pack cannot be installed safely."""


@dataclass(frozen=True)
class NodePackInstallResult:
    target: Path
    fingerprint: str
    changed: bool
    # False when only documentation/licence files differed: the files are
    # refreshed, but the running ComfyUI already executes identical code.
    restart_required: bool = True


def bundled_node_pack_path(project_root: Path | str | None = None) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parent.parent
    )
    return root / "comfy_custom_nodes" / PACK_ID


def pack_files(source: Path | str, *, runtime_only: bool = False) -> list[Path]:
    """Return the fingerprinted pack files in a stable, platform-neutral order."""

    source_path = Path(source)
    suffixes = _RUNTIME_SUFFIXES if runtime_only else _PACK_SUFFIXES
    return sorted(
        (
            path
            for path in source_path.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.name != OWNER_MARKER
            and path.suffix.casefold() in suffixes
        ),
        key=lambda path: path.relative_to(source_path).as_posix(),
    )


def node_pack_fingerprint(
    source: Path | str,
    *,
    runtime_only: bool = False,
    max_files: int | None = None,
    max_file_bytes: int | None = None,
) -> str:
    """The single pack digest used by the installer and the compatibility UI.

    ``runtime_only`` restricts it to the files ComfyUI imports/reads, which
    decides whether a refreshed pack needs a ComfyUI restart.  The optional
    limits bound a read of an arbitrary installed directory.
    """

    source_path = Path(source).resolve()
    if not source_path.is_dir():
        raise ComfyNodePackError(f"번들 ComfyUI 노드 폴더가 없습니다: {source_path}")
    digest = hashlib.sha256()
    files = pack_files(source_path, runtime_only=runtime_only)
    if not files:
        raise ComfyNodePackError(f"번들 ComfyUI 노드가 비어 있습니다: {source_path}")
    if max_files is not None and len(files) > int(max_files):
        raise ComfyNodePackError(f"ComfyUI 노드 파일이 너무 많습니다: {len(files)}개")
    for path in files:
        if max_file_bytes is not None and path.stat().st_size > int(max_file_bytes):
            raise ComfyNodePackError(f"ComfyUI 노드 파일이 너무 큽니다: {path.name}")
        relative = path.relative_to(source_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _installed_runtime_fingerprint(target: Path) -> str:
    try:
        return node_pack_fingerprint(target, runtime_only=True)
    except (ComfyNodePackError, OSError):
        return ""


def _read_marker(target: Path) -> dict[str, Any] | None:
    marker = target / OWNER_MARKER
    if not marker.is_file():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("owner") != OWNER_ID:
        return None
    return data


def install_bundled_node_pack(
    custom_nodes_root: Path | str,
    *,
    source: Path | str | None = None,
) -> NodePackInstallResult:
    """Atomically copy the bundled pack into one explicit ``custom_nodes`` root.

    The root itself must already exist.  Existing third-party content at the
    reserved pack name is treated as a conflict.  A previous app-owned copy is
    refreshed only when its content fingerprint changes.  ``restart_required``
    is False only when the replaced copy already had byte-identical runtime
    (``.py``/``.json``) files, i.e. a documentation-only refresh.
    """

    root = Path(custom_nodes_root).expanduser().resolve()
    if not root.is_dir():
        raise ComfyNodePackError(f"ComfyUI custom_nodes 폴더가 없습니다: {root}")
    source_path = Path(source).resolve() if source is not None else bundled_node_pack_path()
    if not (source_path / "__init__.py").is_file():
        raise ComfyNodePackError(
            f"번들 ComfyUI 노드 진입점을 찾을 수 없습니다: {source_path / '__init__.py'}"
        )
    fingerprint = node_pack_fingerprint(source_path)
    runtime_fingerprint = node_pack_fingerprint(source_path, runtime_only=True)
    target = root / PACK_ID
    existing = _read_marker(target) if target.exists() else None
    if target.exists() and existing is None:
        raise ComfyNodePackError(
            f"앱 소유가 아닌 같은 이름의 ComfyUI 확장이 있어 덮어쓰지 않았습니다: {target}"
        )
    if existing and existing.get("fingerprint") == fingerprint:
        try:
            installed_fingerprint = node_pack_fingerprint(target)
        except ComfyNodePackError:
            installed_fingerprint = ""
        if installed_fingerprint == fingerprint:
            return NodePackInstallResult(
                target=target, fingerprint=fingerprint, changed=False,
                restart_required=False,
            )
    # Measured on disk, not trusted from the marker: a tampered .py must
    # still force the restart that reloads the repaired code.
    restart_required = (
        existing is None
        or _installed_runtime_fingerprint(target) != runtime_fingerprint
    )

    suffix = uuid.uuid4().hex[:10]
    staging = root / f".{PACK_ID}.staging-{suffix}"
    backup = root / f".{PACK_ID}.backup-{suffix}"
    try:
        shutil.copytree(
            source_path,
            staging,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", OWNER_MARKER),
        )
        marker = {
            "owner": OWNER_ID,
            "packId": PACK_ID,
            "version": PACK_VERSION,
            "fingerprint": fingerprint,
            "runtimeFingerprint": runtime_fingerprint,
        }
        (staging / OWNER_MARKER).write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return NodePackInstallResult(
        target=target, fingerprint=fingerprint, changed=True,
        restart_required=restart_required,
    )
