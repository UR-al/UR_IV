"""Safe import/export of application settings and user-authored presets."""

from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from core.storage_paths import PROJECT_ROOT


MAX_BACKUP_FILE_BYTES = 64 * 1024 * 1024
MAX_BACKUP_TOTAL_BYTES = 256 * 1024 * 1024

CONFIG_EXPORT_FILES = (
    "prompt_settings.json",
    "ui_prefs.json",
    "tab_defaults.json",
    "cond_rules.json",
    "global_weights.json",
    "prompt_order.json",
    "char_global_prefs.json",
    "backend_runtime.json",
    "forge_model_paths.json",
    # 사용자가 작성한 대화/보조/스키마 지침 프리셋(core.instruction_presets). gitignore 로
    # 저장소에서 빠지므로 백업이 유일한 사본이다.
    "instruction_presets.json",
    # ComfyUI 워크플로 컨트롤 편집·품질 프리셋·호환성 기준선 — 사용자가 만든 값(gitignore).
    "comfy_workflow_controls.json",
    "comfy_quality_preset.json",
    "comfy_compatibility_baselines.json",
)
#: 크기가 클 수 있어(스레드당 이미지 ~6MB) 사용자가 고를 때만 내보내는 파일.
OPTIONAL_CONFIG_EXPORT_FILES = (
    "chat_threads.json",
)
CONFIG_IMPORT_FILES = frozenset((
    *CONFIG_EXPORT_FILES, *OPTIONAL_CONFIG_EXPORT_FILES, "gallery_last_folder.txt",
))
# user_data/prompt_presets.json 은 빠졌다 — 생성 프리셋은 presets/ 폴더(core.generation_presets)가
# 주인이고 그 파일은 어디서도 읽지 않는다(audit #179).
USER_FILES = (
    "character_presets.json",
    "prompt_history.json",
    "favorite_tags.json",
    "favorites.json",
    "event_gen_settings.json",
    "search_tab_settings.json",
    "instant_wildcards.json",
)
DIRECTORY_EXPORTS = (
    (("config", "automation"), ("config", "automation")),
    (("config", "state"), ("config", "state")),
    (("config", "profiles"), ("config", "profiles")),
    (("user_data", "creator"), ("user_data", "creator")),
    (("queue_presets",), ("queue_presets",)),
    (("presets",), ("presets",)),
    (("wildcards",), ("wildcards",)),
)
ALLOWED_TREE_PREFIXES = tuple(archive for _source, archive in DIRECTORY_EXPORTS)


class SettingsBackupError(ValueError):
    """Raised when a settings archive is unsafe or exceeds its limits."""


@dataclass(frozen=True)
class ImportTarget:
    archive_name: str
    target: Path
    boundary: Path


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_archive_parts(archive_name: str) -> tuple[str, ...] | None:
    raw = str(archive_name).replace("\\", "/")
    if not raw or "\x00" in raw:
        raise SettingsBackupError("backup entry has an invalid name")
    portable = PurePosixPath(raw)
    parts = portable.parts
    if portable.is_absolute() or not parts or ".." in parts:
        raise SettingsBackupError(f"backup entry escapes its root: {archive_name!r}")
    if any(":" in part for part in parts):
        raise SettingsBackupError(f"backup entry contains an unsafe path: {archive_name!r}")
    return parts


def resolve_import_target(
    project_root: str | os.PathLike[str],
    archive_name: str,
) -> ImportTarget | None:
    """Map one allowlisted archive entry to a symlink-safe local target."""
    root = Path(project_root).expanduser().resolve(strict=False)
    parts = _safe_archive_parts(archive_name)
    if parts is None:
        return None

    boundary_parts: tuple[str, ...] | None = None
    if len(parts) == 2 and parts[0] == "config" and parts[1] in CONFIG_IMPORT_FILES:
        boundary_parts = ("config",)
    elif len(parts) == 2 and parts[0] == "user_data" and parts[1] in USER_FILES:
        boundary_parts = ("user_data",)
    else:
        for prefix in ALLOWED_TREE_PREFIXES:
            if parts[: len(prefix)] == prefix and len(parts) > len(prefix):
                if not str(PurePosixPath(*parts)).lower().endswith((".json", ".txt")):
                    return None
                boundary_parts = prefix
                break

    # Backups made by older versions placed flat files at the ZIP root.
    if boundary_parts is None and len(parts) == 1:
        if parts[0] == "prompt_settings.json":
            parts = ("config", parts[0])
            boundary_parts = ("config",)
        elif parts[0] in USER_FILES:
            parts = ("user_data", parts[0])
            boundary_parts = ("user_data",)
    if boundary_parts is None:
        return None

    boundary = root.joinpath(*boundary_parts).resolve(strict=False)
    if not _is_within(boundary, root):
        raise SettingsBackupError(f"backup boundary escapes project storage: {archive_name!r}")

    lexical_target = root.joinpath(*parts)
    if lexical_target.is_symlink():
        raise SettingsBackupError(f"backup target is a symbolic link: {archive_name!r}")
    target = lexical_target.resolve(strict=False)
    if not _is_within(target, boundary):
        raise SettingsBackupError(f"backup target escapes its storage boundary: {archive_name!r}")
    return ImportTarget(str(PurePosixPath(*parts)), target, boundary)


def _iter_tree_files(root: Path, source_parts: tuple[str, ...]):
    source_root = root.joinpath(*source_parts)
    if not source_root.is_dir() or source_root.is_symlink():
        return
    resolved_source = source_root.resolve(strict=True)
    if not _is_within(resolved_source, root):
        return
    for folder, directories, filenames in os.walk(resolved_source, followlinks=False):
        folder_path = Path(folder)
        directories[:] = [
            name for name in directories
            if not (folder_path / name).is_symlink()
        ]
        for filename in filenames:
            source = folder_path / filename
            if source.is_symlink() or source.suffix.lower() not in {".json", ".txt"}:
                continue
            yield source, source.relative_to(resolved_source).as_posix()


def export_settings_archive(
    destination: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    include_chat: bool = False,
) -> int:
    """설정·사용자 데이터를 ZIP 으로. include_chat 이면 대화 기록(chat_threads.json)도 담는다."""
    root = Path(project_root).expanduser().resolve(strict=False)
    exported = 0
    config_files = CONFIG_EXPORT_FILES + (OPTIONAL_CONFIG_EXPORT_FILES if include_chat else ())
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in config_files:
            source = root / "config" / filename
            resolved = source.resolve(strict=False)
            if (
                source.is_file()
                and not source.is_symlink()
                and _is_within(resolved, root)
            ):
                archive.write(source, f"config/{filename}")
                exported += 1
        for filename in USER_FILES:
            source = root / "user_data" / filename
            resolved = source.resolve(strict=False)
            if (
                source.is_file()
                and not source.is_symlink()
                and _is_within(resolved, root)
            ):
                archive.write(source, f"user_data/{filename}")
                exported += 1
        for source_parts, archive_parts in DIRECTORY_EXPORTS:
            for source, relative in _iter_tree_files(root, source_parts) or ():
                archive_name = PurePosixPath(*archive_parts, relative).as_posix()
                archive.write(source, archive_name)
                exported += 1
    return exported


#: 게시 전에 손보는 복원 항목. 조건식은 가져온 시각을 updatedAt 으로 찍는다 — 백업 안의 시각(내보낸
#: 때, 옛 백업은 없음)을 그대로 두면 그 뒤에 편집한 브라우저 캐시(localStorage)가 재시작 부팅에서 더
#: 최신으로 판정돼 가져온 규칙을 조용히 되덮었다(core/cond_rules_store.py, condRulesSource.ts).
COND_RULES_ARCHIVE_NAME = "config/cond_rules.json"


def _prepare_staged_entry(archive_name: str, staged_path: Path) -> None:
    if archive_name == COND_RULES_ARCHIVE_NAME:
        from core.cond_rules_store import stamp_cond_rules_file

        stamp_cond_rules_file(str(staged_path))


def import_settings_archive(
    source_archive: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> int:
    """Validate, stage, and publish allowlisted settings archive entries."""
    root = Path(project_root).expanduser().resolve(strict=False)
    staged: list[tuple[Path, ImportTarget]] = []
    try:
        with zipfile.ZipFile(source_archive, "r") as archive:
            pending: list[tuple[zipfile.ZipInfo, ImportTarget]] = []
            seen_targets: set[str] = set()
            total_size = 0
            for info in archive.infolist():
                if info.is_dir():
                    continue
                target = resolve_import_target(root, info.filename)
                if target is None:
                    continue
                if info.file_size < 0 or info.file_size > MAX_BACKUP_FILE_BYTES:
                    raise SettingsBackupError(
                        f"backup entry is too large: {info.filename!r}"
                    )
                total_size += info.file_size
                if total_size > MAX_BACKUP_TOTAL_BYTES:
                    raise SettingsBackupError("backup archive exceeds the total size limit")
                target_key = os.path.normcase(str(target.target))
                if target_key in seen_targets:
                    raise SettingsBackupError(
                        f"backup contains duplicate targets: {info.filename!r}"
                    )
                seen_targets.add(target_key)
                pending.append((info, target))

            for info, target in pending:
                target.target.parent.mkdir(parents=True, exist_ok=True)
                checked = resolve_import_target(root, target.archive_name)
                if checked is None or checked.target != target.target:
                    raise SettingsBackupError(
                        f"backup target changed during import: {info.filename!r}"
                    )
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{target.target.name}.import-",
                    dir=target.target.parent,
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                try:
                    copied = 0
                    with archive.open(info) as src, temporary.open("wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            copied += len(chunk)
                            if copied > MAX_BACKUP_FILE_BYTES:
                                raise SettingsBackupError(
                                    f"backup entry expanded past its limit: {info.filename!r}"
                                )
                            dst.write(chunk)
                        dst.flush()
                        os.fsync(dst.fileno())
                    _prepare_staged_entry(checked.archive_name, temporary)
                    staged.append((temporary, checked))
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise

        for temporary, target in staged:
            checked = resolve_import_target(root, target.archive_name)
            if checked is None or checked.target != target.target:
                raise SettingsBackupError(
                    f"backup target changed before publication: {target.archive_name!r}"
                )
            os.replace(temporary, target.target)
        return len(staged)
    finally:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)


__all__ = [
    "CONFIG_EXPORT_FILES",
    "OPTIONAL_CONFIG_EXPORT_FILES",
    "USER_FILES",
    "MAX_BACKUP_FILE_BYTES",
    "MAX_BACKUP_TOTAL_BYTES",
    "SettingsBackupError",
    "export_settings_archive",
    "import_settings_archive",
    "resolve_import_target",
]
