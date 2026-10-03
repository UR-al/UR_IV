"""Bounded, session-only undo snapshots for history's recycle-bin operation.

The original still goes through send2trash. Undo restores a verified snapshot;
it does not remove or guess which item belongs to us in the OS recycle bin.
Only server-issued opaque tokens can select an original destination.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import secrets
import shutil
import stat
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from core.path_safety import resolve_missing_input_path, safe_input_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Snapshot:
    local_path: str
    request_path: str
    snapshot: Path
    size: int
    digest: str
    allowed_exts: frozenset[str] | None
    mode: int
    atime_ns: int
    mtime_ns: int


class ImageTrashUndoManager:
    """At most 32 images / 256 MiB; evicted originals remain in the recycle bin.

    Preparation failure never authorizes deletion. Snapshot files live in a
    private TemporaryDirectory, are removed on eviction/undo/normal shutdown,
    and have no persistent manifest or token that can survive a new session.
    """

    def __init__(self, *, max_entries: int = 32, max_bytes: int = 256 * 1024 * 1024):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError('undo limits must be positive')
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._entries: OrderedDict[str, _Snapshot] = OrderedDict()
        self._bytes = 0
        self._cleanup_failed = False
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            self._cleanup_failed = False

    @staticmethod
    def _identity(stat: os.stat_result) -> tuple:
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _failure(message: str) -> dict:
        return {'ok': False, 'removed': False, 'level': 'error', 'message': message}

    def _discard(self, token: str) -> None:
        entry = self._entries[token]
        # Keep accounting/token if removal fails; never silently exceed the cap.
        entry.snapshot.unlink(missing_ok=True)
        del self._entries[token]
        self._bytes -= entry.size

    def _remove_snapshot(self, snapshot: Path) -> None:
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            # No further snapshots until cleanup succeeds; otherwise orphaned
            # files could silently grow beyond the session's disk budget.
            self._cleanup_failed = True

    def move_to_trash(
        self, local_path: str, *, request_path: str,
        allowed_exts: frozenset[str] | None,
        trash: Callable[[str], dict],
    ) -> dict:
        """Snapshot first, then invoke the existing trash boundary exactly once."""
        with self._lock:
            if self._cleanup_failed:
                return self._failure('되돌리기 임시 파일을 정리하지 못했습니다. 앱을 다시 시작한 뒤 시도해 주세요')
            clean = safe_input_path(local_path, allowed_exts=allowed_exts)
            if not clean:
                return self._failure('허용되지 않은 이미지 경로입니다')
            snapshot = None
            try:
                source = Path(clean)
                initial = source.stat()
                if initial.st_size > self.max_bytes:
                    return self._failure('되돌리기 보관 한도를 초과하여 파일을 이동하지 않았습니다')
                if self._temporary is None:
                    self._temporary = tempfile.TemporaryDirectory(prefix='aistudio-history-undo-')
                directory = Path(self._temporary.name)
                # Evict first so retained + in-progress data never exceeds the cap.
                while self._entries and (
                    len(self._entries) >= self.max_entries or self._bytes + initial.st_size > self.max_bytes
                ):
                    self._discard(next(iter(self._entries)))
                if shutil.disk_usage(directory).free < initial.st_size + 1024 * 1024:
                    return self._failure('되돌리기 사본을 보관할 공간이 부족하여 파일을 이동하지 않았습니다')
                token = secrets.token_urlsafe(32)
                snapshot = directory / token
                digest = hashlib.sha256()
                copied = 0
                with source.open('rb') as reader, snapshot.open('xb') as writer:
                    if self._identity(os.fstat(reader.fileno())) != self._identity(initial):
                        raise OSError('source changed before snapshot')
                    while chunk := reader.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > initial.st_size:
                            raise OSError('source changed during snapshot')
                        writer.write(chunk)
                        digest.update(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
                    if self._identity(os.fstat(reader.fileno())) != self._identity(initial):
                        raise OSError('source changed during snapshot')
                if copied != initial.st_size or self._identity(source.stat()) != self._identity(initial):
                    raise OSError('source changed during snapshot')
                if safe_input_path(clean, allowed_exts=allowed_exts) != clean:
                    raise OSError('source path changed during snapshot')
                # Never copy the original's read-only bit onto the owned cache:
                # Windows then refuses unlink(), including eviction and cleanup.
                # Preserve original metadata separately for the restored file.
                snapshot.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except (OSError, ValueError):
                if snapshot is not None:
                    self._remove_snapshot(snapshot)
                return self._failure('되돌리기 사본을 준비하지 못하여 파일을 이동하지 않았습니다')

            entry = _Snapshot(
                clean, str(request_path), snapshot, copied, digest.hexdigest(), allowed_exts,
                stat.S_IMODE(initial.st_mode), initial.st_atime_ns, initial.st_mtime_ns,
            )
            try:
                result = trash(clean)
            except Exception:
                logger.exception('History trash operation raised unexpectedly')
                if source.exists():
                    self._remove_snapshot(snapshot)
                    return self._failure('휴지통으로 옮기지 못했습니다. 원본 파일은 유지됩니다')
                # The production boundary converts all exceptions to results.
                # For a broken replacement adapter whose outcome is unknown,
                # retain/account the recovery copy rather than leaking or deleting
                # it. Do not invent a successful trash outcome or perform rollback.
                self._entries[token] = entry
                self._bytes += copied
                return self._failure('이동 결과를 확인하지 못했습니다. 원래 경로와 휴지통을 확인해 주세요')
            if result.get('ok') and result.get('removed'):
                self._entries[token] = entry
                self._bytes += copied
                return {**result, 'undo_token': token}
            self._remove_snapshot(snapshot)
            return result

    def restore(self, token: str) -> dict:
        """Restore one session token, without ever overwriting an existing path."""
        with self._lock:
            key = token if isinstance(token, str) else ''
            entry = self._entries.get(key)
            result = {'undo_token': key, 'path': entry.request_path if entry else '',
                      'ok': False, 'restored': False, 'level': 'warning', 'retryable': False}
            if entry is None:
                return {**result, 'message': '되돌리기 보관 기간이 지났습니다. 휴지통에서 복원해 주세요'}
            destination = Path(entry.local_path)
            if destination.exists() or destination.is_symlink():
                return {**result, 'retryable': True,
                        'message': '원래 위치에 파일이 있어 복원하지 않았습니다. 기존 파일을 먼저 옮겨 주세요'}
            clean = resolve_missing_input_path(entry.local_path, allowed_exts=entry.allowed_exts)
            if clean != entry.local_path or not destination.parent.is_dir():
                return {**result, 'retryable': True, 'message': '원래 폴더 경로를 확인할 수 없어 복원하지 않았습니다'}
            created_identity = None
            try:
                digest = hashlib.sha256()
                with entry.snapshot.open('rb') as reader:
                    while chunk := reader.read(1024 * 1024):
                        digest.update(chunk)
                if entry.snapshot.stat().st_size != entry.size or digest.hexdigest() != entry.digest:
                    return {**result, 'message': '되돌리기 사본을 확인할 수 없습니다. 휴지통에서 복원해 주세요'}
                # Exclusive creation is the final, race-safe no-overwrite guard.
                with entry.snapshot.open('rb') as reader, destination.open('xb') as writer:
                    created_identity = self._identity(os.fstat(writer.fileno()))[:2]
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fsync(writer.fileno())
                if destination.is_symlink() or self._identity(destination.stat())[:2] != created_identity:
                    raise OSError('restored destination changed')
                # Windows Python 3.11 does not support utime(follow_symlinks=False).
                # This is the just-created, identity-checked regular output.
                os.utime(destination, ns=(entry.atime_ns, entry.mtime_ns))
                destination.chmod(entry.mode)
            except (OSError, ValueError):
                # Only our own partial output may be removed; keep snapshot/token.
                if created_identity is not None:
                    try:
                        if self._identity(destination.stat())[:2] == created_identity:
                            destination.unlink()
                    except OSError:
                        pass
                return {**result, 'retryable': True, 'level': 'error',
                        'message': '파일을 복원하지 못했습니다. 경로와 여유 공간을 확인한 뒤 다시 시도해 주세요'}
            # Output is complete. Cleanup failure must not turn a successful restore
            # into a retry that could duplicate or overwrite the recovered image.
            try:
                self._discard(key)
            except OSError:
                self._cleanup_failed = True
                del self._entries[key]
                self._bytes -= entry.size
            return {**result, 'ok': True, 'restored': True, 'level': 'info',
                    'message': '이미지를 원래 위치로 복원했습니다'}
