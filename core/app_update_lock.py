"""Process-owned lock for the detached application updater.

Windows uses a named kernel mutex so an interrupted updater can never leave a
stale lock behind.  The small POSIX fallback uses ``flock`` with the same
process-lifetime semantics for development and tests on other platforms.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import IO, Any

from core.storage_paths import PROJECT_ROOT


MUTEX_NAME = r"Local\AIStudioPro.UR_IV.Update"
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102
_SYNCHRONIZE = 0x00100000
_MUTEX_MODIFY_STATE = 0x0001


class UpdateLockBusy(RuntimeError):
    """Raised when another updater owns the process lock."""


class UpdateLock:
    """An acquired update lock that must be held until relaunch is scheduled."""

    def __init__(self, *, handle: int | None = None, stream: IO[bytes] | None = None):
        self._handle = handle
        self._stream = stream
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._handle is not None:
            import ctypes

            ctypes.windll.kernel32.ReleaseMutex(self._handle)
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._stream is not None:
            try:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            finally:
                self._stream.close()
                self._stream = None

    def __enter__(self) -> "UpdateLock":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


def _windows_acquire(timeout: float) -> UpdateLock:
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
    if not handle:
        raise OSError("update mutex could not be created")
    timeout_ms = max(0, min(0xFFFFFFFE, int(timeout * 1000)))
    result = int(kernel32.WaitForSingleObject(handle, timeout_ms))
    if result not in {_WAIT_OBJECT_0, _WAIT_ABANDONED}:
        kernel32.CloseHandle(handle)
        if result == _WAIT_TIMEOUT:
            raise UpdateLockBusy("another updater owns the update mutex")
        raise OSError(f"update mutex wait failed: {result}")
    return UpdateLock(handle=int(handle))


def _posix_acquire(project_root: str | os.PathLike[str], timeout: float) -> UpdateLock:
    import fcntl

    lock_path = Path(project_root).resolve() / "cache" / "updates" / "install.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return UpdateLock(stream=stream)
        except BlockingIOError:
            if time.monotonic() >= deadline:
                stream.close()
                raise UpdateLockBusy("another updater owns the update lock")
            time.sleep(0.05)


def acquire_update_lock(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    timeout: float = 0.0,
) -> UpdateLock:
    """Acquire the updater lock, automatically released if this process dies."""

    if os.name == "nt":
        return _windows_acquire(timeout)
    return _posix_acquire(project_root, timeout)


def is_update_in_progress(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
) -> bool:
    """Return whether another process currently owns the update lock."""

    try:
        lock = acquire_update_lock(project_root, timeout=0.0)
    except UpdateLockBusy:
        return True
    lock.release()
    return False


__all__ = [
    "MUTEX_NAME",
    "UpdateLock",
    "UpdateLockBusy",
    "acquire_update_lock",
    "is_update_in_progress",
]
