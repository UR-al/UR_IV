"""Small process registry used to keep source updates away from live app peers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from core.storage_paths import PROJECT_ROOT
from utils.atomic_json import atomic_write_json, load_json_safe

#: Windows 작업 표시줄 그룹/고정용 AppUserModelID — 두 엔트리포인트(new_main_ui/web_main_ui)가
#: 같은 값을 써야 한 앱으로 묶인다. (예전 예제 문자열 'mycompany.myproduct...'를 대체)
APP_USER_MODEL_ID = "URIV.AIStudioPro"


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        process = kernel32.OpenProcess(0x00100000, False, pid)
        if not process:
            return False
        kernel32.CloseHandle(process)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_identity(pid: int) -> str:
    """Return a PID-reuse-resistant process start token when available."""

    if pid <= 0:
        return ""
    if os.name == "nt":
        import ctypes

        class FILETIME(ctypes.Structure):
            _fields_ = (("low", ctypes.c_ulong), ("high", ctypes.c_ulong))

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
        kernel32.GetProcessTimes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        process = kernel32.OpenProcess(0x00100000, False, pid)
        if not process:
            return ""
        creation, exit_time, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        try:
            if not kernel32.GetProcessTimes(
                process,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return ""
            return str((int(creation.high) << 32) | int(creation.low))
        finally:
            kernel32.CloseHandle(process)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return fields[21] if len(fields) > 21 else ""
    except (OSError, IndexError):
        return ""


def _registry_dir(project_root: str | os.PathLike[str]) -> Path:
    return Path(project_root).resolve() / "cache" / "app_instances"


def register_app_instance(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    pid: int | None = None,
    update_guarded: bool = False,
) -> Path:
    lock = None
    if not update_guarded:
        from core.app_update_lock import acquire_update_lock

        lock = acquire_update_lock(project_root, timeout=120.0)
    try:
        process_id = int(pid or os.getpid())
        directory = _registry_dir(project_root)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / f"{process_id}.json"
        atomic_write_json(str(marker), {
            "pid": process_id,
            "processStarted": process_start_identity(process_id),
            "startedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        })
        return marker
    finally:
        if lock is not None:
            lock.release()


def unregister_app_instance(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    pid: int | None = None,
) -> None:
    marker = _registry_dir(project_root) / f"{int(pid or os.getpid())}.json"
    try:
        marker.unlink(missing_ok=True)
    except OSError:
        pass


def live_app_instance_pids(
    project_root: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    exclude_pid: int | None = None,
) -> list[int]:
    directory = _registry_dir(project_root)
    if not directory.is_dir():
        return []
    live: list[int] = []
    for marker in directory.glob("*.json"):
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        if pid == exclude_pid:
            continue
        marker_data = load_json_safe(str(marker), {})
        expected_start = (
            str(marker_data.get("processStarted") or "")
            if isinstance(marker_data, dict)
            else ""
        )
        actual_start = process_start_identity(pid) if expected_start else ""
        if process_exists(pid) and (not expected_start or actual_start == expected_start):
            live.append(pid)
            continue
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
    return sorted(set(live))


__all__ = [
    "APP_USER_MODEL_ID",
    "live_app_instance_pids",
    "process_exists",
    "process_start_identity",
    "register_app_instance",
    "unregister_app_instance",
]
