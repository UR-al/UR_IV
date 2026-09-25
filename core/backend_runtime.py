"""App-owned Forge Neo and ComfyUI runtime lifecycle management.

This module deliberately sits above :mod:`backends`: the existing backend
classes own the HTTP generation protocol, while ``BackendRuntimeManager`` owns
only app-managed source trees, virtual environments and child processes.

The public interface is intentionally small: ``snapshot()``, ``configure()``
and ``execute()``.  Git, package installation, process creation and readiness
checks remain replaceable behind ``LocalRuntimeAdapter`` for deterministic
tests.
"""
from __future__ import annotations

import json
import os
import shlex
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse, urlsplit, urlunsplit

from core.backend_probe import HEALTH_PATHS
from core.process_guard import app_job, clear_pid_file, marker_for, sweep_orphan, write_pid_file
from core.launch_args import LaunchArgsImport, discover_launch_args, merge_args
from core.storage_paths import config_file
from utils.atomic_json import atomic_write_json, load_json_safe


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = config_file(
    "backend_runtime.json",
    legacy_paths="user_data/backend_runtime.json",
)

FORGE_REPOSITORY = "https://github.com/Haoming02/sd-webui-forge-classic.git"
COMFYUI_REPOSITORY = "https://github.com/Comfy-Org/ComfyUI.git"

ENGINE_ALIASES = {
    "forge": "forge",
    "forge_neo": "forge",
    "forge-neo": "forge",
    "comfy": "comfyui",
    "comfyui": "comfyui",
}


@dataclass(frozen=True)
class EngineDefinition:
    key: str
    name: str
    repository: str
    branch: str
    protocol: str
    preferred_port: int
    health_path: str
    extension_folder: str
    entrypoint: str


@dataclass(frozen=True)
class RuntimeLocation:
    """Resolved executable layout for one managed or linked installation.

    ``install_root`` is the outer installation boundary selected/detected for
    display.  ``source_root`` is the directory containing the entrypoint and
    ``python_path`` is the interpreter used to launch it.  Linked locations are
    only inspected; every file this module writes remains under ``runtime_root``.
    """

    install_root: Path
    source_root: Path
    python_path: Path
    portable: bool = False

    @property
    def valid(self) -> bool:
        return self.source_root.is_dir() and self.python_path.is_file()


ENGINE_DEFINITIONS: dict[str, EngineDefinition] = {
    "forge": EngineDefinition(
        key="forge",
        name="Forge Neo",
        repository=FORGE_REPOSITORY,
        branch="neo",
        protocol="webui",
        preferred_port=17860,
        health_path=HEALTH_PATHS["forge"],
        extension_folder="extensions",
        entrypoint="launch.py",
    ),
    "comfyui": EngineDefinition(
        key="comfyui",
        name="ComfyUI",
        repository=COMFYUI_REPOSITORY,
        branch="master",
        protocol="comfyui",
        preferred_port=18188,
        health_path=HEALTH_PATHS["comfyui"],
        extension_folder="custom_nodes",
        entrypoint="main.py",
    ),
}


#: execute 결과의 ``state`` 를 마지막에 한 번 계산하는 스냅샷으로 채우라는 표시.
_FINAL_SNAPSHOT = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_runtime_root() -> Path:
    override = str(os.environ.get("AISTUDIO_MANAGED_BACKENDS_DIR", "") or "").strip()
    if override:
        return Path(os.path.abspath(os.path.expandvars(os.path.expanduser(override))))
    return PROJECT_ROOT / "user_data" / "managed_backends"


def _canonical_engine(value: str) -> str:
    key = ENGINE_ALIASES.get(str(value or "").strip().casefold(), "")
    if not key:
        raise BackendRuntimeError("INVALID_ENGINE", f"지원하지 않는 엔진입니다: {value}")
    return key


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class BackendRuntimeError(RuntimeError):
    """Structured lifecycle error safe to send across QWebChannel."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ):
        self.code = str(code)
        self.stage = str(stage or "")
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        super().__init__(str(message))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "retryable": self.retryable,
            "message": str(self),
            "details": self.details,
        }


@dataclass
class CommandResult:
    returncode: int
    output: str = ""


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


class RuntimeAdapter(Protocol):
    def which(self, executable: str) -> str | None: ...

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult: ...

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> ProcessHandle: ...

    def probe(self, url: str, path: str, timeout: float = 2.0) -> bool: ...

    def port_available(self, host: str, port: int) -> bool: ...


class LocalRuntimeAdapter:
    """Production adapter for local commands, processes and loopback HTTP."""

    def __init__(self) -> None:
        self._command_lock = threading.Lock()
        self._commands: set[subprocess.Popen] = set()
        self._shutdown = threading.Event()

    @staticmethod
    def _terminate_command(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            # The PID is taken only from a live Popen created by this adapter.
            # /T also stops pip/git/bootstrap descendants instead of orphaning them.
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        else:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Cancel command trees created by this adapter; never consult PID files."""
        self._shutdown.set()
        with self._command_lock:
            commands = tuple(self._commands)
        for process in commands:
            self._terminate_command(process)

    def which(self, executable: str) -> str | None:
        return shutil.which(executable)

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        if self._shutdown.is_set():
            raise BackendRuntimeError(
                "COMMAND_CANCELLED", "앱 종료 요청으로 명령 실행을 취소했습니다", stage="command"
            )
        command = [str(item) for item in argv]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd else None,
                env=dict(env) if env is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
        except OSError as exc:
            raise BackendRuntimeError(
                "COMMAND_START_FAILED",
                f"명령을 시작하지 못했습니다: {command[0]}",
                stage="command",
                details={"reason": str(exc)},
            ) from exc
        with self._command_lock:
            self._commands.add(process)

        try:
            if self._shutdown.is_set():
                self._terminate_command(process)
                raise BackendRuntimeError(
                    "COMMAND_CANCELLED", "앱 종료 요청으로 명령 실행을 취소했습니다",
                    stage="command",
                )

            lines: list[str] = []
            started = time.monotonic()
            output_queue: queue.Queue[str | None] = queue.Queue()
            assert process.stdout is not None

            def _read_output() -> None:
                try:
                    for raw_line in iter(process.stdout.readline, ""):
                        output_queue.put(raw_line)
                finally:
                    output_queue.put(None)

            reader = threading.Thread(target=_read_output, daemon=True, name="backend-command-output")
            reader.start()
            stream_done = False
            while not stream_done or process.poll() is None:
                try:
                    line = output_queue.get(timeout=0.1)
                except queue.Empty:
                    line = ""
                if line is None:
                    stream_done = True
                elif line:
                    clean = line.rstrip("\r\n")
                    lines.append(clean)
                    if len(lines) > 500:
                        del lines[:100]
                    if on_line:
                        on_line(clean)
                if self._shutdown.is_set():
                    self._terminate_command(process)
                    raise BackendRuntimeError(
                        "COMMAND_CANCELLED", "앱 종료 요청으로 명령 실행을 취소했습니다",
                        stage="command",
                    )
                if timeout is not None and time.monotonic() - started > timeout:
                    self._terminate_command(process)
                    raise BackendRuntimeError(
                        "COMMAND_TIMEOUT",
                        f"명령 실행 제한 시간({int(timeout)}초)을 초과했습니다",
                        stage="command",
                        retryable=True,
                    )
            reader.join(timeout=1)
            return CommandResult(process.returncode or 0, "\n".join(lines))
        finally:
            with self._command_lock:
                self._commands.discard(process)
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
    ) -> ProcessHandle:
        if self._shutdown.is_set():
            raise BackendRuntimeError(
                "PROCESS_START_CANCELLED", "앱 종료 요청으로 백엔드 실행을 취소했습니다",
                stage="launch",
            )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(
                [str(item) for item in argv],
                cwd=str(cwd),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        except Exception:
            log_handle.close()
            raise
        # Keep the file object alive with the process without exposing it publicly.
        setattr(process, "_aistudio_log_handle", log_handle)
        return process

    def probe(self, url: str, path: str, timeout: float = 2.0) -> bool:
        # 시작 게이트와 같은 probe — 루프백은 connect 만 짧게 끊는다(core/backend_probe.py).
        from core.backend_probe import probe_url

        return probe_url(url, path, timeout=timeout)

    def port_available(self, host: str, port: int) -> bool:
        # Windows 는 SO_EXCLUSIVEADDRUSE 로 host 와 와일드카드를 모두 시험한다 — core/port_probe.py
        from core.port_probe import port_available

        return port_available(host, port)


@dataclass
class _OwnedProcess:
    process: ProcessHandle
    endpoint: str
    port: int
    nonce: str
    started_at: str
    log_path: Path


ProgressCallback = Callable[[dict[str, Any]], None]


class BackendRuntimeManager:
    """Deep module owning both managed engine lifecycles behind one interface."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        config_path: Path | str | None = None,
        runtime_root: Path | str | None = None,
        adapter: RuntimeAdapter | None = None,
        health_timeout: float = 180.0,
        reserved_ports: Callable[[], Any] | None = None,
    ):
        self.config_path = Path(config_path or CONFIG_PATH)
        self.runtime_root = Path(runtime_root or _default_runtime_root()).resolve()
        self.adapter: RuntimeAdapter = adapter or LocalRuntimeAdapter()
        self.health_timeout = float(health_timeout)
        # 앱의 다른 구성요소가 쓰기로 한 포트(아직 bind 전일 수 있음)를 돌려주는 함수.
        # 앱 싱글턴은 켜 둔 Generation API 포트를 넘긴다 — get_backend_runtime_manager.
        self._reserved_ports_source = reserved_ports
        self._state_lock = threading.RLock()
        # An operation for either engine may stop/restart the other engine.  Keep
        # the entire managed-runtime transaction linearizable across engines;
        # per-engine locks alone allow UPDATE/extension restart to interleave with
        # a cross-engine START and leave the app pointing at a stopped endpoint.
        self._operation_gate = threading.Lock()
        self._operation_locks = {key: threading.Lock() for key in ENGINE_DEFINITIONS}
        self._process_lock = threading.RLock()
        self._start_gate = threading.Lock()
        self._shutdown_requested = threading.Event()
        self._processes: dict[str, _OwnedProcess] = {}
        self._start_cancel = {key: threading.Event() for key in ENGINE_DEFINITIONS}
        self._healthy: dict[str, bool] = {key: False for key in ENGINE_DEFINITIONS}
        self._busy: dict[str, bool] = {key: False for key in ENGINE_DEFINITIONS}
        self._last_messages: dict[str, str] = {key: "" for key in ENGINE_DEFINITIONS}
        # 이전 release(.trash-*) 백그라운드 삭제 스레드 — core/release_trash.py
        self._release_purge_thread: threading.Thread | None = None
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._cleanup_inactive_releases()
        self._ensure_primary_model_engine()

    # ------------------------------------------------------------------ store

    def _default_engine_state(self, definition: EngineDefinition) -> dict[str, Any]:
        return {
            "release": "",
            "sourceMode": "managed",
            "existingRoot": "",
            "autoStart": False,
            "port": definition.preferred_port,
            "extensionDir": "",
            "extensionDirApproved": False,
            "remoteVersion": "",
            "remoteCommit": "",
            "lastChecked": "",
            "lastError": "",
            # 사용자가 Settings 에서 덧붙이는 실행 인자 (예: "--xformers --medvram").
            # 문자열 하나로 저장하고 기동 직전에 shlex 로 쪼갠다 — 따옴표 경로도 산다.
            "extraArgs": "",
            # 연결한 설치의 실행 배치 파일(webui-user.bat / run_nvidia_gpu*.bat) 인자를 자동으로 가져올지.
            "importLaunchArgs": True,
            # ComfyUI 만: `--fast fp16_accumulation` (run_nvidia_gpu_fast_fp16_accumulation.bat 와 같은 실행).
            "fastFp16": False,
        }

    def _load_state(self) -> dict[str, Any]:
        raw = load_json_safe(str(self.config_path), {})
        if not isinstance(raw, dict):
            raw = {}
        engines = raw.get("engines")
        if not isinstance(engines, dict):
            engines = {}
        normalised: dict[str, Any] = {
            "schemaVersion": self.SCHEMA_VERSION,
            "activeEngine": ENGINE_ALIASES.get(
                str(raw.get("activeEngine", "") or "").casefold(), ""
            ),
            "primaryModelEngine": ENGINE_ALIASES.get(
                str(raw.get("primaryModelEngine", "forge") or "forge").casefold(),
                "forge",
            ),
            "engines": {},
        }
        for key, definition in ENGINE_DEFINITIONS.items():
            item = self._default_engine_state(definition)
            saved = engines.get(key)
            if not isinstance(saved, dict) and key == "forge":
                saved = engines.get("forge_neo")
            if isinstance(saved, dict):
                for field in item:
                    if field in saved:
                        item[field] = saved[field]
            source_mode = str(item.get("sourceMode", "managed") or "managed").casefold()
            if source_mode not in {"managed", "existing"}:
                source_mode = "managed"
                item["lastError"] = "유효하지 않은 설치 소스 설정을 무시했습니다"
            item["sourceMode"] = source_mode
            existing_root = str(item.get("existingRoot", "") or "").strip()
            if existing_root:
                try:
                    item["existingRoot"] = str(
                        self._normalise_existing_root(existing_root)
                    )
                except BackendRuntimeError:
                    item["existingRoot"] = ""
                    item["sourceMode"] = "managed"
                    if not bool(item.get("extensionDirApproved")):
                        item["extensionDir"] = ""
                    item["lastError"] = "유효하지 않거나 안전하지 않은 기존 설치 설정을 무시했습니다"
            elif source_mode == "existing":
                item["sourceMode"] = "managed"
            try:
                item["port"] = int(item["port"])
            except (TypeError, ValueError):
                item["port"] = definition.preferred_port
            if not 1 <= item["port"] <= 65535:
                item["port"] = definition.preferred_port
            release = str(item.get("release", "") or "").strip()
            if release and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release):
                release = ""
                item["lastError"] = "안전하지 않은 runtime release 설정을 무시했습니다"
            item["release"] = release
            extension_dir = str(item.get("extensionDir", "") or "").strip()
            saved_has_approval = isinstance(saved, dict) and "extensionDirApproved" in saved
            if extension_dir and not saved_has_approval:
                # Before schema v2 every non-empty value came from the explicit
                # save-extension-dir flow, so preserve that user's approval.
                item["extensionDirApproved"] = True
            if extension_dir:
                try:
                    extension_path = Path(
                        os.path.abspath(
                            os.path.expandvars(os.path.expanduser(extension_dir))
                        )
                    ).resolve()
                    item["extensionDir"] = str(
                        self._validate_extension_root(
                            extension_dir,
                            create_managed=_is_relative_to(extension_path, self.runtime_root),
                            engine=key,
                            require_exists=False,
                            require_writable=bool(item.get("extensionDirApproved")),
                        )
                    )
                except BackendRuntimeError:
                    item["extensionDir"] = ""
                    item["extensionDirApproved"] = False
                    item["lastError"] = "유효하지 않은 확장 폴더 설정을 무시했습니다"
            item["extensionDirApproved"] = bool(item.get("extensionDirApproved"))
            item["autoStart"] = bool(item["autoStart"])
            item["importLaunchArgs"] = bool(item.get("importLaunchArgs", True))
            item["fastFp16"] = bool(item.get("fastFp16", False))
            try:
                item["extraArgs"] = self._normalise_extra_args(item.get("extraArgs", ""))
            except BackendRuntimeError:
                item["extraArgs"] = ""
                item["lastError"] = "해석할 수 없는 실행 인자 설정을 무시했습니다"
            normalised["engines"][key] = item
        forge_extension = Path(
            normalised["engines"]["forge"].get("extensionDir")
            or self._default_extension_root("forge")
        )
        comfy_extension = Path(
            normalised["engines"]["comfyui"].get("extensionDir")
            or self._default_extension_root("comfyui")
        )
        if self._extension_roots_overlap(forge_extension, comfy_extension):
            normalised["engines"]["comfyui"]["extensionDir"] = ""
            normalised["engines"]["comfyui"]["extensionDirApproved"] = False
            normalised["engines"]["comfyui"]["lastError"] = (
                "Forge와 겹치는 ComfyUI 확장 폴더 설정을 무시했습니다"
            )
        auto_engines = [
            key for key in ENGINE_DEFINITIONS
            if normalised["engines"][key].get("autoStart")
        ]
        if len(auto_engines) > 1:
            preferred = normalised.get("activeEngine")
            keep = preferred if preferred in auto_engines else auto_engines[0]
            for key in auto_engines:
                normalised["engines"][key]["autoStart"] = key == keep
        return normalised

    def _save_state(self) -> None:
        with self._state_lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(str(self.config_path), self._state, indent=2)

    def _cleanup_inactive_releases(self) -> None:
        """Retire unreferenced app-owned release trees left by updates/crashes.

        생성자(시작 시 GUI 스레드)에서 불린다. 수 GB 트리를 여기서 지우지 않고
        ``.trash-*`` 로 이름만 바꾼 뒤 데몬 스레드가 지운다 — core/release_trash.py.
        이름을 바꾸는 이 단계는 동기라, 이후 설치 작업이 만드는 후보 release 와
        삭제 스레드가 겹치지 않는다(삭제 스레드는 ``.trash-*`` 만 지운다).
        """
        from core.release_trash import is_trash, move_to_trash

        roots: list[Path] = []
        for engine in ENGINE_DEFINITIONS:
            releases_root = (self._engine_root(engine) / "releases").resolve()
            if not releases_root.is_dir():
                continue
            roots.append(releases_root)
            active_id = str(self._state["engines"][engine].get("release", "") or "")
            active = (releases_root / active_id).resolve() if active_id else None
            try:
                children = tuple(releases_root.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    if is_trash(child) or not child.is_dir():
                        continue
                    resolved = child.resolve()
                    if active is not None and resolved == active:
                        continue
                    if not _is_relative_to(resolved, releases_root):
                        continue
                    move_to_trash(resolved, releases_root)
                except OSError:
                    continue
        self._purge_retired_releases(roots)

    def _purge_retired_releases(self, roots: list[Path]) -> None:
        from core.release_trash import purge_trash_async

        thread = purge_trash_async(roots)
        if thread is not None:
            self._release_purge_thread = thread

    def _retire_release(self, engine: str, release_id: str) -> None:
        """업데이트로 비활성이 된 이전 release 를 즉시 치운다(삭제는 백그라운드)."""
        from core.release_trash import move_to_trash

        release_id = str(release_id or "")
        if not release_id or release_id == str(self._state["engines"][engine].get("release") or ""):
            return
        releases_root = (self._engine_root(engine) / "releases").resolve()
        candidate = (releases_root / release_id).resolve()
        if candidate == releases_root or not _is_relative_to(candidate, releases_root):
            return
        if move_to_trash(candidate, releases_root) is not None:
            self._purge_retired_releases([releases_root])

    # --------------------------------------------------------------- filesystem

    def _engine_root(self, engine: str) -> Path:
        return self.runtime_root / engine

    def _release_root(self, engine: str, release: str | None = None) -> Path:
        release_id = release or str(self._state["engines"][engine].get("release", ""))
        return self._engine_root(engine) / "releases" / release_id

    @staticmethod
    def _environment_python_candidates(root: Path) -> tuple[Path, ...]:
        """Return common venv interpreter locations without assuming host OS."""
        return (
            root / "venv" / "Scripts" / "python.exe",
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "bin" / "python",
            root / ".venv" / "bin" / "python",
        )

    @staticmethod
    def _first_file(candidates: Sequence[Path]) -> Path | None:
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _normalise_existing_root(self, value: str | os.PathLike[str]) -> Path:
        raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
        if not raw or not Path(raw).is_absolute():
            raise BackendRuntimeError(
                "LINKED_INSTALL_INVALID",
                "기존 설치 폴더는 절대 경로여야 합니다",
                stage="configure",
            )
        root = Path(os.path.abspath(os.path.normpath(raw))).resolve()
        if not root.is_dir():
            raise BackendRuntimeError(
                "LINKED_INSTALL_INVALID",
                "지정한 기존 설치 폴더가 존재하지 않습니다",
                stage="configure",
                details={"path": str(root)},
            )
        broad_anchors = {
            Path(root.anchor).resolve(),
            Path.home().resolve(),
            PROJECT_ROOT.resolve(),
            self.runtime_root.resolve(),
        }
        if root in broad_anchors:
            raise BackendRuntimeError(
                "LINKED_INSTALL_PATH_UNSAFE",
                "드라이브 루트·홈·앱 루트·관리형 런타임 루트 자체는 기존 설치 폴더로 지정할 수 없습니다",
                stage="configure",
                details={"path": str(root)},
            )
        return root

    def _detect_existing_install(
        self,
        engine: str,
        value: str | os.PathLike[str],
    ) -> RuntimeLocation:
        """Resolve supported linked layouts without changing the selected tree."""
        root = self._normalise_existing_root(value)
        definition = ENGINE_DEFINITIONS[engine]

        if engine == "forge":
            python = self._first_file(self._environment_python_candidates(root))
            if (root / definition.entrypoint).is_file() and python is not None:
                return RuntimeLocation(root, root, python)
        else:
            # Git/venv layout: the selected directory itself contains main.py.
            direct_python = self._first_file(self._environment_python_candidates(root))
            if (root / definition.entrypoint).is_file() and direct_python is not None:
                return RuntimeLocation(root, root, direct_python)

            # Windows portable can be selected either at the outer bundle or at
            # its inner ComfyUI source directory.  Both historical spellings of
            # python_embeded are accepted.
            portable_pairs: list[tuple[Path, Path]] = []
            inner = root / "ComfyUI"
            if (inner / definition.entrypoint).is_file():
                nested_venv = self._first_file(self._environment_python_candidates(inner))
                if nested_venv is not None:
                    return RuntimeLocation(root, inner.resolve(), nested_venv)
                portable_pairs.extend(
                    (inner, root / folder / "python.exe")
                    for folder in ("python_embeded", "python_embedded")
                )
            if (root / definition.entrypoint).is_file():
                outer = root.parent
                portable_pairs.extend(
                    (root, outer / folder / "python.exe")
                    for folder in ("python_embeded", "python_embedded")
                )
            for source, python in portable_pairs:
                if python.is_file():
                    # 후보 python 은 항상 python_embed(d)ed 폴더 안에 있으므로 번들
                    # 루트는 그 폴더의 부모다(바깥 번들 / 안쪽 ComfyUI 선택 모두).
                    install_root = python.parent.parent
                    return RuntimeLocation(
                        install_root.resolve(), source.resolve(), python.resolve(), portable=True
                    )

        expected = (
            "launch.py와 venv/.venv Python"
            if engine == "forge"
            else "main.py와 .venv/venv Python 또는 Windows portable Python"
        )
        raise BackendRuntimeError(
            "LINKED_INSTALL_INVALID",
            f"{definition.name} 설치 구조를 확인할 수 없습니다 ({expected})",
            stage="configure",
            details={"path": str(root)},
        )

    def _runtime_location(
        self,
        engine: str,
        *,
        require_valid: bool = False,
    ) -> RuntimeLocation | None:
        saved = self._state["engines"][engine]
        if saved.get("sourceMode") == "existing":
            root = str(saved.get("existingRoot", "") or "")
            if root:
                try:
                    return self._detect_existing_install(engine, root)
                except BackendRuntimeError:
                    if require_valid:
                        raise
                    return None
            if require_valid:
                raise BackendRuntimeError(
                    "LINKED_INSTALL_INVALID",
                    "기존 설치 폴더가 지정되지 않았습니다",
                    stage="start",
                )
            return None
        release = str(saved.get("release", "") or "")
        if not release:
            return None
        release_root = self._release_root(engine, release).resolve()
        source = (release_root / "source").resolve()
        python = self._venv_python((release_root / "venv").resolve()).resolve()
        location = RuntimeLocation(release_root, source, python)
        entrypoint = source / ENGINE_DEFINITIONS[engine].entrypoint
        if location.valid and entrypoint.is_file():
            return location
        if require_valid:
            raise BackendRuntimeError(
                "NOT_INSTALLED", "관리형 백엔드 설치가 완전하지 않습니다", stage="start"
            )
        return None

    def _source_root(self, engine: str) -> Path:
        location = self._runtime_location(engine)
        if location is not None:
            return location.source_root
        return self._release_root(engine) / "source"

    def _venv_root(self, engine: str) -> Path:
        return self._release_root(engine) / "venv"

    def _python_path(self, engine: str) -> Path:
        location = self._runtime_location(engine)
        if location is not None:
            return location.python_path
        return self._venv_python(self._venv_root(engine))

    @staticmethod
    def _venv_python(venv_root: Path) -> Path:
        return venv_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _data_root(self, engine: str) -> Path:
        return self._engine_root(engine) / "data"

    def _default_extension_root(self, engine: str) -> Path:
        folder = ENGINE_DEFINITIONS[engine].extension_folder
        return self._engine_root(engine) / "shared" / folder

    def _extension_root(self, engine: str) -> Path:
        configured = str(self._state["engines"][engine].get("extensionDir", "") or "").strip()
        return Path(configured).resolve() if configured else self._default_extension_root(engine).resolve()

    def _extension_mount(self, engine: str) -> Path:
        return self._data_root(engine) / ENGINE_DEFINITIONS[engine].extension_folder

    @staticmethod
    def _extension_roots_overlap(first: Path, second: Path) -> bool:
        first = first.resolve()
        second = second.resolve()
        return _is_relative_to(first, second) or _is_relative_to(second, first)

    def _validate_extension_root(
        self,
        value: str | os.PathLike[str],
        *,
        create_managed: bool = False,
        engine: str | None = None,
        require_exists: bool = True,
        require_writable: bool = True,
    ) -> Path:
        raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
        if not raw or not Path(raw).is_absolute():
            raise BackendRuntimeError(
                "EXTENSION_PATH_INVALID", "확장 폴더는 절대 경로여야 합니다", stage="configure"
            )
        path = Path(os.path.abspath(os.path.normpath(raw))).resolve()
        anchors = {Path(path.anchor).resolve(), Path.home().resolve(), PROJECT_ROOT.resolve()}
        if path in anchors:
            raise BackendRuntimeError(
                "EXTENSION_PATH_UNSAFE", "드라이브 루트·홈·앱 루트는 확장 폴더로 사용할 수 없습니다",
                stage="configure",
            )
        if _is_relative_to(path, PROJECT_ROOT.resolve()) and not _is_relative_to(path, self.runtime_root):
            raise BackendRuntimeError(
                "EXTENSION_PATH_UNSAFE",
                "앱 소스/설정 하위 폴더는 외부 확장 폴더로 사용할 수 없습니다",
                stage="configure",
            )
        if _is_relative_to(path, self.runtime_root):
            allowed = bool(
                engine
                and _is_relative_to(path, (self._engine_root(engine) / "shared").resolve())
            )
            if not allowed:
                raise BackendRuntimeError(
                    "EXTENSION_PATH_UNSAFE",
                    "관리형 런타임의 구조/소스 폴더는 확장 폴더로 사용할 수 없습니다",
                    stage="configure",
                )
        if not path.exists():
            if create_managed and _is_relative_to(path, self.runtime_root):
                path.mkdir(parents=True, exist_ok=True)
            elif require_exists:
                raise BackendRuntimeError(
                    "EXTENSION_PATH_INVALID", "지정한 확장 폴더가 존재하지 않습니다", stage="configure"
                )
            else:
                return path
        if not path.is_dir():
            raise BackendRuntimeError(
                "EXTENSION_PATH_INVALID", "지정한 경로가 폴더가 아닙니다", stage="configure"
            )
        if require_writable and not os.access(path, os.W_OK):
            raise BackendRuntimeError(
                "EXTENSION_PATH_READ_ONLY", "지정한 확장 폴더에 쓸 수 없습니다", stage="configure"
            )
        return path

    @staticmethod
    def _is_reparse_directory(path: Path) -> bool:
        if path.is_symlink():
            return True
        if os.name != "nt":
            return False
        try:
            attrs = os.lstat(path).st_file_attributes
            return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
        except (AttributeError, OSError):
            return False

    def _ensure_extension_mount(self, engine: str) -> Path:
        target = self._extension_root(engine)
        state = self._state["engines"][engine]
        target = self._validate_extension_root(
            target,
            create_managed=_is_relative_to(target.resolve(), self.runtime_root),
            engine=engine,
            require_writable=bool(
                _is_relative_to(target.resolve(), self.runtime_root)
                or state.get("extensionDirApproved")
            ),
        )
        mount = self._extension_mount(engine)
        mount.parent.mkdir(parents=True, exist_ok=True)
        # 대상이 사라진 junction(연결했던 설치를 지우거나 옮긴 경우)은 exists()·is_symlink() 가 모두
        # False 다 — reparse 속성으로도 봐야 지우고 새로 연결한다(안 그러면 그 자리에 만들다 실패).
        if mount.exists() or mount.is_symlink() or self._is_reparse_directory(mount):
            try:
                if mount.resolve() == target.resolve():
                    return target
            except OSError:
                pass
            if self._is_reparse_directory(mount):
                os.rmdir(mount)
            elif mount.is_dir() and not any(mount.iterdir()):
                mount.rmdir()
            else:
                raise BackendRuntimeError(
                    "EXTENSION_MOUNT_CONFLICT",
                    f"관리형 확장 연결 위치에 기존 파일이 있습니다: {mount}",
                    stage="extension_mount",
                )
        if os.name == "nt":
            powershell = self.adapter.which("powershell.exe") or self.adapter.which("pwsh.exe")
            if not powershell:
                raise BackendRuntimeError(
                    "EXTENSION_MOUNT_FAILED",
                    "확장 폴더 연결을 만들 PowerShell을 찾지 못했습니다",
                    stage="extension_mount",
                )
            junction_env = os.environ.copy()
            junction_env["AISTUDIO_JUNCTION_PATH"] = str(mount)
            junction_env["AISTUDIO_JUNCTION_TARGET"] = str(target)
            result = self.adapter.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "$ErrorActionPreference='Stop'; "
                    "New-Item -ItemType Junction -Path $env:AISTUDIO_JUNCTION_PATH "
                    "-Target $env:AISTUDIO_JUNCTION_TARGET | Out-Null",
                ],
                cwd=mount.parent,
                env=junction_env,
                timeout=30,
            )
            if result.returncode != 0 or not mount.exists():
                raise BackendRuntimeError(
                    "EXTENSION_MOUNT_FAILED",
                    "확장 폴더 연결(junction)을 만들지 못했습니다",
                    stage="extension_mount",
                    details={"output": result.output[-1000:]},
                )
        else:
            mount.symlink_to(target, target_is_directory=True)
        return target

    # ---------------------------------------------------------------- snapshot

    # .git 파일 읽기는 core/git_metadata.py 한 곳에 둔다(app_updater 표시용 정체성과 공유).
    @staticmethod
    def _git_dir(source: Path) -> Path | None:
        from core.git_metadata import git_dir

        return git_dir(source)

    @staticmethod
    def _read_git_ref(git_dir: Path, ref: str) -> str:
        from core.git_metadata import read_ref

        return read_ref(git_dir, ref)

    def _local_git_info(self, source: Path) -> tuple[str, str, str]:
        """Read local Git metadata without spawning processes or contacting remotes."""
        from core.git_metadata import head_info

        commit, branch = head_info(source)
        return (commit[:12] if commit else ""), commit, branch

    @staticmethod
    def _declared_local_version(engine: str, source: Path) -> str:
        """Read a bounded local version declaration without importing source code."""
        if engine != "comfyui" or not source.is_dir():
            return ""
        candidates = (
            (source / "comfyui_version.py", r"(?m)^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]"),
            (source / "pyproject.toml", r"(?m)^\s*version\s*=\s*['\"]([^'\"]+)['\"]"),
        )
        for path, pattern in candidates:
            try:
                if not path.is_file() or path.stat().st_size > 256 * 1024:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            match = re.search(pattern, text)
            if match:
                version = match.group(1).strip()
                if re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,63}", version):
                    return version
        return ""

    def _git_origin_url(self, source: Path) -> str:
        from core.git_metadata import remote_url

        return remote_url(source, "origin")

    @staticmethod
    def _sanitize_repository_url(value: str) -> str:
        """Remove credentials/query secrets before state reaches disk, Vue or web mode."""
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parts = urlsplit(raw)
            if parts.scheme and parts.hostname:
                host = parts.hostname
                if ":" in host and not host.startswith("["):
                    host = f"[{host}]"
                try:
                    if parts.port is not None:
                        host = f"{host}:{parts.port}"
                except ValueError:
                    pass
                return urlunsplit((parts.scheme, host, parts.path, "", ""))
        except ValueError:
            pass
        if "@" in raw:
            return raw.rsplit("@", 1)[1]
        return raw.split("?", 1)[0].split("#", 1)[0]

    def _capture(self, argv: Sequence[str], *, cwd: Path | None = None) -> str:
        try:
            result = self.adapter.run(argv, cwd=cwd, timeout=30)
            return result.output.strip().splitlines()[-1] if result.returncode == 0 and result.output.strip() else ""
        except Exception:
            return ""

    def _process_running(self, engine: str) -> bool:
        with self._process_lock:
            owned = self._processes.get(engine)
            if owned is None:
                return False
            if owned.process.poll() is None:
                return True
            self._close_process_log(owned.process)
            self._processes.pop(engine, None)
            self._healthy[engine] = False
            return False

    def _extension_state(self, engine: str) -> list[dict[str, Any]]:
        root = self._extension_root(engine)
        if not root.is_dir():
            return []
        result: list[dict[str, Any]] = []
        try:
            children = sorted(root.iterdir(), key=lambda item: item.name.casefold())
        except OSError:
            return []
        for child in children:
            if not child.is_dir() or child.name.startswith("."):
                continue
            version, commit, _branch = self._local_git_info(child)
            repo_url = self._sanitize_repository_url(self._git_origin_url(child))
            # Snapshot must remain local-only.  A remote commit is only retained by explicit
            # check operations in the extension marker, never fetched here.
            marker = self._extension_marker(engine, child.name)
            marker_data = load_json_safe(str(marker), {}) if marker.is_file() else {}
            cached_remote = ""
            if isinstance(marker_data, dict) and str(marker_data.get("repoUrl") or "") == repo_url:
                cached_remote = str(marker_data.get("remoteCommit", "") or "")
            result.append({
                "id": child.name,
                "name": child.name,
                "path": str(child),
                "repoUrl": repo_url,
                "version": version or (commit[:12] if commit else "local"),
                "commit": commit,
                "remoteCommit": cached_remote,
                "updateAvailable": bool(commit and cached_remote and commit != cached_remote),
                "status": "git" if (child / ".git").is_dir() else "local / version unknown",
                "busy": False,
            })
        return result

    def _extension_marker(self, engine: str, extension_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", extension_id)[:128]
        return self._engine_root(engine) / "extension_state" / f"{safe_id}.json"

    def _linked_extension_root(
        self, engine: str, location: RuntimeLocation
    ) -> Path | None:
        candidate = location.source_root / ENGINE_DEFINITIONS[engine].extension_folder
        return candidate.resolve() if candidate.is_dir() else None

    @staticmethod
    def _append_existing_paths(
        output: list[Path],
        root: Path,
        names: Sequence[str],
    ) -> None:
        if not root.is_dir():
            return
        try:
            children = {
                child.name.casefold(): child.resolve()
                for child in root.iterdir()
                if child.is_dir()
            }
        except OSError:
            return
        for name in names:
            candidate = children.get(name.casefold())
            if candidate is not None:
                output.append(candidate)

    @staticmethod
    def _dedupe_paths(paths: Sequence[Path]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for path in paths:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_dir():
                continue
            marker = os.path.normcase(str(resolved))
            if marker in seen:
                continue
            seen.add(marker)
            result.append(str(resolved))
        return result

    def _model_paths(
        self,
        engine: str,
        location: RuntimeLocation | None,
    ) -> dict[str, list[str]]:
        """Return configured, runtime-local, and final fallback model directories."""
        paths: dict[str, list[Path]] = {
            "checkpoints": [],
            "diffusion_models": [],
            "loras": [],
            "vae": [],
            "text_encoders": [],
            "upscale_models": [],
        }
        app_fallback_paths: dict[str, Path] = {}
        if engine == "forge":
            # Preserve the pre-existing Settings contract and its priority.
            try:
                from core.forge_modules import get_forge_paths

                configured = get_forge_paths()
                mapping = {
                    "checkpoint_dir": "checkpoints",
                    "lora_dir": "loras",
                    "vae_dir": "vae",
                    "text_encoder_dir": "text_encoders",
                }
                for old_key, category in mapping.items():
                    value = configured.get(old_key)
                    if value is not None:
                        paths[category].append(Path(value))
            except Exception:
                # A malformed optional legacy setting must not hide the runtime.
                pass

        # When neither of the known Forge model roots exists, forge_modules
        # selects and creates one app-owned library.  Project that same physical
        # fallback into both engines so a Comfy-only installation does not need
        # a duplicate model tree.  Explicit Settings/linked/runtime paths stay
        # ahead of this final fallback.
        try:
            from core.forge_modules import (
                get_app_model_paths,
                get_app_models_root,
                get_forge_root,
            )

            if get_forge_root().resolve() == get_app_models_root().resolve():
                app_fallback_paths = get_app_model_paths()
        except Exception:
            # Snapshot/model discovery must remain usable if the optional
            # fallback directory cannot be inspected on a read-only volume.
            app_fallback_paths = {}

        model_roots: list[Path] = []
        if location is not None:
            model_roots.append(location.source_root / "models")
        model_roots.append(self._data_root(engine) / "models")

        for model_root in model_roots:
            self._append_existing_paths(
                paths["upscale_models"], model_root,
                ("ESRGAN", "upscale_models") if engine == "forge" else ("upscale_models",),
            )
            if engine == "forge":
                self._append_existing_paths(
                    paths["checkpoints"], model_root, ("Stable-diffusion", "checkpoints")
                )
                # Forge 는 Anima·Krea2·Flux 같은 UNET 전용 파일도 Stable-diffusion 에 둔다.
                # ComfyUI 의 UNETLoader 는 diffusion_models 만 보므로 그 폴더를 여기에도 비춘다 —
                # 안 그러면 ComfyUI 워크플로에서 같은 파일이 '없는 모델' 이 된다.
                self._append_existing_paths(
                    paths["diffusion_models"], model_root,
                    ("diffusion_models", "unet", "UNET", "Stable-diffusion"),
                )
                self._append_existing_paths(paths["loras"], model_root, ("Lora", "loras", "LoRA"))
                self._append_existing_paths(paths["vae"], model_root, ("VAE", "vae"))
                self._append_existing_paths(
                    paths["text_encoders"],
                    model_root,
                    ("text_encoder", "text_encoders", "clip", "CLIP"),
                )
            else:
                self._append_existing_paths(paths["checkpoints"], model_root, ("checkpoints",))
                self._append_existing_paths(
                    paths["diffusion_models"], model_root, ("diffusion_models", "unet")
                )
                self._append_existing_paths(paths["loras"], model_root, ("loras",))
                self._append_existing_paths(paths["vae"], model_root, ("vae",))
                self._append_existing_paths(
                    paths["text_encoders"], model_root, ("text_encoders", "clip")
                )
        for category, fallback_path in app_fallback_paths.items():
            if category in paths:
                paths[category].append(fallback_path)
        return {category: self._dedupe_paths(values) for category, values in paths.items()}

    def _model_engine_order(self) -> tuple[str, ...]:
        primary = str(self._state.get("primaryModelEngine") or "forge")
        return (primary,) + tuple(key for key in ENGINE_DEFINITIONS if key != primary)

    def _combined_model_paths(self) -> dict[str, list[str]]:
        combined = {
            "checkpoints": [],
            "diffusion_models": [],
            "loras": [],
            "vae": [],
            "text_encoders": [],
            "upscale_models": [],
        }
        seen = {category: set() for category in combined}
        for engine in self._model_engine_order():
            own = self._model_paths(engine, self._runtime_location(engine))
            for category, values in own.items():
                for value in values:
                    marker = os.path.normcase(str(Path(value).resolve()))
                    if marker in seen[category]:
                        continue
                    seen[category].add(marker)
                    combined[category].append(value)
        return combined

    def _install_state(self, key: str) -> dict[str, Any]:
        """installed·sourceMode·확장 폴더 쓰기 가능 여부 — snapshot() 의 가벼운 부분.

        내부 작업(install/update/start/extension)은 이 몇 필드만 필요하다. 전체
        snapshot() 은 확장마다 .git·marker 를 읽고 모델 경로를 탐색해 한 번에 약 20ms 가
        걸리므로, 작업 한 번에 여러 번 부르지 않도록 이 헬퍼를 snapshot() 과 공유한다.
        """
        definition = ENGINE_DEFINITIONS[key]
        with self._state_lock:
            saved = self._state["engines"][key]
            location = self._runtime_location(key)
            source = location.source_root if location is not None else Path()
            python_path = location.python_path if location is not None else Path()
            installed = bool(
                location is not None
                and (source / definition.entrypoint).is_file()
                and python_path.is_file()
            )
            extension_root = self._extension_root(key)
            extension_approved = bool(saved.get("extensionDirApproved"))
            extension_writable = bool(
                (_is_relative_to(extension_root.resolve(), self.runtime_root) or extension_approved)
                and (not extension_root.exists() or os.access(extension_root, os.W_OK))
            )
            return {
                "location": location,
                "source": source,
                "pythonPath": python_path,
                "installed": installed,
                "sourceMode": str(saved.get("sourceMode", "managed") or "managed"),
                "extensionRoot": extension_root,
                "extensionDirApproved": extension_approved,
                "extensionDirExternal": extension_root != self._default_extension_root(key).resolve(),
                "extensionWritable": extension_writable,
            }

    def snapshot(self) -> dict[str, Any]:
        """Return cached/local state only; never contacts GitHub or PyPI."""
        with self._state_lock:
            engines: dict[str, Any] = {}
            for key, definition in ENGINE_DEFINITIONS.items():
                saved = self._state["engines"][key]
                basics = self._install_state(key)
                source_mode = basics["sourceMode"]
                existing_root = str(saved.get("existingRoot", "") or "")
                location = basics["location"]
                detected = self._detected_launch_args(key, location, saved)
                source = basics["source"]
                python_path = basics["pythonPath"]
                installed = basics["installed"]
                version, commit, branch = self._local_git_info(source) if installed else ("", "", "")
                if installed and not version:
                    version = self._declared_local_version(key, source)
                with self._process_lock:
                    running = self._process_running(key)
                    owned_process = self._processes.get(key) if running else None
                    healthy = bool(owned_process is not None and self._healthy.get(key))
                port = int(saved.get("port") or definition.preferred_port)
                endpoint = owned_process.endpoint if owned_process else f"http://127.0.0.1:{port}"
                remote_commit = str(saved.get("remoteCommit", "") or "")
                update_available = bool(commit and remote_commit and commit != remote_commit)
                if not saved.get("lastChecked"):
                    update_status = "Not checked"
                elif update_available:
                    update_status = "Update available"
                else:
                    update_status = "Up to date"
                extension_root = basics["extensionRoot"]
                extension_approved = basics["extensionDirApproved"]
                extension_writable = basics["extensionWritable"]
                engines[key] = {
                    "engine": key,
                    "name": definition.name,
                    "protocol": definition.protocol,
                    "installed": installed,
                    "running": running,
                    "healthy": healthy,
                    "owned": running,
                    "busy": bool(self._busy.get(key)),
                    "active": self._state.get("activeEngine") == key,
                    "autoStart": bool(saved.get("autoStart")),
                    "sourceMode": source_mode,
                    "existingRoot": existing_root,
                    "root": str(self._engine_root(key)),
                    "installRoot": str(location.install_root) if location is not None else "",
                    "sourceRoot": str(source) if location is not None else "",
                    "pythonPath": str(python_path) if location is not None else "",
                    "portable": bool(location is not None and location.portable),
                    "dataRoot": str(self._data_root(key)),
                    "modelPaths": self._model_paths(key, location),
                    "apiUrl": endpoint,
                    "port": port,
                    "version": version or (commit[:12] if commit else ""),
                    "commit": commit,
                    "branch": branch or definition.branch,
                    "remoteVersion": str(saved.get("remoteVersion", "") or ""),
                    "remoteCommit": remote_commit,
                    "lastChecked": str(saved.get("lastChecked", "") or ""),
                    "updateAvailable": update_available,
                    "updateStatus": update_status,
                    "extensionDir": str(extension_root),
                    "defaultExtensionDir": str(self._default_extension_root(key)),
                    "extensionDirExternal": basics["extensionDirExternal"],
                    "extensionDirApproved": extension_approved,
                    "extensionWritable": extension_writable,
                    "extensions": self._extension_state(key),
                    "message": self._last_messages.get(key) or str(saved.get("lastError", "") or ""),
                    "logPath": str(owned_process.log_path) if owned_process else "",
                    "extraArgs": str(saved.get("extraArgs", "") or ""),
                    "importLaunchArgs": bool(saved.get("importLaunchArgs", True)),
                    "fastFp16": bool(saved.get("fastFp16", False)),
                    "detectedArgs": " ".join(detected.args),
                    "detectedArgsSource": detected.source,
                    "detectedArgsDropped": " ".join(detected.dropped),
                }
            return {
                "ok": True,
                "schemaVersion": self.SCHEMA_VERSION,
                "activeEngine": self._state.get("activeEngine") or "",
                "primaryModelEngine": self._state.get("primaryModelEngine") or "forge",
                "runtimeRoot": str(self.runtime_root),
                "engines": engines,
            }

    # -------------------------------------------------------------- configure

    def configure(
        self, engine: str, patch: Mapping[str, Any], *, return_snapshot: bool = True
    ) -> dict[str, Any]:
        """설정 저장. 공개 호출은 전체 snapshot 을 돌려준다(계약).

        execute 안의 내부 호출은 ``return_snapshot=False`` 로 스냅샷 계산을 건너뛴다 —
        execute 가 끝에서 한 번 계산한 스냅샷을 결과의 ``state`` 로 재사용한다.
        """
        key = _canonical_engine(engine)
        if not isinstance(patch, Mapping):
            raise BackendRuntimeError("INVALID_PAYLOAD", "설정 payload가 객체가 아닙니다")
        with self._state_lock:
            saved = self._state["engines"][key]
            source_change = "sourceMode" in patch or "existingRoot" in patch
            if source_change:
                running_engine = self._owned_running_engine()
                if running_engine:
                    raise BackendRuntimeError(
                        "MODEL_TOPOLOGY_TRANSACTION_REQUIRED",
                        "실행 중인 백엔드의 공유 모델 경로를 바꾸려면 set_install_root 또는 use_managed_install 작업을 사용하세요",
                        stage="configure",
                        details={"runningEngine": running_engine},
                    )
                source_mode = str(
                    patch.get("sourceMode", saved.get("sourceMode", "managed"))
                    or "managed"
                ).casefold()
                if source_mode not in {"managed", "existing"}:
                    raise BackendRuntimeError(
                        "INVALID_SOURCE_MODE", "설치 소스는 managed 또는 existing이어야 합니다"
                    )
                if source_mode == "existing":
                    root_value = patch.get("existingRoot", saved.get("existingRoot", ""))
                    location = self._detect_existing_install(key, str(root_value or ""))
                    linked_extensions = (
                        self._linked_extension_root(key, location)
                        if not saved.get("extensionDirApproved")
                        else None
                    )
                    if linked_extensions is not None:
                        for other in ENGINE_DEFINITIONS:
                            if other != key and self._extension_roots_overlap(
                                linked_extensions, self._extension_root(other)
                            ):
                                raise BackendRuntimeError(
                                    "EXTENSION_PATH_CONFLICT",
                                    "Forge extensions와 ComfyUI custom_nodes는 서로 분리된 폴더여야 합니다",
                                    stage="configure",
                                )
                    saved["sourceMode"] = "existing"
                    saved["existingRoot"] = str(
                        self._normalise_existing_root(str(root_value or ""))
                    )
                    if not saved.get("extensionDirApproved"):
                        saved["extensionDir"] = (
                            str(linked_extensions) if linked_extensions is not None else ""
                        )
                else:
                    saved["sourceMode"] = "managed"
                    saved["existingRoot"] = ""
                    if not saved.get("extensionDirApproved"):
                        saved["extensionDir"] = ""
                saved["remoteVersion"] = ""
                saved["remoteCommit"] = ""
                saved["lastChecked"] = ""
            if "autoStart" in patch:
                enabled = bool(patch.get("autoStart"))
                saved["autoStart"] = enabled
                if enabled:
                    for other in ENGINE_DEFINITIONS:
                        if other != key:
                            self._state["engines"][other]["autoStart"] = False
            if "extensionDir" in patch:
                value = str(patch.get("extensionDir", "") or "").strip()
                if value:
                    approved = bool(patch.get("extensionDirApproved", True))
                    candidate = self._validate_extension_root(
                        value, engine=key, require_writable=approved
                    )
                    for other in ENGINE_DEFINITIONS:
                        if other == key:
                            continue
                        other_root = self._extension_root(other)
                        if self._extension_roots_overlap(candidate, other_root):
                            raise BackendRuntimeError(
                                "EXTENSION_PATH_CONFLICT",
                                "Forge extensions와 ComfyUI custom_nodes는 서로 분리된 폴더여야 합니다",
                                stage="configure",
                            )
                    saved["extensionDir"] = str(candidate)
                    saved["extensionDirApproved"] = approved
                else:
                    saved["extensionDir"] = ""
                    saved["extensionDirApproved"] = bool(
                        patch.get("extensionDirApproved", False)
                    )
                    self._validate_extension_root(
                        self._default_extension_root(key), create_managed=True, engine=key
                    )
            if "extraArgs" in patch:
                saved["extraArgs"] = self._normalise_extra_args(patch.get("extraArgs", ""))
            if "importLaunchArgs" in patch:
                saved["importLaunchArgs"] = bool(patch.get("importLaunchArgs"))
            if "fastFp16" in patch:
                saved["fastFp16"] = bool(patch.get("fastFp16"))
            if patch.get("active") is True:
                self._state["activeEngine"] = key
            self._save_state()
        return self.snapshot() if return_snapshot else {}

    @staticmethod
    def _normalise_extra_args(value: object) -> str:
        """실행 인자 문자열을 다듬는다 — 쪼개지지 않는 값(짝 안 맞는 따옴표)은 거부한다.

        기동 직전이 아니라 **저장할 때** 검사해야 사용자가 그 자리에서 고칠 수 있다.
        줄바꿈은 공백으로 — 인자 하나가 줄을 넘는 일은 없고, 붙여넣기에 흔히 섞인다.
        """
        text = " ".join(str(value or "").split())
        if not text:
            return ""
        if len(text) > 2000:
            raise BackendRuntimeError("INVALID_EXTRA_ARGS", "실행 인자가 너무 깁니다 (2000자 제한)")
        try:
            shlex.split(text, posix=False)
        except ValueError as exc:
            raise BackendRuntimeError("INVALID_EXTRA_ARGS", f"실행 인자를 해석할 수 없습니다: {exc}") from exc
        return text

    def _ensure_primary_model_engine(self, preferred: str = "") -> str:
        """Keep primary independent from active while preferring a usable source."""
        usable = [
            key for key in ENGINE_DEFINITIONS
            if self._runtime_location(key) is not None
        ]
        current = str(self._state.get("primaryModelEngine") or "forge")
        if current in usable:
            return current
        if preferred in usable:
            selected = preferred
        elif usable:
            selected = usable[0]
        else:
            selected = "forge"
        if selected != current:
            self._state["primaryModelEngine"] = selected
            self._save_state()
        return selected

    def _owned_running_engine(self) -> str:
        """Return the single app-owned model consumer, if one is running."""
        return next(
            (key for key in ENGINE_DEFINITIONS if self._process_running(key)), ""
        )

    def _restore_model_topology(
        self,
        changed_engine: str,
        previous_engine_state: Mapping[str, Any],
        previous_primary: str,
        running_engine: str,
        on_progress: ProgressCallback | None,
    ) -> str:
        """Restore saved topology and the process that consumed its model paths.

        Model paths from both engines are projected into whichever owned backend
        is running.  A rollback therefore is not complete until that same process
        is healthy again with the restored launch specification.
        """
        errors: list[str] = []
        try:
            with self._state_lock:
                self._state["engines"][changed_engine] = dict(previous_engine_state)
                self._state["primaryModelEngine"] = previous_primary
                self._save_state()
        except Exception as exc:
            # Keep restoring the mount/process from the already-restored in-memory
            # state even when persistence itself is temporarily unavailable.
            errors.append(f"state persistence: {exc}")
        try:
            self._ensure_extension_mount(changed_engine)
        except Exception as exc:
            errors.append(f"extension mount: {exc}")
        if running_engine:
            if self._runtime_location(running_engine) is None:
                errors.append("previous runtime is no longer available")
            else:
                try:
                    self._start(
                        running_engine,
                        on_progress,
                        install_if_missing=False,
                    )
                except Exception as exc:
                    errors.append(f"backend restart: {exc}")
        return " · ".join(errors)

    def _set_install_source(
        self,
        engine: str,
        *,
        source_mode: str,
        existing_root: str = "",
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        if source_mode == "existing":
            # Validate before stopping a healthy child or mutating saved state.
            self._detect_existing_install(engine, existing_root)
        previous_engine = dict(self._state["engines"][engine])
        previous_primary = str(self._state.get("primaryModelEngine") or "forge")
        running_engine = self._owned_running_engine()
        if running_engine:
            self._stop(running_engine, on_progress)
        try:
            patch: dict[str, Any] = {"sourceMode": source_mode}
            if source_mode == "existing":
                patch["existingRoot"] = existing_root
            # 결과 스냅샷은 execute 가 끝에서 한 번만 계산한다(result['state'] 와 공유).
            self.configure(engine, patch, return_snapshot=False)
            self._ensure_primary_model_engine(engine)
            self._ensure_extension_mount(engine)
            location = self._runtime_location(engine)
            # Switching the running engine to an unavailable managed slot is an
            # intentional stop.  An opposite running engine, however, must always
            # be restarted because its shared model argv/config changed too.
            if running_engine and (running_engine != engine or location is not None):
                self._start(running_engine, on_progress, install_if_missing=False)
        except Exception as exc:
            rollback_error = self._restore_model_topology(
                engine,
                previous_engine,
                previous_primary,
                running_engine,
                on_progress,
            )
            if rollback_error:
                raise BackendRuntimeError(
                    "MODEL_TOPOLOGY_ROLLBACK_FAILED",
                    "설치 소스 적용과 이전 모델 경로/백엔드 복구가 모두 실패했습니다",
                    stage="rollback",
                    retryable=True,
                    details={"reason": str(exc), "rollbackReason": rollback_error},
                ) from exc
            raise
        mode_label = "기존 설치" if source_mode == "existing" else "앱 관리형 설치"
        return {
            "message": f"{ENGINE_DEFINITIONS[engine].name}을 {mode_label}로 전환했습니다",
            "state": _FINAL_SNAPSHOT,
        }

    def _set_primary_model_engine(
        self,
        engine: str,
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        if self._runtime_location(engine) is None:
            raise BackendRuntimeError(
                "PRIMARY_MODEL_SOURCE_UNAVAILABLE",
                "설치되어 실행 가능한 백엔드만 메인 모델 UI로 지정할 수 있습니다",
                stage="set_primary_model_engine",
            )
        previous = str(self._state.get("primaryModelEngine") or "forge")
        if previous == engine:
            return {"message": f"{ENGINE_DEFINITIONS[engine].name}가 이미 메인 모델 UI입니다"}
        running_engine = next(
            (key for key in ENGINE_DEFINITIONS if self._process_running(key)), ""
        )
        if running_engine:
            self._stop(running_engine, on_progress)
        self._state["primaryModelEngine"] = engine
        self._save_state()
        try:
            if running_engine:
                self._start(running_engine, on_progress, install_if_missing=False)
        except Exception:
            self._state["primaryModelEngine"] = previous
            self._save_state()
            if running_engine:
                try:
                    self._start(running_engine, on_progress, install_if_missing=False)
                except Exception:
                    pass
            raise
        return {"message": f"{ENGINE_DEFINITIONS[engine].name}를 메인 모델 UI로 지정했습니다"}

    # -------------------------------------------------------------- operations

    def execute(
        self,
        engine: str,
        action: str,
        payload: Mapping[str, Any] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        key = _canonical_engine(engine)
        action_name = str(action or "").strip().casefold()
        payload = dict(payload or {})
        if not self._operation_gate.acquire(blocking=False):
            raise BackendRuntimeError(
                "OPERATION_BUSY",
                "다른 관리형 백엔드 작업이 진행 중입니다",
                stage=action_name,
                retryable=True,
            )
        lock = self._operation_locks[key]
        if not lock.acquire(blocking=False):
            self._operation_gate.release()
            raise BackendRuntimeError(
                "OPERATION_BUSY", f"{ENGINE_DEFINITIONS[key].name} 작업이 이미 진행 중입니다",
                stage=action_name, retryable=True,
            )
        self._busy[key] = True
        try:
            self._progress(on_progress, key, action_name, "start", f"{action_name} 작업을 시작합니다", 0)
            if action_name == "install":
                result = self._install(key, on_progress)
            elif action_name == "update":
                result = self._update(key, on_progress)
            elif action_name == "check_update":
                result = self._check_update(key, on_progress)
            elif action_name in {"start", "use"}:
                # Explicit Start/Use may prepare a missing runtime. Startup auto-start
                # passes ``startup=true`` and must never trigger downloads implicitly.
                install_if_missing = bool(
                    payload.get("installIfMissing", not bool(payload.get("startup", False)))
                )
                result = self._start(key, on_progress, install_if_missing=install_if_missing)
                startup = bool(payload.get("startup", False))
                # START normally only launches.  If launching this engine had to
                # stop another managed engine, however, leaving the app connected
                # to the stopped endpoint would be invalid.  That replacement is
                # an actual backend switch and must activate/reconnect the new one.
                activate = bool(
                    action_name == "use"
                    or startup
                    or result.get("replacedEngine")
                )
                if activate:
                    patch: dict[str, Any] = {"active": True}
                    if "autoStart" in payload:
                        # Only USE owns this optional preference mutation.  START
                        # must never change auto-start just because it switched.
                        if action_name == "use":
                            patch["autoStart"] = bool(payload.get("autoStart"))
                    self.configure(key, patch, return_snapshot=False)
                result["activate"] = activate
            elif action_name == "stop":
                result = self._stop(key, on_progress)
            elif action_name == "set_auto_start":
                self.configure(key, {"autoStart": bool(payload.get("autoStart"))}, return_snapshot=False)
                result = {"message": "자동 시작 설정을 저장했습니다", "state": _FINAL_SNAPSHOT}
            elif action_name == "set_launch_options":
                option_patch = {
                    name: bool(payload.get(name))
                    for name in ("importLaunchArgs", "fastFp16")
                    if name in payload
                }
                self.configure(key, option_patch, return_snapshot=False)
                running = self._process_running(key)
                result = {
                    "message": "실행 옵션을 저장했습니다" + (" — 다음 시작부터 적용됩니다" if running else ""),
                    "state": _FINAL_SNAPSHOT,
                }
            elif action_name == "set_extra_args":
                # 저장만 한다. 돌고 있는 프로세스엔 다음 기동부터 적용된다 —
                # 인자를 바꿨다고 백엔드를 말없이 재시작하면 진행 중인 생성이 죽는다.
                self.configure(key, {"extraArgs": payload.get("extraArgs", "")}, return_snapshot=False)
                running = self._process_running(key)
                result = {
                    "message": "실행 인자를 저장했습니다" + (" — 다음 시작부터 적용됩니다" if running else ""),
                    "state": _FINAL_SNAPSHOT,
                }
            elif action_name == "set_install_root":
                install_root = str(
                    payload.get("existingRoot")
                    or payload.get("installRoot")
                    or payload.get("path")
                    or ""
                )
                result = self._set_install_source(
                    key,
                    source_mode="existing",
                    existing_root=install_root,
                    on_progress=on_progress,
                )
            elif action_name == "use_managed_install":
                result = self._set_install_source(
                    key, source_mode="managed", on_progress=on_progress
                )
            elif action_name == "set_primary_model_engine":
                requested = str(
                    payload.get("primaryModelEngine")
                    or payload.get("engine")
                    or key
                )
                result = self._set_primary_model_engine(
                    _canonical_engine(requested), on_progress
                )
            elif action_name == "save_extension_dir":
                was_running = self._process_running(key)
                previous = str(self._state["engines"][key].get("extensionDir", "") or "")
                previous_approved = bool(
                    self._state["engines"][key].get("extensionDirApproved")
                )
                if was_running:
                    self._stop(key, on_progress)
                try:
                    self.configure(
                        key, {"extensionDir": payload.get("extensionDir", "")}, return_snapshot=False
                    )
                    self._ensure_extension_mount(key)
                except Exception:
                    self.configure(
                        key,
                        {
                            "extensionDir": previous,
                            "extensionDirApproved": previous_approved,
                        },
                        return_snapshot=False,
                    )
                    try:
                        self._ensure_extension_mount(key)
                    except Exception:
                        pass
                    if was_running:
                        try:
                            self._start(key, on_progress, install_if_missing=False)
                        except Exception:
                            pass
                    raise
                if was_running:
                    self._start(key, on_progress, install_if_missing=False)
                result = {"message": "확장 폴더를 저장했습니다", "state": _FINAL_SNAPSHOT}
            elif action_name == "install_extension":
                result = self._install_extension(key, payload, on_progress)
            elif action_name in {"check_extension", "check_extensions"}:
                result = self._check_extension(key, payload, on_progress)
            elif action_name == "update_extension":
                result = self._update_extension(key, payload, on_progress)
            else:
                raise BackendRuntimeError("INVALID_ACTION", f"지원하지 않는 작업입니다: {action}")
            message = str(result.get("message", "완료"))
            self._last_messages[key] = message
            self._state["engines"][key]["lastError"] = ""
            self._save_state()
            # 결과 스냅샷은 한 번만 계산한다 — 설정 작업의 ``state`` 도 같은 값을 쓴다.
            snapshot = self.snapshot()
            if result.get("state") is _FINAL_SNAPSHOT:
                result["state"] = snapshot
            final = {
                "ok": True,
                "engine": key,
                "action": action_name,
                **result,
                "snapshot": snapshot,
            }
            self._progress(on_progress, key, action_name, "complete", message, 100)
            return final
        except BackendRuntimeError as exc:
            self._last_messages[key] = str(exc)
            self._state["engines"][key]["lastError"] = str(exc)
            self._save_state()
            self._progress(on_progress, key, action_name, "error", str(exc), 100, error=exc.as_dict())
            raise
        except Exception as exc:
            wrapped = BackendRuntimeError(
                "RUNTIME_OPERATION_FAILED", str(exc), stage=action_name, retryable=True
            )
            self._last_messages[key] = str(wrapped)
            self._state["engines"][key]["lastError"] = str(wrapped)
            self._save_state()
            self._progress(on_progress, key, action_name, "error", str(wrapped), 100, error=wrapped.as_dict())
            raise wrapped from exc
        finally:
            self._busy[key] = False
            lock.release()
            self._operation_gate.release()

    def _progress(
        self,
        callback: ProgressCallback | None,
        engine: str,
        action: str,
        phase: str,
        message: str,
        progress: int,
        **extra: Any,
    ) -> None:
        if not callback:
            return
        event = {
            "engine": engine,
            "action": action,
            "phase": phase,
            "message": str(message)[:1000],
            "progress": max(0, min(100, int(progress))),
            **extra,
        }
        try:
            callback(event)
        except Exception:
            pass

    def _command_progress(
        self,
        callback: ProgressCallback | None,
        engine: str,
        action: str,
        phase: str,
        base_progress: int,
    ) -> Callable[[str], None]:
        last_emit = [0.0]

        def emit(line: str) -> None:
            now = time.monotonic()
            if now - last_emit[0] < 0.2:
                return
            last_emit[0] = now
            clean = re.sub(r"https://[^\s/@]+@", "https://***@", str(line))
            self._progress(callback, engine, action, phase, clean[-800:], base_progress)

        return emit

    # ----------------------------------------------------------- install/update

    def _require_tools(self) -> None:
        if not self.adapter.which("git"):
            raise BackendRuntimeError(
                "GIT_NOT_FOUND", "Git을 찾을 수 없습니다", stage="preflight"
            )

    def _new_release_id(self, commit_hint: str = "candidate") -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^0-9a-f]", "", commit_hint.casefold())[:12] or "candidate"
        return f"{stamp}-{safe}-{uuid.uuid4().hex[:6]}"

    def _remote_head(self, definition: EngineDefinition) -> str:
        result = self.adapter.run(
            ["git", "ls-remote", definition.repository, f"refs/heads/{definition.branch}"],
            timeout=60,
        )
        if result.returncode != 0 or not result.output.strip():
            raise BackendRuntimeError(
                "VERSION_CHECK_FAILED", "원격 버전을 확인하지 못했습니다",
                stage="version_check", retryable=True,
                details={"output": result.output[-1000:]},
            )
        return result.output.strip().split()[0]

    def _prepare_release(self, engine: str, on_progress: ProgressCallback | None) -> tuple[str, str]:
        self._require_tools()
        definition = ENGINE_DEFINITIONS[engine]
        action = "install" if not self._state["engines"][engine].get("release") else "update"
        self._progress(on_progress, engine, action, "version_check", "원격 소스 버전을 확인합니다", 5)
        remote_commit = self._remote_head(definition)
        release_id = self._new_release_id(remote_commit)
        release_root = self._release_root(engine, release_id)
        source = release_root / "source"
        venv = release_root / "venv"
        release_root.mkdir(parents=True, exist_ok=False)
        try:
            self._progress(on_progress, engine, action, "download", "소스 코드를 격리 폴더에 다운로드합니다", 15)
            result = self.adapter.run(
                [
                    "git", "clone", "--branch", definition.branch, "--single-branch",
                    definition.repository, str(source),
                ],
                cwd=release_root,
                timeout=1800,
                on_line=self._command_progress(on_progress, engine, action, "download", 20),
            )
            if result.returncode != 0:
                raise BackendRuntimeError(
                    "SOURCE_FETCH_FAILED", "백엔드 소스 다운로드에 실패했습니다",
                    stage="download", retryable=True,
                    details={"output": result.output[-2000:]},
                )
            actual_commit = self._capture(["git", "rev-parse", "HEAD"], cwd=source)
            if actual_commit and actual_commit != remote_commit:
                raise BackendRuntimeError(
                    "SOURCE_REVISION_MISMATCH", "다운로드한 소스 버전이 확인한 원격 버전과 다릅니다",
                    stage="verify",
                )
            self._create_venv(engine, venv, on_progress, action)
            self._bootstrap(engine, source, venv, on_progress, action)
            manifest = {
                "schemaVersion": 1,
                "engine": engine,
                "repository": definition.repository,
                "branch": definition.branch,
                "commit": actual_commit or remote_commit,
                "createdAt": _utc_now(),
            }
            atomic_write_json(str(release_root / "release.json"), manifest, indent=2)
            return release_id, actual_commit or remote_commit
        except Exception:
            # Only the not-yet-active candidate is removed.  Active releases and all
            # shared user data remain untouched.
            if release_root.exists() and _is_relative_to(release_root.resolve(), (self._engine_root(engine) / "releases").resolve()):
                shutil.rmtree(release_root, ignore_errors=True)
            raise

    def _create_venv(
        self,
        engine: str,
        venv: Path,
        on_progress: ProgressCallback | None,
        action: str,
    ) -> None:
        self._progress(on_progress, engine, action, "venv", "전용 Python 환경을 만듭니다", 40)
        uv = self.adapter.which("uv")
        if uv:
            argv = [uv, "venv", str(venv), "--python", "3.13", "--seed"]
        else:
            py = self.adapter.which("py")
            if py:
                argv = [py, "-3.13", "-m", "venv", str(venv)]
            elif Path(sys.executable).name.casefold().startswith("python"):
                argv = [sys.executable, "-m", "venv", str(venv)]
            else:
                raise BackendRuntimeError(
                    "PYTHON_NOT_FOUND",
                    "Python 3.13 또는 uv를 찾을 수 없습니다",
                    stage="venv",
                )
        result = self.adapter.run(
            argv,
            cwd=venv.parent,
            timeout=600,
            on_line=self._command_progress(on_progress, engine, action, "venv", 45),
        )
        if result.returncode != 0 or not self._venv_python(venv).is_file():
            raise BackendRuntimeError(
                "VENV_CREATE_FAILED", "전용 Python 환경 생성에 실패했습니다",
                stage="venv", details={"output": result.output[-2000:]},
            )

    def _pip_install(
        self,
        engine: str,
        python: Path,
        requirements: Path,
        on_progress: ProgressCallback | None,
        action: str,
        progress: int,
    ) -> None:
        if not requirements.is_file():
            return
        uv = self.adapter.which("uv")
        if uv:
            argv = [uv, "pip", "install", "--python", str(python), "-r", str(requirements)]
        else:
            argv = [str(python), "-m", "pip", "install", "-r", str(requirements)]
        result = self.adapter.run(
            argv,
            cwd=requirements.parent,
            timeout=3600,
            on_line=self._command_progress(on_progress, engine, action, "dependencies", progress),
        )
        if result.returncode != 0:
            raise BackendRuntimeError(
                "PACKAGE_INSTALL_FAILED", f"의존성 설치에 실패했습니다: {requirements.name}",
                stage="dependencies", retryable=True,
                details={"output": result.output[-3000:]},
            )

    def _bootstrap(
        self,
        engine: str,
        source: Path,
        venv: Path,
        on_progress: ProgressCallback | None,
        action: str,
    ) -> None:
        python = self._venv_python(venv)
        self._progress(on_progress, engine, action, "dependencies", "백엔드 의존성을 설치합니다", 55)
        if engine == "comfyui":
            self._pip_install(engine, python, source / "requirements.txt", on_progress, action, 60)
            self._pip_install(engine, python, source / "manager_requirements.txt", on_progress, action, 72)
            result = self.adapter.run(
                [str(python), "-c", "import sys; import torch; print(sys.version.split()[0]); print(torch.__version__)"],
                cwd=source,
                timeout=120,
            )
        else:
            data_root = self._data_root(engine)
            data_root.mkdir(parents=True, exist_ok=True)
            argv = [str(python), str(source / "launch.py"), "--exit", "--api", "--data-dir", str(data_root)]
            result = self.adapter.run(
                argv,
                cwd=source,
                timeout=5400,
                on_line=self._command_progress(on_progress, engine, action, "dependencies", 65),
            )
        if result.returncode != 0:
            raise BackendRuntimeError(
                "BOOTSTRAP_FAILED", f"{ENGINE_DEFINITIONS[engine].name} 초기 구성에 실패했습니다",
                stage="verify", retryable=True,
                details={"output": result.output[-3000:]},
            )
        self._progress(on_progress, engine, action, "verify", "격리 런타임 구성을 확인했습니다", 88)

    def _install(
        self,
        engine: str,
        on_progress: ProgressCallback | None,
        *,
        restart_model_consumer: bool = True,
    ) -> dict[str, Any]:
        current = self._install_state(engine)
        if current["installed"]:
            return {"message": f"{ENGINE_DEFINITIONS[engine].name}가 이미 설치되어 있습니다"}
        if current.get("sourceMode") == "existing":
            raise BackendRuntimeError(
                "LINKED_INSTALL_INVALID",
                "연결한 기존 설치를 찾을 수 없습니다. 경로를 다시 지정하거나 앱 관리형 설치로 전환하세요",
                stage="install",
            )
        previous_engine = dict(self._state["engines"][engine])
        previous_primary = str(self._state.get("primaryModelEngine") or "forge")
        release, commit = self._prepare_release(engine, on_progress)
        running_engine = self._owned_running_engine() if restart_model_consumer else ""
        if running_engine:
            self._stop(running_engine, on_progress)
        try:
            with self._state_lock:
                saved = self._state["engines"][engine]
                saved["release"] = release
                saved["remoteCommit"] = commit
                saved["remoteVersion"] = commit[:12]
                saved["lastChecked"] = _utc_now()
                self._save_state()
            self._ensure_primary_model_engine(engine)
            self._ensure_extension_mount(engine)
            if running_engine:
                self._start(running_engine, on_progress, install_if_missing=False)
        except Exception as exc:
            rollback_error = self._restore_model_topology(
                engine,
                previous_engine,
                previous_primary,
                running_engine,
                on_progress,
            )
            if rollback_error:
                raise BackendRuntimeError(
                    "MODEL_TOPOLOGY_ROLLBACK_FAILED",
                    "설치 적용과 이전 모델 경로/백엔드 복구가 모두 실패했습니다",
                    stage="rollback",
                    retryable=True,
                    details={"reason": str(exc), "rollbackReason": rollback_error},
                ) from exc
            raise
        return {"message": f"{ENGINE_DEFINITIONS[engine].name} 설치를 완료했습니다", "release": release}

    def _check_update(self, engine: str, on_progress: ProgressCallback | None) -> dict[str, Any]:
        definition = ENGINE_DEFINITIONS[engine]
        self._require_tools()
        self._progress(on_progress, engine, "check_update", "version_check", "원격 버전을 확인합니다", 30)
        state = self._state["engines"][engine]
        source = self._source_root(engine)
        if state.get("sourceMode") == "existing":
            location = self._runtime_location(engine, require_valid=True)
            assert location is not None
            source = location.source_root
            _version, local, branch = self._local_git_info(source)
            repository = self._git_origin_url(source)
            if not repository or not local:
                raise BackendRuntimeError(
                    "LINKED_VERSION_UNAVAILABLE",
                    "기존 설치가 Git 저장소가 아니어서 원격 버전을 확인할 수 없습니다",
                    stage="version_check",
                )
            branch = branch or definition.branch
            result = self.adapter.run(
                ["git", "ls-remote", repository, f"refs/heads/{branch}"], timeout=60
            )
            if result.returncode != 0 or not result.output.strip():
                raise BackendRuntimeError(
                    "VERSION_CHECK_FAILED",
                    "기존 설치의 원격 버전을 확인하지 못했습니다",
                    stage="version_check",
                    retryable=True,
                    details={"output": result.output[-1000:]},
                )
            remote = result.output.strip().split()[0]
        else:
            remote = self._remote_head(definition)
            local = self._capture(["git", "rev-parse", "HEAD"], cwd=source) if source.is_dir() else ""
        with self._state_lock:
            saved = self._state["engines"][engine]
            saved["remoteCommit"] = remote
            saved["remoteVersion"] = remote[:12]
            saved["lastChecked"] = _utc_now()
            self._save_state()
        available = bool(local and remote != local)
        if not local:
            message = f"원격 최신 버전은 {remote[:12]}입니다 · 아직 설치되지 않았습니다"
        else:
            message = "업데이트가 있습니다" if available else "현재 버전이 최신입니다"
        return {
            "message": message,
            "localCommit": local,
            "remoteCommit": remote,
            "updateAvailable": available,
        }

    def _update(self, engine: str, on_progress: ProgressCallback | None) -> dict[str, Any]:
        state = self._install_state(engine)
        if state.get("sourceMode") == "existing":
            raise BackendRuntimeError(
                "LINKED_UPDATE_UNSUPPORTED",
                "연결한 기존 설치는 앱에서 수정하지 않습니다. 해당 설치의 업데이트 방식을 이용하세요",
                stage="update",
            )
        if not state["installed"]:
            return self._install(engine, on_progress)
        check = self._check_update(engine, on_progress)
        if not check["updateAvailable"]:
            return {"message": "현재 버전이 최신입니다", **check}
        previous_engine = dict(self._state["engines"][engine])
        previous_primary = str(self._state.get("primaryModelEngine") or "forge")
        running_engine = self._owned_running_engine()
        release, commit = self._prepare_release(engine, on_progress)
        if running_engine:
            self._stop(running_engine, on_progress)
        try:
            with self._state_lock:
                self._state["engines"][engine]["release"] = release
                self._state["engines"][engine]["remoteCommit"] = commit
                self._state["engines"][engine]["remoteVersion"] = commit[:12]
                self._state["engines"][engine]["lastChecked"] = _utc_now()
                self._save_state()
            if running_engine:
                self._start(running_engine, on_progress, install_if_missing=False)
        except Exception as exc:
            rollback_error = self._restore_model_topology(
                engine,
                previous_engine,
                previous_primary,
                running_engine,
                on_progress,
            )
            if rollback_error:
                raise BackendRuntimeError(
                    "ROLLBACK_FAILED",
                    "업데이트 실행 확인과 이전 버전 재시작이 모두 실패했습니다",
                    stage="rollback", retryable=True,
                    details={"reason": str(exc), "rollbackReason": rollback_error},
                ) from exc
            raise BackendRuntimeError(
                "UPDATE_VERIFY_FAILED", "업데이트 실행 확인에 실패해 이전 버전으로 되돌렸습니다",
                stage="rollback", retryable=True, details={"reason": str(exc)},
            ) from exc
        # 이전 release 는 다음 실행까지 남기지 않는다 — 다음 시작 때 GUI 스레드가 수 GB 를
        # 지우던 문제를 없애고, 삭제 자체는 백그라운드 스레드가 한다.
        self._retire_release(engine, str(previous_engine.get("release") or ""))
        return {"message": f"{ENGINE_DEFINITIONS[engine].name} 업데이트를 완료했습니다", "release": release}

    # --------------------------------------------------------------- launching

    def _reserved_ports(self) -> frozenset[int]:
        """다른 구성요소가 쓰기로 한 포트 — 실패해도 포트 선택을 막지 않는다."""
        source = self._reserved_ports_source
        if source is None:
            return frozenset()
        try:
            return frozenset(int(port) for port in (source() or ()))
        except Exception:
            return frozenset()

    def _choose_port(self, engine: str) -> int:
        preferred = int(self._state["engines"][engine].get("port") or ENGINE_DEFINITIONS[engine].preferred_port)
        # 켜 둔 Generation API 포트는 아직 bind 전이어도 건너뛴다. 앱 시작 때는 엔진 자동
        # 시작(여기)이 API 서버 bind(start_if_enabled)보다 먼저라, bind 시험만으로는 API 가
        # 곧 쓸 포트를 가져가 버린다(core/generation_api_port.py).
        reserved = self._reserved_ports()
        for port in range(preferred, preferred + 50):
            if port in reserved:
                continue
            if self.adapter.port_available("127.0.0.1", port):
                return port
        raise BackendRuntimeError(
            "PORT_UNAVAILABLE", f"{preferred}번부터 사용 가능한 로컬 포트를 찾지 못했습니다",
            stage="start", retryable=True,
        )

    def _write_comfy_model_paths(self) -> Path:
        paths = self._combined_model_paths()
        output = self._data_root("comfyui") / "aistudio_extra_model_paths.yaml"
        output.parent.mkdir(parents=True, exist_ok=True)
        # JSON is a valid YAML 1.2 document and avoids adding a YAML writer dependency.
        shared: dict[str, str] = {"base_path": ""}
        shared.update({
            category: "\n".join(values)
            for category, values in paths.items()
            if values
        })
        payload = {"aistudio_shared": shared}
        # 공용 원자 쓰기(fsync + 실패 시 tmp 정리) — 같은 '<파일>.tmp' 이름·텍스트 모드.
        atomic_write_json(str(output), payload, indent=2)
        return output

    @staticmethod
    def _disable_forge_browser_launch(data_root: Path) -> None:
        """`<data>/config.json` 의 auto_launch_browser 를 Disable 로. 다른 키는 건드리지 않는다.

        사용자가 설정 화면에서 값을 바꿔 두었더라도 되돌린다 — 이 프로세스는 앱이
        띄우는 것이라 브라우저가 열려야 할 이유가 없다. 파일이 깨져 있으면 손대지 않는다:
        Forge 가 스스로 고치게 두는 편이 조용히 덮어쓰는 것보다 낫다.
        """
        path = data_root / "config.json"
        current: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            if not isinstance(loaded, dict):
                return
            current = loaded
        if current.get("auto_launch_browser") == "Disable":
            return
        current["auto_launch_browser"] = "Disable"
        try:
            atomic_write_json(str(path), current)
        except OSError:
            pass

    def _launch_argv(self, engine: str, port: int) -> tuple[list[str], Path, dict[str, str]]:
        definition = ENGINE_DEFINITIONS[engine]
        location = self._runtime_location(engine, require_valid=True)
        assert location is not None
        source = location.source_root
        python = location.python_path
        data = self._data_root(engine)
        data.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["AISTUDIO_MANAGED_ENGINE"] = engine
        env["AISTUDIO_LAUNCH_NONCE"] = uuid.uuid4().hex
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # 한국어 Windows 콘솔(CP949)에서 커스텀 노드가 이모지·CJK 를 print 하면
        # UnicodeEncodeError 로 엔진이 기동 중에 죽는다 — 자식의 stdout/stderr 만 UTF-8 로 고정.
        # PYTHONUTF8 은 일부러 안 건드린다: open()/subprocess 기본 인코딩까지 바꿔 CP949 로
        # 저장된 확장 설정 파일을 못 읽게 만들 수 있다. setdefault: 사용자 값은 존중.
        env.setdefault("PYTHONIOENCODING", "utf-8")
        if engine == "forge":
            # Forge 의 `auto_launch_browser` 기본값은 "Local" — 서버가 로컬이면 기동할 때마다
            # OS 브라우저를 연다. 앱은 API 만 쓰므로 그 창은 매번 낯선 탭 하나일 뿐이다.
            # 설정 파일은 --data-dir 아래 것을 읽으니 거기에 Disable 을 적어 둔다.
            self._disable_forge_browser_launch(data)
            argv = [
                str(python), str(source / definition.entrypoint),
                "--api", "--api-server-stop", "--port", str(port),
                "--data-dir", str(data), "--theme", "dark",
            ]
            if self._state["engines"][engine].get("sourceMode") == "existing":
                # A linked environment is used as-is.  Do not let launch.py run
                # its implicit Git/pip preparation against the user's venv.
                argv.append("--skip-prepare-environment")
            flag_map = {
                "checkpoints": "--ckpt-dirs",
                "diffusion_models": "--ckpt-dirs",
                "loras": "--lora-dirs",
                "vae": "--vae-dirs",
                "text_encoders": "--text-encoder-dirs",
            }
            seen: dict[str, set[str]] = {flag: set() for flag in set(flag_map.values())}
            for source_engine in self._model_engine_order():
                own = self._model_paths(source_engine, self._runtime_location(source_engine))
                for category, flag in flag_map.items():
                    for path in own[category]:
                        marker = os.path.normcase(str(Path(path).resolve()))
                        if marker in seen[flag]:
                            continue
                        seen[flag].add(marker)
                        argv.extend([flag, path])
        else:
            extra_paths = self._write_comfy_model_paths()
            argv = [str(python)]
            if location.portable:
                argv.append("-s")
            argv.append(str(source / definition.entrypoint))
            if location.portable:
                argv.append("--windows-standalone-build")
            argv.extend([
                "--listen", "127.0.0.1", "--port", str(port),
                "--base-directory", str(data),
                "--extra-model-paths-config", str(extra_paths),
            ])
            # ComfyUI-Manager can run deferred node deletion/install scripts and
            # dependency resolution at startup.  On a linked runtime those writes
            # would target the user's custom_nodes junction and Python environment,
            # bypassing this manager's explicit external-folder approval boundary.
            if self._state["engines"][engine].get("sourceMode") == "managed":
                argv.append("--enable-manager")
        # 배치 파일에서 가져온 인자 → 사용자 인자 순으로 맨 뒤에. 같은 플래그가 겹치면 대개
        # 뒤의 것이 이기므로 사용자가 앱의 기본값도, 배치 파일 값도 덮어쓸 수 있다.
        saved = self._state["engines"][engine]
        imported = self._detected_launch_args(engine, location, saved).args
        extra = str(saved.get("extraArgs", "") or "")
        user_args = shlex.split(extra, posix=False) if extra else []
        argv.extend(merge_args(imported, user_args))
        return argv, source, env

    def _detected_launch_args(self, engine: str, location: RuntimeLocation | None, saved: Mapping[str, Any]) -> LaunchArgsImport:
        """연결 모드면 설치 루트의 실행 배치 파일을 읽는다. 관리형은 배치 파일이 없다 — ComfyUI fp16 토글만."""
        fast_fp16 = engine == "comfyui" and bool(saved.get("fastFp16"))
        roots: list[Path] = []
        if (
            location is not None
            and str(saved.get("sourceMode", "managed") or "managed") == "existing"
            and bool(saved.get("importLaunchArgs", True))
        ):
            roots = [location.install_root, location.source_root]
        if not roots and not fast_fp16:
            return LaunchArgsImport()
        return discover_launch_args(engine, roots, fast_fp16=fast_fp16)

    def _start(
        self,
        engine: str,
        on_progress: ProgressCallback | None,
        *,
        install_if_missing: bool,
    ) -> dict[str, Any]:
        if self._shutdown_requested.is_set():
            raise BackendRuntimeError(
                "START_CANCELLED", "앱 종료 요청으로 백엔드 시작을 취소했습니다", stage="start"
            )
        # Forge and Comfy have independent UI operation queues.  Serialize only
        # their start transitions so near-simultaneous clicks cannot launch both
        # GPU runtimes before either process becomes visible in ``_processes``.
        # STOP does not acquire this gate and can still cancel a health wait.
        with self._start_gate:
            if self._shutdown_requested.is_set():
                raise BackendRuntimeError(
                    "START_CANCELLED", "앱 종료 요청으로 백엔드 시작을 취소했습니다", stage="start"
                )
            previous_engine = next(
                (
                    other for other in ENGINE_DEFINITIONS
                    if other != engine and self._process_running(other)
                ),
                "",
            )
            try:
                return self._start_serialized(
                    engine, on_progress, install_if_missing=install_if_missing
                )
            except Exception as exc:
                if previous_engine and not self._shutdown_requested.is_set():
                    try:
                        self._start_serialized(
                            previous_engine, on_progress, install_if_missing=False
                        )
                    except Exception as restore_exc:
                        raise BackendRuntimeError(
                            "BACKEND_SWITCH_ROLLBACK_FAILED",
                            "새 백엔드 시작과 이전 백엔드 복구가 모두 실패했습니다",
                            stage="rollback",
                            retryable=True,
                            details={
                                "reason": str(exc),
                                "rollbackReason": str(restore_exc),
                                "previousEngine": previous_engine,
                            },
                        ) from exc
                raise

    def _start_serialized(
        self,
        engine: str,
        on_progress: ProgressCallback | None,
        *,
        install_if_missing: bool,
    ) -> dict[str, Any]:
        cancel = self._start_cancel[engine]
        cancel.clear()
        with self._process_lock:
            if self._process_running(engine):
                owned = self._processes[engine]
                return {
                    "message": f"{ENGINE_DEFINITIONS[engine].name}가 이미 실행 중입니다",
                    "apiUrl": owned.endpoint,
                }

        state = self._install_state(engine)
        if not state["installed"]:
            if not install_if_missing:
                raise BackendRuntimeError(
                    "NOT_INSTALLED", "관리형 백엔드가 설치되지 않았습니다", stage="start"
                )
            self._install(
                engine,
                on_progress,
                restart_model_consumer=False,
            )
        if cancel.is_set():
            raise BackendRuntimeError(
                "START_CANCELLED", "앱 종료 요청으로 백엔드 시작을 취소했습니다", stage="start"
            )

        # A single app-managed GPU backend is active at a time. External processes
        # are invisible to this map and are never touched.
        replaced_engine = ""
        for other in ENGINE_DEFINITIONS:
            if other != engine and self._process_running(other):
                self._stop(other, on_progress)
                replaced_engine = other
        self._ensure_extension_mount(engine)
        # 지난 실행이 강제 종료돼 남긴 *우리* 프로세스(표식 = data-dir)가 포트를 쥐고 있으면
        # 먼저 치운다 — 안 치우면 포트가 17863 처럼 밀리고 GPU 를 둘이 나눠 쓴다.
        swept = sweep_orphan(self._pid_file(engine), marker_for(self._data_root(engine)))
        if swept.startswith("killed-orphan"):
            self._progress(on_progress, engine, "start", "cleanup", "지난 실행이 남긴 백엔드 프로세스를 정리했습니다", 10)
            time.sleep(1.5)   # 포트가 돌아올 시간
        port = self._choose_port(engine)
        argv, cwd, env = self._launch_argv(engine, port)
        log_dir = self._engine_root(engine) / "logs"
        log_path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}.log"
        if cancel.is_set():
            raise BackendRuntimeError(
                "START_CANCELLED", "앱 종료 요청으로 백엔드 시작을 취소했습니다", stage="start"
            )
        self._progress(on_progress, engine, "start", "launch", "격리 백엔드 프로세스를 시작합니다", 20)
        try:
            process = self.adapter.start(argv, cwd=cwd, env=env, log_path=log_path)
        except OSError as exc:
            raise BackendRuntimeError(
                "PROCESS_START_FAILED", "백엔드 프로세스를 시작하지 못했습니다",
                stage="launch", retryable=True, details={"reason": str(exc)},
            ) from exc
        self._guard_process(engine, process, argv)
        endpoint = f"http://127.0.0.1:{port}"
        owned = _OwnedProcess(
            process=process,
            endpoint=endpoint,
            port=port,
            nonce=env["AISTUDIO_LAUNCH_NONCE"],
            started_at=_utc_now(),
            log_path=log_path,
        )
        with self._process_lock:
            self._processes[engine] = owned
            self._healthy[engine] = False

        deadline = time.monotonic() + self.health_timeout
        while time.monotonic() < deadline:
            with self._process_lock:
                still_owned = self._processes.get(engine) is owned
            if cancel.is_set() or not still_owned:
                self._stop(engine, on_progress)
                raise BackendRuntimeError(
                    "START_CANCELLED", "앱 종료 요청으로 백엔드 시작을 취소했습니다",
                    stage="health",
                )
            returncode = process.poll()
            if returncode is not None:
                with self._process_lock:
                    if self._processes.get(engine) is owned:
                        self._processes.pop(engine, None)
                    self._healthy[engine] = False
                self._close_process_log(process)
                raise BackendRuntimeError(
                    "PROCESS_EXITED_EARLY", f"백엔드가 준비 전에 종료되었습니다 (code {returncode})",
                    stage="health", retryable=True,
                    details={"logPath": str(log_path)},
                )
            if self.adapter.probe(endpoint, ENGINE_DEFINITIONS[engine].health_path, timeout=2):
                with self._process_lock:
                    if cancel.is_set() or self._processes.get(engine) is not owned:
                        continue
                    self._healthy[engine] = True
                with self._state_lock:
                    self._state["engines"][engine]["port"] = port
                    self._save_state()
                result = {
                    "message": f"{ENGINE_DEFINITIONS[engine].name}가 준비되었습니다",
                    "apiUrl": endpoint,
                    "protocol": ENGINE_DEFINITIONS[engine].protocol,
                    "pid": process.pid,
                    "logPath": str(log_path),
                }
                if replaced_engine:
                    result["replacedEngine"] = replaced_engine
                return result
            elapsed = self.health_timeout - max(0, deadline - time.monotonic())
            pct = 25 + int(min(65, elapsed / max(1.0, self.health_timeout) * 65))
            self._progress(on_progress, engine, "start", "health", "API 준비를 기다리는 중입니다", pct)
            cancel.wait(0.5)
        self._stop(engine, on_progress)
        raise BackendRuntimeError(
            "ENGINE_HEALTH_TIMEOUT", "백엔드 API가 제한 시간 내 준비되지 않았습니다",
            stage="health", retryable=True, details={"logPath": str(log_path)},
        )

    def _pid_file(self, engine: str) -> Path:
        """지난 실행의 PID 와 표식 — data-dir 바깥(엔진 루트)에 둔다. Forge 가 만질 일이 없게."""
        return self._engine_root(engine) / "app_process.pid"

    def _guard_process(self, engine: str, process: ProcessHandle, argv: Sequence[str]) -> None:
        """앱이 어떻게 죽든 자식이 따라 죽게(Job Object) + 다음 시작 청소용 PID 파일.

        둘 다 실패해도 시작은 계속된다 — 안전장치가 없다고 백엔드를 못 쓰게 하진 않는다.
        PID 파일은 표식(data-dir)이 실제 argv 에 있을 때만 쓴다: 그래야 청소 때 명령줄로
        우리 것임을 확인할 수 있다.
        """
        job = app_job()
        if not job.assign(process) and job.error:
            print(f"[Runtime] Job Object 미적용 — 강제 종료 시 {engine} 가 남을 수 있음: {job.error}")
        marker = marker_for(self._data_root(engine))
        pid = getattr(process, "pid", None)
        if pid and any(str(item) == marker for item in argv):
            try:
                write_pid_file(self._pid_file(engine), int(pid), marker)
            except OSError as exc:
                print(f"[Runtime] PID 파일 기록 실패: {exc}")

    @staticmethod
    def _close_process_log(process: ProcessHandle) -> None:
        handle = getattr(process, "_aistudio_log_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _stop(self, engine: str, on_progress: ProgressCallback | None) -> dict[str, Any]:
        self._start_cancel[engine].set()
        with self._process_lock:
            owned = self._processes.pop(engine, None)
            self._healthy[engine] = False
        if owned is None or owned.process.poll() is not None:
            was_owned = owned is not None
            if owned is not None:
                self._close_process_log(owned.process)
            return {
                "message": f"{ENGINE_DEFINITIONS[engine].name}가 이미 중지되어 있습니다",
                # A dead handle created by this manager still identifies the
                # endpoint as app-owned.  A fresh manager has no handle at all
                # and must not disturb an external user.bat process/UI.
                "stopped": was_owned,
                "owned": was_owned,
            }

        self._progress(on_progress, engine, "stop", "stopping", "앱이 시작한 프로세스를 종료합니다", 40)
        process = owned.process
        stopped = False
        try:
            # PID comes from the live Popen handle created by this manager; no stale
            # pid file or unrelated external process is ever passed to taskkill.
            if os.name == "nt" and isinstance(process, subprocess.Popen):
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
                try:
                    process.wait(timeout=12)
                except Exception:
                    pass
            else:
                try:
                    process.terminate()
                    process.wait(timeout=12)
                except Exception:
                    pass
            if process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
            stopped = process.poll() is not None
        finally:
            if not stopped:
                # Preserve the live handle so a later STOP/quit can retry.  Never
                # report success after losing ownership of a surviving process.
                with self._process_lock:
                    self._processes.setdefault(engine, owned)
            else:
                self._close_process_log(process)
                clear_pid_file(self._pid_file(engine))
        if not stopped:
            raise BackendRuntimeError(
                "PROCESS_STOP_FAILED",
                f"{ENGINE_DEFINITIONS[engine].name} 프로세스 트리를 종료하지 못했습니다",
                stage="stop",
                retryable=True,
                details={"pid": process.pid, "logPath": str(owned.log_path)},
            )
        return {
            "message": f"{ENGINE_DEFINITIONS[engine].name}를 중지했습니다",
            "stopped": True,
            "owned": True,
        }

    def stop_all_owned(self, on_progress: ProgressCallback | None = None) -> None:
        """Stop only backend/command handles created by this manager instance."""
        self._shutdown_requested.set()
        for cancel in self._start_cancel.values():
            cancel.set()
        shutdown_adapter = getattr(self.adapter, "shutdown", None)
        if callable(shutdown_adapter):
            try:
                shutdown_adapter()
            except Exception:
                pass
        for engine in tuple(ENGINE_DEFINITIONS):
            try:
                self._stop(engine, on_progress)
            except Exception:
                pass

    # --------------------------------------------------------------- extensions

    @staticmethod
    def _validate_repository_url(value: str) -> tuple[str, str]:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise BackendRuntimeError(
                "INVALID_EXTENSION_SOURCE",
                "확장은 인증정보가 없는 HTTPS Git 저장소 URL만 설치할 수 있습니다",
                stage="extension",
            )
        name = Path(parsed.path.rstrip("/")).name
        if name.casefold().endswith(".git"):
            name = name[:-4]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name or ""):
            raise BackendRuntimeError(
                "INVALID_EXTENSION_SOURCE", "확장 저장소 이름이 안전하지 않습니다", stage="extension"
            )
        return url, name

    def _extension_by_id(self, engine: str, payload: Mapping[str, Any]) -> Path:
        extension_id = str(payload.get("id") or payload.get("extensionId") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", extension_id):
            raise BackendRuntimeError("INVALID_EXTENSION_TARGET", "확장 ID가 올바르지 않습니다")
        root = self._extension_root(engine).resolve()
        target = (root / extension_id).resolve()
        if not _is_relative_to(target, root) or not target.is_dir():
            raise BackendRuntimeError("INVALID_EXTENSION_TARGET", "설치된 확장을 찾을 수 없습니다")
        return target

    def _install_extension(
        self,
        engine: str,
        payload: Mapping[str, Any],
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        state = self._install_state(engine)
        installed = bool(state["installed"])
        external_directory = bool(state.get("extensionDirExternal"))
        if not installed and not external_directory:
            raise BackendRuntimeError(
                "NOT_INSTALLED",
                "관리형 백엔드를 설치하거나 기존 백엔드의 확장 폴더를 먼저 지정하세요",
                stage="extension",
            )
        if not state.get("extensionWritable"):
            raise BackendRuntimeError(
                "EXTENSION_WRITE_NOT_APPROVED",
                "자동 탐지한 기존 설치의 확장 폴더는 읽기 전용입니다. 저장할 확장 폴더를 명시적으로 지정하세요",
                stage="extension",
            )
        url, name = self._validate_repository_url(str(payload.get("repoUrl") or payload.get("url") or ""))
        root = self._ensure_extension_mount(engine) if installed else self._validate_extension_root(
            self._extension_root(engine), engine=engine
        )
        target = (root / name).resolve()
        if not _is_relative_to(target, root.resolve()):
            raise BackendRuntimeError("INVALID_EXTENSION_TARGET", "확장 설치 경로가 루트를 벗어납니다")
        if target.exists():
            raise BackendRuntimeError("EXTENSION_ALREADY_EXISTS", f"이미 설치된 확장입니다: {name}")
        staging = root / f".{name}.aistudio-{uuid.uuid4().hex[:8]}"
        was_running = self._process_running(engine)
        if was_running:
            self._stop(engine, on_progress)
        self._progress(on_progress, engine, "install_extension", "download", f"{name} 저장소를 다운로드합니다", 20)
        dependencies_pending = False
        requirements_path = ""
        try:
            result = self.adapter.run(
                ["git", "clone", url, str(staging)], cwd=root, timeout=900,
                on_line=self._command_progress(on_progress, engine, "install_extension", "download", 35),
            )
            if result.returncode != 0:
                raise BackendRuntimeError(
                    "EXTENSION_FETCH_FAILED", f"{name} 다운로드에 실패했습니다",
                    stage="extension", retryable=True,
                    details={"output": result.output[-2000:]},
                )
            requirements = staging / "requirements.txt"
            if requirements.is_file():
                if installed and state.get("sourceMode") == "managed":
                    self._pip_install(
                        engine, self._python_path(engine), requirements,
                        on_progress, "install_extension", 65,
                    )
                else:
                    dependencies_pending = True
            os.replace(staging, target)
            if dependencies_pending:
                requirements_path = str(target / "requirements.txt")
        except Exception:
            if staging.exists() and _is_relative_to(staging.resolve(), root.resolve()):
                shutil.rmtree(staging, ignore_errors=True)
            if was_running:
                try:
                    self._start(engine, on_progress, install_if_missing=False)
                except Exception:
                    pass
            raise
        if was_running:
            self._start(engine, on_progress, install_if_missing=False)
        if dependencies_pending:
            message = (
                f"{name} 확장 코드를 설치했습니다 · 기존 백엔드 Python에서 "
                "requirements.txt 설치가 필요합니다"
            )
        else:
            message = f"{name} 확장을 설치했습니다"
            if was_running:
                message += " · 백엔드를 재시작해 적용했습니다"
        return {
            "message": message,
            "extensionId": name,
            "restartRequired": False,
            "dependenciesInstalled": not dependencies_pending,
            "dependenciesPending": dependencies_pending,
            "requirementsPath": requirements_path,
        }

    def _check_extension(
        self,
        engine: str,
        payload: Mapping[str, Any],
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        target = self._extension_by_id(engine, payload)
        if not (target / ".git").is_dir():
            raise BackendRuntimeError("EXTENSION_NOT_GIT", "Git으로 설치된 확장만 버전을 확인할 수 있습니다")
        url = self._git_origin_url(target)
        if not url:
            raise BackendRuntimeError("EXTENSION_REMOTE_MISSING", "확장 원격 저장소가 없습니다")
        self._progress(on_progress, engine, "check_extension", "version_check", f"{target.name} 버전을 확인합니다", 40)
        remote_result = self.adapter.run(["git", "ls-remote", url, "HEAD"], cwd=target, timeout=60)
        if remote_result.returncode != 0 or not remote_result.output.strip():
            raise BackendRuntimeError("VERSION_CHECK_FAILED", "확장 원격 버전을 확인하지 못했습니다", retryable=True)
        remote = remote_result.output.strip().split()[0]
        local = self._capture(["git", "rev-parse", "HEAD"], cwd=target)
        marker = self._extension_marker(engine, target.name)
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            str(marker),
            {
                "repoUrl": self._sanitize_repository_url(url),
                "remoteCommit": remote,
                "lastChecked": _utc_now(),
            },
            indent=2,
        )
        available = bool(local and remote != local)
        return {
            "message": f"{target.name}: " + ("업데이트가 있습니다" if available else "최신입니다"),
            "extensionId": target.name,
            "localCommit": local,
            "remoteCommit": remote,
            "updateAvailable": available,
        }

    def _update_extension(
        self,
        engine: str,
        payload: Mapping[str, Any],
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        state = self._install_state(engine)
        installed = bool(state["installed"])
        if not installed and not state.get("extensionDirExternal"):
            raise BackendRuntimeError(
                "NOT_INSTALLED",
                "관리형 백엔드를 설치하거나 기존 백엔드의 확장 폴더를 먼저 지정하세요",
                stage="extension_update",
            )
        if not state.get("extensionWritable"):
            raise BackendRuntimeError(
                "EXTENSION_WRITE_NOT_APPROVED",
                "자동 탐지한 기존 설치의 확장 폴더는 읽기 전용입니다. 저장할 확장 폴더를 명시적으로 지정하세요",
                stage="extension_update",
            )
        target = self._extension_by_id(engine, payload)
        if not (target / ".git").is_dir():
            raise BackendRuntimeError("EXTENSION_NOT_GIT", "Git으로 설치된 확장만 업데이트할 수 있습니다")
        dirty = self._capture(["git", "status", "--porcelain"], cwd=target)
        if dirty:
            raise BackendRuntimeError(
                "EXTENSION_DIRTY", "수정된 파일이 있어 확장을 자동 업데이트하지 않습니다",
                stage="extension_update",
            )
        was_running = self._process_running(engine)
        if was_running:
            self._stop(engine, on_progress)
        dependencies_pending = False
        requirements_path = ""
        try:
            self._progress(on_progress, engine, "update_extension", "download", f"{target.name}을 업데이트합니다", 35)
            result = self.adapter.run(
                ["git", "pull", "--ff-only"], cwd=target, timeout=600,
                on_line=self._command_progress(on_progress, engine, "update_extension", "download", 50),
            )
            if result.returncode != 0:
                raise BackendRuntimeError(
                    "EXTENSION_UPDATE_FAILED", "확장을 fast-forward 업데이트하지 못했습니다",
                    stage="extension_update", retryable=True,
                    details={"output": result.output[-2000:]},
                )
            requirements = target / "requirements.txt"
            if requirements.is_file():
                if installed and state.get("sourceMode") == "managed":
                    self._pip_install(
                        engine, self._python_path(engine), requirements,
                        on_progress, "update_extension", 75,
                    )
                else:
                    dependencies_pending = True
                    requirements_path = str(requirements)
            checked = self._check_extension(engine, {"id": target.name}, on_progress)
        except Exception:
            if was_running:
                try:
                    self._start(engine, on_progress, install_if_missing=False)
                except Exception:
                    pass
            raise
        if was_running:
            self._start(engine, on_progress, install_if_missing=False)
        if dependencies_pending:
            message = (
                f"{target.name} 확장 코드를 업데이트했습니다 · 기존 백엔드 Python에서 "
                "requirements.txt 설치가 필요합니다"
            )
        else:
            message = f"{target.name} 확장을 업데이트했습니다"
            if was_running:
                message += " · 백엔드를 재시작해 적용했습니다"
        return {
            "message": message,
            "extensionId": target.name,
            "restartRequired": False,
            "dependenciesInstalled": not dependencies_pending,
            "dependenciesPending": dependencies_pending,
            "requirementsPath": requirements_path,
            **{key: value for key, value in checked.items() if key not in {"message"}},
        }


_MANAGER: BackendRuntimeManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_backend_runtime_manager() -> BackendRuntimeManager:
    """Process-wide manager preserving the owned ``Popen`` handles."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            # 켜 둔 Generation API 의 포트는 시작 순서와 무관하게 엔진 포트 선택에서 뺀다.
            from core.generation_api_port import reserved_ports as generation_api_reserved_ports

            _MANAGER = BackendRuntimeManager(reserved_ports=generation_api_reserved_ports)
        return _MANAGER
