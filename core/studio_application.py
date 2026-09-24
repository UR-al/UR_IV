"""Transport-neutral application seam shared by desktop and web adapters.

``StudioApplication`` deliberately has no Qt or HTTP dependency.  Adapters
authenticate a caller, build a :class:`CallContext`, and translate JSON to the
single request envelope accepted by :meth:`StudioApplication.invoke`.
"""

from __future__ import annotations

import copy
import logging
import re
import threading
import uuid
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
NATIVE_CAPABILITY = "native"
_REQUEST_KEYS = frozenset({"version", "requestId", "operation", "input"})
_ENGINE_ALIASES = {
    "forge": "forge",
    "forge_neo": "forge",
    "forge-neo": "forge",
    "comfy": "comfyui",
    "comfyui": "comfyui",
}
_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\r\n\"'<>|]+")
_UNC_PATH = re.compile(r"\\\\[^\s\\/]+[\\/][^\r\n\"'<>|]+")


@dataclass(frozen=True, slots=True)
class CallContext:
    """Identity and authority supplied by a trusted transport adapter."""

    principal_id: str
    transport: str
    capabilities: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        principal = str(self.principal_id or "").strip()
        transport = str(self.transport or "").strip()
        if not principal:
            raise ValueError("principal_id is required")
        if not transport:
            raise ValueError("transport is required")
        capabilities = frozenset(
            str(value).strip() for value in self.capabilities if str(value).strip()
        )
        object.__setattr__(self, "principal_id", principal)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "capabilities", capabilities)


class StudioApplicationError(RuntimeError):
    """Structured application error safe to translate at any adapter."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(str(message))
        self.code = str(code or "INTERNAL")
        self.message = str(message or "작업을 완료하지 못했습니다")
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            result["details"] = copy.deepcopy(self.details)
        return result


@dataclass(slots=True)
class _Subscriber:
    context: CallContext
    sink: Callable[[dict[str, Any]], None]
    queue: deque[dict[str, Any]] = field(default_factory=deque)
    draining: bool = False
    active: bool = True


_OPERATIONS: dict[str, dict[str, Any]] = {
    "sync.bootstrap": {"kind": "query", "native": False},
    "runtime.snapshot": {"kind": "query", "native": False},
    "runtime.execute": {"kind": "job", "native": True},
    "generation_api.snapshot": {"kind": "query", "native": False},
    "generation_api.execute": {"kind": "job", "native": True},
    "app_update.snapshot": {"kind": "query", "native": False},
    "app_update.execute": {"kind": "job", "native": True},
    "model_paths.snapshot": {"kind": "query", "native": False},
    "model_paths.save": {"kind": "command", "native": True},
    "model_paths.reset": {"kind": "command", "native": True},
    "model_paths.refresh": {"kind": "command", "native": True},
    "native.pick_directory": {"kind": "command", "native": True},
}


class StudioApplication:
    """Deep application module behind QWebChannel and HTTP adapters.

    Long operations return an ``accepted`` reply and finish on a daemon worker.
    Their lifecycle is observable through the one ordered event journal exposed
    by :meth:`subscribe`.
    """

    def __init__(
        self,
        host: Any = None,
        runtime_manager: Any = None,
        generation_api_manager: Any = None,
        app_update_manager: Any = None,
    ) -> None:
        self._host = host
        self._runtime_manager = runtime_manager
        self._generation_api_manager = generation_api_manager
        self._app_update_manager = app_update_manager
        self._dependency_lock = threading.Lock()

        self._lock = threading.RLock()
        self._event_epoch = uuid.uuid4().hex
        self._seq = 0
        self._journal: deque[dict[str, Any]] = deque(maxlen=1024)
        self._subscribers: dict[int, _Subscriber] = {}
        self._next_subscription = 1
        self._workers: set[threading.Thread] = set()
        self._generation_job_lock = threading.Lock()
        self._generation_job_id = ""
        self._app_update_job_lock = threading.Lock()
        self._app_update_job_id = ""

    # ------------------------------------------------------------------ Interface

    @property
    def event_epoch(self) -> str:
        """Opaque identity for this in-memory event journal instance."""

        return self._event_epoch

    def describe(self, context: CallContext) -> dict[str, Any]:
        context = self._require_context(context)
        native = self._has_capability(context, NATIVE_CAPABILITY)
        with self._lock:
            cursor = self._seq
        return {
            "version": PROTOCOL_VERSION,
            "principalId": context.principal_id,
            "transport": context.transport,
            "capabilities": sorted(context.capabilities),
            "nativeOperations": native,
            "eventEpoch": self._event_epoch,
            "eventCursor": cursor,
            "topics": [
                "runtime.operation",
                "generation_api.operation",
                "app_update.operation",
                "model_paths.changed",
            ],
            "operations": [
                {
                    "name": name,
                    "kind": spec["kind"],
                    "available": native or not spec["native"],
                    "requiredCapability": NATIVE_CAPABILITY if spec["native"] else "",
                }
                for name, spec in _OPERATIONS.items()
            ],
        }

    def invoke(
        self,
        context: CallContext,
        request_mapping: Mapping[str, Any],
    ) -> dict[str, Any]:
        request_id = ""
        try:
            context = self._require_context(context)
            request = self._validate_envelope(request_mapping)
            request_id = request["requestId"]
            operation = request["operation"]
            values = request["input"]
            spec = _OPERATIONS.get(operation)
            if spec is None:
                raise StudioApplicationError(
                    "NOT_FOUND", f"지원하지 않는 operation입니다: {operation}"
                )
            if spec["native"] and not self._has_capability(context, NATIVE_CAPABILITY):
                raise StudioApplicationError(
                    "FORBIDDEN",
                    "이 작업은 데스크톱 native 권한이 필요합니다",
                )

            if operation == "sync.bootstrap":
                self._require_input_keys(values, allowed={"includeAppUpdate"})
                include_app_update = values.get("includeAppUpdate", True)
                if not isinstance(include_app_update, bool):
                    raise StudioApplicationError(
                        "INVALID_ARGUMENT", "includeAppUpdate는 true/false여야 합니다"
                    )
                data = {
                    "eventEpoch": self._event_epoch,
                    "description": self.describe(context),
                    "runtime": self._runtime_snapshot(context),
                    "generationApi": self._generation_api_snapshot(context),
                    "modelPaths": self._model_paths_snapshot(context),
                }
                # 앱 업데이트 스냅샷은 git 을 실행한다 — 쓰지 않는 호출자(Settings)는 뺄 수 있다.
                if include_app_update:
                    data["appUpdate"] = self._app_update_snapshot(context)
                return self._reply(request_id, "ok", data=data)

            if operation == "runtime.snapshot":
                self._require_input_keys(values)
                return self._reply(
                    request_id, "ok", data=self._runtime_snapshot(context)
                )

            if operation == "runtime.execute":
                self._require_input_keys(
                    values,
                    allowed={"engine", "action", "payload"},
                    required={"engine", "action"},
                )
                engine = self._canonical_engine(values["engine"])
                action = self._required_text(values, "action", limit=100)
                payload = self._mapping_value(values, "payload", default={})
                return self._start_runtime_job(
                    context, request_id, engine, action, payload
                )

            if operation == "generation_api.snapshot":
                self._require_input_keys(values)
                return self._reply(
                    request_id, "ok", data=self._generation_api_snapshot(context)
                )

            if operation == "generation_api.execute":
                self._require_input_keys(
                    values,
                    allowed={"action", "payload"},
                    required={"action"},
                )
                action = self._required_text(values, "action", limit=100)
                payload = self._mapping_value(values, "payload", default={})
                return self._start_generation_api_job(
                    context, request_id, action, payload
                )

            if operation == "app_update.snapshot":
                self._require_input_keys(values)
                return self._reply(
                    request_id, "ok", data=self._app_update_snapshot(context)
                )

            if operation == "app_update.execute":
                self._require_input_keys(
                    values,
                    allowed={"action", "payload"},
                    required={"action"},
                )
                action = self._required_text(values, "action", limit=100)
                payload = self._mapping_value(values, "payload", default={})
                return self._start_app_update_job(
                    context, request_id, action, payload
                )

            if operation == "model_paths.snapshot":
                self._require_input_keys(values)
                return self._reply(
                    request_id, "ok", data=self._model_paths_snapshot(context)
                )

            if operation == "model_paths.save":
                self._require_input_keys(values, allowed={"paths"}, required={"paths"})
                paths = self._mapping_value(values, "paths")
                state = self._save_model_paths(paths)
                self._refresh_model_widgets(required=False)
                public = self._public_model_paths(state, context)
                event = self._publish(
                    topic="model_paths.changed",
                    event_type="changed",
                    operation=operation,
                    request_id=request_id,
                    data={"snapshot": state},
                )
                return self._reply(request_id, "ok", data=public, seq=event["seq"])

            if operation == "model_paths.reset":
                self._require_input_keys(values)
                state = self._reset_model_paths()
                self._refresh_model_widgets(required=False)
                public = self._public_model_paths(state, context)
                event = self._publish(
                    topic="model_paths.changed",
                    event_type="changed",
                    operation=operation,
                    request_id=request_id,
                    data={"snapshot": state},
                )
                return self._reply(request_id, "ok", data=public, seq=event["seq"])

            if operation == "model_paths.refresh":
                self._require_input_keys(values)
                self._refresh_model_widgets(required=True)
                state = self._raw_model_paths_snapshot()
                public = self._public_model_paths(state, context)
                event = self._publish(
                    topic="model_paths.changed",
                    event_type="changed",
                    operation=operation,
                    request_id=request_id,
                    data={"snapshot": state},
                )
                return self._reply(request_id, "ok", data=public, seq=event["seq"])

            if operation == "native.pick_directory":
                self._require_input_keys(
                    values,
                    allowed={"purpose", "engine", "key"},
                    required={"purpose"},
                )
                data = self._pick_directory(values)
                return self._reply(request_id, "ok", data=data)

            raise StudioApplicationError("NOT_FOUND", "지원하지 않는 operation입니다")
        except Exception as exc:
            error = self._normalise_error(exc, context if isinstance(context, CallContext) else None)
            return self._reply(request_id, "error", error=error.as_dict())

    def subscribe(
        self,
        context: CallContext,
        sink: Callable[[dict[str, Any]], None],
        after_seq: int = 0,
    ) -> Callable[[], None]:
        """Replay ``seq > after_seq`` and then deliver live events in order.

        The returned callable is idempotent and thread-safe.  It prevents future
        delivery; a sink invocation already in progress is allowed to finish.
        """

        context = self._require_context(context)
        if not callable(sink):
            raise TypeError("sink must be callable")
        if isinstance(after_seq, bool):
            raise TypeError("after_seq must be a non-negative integer")
        try:
            cursor = int(after_seq)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("after_seq must be a non-negative integer") from exc
        if cursor < 0:
            raise ValueError("after_seq must be a non-negative integer")

        with self._lock:
            earliest_seq = (
                int(self._journal[0]["seq"])
                if self._journal
                else self._seq + 1
            )
            if cursor > self._seq or (
                self._journal and cursor < earliest_seq - 1
            ):
                raise StudioApplicationError(
                    "CURSOR_EXPIRED",
                    "event cursor가 현재 journal 범위와 일치하지 않습니다",
                    retryable=True,
                    details={
                        "earliestSeq": earliest_seq,
                        "currentSeq": self._seq,
                    },
                )
            handle = self._next_subscription
            self._next_subscription += 1
            subscriber = _Subscriber(context=context, sink=sink, draining=True)
            subscriber.queue.extend(
                copy.deepcopy(event) for event in self._journal if event["seq"] > cursor
            )
            self._subscribers[handle] = subscriber

        self._drain_subscriber(subscriber)

        unsubscribe_lock = threading.Lock()
        unsubscribed = False

        def unsubscribe() -> None:
            nonlocal unsubscribed
            with unsubscribe_lock:
                if unsubscribed:
                    return
                unsubscribed = True
            with self._lock:
                current = self._subscribers.pop(handle, None)
                if current is not None:
                    current.active = False
                    current.queue.clear()

        return unsubscribe

    # ----------------------------------------------------------- request validation

    @staticmethod
    def _require_context(context: CallContext) -> CallContext:
        if not isinstance(context, CallContext):
            raise TypeError("context must be CallContext")
        return context

    @staticmethod
    def _validate_envelope(request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise StudioApplicationError("INVALID_ARGUMENT", "request는 객체여야 합니다")
        keys = set(request)
        missing = sorted(_REQUEST_KEYS - keys)
        unknown = sorted(str(key) for key in keys - _REQUEST_KEYS)
        if missing or unknown:
            details: dict[str, Any] = {}
            if missing:
                details["missing"] = missing
            if unknown:
                details["unknown"] = unknown
            raise StudioApplicationError(
                "INVALID_ARGUMENT",
                "request envelope가 올바르지 않습니다",
                details=details,
            )
        version = request.get("version")
        if isinstance(version, bool) or version != PROTOCOL_VERSION:
            raise StudioApplicationError(
                "UNSUPPORTED_VERSION", f"지원하는 protocol version은 {PROTOCOL_VERSION}입니다"
            )
        request_id = str(request.get("requestId") or "").strip()
        if not request_id or len(request_id) > 128 or any(ord(ch) < 32 for ch in request_id):
            raise StudioApplicationError(
                "INVALID_ARGUMENT", "requestId는 1~128자의 문자열이어야 합니다"
            )
        operation = str(request.get("operation") or "").strip()
        if not operation or len(operation) > 128:
            raise StudioApplicationError("INVALID_ARGUMENT", "operation이 필요합니다")
        values = request.get("input")
        if not isinstance(values, Mapping):
            raise StudioApplicationError("INVALID_ARGUMENT", "input은 객체여야 합니다")
        return {
            "version": PROTOCOL_VERSION,
            "requestId": request_id,
            "operation": operation,
            "input": dict(values),
        }

    @staticmethod
    def _require_input_keys(
        values: Mapping[str, Any],
        *,
        allowed: set[str] | None = None,
        required: set[str] | None = None,
    ) -> None:
        allowed = allowed or set()
        required = required or set()
        keys = set(values)
        unknown = sorted(str(key) for key in keys - allowed)
        missing = sorted(required - keys)
        if unknown or missing:
            details: dict[str, Any] = {}
            if missing:
                details["missing"] = missing
            if unknown:
                details["unknown"] = unknown
            raise StudioApplicationError(
                "INVALID_ARGUMENT", "operation input이 올바르지 않습니다", details=details
            )

    @staticmethod
    def _required_text(values: Mapping[str, Any], key: str, *, limit: int) -> str:
        raw = values.get(key)
        if not isinstance(raw, str):
            raise StudioApplicationError("INVALID_ARGUMENT", f"input.{key}는 문자열이어야 합니다")
        value = raw.strip()
        if not value or len(value) > limit:
            raise StudioApplicationError(
                "INVALID_ARGUMENT", f"input.{key}는 1~{limit}자여야 합니다"
            )
        return value

    @staticmethod
    def _mapping_value(
        values: Mapping[str, Any],
        key: str,
        *,
        default: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = values.get(key, default)
        if not isinstance(raw, Mapping):
            raise StudioApplicationError("INVALID_ARGUMENT", f"input.{key}는 객체여야 합니다")
        return copy.deepcopy(dict(raw))

    @staticmethod
    def _canonical_engine(value: Any) -> str:
        key = _ENGINE_ALIASES.get(str(value or "").strip().casefold(), "")
        if not key:
            raise StudioApplicationError(
                "INVALID_ARGUMENT", f"지원하지 않는 runtime engine입니다: {value}"
            )
        return key

    @staticmethod
    def _has_capability(context: CallContext, capability: str) -> bool:
        if "*" in context.capabilities or capability in context.capabilities:
            return True
        namespace = capability.split(":", 1)[0]
        return f"{namespace}:*" in context.capabilities

    # ----------------------------------------------------------- manager snapshots

    def _get_runtime_manager(self) -> Any:
        if self._runtime_manager is None:
            with self._dependency_lock:
                if self._runtime_manager is None:
                    from core.backend_runtime import get_backend_runtime_manager

                    self._runtime_manager = get_backend_runtime_manager()
        return self._runtime_manager

    def _get_generation_api_manager(self) -> Any:
        if self._generation_api_manager is None:
            with self._dependency_lock:
                if self._generation_api_manager is None:
                    from core.generation_api import get_generation_api_manager

                    self._generation_api_manager = get_generation_api_manager()
        return self._generation_api_manager

    def _get_app_update_manager(self) -> Any:
        if self._app_update_manager is None:
            with self._dependency_lock:
                if self._app_update_manager is None:
                    from core.app_updater import get_app_update_manager

                    self._app_update_manager = get_app_update_manager()
        return self._app_update_manager

    def _runtime_snapshot(self, context: CallContext) -> dict[str, Any]:
        manager = self._get_runtime_manager()
        snapshot = manager.snapshot()
        if not isinstance(snapshot, Mapping):
            raise StudioApplicationError("INTERNAL", "runtime snapshot 형식이 올바르지 않습니다")
        normalised = self._normalise_runtime_snapshot(snapshot)
        return self._public_runtime(normalised, context)

    def _generation_api_snapshot(self, context: CallContext) -> dict[str, Any]:
        manager = self._get_generation_api_manager()
        include_secret = self._has_capability(context, NATIVE_CAPABILITY)
        try:
            snapshot = manager.snapshot(include_secret=include_secret)
        except TypeError:
            snapshot = manager.snapshot(include_secret)
        if not isinstance(snapshot, Mapping):
            raise StudioApplicationError(
                "INTERNAL", "generation API snapshot 형식이 올바르지 않습니다"
            )
        return self._public_generation_api(dict(snapshot), context)

    def _app_update_snapshot(self, context: CallContext) -> dict[str, Any]:
        snapshot = self._get_app_update_manager().snapshot()
        if not isinstance(snapshot, Mapping):
            raise StudioApplicationError(
                "INTERNAL", "app update snapshot 형식이 올바르지 않습니다"
            )
        return self._public_app_update(dict(snapshot), context)

    def _model_paths_snapshot(self, context: CallContext) -> dict[str, Any]:
        return self._public_model_paths(self._raw_model_paths_snapshot(), context)

    @staticmethod
    def _raw_model_paths_snapshot() -> dict[str, Any]:
        from core.forge_modules import get_forge_path_state

        state = get_forge_path_state()
        if not isinstance(state, Mapping):
            raise StudioApplicationError("INTERNAL", "model path snapshot 형식이 올바르지 않습니다")
        return copy.deepcopy(dict(state))

    @staticmethod
    def _save_model_paths(paths: Mapping[str, Any]) -> dict[str, Any]:
        from core.forge_modules import get_forge_path_state, save_forge_paths

        save_forge_paths(paths)
        return copy.deepcopy(get_forge_path_state())

    @staticmethod
    def _reset_model_paths() -> dict[str, Any]:
        from core.forge_modules import get_forge_path_state, reset_forge_paths

        reset_forge_paths()
        return copy.deepcopy(get_forge_path_state())

    # -------------------------------------------------------------- native hosting

    def _refresh_model_widgets(self, *, required: bool) -> None:
        refresh = getattr(self._host, "refresh_model_widgets", None)
        if not callable(refresh):
            if required:
                raise StudioApplicationError(
                    "UNAVAILABLE", "native model refresh host를 사용할 수 없습니다", retryable=True
                )
            return
        refresh()

    def _pick_directory(self, values: Mapping[str, Any]) -> dict[str, Any]:
        purpose = self._required_text(values, "purpose", limit=64)
        if purpose not in {"runtime_install", "runtime_extension", "model_path"}:
            raise StudioApplicationError(
                "INVALID_ARGUMENT", "지원하지 않는 directory purpose입니다"
            )
        pick = getattr(self._host, "pick_directory", None)
        if not callable(pick):
            raise StudioApplicationError(
                "UNAVAILABLE", "native directory picker를 사용할 수 없습니다", retryable=True
            )

        selector = ""
        current = ""
        if purpose in {"runtime_install", "runtime_extension"}:
            if "key" in values:
                raise StudioApplicationError(
                    "INVALID_ARGUMENT", "runtime directory에는 key를 사용할 수 없습니다"
                )
            engine = self._canonical_engine(values.get("engine"))
            selector = engine
            snapshot = self._normalise_runtime_snapshot(self._get_runtime_manager().snapshot())
            engine_state = snapshot.get("engines", {}).get(engine, {})
            if purpose == "runtime_install":
                current = str(
                    engine_state.get("installRoot")
                    or engine_state.get("existingRoot")
                    or ""
                )
            else:
                current = str(engine_state.get("extensionDir") or "")
        else:
            if "engine" in values:
                raise StudioApplicationError(
                    "INVALID_ARGUMENT", "model path directory에는 engine을 사용할 수 없습니다"
                )
            from core.forge_modules import FORGE_PATH_KEYS

            key = str(values.get("key") or "").strip()
            if key not in FORGE_PATH_KEYS:
                raise StudioApplicationError(
                    "INVALID_ARGUMENT", "지원하지 않는 model path key입니다"
                )
            selector = key
            current = str(self._raw_model_paths_snapshot().get("paths", {}).get(key, ""))

        selected = pick(purpose, selector, current)
        if selected is None or not str(selected).strip():
            return {"ok": True, "cancelled": True, "purpose": purpose}
        return {
            "ok": True,
            "cancelled": False,
            "purpose": purpose,
            "path": str(selected).strip(),
            "engine" if purpose.startswith("runtime_") else "key": selector,
        }

    # --------------------------------------------------------------- async jobs

    def _start_runtime_job(
        self,
        context: CallContext,
        request_id: str,
        engine: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        startup = bool(payload.get("startup", False))

        def work_body() -> None:
            # manager 획득·started 발행도 안쪽 try 에 둔다 — 초기화 실패도 error 이벤트가 된다.
            try:
                manager = self._get_runtime_manager()
                started_snapshot = self._safe_runtime_snapshot()
                self._publish(
                    topic="runtime.operation",
                    event_type="started",
                    operation="runtime.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "engine": engine,
                        "action": action,
                        "snapshot": started_snapshot,
                    },
                )
                # 데스크톱 host 는 started/progress 를 시작 자동기동(startup) 안내에만 쓴다
                # (generator_main._on_backend_runtime_event) — 그 밖엔 보내지 않는다.
                if startup:
                    self._notify_runtime_host({
                        "engine": engine,
                        "type": "started",
                        "action": action,
                        "operationId": job_id,
                        "message": f"{engine} {action} 작업을 시작했습니다.",
                        "startup": startup,
                        "snapshot": started_snapshot,
                        "state": self._runtime_engine_state(started_snapshot, engine),
                    })
                from core.progress_snapshot_gate import ProgressSnapshotGate

                progress_snapshot = ProgressSnapshotGate()

                def progress(update: Any) -> None:
                    data = dict(update) if isinstance(update, Mapping) else {"message": str(update)}
                    event_data: dict[str, Any] = {
                        "engine": engine,
                        "action": action,
                        "update": data,
                    }
                    # 스냅샷(엔진 전체 스캔, 약 20ms)은 단계가 바뀔 때나 2초에 한 번만 싣는다.
                    # 명령 출력(0.2초 간격)·health 루프마다 새로 계산하던 것이 비용과 journal
                    # 메모리의 대부분이었다. Settings 는 스냅샷 없는 progress 도 처리한다.
                    if progress_snapshot.due(data):
                        event_data["snapshot"] = self._safe_runtime_snapshot()
                    self._publish(
                        topic="runtime.operation",
                        event_type="progress",
                        operation="runtime.execute",
                        job_id=job_id,
                        request_id=request_id,
                        data=event_data,
                    )
                    if startup:
                        self._notify_runtime_host({
                            **copy.deepcopy(data),
                            "engine": engine,
                            "type": "progress",
                            "action": action,
                            "operationId": job_id,
                            "startup": startup,
                        })

                result = manager.execute(engine, action, payload, on_progress=progress)
                if isinstance(result, Mapping) and result.get("ok") is False:
                    raise StudioApplicationError(
                        "OPERATION_FAILED",
                        str(result.get("message") or result.get("error") or "runtime 작업 실패"),
                    )
                snapshot = self._normalise_runtime_snapshot(manager.snapshot())
                self._publish(
                    topic="runtime.operation",
                    event_type="completed",
                    operation="runtime.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "engine": engine,
                        "action": action,
                        "result": copy.deepcopy(result),
                        "snapshot": snapshot,
                    },
                )
                result_mapping = (
                    {
                        str(key): copy.deepcopy(value)
                        for key, value in result.items()
                        if str(key) != "snapshot"
                    }
                    if isinstance(result, Mapping)
                    else {}
                )
                self._notify_runtime_host({
                    "engine": engine,
                    "type": "completed",
                    "action": action,
                    "operationId": job_id,
                    "ok": True,
                    "result": result_mapping,
                    "error": None,
                    "message": str(result_mapping.get("message") or ""),
                    "startup": startup,
                    "activate": bool(result_mapping.get("activate", False)),
                    "state": self._runtime_engine_state(snapshot, engine),
                    "snapshot": snapshot,
                })
            except Exception as exc:
                error = self._normalise_error(exc, context)
                snapshot = self._safe_runtime_snapshot()
                self._publish(
                    topic="runtime.operation",
                    event_type="error",
                    operation="runtime.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "engine": engine,
                        "action": action,
                        "error": error.as_dict(),
                        "snapshot": snapshot,
                    },
                )
                self._notify_runtime_host({
                    "engine": engine,
                    "type": "error",
                    "action": action,
                    "operationId": job_id,
                    "ok": False,
                    "result": {},
                    "error": error.as_dict(),
                    "message": error.message,
                    "startup": startup,
                    "activate": False,
                    "state": self._runtime_engine_state(snapshot, engine),
                    "snapshot": snapshot,
                })

        def work() -> None:
            try:
                work_body()
            except Exception as exc:
                error = self._report_job_failure(
                    topic="runtime.operation",
                    operation="runtime.execute",
                    job_id=job_id,
                    request_id=request_id,
                    context=context,
                    exc=exc,
                    data={"engine": engine, "action": action},
                )
                self._notify_runtime_host({
                    "engine": engine,
                    "type": "error",
                    "action": action,
                    "operationId": job_id,
                    "ok": False,
                    "result": {},
                    "error": error,
                    "message": str(error.get("message") or ""),
                    "startup": startup,
                    "activate": False,
                    "state": {},
                })

        return self._accept_job(
            topic="runtime.operation",
            operation="runtime.execute",
            job_id=job_id,
            request_id=request_id,
            worker_name=f"studio-runtime-{engine}-{action}",
            work=work,
            accepted_data={"engine": engine, "action": action},
        )

    @staticmethod
    def _runtime_engine_state(snapshot: Mapping[str, Any], engine: str) -> dict[str, Any]:
        engines = snapshot.get("engines")
        if not isinstance(engines, Mapping):
            return {}
        state = engines.get(engine)
        return copy.deepcopy(dict(state)) if isinstance(state, Mapping) else {}

    def _notify_runtime_host(self, payload: Mapping[str, Any]) -> None:
        """Preserve desktop runtime side effects behind an optional host seam.

        The application layer remains Qt-free.  Desktop composition can observe
        the established lifecycle payload and reconnect the newly started
        backend, invalidate model caches, or disable generation after a stop.
        """

        callback = getattr(self._host, "handle_runtime_event", None)
        if not callable(callback):
            return
        try:
            callback(copy.deepcopy(dict(payload)))
        except Exception:
            logger.exception("Studio runtime host callback failed")

    def _start_generation_api_job(
        self,
        context: CallContext,
        request_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        with self._generation_job_lock:
            active_job_id = self._generation_job_id
            if active_job_id:
                raise StudioApplicationError(
                    "OPERATION_BUSY",
                    "생성 API 설정 작업이 이미 진행 중입니다",
                    retryable=True,
                    details={"activeJobId": active_job_id},
                )
            self._generation_job_id = job_id

        def work_body() -> None:
            # manager 획득·started 발행도 안쪽 try 에 둔다 — 초기화 실패도 error 이벤트가 된다.
            try:
                manager = self._get_generation_api_manager()
                self._publish(
                    topic="generation_api.operation",
                    event_type="started",
                    operation="generation_api.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={"action": action, "snapshot": self._safe_generation_snapshot(True)},
                )
                result = manager.execute(action, payload)
                if isinstance(result, Mapping) and result.get("ok") is False:
                    raise StudioApplicationError(
                        "OPERATION_FAILED",
                        str(result.get("message") or result.get("error") or "generation API 작업 실패"),
                    )
                snapshot = self._snapshot_generation_manager(manager, include_secret=True)
                self._publish(
                    topic="generation_api.operation",
                    event_type="completed",
                    operation="generation_api.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "action": action,
                        "result": copy.deepcopy(result),
                        "snapshot": snapshot,
                    },
                )
            except Exception as exc:
                error = self._normalise_error(exc, context)
                self._publish(
                    topic="generation_api.operation",
                    event_type="error",
                    operation="generation_api.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "action": action,
                        "error": error.as_dict(),
                        "snapshot": self._safe_generation_snapshot(True),
                    },
                )

        def work() -> None:
            try:
                work_body()
            except Exception as exc:
                self._report_job_failure(
                    topic="generation_api.operation",
                    operation="generation_api.execute",
                    job_id=job_id,
                    request_id=request_id,
                    context=context,
                    exc=exc,
                    data={"action": action},
                )
            finally:
                self._release_generation_job(job_id)

        return self._accept_job(
            topic="generation_api.operation",
            operation="generation_api.execute",
            job_id=job_id,
            request_id=request_id,
            worker_name=f"studio-generation-api-{action}",
            work=work,
            accepted_data={"action": action},
            release_lock=lambda: self._release_generation_job(job_id),
        )

    def _release_generation_job(self, job_id: str) -> None:
        with self._generation_job_lock:
            if self._generation_job_id == job_id:
                self._generation_job_id = ""

    def _start_app_update_job(
        self,
        context: CallContext,
        request_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        with self._app_update_job_lock:
            active_job_id = self._app_update_job_id
            if active_job_id:
                raise StudioApplicationError(
                    "OPERATION_BUSY",
                    "업데이트 작업이 이미 진행 중입니다",
                    retryable=True,
                    details={"activeJobId": active_job_id},
                )
            self._app_update_job_id = job_id

        def work_body() -> None:
            # manager 획득·started 발행도 안쪽 try 에 둔다 — 초기화 실패도 error 이벤트가 된다.
            try:
                manager = self._get_app_update_manager()
                self._publish(
                    topic="app_update.operation",
                    event_type="started",
                    operation="app_update.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={"action": action, "snapshot": self._safe_app_update_snapshot()},
                )

                from core.progress_snapshot_gate import ProgressSnapshotGate

                progress_snapshot = ProgressSnapshotGate()

                def progress(update: Any) -> None:
                    data = dict(update) if isinstance(update, Mapping) else {"message": str(update)}
                    event_data: dict[str, Any] = {"action": action, "update": data}
                    # 업데이트 스냅샷은 git 을 실행한다 — 단계가 바뀔 때만 다시 싣는다.
                    if progress_snapshot.due(data):
                        event_data["snapshot"] = self._safe_app_update_snapshot()
                    self._publish(
                        topic="app_update.operation",
                        event_type="progress",
                        operation="app_update.execute",
                        job_id=job_id,
                        request_id=request_id,
                        data=event_data,
                    )

                result = manager.execute(action, payload, on_progress=progress)
                if isinstance(result, Mapping) and result.get("ok") is False:
                    raise StudioApplicationError(
                        "OPERATION_FAILED",
                        str(result.get("message") or result.get("error") or "업데이트 작업 실패"),
                    )
                snapshot = self._safe_app_update_snapshot()
                result_mapping = copy.deepcopy(dict(result)) if isinstance(result, Mapping) else {}
                self._publish(
                    topic="app_update.operation",
                    event_type="completed",
                    operation="app_update.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={"action": action, "result": result_mapping, "snapshot": snapshot},
                )
                if bool(result_mapping.get("restartRequired")):
                    self._request_app_update_restart(result_mapping)
            except Exception as exc:
                error = self._normalise_error(exc, context)
                self._publish(
                    topic="app_update.operation",
                    event_type="error",
                    operation="app_update.execute",
                    job_id=job_id,
                    request_id=request_id,
                    data={
                        "action": action,
                        "error": error.as_dict(),
                        "snapshot": self._safe_app_update_snapshot(),
                    },
                )

        def work() -> None:
            try:
                work_body()
            except Exception as exc:
                self._report_job_failure(
                    topic="app_update.operation",
                    operation="app_update.execute",
                    job_id=job_id,
                    request_id=request_id,
                    context=context,
                    exc=exc,
                    data={"action": action},
                )
            finally:
                self._release_app_update_job(job_id)

        return self._accept_job(
            topic="app_update.operation",
            operation="app_update.execute",
            job_id=job_id,
            request_id=request_id,
            worker_name=f"studio-app-update-{action}",
            work=work,
            accepted_data={"action": action},
            release_lock=lambda: self._release_app_update_job(job_id),
        )

    def _release_app_update_job(self, job_id: str) -> None:
        with self._app_update_job_lock:
            if self._app_update_job_id == job_id:
                self._app_update_job_id = ""

    def _request_app_update_restart(self, result: Mapping[str, Any]) -> None:
        callback = getattr(self._host, "request_app_restart", None)
        if not callable(callback):
            logger.warning("App update completed without a native restart host")
            return
        try:
            callback(copy.deepcopy(dict(result)))
        except Exception:
            logger.exception("Studio app update restart callback failed")

    def _accept_job(
        self,
        *,
        topic: str,
        operation: str,
        job_id: str,
        request_id: str,
        worker_name: str,
        work: Callable[[], None],
        accepted_data: Mapping[str, Any],
        release_lock: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        """세 작업(runtime·generation API·app update)이 공유하는 접수 꼬리.

        워커를 게이트 뒤에 띄우고 ``accepted`` 를 발행한 뒤에야 풀어 준다 — 이벤트 순서가
        accepted → started → … 로 보장된다. 실패하면 워커를 취소하고 작업 락을 푼다.
        """
        try:
            release_worker = self._launch_worker(worker_name, work)
        except Exception:
            if release_lock is not None:
                release_lock()
            raise
        try:
            accepted = self._publish(
                topic=topic,
                event_type="accepted",
                operation=operation,
                job_id=job_id,
                request_id=request_id,
                data=dict(accepted_data),
            )
        except Exception:
            release_worker(False)
            if release_lock is not None:
                release_lock()
            raise
        release_worker(True)
        reply = self._reply(
            request_id,
            "accepted",
            data={"jobId": job_id, "topic": topic},
            seq=accepted["seq"],
        )
        reply["job"] = {
            "id": job_id,
            "operation": operation,
            "state": "queued",
        }
        return reply

    def _report_job_failure(
        self,
        *,
        topic: str,
        operation: str,
        job_id: str,
        request_id: str,
        context: CallContext,
        exc: BaseException,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """work 본문이 실패를 error 이벤트로 바꾸는 것까지 실패했을 때의 최후 방어.

        본문은 manager 획득부터 모든 예외를 스스로 error 로 보고한다. 여기 오는 것은 그 보고
        (스냅샷 계산·발행·host 통보) 자체가 깨진 경우뿐이므로, 스냅샷 없이 error 한 번만
        시도해 UI busy 가 풀릴 기회를 남기고 로그를 남긴다.
        """
        logger.exception("Studio %s job %s failed while reporting its outcome", operation, job_id)
        try:
            error = self._normalise_error(exc, context).as_dict()
        except Exception:
            error = {"code": "INTERNAL", "message": "작업을 완료하지 못했습니다", "retryable": False}
        try:
            self._publish(
                topic=topic,
                event_type="error",
                operation=operation,
                job_id=job_id,
                request_id=request_id,
                data={**dict(data), "error": error},
            )
        except Exception:
            logger.exception("Studio %s job %s could not publish its failure", operation, job_id)
        return error

    def _launch_worker(
        self, name: str, target: Callable[[], None]
    ) -> Callable[[bool], None]:
        thread: threading.Thread
        start_gate = threading.Event()
        cancelled = threading.Event()

        def wrapped() -> None:
            try:
                start_gate.wait()
                if cancelled.is_set():
                    return
                target()
            finally:
                with self._lock:
                    self._workers.discard(thread)

        thread = threading.Thread(target=wrapped, name=name[:120], daemon=True)
        with self._lock:
            self._workers.add(thread)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._workers.discard(thread)
            raise

        def release(run: bool = True) -> None:
            if not run:
                cancelled.set()
            start_gate.set()

        return release

    def _safe_runtime_snapshot(self) -> dict[str, Any]:
        try:
            return self._normalise_runtime_snapshot(self._get_runtime_manager().snapshot())
        except Exception as exc:
            return {"ok": False, "error": self._normalise_error(exc, None).as_dict()}

    def _safe_generation_snapshot(self, include_secret: bool) -> dict[str, Any]:
        try:
            return self._snapshot_generation_manager(
                self._get_generation_api_manager(), include_secret=include_secret
            )
        except Exception as exc:
            return {"ok": False, "error": self._normalise_error(exc, None).as_dict()}

    def _safe_app_update_snapshot(self) -> dict[str, Any]:
        try:
            return self._snapshot_app_update_manager(self._get_app_update_manager())
        except Exception as exc:
            return {"ok": False, "error": self._normalise_error(exc, None).as_dict()}

    @staticmethod
    def _snapshot_generation_manager(manager: Any, *, include_secret: bool) -> dict[str, Any]:
        try:
            value = manager.snapshot(include_secret=include_secret)
        except TypeError:
            value = manager.snapshot(include_secret)
        return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    @staticmethod
    def _snapshot_app_update_manager(manager: Any) -> dict[str, Any]:
        value = manager.snapshot()
        return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}

    # ------------------------------------------------------------ ordered events

    def _publish(
        self,
        *,
        topic: str,
        event_type: str,
        operation: str,
        data: Mapping[str, Any] | None = None,
        job_id: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        to_drain: list[_Subscriber] = []
        with self._lock:
            self._seq += 1
            event: dict[str, Any] = {
                "version": PROTOCOL_VERSION,
                "eventEpoch": self._event_epoch,
                "seq": self._seq,
                "topic": str(topic),
                "type": str(event_type),
                "operation": str(operation),
                "data": copy.deepcopy(dict(data or {})),
            }
            if job_id:
                event["jobId"] = str(job_id)
            if request_id:
                event["requestId"] = str(request_id)
            self._journal.append(copy.deepcopy(event))
            for subscriber in self._subscribers.values():
                if not subscriber.active:
                    continue
                subscriber.queue.append(copy.deepcopy(event))
                if not subscriber.draining:
                    subscriber.draining = True
                    to_drain.append(subscriber)
        for subscriber in to_drain:
            self._drain_subscriber(subscriber)
        return event

    def _drain_subscriber(self, subscriber: _Subscriber) -> None:
        while True:
            with self._lock:
                if not subscriber.active:
                    subscriber.queue.clear()
                    subscriber.draining = False
                    return
                if not subscriber.queue:
                    subscriber.draining = False
                    return
                event = subscriber.queue.popleft()
            public_event = self._public_event(event, subscriber.context)
            try:
                subscriber.sink(public_event)
            except Exception:
                logger.exception("StudioApplication event subscriber failed")

    # --------------------------------------------------------- public projections

    @classmethod
    def _normalise_runtime_snapshot(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = copy.deepcopy(dict(raw))
        engines = snapshot.get("engines")
        if isinstance(engines, Mapping):
            mapped: dict[str, Any] = {}
            for source_key, source_value in engines.items():
                key = _ENGINE_ALIASES.get(str(source_key).strip().casefold(), "")
                if not key and isinstance(source_value, Mapping):
                    key = _ENGINE_ALIASES.get(
                        str(source_value.get("engine") or "").strip().casefold(), ""
                    )
                if not key:
                    continue
                state = copy.deepcopy(dict(source_value)) if isinstance(source_value, Mapping) else {}
                state["engine"] = key
                mapped[key] = state
            snapshot["engines"] = mapped
        for key in ("activeEngine", "primaryModelEngine"):
            value = str(snapshot.get(key) or "").strip().casefold()
            if value:
                snapshot[key] = _ENGINE_ALIASES.get(value, value)
        return snapshot

    def _public_runtime(self, raw: Mapping[str, Any], context: CallContext) -> dict[str, Any]:
        snapshot = self._normalise_runtime_snapshot(raw)
        native = self._has_capability(context, NATIVE_CAPABILITY)
        snapshot["nativeOperations"] = native
        if native:
            return snapshot
        snapshot.pop("runtimeRoot", None)
        unsafe = {
            "existingRoot", "root", "installRoot", "sourceRoot", "pythonPath",
            "dataRoot", "apiUrl", "extensionDir", "defaultExtensionDir",
            "logPath", "modelPaths",
        }
        engines = snapshot.get("engines", {})
        if isinstance(engines, dict):
            for state in engines.values():
                if not isinstance(state, dict):
                    continue
                state["installRootConfigured"] = bool(
                    state.get("installRoot") or state.get("existingRoot")
                )
                state["extensionDirConfigured"] = bool(state.get("extensionDir"))
                state["apiUrlConfigured"] = bool(state.get("apiUrl"))
                paths = state.get("modelPaths")
                if isinstance(paths, Mapping):
                    state["modelPathCounts"] = {
                        str(key): len(value) if isinstance(value, (list, tuple)) else 0
                        for key, value in paths.items()
                    }
                for key in unsafe:
                    state.pop(key, None)
                state["message"] = self._sanitise_external_text(state.get("message", ""))
                extensions = state.get("extensions")
                if isinstance(extensions, list):
                    state["extensions"] = [
                        self._redact_generic(item) for item in extensions
                    ]
        return self._redact_generic(snapshot)

    def _public_generation_api(
        self, raw: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        snapshot = copy.deepcopy(dict(raw))
        native = self._has_capability(context, NATIVE_CAPABILITY)
        snapshot["nativeOperations"] = native
        if native:
            return snapshot
        config = snapshot.get("config")
        if isinstance(config, dict):
            config.pop("token", None)
            safe_targets = []
            for value in config.get("targets", []):
                if not isinstance(value, Mapping):
                    continue
                target = copy.deepcopy(dict(value))
                target["urlConfigured"] = bool(target.pop("url", ""))
                target["workflowConfigured"] = bool(target.pop("workflowPath", ""))
                target["img2imgWorkflowConfigured"] = bool(
                    target.pop("img2imgWorkflowPath", "")
                )
                safe_targets.append(self._redact_generic(target))
            config["targets"] = safe_targets
        return self._redact_generic(snapshot)

    def _public_app_update(
        self, raw: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        snapshot = copy.deepcopy(dict(raw))
        native = self._has_capability(context, NATIVE_CAPABILITY)
        snapshot["nativeOperations"] = native
        if native:
            return self._redact_generic(snapshot)

        # Web transports only need release presentation state.  Use an
        # allowlist so future native fields (commit ids, helper process ids,
        # checkout paths) cannot become public by accident.
        public_keys = {
            "ok", "repository", "repositoryUrl", "releasesUrl",
            "currentVersion", "developmentBuild", "latestVersion", "tagName",
            "releaseName", "releaseUrl", "publishedAt", "notes",
            "updateAvailable", "notificationAvailable", "skipped",
            "skippedVersion", "autoCheck", "intervalHours", "lastCheckedAt",
            "shouldAutoCheck", "busy", "busyAction", "installReason", "lastResult",
        }
        public = {
            key: copy.deepcopy(value)
            for key, value in snapshot.items()
            if key in public_keys
        }
        version = str(public.get("currentVersion") or "0.0.0")
        public["currentDisplay"] = (
            f"v{version} · 개발 빌드"
            if bool(public.get("developmentBuild"))
            else f"v{version}"
        )
        public["nativeOperations"] = False
        public["canInstall"] = False
        if public.get("updateAvailable"):
            public["installReason"] = "업데이트 설치는 데스크톱 앱에서만 사용할 수 있습니다."
        return self._redact_generic(public)

    def _public_model_paths(
        self, raw: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        state = copy.deepcopy(dict(raw))
        native = self._has_capability(context, NATIVE_CAPABILITY)
        state["ok"] = True
        state["nativeOperations"] = native
        if native:
            return state
        paths = state.pop("paths", {})
        state.pop("defaults", None)
        if isinstance(paths, Mapping):
            state["configured"] = {
                str(key): bool(str(value or "")) for key, value in paths.items()
            }
        return self._redact_generic(state)

    def _public_event(self, event: Mapping[str, Any], context: CallContext) -> dict[str, Any]:
        result = copy.deepcopy(dict(event))
        if self._has_capability(context, NATIVE_CAPABILITY):
            return result
        data = result.get("data")
        if not isinstance(data, Mapping):
            return result
        topic = str(result.get("topic") or "")
        if topic == "runtime.operation":
            result["data"] = self._redact_runtime_event_data(data, context)
        elif topic == "generation_api.operation":
            result["data"] = self._redact_generation_event_data(data, context)
        elif topic == "app_update.operation":
            result["data"] = self._public_app_update_event_data(data, context)
        elif topic == "model_paths.changed":
            mapped = copy.deepcopy(dict(data))
            if isinstance(mapped.get("snapshot"), Mapping):
                mapped["snapshot"] = self._public_model_paths(mapped["snapshot"], context)
            result["data"] = self._redact_generic(mapped)
        else:
            result["data"] = self._redact_generic(data)
        return result

    def _public_app_update_event_data(
        self, data: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        """Project updater events to presentation-only fields for web clients."""

        mapped: dict[str, Any] = {}
        action = str(data.get("action") or "")
        if action:
            mapped["action"] = action
        if isinstance(data.get("snapshot"), Mapping):
            mapped["snapshot"] = self._public_app_update(data["snapshot"], context)
        update = data.get("update")
        if isinstance(update, Mapping):
            mapped["update"] = {
                key: copy.deepcopy(update[key])
                for key in ("stage", "message")
                if key in update
            }
        result = data.get("result")
        if isinstance(result, Mapping):
            safe_result = {
                key: copy.deepcopy(result[key])
                for key in ("ok", "action", "message", "restartRequired", "targetVersion")
                if key in result
            }
            if isinstance(result.get("snapshot"), Mapping):
                safe_result["snapshot"] = self._public_app_update(result["snapshot"], context)
            mapped["result"] = safe_result
        if isinstance(data.get("error"), Mapping):
            mapped["error"] = self._redact_event_result(data["error"])
        return self._redact_generic(mapped)

    def _redact_runtime_event_data(
        self, data: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        mapped = copy.deepcopy(dict(data))
        if isinstance(mapped.get("snapshot"), Mapping):
            mapped["snapshot"] = self._public_runtime(mapped["snapshot"], context)
        if isinstance(mapped.get("result"), Mapping):
            result = copy.deepcopy(dict(mapped["result"]))
            if isinstance(result.get("snapshot"), Mapping):
                result["snapshot"] = self._public_runtime(result["snapshot"], context)
            if isinstance(result.get("state"), Mapping):
                result["state"] = self._public_runtime(result["state"], context)
            mapped["result"] = self._redact_event_result(result)
        if "update" in mapped:
            mapped["update"] = self._redact_event_result(mapped["update"])
        if "error" in mapped:
            mapped["error"] = self._redact_event_result(mapped["error"])
        return self._redact_generic(mapped)

    def _redact_generation_event_data(
        self, data: Mapping[str, Any], context: CallContext
    ) -> dict[str, Any]:
        mapped = copy.deepcopy(dict(data))
        if isinstance(mapped.get("snapshot"), Mapping):
            mapped["snapshot"] = self._public_generation_api(mapped["snapshot"], context)
        if isinstance(mapped.get("result"), Mapping):
            result = copy.deepcopy(dict(mapped["result"]))
            if isinstance(result.get("state"), Mapping):
                result["state"] = self._public_generation_api(result["state"], context)
            mapped["result"] = self._redact_event_result(result)
        if "update" in mapped:
            mapped["update"] = self._redact_event_result(mapped["update"])
        if "error" in mapped:
            mapped["error"] = self._redact_event_result(mapped["error"])
        return self._redact_generic(mapped)

    @classmethod
    def _redact_event_result(cls, value: Any) -> Any:
        """Remove native locations from arbitrary manager result extensions.

        Runtime and generation managers can add nested result metadata over
        time.  Their public snapshots have explicit projections, but unknown
        result fields must remain fail-closed for web subscribers instead of
        relying on every future manager author to remember transport security.
        """

        value = cls._redact_generic(value)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                normalised = str(key).replace("_", "").replace("-", "").casefold()
                exposes_location = (
                    normalised in {
                        "endpoint", "cwd", "pid", "process", "processid",
                        "processhandle", "argv", "args", "command", "commandline",
                        "executable", "environment", "env", "nonce", "host",
                        "hostname", "port", "address", "apikey", "accesskey",
                        "refreshtoken", "accesstoken", "clientsecret", "cookie",
                        "cookies", "credential", "credentials", "session", "sessionid",
                    }
                    or normalised.endswith(
                        (
                            "url", "uri", "path", "paths", "root", "directory", "dir",
                            "host", "port", "address", "args", "apikey", "clientsecret",
                            "token", "secret", "cookie", "cookies", "credential",
                            "credentials", "session", "sessionid",
                            "pid", "processid",
                        )
                    )
                )
                if exposes_location:
                    continue
                result[str(key)] = cls._redact_event_result(item)
            return result
        if isinstance(value, list):
            return [cls._redact_event_result(item) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_event_result(item) for item in value]
        return copy.deepcopy(value)

    @classmethod
    def _redact_generic(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                normalised = str(key).replace("_", "").replace("-", "").casefold()
                if normalised in {
                    "token", "apitoken", "authorization", "secret", "password",
                    "workflowpath", "img2imgworkflowpath", "localpath", "filepath",
                    "pythonpath", "logpath",
                }:
                    continue
                result[str(key)] = cls._redact_generic(item)
            return result
        if isinstance(value, list):
            return [cls._redact_generic(item) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_generic(item) for item in value]
        if isinstance(value, str):
            return cls._sanitise_external_text(value)
        return copy.deepcopy(value)

    @staticmethod
    def _sanitise_external_text(value: Any) -> str:
        text = str(value or "").replace("\x00", "")
        text = _UNC_PATH.sub("[local path]", text)
        text = _WINDOWS_PATH.sub("[local path]", text)
        return text[:100_000]

    # --------------------------------------------------------------- reply/errors

    def _reply(
        self,
        request_id: str,
        status: str,
        *,
        data: Any = None,
        error: Mapping[str, Any] | None = None,
        seq: int | None = None,
    ) -> dict[str, Any]:
        if seq is None:
            with self._lock:
                seq = self._seq
        reply: dict[str, Any] = {
            "version": PROTOCOL_VERSION,
            "requestId": str(request_id or ""),
            "status": status,
            "seq": int(seq),
        }
        if status == "error":
            reply["error"] = copy.deepcopy(dict(error or {}))
        else:
            reply["data"] = copy.deepcopy(data)
        return reply

    def _normalise_error(
        self, exc: Exception, context: CallContext | None
    ) -> StudioApplicationError:
        if isinstance(exc, StudioApplicationError):
            error = exc
        else:
            raw: Mapping[str, Any] | None = None
            as_dict = getattr(exc, "as_dict", None)
            if callable(as_dict):
                try:
                    candidate = as_dict()
                    raw = candidate if isinstance(candidate, Mapping) else None
                except Exception:
                    raw = None
            errors = getattr(exc, "errors", None)
            if isinstance(errors, Mapping):
                error = StudioApplicationError(
                    "INVALID_ARGUMENT", str(exc), details={"fields": dict(errors)}
                )
            elif raw is not None:
                error = StudioApplicationError(
                    str(raw.get("code") or "OPERATION_FAILED"),
                    str(raw.get("message") or exc),
                    retryable=bool(raw.get("retryable", False)),
                    details=raw.get("details") if isinstance(raw.get("details"), Mapping) else None,
                )
            else:
                name = type(exc).__name__
                code = {
                    "GenerationValidationError": "INVALID_ARGUMENT",
                    "GenerationNotFoundError": "NOT_FOUND",
                    "GenerationQueueFullError": "RESOURCE_EXHAUSTED",
                    "GenerationConflictError": "CONFLICT",
                    "FileNotFoundError": "NOT_FOUND",
                    "PermissionError": "FORBIDDEN",
                    "TimeoutError": "DEADLINE_EXCEEDED",
                }.get(name, "INTERNAL")
                retryable = code in {"RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "INTERNAL"}
                error = StudioApplicationError(code, str(exc), retryable=retryable)

        if context is not None and self._has_capability(context, NATIVE_CAPABILITY):
            return error
        return StudioApplicationError(
            error.code,
            self._sanitise_external_text(error.message),
            retryable=error.retryable,
            details=self._redact_event_result(error.details),
        )


__all__ = [
    "CallContext",
    "NATIVE_CAPABILITY",
    "PROTOCOL_VERSION",
    "StudioApplication",
    "StudioApplicationError",
]
