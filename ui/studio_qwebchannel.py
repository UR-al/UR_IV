"""QWebChannel adapter for the transport-neutral Studio application Interface.

The adapter deliberately exposes one request slot and one event signal.  It
does not know any domain operation names: validation, authorization, job
lifecycle, and dispatch all remain in :mod:`core.studio_application`.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Mapping
from typing import Any

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from ui.native_dialogs import select_directory

from core.studio_application import (
    CallContext,
    StudioApplication,
    StudioApplicationError,
)


logger = logging.getLogger(__name__)


def _json_text(value: Any) -> str:
    """Encode only contract-safe JSON; never silently stringify bad values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _error_reply(
    code: str,
    message: str,
    *,
    request_id: str = "",
    event_epoch: str = "",
) -> dict[str, Any]:
    reply = {
        "version": 1,
        "requestId": request_id,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "retryable": False,
        },
        "seq": 0,
    }
    if event_epoch:
        reply["eventEpoch"] = event_epoch
    return reply


class DesktopNativeHost(QObject):
    """Desktop-only host capabilities used by StudioApplication operations."""

    _refreshRequested = pyqtSignal()
    _runtimeEventRequested = pyqtSignal(str)
    _appRestartRequested = pyqtSignal(str)

    def __init__(self, window: QObject, vue_bridge: QObject):
        super().__init__(window)
        self._window = window
        self._vue_bridge = vue_bridge
        self._refreshRequested.connect(self._refresh_model_widgets_on_qt_thread)
        self._runtimeEventRequested.connect(self._forward_runtime_event_on_qt_thread)
        self._appRestartRequested.connect(self._restart_app_on_qt_thread)

    @staticmethod
    def _dialog_title(kind: str, selector: str) -> str:
        engine_names = {
            "forge": "Forge Neo",
            "forge_neo": "Forge Neo",
            "comfy": "ComfyUI",
            "comfyui": "ComfyUI",
        }
        model_names = {
            "checkpoint_dir": "Checkpoint / Model",
            "lora_dir": "LoRA",
            "vae_dir": "VAE",
            "text_encoder_dir": "Text Encoder",
        }
        if kind == "runtime_install":
            return f"{engine_names.get(selector, selector or 'Backend')} 설치 폴더 선택"
        if kind == "runtime_extension":
            return f"{engine_names.get(selector, selector or 'Backend')} 확장 폴더 선택"
        if kind == "model_path":
            return f"Forge Neo {model_names.get(selector, selector or '모델')} 폴더 선택"
        return "폴더 선택"

    def pick_directory(self, kind: str, selector: str, current: str) -> str | None:
        """Open the native directory picker for an authorized desktop request."""

        selected = select_directory(
            self._window,
            self._dialog_title(str(kind or ""), str(selector or "")),
            str(current or ""),
        )
        return str(selected) if selected else None

    def refresh_model_widgets(self) -> None:
        """Refresh legacy proxy choices after model-path state changes."""

        self._refreshRequested.emit()

    def handle_runtime_event(self, payload: Mapping[str, Any]) -> None:
        """Relay Studio runtime events to the legacy Vue signal safely."""

        try:
            self._runtimeEventRequested.emit(_json_text(dict(payload)))
        except (RuntimeError, TypeError, ValueError):
            logger.warning("Studio runtime event could not reach the Qt bridge", exc_info=True)

    def request_app_restart(self, payload: Mapping[str, Any]) -> None:
        """Schedule the established clean shutdown path after UI event delivery."""

        try:
            self._appRestartRequested.emit(_json_text(dict(payload)))
        except (RuntimeError, TypeError, ValueError):
            logger.warning("App update restart request could not reach Qt", exc_info=True)

    @pyqtSlot()
    def _refresh_model_widgets_on_qt_thread(self) -> None:
        refresh = getattr(self._vue_bridge, "_refresh_forge_module_widgets", None)
        if callable(refresh):
            refresh()

    @pyqtSlot(str)
    def _forward_runtime_event_on_qt_thread(self, payload_json: str) -> None:
        signal = getattr(self._vue_bridge, "backendRuntimeEvent", None)
        emit = getattr(signal, "emit", None)
        if not callable(emit):
            return
        try:
            emit(payload_json)
        except RuntimeError:
            # The bridge may already be gone while a daemon worker is finishing.
            logger.debug("Vue runtime event signal is already disposed", exc_info=True)

    @pyqtSlot(str)
    def _restart_app_on_qt_thread(self, _payload_json: str) -> None:
        quit_app = getattr(self._window, "_quit_app", None)
        if not callable(quit_app):
            logger.error("App update restart requested without _quit_app host")
            return
        # Let the completed event and its final toast render before shutdown.
        QTimer.singleShot(1200, quit_app)


class StudioQWebChannelAdapter(QObject):
    """JSON-only QWebChannel Adapter for a fixed authenticated context."""

    event = pyqtSignal(str)

    def __init__(
        self,
        application: StudioApplication,
        context: CallContext,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._application = application
        self._context = context
        self._unsubscribe_lock = threading.RLock()
        self._resume_lock = threading.Lock()
        self._unsubscribe = None
        self._subscription_generation = 0
        self._closed = False
        self.destroyed.connect(self._dispose_subscription)

    @property
    def context(self) -> CallContext:
        return self._context

    @pyqtSlot(result=str)
    def describe(self) -> str:
        try:
            return _json_text(self._application.describe(self._context))
        except Exception:
            logger.exception("Studio Interface describe failed")
            return _json_text(
                _error_reply(
                    "INTERNAL",
                    "Studio 기능 정보를 불러오지 못했습니다.",
                    event_epoch=str(getattr(self._application, "event_epoch", "") or ""),
                )
            )

    @pyqtSlot(str, result=str)
    def invoke(self, request_json: str) -> str:
        request_id = ""
        try:
            request = json.loads(
                request_json,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant: {value}")
                ),
            )
            if not isinstance(request, Mapping):
                raise TypeError("request root must be an object")
            request = dict(request)
            request_id = str(request.get("requestId") or "")
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            return _json_text(
                _error_reply(
                    "INVALID_JSON",
                    "요청이 올바른 JSON 객체가 아닙니다.",
                    request_id=request_id,
                )
            )

        try:
            return _json_text(self._application.invoke(self._context, request))
        except Exception:
            logger.exception("Studio Interface invoke failed: %s", request_id)
            return _json_text(
                _error_reply(
                    "INTERNAL",
                    "Studio 요청을 처리하지 못했습니다.",
                    request_id=request_id,
                )
            )

    @pyqtSlot(int, result=str)
    def resume(self, after_seq: int) -> str:
        """Replace the event subscription and replay events after ``after_seq``."""

        event_epoch = self._application.event_epoch
        if isinstance(after_seq, bool) or not isinstance(after_seq, int) or after_seq < 0:
            return _json_text(
                _error_reply(
                    "INVALID_CURSOR",
                    "event cursor는 0 이상의 정수여야 합니다.",
                    event_epoch=event_epoch,
                )
            )
        try:
            self._replace_subscription(after_seq)
        except StudioApplicationError as exc:
            current_seq = exc.details.get("currentSeq", 0)
            return _json_text({
                "version": 1,
                "requestId": "",
                "status": "error",
                "error": exc.as_dict(),
                "seq": int(current_seq) if isinstance(current_seq, int) else 0,
                "eventEpoch": event_epoch,
            })
        except RuntimeError:
            return _json_text(
                _error_reply(
                    "UNAVAILABLE",
                    "Studio event transport가 종료되었습니다.",
                    event_epoch=event_epoch,
                )
            )
        except Exception:
            logger.exception("Studio event resume failed: %s", after_seq)
            return _json_text(
                _error_reply(
                    "INTERNAL",
                    "Studio event 구독을 복구하지 못했습니다.",
                    event_epoch=event_epoch,
                )
            )
        return _json_text({
            "version": 1,
            "status": "ok",
            "afterSeq": after_seq,
            "eventEpoch": event_epoch,
        })

    def close(self) -> None:
        self._dispose_subscription()

    def _dispose_subscription(self, *_args: Any) -> None:
        with self._unsubscribe_lock:
            if self._closed:
                return
            self._closed = True
            self._subscription_generation += 1
            unsubscribe = self._unsubscribe
            self._unsubscribe = None
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception:
                logger.debug("Studio event unsubscribe failed", exc_info=True)

    def _replace_subscription(self, after_seq: int) -> None:
        """Swap subscriptions without a journal gap and invalidate stale sinks."""

        with self._resume_lock:
            with self._unsubscribe_lock:
                if self._closed:
                    raise RuntimeError("adapter is closed")
                self._subscription_generation += 1
                generation = self._subscription_generation
                previous = self._unsubscribe
                self._unsubscribe = None
            if previous is not None:
                try:
                    previous()
                except Exception:
                    logger.debug("Studio event unsubscribe failed", exc_info=True)

            unsubscribe = self._application.subscribe(
                self._context,
                lambda payload, current=generation: self._forward_event(current, payload),
                after_seq=after_seq,
            )
            with self._unsubscribe_lock:
                discard = self._closed or generation != self._subscription_generation
                if not discard:
                    self._unsubscribe = unsubscribe
            if discard:
                try:
                    unsubscribe()
                except Exception:
                    logger.debug("Studio event unsubscribe failed", exc_info=True)
                raise RuntimeError("adapter closed during event resume")

    def _forward_event(self, generation: int, payload: dict[str, Any]) -> None:
        with self._unsubscribe_lock:
            if self._closed or generation != self._subscription_generation:
                return
        try:
            self.event.emit(_json_text(payload))
        except (RuntimeError, TypeError, ValueError):
            # RuntimeError is expected when a worker finishes during QObject
            # teardown.  Contract-invalid events are logged and never coerced.
            logger.warning("Studio event could not cross QWebChannel", exc_info=True)


__all__ = ["DesktopNativeHost", "StudioQWebChannelAdapter"]
