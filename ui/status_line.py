# ui/status_line.py
"""``show_status`` 의 실제 표시 경로 — Vue 하단 계기 스트립 한 줄 + 필요할 때만 토스트.

- :func:`publish_status` — show_status 문구를 ``vue_bridge.statusMessage`` 로 보낸다(core/status_message.py
  페이로드). 스텝마다 오는 진행 문구도 있으므로 토스트가 아니다.
- :func:`notify_user` — 사용자가 누른 동작의 결과(실패·빈 결과·완료)를 놓치면 안 되는 곳에서
  ``showNotification`` 토스트를 명시적으로 띄운다. 문구는 UI 정책대로 절대 경로를 가린다.

로그: 경고·오류 수준 문구만 원문(경로 포함)으로 ``status`` 로거에 WARNING 으로 남긴다. 진행·완료
문구(샘플링 스텝마다 한 줄)까지 남기면 콘솔과 app.log 가 다시 소음으로 덮인다. 로거를 쓰는 이유는
예외를 내지 않아서다 — show_status 는 except 블록·생성 경로에서도 불리는데, 이모지가 든 print() 는
cp949 로 리다이렉트된 stdout 에서 UnicodeEncodeError 로 호출자를 깰 수 있다.

스레드: QWebChannel 에 등록된 vue_bridge 의 시그널은 GUI 스레드에서만 emit 한다. 워커 스레드에서
불리면 bridge 스레드에 사는 중계 객체의 시그널로 넘겨 Qt 가 GUI 스레드로 옮겨 전달하게 한다.
"""
from __future__ import annotations

import json
import logging
import threading

from core.status_message import DEFAULT_TIMEOUT_MS, build_status_payload

_RELAY_ATTR = "_status_line_relay"
_RELAY_LOCK = threading.Lock()
_LOGGED_LEVELS = frozenset({"warning", "error"})
logger = logging.getLogger("status")


def _relay_for(bridge):
    """bridge 스레드에 사는 중계 객체(한 번 만들어 bridge 에 붙여 둔다)."""
    relay = getattr(bridge, _RELAY_ATTR, None)
    if relay is not None:
        return relay
    from PyQt6.QtCore import QObject, pyqtSignal

    class _StatusRelay(QObject):
        deliver = pyqtSignal(str)

    with _RELAY_LOCK:
        relay = getattr(bridge, _RELAY_ATTR, None)
        if relay is None:
            relay = _StatusRelay()
            target_thread = bridge.thread()
            if relay.thread() is not target_thread:
                relay.moveToThread(target_thread)   # 만든 스레드에서만 옮길 수 있다 — 지금 여기
            # 시그널→시그널 연결: 받는 쪽(bridge)이 GUI 스레드라 워커에서 emit 하면 큐로 넘어간다.
            relay.deliver.connect(bridge.statusMessage)
            setattr(bridge, _RELAY_ATTR, relay)
    return relay


def _on_bridge_thread(bridge) -> bool:
    try:
        from PyQt6.QtCore import QThread

        return QThread.currentThread() is bridge.thread()
    except Exception:
        return True


def publish_status(host, message, timeout_ms=DEFAULT_TIMEOUT_MS, *, level=None) -> None:
    """상태줄에 한 줄을 보인다. 경고·오류는 로그에도 원문으로 남긴다. 절대 예외를 내지 않는다."""
    payload = build_status_payload(message, timeout_ms, level)
    if payload is None:
        return
    if payload["level"] in _LOGGED_LEVELS:
        try:
            logger.warning("[Status] %s", " ".join(str(message).split()))
        except Exception:
            pass
    bridge = getattr(host, "vue_bridge", None)
    signal = getattr(bridge, "statusMessage", None)
    if signal is None:
        return
    # 마지막 한 줄을 남겨 둔다 — Vue 가 뜨기 전(설정 불러오기 실패 등)이나 웹 클라이언트가 붙기
    # 전에 보낸 문구도 StatusStrip 이 마운트될 때 getStatusMessage 로 읽어 보인다.
    try:
        bridge._last_status_payload = payload
    except Exception:
        pass
    encoded = json.dumps(payload, ensure_ascii=False)
    try:
        if _on_bridge_thread(bridge):
            signal.emit(encoded)
        else:
            _relay_for(bridge).deliver.emit(encoded)
    except Exception as exc:
        # 종료 중 bridge 가 먼저 사라졌다(RuntimeError) 등 — 상태 한 줄 때문에 호출자를 깨지 않는다
        logger.debug("status line not delivered: %s", exc)


def notify_user(host, level: str, message: str) -> None:
    """사용자 동작의 결과를 토스트로 알린다(경로 가림). vue_bridge 가 없으면 아무것도 안 한다."""
    from core.error_handler import sanitize_for_ui

    bridge = getattr(host, "vue_bridge", None)
    signal = getattr(bridge, "showNotification", None)
    if signal is None:
        return
    text = sanitize_for_ui(str(message or ""), 200)
    if not text:
        return
    try:
        signal.emit(level if level in ("info", "success", "warning", "error") else "info", text)
    except RuntimeError:
        pass


__all__ = ["notify_user", "publish_status"]
