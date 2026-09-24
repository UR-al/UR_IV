# core/status_message.py
"""상태줄 메시지 페이로드 — ``show_status`` 가 Vue 하단 계기 스트립으로 보내는 한 줄(Qt 비의존).

Vue 전환 뒤 ``show_status`` 는 모든 속성이 no-op 인 더미 라벨에만 써서, 79곳의 진행·완료·실패
문구가 화면 어디에도 나오지 않았다. 지금은 이 페이로드를 ``statusMessage`` 시그널로 보내고
StatusStrip 이 한 줄로 보여 준다(토스트가 아니다 — 스텝마다 오는 진행 문구도 있어서다).

페이로드: ``{"text": str, "level": "info"|"success"|"warning"|"error", "timeoutMs": int, "at": epoch_ms}``
- text 는 UI 정책대로 절대 경로를 가리고 길이를 자른다(core.error_handler.sanitize_for_ui).
- level 을 주지 않으면 문구에서 추정한다(❌/실패 → error, ⚠ → warning, ✅/완료 → success).
- timeoutMs 가 0 이면 다음 메시지가 올 때까지 남는다(기존 show_status 의 timeout_ms=0 의미).
"""
from __future__ import annotations

import time
from typing import Optional

STATUS_LEVELS = ("info", "success", "warning", "error")
DEFAULT_TIMEOUT_MS = 5000
MAX_TIMEOUT_MS = 60_000
MAX_STATUS_CHARS = 200

_ERROR_MARKERS = ("❌", "실패", "오류", "failed", "error")
_WARNING_MARKERS = ("⚠", "경고", "warning")
_SUCCESS_MARKERS = ("✅", "완료", "success")


def infer_status_level(text: str) -> str:
    """문구로 수준을 추정한다 — 실패가 완료보다 우선한다('완료 … 실패 3건' 은 error)."""
    lowered = str(text or "").casefold()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return "error"
    if any(marker in lowered for marker in _WARNING_MARKERS):
        return "warning"
    if any(marker in lowered for marker in _SUCCESS_MARKERS):
        return "success"
    return "info"


def _timeout(value) -> int:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_MS
    if ms <= 0:
        return 0
    return min(ms, MAX_TIMEOUT_MS)


def build_status_payload(message, timeout_ms=DEFAULT_TIMEOUT_MS, level: Optional[str] = None,
                         *, now_ms: Optional[int] = None) -> Optional[dict]:
    """상태줄 페이로드. 보낼 내용이 없으면(빈 문자열·공백) None.

    ``at``(epoch ms)은 늦게 붙은 화면(페이지 로드 전 문구·웹 재접속)이 getStatusMessage 로 마지막
    문구를 읽을 때 남은 표시 시간을 계산하는 데 쓴다.
    """
    from core.error_handler import sanitize_for_ui

    raw = " ".join(str(message or "").split())   # 줄바꿈·연속 공백 → 한 줄
    if not raw:
        return None
    text = sanitize_for_ui(raw, MAX_STATUS_CHARS)
    resolved = level if level in STATUS_LEVELS else infer_status_level(raw)
    at = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return {"text": text, "level": resolved, "timeoutMs": _timeout(timeout_ms), "at": at}


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "MAX_STATUS_CHARS",
    "STATUS_LEVELS",
    "build_status_payload",
    "infer_status_level",
]
