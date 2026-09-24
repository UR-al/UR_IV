"""진행(progress) 이벤트에 무거운 스냅샷을 언제 실을지 정한다 (Qt 없음).

관리형 백엔드 작업은 명령 출력마다(최대 5Hz)·health 루프마다 progress 를 보낸다.
이벤트마다 런타임 스냅샷(두 엔진 전체 스캔, 약 20ms)을 새로 계산하면 부팅 한 번에
수십~수백 회가 되고, 이벤트 journal(deque 1024)에 사본이 쌓인다. 화면은 단계(phase)가
바뀔 때만 스냅샷이 새로 필요하므로, 단계가 바뀌었거나 일정 시간이 지났을 때만 싣는다.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

DEFAULT_INTERVAL_SECONDS = 2.0


class ProgressSnapshotGate:
    """``due(update)`` 가 True 일 때만 스냅샷을 계산해 싣는다."""

    def __init__(
        self,
        *,
        interval: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._interval = float(interval)
        self._clock = clock
        self._lock = threading.Lock()
        self._last_phase: str | None = None
        self._last_at: float | None = None

    @staticmethod
    def phase_of(update: Mapping[str, Any]) -> str:
        return str(update.get("phase") or update.get("stage") or "")

    def due(self, update: Mapping[str, Any]) -> bool:
        phase = self.phase_of(update) if isinstance(update, Mapping) else ""
        now = self._clock()
        with self._lock:
            if (
                self._last_at is None
                or phase != self._last_phase
                or now - self._last_at >= self._interval
            ):
                self._last_phase = phase
                self._last_at = now
                return True
            return False


__all__ = ["DEFAULT_INTERVAL_SECONDS", "ProgressSnapshotGate"]
