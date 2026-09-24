# core/coalesced_call.py
"""여러 번 요청돼도 예약된 한 번만 실행하는 호출 — 순수 로직 (Qt 없음).

한 이벤트 핸들러 안에서 같은 일(전체 목록 Vue 전송 · JSON 저장)을 N번 부르면 N번 하던 것을
'다음 이벤트 루프 턴에 한 번'(delay_ms=0) 또는 '잠시 모았다가 한 번'(delay_ms>0)으로 합친다.
예약은 주입한 ``schedule(ms, fn)`` 으로 한다 — Qt 에선 ``QTimer.singleShot``, 테스트에선 가짜.

- :meth:`request` — 아직 예약이 없으면 한 번 예약한다. 이미 있으면 아무것도 안 한다(합쳐짐).
- :meth:`flush` — 예약된 실행을 지금 당장 한다(종료 직전 저장처럼 미룰 수 없을 때).
  이미 걸어 둔 예약 콜백은 무효가 되어 나중에 와도 두 번 실행하지 않는다.
- :meth:`cancel` — 예약만 버린다.

실행 중 예외가 나도 다음 요청은 다시 예약된다(대기 표시를 실행 전에 내린다).
"""
from __future__ import annotations

from typing import Callable

Scheduler = Callable[[int, Callable[[], None]], None]


class CoalescedCall:
    def __init__(self, fn: Callable[[], None], *, schedule: Scheduler, delay_ms: int = 0) -> None:
        self._fn = fn
        self._schedule = schedule
        self._delay_ms = max(0, int(delay_ms))
        self._pending = False
        self._token = 0

    @property
    def pending(self) -> bool:
        return self._pending

    def request(self) -> bool:
        """실행을 예약한다. 새로 예약했으면 True, 이미 예약돼 있어 합쳐졌으면 False."""
        if self._pending:
            return False
        self._pending = True
        self._token += 1
        token = self._token
        self._schedule(self._delay_ms, lambda: self._fire(token))
        return True

    def _fire(self, token: int) -> None:
        if not self._pending or token != self._token:
            return   # flush/cancel 로 이미 처리됐거나 더 새 예약이 있다
        self._pending = False
        self._fn()

    def flush(self) -> bool:
        """예약된 실행이 있으면 지금 실행한다. 실행했으면 True."""
        if not self._pending:
            return False
        self._pending = False
        self._token += 1   # 걸려 있는 예약 콜백을 무효화
        self._fn()
        return True

    def cancel(self) -> None:
        self._pending = False
        self._token += 1


def qt_scheduler() -> Scheduler:
    """메인 스레드 이벤트 루프 예약자 (``QTimer.singleShot``)."""
    from PyQt6.QtCore import QTimer

    return QTimer.singleShot


__all__ = ["CoalescedCall", "Scheduler", "qt_scheduler"]
