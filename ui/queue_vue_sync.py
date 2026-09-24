# ui/queue_vue_sync.py
"""대기열 → Vue 동기화 — 한 이벤트 루프 턴에 한 번으로 합친다.

예전 ``_sync_queue_to_vue`` 는 부를 때마다 대기열 전체(프롬프트 최대 4000자)를 직렬화해
``queueUpdated`` 로 보냈고, 항목 하나가 추가될 때마다 ``queueItemAdded`` 도 따로 보냈다.
XYZ(최대 256잡)·이벤트 시나리오(최대 10,000)처럼 한 핸들러에서 N개를 넣으면 전체 목록이 N번
직렬화·전송됐다(N² 크기). 지금은 요청만 표시해 두고 다음 턴에 **최신 상태 한 번**을 보낸다.
payload 형태(``core.queue_model.vue_queue_state``)는 그대로다.

- ``request_state()`` — queueUpdated 예약(같은 턴의 요청은 합쳐짐)
- ``item_added(item)`` — queueItemAdded 예약(Vue 는 핀 강조에만 쓴다 — 마지막 항목 하나만 보낸다)
- 둘 다 QWebChannel 에 등록된 브리지 시그널이라 메인 스레드 타이머에서만 emit 한다.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from core.coalesced_call import CoalescedCall, Scheduler, qt_scheduler
from core.queue_model import queue_item_added_payload, vue_queue_state


class QueueVueSync:
    def __init__(self, host: Any, *, schedule: Optional[Scheduler] = None) -> None:
        self._host = host
        schedule = schedule or qt_scheduler()
        self._state_call = CoalescedCall(self._emit_state, schedule=schedule, delay_ms=0)
        self._added_call = CoalescedCall(self._emit_added, schedule=schedule, delay_ms=0)
        self._last_added: Optional[dict] = None

    def request_state(self) -> None:
        self._state_call.request()

    def item_added(self, item: Any) -> None:
        if isinstance(item, dict):
            self._last_added = item
            self._added_call.request()

    def flush(self) -> None:
        """예약된 전송을 지금 한다(테스트·종료 직전용)."""
        self._added_call.flush()
        self._state_call.flush()

    def build_state(self) -> Optional[dict]:
        panel = getattr(self._host, 'queue_panel', None)
        if panel is None:
            return None
        manager = getattr(self._host, 'queue_manager', None)
        index_of_processing = getattr(panel, 'processing_index', None)
        processing = index_of_processing() if callable(index_of_processing) else -1
        return vue_queue_state(
            list(getattr(panel, 'queue_items', []) or []),
            running=bool(manager is not None and getattr(manager, 'is_running', False)),
            paused=bool(manager is not None and getattr(manager, 'is_paused', False)),
            # 이번 실행의 '라이브' 완료 수 (이전 실행 종료값이 아니라) — 진행률 정확화
            completed=int(getattr(manager, 'generated_count', 0) or 0) if manager is not None else 0,
            processing_index=processing,
            # 자동화가 돌면 대기열 '시작'은 거절된다(ui/queue_coordination) — Vue 가 버튼을 끈다
            automating=bool(getattr(self._host, 'is_automating', False)),
        )

    def _bridge(self):
        return getattr(self._host, 'vue_bridge', None)

    def _emit_state(self) -> None:
        bridge = self._bridge()
        if bridge is None:
            return
        try:
            state = self.build_state()
            if state is not None:
                bridge.queueUpdated.emit(json.dumps(state))
        except Exception as exc:
            show = getattr(self._host, 'show_status', None)
            if callable(show):
                try:
                    show(f"queue sync error: {exc}")
                except Exception:
                    pass

    def _emit_added(self) -> None:
        item, self._last_added = self._last_added, None
        bridge = self._bridge()
        if bridge is None or item is None:
            return
        try:
            bridge.queueItemAdded.emit(json.dumps(queue_item_added_payload(item)))
        except Exception:
            pass


__all__ = ['QueueVueSync']
