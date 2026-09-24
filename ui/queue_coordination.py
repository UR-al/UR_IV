# ui/queue_coordination.py
"""대기열 매니저 ↔ 자동화 '큐 우선' 조정 — 같은 대기열을 두 소비자가 다투지 않게 한다.

대기열은 두 곳이 소비한다.
- 대기열 매니저(widgets/queue_manager.py) — 대기열 패널의 '시작'
- 자동화 '큐 우선'(ui/generator_actions.py _continue_automation) — 자동화가 돌면 덱보다 대기열을 먼저 낸다

생성 워커는 하나라 둘이 같이 돌면 같은 항목을 두 번 보내고, 실패한 쪽이 남의 '생성 중' 표시를 풀었다.
그 뒤 그 항목을 지우면 자동화가 맨 앞 항목(생성하지 않은 다음 항목)을 지웠다. 그래서
- 자동화가 도는 동안 대기열 '시작'은 거절한다(자동화가 대기열을 먼저 처리하므로 필요도 없다),
- 대기열이 도는 동안(일시정지 포함) 자동화 시작은 거절한다.
'생성 중' 표시 자체도 주인이 있다(core.queue_model.QUEUE_OWNER · AUTOMATION_OWNER).
"""
from __future__ import annotations

from typing import Any

#: 자동화가 도는 동안 대기열 '시작'을 누르면
QUEUE_BLOCKED_BY_AUTOMATION = '자동화가 대기열 항목을 먼저 처리하고 있어 대기열을 따로 시작하지 않습니다'
#: 대기열이 도는 동안 자동화를 시작하면
AUTOMATION_BLOCKED_BY_QUEUE = '대기열이 실행 중입니다 — 대기열을 중지하거나 끝난 뒤 자동화를 시작하세요'
#: 생성 중인 항목을 지우려 했을 때(선택 삭제 · 전체 비우기)
RUNNING_ITEM_KEPT = '생성 중인 항목은 지울 수 없어 남겨 두었습니다 — 생성이 끝난 뒤 지우세요'


def queue_start_refusal(host: Any) -> str:
    """대기열 매니저 ``start_blocker`` — 자동화가 돌면 거절 사유, 아니면 빈 문자열."""
    return QUEUE_BLOCKED_BY_AUTOMATION if getattr(host, 'is_automating', False) else ''


def automation_start_refusal(host: Any) -> str:
    """자동화 시작 전 확인 — 대기열 매니저가 돌면(일시정지 포함) 거절 사유, 아니면 빈 문자열."""
    manager = getattr(host, 'queue_manager', None)
    return AUTOMATION_BLOCKED_BY_QUEUE if getattr(manager, 'is_running', False) else ''


def running_item_kept(panel: Any, requested_ids: Any = None) -> bool:
    """삭제·비우기 뒤에도 생성 중인 항목이 대기열에 남았는가.

    ``requested_ids`` 를 주면 그 항목이 요청에 들어 있었을 때만 True(선택 삭제),
    없으면 남아 있기만 하면 True(전체 비우기).
    """
    running_id = getattr(panel, 'current_processing_id', None)
    if running_id is None:
        return False
    if requested_ids is not None:
        try:
            if running_id not in requested_ids:
                return False
        except TypeError:
            return False
    getter = getattr(panel, 'get_item_by_id', None)
    return bool(callable(getter) and getter(running_id) is not None)


def announce(host: Any, message: str, level: str = 'warning') -> None:
    """상태줄 + 토스트로 알린다(웹 모드에도 닿는다). 둘 다 없으면 조용히 넘어간다."""
    if not message:
        return
    show = getattr(host, 'show_status', None)
    if callable(show):
        try:
            show(message)
        except Exception:
            pass
    from ui.status_line import notify_user
    notify_user(host, level, message)


__all__ = [
    'AUTOMATION_BLOCKED_BY_QUEUE',
    'QUEUE_BLOCKED_BY_AUTOMATION',
    'RUNNING_ITEM_KEPT',
    'announce',
    'automation_start_refusal',
    'queue_start_refusal',
    'running_item_kept',
]
