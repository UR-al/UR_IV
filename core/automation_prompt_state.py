# core/automation_prompt_state.py
"""자동화 조종석 '다음 프롬프트' 편집이 뜻이 있는가 — 순수 규칙(Qt 무관).

조종석은 automationStatus.prompt(총 프롬프트 상자, 걸린 일회성 덮어쓰기가 있으면 그 값)를
'다음 프롬프트'로 보여 주고 편집을 받는다. 그런데 한 장이 생성 중이거나 생성이 끝나 큐 항목
앞에서 멈춘('continue') 동안 그 상자는 **방금 보낸** 프롬프트다. 반복(repeat)이 남지 않았으면
다음 덱 장은 새로 뽑으므로, 그때 건 편집은 뽑는 순간 버려진다(_discard_stale_prompt_override) —
사용자는 고쳤다고 믿는데 본 적 없는 프롬프트가 나간다. 이 판정이 False 면 패널은 편집을 막고
'생성 중인 프롬프트'로 표시한다(frontend/src/components/AutomationPanel.vue · prompt_is_next).

일시정지는 이 상태에 오래 머물지 않는다 — 덱 장 사이에서 멈추면 _run_automation_cycle 이 다음
프롬프트를 뽑아 보여 준 뒤 서므로(판정 True), 멈춘 채 다음 장을 손볼 수 있다.
"""
from __future__ import annotations


def prompt_is_next(
    *,
    generating: bool,
    held_after_generation: bool,
    processing_queue: bool,
    queue_dirtied: bool,
    override_pending: bool,
    override_used: bool,
    current_repeat: int,
    repeat_per_prompt: int,
) -> bool:
    """패널이 보여 주는 프롬프트가 '다음 덱 장'에 그대로 쓰이는가.

    - generating: 한 장이 생성 중(워커가 아직 결과를 내지 않음)
    - held_after_generation: 생성이 끝난 뒤 큐 항목 앞에서 일시정지('continue')
    - processing_queue / queue_dirtied: 큐 항목이 생성 중 / 큐 항목이 UI 프롬프트를 바꿔 놓음
    - override_pending: 아직 안 쓰인 일회성 덮어쓰기가 걸려 있다(패널은 그 값을 보여 준다)
    - override_used: 지금(또는 방금) 장이 덮어쓰기를 썼다 — 상자는 다음 사이클에 원래대로 돌아간다
    - current_repeat / repeat_per_prompt: 이 프롬프트로 몇 번째 장인가 / 프롬프트당 장 수
    """
    if not (generating or held_after_generation):
        # 시작 전 · 사이 간격 대기 · 뽑은 뒤 일시정지 — 상자가 곧 다음 장이다
        return True
    try:
        repeat_remains = int(current_repeat) < max(1, int(repeat_per_prompt))
    except (TypeError, ValueError):
        repeat_remains = False
    if not repeat_remains:
        return False        # 다음 덱 장은 새로 뽑는다 — 보이는 건 방금/지금 생성한 프롬프트
    if override_pending:
        return True         # 반복 중 걸어 둔 덮어쓰기 — 다음 반복 장이 그 값으로 나간다
    # 큐 항목이 보이는 중이거나, 이번 장이 쓴 덮어쓰기가 보이는 중이면 다음 반복 장(원래 프롬프트)과 다르다
    return not (processing_queue or queue_dirtied or override_used)


__all__ = ['prompt_is_next']
