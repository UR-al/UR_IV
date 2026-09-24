# core/automation_retry.py
"""자동화 생성 실패 재시도 판정 — 순수 로직(Qt 무관).

자동화 중 한 장이 실패하면 설정의 max_retries 까지 지수 백오프(2 → 4 → 8초, 최대 30초)로
같은 장을 다시 시도한다. 재시도는 반복 카운터·자연어 변환을 다시 돌리지 않는 생성 진입점
(ActionsMixin._automation_start_generation)으로 들어가야 한다 — 예전엔 _automation_generate 로
들어가 repeat_per_prompt ≥ 2 일 때 실패 한 번마다 그 프롬프트가 한 장씩 모자랐다.
"""
from __future__ import annotations

from typing import Optional

MAX_BACKOFF_SECONDS = 30.0


def retry_backoff_seconds(attempt: int) -> float:
    """attempt 번째(1부터) 재시도 전 대기 — 2**attempt 초, 최대 30초."""
    return min(2.0 ** max(1, int(attempt)), MAX_BACKOFF_SECONDS)


def plan_retry(retry_count: int, max_retries) -> Optional[tuple[int, float]]:
    """실패 직후 판정.

    :param retry_count: 이 장에서 이미 한 재시도 횟수
    :param max_retries: 설정값(숫자가 아니거나 음수면 0)
    :return: 재시도하면 ``(새 재시도 횟수, 대기 초)``, 소진했으면 None(호출자는 횟수를 0 으로)
    """
    try:
        limit = max(0, int(max_retries or 0))
    except (TypeError, ValueError):
        limit = 0
    done = max(0, int(retry_count or 0))
    if done >= limit:
        return None
    attempt = done + 1
    return attempt, retry_backoff_seconds(attempt)


__all__ = ['MAX_BACKOFF_SECONDS', 'plan_retry', 'retry_backoff_seconds']
