"""
RFP Engine — NAIA 1.5의 RFP(Random Function Process)에서 가져온 프롬프트 정리 알고리즘.

지금 남은 것은 생성 때마다 도는 중복 정리 하나다:
``make_dedupe_hook()`` → ``core.standard_hooks.register_standard_hooks`` 가 PromptPipeline 의
FINAL 단계에 등록한다(메인 태그 순서 보존 중복 제거 + prefix/postfix 와 겹치는 태그 제거).

모든 함수는 **순수 함수** (전역 상태 없음, 입력 → 출력), PyQt/Vue 의존성 없음.

(NAIA 의 조건부 명령 DSL — parse_conditional_command·evaluate_condition·execute_command — 과
filter_contains·split_tags·join_tags 는 호출자 없이 남아 있어 지웠다. 조건부 프롬프트는
utils.condition_block/cond_rules 가 담당한다.)
"""
from __future__ import annotations

from typing import Callable


# ─────────────────────────────────────────────────────────────────
# 중복 제거
# ─────────────────────────────────────────────────────────────────

def dedupe(tags: list[str]) -> list[str]:
    """순서 보존 중복 제거."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def remove_collisions(main_tags: list[str],
                      prefix_tags: list[str],
                      postfix_tags: list[str]) -> list[str]:
    """main_tags에서 prefix/postfix와 충돌하는 태그 제거.

    NAIA 원본의 ``general = [item for item in general if item not in prompt_duplicate_check]``
    동작. 비교는 ``{}[]`` 가중치 마커 제거 후 수행.
    """
    strip_marks = str.maketrans("", "", "{}[]")
    blocked = set()
    for t in list(prefix_tags) + list(postfix_tags):
        blocked.add(t.translate(strip_marks).strip())
    return [t for t in main_tags if t not in blocked]


# ─────────────────────────────────────────────────────────────────
# PromptPipeline 훅 팩토리
# ─────────────────────────────────────────────────────────────────

def make_dedupe_hook() -> Callable:
    """main_tags 중복 제거 + prefix/postfix 충돌 제거 훅.

    core.standard_hooks 가 FINAL 단계(priority 90)에 ``standard_dedupe`` 로 등록한다 —
    와일드카드·``$$name$$`` 확장이 끝난 뒤의 태그를 정리해야 하기 때문.
    """
    def hook(ctx) -> None:
        ctx.main_tags = remove_collisions(
            dedupe(ctx.main_tags), ctx.prefix_tags, ctx.postfix_tags
        )
    return hook
