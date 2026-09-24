"""
Standard Hooks — 시작 시 PromptPipeline에 표준 훅을 일괄 등록.

기존 generator_prompts.py / generator_generation.py를 비파괴적으로 보강.
파이프라인은 기존 처리 *뒤*에 호출되므로 회귀 위험 없음.

등록 훅 (priority 낮을수록 먼저):
- POST_PROCESSING (80)  : ``$$name$$`` 인스턴트 와일드카드 확장
                          (core.instant_wildcards.ensure_hook_registered — 부팅 때 즉시 등록)
- FINAL          (90)   : 메인 태그 중복 제거 + prefix/postfix 충돌 제거
                          (core.rfp_engine.make_dedupe_hook)

조건부 프롬프트는 여기가 아니라 utils.condition_block(cond_rules)이 맡는다 — rfp_engine 에 있던
두 번째 조건 DSL 은 호출자 없이 남아 있어 지웠다.
"""
from __future__ import annotations

from utils.app_logger import get_logger

_logger = get_logger("std_hooks")

DEDUPE_HOOK_NAME = "standard_dedupe"


def register_standard_hooks(pipeline=None, instant_wildcards=None) -> int:
    """표준 훅 일괄 등록. 멱등 — 훅 이름으로 확인해 여러 번 호출돼도 한 번만 등록.

    :param pipeline: 등록 대상(기본 프로세스 전역 파이프라인 — 테스트는 새 인스턴스 주입)
    :param instant_wildcards: ``$$name$$`` 매니저(기본 프로세스 싱글톤)
    :return: 새로 등록된 훅 개수
    """
    from core.prompt_pipeline import get_pipeline, HookPoint
    from core.rfp_engine import make_dedupe_hook

    pl = pipeline if pipeline is not None else get_pipeline()
    n = 0

    # FINAL — 중복 제거 + prefix/postfix 충돌 제거
    if not pl.has_hook(HookPoint.FINAL, DEDUPE_HOOK_NAME):
        pl.register(
            HookPoint.FINAL,
            make_dedupe_hook(),
            priority=90,
            name=DEDUPE_HOOK_NAME,
        )
        n += 1

    # POST_PROCESSING — $$name$$ 인스턴트 와일드카드. 저장 경로를 못 여는 등(StoragePathError)
    # 실패해도 나머지 훅과 부팅은 계속된다 — 그때 $$name$$ 은 원문 그대로 남는다.
    try:
        from core.instant_wildcards import ensure_hook_registered
        if ensure_hook_registered(pl, instant_wildcards):
            n += 1
    except Exception:
        _logger.exception("인스턴트 와일드카드 훅 등록 실패 — $$name$$ 치환 없이 계속")

    if n:
        _logger.info(f"표준 훅 {n}개 등록")
    return n


def run_pipeline_on_text(text: str, settings: dict | None = None) -> str:
    """문자열 프롬프트를 PromptContext로 감싸 파이프라인 실행 후 다시 문자열.

    생성 흐름의 컨버전스 포인트에서 호출 — 기존 처리 결과를 받아 추가 가공.

    빈 입력은 그대로 반환. 예외 시 원본 반환 (보수적).
    """
    if not text or not text.strip():
        return text
    try:
        from core.prompt_pipeline import get_pipeline, HookPoint, PromptContext
        ctx = PromptContext(
            main_tags=[t.strip() for t in text.split(",") if t.strip()],
            settings=settings or {},
        )
        pl = get_pipeline()
        # PRE/FIT_RESOLUTION/POST는 사용자가 직접 훅 추가 시에만 동작
        pl.execute(HookPoint.PRE_PROCESSING, ctx)
        pl.execute(HookPoint.FIT_RESOLUTION, ctx)
        pl.execute(HookPoint.POST_PROCESSING, ctx)
        pl.execute(HookPoint.AFTER_WILDCARD, ctx)
        pl.execute(HookPoint.FINAL, ctx)
        return ", ".join(ctx.main_tags)
    except Exception:
        _logger.exception("pipeline 실행 실패 — 원본 반환")
        return text
