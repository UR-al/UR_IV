# core/cancellable_call.py
"""백엔드 어댑터에 ``cancel_check`` 를 '받을 수 있을 때만' 넘기는 공용 헬퍼 (순수, Qt 비의존).

``AbstractBackend.txt2img/img2img`` 와 ``run_krea2_generation`` 은 ``cancel_check`` 를 받는다.
하지만 duck-typed 어댑터(테스트 fake, ``get_object_info`` 같은 보조 메서드, 옛 확장 어댑터)는
안 받는다. 예전엔 같은 ``inspect.signature`` 판정이 generation_api 와 krea2_generation 에
복제돼 있었고, GUI 생성 워커·채팅 생성은 아예 ``cancel_check`` 를 넘기지 않아 모델 전환
중이나 큐 투입 직후에 누른 취소가 사라졌다(감사 #31).

취소 확인 시의 예외 타입은 호출부마다 다르다(GenerationConflictError / RuntimeError) —
API 작업 상태·응답 매핑에 쓰이므로 여기서는 판정과 호출만 공유하고 예외는 호출부가 고른다.
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Optional


def accepts_keyword(method: Callable[..., Any], name: str) -> bool:
    """``method`` 가 키워드 인자 ``name`` 을 받을 수 있는지(``**kwargs`` 포함)."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == name or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def call_with_optional_cancel(
    method: Callable[..., Any],
    *args: Any,
    cancel_check: Optional[Callable[[], bool]] = None,
    **kwargs: Any,
) -> Any:
    """``method(*args, **kwargs)`` 를 호출하되, 받을 수 있으면 ``cancel_check`` 도 넘긴다.

    ``cancel_check`` 가 None 이거나 어댑터가 받지 못하면 넘기지 않는다(옛 어댑터 호환).
    호출 전 취소 확인은 하지 않는다 — 호출부가 자기 예외/결과 형식으로 먼저 확인한다.
    """
    if cancel_check is not None and accepts_keyword(method, "cancel_check"):
        return method(*args, cancel_check=cancel_check, **kwargs)
    return method(*args, **kwargs)
