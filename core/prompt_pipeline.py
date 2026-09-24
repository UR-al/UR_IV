"""
PromptPipeline — 우선순위 기반 5단계 프롬프트 처리 훅 시스템.

목적
----
프롬프트 생성 중간 단계에 모듈이 안전하게 개입하는 표준 방법.
UR_IV의 ``ui/generator_prompts.py`` 등에 흩어진 ad-hoc 처리를
점진적으로 이 파이프라인의 훅으로 이전.

5 단계 (실행 순서)
-------------------
1. ``PRE_PROCESSING``    — 원본 row/입력 → 초기 가공 (제외 필터 사전 적용 등)
2. ``FIT_RESOLUTION``    — 해상도/비율에 맞춘 태그 보정
3. ``POST_PROCESSING``   — 와일드카드 확장 전 가공 (조건부 규칙, 가중치 조정)
4. ``AFTER_WILDCARD``    — 와일드카드 확장 후 가공 (이력 추적, 후처리)
5. ``FINAL``             — 최종 포맷팅 직전 (LoRA stack 합침, 정렬 등)

설계
----
- 훅은 우선순위 ``priority`` 정수로 정렬 (낮을수록 먼저).
- 동일 priority 내에서는 등록 순서.
- 훅 예외는 격리 — 다른 훅은 계속 실행.
- 훅이 ``context.skip_remaining = True``로 설정하면 같은 단계의 후속 훅 스킵.
- 모든 훅은 ``PromptContext``를 직접 변경 (불변 패턴 아님 — 성능 우선).

사용 예
-------
>>> pipeline = get_pipeline()
>>> def add_quality_tags(ctx: PromptContext) -> None:
...     ctx.prefix_tags.insert(0, "masterpiece, best quality")
>>> pipeline.register(HookPoint.PRE_PROCESSING, add_quality_tags, priority=10)
>>>
>>> ctx = PromptContext(source_row=row, settings=settings)
>>> pipeline.execute(HookPoint.PRE_PROCESSING, ctx)
>>> # ... ctx.prefix_tags / main_tags / postfix_tags 사용
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from utils.app_logger import get_logger

_logger = get_logger("prompt_pipeline")


class HookPoint(Enum):
    """프롬프트 파이프라인 훅 포인트."""
    PRE_PROCESSING = "pre_processing"
    FIT_RESOLUTION = "fit_resolution"
    POST_PROCESSING = "post_processing"
    AFTER_WILDCARD = "after_wildcard"
    FINAL = "final"


@dataclass
class PromptContext:
    """파이프라인 단계 간 전달되는 상태.

    훅은 이 객체의 필드를 직접 변경한다.
    """
    # 원본 입력 (DataFrame row, dict, 또는 None)
    source_row: Any = None
    # 사용자 설정 / 탭 파라미터
    settings: dict = field(default_factory=dict)

    # 분리된 태그 슬롯 — 훅이 각자 가공
    prefix_tags: list[str] = field(default_factory=list)
    main_tags: list[str] = field(default_factory=list)
    postfix_tags: list[str] = field(default_factory=list)
    negative_tags: list[str] = field(default_factory=list)

    # 최종 결과 (FINAL 단계에서 채움)
    final_prompt: str = ""
    final_negative: str = ""

    # 자유 메타 (훅 간 데이터 공유용)
    metadata: dict = field(default_factory=dict)

    # 훅이 True로 설정하면 같은 단계의 후속 훅 스킵
    skip_remaining: bool = False

    def merge_all_tags(self) -> str:
        """prefix + main + postfix를 쉼표로 합쳐 반환 (편의 메서드)."""
        parts = [", ".join(self.prefix_tags),
                 ", ".join(self.main_tags),
                 ", ".join(self.postfix_tags)]
        return ", ".join(p for p in parts if p)


@dataclass
class Hook:
    """등록된 훅. ``priority`` 낮을수록 먼저 실행."""
    callback: Callable[[PromptContext], None]
    priority: int = 100
    name: str = ""

    def __repr__(self) -> str:
        n = self.name or getattr(self.callback, "__name__", "<lambda>")
        return f"Hook({n!r}, priority={self.priority})"


class PromptPipeline:
    """5단계 훅을 관리·실행하는 컨테이너.

    스레드 안전. 단, 훅 콜백 자체는 호출 스레드에서 동기 실행.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[Hook]] = {p: [] for p in HookPoint}
        self._lock = threading.RLock()

    def register(
        self,
        point: HookPoint,
        callback: Callable[[PromptContext], None],
        priority: int = 100,
        name: str = "",
    ) -> Hook:
        """훅 등록. 반환값을 ``unregister``에 넘겨 해제 가능.

        :param point: 훅을 끼울 단계
        :param callback: ``(PromptContext) -> None``. 컨텍스트를 직접 변경.
        :param priority: 낮을수록 먼저. 기본 100.
        :param name: 디버깅용 이름 (로그/repr에 표시).
        """
        if not isinstance(point, HookPoint):
            raise TypeError(f"point must be HookPoint, got {type(point).__name__}")
        if not callable(callback):
            raise TypeError("callback must be callable")

        hook = Hook(callback=callback, priority=priority, name=name)
        with self._lock:
            self._hooks[point].append(hook)
            # priority 안정 정렬 — 동일 priority 내 등록 순서 보존
            self._hooks[point].sort(key=lambda h: h.priority)
        return hook

    def unregister(self, point: HookPoint, hook: Hook) -> bool:
        """등록된 훅 해제. 성공 시 True."""
        with self._lock:
            hooks = self._hooks[point]
            try:
                hooks.remove(hook)
                return True
            except ValueError:
                return False

    def unregister_all(self, point: Optional[HookPoint] = None) -> int:
        """모든 훅 해제. point 지정 시 해당 단계만. 해제된 개수 반환."""
        with self._lock:
            if point is None:
                total = sum(len(h) for h in self._hooks.values())
                for p in self._hooks:
                    self._hooks[p].clear()
                return total
            count = len(self._hooks[point])
            self._hooks[point].clear()
            return count

    def has_hook(self, point: HookPoint, name: str) -> bool:
        """``name`` 으로 등록된 훅이 point 단계에 있는가 — 멱등 등록용(register 는 중복을 거르지 않는다)."""
        with self._lock:
            return any(h.name == name for h in self._hooks[point])

    def hook_count(self, point: Optional[HookPoint] = None) -> int:
        with self._lock:
            if point is None:
                return sum(len(h) for h in self._hooks.values())
            return len(self._hooks[point])

    def execute(self, point: HookPoint, context: PromptContext) -> PromptContext:
        """해당 단계의 훅들을 우선순위 순으로 실행.

        - 각 훅 예외는 격리 (로그만 남기고 계속).
        - 훅이 ``context.skip_remaining = True`` 설정 시 같은 단계의 후속 훅 스킵.
        - 단계 진입 시 ``skip_remaining``은 항상 False로 리셋.

        반환값은 동일 context 객체 (체이닝 편의용).
        """
        with self._lock:
            hooks_snapshot = list(self._hooks[point])

        context.skip_remaining = False
        for hook in hooks_snapshot:
            if context.skip_remaining:
                break
            try:
                hook.callback(context)
            except Exception:
                _logger.exception(
                    f"hook {hook!r} at {point.value} raised — continuing"
                )
        return context

    def execute_all(self, context: PromptContext) -> PromptContext:
        """5단계를 정의 순서대로 모두 실행. 편의 메서드."""
        for point in HookPoint:  # Enum은 정의 순서 보존
            self.execute(point, context)
        return context


# ─────────────────────────────────────────────
# 모듈 레벨 싱글톤
# ─────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_pipeline_instance: Optional[PromptPipeline] = None


def get_pipeline() -> PromptPipeline:
    """프로세스 전역 PromptPipeline 인스턴스."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = PromptPipeline()
    return _pipeline_instance
