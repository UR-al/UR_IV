# core/prompt_deep_clean.py
"""딥 프롬프트 클리너 — 프롬프트 패널 [최적화] 의 본체(VueBridge.deepCleanPrompt). Qt 비의존.

최적화 대상은 **메인 태그 한 칸**이다. 예전 호출은 7칸 합본(최종 프롬프트)을 정리한 결과를
메인 칸에 통째로 써서 인물수·캐릭터·작품·작가·접두·접미 태그가 메인에 영구 복제됐다(캐릭터를
바꾸거나 작가를 지워도 옛 태그가 메인에 남아 생성에 섞였다).

이제 나머지 6칸은 ``context`` 로만 받는다.
  - context 태그를 같은 정규화(소문자, 공백→밑줄)로 먼저 '본 것'에 넣어, 다른 칸에 이미 있는
    태그는 메인에서 뺀다(합본에 두 번 들어가지 않게).
  - 충돌 검사(머리색·눈색 …)는 메인 ∪ context 로 한다 — 생성 프롬프트 전체 기준.
  - 결과(optimized)는 메인 태그만 담는다. context 는 절대 결과에 들어가지 않는다.
``context`` 는 선택 키라 옛 호출(prompt 만)도 그대로 동작한다.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

_CONFLICT_GROUPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (('black_hair', 'blonde_hair', 'brown_hair', 'red_hair', 'blue_hair', 'green_hair', 'white_hair',
      'pink_hair', 'purple_hair', 'silver_hair', 'orange_hair', 'grey_hair'), '머리색'),
    (('blue_eyes', 'red_eyes', 'green_eyes', 'brown_eyes', 'yellow_eyes', 'purple_eyes', 'pink_eyes',
      'grey_eyes', 'black_eyes', 'orange_eyes'), '눈색'),
    (('short_hair', 'long_hair', 'very_long_hair', 'medium_hair'), '머리 길이'),
    (('standing', 'sitting', 'lying', 'kneeling', 'squatting'), '포즈'),
    (('day', 'night', 'sunset', 'sunrise'), '시간'),
    (('indoors', 'outdoors'), '장소'),
)
_QUALITY_TAGS = frozenset({'masterpiece', 'best_quality', 'high_quality', 'absurdres', 'highres'})
_COUNT_TAGS = frozenset({'1girl', '2girls', '3girls', '1boy', '2boys', 'solo', 'multiple_girls', 'multiple_boys'})


def tag_key(tag: str) -> str:
    """비교 키 — 소문자, 공백→밑줄(예전 구현과 같은 정규화)."""
    return str(tag).strip().lower().replace(' ', '_')


def split_tags(text: object) -> list[str]:
    return [t.strip() for t in str(text or '').split(',') if t.strip()]


def _context_tags(context: Union[None, str, Iterable[object]]) -> list[str]:
    if context is None:
        return []
    if isinstance(context, str):
        return split_tags(context)
    out: list[str] = []
    try:
        for part in context:
            if isinstance(part, str):
                out.extend(split_tags(part))
    except TypeError:
        return []
    return out


def deep_clean_prompt(prompt: object, context: Union[None, str, Iterable[object]] = None) -> dict:
    """메인 태그 ``prompt`` 를 정리한다.

    반환: ``{optimized, removed, removed_from_context, conflicts, tag_count}``
      - optimized: 정리된 메인 태그(쉼표+공백 결합). 전부 다른 칸과 겹치면 빈 문자열.
      - removed: 메인에서 뺀 태그 수(메인 안의 중복 + 다른 칸과 겹친 것)
      - removed_from_context: 그중 다른 칸에 이미 있어서 뺀 수
      - conflicts: [{group, tags}] — 메인 ∪ context 기준
      - tag_count: optimized 의 태그 수
    """
    tags = split_tags(prompt)
    ctx = _context_tags(context)
    ctx_keys = {tag_key(t) for t in ctx}

    seen: set[str] = set(ctx_keys)
    unique: list[str] = []
    from_context = 0
    for tag in tags:
        key = tag_key(tag)
        if key in seen:
            if key in ctx_keys:
                from_context += 1
            continue
        seen.add(key)
        unique.append(tag)

    present = {tag_key(t) for t in unique} | ctx_keys
    conflicts = []
    for group, label in _CONFLICT_GROUPS:
        found = [t for t in group if t in present]
        if len(found) > 1:
            conflicts.append({'group': label, 'tags': found})

    # 순서 재배치: 인물수 → 품질 → 나머지(원래 순서 유지)
    count, quality, other = [], [], []
    for tag in unique:
        key = tag_key(tag)
        if key in _QUALITY_TAGS:
            quality.append(tag)
        elif key in _COUNT_TAGS:
            count.append(tag)
        else:
            other.append(tag)
    optimized = count + quality + other
    return {
        'optimized': ', '.join(optimized),
        'removed': len(tags) - len(unique),
        'removed_from_context': from_context,
        'conflicts': conflicts,
        'tag_count': len(optimized),
    }


def deep_clean_request(data: Optional[dict]) -> dict:
    """브리지 페이로드 ``{prompt, context?}`` → deep_clean_prompt 결과."""
    if not isinstance(data, dict):
        raise ValueError('요청은 JSON 객체여야 합니다')
    return deep_clean_prompt(data.get('prompt', ''), data.get('context'))
