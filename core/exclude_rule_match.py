# core/exclude_rule_match.py
"""제외 규칙 미리보기 — 규칙 하나가 태그 사전에서 어떤 태그를 지우는지 (Qt 없음).

VueBridge.getExcludeMatches 가 쓴다. 규칙을 가리는 일은 프롬프트 적용과 같은 파서
(core.exclude_rules.parse_exclude_rule)가 한다 — 미리보기와 적용이 규칙 종류를 다르게 읽지 않는다.

  *word    완전 일치          _word_   포함(명시)
  _word    접미 (…word)       word_    접두 (word…)
  word     포함               ~…       유지 규칙 — 지우는 태그 없음

태그 사전은 소문자·밑줄 표기다(VueBridge._exclude_vocabulary). 파서의 비교용 키워드(소문자·공백
표기, 밑줄 없음)를 공백→밑줄로 옮겨 사전 표기 그대로 비교한다. 공백↔밑줄은 글자 단위 일대일이라
포함·접두·접미·일치 결과는 적용 쪽(공백 표기 비교)과 같다 — 단, 앞뒤에 밑줄이 붙은 사전 태그는
적용 쪽이 그 밑줄을 잘라 내고 비교하므로 드물게 다를 수 있다.
완전 일치는 집합 조회 한 번이면 되므로 사전 전체를 훑지 않는다. 미리보기는 일반(general) 칸 기준이다
(고유명사 칸의 '포함 규칙은 전체 일치만' 규칙은 적용하지 않는다).
"""
from __future__ import annotations

from typing import Collection, Iterable

from core.exclude_rules import CONTAINS, EXACT, PREFIX, SUFFIX, ExcludeRule, parse_exclude_rule


def vocabulary_form(keyword: str) -> str:
    """파서의 비교용 키워드(공백 표기) → 태그 사전 표기(밑줄 표기)."""
    return keyword.replace(" ", "_")


def _removal_rule(rule: object) -> ExcludeRule | None:
    """지우는 규칙만 — 빈 칸·키워드 없는 규칙·~유지 규칙은 None."""
    parsed = parse_exclude_rule(rule)
    if parsed is None or parsed.keep:
        return None
    return parsed


def normalize_exclude_rule(rule: object) -> str:
    """규칙 → 결과 캐시 키. 같은 규칙으로 읽히는 표기(`_HAIR`·` _hair `·`_ hair`)는 같은 키다.

    지우는 태그가 없는 규칙(빈 칸, `~` 유지 규칙, 키워드가 빈 `_`·`__`)은 "".
    """
    parsed = _removal_rule(rule)
    if parsed is None:
        return ""
    return f"{parsed.match}:{vocabulary_form(parsed.keyword)}"


def match_exclude_rule(rule: object, vocabulary: Collection[str]) -> list[str]:
    """`vocabulary`(소문자·밑줄 표기) 중 규칙에 걸리는 태그를 정렬해 돌려준다."""
    parsed = _removal_rule(rule)
    if parsed is None:
        return []
    keyword = vocabulary_form(parsed.keyword)

    if parsed.match == EXACT:
        # 완전 일치 — 선형 스캔 대신 집합 조회
        return [keyword] if keyword in vocabulary else []

    # 77만 개를 훑는 경로라 비교식을 인라인으로 둔다(keyword_matches 와 같은 정의)
    matches: Iterable[str]
    if parsed.match == CONTAINS:
        matches = (tag for tag in vocabulary if keyword in tag)
    elif parsed.match == PREFIX:
        matches = (tag for tag in vocabulary if tag.startswith(keyword))
    elif parsed.match == SUFFIX:
        matches = (tag for tag in vocabulary if tag.endswith(keyword))
    else:  # pragma: no cover — RULE_FORMS 에 없는 비교 방식
        raise ValueError(f"unknown exclude match kind: {parsed.match!r}")
    return sorted(matches)


__all__ = ["match_exclude_rule", "normalize_exclude_rule", "vocabulary_form"]
