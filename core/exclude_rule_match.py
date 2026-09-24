# core/exclude_rule_match.py
"""제외 규칙 미리보기 — 규칙 하나가 태그 사전에서 어떤 태그를 지우는지 (Qt 없음).

VueBridge.getExcludeMatches 가 쓴다. 규칙 문법은 프롬프트 제외 규칙과 같다.

  *word    완전 일치          _word_   포함(명시)
  _word    접미 (…word)       word_    접두 (word…)
  word     포함               ~…       유지 규칙 — 지우는 태그 없음

비교는 소문자·밑줄 표기로 한다(태그 사전도 그 표기로 만든다).
완전 일치는 집합 조회 한 번이면 되므로 사전 전체를 훑지 않는다.
"""
from __future__ import annotations

from typing import Collection, Iterable


def normalize_exclude_rule(rule: object) -> str:
    """규칙 → 비교용 키 (앞뒤 공백 제거, 소문자, 공백→밑줄)."""
    if not isinstance(rule, str):
        return ""
    return rule.strip().lower().replace(" ", "_")


def match_exclude_rule(rule: object, vocabulary: Collection[str]) -> list[str]:
    """`vocabulary` 중 규칙에 걸리는 태그를 정렬해 돌려준다."""
    key = normalize_exclude_rule(rule)
    if not key or key.startswith("~"):
        return []

    if key.startswith("*"):
        keyword = key[1:]
        # 완전 일치 — 선형 스캔 대신 집합 조회
        return [keyword] if keyword in vocabulary else []

    matches: Iterable[str]
    if key.startswith("_") and key.endswith("_") and len(key) > 2:
        keyword = key[1:-1]
        matches = (tag for tag in vocabulary if keyword in tag)
    elif key.startswith("_"):
        keyword = key[1:]
        matches = (tag for tag in vocabulary if tag.endswith(keyword))
    elif key.endswith("_"):
        keyword = key[:-1]
        matches = (tag for tag in vocabulary if tag.startswith(keyword))
    else:
        matches = (tag for tag in vocabulary if key in tag)
    return sorted(matches)


__all__ = ["match_exclude_rule", "normalize_exclude_rule"]
