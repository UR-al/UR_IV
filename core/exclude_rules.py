# core/exclude_rules.py
"""제외 규칙 (9종) — 문법의 단일 출처 (Qt 없음).

`제외 (로컬)` 칸의 텍스트를 규칙으로 나누고(split), 규칙마다 종류를 가리고(parse),
데이터 태그 목록에 적용한다(filter). 쓰는 곳:

  * 프롬프트 적용 — ui/generator_prompts.apply_prompt_from_data
  * 관리 창 미리보기 — core/exclude_rule_match → VueBridge.getExcludeMatches
  * 프론트 관리 창의 나누기·분류는 frontend/src/utils/excludeRules.ts 가 같은 정의로 갖는다.
    두 구현은 frontend/src/utils/excludeRules.cases.json 골든 사례로 묶여 있다
    (tests/test_exclude_rules.py 와 excludeRules.test.ts 가 같은 파일을 읽는다).

  제외                          예외 (유지)
  word     포함                 ~word     완전 일치 유지
  *word    완전 일치            ~_word    접미 유지 (…word)
  _word    접미 (…word)         ~word_    접두 유지 (word…)
  word_    접두 (word…)         ~_word_   포함 유지
  _word_   포함 (명시)

나누기: 쉼표와 줄바꿈만 구분자다. 공백으로는 절대 나누지 않고(`long hair` 는 규칙 하나),
밑줄도 파싱 때 바꾸지 않는다 — 앞뒤 밑줄이 규칙 종류를 정하기 때문이다.
(옛 코드는 태그 문자열용 to_list 를 빌려 써서, 쉼표가 없으면 공백으로 나누고 밑줄을 공백으로
바꿨다. 그래서 규칙 하나짜리 `_short` 가 포함 규칙 `short` 로 바뀌었다.)
예외 하나: 키워드 없이 표시만 있는 조각(`~`, `*`, `_`, `~_` …)이 줄 끝에 있으면 쉼표를 만나기 전의
다음 조각과 한 규칙이다(`x, ~` 줄 + `solo` 줄 = `~solo` 유지). 옛 적용은 쉼표 목록 안의 줄바꿈을
공백으로 봤기 때문에, 줄바꿈 구분을 더하면서 이 모양이 `solo`(포함 제외)로 뒤집히지 않게 한다.
표시만 있는 조각이 끝이나 쉼표 앞에 있으면 규칙이 없으므로 버린다.

비교할 때만 소문자 + 밑줄→공백 + 앞뒤 공백 제거로 맞춘다. 적용 순서:
예외(유지) 규칙이 먼저 → 완전 일치 → 포함 → 접두 → 접미.
고유명사 칸(캐릭터·작품·작가, ``proper=True``)에서는 `word`·`_word_` 포함 규칙이 태그 전체와
같을 때만 걸린다('tank' 가 캐릭터 'wakan tanka' 를 지우지 않게). 완전 일치·접두·접미는
태그 경계 기준이라 모든 칸에 그대로 적용된다. 예외 규칙도 모든 칸에 같다.

키워드가 비는 규칙(`_`, `__`, `*`, `~` 등)은 버린다 — 옛 코드는 `__`·`___` 를 빈 접미·포함
규칙으로 남겨 모든 태그를 지웠다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# 비교 방식
EXACT = "exact"
CONTAINS = "contains"
PREFIX = "prefix"
SUFFIX = "suffix"

# 문법 9종 → (비교 방식, 유지 규칙인가)
RULE_FORMS: dict[str, tuple[str, bool]] = {
    "contains": (CONTAINS, False),            # word
    "exact": (EXACT, False),                  # *word
    "suffix": (SUFFIX, False),                # _word
    "prefix": (PREFIX, False),                # word_
    "contains_explicit": (CONTAINS, False),   # _word_
    "keep_exact": (EXACT, True),              # ~word
    "keep_suffix": (SUFFIX, True),            # ~_word
    "keep_prefix": (PREFIX, True),            # ~word_
    "keep_contains": (CONTAINS, True),        # ~_word_
}

# 규칙 구분자 — 쉼표와 줄바꿈. 공백은 구분자가 아니다. (excludeRules.ts 와 같은 정의)
_SEPARATORS = re.compile(r"[,\r\n]")


def _rule_spans(text: str) -> list[tuple[int, int]]:
    """규칙 자리 [start, end) 목록 (앞뒤 공백 제외) — excludeRules.ts 의 excludeRuleSpans 와 같다.

    쉼표·줄바꿈으로 나누고 빈 조각은 버린다. 표시만 있는 조각(parse_exclude_rule 이 None)이
    줄바꿈 앞에 있으면 쉼표를 만나기 전의 다음 조각까지 한 자리로 잇는다.
    """
    spans: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None   # 줄 끝에 걸린 표시만 있는 조각 — 다음 조각과 잇는다
    pos = 0
    while True:
        m = _SEPARATORS.search(text, pos)
        seg_end = m.start() if m else len(text)
        sep = m.group() if m else ""
        raw = text[pos:seg_end]
        body = raw.strip()
        if body:
            start = pos + len(raw) - len(raw.lstrip())
            cur = (start, start + len(body))
            if pending is not None:
                cur = (pending[0], cur[1])
                pending = None
            if sep in ("\n", "\r") and parse_exclude_rule(text[cur[0]:cur[1]]) is None:
                pending = cur
            else:
                spans.append(cur)
        elif sep == "," and pending is not None:
            spans.append(pending)   # 표시만 있는 조각 뒤에 쉼표 — 잇지 않는다
            pending = None
        if m is None:
            break
        pos = m.end()
    if pending is not None:
        spans.append(pending)
    return spans


def split_exclude_rules(text: object) -> list[str]:
    """`제외 (로컬)` 텍스트 → 규칙 문자열 목록 (앞뒤 공백 제거, 빈 항목 버림, 원문 보존)."""
    if not isinstance(text, str):
        return []
    return [text[start:end] for start, end in _rule_spans(text)]


def normalize_exclude_tag(text: str) -> str:
    """비교용 표기 — 밑줄→공백, 앞뒤 공백 제거, 소문자. 규칙 키워드와 태그에 같이 쓴다."""
    return text.replace("_", " ").strip().lower()


def keyword_matches(match: str, text: str, keyword: str) -> bool:
    """비교 방식 하나 — `text`·`keyword` 는 같은 표기(normalize_exclude_tag)여야 한다."""
    if match == EXACT:
        return text == keyword
    if match == CONTAINS:
        return keyword in text
    if match == PREFIX:
        return text.startswith(keyword)
    if match == SUFFIX:
        return text.endswith(keyword)
    raise ValueError(f"unknown exclude match kind: {match!r}")


@dataclass(frozen=True)
class ExcludeRule:
    """파싱된 규칙 하나. `keyword` 는 비교용 표기이고 비어 있지 않다."""

    text: str      # 원문 (앞뒤 공백 제거)
    form: str      # RULE_FORMS 의 키 (9종)
    keyword: str   # normalize_exclude_tag 표기

    @property
    def match(self) -> str:
        return RULE_FORMS[self.form][0]

    @property
    def keep(self) -> bool:
        return RULE_FORMS[self.form][1]

    def matches(self, normalized_tag: str, *, proper: bool = False) -> bool:
        """정규화된 태그에 이 규칙이 걸리는가 (유지·제외 여부와 무관한 순수 매칭)."""
        if proper and not self.keep and self.match == CONTAINS:
            # 고유명사 칸 — 포함 규칙은 태그 전체 일치만
            return normalized_tag == self.keyword
        return keyword_matches(self.match, normalized_tag, self.keyword)


def _classify(rule: str) -> tuple[str, str]:
    """(9종 이름, 키워드 원문). `rule` 은 앞뒤 공백이 없는 비지 않은 문자열."""
    if rule.startswith("~"):
        inner = rule[1:].strip()
        if inner.startswith("_") and inner.endswith("_") and len(inner) > 2:
            return "keep_contains", inner[1:-1]
        if inner.startswith("_"):
            return "keep_suffix", inner[1:]
        if inner.endswith("_"):
            return "keep_prefix", inner[:-1]
        return "keep_exact", inner
    if rule.startswith("*"):
        return "exact", rule[1:]
    if rule.startswith("_") and rule.endswith("_") and len(rule) > 2:
        return "contains_explicit", rule[1:-1]
    if rule.startswith("_"):
        return "suffix", rule[1:]
    if rule.endswith("_"):
        return "prefix", rule[:-1]
    return "contains", rule


def parse_exclude_rule(text: object) -> ExcludeRule | None:
    """규칙 하나 → ExcludeRule. 빈 칸이거나 키워드가 비면 None (나누지 않는다)."""
    if not isinstance(text, str):
        return None
    rule = text.strip()
    if not rule:
        return None
    form, raw_keyword = _classify(rule)
    keyword = normalize_exclude_tag(raw_keyword)
    if not keyword:
        return None
    return ExcludeRule(text=rule, form=form, keyword=keyword)


class ExcludeRuleSet:
    """규칙 묶음 — 태그 목록에 제외 규칙을 적용한다."""

    def __init__(self, rules: Iterable[ExcludeRule]):
        self.rules: tuple[ExcludeRule, ...] = tuple(rules)

        def keywords(keep: bool, match: str) -> tuple[str, ...]:
            return tuple(r.keyword for r in self.rules if r.keep == keep and r.match == match)

        # 규칙 수백 개 × 태그 수십 개를 GUI 스레드에서 매 사이클 돈다 — 규칙마다 메서드를 부르지 않고
        # 종류별 키워드 묶음으로 본다(일치=집합, 접두·접미=str.startswith/endswith 튜플).
        # 의미는 ExcludeRule.matches 와 같다(tests/test_exclude_rules 가 둘을 대조한다).
        self._keep_exact = frozenset(keywords(True, EXACT))
        self._keep_prefix = keywords(True, PREFIX)
        self._keep_suffix = keywords(True, SUFFIX)
        self._keep_contains = keywords(True, CONTAINS)
        self._exact = frozenset(keywords(False, EXACT))
        self._contains = keywords(False, CONTAINS)
        self._contains_whole = frozenset(self._contains)   # 고유명사 칸 — 태그 전체 일치만
        self._prefix = keywords(False, PREFIX)
        self._suffix = keywords(False, SUFFIX)

    def __len__(self) -> int:
        return len(self.rules)

    def __bool__(self) -> bool:
        return bool(self.rules)

    def _kept(self, nt: str) -> bool:
        return (nt in self._keep_exact
                or nt.endswith(self._keep_suffix)
                or nt.startswith(self._keep_prefix)
                or any(c in nt for c in self._keep_contains))

    def excludes(self, tag: str, *, proper: bool = False) -> bool:
        """이 태그를 지우는가. 예외(유지) 규칙이 먼저다."""
        nt = normalize_exclude_tag(tag)
        if self._kept(nt):
            return False
        if nt in self._exact:
            return True
        if proper:
            if nt in self._contains_whole:
                return True
        elif any(c in nt for c in self._contains):
            return True
        # 접두·접미는 태그 경계 기준이라 고유명사 칸에도 그대로 적용한다
        return nt.startswith(self._prefix) or nt.endswith(self._suffix)

    def filter_tags(self, tags: Iterable[str], *, proper: bool = False) -> list[str]:
        """제외되지 않은 태그만 원래 순서·원래 표기로 돌려준다."""
        if not self.rules:
            return list(tags)
        return [t for t in tags if not self.excludes(t, proper=proper)]


def parse_exclude_rules(source: object) -> ExcludeRuleSet:
    """`제외 (로컬)` 텍스트(또는 이미 나눈 규칙 목록) → ExcludeRuleSet."""
    if isinstance(source, str):
        items: Iterable[object] = split_exclude_rules(source)
    elif source is None:
        items = ()
    else:
        items = source  # type: ignore[assignment]
    parsed = (parse_exclude_rule(item) for item in items)
    return ExcludeRuleSet(rule for rule in parsed if rule is not None)


__all__ = [
    "CONTAINS", "EXACT", "PREFIX", "SUFFIX", "RULE_FORMS",
    "ExcludeRule", "ExcludeRuleSet",
    "keyword_matches", "normalize_exclude_tag",
    "parse_exclude_rule", "parse_exclude_rules", "split_exclude_rules",
]
