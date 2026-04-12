# core/tag_matcher.py
"""통합 태그 매칭 엔진 — Search / Event Gen 공용

문법:
  word      → 포함 매칭 (tag에 word가 포함된 모든 것)
  *word     → 완전 일치 (정확히 word인 태그만)
  _word     → 접미 매칭 (word로 끝나는 태그, e.g. short hair, long hair)
  word_     → 접두 매칭 (word로 시작하는 태그, e.g. hair ornament)
  _word_    → 포함 매칭 (명시적, word와 동일)
  [A, B]    → AND 그룹 (A와 B 모두 존재)
  [A|B]     → OR 그룹 (A 또는 B 중 하나 이상)
  쉼표(,)   → AND (대괄호 밖에서)
"""
import re
import pandas as pd


def _normalize(text: str) -> str:
    """밑줄을 공백으로, 소문자"""
    return text.strip().lower().replace('_', ' ')


def _match_single_tag(pattern: str, tag_text: str) -> bool:
    """단일 패턴을 태그 문자열에 대해 매칭
    tag_text: 콤마 또는 공백으로 구분된 전체 태그 문자열 (소문자, 밑줄→공백 변환 완료)
    """
    pattern = pattern.strip()
    if not pattern:
        return True

    # *word → 완전 일치
    if pattern.startswith('*'):
        target = _normalize(pattern[1:])
        # 콤마로 분리 후 정확 비교 (태그 내 공백 보존)
        tags = [t.strip() for t in tag_text.split(',') if t.strip()]
        return target in tags

    # _word_ → 포함 (명시적)
    if pattern.startswith('_') and pattern.endswith('_') and len(pattern) > 2:
        target = _normalize(pattern[1:-1])
        return target in tag_text

    # _word → 접미 (word로 끝나는 태그)
    if pattern.startswith('_') and not pattern.endswith('_'):
        target = _normalize(pattern[1:])
        tags = [t.strip() for t in tag_text.split(',') if t.strip()]
        return any(t.endswith(target) for t in tags)

    # word_ → 접두 (word로 시작하는 태그)
    if pattern.endswith('_') and not pattern.startswith('_'):
        target = _normalize(pattern[:-1])
        tags = [t.strip() for t in tag_text.split(',') if t.strip()]
        return any(t.startswith(target) for t in tags)

    # 기본: 포함 매칭
    target = _normalize(pattern)
    return target in tag_text


def parse_query(query: str) -> list:
    """쿼리 문자열을 파싱하여 조건 리스트로 변환

    Returns: [ {'type': 'and'|'or'|'single', 'terms': [str, ...]} ]
    """
    if not query or not query.strip():
        return []

    # 대괄호 밖의 쉼표로 분리
    parts = re.split(r',\s*(?![^\[]*\])', query.strip())
    conditions = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # [A|B|C] → OR 그룹
        m = re.match(r'^\[(.+)\]$', part)
        if m:
            inner = m.group(1)
            if '|' in inner:
                terms = [t.strip() for t in inner.split('|') if t.strip()]
                conditions.append({'type': 'or', 'terms': terms})
            elif ',' in inner:
                terms = [t.strip() for t in inner.split(',') if t.strip()]
                conditions.append({'type': 'and', 'terms': terms})
            else:
                conditions.append({'type': 'single', 'terms': [inner.strip()]})
        else:
            conditions.append({'type': 'single', 'terms': [part]})

    return conditions


def match_query(query: str, tag_text: str) -> bool:
    """쿼리 전체를 태그 텍스트에 대해 매칭

    query: 사용자 입력 (예: "[short hair|long hair], blue eyes")
    tag_text: 태그 문자열 (예: "1girl, short hair, blue eyes, school uniform")
    Returns: True if all conditions match
    """
    conditions = parse_query(query)
    if not conditions:
        return True

    normalized = _normalize(tag_text)

    for cond in conditions:
        if cond['type'] == 'or':
            # OR: 하나 이상 매칭
            if not any(_match_single_tag(t, normalized) for t in cond['terms']):
                return False
        elif cond['type'] == 'and':
            # AND: 모두 매칭
            if not all(_match_single_tag(t, normalized) for t in cond['terms']):
                return False
        else:
            # single
            if not _match_single_tag(cond['terms'][0], normalized):
                return False

    return True


def filter_dataframe(df: pd.DataFrame, col: str, query: str) -> pd.Series:
    """DataFrame에 쿼리를 적용하여 Boolean mask 반환

    df: DataFrame
    col: 태그 문자열이 있는 컬럼명
    query: 사용자 입력 쿼리
    Returns: pd.Series[bool]
    """
    conditions = parse_query(query)
    if not conditions:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)
    col_lower = df[col].fillna('').str.lower()

    for cond in conditions:
        if cond['type'] == 'or':
            or_mask = pd.Series(False, index=df.index)
            for term in cond['terms']:
                or_mask |= _apply_pattern(col_lower, term)
            mask &= or_mask
        elif cond['type'] == 'and':
            for term in cond['terms']:
                mask &= _apply_pattern(col_lower, term)
        else:
            mask &= _apply_pattern(col_lower, cond['terms'][0])

    return mask


def _apply_pattern(col_series: pd.Series, pattern: str) -> pd.Series:
    """단일 패턴을 Series에 적용하여 Boolean mask 반환
    col_series: 소문자 변환된 원본 (danbooru: 공백+밑줄, SD: 콤마+공백)
    양쪽 형식 모두 지원: 패턴을 공백/밑줄 양쪽으로 매칭
    """
    pattern = pattern.strip()
    if not pattern:
        return pd.Series(True, index=col_series.index)

    def _both(text):
        """공백과 밑줄 양쪽 버전의 정규식 OR 패턴"""
        t1 = re.escape(text.lower().replace('_', ' '))
        t2 = re.escape(text.lower().replace(' ', '_'))
        return f'(?:{t1}|{t2})' if t1 != t2 else t1

    # 태그 경계: 시작/끝 또는 구분자(콤마, 공백)
    SEP_L = r'(?:^|[,\s]\s*)'
    SEP_R = r'(?:\s*[,\s]|$)'

    # *word → 완전 일치
    if pattern.startswith('*'):
        target = pattern[1:].strip()
        pat = SEP_L + _both(target) + SEP_R
        return col_series.str.contains(pat, regex=True, na=False)

    # _word_ → 포함 (명시적)
    if pattern.startswith('_') and pattern.endswith('_') and len(pattern) > 2:
        target = pattern[1:-1].strip()
        t1 = target.lower().replace('_', ' ')
        t2 = target.lower().replace(' ', '_')
        return col_series.str.contains(t1, regex=False, na=False) | col_series.str.contains(t2, regex=False, na=False)

    # _word → 접미 (태그가 word로 끝남)
    if pattern.startswith('_') and not pattern.endswith('_'):
        target = pattern[1:].strip()
        pat = _both(target) + SEP_R
        return col_series.str.contains(pat, regex=True, na=False)

    # word_ → 접두 (태그가 word로 시작)
    if pattern.endswith('_') and not pattern.startswith('_'):
        target = pattern[:-1].strip()
        pat = SEP_L + _both(target)
        return col_series.str.contains(pat, regex=True, na=False)

    # 기본: 포함 매칭 (양쪽 형식)
    t1 = pattern.lower().replace('_', ' ')
    t2 = pattern.lower().replace(' ', '_')
    return col_series.str.contains(t1, regex=False, na=False) | col_series.str.contains(t2, regex=False, na=False)
