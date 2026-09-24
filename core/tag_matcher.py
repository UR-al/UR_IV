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

pandas 는 마스크를 만드는 함수 안에서 import 한다(주석의 타입은 TYPE_CHECKING 전용) —
parse_query 만 쓰는 곳이나 창 표시 전 import 경로(core.event_data_loader)가 pandas 로드를
떠안지 않게 한다.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _normalize(text: str) -> str:
    """밑줄을 공백으로, 소문자"""
    return text.strip().lower().replace('_', ' ')


def parse_query(query: str) -> list:
    """쿼리 문자열을 파싱하여 조건 리스트로 변환

    Returns: [ {'type': 'and'|'or'|'single', 'terms': [str, ...], 'has_wildcard'?: bool} ]

    특수 케이스: OR 그룹 안에 빈 토큰이 있으면 (`[A|B|]` 처럼 trailing |)
      → has_wildcard=True 로 표시 → '이 필드 조건은 매칭 안 되어도 통과'
      → 즉 이 OR 그룹이 항상 True를 반환하므로 AND 모드에서도 해당 필드 면제 효과
      예) AND 모드, copyright=`[tears of themis|alchemy stars|]`
          → '작품이 TOT 또는 AS 또는 작품 무관' (copyright 조건 사실상 면제)
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
                raw = [t.strip() for t in inner.split('|')]
                # 와일드카드는 trailing 빈 토큰(`|]` 직전)에서만 인식 — 명시적 신호로 한정.
                # 중간 `||`(이중 파이프)는 사용자 오타/정렬 흔적이라 와일드카드 발동 X,
                # 그냥 빈 토큰은 필터링하여 정리. 양 끝의 leading `|`도 동일하게 무시.
                #   예) [A|B|]     → has_wildcard=True  (trailing empty)
                #   예) [A||B]     → has_wildcard=False (중간 || 는 그냥 정리)
                #   예) [|A|B]     → has_wildcard=False (leading empty 는 무시)
                #   예) [|A|B|]    → has_wildcard=True  (trailing empty 가 있으면 wildcard)
                has_wildcard = len(raw) > 0 and raw[-1] == ''
                terms = [t for t in raw if t]
                conditions.append({
                    'type': 'or',
                    'terms': terms,
                    'has_wildcard': has_wildcard,
                })
            elif ',' in inner:
                # AND 그룹 — 빈 토큰은 그냥 무시 (AND에서 True&X=X 이므로 영향 없음)
                terms = [t.strip() for t in inner.split(',') if t.strip()]
                conditions.append({'type': 'and', 'terms': terms})
            else:
                conditions.append({'type': 'single', 'terms': [inner.strip()]})
        else:
            conditions.append({'type': 'single', 'terms': [part]})

    return conditions


def _both(text: str) -> str:
    """공백형·밑줄형 양쪽을 잡는 정규식(이스케이프됨). 두 형태가 같으면 하나만."""
    t1 = re.escape(text.lower().replace('_', ' '))
    t2 = re.escape(text.lower().replace(' ', '_'))
    return f'(?:{t1}|{t2})' if t1 != t2 else t1


def contains_tag_text(col_series: pd.Series, text: str) -> pd.Series:
    """태그 문자열 부분일치 — 공백형(long hair)·밑줄형(long_hair) 어느 쪽이든.

    col_series 는 소문자 기준(질의만 여기서 소문자로 맞춘다). 두 형태를 정규식 교대
    하나로 묶어 컬럼을 **한 번만** 스캔한다 — 형태마다 str.contains 를 따로 돌리면
    같은 전체 스캔을 두 번 해 plain 텀 비용이 두 배가 된다. 텍스트는 이스케이프되므로
    '(', '+', '?' 같은 문자도 글자 그대로 매칭된다.
    """
    return col_series.str.contains(_both(text), regex=True, na=False)


def _apply_or_plain(col_series: pd.Series, terms: list) -> pd.Series:
    """연산자(*, _ 접두/접미) 없는 plain 텀 여러 개를 '하나의 정규식 교대'로 1회 매칭(OR).
    큰 OR 그룹(수백 캐릭터)에서 텀마다 전체 컬럼을 재스캔하던 것을 1회로 단축 — 9.2M행에서 수십 배 빠름.
    의미는 _apply_pattern 기본 경로와 동일(공백/언더스코어 양쪽 부분일치).
    """
    import pandas as pd

    alts = []
    for t in terms:
        tl = (t or '').strip().lower()
        if not tl:
            continue
        a = re.escape(tl.replace('_', ' '))
        b = re.escape(tl.replace(' ', '_'))
        alts.append(a)
        if b != a:
            alts.append(b)
    if not alts:
        return pd.Series(False, index=col_series.index)
    pat = '(?:' + '|'.join(alts) + ')'
    return col_series.str.contains(pat, regex=True, na=False)


def _eval_condition(col_lower: pd.Series, cond: dict, index) -> pd.Series:
    """단일 조건(dict)을 평가하여 Boolean mask 반환.
    cond['type']: 'or' | 'and' | 'single'
    """
    import pandas as pd

    if cond['type'] == 'or':
        # has_wildcard=True 면 [A|B|] 같은 빈 토큰 포함 그룹 → 무조건 통과
        if cond.get('has_wildcard'):
            return pd.Series(True, index=index)
        # [A|B] — 명시적 OR. 성능: plain 텀들은 정규식 교대로 1회 스캔, 연산자 텀만 개별 처리.
        plain, special = [], []
        for term in cond['terms']:
            ts = (term or '').strip()
            if not ts:
                continue
            (special if (ts[0] in '*_' or ts[-1] == '_') else plain).append(ts)
        cm = pd.Series(False, index=index)
        if plain:
            cm |= _apply_or_plain(col_lower, plain)
        for term in special:
            cm |= _apply_pattern(col_lower, term)
        return cm
    elif cond['type'] == 'and':
        # [A,B] — 명시적 AND 그룹
        cm = pd.Series(True, index=index)
        for term in cond['terms']:
            cm &= _apply_pattern(col_lower, term)
        return cm
    else:
        # single — bare 콤마 토큰
        return _apply_pattern(col_lower, cond['terms'][0])


def filter_dataframe(df: pd.DataFrame, col: str, query: str,
                     default_combine: str = 'and',
                     col_lower: pd.Series = None) -> pd.Series:
    """DataFrame에 쿼리를 적용하여 Boolean mask 반환

    :param df: DataFrame
    :param col: 태그 문자열이 있는 컬럼명
    :param query: 사용자 입력 쿼리
    :param default_combine: 'and' (콤마=AND, 기본) | 'or' (콤마=OR)
        - 명시적 대괄호 그룹은 모드 무관:
          [A|B] → 항상 OR, [A,B] → 항상 AND
        - bare 콤마(대괄호 밖) 토큰 결합 방식만 모드에 따라 변경
    :param col_lower: 미리 lowercase한 시리즈 (성능 최적화용).
        넘기면 .str.lower() 호출 생략. None이면 내부에서 계산.

    구현: 누적식 평가 (마스크를 한 번에 들고 다니지 않음).
      AND 모드: True에서 시작해 &= 누적 → short-circuit 비슷한 효과
      OR  모드: False에서 시작해 |= 누적
    """
    import pandas as pd

    conditions = parse_query(query)
    if not conditions:
        return pd.Series(True, index=df.index)

    if col_lower is None:
        col_lower = df[col].fillna('').str.lower()
    use_or = (str(default_combine).lower() == 'or')

    if use_or:
        # OR 모드: False에서 시작해 누적 |=
        result = pd.Series(False, index=df.index)
        for cond in conditions:
            result |= _eval_condition(col_lower, cond, df.index)
        return result
    else:
        # AND 모드: True에서 시작해 누적 &= (기존과 동일한 메모리 패턴)
        result = pd.Series(True, index=df.index)
        for cond in conditions:
            result &= _eval_condition(col_lower, cond, df.index)
        return result


def _apply_pattern(col_series: pd.Series, pattern: str) -> pd.Series:
    """단일 패턴을 Series에 적용하여 Boolean mask 반환
    col_series: 소문자 변환된 원본 (danbooru: 공백+밑줄, SD: 콤마+공백)
    양쪽 형식 모두 지원: 패턴을 공백/밑줄 양쪽으로 매칭
    """
    pattern = pattern.strip()
    if not pattern:
        import pandas as pd

        return pd.Series(True, index=col_series.index)

    # 태그 경계: 시작/끝 또는 구분자(콤마, 공백)
    SEP_L = r'(?:^|[,\s]\s*)'
    SEP_R = r'(?:\s*[,\s]|$)'

    # *word → 완전 일치
    if pattern.startswith('*'):
        target = pattern[1:].strip()
        pat = SEP_L + _both(target) + SEP_R
        return col_series.str.contains(pat, regex=True, na=False)

    # _word_ → 포함 (명시적) — 두 형태를 한 번의 스캔으로
    if pattern.startswith('_') and pattern.endswith('_') and len(pattern) > 2:
        return contains_tag_text(col_series, pattern[1:-1].strip())

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

    # 기본: 포함 매칭 (양쪽 형식) — 두 형태를 한 번의 스캔으로
    return contains_tag_text(col_series, pattern)
