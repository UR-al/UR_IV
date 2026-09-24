# core/search_rows.py
"""Search 결과 행 정규화 — 검색 워커·bridge·parquet 가져오기가 공유하는 단일 규칙.

행 계약(Vue SearchView·자동화 덱·디스크 캐시가 모두 이 모양을 쓴다):

    {copyright, character, artist, general, rating: str,
     image_width, image_height: int | None}

- 태그 4종은 ``tag_string_X`` 가 비어 있지 않으면 그것을, 아니면 ``X`` 를 쓴다.
  (검색 shard 는 ``X`` 만 갖지만, 외부/구형 parquet 가져오기는 둘 중 하나를 가질 수 있다.)
- 결측값(None·NaN·pd.NA)은 문자열 ``''`` / 해상도 ``None`` 으로 바꾼다. pandas 3 의
  문자열 dtype 은 결측을 NaN 으로 돌려주므로 ``str()`` 하면 ``'nan'`` 이 새어 나간다.
- 해상도는 양의 정수만 인정한다(없으면 자동 해상도 폴백).

순수 로직 — Qt 의존 없음.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

# 검색 결과 행 상한(무작위 표본). 워커가 이 값으로 자르고, 사용자가 '무제한'을 고르면
# 끈다. 자동화 풀과 같은 리스트라 낮추지 않는다.
SEARCH_RESULT_CAP = 500_000

TAG_FIELDS = ('copyright', 'character', 'artist', 'general')
SEARCH_ROW_KEYS = TAG_FIELDS + ('rating', 'image_width', 'image_height')


class NormalizedSearchRows(list):
    """이미 :func:`normalize_search_rows_in_place` 를 거친 결과 목록 표식.

    검색 워커가 워커 스레드에서 정규화를 끝낸 뒤 이 타입으로 내보내면, bridge 는
    GUI 스레드에서 수십만 행 루프를 다시 돌지 않는다.
    """

    __slots__ = ()


def is_missing(value: Any) -> bool:
    """None / NaN / pd.NA / NaT 판정. 배열 같은 값은 결측이 아닌 것으로 본다."""
    if value is None:
        return True
    if isinstance(value, float):   # numpy.float64 도 float 하위형
        return math.isnan(value)
    if isinstance(value, (str, bytes, int)):
        return False
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except (TypeError, ValueError, ImportError):
        return False


def text_value(value: Any) -> str:
    """결측이면 '', 아니면 str(value)."""
    if is_missing(value):
        return ''
    return value if isinstance(value, str) else str(value)


def dimension_value(value: Any) -> int | None:
    """image_width/height → 양의 int, 아니면 None."""
    if is_missing(value) or value == '':
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _tag_value(row: Mapping, field: str) -> str:
    primary = row.get(f'tag_string_{field}')
    if not is_missing(primary) and primary != '':
        return text_value(primary)
    return text_value(row.get(field))


def normalize_search_row(row: Mapping) -> dict:
    """한 행을 Search 행 계약의 새 dict 로 만든다."""
    return {
        'copyright': _tag_value(row, 'copyright'),
        'character': _tag_value(row, 'character'),
        'artist': _tag_value(row, 'artist'),
        'general': _tag_value(row, 'general'),
        'rating': text_value(row.get('rating')),
        'image_width': dimension_value(row.get('image_width')),
        'image_height': dimension_value(row.get('image_height')),
    }


def normalize_search_rows_in_place(rows: list) -> list:
    """list[dict] 를 제자리에서 정규화한다(같은 list·같은 dict 객체 재사용).

    수십만 행에서 두 번째 list[dict] 를 만들지 않아 피크 메모리가 늘지 않는다.
    dict 가 아닌 항목은 제거한다. 반환값은 인자로 받은 그 list 다.
    """
    write_idx = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized = normalize_search_row(row)
        row.clear()
        row.update(normalized)
        rows[write_idx] = row
        write_idx += 1
    if write_idx < len(rows):
        del rows[write_idx:]
    return rows


def search_rows_from_frame(frame, cap: int | None = None) -> list[dict]:
    """DataFrame(외부 parquet 등) → 정규화된 Search 행 목록.

    ``to_dict('records')`` 는 iterrows 보다 수 배 빠르고, 결측은 정규화가 처리하므로
    dtype 을 바꾸는 fillna 가 필요 없다. ``cap`` 이 있으면 검색 워커와 같은 규칙으로
    (행 dict 를 만들기 전에) 무작위 표본으로 자른다 — 앞부분만 자르면 편향된다.
    """
    if cap is not None and cap > 0 and len(frame) > cap:
        frame = frame.sample(n=cap)
    return normalize_search_rows_in_place(frame.to_dict('records'))


__all__ = [
    'NormalizedSearchRows',
    'SEARCH_RESULT_CAP',
    'SEARCH_ROW_KEYS',
    'TAG_FIELDS',
    'dimension_value',
    'is_missing',
    'normalize_search_row',
    'normalize_search_rows_in_place',
    'search_rows_from_frame',
    'text_value',
]
