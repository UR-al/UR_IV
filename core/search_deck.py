# core/search_deck.py
"""자동화 프롬프트 덱 — 등급(rating) 필터 · 셔플 · 재구성의 단일 규칙 (순수 로직, Qt 무관).

덱(``shuffled_prompt_deck``)은 풀(``filtered_results``)에서 **등급 필터를 통과한 행**을 섞은
목록이고, 자동화·랜덤 프롬프트가 뒤에서 하나씩 pop 한다. 예전엔 이 재구성이 9곳에 복제돼
새 검색·가져오기·자동화 시작·시작 복원은 등급 필터를 빼먹고, 필터 변경은 매번 덱을 새로 섞어
진행도를 날렸다. 이제 모든 경로가 여기 함수(:func:`refill_owner_deck` ·
:func:`set_owner_rating_filter`)를 쓴다.

- :func:`rating_ok` — 등급이 비었거나 'nan'(모름)이면 통과. 'general' 같은 전체 이름도 받는다.
- :func:`build_deck` — 풀 전체로 새 덱(필터 → 셔플).
- :func:`refilter_deck` — 필터만 바뀔 때: 남은 덱에서 새 필터에 안 맞는 행을 빼고, 새로 허용된
  등급의 행만 풀에서 더한다 — 이미 뽑은 진행도는 유지된다.
- :class:`DeckSaveThrottle` — 뽑을 때마다 남은 덱 전체(수만 행 인덱스)를 다시 쓰지 않도록
  저장 시점을 모은다(N회마다 / T초마다 / 덱이 비면). 멈춤·종료 때는 호출자가 flush 한다.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Iterable, Optional

ALL_RATINGS = frozenset({'g', 's', 'q', 'e'})
_RATING_ALIASES = {
    'general': 'g', 'sensitive': 's', 'questionable': 'q', 'explicit': 'e',
}

NO_ELIGIBLE_MESSAGE = (
    '등급 필터를 통과하는 검색 결과가 없습니다 — 등급 필터를 넓히거나 다시 검색하세요.'
)


def normalize_ratings(ratings: Optional[Iterable[Any]]) -> frozenset:
    """등급 필터 값 → ``{'g','s','q','e'}`` 의 부분집합. None 이면 전체(필터 없음)."""
    if ratings is None:
        return ALL_RATINGS
    out = set()
    for value in ratings:
        key = _norm_rating(value)
        if key in ALL_RATINGS:
            out.add(key)
    return frozenset(out)


def _norm_rating(value: Any) -> str:
    text = str(value if value is not None else '').strip().lower()
    if text in ('', 'nan', 'none', 'null'):
        return ''
    return _RATING_ALIASES.get(text, text)


def row_rating(row: Any) -> str:
    """행의 등급 — 모르면 ''."""
    if not isinstance(row, dict):
        return ''
    return _norm_rating(row.get('rating'))


def rating_ok(row: Any, ratings: Iterable[str]) -> bool:
    """등급 필터 통과 여부. 등급을 모르는 행(빈 값·'nan')은 거를 근거가 없어 통과시킨다."""
    rating = row_rating(row)
    return not rating or rating in ratings


def eligible_rows(pool: Optional[list], ratings: Iterable[str]) -> list:
    allowed = frozenset(ratings)
    return [row for row in (pool or []) if rating_ok(row, allowed)]


def build_deck(pool: Optional[list], ratings: Iterable[str], rng: Optional[random.Random] = None) -> list:
    """풀 전체로 새 덱 — 등급 필터 통과 행을 섞은 새 목록(풀은 건드리지 않는다)."""
    deck = eligible_rows(pool, ratings)
    (rng or random).shuffle(deck)
    return deck


def refilter_deck(
    deck: Optional[list],
    pool: Optional[list],
    old_ratings: Iterable[str],
    new_ratings: Iterable[str],
    rng: Optional[random.Random] = None,
) -> list:
    """등급 필터만 바뀔 때의 새 덱 — 진행도(이미 뽑은 행)를 유지한다.

    - 같은 필터면 남은 덱을 그대로(같은 객체) 돌려준다 — 다시 섞지 않는다.
    - 좁아지면: 남은 덱에서 새 필터에 안 맞는 행만 뺀다(순서 유지).
    - 넓어지면: 새로 허용된 등급(예전 필터엔 안 맞고 새 필터엔 맞는)의 풀 행을 섞어 넣는다.
    """
    old = frozenset(old_ratings)
    new = frozenset(new_ratings)
    current = list(deck or [])
    if old == new:
        return deck if isinstance(deck, list) else current
    kept = [row for row in current if rating_ok(row, new)]
    added = [row for row in (pool or []) if rating_ok(row, new) and not rating_ok(row, old)]
    if not added:
        return kept
    merged = kept + added
    (rng or random).shuffle(merged)
    return merged


def prompt_bundle_from_row(row: Any) -> dict:
    """검색 결과 한 행(Vue apply_search_result·add_search_to_queue 페이로드) → apply_prompt_from_data 번들.

    덱에서 뽑을 때와 같은 행 전체(태그 4종 + rating + image_width/height)를 넘긴다 — 예전엔
    태그 4종만 골라 넘겨 '자동 해상도(Parquet H/W)'가 검색 적용 경로에서만 빠졌다.
    'nan' 문자열(외부 parquet 누출)은 빈 값으로.
    """
    from core.search_rows import TAG_FIELDS, normalize_search_row
    bundle = normalize_search_row(row if isinstance(row, dict) else {})
    for key in TAG_FIELDS:
        if bundle[key].strip().lower() == 'nan':
            bundle[key] = ''
    return bundle


# ── owner(메인 윈도우) 헬퍼 — getattr 로만 닿아 테스트 대역에도 그대로 쓴다 ─────────────
# owner 속성 계약: filtered_results(풀) · shuffled_prompt_deck(덱) · _rating_filter(등급 set)
# · _deck_built_ratings(덱을 만든 등급 — 필터 변경 시 '무엇이 새로 허용됐나' 판정)
# · 선택: _save_deck_state() · _emit_auto_status()

def owner_ratings(owner: Any) -> frozenset:
    return normalize_ratings(getattr(owner, '_rating_filter', None))


def _call_optional(owner: Any, name: str) -> None:
    fn = getattr(owner, name, None)
    if not callable(fn):
        return
    try:
        if name == '_emit_auto_status' and callable(getattr(owner, '_auto_is_waiting', None)):
            # 자동화 대기 중이면 카운트다운 표시가 끊기지 않게 대기 상태를 그대로 싣는다
            fn(waiting=bool(owner._auto_is_waiting()))
        else:
            fn()
    except Exception as exc:   # 저장·상태 알림 실패가 덱 갱신을 되돌리지 않게
        print(f"[Deck] {name} 실패: {exc}")


def refill_owner_deck(owner: Any, pool: Optional[list] = None, *, save: bool = True,
                      emit: bool = True, rng: Optional[random.Random] = None) -> list:
    """덱 재구성의 단일 경로 — 풀(기본 owner.filtered_results) → 등급 필터 → 셔플 → 저장 → 상태.

    새 검색 · 가져오기 · Vue 필터 · 덱 초기화 · 소진 후 리필 · 자동화 시작 · 시작 복원이 모두
    이 함수를 부른다. 반환: 새 덱(비어 있으면 등급 필터를 통과한 행이 없다는 뜻).
    """
    source = getattr(owner, 'filtered_results', None) if pool is None else pool
    ratings = owner_ratings(owner)
    deck = build_deck(source, ratings, rng)
    owner.shuffled_prompt_deck = deck
    owner._deck_built_ratings = ratings
    if save:
        _call_optional(owner, '_save_deck_state')
    if emit:
        _call_optional(owner, '_emit_auto_status')
    return deck


def set_owner_rating_filter(owner: Any, ratings: Optional[Iterable[Any]],
                            rng: Optional[random.Random] = None) -> bool:
    """Vue set_rating_filter — 필터를 바꾸고 덱을 진행도를 지키며 맞춘다.

    같은 필터가 다시 오면(Vue 마운트·웹 탭마다 보낸다) 아무것도 하지 않는다 — 예전엔 매번
    덱을 새로 섞어 '얼마나 뽑았는지'가 초기화됐다. 바뀌었으면 :func:`refilter_deck` 후 저장·알림.
    반환: 덱을 바꿨으면 True.
    """
    new = normalize_ratings(ALL_RATINGS if ratings is None else ratings)
    built = getattr(owner, '_deck_built_ratings', None)
    old = built if built is not None else owner_ratings(owner)
    owner._rating_filter = set(new)
    pool = getattr(owner, 'filtered_results', None)
    deck = getattr(owner, 'shuffled_prompt_deck', None)
    if not pool:
        owner._deck_built_ratings = new
        return False
    if built is None:
        # 어떤 필터로 만든 덱인지 모른다(옛 덱 파일) — 새 필터에 안 맞는 행만 걸러 둔다.
        new_deck = [row for row in (deck or []) if rating_ok(row, new)]
        changed = new_deck != list(deck or [])
    elif new == old:
        return False
    else:
        new_deck = refilter_deck(deck, pool, old, new, rng)
        changed = True
    owner.shuffled_prompt_deck = new_deck
    owner._deck_built_ratings = new
    if changed:
        _call_optional(owner, '_save_deck_state')
        _call_optional(owner, '_emit_auto_status')
    return changed


def deck_pool_size(owner: Any) -> int:
    """덱 진행 표시의 '전체' — 풀 중 지금 등급 필터를 통과하는 행 수(필터 전 풀 크기가 아니다).

    풀 목록 객체·길이·필터가 같으면 캐시를 쓴다(대기 중 100ms 마다 불린다).
    """
    pool = getattr(owner, 'filtered_results', None) or []
    ratings = owner_ratings(owner)
    cache = getattr(owner, '_deck_pool_size_cache', None)
    if cache and cache[0] is pool and cache[1] == len(pool) and cache[2] == ratings:
        return cache[3]
    size = len(eligible_rows(pool, ratings))
    try:
        owner._deck_pool_size_cache = (pool, len(pool), ratings, size)
    except Exception:
        pass
    return size


class DeckSaveThrottle:
    """덱 진행도 저장 시점 결정 — 뽑을 때마다가 아니라 모아서 저장한다.

    ``note_draw`` 가 True 를 돌려주면 호출자가 저장하고 ``mark_saved`` 를 부른다.
    저장 조건: 마지막 저장 뒤 ``every`` 번 뽑았거나, ``interval`` 초가 지났거나, 덱이 비었을 때.
    멈춤·종료처럼 확실히 남겨야 할 때는 ``dirty`` 를 보고 즉시 저장(flush)한다.
    """

    def __init__(self, every: int = 20, interval: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.every = max(1, int(every))
        self.interval = float(interval)
        self._clock = clock
        self._pending = 0
        self._last = clock()

    @property
    def dirty(self) -> bool:
        return self._pending > 0

    def note_draw(self, remaining: int) -> bool:
        self._pending += 1
        return (
            self._pending >= self.every
            or remaining <= 0
            or (self._clock() - self._last) >= self.interval
        )

    def mark_saved(self) -> None:
        self._pending = 0
        self._last = self._clock()


__all__ = [
    'ALL_RATINGS',
    'DeckSaveThrottle',
    'NO_ELIGIBLE_MESSAGE',
    'build_deck',
    'deck_pool_size',
    'eligible_rows',
    'normalize_ratings',
    'owner_ratings',
    'rating_ok',
    'refill_owner_deck',
    'refilter_deck',
    'row_rating',
    'set_owner_rating_filter',
]
