# core/search_session.py
"""Search 런타임 스냅숏 상태 — 메인 윈도우가 들고 있는 검색 결과/덱의 단일 규칙.

owner(메인 윈도우) 속성 계약:

- ``filtered_results``          활성(필터 적용) 행 — 자동화 풀, 디스크 active 캐시와 같은 내용
- ``shuffled_prompt_deck``      남은 덱 (호출자가 관리)
- ``_search_dataset_identity``  ``{'label', 'fingerprint'}`` — 결과를 만든 manifest
- ``_search_snapshot_id``       32 hex — 검색/가져오기 한 번의 lineage
- ``_search_base_results``      필터 전 base 행. Vue 의 ``results`` 배열과 **같은 순서**라
                                Vue 는 필터 결과를 행 전체가 아니라 base 인덱스로 보낸다.
- ``_search_base_snapshot_id``  base 가 속한 snapshot (다른 snapshot 의 base 재사용 방지)

base 결정 규칙은 Vue ``resolveSearchCacheRestore`` 와 같다: full 캐시가 있으면 full,
없으면 active. 새 검색·가져오기에서는 결과 목록 그 자체가 base 다. Python 이 기록하는 base 는
``loadFullResults`` 가 Vue 에 돌려준 배열과 같아야 한다(:func:`serve_runtime_base`).

메모리: 재시작 복원 뒤 active 와 full 은 따로 파싱된 두 벌이다. base 를 기록할 때 active 가
base 의 순서 보존 부분열(값 동일)이면 active·덱 행을 base 행 객체로 바꿔 한 벌만 남긴다.

순수 로직 — Qt 의존 없음. 저장소는 ``SearchResultStore`` 인스턴스를 주입받는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_SAFE_SNAPSHOT_ID = re.compile(r"\A[A-Fa-f0-9]{32}\Z")

# update_prompt_deck 거부 사유 — Vue 는 이미 화면 목록을 바꾸고 '필터 적용' 토스트를 띄운
# 뒤라, 자동화 덱이 그대로라는 점을 분명히 알린다(호출자는 알림으로도 보낸다).
LINEAGE_MISMATCH_MESSAGE = (
    '검색 결과 lineage가 현재 결과와 일치하지 않아 '
    '지연된 필터 요청을 무시했습니다(자동화 덱은 그대로입니다).'
)
BASE_MISMATCH_MESSAGE = (
    '검색 결과 기준 목록이 현재 결과와 맞지 않아 자동화 덱에 필터를 적용하지 못했습니다'
    '(화면 목록만 바뀜). 검색을 다시 실행해 주세요.'
)


def runtime_lineage(owner: Any) -> dict | None:
    """owner 의 현재 lineage — update_prompt_deck 페이로드의 ``lineage`` 와 비교하는 형태."""
    identity = getattr(owner, '_search_dataset_identity', None)
    if not isinstance(identity, dict):
        return None
    return {
        'label': identity.get('label'),
        'fingerprint': identity.get('fingerprint'),
        'snapshot_id': getattr(owner, '_search_snapshot_id', None),
    }


def _valid_snapshot_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_SAFE_SNAPSHOT_ID.fullmatch(value))


def runtime_snapshot_is_current(owner: Any, store: Any) -> bool:
    """owner 가 유효한 스냅숏을 들고 있고, 그 데이터셋이 지금 활성 manifest 와 같은가.

    참이면 디스크 캐시를 다시 파싱할 필요가 없다 — 메모리가 디스크와 같거나 더 최신이다
    (디스크 쓰기는 백그라운드 스레드에서 뒤따른다).
    """
    if owner is None or not isinstance(getattr(owner, 'filtered_results', None), list):
        return False
    identity = getattr(owner, '_search_dataset_identity', None)
    if not isinstance(identity, dict):
        return False
    if not _valid_snapshot_id(getattr(owner, '_search_snapshot_id', None)):
        return False
    try:
        current = store.dataset_info()
    except Exception:
        return False
    return (
        isinstance(current, dict)
        and identity.get('label') == current.get('label')
        and identity.get('fingerprint') == current.get('fingerprint')
    )


def publish_snapshot(
    owner: Any,
    *,
    active: list,
    base: list | None,
    identity: dict,
    snapshot_id: str,
) -> None:
    """새 스냅숏을 owner 에 게시한다(덱은 호출자가 채운다)."""
    owner._search_dataset_identity = identity
    owner._search_snapshot_id = snapshot_id
    owner.filtered_results = active
    set_runtime_base(owner, base)


def set_runtime_base(owner: Any, base: list | None) -> None:
    """현재 스냅숏의 base 행을 기록한다(None 이면 '아직 모름')."""
    owner._search_base_results = base
    owner._search_base_snapshot_id = (
        getattr(owner, '_search_snapshot_id', None) if base is not None else None
    )


def runtime_base_rows(owner: Any) -> list | None:
    """현재 스냅숏에 속한 base 행 — 모르거나 다른 스냅숏 것이면 None."""
    base = getattr(owner, '_search_base_results', None)
    if not isinstance(base, list):
        return None
    snapshot_id = getattr(owner, '_search_snapshot_id', None)
    if snapshot_id is None or getattr(owner, '_search_base_snapshot_id', None) != snapshot_id:
        return None
    return base


def restore_runtime_from_disk(owner: Any, store: Any) -> list:
    """디스크 active 캐시 → owner 런타임 상태(filtered_results·덱·lineage). 직렬화는 하지 않는다.

    base 는 여기서 읽지 않는다(시작 시 full 캐시까지 파싱하지 않도록 지연) — 필요할 때
    :func:`load_runtime_base` 가 스냅숏 id 로 full 만 읽는다. 반환값은 active 행.
    """
    parsed = store.load_active()
    snapshot_id = store.last_snapshot_id
    identity = store.last_dataset_identity
    if (
        snapshot_id is None
        or owner is None
        or not hasattr(owner, 'filtered_results')
    ):
        return parsed
    publish_snapshot(
        owner,
        active=parsed,
        base=None,
        identity=identity,
        snapshot_id=snapshot_id,
    )
    # 저장된 덱 진행도 복원('얼마나 뽑았는지' 유지). 실패(파일 없음/풀 크기 변경)면 새 덱 —
    # 다른 재구성 경로와 같이 등급 필터를 적용한다(core.search_deck). 저장은 첫 뽑기 때.
    restore_deck = getattr(owner, '_restore_deck_state', None)
    if not (callable(restore_deck) and restore_deck()):
        from core.search_deck import refill_owner_deck
        refill_owner_deck(owner, parsed, save=False, emit=False)
    return parsed


def load_runtime_base(owner: Any, store: Any) -> list | None:
    """현재 스냅숏의 base 행(Vue '필터 해제' 베이스와 같은 배열). owner 스냅숏이 없으면 None.

    메모리에 있으면 그대로, 없으면 full 캐시를 스냅숏 id 로 한 번만 읽고(active 재파싱 없음)
    기록한다. full 이 없거나 다른 스냅숏이면 active(=filtered_results)가 base 다.
    """
    if not runtime_snapshot_is_current(owner, store):
        return None
    base = runtime_base_rows(owner)
    if base is not None:
        return base
    try:
        full = store.load_full(expected_snapshot_id=owner._search_snapshot_id)
    except Exception as exc:
        # 읽기 실패도 'full 없음'과 같게 — Vue 도 같은 응답(active)을 base 로 삼으므로
        # 두 쪽 base 가 어긋나 인덱스 필터가 거부되는 일이 없다.
        print(f"[Search] full cache unreadable: {exc}")
        full = []
    if getattr(store, 'last_error', None):
        print(f"[Search] full cache ignored: {store.last_error}")
    if full:
        adopt_runtime_base(owner, full)
        return full
    base = list(owner.filtered_results)
    set_runtime_base(owner, base)
    return base


def serve_runtime_base(owner: Any, store: Any) -> list:
    """``loadFullResults`` 응답 — Vue '필터 해제' base 로 쓸 배열을 돌려주고, owner 에도 기록한다.

    owner 스냅숏이 지금 manifest 와 같으면 :func:`load_runtime_base`. 아니면(그 사이 manifest
    가 바뀌었거나 읽기 실패) 디스크 쌍을 Vue 규칙(full → active)으로 읽어 돌려주되, Vue 가 그
    응답으로 정할 base 를 owner 에도 기록한다 — 그래야 이후 인덱스 필터가 base 없음으로 계속
    거부되지 않는다.

    - 응답 행이 owner 스냅숏의 것이면 그 행이 base.
    - 응답이 비면 Vue 는 loadLastSearchResults 응답(= owner.filtered_results)을 base 로 쓴다.
    - 다른 스냅숏의 행이면 기록하지 않는다(인덱스 필터는 base_size/lineage 로 거부된다).
    """
    base = load_runtime_base(owner, store)
    if base is not None:
        return base
    served = store.load_full()
    if not served:
        served = store.load_active()
    if getattr(store, 'last_error', None):
        print(f"[Search] full cache ignored: {store.last_error}")
    _record_served_base(owner, served, getattr(store, 'last_snapshot_id', None))
    return served


def _record_served_base(owner: Any, served: list, served_snapshot_id: Any) -> None:
    if owner is None or not isinstance(getattr(owner, 'filtered_results', None), list):
        return
    snapshot_id = getattr(owner, '_search_snapshot_id', None)
    if not _valid_snapshot_id(snapshot_id):
        return
    if not isinstance(getattr(owner, '_search_dataset_identity', None), dict):
        return
    if served:
        if (
            _valid_snapshot_id(served_snapshot_id)
            and served_snapshot_id.lower() == snapshot_id.lower()
        ):
            adopt_runtime_base(owner, served)
        return
    if owner.filtered_results:
        set_runtime_base(owner, list(owner.filtered_results))


def share_base_rows(rows: list, base: list) -> list | None:
    """rows 가 base 의 순서 보존 부분열(행 값이 같음)이면 base 의 행 객체로 이룬 목록, 아니면 None.

    같은 길이면 내용이 같을 때만 ``base`` 자체를 돌려준다(필터 없음 — 새 검색처럼 active 와
    base 가 한 목록). 비교는 값(==) 으로 해서 증명된 경우에만 바꾸므로, 순서가 다르거나
    base 에 없는 행이 하나라도 있으면 None — 호출자는 두 벌을 그대로 둔다.
    """
    if not isinstance(rows, list) or not isinstance(base, list) or len(rows) > len(base):
        return None
    if len(rows) == len(base):
        return base if rows == base else None
    shared = []
    append = shared.append
    remaining = iter(base)
    for row in rows:
        for candidate in remaining:
            if candidate is row or candidate == row:
                append(candidate)
                break
        else:
            return None
    return shared


def adopt_runtime_base(owner: Any, base: list) -> None:
    """디스크에서 따로 파싱한 base 를 기록하고, active·덱 행을 base 행 객체로 합친다.

    재시작 복원 뒤 active(filtered_results)와 full 은 같은 행을 두 벌 들고 있다(4만 행에 약
    40MB, 50만 cap 근처면 수백 MB). active 가 base 의 부분열이면 위치는 그대로 두고 행 객체만
    base 것으로 바꾼다 — 덱 저장 인덱스(filtered_results 위치)도 그대로 유효하다.
    """
    active = getattr(owner, 'filtered_results', None)
    if isinstance(active, list) and active and active is not base:
        shared = share_base_rows(active, base)
        if shared is not None and any(a is not b for a, b in zip(active, shared)):
            deck = getattr(owner, 'shuffled_prompt_deck', None)
            if isinstance(deck, list) and deck:
                by_id = {id(old): new for old, new in zip(active, shared)}
                owner.shuffled_prompt_deck = [by_id.get(id(row), row) for row in deck]
            owner.filtered_results = shared
            # 풀 목록 객체가 바뀌었으니 덱 저장(id→index 캐시 포함)을 새 목록 기준으로 갱신한다.
            save_deck = getattr(owner, '_save_deck_state', None)
            if callable(save_deck):
                save_deck()
    set_runtime_base(owner, base)


@dataclass(frozen=True)
class DeckUpdate:
    """update_prompt_deck 해석 결과. rows 가 None 이면 덱을 바꾸지 않는다."""

    rows: list | None
    error: str | None = None


def resolve_prompt_deck_update(
    owner: Any,
    payload: Any,
    *,
    store_factory: Callable[[], Any] | None = None,
) -> DeckUpdate:
    """Vue update_prompt_deck 페이로드 → 새 활성 행.

    두 형태를 받는다.
    - ``{indices, base_size, lineage}``: base 행 인덱스(권장 — 수만 행에도 수백 KB).
    - ``{results, lineage}``: 행 전체(하위 호환).
    lineage 가 현재 스냅숏과 다르면(늦게 도착한 옛 필터) 거부한다.

    lineage 는 맞는데 base 를 아직 모르면(loadFullResults 가 base 를 기록하지 못한 경우)
    ``store_factory`` 로 :func:`load_runtime_base` 를 한 번 시도한다 — loadFullResults 와 같은
    규칙이라 같은 배열이 나오고, 그래도 다르면 base_size 검사가 거부한다.
    """
    if not isinstance(payload, dict):
        return DeckUpdate(None)
    has_indices = 'indices' in payload
    results = payload.get('results')
    has_results = 'results' in payload and isinstance(results, list)
    if not has_indices and not has_results:
        return DeckUpdate(None)
    if payload.get('lineage') != runtime_lineage(owner):
        return DeckUpdate(None, LINEAGE_MISMATCH_MESSAGE)
    if not has_indices:
        return DeckUpdate(results)

    base = runtime_base_rows(owner)
    if base is None and store_factory is not None:
        try:
            base = load_runtime_base(owner, store_factory())
        except Exception as exc:
            print(f"[Search] runtime base reload failed: {exc}")
            base = None
    indices = payload.get('indices')
    base_size = payload.get('base_size')
    if (
        base is None
        or type(base_size) is not int
        or base_size != len(base)
        or not isinstance(indices, list)
    ):
        return DeckUpdate(None, BASE_MISMATCH_MESSAGE)
    size = len(base)
    rows = []
    append = rows.append
    for index in indices:
        if type(index) is not int or index < 0 or index >= size:
            return DeckUpdate(None, BASE_MISMATCH_MESSAGE)
        append(base[index])
    return DeckUpdate(rows)


__all__ = [
    'BASE_MISMATCH_MESSAGE',
    'DeckUpdate',
    'LINEAGE_MISMATCH_MESSAGE',
    'adopt_runtime_base',
    'load_runtime_base',
    'publish_snapshot',
    'resolve_prompt_deck_update',
    'restore_runtime_from_disk',
    'runtime_base_rows',
    'runtime_lineage',
    'runtime_snapshot_is_current',
    'serve_runtime_base',
    'set_runtime_base',
    'share_base_rows',
]
