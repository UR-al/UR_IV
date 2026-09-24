# core/queue_model.py
"""대기열 상태 모델 — 순수 로직 (Qt 없음).

화면에 보이는 대기열은 Vue(``QueuePanel.vue``)가 그리고, 정본 상태는 이 모델이 들고 있다.
Qt 쪽 ``widgets.queue_panel.QueuePanel`` 은 이 모델을 감싸 변경 알림(queue_changed)과
디스크 저장(디바운스)만 붙인다. 예전 QueuePanel 은 숨은 QWidget 이라 항목이 바뀔 때마다
보이지 않는 카드 N장을 다시 만들어, XYZ 256칸 추가가 O(N²)(약 20초 정지, RSS 수 GB)였다.

불변식
- 항목 ``id`` 는 모델이 발급하고 대기열 안에서 겹치지 않는다(추가 payload 의 ``id`` 는 무시).
- '실행 중' 항목(``processing_id``)은 사용자 삭제·비우기·순서 바꾸기로 움직이지 않는다 —
  생성이 끝나면 대기열 매니저가 :meth:`QueueModel.consume` 으로 **그 항목**을 지운다.
  (예전엔 보호 가드가 존재하지 않는 속성을 읽어 무력했고, 실행 중 항목을 지우면 완료 시
  '다음' 항목이 생성 없이 지워졌다.)
- 실행 중 항목이 대기열에서 빠지면 '실행 중' 표시도 함께 풀린다(낡은 표시가 삭제를 막지 않게).
- '실행 중' 표시에는 **주인**(``processing_owner``)이 있다 — 대기열 매니저(:data:`QUEUE_OWNER`)와
  자동화 '큐 우선'(:data:`AUTOMATION_OWNER`)이 같은 대기열을 소비한다. 주인이 아닌 쪽은 표시를
  빼앗거나(:meth:`QueueModel.claim`) 풀지(:meth:`QueueModel.release`) 못한다. 예전엔 주인 없는 한
  칸이라, 자동화가 생성 중인 항목 Q 를 대기열 '시작'이 다시 보냈다가 실패하며 표시를 풀었고, Q 를
  지운 뒤 자동화가 맨 앞 항목을 지우는 바람에 생성하지 않은 다음 항목이 사라졌다.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Iterable, Mapping, Optional

#: '실행 중' 표시의 주인 — 대기열 매니저(widgets/queue_manager.py)
QUEUE_OWNER = 'queue'
#: '실행 중' 표시의 주인 — 자동화 '큐 우선' 경로(ui/generator_actions.py)
AUTOMATION_OWNER = 'automation'

#: Vue 로 보낼 때 프롬프트 계열은 이만큼, 나머지 문자열은 _OTHER_TEXT_LIMIT 까지만 자른다
PROMPT_TEXT_LIMIT = 4000
_OTHER_TEXT_LIMIT = 500
_PROMPT_KEYS = ('prompt', 'negative_prompt')
#: queueItemAdded(핀 강조용) payload 의 문자열 길이 제한
ADDED_TEXT_LIMIT = 200

_GROUP_DEFAULTS = (
    ('group_id', ''),
    ('group_index', 1),
    ('group_total', 1),
    ('is_last_of_group', True),
)


def _random_id() -> str:
    return uuid.uuid4().hex[:8]


class QueueModel:
    """대기열 항목 목록 + '실행 중' 항목 표시."""

    def __init__(self, items: Optional[Iterable[Any]] = None, *,
                 id_factory: Optional[Callable[[], str]] = None) -> None:
        self._items: list[dict] = []
        self._by_id: dict[str, dict] = {}
        self._processing_id: Optional[str] = None
        self._processing_owner: Optional[str] = None
        self._id_factory = id_factory or _random_id
        if items is not None:
            self.replace(items)

    # ── 조회 ──────────────────────────────────────────────────────────────
    @property
    def items(self) -> list[dict]:
        """현재 항목 목록(정본 그대로). 읽기 전용으로 쓴다 — 바꿀 땐 모델 메서드로."""
        return self._items

    def snapshot(self) -> list[dict]:
        return [dict(item) for item in self._items]

    def count(self) -> int:
        return len(self._items)

    def is_empty(self) -> bool:
        return not self._items

    def first(self) -> Optional[dict]:
        return self._items[0] if self._items else None

    def get(self, item_id: Any) -> Optional[dict]:
        if item_id is None:
            return None
        try:
            return self._by_id.get(item_id)
        except TypeError:   # 해시할 수 없는 값
            return None

    def index_of(self, item_id: Any) -> int:
        target = self.get(item_id)
        if target is None:
            return -1
        for index, item in enumerate(self._items):
            if item is target:
                return index
        return -1

    @property
    def processing_id(self) -> Optional[str]:
        return self._processing_id

    @property
    def processing_owner(self) -> Optional[str]:
        """'실행 중' 표시를 건 쪽(:data:`QUEUE_OWNER` · :data:`AUTOMATION_OWNER`). 옛 호출이면 None."""
        return self._processing_owner

    @property
    def is_processing(self) -> bool:
        return self._processing_id is not None

    def processing_index(self) -> int:
        return self.index_of(self._processing_id)

    # ── 변경 ──────────────────────────────────────────────────────────────
    def _new_id(self) -> str:
        for _ in range(1000):
            candidate = str(self._id_factory())
            if candidate and candidate not in self._by_id:
                return candidate
        raise RuntimeError('대기열 항목 id 를 만들지 못했습니다')

    def _forget(self, item: dict) -> None:
        item_id = item.get('id')
        self._by_id.pop(item_id, None)
        if item_id is not None and item_id == self._processing_id:
            self._processing_id = None
            self._processing_owner = None

    def add(self, item_data: Mapping[str, Any]) -> dict:
        """항목을 맨 뒤에 넣고 저장된 항목(dict)을 돌려준다.

        그룹 필드 기본값은 '단독 항목'이며 ``item_data`` 가 주면 그 값을 따른다.
        ``id`` 는 늘 새로 발급한다 — 복제·재등록 payload 에 옛 id 가 있어도 겹치지 않게.
        """
        if not isinstance(item_data, Mapping):
            raise TypeError('대기열 항목은 dict 여야 합니다')
        item: dict = {'id': self._new_id()}
        for key, default in _GROUP_DEFAULTS:
            item[key] = default
        for key, value in item_data.items():
            if key != 'id':
                item[key] = value
        if not item.get('group_id'):
            item['group_id'] = ''
        self._items.append(item)
        self._by_id[item['id']] = item
        return item

    def remove_ids(self, item_ids: Iterable[Any]) -> int:
        """여러 항목 삭제 — 실행 중 항목은 보호. 지운 개수를 돌려준다."""
        targets = {item_id for item_id in (item_ids or ())
                   if isinstance(item_id, str) and item_id in self._by_id}
        targets.discard(self._processing_id)
        if not targets:
            return 0
        kept: list[dict] = []
        removed = 0
        for item in self._items:
            if item.get('id') in targets:
                self._forget(item)
                removed += 1
            else:
                kept.append(item)
        self._items = kept
        return removed

    def move(self, item_id: Any, direction: Any) -> bool:
        """한 칸 위(``'up'``/-1)나 아래(``'down'``/+1)로. 실행 중 항목과는 자리를 바꾸지 않는다."""
        step = -1 if direction in ('up', -1) else 1 if direction in ('down', 1) else 0
        if step == 0:
            return False
        index = self.index_of(item_id)
        target = index + step
        if index < 0 or target < 0 or target >= len(self._items):
            return False
        running = self._processing_id
        if running is not None and running in (self._items[index].get('id'), self._items[target].get('id')):
            return False
        self._items[index], self._items[target] = self._items[target], self._items[index]
        return True

    def pop_first(self) -> Optional[dict]:
        """맨 앞 항목을 꺼낸다(자동화 '큐 우선' 경로가 처리한 항목 정리용)."""
        if not self._items:
            return None
        item = self._items.pop(0)
        self._forget(item)
        return item

    def consume(self, item_id: Any) -> Optional[dict]:
        """생성이 끝난 항목을 지운다 — 실행 중 보호와 무관하게 **그 항목만**. 없으면 None."""
        index = self.index_of(item_id)
        if index < 0:
            return None
        item = self._items.pop(index)
        self._forget(item)
        return item

    def update_fields(self, item_id: Any, fields: Mapping[str, Any]) -> bool:
        """항목 필드 수정(``id`` 는 바꾸지 않는다). 항목이 있으면 True."""
        item = self.get(item_id)
        if item is None:
            return False
        for key, value in (fields or {}).items():
            if key != 'id':
                item[key] = value
        return True

    def clear(self) -> int:
        """전체 비우기 — 실행 중 항목은 남긴다. 지운 개수를 돌려준다."""
        running = self.get(self._processing_id)
        removed = len(self._items) - (1 if running is not None else 0)
        self._items = [running] if running is not None else []
        self._by_id = {running['id']: running} if running is not None else {}
        return removed

    def replace(self, items: Iterable[Any]) -> int:
        """복구용 — 목록을 통째로 바꾼다. dict 만 받고, id 가 없거나 겹치면 새로 발급한다.

        '실행 중' 표시는 풀린다(복구된 항목은 아직 아무도 생성하지 않는다).
        """
        self._items = []
        self._by_id = {}
        self._processing_id = None
        self._processing_owner = None
        for raw in items or ():
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            item_id = item.get('id')
            if not isinstance(item_id, str) or not item_id or item_id in self._by_id:
                item['id'] = self._new_id()
            self._items.append(item)
            self._by_id[item['id']] = item
        return len(self._items)

    def set_processing(self, item_id: Optional[Any], owner: Optional[str] = None) -> bool:
        """'실행 중' 항목을 주인 확인 없이 바꾼다(None=해제). 바뀌었으면 True.

        옛 호출·복구용이다. 대기열 매니저와 자동화는 :meth:`claim` / :meth:`release` 로 자기 표시만
        다룬다 — 이 메서드로 남의 표시를 지우면 그 항목을 지울 수 있게 되어 소비가 어긋난다.
        """
        value = None if item_id is None else item_id
        new_owner = owner if value is not None else None
        if value == self._processing_id and new_owner == self._processing_owner:
            return False
        self._processing_id = value
        self._processing_owner = new_owner
        return True

    def claim(self, item_id: Any, owner: str) -> bool:
        """``owner`` 가 ``item_id`` 를 '실행 중'으로 표시한다. 표시를 쥐었으면 True.

        - 대기열에 없는 항목은 표시하지 않는다(False).
        - 다른 주인의 표시가 있으면 빼앗지 않는다(False) — 그쪽이 지금 그 항목(또는 다른 항목)을
          생성하고 있다. 같은 주인이면 대상을 바꾼다(다음 항목으로 넘어감).
        - 주인 없는 옛 표시(``set_processing`` 으로 건 것)는 넘겨받는다.
        """
        if not owner or self.get(item_id) is None:
            return False
        if self._processing_id is not None and self._processing_owner not in (None, owner):
            return False
        self._processing_id = item_id
        self._processing_owner = owner
        return True

    def release(self, item_id: Any, owner: Optional[str] = None) -> bool:
        """``item_id`` 의 '실행 중' 표시를 푼다 — 그 항목이 표시돼 있고 주인이 맞을 때만. 풀었으면 True.

        ``owner`` 가 None 이면 주인을 따지지 않는다(옛 호출). 다른 항목·다른 주인의 표시는 그대로 둔다.
        """
        if item_id is None or self._processing_id != item_id:
            return False
        if owner is not None and self._processing_owner not in (None, owner):
            return False
        self._processing_id = None
        self._processing_owner = None
        return True

    def held_by_other(self, owner: str) -> bool:
        """``owner`` 가 아닌 쪽이 '실행 중' 표시를 쥐고 있는가(주인 없는 옛 표시는 아니다)."""
        return (self._processing_id is not None and self._processing_owner is not None
                and self._processing_owner != owner)


# ── 영속화 · Vue 직렬화 헬퍼 ────────────────────────────────────────────────

def restorable_items(document: Any) -> list[dict]:
    """세션 캐시(``{"items": [...]}``)에서 복구할 항목 — dict 가 아니면 버린다."""
    if not isinstance(document, Mapping):
        return []
    items = document.get('items')
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def state_document(items: Iterable[Mapping[str, Any]]) -> dict:
    return {'items': list(items)}


def auto_restore_enabled(prefs: Any) -> bool:
    """ui_prefs.json 의 ``queue_auto_restore`` (UI 없음 — 파일을 직접 고쳐 끄는 탈출구, 기본 켬)."""
    if not isinstance(prefs, Mapping):
        return True
    return bool(prefs.get('queue_auto_restore', True))


def _vue_safe_item(item: Mapping[str, Any]) -> dict:
    out: dict = {}
    for key, value in item.items():
        if isinstance(value, str):
            out[key] = value[:PROMPT_TEXT_LIMIT] if key in _PROMPT_KEYS else value[:_OTHER_TEXT_LIMIT]
        elif isinstance(value, (bool, int, float)):
            out[key] = value
    return out


def vue_queue_state(items: Iterable[Mapping[str, Any]], *, running: bool, paused: bool,
                    completed: int, processing_index: int = -1, automating: bool = False) -> dict:
    """``queueUpdated`` payload — 원시 타입만 통과시키고 긴 문자열은 자른다.

    ``current_index`` 는 예전 의미 그대로(대기열 매니저가 돌면 0번이 현재 항목).
    ``processing_index`` 는 실제로 생성 중인 항목(자동화 '큐 우선' 포함) — 없으면 -1.
    Vue 는 이 항목의 삭제·순서 바꾸기를 막는다(백엔드도 막는다).
    ``automation`` 은 자동화가 돌고 있는가 — 자동화가 대기열을 먼저 처리하므로 그동안 대기열
    '시작'은 거절된다(Vue 는 버튼을 끈다. 이후 변화는 automationStatus 로 따라간다).
    """
    safe_items = [_vue_safe_item(item) for item in items]
    running = bool(running)
    processing_index = int(processing_index) if 0 <= int(processing_index) < len(safe_items) else -1
    return {
        'items': safe_items,
        'running': running,
        'paused': bool(paused),
        'current_index': 0 if (running and safe_items) else -1,
        'completed': int(completed or 0),
        'processing_index': processing_index,
        'automation': bool(automating),
    }


def queue_item_added_payload(item: Mapping[str, Any]) -> dict:
    """``queueItemAdded`` payload(Vue 는 핀 강조에만 쓴다) — 원시 타입만, 문자열로 짧게."""
    return {key: str(value)[:ADDED_TEXT_LIMIT] for key, value in item.items()
            if isinstance(value, (str, int, float, bool))}


def periodic_unload_due(generated: int, every_n: int, *, remaining: int) -> bool:
    """대기열 '정기 정리' — N장마다 체크포인트를 VRAM 에서 내릴 차례인가.

    남은 항목이 없으면(이번이 마지막 장) 내리지 않는다: 이어서 만들 장이 없는데 내리면 다음
    수동 생성이 다시 올리는 시간만 는다(끝난 뒤 언로드는 '생성 후 모델 언로드' 설정의 몫).
    """
    try:
        every = int(every_n)
        count = int(generated)
    except (TypeError, ValueError):
        return False
    if every <= 0 or count <= 0 or remaining <= 0:
        return False
    return count % every == 0


__all__ = [
    'ADDED_TEXT_LIMIT',
    'AUTOMATION_OWNER',
    'PROMPT_TEXT_LIMIT',
    'QUEUE_OWNER',
    'QueueModel',
    'auto_restore_enabled',
    'periodic_unload_due',
    'queue_item_added_payload',
    'restorable_items',
    'state_document',
    'vue_queue_state',
]
