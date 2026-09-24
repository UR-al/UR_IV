# widgets/queue_panel.py
"""
대기열 상태 저장소 — 화면이 없는 QObject. 보이는 대기열은 Vue(``QueuePanel.vue``)가 그린다.

예전엔 setParent(None) 으로 숨겨 둔 QWidget 이었다. 한 번도 표시되지 않는데도 항목이 바뀔
때마다 카드(QueueItemCard) 전부를 지우고 다시 만들고, 대기열 전체 JSON 을 쓰고, 전체 목록을
Vue 로 보냈다 — XYZ 256칸을 한 칸씩 넣으면 카드 32,896장(O(N²)), 약 20초 정지·RSS 수 GB.
편집 다이얼로그·드래그·프리셋 같은 위젯 입력 경로도 화면이 없어 도달할 수 없었다.

지금은
- 상태: :class:`core.queue_model.QueueModel` (순수 로직 — 실행 중 항목 보호·id 유일성)
- 변경 알림: ``queue_changed(int)`` — 목록을 직렬화하지 않아 싸다. Vue 전송은 받는 쪽
  (``ui.queue_vue_sync``)이 한 이벤트 루프 턴에 한 번으로 합친다.
- 디스크 저장: :data:`PERSIST_DELAY_MS` 동안 모았다가 한 번 쓴다. 앱 종료 때는
  :meth:`QueuePanel.flush_to_disk` 로 즉시 쓴다(데스크톱 ``_quit_app`` · 웹 모드 aboutToQuit).

이름(QueuePanel)과 모듈 경로는 호환을 위해 그대로 둔다 — ``ui.generator_main`` 이 이 이름으로
만들고, 테스트가 ``ui.generator_main.QueuePanel`` 을 patch 한다.
"""
import json
import logging
import os

from PyQt6.QtCore import QObject, pyqtSignal

from core.coalesced_call import CoalescedCall, qt_scheduler
from core.queue_model import QueueModel, auto_restore_enabled, restorable_items, state_document
from utils.atomic_json import atomic_write_json

logger = logging.getLogger(__name__)

# 앱 크래시 후에도 대기열 복구에 쓰는 세션 캐시
from core.storage_paths import cache_file

_QUEUE_STATE_PATH = str(cache_file(
    "session/queue_state.json",
    legacy_paths="config/queue_state.json",
))

#: 대기열 변경을 모아 디스크에 쓰는 간격 — 한 핸들러 안의 일괄 추가(XYZ·이벤트 시나리오)는 한 번만 쓴다
PERSIST_DELAY_MS = 250


class QueuePanel(QObject):
    """대기열 상태 저장소 (화면 없음) — 대기열 매니저와 Vue 동기화의 정본."""

    # 대기열 변경(추가/삭제/순서/수정/실행 중 표시) — 인자는 항목 수. 목록은 싣지 않는다.
    queue_changed = pyqtSignal(int)

    def __init__(self, parent=None, *, state_path=None, prefs_path=None, schedule=None,
                 restore: bool = True):
        super().__init__(parent)
        self._model = QueueModel()
        self._state_path = str(state_path or _QUEUE_STATE_PATH)
        self._prefs_path = str(prefs_path) if prefs_path else None
        self._persist_call = CoalescedCall(
            self._persist_to_disk, schedule=schedule or qt_scheduler(), delay_ms=PERSIST_DELAY_MS,
        )
        if restore:
            self._restore_from_disk()

    # ── 조회 (호환 이름) ──

    @property
    def queue_items(self) -> list:
        """현재 항목 목록(정본). 읽기 전용 — 바꿀 땐 add/remove/update/clear 메서드로."""
        return self._model.items

    @property
    def is_processing(self) -> bool:
        return self._model.is_processing

    @property
    def current_processing_id(self):
        return self._model.processing_id

    @property
    def processing_owner(self):
        """'실행 중' 표시의 주인 — core.queue_model.QUEUE_OWNER · AUTOMATION_OWNER (옛 표시는 None)."""
        return self._model.processing_owner

    def processing_held_by_other(self, owner: str) -> bool:
        """``owner`` 가 아닌 쪽이 지금 대기열 항목을 생성하고 있는가."""
        return self._model.held_by_other(owner)

    def processing_index(self) -> int:
        return self._model.processing_index()

    def count(self) -> int:
        return self._model.count()

    def is_empty(self) -> bool:
        return self._model.is_empty()

    def get_first_item(self):
        return self._model.first()

    def get_item_by_id(self, item_id: str):
        return self._model.get(item_id)

    # ── 변경 알림 · 영속화 ──

    def _changed(self, *, persist: bool = True) -> None:
        if persist:
            self._persist_call.request()
        self.queue_changed.emit(self._model.count())

    def flush_to_disk(self) -> bool:
        """모아 둔 저장을 지금 쓴다(앱 종료 직전). 쓸 것이 있었으면 True."""
        return self._persist_call.flush()

    def _persist_to_disk(self) -> None:
        """현재 대기열을 JSON으로 저장 (공용 atomic_write_json: fsync + 실패 시 tmp 정리).

        직렬화할 수 없는 값(TypeError/ValueError)도 대기열 조작을 끊지 않게 경고만 남긴다.
        """
        try:
            atomic_write_json(self._state_path, state_document(self._model.items), indent=None)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("queue persist failed: %s", e)

    def _read_prefs(self):
        # 경로·로더는 core.ui_prefs 한 곳(없거나 깨졌으면 {} → 자동 복구 켬)
        from core.ui_prefs import read_ui_prefs
        return read_ui_prefs(self._prefs_path)

    def _restore_from_disk(self) -> None:
        """앱 시작 시 미완료 대기열을 복구.

        ui_prefs.json 의 ``queue_auto_restore`` (UI 없음 — 파일을 직접 고쳐 끄는 탈출구, 기본 true):
          - true : 다이얼로그 없이 자동 복구
          - false: 다이얼로그로 확인
        """
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("queue restore read failed: %s", e)
            return
        items = restorable_items(data)
        if not items:
            return

        try:
            auto_restore = auto_restore_enabled(self._read_prefs())
        except Exception:
            auto_restore = True

        if not auto_restore:
            # 사용자가 명시적으로 끈 경우 — 확인 다이얼로그(화면 없는 객체라 부모 없이 띄운다)
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                None, "대기열 복구",
                f"이전 세션의 미완료 대기열 {len(items)}개 항목을 복구할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                try:
                    os.remove(self._state_path)
                except OSError:
                    pass
                return

        restored = self._model.replace(items)
        # 복구 전용 필드 보정(겹친 id 재발급 등)이 있었을 수 있다 — 다음 저장 때 반영된다
        self._changed(persist=False)
        logger.info("대기열 복구: %d개 항목", restored)

    # ── 변경 ──
    # 모든 변경은 O(항목 수) 이하이고 화면을 다시 그리지 않는다. 알림은 즉시(싸다), 저장은 모아서.

    def add_single_item(self, item_data: dict) -> str:
        """항목 하나를 맨 뒤에 넣고 id 를 돌려준다.

        ``ui.generator_main._setup_queue`` 가 인스턴스에서 이 메서드를 감싸 ComfyUI 워크플로 컨트롤
        동결과 queueItemAdded 알림을 붙인다 — 일괄 추가도 반드시 이 메서드를 한 건씩 거친다
        (그 래퍼를 우회하는 일괄 API 를 만들지 않는다).
        """
        item = self._model.add(item_data)
        self._changed()
        return item['id']

    def remove_items_by_ids(self, item_ids: list) -> int:
        """여러 item_id를 한 번에 삭제 — 실행 중 항목은 보호한다. 지운 개수를 돌려준다."""
        removed = self._model.remove_ids(item_ids or ())
        if removed:
            self._changed()
        return removed

    def move_item_up(self, item_id: str) -> bool:
        """한 칸 앞으로. 첫 번째이거나 실행 중 항목과 자리를 바꾸게 되면 False."""
        moved = self._model.move(item_id, 'up')
        if moved:
            self._changed()
        return moved

    def move_item_down(self, item_id: str) -> bool:
        """한 칸 뒤로. 마지막이거나 실행 중 항목과 자리를 바꾸게 되면 False."""
        moved = self._model.move(item_id, 'down')
        if moved:
            self._changed()
        return moved

    def update_item(self, item_id: str, fields: dict) -> bool:
        """항목 필드 수정(Vue 편집 모달). 항목이 있으면 True."""
        found = self._model.update_fields(item_id, fields or {})
        if found:
            self._changed()
        return found

    def clear_items(self) -> int:
        """전체 비우기(Vue 가 이미 확인을 받았다) — 실행 중 항목은 남긴다. 지운 개수."""
        removed = self._model.clear()
        if removed:
            self._changed()
        return removed

    def remove_first_item(self):
        """맨 앞 항목 제거 — 자동화 '큐 우선' 경로가 방금 처리한 항목을 정리할 때 쓴다."""
        removed = self._model.pop_first()
        if removed is not None:
            self._changed()
        return removed

    def consume_item(self, item_id: str):
        """생성이 끝난 **그 항목**을 지운다(대기열 매니저). 이미 없으면 None — 다른 항목은 건드리지 않는다."""
        removed = self._model.consume(item_id)
        if removed is not None:
            self._changed()
        return removed

    # ── 실행 중 표시 ──

    def set_processing(self, is_processing: bool, item_id: str = None, *, owner: str = None):
        """'실행 중' 항목 표시(주인 확인 없음 — 옛 호출용). 실행 중 항목은 삭제·비우기·순서 바꾸기에서 보호된다.

        ``item_id`` 없이 켜면(옛 호출 방식) 맨 앞 항목을 실행 중으로 본다. 표시가 바뀌면 Vue 가
        그 행의 삭제·이동 버튼을 막도록 변경 알림을 보낸다(디스크에는 쓰지 않는다 — 복구된
        항목은 아무도 생성 중이 아니다). 대기열 매니저·자동화는 남의 표시를 지우지 않도록
        :meth:`claim_processing` / :meth:`release_processing` 을 쓴다.
        """
        target = None
        if is_processing:
            target = item_id
            if target is None:
                first = self._model.first()
                target = first.get('id') if first else None
        if self._model.set_processing(target, owner):
            self._changed(persist=False)

    def claim_processing(self, item_id, owner: str) -> bool:
        """``owner`` 가 ``item_id`` 를 생성한다고 표시한다. 다른 주인이 표시를 쥐고 있으면 False.

        대기열 매니저(QUEUE_OWNER)와 자동화 '큐 우선'(AUTOMATION_OWNER)이 같은 대기열을 소비한다 —
        한쪽이 생성 중인 항목을 다른 쪽이 다시 보내거나 그 표시를 풀면, 표시가 사라진 항목을 지운 뒤
        완료 처리가 엉뚱한 항목을 지웠다.
        """
        before = (self._model.processing_id, self._model.processing_owner)
        claimed = self._model.claim(item_id, owner)
        if claimed and before != (self._model.processing_id, self._model.processing_owner):
            self._changed(persist=False)
        return claimed

    def release_processing(self, item_id, owner: str = None) -> bool:
        """``item_id`` 가 아직 '실행 중'이고 주인이 ``owner`` 면 표시를 푼다. 풀었으면 True.

        다른 항목·다른 주인의 표시는 건드리지 않는다(``owner`` 가 None 이면 주인을 따지지 않는다).
        자동화를 멈추면 큐 우선 항목은 소비되지 않고 대기열에 남는다 — 표시가 남으면 그 항목을
        지울 수 없게 된다.
        """
        if not self._model.release(item_id, owner):
            return False
        self._changed(persist=False)
        return True
