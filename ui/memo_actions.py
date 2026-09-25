"""메모 액션 — Vue ``requestAction('memo_*')`` 의 Python 쪽(공유 메모 계약).

GeneratorMainUI 에 믹스인으로 얹힌다(ui/model_download_actions.py 와 같은 방식). 실제 일은 Qt 를 모르는
core.memo_sync_service.MemoSyncService 가 한다 — 여기는 브리지 배선과 '지금 동기화할 Forge 가 어디인가'
만 정한다.

액션: memo_list {} · memo_save {id, title, text, base_updated_at, editor?, request?} · memo_delete {id} ·
memo_sync {}. ``editor`` = 저장을 보낸 편집기 식별자(같은 낡은 base 의 글을 편집기마다 따로 사본에 모은다),
``request`` = 이 저장의 식별자(답의 ``saved`` 로 돌아온다).
시그널(ui/vue_bridge.py): memoState — JSON {memos:[...], sync:{available, target, syncing,
last_synced_at, error}, saved?:{request, id, conflict_of}} — ``saved`` 는 request 를 붙인 memo_save 의 답에만.
저장 충돌·오류 안내는 기존 showNotification 으로 보낸다.

브리지 계약: tests/test_bridge_contract.py 가 아래 ``action in (...)`` 의 이름을 AST 로 읽는다 —
비교 대상은 리터럴로 둘 것.
"""
from __future__ import annotations

import json
import threading

from PyQt6.QtCore import QObject, Qt, pyqtSignal

_INIT_LOCK = threading.Lock()


class _MemoGuiRelay(QObject):
    """동기화 워커가 'GUI 스레드에서 이것을 불러 달라'고 넘기는 통로(큐 연결).

    memoState 는 GUI 스레드에서 보내는 순간의 상태로 만들어야 순서가 뒤집히지 않는다
    (core.memo_sync_service 모듈 설명 참조).
    """

    call = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(self._run, Qt.ConnectionType.QueuedConnection)

    def _run(self, fn) -> None:
        fn()


class MemoActionsMixin:
    def _memo_dispatcher(self):
        """GUI 스레드로 넘기는 함수. 창(QObject)이 아니면(테스트 호스트) None — 부른 스레드에서 바로 보낸다."""
        if not isinstance(self, QObject):
            return None
        relay = _MemoGuiRelay()
        relay.moveToThread(self.thread())   # 워커에서 처음 불려도 슬롯은 창의 스레드에서 돈다
        self._memo_gui_relay = relay   # 부모 없이 만들었으니 참조를 쥔다
        return relay.call.emit

    def _memo_service(self):
        with _INIT_LOCK:
            service = getattr(self, "_memo_sync_service", None)
            if service is None:
                from core.memo_store import MemoStore
                from core.memo_sync_service import MemoSyncService

                store_factory = getattr(self, "_memo_store_factory", MemoStore)
                service = MemoSyncService(
                    store_factory(),
                    target_provider=self._memo_sync_target,
                    emit=self._memo_emit_state,
                    notify=self._memo_notify,
                    dispatch=self._memo_dispatcher(),
                )
                self._memo_sync_service = service
            return service

    def _memo_emit_state(self, state: dict) -> None:
        signal = getattr(getattr(self, "vue_bridge", None), "memoState", None)
        if signal is None:
            return
        try:
            signal.emit(json.dumps(state, ensure_ascii=False))
        except RuntimeError:
            pass  # 종료 중 QObject 가 이미 사라졌다

    def _memo_notify(self, level: str, message: str) -> None:
        signal = getattr(getattr(self, "vue_bridge", None), "showNotification", None)
        if signal is None:
            return
        try:
            signal.emit(level, message)
        except RuntimeError:
            pass

    def _memo_sync_target(self):
        """지금 활성 백엔드가 Forge/WebUI 면 그 주소의 메모 클라이언트, ComfyUI 면 로컬 전용."""
        from backends import BackendType, get_backend, get_backend_type
        from core.forge_memo_client import ForgeMemoClient
        from core.memo_sync_service import MemoSyncTarget

        if get_backend_type() != BackendType.WEBUI:
            return MemoSyncTarget.local()
        try:
            client = ForgeMemoClient(getattr(get_backend(), "api_url", ""))
        except ValueError:
            return MemoSyncTarget.local()
        return MemoSyncTarget(kind="forge", server=client.base_url, client=client,
                              connected=bool(getattr(self, "_backend_connected", False)))

    def _handle_memo_action(self, action: str, payload: dict) -> bool:
        if action in ("memo_list", "memo_save", "memo_delete", "memo_sync"):
            pass
        else:
            return False
        if getattr(self, "_memo_closed", False):
            return True
        payload = payload if isinstance(payload, dict) else {}
        service = self._memo_service()
        if action == "memo_list":
            service.handle_list()
        elif action == "memo_save":
            service.handle_save(payload)
        elif action == "memo_delete":
            service.handle_delete(payload)
        else:
            service.handle_sync()
        return True

    def _memo_backend_connected(self) -> None:
        """백엔드 연결 성공(on_webui_info_loaded) — Forge 면 곧바로 메모를 맞춘다."""
        if getattr(self, "_memo_closed", False):
            return
        self._memo_service().backend_changed()

    def _shutdown_memo_sync(self) -> None:
        with _INIT_LOCK:
            self._memo_closed = True
            service = getattr(self, "_memo_sync_service", None)
        if service is not None:
            service.shutdown()
