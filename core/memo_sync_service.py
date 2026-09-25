"""메모 액션 처리 + 백그라운드 동기화 조율 — Qt 를 모르는 순수 모듈(ui/memo_actions.py 가 얹는다).

- 저장·삭제는 호출한 스레드(GUI)에서 로컬 파일에 바로 쓰고 memoState 를 보낸다(오프라인에서도 동작).
- 네트워크 동기화는 늘 별도 스레드에서 한 번에 하나만 돈다. 도는 중에 또 요청되면 끝난 뒤 한 번 더
  돈다(요청을 합친다). 저장 뒤 동기화는 ``debounce_s`` 만큼 모았다가 한 번 — 타자 칠 때마다
  요청하지 않게.
- 동기화 대상은 ``target_provider()`` 가 매번 정한다: ComfyUI 면 로컬 전용(available=false),
  Forge/WebUI 인데 아직 연결 전이면 자동 동기화는 건너뛴다(명시적 memo_sync 는 시도한다).
- memoState 는 늘 GUI 스레드에서, 보내는 그 순간의 상태로 만든다. 워커는 ``dispatch`` 로 'GUI 스레드에서
  보내 달라'고만 한다 — 워커가 만든 payload 를 큐로 넘기면 그 사이 처리된 저장의 더 새 memoState 보다
  늦게 도착해 화면이 옛 목록으로 돌아간다(편집 중인 초안이 옛 글로 바뀌거나 방금 만든 메모가 닫힌다).

memoState 모양(공유 메모 계약)::

    {"memos": [{id,title,text,created_at,updated_at}],
     "sync": {"available": bool, "target": "forge"|"local", "syncing": bool,
              "last_synced_at": str|null, "error": str|null},
     "saved": {"request": str, "id": str, "conflict_of": str}}   # memo_save 의 답에만

``saved`` — 그 저장(화면이 붙인 ``request``)이 어느 메모에 들어갔나. 충돌 사본이 됐으면 id 가 사본, conflict_of
가 원래 메모다. 화면은 이것으로만 초안을 사본으로 옮긴다 — 본문이 같은 남의 사본을 제 것으로 오인하지 않게.
"""
from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional

from core.forge_memo_client import ForgeMemoError, MemoRoutesUnavailable
from core.memo_store import MemoStore, MemoStoreUnavailableError, MemoValidationError
from core.memo_sync import run_memo_sync

LOCAL_ONLY_MESSAGE = "ComfyUI 에 연결된 동안에는 메모를 이 PC 에만 저장합니다"
UNEXPECTED_MESSAGE = "메모 동기화 중 예상하지 못한 오류가 났습니다 — 메모는 이 PC 에 저장돼 있습니다"
CONFLICT_MESSAGE = "다른 곳에서 먼저 바뀐 메모가 있어 내 편집을 '(충돌 사본)' 으로 따로 저장했습니다"
DEFAULT_DEBOUNCE_S = 2.0


@dataclass(frozen=True)
class MemoSyncTarget:
    """동기화 대상. kind='forge' 면 client·server 가 있다. connected=False 면 자동 동기화를 건너뛴다."""

    kind: str = "local"
    server: str = ""
    client: Any = None
    connected: bool = False

    @classmethod
    def local(cls) -> "MemoSyncTarget":
        return cls()


Emit = Callable[[dict], None]
Notify = Callable[[str, str], None]
# fn 을 GUI 스레드(브리지 시그널의 주인)에서 나중에 부른다. None 이면 부른 스레드에서 바로(테스트·Qt 없음).
Dispatch = Callable[[Callable[[], None]], None]


def _start_daemon(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True, name="memo-sync").start()


def _daemon_timer(delay: float, fn: Callable[[], None]):
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    return timer


class MemoSyncService:
    def __init__(self, store: MemoStore, *, target_provider: Callable[[], Optional[MemoSyncTarget]],
                 emit: Emit, notify: Optional[Notify] = None,
                 debounce_s: float = DEFAULT_DEBOUNCE_S,
                 timer_factory: Callable[[float, Callable[[], None]], Any] = _daemon_timer,
                 start_thread: Callable[[Callable[[], None]], None] = _start_daemon,
                 dispatch: Optional[Dispatch] = None) -> None:
        self._store = store
        self._target_provider = target_provider
        self._emit_fn = emit
        self._notify_fn = notify
        self._dispatch = dispatch
        self._emit_pending = False
        self._debounce_s = max(0.0, float(debounce_s))
        self._timer_factory = timer_factory
        self._start_thread = start_thread
        self._lock = threading.Lock()
        self._state = {"available": False, "target": "local", "syncing": False, "error": None}
        self._server = ""
        self._timer = None
        self._running = False
        self._rerun = False
        self._explicit = False
        self._closed = False

    # ── 상태 ──
    def payload(self, saved: Optional[dict] = None) -> dict:
        with self._lock:
            sync = dict(self._state)
            server = self._server
        try:
            memos = self._store.list_public()
            last = self._store.last_synced_at(server) if server else None
        except MemoStoreUnavailableError as exc:
            memos, last = [], None
            sync["error"] = str(exc)
        sync["last_synced_at"] = last
        state = {"memos": memos, "sync": sync}
        if saved is not None:
            state["saved"] = saved
        return state

    def emit_state(self, saved: Optional[dict] = None) -> None:
        """지금 상태로 memoState 를 보낸다 — GUI 스레드(액션 처리)에서 부른다. 워커는 _emit_from_worker."""
        with self._lock:
            if self._closed:
                return
        try:
            self._emit_fn(self.payload(saved))
        except Exception as exc:   # 브리지가 닫히는 중 등 — 동기화를 멈추지 않는다
            print(f"[Memo] memoState 전송 실패(무시): {exc}")

    def _emit_from_worker(self) -> None:
        """워커 스레드의 상태 알림 — GUI 스레드에서 보내는 순간의 상태로 만들게 넘긴다.

        이미 대기 중인 알림이 있으면 합친다: 그것이 보낼 때 가장 새 상태를 읽는다.
        """
        if self._dispatch is None:
            self.emit_state()
            return
        with self._lock:
            if self._closed or self._emit_pending:
                return
            self._emit_pending = True
        try:
            self._dispatch(self._emit_dispatched)
        except Exception as exc:   # 종료 중(릴레이 객체가 사라짐) 등
            with self._lock:
                self._emit_pending = False
            print(f"[Memo] memoState 예약 실패(무시): {exc}")

    def _emit_dispatched(self) -> None:
        with self._lock:
            self._emit_pending = False   # 보내기 전에 푼다 — 이 뒤의 변경은 새 알림을 예약한다
        self.emit_state()

    def _notify(self, level: str, message: str) -> None:
        if self._notify_fn is None:
            return
        try:
            self._notify_fn(level, message)
        except Exception:
            pass

    def _set_state(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)

    # ── 액션 ──
    def handle_list(self) -> None:
        self.emit_state()
        self.request_sync()

    def handle_save(self, payload: dict) -> None:
        payload = payload if isinstance(payload, dict) else {}
        try:
            result = self._store.save(payload.get("id"), payload.get("title"), payload.get("text"),
                                      payload.get("base_updated_at"), editor=payload.get("editor"))
        except (MemoValidationError, MemoStoreUnavailableError) as exc:
            self._notify("error", str(exc))
            self.emit_state()
            return
        except Exception as exc:
            # 디스크 오류(OSError)든 예상 밖이든 알리고 지금 상태(저장소가 되돌린 판)를 보낸다 — 화면은
            # 확인이 없으니 같은 저장을 다시 보낸다. 브리지(onAction)는 예외를 찍기만 해서 조용히 사라진다.
            if not isinstance(exc, OSError):
                traceback.print_exc()
            self._notify("error", f"메모를 저장하지 못했습니다({exc.__class__.__name__})")
            self.emit_state()
            return
        if result.conflict and result.changed:   # 같은 저장을 다시 보낸 것이면 다시 알리지 않는다
            self._notify("info", CONFLICT_MESSAGE)
        request = payload.get("request")
        saved = None
        if isinstance(request, str) and 0 < len(request) <= 80:
            saved = {"request": request, "id": result.memo["id"], "conflict_of": result.conflict_of}
        self.emit_state(saved)
        if result.changed:
            self.request_sync(delay=self._debounce_s)

    def handle_delete(self, payload: dict) -> None:
        payload = payload if isinstance(payload, dict) else {}
        try:
            removed = self._store.delete(payload.get("id"))
        except MemoStoreUnavailableError as exc:
            self._notify("error", str(exc))
            removed = None
        except Exception as exc:
            if not isinstance(exc, OSError):
                traceback.print_exc()
            self._notify("error", f"메모를 지우지 못했습니다({exc.__class__.__name__})")
            removed = None
        self.emit_state()
        if removed is not None:
            self.request_sync(delay=self._debounce_s)

    def handle_sync(self) -> None:
        self.request_sync(explicit=True)

    def backend_changed(self) -> None:
        """백엔드 연결(또는 전환) 직후 — 대상이 바뀌었을 수 있으니 상태를 비우고 다시 맞춘다."""
        self._set_state(available=False, target="local", error=None)
        self.request_sync()

    # ── 스케줄 ──
    def request_sync(self, *, explicit: bool = False, delay: float = 0.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._explicit = self._explicit or explicit
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if delay > 0:
                timer = self._timer_factory(delay, self._kick)
                self._timer = timer
                timer.start()
                return
        self._kick()

    def _kick(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._timer = None
            if self._running:
                self._rerun = True
                return
            self._running = True
        try:
            self._start_thread(self._worker)
        except Exception:
            with self._lock:
                self._running = False
            raise

    def _worker(self) -> None:
        while True:
            with self._lock:
                explicit, self._explicit = self._explicit, False
                self._rerun = False
            try:
                self._run_once(explicit)
            except Exception:
                traceback.print_exc()
                self._set_state(syncing=False, available=False, target="local", error=UNEXPECTED_MESSAGE)
                self._emit_from_worker()
            with self._lock:
                if self._rerun and not self._closed:
                    continue
                self._running = False
                return

    def _run_once(self, explicit: bool) -> None:
        target = self._target_provider()
        if target is None or target.kind != "forge" or target.client is None:
            self._set_state(available=False, target="local",
                            error=LOCAL_ONLY_MESSAGE if explicit else None)
            self._emit_from_worker()
            return
        with self._lock:
            self._server = target.server
        if not target.connected and not explicit:
            # 아직 연결 전 — 자동 동기화는 조용히 건너뛴다(연결되면 backend_changed 가 다시 부른다).
            self._set_state(available=False, target="local", error=None)
            self._emit_from_worker()
            return
        self._set_state(syncing=True)
        self._emit_from_worker()
        try:
            report = run_memo_sync(self._store, target.client, target.server)
        except MemoRoutesUnavailable as exc:
            self._set_state(syncing=False, available=False, target="local",
                            error=str(exc) if explicit else None)
        except (ForgeMemoError, MemoStoreUnavailableError) as exc:
            self._set_state(syncing=False, available=False, target="local", error=str(exc))
        except OSError as exc:
            self._set_state(syncing=False, available=False, target="local",
                            error=f"메모 파일을 쓰지 못했습니다({exc.__class__.__name__})")
        else:
            error = None
            if report.rejected:
                error = (f"메모 {len(report.rejected)}개를 Forge 가 받지 않아 이 PC 에만 있습니다 — "
                         f"{report.rejected[0][1]}")
            self._set_state(syncing=False, available=True, target="forge", error=error)
            if report.conflicts:
                self._notify("info", CONFLICT_MESSAGE)
        self._emit_from_worker()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


__all__ = [
    "CONFLICT_MESSAGE", "LOCAL_ONLY_MESSAGE", "MemoSyncService", "MemoSyncTarget",
]
