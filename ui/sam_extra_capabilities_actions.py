"""sam-extra 확장 기능 스냅샷 — WebUI 연결 때 워커에서 GET 으로 확인하고 Vue 에 알린다.

- 결과는 메인 UI 의 ``self.sam_extra_capabilities`` (``SamExtraCapabilities`` | None)에 둔다.
  뒤 패키지(P2/P3/P7-P9)는 페이로드를 만들 때 이 값의 ``may_use(feature)`` 를 본다
  (모르면 True — 지금처럼 보낸다).
- Vue 이벤트 ``samExtraCapabilities`` 로 스냅샷 전체(JSON)를 보낸다. URL 은 넣지 않는다(웹 모드).
- Vue 액션 ``sam_extra_capabilities_get`` {refresh?: bool}: refresh 면 다시 수집, 아니면 마지막
  스냅샷을 다시 보낸다(늦게 붙은 구독자·웹 재접속용). GUI 스레드는 네트워크를 기다리지 않는다.
  - 지금 백엔드의 확인이 이미 도는 중이면 어느 요청도 새 확인을 시작하지 않는다 — 그 결과가 곧
    이벤트로 간다. (새 확인이 serial 을 올리면 연결 훅의 강제 확인 결과가 버려지고, 캐시의 옛
    스냅샷이 다음 연결까지 남는다.)
  - refresh 는 웹 클라이언트도 보낼 수 있어서 ``ACTION_REFRESH_MIN_INTERVAL_S`` 간격으로 제한한다.
    이 수동 새로고침만 ``/sdapi/v1/extensions`` (Forge 가 확장 목록을 다시 스캔하는 GET) 캐시를 버린다.
- 백엔드 변경(BACKEND_CHANGED / BACKEND_URL_CHANGED)은 스냅샷과 공용 스냅샷 캐시를 버린다 —
  재연결 뒤 캐시가 바뀌기 전 스냅샷을 내주지 않게.
"""
from __future__ import annotations

import json
import threading
import time

from core.sam_extra_capabilities import (
    STATUS_NOT_APPLICABLE, STATUS_UNKNOWN, SamExtraCapabilities, unknown_capabilities,
)

_INIT_LOCK = threading.Lock()
ACTION_REFRESH_MIN_INTERVAL_S = 30.0   # Vue 액션(웹 클라이언트 포함)의 강제 새로고침 최소 간격
_clock = time.monotonic                # 테스트가 바꾼다


class SamExtraCapabilitiesActionsMixin:
    def _handle_sam_extra_capabilities_action(self, action, payload):
        if action != "sam_extra_capabilities_get":
            return False
        payload = payload if isinstance(payload, dict) else {}
        self._sam_extra_ensure()
        if self._sam_extra_probe_pending():
            # 지금 백엔드의 확인이 도는 중 — 그 결과가 곧 간다. 지금 값(보통 '모름')만 다시 보낸다.
            self._emit_sam_extra_capabilities()
        elif payload.get("refresh") is True and self._sam_extra_take_manual_refresh():
            self._refresh_sam_extra_capabilities(force=True, manual=True)
        elif (getattr(self, "sam_extra_capabilities", None) is None
              and getattr(self, "_backend_connected", False)):
            # 연결은 됐는데 결과가 없다(확인 실패 등) — 캐시(TTL)로 채운다. 백엔드가 바뀌면 캐시도
            # 버리므로(_invalidate_sam_extra_capabilities) 바뀌기 전 스냅샷이 나오지 않는다.
            self._refresh_sam_extra_capabilities(force=False)
        else:
            # 연결 전(시작 게이트 등)에는 서버를 두드리지 않는다 — 연결 훅이 곧 확인한다.
            # 간격 제한에 걸린 refresh 도 여기로 온다(마지막 스냅샷을 다시 보낸다).
            self._emit_sam_extra_capabilities()
        return True

    def _sam_extra_ensure(self):
        with _INIT_LOCK:
            if hasattr(self, "_sam_extra_lock"):
                return
            self._sam_extra_lock = threading.Lock()
            # 읽기+보내기를 한 덩어리로 — 워커의 새 결과 뒤에 옛 값이 늦게 나가지 않게 순서를 지킨다.
            self._sam_extra_emit_lock = threading.RLock()
            self._sam_extra_serial = 0
            if not hasattr(self, "sam_extra_capabilities"):
                self.sam_extra_capabilities = None
        try:
            from core.app_context import Events, get_context
            context = get_context()
            self._sam_extra_subscriptions = [
                context.subscribe(event, lambda _data: self._invalidate_sam_extra_capabilities())
                for event in (Events.BACKEND_CHANGED, Events.BACKEND_URL_CHANGED)]
        except Exception as exc:  # 컨텍스트가 없는 테스트 더블 — 연결 훅만으로도 동작한다
            print(f"[sam-extra] 백엔드 변경 구독 실패(무시): {exc}", flush=True)

    def _sam_extra_probe_pending(self) -> bool:
        """지금 serial(지금 백엔드)의 확인 워커가 아직 결과를 내지 않았을 수 있다."""
        with self._sam_extra_lock:
            worker = getattr(self, "_sam_extra_worker", None)
            current = getattr(self, "_sam_extra_worker_serial", None) == self._sam_extra_serial
        return bool(current and worker is not None and (worker.ident is None or worker.is_alive()))

    def _sam_extra_take_manual_refresh(self) -> bool:
        """액션의 강제 새로고침을 지금 해도 되나 (되면 시각을 기록한다)."""
        now = _clock()
        with self._sam_extra_lock:
            last = getattr(self, "_sam_extra_manual_refresh_at", None)
            if last is not None and now - last < ACTION_REFRESH_MIN_INTERVAL_S:
                return False
            self._sam_extra_manual_refresh_at = now
        return True

    def _emit_sam_extra_capabilities(self):
        self._sam_extra_ensure()
        signal = getattr(getattr(self, "vue_bridge", None), "samExtraCapabilities", None)
        if signal is None:
            return
        with self._sam_extra_emit_lock:
            capabilities = getattr(self, "sam_extra_capabilities", None) or unknown_capabilities()
            try:
                signal.emit(json.dumps(capabilities.to_dict(), ensure_ascii=False))
            except RuntimeError:
                pass  # 종료 중 QObject 가 이미 사라졌다

    def _store_sam_extra_capabilities(self, capabilities: SamExtraCapabilities, serial: int) -> bool:
        with self._sam_extra_lock:
            if serial != self._sam_extra_serial:
                return False  # 더 새 요청이나 백엔드 변경이 있었다 — 옛 결과는 버린다
            self.sam_extra_capabilities = capabilities
        self._emit_sam_extra_capabilities()
        return True

    def _invalidate_sam_extra_capabilities(self):
        self._sam_extra_ensure()
        with self._sam_extra_lock:
            self._sam_extra_serial += 1
            self.sam_extra_capabilities = None
        try:
            from core.sam_extra_probe import invalidate_capabilities
            invalidate_capabilities()   # 이벤트에 옛 주소가 없다 — 전부 버린다(연결 훅이 곧 다시 묻는다)
        except Exception as exc:
            print(f"[sam-extra] 스냅샷 캐시 무효화 실패(무시): {exc}", flush=True)
        self._emit_sam_extra_capabilities()

    def _refresh_sam_extra_capabilities(self, *, force: bool = False, manual: bool = False):
        """지금 백엔드의 스냅샷을 워커에서 다시 얻는다. GUI 스레드를 막지 않는다.

        ``manual`` = 사용자의 수동 새로고침(간격 제한은 호출하는 쪽) — 확장 목록 캐시도 버린다.
        """
        self._sam_extra_ensure()
        from backends import BackendType, get_backend, get_backend_type

        with self._sam_extra_lock:
            self._sam_extra_serial += 1
            serial = self._sam_extra_serial
        if get_backend_type() != BackendType.WEBUI:
            self._store_sam_extra_capabilities(unknown_capabilities(STATUS_NOT_APPLICABLE), serial)
            return
        base_url = str(getattr(get_backend(), "api_url", "") or "")
        if not base_url:
            self._store_sam_extra_capabilities(unknown_capabilities(STATUS_UNKNOWN), serial)
            return
        if manual:
            from core.sam_extra_probe import invalidate_extensions
            invalidate_extensions(base_url)
        fetch = getattr(self, "_sam_extra_fetch", None)

        def work():
            try:
                if fetch is not None:
                    capabilities = fetch(base_url, force)
                else:
                    from core.sam_extra_probe import get_capabilities
                    capabilities = get_capabilities(base_url, refresh=force)
            except Exception as exc:
                capabilities = unknown_capabilities("error", error=type(exc).__name__)
            if self._store_sam_extra_capabilities(capabilities, serial) and capabilities.known:
                flags = [name for name, on in capabilities.to_dict()["features"].items() if on]
                print(f"[sam-extra] 기능 확인: installed={capabilities.installed} "
                      f"features={','.join(flags) or '-'} warnings={len(capabilities.warnings)}", flush=True)

        worker = threading.Thread(target=work, daemon=True, name="sam-extra-capabilities")
        with self._sam_extra_lock:
            self._sam_extra_worker = worker
            self._sam_extra_worker_serial = serial
        worker.start()
