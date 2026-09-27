# ui/boot_notices.py
"""부팅 중에 생긴 알림을 **Vue 가 준비되고 메인 창이 보인 뒤에** 띄운다 (critic A5).

``load_settings()`` 는 ``GeneratorMainUI.__init__`` 에서 Vue 로드보다 먼저 돈다. 그때 ``showNotification`` 을 내면
Vue 가 아직 그 시그널을 구독하지 않아 영구히 사라진다. 그래서 알림을 host 에 쌓아 두고 두 조건이 모두 참일 때 띄운다.

1. **Vue 준비** — Vue 는 ``uiPrefsLoaded`` 를 받아 LoRA 스택을 복원한 뒤 늘 ``set_lora_stack`` 을 보낸다
   (frontend/src/composables/useLoraStack.js restoreFromPrefs — 빈 스택도 보낸다). 그 처리 가지가
   ``flush_boot_notices`` 를 부른다. App.vue 는 그보다 먼저 ``showNotification`` 을 구독한다(onMounted 에서
   uiPrefsLoaded 구독 앞).
2. **창 표시** — 데스크톱은 시작 시퀀스(``_run_startup_sequence``: 창을 숨긴 채 Vue 로드 → 백엔드 연결을 최대 8초
   대기, 중첩 이벤트 루프라 WebChannel 메시지는 처리된다)가 끝난 뒤에야 ``new_main_ui.main()`` 이 ``showMaximized()``
   한다. 그래서 Vue 준비 신호는 보통 스플래시 단계에 오고, 토스트 수명은 3초(frontend/src/composables/useToasts.ts
   ``TOAST_TTL_MS``)라 그때 띄우면 사용자가 창을 보기 전에 사라진다. 창이 숨어 있으면 준비만 기록하고, 창을 띄운 뒤
   ``schedule_flush_after_show`` 가 잠깐 뒤(``SHOW_FLUSH_DELAY_MS``) 띄운다. 웹 모드(``web_mode``)는 호스트 창을
   띄우지 않으므로(브라우저가 화면) Vue 준비만 본다.

``set_lora_stack`` 은 세션 중에도 오지만 쌓인 알림이 없으면 아무 일도 하지 않는다(한 번 비우면 끝).
쌓는 것은 ``Notice`` 또는 ``DeferredNotice``(띄울 때 문구를 정한다 — 그때의 백엔드처럼 부팅 뒤에야 정해지는 것으로
고르고, None 이면 띄우지 않는다)다. 새 브리지 이름은 없다. GUI 스레드에서만 부른다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PENDING_ATTR = "_boot_notices_pending"
_VUE_READY_ATTR = "_boot_vue_ready"
# 창을 띄운 뒤 첫 그리기가 끝나고 토스트가 보이도록 조금 기다린다(토스트 수명 3초보다 충분히 짧게)
SHOW_FLUSH_DELAY_MS = 1000


@dataclass(frozen=True)
class DeferredNotice:
    """띄울 때 문구를 정하는 부팅 알림 — ``build()`` 가 Notice 또는 None(띄우지 않음)을 돌려준다."""

    key: str
    build: Callable[[], Any]


def defer_boot_notice(host, notice: Any) -> None:
    """Vue 준비·창 표시 뒤에 띄울 알림 하나(``Notice`` 또는 ``DeferredNotice``)를 쌓는다(같은 key 는 한 번만)."""
    if notice is None:
        return
    pending = getattr(host, _PENDING_ATTR, None)
    if not isinstance(pending, list):
        pending = []
    key = getattr(notice, "key", None)
    if key is not None and any(getattr(item, "key", None) == key for item in pending):
        return
    pending.append(notice)
    try:
        setattr(host, _PENDING_ATTR, pending)
    except Exception:
        pass


def pending_boot_notices(host) -> list:
    """아직 띄우지 않은 항목(``Notice`` 또는 ``DeferredNotice``) 사본."""
    pending = getattr(host, _PENDING_ATTR, None)
    return list(pending) if isinstance(pending, list) else []


def _window_shown(host) -> bool:
    """토스트를 볼 화면이 있나. 웹 모드는 브라우저가 화면이라 늘 참, ``isVisible`` 이 없는 호스트(테스트 더블)도 참."""
    if getattr(host, "web_mode", False):
        return True
    is_visible = getattr(host, "isVisible", None)
    if not callable(is_visible):
        return True
    try:
        return bool(is_visible())
    except Exception:   # 종료 중 — C++ 객체가 이미 사라졌다
        return False


def _resolve(item: Any) -> Any:
    if isinstance(item, DeferredNotice):
        try:
            return item.build()
        except Exception:
            logger.debug("부팅 알림 '%s' 문구 결정 실패(무시)", item.key, exc_info=True)
            return None
    return item


def _show_pending(host) -> int:
    pending = pending_boot_notices(host)
    if not pending:
        return 0
    try:
        setattr(host, _PENDING_ATTR, [])
    except Exception:
        pass
    notices = [notice for notice in (_resolve(item) for item in pending) if notice is not None]
    if not notices:
        return 0
    try:
        from core.sam_extra_notices import PRE_GENERATION_NOTICE_TTL_S
        from ui.sam_extra_notices_ui import show_notices
        return show_notices(host, notices, ttl=PRE_GENERATION_NOTICE_TTL_S)
    except Exception:
        logger.debug("부팅 알림 표시 실패(무시)", exc_info=True)
        return 0


def flush_boot_notices(host) -> int:
    """Vue 준비 신호(``set_lora_stack``)에서 부른다. 창이 보이면 쌓인 알림을 띄우고 비운다. 창이 아직 숨어 있으면
    준비만 기록하고 0 — 창을 띄운 뒤 ``flush_after_window_shown`` 이 띄운다. 여러 번 불러도 된다(비었으면 0)."""
    try:
        setattr(host, _VUE_READY_ATTR, True)
    except Exception:
        pass
    if not pending_boot_notices(host):
        return 0
    if not _window_shown(host):
        logger.debug("부팅 알림 %d건 — 메인 창이 아직 숨어 있어 창을 띄운 뒤로 미룬다",
                     len(pending_boot_notices(host)))
        return 0
    return _show_pending(host)


def flush_after_window_shown(host) -> int:
    """창을 띄운 뒤에 부른다. Vue 가 이미 준비됐으면(부팅 복원 신호를 받았으면) 쌓인 알림을 띄운다. 아직이면 0 —
    그 신호의 ``flush_boot_notices`` 가 (창이 이제 보이므로) 바로 띄운다."""
    if not getattr(host, _VUE_READY_ATTR, False) or not _window_shown(host):
        return 0
    return _show_pending(host)


def schedule_flush_after_show(host, delay_ms: int = SHOW_FLUSH_DELAY_MS) -> None:
    """``window.showMaximized()`` 바로 뒤에 부른다(new_main_ui.main) — ``delay_ms`` 뒤 ``flush_after_window_shown``."""
    try:
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(int(delay_ms), lambda: flush_after_window_shown(host))
    except Exception:
        logger.debug("부팅 알림 표시 예약 실패(무시)", exc_info=True)


__all__ = ["DeferredNotice", "SHOW_FLUSH_DELAY_MS", "defer_boot_notice", "flush_after_window_shown",
           "flush_boot_notices", "pending_boot_notices", "schedule_flush_after_show"]
