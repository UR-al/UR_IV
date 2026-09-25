"""sam-extra 알림(``core/sam_extra_notices``) → 기존 토스트(``vue_bridge.showNotification``).

새 브리지 이름은 없다. 모든 함수는 GUI 스레드에서 부른다(워커 결과 시그널의 슬롯, 생성 시작 직전).

- ``show_result_notices(host, gen_info)``  T2I·I2I·인페인트 결과: 백엔드가 generation info 에 실어 둔 알림
- ``check_before_generation(host, payload)``  보내기 직전 경고 (WebUI 백엔드에서만, 생성은 막지 않는다)
- ``check_standalone_sam3(host, settings)``  단독 SAM3·배치·Refine 시작 전 — 체크포인트 자동 다운로드 경고
- ``relay_worker_result(host, signal_name, text)``  워커 결과 JSON 의 ``notices`` 를 띄우고 그대로 Vue 로 보낸다

같은 알림은 ``NoticeThrottle`` 로 잠시 한 번만 띄운다(배치·자동화가 장마다 같은 토스트를 쌓지 않게).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping

from core import sam_extra_notices as notices_core
from core.sam_extra_notices import Notice

logger = logging.getLogger(__name__)

_THROTTLE_ATTR = "_sam_extra_notice_throttle"
_TOAST_MAX_LEN = 600


def _throttle(host) -> notices_core.NoticeThrottle:
    throttle = getattr(host, _THROTTLE_ATTR, None)
    if not isinstance(throttle, notices_core.NoticeThrottle):
        throttle = notices_core.NoticeThrottle()
        try:
            setattr(host, _THROTTLE_ATTR, throttle)
        except Exception:
            pass
    return throttle


def show_notices(host, notices: Iterable[Notice], *, ttl: float = notices_core.RESULT_NOTICE_TTL_S) -> int:
    """알림을 토스트로 띄운다(억제된 것 제외). 띄운 수."""
    signal = getattr(getattr(host, "vue_bridge", None), "showNotification", None)
    shown = 0
    throttle = _throttle(host)
    for notice in notices:
        if not isinstance(notice, Notice) or not throttle.allow(notice, notices_core.notice_ttl(notice, ttl)):
            continue
        log = logger.info if notice.level == notices_core.LEVEL_INFO else logger.warning
        log("[sam-extra] %s: %s", notice.code, notice.message)
        if signal is None:
            continue
        try:
            from core.error_handler import sanitize_for_ui
            signal.emit(notice.level, sanitize_for_ui(notice.message, max_len=_TOAST_MAX_LEN))
            shown += 1
        except RuntimeError:
            pass   # 종료 중 QObject 가 이미 사라졌다
    return shown


def show_result_notices(host, gen_info: Any) -> int:
    """생성 결과 info 의 알림(``INFO_KEY``)을 띄운다."""
    try:
        return show_notices(host, notices_core.notices_from_info(gen_info))
    except Exception:
        logger.debug("sam-extra 결과 알림 실패(무시)", exc_info=True)
        return 0


def _webui_context(host) -> tuple[bool, Any, bool]:
    """(WebUI 백엔드인가, 기능 스냅샷|None, 원격 Forge 인가). 네트워크 없음."""
    from backends import BackendType, get_backend, get_backend_type

    if get_backend_type() != BackendType.WEBUI:
        return False, None, False
    api_url = str(getattr(get_backend(), "api_url", "") or "")
    capabilities = getattr(host, "sam_extra_capabilities", None)
    if capabilities is None and api_url:
        try:
            from core.sam_extra_probe import peek_capabilities
            capabilities = peek_capabilities(api_url)   # 캐시만 본다 — HTTP 없음
        except Exception:
            capabilities = None
    return True, capabilities, not notices_core.is_loopback_url(api_url)


def check_before_generation(host, payload: Any) -> int:
    """보내기 직전 payload 경고를 띄운다. 생성은 막지 않는다(실패해도 조용히 0)."""
    try:
        is_webui, capabilities, remote = _webui_context(host)
        if not is_webui or not isinstance(payload, Mapping):
            return 0
        notices = notices_core.pre_generation_notices(payload, capabilities=capabilities, remote=remote)
        return show_notices(host, notices, ttl=notices_core.PRE_GENERATION_NOTICE_TTL_S)
    except Exception:
        logger.debug("sam-extra 생성 전 검사 실패(무시)", exc_info=True)
        return 0


def check_standalone_sam3(host, settings: Any) -> int:
    """단독 SAM3·배치 SAM3·Refine 시작 전: 체크포인트가 이름만 남은 'sam3.pt' 면 HF 다운로드 경고."""
    try:
        is_webui, _capabilities, remote = _webui_context(host)
        if not is_webui:
            return 0
        value = (settings or {}).get("sam3_checkpoint") if isinstance(settings, Mapping) else None
        from core.forge_modules import resolve_sam3_checkpoint
        resolved = resolve_sam3_checkpoint(str(value or "").strip() or notices_core.HF_CHECKPOINT_NAME)
        notice = notices_core.sam3_checkpoint_notice(resolved, remote=remote)
        return show_notices(host, [notice] if notice else [], ttl=notices_core.PRE_GENERATION_NOTICE_TTL_S)
    except Exception:
        logger.debug("sam-extra 단독 SAM3 사전 검사 실패(무시)", exc_info=True)
        return 0


def relay_worker_result(host, signal_name: str, text: str) -> None:
    """워커 결과 JSON → (``notices`` 토스트) → Vue 시그널 그대로. JSON 이 아니어도 그대로 보낸다."""
    try:
        data = json.loads(text) if isinstance(text, str) else None
        raw = data.get("notices") if isinstance(data, Mapping) else None
        if isinstance(raw, list):
            show_notices(host, [n for n in (Notice.from_dict(item) for item in raw) if n is not None])
    except Exception:
        logger.debug("sam-extra 워커 알림 실패(무시)", exc_info=True)
    signal = getattr(getattr(host, "vue_bridge", None), signal_name, None)
    if signal is not None:
        signal.emit(text)


__all__ = [
    "check_before_generation", "check_standalone_sam3", "relay_worker_result", "show_notices",
    "show_result_notices",
]
