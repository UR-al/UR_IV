"""VRAM 게이지 수동 언로드 — 백엔드 HTTP 는 워커 스레드, 결과 표시는 메인 스레드.

예전에는 ``unload_model_request`` 가 메인 스레드에서 동기 HTTP 를 보냈다(WebUI 는
cleanup 30+20+10초 타임아웃이라 응답이 없으면 UI 가 최대 약 60초 멈춤). 반환값도 버려
실패해도 '요청됨'으로만 보였다.

지금은 ``core.post_generation.start_post_generation_unload`` 의 대기 핸들을 같이 쓴다 —
그래서 언로드가 날아가는 동안 시작된 생성 워커는 run() 초입에서 이 요청을 기다린다
(샘플링 도중 언로드가 끼어드는 경쟁 방지). 결과는 QTimer 100ms 핸드오프로 메인 스레드에서
``show_status`` 와 Vue 토스트(``showNotification``)로 알린다 — QWebChannel 에 등록된
vue_bridge 를 워커 스레드에서 emit 하지 않는다.
"""
from __future__ import annotations

from typing import Callable, Optional

POLL_MS = 100

Scheduler = Callable[[int, Callable[[], None]], None]


def _notify(host, level: str, message: str) -> None:
    bridge = getattr(host, "vue_bridge", None)
    signal = getattr(bridge, "showNotification", None)
    if signal is not None:
        try:
            signal.emit(level, message)
        except Exception:
            pass


def _report_busy(host) -> None:
    host.show_status("Model unload skipped: a generation is running.")
    _notify(host, "warning", "생성 중이라 모델을 언로드하지 않았습니다 — 생성이 끝난 뒤 다시 시도하세요")


def _default_scheduler() -> Scheduler:
    from PyQt6.QtCore import QTimer

    return QTimer.singleShot


def request_manual_backend_unload(host, backend, *, schedule: Optional[Scheduler] = None):
    """백엔드 모델 언로드를 워커로 보내고 끝나면 메인 스레드에서 결과를 알린다.

    반환: 시작한 스레드, 이미 언로드가 진행 중이거나 생성 중이라 보내지 않았으면 ``None``.
    ``schedule(ms, fn)`` 은 메인 스레드 타이머(기본 ``QTimer.singleShot``) — 테스트 주입용.

    생성 작업(T2I·인페인트·I2I·채팅·Creator 등)이 공유 GPU 리스를 쥐고 있거나 리스 밖의 후처리 작업
    (ADetailer·SAM3·Refine·배치 업스케일, ``backend_job``)이 도는 중이면 보내지 않는다 —
    Forge unload-checkpoint 는 queue_lock 없이 모델을 내려 샘플링 중인 작업을 깨뜨린다.
    여기서 먼저 보고, 그 사이에 시작된 작업은 언로드 스레드의 ``try_hold`` 가 다시 거른다.
    """
    from core.post_generation import (
        MANUAL_UNLOAD_METHODS, UNLOAD_SKIPPED_BUSY, start_post_generation_unload,
    )
    from core.resource_coordinator import get_generation_coordinator

    if backend is None:
        host.show_status("Model unload failed: no backend.")
        _notify(host, "error", "연결된 백엔드가 없어 모델을 언로드하지 못했습니다")
        return None

    if get_generation_coordinator().unload_blocked():
        _report_busy(host)
        return None

    outcome: list[bool] = []
    thread = start_post_generation_unload(
        backend, methods=MANUAL_UNLOAD_METHODS, on_done=outcome.append,
    )
    if thread is None:
        host.show_status("Model unload already in progress.")
        _notify(host, "info", "이미 모델 언로드가 진행 중입니다")
        return None

    host.show_status("Model unload requested.")
    schedule = schedule or _default_scheduler()

    def _poll() -> None:
        if thread.is_alive():
            schedule(POLL_MS, _poll)
            return
        if outcome and outcome[0] is UNLOAD_SKIPPED_BUSY:
            _report_busy(host)
            return
        ok = bool(outcome and outcome[0])
        if ok:
            host.show_status("Model unloaded.")
            _notify(host, "success", "백엔드 모델 언로드 완료")
        else:
            host.show_status("Model unload failed.")
            _notify(host, "error", "백엔드 모델 언로드 실패 — 백엔드 연결을 확인하세요")

    schedule(POLL_MS, _poll)
    return thread
