# core/post_generation.py
"""생성 후 모델 언로드 판단 — 순수 로직 (Qt 없음).

설정 `unloadModelsAfterGen` 이 켜져 있으면 한 번의 생성이 끝났을 때 백엔드 모델을
VRAM 에서 내린다. 단, 자동화나 대기열이 이어서 다음 장을 만들 때는 내리지 않는다 —
매 장마다 내렸다 올리면 로딩 시간만 늘고(사용자가 이미 해롭다고 실측) 이득이 없다.
연속 작업의 **마지막 장 뒤에만** 내린다. 호출 지점은 세 곳:
단일 생성 완료(on_generation_finished) · 자동화 종료(_stop_automation) · 대기열 종료(_on_queue_completed).

언로드와 생성은 공유 GPU 리스(core.resource_coordinator)로 서로 배타다. 언로드 스레드는
``try_hold`` 로 리스를 잡은 동안에만 HTTP 를 보내고, 다른 생성(인페인트·I2I·채팅·Creator·
생성 API·캡션 등 ``gen_worker`` 밖의 작업 포함)이 쥐고 있으면 보내지 않고 건너뛴다 — Forge 의
unload-checkpoint 는 queue_lock 없이 모델을 내려 샘플링 중인 작업을 깨뜨린다.
생성 쪽은 :func:`reserve_generation_lease` 로 진행 중인 언로드를 기다린 뒤 리스를 잡는다.
리스를 잡지 않는 후처리 작업(ADetailer·SAM3·Refine·배치 업스케일)은 ``backend_job_guard`` 로
등록한다 — 도는 동안에는 ``try_hold`` 가 실패해 언로드를 건너뛰고, 진행 중인 언로드는 기다린다.
"""
from __future__ import annotations

import threading
import time
from contextlib import ExitStack, contextmanager
from typing import Callable, Iterator, Mapping

PREF_KEY = "unloadModelsAfterGen"

#: 언로드 스레드가 리스를 잡을 때 쓰는 소유자 이름(ResourceState.owner).
UNLOAD_HOLD_OWNER = "post-gen-unload"
#: 방금 끝난 워커가 리스를 놓는 짧은 틈(결과 시그널을 리스 안에서 쏜 실패 경로)을 기다리는 시간.
#: 이보다 오래 쥐고 있으면 진짜 생성 중이다 — 기다리지 않고 건너뛴다.
UNLOAD_HOLD_GRACE_SECONDS = 0.5


class _UnloadSkipped:
    """``on_done`` 에 넘기는 '생성이 GPU 를 쓰고 있어 건너뜀' 표시.

    거짓(bool False)이라 이 값을 모르는 호출자는 '언로드 안 됨'으로 읽는다. 구분하려면
    ``ok is UNLOAD_SKIPPED_BUSY`` 로 비교한다.
    """

    __slots__ = ()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNLOAD_SKIPPED_BUSY"


UNLOAD_SKIPPED_BUSY = _UnloadSkipped()

#: '생성 후 언로드' · 대기열 정기 정리(widgets/queue_manager.py) — 체크포인트 언로드가 먼저다.
POST_GENERATION_UNLOAD_METHODS = ("unload_checkpoint", "unload_models", "unload")
#: VRAM 게이지 수동 언로드. Forge 의 unload_models 도 이제 unload_checkpoint 와 같다(예전의
#: LoRA 재스캔·options POST 는 VRAM 을 회수하지 않고 Forge config.json 만 다시 써서 뺐다).
MANUAL_UNLOAD_METHODS = ("unload_models", "unload")

#: 진행 중인 '생성 후 언로드' HTTP 스레드 — 다음 생성 워커가 시작 전에 기다린다.
_pending_lock = threading.Lock()
_pending_unload: threading.Thread | None = None


def unload_after_generation_enabled(prefs: Mapping | None) -> bool:
    if not isinstance(prefs, Mapping):
        return False
    return bool(prefs.get(PREF_KEY, False))


def should_unload_after_generation(
    prefs: Mapping | None,
    *,
    automating: bool,
    queue_running: bool,
    worker_running: bool = False,
    generation_active: bool = False,
) -> bool:
    """이번 생성 뒤에 모델을 내릴지.

    - 설정이 꺼져 있으면 False
    - 자동화가 계속 돌거나 대기열에 다음 작업이 있으면 False (마지막 장 뒤에 내린다)
    - 생성 워커가 아직 도는 중(수동 중지 직후 등)이면 False — 샘플링 중 언로드는 위험
    - 다른 생성 작업이 공유 GPU 리스를 쥐고 있으면 False — ``gen_worker`` 밖의 인페인트·I2I·
      채팅·Creator·생성 API 등(예: 인페인트 중 T2I 를 눌러 바로 '사용 중' 실패한 뒤)
    """
    if not unload_after_generation_enabled(prefs):
        return False
    if automating or queue_running or worker_running or generation_active:
        return False
    return True


def unload_backend_models(backend, methods=POST_GENERATION_UNLOAD_METHODS) -> bool:
    """백엔드 종류를 모르고도 언로드.

    기본 우선순위: unload_checkpoint (Forge unload-checkpoint · ComfyUI /free) → unload_models
    (unload_checkpoint 가 없는 백엔드용) → unload.
    ``methods`` 로 다른 순서(수동 언로드는 :data:`MANUAL_UNLOAD_METHODS`)를 줄 수 있다.
    """
    if backend is None:
        return False
    for name in methods:
        fn = getattr(backend, name, None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                return False
    return False


def unload_in_progress() -> bool:
    """'생성 후 언로드' 요청이 아직 날아가는 중인지."""
    with _pending_lock:
        return _pending_unload is not None and _pending_unload.is_alive()


def _shared_coordinator():
    # 모듈 속성으로 찾는다 — 테스트가 core.resource_coordinator.get_generation_coordinator 를 바꿔 끼운다
    import core.resource_coordinator as resource_coordinator
    return resource_coordinator.get_generation_coordinator()


def start_post_generation_unload(backend, *, on_done: Callable[[bool], None] | None = None,
                                 methods=POST_GENERATION_UNLOAD_METHODS, coordinator=None):
    """언로드 HTTP(Forge unload-checkpoint 최대 30초 · ComfyUI /free)를 데몬 스레드로 보내고
    대기 핸들로 등록한다. 이미 하나가 진행 중이면 새로 보내지 않고 ``None``.

    UI 스레드는 기다리지 않는다 — 다음 생성 워커가 run() 초입에서
    :func:`wait_for_pending_unload` 로 기다린다(샘플링과 언로드가 겹치지 않게).
    VRAM 게이지 수동 언로드도 ``methods=MANUAL_UNLOAD_METHODS`` 로 같은 대기 핸들을 쓴다.

    HTTP 는 공유 GPU 리스를 ``try_hold`` 로 잡은 동안에만 보낸다. 다른 생성 작업이 쥐고 있으면
    (:data:`UNLOAD_HOLD_GRACE_SECONDS` 안에 놓지 않으면) 보내지 않고 ``on_done(UNLOAD_SKIPPED_BUSY)``.
    ``coordinator`` 는 테스트 주입용(기본: 공유 코디네이터).
    """
    global _pending_unload

    def _run():
        from core.resource_coordinator import ResourceBusyError
        coord = coordinator if coordinator is not None else _shared_coordinator()
        try:
            with coord.try_hold(UNLOAD_HOLD_OWNER, timeout=UNLOAD_HOLD_GRACE_SECONDS):
                ok = unload_backend_models(backend, methods)
        except ResourceBusyError:
            ok = UNLOAD_SKIPPED_BUSY
        if on_done is not None:
            try:
                on_done(ok)
            except Exception:
                pass

    with _pending_lock:
        if _pending_unload is not None and _pending_unload.is_alive():
            return None
        thread = threading.Thread(target=_run, name="post-gen-unload", daemon=True)
        _pending_unload = thread
        thread.start()
        return thread


def wait_for_pending_unload(timeout: float = 30.0, *, cancelled: Callable[[], bool] | None = None,
                            poll: float = 0.1) -> bool:
    """진행 중인 언로드가 끝날 때까지 **호출한 워커 스레드에서** 기다린다.

    ``True`` = 더 기다릴 언로드가 없다. ``cancelled()`` 가 참이 되거나 ``timeout`` 을 넘기면
    ``False`` — 호출자는 취소 여부를 다시 확인하고, 시간 초과면 그대로 진행한다(예전 join 과 같다).
    GUI 스레드에서 부르지 않는다.
    """
    with _pending_lock:
        thread = _pending_unload
    if thread is None or not thread.is_alive():
        return True
    deadline = time.monotonic() + max(0.0, float(timeout))
    step = max(0.01, float(poll))
    while thread.is_alive():
        if cancelled is not None and cancelled():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        thread.join(min(step, remaining))
    return True


@contextmanager
def reserve_generation_lease(owner: str, *, coordinator=None, unload_llm: bool = False,
                             cancelled: Callable[[], bool] | None = None) -> Iterator[object]:
    """진행 중인 '생성 후 언로드'를 기다린 뒤 공유 GPU 리스를 잡는다(``reserve(timeout=0)`` —
    다른 생성 작업은 기다리지 않고 곧바로 ResourceBusyError).

    워커 스레드 전용 — 언로드를 최대 30초 기다릴 수 있으니 GUI 스레드에서 부르지 않는다.
    기다림과 reserve 사이에 막 시작된 언로드가 리스를 쥐었으면(ResourceBusyError) 그 언로드가
    끝나길 한 번 더 기다렸다가 다시 잡는다 — 그 틈에 걸린 생성이 '사용 중'으로 실패하지 않게.
    진짜 다른 생성이 쥐고 있으면 두 번째 시도도 ResourceBusyError 그대로다(기다림은 즉시 끝난다).
    취소는 호출자가 리스 안에서 다시 확인한다(첫 기다림 뒤와 같은 규칙).
    """
    from core.resource_coordinator import ResourceBusyError
    coord = coordinator if coordinator is not None else _shared_coordinator()
    wait_for_pending_unload(cancelled=cancelled)
    with ExitStack() as stack:
        busy = False
        try:
            state = stack.enter_context(coord.reserve(owner, unload_llm=unload_llm, timeout=0))
        except ResourceBusyError:
            busy = True
        if busy:
            wait_for_pending_unload(cancelled=cancelled)
            state = stack.enter_context(coord.reserve(owner, unload_llm=unload_llm, timeout=0))
        yield state
