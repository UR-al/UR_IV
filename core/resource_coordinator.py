"""Serialize heavyweight generation and coordinate owned GPU resources."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, Lock, RLock
from typing import Callable, Iterator, Optional
import logging
import time


class ResourceBusyError(RuntimeError):
    """Another Creator job currently owns the GPU generation lease."""


class ResourceTransitionError(RuntimeError):
    """An owned resource could not be moved to the required state."""


#: ``try_hold`` 로 잡은 리스의 단계 이름 — 생성이 아니라 짧은 백엔드 작업(모델 언로드)이 쥐고 있다.
HOLD_PHASE = "unloading"

_BUSY_MESSAGE = "다른 생성 작업이 GPU 리소스를 사용 중입니다"
_HOLD_BUSY_MESSAGE = "백엔드 모델 언로드가 진행 중입니다 — 끝난 뒤 다시 시도하세요"
_BACKEND_JOB_BUSY_MESSAGE = "후처리 백엔드 작업(ADetailer·SAM3·Refine·업스케일)이 진행 중입니다"

#: 모델 언로드 HTTP 한 번의 requests 타임아웃(초) — Forge unload-checkpoint(backends/webui_backend.py).
#: requests 는 이 값을 연결과 응답 읽기에 **각각** 쓰므로, 언로드 스레드가 ``try_hold`` 를 쥐는 시간은
#: 길어야 약 2배다. ComfyUI /free 는 즉시 ACK 라 더 짧다(10초).
UNLOAD_HTTP_TIMEOUT_SECONDS = 30.0

#: 후처리 백엔드 작업(:meth:`GenerationResourceCoordinator.backend_job`)이 진행 중인 모델 언로드 hold 를
#: 기다리는 최대 시간 — 언로드 HTTP 가 스스로 끝나는 상한(연결+읽기 = 2×UNLOAD_HTTP_TIMEOUT_SECONDS)보다
#: 길게 둔다. 그래서 정상적인 언로드는 늘 이보다 먼저 hold 를 놓고, 이 시간을 넘기는 것은 hold 가 멈춘
#: (언로드 훅이 돌아오지 않는) 경우뿐이다 — 그때 후처리는 언로드와 겹쳐 돌지 않고 ResourceBusyError 로
#: 실패한다(fail-closed). 예전엔 30초(= HTTP 타임아웃과 같은 값)를 넘기면 hold 가 쥐어진 채로 그대로
#: 진행했다. 생성 쪽도 hold 와 겹치지 않는다(reserve(timeout=0) → ResourceBusyError).
BACKEND_JOB_UNLOAD_WAIT_SECONDS = 2 * UNLOAD_HTTP_TIMEOUT_SECONDS + 5.0


@dataclass(frozen=True)
class ResourceState:
    phase: str
    owner: str = ""
    llm_unloaded: bool = False
    since: float = 0.0


class GenerationResourceCoordinator:
    """Own a single generation lease and an optional LLM unload hook.

    External Forge/Comfy processes are intentionally outside this interface.
    Only hooks explicitly supplied by the application are invoked, preventing
    the coordinator from killing a process it does not own.

    ``before_generation`` releases GPU memory this process itself holds (for
    example the editor's cached SAM3 bundle) right after the lease is acquired.
    It is best-effort: a failing hook never blocks or fails the generation.
    """

    def __init__(
        self,
        unload_llm: Optional[Callable[[], bool]] = None,
        on_state: Optional[Callable[[ResourceState], None]] = None,
        before_generation: Optional[Callable[[], object]] = None,
    ) -> None:
        self._lease = Lock()
        self._state_lock = RLock()
        self._unload_llm = unload_llm
        self._on_state = on_state
        self._before_generation = before_generation
        self._state = ResourceState("idle", since=time.time())
        # 리스를 잡지 않는 후처리 백엔드 작업(backend_job) 수와 언로드 hold 여부 — try_hold 와
        # backend_job 이 같은 조건 변수 아래에서 서로를 본다(한쪽이 확인한 뒤 다른 쪽이 끼어들지 못한다).
        self._jobs = Condition(Lock())
        self._backend_jobs = 0
        self._holding = False

    def configure(
        self,
        *,
        unload_llm: Optional[Callable[[], bool]] = None,
        on_state: Optional[Callable[[ResourceState], None]] = None,
        before_generation: Optional[Callable[[], object]] = None,
    ) -> None:
        """Attach application hooks without replacing the shared lease.

        The first ordinary generation may happen before Creator Studio opens.
        Late configuration therefore updates lifecycle callbacks while keeping
        the exact same lock and state object used by every generation path.
        """

        with self._state_lock:
            if unload_llm is not None:
                self._unload_llm = unload_llm
            if on_state is not None:
                self._on_state = on_state
            if before_generation is not None:
                self._before_generation = before_generation

    @property
    def state(self) -> ResourceState:
        with self._state_lock:
            return self._state

    @contextmanager
    def reserve(
        self,
        owner: str,
        *,
        unload_llm: bool = True,
        timeout: float = 0.0,
    ) -> Iterator[ResourceState]:
        owner = str(owner or "creator")[:120]
        acquired = self._lease.acquire(timeout=max(0.0, float(timeout)))
        if not acquired:
            raise ResourceBusyError(self._busy_message())

        did_unload = False
        try:
            self._set_state("preparing", owner, False)
            self._release_in_process_models()
            if unload_llm and self._unload_llm is not None:
                did_unload = bool(self._unload_llm())
                if not did_unload:
                    raise ResourceTransitionError("Ollama 모델 언로드를 확인하지 못했습니다")
            running = self._set_state("running", owner, did_unload)
            yield running
        finally:
            self._set_state("releasing", owner, did_unload)
            self._set_state("idle", "", False)
            self._lease.release()

    @contextmanager
    def try_hold(self, owner: str, *, timeout: float = 0.0) -> Iterator[ResourceState]:
        """생성이 아닌 짧은 백엔드 작업(모델 언로드)이 리스를 잡는다 — 생성과 서로 배타.

        생성 작업이 리스를 쥐고 있으면(``timeout`` 안에 못 잡으면) ResourceBusyError — 호출자는
        그 작업을 건너뛴다. 샘플링 도중 Forge unload-checkpoint 가 끼어들면 queue_lock 없이
        모델을 내려 진행 중인 인페인트·I2I 가 깨진다.
        ``before_generation``·``on_state`` 훅은 부르지 않는다 — 편집기 SAM3 캐시를 건드릴 이유가 없고,
        Creator 상태 UI 에 가짜 '생성 중'이 뜨지 않게. 단계는 :data:`HOLD_PHASE`.

        리스를 잡지 않는 후처리 백엔드 작업(:meth:`backend_job` — ADetailer·SAM3·Refine·배치 업스케일)이
        도는 중이어도 같은 이유로 ResourceBusyError 다(남은 ``timeout`` 만큼은 끝나길 기다린다).
        hold 를 쥔 동안 새 후처리 작업은 시작하지 않고 hold 가 풀리길 기다린다.
        """
        owner = str(owner or "hold")[:120]
        wait = max(0.0, float(timeout))
        deadline = time.monotonic() + wait
        if not self._lease.acquire(timeout=wait):
            raise ResourceBusyError(self._busy_message())
        try:
            with self._jobs:
                while self._backend_jobs:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ResourceBusyError(_BACKEND_JOB_BUSY_MESSAGE)
                    self._jobs.wait(remaining)
                self._holding = True
        except BaseException:
            self._lease.release()
            raise
        try:
            state = ResourceState(HOLD_PHASE, owner, False, time.time())
            with self._state_lock:
                self._state = state
            yield state
        finally:
            with self._state_lock:
                self._state = ResourceState("idle", since=time.time())
            with self._jobs:
                self._holding = False
                self._jobs.notify_all()
            self._lease.release()

    @contextmanager
    def backend_job(self, owner: str = "", *,
                    wait_timeout: Optional[float] = None) -> Iterator[None]:
        """리스를 잡지 않는 Forge/ComfyUI 후처리 작업 한 건(ADetailer·SAM3·Refine·배치 업스케일).

        생성과는 나란히 돈다(리스를 잡지 않는다 — 생성 중에 실패하지 않게). 배타는 모델 언로드
        (:meth:`try_hold`)와만이다 — Forge unload-checkpoint 는 queue_lock 없이 모델을 내려 샘플링 중인
        img2img/extras 작업을 깨뜨린다.
        - 언로드가 hold 를 쥐고 있으면 끝나길 기다렸다 시작한다(최대 ``wait_timeout``, 기본
          :data:`BACKEND_JOB_UNLOAD_WAIT_SECONDS` — 언로드 HTTP 의 상한보다 길다). 그래도 hold 가 안
          풀리면(멈춘 훅) 언로드와 겹쳐 돌지 않고 ResourceBusyError — 워커가 오류로 알린다.
          예전엔 시간이 지나면 hold 가 쥐어진 채로 그대로 진행했다.
        - 도는 동안에는 try_hold 가 ResourceBusyError — 언로드 스레드는 HTTP 를 보내지 않고 건너뛴다.
        워커 스레드 전용(언로드를 기다릴 수 있다). 여러 작업이 동시에 들어올 수 있다.
        """
        if wait_timeout is None:
            # 호출 때 읽는다 — 테스트가 모듈 상수를 바꿔 끼울 수 있게(기본 인자는 정의 때 굳는다)
            wait_timeout = BACKEND_JOB_UNLOAD_WAIT_SECONDS
        deadline = time.monotonic() + max(0.0, float(wait_timeout))
        with self._jobs:
            while self._holding:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # 아직 아무것도 세지 않았다 — 되돌릴 것 없이 실패한다
                    logging.getLogger(__name__).warning(
                        "backend job %s refused: a model unload is still holding the backend", owner or "?")
                    raise ResourceBusyError(_HOLD_BUSY_MESSAGE)
                self._jobs.wait(remaining)
            self._backend_jobs += 1
        try:
            yield
        finally:
            with self._jobs:
                self._backend_jobs -= 1
                self._jobs.notify_all()

    def is_busy(self) -> bool:
        """누군가(생성 작업이든 언로드 hold 든) 리스를 쥐고 있는가 — 스냅숏, 판정용 힌트."""
        return self.state.phase != "idle"

    def generation_active(self) -> bool:
        """생성 작업이 리스를 쥐고 있는가(언로드 hold 는 제외) — 스냅숏, 판정용 힌트.

        확정적인 배타는 ``reserve``/``try_hold`` 가 한다. 이것은 UI 가 '곧 건너뛸 요청'을 알리지
        않게 미리 보는 용도다.
        """
        return self.state.phase not in ("idle", HOLD_PHASE)

    def backend_jobs_active(self) -> bool:
        """리스 밖의 후처리 백엔드 작업(:meth:`backend_job`)이 도는 중인가 — 스냅숏."""
        with self._jobs:
            return self._backend_jobs > 0

    def unload_blocked(self) -> bool:
        """지금 모델을 내리면 진행 중인 작업을 깨뜨리는가 — 생성 리스 보유 또는 후처리 작업 진행.

        스냅숏, 판정용 힌트(언로드 요청을 아예 보내지 않고 '건너뜀'을 알리는 용도). 확정적인 배타는
        언로드 스레드의 :meth:`try_hold` 가 한다.
        """
        return self.generation_active() or self.backend_jobs_active()

    def _busy_message(self) -> str:
        return _HOLD_BUSY_MESSAGE if self.state.phase == HOLD_PHASE else _BUSY_MESSAGE

    def _release_in_process_models(self) -> None:
        with self._state_lock:
            hook = self._before_generation
        if hook is None:
            return
        try:
            hook()
        except Exception:
            # Freeing cached editor models is an optimisation; it must never
            # strand the lease or fail the generation that asked for it.
            pass

    def _set_state(self, phase: str, owner: str, llm_unloaded: bool) -> ResourceState:
        state = ResourceState(phase, owner, llm_unloaded, time.time())
        with self._state_lock:
            self._state = state
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:
                pass
        return state


def release_in_process_vision_models() -> None:
    """생성 직전, 이 앱 프로세스가 직접 올린 편집기 비전 모델(SAM3 번들 ~3.4GB)을 반납한다.

    Forge/ComfyUI는 별도 프로세스라 우리 프로세스의 CUDA 캐시를 쓸 수 없다 —
    ANIMA+SAM3+Forge가 16GB를 넘겨 OOM이 나던 시나리오(2e9e52b31)를 막는다.
    외부 프로세스는 건드리지 않는다. 캐시가 비어 있으면 비용이 없다.
    """
    from core.model_cache import release_for_generation
    release_for_generation()


def release_before_backend_job(job: str = "") -> bool:
    """GPU 리스를 잡지 않는 Forge/ComfyUI 후처리 작업 직전에 부른다. 절대 예외를 내지 않는다.

    SAM3·ADetailer·Refine·배치 업스케일 워커는 reserve()를 거치지 않는다 — 생성과 나란히
    Forge 큐에 넣어 온 동작이라 timeout=0 리스로 바꾸면 생성 중에는 실패하게 된다.
    대신 작업마다 이 프로세스가 쥔 편집기 SAM3 번들(~3.4GB)을 먼저 반납해, Forge가 올리는
    SAM3/ADetailer 모델과 겹쳐 16GB를 넘기지 않게 한다(편집기가 추론 중이면 끝나는 즉시).
    캐시가 비었으면 비용이 없다. 반환: 반납 호출이 성공했는지(실패해도 작업은 진행).
    모델 언로드와의 배타는 따로 — 백엔드 HTTP 를 :func:`backend_job_guard` 로 감싼다.
    """
    try:
        release_in_process_vision_models()
        return True
    except Exception:
        logging.getLogger(__name__).warning(
            "in-process vision model release failed before %s", job or "backend job", exc_info=True)
        return False


@contextmanager
def backend_job_guard(job: str = "") -> Iterator[None]:
    """리스를 잡지 않는 후처리 백엔드 HTTP 한 건을 공유 코디네이터의 ``backend_job`` 으로 감싼다.

    진행 중인 모델 언로드(생성 후·대기열 정기 정리·VRAM 수동 언로드)가 끝나길 기다렸다 보내고, 보내는
    동안에는 새 언로드가 건너뛴다 — 생성과는 계속 나란히 돈다. 워커 스레드 전용.
    """
    # 모듈 전역으로 찾는다 — 테스트가 core.resource_coordinator.get_generation_coordinator 를 바꿔 끼운다
    with get_generation_coordinator().backend_job(job or "backend-job"):
        yield


_SHARED_GENERATION_COORDINATOR = GenerationResourceCoordinator(
    before_generation=release_in_process_vision_models,
)


def get_generation_coordinator(
    *,
    unload_llm: Optional[Callable[[], bool]] = None,
    on_state: Optional[Callable[[ResourceState], None]] = None,
    before_generation: Optional[Callable[[], object]] = None,
) -> GenerationResourceCoordinator:
    """Return the process-wide GPU generation lease and optionally configure it."""

    _SHARED_GENERATION_COORDINATOR.configure(
        unload_llm=unload_llm,
        on_state=on_state,
        before_generation=before_generation,
    )
    return _SHARED_GENERATION_COORDINATOR
