# workers/generation_worker.py
import copy
import json
import logging
import threading
import time
from contextlib import contextmanager

from PyQt6.QtCore import QThread, pyqtSignal

from backends import get_backend
from core.cancellable_call import call_with_optional_cancel
from core.error_handler import sanitize_for_ui
from core.post_generation import reserve_generation_lease, wait_for_pending_unload
from core.resource_coordinator import ResourceBusyError, get_generation_coordinator

logger = logging.getLogger(__name__)


class WebUIInfoWorker(QThread):
    """서버 정보 로드 워커"""
    info_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def run(self):
        """백엔드 API에서 모델, 샘플러 등 정보 가져오기"""
        try:
            backend = get_backend()
            info = backend.get_info()
            self.info_ready.emit({
                'models': info.models,
                'samplers': info.samplers,
                'schedulers': info.schedulers,
                'upscalers': info.upscalers,
                'options': info.options,
                'vae': info.vae,
                'checkpoints': info.checkpoints,
            })
        except Exception as e:
            logger.exception("WebUIInfoWorker failed")
            self.error_occurred.emit(sanitize_for_ui(e))


class _CancellableMixin:
    """QThread 생성 워커에 공통으로 쓰는 취소 플래그."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True
        # 플래그만으론 이미 보낸 HTTP 요청이 끝까지 돈다 — 백엔드에 실제 중단 요청.
        # fire-and-forget: 메인 스레드를 막지 않고, 실패해도 무시.
        import threading

        def _do_interrupt():
            try:
                owned_interrupt = getattr(self, '_interrupt_owned_backend', None)
                if owned_interrupt:
                    owned_interrupt()
                else:
                    get_backend().interrupt()
            except Exception:
                logger.debug("backend interrupt 실패(무시)", exc_info=True)
        threading.Thread(target=_do_interrupt, daemon=True).start()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def _cancel_check(self) -> bool:
        """백엔드에 넘기는 cancel_check — 모델 전환 중·큐 투입 직후의 취소도 잡는다.

        전역 interrupt 한 번은 WebUI 가 아직 inflight 가 아닐 때(체크포인트 전환 중) 버려지고,
        Forge 가 state.begin() 하기 전에 도착하면 플래그가 지워진다. 백엔드는 이 콜백으로
        발송 전·전환 후에 다시 확인하고, 우리 작업이 실제로 돌기 시작한 뒤에만 중단한다.
        """
        return self._cancelled


class _GenerationDeferred(RuntimeError):
    """A queued snapshot could not be submitted; keep it for explicit resume."""


class GenerationFlowWorker(QThread, _CancellableMixin):
    """이미지 생성 워커"""
    finished = pyqtSignal(object, dict)
    progress = pyqtSignal(int, int, object)  # step, total_steps, preview_bytes|None

    def __init__(self, model_name: str, payload: dict, *, backend=None):
        QThread.__init__(self)
        _CancellableMixin.__init__(self)
        self.model_name = model_name
        self.payload = copy.deepcopy(payload)
        self._backend_snapshot = backend
        self._backend_lock = threading.RLock()
        self._owned_backend = None
        self._start_time: float | None = None
        self._result_emitted = False

    def _emit_result(self, result, info):
        info = dict(info or {})
        xyz_info = self.payload.get("_xyz_info")
        if isinstance(xyz_info, dict):
            info["_xyz_info"] = copy.deepcopy(xyz_info)
        self._result_emitted = True
        self.finished.emit(result, info)

    def _interrupt_owned_backend(self):
        # Hold only this worker's lease lock, never a GUI lock, during HTTP.
        with self._backend_lock:
            if self._owned_backend is not None:
                self._owned_backend.interrupt()

    @contextmanager
    def _backend_lease(self, backend):
        # run() 초입의 기다림 뒤에 막 시작된 '생성 후 언로드'가 리스를 쥐었으면 그걸 기다렸다 다시 잡는다
        with reserve_generation_lease("txt2img", coordinator=get_generation_coordinator(),
                                      cancelled=lambda: self.is_cancelled):
            with self._backend_lock:
                if self._backend_snapshot is not None and get_backend() is not self._backend_snapshot:
                    raise _GenerationDeferred("XYZ 작업의 백엔드가 변경되었습니다. 원래 백엔드를 선택하고 대기열을 재개하세요.")
                self._owned_backend = backend
            try:
                yield
            finally:
                with self._backend_lock:
                    self._owned_backend = None

    def run(self):
        self._start_time = time.monotonic()
        dispatched = False
        try:
            backend = self._backend_snapshot if self._backend_snapshot is not None else get_backend()
            payload = dict(self.payload)
            if "_chat_deferred_prompt" in payload:
                from core.chat_generation import prepare_prompt_payload
                payload = prepare_prompt_payload(payload)
            xyz_info = payload.pop("_xyz_info", None)
            generation_family = str(payload.pop("_generation_family", "standard") or "standard").lower()

            def on_progress(step: int, total: int, preview):
                if self.is_cancelled:
                    return
                self.progress.emit(step, total, preview)

            # '생성 후 언로드' 요청이 아직 날아가는 중이면 여기(워커 스레드)서 기다린다 —
            # 샘플링이 언로드 위에 겹치지 않게. 기다리는 동안의 취소는 바로 아래에서 다시 본다.
            wait_for_pending_unload(cancelled=lambda: self.is_cancelled)
            if self.is_cancelled:
                self._emit_result("생성 취소됨", {'cancelled': True})
                return

            with self._backend_lease(backend):
                if self.is_cancelled:
                    self._emit_result("생성 취소됨", {'cancelled': True})
                    return
                dispatched = True
                # cancel_check 를 넘겨야 모델 전환 중·발송 직후의 취소가 사라지지 않는다
                # (받지 못하는 옛 어댑터·fake 에는 넘기지 않는다 — core/cancellable_call.py).
                if generation_family == "krea2":
                    from core.krea2_generation import run_krea2_generation

                    result = call_with_optional_cancel(
                        run_krea2_generation, backend, "t2i", payload,
                        progress_callback=on_progress, cancel_check=self._cancel_check,
                    )
                else:
                    result = call_with_optional_cancel(
                        backend.txt2img, self.model_name, payload,
                        progress_callback=on_progress, cancel_check=self._cancel_check,
                    )

                # 취소 후 도착한 결과(interrupt의 부분 이미지 포함)는 성공으로 emit하지 않음
                # — 디스크 저장/히스토리/성공 통계/자동화 계속으로 이어지던 버그 방지
                if self.is_cancelled:
                    self._emit_result("생성 취소됨", {'cancelled': True})
                    return

                if not result.success:
                    self._emit_result(result.error, {})
                    return

                # ADetailer/SAM3 는 요청 안의 alwayson_scripts 로 이미 적용돼 돌아온다
                # (ui/generator_generation._apply_postprocess_chain) — 별도 후처리 단계 없음.
                final_image = result.image_data
            if self.is_cancelled:
                self._emit_result("생성 취소됨", {'cancelled': True})
                return
            info = dict(result.info or {})
            if xyz_info:
                info['_xyz_info'] = xyz_info
            self._emit_result(final_image, info)

        except Exception as e:
            if self.is_cancelled:
                # interrupt로 인한 요청 중단 예외는 '취소'로 보고
                self._emit_result("생성 취소됨", {'cancelled': True})
                return
            if (self._backend_snapshot is not None and not dispatched
                    and isinstance(e, (_GenerationDeferred, ResourceBusyError))):
                self._emit_result(sanitize_for_ui(e), {'_queue_deferred': True})
                return
            logger.exception("GenerationFlowWorker failed")
            self._emit_result(f"이미지 생성 중 오류: {sanitize_for_ui(e)}", {})


class Img2ImgFlowWorker(QThread, _CancellableMixin):
    """img2img / inpaint 생성 워커"""
    finished = pyqtSignal(object, dict)
    progress = pyqtSignal(int, int, object)

    def __init__(self, model_name: str, payload: dict):
        QThread.__init__(self)
        _CancellableMixin.__init__(self)
        self.model_name = model_name
        self.payload = payload

    def run(self):
        try:
            backend = get_backend()
            payload = dict(self.payload)
            generation_family = str(payload.pop("_generation_family", "standard") or "standard").lower()

            def on_progress(step: int, total: int, preview):
                if self.is_cancelled:
                    return
                self.progress.emit(step, total, preview)

            # T2I 와 같은 이유로 진행 중인 '생성 후 언로드'를 먼저 기다린다(예전엔 이 경로에 대기가 없었다)
            wait_for_pending_unload(cancelled=lambda: self.is_cancelled)
            if self.is_cancelled:
                self.finished.emit("생성 취소됨", {'cancelled': True})
                return

            # 리스는 '생성 후 언로드'와 서로 배타 — 기다림 뒤 막 시작된 언로드면 끝나길 기다려 다시 잡는다
            with reserve_generation_lease("img2img", coordinator=get_generation_coordinator(),
                                          cancelled=lambda: self.is_cancelled):
                # T2I 와 같은 이유로 cancel_check 를 넘긴다(모델 전환 중·발송 직후 취소)
                if generation_family == "krea2":
                    from core.krea2_generation import run_krea2_generation

                    result = call_with_optional_cancel(
                        run_krea2_generation, backend, "i2i", payload,
                        progress_callback=on_progress, cancel_check=self._cancel_check,
                    )
                else:
                    result = call_with_optional_cancel(
                        backend.img2img, self.model_name, payload,
                        progress_callback=on_progress, cancel_check=self._cancel_check,
                    )

                if self.is_cancelled:
                    self.finished.emit("생성 취소됨", {'cancelled': True})
                    return

                if not result.success:
                    self.finished.emit(result.error, {})
                    return

                final_image = result.image_data
            if self.is_cancelled:
                self.finished.emit("생성 취소됨", {'cancelled': True})
                return
            info = dict(result.info or {})
            self.finished.emit(final_image, info)

        except Exception as e:
            if self.is_cancelled:
                self.finished.emit("생성 취소됨", {'cancelled': True})
                return
            logger.exception("Img2ImgFlowWorker failed")
            self.finished.emit(f"img2img 생성 중 오류: {sanitize_for_ui(e)}", {})
