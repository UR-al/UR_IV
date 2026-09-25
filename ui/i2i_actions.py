"""Vue I2I(``generate_i2i``) — 은퇴한 숨은 PyQt Img2ImgTab 을 대신하는 Qt 접착층.

요청 전체는 ``core.i2i_payload.build_i2i_request`` 가 Vue 페이로드만 보고 만든다(이미지·참조
디코딩, 크기, 프롬프트 폴백, denoise/steps/cfg/seed/resize, Krea2 필드). 여기서는
  1) T2I 프롬프트 폴백 값과 확장(NegPiP/ADetailer/SAM3/Anima — ``apply_alwayson_extensions``)을
     얹고,
  2) ``Img2ImgFlowWorker`` 를 창에 붙잡아 두고(GC 크래시 방지) 실행 중이면 새 요청을 거절하고
     (예전 탭은 실행 중 워커의 신호를 끊고 GUI 를 최대 2초 막은 뒤 참조를 버려, 앞 결과가 사라지고
     실행 중 QThread 가 GC 될 수 있었다),
  3) 결과를 출력 폴더에 새 파일로 저장해 Vue 히스토리로 보낸다.
인페인트(``ui/inpaint_actions.py``)와 같은 구조다.

거절만 있고 멈출 길이 없으면 멈춘 백엔드 호출 하나가 이후 I2I 를 백엔드 타임아웃(Forge img2img
600초)까지 막는다. 그래서 진행 상태를 ``i2iJobState`` ({running, cancelling}) 로 I2IView 에 보내고,
화면의 취소 버튼이 ``cancel_i2i`` → ``cancel_vue_i2i`` 로 워커를 취소한다(플래그 + 백엔드 interrupt —
T2I 취소와 같은 ``_CancellableMixin``). 입력 이미지의 PNG 재압축은 워커가 ``prepare_payload`` 로 한다.
"""
from __future__ import annotations

import json
import os
import random
import time

from core.i2i_payload import FAMILY_KREA2, build_i2i_request
from core.image_payload import ImagePayloadError
from core.output_files import write_new_file
from core.result_image import result_image_size
from ui.batch_jobs import notify, worker_running
from utils.app_logger import get_logger

logger = get_logger('i2i')

WORKER_ATTR = '_vue_i2i_worker'
JOB_STATE_SIGNAL = 'i2iJobState'
# finished 를 보낸 뒤 run() 이 돌아오기까지 기다리는 한도(보통 즉시) — 그 전에 참조를 버리면
# 'QThread destroyed while running' 으로 죽는다.
FINISHING_WAIT_MS = 2000
_DONE_FLAG = '_i2i_finished'
_CANCEL_FLAG = '_i2i_cancel_requested'

BUSY_MESSAGE = 'I2I 생성이 이미 진행 중입니다 — 끝나길 기다리거나 I2I 화면의 취소 버튼으로 멈추세요'
FINISHING_MESSAGE = '이전 I2I 작업을 마무리하는 중입니다 — 잠시 후 다시 시도하세요'


def _job_state_json(running: bool, cancelling: bool = False) -> str:
    """``i2iJobState`` 페이로드 — frontend/src/utils/i2iJobState.ts 가 읽는다."""
    return json.dumps({'running': bool(running), 'cancelling': bool(running and cancelling)})


def emit_job_state(mw, running: bool, cancelling: bool = False) -> None:
    bridge = getattr(mw, 'vue_bridge', None)
    signal = getattr(bridge, JOB_STATE_SIGNAL, None)
    if signal is None:
        return
    try:
        signal.emit(_job_state_json(running, cancelling))
    except Exception:
        pass


def _plain_text(widget) -> str:
    try:
        return widget.toPlainText() if widget is not None else ''
    except Exception:
        return ''


def _output_dir() -> str:
    import config
    return str(getattr(config, 'OUTPUT_DIR', '') or '')


def is_running(mw) -> bool:
    """생성이 아직 끝나지 않았는지 — finished 를 이미 보낸 워커(run() 이 돌아오는 중)는 끝난 것으로 본다.

    isRunning() 만 보면 finished 직후 잠깐 True 라, 그 사이 요청이 거절되면서 화면이 다시
    '실행 중'으로 바뀐 채 풀리지 않는다(끝 알림은 이미 지나갔다).
    """
    if not worker_running(mw, WORKER_ATTR):
        return False
    return not getattr(getattr(mw, WORKER_ATTR, None), _DONE_FLAG, False)


def _cancel_requested(mw) -> bool:
    return bool(getattr(getattr(mw, WORKER_ATTR, None), _CANCEL_FLAG, False))


def _release_finished_worker(mw) -> bool:
    """finished 를 보낸 이전 워커의 run() 이 돌아오길 잠깐 기다린다. 끝났으면(또는 없으면) True.

    참조를 새 워커로 바꾸기 전에 해야 한다 — 실행 중 QThread 가 GC 되면 앱이 죽는다.
    """
    previous = getattr(mw, WORKER_ATTR, None)
    if previous is None:
        return True
    try:
        if not previous.isRunning():
            return True
        return bool(previous.wait(FINISHING_WAIT_MS))
    except RuntimeError:   # C++ 객체가 이미 지워짐
        return True


def cancel_vue_i2i(mw) -> bool:
    """I2IView 취소 버튼(``cancel_i2i``) — 진행 중인 I2I 워커를 취소한다. 취소를 요청했으면 True.

    ``cancel()`` 은 플래그 + 백엔드 interrupt 다(T2I 취소와 같다). 워커가 곧 cancelled 로 끝나면
    ``_on_finished`` 가 상태를 풀고 '취소되었습니다'를 알린다.
    """
    worker = getattr(mw, WORKER_ATTR, None)
    if not is_running(mw):
        emit_job_state(mw, False)   # 화면이 옛 상태를 들고 있었으면 맞춘다
        notify(mw, 'info', '취소할 I2I 생성이 없습니다')
        return False
    cancel = getattr(worker, 'cancel', None)
    if callable(cancel):
        try:
            cancel()
        except Exception as exc:
            logger.warning("i2i cancel failed: %s", exc)
            notify(mw, 'error', f'I2I 생성을 취소하지 못했습니다: {exc}')
            return False
    try:
        setattr(worker, _CANCEL_FLAG, True)
    except Exception:
        pass
    emit_job_state(mw, True, cancelling=True)
    notify(mw, 'info', 'I2I 생성을 취소하는 중…')
    return True


def _model_name(mw) -> str:
    combo = getattr(mw, 'model_combo', None)
    if combo is None:
        return ''
    try:
        return combo.currentText()
    except Exception:
        return ''


def start_vue_i2i(mw, payload: dict, *, worker_factory=None) -> bool:
    """Vue I2IView 페이로드로 img2img(또는 Krea2 아이덴티티 편집)를 시작한다. 시작했으면 True."""
    if is_running(mw):
        # 화면이 상태를 놓쳤어도(새로고침·웹 재접속) 취소 버튼이 보이게 다시 알린다
        emit_job_state(mw, True, cancelling=_cancel_requested(mw))
        notify(mw, 'warning', BUSY_MESSAGE)
        return False
    if not _release_finished_worker(mw):
        # 끝 알림은 보냈는데 run() 이 한도 안에 돌아오지 않았다 — 참조를 버리지 않고 거절한다.
        # 작업은 끝났으므로 화면은 '실행 중'으로 묶지 않는다(풀어 줄 알림이 다시 오지 않는다).
        emit_job_state(mw, False)
        notify(mw, 'warning', FINISHING_MESSAGE)
        return False
    try:
        request = build_i2i_request(
            payload or {},
            main_prompt=_plain_text(getattr(mw, 'total_prompt_display', None)),
            main_negative=_plain_text(getattr(mw, 'neg_prompt_text', None)),
        )
    except ImagePayloadError as exc:
        notify(mw, 'error', str(exc))
        return False

    backend_payload = request.payload
    # 확장(NegPiP/ADetailer/SAM3/Anima Guidance) — Forge Neo 와 같게 img2img 에도 alwayson_scripts 로.
    # 확장 적용 실패가 I2I 생성 자체를 막지는 않는다(예전 탭과 같은 규칙).
    apply_extensions = getattr(mw, 'apply_alwayson_extensions', None)
    if callable(apply_extensions):
        try:
            apply_extensions(backend_payload)
        except Exception as exc:
            logger.warning("I2I 확장 적용 실패 (확장 없이 진행): %s", exc)

    if worker_factory is None:
        from workers.generation_worker import Img2ImgFlowWorker
        worker_factory = Img2ImgFlowWorker
    # 입력 이미지의 PNG 재압축(필요할 때만)은 워커 스레드가 백엔드 호출 전에 한다
    worker = worker_factory(_model_name(mw), backend_payload,
                            prepare=request.prepare_payload if request.needs_prepare else None)
    setattr(mw, WORKER_ATTR, worker)
    worker.finished.connect(lambda result, info, w=worker: _on_finished(mw, w, result, info))
    worker.start()
    emit_job_state(mw, True)
    kind = 'Krea2 아이덴티티 편집' if request.family == FAMILY_KREA2 else 'I2I 생성'
    notify(mw, 'info', f'{kind} 중… ({request.width}×{request.height})')
    return True


def save_i2i_result(data: bytes, output_dir: str) -> str:
    """결과 바이트를 출력 폴더에 새 파일로 쓴다(덮어쓰지 않음). 백엔드가 보낸 바이트를 그대로 쓰므로
    Forge/ComfyUI 가 넣은 생성 메타데이터가 남는다."""
    name = f"i2i_{int(time.time())}_{random.randint(100, 999)}.png"
    return write_new_file(os.path.join(output_dir, name), data)


def _on_finished(mw, worker, result, gen_info) -> None:
    # 워커 참조는 여기서 끊지 않는다 — 이 시그널은 run() 안에서 나오므로 스레드가 아직 돌고 있다.
    # 끝났다는 표시만 남기고, 다음 요청이 _release_finished_worker 로 run() 복귀를 기다린 뒤 교체한다.
    try:
        setattr(worker, _DONE_FLAG, True)
    except Exception:
        pass
    if getattr(mw, WORKER_ATTR, None) is worker:
        emit_job_state(mw, False)
    info = gen_info if isinstance(gen_info, dict) else {}
    if isinstance(result, (bytes, bytearray)) and result:
        try:
            filepath = save_i2i_result(bytes(result), _output_dir())
        except OSError as exc:
            logger.error("i2i result save failed: %s", exc)
            notify(mw, 'error', f'I2I 결과 저장 실패: {exc}')
            return
        # 해상도는 결과 헤더(요청 크기는 폴백, core.result_image). 결과는 Vue 히스토리로만 간다.
        width, height = result_image_size(result, info)
        bridge = getattr(mw, 'vue_bridge', None)
        if bridge is not None:
            try:
                bridge.send_image(filepath, width, height, info.get('seed', -1))
            except Exception as exc:
                logger.warning("i2i send_image failed: %s", exc)
        notify(mw, 'success', 'I2I 생성 완료')
        return
    if info.get('cancelled'):
        notify(mw, 'info', 'I2I 생성이 취소되었습니다')
        return
    notify(mw, 'error', f'I2I 생성 실패: {result}')
