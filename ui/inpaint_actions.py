"""Vue 인페인트(``generate_inpaint``) — 숨은 레거시 InpaintTab 을 거치지 않는 Qt 접착층.

요청 전체는 ``core.inpaint_payload.build_inpaint_request`` 가 Vue 페이로드만 보고 만든다
(마스크 초기값·인페인트 범위·steps/cfg/seed/negative·실제 이미지 크기). 여기서는
  1) T2I 프롬프트 폴백과 확장(NegPiP/ADetailer/SAM3/Anima — ``apply_alwayson_extensions``)을
     얹고,
  2) ``Img2ImgFlowWorker`` 를 창에 붙잡아 두고(GC 크래시 방지) 실행 중이면 새 요청을 거절하고,
  3) 결과를 출력 폴더에 새 파일로 저장해 Vue 히스토리로 보낸다.
"""
from __future__ import annotations

import os
import random
import time

from core.image_payload import ImagePayloadError
from core.inpaint_payload import build_inpaint_request
from core.output_files import write_new_file
from core.result_image import result_image_size
from utils.app_logger import get_logger

logger = get_logger('inpaint')

WORKER_ATTR = '_vue_inpaint_worker'


def _notify(mw, kind: str, message: str) -> None:
    bridge = getattr(mw, 'vue_bridge', None)
    if bridge is None:
        return
    try:
        bridge.showNotification.emit(kind, message)
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
    worker = getattr(mw, WORKER_ATTR, None)
    try:
        return bool(worker is not None and worker.isRunning())
    except RuntimeError:
        return False


def start_vue_inpaint(mw, payload: dict, *, worker_factory=None) -> bool:
    """Vue InpaintView 페이로드로 인페인트를 시작한다. 시작했으면 True."""
    if is_running(mw):
        _notify(mw, 'warning', '인페인트가 이미 진행 중입니다 — 끝난 뒤 다시 시도하세요')
        return False
    try:
        request = build_inpaint_request(
            payload or {},
            main_prompt=_plain_text(getattr(mw, 'total_prompt_display', None)),
            main_negative=_plain_text(getattr(mw, 'neg_prompt_text', None)),
        )
    except ImagePayloadError as exc:
        _notify(mw, 'error', str(exc))
        return False

    backend_payload = request.payload
    # 확장(NegPiP/ADetailer/SAM3/Anima Guidance) — Forge Neo i2i 탭과 같게 alwayson 으로.
    # 확장 적용 실패가 인페인트 자체를 막지는 않는다(레거시 탭과 같은 규칙).
    apply_extensions = getattr(mw, 'apply_alwayson_extensions', None)
    if callable(apply_extensions):
        try:
            apply_extensions(backend_payload)
        except Exception as exc:
            logger.warning("Inpaint 확장 적용 실패 (확장 없이 진행): %s", exc)
    # sam-extra 생성 전 경고(CFG≈1 의 SMC/APG/CWM, sam3.pt 자동 다운로드, 없는 스크립트) — 막지는 않는다
    from ui.sam_extra_notices_ui import check_before_generation
    check_before_generation(mw, backend_payload)

    model_name = ''
    combo = getattr(mw, 'model_combo', None)
    if combo is not None:
        try:
            model_name = combo.currentText()
        except Exception:
            model_name = ''

    if worker_factory is None:
        from workers.generation_worker import Img2ImgFlowWorker
        worker_factory = Img2ImgFlowWorker
    worker = worker_factory(model_name, backend_payload)
    setattr(mw, WORKER_ATTR, worker)
    worker.finished.connect(lambda result, info, w=worker: _on_finished(mw, w, result, info))
    worker.start()
    _notify(mw, 'info', f'인페인트 생성 중… ({request.width}×{request.height})')
    return True


def save_inpaint_result(data: bytes, output_dir: str) -> str:
    """결과 바이트를 출력 폴더에 새 파일로 쓴다(덮어쓰지 않음). 백엔드가 보낸 바이트를
    그대로 쓰므로 Forge/ComfyUI 가 넣은 생성 메타데이터가 남는다."""
    name = f"inpaint_{int(time.time())}_{random.randint(100, 999)}.png"
    return write_new_file(os.path.join(output_dir, name), data)


def _on_finished(mw, worker, result, gen_info) -> None:
    # 워커 참조는 여기서 끊지 않는다 — 이 시그널은 run() 안에서 나오므로 스레드가 아직
    # 돌고 있다. 참조가 사라져 GC 되면 'QThread destroyed while running' 으로 죽는다.
    # 다음 요청은 isRunning() 으로 판정하고, 그때 끝난 워커를 교체한다.
    info = gen_info if isinstance(gen_info, dict) else {}
    if isinstance(result, (bytes, bytearray)) and result:
        try:
            filepath = save_inpaint_result(bytes(result), _output_dir())
        except OSError as exc:
            logger.error("inpaint result save failed: %s", exc)
            _notify(mw, 'error', f'인페인트 결과 저장 실패: {exc}')
            return
        # 해상도는 헤더만 읽는다(요청 크기는 폴백). 결과는 Vue 히스토리로만 간다 — 숨은 PyQt
        # 갤러리(add_image_to_gallery → 풀해상도 QPixmap 을 최대 100장 들고 있던 ThumbnailItem)는 은퇴했다.
        width, height = result_image_size(result, info)
        bridge = getattr(mw, 'vue_bridge', None)
        if bridge is not None:
            try:
                bridge.send_image(filepath, width, height, info.get('seed', -1))
            except Exception as exc:
                logger.warning("inpaint send_image failed: %s", exc)
        _notify(mw, 'success', '인페인트 생성 완료')
        from ui.sam_extra_notices_ui import show_result_notices
        show_result_notices(mw, info)   # 'SAM3 Error'·'Anima38: off'·PAG 누락
        return
    if info.get('cancelled'):
        _notify(mw, 'info', '인페인트가 취소되었습니다')
        return
    _notify(mw, 'error', f'인페인트 실패: {result}')
