# workers/sam3_worker.py
"""SAM3 단독 실행 워커 — 이미지 경로 → SAM3 적용 → 결과 저장"""
import os
import json
import base64
import logging
import threading
from PyQt6.QtCore import QThread, pyqtSignal

from core.error_handler import sanitize_for_ui
from core.image_metadata import read_applicable_prompts
from core.resource_coordinator import backend_job_guard, release_before_backend_job

logger = logging.getLogger(__name__)


def _get_output_path(src_path: str, output_folder: str = '') -> str:
    """SAM3 결과 저장 경로 생성"""
    d = os.path.dirname(src_path)
    name, ext = os.path.splitext(os.path.basename(src_path))
    output_dir = output_folder or os.path.join(d, 'sam3')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{name}_sam3{ext}")


def _to_posix(path: str) -> str:
    return path.replace('\\', '/')


def _prepare_settings(settings: dict, image_path: str) -> tuple[dict, str]:
    """EXIF 프롬프트 적용 → (settings, exif_warning).

    프롬프트는 core.image_metadata 한 곳에서 읽는다(네거티브 없는 A1111, JPEG/WebP
    UserComment, IDAT 뒤 텍스트, ComfyUI 그래프 역할 판정). 읽지 못했거나 ComfyUI 그래프가
    모호하면 추측하지 않고 경고를 결과 JSON 으로 돌려준다 — BatchView 가 토스트로 알린다.
    """
    settings = dict(settings)
    settings.setdefault('sam3_mode', 'Inpaint')
    settings.setdefault('sam3_mask_mode', 'Individual')
    settings.setdefault('sam3_prompt', 'face')
    warning = ''
    if settings.get('use_exif_prompt'):
        prompt, negative, warning = read_applicable_prompts(image_path)
        if prompt and not settings.get('sam3_inpaint_prompt'):
            settings['sam3_inpaint_prompt'] = prompt
        if negative and not settings.get('sam3_negative_prompt'):
            settings['sam3_negative_prompt'] = negative
    return settings, warning


def _with_exif_warning(result: dict, warning: str) -> dict:
    if warning:
        result['exif_warning'] = warning
    return result


class Sam3SingleWorker(QThread):
    """단일 이미지 SAM3 처리"""
    finished = pyqtSignal(str)

    def __init__(self, image_path: str, settings: dict, parent=None):
        super().__init__(parent)
        self._path = image_path
        self._settings = settings

    def run(self):
        try:
            from backends import get_backend
            backend = get_backend()
            if not backend:
                self.finished.emit(json.dumps({'error': '백엔드 연결 없음'}))
                return

            settings, exif_warning = _prepare_settings(self._settings, self._path)

            with open(self._path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()

            # Forge가 SAM3를 올리기 전에 앱 프로세스의 편집기 SAM3 번들(~3.4GB)을 반납
            release_before_backend_job('sam3')
            # 모델 언로드(생성 후·대기열 정리·수동)와 배타 — 진행 중이면 기다리고, 도는 동안엔 언로드가 건너뛴다
            with backend_job_guard('sam3'):
                result_b64 = backend.sam3(image_b64, settings)

            output_path = _get_output_path(self._path, settings.get('output_folder', ''))
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(result_b64))

            self.finished.emit(json.dumps(_with_exif_warning({
                'before': _to_posix(self._path),
                'after': _to_posix(output_path),
                'output_path': _to_posix(output_path),
            }, exif_warning), ensure_ascii=False))
        except Exception as e:
            logger.exception("Sam3SingleWorker failed")
            self.finished.emit(json.dumps({'error': sanitize_for_ui(e)}))


class Sam3BatchWorker(QThread):
    """배치 이미지 SAM3 처리"""
    progress = pyqtSignal(int, int)
    single_done = pyqtSignal(str)
    all_done = pyqtSignal()

    def __init__(self, paths: list, settings: dict, parent=None):
        super().__init__(parent)
        self._paths = paths
        self._settings = settings
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        from backends import get_backend
        backend = get_backend()
        if not backend:
            self.single_done.emit(json.dumps({'error': '백엔드 연결 없음'}))
            return

        total = len(self._paths)
        for i, path in enumerate(self._paths):
            if self._stop_event.is_set():
                break
            try:
                settings, exif_warning = _prepare_settings(self._settings, path)

                with open(path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode()

                # 항목마다 — 배치 도중 편집기에서 SAM3를 다시 올렸어도 Forge 작업과 겹치지 않게
                release_before_backend_job('sam3-batch')
                with backend_job_guard('sam3-batch'):          # 모델 언로드와 배타(항목마다)
                    result_b64 = backend.sam3(image_b64, settings)
                output_path = _get_output_path(path, settings.get('output_folder', ''))
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(result_b64))

                self.single_done.emit(json.dumps(_with_exif_warning({
                    'before': _to_posix(path),
                    'after': _to_posix(output_path),
                    'output_path': _to_posix(output_path),
                    'index': i,
                }, exif_warning), ensure_ascii=False))
            except Exception as e:
                logger.exception("Sam3BatchWorker failed for %s", os.path.basename(path))
                self.single_done.emit(json.dumps({
                    'error': sanitize_for_ui(e),
                    'path': _to_posix(path),
                    'index': i,
                }))

            self.progress.emit(i + 1, total)

        self.all_done.emit()
