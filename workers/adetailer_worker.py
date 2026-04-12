# workers/adetailer_worker.py
"""ADetailer 단독 실행 워커 — 이미지 경로 → ADetailer 적용 → 결과 저장"""
import os
import json
import base64
import threading
from PyQt6.QtCore import QThread, pyqtSignal


def _read_exif_prompts(image_path: str):
    """PNG EXIF에서 positive/negative prompt 추출"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        raw = img.info.get('parameters', '')
        if raw and 'Steps:' in raw:
            parts = raw.split('\nNegative prompt: ')
            prompt = parts[0].strip()
            negative = ''
            if len(parts) > 1:
                sub = parts[1].split('\nSteps: ')
                negative = sub[0].strip()
            return prompt, negative
    except Exception as e:
        print(f"[AD Worker] EXIF 읽기 실패 ({os.path.basename(image_path)}): {e}")
    return '', ''


def _get_output_path(src_path: str, output_folder: str = '') -> str:
    """AD 결과 저장 경로 생성"""
    d = os.path.dirname(src_path)
    name, ext = os.path.splitext(os.path.basename(src_path))
    output_dir = output_folder or os.path.join(d, 'adetailer')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{name}_ad{ext}")


def _to_posix(path: str) -> str:
    return path.replace('\\', '/')


def _prepare_settings(settings: dict, image_path: str) -> dict:
    """EXIF 프롬프트 적용"""
    settings = dict(settings)
    if settings.get('use_exif_prompt'):
        prompt, negative = _read_exif_prompts(image_path)
        settings['ad_prompt'] = prompt
        settings['ad_negative'] = negative
    return settings


class ADetailerSingleWorker(QThread):
    """단일 이미지 ADetailer 처리"""
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

            settings = _prepare_settings(self._settings, self._path)

            with open(self._path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()

            result_b64 = backend.adetailer(image_b64, settings)

            output_path = _get_output_path(self._path, settings.get('output_folder', ''))
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(result_b64))

            self.finished.emit(json.dumps({
                'before': _to_posix(self._path),
                'after': _to_posix(output_path),
                'output_path': _to_posix(output_path),
            }))
        except Exception as e:
            self.finished.emit(json.dumps({'error': str(e)}))


class ADetailerBatchWorker(QThread):
    """배치 이미지 ADetailer 처리"""
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
                settings = _prepare_settings(self._settings, path)

                with open(path, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode()

                result_b64 = backend.adetailer(image_b64, settings)
                output_path = _get_output_path(path, settings.get('output_folder', ''))
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(result_b64))

                self.single_done.emit(json.dumps({
                    'before': _to_posix(path),
                    'after': _to_posix(output_path),
                    'output_path': _to_posix(output_path),
                    'index': i,
                }))
            except Exception as e:
                self.single_done.emit(json.dumps({
                    'error': str(e),
                    'path': _to_posix(path),
                    'index': i,
                }))

            self.progress.emit(i + 1, total)

        self.all_done.emit()
