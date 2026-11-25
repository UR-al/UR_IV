# workers/adetailer_worker.py
"""ADetailer 단독 실행 워커 — 이미지 경로 → ADetailer 적용 → 결과 저장"""
import os
import json
import base64
import threading
from PyQt6.QtCore import QThread, pyqtSignal

from core.cancellable_call import call_with_optional_cancel
from core.image_metadata import read_applicable_prompts
from core.output_files import write_new_file
from core.path_safety import UnsafePathError, safe_input_path, safe_output_dir
from core.resource_coordinator import backend_job_guard, release_before_backend_job

ALLOWED_IMAGE_EXTS: frozenset[str] = frozenset((
    '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff', '.avif', '.apng',
))


def _get_output_path(src_path: str, output_folder: str = '') -> str:
    """AD 결과 저장 경로 생성 (항상 .png — 백엔드가 PNG 바이트를 반환)"""
    d = os.path.dirname(src_path)
    name, _ext = os.path.splitext(os.path.basename(src_path))
    output_dir = safe_output_dir(output_folder or os.path.join(d, 'adetailer'), create=True)
    return os.path.join(output_dir, f"{name}_ad.png")


def _to_posix(path: str) -> str:
    return path.replace('\\', '/')


def _prepare_settings(settings: dict, image_path: str) -> tuple[dict, str]:
    """EXIF 프롬프트 적용 → (settings, exif_warning).

    프롬프트는 core.image_metadata 한 곳에서 읽는다(네거티브 없는 A1111, JPEG/WebP
    UserComment, IDAT 뒤 텍스트, ComfyUI 그래프). 못 읽거나 모호하면 빈 프롬프트로 돌되
    경고를 결과 JSON 에 실어 BatchView 가 토스트로 알린다.
    """
    settings = dict(settings)
    warning = ''
    if settings.get('use_exif_prompt'):
        prompt, negative, warning = read_applicable_prompts(image_path)
        settings['ad_prompt'] = prompt
        settings['ad_negative'] = negative
    return settings, warning


def _with_exif_warning(result: dict, warning: str) -> dict:
    if warning:
        result['exif_warning'] = warning
    return result


def _with_notices(result: dict, result_b64) -> dict:
    """sam-extra 알림(메인 생성 설정 전달 제외·재시도, PAG 누락 등)은 결과 JSON 의 notices 로 —
    GUI 가 토스트로 띄운다(ui/sam_extra_notices_ui.relay_worker_result). ComfyUI 결과(평범한 str)는 없다."""
    from core.sam_extra_notices import notices_of, notices_to_dicts

    notices = notices_to_dicts(notices_of(result_b64))
    if notices:
        result['notices'] = notices
    return result


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

            safe_src = safe_input_path(self._path, allowed_exts=ALLOWED_IMAGE_EXTS)
            if not safe_src:
                self.finished.emit(json.dumps({'error': '유효하지 않은 입력 이미지 경로'}))
                return

            settings, exif_warning = _prepare_settings(self._settings, safe_src)

            with open(safe_src, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()

            # Forge가 ADetailer 모델을 올리기 전에 앱 프로세스의 편집기 SAM3 번들(~3.4GB)을 반납
            release_before_backend_job('adetailer')
            # 모델 언로드(생성 후·대기열 정리·수동)와 배타 — 진행 중이면 기다리고, 도는 동안엔 언로드가 건너뛴다
            with backend_job_guard('adetailer'):
                result_b64 = backend.adetailer(image_b64, settings)

            target_path = _get_output_path(safe_src, settings.get('output_folder', ''))
            output_path = write_new_file(target_path, base64.b64decode(result_b64))

            self.finished.emit(json.dumps(_with_notices(_with_exif_warning({
                'before': _to_posix(safe_src),
                'after': _to_posix(output_path),
                'output_path': _to_posix(output_path),
            }, exif_warning), result_b64), ensure_ascii=False))
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
        # 완료 알림을 실제 결과로 — 백엔드 없음·출력 폴더 오류·0장 성공 시 '완료(N장)' 오탐 방지
        from core.batch_job_state import JobProgress
        self.job = JobProgress(
            job='ADetailer 배치',
            total=len(paths),
            output_dir=str((settings or {}).get('output_folder') or '').replace('\\', '/'),
        )
        self.first_error = ''

    def stop(self):
        self._stop_event.set()

    def _record(self, ok: bool, error: str = '') -> None:
        self.job.record(ok)
        if not ok and error and not self.first_error:
            self.first_error = error

    def completion_notice(self) -> tuple[str, str]:
        """(showNotification kind, 문구) — 실제 성공·실패·건너뜀 수로 (``all_done`` 뒤 GUI 스레드에서 부른다)."""
        from core.batch_job_state import completion_notice
        return completion_notice(self.job, self.first_error)

    def run(self):
        from backends import get_backend
        backend = get_backend()
        if not backend:
            self._record(False, '백엔드 연결 없음')
            self.single_done.emit(json.dumps({'error': '백엔드 연결 없음'}))
            self.job.finish()
            self.progress.emit(0, 0)
            self.all_done.emit()
            return

        total = len(self._paths)
        explicit_out = (self._settings or {}).get('output_folder', '')
        if explicit_out:
            try:
                safe_output_dir(explicit_out, create=True)
            except UnsafePathError as e:
                message = f'유효하지 않은 출력 폴더: {e}'
                self._record(False, message)
                self.single_done.emit(json.dumps({'error': message}))
                self.job.finish()
                self.progress.emit(0, 0)
                self.all_done.emit()
                return

        for i, path in enumerate(self._paths):
            if self._stop_event.is_set():
                break
            try:
                safe_src = safe_input_path(path, allowed_exts=ALLOWED_IMAGE_EXTS)
                if not safe_src:
                    raise ValueError('유효하지 않은 입력 이미지 경로')

                settings, exif_warning = _prepare_settings(self._settings, safe_src)

                with open(safe_src, 'rb') as f:
                    image_b64 = base64.b64encode(f.read()).decode()

                # 항목마다 — 배치 도중 편집기에서 SAM3를 다시 올렸어도 Forge 작업과 겹치지 않게
                release_before_backend_job('adetailer-batch')
                with backend_job_guard('adetailer-batch'):     # 모델 언로드와 배타(항목마다)
                    # 받을 수 있는 백엔드(WebUI)에는 중지 확인을 넘긴다 — 거절된 전달 블록을 빼고 다시 보내기 전에 본다
                    result_b64 = call_with_optional_cancel(backend.adetailer, image_b64, settings,
                                                           cancel_check=self._stop_event.is_set)
                target_path = _get_output_path(safe_src, settings.get('output_folder', ''))
                output_path = write_new_file(target_path, base64.b64decode(result_b64))

                self._record(True)
                self.single_done.emit(json.dumps(_with_notices(_with_exif_warning({
                    'before': _to_posix(safe_src),
                    'after': _to_posix(output_path),
                    'output_path': _to_posix(output_path),
                    'index': i,
                }, exif_warning), result_b64), ensure_ascii=False))
            except Exception as e:
                self._record(False, str(e))
                self.single_done.emit(json.dumps({
                    'error': str(e),
                    'path': _to_posix(str(path)),
                    'index': i,
                }))

            self.progress.emit(i + 1, total)

        self.job.finish()
        if self.job.done < total:
            self.progress.emit(self.job.done, self.job.done)
        self.all_done.emit()
