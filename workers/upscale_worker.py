# workers/upscale_worker.py
import os
import base64
import logging
from PyQt6.QtCore import QThread, pyqtSignal
from backends import get_backend
from core.path_safety import safe_input_path, safe_output_dir, UnsafePathError
from core.resource_coordinator import backend_job_guard, release_before_backend_job
from core.error_handler import sanitize_for_ui
from core.output_files import write_new_file
# ADetailer 슬롯 본문은 Qt 의존 없는 core/adetailer_args 한 벌이다(WebUIBackend 폴백은
# core.adetailer_args.slot_from_settings 를 직접 부른다). 옛 _build_adetailer_slot 별칭은 두지 않는다.

logger = logging.getLogger(__name__)


def _image_to_base64(image_path: str) -> str:
    """이미지 파일을 base64 문자열로 변환"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class BatchUpscaleWorker(QThread):
    """배치 업스케일/ADetailer/SAM3 워커"""
    single_finished = pyqtSignal(int, bool, str)  # index, success, message
    all_finished = pyqtSignal()
    progress = pyqtSignal(int, int)  # current, total

    def __init__(self, image_paths: list, settings: dict):
        """
        settings = {
            'mode': 'upscale_only' | 'adetailer_only' | 'sam3_only' | 'both',
            'upscaler_name': str,
            'scale_mode': 'factor' | 'size',
            'scale_factor': float,
            'target_width': int,
            'target_height': int,
            'ad_model': str,
            'ad_confidence': float,
            'ad_denoise': float,
            'ad_prompt': str,
            'output_folder': str,
        }
        """
        super().__init__()
        self.image_paths = image_paths
        self.settings = settings
        self._stop_requested = False

    def run(self):
        """배치 처리 실행"""
        total = len(self.image_paths)

        try:
            output_folder = safe_output_dir(self.settings.get('output_folder', ''), create=True)
        except UnsafePathError as e:
            logger.error("batch upscale: invalid output_folder: %s", e)
            for i in range(total):
                self.single_finished.emit(i, False, "출력 폴더가 유효하지 않습니다")
            self.all_finished.emit()
            return

        for i, path in enumerate(self.image_paths):
            if self._stop_requested:
                break

            self.progress.emit(i, total)
            try:
                safe_src = safe_input_path(path)
                if not safe_src:
                    raise ValueError("유효하지 않은 입력 이미지 경로")

                b64_image = _image_to_base64(safe_src)
                result_b64 = b64_image
                mode = self.settings['mode']
                backend = get_backend()
                applied_steps = []

                # 항목마다 Forge 작업(업스케일·ADetailer·SAM3) 전에 앱 프로세스의
                # 편집기 SAM3 번들(~3.4GB)을 반납 — 배치 도중 편집기에서 다시 올렸어도 겹치지 않게
                release_before_backend_job('batch-upscale')

                # 항목의 Forge 작업 전체를 모델 언로드(생성 후·대기열 정리·수동)와 배타로 —
                # 진행 중인 언로드는 기다리고, 도는 동안엔 언로드가 건너뛴다
                with backend_job_guard('batch-upscale'):
                    # 업스케일
                    if mode in ('upscale_only', 'both'):
                        result_b64 = backend.upscale(result_b64, self.settings)
                        applied_steps.append('upscaled')

                    # ADetailer
                    if mode in ('adetailer_only', 'both') and self.settings.get('ad_enabled', True):
                        result_b64 = backend.adetailer(result_b64, self.settings)
                        applied_steps.append('ad')

                    # SAM3
                    if mode in ('sam3_only', 'both') and self.settings.get('sam3_enabled', True):
                        result_b64 = backend.sam3(result_b64, self.settings)
                        applied_steps.append('sam3')

                # 저장 (output_folder 아래로만 허용). 같은 이름의 이전 결과·원본을
                # 덮어쓰지 않고 _2, _3 … 새 파일로 쓴다.
                basename = os.path.splitext(os.path.basename(safe_src))[0]
                suffix = "_" + "_".join(applied_steps) if applied_steps else "_result"
                output_path = write_new_file(
                    os.path.join(output_folder, f"{basename}{suffix}.png"),
                    base64.b64decode(result_b64),
                )

                self.single_finished.emit(i, True, os.path.basename(output_path))

            except Exception as e:
                logger.warning("upscale failed for %s: %s", path, e)
                # 사용자에게 실제 원인을 보여 준다(경로·토큰은 sanitize_for_ui 가 가린다).
                self.single_finished.emit(i, False, sanitize_for_ui(str(e)) or "처리 실패 (로그 참조)")

        self.progress.emit(total, total)
        self.all_finished.emit()

    def request_stop(self):
        """처리 중지 요청"""
        self._stop_requested = True
