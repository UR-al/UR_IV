# workers/refine_worker.py
"""SAM3 Refine 워커 — 이미지 경로 + Refine 설정 → 재손질 결과 저장.

sam-extra 워크플로 2(Refine 패널)를 앱에서 구현한 경로. 확장의 Refine 패널은
Gradio 전용이라 HTTP로 호출할 수 없어, 같은 결과가 나오도록 앱이 직접 payload를
만들어 `/sdapi/v1/img2img` + `alwayson_scripts["SAM3 Mask"]`로 보낸다.

체인 refine(결과를 다시 입력으로) 을 지원하려면 결과 경로를 그대로 돌려주면 된다.
"""
import base64
import json
import logging
import os

from PyQt6.QtCore import QThread, pyqtSignal

from core.error_handler import sanitize_for_ui
from core.resource_coordinator import backend_job_guard, release_before_backend_job

logger = logging.getLogger(__name__)


def _output_path(src_path: str, output_folder: str = '') -> str:
    """결과 저장 경로. 같은 이름이 있으면 _2, _3... 으로 비켜간다
    (체인 refine을 반복해도 이전 결과를 덮어쓰지 않게)."""
    base_dir = output_folder or os.path.join(os.path.dirname(src_path), 'refine')
    os.makedirs(base_dir, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(src_path))
    candidate = os.path.join(base_dir, f"{name}_refine{ext}")
    index = 2
    while os.path.exists(candidate):
        candidate = os.path.join(base_dir, f"{name}_refine_{index}{ext}")
        index += 1
    return candidate


def _to_posix(path: str) -> str:
    return path.replace('\\', '/')


class RefineWorker(QThread):
    """단일 이미지 Refine"""
    finished = pyqtSignal(str)   # JSON {before, after, prompt, negative_prompt} 또는 {error}

    def __init__(self, image_path: str, settings: dict, parent=None):
        super().__init__(parent)
        self._path = image_path
        self._settings = settings or {}

    def run(self):
        try:
            from backends import get_backend
            backend = get_backend()
            if not backend:
                self.finished.emit(json.dumps({'error': '백엔드 연결 없음'}))
                return
            if not hasattr(backend, 'refine'):
                self.finished.emit(json.dumps({'error': '이 백엔드는 Refine을 지원하지 않습니다'}))
                return
            if not os.path.exists(self._path):
                self.finished.emit(json.dumps({'error': f'이미지를 찾을 수 없습니다: {self._path}'}))
                return

            # 결과 프롬프트를 UI에 그대로 보여준다 — Target 수술이 의도대로 됐는지
            # 사용자가 눈으로 확인할 수 있어야 한다(확장도 콘솔에 같은 걸 찍는다)
            from core.refine_prompt import build_refine_prompts
            preview = build_refine_prompts(
                main_prompt=self._settings.get('main_prompt', ''),
                main_negative=self._settings.get('main_negative', ''),
                target=self._settings.get('target', ''),
                replacement=self._settings.get('replacement', ''),
                negative=self._settings.get('negative', ''),
                inherit_main=bool(self._settings.get('inherit_main', True)),
                inherit_negative=bool(self._settings.get('inherit_negative', True)),
            )

            with open(self._path, 'rb') as fh:
                image_b64 = base64.b64encode(fh.read()).decode()

            # Forge img2img + SAM3 Mask 전에 앱 프로세스의 편집기 SAM3 번들(~3.4GB)을 반납
            release_before_backend_job('refine')
            # 모델 언로드(생성 후·대기열 정리·수동)와 배타 — 진행 중이면 기다리고, 도는 동안엔 언로드가 건너뛴다
            with backend_job_guard('refine'):
                result_b64 = backend.refine(image_b64, self._settings)

            out_path = _output_path(self._path, self._settings.get('output_folder', ''))
            with open(out_path, 'wb') as fh:
                fh.write(base64.b64decode(result_b64))

            self.finished.emit(json.dumps({
                'before': _to_posix(self._path),
                'after': _to_posix(out_path),
                'output_path': _to_posix(out_path),
                'prompt': preview['prompt'],
                'negative_prompt': preview['negative_prompt'],
                'detect_tokens': preview['detect_tokens'],
            }))
        except NotImplementedError as e:
            self.finished.emit(json.dumps({'error': str(e)}))
        except Exception as e:
            logger.exception("RefineWorker failed")
            self.finished.emit(json.dumps({'error': sanitize_for_ui(e)}))
