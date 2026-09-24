# workers/batch_image_worker.py
"""Vue 일괄 처리(리사이즈/포맷 변환) 워커 — 처리 규칙은 ``core.batch_image_ops``.

숨은 레거시 BatchTab/``tabs.editor.batch_worker`` 는 이 워커로 대체된 뒤 은퇴해 삭제됐다
(tests/test_legacy_editor_retirement.py).
"""
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from core.batch_image_ops import BatchJob, batch_item_result
from core.path_safety import safe_input_path


class BatchImageWorker(QThread):
    """파일마다 새 결과 파일을 만든다 — 원본은 절대 덮어쓰지 않는다."""

    file_done = pyqtSignal(int, bool, str)   # index, ok, 결과 경로 또는 오류 문구
    all_done = pyqtSignal()

    def __init__(self, files: list, job: BatchJob, output_dir: str, parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.job = job
        self.output_dir = output_dir
        self._stop = threading.Event()

    def cancel(self):
        self._stop.set()

    def run(self):
        for index, path in enumerate(self.files):
            if self._stop.is_set():
                break
            # 경로 검증·처리·실패 문구 정리(파일 이름만, sanitize_for_ui)는 core 한 곳에서.
            ok, message = batch_item_result(path, self.output_dir, self.job, resolve_path=safe_input_path)
            self.file_done.emit(index, ok, message)
        self.all_done.emit()
