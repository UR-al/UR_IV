"""Vue 일괄 처리(``start_batch`` — 리사이즈/포맷 변환). 숨은 레거시 BatchTab 을 거치지 않는다.

결과는 ``<출력 폴더>/batch`` 에 새 파일로만 쓰고 원본 메타데이터를 보존한다
(``core.batch_image_ops``). 실행 중 재요청 거절·진행 상태·단일 완료 알림은 ``ui.batch_jobs``.
"""
from __future__ import annotations

import os

from core.batch_image_ops import BatchJobError, batch_output_dir, parse_batch_job
from core.upscale_settings import input_files
from ui.batch_jobs import JobTracker, notify, replace_finished_worker, worker_running

WORKER_ATTR = '_vue_batch_worker'


def _output_root() -> str:
    import config
    return str(getattr(config, 'OUTPUT_DIR', '') or '')


def start_vue_batch(mw, payload: dict, *, worker_factory=None, output_root: str | None = None) -> bool:
    """Vue BatchView 페이로드로 일괄 처리를 시작한다. 시작했으면 True."""
    if worker_running(mw, WORKER_ATTR):
        notify(mw, 'warning', '배치가 이미 진행 중입니다 — 끝난 뒤 다시 시도하세요')
        return False
    files = input_files(payload or {})
    if not files:
        notify(mw, 'error', '배치: 처리할 파일이 없습니다')
        return False
    try:
        job = parse_batch_job(payload or {})
    except BatchJobError as exc:
        notify(mw, 'error', str(exc))
        return False
    out_dir = batch_output_dir(output_root if output_root is not None else _output_root())
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        notify(mw, 'error', f'배치: 출력 폴더를 만들 수 없습니다 ({exc})')
        return False

    if worker_factory is None:
        from workers.batch_image_worker import BatchImageWorker
        worker_factory = BatchImageWorker
    worker = worker_factory(files, job, out_dir)
    tracker = JobTracker(mw, 'batch', len(files), out_dir)
    worker.file_done.connect(lambda _index, ok, message: tracker.on_file(ok, message))
    worker.all_done.connect(tracker.on_done)
    replace_finished_worker(mw, WORKER_ATTR, worker)
    worker.start()
    tracker.emit_state()
    notify(mw, 'info', f'배치 시작: {len(files)}개 → {tracker.progress.output_dir}')
    return True
