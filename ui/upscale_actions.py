"""Vue 업스케일(``start_upscale``). 숨은 레거시 UpscaleTab 을 거치지 않는다.

예전 경로는 숨은 탭의 채워진 적 없는 콤보·존재한 적 없는 ``spin_scale`` 을 거쳐 늘
``upscaler_name='(로드 필요)'``·2배로 나갔고, 모든 파일이 실패해도 성공 토스트와 모달이
떴다. 이제 설정은 Vue 페이로드만으로 만들고(``core.upscale_settings``), 결과는
``<출력 폴더>/upscale`` 에 새 파일로 쓰며, 알림은 실제 성공/실패 수로 한 번만 낸다.
"""
from __future__ import annotations

import os

from core.upscale_settings import build_upscale_settings, input_files
from ui.batch_jobs import JobTracker, notify, replace_finished_worker, worker_running

WORKER_ATTR = '_vue_upscale_worker'
UPSCALE_SUBDIR = 'upscale'


def _output_root() -> str:
    import config
    return str(getattr(config, 'OUTPUT_DIR', '') or '')


def start_vue_upscale(mw, payload: dict, *, worker_factory=None, output_root: str | None = None) -> bool:
    """Vue BatchView 업스케일 페이로드 ``{files, upscaler, scale}`` 로 시작한다. 시작했으면 True."""
    if worker_running(mw, WORKER_ATTR):
        notify(mw, 'warning', '업스케일이 이미 진행 중입니다 — 끝난 뒤 다시 시도하세요')
        return False
    files = input_files(payload or {})
    if not files:
        notify(mw, 'error', '업스케일: 처리할 파일이 없습니다')
        return False
    out_dir = os.path.join(output_root if output_root is not None else _output_root(), UPSCALE_SUBDIR)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        notify(mw, 'error', f'업스케일: 출력 폴더를 만들 수 없습니다 ({exc})')
        return False
    settings = build_upscale_settings(payload or {}, out_dir)

    if worker_factory is None:
        from workers.upscale_worker import BatchUpscaleWorker
        worker_factory = BatchUpscaleWorker
    worker = worker_factory(files, settings)
    tracker = JobTracker(mw, 'upscale', len(files), out_dir)
    worker.single_finished.connect(lambda _index, ok, message: tracker.on_file(ok, message))
    worker.all_finished.connect(tracker.on_done)
    replace_finished_worker(mw, WORKER_ATTR, worker)
    worker.start()
    tracker.emit_state()
    notify(mw, 'info', f"업스케일 시작: {len(files)}개 · {settings['upscaler_name']} "
                       f"{settings['scale_factor']:g}x")
    return True
