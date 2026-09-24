"""Vue 일괄 처리 탭의 장기 작업(배치·업스케일) 공용 Qt 접착층.

* 워커를 창 속성에 붙잡아 둔다 — 참조가 끊겨 실행 중인 QThread 가 GC 되면 앱이 죽는다.
* 실행 중이면 새 요청을 거절한다(재클릭으로 워커 두 개가 같은 파일을 같은 경로에 쓰던 문제).
* 진행 상태를 ``batchJobState`` 로 보내고, 끝나면 **한 번만** 실제 결과로 알린다
  (``core.batch_job_state``). 숨은 레거시 탭의 QMessageBox 모달은 쓰지 않는다.
"""
from __future__ import annotations

from core.batch_job_state import JobProgress, completion_notice


def notify(mw, kind: str, message: str) -> None:
    bridge = getattr(mw, 'vue_bridge', None)
    if bridge is None:
        return
    try:
        bridge.showNotification.emit(kind, message)
    except Exception:
        pass


def worker_running(mw, attr: str) -> bool:
    worker = getattr(mw, attr, None)
    try:
        return bool(worker is not None and worker.isRunning())
    except RuntimeError:   # C++ 객체가 이미 지워짐
        return False


def replace_finished_worker(mw, attr: str, worker) -> None:
    """끝난 이전 워커의 시그널을 끊고 새 워커로 교체한다(실행 중이면 호출하지 말 것)."""
    previous = getattr(mw, attr, None)
    if previous is not None and previous is not worker:
        for name in ('progress', 'single_finished', 'all_finished', 'file_done', 'all_done'):
            signal = getattr(previous, name, None)
            if signal is None:
                continue
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass
    setattr(mw, attr, worker)


class JobTracker:
    """파일별 결과를 모아 ``batchJobState`` 를 보내고 완료 알림을 한 번 낸다."""

    def __init__(self, mw, job: str, total: int, output_dir: str):
        self._mw = mw
        self.progress = JobProgress(job=job, total=int(total), output_dir=output_dir.replace('\\', '/'))
        self.first_error = ''
        self._finished = False

    def emit_state(self) -> None:
        bridge = getattr(self._mw, 'vue_bridge', None)
        signal = getattr(bridge, 'batchJobState', None)
        if signal is None:
            return
        try:
            signal.emit(self.progress.to_json())
        except Exception:
            pass

    def on_file(self, ok: bool, message: str = '') -> None:
        if self._finished:
            return
        self.progress.record(bool(ok))
        if not ok and not self.first_error and message:
            self.first_error = str(message)
        self.emit_state()

    def on_done(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.progress.finish()
        self.emit_state()
        kind, message = completion_notice(self.progress, self.first_error)
        notify(self._mw, kind, message)
