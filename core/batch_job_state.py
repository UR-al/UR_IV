"""Vue 일괄 처리·업스케일 작업의 진행 상태와 완료 알림 문구 (순수 로직).

``batchJobState`` 시그널 JSON:
    {job: 'batch'|'upscale', running, done, total, success, failed, output_dir}
Vue BatchView 는 이걸로 진행률을 보여 주고, 실행 중에는 시작 버튼을 막는다(재클릭으로
워커 두 개가 같은 파일을 같은 경로에 동시에 쓰던 문제).

완료 알림은 **한 번만**, 실제 결과로 정한다 — 예전에는 모든 파일이 실패해도 '완료'
토스트(+ 숨은 탭의 애플리케이션 모달)가 떴다.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

JOB_LABELS = {'batch': '배치', 'upscale': '업스케일'}


@dataclass
class JobProgress:
    job: str
    total: int
    output_dir: str = ''
    running: bool = True
    done: int = 0
    success: int = 0
    failed: int = 0
    stopped: bool = False

    def record(self, ok: bool) -> None:
        self.done += 1
        if ok:
            self.success += 1
        else:
            self.failed += 1

    def finish(self) -> None:
        self.running = False
        if self.done < self.total:
            self.stopped = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def completion_notice(progress: JobProgress, first_error: str = '') -> tuple[str, str]:
    """(showNotification kind, 문구). 성공만 → success, 섞임/중단 → warning, 전부 실패 → error."""
    label = JOB_LABELS.get(progress.job, progress.job)
    where = f' → {progress.output_dir}' if progress.output_dir else ''
    skipped = max(0, progress.total - progress.done)
    if progress.success and not progress.failed and not skipped:
        return 'success', f'{label} 완료: {progress.success}개{where}'
    if progress.success == 0:
        reason = f' — {first_error}' if first_error else ''
        if progress.done == 0:
            return 'warning', f'{label} 중단: 처리한 파일이 없습니다'
        return 'error', f'{label} 실패: {progress.failed}개 모두 실패{reason}'
    parts = [f'성공 {progress.success}']
    if progress.failed:
        parts.append(f'실패 {progress.failed}')
    if skipped:
        parts.append(f'건너뜀 {skipped}')
    reason = f' — {first_error}' if first_error and progress.failed else ''
    return 'warning', f'{label} 일부 완료: {" · ".join(parts)}{where}{reason}'
