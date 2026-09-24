/**
 * `batchJobState` 이벤트(Vue 일괄 처리·업스케일 진행) 파서 — 순수 로직.
 * Python: core/batch_job_state.JobProgress.to_json()
 *   {job, total, output_dir, running, done, success, failed, stopped}
 */
import type { BatchJobStatePayload } from '../types/bridge'

export type BatchJobKind = 'batch' | 'upscale'

function count(value: unknown): number {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0
}

export function parseBatchJobState(raw: unknown): BatchJobStatePayload | null {
  let data: any = raw
  if (typeof raw === 'string') {
    try { data = JSON.parse(raw) } catch { return null }
  }
  if (!data || typeof data !== 'object') return null
  if (data.job !== 'batch' && data.job !== 'upscale') return null
  return {
    job: data.job,
    running: data.running === true,
    total: count(data.total),
    done: count(data.done),
    success: count(data.success),
    failed: count(data.failed),
    stopped: data.stopped === true,
    output_dir: typeof data.output_dir === 'string' ? data.output_dir : '',
  }
}

/** 진행률 0~100 (total 이 0 이면 0). */
export function jobPercent(state: BatchJobStatePayload | null | undefined): number {
  if (!state || state.total <= 0) return 0
  return Math.min(100, Math.round((state.done / state.total) * 100))
}

/** 진행 줄 문구 — '3/10 · 실패 1'. */
export function jobProgressText(state: BatchJobStatePayload | null | undefined): string {
  if (!state) return ''
  const parts = [`${state.done}/${state.total}`]
  if (state.failed) parts.push(`실패 ${state.failed}`)
  return parts.join(' · ')
}
