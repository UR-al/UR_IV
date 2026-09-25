/**
 * `i2iJobState` 이벤트(Vue I2I 진행) 파서와 버튼 문구 — 순수 로직.
 * Python: ui/i2i_actions.emit_job_state → {running, cancelling}
 *
 * 실행 중 재요청은 Python 이 거절한다. 예전엔 화면이 실행 여부를 몰라 시작 버튼이 늘 눌렸고
 * 취소할 길도 없어서, 멈춘 백엔드 호출 하나가 이후 I2I 를 백엔드 타임아웃(600초)까지 막았다.
 */
import type { I2IJobStatePayload } from '../types/bridge'

export const IDLE_I2I_JOB: I2IJobStatePayload = Object.freeze({ running: false, cancelling: false })

/** JSON 문자열/객체 → 상태. 깨진 값은 null(상태를 바꾸지 않는다). */
export function parseI2IJobState(raw: unknown): I2IJobStatePayload | null {
  let data: any = raw
  if (typeof raw === 'string') {
    try { data = JSON.parse(raw) } catch { return null }
  }
  if (!data || typeof data !== 'object' || typeof data.running !== 'boolean') return null
  const running = data.running
  return { running, cancelling: running && data.cancelling === true }
}

/** 시작 버튼 문구 — 이미지가 없으면 안내, 실행 중이면 진행 표시. */
export function i2iGenerateLabel(state: I2IJobStatePayload, hasImage: boolean, isKrea2: boolean): string {
  if (state.cancelling) return '취소하는 중…'
  if (state.running) return isKrea2 ? 'Krea2 아이덴티티 편집 중…' : 'I2I 생성 중…'
  if (!hasImage) return '이미지를 먼저 올리세요'
  return isKrea2 ? 'Krea2 아이덴티티 편집 시작' : 'I2I 생성 시작'
}
