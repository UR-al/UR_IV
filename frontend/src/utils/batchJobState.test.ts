import { describe, expect, it } from 'vitest'
import { jobPercent, jobProgressText, parseBatchJobState } from './batchJobState'

describe('parseBatchJobState', () => {
  it('Python JobProgress JSON 을 읽는다', () => {
    const state = parseBatchJobState(JSON.stringify({
      job: 'upscale', total: 4, output_dir: 'C:/out/upscale', running: true,
      done: 1, success: 1, failed: 0, stopped: false,
    }))
    expect(state).toEqual({
      job: 'upscale', running: true, total: 4, done: 1, success: 1, failed: 0,
      stopped: false, output_dir: 'C:/out/upscale',
    })
  })

  it('모르는 작업·깨진 JSON 은 null', () => {
    expect(parseBatchJobState('{')).toBeNull()
    expect(parseBatchJobState(JSON.stringify({ job: 'caption' }))).toBeNull()
    expect(parseBatchJobState(null)).toBeNull()
  })

  it('숫자가 아니거나 음수인 카운트는 0', () => {
    const state = parseBatchJobState({ job: 'batch', total: 'x', done: -2, running: 'yes' })
    expect(state).toMatchObject({ total: 0, done: 0, running: false })
  })
})

describe('진행 표시', () => {
  it('퍼센트와 문구', () => {
    const state = parseBatchJobState({ job: 'batch', total: 10, done: 3, failed: 1, running: true })
    expect(jobPercent(state)).toBe(30)
    expect(jobProgressText(state)).toBe('3/10 · 실패 1')
    expect(jobPercent(null)).toBe(0)
    expect(jobProgressText(null)).toBe('')
  })
})
