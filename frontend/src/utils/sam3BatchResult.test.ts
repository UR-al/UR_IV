import { describe, expect, it } from 'vitest'
import { createSam3ErrorToasts, sam3ResultOutcome } from './sam3BatchResult'

describe('sam3ResultOutcome', () => {
  it('a single run ends with its one result, success or error', () => {
    expect(sam3ResultOutcome({ before: 'a.png', after: 'b.png' })).toMatchObject({ index: null, ok: true, endsRun: true })
    expect(sam3ResultOutcome({ error: '백엔드 연결 없음' })).toMatchObject({ index: null, ok: false, endsRun: true })
  })

  it('a failed batch item does not end the run — the worker keeps going', () => {
    const outcome = sam3ResultOutcome({ error: 'SAM3 실패 — VRAM 부족', path: 'C:/in/a.png', index: 0 })
    expect(outcome).toMatchObject({ index: 0, ok: false, endsRun: false, skipped: 0, file: 'a.png' })
    expect(sam3ResultOutcome({ before: 'x', after: 'y', index: 3 })).toMatchObject({ index: 3, ok: true, endsRun: false })
  })

  it('a batch stopped on a repeating failure ends the run and reports what was skipped', () => {
    const outcome = sam3ResultOutcome({ error: 'KeyError', path: 'a.png', index: 1, batch_stopped: true, skipped: 4 })
    expect(outcome).toMatchObject({ index: 1, ok: false, endsRun: true, skipped: 4 })
  })

  it('ignores malformed indices and flags', () => {
    expect(sam3ResultOutcome({ error: 'x', index: '2' as unknown as number }).index).toBeNull()
    expect(sam3ResultOutcome({ error: 'x', index: 2, batch_stopped: 'yes' }).endsRun).toBe(false)
    expect(sam3ResultOutcome(null)).toMatchObject({ index: null, ok: true, endsRun: true })
  })
})

describe('createSam3ErrorToasts', () => {
  it('shows each batch failure cause once per run, single errors always', () => {
    const toasts = createSam3ErrorToasts()
    const first = sam3ResultOutcome({ error: 'VRAM 부족', path: 'C:/a.png', index: 0 })
    const again = sam3ResultOutcome({ error: 'VRAM 부족', path: 'C:/b.png', index: 1 })
    expect(toasts.note(first)).toBe('SAM3 오류 · a.png: VRAM 부족')
    expect(toasts.note(again)).toBeNull()
    expect(toasts.note(sam3ResultOutcome({ before: 'c.png', after: 'd.png', index: 2 }))).toBeNull()
    expect(toasts.note(sam3ResultOutcome({ error: '연결 없음' }))).toBe('SAM3 오류: 연결 없음')
    toasts.reset()
    expect(toasts.note(again)).toBe('SAM3 오류 · b.png: VRAM 부족')
  })

  it('always announces the stop, even for a cause already shown', () => {
    const toasts = createSam3ErrorToasts()
    toasts.note(sam3ResultOutcome({ error: 'KeyError', path: 'a.png', index: 0 }))
    const stop = toasts.note(sam3ResultOutcome({ error: 'KeyError', path: 'b.png', index: 1, batch_stopped: true, skipped: 2 }))
    expect(stop).toContain('남은 2장')
  })
})
