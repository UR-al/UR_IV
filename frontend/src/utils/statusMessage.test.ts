import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createStatusLine, parseStatusMessage, statusRemainingMs, type StatusMessage } from './statusMessage'

describe('parseStatusMessage', () => {
  it('reads the Python payload', () => {
    expect(parseStatusMessage('{"text":"✅ 설정 저장","level":"success","timeoutMs":3000,"at":1000}'))
      .toEqual({ text: '✅ 설정 저장', level: 'success', timeoutMs: 3000, at: 1000 })
  })

  it('falls back to info, sticky and unknown time for odd fields', () => {
    expect(parseStatusMessage({ text: ' 진행 중 ', level: 'loud', timeoutMs: -5 }))
      .toEqual({ text: '진행 중', level: 'info', timeoutMs: 0, at: 0 })
  })

  it('rejects empty, broken and text-less payloads', () => {
    expect(parseStatusMessage('')).toBeNull()
    expect(parseStatusMessage('{}')).toBeNull()
    expect(parseStatusMessage('{"text":"   "}')).toBeNull()
    expect(parseStatusMessage('not json')).toBeNull()
    expect(parseStatusMessage(null)).toBeNull()
  })
})

describe('statusRemainingMs', () => {
  const msg = { text: 'x', level: 'info' as const, timeoutMs: 5000, at: 10_000 }

  it('live events count from arrival regardless of the sender clock', () => {
    expect(statusRemainingMs(msg, 999_999, true)).toBe(5000)
  })

  it('a replayed message only shows for what is left of its time', () => {
    expect(statusRemainingMs(msg, 12_000, false)).toBe(3000)
    expect(statusRemainingMs(msg, 20_000, false)).toBe(0)
  })

  it('a clock behind the sender never stretches the time past timeoutMs', () => {
    expect(statusRemainingMs(msg, 1_000, false)).toBe(5000)
  })

  it('sticky messages (timeout 0) never expire', () => {
    expect(statusRemainingMs({ ...msg, timeoutMs: 0 }, 99_999_999, false)).toBe(Number.POSITIVE_INFINITY)
  })
})

describe('createStatusLine', () => {
  beforeEach(() => { vi.useFakeTimers(); vi.setSystemTime(100_000) })
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers() })

  function harness() {
    const seen: Array<string | null> = []
    const line = createStatusLine({ onChange: (m: StatusMessage | null) => seen.push(m ? m.text : null) })
    return { line, seen }
  }
  const payload = (text: string, timeoutMs: number, at = 100_000) => JSON.stringify({ text, level: 'info', timeoutMs, at })

  it('shows a live line and clears it after its own timeout', async () => {
    const { line, seen } = harness()
    line.live(payload('생성 중', 3000))
    expect(line.current?.text).toBe('생성 중')
    await vi.advanceTimersByTimeAsync(2999)
    expect(line.current?.text).toBe('생성 중')
    await vi.advanceTimersByTimeAsync(1)
    expect(line.current).toBeNull()
    expect(seen).toEqual(['생성 중', null])
  })

  it("a newer line replaces the older one and the older line's timer never clears it", async () => {
    const { line, seen } = harness()
    line.live(payload('첫 줄', 1000))
    await vi.advanceTimersByTimeAsync(500)
    line.live(payload('둘째 줄', 5000))
    await vi.advanceTimersByTimeAsync(1000)   // 첫 줄의 만료 시각이 지났다
    expect(line.current?.text).toBe('둘째 줄')
    expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(4000)
    expect(line.current).toBeNull()
    expect(seen).toEqual(['첫 줄', '둘째 줄', null])
  })

  it('sticky lines (timeout 0) stay until the next line', async () => {
    const { line } = harness()
    line.live(payload('❌ 설정 불러오기 실패', 0))
    await vi.advanceTimersByTimeAsync(10 * 60_000)
    expect(line.current?.text).toBe('❌ 설정 불러오기 실패')
    expect(vi.getTimerCount()).toBe(0)
    line.live(payload('✅ 설정이 저장되었습니다.', 3000))
    expect(line.current?.text).toBe('✅ 설정이 저장되었습니다.')
  })

  it('a replayed line shows only for its remaining time and never over a live one', async () => {
    const { line, seen } = harness()
    line.replay(payload('옛 문구', 5000, 97_000))   // 3초 전에 보낸 5초짜리 → 2초 남음
    expect(line.current?.text).toBe('옛 문구')
    await vi.advanceTimersByTimeAsync(2000)
    expect(line.current).toBeNull()

    line.replay(payload('이미 지난 문구', 1000, 90_000))
    expect(line.current).toBeNull()

    line.live(payload('실시간', 5000))
    line.replay(payload('늦게 온 재생', 0))
    expect(line.current?.text).toBe('실시간')
    expect(seen).toEqual(['옛 문구', null, '실시간'])
  })

  it('ignores broken payloads and everything after dispose', async () => {
    const { line, seen } = harness()
    line.live('not json')
    line.live('{"text":"  "}')
    line.live(payload('남은 줄', 1000))
    line.dispose()
    expect(vi.getTimerCount()).toBe(0)
    line.live(payload('닫힌 뒤', 1000))
    line.replay(payload('닫힌 뒤 재생', 1000))
    expect(seen).toEqual(['남은 줄'])
  })
})
