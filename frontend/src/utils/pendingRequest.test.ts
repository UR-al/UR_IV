import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPendingRequest } from './pendingRequest'

describe('createPendingRequest', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers() })

  it("an early-finished request's timer never times out the next request", async () => {
    const timeouts = vi.fn()
    const request = createPendingRequest<string>({ timeoutMs: 65000, onTimeout: timeouts })
    const first = request.start()
    await vi.advanceTimersByTimeAsync(5000)
    request.finish('A result')
    await expect(first).resolves.toBe('A result')

    const second = request.start()
    await vi.advanceTimersByTimeAsync(60001)   // A 시작 65초 시점을 지난다
    expect(timeouts).not.toHaveBeenCalled()
    expect(request.pending).toBe(true)
    request.finish('B result')
    await expect(second).resolves.toBe('B result')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('times out only its own pending request and reports it once', async () => {
    const timeouts = vi.fn()
    const states: boolean[] = []
    const request = createPendingRequest<string>({
      timeoutMs: 1000, onTimeout: timeouts, onPendingChange: (v) => states.push(v),
    })
    const waiting = request.start()
    await vi.advanceTimersByTimeAsync(1000)
    await expect(waiting).resolves.toBeNull()
    expect(timeouts).toHaveBeenCalledTimes(1)
    expect(states).toEqual([true, false])
    expect(request.finish('late result')).toBe(false)
  })

  it('starting again settles the previous wait and clears its timer', async () => {
    const request = createPendingRequest<string>({ timeoutMs: 1000 })
    const first = request.start()
    const second = request.start()
    await expect(first).resolves.toBeNull()
    expect(vi.getTimerCount()).toBe(1)
    request.finish('ok')
    await expect(second).resolves.toBe('ok')
  })

  it('ignores a stale timer callback even if the host fires it after clearing', async () => {
    const callbacks: Array<() => void> = []
    const request = createPendingRequest<string>({
      timeoutMs: 10,
      setTimer: (callback) => { callbacks.push(callback); return callbacks.length },
      clearTimer: () => {},   // 해제가 먹지 않는 호스트
    })
    const first = request.start()
    request.finish('A')
    await first
    const second = request.start()
    callbacks[0]!()   // A 의 타이머가 늦게 불린다
    expect(request.pending).toBe(true)
    request.finish('B')
    await expect(second).resolves.toBe('B')
  })
})
