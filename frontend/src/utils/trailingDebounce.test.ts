import { describe, expect, it, vi } from 'vitest'
import { createTrailingDebounce } from './trailingDebounce'

function manualTimers() {
  let next = 1
  const pending = new Map<number, () => void>()
  return {
    pending,
    set: (fn: () => void) => { const id = next++; pending.set(id, fn); return id },
    clear: (id: unknown) => { pending.delete(id as number) },
    fire() { const fns = [...pending.values()]; pending.clear(); fns.forEach(fn => fn()) },
  }
}

describe('createTrailingDebounce', () => {
  it('runs once with the last arguments after the burst settles', () => {
    const timers = manualTimers()
    const fn = vi.fn()
    const debounced = createTrailingDebounce(fn, 300, timers)
    debounced(1); debounced(2); debounced(3)
    expect(fn).not.toHaveBeenCalled()
    expect(timers.pending.size).toBe(1)   // 이전 타이머는 지워진다
    timers.fire()
    expect(fn).toHaveBeenCalledTimes(1)
    expect(fn).toHaveBeenCalledWith(3)
    expect(debounced.pending()).toBe(false)
  })

  it('flush runs the pending call now; cancel drops it', () => {
    const timers = manualTimers()
    const fn = vi.fn()
    const debounced = createTrailingDebounce(fn, 300, timers)
    expect(debounced.flush()).toBe(false)
    debounced('a')
    expect(debounced.flush()).toBe(true)
    expect(fn).toHaveBeenCalledWith('a')
    debounced('b')
    debounced.cancel()
    timers.fire()
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('works with real timers', () => {
    vi.useFakeTimers()
    try {
      const fn = vi.fn()
      const debounced = createTrailingDebounce(fn, 250)
      debounced('x')
      vi.advanceTimersByTime(200)
      debounced('y')
      vi.advanceTimersByTime(200)
      expect(fn).not.toHaveBeenCalled()
      vi.advanceTimersByTime(60)
      expect(fn).toHaveBeenCalledWith('y')
    } finally {
      vi.useRealTimers()
    }
  })
})
