import { describe, expect, it, vi } from 'vitest'
import { effectScope, nextTick, ref, watch } from 'vue'
import { createRefCountedSingleton } from './refCountedSingleton'

describe('createRefCountedSingleton', () => {
  function fixture() {
    const created = vi.fn()
    const disposed = vi.fn()
    const seen: number[] = []
    const use = createRefCountedSingleton(() => {
      created()
      const value = ref(0)
      watch(value, v => seen.push(v), { flush: 'sync' })
      return { api: { value }, dispose: disposed }
    })
    return { use, created, disposed, seen }
  }

  it('shares one instance between users and disposes after the last one leaves', () => {
    const { use, created, disposed, seen } = fixture()
    const a = effectScope(); const b = effectScope()
    const first = a.run(use)!
    const second = b.run(use)!
    expect(first).toBe(second)
    expect(created).toHaveBeenCalledTimes(1)
    expect(use.users()).toBe(2)

    a.stop()
    expect(disposed).not.toHaveBeenCalled()
    first.value.value = 1
    expect(seen).toEqual([1])            // 남은 사용자가 있으니 watch 도 산다

    b.stop()
    expect(disposed).toHaveBeenCalledTimes(1)
    first.value.value = 2
    expect(seen).toEqual([1])            // scope 째 멈췄다 — 컴포넌트에 묶이지 않은 watch 도 풀린다
    expect(use.users()).toBe(0)
  })

  it('creates a fresh instance for the next user after disposal', async () => {
    const { use, created } = fixture()
    const a = effectScope()
    const first = a.run(use)!
    a.stop()
    const b = effectScope()
    const second = b.run(use)!
    expect(second).not.toBe(first)
    expect(created).toHaveBeenCalledTimes(2)
    b.stop()
    await nextTick()
  })

  it('a singleton used from inside another one is released together with it', () => {
    const inner = fixture()
    const outerDisposed = vi.fn()
    const outer = createRefCountedSingleton(() => ({ api: { inner: inner.use() }, dispose: outerDisposed }))
    const view = effectScope()
    view.run(() => { inner.use(); outer() })
    expect(inner.use.users()).toBe(2)
    view.stop()
    expect(outerDisposed).toHaveBeenCalledTimes(1)
    expect(inner.disposed).toHaveBeenCalledTimes(1)
    expect(inner.use.users()).toBe(0)
  })
})
