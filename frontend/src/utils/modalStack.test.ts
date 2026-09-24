import { describe, expect, it, vi } from 'vitest'
import { createModalStack } from './modalStack'

describe('createModalStack', () => {
  it('is empty until a layer opens, and ESC does nothing then', () => {
    const stack = createModalStack()
    expect(stack.isAnyOpen()).toBe(false)
    expect(stack.closeTop()).toBe(false)
  })

  it('ESC closes only the topmost closable layer', () => {
    const stack = createModalStack()
    const closeA = vi.fn()
    const closeB = vi.fn()
    stack.open({ close: closeA })
    stack.open({ close: closeB })
    expect(stack.closeTop()).toBe(true)
    expect(closeB).toHaveBeenCalledTimes(1)
    expect(closeA).not.toHaveBeenCalled()
  })

  it('a self-handling layer (no close) swallows ESC instead of closing what is behind it', () => {
    const stack = createModalStack()
    const behind = vi.fn()
    stack.open({ close: behind })
    stack.open({})
    expect(stack.closeTop()).toBe(true)
    expect(behind).not.toHaveBeenCalled()
  })

  it('release removes exactly its own layer, and a second release is a no-op', () => {
    const stack = createModalStack()
    const closeA = vi.fn()
    const releaseA = stack.open({ close: closeA })
    const releaseB = stack.open({})
    releaseA()
    releaseA()
    expect(stack.size()).toBe(1)
    releaseB()
    expect(stack.isAnyOpen()).toBe(false)
    expect(stack.closeTop()).toBe(false)
    expect(closeA).not.toHaveBeenCalled()
  })

  it('the same close function opened twice is tracked as two layers', () => {
    const stack = createModalStack()
    const close = vi.fn()
    const first = stack.open({ close })
    stack.open({ close })
    first()
    expect(stack.size()).toBe(1)
  })
})
