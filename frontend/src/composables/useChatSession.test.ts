import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope } from 'vue'

const mocks = vi.hoisted(() => {
  const handlers = new Map<string, Set<(json: string) => void>>()
  return {
    handlers,
    actionSpy: vi.fn(),
    onBackendEvent: vi.fn((event: string, fn: (json: string) => void) => {
      if (!handlers.has(event)) handlers.set(event, new Set())
      handlers.get(event)!.add(fn)
      return () => { handlers.get(event)?.delete(fn) }
    }),
    emit(event: string, payload: unknown) {
      const raw = typeof payload === 'string' ? payload : JSON.stringify(payload)
      for (const fn of [...(handlers.get(event) || [])]) fn(raw)
    },
  }
})

vi.mock('../bridge.js', () => ({ onBackendEvent: mocks.onBackendEvent }))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))

describe('useChatSession save throttling (P-01)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    mocks.handlers.clear()
    mocks.actionSpy.mockReset()
    const storage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
    vi.stubGlobal('localStorage', storage)
    vi.stubGlobal('window', {
      localStorage: storage,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    })
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('does not trigger chat_save on streaming tokens, and saves on sendMessage and onDone', async () => {
    vi.resetModules()
    const { useChatSession } = await import('./useChatSession')
    const scope = effectScope()
    const session = scope.run(() => useChatSession())!

    mocks.emit('chatThreads', [])
    vi.advanceTimersByTime(1000)
    expect(mocks.actionSpy.mock.calls.filter(([name]) => name === 'chat_save')).toHaveLength(0)

    const thread = session.active.value!
    session.sendMessage(thread, 'hello', [], { mode: 'text' } as any)
    vi.advanceTimersByTime(700)
    expect(mocks.actionSpy.mock.calls.filter(([name]) => name === 'chat_save')).toHaveLength(1)

    const reqId = session.busyId.value
    mocks.actionSpy.mockClear()

    // 50개 토큰 스트리밍 동안 chat_save 는 호출되지 않아야 한다
    for (let i = 0; i < 50; i++) {
      mocks.emit('chatToken', { id: reqId, text: 'a' })
      vi.advanceTimersByTime(100)
    }
    vi.advanceTimersByTime(1000)
    expect(mocks.actionSpy.mock.calls.filter(([name]) => name === 'chat_save')).toHaveLength(0)

    // 완료(chatDone) 시점에만 1회 저장한다
    mocks.emit('chatDone', { id: reqId, ok: true, content: 'a'.repeat(50) })
    vi.advanceTimersByTime(700)
    expect(mocks.actionSpy.mock.calls.filter(([name]) => name === 'chat_save')).toHaveLength(1)

    scope.stop()
  })
})
