import { describe, expect, it, vi } from 'vitest'
import {
  AUTOMATION_KEYS, automationPatchFromServer, automationSyncPayload, hydrateAutomationSettings, markAutomationKeys,
} from './automationSettings'

describe('automationPatchFromServer', () => {
  it('keeps valid fields including cleanupEveryN', () => {
    expect(automationPatchFromServer('{"mode":"timer","limit":30,"repeat":2,"delay":0,"allowDupes":true,"autoResetDeck":false,"maxRetries":0,"cleanupEveryN":25}'))
      .toEqual({ mode: 'timer', limit: 30, repeat: 2, delay: 0, allowDupes: true, autoResetDeck: false, maxRetries: 0, cleanupEveryN: 25 })
  })

  it('drops invalid values and garbage payloads', () => {
    expect(automationPatchFromServer({ mode: 'forever', limit: 0, repeat: -1, delay: -2, maxRetries: 'x', cleanupEveryN: -3 })).toEqual({})
    expect(automationPatchFromServer('not json')).toEqual({})
    expect(automationPatchFromServer('[]')).toEqual({})
  })
})

describe('hydrateAutomationSettings', () => {
  const none = () => new Set<string>()

  it('applies the file values before the first sync', () => {
    const order: string[] = []
    hydrateAutomationSettings({
      request: cb => cb('{"limit":42,"cleanupEveryN":5}'),
      editedKeys: none,
      apply: patch => order.push(`apply:${JSON.stringify(patch)}`),
      onSettled: hydrated => order.push(`sync:${hydrated}`),
      setTimer: () => 0,
    })
    expect(order).toEqual(['apply:{"limit":42,"cleanupEveryN":5}', 'sync:true'])
  })

  it('keeps the keys the user edited while the request was in flight but applies the rest', () => {
    const edited = new Set<string>()
    let reply: ((json: string) => void) | null = null
    const apply = vi.fn()
    const onSettled = vi.fn()
    hydrateAutomationSettings({ request: cb => { reply = cb }, editedKeys: () => edited, apply, onSettled, setTimer: () => 0 })
    edited.add('delay')   // 사용자가 먼저 delay 만 고쳤다
    reply!('{"limit":99,"delay":5,"cleanupEveryN":7}')
    expect(apply).toHaveBeenCalledWith({ limit: 99, cleanupEveryN: 7 })
    expect(onSettled).toHaveBeenCalledTimes(1)
    expect(onSettled).toHaveBeenCalledWith(true)
  })

  it('times out unhydrated, then still applies a late answer once and reports it hydrated', () => {
    let fireTimeout: (() => void) | null = null
    let reply: ((json: string) => void) | null = null
    const apply = vi.fn()
    const onSettled = vi.fn()
    hydrateAutomationSettings({
      request: cb => { reply = cb },
      editedKeys: none,
      apply,
      onSettled,
      setTimer: fn => { fireTimeout = fn; return 1 },
    })
    expect(onSettled).not.toHaveBeenCalled()
    fireTimeout!()
    fireTimeout!()
    expect(onSettled).toHaveBeenCalledTimes(1)
    expect(onSettled).toHaveBeenCalledWith(false)
    reply!('{"limit":42}')
    reply!('{"limit":1}')
    expect(apply).toHaveBeenCalledTimes(1)
    expect(apply).toHaveBeenCalledWith({ limit: 42 })
    expect(onSettled.mock.calls).toEqual([[false], [true]])
  })

  it('an answer before the timeout settles once; the timer then does nothing', () => {
    let fireTimeout: (() => void) | null = null
    const onSettled = vi.fn()
    hydrateAutomationSettings({
      request: cb => cb('{"limit":42}'),
      editedKeys: none,
      apply: vi.fn(),
      onSettled,
      setTimer: fn => { fireTimeout = fn; return 1 },
    })
    fireTimeout!()
    expect(onSettled.mock.calls).toEqual([[true]])
  })

  it('a throwing getter gives up once', () => {
    const onSettled = vi.fn()
    hydrateAutomationSettings({ request: () => { throw new Error('x') }, editedKeys: none, apply: vi.fn(), onSettled, setTimer: () => 0 })
    expect(onSettled.mock.calls).toEqual([[false]])
  })

  it('syncs immediately (unhydrated) when the backend has no getter (older host)', () => {
    const onSettled = vi.fn()
    hydrateAutomationSettings({ request: null, editedKeys: none, apply: vi.fn(), onSettled })
    expect(onSettled).toHaveBeenCalledWith(false)
  })
})

describe('automationSyncPayload', () => {
  const ui = { mode: 'count', limit: 10, repeat: 1, delay: 2.5, allowDupes: false, autoResetDeck: false, maxRetries: 2, cleanupEveryN: 0 } as const

  it('sends nothing persisted before hydration — defaults never reach the file', () => {
    expect(automationSyncPayload({ ...ui }, new Set())).toEqual({})
  })

  it('sends only the known keys', () => {
    const known = new Set<keyof typeof ui>()
    markAutomationKeys(known, { delay: 2.5, junk: 1 })
    expect(automationSyncPayload({ ...ui }, known)).toEqual({ delay: 2.5 })
  })

  it('sends every key once all are known, with safe numbers', () => {
    const known = new Set(AUTOMATION_KEYS)
    expect(automationSyncPayload({ ...ui, limit: Number.NaN, maxRetries: -1 }, known))
      .toEqual({ ...ui, limit: 10, maxRetries: 2 })
  })
})
