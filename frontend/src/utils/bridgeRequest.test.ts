import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createLatestRequest, parseBridgeReply, wasAbandoned } from './bridgeRequest'

interface Reply { requestId?: string; value?: number; error?: string }

describe('createLatestRequest', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.clearAllTimers(); vi.useRealTimers() })

  it('resolves only with the reply that carries its own request id', async () => {
    const request = createLatestRequest<Reply>({ timeoutMs: 1000, prefix: 'loras', randomSuffix: () => 'x' })
    const { id, done } = request.begin()
    expect(id.startsWith('loras-')).toBe(true)
    expect(request.pending).toBe(true)

    expect(request.receive(JSON.stringify({ requestId: 'someone-else', value: 1 }))).toBe(false)
    expect(request.receive('{broken')).toBe(false)
    expect(request.receive(null)).toBe(false)
    expect(request.pending).toBe(true)

    expect(request.receive(JSON.stringify({ requestId: id, value: 2 }))).toBe(true)
    await expect(done).resolves.toEqual({ requestId: id, value: 2 })
    expect(request.pending).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('a newer request ends the older wait with null and ignores the late old reply', async () => {
    const request = createLatestRequest<Reply>({ timeoutMs: 1000 })
    const first = request.begin()
    const second = request.begin()
    expect(first.id).not.toBe(second.id)
    await expect(first.done).resolves.toBeNull()

    // 옛 요청의 결과가 늦게 와도 새 대기를 끝내지 않는다
    expect(request.receive({ requestId: first.id, value: 1 })).toBe(false)
    expect(request.pending).toBe(true)
    expect(request.receive({ requestId: second.id, value: 2 })).toBe(true)
    await expect(second.done).resolves.toEqual({ requestId: second.id, value: 2 })
  })

  it('times out with null so a loading flag can never stay on forever', async () => {
    const onTimeout = vi.fn()
    const request = createLatestRequest<Reply>({ timeoutMs: 500, onTimeout })
    const { id, done } = request.begin()
    await vi.advanceTimersByTimeAsync(499)
    expect(request.pending).toBe(true)
    await vi.advanceTimersByTimeAsync(1)
    await expect(done).resolves.toBeNull()
    expect(onTimeout).toHaveBeenCalledTimes(1)
    expect(request.receive({ requestId: id, value: 3 })).toBe(false)
  })

  it("an earlier request's timer never times out the next request", async () => {
    const onTimeout = vi.fn()
    const request = createLatestRequest<Reply>({ timeoutMs: 1000, onTimeout })
    const first = request.begin()
    await vi.advanceTimersByTimeAsync(600)
    request.receive({ requestId: first.id })
    await first.done
    const second = request.begin()
    await vi.advanceTimersByTimeAsync(600)   // 첫 요청 시작 1200ms — 첫 타이머는 해제됐다
    expect(onTimeout).not.toHaveBeenCalled()
    expect(request.pending).toBe(true)
    request.cancel()
    await expect(second.done).resolves.toBeNull()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('tells a cancel apart from a timeout (closing a modal must not report "no response")', async () => {
    const onTimeout = vi.fn()
    const request = createLatestRequest<Reply>({ timeoutMs: 500, onTimeout })
    const ticket = request.begin()
    expect(ticket.outcome()).toBe('pending')
    request.cancel()
    await expect(ticket.done).resolves.toBeNull()
    expect(ticket.outcome()).toBe('cancelled')
    expect(wasAbandoned(ticket.outcome())).toBe(true)
    await vi.advanceTimersByTimeAsync(1000)
    expect(onTimeout).not.toHaveBeenCalled()
    expect(ticket.outcome()).toBe('cancelled')   // 늦은 타이머가 결말을 바꾸지 않는다
  })

  it('records why each request ended — replied, timeout, superseded', async () => {
    const request = createLatestRequest<Reply>({ timeoutMs: 500 })
    const replied = request.begin()
    request.receive({ requestId: replied.id, value: 1 })
    await replied.done
    expect(replied.outcome()).toBe('replied')
    expect(wasAbandoned(replied.outcome())).toBe(false)

    const older = request.begin()
    const newer = request.begin()
    await expect(older.done).resolves.toBeNull()
    expect(older.outcome()).toBe('superseded')
    expect(wasAbandoned(older.outcome())).toBe(true)
    expect(newer.outcome()).toBe('pending')

    await vi.advanceTimersByTimeAsync(500)
    await expect(newer.done).resolves.toBeNull()
    expect(newer.outcome()).toBe('timeout')
    expect(wasAbandoned(newer.outcome())).toBe(false)   // 시간 초과만 사용자에게 알릴 실패다
    expect(older.outcome()).toBe('superseded')          // 뒤 요청의 결말이 앞 요청을 덮지 않는다
  })

  it('the outcome is already final when the awaiting caller resumes', async () => {
    const request = createLatestRequest<Reply>({ timeoutMs: 500 })
    const ticket = request.begin()
    const seen = ticket.done.then(() => ticket.outcome())
    request.cancel()
    request.begin()   // 취소 직후 새 요청이 시작돼도 앞 요청의 결말은 그대로
    await expect(seen).resolves.toBe('cancelled')
    request.cancel()
  })

  it('ids are unique across instances even with the same prefix (web clients share signals)', () => {
    const a = createLatestRequest<Reply>({ timeoutMs: 1000, prefix: 'gif' })
    const b = createLatestRequest<Reply>({ timeoutMs: 1000, prefix: 'gif' })
    const ids = new Set([a.begin().id, b.begin().id, a.begin().id])
    expect(ids.size).toBe(3)
    a.cancel(); b.cancel()
  })
})

describe('parseBridgeReply', () => {
  it('accepts JSON objects and objects, rejects the rest', () => {
    expect(parseBridgeReply('{"requestId":"a"}')).toEqual({ requestId: 'a' })
    expect(parseBridgeReply({ requestId: 'b' })).toEqual({ requestId: 'b' })
    expect(parseBridgeReply('[1,2]')).toBeNull()
    expect(parseBridgeReply('nope')).toBeNull()
    expect(parseBridgeReply('')).toBeNull()
    expect(parseBridgeReply(undefined)).toBeNull()
  })
})
