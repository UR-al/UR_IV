import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createToasts, MAX_TOASTS, MAX_TOAST_HISTORY, TOAST_TTL_MS, useToasts } from './useToasts'

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers() })

describe('createToasts', () => {
  it('shows a toast, records it and expires it after the ttl', () => {
    const t = createToasts()
    t.addToast('success', 'saved')
    expect(t.toasts.value.map(x => x.msg)).toEqual(['saved'])
    expect(t.toastHistory.value.map(x => x.msg)).toEqual(['saved'])
    expect(t.unread.value).toBe(1)
    vi.advanceTimersByTime(TOAST_TTL_MS - 1)
    expect(t.toasts.value).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(t.toasts.value).toHaveLength(0)
    expect(t.toastHistory.value).toHaveLength(1)   // 기록은 남는다
  })

  it('folds a repeat of the last toast into a counter and restarts its timer (no new history row)', () => {
    const t = createToasts()
    t.addToast('error', 'boom')
    vi.advanceTimersByTime(TOAST_TTL_MS - 100)
    t.addToast('error', 'boom')
    expect(t.toasts.value).toHaveLength(1)
    expect(t.toasts.value[0].count).toBe(2)
    expect(t.toastHistory.value).toHaveLength(1)
    expect(t.unread.value).toBe(1)
    vi.advanceTimersByTime(200)
    expect(t.toasts.value).toHaveLength(1)   // 다시 센다
    vi.advanceTimersByTime(TOAST_TTL_MS)
    expect(t.toasts.value).toHaveLength(0)
  })

  it('keeps at most MAX_TOASTS on screen, dropping the oldest', () => {
    const t = createToasts()
    for (let i = 0; i < MAX_TOASTS + 2; i++) t.addToast('info', `m${i}`)
    expect(t.toasts.value.map(x => x.msg)).toEqual(['m2', 'm3', 'm4', 'm5', 'm6'])
  })

  it('caps the history and puts the newest first', () => {
    const t = createToasts()
    for (let i = 0; i < MAX_TOAST_HISTORY + 3; i++) t.addToast('info', `m${i}`)
    expect(t.toastHistory.value).toHaveLength(MAX_TOAST_HISTORY)
    expect(t.toastHistory.value[0].msg).toBe(`m${MAX_TOAST_HISTORY + 2}`)
  })

  it('opening the panel clears the unread count, and toasts while open do not count', () => {
    const t = createToasts()
    t.addToast('info', 'a')
    t.toggleNotifPanel()
    expect(t.showNotifPanel.value).toBe(true)
    expect(t.unread.value).toBe(0)
    t.addToast('info', 'b')
    expect(t.unread.value).toBe(0)
    t.toggleNotifPanel()
    t.addToast('info', 'c')
    expect(t.unread.value).toBe(1)
    t.clearNotifHistory()
    expect(t.toastHistory.value).toEqual([])
  })

  it('removeToast and clearAllToasts cancel the pending timers', () => {
    const t = createToasts()
    t.addToast('info', 'a')
    t.addToast('info', 'b')
    t.removeToast(t.toasts.value[0].id)
    expect(t.toasts.value.map(x => x.msg)).toEqual(['b'])
    t.clearAllToasts()
    expect(t.toasts.value).toEqual([])
    expect(vi.getTimerCount()).toBe(0)
  })
})

it('useToasts is one shared list for the whole app', () => {
  expect(useToasts()).toBe(useToasts())
})
