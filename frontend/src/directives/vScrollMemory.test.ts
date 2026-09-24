import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { SCROLL_RESTORE_RETRY_MS, vScrollMemory } from './vScrollMemory'

function fakeEl() {
  const listeners: Record<string, Array<() => void>> = {}
  return {
    scrollTop: 0,
    listeners,
    addEventListener: (name: string, cb: () => void) => { (listeners[name] ||= []).push(cb) },
    removeEventListener: (name: string, cb: () => void) => { listeners[name] = (listeners[name] || []).filter(f => f !== cb) },
    scroll(top: number) { this.scrollTop = top; for (const cb of listeners.scroll || []) cb() },
  }
}
const hook = (name: 'mounted' | 'updated' | 'beforeUnmount', el: any, value?: string) =>
  (vScrollMemory as any)[name](el, { value })

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers() })

it('restores the last scroll position of the same key when a panel mounts again', async () => {
  const first = fakeEl()
  hook('mounted', first, 'history-a')
  first.scroll(240)
  hook('beforeUnmount', first)
  expect(first.listeners.scroll).toEqual([])      // 리스너를 떼고 나간다

  const second = fakeEl()
  hook('mounted', second, 'history-a')
  await nextTick()
  expect(second.scrollTop).toBe(240)
  second.scrollTop = 0                            // 늦게 채워진 콘텐츠가 위치를 0 으로 되돌려도
  vi.advanceTimersByTime(SCROLL_RESTORE_RETRY_MS)
  expect(second.scrollTop).toBe(240)              // 한 번 더 복원한다
})

it('a key without history is left alone', async () => {
  const el = fakeEl()
  el.scrollTop = 5
  hook('mounted', el, 'never-seen')
  await nextTick()
  vi.runAllTimers()
  expect(el.scrollTop).toBe(5)
})

it('an update that snapped the list back to the top is undone', async () => {
  const el = fakeEl()
  hook('mounted', el, 'history-b')
  el.scroll(90)
  el.scrollTop = 0                                // 재렌더로 0 이 됐다(스크롤 이벤트 없이)
  hook('updated', el)
  await nextTick()
  expect(el.scrollTop).toBe(90)
})
