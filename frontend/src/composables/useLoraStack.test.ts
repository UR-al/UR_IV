import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

// 생성 LoRA 의 단일 소스는 Python _vue_lora_entries — ui_prefs 복원 뒤 반드시 Python 에 보내야 한다.
// 웹 모드에선 ui_prefs 가 getInitialConfig 응답으로 늦게 오고 Python 호스트는 새 클라이언트마다
// 런타임 복원을 돌리지 않아서, 이 브라우저의 낡은 localStorage 스택이 공유 스택을 덮으면 안 된다.
const mocks = vi.hoisted(() => ({
  actionSpy: vi.fn(),
  handlers: new Map<string, (...args: any[]) => void>(),
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))
vi.mock('../bridge.js', () => ({
  onBackendEvent: (name: string, cb: (...args: any[]) => void) => { mocks.handlers.set(name, cb); return () => {} },
}))

import { LORA_SAVE_DEBOUNCE_MS, useLoraStack } from './useLoraStack.js'

const STALE = [{ name: 'stale_lora', weight: 50, enabled: true, triggerWords: [] }]
let storage: Map<string, string>

function makeStack() {
  const saveUiPrefs = vi.fn()
  const api = useLoraStack({ storeWidgets: {}, addToast: () => {}, saveUiPrefs })
  return { ...api, saveUiPrefs }
}

function loraSyncs() {
  return mocks.actionSpy.mock.calls.filter(([name]) => name === 'set_lora_stack').map(([, payload]) => payload.entries)
}

beforeEach(() => {
  mocks.actionSpy.mockReset()
  mocks.handlers.clear()
  storage = new Map([['loraStack', JSON.stringify(STALE)]])
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (k: string) => storage.get(k) ?? null,
      setItem: (k: string, v: string) => { storage.set(k, String(v)) },
    },
  })
})
afterEach(() => { vi.unstubAllGlobals() })

it('never pushes the localStorage fallback stack on its own before ui_prefs arrive', async () => {
  const { loraStack } = makeStack()
  await nextTick()
  expect(loraStack.map(l => l.name)).toEqual(['stale_lora'])
  expect(loraSyncs()).toEqual([])
})

it('pushes the ui_prefs stack to Python exactly once after restoring it, without echoing a save', async () => {
  const { loraStack, restoreFromPrefs, saveUiPrefs } = makeStack()
  restoreFromPrefs({ loraStack: [
    { name: 'shared_a', weight: 85, enabled: true, triggerWords: ['tw'] },
    { name: 'shared_b', weight: 100, enabled: false, triggerWords: [] },
  ] })
  expect(loraSyncs()).toEqual([])   // 복원 플래그가 풀린 뒤(nextTick)에 보낸다
  await nextTick(); await nextTick()
  expect(loraStack.map(l => l.name)).toEqual(['shared_a', 'shared_b'])
  expect(loraSyncs()).toEqual([[
    { name: 'shared_a', weight: 0.85, enabled: true, triggerWords: ['tw'] },
    { name: 'shared_b', weight: 1, enabled: false, triggerWords: [] },
  ]])
  expect(saveUiPrefs).not.toHaveBeenCalled()
  expect(JSON.parse(storage.get('loraStack') || '[]').map((l: any) => l.name)).toEqual(['shared_a', 'shared_b'])
})

it('pushes an empty ui_prefs stack so a stale shared stack cannot survive', async () => {
  const { loraStack, restoreFromPrefs } = makeStack()
  restoreFromPrefs({ loraStack: [] })
  await nextTick(); await nextTick()
  expect(loraStack.length).toBe(0)
  expect(loraSyncs()).toEqual([[]])
})

it('pushes the on-screen stack when ui_prefs has never stored one', async () => {
  const { restoreFromPrefs } = makeStack()
  restoreFromPrefs({ theme: 'dark' })
  await nextTick(); await nextTick()
  expect(loraSyncs()).toEqual([[{ name: 'stale_lora', weight: 0.5, enabled: true, triggerWords: [] }]])
})

it('ignores a missing prefs payload', async () => {
  const { restoreFromPrefs } = makeStack()
  restoreFromPrefs(null)
  await nextTick(); await nextTick()
  expect(loraSyncs()).toEqual([])
})

it('resumes normal save + sync for user edits after a restore', async () => {
  vi.useFakeTimers()
  try {
    const { loraStack, restoreFromPrefs, saveUiPrefs } = makeStack()
    restoreFromPrefs({ loraStack: [{ name: 'a', weight: 80, enabled: true, triggerWords: [] }] })
    await nextTick(); await nextTick()
    mocks.actionSpy.mockReset()
    loraStack[0].enabled = false
    await nextTick(); await nextTick()
    expect(loraSyncs()).toEqual([[{ name: 'a', weight: 0.8, enabled: false, triggerWords: [] }]])
    vi.advanceTimersByTime(LORA_SAVE_DEBOUNCE_MS)
    expect(saveUiPrefs).toHaveBeenCalledWith({ loraStack: [{ name: 'a', weight: 80, enabled: false, triggerWords: [] }] })
  } finally {
    vi.useRealTimers()
  }
})

// 감사 #130: 가중치 슬라이더 드래그(input 이벤트마다)가 틱마다 GUI 스레드의 ui_prefs.json RMW 를
// 부르지 않는다. Python 생성 스택(set_lora_stack)은 틱마다 즉시 — 드래그 직후 생성이 옛 값으로 나가면 안 된다.
it('debounces the file save during a weight drag but syncs Python on every tick', async () => {
  vi.useFakeTimers()
  try {
    const { loraStack, restoreFromPrefs, saveUiPrefs } = makeStack()
    restoreFromPrefs({ loraStack: [{ name: 'a', weight: 80, enabled: true, triggerWords: [] }] })
    await nextTick(); await nextTick()
    mocks.actionSpy.mockReset()
    for (const w of [81, 82, 83, 84, 85]) {
      loraStack[0].weight = w
      await nextTick()
    }
    expect(loraSyncs().map(entries => entries[0].weight)).toEqual([0.81, 0.82, 0.83, 0.84, 0.85])
    expect(saveUiPrefs).not.toHaveBeenCalled()
    vi.advanceTimersByTime(LORA_SAVE_DEBOUNCE_MS)
    expect(saveUiPrefs).toHaveBeenCalledTimes(1)
    expect(saveUiPrefs).toHaveBeenCalledWith({ loraStack: [{ name: 'a', weight: 85, enabled: true, triggerWords: [] }] })
    expect(JSON.parse(storage.get('loraStack') || '[]')[0].weight).toBe(85)   // 캐시는 즉시
  } finally {
    vi.useRealTimers()
  }
})

it('adding a LoRA saves and syncs once (no explicit + watch double save)', async () => {
  vi.useFakeTimers()
  try {
    const { addLoraToStack, restoreFromPrefs, saveUiPrefs } = makeStack()
    restoreFromPrefs({ loraStack: [] })
    await nextTick(); await nextTick()
    mocks.actionSpy.mockReset()
    addLoraToStack('new_lora', 0.7, ['tw'])
    await nextTick()
    vi.advanceTimersByTime(LORA_SAVE_DEBOUNCE_MS)
    expect(saveUiPrefs).toHaveBeenCalledTimes(1)
    expect(loraSyncs()).toEqual([[{ name: 'new_lora', weight: 0.7, enabled: true, triggerWords: ['tw'] }]])
  } finally {
    vi.useRealTimers()
  }
})

it('a ui_prefs restore drops a pending save of the pre-restore stack (file wins)', async () => {
  vi.useFakeTimers()
  try {
    const { loraStack, restoreFromPrefs, saveUiPrefs } = makeStack()
    restoreFromPrefs({ loraStack: [{ name: 'a', weight: 80, enabled: true, triggerWords: [] }] })
    await nextTick(); await nextTick()
    loraStack[0].weight = 90
    await nextTick()
    restoreFromPrefs({ loraStack: [{ name: 'b', weight: 50, enabled: true, triggerWords: [] }] })
    await nextTick(); await nextTick()
    vi.advanceTimersByTime(LORA_SAVE_DEBOUNCE_MS * 2)
    expect(saveUiPrefs).not.toHaveBeenCalled()
    expect(loraStack.map(l => l.name)).toEqual(['b'])
  } finally {
    vi.useRealTimers()
  }
})
