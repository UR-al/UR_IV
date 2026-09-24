import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

// 고해상도 배율: Python 배율(set_high_res_factor)은 즉시, 파일 저장(save_ui_prefs)은 디바운스.
// 부팅 땐 파일 값이 이기고 같은 값을 되쓰지 않는다. 예전엔 마운트 400ms 뒤 localStorage 값을
// 파일에 무조건 써서 캐시가 파일을 덮을 수 있었다(감사 #107·#130).
const mocks = vi.hoisted(() => ({ actionSpy: vi.fn() }))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))

import { HIGH_RES_SAVE_DEBOUNCE_MS, useHighRes } from './useHighRes.js'

let storage: Map<string, string>

function make() {
  const saveUiPrefs = vi.fn()
  const api = useHighRes({ storeWidgets: { width_input: '1000', height_input: '800' }, saveUiPrefs })
  return { ...api, saveUiPrefs }
}
const pushes = () => mocks.actionSpy.mock.calls.filter(([name]) => name === 'set_high_res_factor').map(([, p]) => p)

beforeEach(() => {
  vi.useFakeTimers()
  mocks.actionSpy.mockReset()
  storage = new Map([['highRes.enabled', 'true'], ['highRes.factor', '1.5']])
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => { storage.set(k, String(v)) },
  })
})
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

it('does not push or save anything on its own at mount', async () => {
  const { saveUiPrefs } = make()
  await nextTick()
  vi.advanceTimersByTime(2000)
  expect(pushes()).toEqual([])
  expect(saveUiPrefs).not.toHaveBeenCalled()
})

it('drag ticks push Python immediately but save the file once after the burst', async () => {
  const { highResFactor, saveUiPrefs } = make()
  for (const f of [1.55, 1.6, 1.65, 1.7]) {
    highResFactor.value = f
    await nextTick()
  }
  expect(pushes().map(p => p.factor)).toEqual([1.55, 1.6, 1.65, 1.7])
  expect(saveUiPrefs).not.toHaveBeenCalled()
  expect(storage.get('highRes.factor')).toBe('1.7')   // 캐시는 즉시
  vi.advanceTimersByTime(HIGH_RES_SAVE_DEBOUNCE_MS)
  expect(saveUiPrefs).toHaveBeenCalledTimes(1)
  expect(saveUiPrefs).toHaveBeenCalledWith({ highResEnabled: true, highResFactor: 1.7 })
})

it('restores the file values, pushes them once and never echoes a save', async () => {
  const { highResEnabled, highResFactor, restoreFromPrefs, saveUiPrefs } = make()
  restoreFromPrefs({ highResEnabled: false, highResFactor: 2 })
  await nextTick(); await nextTick()
  expect(highResEnabled.value).toBe(false)
  expect(highResFactor.value).toBe(2)
  expect(pushes()).toEqual([{ enabled: false, factor: 2 }])
  expect(storage.get('highRes.enabled')).toBe('false')
  vi.advanceTimersByTime(HIGH_RES_SAVE_DEBOUNCE_MS * 2)
  expect(saveUiPrefs).not.toHaveBeenCalled()
})

it('a restore cancels a pending save of the pre-restore value (file wins)', async () => {
  const { highResFactor, restoreFromPrefs, saveUiPrefs } = make()
  highResFactor.value = 1.8
  await nextTick()
  restoreFromPrefs({ highResEnabled: true, highResFactor: 1.3 })
  await nextTick(); await nextTick()
  vi.advanceTimersByTime(HIGH_RES_SAVE_DEBOUNCE_MS * 2)
  expect(saveUiPrefs).not.toHaveBeenCalled()
  expect(highResFactor.value).toBe(1.3)
})

it('migrates the on-screen cache into ui_prefs once when the file never stored it', async () => {
  const { restoreFromPrefs, saveUiPrefs } = make()
  restoreFromPrefs({ theme: 'dark' })
  expect(saveUiPrefs).toHaveBeenCalledWith({ highResEnabled: true, highResFactor: 1.5 })
  expect(pushes()).toEqual([{ enabled: true, factor: 1.5 }])
})
