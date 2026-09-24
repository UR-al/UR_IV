import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createPresetManager } from './usePresetManager'

function setup() {
  const requestAction = vi.fn()
  const bk = {
    getPresetList: vi.fn((cb: (json: string) => void) => cb(JSON.stringify(['a', 'b']))),
    getPresetData: vi.fn((name: string, cb: (json: string) => void) => cb(JSON.stringify({ name, steps: 20 }))),
  }
  const mgr = createPresetManager({ getBackend: async () => bk, requestAction })
  return { mgr, bk, requestAction }
}

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

it('opening the manager shows it and loads the preset list', async () => {
  const { mgr } = setup()
  mgr.openPresetManager()
  expect(mgr.showPresetManager.value).toBe(true)
  await vi.runAllTimersAsync()
  expect(mgr.presetList.value).toEqual(['a', 'b'])
})

it('selecting a preset loads its preview; applying it closes the manager', async () => {
  const { mgr, requestAction } = setup()
  mgr.openPresetManager()
  mgr.selectPreset('b')
  await vi.runAllTimersAsync()
  expect(mgr.selectedPreset.value).toBe('b')
  expect(mgr.presetPreview.value).toEqual({ name: 'b', steps: 20 })
  mgr.loadSelectedPreset()
  expect(requestAction).toHaveBeenCalledWith('load_preset_by_name', { name: 'b' })
  expect(mgr.showPresetManager.value).toBe(false)
})

it('does nothing without a selection', () => {
  const { mgr, requestAction } = setup()
  mgr.loadSelectedPreset()
  mgr.deleteSelectedPreset()
  expect(requestAction).not.toHaveBeenCalled()
})

it('deletes only after confirmation and clears the selection', async () => {
  const { mgr, requestAction } = setup()
  mgr.openPresetManager()
  await vi.runAllTimersAsync()
  mgr.selectPreset('a')
  await vi.runAllTimersAsync()
  vi.stubGlobal('confirm', () => false)
  mgr.deleteSelectedPreset()
  expect(requestAction).not.toHaveBeenCalled()
  vi.stubGlobal('confirm', () => true)
  mgr.deleteSelectedPreset()
  expect(requestAction).toHaveBeenCalledWith('delete_preset', { name: 'a' })
  expect(mgr.presetList.value).toEqual(['b'])
  expect(mgr.selectedPreset.value).toBe('')
  expect(mgr.presetPreview.value).toBeNull()
})

it('saving a new preset sends the typed name and re-reads the list instead of guessing it', async () => {
  const { mgr, bk, requestAction } = setup()
  vi.stubGlobal('prompt', () => null)
  mgr.saveNewPreset()
  expect(requestAction).not.toHaveBeenCalled()
  vi.stubGlobal('prompt', () => 'my preset')
  mgr.saveNewPreset()
  expect(requestAction).toHaveBeenCalledWith('save_preset_by_name', { name: 'my preset' })
  expect(mgr.presetList.value).toEqual([])        // 낙관적으로 넣지 않는다
  await vi.advanceTimersByTimeAsync(200)
  expect(bk.getPresetList).toHaveBeenCalledTimes(1)
  expect(mgr.presetList.value).toEqual(['a', 'b'])
})
