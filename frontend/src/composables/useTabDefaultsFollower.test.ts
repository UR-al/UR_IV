import { describe, expect, it } from 'vitest'
import { startTabDefaultsFollower } from './useTabDefaultsFollower'
import { FACTORY_TAB_DEFAULTS, TAB_DEFAULTS_EVENT, followDefault, type TabDefaults } from '../utils/tabDefaults'

function flush() {
  return new Promise(resolve => setTimeout(resolve, 0))
}

describe('startTabDefaultsFollower', () => {
  it('applies the saved defaults once, then follows Settings saves for untouched values only', async () => {
    const target = new EventTarget()
    const view = { denoising: FACTORY_TAB_DEFAULTS.denoising, snapRadius: FACTORY_TAB_DEFAULTS.snapRadius }
    const applied: Array<[TabDefaults, TabDefaults]> = []
    const stop = startTabDefaultsFollower((next, prev) => {
      applied.push([next, prev])
      view.denoising = followDefault(view.denoising, prev.denoising, next.denoising)
      view.snapRadius = followDefault(view.snapRadius, prev.snapRadius, next.snapRadius)
    }, {
      loadBackend: async () => ({ getTabDefaults: (cb: (json: string) => void) => cb('{"denoising": 0.4, "snapRadius": 12}') }),
      target: target as any,
    })
    await flush()
    expect(view.denoising).toBe(0.4)                 // 파일 값이 I2I 초기값이 된다(예전엔 늘 0.75)

    view.snapRadius = 30                              // 사용자가 에디터에서 직접 바꿈
    target.dispatchEvent(Object.assign(new Event(TAB_DEFAULTS_EVENT), { detail: { denoising: 0.6, snapRadius: 18 } }))
    expect(view.denoising).toBe(0.6)                 // 손대지 않은 값은 새 기본값을 따른다
    expect(view.snapRadius).toBe(30)                 // 바꾼 값은 그대로

    stop()
    target.dispatchEvent(Object.assign(new Event(TAB_DEFAULTS_EVENT), { detail: { denoising: 0.9 } }))
    expect(view.denoising).toBe(0.6)
    expect(applied.length).toBe(2)
  })

  it('survives a missing backend (factory values, no throw)', async () => {
    const seen: TabDefaults[] = []
    const stop = startTabDefaultsFollower(next => { seen.push(next) }, { loadBackend: async () => null, target: null })
    await flush()
    expect(seen[0]).toEqual(FACTORY_TAB_DEFAULTS)
    stop()
  })
})
