import { describe, expect, it } from 'vitest'
import {
  FACTORY_TAB_DEFAULTS,
  TAB_DEFAULTS_EVENT,
  announceTabDefaults,
  diffTabDefaults,
  editorDefaultsFrom,
  followDefault,
  normalizeTabDefaults,
  readTabDefaults,
} from './tabDefaults'

describe('normalizeTabDefaults', () => {
  it('fills missing keys with factory values and drops retired ones', () => {
    const d = normalizeTabDefaults({ steps: '28', denoising: 0.4, defaultRating: 'e', negpip_enabled: true })
    expect(d.steps).toBe(28)
    expect(d.denoising).toBe(0.4)
    expect(d.cfg).toBe(FACTORY_TAB_DEFAULTS.cfg)
    expect(d).not.toHaveProperty('defaultRating')
    expect(d).not.toHaveProperty('negpip_enabled')
  })

  it('clamps numbers, rounds integers and survives garbage', () => {
    const d = normalizeTabDefaults({ steps: 9999, width: 1000.6, yoloConf: 7, denoising: -1, seed: '', cfg: 'abc' })
    expect(d.steps).toBe(500)
    expect(d.width).toBe(1001)
    expect(d.yoloConf).toBe(1)
    expect(d.denoising).toBe(0)
    expect(d.seed).toBe('-1')
    expect(d.cfg).toBe(FACTORY_TAB_DEFAULTS.cfg)
    expect(normalizeTabDefaults(null)).toEqual(FACTORY_TAB_DEFAULTS)
    expect(normalizeTabDefaults([1, 2])).toEqual(FACTORY_TAB_DEFAULTS)
  })
})

describe('diffTabDefaults', () => {
  it('sends only the keys the user changed', () => {
    const synced = normalizeTabDefaults({ steps: 30, cfg: 5 })
    const current = { ...synced, denoising: 0.5 }
    expect(diffTabDefaults(current, synced)).toEqual({ denoising: 0.5 })
  })

  it('returns null when nothing changed (mount/reload must not save or toast)', () => {
    const synced = normalizeTabDefaults({ steps: 30 })
    expect(diffTabDefaults({ ...synced }, synced)).toBeNull()
  })

  it('does not save a number field that is being cleared', () => {
    const synced = normalizeTabDefaults({})
    const current = { ...synced, steps: '' as unknown as number, cfg: 6 }
    expect(diffTabDefaults(current, synced)).toEqual({ cfg: 6 })
  })

  it('never resends stale T2I values that the global save just refreshed', () => {
    // 파일(전역 저장 반영): steps 40 — Settings 는 다시 읽어 synced=40, 사용자가 denoising 만 바꿈
    const synced = normalizeTabDefaults({ steps: 40 })
    const patch = diffTabDefaults({ ...synced, denoising: 0.3 }, synced)
    expect(patch).not.toHaveProperty('steps')
  })
})

describe('editor defaults', () => {
  it('converts YOLO confidence to the panel percent scale', () => {
    const e = editorDefaultsFrom(normalizeTabDefaults({ yoloConf: 0.4, brushSize: 32, effectStrength: 9, snapRadius: 20 }))
    expect(e).toEqual({ brushSize: 32, effectStrength: 9, detectConf: 40, snapRadius: 20 })
  })

  it('follows a new default only while the user has not touched the value', () => {
    expect(followDefault(20, 20, 32)).toBe(32)
    expect(followDefault(25, 20, 32)).toBe(25)
  })
})

describe('bridge helpers', () => {
  it('reads through the backend and falls back to factory values', async () => {
    const backend = { getTabDefaults: (cb: (json: string) => void) => cb('{"denoising": 0.4}') }
    expect((await readTabDefaults(backend)).denoising).toBe(0.4)
    expect(await readTabDefaults(null)).toEqual(FACTORY_TAB_DEFAULTS)
    expect(await readTabDefaults({ getTabDefaults: (cb) => cb('not json') })).toEqual(FACTORY_TAB_DEFAULTS)
  })

  it('announces saved values on the window event', () => {
    const events: Event[] = []
    announceTabDefaults(normalizeTabDefaults({ snapRadius: 8 }), { dispatchEvent: (e: Event) => { events.push(e); return true } })
    expect(events[0].type).toBe(TAB_DEFAULTS_EVENT)
    expect((events[0] as CustomEvent).detail.snapRadius).toBe(8)
  })
})
