import { describe, expect, it } from 'vitest'
import {
  COLOR_MATCH_DEFAULTS, COLOR_MATCH_MESSAGES, COLOR_MATCH_STORAGE_KEY, clampColorMatch, colorMatchNeedsMask,
  colorMatchPlan, loadColorMatchPrefs, referenceLabel, saveColorMatchPrefs, type ColorMatchConfig,
} from './colorMatch'

const cfg = (over: Partial<ColorMatchConfig> = {}): ColorMatchConfig => ({ ...COLOR_MATCH_DEFAULTS, ...over })

function memoryStorage(init: Record<string, string> = {}) {
  const data = { ...init }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => { data[k] = String(v) },
  }
}

describe('colorMatchPlan', () => {
  it('reference mode sends the path and no mask by default', () => {
    expect(colorMatchPlan(cfg({ strength: 70 }), 'C:/a/ref.png', false)).toEqual({
      ok: true, withMask: false, params: { mode: 'reference', strength: 70, reference_path: 'C:/a/ref.png' },
    })
  })

  it('reference mode needs a loaded image', () => {
    expect(colorMatchPlan(cfg(), '  ', true)).toEqual({ ok: false, message: COLOR_MATCH_MESSAGES.noReference })
  })

  it('"칠한 영역만" sends the mask, and refuses without one', () => {
    const plan = colorMatchPlan(cfg({ maskOnly: true }), 'r.png', true)
    expect(plan.ok && plan.withMask).toBe(true)
    expect(colorMatchPlan(cfg({ maskOnly: true }), 'r.png', false))
      .toEqual({ ok: false, message: COLOR_MATCH_MESSAGES.needMask })
  })

  it('ring mode always needs the mask and sends the band width', () => {
    expect(colorMatchPlan(cfg({ mode: 'ring', ringPx: 24 }), '', true)).toEqual({
      ok: true, withMask: true, params: { mode: 'ring', strength: 100, ring_px: 24 },
    })
    expect(colorMatchPlan(cfg({ mode: 'ring' }), 'r.png', false))
      .toEqual({ ok: false, message: COLOR_MATCH_MESSAGES.ringNeedsMask })
  })

  it('clamps out-of-range values before sending', () => {
    const plan = colorMatchPlan(cfg({ mode: 'ring', strength: 500, ringPx: 1 }), '', true)
    expect(plan).toEqual({ ok: true, withMask: true, params: { mode: 'ring', strength: 100, ring_px: 4 } })
  })
})

describe('config helpers', () => {
  it('needs a mask for ring mode or "칠한 영역만"', () => {
    expect(colorMatchNeedsMask(cfg())).toBe(false)
    expect(colorMatchNeedsMask(cfg({ maskOnly: true }))).toBe(true)
    expect(colorMatchNeedsMask(cfg({ mode: 'ring' }))).toBe(true)
  })

  it('clampColorMatch falls back to defaults for junk', () => {
    expect(clampColorMatch(null)).toEqual(COLOR_MATCH_DEFAULTS)
    expect(clampColorMatch({ mode: 'weird', strength: '50', ringPx: 99.6, maskOnly: 'yes' }))
      .toEqual({ mode: 'reference', strength: 100, ringPx: 64, maskOnly: false })
  })

  it('shows the file name or the empty label', () => {
    expect(referenceLabel('C:\\사진\\참조 이미지.png')).toBe('참조 이미지.png')
    expect(referenceLabel('')).toBe('불러온 이미지 없음')
  })
})

describe('prefs', () => {
  it('round-trips everything but the reference path', () => {
    const st = memoryStorage()
    saveColorMatchPrefs(cfg({ mode: 'ring', strength: 40, ringPx: 32, maskOnly: true }), st)
    expect(JSON.parse(st.data[COLOR_MATCH_STORAGE_KEY])).toEqual({ mode: 'ring', strength: 40, ringPx: 32, maskOnly: true })
    expect(loadColorMatchPrefs(st)).toEqual({ mode: 'ring', strength: 40, ringPx: 32, maskOnly: true })
  })

  it('survives broken JSON and blocked storage', () => {
    expect(loadColorMatchPrefs(memoryStorage({ [COLOR_MATCH_STORAGE_KEY]: '{oops' }))).toEqual(COLOR_MATCH_DEFAULTS)
    const blocked = { getItem: () => { throw new Error('x') }, setItem: () => { throw new Error('x') } }
    expect(loadColorMatchPrefs(blocked)).toEqual(COLOR_MATCH_DEFAULTS)
    expect(() => saveColorMatchPrefs(cfg(), blocked)).not.toThrow()
  })
})
