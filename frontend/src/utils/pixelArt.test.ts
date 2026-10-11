import { describe, expect, it } from 'vitest'
import {
  PIXEL_ART_DEFAULTS, PIXEL_ART_STORAGE_KEY, clampPixelArt, colorsLabel, loadPixelArtPrefs, pixelSizeLabel,
  savePixelArtPrefs,
} from './pixelArt'

function memoryStorage(init: Record<string, string> = {}) {
  const data = { ...init }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => { data[k] = String(v) },
  }
}

describe('clampPixelArt', () => {
  it('keeps valid values and rounds', () => {
    expect(clampPixelArt({ pixel_size: 12.4, colors: 7.6 })).toEqual({ pixel_size: 12, colors: 8 })
  })
  it('clamps to the backend ranges', () => {
    expect(clampPixelArt({ pixel_size: 1, colors: 999 })).toEqual({ pixel_size: 2, colors: 64 })
    expect(clampPixelArt({ pixel_size: 500, colors: -4 })).toEqual({ pixel_size: 64, colors: 0 })
  })
  it('falls back to defaults for junk', () => {
    expect(clampPixelArt(null)).toEqual(PIXEL_ART_DEFAULTS)
    expect(clampPixelArt({ pixel_size: 'abc', colors: NaN })).toEqual(PIXEL_ART_DEFAULTS)
    expect(clampPixelArt({ pixel_size: '16' })).toEqual({ pixel_size: 16, colors: 16 })
  })
})

describe('labels', () => {
  it('shows "끔" when the palette is off', () => {
    expect(colorsLabel(0)).toBe('끔')
    expect(colorsLabel(1)).toBe('끔')
    expect(colorsLabel(16)).toBe('16')
    expect(pixelSizeLabel(8)).toBe('8px')
  })
})

describe('prefs', () => {
  it('round-trips through storage', () => {
    const st = memoryStorage()
    savePixelArtPrefs({ pixel_size: 24, colors: 0 }, st)
    expect(JSON.parse(st.data[PIXEL_ART_STORAGE_KEY])).toEqual({ pixel_size: 24, colors: 0 })
    expect(loadPixelArtPrefs(st)).toEqual({ pixel_size: 24, colors: 0 })
  })
  it('survives broken JSON and blocked storage', () => {
    expect(loadPixelArtPrefs(memoryStorage({ [PIXEL_ART_STORAGE_KEY]: 'not json' }))).toEqual(PIXEL_ART_DEFAULTS)
    const blocked = { getItem: () => { throw new Error('x') }, setItem: () => { throw new Error('x') } }
    expect(loadPixelArtPrefs(blocked)).toEqual(PIXEL_ART_DEFAULTS)
    expect(() => savePixelArtPrefs(PIXEL_ART_DEFAULTS, blocked)).not.toThrow()
    expect(loadPixelArtPrefs(null)).toEqual(PIXEL_ART_DEFAULTS)
  })
})
