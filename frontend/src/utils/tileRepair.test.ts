import { describe, expect, it } from 'vitest'
import {
  TILE_REPAIR_DEFAULT_PROMPT, TILE_REPAIR_INCREMENTS, TILE_REPAIR_RANGES, clampTileRepairNumber,
  defaultChoiceLabel, defaultTileRepairSettings, normalizeTileRepairSeed, parseTileRepairEvent,
  readTileRepairOptions, tileRepairSource,
} from './tileRepair'

describe('tile repair defaults and bounds', () => {
  it('match the originals', () => {
    const d = defaultTileRepairSettings()
    // origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
    expect([d.negative_prompt, d.steps, d.cfg_scale, d.flow_shift]).toEqual(['', 50, 3.5, 5.0])
    // origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170
    expect(d.multiplier).toBe(1.0)
    // origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:130 — strength -10..10 step .01
    expect(TILE_REPAIR_RANGES.multiplier).toEqual([-10, 10])
    expect(TILE_REPAIR_INCREMENTS.multiplier).toBe(0.01)
    expect(d.prompt).toBe(TILE_REPAIR_DEFAULT_PROMPT)
    expect([d.short_side, d.seed, d.unload_forge_before]).toEqual([1024, -1, true])
    expect([d.model, d.dit, d.text_encoder, d.vae]).toEqual(['', '', '', ''])
  })

  it('returns a fresh object each time', () => {
    const a = defaultTileRepairSettings()
    a.steps = 1
    expect(defaultTileRepairSettings().steps).toBe(50)
  })
})

describe('clampTileRepairNumber', () => {
  it('clamps into the range and rounds integer fields', () => {
    expect(clampTileRepairNumber('multiplier', '12')).toBe(10)
    expect(clampTileRepairNumber('multiplier', -11)).toBe(-10)
    expect(clampTileRepairNumber('multiplier', '-2.25')).toBe(-2.25)
    expect(clampTileRepairNumber('steps', '20.6')).toBe(21)
    expect(clampTileRepairNumber('short_side', 100)).toBe(256)
    expect(clampTileRepairNumber('cfg_scale', '')).toBeNull()
    expect(clampTileRepairNumber('cfg_scale', 'abc')).toBeNull()
  })
})

describe('normalizeTileRepairSeed', () => {
  it('keeps -1 or a whole number from 0', () => {
    expect(normalizeTileRepairSeed('42')).toBe(42)
    expect(normalizeTileRepairSeed(' -1 ')).toBe(-1)
    expect(normalizeTileRepairSeed('-5')).toBe(-1)
    expect(normalizeTileRepairSeed('1.5')).toBe(-1)
    expect(normalizeTileRepairSeed('')).toBe(-1)
  })
})

describe('tileRepairSource', () => {
  it('prefers the local path, then an uploaded data URL', () => {
    expect(tileRepairSource('C:/a.png', 'data:image/png;base64,AA')).toEqual({ image_path: 'C:/a.png', image: '' })
    expect(tileRepairSource('', 'data:image/webp;base64,AA')).toEqual({ image_path: '', image: 'data:image/webp;base64,AA' })
    expect(tileRepairSource('', 'file:///C:/a.png')).toBeNull()
    expect(tileRepairSource('', '')).toBeNull()
  })
})

describe('event and options parsing', () => {
  it('accepts only tile repair events', () => {
    expect(parseTileRepairEvent('{"action":"tile_repair_run","requestId":"r","ok":true}')?.ok).toBe(true)
    expect(parseTileRepairEvent({ action: 'tile_repair_cancel', requestId: 'r', ok: true })?.action).toBe('tile_repair_cancel')
    expect(parseTileRepairEvent('{"action":"relight_preview","requestId":"r","ok":true}')).toBeNull()
    expect(parseTileRepairEvent('{"action":"tile_repair_run","ok":true}')).toBeNull()
    expect(parseTileRepairEvent('[1]')).toBeNull()
    expect(parseTileRepairEvent('{broken')).toBeNull()
  })

  it('checks the option lists', () => {
    const good = { models: ['a'], dit: [], text_encoder: [], vae: [], default_model: 'a' }
    expect(readTileRepairOptions(good)).toBe(good)
    expect(readTileRepairOptions({ ...good, vae: 'x' })).toBeNull()
    expect(readTileRepairOptions({ ...good, models: [1] })).toBeNull()
    expect(readTileRepairOptions(null)).toBeNull()
  })

  it('labels the extension default choice', () => {
    expect(defaultChoiceLabel('animaTileRepair_v20.safetensors')).toBe('기본값 (animaTileRepair_v20.safetensors)')
    expect(defaultChoiceLabel(null)).toContain('기본값')
  })
})
