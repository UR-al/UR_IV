import { describe, expect, it } from 'vitest'
import cases from './animaSpeed.cases.json'
import {
  COMFY_NOTE, DEFAULTS, DEFAULT_SETTINGS, DELTA_RANGE, DIVISOR_RANGE, FORGE_NOTE, INVALID_NOTE, KREA2_NOTE,
  LABELS, MODES, MODE_OPTIONS, OLD_FORGE_NOTE, PRESET_A, PRESET_NAMES, THRESHOLDS, THRESHOLD_OPTIONS, TRANSFORMS,
  TRANSFORM_OPTIONS, VALIDATION_MESSAGES, WIDGET_IDS, cardNote, isDefault, numbers, parseSettings, pyFloat,
  readSettings, summary, validate, validationMessages, visibleFields, type SpeedKey, type SpeedSettings,
} from './animaSpeed'

const card = (patch: Partial<SpeedSettings> = {}): SpeedSettings => ({ ...DEFAULT_SETTINGS, enabled: true, ...patch })

describe('tables', () => {
  it('widget ids, defaults and choices', () => {
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_aspeed_'))).toBe(true)
    expect(Object.keys(DEFAULTS).sort()).toEqual(Object.keys(WIDGET_IDS).sort())
    expect(Object.keys(LABELS).sort()).toEqual(Object.keys(WIDGET_IDS).sort())
    expect(parseSettings(DEFAULTS)).toEqual(DEFAULT_SETTINGS)
    expect(MODE_OPTIONS.map(o => o.key)).toEqual(MODES)
    expect(THRESHOLD_OPTIONS.map(o => o.key)).toEqual(THRESHOLDS)
    expect(TRANSFORM_OPTIONS.map(o => o.key)).toEqual(TRANSFORMS)
    expect(Object.keys(PRESET_A).sort()).toEqual([...PRESET_NAMES].sort())
    expect(DELTA_RANGE).toEqual({ min: 0.0001, max: 0.5, step: 0.0001 })
    expect(DIVISOR_RANGE).toEqual({ min: 0.5, max: 2, step: 0.01 })
    expect(Object.keys(VALIDATION_MESSAGES)).toHaveLength(9)     // sam-extra 0.33.0: no script_error (a/0 is a field error)
    expect(INVALID_NOTE).toContain('적용되지 않습니다')
  })
})

describe('golden cases shared with core/anima_speed.py', () => {
  it('normalizes the card values like parse_settings', () => {
    for (const item of cases.settings) {
      expect(parseSettings(item.widgets as Partial<Record<SpeedKey, unknown>>), item.name).toEqual(item.settings)
    }
  })

  it('validates like validate (codes and their order)', () => {
    for (const item of cases.settings) {
      expect(validate(parseSettings(item.widgets as Partial<Record<SpeedKey, unknown>>)), item.name).toEqual(item.errors)
    }
    expect(cases.settings.length).toBeGreaterThanOrEqual(60)
  })
})

describe('numbers', () => {
  it('reads python float syntax only', () => {
    expect(['1', ' 1 ', '1.', '.5', '+1', '-0.5', '1e3', '1E-3', '1_000', '1e1_0', '-inf', 'Infinity', 'nan'].map(pyFloat))
      .toEqual([1, 1, 1, 0.5, 1, -0.5, 1000, 0.001, 1000, 1e10, -Infinity, Infinity, NaN])
    for (const bad of ['', ' ', '0x1', '1__0', '_1', '1_', '1._5', 'e5', '.', 'inf1', '1/2', null, undefined]) {
      expect(pyFloat(bad), String(bad)).toBeNull()
    }
  })

  it('splits on comma and semicolon, reads a/b, stops at the first failure', () => {
    expect(numbers('1/4, 1/2; 1')).toEqual({ values: [0.25, 0.5, 1] })
    expect(numbers(' , ')).toEqual({ values: [] })
    expect(numbers('1/0,x')).toEqual({ error: 'zero' })
    expect(numbers('x,1/0')).toEqual({ error: 'value' })
    expect(numbers('0.5,inf')).toEqual({ error: 'value' })
    expect(numbers('nan/0')).toEqual({ error: 'zero' })
  })
})

describe('card display', () => {
  it('reads the store by widget id and messages follow the codes', () => {
    const store = { [WIDGET_IDS.enabled]: 'true', [WIDGET_IDS.scales]: '0.5', [WIDGET_IDS.delta]: '0', other: 'x' }
    const settings = readSettings(store)
    expect([settings.enabled, settings.scales, settings.delta]).toEqual([true, '0.5', 0])
    expect(validationMessages(settings)).toEqual([VALIDATION_MESSAGES.scales, VALIDATION_MESSAGES.delta])
    expect(readSettings(undefined)).toEqual(DEFAULT_SETTINGS)
  })

  it('shows only the fields the extension reads', () => {
    expect(visibleFields(card())).toEqual(['mode', 'threshold', 'preset', 'scales', 'delta', 'sigma_divisor', 'adaptive',
      'hires', 'apply_img2img', 'transform', 'seed'])
    expect(visibleFields(card({ threshold: 'delta_optimal', preset: 'custom' }))).toEqual(['mode', 'threshold', 'preset',
      'scales', 'delta', 'adaptive', 'hires', 'apply_img2img', 'transform', 'spectrum_A', 'spectrum_beta', 'seed'])
    expect(visibleFields(card({ threshold: 'manual', preset: 'custom' }))).toEqual(['mode', 'threshold', 'scales',
      'manual_sigmas', 'hires', 'apply_img2img', 'transform', 'seed'])
  })

  it('summary badge', () => {
    expect(summary(card({ enabled: false }))).toBe('')
    expect(summary(card())).toBe('transition · neo_shift · 0.5→1')
    expect(summary(card({ scales: '1/4,0.5,1.0' }))).toBe('transition · neo_shift · 0.25→0.5→1')
    expect(summary(card({ mode: 'respace', threshold: 'manual' }))).toBe('respace · manual σ 0.7')
    expect(summary(card({ scales: '0.5' }))).toBe('⚠ 설정 오류')
  })

  it('reset state and backend line', () => {
    expect(isDefault(parseSettings(DEFAULTS))).toBe(true)
    expect(isDefault(card({ enabled: false, seed: 3 }))).toBe(false)
    const forge = (speed: boolean) => ({ status: 'ok', known: true, features: { speed } }) as const
    expect(cardNote(forge(true) as never)).toBe(FORGE_NOTE)
    expect(cardNote(forge(false) as never)).toBe(OLD_FORGE_NOTE)
    expect(cardNote(null)).toBe(FORGE_NOTE)                                       // 모르면 막지 않는다
    expect(cardNote({ status: 'not_applicable', known: false } as never)).toBe(COMFY_NOTE)             // 계획 C6
    expect(cardNote(forge(true) as never, true)).toBe(KREA2_NOTE)
  })
})
