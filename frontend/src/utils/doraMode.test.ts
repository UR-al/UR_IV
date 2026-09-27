import { describe, expect, it } from 'vitest'
import type { SamExtraCapabilitiesEvent } from '../types/bridge'
import {
  DEFAULTS, INSERT_OPTIONS, MODE_OPTIONS, SCOPE_OPTIONS, WIDGET_IDS, cardStatus, isStock, keyOf, labelOf,
  readValues, showWeakFields, summary, type DoraValues,
} from './doraMode'

function caps(over: Partial<SamExtraCapabilitiesEvent> = {}, dora = true): SamExtraCapabilitiesEvent {
  return {
    status: 'ok', known: true, checked_at: 't', installed: true,
    features: { dora } as SamExtraCapabilitiesEvent['features'],
    anima_guidance_argc: null, detail_daemon_hires: null, version: {}, scripts: {}, sam3_keys: {},
    options: {}, options_known: false, gradio_api: [], choices: {}, warnings: [], errors: {},
    ...over,
  }
}

const values = (over: Partial<DoraValues> = {}): DoraValues => ({ ...DEFAULTS, ...over })

describe('doraMode options', () => {
  it('keeps the extension radio order and keys', () => {
    expect(MODE_OPTIONS.map(o => o.key)).toEqual(['lycoris', 'forge_fp32', 'forge', 'no_magnitude'])
    expect(INSERT_OPTIONS.map(o => o.key)).toEqual(['keep', 'additive', 'skip', 'weak'])
    expect(SCOPE_OPTIONS.map(o => o.key)).toEqual(['attn', 'attn_mlp', 'all'])
  })

  it('maps labels and keys both ways (unknown passes through)', () => {
    expect(labelOf(MODE_OPTIONS, 'lycoris')).toBe('LyCORIS (학습과 동일 · fp32)')
    expect(keyOf(INSERT_OPTIONS, '약한 복사 (브리지식 · 강도·범위 조절)')).toBe('weak')
    expect(labelOf(SCOPE_OPTIONS, 'mystery')).toBe('mystery')
    expect(keyOf(SCOPE_OPTIONS, 'mystery')).toBe('mystery')
  })

  it('app defaults follow the Forge txt2img ui-config (on, LyCORIS, keep, i2i off)', () => {
    expect(DEFAULTS).toEqual({
      enabled: 'true', mode: 'lycoris', inserted: 'keep', weak_strength: '0.12', weak_scope: 'attn',
      apply_img2img: 'false',
    })
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_dora_'))).toBe(true)
  })
})

describe('readValues', () => {
  it('falls back to defaults for empty or unknown cells', () => {
    expect(readValues({})).toEqual(DEFAULTS)
    const read = readValues({ _dora_enabled: 'false', _dora_mode: 'bogus', _dora_inserted: 'weak', _dora_weak_strength: 0.3 })
    expect(read.enabled).toBe('false')
    expect(read.mode).toBe('lycoris')
    expect(read.inserted).toBe('weak')
    expect(read.weak_strength).toBe('0.3')   // Vue 숫자 입력은 number 로 온다
  })
})

describe('isStock / summary / weak fields', () => {
  it('treats off or forge+keep as stock', () => {
    expect(isStock(values({ enabled: 'false' }))).toBe(true)
    expect(isStock(values({ mode: 'forge', inserted: 'keep' }))).toBe(true)
    expect(isStock(values({ mode: 'forge', inserted: 'weak' }))).toBe(false)
    expect(isStock(values())).toBe(false)
  })

  it('summarises the effective mode', () => {
    expect(summary(values())).toBe('LyCORIS · 그대로 복제')
    expect(summary(values({ enabled: 'false' }))).toBe('')
    expect(summary(values({ mode: 'forge', inserted: 'weak', weak_strength: '0.08', weak_scope: 'attn_mlp' })))
      .toBe('순정 · 약한 복사 0.08 어텐션+MLP')
  })

  it('shows strength and scope only for weak copy', () => {
    expect(showWeakFields(values())).toBe(false)
    expect(showWeakFields(values({ inserted: 'weak' }))).toBe(true)
  })
})

describe('cardStatus', () => {
  it('off wins over everything', () => {
    expect(cardStatus(null, values({ enabled: 'false' })).status).toBe('off')
  })
  it('ComfyUI is not applicable', () => {
    expect(cardStatus(caps({ status: 'not_applicable', known: false }), values()).status).toBe('comfy')
  })
  it('unknown snapshot sends stock', () => {
    expect(cardStatus(null, values()).status).toBe('unknown')
    expect(cardStatus(caps({ status: 'unknown', known: false }), values()).status).toBe('unknown')
  })
  it('missing script and ok', () => {
    expect(cardStatus(caps({}, false), values()).status).toBe('missing')
    expect(cardStatus(caps(), values()).status).toBe('ok')
  })
})
