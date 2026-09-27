import { describe, expect, it } from 'vitest'
import type { SamExtraCapabilitiesEvent } from '../types/bridge'
import {
  COMFY_ADAPTER_FALLBACK, COMFY_ADAPTER_PROPERTY, DEFAULTS, DEFAULT_ADAPTER, KIND_PROPERTY, STRENGTH_RANGE, WIDGET_IDS,
  adapterChoices, cardStatus, cardView, modelKind, readValues, summary, type Anima38Values,
} from './anima38Card'

function caps(over: Partial<SamExtraCapabilitiesEvent> = {}, anima38 = true): SamExtraCapabilitiesEvent {
  return {
    status: 'ok', known: true, checked_at: 't', installed: true,
    features: { anima38 } as SamExtraCapabilitiesEvent['features'],
    anima_guidance_argc: null, detail_daemon_hires: null, version: {}, scripts: {}, sam3_keys: {},
    options: {}, options_known: false, gradio_api: [], choices: {}, warnings: [], errors: {},
    ...over,
  }
}

const values = (over: Partial<Anima38Values> = {}): Anima38Values => ({ ...DEFAULTS, ...over })

describe('anima38Card constants', () => {
  it('app defaults follow the Forge txt2img ui-config with v1 off (D1=B)', () => {
    expect(DEFAULTS).toEqual({
      enabled: 'false', adapter: DEFAULT_ADAPTER, strength: '1', negative: 'true', negative_strength: '1',
      bypass: 'false', apply_img2img: 'false',
    })
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_a38_'))).toBe(true)
    expect(KIND_PROPERTY).toBe('animaKinds')
    expect(STRENGTH_RANGE).toEqual({ min: 0, max: 2, step: 0.05 })
  })
})

describe('readValues / modelKind', () => {
  it('falls back to defaults for empty cells and keeps store strings', () => {
    expect(readValues({})).toEqual(DEFAULTS)
    const read = readValues({ _a38_negative: 'false', _a38_strength: 0.65, _a38_adapter: '' })
    expect(read.negative).toBe('false')
    expect(read.strength).toBe('0.65')          // Vue 숫자 입력은 number 로 온다
    expect(read.adapter).toBe(DEFAULT_ADAPTER)
  })

  it('reads the kind of the current model, unknown otherwise', () => {
    const kinds = { 'Anima-3.8B-v1.1.safetensors': 'v2', 'sdxl.safetensors': 'other', bad: 'mystery' }
    expect(modelKind(kinds, 'Anima-3.8B-v1.1.safetensors')).toBe('v2')
    expect(modelKind(kinds, 'sdxl.safetensors')).toBe('other')
    expect(modelKind(kinds, 'bad')).toBe('unknown')
    expect(modelKind(kinds, 'missing')).toBe('unknown')
    expect(modelKind(undefined, 'x')).toBe('unknown')
    expect(modelKind('', 'x')).toBe('unknown')
  })
})

describe('summary follows the Python effective rule', () => {
  it('v2 shows only negative connector or bypass', () => {
    expect(summary('v2', values())).toBe('부정 커넥터')
    expect(summary('v2', values({ negative: 'false' }))).toBe('')
    expect(summary('v2', values({ bypass: 'true' }))).toBe('Bypass')
    expect(summary('v2', values({ negative: 'false', enabled: 'true' }))).toBe('')   // v2 는 v1 을 쓰지 않는다
  })

  it('non-bundle Anima shows v1 only when enabled and not bypassed', () => {
    expect(summary('anima', values())).toBe('')
    expect(summary('anima', values({ enabled: 'true', strength: '0.65' }))).toBe('v1 0.65 · 부정 1')
    expect(summary('anima', values({ enabled: 'true', negative: 'false' }))).toBe('v1 1')
    expect(summary('anima', values({ enabled: 'true', bypass: 'true' }))).toBe('')
  })

  it('other is empty and unknown lists everything on', () => {
    expect(summary('other', values({ enabled: 'true' }))).toBe('')
    expect(summary('unknown', values({ enabled: 'true' }))).toBe('부정 커넥터 · v1 1')
    expect(summary('unknown', values({ bypass: 'true' }))).toBe('Bypass')
  })
})

describe('cardView', () => {
  it('hides for non-Anima and Krea2, keeps the fields that each kind uses', () => {
    expect(cardView('other', values()).visible).toBe(false)
    expect(cardView('v2', values(), true).visible).toBe(false)
    expect(cardView('v2', values())).toEqual({
      visible: true, showNegativeStrength: false, v2FixedNote: true, showBypass: true, v1BypassedNote: false,
      showV1: false,
    })
    expect(cardView('anima', values())).toEqual({
      visible: true, showNegativeStrength: true, v2FixedNote: false, showBypass: false, v1BypassedNote: false,
      showV1: true,
    })
    expect(cardView('unknown', values({ negative: 'false' }))).toEqual({
      visible: true, showNegativeStrength: false, v2FixedNote: false, showBypass: true, v1BypassedNote: false,
      showV1: true,
    })
  })

  it('keeps Bypass visible on non-bundle Anima while it is on — it still turns v1 off (P9 review 3)', () => {
    // 3.8B 에서 Bypass 를 켠 채 2.9B·UR_ANIMA 로 바꾸고 v1 을 켜면 파이썬 plan 은 블록을 보내지 않는다
    // (확장 process_batch 가 bypass 면 v1 설치 전에 돌아간다). 되돌릴 칸이 숨으면 안 된다.
    const stuck = cardView('anima', values({ enabled: 'true', bypass: 'true' }))
    expect(stuck.showBypass).toBe(true)
    expect(stuck.v1BypassedNote).toBe(true)
    expect(summary('anima', values({ enabled: 'true', bypass: 'true' }))).toBe('')
    const bypassOnly = cardView('anima', values({ bypass: 'true' }))
    expect(bypassOnly.showBypass).toBe(true)
    expect(bypassOnly.v1BypassedNote).toBe(false)                 // v1 을 안 켰으면 잃는 것이 없다
    expect(cardView('anima', values({ enabled: 'true' })).showBypass).toBe(false)   // 끈 Bypass 는 계속 숨긴다
    expect(cardView('v2', values({ bypass: 'true', enabled: 'true' })).v1BypassedNote).toBe(false)
  })
})

describe('adapterChoices', () => {
  it('treats the single fallback name as unverified and keeps the saved value', () => {
    expect(adapterChoices(null, DEFAULT_ADAPTER)).toEqual({ options: [DEFAULT_ADAPTER], unverified: true })
    expect(adapterChoices(caps({ choices: { anima38_adapters: [DEFAULT_ADAPTER] } }), DEFAULT_ADAPTER).unverified)
      .toBe(true)
    const real = adapterChoices(caps({ choices: { anima38_adapters: ['a.safetensors', 'b.safetensors'] } }), 'old.safetensors')
    expect(real).toEqual({ options: ['old.safetensors', 'a.safetensors', 'b.safetensors'], unverified: false })
  })

  it('uses the ComfyUI object_info adapters when Python sent them (P9 review 2)', () => {
    expect(COMFY_ADAPTER_PROPERTY).toBe('comfyAdapters')
    const comfy = caps({ status: 'not_applicable', known: false })
    const alt = 'text/MyAnima_v1_adapter.safetensors'
    // 기본 이름이 아닌 어댑터만 있는 ComfyUI — 예전에는 기본 이름 하나뿐이라 v1 을 켜면 컴파일 오류였다
    expect(adapterChoices(comfy, DEFAULT_ADAPTER, [alt])).toEqual({ options: [DEFAULT_ADAPTER, alt], unverified: false })
    expect(adapterChoices(comfy, alt, [alt])).toEqual({ options: [alt], unverified: false })
    expect(adapterChoices(comfy, alt, [COMFY_ADAPTER_FALLBACK]).unverified).toBe(true)   // 팩 폴백 이름뿐
    expect(adapterChoices(comfy, alt, []).unverified).toBe(true)
    // 배열이 아니면(Forge·모름) 기능 스냅샷 규칙 그대로
    const forge = caps({ choices: { anima38_adapters: ['a.safetensors'] } })
    expect(adapterChoices(forge, 'a.safetensors', null)).toEqual({ options: ['a.safetensors'], unverified: false })
  })

  it('never offers the ComfyUI pack placeholder when no adapter is installed (P9 review round 2, 1)', () => {
    const comfy = caps({ status: 'not_applicable', known: false })
    // 팩이 못 찾으면 콤보는 자리표시자뿐 — 파이썬(comfy_adapter_choices)이 빼고 빈 배열을 보낸다
    expect(adapterChoices(comfy, DEFAULT_ADAPTER, [])).toEqual({ options: [DEFAULT_ADAPTER], unverified: true })
    const saved = 'text/MyAnima_v1_adapter.safetensors'
    expect(adapterChoices(comfy, saved, [])).toEqual({ options: [saved, DEFAULT_ADAPTER], unverified: true })
    expect(adapterChoices(comfy, DEFAULT_ADAPTER, []).options).not.toContain(COMFY_ADAPTER_FALLBACK)
    // 이름이 그대로 오면 같은 이름의 파일이 text_encoders 에 있다 — 고를 수 있어야 한다(컴파일 오류가 이 이름을 권한다)
    expect(adapterChoices(comfy, DEFAULT_ADAPTER, [COMFY_ADAPTER_FALLBACK]))
      .toEqual({ options: [DEFAULT_ADAPTER, COMFY_ADAPTER_FALLBACK], unverified: true })
  })
})

describe('cardStatus', () => {
  it('orders comfy, missing, unknown snapshot, then model kind', () => {
    expect(cardStatus(caps({ status: 'not_applicable', known: false }), 'v2').status).toBe('comfy')
    expect(cardStatus(caps({}, false), 'v2').status).toBe('missing')
    expect(cardStatus(null, 'v2').status).toBe('unknown_caps')
    expect(cardStatus(caps({ known: false, status: 'unknown' }), 'v2').status).toBe('unknown_caps')
    expect(cardStatus(caps(), 'v2').status).toBe('v2')
    expect(cardStatus(caps(), 'anima').status).toBe('anima')
    expect(cardStatus(caps(), 'unknown').status).toBe('unknown_model')
  })

  it('Comfy wording says missing files fail before queueing (critic B8)', () => {
    expect(cardStatus(caps({ status: 'not_applicable', known: false }), 'anima').text).toContain('생성 전에 오류')
  })
})
