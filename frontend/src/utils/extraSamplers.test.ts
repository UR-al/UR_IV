import { describe as group, expect, it } from 'vitest'
import cases from './extraSamplers.cases.json'
import {
  ARG_COUNT, CFG_PP_LABELS, COMFY_PARAMS_NOTE, DEFAULTS, DEFAULT_SETTINGS, DY_CFGPP_HINT, DY_PLAIN_HINT, ER_SDE_LABELS,
  ETA_LABELS, ETA_RANGE, FLOW_ODE_HINT, FLOW_ODE_LABELS, IPNDM_HINT, KREA2_NOTE, NOISE_RANGE, OLD_FORGE_NOTE, PASS_WIDGETS,
  RESTART_HINT, SAMPLERS, SCRIPT_TITLE, SMEA_CFGPP_HINT, STAGE_OPTIONS, STAGE_RANGE, TUNABLE_HINT, UD10_HINT, WIDGET_IDS,
  FORGE_PARAMS_NOTE, cardNote, cfgWarning, describe, hiddenNote, isCfgPp, isDefault, paramsNote, parseValues,
  passesUsing, readValues, resolveSampler, samplerHints, samplerKind, subtitle, usesEta, widgetNumber, windowEmpty, windowSummary, windowSupported,
  windowUnsupported, type XsampKey,
} from './extraSamplers'

const valuesOf = (raw: Record<string, string>) => readValues(Object.fromEntries(Object.entries(raw)
  .map(([key, value]) => [WIDGET_IDS[key as XsampKey], value])))

group('tables', () => {
  it('widget ids, defaults, ranges', () => {
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_xsamp_'))).toBe(true)
    expect(Object.keys(WIDGET_IDS)).toEqual(['max_stage', 'eta', 'noise_window', 'noise_start', 'noise_end'])
    expect(Object.keys(WIDGET_IDS)).toHaveLength(ARG_COUNT)
    expect(DEFAULTS).toEqual({ max_stage: '3', eta: '1', noise_window: '0', noise_start: '0.2', noise_end: '0.8' })
    expect(parseValues(DEFAULTS)).toEqual(DEFAULT_SETTINGS)
    expect(STAGE_RANGE).toEqual({ min: 1, max: 3, step: 1 })
    expect(ETA_RANGE).toEqual({ min: 0, max: 10, step: 0.01 })
    expect(NOISE_RANGE).toEqual({ min: 0, max: 1, step: 0.01 })
    expect(STAGE_OPTIONS).toEqual(['1', '2', '3'])
    expect(SCRIPT_TITLE).toBe('extra samplers')
  })

  it('eighteen samplers resolve from label, Comfy name and alias; kinds and groups agree with the rows', () => {
    expect(SAMPLERS).toHaveLength(18)
    for (const row of SAMPLERS) {
      for (const text of [row.label, row.name, row.label.toUpperCase(), ` ${row.name} `, ...(row.alias ? [row.alias] : [])]) {
        expect(resolveSampler(text), text).toBe(row.label)
        expect(samplerKind(text), text).toBe(row.kind)
      }
    }
    expect(SAMPLERS.filter(r => r.kind === 'er_sde').map(r => r.label)).toEqual(ER_SDE_LABELS)
    expect(ETA_LABELS.every(label => ER_SDE_LABELS.includes(label))).toBe(true)
    expect(SAMPLERS.filter(r => r.alias).map(r => r.alias)).toEqual(['k_euler_dy', 'k_euler_smea_dy', 'k_dpmpp_2m_sde_heun'])
    expect(FLOW_ODE_LABELS.every(label => SAMPLERS.some(r => r.label === label && r.kind === ''))).toBe(true)
    expect(CFG_PP_LABELS.map(isCfgPp)).toEqual([true, true, true])
    expect([isCfgPp('Euler Dy'), isCfgPp('euler_cfg_pp'), isCfgPp('')]).toEqual([false, false, false])
    expect([resolveSampler(''), resolveSampler(undefined), samplerKind('Euler a')]).toEqual([null, null, null])
  })
})

group('golden cases made by core/extra_samplers.py', () => {
  it('resolve', () => {
    for (const { text, label, kind } of cases.resolve) {
      expect(resolveSampler(text), text).toBe(label)
      expect(samplerKind(text), text).toBe(kind)
    }
    expect(cases.resolve.length).toBeGreaterThanOrEqual(40)
  })

  it('passes', () => {
    for (const item of cases.passes) {
      const got = passesUsing(item.widgets as Record<string, unknown>).map(p => [p.pass, p.sampler])
      expect(got, item.name).toEqual(item.passes)
    }
    expect(cases.passes.length).toBeGreaterThanOrEqual(20)
  })

  it('settings: parse, empty window, describe', () => {
    const samplers = cases.describe_samplers as Record<string, string[] | null>
    for (const item of cases.settings) {
      const settings = parseValues(valuesOf(item.values as Record<string, string>))
      expect(settings, item.name).toEqual(item.settings)
      expect(windowEmpty(settings), item.name).toBe(item.window_empty)
      for (const [key, expected] of Object.entries(item.describe)) {
        const list = samplers[key]
        expect(describe(settings, list ?? undefined), `${item.name} / ${key}`).toBe(expected)
      }
    }
    expect(cases.settings.length).toBeGreaterThanOrEqual(40)
  })

  it('noise window support from the snapshot', () => {
    const on = parseValues({ ...DEFAULTS, noise_window: '1' })
    for (const item of cases.window) {
      const caps = item.caps as never
      expect(windowSupported(caps), item.name).toBe(item.supported)
      expect(windowUnsupported(on, caps), item.name).toBe(item.unsupported_when_on)
      expect(windowUnsupported(DEFAULT_SETTINGS, caps), item.name).toBe(item.unsupported_when_off)
    }
    expect(windowSupported(null)).toBeNull()
  })
})

group('values', () => {
  it('readValues: missing or null → defaults, everything else as text', () => {
    expect(readValues({})).toEqual(DEFAULTS)
    expect(readValues({ [WIDGET_IDS.eta]: 0.5, [WIDGET_IDS.max_stage]: null }))
      .toEqual({ ...DEFAULTS, eta: '0.5' })
    expect(readValues({ [WIDGET_IDS.noise_window]: true }).noise_window).toBe('true')
    expect(parseValues(readValues({ [WIDGET_IDS.noise_window]: true })).noise_window).toBe(true)
  })

  it('isDefault ignores number spelling, not an emptied field', () => {
    expect(isDefault(DEFAULTS)).toBe(true)
    expect(isDefault({ ...DEFAULTS, eta: '1.0', max_stage: '3.0', noise_start: '0.20', noise_window: 'false' })).toBe(true)
    expect(isDefault({ ...DEFAULTS, eta: '' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, max_stage: '2' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, noise_window: '1' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, noise_end: '0.9' })).toBe(false)
  })

  it('window summary and number spelling follow python', () => {
    expect(windowSummary(DEFAULT_SETTINGS)).toBe('')
    expect(windowSummary({ ...DEFAULT_SETTINGS, noise_window: true })).toBe('잡음 0.2–0.8')
    expect(windowSummary({ ...DEFAULT_SETTINGS, noise_window: true, noise_start: 0.8 })).toBe('잡음 0.8–0.8 (빈 구간)')
    expect([0, 1, 2.5, 0.1, 1e-5, 1.5e-5, 0.0001, 1e16].map(widgetNumber))
      .toEqual(['0', '1', '2.5', '0.1', '1e-05', '1.5e-05', '0.0001', '1e+16'])
    expect(hiddenNote({ ...DEFAULT_SETTINGS, max_stage: 2 }))
      .toBe('ER SDE 값(차수 2 · η 1)은 ER SDE (Reverse-time)·(ODE)·(Tunable) 을 고를 때만 쓰입니다.')
  })
})

group('passes, hints and notes', () => {
  it('subtitle lists each pass once; η follows Reverse-time and Tunable only', () => {
    const passes = passesUsing({ sampler_combo: 'ER SDE (ODE)', hires_options_group: 'true',
      hires_sampler_combo: 'er_sde_tunable', adetailer_group: 'true', ad_slot1_group: 'true' })
    expect(subtitle(passes)).toBe('사용: 메인 · Hires · ADetailer 1')
    expect(usesEta(passes)).toBe(true)
    expect(usesEta(passesUsing({ sampler_combo: 'ER SDE (ODE)' }))).toBe(false)
    expect(subtitle([])).toBe('')
    expect(Object.keys(PASS_WIDGETS)).toContain('sam3UseSampler')
  })

  it('one-line hints follow main and a separate hires sampler', () => {
    expect(samplerHints({ sampler_combo: 'ER SDE (Reverse-time)' })).toEqual([])
    expect(samplerHints({ sampler_combo: 'ER SDE (Tunable)' })).toEqual([TUNABLE_HINT])
    expect(samplerHints({ sampler_combo: 'Euler', hires_options_group: 'true', hires_sampler_combo: 'IPNDM' }))
      .toEqual([IPNDM_HINT])
    expect(samplerHints({ sampler_combo: 'Euler', hires_options_group: 'false', hires_sampler_combo: 'IPNDM' })).toEqual([])
    expect(samplerHints({ sampler_combo: 'ipndm_v' })).toEqual([])
    expect(samplerHints({ sampler_combo: 'dpmpp_3m_flow_ode', hires_options_group: 'true',
      hires_sampler_combo: 'DPM++ 2M (flow ODE)' })).toEqual([FLOW_ODE_HINT])
    expect(samplerHints({ sampler_combo: 'Restart (flow)' })).toEqual([RESTART_HINT])
    expect(samplerHints({ sampler_combo: 'Euler Dy', hires_options_group: 'true', hires_sampler_combo: 'Euler SMEA Dy' }))
      .toEqual([DY_PLAIN_HINT])
    expect(samplerHints({ sampler_combo: 'Euler Dy CFG++', hires_options_group: 'true',
      hires_sampler_combo: 'euler_smea_dy_cfg_pp' })).toEqual([DY_CFGPP_HINT, SMEA_CFGPP_HINT])
    expect(samplerHints({ sampler_combo: 'CFG++ UD10 AB', cfg_input: 2 })).toEqual([UD10_HINT])
  })

  it('CFG++ samplers warn above CFG 2', () => {
    expect(samplerHints({ sampler_combo: 'CFG++ UD10 AB', cfg_input: '2.5' })).toEqual([UD10_HINT, cfgWarning(2.5)])
    expect(samplerHints({ sampler_combo: 'Euler Dy CFG++', cfg_input: 7 })).toEqual([DY_CFGPP_HINT, cfgWarning(7)])
    expect(samplerHints({ sampler_combo: 'Euler Dy', cfg_input: 7 })).toEqual([DY_PLAIN_HINT])
    expect(samplerHints({ sampler_combo: 'Euler SMEA Dy CFG++', cfg_input: 'x' })).toEqual([SMEA_CFGPP_HINT])
    expect(cfgWarning(2.5)).toBe('CFG++ 샘플러는 CFG 1~2 를 권장합니다 (지금 2.5)')
  })

  it('card note: Krea2, old Forge, otherwise none; ComfyUI gets the params note instead', () => {
    const forge = (has: boolean) => ({ status: 'ok', known: true, features: { extra_samplers: has } }) as never
    expect(cardNote(forge(true))).toBe('')
    expect(cardNote(forge(false))).toBe(OLD_FORGE_NOTE)
    expect(cardNote({ status: 'unknown', known: false, features: {} } as never)).toBe('')
    expect(cardNote(null)).toBe('')
    // 계획 C6 — ComfyUI 는 경고가 아니다: 안내 줄이 Forge 설정 줄 대신 ComfyUI 줄이 된다
    expect(cardNote({ status: 'not_applicable', known: false, features: {} } as never)).toBe('')
    expect(paramsNote({ status: 'not_applicable' } as never)).toBe(COMFY_PARAMS_NOTE)
    expect(paramsNote({ status: 'ok' } as never)).toBe(FORGE_PARAMS_NOTE)
    expect(paramsNote(null)).toBe(FORGE_PARAMS_NOTE)
    expect(cardNote(forge(true), true)).toBe(KREA2_NOTE)
  })
})
