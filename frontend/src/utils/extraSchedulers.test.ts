import { describe, expect, it } from 'vitest'
import cases from './extraSchedulers.cases.json'
import {
  BETA_RANGE, COMFY_PENDING_NOTE, DEFAULTS, FLOW_COSMOS_DYNAMIC_HINT, FLOW_COSMOS_DYNAMIC_RHO_ABOVE, FLOW_COSMOS_FIELDS,
  FLOW_COSMOS_HINT, FLOW_COSMOS_SCHEDULERS, FLOW_HINT, FLOW_UNSUITABLE, KARRAS_DYNAMIC_HINT, KREA2_NOTE, MODE_OPTIONS,
  MU_RANGE, OLD_FORGE_NOTE, PASS_WIDGETS, REACT_DYNSF_HINT, RHO_RANGE, SCHEDULERS, SIGMA_MAX_RANGE, SIGMA_MIN_RANGE,
  VALUE_SCHEDULERS, WIDGET_IDS, cardNote, flag, flowCosmosDynamicRhoOk, flowCosmosRangeOk, isDefault, passLabel,
  passesUsing, readValues, resolveScheduler, schedulerHints, subtitle, usesCustom, usesFlowCosmos, usesFlowCosmosDynamic,
  usesLaplace,
} from './extraSchedulers'

describe('tables', () => {
  it('widget ids, defaults, ranges, modes', () => {
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_xsched_'))).toBe(true)
    expect(Object.keys(WIDGET_IDS)).toEqual(['custom_mode', 'custom_expression', 'custom_sigmas', 'laplace_mu', 'laplace_beta',
      'flow_cosmos_rho', 'flow_cosmos_sigma_max', 'flow_cosmos_sigma_min'])
    expect(DEFAULTS).toEqual({
      custom_mode: 'expression', custom_expression: 'M * (m / M) ** x', custom_sigmas: '[1.0, 0.6, 0.25, 0.1, 0.0]',
      laplace_mu: '0', laplace_beta: '0.5', flow_cosmos_rho: '7', flow_cosmos_sigma_max: '80', flow_cosmos_sigma_min: '0.002',
    })
    expect(MU_RANGE).toEqual({ min: -10, max: 10, step: 0.1 })
    expect(BETA_RANGE).toEqual({ min: 0, max: 10, step: 0.1 })
    expect(RHO_RANGE).toEqual({ min: 1, max: 15, step: 0.1 })
    expect(SIGMA_MAX_RANGE).toEqual({ min: 1, max: 1000, step: 0.1 })
    expect(SIGMA_MIN_RANGE).toEqual({ min: 0.0001, max: 1, step: 0.0001 })
    expect(FLOW_COSMOS_FIELDS.map(f => f.key)).toEqual(['flow_cosmos_rho', 'flow_cosmos_sigma_max', 'flow_cosmos_sigma_min'])
    expect(MODE_OPTIONS.map(o => o.key)).toEqual(['expression', 'sigmas'])
  })

  it('every scheduler resolves from its name, label and alias; value/flow sets and React DynSF split the nine', () => {
    for (const row of SCHEDULERS) {
      const texts = [row.name, row.label, row.label.toUpperCase(), ...(row.alias ? [row.alias] : [])]
      for (const text of texts) expect(resolveScheduler(text)).toBe(row.name)
    }
    expect(SCHEDULERS).toHaveLength(9)
    expect(SCHEDULERS.filter(r => !r.alias).map(r => r.name))
      .toEqual(['react_cosinusoidal_dynsf', 'flow_cosmos_rho7', 'flow_cosmos_dynamic'])
    expect([...VALUE_SCHEDULERS, ...FLOW_UNSUITABLE, 'react_cosinusoidal_dynsf'].sort())
      .toEqual(SCHEDULERS.map(r => r.name).sort())
    expect(FLOW_COSMOS_SCHEDULERS.every(name => VALUE_SCHEDULERS.includes(name))).toBe(true)
    expect([resolveScheduler(''), resolveScheduler(undefined)]).toEqual([null, null])   // 별칭 없는 줄이 빈 글자를 먹지 않는다
  })
})

describe('golden cases shared with core/extra_schedulers.py', () => {
  it('resolve', () => {
    for (const { text, name } of cases.resolve) expect(resolveScheduler(text), String(text)).toBe(name)
  })

  it('passes', () => {
    for (const item of cases.passes) {
      const got = passesUsing(item.widgets as Record<string, unknown>).map(p => [p.pass, p.scheduler])
      expect(got, item.name).toEqual(item.passes)
    }
  })

  it('flow cosmos checks on the sent (parsed, clamped) values', () => {
    for (const item of cases.flow_cosmos) {
      const values = readValues(Object.fromEntries(Object.entries(item.values)
        .map(([key, value]) => [WIDGET_IDS[key as keyof typeof WIDGET_IDS], value])))
      expect(flowCosmosRangeOk(values), item.name).toBe(item.range_ok)
      expect(flowCosmosDynamicRhoOk(values), item.name).toBe(item.dynamic_rho_ok)
    }
    expect(cases.flow_cosmos.length).toBeGreaterThanOrEqual(10)
    expect(FLOW_COSMOS_DYNAMIC_RHO_ABOVE).toBe(2)
  })
})

describe('passes and labels', () => {
  it('checks accept the store strings and plain booleans', () => {
    expect([flag('true'), flag(' TRUE '), flag(true), flag(1), flag('on')]).toEqual([true, true, true, true, true])
    expect([flag('false'), flag(''), flag(undefined), flag(null), flag(0), flag(false)]).toEqual([false, false, false, false, false, false])
    expect(passesUsing({ [PASS_WIDGETS.main]: 'custom', [PASS_WIDGETS.hiresGroup]: true,
      [PASS_WIDGETS.hiresScheduler]: 'Laplace' })).toEqual([{ pass: 'main', scheduler: 'custom' }, { pass: 'hires', scheduler: 'laplace' }])
  })

  it('subtitle lists each pass once in order; laplace/custom rows follow what is used', () => {
    const passes = passesUsing({ scheduler_combo: 'laplace', hires_options_group: 'true', hires_scheduler_combo: 'custom',
      adetailer_group: 'true', ad_slot1_group: 'true', _ad_s1_use_sampler: 'false' })
    expect(subtitle(passes)).toBe('사용: 메인 · Hires · ADetailer 1')
    expect([usesLaplace(passes), usesCustom(passes)]).toEqual([true, true])
    expect(subtitle([])).toBe('')
    expect(usesCustom(passesUsing({ scheduler_combo: 'laplace' }))).toBe(false)
    expect(usesFlowCosmos(passes)).toBe(false)
    const cosmos = passesUsing({ scheduler_combo: 'Flow Cosmos rho7', hires_options_group: 'true',
      hires_scheduler_combo: 'flow_cosmos_dynamic' })
    expect(subtitle(cosmos)).toBe('사용: 메인 · Hires')
    expect([usesFlowCosmos(cosmos), usesFlowCosmosDynamic(cosmos), usesLaplace(cosmos)]).toEqual([true, true, false])
    expect(usesFlowCosmosDynamic(passesUsing({ scheduler_combo: 'flow_cosmos_rho7' }))).toBe(false)
    expect(['main', 'hires', 'adetailer_2', 'sam3', 'other'].map(passLabel)).toEqual(['메인', 'Hires', 'ADetailer 2', 'SAM3', 'other'])
  })
})

describe('values', () => {
  it('readValues: missing → defaults, mode labels → keys, empty text stays a value', () => {
    expect(readValues({})).toEqual(DEFAULTS)
    expect(readValues({ [WIDGET_IDS.custom_mode]: 'Sigma list' }).custom_mode).toBe('sigmas')
    expect(readValues({ [WIDGET_IDS.custom_mode]: 'weird' }).custom_mode).toBe('expression')
    expect(readValues({ [WIDGET_IDS.custom_expression]: '' }).custom_expression).toBe('')
    expect(readValues({ [WIDGET_IDS.laplace_mu]: 1.5 }).laplace_mu).toBe('1.5')
    expect(readValues({ [WIDGET_IDS.flow_cosmos_sigma_min]: 0.006 }).flow_cosmos_sigma_min).toBe('0.006')
  })

  it('isDefault ignores number spelling and outer spaces', () => {
    expect(isDefault(DEFAULTS)).toBe(true)
    expect(isDefault({ ...DEFAULTS, laplace_beta: '0.50', custom_expression: ' M * (m / M) ** x ' })).toBe(true)
    expect(isDefault({ ...DEFAULTS, laplace_mu: '0.1' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, custom_mode: 'sigmas' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, flow_cosmos_rho: '7.0', flow_cosmos_sigma_max: '80.00' })).toBe(true)
    expect(isDefault({ ...DEFAULTS, flow_cosmos_rho: '5' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, flow_cosmos_sigma_max: '240' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, flow_cosmos_sigma_min: '0.006' })).toBe(false)
  })
})

describe('hints and notes', () => {
  it('flow-unsuitable and Karras Dynamic hints follow main and a separate hires scheduler', () => {
    expect(schedulerHints({ scheduler_combo: 'Laplace' })).toEqual([])
    expect(schedulerHints({ scheduler_combo: 'phi' })).toEqual([FLOW_HINT])
    expect(schedulerHints({ scheduler_combo: 'simple', hires_options_group: 'true', hires_scheduler_combo: 'Karras Dynamic' }))
      .toEqual([FLOW_HINT, KARRAS_DYNAMIC_HINT])
    expect(schedulerHints({ scheduler_combo: 'simple', hires_options_group: 'false', hires_scheduler_combo: 'Cosine' })).toEqual([])
  })

  it('v0.33.0 schedulers have their own hints; values come from the fields', () => {
    expect(schedulerHints({ scheduler_combo: 'React Cosinusoidal DynSF' })).toEqual([REACT_DYNSF_HINT])
    expect(schedulerHints({ scheduler_combo: 'flow_cosmos_rho7' })).toEqual([FLOW_COSMOS_HINT])
    expect(schedulerHints({ scheduler_combo: 'Flow Cosmos Dynamic', hires_options_group: 'true',
      hires_scheduler_combo: 'Flow Cosmos rho7' })).toEqual([FLOW_COSMOS_HINT, FLOW_COSMOS_DYNAMIC_HINT])
    expect(schedulerHints({ scheduler_combo: 'karras_dynamic', hires_options_group: 'true',
      hires_scheduler_combo: 'flow_cosmos_dynamic' })).toEqual([FLOW_HINT, KARRAS_DYNAMIC_HINT, FLOW_COSMOS_DYNAMIC_HINT])
    expect(FLOW_COSMOS_DYNAMIC_HINT).toContain('2 보다 커야')
  })

  it('card note: Krea2, ComfyUI pending, old Forge, otherwise none', () => {
    const forge = (has: boolean) => ({ status: 'ok', known: true, features: { extra_schedulers: has } }) as never
    expect(cardNote(forge(true))).toBe('')
    expect(cardNote(forge(false))).toBe(OLD_FORGE_NOTE)
    expect(cardNote({ status: 'unknown', known: false, features: {} } as never)).toBe('')
    expect(cardNote(null)).toBe('')
    expect(cardNote({ status: 'not_applicable', known: false, features: {} } as never)).toBe(COMFY_PENDING_NOTE)
    expect(cardNote(forge(true), true)).toBe(KREA2_NOTE)
  })
})
