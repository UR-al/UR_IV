import { describe, expect, it } from 'vitest'
import cases from './extraSchedulers.cases.json'
import {
  BETA_RANGE, COMFY_PENDING_NOTE, DEFAULTS, FLOW_HINT, FLOW_UNSUITABLE, KARRAS_DYNAMIC_HINT, KREA2_NOTE, MODE_OPTIONS,
  MU_RANGE, OLD_FORGE_NOTE, PASS_WIDGETS, SCHEDULERS, VALUE_SCHEDULERS, WIDGET_IDS, cardNote, flag, isDefault,
  passLabel, passesUsing, readValues, resolveScheduler, schedulerHints, subtitle, usesCustom, usesLaplace,
} from './extraSchedulers'

describe('tables', () => {
  it('widget ids, defaults, ranges, modes', () => {
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_xsched_'))).toBe(true)
    expect(Object.keys(WIDGET_IDS)).toEqual(['custom_mode', 'custom_expression', 'custom_sigmas', 'laplace_mu', 'laplace_beta'])
    expect(DEFAULTS).toEqual({
      custom_mode: 'expression', custom_expression: 'M * (m / M) ** x', custom_sigmas: '[1.0, 0.6, 0.25, 0.1, 0.0]',
      laplace_mu: '0', laplace_beta: '0.5',
    })
    expect(MU_RANGE).toEqual({ min: -10, max: 10, step: 0.1 })
    expect(BETA_RANGE).toEqual({ min: 0, max: 10, step: 0.1 })
    expect(MODE_OPTIONS.map(o => o.key)).toEqual(['expression', 'sigmas'])
  })

  it('every scheduler resolves from its name, label and alias; value/flow sets split the six', () => {
    for (const row of SCHEDULERS) {
      for (const text of [row.name, row.label, row.alias, row.label.toUpperCase()]) expect(resolveScheduler(text)).toBe(row.name)
    }
    expect([...VALUE_SCHEDULERS, ...FLOW_UNSUITABLE].sort()).toEqual(SCHEDULERS.map(r => r.name).sort())
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
  })

  it('isDefault ignores number spelling and outer spaces', () => {
    expect(isDefault(DEFAULTS)).toBe(true)
    expect(isDefault({ ...DEFAULTS, laplace_beta: '0.50', custom_expression: ' M * (m / M) ** x ' })).toBe(true)
    expect(isDefault({ ...DEFAULTS, laplace_mu: '0.1' })).toBe(false)
    expect(isDefault({ ...DEFAULTS, custom_mode: 'sigmas' })).toBe(false)
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
