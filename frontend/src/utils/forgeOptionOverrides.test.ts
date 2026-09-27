import { describe, expect, it } from 'vitest'
import {
  CHOICES, DESCRIPTIONS, GROUPS, PREF_KEY, SPECS, choiceFromLabel, choiceLabel, choiceOf, isBlocked,
  normalizeOverrides, optionStatus, rowWarning, specsOf, summary, withChoice,
} from './forgeOptionOverrides'

const KEEP_IN_RAM = 'sam3_unload_keep_in_ram'
const DEDUP = 'sam3_guidance_pag_prefix_dedup'
const DAVE = 'sam3_guidance_dave_pre_dd_sigma'

const known = (options: Record<string, unknown>, optionsKnown = true) =>
  ({ status: 'ok', known: true, options_known: optionsKnown, options }) as const

describe('forgeOptionOverrides table', () => {
  it('has eight boolean options in three groups with a description each', () => {
    expect(PREF_KEY).toBe('forgeOptionOverrides')
    expect(SPECS).toHaveLength(8)
    expect(new Set(SPECS.map(s => s.key)).size).toBe(8)
    expect(GROUPS.map(g => g.id)).toEqual(['memory', 'result_minor', 'result'])
    expect(GROUPS.flatMap(g => specsOf(g.id)).map(s => s.key)).toEqual(SPECS.map(s => s.key))
    for (const spec of SPECS) expect(DESCRIPTIONS[spec.key]?.length).toBeGreaterThan(10)
    expect(specsOf('result').map(s => s.key)).toContain(DAVE)
    expect(CHOICES.map(c => c.id)).toEqual(['follow', 'on', 'off'])
  })
})

describe('normalizeOverrides / withChoice', () => {
  it('keeps real booleans on spec keys only (1, "true", null and unknown keys = follow)', () => {
    const raw = { [DEDUP]: false, [KEEP_IN_RAM]: 'false', [DAVE]: 1, sd_model_checkpoint: true, sam3_x: null }
    const before = JSON.stringify(raw)
    expect(normalizeOverrides(raw)).toEqual({ [DEDUP]: false })
    expect(JSON.stringify(raw)).toBe(before)
    expect(normalizeOverrides(null)).toEqual({})
    expect(normalizeOverrides([true])).toEqual({})
  })

  it('follow deletes the key, on/off store booleans, input is not mutated', () => {
    const start = { [DEDUP]: false }
    const on = withChoice(start, KEEP_IN_RAM, 'on')
    expect(on).toEqual({ [KEEP_IN_RAM]: true, [DEDUP]: false })
    expect(start).toEqual({ [DEDUP]: false })
    expect(withChoice(on, DEDUP, 'follow')).toEqual({ [KEEP_IN_RAM]: true })
    expect(withChoice({}, DAVE, 'off')).toEqual({ [DAVE]: false })
    expect(withChoice({}, 'sd_model_checkpoint', 'on')).toEqual({})
    expect(choiceOf({ [DAVE]: false }, DAVE)).toBe('off')
    expect(choiceOf({ [DAVE]: true }, DAVE)).toBe('on')
    expect(choiceOf({}, DAVE)).toBe('follow')
  })

  it('labels round-trip', () => {
    for (const choice of CHOICES) expect(choiceFromLabel(choiceLabel(choice.id))).toBe(choice.id)
    expect(choiceFromLabel('???')).toBe('follow')
  })

  it('summary counts overrides', () => {
    expect(summary({})).toBe('모두 Forge 설정 따름')
    expect(summary({ [DEDUP]: false, [DAVE]: true })).toBe('2개 덮어씀')
  })
})

describe('optionStatus', () => {
  it('comfy, unknown, unverified, missing, ok — like plan_overrides', () => {
    expect(optionStatus({ status: 'not_applicable', known: false }, DEDUP).state).toBe('comfy')
    expect(optionStatus(null, DEDUP).state).toBe('unknown')
    expect(optionStatus({ status: 'unknown', known: false }, DEDUP).state).toBe('unknown')
    expect(optionStatus(known({}, false), DEDUP).state).toBe('unverified')
    expect(optionStatus(known({ sam3_other: true }), DEDUP).state).toBe('missing')
    const ok = optionStatus(known({ [DEDUP]: false }), DEDUP)
    expect(ok).toMatchObject({ state: 'ok', forgeValue: false })
    expect(ok.text).toContain('끔')
    expect(optionStatus(known({ [DEDUP]: 'x' }), DEDUP).forgeValue).toBeNull()
  })

  it('blocked only when a value is chosen but cannot be sent', () => {
    const missing = optionStatus(known({}), DEDUP)
    expect(isBlocked(missing, 'off')).toBe(true)
    expect(isBlocked(missing, 'follow')).toBe(false)
    expect(isBlocked(optionStatus(known({ [DEDUP]: true }), DEDUP), 'off')).toBe(false)
    expect(isBlocked(optionStatus(null, DEDUP), 'on')).toBe(true)
  })
})

describe('rowWarning', () => {
  it('keep_in_ram on while Forge is off warns about the restore callback', () => {
    expect(rowWarning(KEEP_IN_RAM, 'on', false)).toContain('한 요청 안')
    expect(rowWarning(KEEP_IN_RAM, 'on', true)).toBe('')
    expect(rowWarning(KEEP_IN_RAM, 'off', false)).toBe('')
  })

  it('keep_resident on, sparse re-merge and DAVE off have their notes', () => {
    expect(rowWarning('sam3_anima38_keep_resident', 'on', null)).toContain('학습')
    expect(rowWarning('sam3_anima38_keep_resident', 'follow', null)).toBe('')
    expect(rowWarning('sam3_anima_sparse_lora_forge_guess', 'on', false)).toContain('다시 합칩니다')
    expect(rowWarning(DAVE, 'off', true)).toContain('무너집니다')
    expect(rowWarning(DAVE, 'on', true)).toBe('')
  })
})
