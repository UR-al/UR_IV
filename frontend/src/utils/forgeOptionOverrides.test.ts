import { describe, expect, it } from 'vitest'
import {
  CHOICES, DESCRIPTIONS, GROUPS, PREF_KEY, SPECS, accepts, choiceFromLabel, choiceLabel, choiceOf, isBlocked,
  isEnumSpec, normalizeOverrides, optionStatus, rowChoices, rowWarning, specOf, specsOf, summary, withChoice,
} from './forgeOptionOverrides'

const KEEP_IN_RAM = 'sam3_unload_keep_in_ram'
const DEDUP = 'sam3_guidance_pag_prefix_dedup'
const DAVE = 'sam3_guidance_dave_pre_dd_sigma'
const DEVICE = 'sam3_degrid_device'
const PRECISION = 'sam3_degrid_gpu_precision'
const KEEP_LOADED = 'sam3_degrid_keep_loaded'
const spec = (key: string) => specOf(key)!

const known = (options: Record<string, unknown>, optionsKnown = true) =>
  ({ status: 'ok', known: true, options_known: optionsKnown, options }) as const

describe('forgeOptionOverrides table', () => {
  it('has thirteen options (checkbox or radio) in three groups with a description each', () => {
    expect(PREF_KEY).toBe('forgeOptionOverrides')
    expect(SPECS).toHaveLength(13)
    expect(new Set(SPECS.map(s => s.key)).size).toBe(13)
    for (const item of SPECS) expect(accepts(item, item.default)).toBe(true)
    expect(SPECS.filter(isEnumSpec).map(s => s.key)).toEqual([DEVICE, PRECISION])
    expect(specsOf('result_minor').map(s => s.key)).toContain(PRECISION)
    expect(spec(PRECISION).infotext).toBe('')
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

  it('labels round-trip per row', () => {
    for (const item of SPECS) {
      for (const choice of rowChoices(item)) expect(choiceFromLabel(item, choiceLabel(item, choice.id))).toBe(choice.id)
      expect(choiceFromLabel(item, '???')).toBe('follow')
    }
    expect(choiceLabel(spec(DEDUP), 'nonsense')).toBe('Forge 설정 따름')
  })

  it('summary counts overrides', () => {
    expect(summary({})).toBe('모두 Forge 설정 따름')
    expect(summary({ [DEDUP]: false, [DAVE]: true })).toBe('2개 덮어씀')
  })
})

describe('radio (enum) rows', () => {
  it('rowChoices: checkbox = follow/on/off, radio = follow + extension choices in order', () => {
    expect(rowChoices(spec(DEDUP)).map(c => c.id)).toEqual(CHOICES.map(c => c.id))
    expect(rowChoices(spec(DEVICE))).toEqual([
      { id: 'follow', label: 'Forge 설정 따름' },
      { id: 'auto', label: 'GPU (Forge 장치)' },
      { id: 'cpu', label: 'CPU (VRAM 안 씀, 느림)' },
    ])
    expect(rowChoices(spec(PRECISION)).map(c => c.id)).toEqual(['follow', 'fp32', 'fp16'])
    expect(rowChoices(spec(KEEP_LOADED)).map(c => c.id)).toEqual(['follow', 'on', 'off'])
  })

  it('normalize keeps only choice strings on radio rows and booleans on checkbox rows', () => {
    const raw = { [DEVICE]: 'cpu', [PRECISION]: true, [KEEP_LOADED]: 'true', [DEDUP]: 'auto' }
    expect(normalizeOverrides(raw)).toEqual({ [DEVICE]: 'cpu' })
    expect(normalizeOverrides({ [DEVICE]: 'CPU', [PRECISION]: 'fp64' })).toEqual({})
    expect(normalizeOverrides({ [PRECISION]: 'fp16', [KEEP_LOADED]: false })).toEqual({ [KEEP_LOADED]: false, [PRECISION]: 'fp16' })
    expect(Object.keys(normalizeOverrides({ [PRECISION]: 'fp16', [DEVICE]: 'auto' }))).toEqual([DEVICE, PRECISION])
  })

  it('choiceOf / withChoice round-trip radio values; foreign choices are ignored', () => {
    const cpu = withChoice({}, DEVICE, 'cpu')
    expect(cpu).toEqual({ [DEVICE]: 'cpu' })
    expect(choiceOf(cpu, DEVICE)).toBe('cpu')
    expect(withChoice(cpu, DEVICE, 'follow')).toEqual({})
    expect(withChoice(cpu, DEVICE, 'on')).toEqual({ [DEVICE]: 'cpu' })        // 체크박스 선택지는 라디오에 없다
    expect(withChoice(cpu, DEVICE, 'gpu')).toEqual({ [DEVICE]: 'cpu' })
    expect(withChoice({}, DEDUP, 'cpu')).toEqual({})                          // 라디오 값은 체크박스에 없다
    expect(withChoice(cpu, KEEP_LOADED, 'on')).toEqual({ [DEVICE]: 'cpu', [KEEP_LOADED]: true })
    expect(choiceOf({ [DEVICE]: true }, DEVICE)).toBe('follow')
    expect(summary({ [DEVICE]: 'cpu', [PRECISION]: 'fp16', [DEDUP]: false })).toBe('3개 덮어씀')
  })

  it('status shows the Forge radio value as text; blocked rules are unchanged', () => {
    const ok = optionStatus(known({ [DEVICE]: 'cpu' }), DEVICE)
    expect(ok).toMatchObject({ state: 'ok', forgeValue: 'cpu', text: 'Forge 값: cpu (Forge 시작 시점)' })
    expect(optionStatus(known({ [DEVICE]: true }), DEVICE).forgeValue).toBeNull()
    expect(optionStatus(known({ [KEEP_LOADED]: 'x' }), KEEP_LOADED).forgeValue).toBeNull()
    expect(isBlocked(optionStatus(known({}), PRECISION), 'fp16')).toBe(true)
    expect(isBlocked(optionStatus(known({ [PRECISION]: 'fp32' }), PRECISION), 'fp16')).toBe(false)
  })

  it('warnings for cpu and fp16', () => {
    expect(rowWarning(DEVICE, 'cpu', 'auto')).toContain('6.5초')
    expect(rowWarning(DEVICE, 'auto', 'cpu')).toBe('')
    expect(rowWarning(PRECISION, 'fp16', 'fp32')).toContain('fp32 로 다시')
    expect(rowWarning(PRECISION, 'fp32', 'fp16')).toBe('')
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
