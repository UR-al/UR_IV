import { describe, expect, it } from 'vitest'
import stateGolden from '../../../tests/fixtures/colorcraft_state_golden.json'
import cases from './colorcraft.cases.json'
import {
  COMBO_TAGS, DEFAULT_STATE, MARK_EDITED, MARK_ON, MASK_TAGS, MODIFIER_TAGS, NO_ENTRIES_NOTE, OFF_NOTE, OLD_FORGE_NOTE,
  COMFY_PENDING_NOTE, KREA2_NOTE, SCHEMA, STATE_PATHS, StateError, WIDGET_IDS, badge, buildChain, canonicalState,
  cardStatus, choiceLabel, comboRefChoices, configFromMapping, configFromState, defaultConfig, dumpState, edited,
  formatG10, outOfRange, overview, parseState, readCard, resetNode, scheduleValues, setPath, toInfotext, visibleGroups,
} from './colorcraft'

const caps = (colorcraft: boolean, status: 'ok' | 'not_applicable' = 'ok') =>
  ({ status, known: true, features: { colorcraft } as Record<string, boolean> }) as never

describe('tables', () => {
  it('widget ids, tags and state paths', () => {
    expect(Object.values(WIDGET_IDS)).toEqual(['_cc_enabled', '_cc_masking', '_cc_img2img', '_cc_state'])
    expect(MODIFIER_TAGS).toHaveLength(10)
    expect(MASK_TAGS).toHaveLength(10)
    expect(COMBO_TAGS).toHaveLength(5)
    expect(STATE_PATHS).toHaveLength(10 * 44 + 10 * 10 + 5 * 7)
    expect(STATE_PATHS[0]).toBe('I.active')
    expect(STATE_PATHS[STATE_PATHS.length - 1]).toBe('C5.normalize')
    expect(comboRefChoices(0)).toEqual(['none', ...MASK_TAGS])
    expect(comboRefChoices(2)).toEqual(['none', ...MASK_TAGS, 'C1', 'C2'])
  })

  it('visible groups follow the Type with Advanced gates', () => {
    expect(visibleGroups('punch').map(g => g.group)).toEqual(['contrast', 'detail'])
    expect(visibleGroups('basic').map(g => g.group)).toEqual(['contrast', 'color_shift'])
    const advanced = visibleGroups('advanced')
    expect(advanced.map(g => g.group)).toEqual(SCHEMA.groupOrder)
    expect(advanced.find(g => g.group === 'chroma_plus')!.gate?.name).toBe('more_colors')
    expect(advanced.find(g => g.group === 'color_shift')!.gate?.name).toBe('color_shift')
    const shift = visibleGroups('shift')
    expect(shift[0]!.gate).toBeNull()                                  // Shift 노드에는 스위치가 없다
    expect(shift[0]!.fields.map(f => f.name)).not.toContain('color_shift')
    expect(choiceLabel('kind', 'Punch')).toBe('Punch (대비·디테일)')
    expect(choiceLabel('kind', 'Nope')).toBe('Nope')
  })
})

describe('state string golden shared with core/colorcraft.py', () => {
  it('canonical(input) == state and parse(state) == paths', () => {
    expect(stateGolden.version).toBe(SCHEMA.version)
    for (const item of stateGolden.cases) {
      expect(canonicalState(item.input), item.name).toBe(item.state)
      expect(parseState(item.state), item.name).toEqual(item.paths)
      expect(canonicalState(item.state), item.name).toBe(item.state)
      expect(dumpState(configFromMapping(item.paths as Record<string, unknown>)), item.name).toBe(item.state)
    }
    expect(stateGolden.cases.length).toBeGreaterThanOrEqual(20)
  })

  it('refuses what is not a state', () => {
    expect(() => parseState('{not json')).toThrow(StateError)
    expect(() => parseState('[1]')).toThrow(StateError)
    expect(() => parseState('{"v":2}')).toThrow(StateError)
    expect(() => parseState('{"v":true}')).toThrow(StateError)
    expect(() => parseState(12)).toThrow(StateError)
    expect(parseState('  ')).toEqual({})
    expect(parseState({ v: 1, 'I.exposure': 0.3, rev: 'x' })).toEqual({ 'I.exposure': 0.3 })
  })
})

describe('card golden shared with core/colorcraft.py (overview, chains)', () => {
  it('marks, summary text and per-pass chain tags', () => {
    for (const item of cases.configs) {
      const config = configFromState(item.enabled, item.masking, item.state)
      const view = overview(config)
      expect(view.modifierMarks.map(m => [m.label, m.tag]), item.name).toEqual(item.overview.modifierMarks)
      expect(view.maskMarks.map(m => [m.label, m.tag]), item.name).toEqual(item.overview.maskMarks)
      expect(view.text, item.name).toBe(item.overview.text)
      expect(buildChain(config, false), item.name).toEqual(item.base)
      expect(buildChain(config, true), item.name).toEqual(item.hires)
      expect(dumpState(config), item.name).toBe(item.state)                    // 상태 문자열은 고정점
      expect(toInfotext(config), item.name).toBe(item.infotext)
    }
    expect(cases.configs.length).toBeGreaterThanOrEqual(80)
  })

  it('schedule preview = make_schedule (display only)', () => {
    for (const item of cases.schedules) {
      const got = scheduleValues(item.modifier, item.steps)
      if (item.values === null) {
        expect(got, item.name).toBeNull()
        continue
      }
      expect(got, item.name).toHaveLength(item.values.length)
      // 보기 전용 — cos·거듭제곱의 마지막 자리(V8 대 numpy)만 다를 수 있다
      got!.forEach((value, i) => expect(Math.abs(value - item.values![i]!), `${item.name}[${i}]`).toBeLessThan(1e-12))
    }
  })
})

describe('card helpers', () => {
  it("formats floats like python format(x, '.10g')", () => {
    const cases: Array<[number, string]> = [[0, '0'], [-0, '-0'], [0.3, '0.3'], [1, '1'], [-0.2, '-0.2'], [1e-7, '1e-07'],
      [0.0001, '0.0001'], [0.00001, '1e-05'], [123456789012, '1.23456789e+11'], [1234567890, '1234567890'],
      [12345678901, '1.23456789e+10'], [0.1 + 0.2, '0.3'], [0.1234567891234, '0.1234567891'], [1 / 3, '0.3333333333'],
      [2.5e21, '2.5e+21'], [-12.5, '-12.5'], [64, '64'], [9.9999999999, '10']]
    for (const [value, text] of cases) expect(formatG10(value), String(value)).toBe(text)
  })

  it('reads the widget store like parse_settings / config_of', () => {
    const read = readCard({ _cc_enabled: 'true', _cc_masking: '1', _cc_state: '{"v":1,"I.exposure":0.30}' })
    expect([read.enabled, read.masking, read.img2img, read.state]).toEqual([true, true, false, '{"v":1,"I.exposure":0.3}'])
    expect(read.config.modifiers[0]!.exposure).toBe(0.3)
    const broken = readCard({ _cc_enabled: 'true', _cc_state: '{oops' })
    expect(broken.state).toBe(DEFAULT_STATE)
    expect(broken.stateError).toContain('JSON')
    expect(readCard({}).enabled).toBe(false)
  })

  it('setPath coerces without clamping and resetNode restores defaults', () => {
    let config = setPath(defaultConfig(), 'I.exposure', '12.5')
    expect(config.modifiers[0]!.exposure).toBe(12.5)
    config = setPath(config, 'II.kind', 'chroma_plus')
    expect(config.modifiers[1]!.kind).toBe('Chroma Plus')
    // 숫자 글자는 파이썬 float() 문법 — 빈 칸·16진수는 기본값, 밑줄은 숫자 사이에서 읽힌다
    expect(setPath(config, 'I.strength', '').modifiers[0]!.strength).toBe(1)
    expect(setPath(config, 'I.clarity', '0x10').modifiers[0]!.clarity).toBe(0)
    expect(setPath(config, 'I.clarity', '1_0').modifiers[0]!.clarity).toBe(10)
    expect(setPath(config, 'I.clarity', 'inf').modifiers[0]!.clarity).toBe(0)
    config = setPath(config, 'C2.mask_a', 'C3')                               // 뒤 조합은 고를 수 없다
    expect(config.combos[1]!.mask_a).toBe('none')
    expect(setPath(config, 'Z.nope', 1)).toBe(config)
    expect(edited(config, 'I')).toBe(true)
    const reset = resetNode(config, 'I')
    expect(edited(reset, 'I')).toBe(false)
    expect(reset.modifiers[0]!.active).toBe(true)                             // 수정자 I 은 켜진 채 시작한다
    expect(outOfRange(SCHEMA.modifierFields.find(f => f.name === 'exposure')!, 12.5)).toBe(true)
    expect(outOfRange(SCHEMA.modifierFields.find(f => f.name === 'exposure')!, -4)).toBe(false)
  })

  it('card status and badge', () => {
    const off = defaultConfig()
    expect(cardStatus(off, null)).toEqual({ level: 'off', text: OFF_NOTE })
    const nothing = configFromState(true, false, '')
    expect(cardStatus(nothing, null)).toEqual({ level: 'warn', text: NO_ENTRIES_NOTE })
    const on = configFromState(true, false, '{"v":1,"I.exposure":0.3,"II.active":true,"II.pass":"Hires","II.kind":"Punch","II.clarity":0.5}')
    expect(cardStatus(on, null)).toEqual({ level: 'ok', text: 'Forge 로 보냅니다 — 수정자 2개(Base 1·Hires 1)' })
    expect(cardStatus(on, caps(true))).toEqual(cardStatus(on, null))
    expect(cardStatus(on, caps(false)).text).toBe(OLD_FORGE_NOTE)
    expect(cardStatus(on, caps(true, 'not_applicable')).text).toBe(COMFY_PENDING_NOTE)
    expect(cardStatus(on, null, true).text).toBe(KREA2_NOTE)
    expect(badge(off)).toBe('')
    expect(badge(on)).toBe('I Advanced, II Punch (Hires)')
    expect(badge(configFromState(true, false, '{"v":1,"I.active":false}'))).toBe('켠 수정자 없음')
    expect([MARK_ON, MARK_EDITED]).toEqual(['●', '○'])
  })
})
