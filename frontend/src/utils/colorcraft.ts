/**
 * Colorcraft 카드의 순수 로직 — core/colorcraft.py 의 거울(상태 문자열·정규화·체인 판정·한눈에 보기 표시).
 *
 * - 표(칸 이름·기본값·범위·선택지·묶음·한국어 라벨)는 colorcraftSchema.generated.ts(파이썬 schema_ts() 가 만든다)에서 온다.
 * - 상태 문자열(위젯 _cc_state): 기본값과 다른 경로만, 경로 순서, "v":1 이 앞 — 파이썬 dump_state 와 같은 바이트
 *   (tests/fixtures/colorcraft_state_golden.json 을 양쪽이 왕복). 숫자는 JSON.stringify 글자, -0 은 0.
 * - 값 정규화는 확장 spec.coerce 규칙(숫자는 파이썬 float() 문법의 유한한 수 — 아니면 기본값, 참/거짓 낱말, 선택지는
 *   대소문자 무시, 종류는 노드 id 도). 범위로 자르지 않는다(확장도 자르지 않는다).
 * - 체인(buildChain)·한눈에 보기(overview)·카드 상태 줄(cardStatus)은 파이썬 build_chain·overview·plan 과 같은 규칙 —
 *   utils/colorcraft.cases.json(파이썬이 만든 기대값)을 vitest 와 tests/test_colorcraft_card_mirror.py 가 같이 읽는다.
 * - 스케줄 미리보기(scheduleValues)는 업스트림 muerrilla/ComfyUI-Colorcraft@d28ac6a lib_colorcraft/schedule.py
 *   make_schedule(MIT, 아래 고지)을 TS 로 옮긴 것 — 보기 전용(보내는 값은 바꾸지 않는다).
 *
 * make_schedule 원본: MIT License, Copyright (c) 2026 Sahand Ahmadian Tehrani (Muerrilla). Permission is hereby granted,
 * free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall
 * be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
 * OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
import type { SamExtraCapabilitiesEvent } from '../types/bridge'
import { pyFloat } from './animaSpeed'
import { COLORCRAFT_SCHEMA as SCHEMA, type ColorcraftField } from './colorcraftSchema.generated'

export { COLORCRAFT_SCHEMA as SCHEMA, type ColorcraftField } from './colorcraftSchema.generated'

export type ColorcraftValue = number | boolean | string
export type ColorcraftNode = Record<string, ColorcraftValue>

export interface ColorcraftConfig {
  enabled: boolean
  masking: boolean
  modifiers: ColorcraftNode[]
  leaves: ColorcraftNode[]
  combos: ColorcraftNode[]
}

export type CardKey = 'enabled' | 'masking' | 'img2img' | 'state'
export const WIDGET_IDS: Readonly<Record<CardKey, string>> = {
  enabled: '_cc_enabled', masking: '_cc_masking', img2img: '_cc_img2img', state: '_cc_state',
}
export const DEFAULT_STATE = ''

export const MODIFIER_TAGS = SCHEMA.modifierTags
export const MASK_TAGS = SCHEMA.maskTags
export const COMBO_TAGS = SCHEMA.comboTags
export const NODE_TAGS: readonly string[] = [...MODIFIER_TAGS, ...MASK_TAGS, ...COMBO_TAGS]
export const MASK_NONE = SCHEMA.maskNone
export const MARK_ON = SCHEMA.marks.on
export const MARK_EDITED = SCHEMA.marks.edited
export const INCOMPLETE = SCHEMA.marks.incomplete

const byName = (fields: readonly ColorcraftField[]) => new Map(fields.map(f => [f.name, f] as const))
const MODIFIER_BY_NAME = byName(SCHEMA.modifierFields)
const LEAF_BY_NAME = byName(SCHEMA.leafFields)
const COMBO_BY_NAME = byName(SCHEMA.comboFields)
const COMBO_REF_NAMES = new Set(['mask_a', 'mask_b'])
const ADVANCED_KINDS = new Set(SCHEMA.advancedKinds)
const CHROMA_PLUS_NAMES = ['temp_plus_tint', 'temp_minus_tint', 'lab_a', 'lab_b', 'lab_a_plus_b', 'lab_a_minus_b']

export class StateError extends Error {}

// ── 값 정규화 (확장 spec.coerce — core/colorcraft._finite·_truthy·_choice) ─────────────────────
const TRUE_WORDS = new Set(['true', '1', 'yes', 'on'])
const FALSE_WORDS = new Set(['false', '0', 'no', 'off', ''])

function finite(value: unknown, fallback: number): number {
  let number: number | null
  if (typeof value === 'boolean') number = value ? 1 : 0
  else if (typeof value === 'number') number = value
  else if (typeof value === 'string') number = pyFloat(value)
  else number = null
  return number !== null && Number.isFinite(number) ? number : fallback
}

function truthy(value: unknown, fallback: boolean): boolean {
  if (typeof value === 'boolean') return value
  if (value === null || value === undefined) return fallback
  if (typeof value === 'number') return value !== 0
  if (typeof value !== 'string') return fallback
  const word = value.trim().toLowerCase()
  if (TRUE_WORDS.has(word)) return true
  if (FALSE_WORDS.has(word)) return false
  return fallback
}

function choice(field: ColorcraftField, value: unknown, choices?: readonly string[]): string {
  const options = choices ?? field.choices
  if (typeof value === 'string' && options.includes(value)) return value
  const text = value === null || value === undefined ? '' : String(value).trim()
  const folded = new Map(options.map(option => [option.toLowerCase(), option] as const))
  const hit = folded.get(text.toLowerCase())
  if (hit !== undefined) return hit
  if (field.name === 'kind') {
    const nodeId = text.toLowerCase().split(' ').join('_')
    const label = SCHEMA.kindLabels[nodeId]
    if (label !== undefined) return label
  }
  return field.default as string
}

/** 칸 ``field`` 의 형으로. 쓸 수 없으면 칸 기본값(확장 spec.coerce). */
export function coerce(field: ColorcraftField, value: unknown, choices?: readonly string[]): ColorcraftValue {
  if (field.kind === 'float') return finite(value, field.default as number)
  if (field.kind === 'bool') return truthy(value, field.default as boolean)
  return choice(field, value, choices)
}

export function comboRefChoices(index: number): readonly string[] {
  return SCHEMA.comboRefChoices[index] ?? [MASK_NONE]
}

function coerceCombo(index: number, field: ColorcraftField, value: unknown): ColorcraftValue {
  return coerce(field, value, COMBO_REF_NAMES.has(field.name) ? comboRefChoices(index) : undefined)
}

export function modifierDefault(index: number, name: string): ColorcraftValue {
  if (name === 'active') return SCHEMA.activeDefaults[index] ?? false
  return MODIFIER_BY_NAME.get(name)!.default
}

// ── 설정 묶음 ─────────────────────────────────────────────────────────────────
export function defaultConfig(): ColorcraftConfig {
  return {
    enabled: false,
    masking: false,
    modifiers: MODIFIER_TAGS.map((_tag, i) =>
      Object.fromEntries(SCHEMA.modifierFields.map(f => [f.name, modifierDefault(i, f.name)]))),
    leaves: MASK_TAGS.map(() => Object.fromEntries(SCHEMA.leafFields.map(f => [f.name, f.default]))),
    combos: COMBO_TAGS.map(() => Object.fromEntries(SCHEMA.comboFields.map(f => [f.name, f.default]))),
  }
}

export function copyConfig(config: ColorcraftConfig): ColorcraftConfig {
  return {
    enabled: config.enabled, masking: config.masking,
    modifiers: config.modifiers.map(m => ({ ...m })), leaves: config.leaves.map(l => ({ ...l })),
    combos: config.combos.map(c => ({ ...c })),
  }
}

export type NodeKind = 'modifier' | 'leaf' | 'combo'

export function nodeKind(tag: string): NodeKind {
  if (MODIFIER_TAGS.includes(tag)) return 'modifier'
  if (MASK_TAGS.includes(tag)) return 'leaf'
  if (COMBO_TAGS.includes(tag)) return 'combo'
  throw new Error(`Colorcraft 노드가 아닙니다: ${tag}`)
}

export function nodeFields(tag: string): readonly ColorcraftField[] {
  const kind = nodeKind(tag)
  return kind === 'modifier' ? SCHEMA.modifierFields : kind === 'leaf' ? SCHEMA.leafFields : SCHEMA.comboFields
}

export function nodeValues(config: ColorcraftConfig, tag: string): ColorcraftNode {
  const kind = nodeKind(tag)
  if (kind === 'modifier') return config.modifiers[MODIFIER_TAGS.indexOf(tag)]!
  if (kind === 'leaf') return config.leaves[MASK_TAGS.indexOf(tag)]!
  return config.combos[COMBO_TAGS.indexOf(tag)]!
}

export function nodeDefault(tag: string, field: ColorcraftField): ColorcraftValue {
  if (MODIFIER_TAGS.includes(tag)) return modifierDefault(MODIFIER_TAGS.indexOf(tag), field.name)
  return field.default
}

/** 칸 ``field`` 의 실제 선택지 — 조합의 Mask A/B 는 조합마다 다르다(앞선 조합만). */
export function nodeChoices(tag: string, field: ColorcraftField): readonly string[] {
  if (COMBO_TAGS.includes(tag) && COMBO_REF_NAMES.has(field.name)) return comboRefChoices(COMBO_TAGS.indexOf(tag))
  return field.choices
}

export const STATE_PATHS: readonly string[] = NODE_TAGS.flatMap(tag => nodeFields(tag).map(f => `${tag}.${f.name}`))
const STATE_PATH_SET = new Set(STATE_PATHS)

function splitPath(path: string): [string, string] {
  const at = path.indexOf('.')
  return at < 0 ? [path, ''] : [path.slice(0, at), path.slice(at + 1)]
}

/** ``{"enabled": …, "I.exposure": 0.3, …}`` → 설정(패널 기본값 위 — 수정자 I 켜짐, 없는 경로 기본값, 모르는 경로 버림).
 *  ``enabled`` 가 없으면 켬(확장 config_from_mapping). null 값은 건너뛴다. */
export function configFromMapping(mapping: Record<string, unknown>): ColorcraftConfig {
  const config = defaultConfig()
  config.enabled = truthy(mapping.enabled ?? true, true)
  config.masking = truthy(mapping.masking ?? false, false)
  for (const [key, value] of Object.entries(mapping)) {
    if (value === null || value === undefined) continue
    const [tag, name] = splitPath(key)
    if (MODIFIER_TAGS.includes(tag) && MODIFIER_BY_NAME.has(name)) {
      config.modifiers[MODIFIER_TAGS.indexOf(tag)]![name] = coerce(MODIFIER_BY_NAME.get(name)!, value)
    } else if (MASK_TAGS.includes(tag) && LEAF_BY_NAME.has(name)) {
      config.leaves[MASK_TAGS.indexOf(tag)]![name] = coerce(LEAF_BY_NAME.get(name)!, value)
    } else if (COMBO_TAGS.includes(tag) && COMBO_BY_NAME.has(name)) {
      const index = COMBO_TAGS.indexOf(tag)
      config.combos[index]![name] = coerceCombo(index, COMBO_BY_NAME.get(name)!, value)
    }
  }
  return config
}

// ── 상태 문자열 (위젯 _cc_state — core/colorcraft.dump_state·parse_state) ───────────────────
function sameValue(field: ColorcraftField, value: ColorcraftValue, fallback: ColorcraftValue): boolean {
  if (field.kind === 'float') return Number(value) === Number(fallback)        // -0 === 0
  return value === fallback
}

/** 기본값과 다른 상태 경로 → 값(경로 순서). 값은 coerce 를 거친다(-0 은 0). */
export function statePaths(config: ColorcraftConfig): Record<string, ColorcraftValue> {
  const out: Record<string, ColorcraftValue> = {}
  for (const tag of NODE_TAGS) {
    const current = nodeValues(config, tag)
    for (const field of nodeFields(tag)) {
      const value = coerce(field, current[field.name], nodeChoices(tag, field))
      if (!sameValue(field, value, nodeDefault(tag, field))) {
        out[`${tag}.${field.name}`] = field.kind === 'float' && value === 0 ? 0 : value
      }
    }
  }
  return out
}

/** 상태 문자열 — 전부 기본값이면 "", 아니면 {"v":1,"I.exposure":0.3,…}. 켬·마스크 켬은 위젯이 따로라 넣지 않는다. */
export function dumpState(config: ColorcraftConfig): string {
  const paths = statePaths(config)
  const keys = Object.keys(paths)
  if (!keys.length) return DEFAULT_STATE
  const body = [`"v":${SCHEMA.version}`, ...keys.map(path => `${JSON.stringify(path)}:${JSON.stringify(paths[path])}`)]
  return `{${body.join(',')}}`
}

/** 상태 문자열(또는 객체) → {경로: 원래 값}(모르는 경로·v·rev 제외). ""/null = {}. 읽을 수 없으면 StateError. */
export function parseState(raw: unknown): Record<string, unknown> {
  if (raw === null || raw === undefined || (typeof raw === 'string' && raw.trim() === '')) return {}
  let data: Record<string, unknown>
  if (typeof raw === 'object' && !Array.isArray(raw)) {
    data = raw as Record<string, unknown>
  } else if (typeof raw === 'string' && raw.trimStart().startsWith('{')) {
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (error) {
      throw new StateError(`Colorcraft 상태가 JSON 이 아닙니다 (${String(error)})`)
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      throw new StateError('Colorcraft 상태가 JSON 객체가 아닙니다')
    }
    data = parsed as Record<string, unknown>
  } else {
    throw new StateError(`Colorcraft 상태가 아닙니다 (${typeof raw})`)
  }
  const version = 'v' in data ? data.v : SCHEMA.version
  if (typeof version === 'boolean' || version !== SCHEMA.version) {
    throw new StateError(`Colorcraft 상태 버전 ${JSON.stringify(version)} 은(는) ${SCHEMA.version} 이 아닙니다`)
  }
  return Object.fromEntries(Object.entries(data).filter(([key]) => STATE_PATH_SET.has(key)))
}

/** 위젯 세 칸 → 설정(경로는 패널 기본값 위에 얹는다). 읽을 수 없는 상태는 StateError. */
export function configFromState(enabled: unknown, masking: unknown, state: unknown): ColorcraftConfig {
  const config = configFromMapping(parseState(state))
  config.enabled = truthy(enabled, false)
  config.masking = truthy(masking, false)
  return config
}

/** 다시 쓴 정규 문자열(경로 순서·숫자 글자·기본값 생략). 읽을 수 없으면 StateError. */
export function canonicalState(raw: unknown): string {
  return dumpState(configFromMapping(parseState(raw)))
}

/** 경로 하나를 바꾼 새 설정(값은 coerce — 범위로 자르지 않는다). 모르는 경로는 그대로 돌려준다. */
export function setPath(config: ColorcraftConfig, path: string, value: unknown): ColorcraftConfig {
  const [tag, name] = splitPath(path)
  if (!NODE_TAGS.includes(tag)) return config
  const field = nodeFields(tag).find(f => f.name === name)
  if (!field) return config
  const next = copyConfig(config)
  nodeValues(next, tag)[name] = coerce(field, value, nodeChoices(tag, field))
  return next
}

/** 노드 하나를 기본값으로(수정자 I 의 켬은 켬 — 확장 기본값). */
export function resetNode(config: ColorcraftConfig, tag: string): ColorcraftConfig {
  const next = copyConfig(config)
  const values = nodeValues(next, tag)
  for (const field of nodeFields(tag)) values[field.name] = nodeDefault(tag, field)
  return next
}

// ── 카드 값 읽기 (위젯 스토어 → 설정) ─────────────────────────────────────────────
export interface CardSettings {
  enabled: boolean
  masking: boolean
  img2img: boolean
  state: string
}

export interface CardRead extends CardSettings {
  config: ColorcraftConfig
  /** 상태 문자열을 읽지 못했으면 그 이유(파이썬처럼 기본값 위의 설정으로 본다). */
  stateError: string | null
}

/** 위젯 스토어 → 카드 값(core/colorcraft.parse_settings·config_of 와 같은 규칙). */
export function readCard(widgets: Record<string, unknown>): CardRead {
  const enabled = truthy(widgets[WIDGET_IDS.enabled], false)
  const masking = truthy(widgets[WIDGET_IDS.masking], false)
  const img2img = truthy(widgets[WIDGET_IDS.img2img], false)
  const raw = widgets[WIDGET_IDS.state]
  let state = DEFAULT_STATE
  let stateError: string | null = null
  try {
    state = raw === null || raw === undefined ? DEFAULT_STATE : canonicalState(raw)
  } catch (error) {
    stateError = error instanceof Error ? error.message : String(error)
  }
  const config = configFromState(enabled, masking, state)
  return { enabled, masking, img2img, state, config, stateError }
}

// ── 패스마다 도는 체인 (core/colorcraft.build_chain) ────────────────────────────
export function kindOf(modifier: ColorcraftNode): string {
  return SCHEMA.kindIds[String(modifier.kind)] ?? 'advanced'
}

export function takesMask(kind: string): boolean {
  return SCHEMA.kindTakesMask[kind] ?? true
}

export function runsOnPass(modifier: ColorcraftNode, isHires: boolean): boolean {
  const which = modifier.pass ?? SCHEMA.passBase
  if (which === SCHEMA.passBoth) return true
  return (which === SCHEMA.passHires) === isHires
}

export type Schedule = Record<'strength' | 'start' | 'end' | 'bias' | 'exponent' | 'start_off' | 'end_off', number>
  & { smooth: boolean }

/** 노드의 스케줄. 고급 스케줄 끔이면 치우침·지수·오프셋을 되돌린다(업스트림 Forge 스크립트 규칙). */
export function scheduleOf(modifier: ColorcraftNode): Schedule {
  const schedule: Schedule = {
    strength: Number(modifier.strength), start: Number(modifier.start), end: Number(modifier.end),
    bias: Number(modifier.bias), exponent: Number(modifier.exponent), start_off: Number(modifier.start_off),
    end_off: Number(modifier.end_off), smooth: Boolean(modifier.smooth),
  }
  if (!modifier.advanced) Object.assign(schedule, { bias: 0.5, exponent: 0, start_off: 0, end_off: 0 })
  return schedule
}

function scheduleIsZero(schedule: Schedule): boolean {
  return schedule.strength === 0 && schedule.start_off === 0 && schedule.end_off === 0
}

function paramsAreNoop(kind: string, modifier: ColorcraftNode): boolean {
  let names: string[]
  if (kind === 'basic') names = ['contrast', 'color_shift_amount']
  else if (kind === 'advanced') {
    names = ['exposure', 'tone_compression', 'contrast', 'clarity', 'sharpness', 'temperature', 'tint', 'vibrance',
      'saturation', 'chroma_contrast']
    if (modifier.more_colors) names.push(...CHROMA_PLUS_NAMES)
    if (modifier.color_shift) names.push('color_shift_amount')
  } else if (kind === 'luma') names = ['exposure', 'tone_compression']
  else if (kind === 'chroma') names = ['temperature', 'tint', 'vibrance', 'saturation', 'chroma_contrast']
  else if (kind === 'chroma_plus') names = [...CHROMA_PLUS_NAMES]
  else if (kind === 'punch') names = ['contrast', 'clarity', 'sharpness']
  else if (kind === 'shift') names = ['color_shift_amount']
  else return true
  return names.every(name => modifier[name] === 0)
}

/** 조합이 풀리는가(잎과 앞선 조합만 — A·B 가 모두 풀려야 한다). */
export function resolvedCombos(config: ColorcraftConfig): Set<string> {
  const resolved = new Set<string>()
  const ok = (ref: ColorcraftValue | undefined) =>
    typeof ref === 'string' && ref !== MASK_NONE && (MASK_TAGS.includes(ref) || resolved.has(ref))
  config.combos.forEach((combo, i) => {
    if (ok(combo.mask_a) && ok(combo.mask_b)) resolved.add(COMBO_TAGS[i]!)
  })
  return resolved
}

function maskResolves(config: ColorcraftConfig, modifier: ColorcraftNode, resolved: Set<string>): boolean {
  if (!config.masking || !takesMask(kindOf(modifier))) return false
  const selection = String(modifier.mask ?? MASK_NONE)
  if (selection === MASK_NONE) return false
  return MASK_TAGS.includes(selection) || resolved.has(selection)
}

export interface PassChain {
  tags: string[]             // 이 패스에서 도는 수정자
  unresolved: string[]       // "II->C3" — 미완성 조합을 고른 수정자(마스크 없이 돈다)
  needsBasis: boolean        // 기저 벡터가 필요한 항목이 있다(Advanced 계열, 또는 마스크를 쓰는 Shift)
}

/** 한 패스의 체인(태그만). 꺼진 탭, 다른 패스의 탭, 아무것도 바꾸지 못하는 항목(스케줄 0·값 0)은 뺀다. */
export function buildChain(config: ColorcraftConfig, isHires: boolean): PassChain {
  const out: PassChain = { tags: [], unresolved: [], needsBasis: false }
  if (!config.enabled) return out
  const resolved = resolvedCombos(config)
  config.modifiers.forEach((modifier, i) => {
    if (!modifier.active || !runsOnPass(modifier, isHires)) return
    const kind = kindOf(modifier)
    if (scheduleIsZero(scheduleOf(modifier)) || paramsAreNoop(kind, modifier)) return
    const masked = maskResolves(config, modifier, resolved)
    if (!masked && config.masking && takesMask(kind) && COMBO_TAGS.includes(String(modifier.mask))) {
      out.unresolved.push(`${MODIFIER_TAGS[i]}->${modifier.mask}`)
    }
    if (ADVANCED_KINDS.has(kind) || (kind === 'shift' && masked)) out.needsBasis = true
    out.tags.push(MODIFIER_TAGS[i]!)
  })
  return out
}

// ── 한눈에 보기 (확장 편집기의 ●/○ 표시와 요약 줄 — core/colorcraft.overview) ─────────────────
/** 켬(active)을 빼고 기본값과 다른 칸이 있나. */
export function edited(config: ColorcraftConfig, tag: string): boolean {
  const values = nodeValues(config, tag)
  return nodeFields(tag).some(f => f.name !== 'active' && values[f.name] !== nodeDefault(tag, f))
}

function maskedSelections(config: ColorcraftConfig): string[] {
  return config.modifiers
    .filter(m => m.active && takesMask(kindOf(m)) && m.mask !== MASK_NONE)
    .map(m => String(m.mask))
}

/** [쓰이는 태그, 미완성 조합 태그] — 켜진 수정자가 실제로 가리는 잎·조합(미완성 조합과 그 너머는 빼고). */
export function usedMasks(config: ColorcraftConfig): [Set<string>, Set<string>] {
  const resolved = resolvedCombos(config)
  const unresolved = new Set(COMBO_TAGS.filter(tag => !resolved.has(tag)))
  const used = new Set<string>()
  if (!config.masking) return [used, unresolved]
  const todo = maskedSelections(config)
  while (todo.length) {
    const tag = todo.pop()!
    if (tag === MASK_NONE || used.has(tag) || unresolved.has(tag)) continue
    used.add(tag)
    if (COMBO_TAGS.includes(tag)) {
      const combo = config.combos[COMBO_TAGS.indexOf(tag)]!
      todo.push(String(combo.mask_a), String(combo.mask_b))
    }
  }
  return [used, unresolved]
}

export interface PickerMark { label: string; tag: string; mark: '' | typeof MARK_ON | typeof MARK_EDITED }

export interface Overview {
  modifierMarks: PickerMark[]
  maskMarks: PickerMark[]
  text: string
}

/** 편집기 고르개의 표시와 요약 줄(확장 panel_state.overview 의 글자 — HTML 은 빼고). */
export function overview(config: ColorcraftConfig): Overview {
  const [used, unresolved] = usedMasks(config)
  const active: string[] = []
  const modifierMarks: PickerMark[] = MODIFIER_TAGS.map((tag, i) => {
    const m = config.modifiers[i]!
    let mark: PickerMark['mark'] = ''
    if (m.active) {
      mark = MARK_ON
      let text = `${tag} ${m.kind}`
      if (config.masking && takesMask(kindOf(m)) && m.mask !== MASK_NONE) {
        text += ` ➜ ${m.mask}` + (unresolved.has(String(m.mask)) ? INCOMPLETE : '')
      }
      if (m.pass !== SCHEMA.passBase) text += ` (${m.pass})`
      active.push(text)
    } else if (edited(config, tag)) {
      mark = MARK_EDITED
    }
    return { label: mark ? `${tag} ${mark}` : tag, tag, mark }
  })
  const maskMarks: PickerMark[] = [...MASK_TAGS, ...COMBO_TAGS].map(tag => {
    const mark: PickerMark['mark'] = used.has(tag) ? MARK_ON : edited(config, tag) ? MARK_EDITED : ''
    return { label: mark ? `${tag} ${mark}` : tag, tag, mark }
  })
  let text = active.length ? `켠 수정자: ${active.join(', ')}` : '켠 수정자 없음'
  if (config.masking) {
    const names = NODE_TAGS.filter(tag => used.has(tag))
    text += ' · 쓰는 마스크: ' + (names.length ? names.join(', ') : '없음')
  }
  if (!config.enabled) text = `꺼짐 — ${text}`
  return { modifierMarks, maskMarks, text }
}

// ── 편집기 묶음 ─────────────────────────────────────────────────────────────────
export interface FieldGroup {
  group: string
  label: string
  gate: ColorcraftField | null       // Advanced 의 스위치(Chroma Plus·색 이동) — 끄면 묶음 값이 쓰이지 않는다
  fields: ColorcraftField[]
}

const GATE_NAMES = new Set(Object.values(SCHEMA.kindGates).flat())

/** 종류 ``kind``(노드 id)가 보이는 값 묶음(확장 KIND_GROUPS, 편집기 순서) — 머리·스케줄 묶음은 빼고. */
export function visibleGroups(kind: string): FieldGroup[] {
  const groups = SCHEMA.kindGroups[kind] ?? []
  const gates = new Set(SCHEMA.kindGates[kind] ?? [])
  return SCHEMA.groupOrder.filter(group => groups.includes(group)).map(group => {
    const inGroup = SCHEMA.modifierFields.filter(f => f.group === group)
    const gate = inGroup.find(f => gates.has(f.name)) ?? null
    return {
      group, label: SCHEMA.groupLabels[group] ?? group, gate,
      fields: inGroup.filter(f => !GATE_NAMES.has(f.name)),
    }
  })
}

export function fieldsOfGroup(group: string): ColorcraftField[] {
  return SCHEMA.modifierFields.filter(f => f.group === group)
}

/** 선택지의 표시 라벨(값은 그대로 보낸다). */
export function choiceLabel(fieldName: string, value: string): string {
  return SCHEMA.choiceLabels[fieldName]?.[value] ?? value
}

/** 확장 슬라이더 범위 밖인가(자르지 않고 보낸다 — 카드가 부드럽게 알린다). */
export function outOfRange(field: ColorcraftField, value: unknown): boolean {
  if (field.kind !== 'float' || typeof value !== 'number') return false
  return (field.min !== null && value < field.min) || (field.max !== null && value > field.max)
}

export const OUT_OF_RANGE_NOTE = '확장 슬라이더 범위 밖 — 그대로 보냅니다'

// ── infotext 문자열 (확장 자기 열쇠 'SAM Extra Colorcraft' — core/colorcraft.to_infotext) ─────────────
export const INFOTEXT_VERSION = 'v1'

/** 켜진 수정자의 선택이 (조합을 거쳐) 닿는 [잎, 조합] — 미완성 조합도 따라간다(infotext 가 적는 마스크). */
export function activeMasks(config: ColorcraftConfig): [string[], string[]] {
  if (!config.masking) return [[], []]
  const reached = new Set<string>()
  const todo = maskedSelections(config)
  while (todo.length) {
    const tag = todo.pop()!
    if (tag === MASK_NONE || reached.has(tag)) continue
    reached.add(tag)
    if (COMBO_TAGS.includes(tag)) {
      const combo = config.combos[COMBO_TAGS.indexOf(tag)]!
      todo.push(String(combo.mask_a), String(combo.mask_b))
    }
  }
  return [MASK_TAGS.filter(t => reached.has(t)), COMBO_TAGS.filter(t => reached.has(t))]
}

/** 수정자의 종류가 읽는 칸 — infotext 에 쓰는 칸(머리·스케줄 전부·그 종류 노드의 값). */
export function writtenModifierFields(modifier: ColorcraftNode): string[] {
  const kind = kindOf(modifier)
  const names = ['kind', 'pass', ...(takesMask(kind) ? ['mask'] : []), 'strength', 'start', 'end', 'advanced',
    ...SCHEMA.shapingNames]
  for (const [, panelName] of SCHEMA.kindParams[kind] ?? []) if (!names.includes(panelName)) names.push(panelName)
  return names
}

/** 파이썬 ``format(x, '.10g')`` — 유효숫자 10자리, 지수가 -4 미만이거나 10 이상이면 ``1e-07`` 꼴. */
export function formatG10(value: number): string {
  if (value === 0) return Object.is(value, -0) ? '-0' : '0'
  const [mantissa, exponentText] = value.toExponential(9).split('e') as [string, string]
  const exponent = Number(exponentText)
  const strip = (text: string) => text.includes('.') ? text.replace(/0+$/, '').replace(/\.$/, '') : text
  if (exponent < -4 || exponent >= 10) {
    return `${strip(mantissa)}e${exponent < 0 ? '-' : '+'}${String(Math.abs(exponent)).padStart(2, '0')}`
  }
  return strip(value.toFixed(9 - exponent))
}

function formatValue(field: ColorcraftField, value: ColorcraftValue): string {
  if (field.kind === 'bool') return value ? '1' : '0'
  if (field.kind === 'float') return formatG10(Number(value))
  return String(value)
}

/** ``v1;mods=I,III;masking=1;masks=M1;combos=;I.exposure=0.3;…`` — 기본값과 다른 값만(확장 to_infotext). 켬은 쓰지 않는다
 *  (이 문자열을 받으면 확장은 켠 것으로 읽는다). Forge 로 보내는 첫 인자이자 이미지에 남는 값이다. */
export function toInfotext(config: ColorcraftConfig): string {
  const active = config.modifiers.flatMap((m, i) => (m.active ? [i] : []))
  const parts = [INFOTEXT_VERSION, `mods=${active.map(i => MODIFIER_TAGS[i]).join(',')}`]
  const [leaves, combos] = activeMasks(config)
  if (config.masking) parts.push('masking=1', `masks=${leaves.join(',')}`, `combos=${combos.join(',')}`)
  for (const i of active) {
    const modifier = config.modifiers[i]!
    for (const name of writtenModifierFields(modifier)) {
      const field = MODIFIER_BY_NAME.get(name)!
      if (modifier[name] !== field.default) parts.push(`${MODIFIER_TAGS[i]}.${name}=${formatValue(field, modifier[name]!)}`)
    }
  }
  for (const [tags, fields, values, all] of [[leaves, SCHEMA.leafFields, config.leaves, MASK_TAGS],
    [combos, SCHEMA.comboFields, config.combos, COMBO_TAGS]] as const) {
    for (const tag of tags) {
      const node = values[all.indexOf(tag)]!
      for (const field of fields) {
        if (node[field.name] !== field.default) parts.push(`${tag}.${field.name}=${formatValue(field, node[field.name]!)}`)
      }
    }
  }
  return parts.join(';')
}

// ── 카드 상태 줄 (core/colorcraft.plan 의 표시판) ─────────────────────────────────
export type CardLevel = 'off' | 'ok' | 'warn'
export interface CardStatus { level: CardLevel; text: string }

export const OFF_NOTE = "꺼져 있습니다 — 켜면 Forge 확장 'Colorcraft (sam-extra)' 로 보냅니다"
export const OLD_FORGE_NOTE = '연결된 Forge 의 sam-extra 에 Colorcraft 가 없어(0.31.0 이상 필요) 보내지 않습니다.'
export const COMFY_OFF_NOTE = '꺼져 있습니다 — 켜면 ComfyUI 앱 노드 팩 1.7.0 의 같은 계산(ForgeNeoColorcraft)으로 돌립니다'
/** ComfyUI 로 보낼 때 상태 줄 머리(계획 C6 — 컴파일러가 항목이 있는 메인 패스마다 노드를 넣는다). */
export const COMFY_TARGET = 'ComfyUI(앱 노드 팩 1.7.0)'
export const KREA2_NOTE = 'Krea2 생성에는 보내지 않습니다.'
export const NO_ENTRIES_NOTE = '켰지만 바뀌는 값이 없습니다(켠 수정자가 없거나 모두 0) — 보내지 않습니다.'

/** 카드 상태 줄 — 꺼짐 / Krea2 / 옛 Forge / 바뀌는 값 없음 / 보냄(수정자 수 — Forge 또는 ComfyUI 앱 노드 팩 1.7.0). */
export function cardStatus(config: ColorcraftConfig,
  caps: Pick<SamExtraCapabilitiesEvent, 'status' | 'known' | 'features'> | null | undefined,
  krea2 = false): CardStatus {
  const comfy = caps?.status === 'not_applicable'
  if (!config.enabled) return { level: 'off', text: comfy ? COMFY_OFF_NOTE : OFF_NOTE }
  if (krea2) return { level: 'warn', text: KREA2_NOTE }
  if (!comfy && caps?.known && caps.features?.colorcraft !== true) return { level: 'warn', text: OLD_FORGE_NOTE }
  const base = buildChain(config, false).tags
  const hires = buildChain(config, true).tags
  if (!base.length && !hires.length) return { level: 'warn', text: NO_ENTRIES_NOTE }
  const count = new Set([...base, ...hires]).size
  return { level: 'ok',
    text: `${comfy ? COMFY_TARGET : 'Forge'} 로 보냅니다 — 수정자 ${count}개(Base ${base.length}·Hires ${hires.length})` }
}

/** 카드 머리 배지 — 켠 수정자 요약(꺼졌으면 ''). */
export function badge(config: ColorcraftConfig): string {
  if (!config.enabled) return ''
  const active = config.modifiers.flatMap((m, i) => m.active
    ? [`${MODIFIER_TAGS[i]} ${m.kind}${m.pass !== SCHEMA.passBase ? ` (${m.pass})` : ''}`] : [])
  if (!active.length) return '켠 수정자 없음'
  return active.length > 3 ? `${active.slice(0, 3).join(', ')} 외 ${active.length - 3}` : active.join(', ')
}

// ── 스케줄 미리보기 (업스트림 make_schedule 의 TS 판 — 보기 전용) ───────────────────────
function roundHalfEven(value: number): number {
  const floor = Math.floor(value)
  const diff = value - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

function linspace(start: number, stop: number, num: number): number[] {
  if (num <= 0) return []
  if (num === 1) return [start]
  const step = (stop - start) / (num - 1)
  const out = Array.from({ length: num }, (_v, i) => i * step + start)
  out[num - 1] = stop
  return out
}

function shape(values: number[], smooth: boolean, exponent: number, amount: number, offset: number): number[] {
  let out = smooth ? values.map(v => 0.5 * (1 - Math.cos(v * Math.PI))) : values
  out = exponent >= 0 ? out.map(v => v ** exponent) : out.map(v => 1 - (1 - v) ** Math.abs(1 / exponent))
  if (out.some(v => v !== 0)) out = out.map(v => v * (amount - offset) + offset)
  return out
}

/** 수정자 하나의 스텝별 값(업스트림 make_schedule — 확장이 패스의 스텝 수로 만드는 배열). 시작·끝·치우침이 0~1 밖이거나
 *  스텝이 없으면 null(확장은 그 값도 쓰지만 미리보기는 그리지 않는다). */
export function scheduleValues(modifier: ColorcraftNode, steps: number): number[] | null {
  const s = scheduleOf(modifier)
  if (!Number.isInteger(steps) || steps < 1) return null
  if ([s.start, s.end, s.bias].some(v => !(v >= 0 && v <= 1))) return null
  const start = Math.min(s.start, s.end)
  const mid = start + s.bias * (s.end - start)
  const [startIdx, midIdx, endIdx] = [start, mid, s.end].map(x => roundHalfEven(x * (steps - 1))) as [number, number,
    number]
  const startValues = shape(linspace(0, 1, midIdx - startIdx + 1), s.smooth, s.exponent, s.strength, s.start_off)
  const endValues = shape(linspace(1, 0, endIdx - midIdx + 1), s.smooth, s.exponent, s.strength, s.end_off)
  const out = new Array<number>(steps).fill(0)
  startValues.forEach((v, i) => { out[startIdx + i] = v })
  endValues.forEach((v, i) => { out[midIdx + i] = v })
  for (let i = 0; i < startIdx; i++) out[i] = s.start_off
  for (let i = endIdx + 1; i < steps; i++) out[i] = s.end_off
  return out
}
