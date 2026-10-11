import { endsInLineComment, stripPromptComments } from './promptComments'

/** Independent prompt steering, not a geometric camera or conditioning model. */
export interface CompositionState {
  azimuth: number
  elevation: number
  distance: number
  roll: number
  framing: number
}

export const DEFAULT_COMPOSITION: Readonly<CompositionState> = Object.freeze({
  azimuth: 0, elevation: 0, distance: 45, roll: 0, framing: 0,
})

export const COMPOSITION_CONTROLS = [
  { key: 'azimuth', label: '방향', min: -180, max: 180, unit: '°' },
  { key: 'elevation', label: '높이', min: -75, max: 75, unit: '°' },
  { key: 'distance', label: '거리 · 크롭', min: 0, max: 100, unit: '%' },
  { key: 'roll', label: '기울기', min: -30, max: 30, unit: '°' },
  { key: 'framing', label: '화면 내 인물 위치', min: -100, max: 100, unit: '%' },
] as const

export const COMPOSITION_PRESETS: ReadonlyArray<{ name: string; state: CompositionState }> = [
  { name: '정면 상반신', state: { ...DEFAULT_COMPOSITION } },
  { name: '낮은 시점 · 전신', state: { azimuth: -35, elevation: -30, distance: 75, roll: 0, framing: 0 } },
  { name: '위에서 · 근접', state: { azimuth: 30, elevation: 40, distance: 15, roll: 0, framing: 0 } },
  { name: '측면 · 여백', state: { azimuth: 90, elevation: 0, distance: 55, roll: 0, framing: -55 } },
  { name: '후면 · 원경', state: { azimuth: 180, elevation: 15, distance: 100, roll: 0, framing: 40 } },
]

export function normalizeComposition(value: unknown): CompositionState {
  const source = value && typeof value === 'object' ? value as Record<string, unknown> : {}
  const result = { ...DEFAULT_COMPOSITION }
  for (const control of COMPOSITION_CONTROLS) {
    const raw = source[control.key]
    const number = typeof raw === 'number' ? raw : Number.NaN
    result[control.key] = Number.isFinite(number)
      ? Math.round(Math.max(control.min, Math.min(control.max, number)))
      : DEFAULT_COMPOSITION[control.key]
  }
  return result
}

export function compositionTags(value: CompositionState): string[] {
  const state = normalizeComposition(value)
  const angle = Math.abs(state.azimuth)
  const side = state.azimuth < 0 ? 'left' : 'right'
  const tags = angle <= 20 ? ['facing viewer']
    : angle < 65 ? [`three-quarter view from the subject's ${side}`]
    : angle <= 115 ? ['from side', 'profile', `view from the subject's ${side}`]
    : angle < 160 ? ['from behind', `rear three-quarter view from the subject's ${side}`]
    : ['from behind']
  if (state.elevation >= 20) tags.push('from above')
  else if (state.elevation <= -20) tags.push('from below')
  if (state.elevation >= 60) tags.push("bird's eye view")
  tags.push(state.distance <= 15 ? 'close-up' : state.distance <= 30 ? 'portrait'
    : state.distance <= 50 ? 'upper body' : state.distance <= 65 ? 'cowboy shot'
    : state.distance <= 85 ? 'full body' : 'wide shot')
  if (Math.abs(state.roll) >= 6) {
    tags.push('dutch angle', `frame tilted ${state.roll < 0 ? 'counterclockwise' : 'clockwise'}`)
  }
  tags.push(Math.abs(state.framing) < 25 ? 'centered composition'
    : `subject on the ${state.framing < 0 ? 'left' : 'right'} side of the frame`)
  return tags
}

/** Split only top-level tags; preserve weighted groups, schedules and wildcards. */
export function splitCompositionPrompt(text: string): string[] {
  const result: string[] = []
  const closing: string[] = []
  const pairs: Record<string, string> = { '(': ')', '[': ']', '{': '}', '<': '>' }
  let start = 0
  for (let index = 0; index < text.length; index++) {
    const char = text[index]!
    if (char === '\\') { index++; continue }
    if (pairs[char]) closing.push(pairs[char]!)
    else if (char === closing[closing.length - 1]) closing.pop()
    else if (char === ',' && !closing.length) {
      result.push(text.slice(start, index).trim())
      start = index + 1
    }
  }
  result.push(text.slice(start).trim())
  return result.filter(Boolean)
}

function tagKey(text: string): string {
  let value = text.trim().toLowerCase().replace(/_/g, ' ')
  // Only unwrap a complete attention expression, never an escaped literal.
  while (value.startsWith('(') && value.endsWith(')') && splitCompositionPrompt(value.slice(1, -1)).length === 1) {
    value = value.slice(1, -1).replace(/:\s*[+-]?(?:\d+\.?\d*|\.\d+)\s*$/, '').trim()
  }
  return value.replace(/\s+/g, ' ')
}

const CONTRAST_GROUPS = [
  ['from above', 'from below'], ['facing viewer', 'from behind'],
  ['close-up', 'portrait', 'upper body', 'cowboy shot', 'full body', 'wide shot'],
  ['centered composition', 'subject on the left side of the frame', 'subject on the right side of the frame'],
]

export function planCompositionAppend(current: string, tags: readonly string[], otherSections = '') {
  // 프롬프트 주석(// · /* */) 안의 글자는 '이미 있는 태그'가 아니다 — 주석을 뗀 글자로 본다
  const present = `${stripPromptComments(current)}, ${stripPromptComments(otherSections)}`
  const keys = new Set(splitCompositionPrompt(present).map(tagKey))
  const generated = new Set(tags.map(tagKey))
  const additions = tags.filter(tag => {
    const key = tagKey(tag)
    if (!key || keys.has(key)) return false
    keys.add(key)
    return true
  })
  const conflicts: string[] = []
  const original = new Set(splitCompositionPrompt(present).map(tagKey))
  for (const group of CONTRAST_GROUPS) {
    const incoming = group.filter(tag => generated.has(tag))
    const previous = group.filter(tag => original.has(tag) && !generated.has(tag))
    if (incoming.length && previous.length) conflicts.push(`${previous.join(' / ')} ↔ ${incoming.join(' / ')}`)
  }
  // 줄 주석으로 끝나면 줄바꿈으로 잇는다 — 쉼표로 이으면 붙인 태그가 주석에 먹힌다
  const separator = !current.trim() ? '' : endsInLineComment(current) ? '\n'
    : /,\s*$/.test(current) ? (/\s$/.test(current) ? '' : ' ') : ', '
  return {
    additions, conflicts,
    // No reserialization: preserve the user's original prompt, syntax and spacing.
    text: additions.length ? current + separator + additions.join(', ') : current,
  }
}

/** Isometric diagram; coordinates remain inside the viewport at every setting. */
export function compositionCameraPoint(value: CompositionState) {
  const state = normalizeComposition(value)
  const angle = state.azimuth * Math.PI / 180
  const elevation = state.elevation * Math.PI / 180
  const radius = 52 + state.distance * 0.28
  return {
    x: 150 + Math.sin(angle) * Math.cos(elevation) * radius,
    y: 88 + Math.cos(angle) * Math.cos(elevation) * radius * 0.33 - Math.sin(elevation) * 50,
    behind: Math.cos(angle) < 0,
  }
}

export function dragComposition(start: CompositionState, deltaX: number, deltaY: number): CompositionState {
  return normalizeComposition({ ...start, azimuth: start.azimuth + deltaX * 0.7, elevation: start.elevation - deltaY * 0.6 })
}
