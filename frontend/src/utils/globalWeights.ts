/**
 * 글로벌 태그 가중치 — 생성 직전 메인 태그에 등록 태그의 가중치 `(tag:w)` 를 붙인다 (#109).
 *
 * 예전 구현은 태그를 정규식으로 이스케이프하려다 문자 클래스의 역슬래시가 두 배로 들어가
 * 실제로는 아무것도 이스케이프하지 못했다: `:(`·`+_+` 를 등록하면 new RegExp 가 SyntaxError 를
 * 던져 **생성 자체가 무반응**이 됐고, `c.c.` 는 무관한 `cxcx` 를 바꿨으며, `\b` 때문에 괄호·
 * 이모티콘 태그는 매칭조차 안 됐다. 여기서는 정규식 없이 **쉼표·줄바꿈 단위 토큰**을 비교한다.
 *
 * - 괄호 깊이 0 의 토큰만 본다 — `(a, b:1.1)` 같은 묶음 강조 안은 건드리지 않는다.
 *   (괄호가 짝이 맞지 않는 프롬프트면 깊이를 무시하고 평평하게 나눈다.)
 * - 비교는 대소문자·밑줄/공백·`\(` 이스케이프를 무시한다('Long_Hair' = 'long hair').
 * - 이미 `(tag:1.2)` 면 숫자만 바꾸고, 맨 태그면 원문 철자 그대로 감싼다.
 * - `<lora:…>` · `BREAK` · `AND` 와 가중치 1.00 은 손대지 않는다.
 */

export interface GlobalWeight {
  tag: string
  /** 백분율(100 = 1.00) */
  weight: number
  [k: string]: any
}

const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)$/

/** 100 → '1.00' */
export function formatWeight(weight: number): string {
  const n = Number(weight)
  return (Number.isFinite(n) ? n / 100 : 1).toFixed(2)
}

function unescapeParens(text: string): string {
  return text.replace(/\\([()[\]])/g, '$1')
}

/** 비교 키 — 소문자, 밑줄→공백, 공백 정리, 괄호 이스케이프 제거 */
export function normalizeTagKey(text: string): string {
  return unescapeParens(String(text ?? ''))
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

interface Segment { text: string; sep: string }

/** 깊이 0 의 쉼표·줄바꿈으로 나눈다. 괄호가 짝이 안 맞으면 null. */
function splitTopLevel(text: string): Segment[] | null {
  const out: Segment[] = []
  let depth = 0
  let angle = 0
  let buf = ''
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '\\' && i + 1 < text.length) { buf += ch + text[i + 1]; i++; continue }
    if (ch === '(' || ch === '[' || ch === '{') depth++
    else if (ch === ')' || ch === ']' || ch === '}') { depth--; if (depth < 0) return null }
    else if (ch === '<') angle++
    else if (ch === '>' && angle > 0) angle--
    if ((ch === ',' || ch === '\n') && depth === 0 && angle === 0) {
      out.push({ text: buf, sep: ch })
      buf = ''
      continue
    }
    buf += ch
  }
  if (depth !== 0 || angle !== 0) return null
  out.push({ text: buf, sep: '' })
  return out
}

function splitFlat(text: string): Segment[] {
  const out: Segment[] = []
  let buf = ''
  for (const ch of text) {
    if (ch === ',' || ch === '\n') { out.push({ text: buf, sep: ch }); buf = ''; continue }
    buf += ch
  }
  out.push({ text: buf, sep: '' })
  return out
}

function parensBalanced(text: string): boolean {
  let depth = 0
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '\\') { i++; continue }
    if (ch === '(') depth++
    else if (ch === ')') { depth--; if (depth < 0) return false }
  }
  return depth === 0
}

/** 감쌀 본문 — 괄호 짝이 맞으면 원문 그대로, 아니면(':(' 등) 이스케이프해 문법을 지킨다. */
function wrapBody(core: string): string {
  if (parensBalanced(core)) return core
  let out = ''
  for (let i = 0; i < core.length; i++) {
    const ch = core[i]
    if (ch === '\\' && i + 1 < core.length) { out += ch + core[i + 1]; i++; continue }
    out += ch === '(' || ch === ')' ? '\\' + ch : ch
  }
  return out
}

/** `(본문:숫자)` 형태면 본문을 돌려준다 */
function weightedInner(core: string): string | null {
  if (!core.startsWith('(') || !core.endsWith(')')) return null
  const body = core.slice(1, -1)
  const colon = body.lastIndexOf(':')
  if (colon <= 0) return null
  const num = body.slice(colon + 1).trim()
  if (!NUMBER.test(num)) return null
  const inner = body.slice(0, colon).trim()
  if (!inner || !parensBalanced(inner)) return null
  return inner
}

function isSpecialToken(core: string): boolean {
  return core.startsWith('<') || core === 'BREAK' || core === 'AND'
}

/** 등록 목록 → 비교키 → 가중치 문자열. 같은 태그는 뒤의 것이 이긴다. 1.00 은 뺀다. */
export function weightMap(weights: readonly GlobalWeight[] | null | undefined): Map<string, string> {
  const map = new Map<string, string>()
  for (const w of weights || []) {
    if (!w || typeof w.tag !== 'string') continue
    const key = normalizeTagKey(w.tag)
    if (!key) continue
    const formatted = formatWeight(w.weight)
    if (formatted === '1.00') map.delete(key)
    else map.set(key, formatted)
  }
  return map
}

function applyToCore(core: string, map: Map<string, string>): string {
  if (!core || isSpecialToken(core)) return core
  const inner = weightedInner(core)
  if (inner !== null) {
    const w = map.get(normalizeTagKey(inner))
    return w ? `(${inner}:${w})` : core
  }
  const w = map.get(normalizeTagKey(core))
  return w ? `(${wrapBody(core)}:${w})` : core
}

/** 프롬프트 ``text`` 에 글로벌 가중치를 적용한 새 텍스트. 던지지 않는다. */
export function applyGlobalWeights(text: string, weights: readonly GlobalWeight[] | null | undefined): string {
  const src = String(text ?? '')
  const map = weightMap(weights)
  if (!src || map.size === 0) return src
  const segments = splitTopLevel(src) ?? splitFlat(src)
  let out = ''
  for (const seg of segments) {
    const lead = seg.text.match(/^\s*/)?.[0] ?? ''
    const trail = seg.text.slice(lead.length).match(/\s*$/)?.[0] ?? ''
    const core = seg.text.slice(lead.length, seg.text.length - trail.length)
    out += lead + applyToCore(core, map) + trail + seg.sep
  }
  return out
}

/**
 * v-for 행 key — 편집 중인 태그 문자열을 key 로 쓰면 글자마다 행이 다시 마운트돼 포커스를
 * 잃는다. 행 객체마다 세션 한정 번호를 붙인다(저장 payload 에는 들어가지 않는다).
 */
export function createRowKeyer() {
  const ids = new WeakMap<object, number>()
  let next = 1
  return (row: object): number => {
    let id = ids.get(row)
    if (id === undefined) { id = next++; ids.set(row, id) }
    return id
  }
}
