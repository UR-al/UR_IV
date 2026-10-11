/**
 * 프롬프트 주석 — `// 줄 끝까지` 와 `/* 블록 *\/`. 생성 요청에는 절대 가지 않는다(파이썬 core/prompt_comments.py 가
 * 두 백엔드 입구에서 뗀다). 여기서는 같은 문법으로 블록 칸에 흐리게 표시하고, 주석 안에서는 자동완성을 끄고,
 * 블록을 나누고 다시 이을 때 줄 주석 뒤 줄바꿈을 지킨다(쉼표로 이으면 다음 태그가 주석에 먹힌다).
 *
 * 문법(파이썬과 같다 — promptComments.cases.json 골든 사례로 묶여 있다):
 * - `//`·`/*` 는 맨 앞, 공백 문자 뒤, ASCII 쉼표 뒤, 블록 주석이 끝난 바로 그 자리에서만 주석을 시작한다.
 *   그래서 `http://`·`a//b`·`__x/*__`·`~/x/~`·`<lora:dir/name:1>` 은 글자 그대로다.
 * - `//` 는 줄 끝(`\n`·`\r`) 앞까지, `/*` 는 처음 만나는 `*\/` 까지. 닫히지 않으면(open) 그 줄 끝까지만.
 * - 줄 주석은 태그를 끝낸다 — 블록 칸이 `a // 메모\nb` 를 두 블록으로 보므로, 뗀 글자도 괄호 밖 줄 주석 왼쪽
 *   내용 뒤에 쉼표를 하나 둔다(`a,\nb`). 왼쪽이 이미 쉼표로 끝나거나, 다음 줄에 내용이 없거나 쉼표로 시작하거나,
 *   왼쪽 끝 낱말이 `BREAK`·`AND` 이면 두지 않는다.
 */

export type CommentKind = 'line' | 'block' | 'open'

export interface CommentSpan {
  start: number
  /** 포함하지 않는 끝(UTF-16 단위) */
  end: number
  kind: CommentKind
}

export interface CommentSegment {
  text: string
  comment: boolean
  /** 닫히지 않은 `/*` 주석 조각 */
  open: boolean
}

const WS = new Set([' ', '\t', '\n', '\r', '\f', '\v', '\u00a0', '\u3000'])
const HWS = new Set([' ', '\t', '\f', '\v', '\u00a0', '\u3000'])
const isEol = (ch: string | undefined) => ch === '\n' || ch === '\r'

function eolEnd(text: string, from: number): number {
  for (let i = from; i < text.length; i++) if (isEol(text[i])) return i
  return text.length
}

/** 주석 구간 목록(원문 순서). 주석이 없으면 빈 배열 — `/` 가 없으면 바로 돌아온다. */
export function commentSpans(text: string): CommentSpan[] {
  if (!text || text.indexOf('/') < 0) return []
  const spans: CommentSpan[] = []
  const n = text.length
  let blockEnd = -1
  let i = text.indexOf('/')
  while (i >= 0 && i < n - 1) {
    const next = text[i + 1]
    const prev = text[i - 1]
    if ((next === '/' || next === '*') && (i === 0 || i === blockEnd || WS.has(prev!) || prev === ',')) {
      if (next === '*') {
        const close = text.indexOf('*/', i + 2)
        if (close >= 0) {
          blockEnd = close + 2
          spans.push({ start: i, end: blockEnd, kind: 'block' })
          i = text.indexOf('/', blockEnd)
          continue
        }
      }
      const end = eolEnd(text, i + 2)
      spans.push({ start: i, end, kind: next === '*' ? 'open' : 'line' })
      i = text.indexOf('/', end)
      continue
    }
    i = text.indexOf('/', i + 1)
  }
  return spans
}

export function hasPromptComment(text: string): boolean {
  return commentSpans(text).length > 0
}

function rstripHws(text: string): string {
  let cut = text.length
  while (cut > 0 && HWS.has(text[cut - 1]!)) cut--
  return text.slice(0, cut)
}

function eolLength(text: string, at: number): number {
  if (text.startsWith('\r\n', at)) return 2
  return isEol(text[at]) ? 1 : 0
}

/** 줄 주석이 태그 경계인지 볼 때 세는 괄호 — `\` 뒤 글자는 세지 않는다(파이썬 _bracket_depth 와 같다) */
const DEPTH_OPEN = '([{'
const DEPTH_CLOSE = ')]}'
/** 줄 주석 왼쪽이 이 낱말로 끝나면 쉼표를 두지 않는다 — 줄 단위 문법(Forge BREAK·AND) */
const LINE_KEYWORD_TAIL = /(?:^|[ \t\n\r\f\v\u00a0\u3000,])(?:BREAK|AND)$/

function bracketDepth(text: string, start: number, end: number, depth: number): number {
  let i = start
  while (i < end) {
    const ch = text[i]!
    if (ch === '\\') { i += 2; continue }
    if (DEPTH_OPEN.includes(ch)) depth++
    else if (DEPTH_CLOSE.includes(ch)) depth = Math.max(0, depth - 1)
    i++
  }
  return depth
}

/** `pos` 뒤에 (공백·줄 끝·이어지는 주석을 건너뛰고) 쉼표가 아닌 내용이 오는가 */
function contentFollows(text: string, spans: CommentSpan[], index: number, pos: number): boolean {
  let k = index + 1
  while (pos < text.length) {
    if (WS.has(text[pos]!)) { pos++; continue }
    if (k < spans.length && spans[k]!.start === pos) { pos = spans[k]!.end; k++; continue }
    return text[pos] !== ','
  }
  return false
}

/**
 * 주석을 뗀 텍스트 — 주석만 있던 줄은 줄째로 없애고, 내용 뒤 줄바꿈은 남긴다(BREAK 줄 유지).
 * 괄호 밖 줄 주석이 내용 뒤에서 줄을 끝내면 그 내용 뒤에 쉼표 하나(블록 경계와 같게).
 * 주석이 없는 줄은 손대지 않고, 결과는 멱등이다. 파이썬 strip_prompt_comments 와 같은 결과.
 */
export function stripPromptComments(text: string): string {
  const spans = commentSpans(text)
  if (!spans.length) return text
  const n = text.length
  let out = ''
  let pos = 0
  let depth = 0
  for (let index = 0; index < spans.length; index++) {
    const { start, end, kind } = spans[index]!
    if (start < pos) continue
    if (start > pos) depth = bracketDepth(text, pos, start, depth)
    out += text.slice(pos, start)
    pos = end
    // (b) 왼쪽이 가로 공백·줄 처음이면 오른쪽 가로 공백은 버린다
    const last = out[out.length - 1]
    if ((last === undefined || HWS.has(last) || isEol(last)) && pos < n && HWS.has(text[pos]!)) {
      while (pos < n && HWS.has(text[pos]!)) pos++
    }
    // (b2) 오른쪽이 쉼표면 왼쪽 가로 공백을 지운다 — 'a /* x */, b' → 'a, b'
    if (text[pos] === ',') out = rstripHws(out)
    // (c) 줄 끝(또는 글 끝)에 닿았으면 왼쪽 가로 공백을 지우고, 주석만 있던 줄은 줄째로 없앤다
    const eol = eolLength(text, pos)
    if (eol || pos >= n) {
      out = rstripHws(out)
      const tail = out[out.length - 1]
      if (tail === undefined || isEol(tail)) {
        if (eol) pos += eol
        else if (out) out = out.slice(0, out.endsWith('\r\n') ? -2 : -1)
      } else if (eol && kind !== 'block' && depth === 0 && tail !== ','
        && !LINE_KEYWORD_TAIL.test(out.slice(-6)) && contentFollows(text, spans, index, pos)) {
        // (d) 괄호 밖 줄 주석이 내용 뒤에서 줄을 끝내면 태그 경계 — 다음 줄 내용과 붙지 않게 쉼표 하나
        out += ','
      }
    }
  }
  return out + text.slice(pos)
}

/**
 * 태그 밑줄 → 공백(블록 칸·자동완성이 넣는 모양). 밑줄 뒤에 `//`·`/*` 가 오는 태그(Danbooru
 * `z.a.t.o._//_i_love_the_world_and_everything_in_it`)는 바꾸면 그 자리부터 주석이 되어 뒤가 잘리므로 그대로 둔다.
 * 파이썬 underscores_to_spaces 와 같다.
 */
export function underscoresToSpaces(tag: string): string {
  if (!tag || !tag.includes('_') || /_(?=\/[/*])/.test(tag)) return tag
  return tag.replace(/_/g, ' ')
}

/** 끝이 줄 주석(또는 닫히지 않은 `/*`)인가 — 뒤에 이어 붙인 글자가 주석에 먹힌다 */
export function endsInLineComment(text: string): boolean {
  const spans = commentSpans(text)
  const last = spans[spans.length - 1]
  return !!last && last.kind !== 'block' && last.end === text.length
}

/**
 * `base` 뒤에 `add` 를 잇는다 — 예전 규칙(끝 쉼표·공백을 떼고 `', '`) 그대로이되, 그렇게 다듬은 끝이
 * 줄 주석이면 줄바꿈으로 잇는다.
 */
export function appendPromptText(base: string, add: string): string {
  if (!base) return add
  if (!add) return base
  const head = base.replace(/,?\s*$/, '')
  return endsInLineComment(head) ? `${head}\n${add}` : `${head}, ${add}`
}

/** 커서(`caret` = 앞 글자 수)가 주석 안인가 — 자동완성을 띄우지 않는다. 줄 주석은 줄 끝 자리까지 포함. */
export function isCaretInComment(text: string, caret: number): boolean {
  if (!text || text.indexOf('/') < 0) return false
  for (const { start, end, kind } of commentSpans(text)) {
    if (start >= caret) break
    if (kind === 'block' ? caret < end : caret <= end) return true
  }
  return false
}

/** 표시용 조각 — 주석과 아닌 부분을 원문 순서로. 주석이 없으면 조각 하나. */
export function commentSegments(text: string): CommentSegment[] {
  const spans = commentSpans(text)
  if (!spans.length) return [{ text, comment: false, open: false }]
  const out: CommentSegment[] = []
  let pos = 0
  for (const { start, end, kind } of spans) {
    if (start > pos) out.push({ text: text.slice(pos, start), comment: false, open: false })
    out.push({ text: text.slice(start, end), comment: true, open: kind === 'open' })
    pos = end
  }
  if (pos < text.length) out.push({ text: text.slice(pos), comment: false, open: false })
  return out
}

/** 주석만 있는가(주석을 떼면 아무것도 없다) */
export function isCommentOnly(text: string): boolean {
  return hasPromptComment(text) && !stripPromptComments(text).trim()
}

const OPEN_BRACKETS = '([{'
const CLOSE_BRACKETS = ')]}'

/** 주석 없는 텍스트의 예전 블록 나누기(TagBlockField.parseText) — 괄호 안·`__와일드카드__` 안 쉼표 보호 */
function splitPlain(p: string): string[] {
  p = p.replace(/__([^_]+)__/g, m => m.replace(/,/g, '\x01'))
  return p.split(',').map(t => t.trim().replace(/\x01/g, ',')).filter(Boolean)
}

/**
 * 프롬프트 칸 → 블록 텍스트 목록. 주석이 없으면 예전 규칙 그대로(괄호·와일드카드 안 쉼표 보호).
 * 주석이 있으면 주석 안 쉼표·괄호는 세지 않고, 괄호 밖 줄 주석 뒤 줄바꿈이 블록 경계다 —
 * `a // 메모\nb` → `['a // 메모', 'b']`. 다시 이을 때는 {@link joinPromptBlocks}.
 */
export function splitPromptBlocks(text: string): string[] {
  if (!text) return []
  const spans = commentSpans(text)
  let depth = 0
  let p = ''
  if (!spans.length) {
    for (const ch of text) {
      if (OPEN_BRACKETS.includes(ch)) depth++
      else if (CLOSE_BRACKETS.includes(ch)) depth = Math.max(0, depth - 1)
      p += (ch === ',' && depth > 0) ? '\x01' : ch
    }
    return splitPlain(p)
  }
  let pos = 0
  for (const { start, end, kind } of spans) {
    for (let i = pos; i < start; i++) {
      const ch = text[i]!
      if (OPEN_BRACKETS.includes(ch)) depth++
      else if (CLOSE_BRACKETS.includes(ch)) depth = Math.max(0, depth - 1)
      p += (ch === ',' && depth > 0) ? '\x01' : ch
    }
    p += text.slice(start, end).replace(/,/g, '\x01')
    pos = end
    if (kind !== 'block' && depth === 0 && pos < text.length) {
      p += ','
      pos += eolLength(text, pos)
    }
  }
  for (let i = pos; i < text.length; i++) {
    const ch = text[i]!
    if (OPEN_BRACKETS.includes(ch)) depth++
    else if (CLOSE_BRACKETS.includes(ch)) depth = Math.max(0, depth - 1)
    p += (ch === ',' && depth > 0) ? '\x01' : ch
  }
  return splitPlain(p)
}

/** 블록 목록 → 칸 텍스트. `', '` 로 잇되 줄 주석으로 끝나는 블록 뒤는 줄바꿈. 주석이 없으면 `join(', ')` 와 같다. */
export function joinPromptBlocks(parts: readonly string[]): string {
  let out = ''
  parts.forEach((part, i) => {
    if (i > 0) out += endsInLineComment(parts[i - 1]!) ? '\n' : ', '
    out += part
  })
  return out
}
