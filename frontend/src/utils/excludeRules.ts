// 제외 규칙 (9종) — `제외 (로컬)` 칸을 규칙으로 나누고 종류를 가리는 프론트 쪽 정의.
//
// 실제 적용은 파이썬 core/exclude_rules.py 가 한다. 관리 창의 규칙 개수·목록·편집, 블록 칸의
// 블록 경계가 적용과 어긋나지 않도록 같은 정의를 여기 둔다. 두 구현은
// excludeRules.cases.json 골든 사례로 묶여 있다(tests/test_exclude_rules.py 도 같은 파일을 읽는다).
//
//   제외                          예외 (유지)
//   word     포함                 ~word     완전 일치 유지
//   *word    완전 일치            ~_word    접미 유지 (…word)
//   _word    접미 (…word)         ~word_    접두 유지 (word…)
//   word_    접두 (word…)         ~_word_   포함 유지
//   _word_   포함 (명시)
//
// 나누기: 쉼표와 줄바꿈만 구분자다. 공백으로는 나누지 않고(`long hair` 는 규칙 하나), 밑줄도
// 파싱 때 바꾸지 않는다 — 앞뒤 밑줄이 규칙 종류를 정한다. 비교할 때만 소문자·밑줄→공백으로 맞춘다.
// 예외 하나: 키워드 없이 표시만 있는 조각(`~`, `*`, `_`, `~_` …)이 줄 끝에 있으면, 쉼표를 만나기 전의
// 다음 조각과 한 규칙이다(`~` 줄 + `solo` 줄 = `~solo`). 옛 적용은 쉼표 목록 안의 줄바꿈을 공백으로 봤다 —
// 그 모양을 줄바꿈 구분 때문에 `solo`(포함 제외)로 뒤집지 않는다. 표시만 있는 조각이 끝이나 쉼표 앞이면 버린다.
//
// 칸을 다시 쓰는 곳(관리 창 편집·삭제, 예외 해제, 블록 칸)은 rewriteExcludeRules 로 바뀐 규칙 자리만
// 고친다 — 한 줄에 한 규칙, 카테고리별 줄 같은 배치를 쉼표 한 줄로 뭉개지 않는다.

export type ExcludeMatchKind = 'exact' | 'contains' | 'prefix' | 'suffix'

export type ExcludeRuleForm =
  | 'contains' | 'exact' | 'suffix' | 'prefix' | 'contains_explicit'
  | 'keep_exact' | 'keep_suffix' | 'keep_prefix' | 'keep_contains'

/** 문법 9종 → 비교 방식과 유지 규칙 여부 (파이썬 RULE_FORMS 와 같은 표) */
export const EXCLUDE_RULE_FORMS: Readonly<Record<ExcludeRuleForm, { match: ExcludeMatchKind; keep: boolean }>> = {
  contains: { match: 'contains', keep: false },
  exact: { match: 'exact', keep: false },
  suffix: { match: 'suffix', keep: false },
  prefix: { match: 'prefix', keep: false },
  contains_explicit: { match: 'contains', keep: false },
  keep_exact: { match: 'exact', keep: true },
  keep_suffix: { match: 'suffix', keep: true },
  keep_prefix: { match: 'prefix', keep: true },
  keep_contains: { match: 'contains', keep: true },
}

export interface ExcludeRule {
  /** 원문 (앞뒤 공백 제거) */
  text: string
  form: ExcludeRuleForm
  match: ExcludeMatchKind
  keep: boolean
  /** 비교용 표기 (소문자, 밑줄→공백, 앞뒤 공백 제거) — 비어 있지 않다 */
  keyword: string
}

// 파이썬 str.strip() 과 같은 공백 집합 — JS trim() 은 ﻿ 를 더 자르고 \x1c-\x1f·\x85 는 남긴다.
const PY_WHITESPACE = '\\t\\n\\v\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000'
const PY_STRIP = new RegExp(`^[${PY_WHITESPACE}]+|[${PY_WHITESPACE}]+$`, 'g')

/** 파이썬 `str.strip()` 과 같은 앞뒤 공백 제거 */
export function pyStrip(text: string): string {
  return text.replace(PY_STRIP, '')
}

/** 규칙 구분자 — 쉼표와 줄바꿈. 공백은 구분자가 아니다. (core/exclude_rules._SEPARATORS 와 같다) */
const SEPARATORS = /[,\r\n]/g
const PY_LEADING = new RegExp(`^[${PY_WHITESPACE}]+`)

/** 원문에서 규칙 하나가 차지하는 자리 [start, end) — 앞뒤 공백은 뺀다 */
export interface ExcludeRuleSpan {
  start: number
  end: number
}

/**
 * `제외 (로컬)` 텍스트 → 규칙 자리 목록 (core/exclude_rules._rule_spans 와 같은 규칙).
 * 쉼표·줄바꿈으로 나누고 빈 조각은 버린다. 표시만 있는 조각(parseExcludeRule 이 null)이 줄바꿈 앞에
 * 있으면 쉼표를 만나기 전의 다음 조각까지 한 자리로 잇는다.
 */
export function excludeRuleSpans(text: unknown): ExcludeRuleSpan[] {
  if (typeof text !== 'string') return []
  const spans: ExcludeRuleSpan[] = []
  let pending: ExcludeRuleSpan | null = null   // 줄 끝에 걸린 표시만 있는 조각 — 다음 조각과 잇는다
  let pos = 0
  for (;;) {
    SEPARATORS.lastIndex = pos
    const m = SEPARATORS.exec(text)
    const segEnd = m ? m.index : text.length
    const sep = m ? m[0] : ''
    const raw = text.slice(pos, segEnd)
    const body = pyStrip(raw)
    if (body) {
      const start = pos + raw.length - raw.replace(PY_LEADING, '').length
      let cur: ExcludeRuleSpan = { start, end: start + body.length }
      if (pending) { cur = { start: pending.start, end: cur.end }; pending = null }
      if ((sep === '\n' || sep === '\r') && parseExcludeRule(text.slice(cur.start, cur.end)) === null) pending = cur
      else spans.push(cur)
    } else if (sep === ',' && pending) {
      spans.push(pending)   // 표시만 있는 조각 뒤에 쉼표 — 잇지 않는다
      pending = null
    }
    if (!m) break
    pos = m.index + 1
  }
  if (pending) spans.push(pending)
  return spans
}

/** `제외 (로컬)` 텍스트 → 규칙 문자열 목록 (앞뒤 공백 제거, 빈 항목 버림, 원문 보존) */
export function splitExcludeRules(text: unknown): string[] {
  if (typeof text !== 'string') return []
  return excludeRuleSpans(text).map(span => text.slice(span.start, span.end))
}

const LINE_BREAK = /\r\n|\r|\n/g

function lineBreakCount(gap: string): number {
  return (gap.match(LINE_BREAK) || []).length
}

/**
 * 규칙 목록을 원래 텍스트에 되써 넣는다 — 바뀌지 않은 규칙 사이의 구분자·줄바꿈을 그대로 둔다.
 * 관리 창 편집·삭제·추가, 예외 켜고 끄기, 블록 칸(TagBlockField 의 join)이 쓴다.
 *
 *   · 앞뒤로 같은 규칙은 손대지 않고 바뀐 구간만 고친다
 *   · 같은 개수로 바뀌면(편집·끌어 옮기기) 규칙만 원래 자리에 바꿔 끼운다 — 구분자는 그대로
 *   · 규칙이 빠지면 그 둘레 구분자 중 줄바꿈이 가장 많은 것 하나만 남긴다(같으면 앞쪽)
 *   · 새 규칙은 ', ' 로 잇는다 — 끝에 더하면 마지막 규칙 뒤(줄 끝 공백·쉼표 등 꼬리는 뒤에 남는다)
 *   · 규칙이 모두 빠지면 빈 칸, 원래 규칙이 없었으면 ', ' 로 이은 목록
 */
export function rewriteExcludeRules(previous: unknown, rules: readonly string[]): string {
  const text = typeof previous === 'string' ? previous : ''
  const next = rules.filter(rule => typeof rule === 'string' && pyStrip(rule) !== '')
  if (next.length === 0) return ''
  const spans = excludeRuleSpans(text)
  const old = spans.map(span => text.slice(span.start, span.end))
  if (old.length === 0) return next.join(', ')

  const n = old.length
  const total = next.length
  let p = 0
  while (p < n && p < total && old[p] === next[p]) p++
  if (p === n && p === total) return text
  let s = 0
  while (s < n - p && s < total - p && old[n - 1 - s] === next[total - 1 - s]) s++
  const k = n - p - s        // 원래 목록에서 바뀐 구간 old[p .. p+k)
  const m = total - p - s    // 새 목록에서 그 자리에 들어가는 next[p .. p+m)

  // gaps[i] = old[i] 와 old[i+1] 사이 원문 (구분자·공백·빈 조각)
  const gaps = spans.slice(1).map((span, i) => text.slice(spans[i].end, span.start))
  if (m > k) {
    // 늘어난 규칙은 바뀐 구간 마지막 자리 뒤(구간이 비었으면 그 자리 앞)에 ', ' 로 잇는다
    gaps.splice(k > 0 ? p + k - 1 : p, 0, ...Array<string>(m - k).fill(', '))
  } else if (m < k) {
    // old[p+m .. p+k) 가 빠진다(앞 m 자리는 새 규칙이 쓴다) — 그 둘레 구분자(앞·사이·뒤)를 지우고,
    // 앞·뒤 규칙이 다 남으면 줄바꿈이 가장 많은 구분자 하나만 둔다(같으면 앞쪽).
    const lo = Math.max(p + m - 1, 0)
    const hi = Math.min(p + k - 1, n - 2)
    const candidates = gaps.slice(lo, hi + 1)
    let kept: string[] = []
    if (candidates.length > k - m) {
      let keep = 0
      candidates.forEach((gap, i) => { if (lineBreakCount(gap) > lineBreakCount(candidates[keep])) keep = i })
      kept = [candidates[keep]]
    }
    gaps.splice(lo, hi - lo + 1, ...kept)
  }
  const out: string[] = [text.slice(0, spans[0].start)]
  next.forEach((rule, i) => {
    if (i > 0) out.push(gaps[i - 1])
    out.push(rule)
  })
  out.push(text.slice(spans[n - 1].end))
  const result = out.join('')
  // 배치를 살리다 규칙이 달라지면 안 된다 — 예: 규칙이 빠져 표시만 있는 `~` 가 다음 줄 규칙과 이어지면
  // (`a, ~, b\nc` 에서 b 삭제 → `a, ~\nc` = `~c`). 그럴 때만 쉼표 한 줄로 쓴다(쉼표 뒤로는 잇지 않는다).
  const joined = next.join(', ')
  return sameList(splitExcludeRules(result), splitExcludeRules(joined)) ? result : joined
}

function sameList(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((item, i) => item === b[i])
}

/** 비교용 표기 — 밑줄→공백, 앞뒤 공백 제거, 소문자. 규칙 키워드와 태그에 같이 쓴다. */
export function normalizeExcludeTag(text: string): string {
  return pyStrip(text.replace(/_/g, ' ')).toLowerCase()
}

function classify(rule: string): [ExcludeRuleForm, string] {
  if (rule.startsWith('~')) {
    const inner = pyStrip(rule.slice(1))
    if (inner.startsWith('_') && inner.endsWith('_') && inner.length > 2) return ['keep_contains', inner.slice(1, -1)]
    if (inner.startsWith('_')) return ['keep_suffix', inner.slice(1)]
    if (inner.endsWith('_')) return ['keep_prefix', inner.slice(0, -1)]
    return ['keep_exact', inner]
  }
  if (rule.startsWith('*')) return ['exact', rule.slice(1)]
  if (rule.startsWith('_') && rule.endsWith('_') && rule.length > 2) return ['contains_explicit', rule.slice(1, -1)]
  if (rule.startsWith('_')) return ['suffix', rule.slice(1)]
  if (rule.endsWith('_')) return ['prefix', rule.slice(0, -1)]
  return ['contains', rule]
}

/** 규칙 하나 → ExcludeRule. 빈 칸이거나 키워드가 비면(`_`·`__`·`*`·`~`) null — 적용 쪽도 버린다. */
export function parseExcludeRule(text: unknown): ExcludeRule | null {
  if (typeof text !== 'string') return null
  const rule = pyStrip(text)
  if (!rule) return null
  const [form, rawKeyword] = classify(rule)
  const keyword = normalizeExcludeTag(rawKeyword)
  if (!keyword) return null
  const { match, keep } = EXCLUDE_RULE_FORMS[form]
  return { text: rule, form, match, keep, keyword }
}

/** 비교 방식 하나 — `text`·`keyword` 는 같은 표기(normalizeExcludeTag)여야 한다 (파이썬 keyword_matches) */
function keywordMatches(match: ExcludeMatchKind, text: string, keyword: string): boolean {
  switch (match) {
    case 'exact': return text === keyword
    case 'contains': return text.includes(keyword)
    case 'prefix': return text.startsWith(keyword)
    case 'suffix': return text.endsWith(keyword)
  }
}

/** 규칙 하나를 끝에 더한다 — 기존 배치는 그대로 두고 마지막 규칙 뒤에 ', ' 로 잇는다. 빈 규칙은 무시. */
export function appendExcludeRule(text: string, rule: string): string {
  if (typeof rule !== 'string' || !pyStrip(rule)) return text
  return rewriteExcludeRules(text, [...splitExcludeRules(text), rule])
}

/** 태그로 새 규칙을 만들 때의 표기 — 공백형으로 써야 앞뒤 밑줄이 접두·접미 표시로 읽히지 않는다 */
function tagAsKeyword(tag: string): string {
  return tag.replace(/_/g, ' ').trim()
}

function isRuleFor(rule: string, form: ExcludeRuleForm, keyword: string): boolean {
  const parsed = parseExcludeRule(rule)
  return parsed !== null && parsed.form === form && parsed.keyword === keyword
}

/** 예외(유지) 규칙 색인 — 칸 텍스트가 바뀔 때 한 번 만들고 태그마다 묻는다(관리 창 매칭 태그는 수천 개). */
export interface ExcludeKeepIndex {
  /**
   * 이 태그를 지우지 않게 하는 예외 규칙 — 적용(core/exclude_rules.ExcludeRuleSet._kept)과 같은 판정.
   * `~태그`(완전 일치)가 있으면 그것, 없으면 걸리는 첫 패턴 예외(`~_x`·`~x_`·`~_x_`), 없으면 null.
   */
  keptBy(tag: string): ExcludeRule | null
}

export function buildExcludeKeepIndex(text: unknown): ExcludeKeepIndex {
  const exact = new Map<string, ExcludeRule>()
  const patterns: ExcludeRule[] = []
  for (const item of splitExcludeRules(text)) {
    const rule = parseExcludeRule(item)
    if (!rule || !rule.keep) continue
    if (rule.match !== 'exact') patterns.push(rule)
    else if (!exact.has(rule.keyword)) exact.set(rule.keyword, rule)
  }
  return {
    keptBy(tag: string): ExcludeRule | null {
      const nt = normalizeExcludeTag(tag)
      if (!nt) return null
      return exact.get(nt) ?? patterns.find(rule => keywordMatches(rule.match, nt, rule.keyword)) ?? null
    },
  }
}

/**
 * 관리 창 태그 클릭 — `~태그` 유지 규칙을 켜고 끈다.
 *   · 이 태그의 `~` 완전 일치 규칙이 있으면 모두 뺀다 (다른 규칙과 줄 배치는 그대로)
 *   · 없는데 패턴 예외(`~tank_`·`~_hair` …)로 이미 유지되면 그대로 둔다 — 중복 `~태그` 를 더하지 않는다.
 *     패턴 규칙은 다른 태그도 지키므로 클릭으로 끄지 않는다(그 규칙을 직접 고쳐야 풀린다).
 *   · 둘 다 아니면 `~태그` 를 끝에 더한다
 */
export function toggleKeepExact(text: string, tag: string): string {
  const keyword = normalizeExcludeTag(tag)
  if (!keyword) return text
  const rules = splitExcludeRules(text)
  if (!rules.some(rule => isRuleFor(rule, 'keep_exact', keyword))) {
    if (buildExcludeKeepIndex(text).keptBy(tag)) return text
    return appendExcludeRule(text, '~' + tagAsKeyword(tag))
  }
  // 뒤에서부터 하나씩 — 흩어진 규칙마다 자기 둘레 구분자만 정리한다
  let out = text
  const current = rules.slice()
  for (let i = current.length - 1; i >= 0; i--) {
    if (!isRuleFor(current[i], 'keep_exact', keyword)) continue
    current.splice(i, 1)
    out = rewriteExcludeRules(out, current)
  }
  return out
}

/** 관리 창 매칭 태그 버튼의 안내 — 클릭이 무엇을 하는지 (toggleKeepExact 와 같은 갈래) */
export function keepToggleHint(keptBy: ExcludeRule | null): string {
  if (!keptBy) return '클릭: 예외(~) 추가 · 우클릭: 완전 일치(*) 제외 추가'
  const rule = keptBy.text.replace(/\s+/g, ' ')
  if (keptBy.form === 'keep_exact') return `예외 '${rule}' — 클릭하면 해제`
  return `예외 규칙 '${rule}' 로 유지됨 — 클릭으로는 풀리지 않는다 (그 규칙을 고친다)`
}

/** `*태그` 완전 일치 제외 규칙을 더한다 — 같은 태그의 완전 일치 규칙이 이미 있으면 그대로. */
export function addExactExcludeRule(text: string, tag: string): string {
  const keyword = normalizeExcludeTag(tag)
  if (!keyword) return text
  if (splitExcludeRules(text).some(rule => isRuleFor(rule, 'exact', keyword))) return text
  return appendExcludeRule(text, '*' + tagAsKeyword(tag))
}

const FORM_COLOR: Readonly<Record<ExcludeRuleForm, string>> = {
  keep_exact: 'bc-action', keep_suffix: 'bc-action', keep_prefix: 'bc-action', keep_contains: 'bc-action',
  exact: 'bc-expression',
  contains: 'bc-nsfw', contains_explicit: 'bc-nsfw',
  suffix: 'bc-body',
  prefix: 'bc-clothing',
}

/** 블록·관리 창 색 — 규칙 종류별. 아무 것도 하지 않는 규칙(키워드가 빔)은 색이 없다. */
export function excludeRuleColorClass(text: string): string {
  const parsed = parseExcludeRule(text)
  return parsed ? FORM_COLOR[parsed.form] : ''
}
