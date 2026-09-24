// 태그 자동완성 공통 로직 — PromptPanel(텍스트 모드)과 TagBlockField(블록 모드)가 같이 쓴다.
//
// 브리지 슬롯 getTagSuggestionsRich 가 [{tag, ko, category, desc, count}] 를 준다({tag, ko} 로
// 정규화). 문자열 목록만 주던 옛 getTagSuggestions 슬롯은 운영 호출자가 없어 제거했다.
// 한글이 섞인 입력은 한 글자부터 검색한다("장" 만 쳐도 장발/장갑…이 뜬다) — 블록 모드는
// IME 조합 중인 글자를 v-model 이 아니라 입력 요소 값에서 읽어야 그 한 글자가 잡힌다.
// 요청 수명주기(디바운스 취소·늦은 응답 버리기)는 composables/useTagAutocomplete.ts 가 맡는다.

export interface TagSuggestion {
  tag: string
  ko?: string
  category?: string
  desc?: string
  count?: number
}

const HANGUL = /[ㄱ-ㆎ가-힣]/

export function hasHangul(text: string): boolean {
  return HANGUL.test(text || '')
}

/** 자동완성을 띄우기 시작하는 최소 글자 수 — 영문 2, 한글 1 */
export function minSuggestPrefix(prefix: string): number {
  return hasHangul(prefix) ? 1 : 2
}

/** 브리지 응답(JSON 문자열 또는 이미 파싱된 값)을 {tag, ko} 목록으로 정규화 */
export function normalizeSuggestions(raw: unknown, limit = 10): TagSuggestion[] {
  let value: unknown = raw
  if (typeof raw === 'string') {
    try { value = JSON.parse(raw) } catch { return [] }
  }
  if (!Array.isArray(value)) return []
  const out: TagSuggestion[] = []
  const seen = new Set<string>()
  for (const item of value) {
    let entry: TagSuggestion | null = null
    if (typeof item === 'string') entry = { tag: item }
    else if (item && typeof item === 'object' && typeof (item as any).tag === 'string') {
      const o = item as any
      entry = { tag: o.tag, ko: typeof o.ko === 'string' ? o.ko : '', category: o.category || '', desc: o.desc || '', count: Number(o.count) || 0 }
    }
    if (!entry || !entry.tag || seen.has(entry.tag)) continue
    seen.add(entry.tag)
    out.push(entry)
    if (out.length >= limit) break
  }
  return out
}

/**
 * 백엔드에 후보를 요청한다(getTagSuggestionsRich). 슬롯이 없거나 호출이 던지면 빈 목록.
 * 콜백은 항상 정규화된 목록으로 한 번 호출된다(오류 시 빈 목록).
 */
export function fetchTagSuggestions(backend: any, prefix: string, cb: (items: TagSuggestion[]) => void): void {
  let delivered = false
  const deliver = (json: unknown) => {
    if (delivered) return
    delivered = true
    cb(normalizeSuggestions(json))   // normalizeSuggestions 는 던지지 않는다; cb 의 예외는 호출자 몫
  }
  try {
    if (backend && typeof backend.getTagSuggestionsRich === 'function') {
      backend.getTagSuggestionsRich(prefix, deliver)
    } else {
      deliver([])
    }
  } catch {
    if (!delivered) deliver([])
  }
}

/** 태그 구분자 — 쉼표와 줄바꿈. 텍스트 모드 자동완성의 '지금 편집 중인 태그' 경계다. */
const TAG_SEPARATOR = /[,\n]/

function isSpace(ch: string | undefined): boolean {
  return ch === ' ' || ch === '\t' || ch === '\r' || ch === '　'
}

/** 커서가 놓인 태그 조각. start~end 는 앞뒤 공백을 뺀 교체 구간, query 는 start~커서. */
export interface TagToken {
  start: number
  end: number
  query: string
}

/**
 * 커서(caret) 위치의 태그 조각을 찾는다 — 마지막 쉼표 뒤가 아니라 **커서가 있는** 조각.
 *
 * 예전 구현은 항상 마지막 쉼표 뒤를 질의로 삼아, 중간 태그를 고치는 동안 마지막 태그의
 * 후보가 떠서 Tab/Enter 가 엉뚱한 태그를 바꿨다. 질의는 조각 시작부터 커서까지만 본다
 * ('long_h|air' → 'long_h'), 교체는 조각 전체('long_hair')를 바꾼다.
 */
export function tagTokenAt(text: string, caret?: number | null): TagToken {
  const src = String(text ?? '')
  const pos = Math.max(0, Math.min(src.length, caret == null || !Number.isFinite(caret) ? src.length : caret))
  let start = pos
  while (start > 0 && !TAG_SEPARATOR.test(src[start - 1])) start--
  let end = pos
  while (end < src.length && !TAG_SEPARATOR.test(src[end])) end++
  while (start < end && isSpace(src[start])) start++
  while (end > start && isSpace(src[end - 1])) end--
  const query = pos > start ? src.slice(start, Math.min(pos, end)).trim() : ''
  return { start, end: Math.max(start, end), query }
}

/**
 * ``token`` 구간을 ``tag`` 로 바꾼 새 텍스트와 커서 위치.
 *
 * - 뒤에 이미 구분자(쉼표·줄바꿈)가 있으면 ', ' 를 붙이지 않는다(중간 편집 시 쉼표 중복 방지).
 * - 텍스트 끝이면 예전처럼 ', ' 를 붙여 다음 태그를 바로 칠 수 있게 한다.
 * - 쉼표 바로 뒤라 공백이 없으면 한 칸 띄운다('a,lo' → 'a, long_hair, ').
 */
export function replaceTagToken(text: string, token: TagToken, tag: string): { text: string; caret: number } {
  const src = String(text ?? '')
  const start = Math.max(0, Math.min(src.length, token.start))
  const end = Math.max(start, Math.min(src.length, token.end))
  let before = src.slice(0, start)
  const after = src.slice(end)
  if (before.endsWith(',')) before += ' '
  const rest = after.replace(/^[ \t\r　]+/, '')
  if (!rest) {
    const next = before + tag + ', '
    return { text: next, caret: next.length }
  }
  if (TAG_SEPARATOR.test(rest[0])) {
    const next = before + tag + after
    return { text: next, caret: before.length + tag.length }
  }
  // 구간 뒤에 구분자 없이 글자가 이어지는 경우는 tagTokenAt 이 만들지 않지만, 방어적으로 구분한다.
  const next = before + tag + ', ' + rest
  return { text: next, caret: before.length + tag.length + 2 }
}

/** 자동완성 요청을 보낸 순간의 텍스트와 커서 조각 — 수락 때 그대로인지 확인하는 기준 */
export interface TagQuery {
  text: string
  start: number
  query: string
}

/** 입력 이벤트 시점의 질의 기록을 만든다 */
export function tagQueryAt(text: string, caret?: number | null): TagQuery {
  const src = String(text ?? '')
  const token = tagTokenAt(src, caret)
  return { text: src, start: token.start, query: token.query }
}

/**
 * 수락하려는 지금, 텍스트와 커서 조각이 후보를 요청했던 그 조각 그대로인가.
 *
 * 디바운스(300ms)와 브리지 왕복 사이에 키보드로 커서를 옮기거나(Home·화살표) 값이 밖에서
 * 바뀌면(Undo·백엔드 갱신) 도착한 후보는 **옛 조각**의 것이다. 그때 지금 커서 조각을 바꾸면
 * 엉뚱한 태그가 교체된다('1girl, smile, lon' 에서 Home → Enter 가 '1girl' 을 'long_hair' 로).
 */
export function isSameTagQuery(recorded: TagQuery | null | undefined, text: string, caret?: number | null): boolean {
  if (!recorded) return false
  const src = String(text ?? '')
  if (src !== recorded.text) return false
  const token = tagTokenAt(src, caret)
  return token.start === recorded.start && token.query === recorded.query
}

/** 입력 없이 커서만 옮기는 키 — 자동완성 요청(대기 중 포함)을 무효로 만든다 */
const CARET_MOVE_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'])

export function isCaretMoveKey(key: string | null | undefined): boolean {
  return CARET_MOVE_KEYS.has(String(key || ''))
}
