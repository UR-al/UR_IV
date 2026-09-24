/**
 * 파일 와일드카드(wildcards/*.txt) 관리자의 순수 규칙 — 파이썬 utils/file_wildcard.py 와 같은 뜻.
 *
 * - 한 줄 = 하나의 그룹(쉼표로 나눈 후보 중 하나를 뽑는다). 생성 때는 모든 그룹에서 하나씩.
 * - '#' 로 시작하는 줄(앞 공백 무시)과 빈 줄은 주석 — 해석에서 빠지지만 파일에는 남아야 한다.
 *   그래서 편집은 백엔드가 준 원문 `lines` 로 하고, 개수 표시만 `tags`(주석 제외)로 한다.
 * - 프롬프트 문법은 `__이름__` 과 `~/이름/~` 가 같은 뜻이다(해석기는 둘 다 푼다). 단 `__이름__` 으로
 *   못 가리키는 이름(밑줄·공백으로 시작/끝, 쉼표, 가운데 `__`)은 `~/이름/~` 만 쓴다(dunderWildcardOk).
 */
export interface WildcardEntry {
  name: string
  file: string
  /** 해석에 쓰이는 줄(주석·빈 줄 제외) — 개수 표시용 */
  tags: string[]
  /** 주석까지 담은 원문 줄 — 편집·저장용(옛 백엔드는 안 보낼 수 있다) */
  lines?: string[]
  [k: string]: unknown
}

export function isWildcardComment(line: string): boolean {
  const s = (line || '').trim()
  return !s || s.startsWith('#')
}

/** 편집기에 올릴 줄 — 원문(lines)이 있으면 그것, 없으면 tags. 복사본을 돌려준다. */
export function editableLines(entry: WildcardEntry | null | undefined): string[] {
  if (!entry) return []
  return [...(Array.isArray(entry.lines) ? entry.lines : entry.tags || [])]
}

/** 저장할 파일 본문 — 주석과 사이 빈 줄은 그대로, 끝의 빈 줄만 걷어낸다. */
export function wildcardContent(lines: string[]): string {
  const out = [...lines]
  while (out.length && !out[out.length - 1].trim()) out.pop()
  return out.join('\n')
}

/** 저장 직후 로컬 목록 갱신값 — 원문과 개수용 줄을 함께 맞춘다. */
export function savedEntryLines(lines: string[]): { lines: string[]; tags: string[] } {
  const kept = wildcardContent(lines).split('\n')
  const raw = kept.length === 1 && kept[0] === '' ? [] : kept
  return { lines: raw, tags: raw.filter(l => !isWildcardComment(l)).map(l => l.trim()) }
}

/**
 * `__이름__` 으로 이 이름을 가리킬 수 있는가 — 파이썬 DUNDER_WILDCARD_PATTERN 과 같은 규칙.
 * 이름이 공백·밑줄로 시작/끝나거나, 쉼표·줄바꿈이 있거나, 가운데에 `__` 가 있으면 해석기가
 * 못 풀거나(글자 그대로 나간다) 다른 파일을 푼다(`___base__` → base.txt). 그때는 `~/이름/~` 만 된다.
 */
export function dunderWildcardOk(name: string): boolean {
  return !!name && !/^[\s_]|[\s_]$/.test(name) && !/[,\n]/.test(name) && !name.includes('__')
}

/** 프롬프트에 넣을 문법 — `__이름__`(안내 기본), 그 표기로 못 가리키는 이름이면 `~/이름/~`. */
export function wildcardSyntax(name: string): string {
  return dunderWildcardOk(name) ? `__${name}__` : `~/${name}/~`
}

/** 문법 안내에 보일 표기들 — 넣는 표기(wildcardSyntax)가 먼저, 같은 뜻의 다른 표기가 뒤. */
export function wildcardSyntaxForms(name: string): string[] {
  const tilde = `~/${name}/~`
  return dunderWildcardOk(name) ? [`__${name}__`, tilde] : [tilde]
}

/**
 * createWildcard 응답 — 새로 만들었는지(created)와 실제 파일 이름. 이미 있던 파일이면 created=false
 * 이고 name 은 디스크의 표기(대소문자까지)다 — 그 항목을 열면 된다. 실패 응답이면 null.
 */
export function createdWildcard(json: string): { name: string; created: boolean } | null {
  try {
    const r = JSON.parse(json)
    if (!r || r.error || !r.ok || typeof r.name !== 'string' || !r.name) return null
    return { name: r.name, created: r.created !== false }
  } catch {
    return null
  }
}

/**
 * saveWildcard/renameWildcard 응답에서 실제 저장된 이름을 꺼낸다(파일명 규칙으로 바뀔 수 있다).
 * 실패 응답이면 null.
 */
export function savedWildcardName(json: string, fallback: string): string | null {
  try {
    const r = JSON.parse(json)
    if (!r || r.error || !r.ok) return null
    return typeof r.name === 'string' && r.name ? r.name : fallback
  } catch {
    return null
  }
}

/**
 * 이름을 돌려주지 않는 슬롯(deleteWildcard 등)의 응답이 성공인가 — `{ok:true}` 이고 error 가 없을 때만.
 * error 응답·빈 응답·JSON 아님은 실패다(ui/vue_bridge.py 는 예외를 `{error}` 로 돌려준다 — 예전엔
 * 삭제 응답을 읽지 않아, 잠긴/읽기 전용 파일이 디스크에 남아도 목록에서 지우고 성공처럼 보였다).
 */
export function wildcardReplyOk(json: string): boolean {
  try {
    const r = JSON.parse(json)
    return !!r && typeof r === 'object' && !r.error && !!r.ok
  } catch {
    return false
  }
}

/** 실패 토스트 문구 — 응답에 error 글이 있으면 `기본: error`, 없으면(빈 응답·JSON 아님 포함) 기본만. */
export function wildcardFailureMessage(json: string, base: string): string {
  try {
    const r = JSON.parse(json)
    if (r && typeof r === 'object' && r.error) return `${base}: ${String(r.error)}`
  } catch {}
  return base
}

/**
 * 관리자의 '사용' — 프롬프트 칸 끝에 문법을 붙인 결과. 끝의 쉼표·공백을 하나로 정리한 뒤
 * `, 문법, ` 을 붙인다(빈 칸이면 `문법, `). 생성 경로(resolve_file_wildcards)가 이 모양을 푼다
 * (tests/test_file_wildcard.py · test_prompt_cleaner_wildcards.py).
 */
export function insertWildcardSyntax(current: string, syntax: string): string {
  return current ? current.replace(/,?\s*$/, '') + ', ' + syntax + ', ' : syntax + ', '
}
