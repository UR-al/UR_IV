/**
 * Search 결과의 character/copyright/artist 칸 분해 — 필터 매니저 옵션·필터 적용 공용.
 *
 * 데이터는 공백 구분 + 태그 안 밑줄(`hatsune_miku kagamine_rin`)이고, 괄호 안 공백은
 * 태그 일부다(`kafka_(honkai:_star_rail)`). 콤마가 있으면 legacy 포맷으로 보고 콤마로만 자른다.
 * 결과는 표시형(밑줄 → 공백)이다.
 */
export function splitDanbooruTags(raw: unknown): string[] {
  const text = String(raw || '').trim()
  if (!text) return []
  if (text.includes(',')) return text.split(',').map(v => v.trim().replace(/_/g, ' ')).filter(Boolean)
  const tags: string[] = []
  let current = ''
  let depth = 0
  for (const ch of text) {
    if (ch === '(') { depth++; current += ch }
    else if (ch === ')') { depth--; current += ch }
    else if (ch === ' ' && depth === 0) {
      if (current.trim()) tags.push(current.trim().replace(/_/g, ' '))
      current = ''
    } else { current += ch }
  }
  if (current.trim()) tags.push(current.trim().replace(/_/g, ' '))
  return tags
}

// 범위(보통 검색 결과 base 배열) → (원문 → 분해 결과). 범위 배열이 버려지면 캐시도 같이 GC 된다.
// 행마다가 아니라 원문 문자열마다 저장한다 — 수십만 행이 같은 작품/캐릭터 문자열을 공유하므로
// 메모리는 고유 문자열 수에 비례하고, 필터 칩을 토글할 때마다 같은 문자열을 다시 쪼개지 않는다.
const splitCaches = new WeakMap<object, Map<string, readonly string[]>>()

/** `scope` 에 묶인 메모이즈된 분해 함수. 같은 scope 면 같은 캐시를 공유한다. */
export function cachedTagSplitter(scope: object): (raw: unknown) => readonly string[] {
  let cache = splitCaches.get(scope)
  if (!cache) {
    cache = new Map()
    splitCaches.set(scope, cache)
  }
  const store = cache
  return (raw: unknown) => {
    const key = raw == null ? '' : String(raw)
    let hit = store.get(key)
    if (hit === undefined) {
      hit = Object.freeze(splitDanbooruTags(key))
      store.set(key, hit)
    }
    return hit
  }
}
