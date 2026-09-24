/**
 * 오른쪽 히스토리 스트립의 키보드 이동(↑/↓, 보조키+↑/↓) — 순수 인덱스 계산(App.vue 에서 추출).
 * 목록의 0 번이 최신이다.
 */

/**
 * 지금 보는 그림에서 한 칸 이동할 인덱스. 선택이 없거나 목록에 없으면 첫 그림(0)부터,
 * 끝에서는 멈춘다. 빈 목록이면 null.
 */
export function stepHistoryIndex(list: readonly string[], current: string, dir: number): number | null {
  if (!list.length) return null
  const idx = list.indexOf(current)
  if (idx < 0) return 0
  return Math.min(list.length - 1, Math.max(0, idx + dir))
}

/** 맨 위(top = 최신) / 맨 아래(bottom = 가장 오래됨). 빈 목록이면 null. */
export function edgeHistoryIndex(list: readonly string[], edge: 'top' | 'bottom' | string): number | null {
  if (!list.length) return null
  return edge === 'top' ? 0 : list.length - 1
}

/** 인덱스가 들어 있는 페이지 번호 */
export function historyPageOf(index: number, perPage: number): number {
  return Math.floor(index / perPage)
}

/** 한 페이지에 보이는 항목 */
export function historyPageSlice<T>(list: readonly T[], page: number, perPage: number): T[] {
  const start = page * perPage
  return list.slice(start, start + perPage)
}
