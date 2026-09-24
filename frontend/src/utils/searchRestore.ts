export interface SearchCacheRestore<T> {
  active: T[]
  base: T[]
  isFiltered: boolean
}

export interface SearchResultLineage {
  label: string
  fingerprint: string
  snapshot_id: string
}

/** 하위 호환 형태 — 필터된 행 전체를 보낸다(수만 행이면 수십 MB). */
export interface SearchDeckRowsUpdate<T> {
  results: T[]
  lineage: SearchResultLineage
}

/**
 * 권장 형태 — 필터 결과를 base(검색 결과 전체) 배열의 인덱스로 보낸다.
 * Python 은 같은 순서의 base 를 들고 있어 `base_size` 가 맞을 때만 받아들인다.
 */
export interface SearchDeckIndexUpdate {
  indices: number[]
  base_size: number
  lineage: SearchResultLineage
}

export type SearchDeckUpdate<T> = SearchDeckRowsUpdate<T> | SearchDeckIndexUpdate

// base 배열 → (행 객체 → 인덱스). 필터를 몇 번 바꿔도 같은 base 면 한 번만 만든다.
const rowIndexCache = new WeakMap<object, Map<unknown, number>>()

/** base 행 객체 → 위치. `toKey` 는 같은 base 에 대해 늘 같은 함수여야 한다(예: Vue toRaw). */
export function searchRowIndex<T extends object>(
  base: readonly T[],
  toKey: (row: T) => unknown = row => row,
): Map<unknown, number> {
  let index = rowIndexCache.get(base)
  if (!index) {
    index = new Map()
    for (let i = 0; i < base.length; i++) {
      const key = toKey(base[i])
      if (!index.has(key)) index.set(key, i)
    }
    rowIndexCache.set(base, index)
  }
  return index
}

/** 두 lineage 가 같은 검색 스냅숏을 가리키는가. */
export function sameSearchLineage(
  a: SearchResultLineage | null | undefined,
  b: SearchResultLineage | null | undefined,
): boolean {
  if (!a || !b) return false
  return a.label === b.label && a.fingerprint === b.fingerprint && a.snapshot_id === b.snapshot_id
}

/** Parse the additive backend lineage event without trusting malformed JSON. */
export function parseSearchResultLineage(raw: string): SearchResultLineage | null {
  try {
    const value = JSON.parse(raw)
    if (!value || typeof value !== 'object') return null
    const label = value.label
    const fingerprint = value.fingerprint
    const snapshotId = value.snapshot_id
    if (typeof label !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(label)) return null
    if (typeof fingerprint !== 'string' || !/^[A-Fa-f0-9]{64}$/.test(fingerprint)) return null
    if (typeof snapshotId !== 'string' || !/^[A-Fa-f0-9]{32}$/.test(snapshotId)) return null
    return {
      label,
      fingerprint: fingerprint.toLowerCase(),
      snapshot_id: snapshotId.toLowerCase(),
    }
  } catch {
    return null
  }
}

/**
 * Bind the exact result lineage to every filter/deck mutation request.
 *
 * `base` 를 주면 모든 행이 base 에 있을 때 인덱스 형태로 보낸다(행 전체 직렬화 없음).
 * base 에 없는 행이 하나라도 있으면(다른 배열에서 온 행) 행 전체 형태로 되돌아간다.
 */
export function buildSearchDeckUpdate<T extends object>(
  results: T[],
  lineage: SearchResultLineage | null,
  base?: readonly T[] | null,
  toKey: (row: T) => unknown = row => row,
): SearchDeckUpdate<T> | null {
  if (!lineage) return null
  if (base) {
    const index = searchRowIndex(base, toKey)
    const indices: number[] = new Array(results.length)
    let complete = true
    for (let i = 0; i < results.length; i++) {
      const position = index.get(toKey(results[i]))
      if (position === undefined) {
        complete = false
        break
      }
      indices[i] = position
    }
    if (complete) {
      return { indices, base_size: base.length, lineage: { ...lineage } }
    }
  }
  return {
    results,
    lineage: { ...lineage },
  }
}

/** Resolve a validated backend active/full pair into recoverable UI state. */
export function resolveSearchCacheRestore<T>(
  active: unknown,
  full: unknown,
): SearchCacheRestore<T> | null {
  if (!Array.isArray(active) || !Array.isArray(full)) return null
  if (active.length === 0 && full.length === 0) return null
  const base = full.length > 0 ? full : active
  return {
    active: [...active] as T[],
    base: [...base] as T[],
    isFiltered: active.length !== base.length,
  }
}
