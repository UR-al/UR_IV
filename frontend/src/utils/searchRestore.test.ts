import { describe, expect, it } from 'vitest'

import {
  buildSearchDeckUpdate,
  parseSearchResultLineage,
  resolveSearchCacheRestore,
  sameSearchLineage,
  searchRowIndex,
} from './searchRestore'


const LINEAGE = {
  label: '2026_07',
  fingerprint: 'a'.repeat(64),
  snapshot_id: 'b'.repeat(32),
}


describe('buildSearchDeckUpdate (index payload)', () => {
  it('sends base indices instead of the filtered rows', () => {
    const base = [{ general: 'a' }, { general: 'b' }, { general: 'c' }, { general: 'd' }]
    const filtered = [base[1], base[3]]

    const payload = buildSearchDeckUpdate(filtered, LINEAGE, base)

    expect(payload).toEqual({ indices: [1, 3], base_size: 4, lineage: LINEAGE })
    expect(payload && 'results' in payload).toBe(false)
  })

  it('keeps duplicate-looking rows apart because it matches row identity', () => {
    const base = [{ general: 'same' }, { general: 'same' }]
    expect(buildSearchDeckUpdate([base[1]], LINEAGE, base)).toEqual({
      indices: [1], base_size: 2, lineage: LINEAGE,
    })
  })

  it('maps wrapped rows through toKey (e.g. Vue toRaw)', () => {
    const base = [{ general: 'a' }, { general: 'b' }]
    const wrapped = new Map(base.map(row => [{ wrapped: row }, row] as const))
    const filtered = [...wrapped.keys()].reverse() as any[]
    const payload = buildSearchDeckUpdate<any>(filtered, LINEAGE, base, row => wrapped.get(row) ?? row)
    expect(payload).toEqual({ indices: [1, 0], base_size: 2, lineage: LINEAGE })
  })

  it('an empty filter result is a valid empty index list', () => {
    const base = [{ general: 'a' }]
    expect(buildSearchDeckUpdate([], LINEAGE, base)).toEqual({ indices: [], base_size: 1, lineage: LINEAGE })
  })

  it('falls back to the full-row payload when a row is not in the base', () => {
    const base = [{ general: 'a' }]
    const stranger = { general: 'not-in-base' }
    expect(buildSearchDeckUpdate([base[0], stranger], LINEAGE, base)).toEqual({
      results: [base[0], stranger],
      lineage: LINEAGE,
    })
  })

  it('never builds a payload without lineage, even with a base', () => {
    expect(buildSearchDeckUpdate([{ general: 'a' }], null, [{ general: 'a' }])).toBeNull()
  })

  it('reuses one row index per base array', () => {
    const base = [{ general: 'a' }, { general: 'b' }]
    expect(searchRowIndex(base)).toBe(searchRowIndex(base))
    expect(searchRowIndex(base).get(base[1])).toBe(1)
    expect(searchRowIndex([...base])).not.toBe(searchRowIndex(base))
  })

  it('lineage copies are detached from the caller object', () => {
    const lineage = { ...LINEAGE }
    const payload = buildSearchDeckUpdate([], lineage, [])
    lineage.snapshot_id = 'c'.repeat(32)
    expect(payload?.lineage.snapshot_id).toBe('b'.repeat(32))
  })
})


describe('sameSearchLineage', () => {
  it('compares every field and rejects missing lineage', () => {
    expect(sameSearchLineage(LINEAGE, { ...LINEAGE })).toBe(true)
    expect(sameSearchLineage(LINEAGE, { ...LINEAGE, snapshot_id: 'c'.repeat(32) })).toBe(false)
    expect(sameSearchLineage(LINEAGE, { ...LINEAGE, fingerprint: 'd'.repeat(64) })).toBe(false)
    expect(sameSearchLineage(null, LINEAGE)).toBe(false)
    expect(sameSearchLineage(LINEAGE, undefined)).toBe(false)
  })
})


describe('resolveSearchCacheRestore', () => {
  it('keeps the full base recoverable after a zero-result active filter', () => {
    const full = [{ general: 'recoverable' }]
    const restored = resolveSearchCacheRestore([], full)

    expect(restored).not.toBeNull()
    expect(restored?.active).toEqual([])
    expect(restored?.base).toEqual(full)
    expect(restored?.isFiltered).toBe(true)

    const afterClearFilters = [...(restored?.base ?? [])]
    expect(afterClearFilters).toEqual(full)
  })

  it('rejects an absent or empty cache pair', () => {
    expect(resolveSearchCacheRestore(null, [])).toBeNull()
    expect(resolveSearchCacheRestore([], [])).toBeNull()
  })

  it('binds a filter payload to the result lineage that produced its rows', () => {
    const lineage = parseSearchResultLineage(JSON.stringify({
      label: '2026_07',
      fingerprint: 'a'.repeat(64),
      snapshot_id: 'b'.repeat(32),
    }))

    expect(buildSearchDeckUpdate([{ general: 'from_a' }], lineage)).toEqual({
      results: [{ general: 'from_a' }],
      lineage: {
        label: '2026_07',
        fingerprint: 'a'.repeat(64),
        snapshot_id: 'b'.repeat(32),
      },
    })
  })

  it('does not build an unbound filter payload', () => {
    expect(parseSearchResultLineage('{"label":"../unsafe"}')).toBeNull()
    expect(buildSearchDeckUpdate([], null)).toBeNull()
  })
})
