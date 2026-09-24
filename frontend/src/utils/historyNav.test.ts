import { describe, expect, it } from 'vitest'
import { edgeHistoryIndex, historyPageOf, historyPageSlice, stepHistoryIndex } from './historyNav'

const LIST = ['a', 'b', 'c', 'd']

describe('stepHistoryIndex', () => {
  it('starts from the newest when nothing (or something unknown) is selected', () => {
    expect(stepHistoryIndex(LIST, '', 1)).toBe(0)
    expect(stepHistoryIndex(LIST, 'zzz', -1)).toBe(0)
  })
  it('moves one step and stops at both ends', () => {
    expect(stepHistoryIndex(LIST, 'b', 1)).toBe(2)
    expect(stepHistoryIndex(LIST, 'b', -1)).toBe(0)
    expect(stepHistoryIndex(LIST, 'a', -1)).toBe(0)
    expect(stepHistoryIndex(LIST, 'd', 1)).toBe(3)
  })
  it('has nothing to do on an empty list', () => {
    expect(stepHistoryIndex([], 'a', 1)).toBeNull()
  })
})

describe('edgeHistoryIndex', () => {
  it('jumps to the newest or the oldest', () => {
    expect(edgeHistoryIndex(LIST, 'top')).toBe(0)
    expect(edgeHistoryIndex(LIST, 'bottom')).toBe(3)
    expect(edgeHistoryIndex([], 'top')).toBeNull()
  })
})

describe('paging', () => {
  it('finds the page of an index and slices a page', () => {
    expect(historyPageOf(0, 5)).toBe(0)
    expect(historyPageOf(4, 5)).toBe(0)
    expect(historyPageOf(5, 5)).toBe(1)
    expect(historyPageSlice(['1', '2', '3', '4', '5', '6', '7'], 1, 5)).toEqual(['6', '7'])
    expect(historyPageSlice(['1', '2'], 3, 5)).toEqual([])
  })
})
