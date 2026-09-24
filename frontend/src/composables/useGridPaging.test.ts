import { describe, expect, it } from 'vitest'
import { fillTarget, keepVisibleCount, scrollTarget } from './useGridPaging'

describe('grid paging', () => {
  it('fills a wide viewport before any thumbnail has loaded', () => {
    // 1000px 폭 / 200px 칸 = 5열, 600px 높이 → 3행 + 1 = 20장
    expect(fillTarget({ total: 100, visible: 10, clientWidth: 1000, clientHeight: 600, scrollHeight: 0, cell: 200 })).toBe(20)
  })

  it('adds a page when loaded cards still do not overflow, and stops once they do', () => {
    const base = { total: 100, visible: 40, clientWidth: 1000, clientHeight: 600, cell: 200 }
    expect(fillTarget({ ...base, scrollHeight: 600 })).toBe(70)
    expect(fillTarget({ ...base, scrollHeight: 900 })).toBe(40)
  })

  it('never grows while detached (no layout) or past the total', () => {
    expect(fillTarget({ total: 100, visible: 40, clientWidth: 0, clientHeight: 0, scrollHeight: 0, cell: 200 })).toBe(40)
    expect(fillTarget({ total: 30, visible: 30, clientWidth: 1000, clientHeight: 600, scrollHeight: 0, cell: 60 })).toBe(30)
  })

  it('loads more near the bottom only', () => {
    expect(scrollTarget({ total: 100, visible: 40, scrollHeight: 2000, scrollTop: 1300, clientHeight: 600 })).toBe(70)
    expect(scrollTarget({ total: 50, visible: 40, scrollHeight: 2000, scrollTop: 1300, clientHeight: 600 })).toBe(50)
    expect(scrollTarget({ total: 100, visible: 40, scrollHeight: 2000, scrollTop: 0, clientHeight: 600 })).toBe(40)
  })

  it('keeps the scroll depth when a list refreshes but shows at least a page', () => {
    expect(keepVisibleCount(120, 500)).toBe(120)
    expect(keepVisibleCount(120, 60)).toBe(60)
    expect(keepVisibleCount(10, 5)).toBe(40)
  })
})
