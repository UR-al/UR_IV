import { describe, expect, it } from 'vitest'
import {
  ONLY_RUNNING_ROW_NOTICE, canMoveDown, canMoveUp, clearConfirmMessage, isRowLocked, removableIds,
} from './queueLocks'

describe('queue row locks', () => {
  it('locks only the row that is generating', () => {
    expect(isRowLocked(0, 0)).toBe(true)
    expect(isRowLocked(1, 0)).toBe(false)
    expect(isRowLocked(0, -1)).toBe(false)
  })

  it('never swaps a row with the generating row', () => {
    // 0번이 생성 중 — 1번은 0번 자리로 못 오고, 0번은 내려가지 못한다
    expect(canMoveUp(1, 0)).toBe(false)
    expect(canMoveDown(0, 3, 0)).toBe(false)
    expect(canMoveUp(2, 0)).toBe(true)
    expect(canMoveDown(1, 3, 0)).toBe(true)
  })

  it('allows free reordering when nothing is generating', () => {
    expect(canMoveUp(1, -1)).toBe(true)
    expect(canMoveDown(0, 3, -1)).toBe(true)
  })

  it('respects list bounds', () => {
    expect(canMoveUp(0, -1)).toBe(false)
    expect(canMoveDown(2, 3, -1)).toBe(false)
    expect(canMoveDown(-1, 3, -1)).toBe(false)
  })

  it('drops the generating id from delete requests', () => {
    expect(removableIds(['a', 'b', 'c'], 'b')).toEqual(['a', 'c'])
    expect(removableIds(new Set(['a', '']), null)).toEqual(['a'])
    expect(removableIds(['a'], 'a')).toEqual([])
  })

  it('does not count the generating row when asking to clear everything', () => {
    expect(clearConfirmMessage(3, -1)).toBe('대기열 3개 항목을 모두 삭제할까요?')
    expect(clearConfirmMessage(3, 0)).toBe('생성 중인 1개를 뺀 2개 항목을 삭제할까요?')
    expect(clearConfirmMessage(1, 0)).toBeNull()          // 생성 중인 행뿐 — 지울 것이 없다
    expect(clearConfirmMessage(0, -1)).toBeNull()
    expect(clearConfirmMessage(2, 5)).toBe('대기열 2개 항목을 모두 삭제할까요?')   // 범위 밖 인덱스는 무시
    expect(ONLY_RUNNING_ROW_NOTICE).toContain('생성 중')
  })
})
