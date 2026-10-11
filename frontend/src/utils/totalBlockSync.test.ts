import { describe, expect, it } from 'vitest'
import { stripPromptComments } from './promptComments'
import { syncFieldToTotal, totalTags } from './totalBlockSync'

/**
 * 최종 프롬프트(블록 모드) → 원래 칸 되쓰기. 최종은 주석을 뗀 합본이라, 주석이 든 칸이 최종과 비교되며
 * 주석·주석 붙은 태그를 잃으면 안 된다. 주석이 없는 칸은 예전 동작(지우기 + 최종 순서로 정렬) 그대로.
 */
describe('syncFieldToTotal', () => {
  it('주석이 없는 칸 — 예전 그대로(사라진 태그 빼기·최종 순서로 정렬·바뀐 게 없으면 null)', () => {
    expect(syncFieldToTotal('a, b, c', 'c, a')).toBe('c, a')
    expect(syncFieldToTotal('a, b', 'x, a, b')).toBeNull()
    expect(syncFieldToTotal(' a, b ', 'a, b')).toBeNull()
    expect(syncFieldToTotal('a', '')).toBe('')
    expect(syncFieldToTotal('', 'a')).toBeNull()
  })

  it('주석이 든 칸 — 최종(주석을 뗀 합본)과 비교해도 주석·주석 붙은 태그가 남는다', () => {
    const field = 'masterpiece, // quality\nbest quality, 1girl // main'
    const total = `char, ${stripPromptComments(field)}, suffix`   // compose_prompt 와 같은 모양
    expect(syncFieldToTotal(field, total)).toBeNull()
  })

  it('최종에서 지운 태그만 빠지고, 주석만 있는 토큰은 남고, 줄 주석 뒤 줄바꿈이 지켜진다', () => {
    const field = '// head\na // why a\nb, c'
    expect(totalTags(stripPromptComments(field)).pieces.has('a')).toBe(true)
    const result = syncFieldToTotal(field, 'b, c')
    expect(result).toBe('// head\nb, c')
    expect(stripPromptComments(result!)).toBe('b, c')
    expect(syncFieldToTotal('x // n\ny, z', 'x, z')).toBe('x // n\nz')
  })

  it('괄호 묶음·블록 주석이 든 태그도 비교한다', () => {
    const field = '(a, b:1.2) // w\nc /* x */ d'
    const total = stripPromptComments(field)
    expect(syncFieldToTotal(field, total)).toBeNull()
    expect(syncFieldToTotal(field, '(a, b:1.2)')).toBe('(a, b:1.2) // w')
  })
})
