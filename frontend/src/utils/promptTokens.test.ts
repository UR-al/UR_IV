import { describe, expect, it } from 'vitest'
import { approxTokens } from './promptTokens'

describe('approxTokens', () => {
  it('빈 문자열과 공백·주석만 있으면 0 을 돌려준다', () => {
    expect(approxTokens('')).toBe(0)
    expect(approxTokens('   , , ')).toBe(0)
    expect(approxTokens('// comment only')).toBe(0)
    expect(approxTokens('/* block comment */')).toBe(0)
  })

  it('쉼표로 나눈 태그의 단어 수와 쉼표 수를 합산한다', () => {
    expect(approxTokens('1girl')).toBe(1)
    expect(approxTokens('1girl, blue eyes, long hair')).toBe(7) // 5 words + 2 commas
  })

  it('인라인·블록 주석을 제외한 토큰만 센다', () => {
    expect(approxTokens('1girl, // ignored tag\nblue eyes /* also ignored */')).toBe(4) // 3 words + 1 comma
  })
})
