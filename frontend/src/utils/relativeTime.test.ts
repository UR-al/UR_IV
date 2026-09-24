import { describe, expect, it } from 'vitest'
import { relativeTimeKo, unreadBadgeText } from './relativeTime'

describe('relativeTimeKo', () => {
  const now = 1_000_000_000_000
  it.each([
    [0, '0초 전'],
    [59_999, '59초 전'],
    [60_000, '1분 전'],
    [59 * 60_000 + 59_999, '59분 전'],
    [3_600_000, '1시간 전'],
    [23 * 3_600_000 + 3_599_999, '23시간 전'],
    [24 * 3_600_000, '1일 전'],
    [3 * 24 * 3_600_000 + 5, '3일 전'],
  ])('%i ms ago -> %s', (ago, text) => {
    expect(relativeTimeKo(now - ago, now)).toBe(text)
  })
})

describe('unreadBadgeText', () => {
  it('shows the count up to nine and 9+ beyond', () => {
    expect(unreadBadgeText(1)).toBe('1')
    expect(unreadBadgeText(9)).toBe('9')
    expect(unreadBadgeText(10)).toBe('9+')
  })
})
