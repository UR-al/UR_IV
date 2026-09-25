import { describe, expect, it } from 'vitest'
import { EMPTY_INFO, displayInfo, formatResolution, formatSeed } from './imageInfo'

describe('formatResolution', () => {
  it('formats positive sizes', () => {
    expect(formatResolution(832, 1216)).toBe('832 × 1216')
    expect(formatResolution('640', '768')).toBe('640 × 768')
  })
  it('returns empty when a side is missing or invalid (never "undefined")', () => {
    expect(formatResolution(undefined, undefined)).toBe('')
    expect(formatResolution(832, undefined)).toBe('')
    expect(formatResolution(0, 512)).toBe('')
    expect(formatResolution('abc', 512)).toBe('')
    expect(formatResolution(null, null)).toBe('')
  })
})

describe('formatSeed', () => {
  it('keeps integer seeds, including -1 and large string seeds', () => {
    expect(formatSeed(123)).toBe('123')
    expect(formatSeed(-1)).toBe('-1')
    expect(formatSeed('18446744073709551615')).toBe('18446744073709551615')
  })
  it('returns empty for missing or non-numeric seeds', () => {
    expect(formatSeed(undefined)).toBe('')
    expect(formatSeed(null)).toBe('')
    expect(formatSeed('')).toBe('')
    expect(formatSeed('abc')).toBe('')
    expect(formatSeed(Number.NaN)).toBe('')
  })
})

describe('displayInfo', () => {
  it('shows a dash for empty values', () => {
    expect(displayInfo('')).toBe(EMPTY_INFO)
    expect(displayInfo(undefined)).toBe(EMPTY_INFO)
    expect(displayInfo('  ')).toBe(EMPTY_INFO)
    expect(displayInfo('832 × 1216')).toBe('832 × 1216')
  })
})
