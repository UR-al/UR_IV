import { expect, it } from 'vitest'
import { normalizeRandomRes } from './randomResolution'

it('rounds to multiples of 8 and labels the entry', () => {
  expect(normalizeRandomRes(830, 1219)).toEqual([832, 1216, '832x1216'])
  expect(normalizeRandomRes(1024, 1024)).toEqual([1024, 1024, '1024x1024'])
})

it('empty or zero inputs fall back to 832x1216', () => {
  expect(normalizeRandomRes(0, null)).toEqual([832, 1216, '832x1216'])
  expect(normalizeRandomRes(undefined, 640)).toEqual([832, 640, '832x640'])
})

it('rejects sides under 256 (after rounding)', () => {
  expect(normalizeRandomRes(200, 1024)).toBeNull()
  expect(normalizeRandomRes(1024, 251)).toBeNull()
  expect(normalizeRandomRes(253, 1024)).toEqual([256, 1024, '256x1024'])
})
