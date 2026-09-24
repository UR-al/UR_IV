import { expect, it } from 'vitest'
import { formatDuration } from './durationFormat'

it.each([
  [0, '0s'],
  [Number.NaN, '0s'],
  [42, '42s'],
  [59.5, '59.5s'],
  [60, '1m 0s'],
  [185.4, '3m 5s'],
  [3599, '59m 59s'],
  [3600, '1h 0m'],
  [8061, '2h 14m'],
])('%s seconds -> %s', (sec, text) => {
  expect(formatDuration(sec)).toBe(text)
})
