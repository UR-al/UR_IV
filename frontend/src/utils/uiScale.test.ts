import { expect, it } from 'vitest'
import { normalizeUiScale } from './uiScale'

it.each([
  ['1.25', 1.25],
  [1.5, 1.5],
  ['0.7', 0.7],
  [2, 2],
  ['0.69', 1],
  [2.01, 1],
  ['abc', 1],
  [null, 1],
  [undefined, 1],
  ['1.1x', 1.1],   // parseFloat 규칙 그대로 — 예전 동작과 같다
])('%s -> %s', (value, scale) => {
  expect(normalizeUiScale(value)).toBe(scale)
})
