import { expect, it } from 'vitest'
import { moveAdjacent } from './listMove'

it('swaps with the neighbour and leaves the input untouched', () => {
  const list = ['a', 'b', 'c']
  expect(moveAdjacent(list, 1, -1)).toEqual(['b', 'a', 'c'])
  expect(moveAdjacent(list, 1, 1)).toEqual(['a', 'c', 'b'])
  expect(list).toEqual(['a', 'b', 'c'])
})

it('refuses moves past either end or from outside the list', () => {
  const list = ['a', 'b']
  expect(moveAdjacent(list, 0, -1)).toBeNull()
  expect(moveAdjacent(list, 1, 1)).toBeNull()
  expect(moveAdjacent(list, 2, -1)).toBeNull()
  expect(moveAdjacent(list, -1, 1)).toBeNull()
  expect(moveAdjacent([], 0, 1)).toBeNull()
})
