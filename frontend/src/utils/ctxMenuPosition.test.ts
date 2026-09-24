import { describe, expect, it } from 'vitest'
import { clampMenuPosition, menuPositionStyle } from './ctxMenuPosition'

const viewport = { width: 1000, height: 800 }
const menu = { width: 220, height: 360 }

describe('clampMenuPosition', () => {
  it('keeps a menu that fits where it was opened', () => {
    expect(clampMenuPosition({ x: 100, y: 100 }, menu, viewport)).toEqual({ x: 100, y: 100 })
  })

  it('pulls a menu opened near the bottom-right corner back inside', () => {
    expect(clampMenuPosition({ x: 950, y: 700 }, menu, viewport)).toEqual({ x: 770, y: 430 })
  })

  it('never leaves the top-left edge when the viewport is smaller than the menu', () => {
    expect(clampMenuPosition({ x: 50, y: 50 }, { width: 400, height: 900 }, { width: 300, height: 600 }))
      .toEqual({ x: 10, y: 8 })
  })

  it('formats a style object', () => {
    expect(menuPositionStyle({ x: 3, y: 4 })).toEqual({ top: '4px', left: '3px' })
  })
})
