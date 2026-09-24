import { describe, expect, it } from 'vitest'
import { isImeComposing } from './imeComposition'

describe('isImeComposing', () => {
  it('flags composing keydowns (isComposing or legacy keyCode 229)', () => {
    expect(isImeComposing({ isComposing: true, keyCode: 13 })).toBe(true)
    expect(isImeComposing({ isComposing: false, keyCode: 229 })).toBe(true)
  })

  it('lets ordinary Enter through', () => {
    expect(isImeComposing({ isComposing: false, keyCode: 13 })).toBe(false)
    expect(isImeComposing({})).toBe(false)
    expect(isImeComposing(null)).toBe(false)
  })
})
