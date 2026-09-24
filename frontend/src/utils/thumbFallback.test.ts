import { describe, expect, it } from 'vitest'
import { fallbackToOriginal, type ImgLike } from './thumbFallback'

function img(src: string): ImgLike & { src: string } {
  const el = {
    src,
    dataset: {} as Record<string, string | undefined>,
    getAttribute: (name: string) => (name === 'src' ? el.src : null),
    setAttribute: (name: string, value: string) => { if (name === 'src') el.src = value },
  }
  return el
}

describe('fallbackToOriginal', () => {
  it('swaps a failed thumbnail for the original once', () => {
    const el = img('aithumb:thumb?path=a.png&width=256')
    expect(fallbackToOriginal(el, 'file:///a.png')).toBe(true)
    expect(el.src).toBe('file:///a.png')
    // 원본까지 실패하면 멈춘다
    expect(fallbackToOriginal(el, 'file:///a.png')).toBe(false)
  })

  it('falls back again when a new thumbnail width fails later', () => {
    const el = img('aithumb:thumb?path=a.png&width=256')
    fallbackToOriginal(el, 'file:///a.png')
    el.src = 'aithumb:thumb?path=a.png&width=384'
    expect(fallbackToOriginal(el, 'file:///a.png')).toBe(true)
  })

  it('ignores missing targets and urls', () => {
    expect(fallbackToOriginal(null, 'x')).toBe(false)
    expect(fallbackToOriginal(img('a'), '')).toBe(false)
  })
})
