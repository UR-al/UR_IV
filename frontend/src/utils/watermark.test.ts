import { describe, expect, it } from 'vitest'
import {
  NO_WATERMARK_IMAGE, WATERMARK_FONTS, watermarkFileLabel, watermarkPreviewToRefresh,
} from './watermark'

describe('watermarkFileLabel', () => {
  it('shows the file name of the loaded watermark image', () => {
    expect(watermarkFileLabel('C:/Users/me/로고 이미지.png')).toBe('로고 이미지.png')
    expect(watermarkFileLabel('C:\\Users\\me\\logo.webp')).toBe('logo.webp')
    expect(watermarkFileLabel('/home/me/pics/mark.png/')).toBe('mark.png')
  })

  it('falls back to the empty label', () => {
    expect(watermarkFileLabel('')).toBe(NO_WATERMARK_IMAGE)
    expect(watermarkFileLabel(null)).toBe(NO_WATERMARK_IMAGE)
    expect(watermarkFileLabel('   ', '없음')).toBe('없음')
  })
})

describe('watermarkPreviewToRefresh', () => {
  it('redraws the kind that was last shown, only when it can be drawn', () => {
    expect(watermarkPreviewToRefresh('text', true, false)).toBe('text')
    expect(watermarkPreviewToRefresh('text', false, true)).toBeNull()
    expect(watermarkPreviewToRefresh('image', false, true)).toBe('image')
    expect(watermarkPreviewToRefresh('image', true, false)).toBeNull()
    expect(watermarkPreviewToRefresh(null, true, true)).toBeNull()
  })
})

describe('WATERMARK_FONTS', () => {
  it('offers a Korean-capable font and no duplicates', () => {
    expect(WATERMARK_FONTS).toContain('맑은 고딕')
    expect(new Set(WATERMARK_FONTS).size).toBe(WATERMARK_FONTS.length)
  })
})
