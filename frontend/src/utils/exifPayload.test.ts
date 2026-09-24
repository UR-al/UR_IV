import { describe, expect, it } from 'vitest'
import { exifFromPayload, exifTabContent } from './exifPayload'

describe('exifFromPayload', () => {
  it('keeps only the bar fields and fills gaps with empty values', () => {
    expect(exifFromPayload({ prompt: 'p', negative: 'n', raw: 'r', params: { core: 'c' }, params_line: 'l', size: '1×1', extra: 1 }))
      .toEqual({ prompt: 'p', negative: 'n', raw: 'r', params: { core: 'c' }, params_line: 'l' })
    expect(exifFromPayload({})).toEqual({ prompt: '', negative: '', raw: '', params: null, params_line: '' })
    expect(exifFromPayload(null)).toEqual({ prompt: '', negative: '', raw: '', params: null, params_line: '' })
  })
})

describe('exifTabContent', () => {
  const exif = exifFromPayload({ prompt: '1girl', negative: '', raw: 'Steps: 20' })
  it('shows the right text per tab', () => {
    expect(exifTabContent(exif, 'positive')).toBe('1girl')
    expect(exifTabContent(exif, 'negative')).toBe('')
    expect(exifTabContent(exif, 'params')).toBe('Steps: 20')
  })
  it('says there is no EXIF when the positive prompt is empty', () => {
    expect(exifTabContent(exifFromPayload({}), 'positive')).toBe('No EXIF data')
  })
})
