import { describe, expect, it } from 'vitest'
import { formatEta, previewDataUrl, progressPercent } from './generationProgress'

describe('progressPercent', () => {
  it('rounds to a whole percent', () => {
    expect(progressPercent(1, 3)).toBe(33)
    expect(progressPercent(2, 3)).toBe(67)
    expect(progressPercent(20, 20)).toBe(100)
  })
})

describe('formatEta', () => {
  it('is unknown before the first step', () => {
    expect(formatEta(5, 0, 20)).toBeNull()
  })
  it('extrapolates from the elapsed time per step', () => {
    expect(formatEta(10, 5, 20)).toBe('ETA 30s')          // 2s/step × 15
    expect(formatEta(30, 5, 20)).toBe('ETA 1m30s')        // 6s/step × 15 = 90s
    expect(formatEta(25, 5, 20)).toBe('ETA 1m15s')
    expect(formatEta(40, 20, 20)).toBe('ETA 0s')
  })
  it('pads seconds in the minute form', () => {
    expect(formatEta(65, 1, 3)).toBe('ETA 2m10s')
    expect(formatEta(61, 1, 2)).toBe('ETA 1m01s')
  })
})

describe('previewDataUrl', () => {
  it('picks the mime from the base64 head', () => {
    expect(previewDataUrl('/9j/AAAA')).toBe('data:image/jpeg;base64,/9j/AAAA')
    expect(previewDataUrl('UklGRxxxx')).toBe('data:image/webp;base64,UklGRxxxx')
    expect(previewDataUrl('iVBORw0K')).toBe('data:image/png;base64,iVBORw0K')
  })
  it('rejects empty and non-string payloads', () => {
    expect(previewDataUrl('')).toBeNull()
    expect(previewDataUrl(null)).toBeNull()
    expect(previewDataUrl(42)).toBeNull()
  })
})
