import { describe, expect, it } from 'vitest'
import {
  PREVIEW_MAX_EDGE, PreviewGate, downscaleMaskNearest, previewDims, shouldDeferPreviewHide,
} from './editorPreview'

describe('previewDims', () => {
  it('matches the backend preview downscale (long edge 1024, floor)', () => {
    // vue_bridge: (max(1, int(w * r)), max(1, int(h * r))) with r = 1024 / long
    expect(previewDims(832, 1216)).toEqual({ w: Math.floor(832 * (1024 / 1216)), h: 1024 })
    expect(previewDims(4096, 2048)).toEqual({ w: 1024, h: 512 })
    expect(previewDims(1024, 1024)).toEqual({ w: 1024, h: 1024 })
    expect(previewDims(640, 480)).toEqual({ w: 640, h: 480 })
    expect(previewDims(100000, 10)).toEqual({ w: 1024, h: 1 })
    expect(PREVIEW_MAX_EDGE).toBe(1024)
  })
})

describe('downscaleMaskNearest', () => {
  it('samples pixel centers', () => {
    // 4×4 → 2×2: 각 2×2 블록의 (1,1) 번째를 본다
    const mask = new Uint8Array([
      0, 0, 0, 0,
      0, 255, 0, 255,
      0, 0, 0, 0,
      0, 0, 0, 255,
    ])
    expect(Array.from(downscaleMaskNearest(mask, 4, 4, 2, 2))).toEqual([255, 255, 0, 255])
  })

  it('keeps a large painted region in proportion', () => {
    const w = 2000, h = 1000
    const mask = new Uint8Array(w * h)
    for (let y = 200; y < 600; y++) for (let x = 500; x < 1500; x++) mask[y * w + x] = 255
    const { w: tw, h: th } = previewDims(w, h)
    const small = downscaleMaskNearest(mask, w, h, tw, th)
    let on = 0
    for (const v of small) if (v) on++
    const expected = (400 * 1000) * (tw * th) / (w * h)
    expect(Math.abs(on - expected) / expected).toBeLessThan(0.02)
  })

  it('returns an empty mask for degenerate sizes', () => {
    expect(downscaleMaskNearest(new Uint8Array(0), 0, 0, 0, 0).length).toBe(0)
  })
})

describe('PreviewGate', () => {
  it('rejects results requested before an invalidation', () => {
    const gate = new PreviewGate()
    const token = gate.current()
    expect(gate.accepts(token)).toBe(true)
    gate.invalidate()   // 프리뷰를 걷었다
    expect(gate.accepts(token)).toBe(false)
    expect(gate.accepts(gate.current())).toBe(true)
  })

  it('does not block results without a token', () => {
    const gate = new PreviewGate()
    gate.invalidate()
    expect(gate.accepts(undefined)).toBe(true)
  })
})

describe('shouldDeferPreviewHide', () => {
  const base = { previewShown: true, baseLoadPending: false, imageSrc: 'a.png?t=1', requestedImageSrc: 'a.png?t=1' }

  it('hides at once when the base image is not changing (tab switch, reset)', () => {
    expect(shouldDeferPreviewHide(base)).toBe(false)
  })

  it('keeps the preview while the committed result is still decoding', () => {
    expect(shouldDeferPreviewHide({ ...base, baseLoadPending: true })).toBe(true)
  })

  it('keeps the preview when the new src arrived but its load has not started yet', () => {
    // 프리뷰 감시가 원본 감시보다 먼저 돈 경우 — 순서에 기대지 않는다
    expect(shouldDeferPreviewHide({ ...base, imageSrc: 'edited_1.png?t=2' })).toBe(true)
  })

  it('never defers a preview that is not on screen', () => {
    expect(shouldDeferPreviewHide({ ...base, previewShown: false, baseLoadPending: true })).toBe(false)
  })

  it('does not wait for an empty src (editor closed)', () => {
    expect(shouldDeferPreviewHide({ ...base, imageSrc: '' })).toBe(false)
  })
})
