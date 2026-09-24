import { describe, expect, it } from 'vitest'
import {
  HOLE_BLACK, HOLE_WHITE, buildMoveBackground, clipRect, composeMoveFrame, holeColorFor,
  type PixelRect,
} from './movePreview'

/** 예전 renderMovePreview 와 같은 결과(매번 bg 에서 새로 합성)를 만드는 기준 구현 */
function composeFull(bg: Uint32Array, src: Uint32Array, mask: Uint8Array, w: number, h: number, dx: number, dy: number) {
  const out = new Uint32Array(bg)
  for (let y = 0; y < h; y++) {
    const ty = y + dy
    if (ty < 0 || ty >= h) continue
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] === 0) continue
      const tx = x + dx
      if (tx < 0 || tx >= w) continue
      out[ty * w + tx] = src[y * w + x]
    }
  }
  return out
}

function maskBBox(mask: Uint8Array, w: number, h: number): PixelRect {
  let x1 = w, y1 = h, x2 = -1, y2 = -1
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!mask[y * w + x]) continue
    x1 = Math.min(x1, x); y1 = Math.min(y1, y); x2 = Math.max(x2, x); y2 = Math.max(y2, y)
  }
  return { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 }
}

/** 결정적 의사 난수 — 테스트가 매번 같은 입력을 본다 */
function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000 }
}

describe('buildMoveBackground', () => {
  it('punches the mask area with the hole color and keeps the rest', () => {
    const w = 4, h = 3
    const src = Uint32Array.from({ length: w * h }, (_, i) => i + 1)
    const mask = new Uint8Array(w * h)
    mask[5] = 255; mask[6] = 255
    const bg = buildMoveBackground(src, mask, w, h, { x: 1, y: 1, w: 2, h: 1 })
    expect(bg[5]).toBe(HOLE_BLACK)
    expect(bg[6]).toBe(HOLE_BLACK)
    expect(bg[0]).toBe(1)
    expect(src[5]).toBe(6)   // 스냅숏은 건드리지 않는다
    const white = buildMoveBackground(src, mask, w, h, { x: 1, y: 1, w: 2, h: 1 }, holeColorFor('white'))
    expect(white[5]).toBe(HOLE_WHITE)
  })
})

describe('clipRect', () => {
  it('clips to the image and rejects empty rects', () => {
    expect(clipRect({ x: -2, y: 1, w: 5, h: 10 }, 4, 4)).toEqual({ x: 0, y: 1, w: 3, h: 3 })
    expect(clipRect({ x: 5, y: 0, w: 2, h: 2 }, 4, 4)).toBeNull()
  })
})

describe('composeMoveFrame', () => {
  it('matches a full recomposition for every frame of a drag, including off-canvas moves', () => {
    const w = 37, h = 23
    const rand = rng(42)
    const src = Uint32Array.from({ length: w * h }, () => (rand() * 0xffffffff) >>> 0)
    const mask = new Uint8Array(w * h)
    // 들쭉날쭉한 조각(반투명 픽셀 포함 — 대입이라 합성하지 않는다)
    for (let y = 5; y < 15; y++) for (let x = 8; x < 20; x++) if (rand() > 0.3) mask[y * w + x] = 255
    const bbox = maskBBox(mask, w, h)
    const bg = buildMoveBackground(src, mask, w, h, bbox)
    const out = new Uint32Array(bg)
    let prev: PixelRect | null = null
    const path: Array<[number, number]> = [
      [0, 0], [1, 0], [3, 2], [10, 5], [30, 10], [60, 0], [-40, -30], [-3, 4], [2, -1], [0, 0],
    ]
    for (const [dx, dy] of path) {
      const frame = composeMoveFrame(out, bg, src, mask, w, h, bbox, prev, dx, dy)
      prev = frame.dest
      expect(Array.from(out)).toEqual(Array.from(composeFull(bg, src, mask, w, h, dx, dy)))
    }
  })

  it('reports only the previous and new piece rects as dirty', () => {
    const w = 100, h = 100
    const src = Uint32Array.from({ length: w * h }, (_, i) => i)
    const mask = new Uint8Array(w * h)
    for (let y = 10; y < 20; y++) for (let x = 10; x < 20; x++) mask[y * w + x] = 255
    const bbox = { x: 10, y: 10, w: 10, h: 10 }
    const bg = buildMoveBackground(src, mask, w, h, bbox)
    const out = new Uint32Array(bg)
    const first = composeMoveFrame(out, bg, src, mask, w, h, bbox, null, 5, 5)
    expect(first.dirty).toEqual([{ x: 15, y: 15, w: 10, h: 10 }])
    const second = composeMoveFrame(out, bg, src, mask, w, h, bbox, first.dest, 50, 60)
    expect(second.dirty).toEqual([{ x: 15, y: 15, w: 10, h: 10 }, { x: 60, y: 70, w: 10, h: 10 }])
    // 화면 전체(10,000픽셀)가 아니라 두 사각형(200픽셀)만 다시 올린다
    const area = second.dirty.reduce((s, r) => s + r.w * r.h, 0)
    expect(area).toBe(200)
  })

  it('handles a piece moved entirely off the canvas', () => {
    const w = 10, h = 10
    const src = Uint32Array.from({ length: w * h }, (_, i) => i + 100)
    const mask = new Uint8Array(w * h)
    mask[55] = 255
    const bbox = { x: 5, y: 5, w: 1, h: 1 }
    const bg = buildMoveBackground(src, mask, w, h, bbox)
    const out = new Uint32Array(bg)
    const f1 = composeMoveFrame(out, bg, src, mask, w, h, bbox, null, 2, 0)
    const f2 = composeMoveFrame(out, bg, src, mask, w, h, bbox, f1.dest, 50, 0)
    expect(f2.dest).toBeNull()
    expect(Array.from(out)).toEqual(Array.from(bg))
  })
})
