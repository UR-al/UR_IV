import { describe, expect, it } from 'vitest'
import {
  MASK_ON, MaskHistory, appendLassoPoint, countMask, dragRect, fillPolygon, fillRect,
  maskToGrayPixels, paintMaskOverlay, pointInPolygon, polygonBBox, rectIsApplicable, rgba32,
  snapshotsOnPress, stampCircle, strokeLine, type MaskBuffer, type Point,
} from './maskOps'

function blank(w: number, h: number): MaskBuffer {
  return { data: new Uint8Array(w * h), w, h }
}

/** 예전 EditorCanvas/InpaintView 의 bbox × pip 채우기 — 스캔라인 결과의 기준 */
function referenceFill(mask: MaskBuffer, pts: Point[], on: boolean): number {
  const { x1, y1, x2, y2 } = polygonBBox(pts, mask.w, mask.h)
  let delta = 0
  for (let y = y1; y < y2; y++) {
    for (let x = x1; x < x2; x++) {
      if (!pointInPolygon(x, y, pts)) continue
      const i = y * mask.w + x
      if (on && mask.data[i] === 0) { mask.data[i] = MASK_ON; delta++ }
      if (!on && mask.data[i] !== 0) { mask.data[i] = 0; delta-- }
    }
  }
  return delta
}

function rng(seed: number) {
  let s = seed >>> 0
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 2 ** 32 }
}

function randomPolygon(rand: () => number, n: number, w: number, h: number, integer = false): Point[] {
  const pts: Point[] = []
  for (let i = 0; i < n; i++) {
    let x = rand() * (w + 20) - 10, y = rand() * (h + 20) - 10
    if (integer) { x = Math.round(x); y = Math.round(y) }
    pts.push({ x, y })
  }
  return pts
}

describe('fillPolygon (scanline) matches the old per-pixel point-in-polygon fill exactly', () => {
  it('random self-intersecting polygons, float and integer vertices, partly outside the image', () => {
    const rand = rng(20260924)
    for (let t = 0; t < 150; t++) {
      const w = 40 + Math.floor(rand() * 40), h = 30 + Math.floor(rand() * 40)
      const pts = randomPolygon(rand, 3 + Math.floor(rand() * 25), w, h, t % 3 === 0)
      const a = blank(w, h), b = blank(w, h)
      const edit = fillPolygon(a, pts, true)
      const ref = referenceFill(b, pts, true)
      expect(a.data).toEqual(b.data)
      expect(edit.delta).toBe(ref)
    }
  })

  it('erasing matches too, and only counts pixels that were on', () => {
    const rand = rng(7)
    for (let t = 0; t < 60; t++) {
      const w = 50, h = 50
      const a = blank(w, h), b = blank(w, h)
      const seed = randomPolygon(rand, 8, w, h)
      fillPolygon(a, seed, true); referenceFill(b, seed, true)
      const cut = randomPolygon(rand, 6 + (t % 10), w, h)
      const edit = fillPolygon(a, cut, false)
      const ref = referenceFill(b, cut, false)
      expect(a.data).toEqual(b.data)
      expect(edit.delta).toBe(ref)
      expect(countMask(a.data)).toBe(countMask(b.data))
    }
  })

  it('handles horizontal edges, duplicate points and a long lasso quickly', () => {
    const square: Point[] = [{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 8, y: 2 }, { x: 8, y: 6 }, { x: 2, y: 6 }]
    const a = blank(12, 10), b = blank(12, 10)
    fillPolygon(a, square, true); referenceFill(b, square, true)
    expect(a.data).toEqual(b.data)
    expect(countMask(a.data)).toBe(6 * 4)

    // 1000 점짜리 원형 올가미(2000×1500) — 예전 방식은 수 초
    const lasso: Point[] = []
    for (let i = 0; i < 1000; i++) {
      const t = (i / 1000) * Math.PI * 2
      lasso.push({ x: 1000 + 900 * Math.cos(t) * (1 + 0.05 * Math.sin(9 * t)), y: 750 + 700 * Math.sin(t) })
    }
    const big = blank(2000, 1500)
    const started = performance.now()
    const edit = fillPolygon(big, lasso, true)
    const elapsed = performance.now() - started
    expect(edit.delta).toBeGreaterThan(1_500_000)
    expect(elapsed).toBeLessThan(400)
  })

  it('returns the clipped bounding box for the caller bookkeeping', () => {
    const edit = fillPolygon(blank(20, 20), [{ x: -5, y: 3.5 }, { x: 10.2, y: 3 }, { x: 4, y: 30 }], true)
    expect([edit.x1, edit.y1, edit.x2, edit.y2]).toEqual([0, 3, 11, 20])
  })
})

describe('circles, strokes and rectangles', () => {
  it('stampCircle turns pixels on/off and reports the delta', () => {
    const m = blank(20, 20)
    const on = stampCircle(m, 10, 10, 3, true)
    expect(on.delta).toBe(countMask(m.data))
    expect(on.delta).toBeGreaterThan(20)
    expect(stampCircle(m, 10, 10, 3, true).delta).toBe(0)
    const off = stampCircle(m, 10, 10, 1, false)
    expect(off.delta).toBeLessThan(0)
    expect(countMask(m.data)).toBe(on.delta + off.delta)
    expect([on.x1, on.y1, on.x2, on.y2]).toEqual([7, 7, 13, 13])
  })

  it('a zero radius still paints (minimum 1px) and off-image circles touch nothing', () => {
    const m = blank(10, 10)
    expect(stampCircle(m, 5, 5, 0, true).delta).toBeGreaterThan(0)
    expect(stampCircle(m, -50, -50, 3, true).delta).toBe(0)
  })

  it('strokeLine is the union of evenly spaced circles', () => {
    const m = blank(60, 20)
    const edit = strokeLine(m, 5, 10, 55, 10, 4, true)
    expect(edit.delta).toBe(countMask(m.data))
    for (let x = 5; x <= 55; x++) expect(m.data[10 * 60 + x]).toBe(MASK_ON)
    expect(edit.x1).toBe(1)
    expect(edit.x2).toBe(59)
  })

  it('fillRect rounds and clips', () => {
    const m = blank(10, 10)
    const edit = fillRect(m, -3, 1.4, 4.6, 3, true)
    expect([edit.x1, edit.y1, edit.x2, edit.y2]).toEqual([0, 1, 5, 3])
    expect(edit.delta).toBe(10)
    expect(fillRect(m, 0, 0, 10, 10, false).delta).toBe(-10)
  })
})

describe('undo policy and history', () => {
  it('only press-painting tools snapshot on pointerdown', () => {
    expect(snapshotsOnPress('brush', 'box')).toBe(true)
    expect(snapshotsOnPress('stamp', 'box')).toBe(true)
    expect(snapshotsOnPress('eraser', 'brush')).toBe(true)
    expect(snapshotsOnPress('eraser', 'box', true)).toBe(true)
    expect(snapshotsOnPress('eraser', 'box')).toBe(false)
    expect(snapshotsOnPress('eraser', 'lasso')).toBe(false)
    expect(snapshotsOnPress('box', 'brush')).toBe(false)
    expect(snapshotsOnPress('lasso', 'brush')).toBe(false)
  })

  it('tiny drags are clicks, not rectangles', () => {
    expect(rectIsApplicable(dragRect(10, 10, 13, 30))).toBe(false)
    expect(rectIsApplicable(dragRect(10, 10, 14.4, 14.6))).toBe(true)
    expect(dragRect(9.6, 20, 2.2, 3.5)).toEqual({ x1: 2, y1: 4, x2: 10, y2: 20 })
  })

  it('MaskHistory undo/redo round-trips and drops stale snapshots', () => {
    const history = new MaskHistory(2)
    const mask = new Uint8Array(4)
    history.save(mask); mask[0] = 255
    history.save(mask); mask[1] = 255
    history.save(mask); mask[2] = 255   // 가장 오래된 것은 밀려난다(max 2)
    expect(history.undoCount).toBe(2)
    expect(history.undo(mask)).toBe(true)
    expect([...mask]).toEqual([255, 255, 0, 0])
    expect(history.redoCount).toBe(1)
    expect(history.redo(mask)).toBe(true)
    expect([...mask]).toEqual([255, 255, 255, 0])
    history.undo(mask)
    history.save(mask)
    expect(history.redoCount).toBe(0)

    const resized = new Uint8Array(9)
    expect(history.undo(resized)).toBe(false)
    expect(history.undoCount).toBe(0)
    expect(history.undo(resized)).toBe(false)
  })
})

describe('lasso points', () => {
  it('drops points closer than 1px to the previous one', () => {
    const pts: Point[] = []
    expect(appendLassoPoint(pts, { x: 1, y: 1 })).toBe(true)
    expect(appendLassoPoint(pts, { x: 1.4, y: 1.4 })).toBe(false)
    expect(appendLassoPoint(pts, { x: 2, y: 1 })).toBe(true)
    expect(appendLassoPoint(pts, { x: 2, y: 1 })).toBe(false)
    expect(pts).toEqual([{ x: 1, y: 1 }, { x: 2, y: 1 }])
  })
})

describe('overlay and PNG pixels', () => {
  it('paints only the requested rectangle', () => {
    const data = new Uint8Array([255, 0, 255, 255])
    const px = new Uint32Array([7, 7, 7, 7])
    const color = rgba32(226, 179, 64, 100)
    paintMaskOverlay(px, data, 2, 0, 0, 2, 1, color)
    expect([...px]).toEqual([color, 0, 7, 7])
    expect(color).toBe(((100 << 24) | (64 << 16) | (179 << 8) | 226) >>> 0)
  })

  it('mask PNG pixels are opaque white/black', () => {
    const out = maskToGrayPixels(new Uint8Array([0, 255, 1]), new Uint32Array(3))
    expect([...out]).toEqual([0xff000000, 0xffffffff, 0xffffffff])
  })
})
