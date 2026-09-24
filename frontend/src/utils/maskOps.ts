/**
 * 마스크 엔진 공용 연산 — EditorCanvas(에디터)와 InpaintView(인페인트)가 같이 쓴다.
 *
 * 두 곳은 같은 마스크 코드를 복사해 쓰다가 개선이 한쪽(에디터)에만 들어갔다:
 *   - undo 시점: 인페인트는 누를 때마다 스냅숏을 떠서 빈 클릭도 undo 단계를 쌓고 redo 를 날렸다
 *   - 올가미 채우기: 두 곳 모두 bbox 전 픽셀 × 꼭짓점 수(point-in-polygon)라 긴 올가미에서 수 초 멈췄다
 * 여기 함수들은 **상태가 없다** — 마스크 버퍼를 받아 칠하고, 건드린 사각형과 켜진 픽셀 수
 * 변화(delta)를 돌려준다. 경계 상자·dirty 영역·픽셀 카운터 같은 부기는 호출부가 한다.
 *
 * 마스크 값은 0(꺼짐) / 255(켜짐)이다.
 */

export interface Point { x: number; y: number }

export interface MaskBuffer { data: Uint8Array; w: number; h: number }

/** 칠하기 결과 — 클립된 사각형(x2/y2 는 배타)과 켜진 픽셀 수의 변화 */
export interface MaskEdit { x1: number; y1: number; x2: number; y2: number; delta: number }

export const MASK_ON = 255

function blankEdit(): MaskEdit {
  return { x1: Infinity, y1: Infinity, x2: -Infinity, y2: -Infinity, delta: 0 }
}

/** 두 결과를 합친다(사각형은 min/max, delta 는 합). `into` 를 고쳐서 돌려준다. */
export function mergeEdit(into: MaskEdit, edit: MaskEdit): MaskEdit {
  if (edit.x1 < into.x1) into.x1 = edit.x1
  if (edit.y1 < into.y1) into.y1 = edit.y1
  if (edit.x2 > into.x2) into.x2 = edit.x2
  if (edit.y2 > into.y2) into.y2 = edit.y2
  into.delta += edit.delta
  return into
}

/** 원 하나를 켜거나(on) 끈다. 반경은 최소 1px. */
export function stampCircle(mask: MaskBuffer, cx: number, cy: number, r: number, on: boolean): MaskEdit {
  const { data, w, h } = mask
  const radius = Math.max(1, r)
  const rr = radius * radius
  const x1 = Math.max(0, Math.floor(cx - radius)), x2 = Math.min(w, Math.ceil(cx + radius))
  const y1 = Math.max(0, Math.floor(cy - radius)), y2 = Math.min(h, Math.ceil(cy + radius))
  let delta = 0
  for (let y = y1; y < y2; y++) {
    const row = y * w
    const dy = y - cy
    const dy2 = dy * dy
    for (let x = x1; x < x2; x++) {
      const dx = x - cx
      if (dx * dx + dy2 > rr) continue
      const i = row + x
      if (on) { if (data[i] === 0) { data[i] = MASK_ON; delta++ } }
      else if (data[i] !== 0) { data[i] = 0; delta-- }
    }
  }
  return { x1, y1, x2, y2, delta }
}

/** 두 점 사이를 반경 r 의 원으로 잇는다(간격 = 반경의 30%). */
export function strokeLine(
  mask: MaskBuffer, x0: number, y0: number, x1: number, y1: number, r: number, on: boolean,
): MaskEdit {
  const dist = Math.hypot(x1 - x0, y1 - y0)
  const steps = Math.max(1, Math.ceil(dist / Math.max(1, r * 0.3)))
  const edit = blankEdit()
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    mergeEdit(edit, stampCircle(mask, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, on))
  }
  return edit
}

/** 사각형 [x1,x2)×[y1,y2) 를 켜거나 끈다. 좌표는 반올림한 뒤 이미지 안으로 자른다. */
export function fillRect(mask: MaskBuffer, x1: number, y1: number, x2: number, y2: number, on: boolean): MaskEdit {
  const { data, w, h } = mask
  const cx1 = Math.max(0, Math.round(x1)), cx2 = Math.min(w, Math.round(x2))
  const cy1 = Math.max(0, Math.round(y1)), cy2 = Math.min(h, Math.round(y2))
  let delta = 0
  for (let y = cy1; y < cy2; y++) {
    const row = y * w
    for (let x = cx1; x < cx2; x++) {
      const i = row + x
      if (on) { if (data[i] === 0) { data[i] = MASK_ON; delta++ } }
      else if (data[i] !== 0) { data[i] = 0; delta-- }
    }
  }
  return { x1: cx1, y1: cy1, x2: cx2, y2: cy2, delta }
}

/** 다각형의 정수 경계 상자(이미지 안으로 자름, x2/y2 배타). */
export function polygonBBox(pts: readonly Point[], w: number, h: number) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const p of pts) {
    if (p.x < minX) minX = p.x
    if (p.y < minY) minY = p.y
    if (p.x > maxX) maxX = p.x
    if (p.y > maxY) maxY = p.y
  }
  return {
    x1: Math.max(0, Math.floor(minX)), y1: Math.max(0, Math.floor(minY)),
    x2: Math.min(w, Math.ceil(maxX)), y2: Math.min(h, Math.ceil(maxY)),
  }
}

/** 짝홀 규칙 point-in-polygon — 픽셀 (x, y) 격자점 기준. fillPolygon 의 기준(golden) 구현. */
export function pointInPolygon(x: number, y: number, poly: readonly Point[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/**
 * 다각형 내부를 켜거나 끈다 — 스캔라인 짝홀 채우기.
 *
 * 행마다 변과의 교차 x 를 모아 정렬하고 [ceil(c₂ₖ), ceil(c₂ₖ₊₁)) 구간만 칠한다. 교차 x 를
 * pointInPolygon 과 **같은 식·같은 연산 순서**로 구해서 결과가 픽셀 단위로 똑같다(vitest golden).
 * 비용이 O(면적 × 꼭짓점)에서 O(행 × 꼭짓점 + 칠한 면적)으로 준다 — 750점 올가미가 2.6초 → 수 ms.
 */
export function fillPolygon(mask: MaskBuffer, pts: readonly Point[], on: boolean): MaskEdit {
  const { data, w, h } = mask
  const box = polygonBBox(pts, w, h)
  const n = pts.length
  let delta = 0
  if (n < 3) return { ...box, delta }
  const xs = new Float64Array(n)
  for (let y = box.y1; y < box.y2; y++) {
    let k = 0
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = pts[i].x, yi = pts[i].y, xj = pts[j].x, yj = pts[j].y
      if ((yi > y) !== (yj > y)) xs[k++] = (xj - xi) * (y - yi) / (yj - yi) + xi
    }
    if (k < 2) continue
    const hits = xs.subarray(0, k)
    hits.sort()
    const row = y * w
    for (let p = 0; p + 1 < k; p += 2) {
      const from = Math.max(box.x1, Math.ceil(hits[p]))
      const to = Math.min(box.x2, Math.ceil(hits[p + 1]))
      for (let x = from; x < to; x++) {
        const i = row + x
        if (on) { if (data[i] === 0) { data[i] = MASK_ON; delta++ } }
        else if (data[i] !== 0) { data[i] = 0; delta-- }
      }
    }
  }
  return { ...box, delta }
}

/** 켜진 픽셀 수(전체 스캔). undo/redo 로 버퍼를 통째로 갈아 끼운 뒤에만 쓴다. */
export function countMask(data: Uint8Array): number {
  let count = 0
  for (let i = 0; i < data.length; i++) if (data[i] !== 0) count++
  return count
}

// ── 올가미 점 ──────────────────────────────────────────────────────────────

/** 직전 점과 이만큼 이상 떨어진 점만 쌓는다(px). 제자리 포인터 이벤트가 꼭짓점을 불리지 않게. */
export const LASSO_MIN_STEP = 1

/** 올가미 점을 쌓는다. 직전 점과 `minStep` 미만이면 버리고 false. */
export function appendLassoPoint(points: Point[], p: Point, minStep: number = LASSO_MIN_STEP): boolean {
  const last = points[points.length - 1]
  if (last && Math.hypot(p.x - last.x, p.y - last.y) < minStep) return false
  points.push({ x: p.x, y: p.y })
  return true
}

// ── undo 정책 ──────────────────────────────────────────────────────────────

/** 이 크기(px) 이하의 사각형 드래그는 '클릭'으로 보고 적용하지 않는다. */
export const MIN_RECT_PX = 3

/**
 * 누르는 순간 마스크가 바뀌는 도구만 pointerdown 에서 undo 스냅숏을 뜬다.
 * 사각형·올가미는 실제로 적용될 때(뗄 때) 뜬다 — 빈 클릭이 빈 undo 단계를 쌓거나 redo 를 날리지 않게.
 */
export function snapshotsOnPress(tool: string, eraserMode: string, eraserRestore = false): boolean {
  return tool === 'brush' || tool === 'stamp'
    || (tool === 'eraser' && (eraserRestore || eraserMode === 'brush'))
}

/** 드래그 두 점 → 반올림한 정규 사각형. */
export function dragRect(ax: number, ay: number, bx: number, by: number) {
  return {
    x1: Math.round(Math.min(ax, bx)), y1: Math.round(Math.min(ay, by)),
    x2: Math.round(Math.max(ax, bx)), y2: Math.round(Math.max(ay, by)),
  }
}

/** 적용할 만큼 큰 사각형인지(가로·세로 모두 MIN_RECT_PX 초과). */
export function rectIsApplicable(rect: { x1: number; y1: number; x2: number; y2: number }): boolean {
  return rect.x2 - rect.x1 > MIN_RECT_PX && rect.y2 - rect.y1 > MIN_RECT_PX
}

/** 마스크 undo/redo — 전체 스냅숏, 최대 `max` 단계. 크기가 다른(이미지가 바뀐 뒤) 스냅숏은 버린다. */
export class MaskHistory {
  private undoStack: Uint8Array[] = []
  private redoStack: Uint8Array[] = []

  constructor(readonly max = 10) {}

  get undoCount(): number { return this.undoStack.length }
  get redoCount(): number { return this.redoStack.length }

  /** 지금 상태를 undo 에 쌓는다. 새 편집이므로 redo 는 비운다. */
  save(current: Uint8Array): void {
    this.undoStack.push(new Uint8Array(current))
    while (this.undoStack.length > this.max) this.undoStack.shift()
    this.redoStack = []
  }

  /** 한 단계 되돌린다. 되돌렸으면 true. 크기가 안 맞는 스냅숏을 만나면 기록을 비우고 false. */
  undo(current: Uint8Array): boolean {
    const snapshot = this.undoStack.pop()
    if (!snapshot) return false
    if (snapshot.length !== current.length) { this.clear(); return false }
    this.redoStack.push(new Uint8Array(current))
    current.set(snapshot)
    return true
  }

  /** 한 단계 다시 한다. 다시 했으면 true. 크기가 안 맞으면 redo 만 비우고 false. */
  redo(current: Uint8Array): boolean {
    const snapshot = this.redoStack.pop()
    if (!snapshot) return false
    if (snapshot.length !== current.length) { this.redoStack = []; return false }
    this.undoStack.push(new Uint8Array(current))
    current.set(snapshot)
    return true
  }

  clear(): void { this.undoStack = []; this.redoStack = [] }
}

// ── 표시·인코딩 ─────────────────────────────────────────────────────────────

/** RGBA 바이트를 리틀 엔디언 32비트 한 칸으로 (ImageData 의 Uint32Array 뷰에 바로 쓴다). */
export function rgba32(r: number, g: number, b: number, a: number): number {
  return ((a & 255) << 24 | (b & 255) << 16 | (g & 255) << 8 | (r & 255)) >>> 0
}

/** 마스크의 [x1,x2)×[y1,y2) 를 오버레이 픽셀로 옮긴다 — 켜진 곳은 `color`, 꺼진 곳은 투명. */
export function paintMaskOverlay(
  pixels: Uint32Array, data: Uint8Array, w: number,
  x1: number, y1: number, x2: number, y2: number, color: number,
): void {
  for (let y = y1; y < y2; y++) {
    const row = y * w
    for (let x = x1; x < x2; x++) {
      const i = row + x
      pixels[i] = data[i] > 0 ? color : 0
    }
  }
}

/** 마스크 → 흑백 불투명 픽셀(켜짐 흰색, 꺼짐 검정). 백엔드로 보내는 PNG 의 픽셀. */
export function maskToGrayPixels(data: Uint8Array, out: Uint32Array): Uint32Array {
  const n = Math.min(data.length, out.length)
  for (let i = 0; i < n; i++) out[i] = data[i] > 0 ? 0xffffffff : 0xff000000
  return out
}

/** 마스크를 흑백 PNG data URL 로 (브라우저 전용 — 캔버스로 인코딩). */
export function encodeMaskPng(data: Uint8Array, w: number, h: number): string {
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')!
  const image = ctx.createImageData(w, h)
  maskToGrayPixels(data, new Uint32Array(image.data.buffer))
  ctx.putImageData(image, 0, 0)
  return canvas.toDataURL('image/png')
}
