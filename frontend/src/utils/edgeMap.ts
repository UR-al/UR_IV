/**
 * 자석 올가미 엣지맵 — EditorView/EditorCanvas 와 InpaintView 가 같이 쓴다.
 *
 * 백엔드 getEdgeMap(Canny, 동기 슬롯)이 흑백 PNG data URL 을 준다. 여기서
 *   - 한 장짜리 캐시(같은 이미지에서 '자석'을 다시 켜도 Canny 를 다시 돌리지 않는다),
 *   - PNG → 흑백 배열 풀기, 커서 근처 엣지로 붙이기(snap)
 * 를 맡는다. 예전 인페인트는 캐시가 없어 토글마다 GUI 스레드에서 Canny 를 다시 돌렸고,
 * 새 이미지를 열어도 옛 엣지맵이 남아 이전 이미지의 윤곽에 달라붙었다.
 */
import { getBackend } from '../bridge.js'
import type { Point } from './maskOps'

export interface EdgeMap { data: Uint8Array; w: number; h: number }

/** 프론트가 쓰는 Canny 임계값 (core/edge_map.py 기본값과 같다) */
export const EDGE_CANNY_LOW = 50
export const EDGE_CANNY_HIGH = 150
/** 자석이 붙는 반경(px) 기본값 */
export const EDGE_SNAP_RADIUS = 12

/** 디코드한 RGBA 픽셀 → 엣지 배열(R 채널). */
export function edgeMapFromRgba(rgba: ArrayLike<number>, w: number, h: number): EdgeMap {
  const data = new Uint8Array(w * h)
  for (let i = 0; i < data.length; i++) data[i] = rgba[i * 4]
  return { data, w, h }
}

/** 반경 r 안에서 가장 가까운 엣지 픽셀(>127)로 붙인다. 엣지맵이 없거나 근처에 엣지가 없으면 그대로. */
export function snapToEdge(edge: EdgeMap | null, x: number, y: number, r: number = EDGE_SNAP_RADIUS): Point {
  if (!edge) return { x, y }
  const { data, w, h } = edge
  let best = Infinity, bx = x, by = y
  const x0 = Math.max(0, Math.floor(x - r)), y0 = Math.max(0, Math.floor(y - r))
  const x1 = Math.min(w, Math.ceil(x + r)), y1 = Math.min(h, Math.ceil(y + r))
  for (let py = y0; py < y1; py++) {
    const row = py * w
    for (let px = x0; px < x1; px++) {
      if (data[row + px] > 127) {
        const d = (px - x) ** 2 + (py - y) ** 2
        if (d < best) { best = d; bx = px; by = py }
      }
    }
  }
  return { x: bx, y: by }
}

/** PNG data URL → 엣지 배열 (브라우저 전용). 실패하면 null. */
export function decodeEdgeMap(b64: string): Promise<EdgeMap | null> {
  return new Promise(resolve => {
    if (!b64) { resolve(null); return }
    const img = new Image()
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d', { willReadFrequently: true })!
        ctx.drawImage(img, 0, 0)
        const id = ctx.getImageData(0, 0, canvas.width, canvas.height)
        resolve(edgeMapFromRgba(id.data, canvas.width, canvas.height))
      } catch { resolve(null) }
    }
    img.onerror = () => resolve(null)
    img.src = b64
  })
}

/** 백엔드 getEdgeMap 호출 → PNG data URL (없거나 실패하면 ''). */
export async function fetchEdgeMapFromBackend(path: string): Promise<string> {
  const backend: any = await getBackend()
  if (!path || typeof backend?.getEdgeMap !== 'function') return ''
  return new Promise(resolve => {
    backend.getEdgeMap(path, EDGE_CANNY_LOW, EDGE_CANNY_HIGH, (b64: string) => resolve(b64 || ''))
  })
}

/**
 * 이미지 경로 한 장짜리 엣지맵 캐시. 같은 경로를 동시에 두 번 요청하면 한 번만 부른다.
 * `clear()` 뒤에 도착한 옛 응답은 캐시에 넣지 않는다(이미지가 바뀐 뒤의 늦은 응답).
 */
export class EdgeMapCache {
  private path = ''
  private b64 = ''
  private pending: { path: string; promise: Promise<string> } | null = null
  private generation = 0

  constructor(private readonly fetcher: (path: string) => Promise<string> = fetchEdgeMapFromBackend) {}

  /** 캐시에 있으면 그 data URL, 없으면 '' (요청하지 않는다). */
  peek(path: string): string {
    return path && path === this.path ? this.b64 : ''
  }

  get(path: string): Promise<string> {
    if (!path) return Promise.resolve('')
    const hit = this.peek(path)
    if (hit) return Promise.resolve(hit)
    if (this.pending && this.pending.path === path) return this.pending.promise
    const gen = this.generation
    const promise = this.fetcher(path).then(
      (b64) => {
        if (this.pending?.promise === promise) this.pending = null
        if (b64 && gen === this.generation) { this.path = path; this.b64 = b64 }
        return b64 || ''
      },
      () => {
        if (this.pending?.promise === promise) this.pending = null
        return ''
      },
    )
    this.pending = { path, promise }
    return promise
  }

  clear(): void {
    this.generation++
    this.path = ''
    this.b64 = ''
    this.pending = null
  }
}
