/**
 * 그리기 도구의 순수 계산 — DOM 의존 없음.
 *
 * PyQt 판은 `widgets/interactive_label.py`(은퇴해 삭제됨)에서 `display_base_image` 를 **직접** 고쳤다.
 * 되돌리려면 이미지 전체 스냅샷을 쌓아야 했고, 투명도 있는 펜은 획이 겹치는 관절마다
 * 진해졌다. Vue 판은 별도 레이어에 그리고 병합할 때 한 번만 합성한다.
 */

export interface FloodFillResult {
  /** 채워질 픽셀이면 1 — 이미지와 같은 크기 */
  mask: Uint8Array
  /** 채워진 영역의 경계 상자 (both inclusive) */
  minX: number
  minY: number
  maxX: number
  maxY: number
  count: number
}

/** `#rrggbb` → [r, g, b]. 못 읽으면 검정. */
export function hexToRgb(hex: string): [number, number, number] {
  const match = /^#?([0-9a-f]{6})$/i.exec(String(hex || '').trim())
  if (!match) return [0, 0, 0]
  const value = parseInt(match[1], 16)
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}

export function rgbToHex(r: number, g: number, b: number): string {
  const clamp = (v: number) => Math.max(0, Math.min(255, Math.round(v)))
  return `#${((1 << 24) | (clamp(r) << 16) | (clamp(g) << 8) | clamp(b)).toString(16).slice(1)}`
}

/**
 * 스캔라인 플러드 필. 씨앗 픽셀 색과의 채널별 차이가 `tolerance` 이내인 4-연결 영역.
 *
 * 이웃과 비교하지 않고 **씨앗과** 비교한다 (OpenCV 의 `FLOODFILL_FIXED_RANGE`).
 * 이웃 비교는 완만한 그라디언트를 타고 이미지 전체로 번진다.
 *
 * 재귀 대신 스택 + 스캔라인이다. 큰 이미지에서 픽셀마다 재귀하면 스택이 터진다.
 */
export function floodFillMask(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  seedX: number,
  seedY: number,
  tolerance = 20,
): FloodFillResult | null {
  const x0 = Math.round(seedX)
  const y0 = Math.round(seedY)
  if (x0 < 0 || y0 < 0 || x0 >= width || y0 >= height) return null

  const seed = (y0 * width + x0) * 4
  const sr = data[seed]
  const sg = data[seed + 1]
  const sb = data[seed + 2]
  const sa = data[seed + 3]

  const mask = new Uint8Array(width * height)
  const stack: number[] = [x0, y0]
  let minX = x0
  let minY = y0
  let maxX = x0
  let maxY = y0
  let count = 0

  const matches = (index: number): boolean => {
    if (mask[index]) return false
    const p = index * 4
    return (
      Math.abs(data[p] - sr) <= tolerance &&
      Math.abs(data[p + 1] - sg) <= tolerance &&
      Math.abs(data[p + 2] - sb) <= tolerance &&
      Math.abs(data[p + 3] - sa) <= tolerance
    )
  }

  while (stack.length) {
    const y = stack.pop() as number
    let x = stack.pop() as number
    let index = y * width + x

    // 이 줄의 왼쪽 끝까지 후퇴
    while (x > 0 && matches(index - 1)) {
      x--
      index--
    }

    let spanAbove = false
    let spanBelow = false
    while (x < width && matches(index)) {
      mask[index] = 1
      count++
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y

      if (y > 0) {
        const above = index - width
        const ok = matches(above)
        if (ok && !spanAbove) {
          stack.push(x, y - 1)
          spanAbove = true
        } else if (!ok) {
          spanAbove = false
        }
      }
      if (y < height - 1) {
        const below = index + width
        const ok = matches(below)
        if (ok && !spanBelow) {
          stack.push(x, y + 1)
          spanBelow = true
        } else if (!ok) {
          spanBelow = false
        }
      }
      x++
      index++
    }
  }

  return count ? { mask, minX, minY, maxX, maxY, count } : null
}

/**
 * 두 점을 잇는 벡터를 따라가는 선형 그라디언트의 캔버스 좌표.
 *
 * 드래그가 한 점에 머물면(길이 0) 그라디언트를 만들 수 없다 — 그때는 null.
 */
export function gradientLine(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): { x1: number; y1: number; x2: number; y2: number } | null {
  if (Math.hypot(x2 - x1, y2 - y1) < 1) return null
  return { x1, y1, x2, y2 }
}

/** 사각형/타원 도구가 쓰는 정규화된 경계 상자 (드래그 방향과 무관하게). */
export function normalizedRect(x1: number, y1: number, x2: number, y2: number) {
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    w: Math.abs(x2 - x1),
    h: Math.abs(y2 - y1),
  }
}

/** 레이어에 그리는 도구인지 — 마스크/선택 도구와 포인터 처리를 갈라야 한다. */
export const DRAW_TOOLS = [
  'pen',
  'line',
  'rect',
  'ellipse',
  'fill',
  'eyedropper',
  'clone_stamp',
  'text_overlay',
  'gradient',
  'heal',
] as const

export type DrawToolName = (typeof DRAW_TOOLS)[number]

export function isDrawTool(tool: string | undefined): tool is DrawToolName {
  return !!tool && (DRAW_TOOLS as readonly string[]).includes(tool)
}
