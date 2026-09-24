import { describe, expect, it } from 'vitest'
import {
  floodFillMask,
  gradientLine,
  hexToRgb,
  isDrawTool,
  normalizedRect,
  rgbToHex,
} from './drawTools'

/** `w×h` 격자를 만든다. `colorAt(x, y)` 가 `[r,g,b]` 를 준다. */
function grid(w: number, h: number, colorAt: (x: number, y: number) => [number, number, number]) {
  const data = new Uint8ClampedArray(w * h * 4)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = colorAt(x, y)
      const i = (y * w + x) * 4
      data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255
    }
  }
  return data
}

describe('hexToRgb / rgbToHex', () => {
  it('왕복해도 같다', () => {
    expect(hexToRgb('#12B886')).toEqual([18, 184, 134])
    expect(rgbToHex(18, 184, 134)).toBe('#12b886')
  })

  it('# 없이도, 대소문자 섞여도 읽는다', () => {
    expect(hexToRgb('ff8800')).toEqual([255, 136, 0])
    expect(hexToRgb('#FF8800')).toEqual([255, 136, 0])
  })

  it('못 읽으면 검정 — 던지지 않는다', () => {
    expect(hexToRgb('nope')).toEqual([0, 0, 0])
    expect(hexToRgb('')).toEqual([0, 0, 0])
    expect(hexToRgb('#abc')).toEqual([0, 0, 0])
  })

  it('범위를 벗어난 값은 자른다', () => {
    expect(rgbToHex(-5, 300, 128.6)).toBe('#00ff81')
  })
})

describe('floodFillMask', () => {
  it('같은 색 영역만 채운다', () => {
    // 왼쪽 절반 흰색, 오른쪽 절반 검정
    const data = grid(8, 4, (x) => (x < 4 ? [255, 255, 255] : [0, 0, 0]))
    const out = floodFillMask(data, 8, 4, 1, 1)!
    expect(out.count).toBe(16)
    expect(out.minX).toBe(0)
    expect(out.maxX).toBe(3)
    expect(out.mask[0 * 8 + 4]).toBe(0)
  })

  it('씨앗과 비교한다 — 완만한 그라디언트를 타고 번지지 않는다', () => {
    // 한 줄이 0,10,20,... 로 서서히 밝아진다. 이웃끼리 비교하면 전부 이어져
    // 이미지 전체가 칠해진다(OpenCV 의 FLOODFILL_FIXED_RANGE 가 막는 것과 같은 문제).
    const data = grid(20, 1, (x) => [x * 10, x * 10, x * 10])
    const out = floodFillMask(data, 20, 1, 0, 0, 20)!
    expect(out.count).toBeLessThan(20)
    expect(out.maxX).toBe(2)   // 0,10,20 까지만 (30 은 씨앗 0 과 30 차이)
  })

  it('4-연결이다 — 대각선으로는 새지 않는다', () => {
    // 체커보드: 대각선끼리만 같은 색
    const data = grid(4, 4, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]))
    const out = floodFillMask(data, 4, 4, 0, 0)!
    expect(out.count).toBe(1)
  })

  it('막힌 구멍은 채우지 않는다', () => {
    // 가운데 3×3 이 테두리로 둘러싸인 형태
    const data = grid(5, 5, (x, y) => {
      const inner = x >= 1 && x <= 3 && y >= 1 && y <= 3
      return inner ? [0, 0, 0] : [255, 255, 255]
    })
    const inside = floodFillMask(data, 5, 5, 2, 2)!
    expect(inside.count).toBe(9)
    const outside = floodFillMask(data, 5, 5, 0, 0)!
    expect(outside.count).toBe(16)
  })

  it('경계 상자가 채운 영역과 맞는다', () => {
    const data = grid(10, 10, (x, y) =>
      (x >= 3 && x <= 6 && y >= 2 && y <= 5 ? [10, 10, 10] : [200, 200, 200]))
    const out = floodFillMask(data, 10, 10, 4, 3)!
    expect([out.minX, out.maxX, out.minY, out.maxY]).toEqual([3, 6, 2, 5])
    expect(out.count).toBe(16)
  })

  it('허용 오차를 넓히면 더 많이 잡는다', () => {
    const data = grid(6, 1, (x) => [x * 8, x * 8, x * 8])
    expect(floodFillMask(data, 6, 1, 0, 0, 0)!.count).toBe(1)
    expect(floodFillMask(data, 6, 1, 0, 0, 40)!.count).toBe(6)
  })

  it('바깥을 찍으면 null', () => {
    const data = grid(4, 4, () => [0, 0, 0])
    expect(floodFillMask(data, 4, 4, -1, 0)).toBeNull()
    expect(floodFillMask(data, 4, 4, 4, 0)).toBeNull()
  })

  it('큰 영역에서도 스택이 터지지 않는다 (재귀였다면 넘쳤다)', () => {
    const data = grid(400, 400, () => [128, 128, 128])
    expect(floodFillMask(data, 400, 400, 200, 200)!.count).toBe(160000)
  })
})

describe('normalizedRect', () => {
  it('드래그 방향과 무관하게 같은 사각형', () => {
    const a = normalizedRect(10, 20, 40, 60)
    const b = normalizedRect(40, 60, 10, 20)
    expect(a).toEqual(b)
    expect(a).toEqual({ x: 10, y: 20, w: 30, h: 40 })
  })
})

describe('gradientLine — 그라디언트 도구가 드래그 벡터를 그대로 쓴다', () => {
  it('드래그 방향 그대로의 시작·끝점', () => {
    expect(gradientLine(10, 20, 40, 60)).toEqual({ x1: 10, y1: 20, x2: 40, y2: 60 })
    expect(gradientLine(40, 60, 10, 20)).toEqual({ x1: 40, y1: 60, x2: 10, y2: 20 })
  })

  it('한 점에 머문 드래그(길이 1 미만)는 그라디언트를 만들지 않는다', () => {
    expect(gradientLine(5, 5, 5, 5)).toBeNull()
    expect(gradientLine(5, 5, 5.6, 5.6)).toBeNull()   // hypot ≈ 0.85
    expect(gradientLine(5, 5, 6, 5)).not.toBeNull()   // 정확히 1 은 그린다
  })
})

describe('isDrawTool — 포인터를 어느 쪽으로 보낼지 가른다', () => {
  it('그리기 도구만 참', () => {
    for (const id of ['pen', 'line', 'rect', 'ellipse', 'fill', 'eyedropper',
      'clone_stamp', 'text_overlay', 'gradient', 'heal']) {
      expect(isDrawTool(id)).toBe(true)
    }
  })

  it('마스크 도구와 빈 값은 거짓 — 아니면 포인터가 마스크 처리로 새어 나간다', () => {
    for (const id of ['box', 'lasso', 'brush', 'eraser', 'stamp', '', undefined]) {
      expect(isDrawTool(id)).toBe(false)
    }
  })
})
