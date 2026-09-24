/**
 * 영역 이동 미리보기의 픽셀 계산 — DOM 의존 없음.
 *
 * 예전 `renderMovePreview` 는 pointermove 마다 이미지 크기 ImageData 를 새로 할당하고
 * 전 화소를 두 번 돌고 전체를 putImageData 했다(4MP 면 프레임당 16~33MB 할당).
 * 그런데 드래그 중에는 마스크도 스냅숏도 바뀌지 않는다. 그래서
 *
 *   1. 이동 시작 때 '구멍 뚫린 배경'(bg)과 재사용할 출력 버퍼를 한 번만 만든다.
 *   2. 매 프레임에는 직전 조각 자리만 bg 로 되돌리고, 새 자리에 조각을 대입한다.
 *   3. 바뀐 두 사각형만 캔버스에 올린다.
 *
 * 조각은 **대입**이다(합성하지 않는다) — 반투명 PNG 에서도 예전 결과와 같다.
 * 버퍼는 32비트 픽셀 뷰(Uint32Array)로 다룬다: 픽셀당 대입 한 번.
 */

/** 반열림 사각형 [x, x+w) × [y, y+h) */
export interface PixelRect { x: number; y: number; w: number; h: number }

/** 리틀엔디언 RGBA 패킹 — 구멍 색(불투명 검정/흰색) */
export const HOLE_BLACK = 0xff000000 >>> 0
export const HOLE_WHITE = 0xffffffff >>> 0

export function holeColorFor(fill: string | undefined): number {
  return String(fill || '').toLowerCase() === 'white' ? HOLE_WHITE : HOLE_BLACK
}

/** 두 사각형을 이미지 안으로 자른다. 비면 null. */
export function clipRect(r: PixelRect, w: number, h: number): PixelRect | null {
  const x1 = Math.max(0, r.x)
  const y1 = Math.max(0, r.y)
  const x2 = Math.min(w, r.x + r.w)
  const y2 = Math.min(h, r.y + r.h)
  if (x2 <= x1 || y2 <= y1) return null
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 }
}

/**
 * 원래 자리를 구멍으로 비운 배경. 마스크 경계 상자 안만 보면 된다(밖은 스냅숏 그대로).
 */
export function buildMoveBackground(
  src: Uint32Array, mask: Uint8Array, w: number, h: number,
  bbox: PixelRect, hole: number = HOLE_BLACK,
): Uint32Array {
  const bg = new Uint32Array(src)
  const box = clipRect(bbox, w, h)
  if (!box) return bg
  for (let y = box.y; y < box.y + box.h; y++) {
    const row = y * w
    for (let x = box.x; x < box.x + box.w; x++) {
      if (mask[row + x] > 0) bg[row + x] = hole
    }
  }
  return bg
}

function copyRect(dst: Uint32Array, src: Uint32Array, w: number, r: PixelRect) {
  for (let y = r.y; y < r.y + r.h; y++) {
    const start = y * w + r.x
    dst.set(src.subarray(start, start + r.w), start)
  }
}

export interface MoveFrame {
  /** 이번 프레임에 조각이 놓인 자리(이미지 안으로 자른 것). 다음 프레임의 prev 로 넘긴다. */
  dest: PixelRect | null
  /** 캔버스에 다시 올려야 하는 사각형들 */
  dirty: PixelRect[]
}

/**
 * 한 프레임 합성. `out` 은 직전 프레임 결과(첫 프레임이면 bg 사본)를 그대로 들고 온다.
 *
 * 불변식: 조각 자리(prevDest) 밖의 `out` 은 항상 `bg` 와 같다. 그래서 직전 자리만 되돌리면
 * 전체를 다시 계산한 것과 같은 결과가 된다.
 */
export function composeMoveFrame(
  out: Uint32Array, bg: Uint32Array, src: Uint32Array, mask: Uint8Array,
  w: number, h: number, bbox: PixelRect, prevDest: PixelRect | null,
  dx: number, dy: number,
): MoveFrame {
  const dirty: PixelRect[] = []
  if (prevDest) {
    copyRect(out, bg, w, prevDest)
    dirty.push(prevDest)
  }
  const box = clipRect(bbox, w, h)
  if (!box) return { dest: null, dirty }
  const dest = clipRect({ x: box.x + dx, y: box.y + dy, w: box.w, h: box.h }, w, h)
  if (!dest) return { dest: null, dirty }
  // 조각 대입 — 목적지가 이미지 안인 원본 좌표만 돈다
  const sx1 = dest.x - dx
  const sy1 = dest.y - dy
  for (let y = sy1; y < sy1 + dest.h; y++) {
    const srcRow = y * w
    const dstRow = (y + dy) * w + dx
    for (let x = sx1; x < sx1 + dest.w; x++) {
      if (mask[srcRow + x] > 0) out[dstRow + x] = src[srcRow + x]
    }
  }
  dirty.push(dest)
  return { dest, dirty }
}
