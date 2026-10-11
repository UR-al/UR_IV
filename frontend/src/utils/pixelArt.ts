/**
 * 픽셀아트 필터 옵션 — ColorPanel 의 '픽셀 크기'·'색 수' 슬라이더와 그 저장값.
 *
 * 범위·기본값은 백엔드 `core/editor_ops.FILTER_OPTIONS['pixelart']` 와 같아야 한다 —
 * tests/test_editor_pixel_art.py 가 아래 상수를 정규식으로 읽어 대조한다(이름을 바꾸면 그쪽도).
 * 색 수 2 미만은 팔레트를 끈다(백엔드도 같다).
 */

export const PIXEL_SIZE_MIN = 2
export const PIXEL_SIZE_MAX = 64
export const PIXEL_SIZE_DEFAULT = 8
export const COLORS_MIN = 0
export const COLORS_MAX = 64
export const COLORS_DEFAULT = 16

export const PIXEL_ART_STORAGE_KEY = 'editorPixelArt'

/** 백엔드 filter 페이로드의 `options` 그대로(키 이름이 snake_case 인 이유). */
export interface PixelArtOptions {
  pixel_size: number
  colors: number
}

export const PIXEL_ART_DEFAULTS: Readonly<PixelArtOptions> = Object.freeze({
  pixel_size: PIXEL_SIZE_DEFAULT,
  colors: COLORS_DEFAULT,
})

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

function clampInt(value: unknown, lo: number, hi: number, fallback: number): number {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value) : NaN
  if (!Number.isFinite(n)) return fallback
  return Math.max(lo, Math.min(hi, Math.round(n)))
}

/** 아무 값 → 범위 안의 정수 옵션. 모르는 키는 버리고 빠진 키는 기본값. */
export function clampPixelArt(raw: unknown): PixelArtOptions {
  const src = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  return {
    pixel_size: clampInt(src.pixel_size, PIXEL_SIZE_MIN, PIXEL_SIZE_MAX, PIXEL_SIZE_DEFAULT),
    colors: clampInt(src.colors, COLORS_MIN, COLORS_MAX, COLORS_DEFAULT),
  }
}

/** 색 수 표시 — 팔레트를 끄는 값(2 미만)은 '끔'. */
export function colorsLabel(n: number): string {
  return n < 2 ? '끔' : String(n)
}

export function pixelSizeLabel(n: number): string {
  return `${n}px`
}

export function loadPixelArtPrefs(storage: StorageLike | null = browserStorage()): PixelArtOptions {
  try {
    const text = storage?.getItem(PIXEL_ART_STORAGE_KEY)
    return clampPixelArt(text ? JSON.parse(text) : null)
  } catch {
    return { ...PIXEL_ART_DEFAULTS }
  }
}

export function savePixelArtPrefs(opts: PixelArtOptions, storage: StorageLike | null = browserStorage()): void {
  try {
    storage?.setItem(PIXEL_ART_STORAGE_KEY, JSON.stringify(clampPixelArt(opts)))
  } catch {
    // 저장소가 막혀 있으면(사생활 모드 등) 이번 세션 값만 쓴다
  }
}
