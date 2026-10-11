/**
 * 참조 색 맞추기 — 패널 설정, 저장값, 그리고 '무엇을 어떻게 보낼지'(colorMatchPlan).
 *
 * 백엔드 `core/editor_color_match.run_color_match` 의 페이로드:
 *   { mode: 'reference'|'ring', strength: 1..100, reference_path?, ring_px? } (+ mask_base64)
 * - 참조 이미지 모드: '칠한 영역만' 이면 마스크를 실어 칠한 곳만, 아니면 이미지 전체를 바꾼다.
 * - 마스크 주변 모드: 마스크가 꼭 있어야 한다(칠한 영역 안쪽 띠를 바로 바깥 띠에 맞춘다).
 * 참조 이미지 경로는 세션에만 둔다(저장하지 않는다) — 나머지 설정만 localStorage 에 남긴다.
 */

export type ColorMatchMode = 'reference' | 'ring'

export interface ColorMatchConfig {
  mode: ColorMatchMode
  /** 1~100 % */
  strength: number
  /** 마스크 주변 모드의 띠 폭(원본 px) */
  ringPx: number
  /** 참조 이미지 모드에서 칠한 영역만 바꿀지 */
  maskOnly: boolean
}

export const STRENGTH_MIN = 1
export const STRENGTH_MAX = 100
export const RING_PX_MIN = 4
export const RING_PX_MAX = 64

export const COLOR_MATCH_DEFAULTS: Readonly<ColorMatchConfig> = Object.freeze({
  mode: 'reference' as ColorMatchMode,
  strength: 100,
  ringPx: 16,
  maskOnly: false,
})

export const COLOR_MATCH_STORAGE_KEY = 'editorColorMatch'

/** 프론트에서 먼저 거르는 경우의 문구 — 백엔드(core/editor_color_match)의 같은 경우와 같은 말이다. */
export const COLOR_MATCH_MESSAGES = Object.freeze({
  noReference: '참조 이미지를 먼저 불러오세요',
  ringNeedsMask: '마스크 주변 모드는 먼저 영역을 칠하세요',
  needMask: '먼저 브러시·올가미로 영역을 칠하세요',
})

export interface ColorMatchParams {
  mode: ColorMatchMode
  strength: number
  reference_path?: string
  ring_px?: number
}

export type ColorMatchPlan =
  | { ok: true; withMask: boolean; params: ColorMatchParams }
  | { ok: false; message: string }

function clampInt(value: unknown, lo: number, hi: number, fallback: number): number {
  const n = typeof value === 'number' ? value : NaN
  if (!Number.isFinite(n)) return fallback
  return Math.max(lo, Math.min(hi, Math.round(n)))
}

export function clampColorMatch(raw: unknown): ColorMatchConfig {
  const src = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  return {
    mode: src.mode === 'ring' ? 'ring' : 'reference',
    strength: clampInt(src.strength, STRENGTH_MIN, STRENGTH_MAX, COLOR_MATCH_DEFAULTS.strength),
    ringPx: clampInt(src.ringPx, RING_PX_MIN, RING_PX_MAX, COLOR_MATCH_DEFAULTS.ringPx),
    maskOnly: typeof src.maskOnly === 'boolean' ? src.maskOnly : COLOR_MATCH_DEFAULTS.maskOnly,
  }
}

/** 이 설정에 마스크가 필요한가 — 패널의 '먼저 칠하세요' 안내와 같은 판정. */
export function colorMatchNeedsMask(cfg: ColorMatchConfig): boolean {
  return cfg.mode === 'ring' || cfg.maskOnly
}

/**
 * 보낼 페이로드와 마스크 동봉 여부. 보낼 수 없으면 이유(사용자 문구).
 * hasMask 는 지금 캔버스에 칠한 영역이 있는지 — 없으면 보내 봐야 백엔드가 같은 이유로 거절한다.
 */
export function colorMatchPlan(cfg: ColorMatchConfig, referencePath: string, hasMask: boolean): ColorMatchPlan {
  const c = clampColorMatch(cfg)
  if (c.mode === 'ring') {
    if (!hasMask) return { ok: false, message: COLOR_MATCH_MESSAGES.ringNeedsMask }
    return { ok: true, withMask: true, params: { mode: 'ring', strength: c.strength, ring_px: c.ringPx } }
  }
  const path = String(referencePath || '').trim()
  if (!path) return { ok: false, message: COLOR_MATCH_MESSAGES.noReference }
  if (c.maskOnly && !hasMask) return { ok: false, message: COLOR_MATCH_MESSAGES.needMask }
  return { ok: true, withMask: c.maskOnly, params: { mode: 'reference', strength: c.strength, reference_path: path } }
}

/** 불러온 참조 이미지 이름 표시. */
export function referenceLabel(path: string): string {
  const p = String(path || '').replace(/\\/g, '/')
  if (!p) return '불러온 이미지 없음'
  return p.substring(p.lastIndexOf('/') + 1) || p
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

export function loadColorMatchPrefs(storage: StorageLike | null = browserStorage()): ColorMatchConfig {
  try {
    const text = storage?.getItem(COLOR_MATCH_STORAGE_KEY)
    return clampColorMatch(text ? JSON.parse(text) : null)
  } catch {
    return { ...COLOR_MATCH_DEFAULTS }
  }
}

export function saveColorMatchPrefs(cfg: ColorMatchConfig, storage: StorageLike | null = browserStorage()): void {
  try {
    storage?.setItem(COLOR_MATCH_STORAGE_KEY, JSON.stringify(clampColorMatch(cfg)))
  } catch {
    // 저장소가 막혀 있으면 이번 세션 값만 쓴다
  }
}
