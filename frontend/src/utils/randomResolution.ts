/** 랜덤 해상도 목록의 한 항목 — [폭, 높이, 표시 이름] (Python set_random_resolutions 와 같은 모양) */
export type RandomResEntry = [number, number, string]

export const DEFAULT_RANDOM_RES_W = 832
export const DEFAULT_RANDOM_RES_H = 1216
/** 한 변의 최소 크기 — 이보다 작으면 추가하지 않는다 */
export const MIN_RANDOM_RES_SIDE = 256

/**
 * 입력한 폭·높이 → 목록에 넣을 항목. 비었거나 0 이면 기본값(832×1216), 8 의 배수로 반올림하고,
 * 한 변이라도 256 보다 작으면 null(추가하지 않음).
 */
export function normalizeRandomRes(width: number | null | undefined, height: number | null | undefined): RandomResEntry | null {
  const w = Math.round((width || DEFAULT_RANDOM_RES_W) / 8) * 8
  const h = Math.round((height || DEFAULT_RANDOM_RES_H) / 8) * 8
  if (w < MIN_RANDOM_RES_SIDE || h < MIN_RANDOM_RES_SIDE) return null
  return [w, h, `${w}x${h}`]
}
