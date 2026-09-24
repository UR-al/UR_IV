/**
 * UI 크기(Chromium zoom) — 설정 값 정규화. 0.7~2.0 밖이거나 숫자가 아니면 1.0.
 * 부팅 캐시(localStorage 'ui.scale') · Settings 슬라이더 이벤트 · ui_prefs.uiScale 이 모두 여기를 거친다.
 */
export function normalizeUiScale(value: unknown): number {
  const n = parseFloat(value as any)
  return (!isNaN(n) && n >= 0.7 && n <= 2.0) ? n : 1.0
}
