/**
 * 생성 진행 표시 — 진행률 · 남은 시간 · 라이브 프리뷰 URL (App.vue 의 generationProgress /
 * generationPreview 핸들러에서 추출).
 */

/** step/total → 0~100 정수 퍼센트 */
export function progressPercent(step: number, total: number): number {
  return Math.round(step / total * 100)
}

/**
 * 간단 ETA — 지금까지 걸린 시간을 스텝 수로 나눠 남은 스텝에 곱한다.
 * 첫 스텝 전(step 0)에는 알 수 없어 null. 60초 이상이면 `ETA 2m05s`, 아니면 `ETA 42s`.
 */
export function formatEta(elapsedSec: number, step: number, total: number): string | null {
  if (!(step > 0)) return null
  const remaining = Math.max(0, elapsedSec / step * (total - step))
  return remaining >= 60
    ? `ETA ${Math.floor(remaining / 60)}m${String(Math.floor(remaining % 60)).padStart(2, '0')}s`
    : `ETA ${remaining.toFixed(0)}s`
}

/**
 * Forge live preview(base64) → data URL. 머리 바이트로 형식을 고른다(JPEG `/9j/` · WebP `UklGR`,
 * 그 밖은 PNG). 문자열이 아니거나 비었으면 null.
 */
export function previewDataUrl(b64: unknown): string | null {
  if (typeof b64 !== 'string' || !b64) return null
  const mime = b64.startsWith('/9j/') ? 'image/jpeg' : b64.startsWith('UklGR') ? 'image/webp' : 'image/png'
  return `data:${mime};base64,${b64}`
}
