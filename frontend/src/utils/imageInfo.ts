/**
 * 뷰어 하단 정보 줄(해상도 · 시드)에 쓰는 표시 문자열.
 *
 * imageGenerated 페이로드에 크기나 시드가 없으면(외부 워크플로 결과 · 목업 등) 예전에는
 * 템플릿 문자열이 그대로 'undefined × undefined', 'undefined' 를 찍었다. 값이 없으면 빈 문자열을
 * 돌려주고, 화면은 빈 값을 '—' 로 보여 준다(EMPTY_INFO).
 */

export const EMPTY_INFO = '—'

function positiveInt(value: unknown): number | null {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN
  return Number.isFinite(n) && n > 0 ? Math.round(n) : null
}

/** 가로·세로가 둘 다 양수일 때만 'W × H', 아니면 ''. */
export function formatResolution(width: unknown, height: unknown): string {
  const w = positiveInt(width)
  const h = positiveInt(height)
  return w && h ? `${w} × ${h}` : ''
}

/** 정수 시드(음수 -1 포함)면 그 문자열, 없거나 숫자가 아니면 ''. 큰 시드는 문자열 그대로 둔다. */
export function formatSeed(seed: unknown): string {
  if (typeof seed === 'number') return Number.isFinite(seed) ? String(Math.trunc(seed)) : ''
  if (typeof seed === 'string') {
    const text = seed.trim()
    return /^-?\d+$/.test(text) ? text : ''
  }
  return ''
}

/** 화면에 찍을 값 — 비었으면 '—'. */
export function displayInfo(value: string | null | undefined): string {
  return value && value.trim() ? value : EMPTY_INFO
}
