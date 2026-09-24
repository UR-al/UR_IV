/** 생성 통계의 '총 시간' — 초를 `42s` · `3m 5s` · `2h 14m` 로 줄인다(0/빈 값은 `0s`). */
export function formatDuration(sec: number): string {
  if (!sec) return '0s'
  if (sec < 60) return sec + 's'
  if (sec < 3600) return Math.floor(sec / 60) + 'm ' + Math.round(sec % 60) + 's'
  return Math.floor(sec / 3600) + 'h ' + Math.floor((sec % 3600) / 60) + 'm'
}
