/** 알림 기록의 '몇 초 전' 표기 — 초 · 분 · 시간 · 일 단위로 내림한다. */
export function relativeTimeKo(ts: number, now: number = Date.now()): string {
  const s = Math.floor((now - ts) / 1000)
  if (s < 60) return `${s}초 전`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}분 전`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h}시간 전`
  return `${Math.floor(h / 24)}일 전`
}

/** 알림 벨 배지 — 10 이상은 '9+' 로 줄인다. */
export function unreadBadgeText(unread: number): string {
  return unread > 9 ? '9+' : String(unread)
}
