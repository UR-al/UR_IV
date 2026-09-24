/**
 * `index` 번째 항목을 한 칸 위(-1) 또는 아래(+1)로 옮긴 **새 배열**. 옮길 수 없으면(맨 위에서 위로,
 * 맨 아래에서 아래로, 범위 밖) null — 호출부는 목록을 그대로 둔다. 프롬프트 섹션 순서 매니저의 ↑/↓.
 */
export function moveAdjacent<T>(list: readonly T[], index: number, delta: -1 | 1): T[] | null {
  const target = index + delta
  if (index < 0 || index >= list.length || target < 0 || target >= list.length) return null
  const out = [...list]
  ;[out[index], out[target]] = [out[target], out[index]]
  return out
}
