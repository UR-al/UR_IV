/**
 * 우클릭 메뉴가 화면 밖으로 나가지 않게 위치를 당긴다 — Gallery·Favorites·히스토리(App.vue) 공용.
 *
 * 예전엔 Favorites 와 App.vue 만 보정했고 Gallery 는 원시 좌표를 써서, 화면 아래쪽에서
 * 우클릭하면 '휴지통으로 이동' 같은 아래 항목이 잘렸다.
 */
export interface MenuSize { width: number; height: number }
export interface ViewportSize { width: number; height: number }
export interface MenuPoint { x: number; y: number }

/** 오른쪽·아래로 넘치면 안쪽으로 당기고(여백 margin), 왼쪽·위로 넘치면 가장자리에 붙인다. */
export function clampMenuPosition(
  point: MenuPoint,
  menu: MenuSize,
  viewport: ViewportSize,
  margin = 10,
  minTop = 8,
): MenuPoint {
  let { x, y } = point
  if (x + menu.width > viewport.width) x = viewport.width - menu.width - margin
  if (y + menu.height > viewport.height) y = viewport.height - menu.height - margin
  if (x < 0) x = margin
  if (y < minTop) y = minTop
  return { x, y }
}

export function menuPositionStyle(point: MenuPoint): { top: string; left: string } {
  return { top: `${point.y}px`, left: `${point.x}px` }
}
