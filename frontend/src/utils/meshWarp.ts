/**
 * 3×3 격자 변형(변형 탭 '격자 변형')의 점 계산 — 격자, 손잡이 고르기, 이웃 사이 클램프,
 * 백엔드 페이로드, 점 이동 되돌리기.
 *
 * 좌표는 캔버스(이미지 내부) 좌표다 — 픽셀 i 가 [i, i+1] 을 덮는다(EditorCanvas.getImagePos).
 * 백엔드(core/editor_mesh_warp)는 정규화 좌표 (x/w, y/h) 를 받는다 — 프리뷰(축소본)와 확정(원본)이
 * 같은 페이로드를 쓰려고. 점은 왼쪽 위부터 행 우선, 4×4 = 16개(3×3 칸).
 *
 * 클램프: 점은 이미지 안에, 같은 행의 왼쪽·오른쪽 이웃 사이(x)와 같은 열의 위·아래 이웃 사이(y)에
 * 이미지 크기의 2% 이상 떨어져 머문다. 그것만으로는 대각선으로 엇갈려 칸이 뒤집힐 수 있어서, 이 점에 붙은
 * 칸(최대 4개)이 볼록하게(뒤집히거나 꼬이거나 오목해지지 않게) 남고 다른 모든 점(대각선 이웃 포함)과 2% 이상
 * 떨어진 곳까지만 간다. 오목한 칸도 얇은 판 스플라인이 크게 접어 버려서(실측) 볼록까지 요구한다
 * (백엔드 core/editor_mesh_warp.check_grid 도 다시 확인한다 — 반올림 여유로 백엔드 쪽 기준이 조금 느슨하다).
 * '가장자리 고정'이면 테두리 12점은 잡히지 않고 안쪽 4점만 움직이고, 테두리는 늘 처음 자리다.
 * 고정을 켜며 테두리를 되돌린 것도 되돌리기 한 칸이고, 그 칸을 되돌리면 고정도 함께 꺼진다(체크도 따라간다).
 */

export const MESH_N = 4
export const MESH_POINTS = MESH_N * MESH_N
/** 이웃 점과의 최소 간격 — 이미지 폭·높이의 비율 */
export const MESH_MIN_GAP = 0.02
/** 점 이동 되돌리기 기록 — 끌기 한 번(움직임마다가 아니라)·초기화·고정으로 되돌린 테두리가 한 칸 */
export const MESH_HISTORY_MAX = 50
export const MESH_PIN_STORAGE_KEY = 'editorMeshPinEdges'
/** 페이로드 반올림 자리(1e-5) — 4K 에서도 0.04px */
const ROUND = 1e5

export interface MeshPoint { x: number; y: number }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/** 처음 격자 — 이미지를 3×3 칸으로 나눈 꼭짓점 16개. */
export function meshGrid(w: number, h: number): MeshPoint[] {
  const pts: MeshPoint[] = []
  for (let r = 0; r < MESH_N; r++) {
    for (let c = 0; c < MESH_N; c++) pts.push({ x: (w * c) / (MESH_N - 1), y: (h * r) / (MESH_N - 1) })
  }
  return pts
}

/** 테두리 점인가 (행·열이 0 또는 3). */
export function isEdgeIndex(i: number): boolean {
  const r = Math.floor(i / MESH_N), c = i % MESH_N
  return r === 0 || c === 0 || r === MESH_N - 1 || c === MESH_N - 1
}

/** (x, y) 에서 반경 r 안의 가장 가까운 손잡이. 고정된 테두리 점은 고르지 않는다. 없으면 -1. */
export function hitMeshHandle(points: readonly MeshPoint[], x: number, y: number, r: number, pinEdges: boolean): number {
  let best = -1
  let bestDist = r * r
  for (let i = 0; i < points.length; i++) {
    if (pinEdges && isEdgeIndex(i)) continue
    const dx = points[i].x - x, dy = points[i].y - y
    const d = dx * dx + dy * dy
    if (d <= bestDist) { bestDist = d; best = i }
  }
  return best
}

/** 점 i 가 꼭짓점인 칸들 — 칸마다 [왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래] 인덱스. */
export function meshCellsAround(i: number): number[][] {
  const r = Math.floor(i / MESH_N), c = i % MESH_N
  const cells: number[][] = []
  for (const r0 of [r - 1, r]) {
    for (const c0 of [c - 1, c]) {
      if (r0 < 0 || c0 < 0 || r0 >= MESH_N - 1 || c0 >= MESH_N - 1) continue
      const tl = r0 * MESH_N + c0
      cells.push([tl, tl + 1, tl + MESH_N + 1, tl + MESH_N])
    }
  }
  return cells
}

/** 칸 꼭짓점 각의 sin 하한 — 백엔드(0)보다 조금 엄격해 반올림한 페이로드가 거절되지 않는다 */
export const MESH_CONVEX_MARGIN = 0.05

/**
 * 칸이 볼록한가 — 네 꼭짓점 모두 처음 격자와 같은 방향으로 돌고, 꼭짓점 각의 sin 이 `margin` 보다 크다.
 * 네 꼭짓점이 모두 같은 방향이면 뒤집힘·꼬임(나비꼴)·오목이 모두 걸러진다. q 는 [왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래].
 */
export function meshCellConvex(q: readonly MeshPoint[], margin = MESH_CONVEX_MARGIN): boolean {
  for (let k = 0; k < 4; k++) {
    const p = q[(k + 3) % 4], c = q[k], n = q[(k + 1) % 4]
    const ax = n.x - c.x, ay = n.y - c.y, bx = p.x - c.x, by = p.y - c.y
    const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by)
    if (la === 0 || lb === 0 || (ax * by - ay * bx) / (la * lb) <= margin) return false
  }
  return true
}

/**
 * 점 i 를 p 에 두어도 격자가 온전한가 — i 에 붙은 칸이 모두 볼록하고, 다른 모든 점(대각선 이웃 포함)과
 * 정규화 거리 `minGap` 이상 떨어져 있다. 비교는 정규화 좌표(x/w, y/h)로 한다(백엔드와 같은 공간).
 */
export function meshMoveSound(points: readonly MeshPoint[], i: number, p: MeshPoint, w: number, h: number,
  minGap = MESH_MIN_GAP): boolean {
  const sx = w > 0 ? 1 / w : 0, sy = h > 0 ? 1 / h : 0
  for (let j = 0; j < points.length; j++) {
    if (j === i) continue
    const dx = (p.x - points[j].x) * sx, dy = (p.y - points[j].y) * sy
    if (dx * dx + dy * dy < minGap * minGap) return false
  }
  return meshCellsAround(i).every(cell =>
    meshCellConvex(cell.map(j => (j === i ? { x: p.x * sx, y: p.y * sy } : { x: points[j].x * sx, y: points[j].y * sy }))))
}

/** 같은 자리로 보는 이분 탐색 횟수 — 2^-20 이면 4K 에서도 1px 의 1/200 */
const SETTLE_STEPS = 20

/**
 * 점 i 를 p 로 옮길 때 실제로 놓일 자리 — 이미지 안, 그리고 이웃(같은 행 좌우·같은 열 위아래)과
 * `minGap` 비율 이상 떨어진 곳. 이웃이 이미 너무 가까우면 그 축은 지금 자리를 지킨다.
 * 그 자리에서 칸이 볼록하지 않거나 다른 점에 너무 붙으면(meshMoveSound), 지금 자리에서 그쪽으로
 * 갈 수 있는 데까지만 간다. 지금 자리부터 이미 어긋나 있으면(밖에서 만든 점) 축 클램프만 쓴다 — 백엔드가 거른다.
 */
export function clampMeshPoint(points: readonly MeshPoint[], i: number, p: MeshPoint, w: number, h: number,
  minGap = MESH_MIN_GAP): MeshPoint {
  const r = Math.floor(i / MESH_N), c = i % MESH_N
  const gx = minGap * w, gy = minGap * h
  const loX = c > 0 ? points[i - 1].x + gx : 0
  const hiX = c < MESH_N - 1 ? points[i + 1].x - gx : w
  const loY = r > 0 ? points[i - MESH_N].y + gy : 0
  const hiY = r < MESH_N - 1 ? points[i + MESH_N].y - gy : h
  const x = loX <= hiX ? clamp(p.x, Math.max(0, loX), Math.min(w, hiX)) : points[i].x
  const y = loY <= hiY ? clamp(p.y, Math.max(0, loY), Math.min(h, hiY)) : points[i].y
  const target = { x, y }
  if (meshMoveSound(points, i, target, w, h, minGap)) return target
  const cur = points[i]
  if (!meshMoveSound(points, i, cur, w, h, minGap)) return target
  let lo = 0, hi = 1
  for (let k = 0; k < SETTLE_STEPS; k++) {
    const mid = (lo + hi) / 2
    const q = { x: cur.x + (target.x - cur.x) * mid, y: cur.y + (target.y - cur.y) * mid }
    if (meshMoveSound(points, i, q, w, h, minGap)) lo = mid
    else hi = mid
  }
  return lo > 0 ? { x: cur.x + (target.x - cur.x) * lo, y: cur.y + (target.y - cur.y) * lo } : { x: cur.x, y: cur.y }
}

/** 백엔드 페이로드 `dst` — [[u, v] × 16], 0~1, 1e-5 로 반올림. */
export function toNormalized(points: readonly MeshPoint[], w: number, h: number): number[][] {
  return points.map(p => [
    Math.round(clamp(w > 0 ? p.x / w : 0, 0, 1) * ROUND) / ROUND,
    Math.round(clamp(h > 0 ? p.y / h : 0, 0, 1) * ROUND) / ROUND,
  ])
}

/** 정규화 격자 [[u, v] × 16] 가 처음 격자 그대로인가(백엔드 항등 판정과 같은 뜻). */
export function isIdentityNormalized(dst: readonly (readonly number[])[], eps = 1e-4): boolean {
  if (dst.length !== MESH_POINTS) return false
  return dst.every((uv, i) => {
    const u = (i % MESH_N) / (MESH_N - 1), v = Math.floor(i / MESH_N) / (MESH_N - 1)
    return Math.abs(uv[0] - u) <= eps && Math.abs(uv[1] - v) <= eps
  })
}

export function isIdentityMesh(points: readonly MeshPoint[], w: number, h: number, eps = 1e-4): boolean {
  return isIdentityNormalized(toNormalized(points, w, h), eps)
}

/** 테두리 점만 처음 격자 자리로 되돌린 사본. */
export function snapEdgesToGrid(points: readonly MeshPoint[], w: number, h: number): MeshPoint[] {
  const grid = meshGrid(w, h)
  return points.map((p, i) => (isEdgeIndex(i) ? grid[i] : { ...p }))
}

const copy = (pts: readonly MeshPoint[]): MeshPoint[] => pts.map(p => ({ x: p.x, y: p.y }))

/** 되돌리기 한 칸 — 점들과, 고정을 켜며 테두리를 되돌린 칸이면 그 전의 고정 상태 */
interface MeshHistoryEntry { points: MeshPoint[]; pinEdges?: boolean }
const samePoints = (a: readonly MeshPoint[], b: readonly MeshPoint[]) =>
  a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y)

/** 격자 변형 한 번(시작 ~ 적용/취소)의 점 상태와 되돌리기 기록. */
export class MeshSession {
  points: MeshPoint[] = []
  dragIdx = -1
  pinEdges = true
  private w = 0
  private h = 0
  private history: MeshHistoryEntry[] = []
  private dragStart: MeshPoint[] | null = null

  get active(): boolean { return this.points.length === MESH_POINTS }
  get canUndo(): boolean { return this.history.length > 0 }

  begin(w: number, h: number, pinEdges: boolean): void {
    this.w = w
    this.h = h
    this.points = meshGrid(w, h)
    this.pinEdges = pinEdges
    this.dragIdx = -1
    this.dragStart = null
    this.history = []
  }

  clear(): void {
    this.points = []
    this.dragIdx = -1
    this.dragStart = null
    this.history = []
  }

  startDrag(i: number): boolean {
    if (!this.active || i < 0 || i >= MESH_POINTS || (this.pinEdges && isEdgeIndex(i))) return false
    this.dragIdx = i
    this.dragStart = copy(this.points)
    return true
  }

  /** 끌고 있는 점을 p 쪽으로(클램프). 자리가 바뀌었으면 true. */
  dragTo(p: MeshPoint): boolean {
    if (this.dragIdx < 0) return false
    const next = clampMeshPoint(this.points, this.dragIdx, p, this.w, this.h)
    const cur = this.points[this.dragIdx]
    if (next.x === cur.x && next.y === cur.y) return false
    this.points[this.dragIdx] = next
    return true
  }

  /** 끌기를 마친다. 실제로 움직였으면 되돌리기 한 칸을 남기고 true. */
  endDrag(): boolean {
    const start = this.dragStart
    this.dragIdx = -1
    this.dragStart = null
    if (!start || samePoints(start, this.points)) return false
    this.pushHistory(start)
    return true
  }

  /**
   * 마지막 한 칸을 되돌린다. 고정을 켜며 테두리를 되돌린 칸이면 고정도 그 전 상태로 돌린다(부르는 쪽이
   * pinEdges 를 보고 체크를 맞춘다). 고정이 켜져 있으면 테두리는 늘 처음 자리 — 손잡이·프리뷰·결과가 같게.
   */
  undo(): boolean {
    const prev = this.history.pop()
    if (!prev) return false
    if (prev.pinEdges !== undefined) this.pinEdges = prev.pinEdges
    this.points = this.pinEdges ? snapEdgesToGrid(prev.points, this.w, this.h) : prev.points
    this.dragIdx = -1
    this.dragStart = null
    return true
  }

  /** 처음 격자로. 이미 그대로면 false(기록도 남기지 않는다). */
  reset(): boolean {
    if (!this.active) return false
    const grid = meshGrid(this.w, this.h)
    if (samePoints(grid, this.points)) return false
    this.pushHistory(copy(this.points))
    this.points = grid
    return true
  }

  /**
   * '가장자리 고정' 전환. 켤 때 옮겨 둔 테두리 점이 있으면 제자리로 돌린다(되돌리기 한 칸 — 그 칸에는 고정을
   * 켜기 전 상태도 적어, 되돌리면 테두리와 함께 고정도 꺼진다. 예전엔 고정이 켜진 채 옮긴 테두리만 되살아나
   * 손잡이(잡을 수 없음)·프리뷰(테두리 고정)·적용(변화 없음)이 서로 달랐다).
   */
  setPinEdges(on: boolean): boolean {
    const before = this.pinEdges
    this.pinEdges = on
    if (!on || !this.active) return false
    const snapped = snapEdgesToGrid(this.points, this.w, this.h)
    if (samePoints(snapped, this.points)) return false
    this.pushHistory(copy(this.points), before)
    this.points = snapped
    return true
  }

  normalized(): number[][] { return toNormalized(this.points, this.w, this.h) }

  isIdentity(): boolean { return isIdentityMesh(this.points, this.w, this.h) }

  private pushHistory(pts: MeshPoint[], pinEdges?: boolean): void {
    this.history.push(pinEdges === undefined ? { points: pts } : { points: pts, pinEdges })
    while (this.history.length > MESH_HISTORY_MAX) this.history.shift()
  }
}

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

/** '가장자리 고정' 저장값 — 기본 켜짐. */
export function loadMeshPinEdges(storage: StorageLike | null = browserStorage()): boolean {
  try {
    return storage?.getItem(MESH_PIN_STORAGE_KEY) !== '0'
  } catch {
    return true
  }
}

export function saveMeshPinEdges(on: boolean, storage: StorageLike | null = browserStorage()): void {
  try {
    storage?.setItem(MESH_PIN_STORAGE_KEY, on ? '1' : '0')
  } catch {
    // 저장소가 막혀 있으면 이번 세션 값만 쓴다
  }
}
