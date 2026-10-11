/**
 * 3×3 격자 변형(변형 탭 '격자 변형')의 점 계산 — 격자, 손잡이 고르기, 이웃 사이 클램프,
 * 백엔드 페이로드, 점 이동 되돌리기.
 *
 * 좌표는 캔버스(이미지 내부) 좌표다 — 픽셀 i 가 [i, i+1] 을 덮는다(EditorCanvas.getImagePos).
 * 백엔드(core/editor_mesh_warp)는 정규화 좌표 (x/w, y/h) 를 받는다 — 프리뷰(축소본)와 확정(원본)이
 * 같은 페이로드를 쓰려고. 점은 왼쪽 위부터 행 우선, 4×4 = 16개(3×3 칸).
 *
 * 클램프: 점은 이미지 안에, 같은 행의 왼쪽·오른쪽 이웃 사이(x)와 같은 열의 위·아래 이웃 사이(y)에
 * 이미지 크기의 2% 이상 떨어져 머문다 — 격자가 접히거나 두 점이 겹치지 않게(백엔드도 다시 확인한다).
 * '가장자리 고정'이면 테두리 12점은 잡히지 않고 안쪽 4점만 움직인다.
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

/**
 * 점 i 를 p 로 옮길 때 실제로 놓일 자리 — 이미지 안, 그리고 이웃(같은 행 좌우·같은 열 위아래)과
 * `minGap` 비율 이상 떨어진 곳. 이웃이 이미 너무 가까우면 그 축은 지금 자리를 지킨다.
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
  return { x, y }
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
const samePoints = (a: readonly MeshPoint[], b: readonly MeshPoint[]) =>
  a.length === b.length && a.every((p, i) => p.x === b[i].x && p.y === b[i].y)

/** 격자 변형 한 번(시작 ~ 적용/취소)의 점 상태와 되돌리기 기록. */
export class MeshSession {
  points: MeshPoint[] = []
  dragIdx = -1
  pinEdges = true
  private w = 0
  private h = 0
  private history: MeshPoint[][] = []
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

  undo(): boolean {
    const prev = this.history.pop()
    if (!prev) return false
    this.points = prev
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

  /** '가장자리 고정' 전환. 켤 때 옮겨 둔 테두리 점이 있으면 제자리로 돌린다(되돌리기 한 칸). */
  setPinEdges(on: boolean): boolean {
    this.pinEdges = on
    if (!on || !this.active) return false
    const snapped = snapEdgesToGrid(this.points, this.w, this.h)
    if (samePoints(snapped, this.points)) return false
    this.pushHistory(copy(this.points))
    this.points = snapped
    return true
  }

  normalized(): number[][] { return toNormalized(this.points, this.w, this.h) }

  isIdentity(): boolean { return isIdentityMesh(this.points, this.w, this.h) }

  private pushHistory(pts: MeshPoint[]): void {
    this.history.push(pts)
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
