/**
 * 격자 변형 가이드 그리기 — 행·열 선과 손잡이. EditorCanvas 의 마스크 오버레이 위에 얹힌다
 * (프리뷰 레이어보다 위라 손잡이가 옮겨진 그림 위에 그대로 보인다).
 *
 * 색은 원근 보정 손잡이와 같은 hex 다 — 임의의 이미지 위에 얹히는 표시라 테마 토큰을 쓰지 않는다
 * (EditorCanvas.drawTransientOverlay 주석과 같은 이유). 크기는 화면 px 기준이라 `px`(화면 1px 이
 * 이미지 몇 px 인지)를 곱한다.
 */
import { MESH_N, MESH_POINTS, isEdgeIndex, type MeshPoint } from './meshWarp'

/** 움직이는 손잡이 반경(화면 px) — 원근 보정 손잡이와 같다 */
export const MESH_HANDLE_PX = 9
/** 고정된 테두리 점 표시(속 빈 네모)의 반 변(화면 px) */
export const MESH_PINNED_PX = 4

const GOLD = '#E2B340'

export interface MeshGuideState {
  points: readonly MeshPoint[]
  dragIdx: number
  pinEdges: boolean
}

export function drawMeshGuide(ctx: CanvasRenderingContext2D, state: MeshGuideState, px: number): void {
  const pts = state.points
  if (pts.length !== MESH_POINTS) return
  ctx.save()

  // 격자선 — 행과 열을 점 순서대로 잇는다(휜 모양이 그대로 보인다)
  ctx.strokeStyle = 'rgba(226, 179, 64, 0.7)'
  ctx.lineWidth = 1.5 * px
  for (let r = 0; r < MESH_N; r++) {
    ctx.beginPath()
    ctx.moveTo(pts[r * MESH_N].x, pts[r * MESH_N].y)
    for (let c = 1; c < MESH_N; c++) ctx.lineTo(pts[r * MESH_N + c].x, pts[r * MESH_N + c].y)
    ctx.stroke()
  }
  for (let c = 0; c < MESH_N; c++) {
    ctx.beginPath()
    ctx.moveTo(pts[c].x, pts[c].y)
    for (let r = 1; r < MESH_N; r++) ctx.lineTo(pts[r * MESH_N + c].x, pts[r * MESH_N + c].y)
    ctx.stroke()
  }

  const handle = MESH_HANDLE_PX * px
  const pinned = MESH_PINNED_PX * px
  for (let i = 0; i < MESH_POINTS; i++) {
    const p = pts[i]
    if (state.pinEdges && isEdgeIndex(i)) {
      ctx.strokeStyle = 'rgba(226, 179, 64, 0.55)'
      ctx.lineWidth = 1.5 * px
      ctx.strokeRect(p.x - pinned, p.y - pinned, pinned * 2, pinned * 2)
      continue
    }
    const active = i === state.dragIdx
    ctx.beginPath()
    ctx.arc(p.x, p.y, handle, 0, Math.PI * 2)
    ctx.fillStyle = active ? GOLD : 'rgba(20, 20, 20, 0.85)'
    ctx.fill()
    ctx.strokeStyle = GOLD
    ctx.lineWidth = 2 * px
    ctx.stroke()
  }
  ctx.restore()
}
