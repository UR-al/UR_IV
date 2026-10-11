/**
 * 격자 변형 손잡이 — EditorCanvas 쪽(포인터·오버레이). `useDrawLayer` 처럼 캔버스 컴포넌트에서
 * 떼어 낸 응집 기능이다. 캔버스는 원근 보정 블록 바로 뒤에 한 줄씩 넘겨 주기만 한다.
 *
 * 점 계산·되돌리기는 utils/meshWarp 의 MeshSession, 그리기는 utils/meshWarpGuide.
 * 점이 움직일 때마다(끄는 중·끝) `onChange(정규화 dst)` — 부모가 디바운스 프리뷰를 건다.
 */
import { MeshSession, hitMeshHandle, type MeshPoint } from '../utils/meshWarp'
import { MESH_HANDLE_PX, drawMeshGuide } from '../utils/meshWarpGuide'

export interface MeshWarpHandleDeps {
  /** 지금 원본 이미지 크기(캔버스 내부 좌표). 이미지가 없으면 null. */
  imageSize: () => { w: number; h: number } | null
  /** 화면 1px 이 이미지 몇 px 인지 — 손잡이 크기·히트 반경을 줌과 무관하게 */
  imagePerScreenPx: () => number
  /** 가이드만 다시 그린다(마스크 픽셀은 그대로) */
  redraw: () => void
  /** 오버레이 전체를 다시 그린다(시작·끝 — 가이드가 남지 않게) */
  redrawAll: () => void
  onChange: (dst: number[][]) => void
}

/** 히트 반경 배율 — 원근 보정 손잡이와 같이 넉넉하게 */
const HIT_SLOP = 1.8

export function useMeshWarpHandles(deps: MeshWarpHandleDeps) {
  const session = new MeshSession()
  let active = false

  const isActive = () => active
  const emitChange = () => deps.onChange(session.normalized())

  function pointerDown(p: MeshPoint): boolean {
    if (!active) return false
    const idx = hitMeshHandle(session.points, p.x, p.y, MESH_HANDLE_PX * deps.imagePerScreenPx() * HIT_SLOP,
      session.pinEdges)
    if (!session.startDrag(idx)) return false
    deps.redraw()
    return true
  }

  function pointerMove(p: MeshPoint): boolean {
    if (!active || session.dragIdx < 0) return false
    if (session.dragTo(p)) {
      deps.redraw()
      emitChange()
    }
    return true
  }

  function pointerUp(): boolean {
    if (!active) return false
    const dragging = session.dragIdx >= 0
    const moved = session.endDrag()
    if (dragging) deps.redraw()
    if (moved) emitChange()
    return dragging
  }

  function draw(ctx: CanvasRenderingContext2D): void {
    if (active) drawMeshGuide(ctx, session, deps.imagePerScreenPx())
  }

  function begin(pinEdges: boolean): boolean {
    const size = deps.imageSize()
    if (!size || size.w <= 0 || size.h <= 0) return false
    session.begin(size.w, size.h, pinEdges)
    active = true
    deps.redrawAll()
    return true
  }

  function stop(): void {
    active = false
    session.clear()
    deps.redrawAll()
  }

  /** 적용 — 정규화 dst 를 돌려주고 끝낸다. 진행 중이 아니면 null. */
  function end(): number[][] | null {
    if (!active) return null
    const dst = session.normalized()
    stop()
    return dst
  }

  function cancel(): void {
    if (active) stop()
  }

  function reset(): boolean {
    if (!active || !session.reset()) return false
    deps.redraw()
    emitChange()
    return true
  }

  /** 마지막 점 이동 되돌리기(Ctrl+Z). 되돌릴 게 없으면 false. */
  function undoStep(): boolean {
    if (!active || !session.undo()) return false
    deps.redraw()
    emitChange()
    return true
  }

  function setPinEdges(on: boolean): void {
    if (!active) { session.pinEdges = on; return }
    const changed = session.setPinEdges(on)
    deps.redraw()
    if (changed) emitChange()
  }

  return { isActive, pointerDown, pointerMove, pointerUp, draw, begin, end, cancel, reset, undoStep, setPinEdges }
}
