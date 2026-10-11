/**
 * 격자 변형 — EditorView 쪽(모드·프리뷰·적용·단축키). 캔버스 쪽 손잡이는 useMeshWarpHandles.
 *
 * 흐름: 시작(다른 모드 취소 → 도구 'mesh_warp' → 프리뷰 걷기 → 캔버스 손잡이) → 점을 끌 때마다
 * 디바운스 프리뷰(`mesh_warp`, 정규화 좌표라 축소본에도 같은 페이로드) → 적용(`doOp` → 결과가 undo
 * 스택에 들어간다) / 취소. 진행 중에는 Ctrl+Z 가 마지막 점 이동을, Esc 가 취소를, Enter 가 적용을 한다.
 * 도구를 바꾸거나 이미지가 바뀌면(undo·열기·다른 작업 결과) 진행 중인 격자는 취소한다 — 옛 그림에 맞춘
 * 점을 새 그림에 적용하지 않게.
 */
import { ref, watch, type Ref } from 'vue'
import { isIdentityNormalized, loadMeshPinEdges, saveMeshPinEdges } from '../utils/meshWarp'

export const MESH_TOOL = 'mesh_warp'

/** EditorCanvas 가 노출하는 격자 변형 API(테스트·부분 목에서 빠져도 되게 선택형). */
export interface MeshCanvasApi {
  beginMeshWarp?: (pinEdges: boolean) => boolean
  endMeshWarp?: () => number[][] | null
  cancelMeshWarp?: () => void
  resetMeshWarp?: () => boolean
  undoMeshStep?: () => boolean
  setMeshPinEdges?: (on: boolean) => void
}

export type MeshToastType = 'info' | 'warning' | 'error' | 'success'

export interface EditorMeshWarpDeps {
  canvas: () => MeshCanvasApi | null | undefined
  hasImage: () => boolean
  currentTool: Ref<string>
  imagePath: Ref<string>
  clearPreview: () => void
  schedulePreview: (operation: string, params: Record<string, unknown>) => void
  doOp: (operation: string, params: Record<string, unknown>) => void
  /** 원근 보정·영역 이동처럼 캔버스를 잡는 다른 모드를 끝낸다 */
  cancelOtherModes: () => void
  toast: (type: MeshToastType, msg: string) => void
}

export const MESH_MESSAGES = Object.freeze({
  started: '점을 끌어 모양을 다듬은 뒤 "적용"을 누르세요',
  unchanged: '움직인 점이 없습니다',
})

function typingInField(): boolean {
  if (typeof document === 'undefined') return false
  const el = document.activeElement as HTMLElement | null
  return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
}

export function useEditorMeshWarp(deps: EditorMeshWarpDeps) {
  const meshActive = ref(false)
  const meshPinEdges = ref(loadMeshPinEdges())

  const payload = (dst: number[][]) => ({ grid: 4, dst, pin_edges: meshPinEdges.value })

  function finish() {
    meshActive.value = false
    if (deps.currentTool.value === MESH_TOOL) deps.currentTool.value = 'box'
  }

  function startMesh() {
    if (!deps.hasImage() || meshActive.value) return
    deps.cancelOtherModes()
    // 프리뷰가 떠 있으면 점을 맞출 원본이 가려진다
    deps.clearPreview()
    if (!deps.canvas()?.beginMeshWarp?.(meshPinEdges.value)) return
    meshActive.value = true
    deps.currentTool.value = MESH_TOOL
    deps.toast('info', MESH_MESSAGES.started)
  }

  function confirmMesh() {
    if (!meshActive.value) return
    const dst = deps.canvas()?.endMeshWarp?.() ?? null
    finish()
    if (!dst || isIdentityNormalized(dst)) {
      deps.clearPreview()
      deps.toast('info', MESH_MESSAGES.unchanged)
      return
    }
    // 프리뷰는 걷지 않는다 — 확정 결과가 그려질 때 캔버스가 걷는다(그새 '적용 전' 그림이 비치지 않게)
    deps.doOp('mesh_warp', payload(dst))
  }

  function cancelMesh() {
    if (!meshActive.value) return
    deps.canvas()?.cancelMeshWarp?.()
    finish()
    deps.clearPreview()
  }

  function resetMesh() {
    if (meshActive.value) deps.canvas()?.resetMeshWarp?.()
  }

  function setMeshPinEdges(on: boolean) {
    meshPinEdges.value = !!on
    saveMeshPinEdges(meshPinEdges.value)
    deps.canvas()?.setMeshPinEdges?.(meshPinEdges.value)
  }

  /** 캔버스의 mesh-changed — 점이 처음 자리면 프리뷰를 걷고, 아니면 디바운스 프리뷰. */
  function onMeshChanged(dst: number[][]) {
    if (!meshActive.value) return
    if (isIdentityNormalized(dst)) { deps.clearPreview(); return }
    deps.schedulePreview('mesh_warp', payload(dst))
  }

  /** 진행 중일 때 가로챌 키. 처리했으면 true(호출자는 더 보지 않는다). */
  function handleMeshKey(e: KeyboardEvent): boolean {
    if (!meshActive.value) return false
    if (e.ctrlKey && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'z') {
      // 이미지 undo 로 넘기지 않는다 — 그러면 그림이 바뀌어 격자가 통째로 취소된다
      e.preventDefault()
      e.stopImmediatePropagation()
      deps.canvas()?.undoMeshStep?.()
      return true
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      cancelMesh()
      return true
    }
    if (e.key === 'Enter' && !typingInField()) {
      e.preventDefault()
      confirmMesh()
      return true
    }
    return false
  }

  watch(deps.currentTool, (tool) => { if (meshActive.value && tool !== MESH_TOOL) cancelMesh() })
  watch(deps.imagePath, () => { if (meshActive.value) cancelMesh() })

  return {
    meshActive, meshPinEdges, startMesh, confirmMesh, cancelMesh, resetMesh, setMeshPinEdges, onMeshChanged,
    handleMeshKey,
  }
}
