/**
 * 참조 색 맞추기 — EditorView 쪽(참조 이미지 경로·프리뷰·적용). 패널은 ColorMatchPanel.
 *
 * '참조 이미지 불러오기' → 호스트 파일 대화상자(editor_load_color_reference) → editorColorReferenceLoaded
 * 로 경로가 돌아온다. 경로는 이 세션에만 둔다(저장하지 않는다). 무엇을 보낼지는 utils/colorMatch 의
 * colorMatchPlan 이 정한다 — 마스크가 필요하면 doOpWithMask(칠한 곳이 없으면 그쪽이 안내한다).
 */
import { onMounted, onUnmounted, ref } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { colorMatchPlan, type ColorMatchConfig } from '../utils/colorMatch'

export type ColorMatchToastType = 'info' | 'warning' | 'error' | 'success'

export interface ColorMatchDeps {
  /** 지금 캔버스에 칠한 영역이 있는가(호출 시점 값) */
  hasMask: () => boolean
  schedulePreview: (operation: string, params: Record<string, unknown>) => void
  scheduleMaskPreview: (operation: string, params: Record<string, unknown>) => void
  doOp: (operation: string, params: Record<string, unknown>) => void
  doOpWithMask: (operation: string, params: Record<string, unknown>) => void
  clearPreview: () => void
  toast: (type: ColorMatchToastType, msg: string) => void
}

export function useColorMatch(deps: ColorMatchDeps) {
  const referencePath = ref('')

  function loadReference() { requestAction('editor_load_color_reference') }

  function previewColorMatch(cfg: ColorMatchConfig) {
    const plan = colorMatchPlan(cfg, referencePath.value, deps.hasMask())
    // 보낼 수 없는 설정의 프리뷰는 조용히 걷는다 — 안내는 패널이, 토스트는 적용 때만
    if (!plan.ok) { deps.clearPreview(); return }
    const params = { ...plan.params }
    if (plan.withMask) deps.scheduleMaskPreview('color_match', params)
    else deps.schedulePreview('color_match', params)
  }

  function applyColorMatch(cfg: ColorMatchConfig) {
    const plan = colorMatchPlan(cfg, referencePath.value, deps.hasMask())
    if (!plan.ok) { deps.toast('warning', plan.message); return }
    const params = { ...plan.params }
    if (plan.withMask) deps.doOpWithMask('color_match', params)
    else deps.doOp('color_match', params)
  }

  let unsubscribe: (() => void) | null = null
  onMounted(() => {
    unsubscribe = onBackendEvent('editorColorReferenceLoaded', (path: string) => {
      referencePath.value = String(path || '')
    })
  })
  onUnmounted(() => { unsubscribe?.(); unsubscribe = null })

  return { referencePath, loadReference, previewColorMatch, applyColorMatch }
}
