import { ref } from 'vue'
import { moveAdjacent } from '../utils/listMove'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

export interface PromptOrderSection { key: string; label: string; [k: string]: any }

/**
 * 프롬프트 섹션 순서 매니저 — App.vue 에서 추출(App.vue 분할 ④).
 * 화면은 components/managers/PromptOrderModal.vue. 목록은 promptOrderLoaded 로 온다(App onMounted 가 `bind()`).
 */
export function createPromptOrder(deps: Pick<ManagerDeps, 'onBackendEvent' | 'requestAction' | 'addToast'>) {
  const showOrderManager = ref(false)
  const promptOrderList = ref<PromptOrderSection[]>([])  // [{key, label}, ...]

  function loadPromptOrder() {
    deps.requestAction('prompt_order_list', {})
  }
  function moveOrderUp(i: number) {
    const next = moveAdjacent(promptOrderList.value, i, -1)
    if (next) promptOrderList.value = next
  }
  function moveOrderDown(i: number) {
    const next = moveAdjacent(promptOrderList.value, i, 1)
    if (next) promptOrderList.value = next
  }
  function saveOrderAndClose() {
    const order = promptOrderList.value.map(s => s.key)
    deps.requestAction('prompt_order_save', { order })
    showOrderManager.value = false
    deps.addToast('success', '프롬프트 순서 저장됨')
  }
  function resetPromptOrder() {
    if (!window.confirm('기본 순서(인물수 → 캐릭터 → 작품 → 작가 → 선행 → 메인 → 후행)로 복원할까요?')) return
    deps.requestAction('prompt_order_reset', {})
    deps.addToast('info', '기본 순서로 복원')
  }

  /** promptOrderLoaded(JSON 배열) — 배열이 아니면 무시한다. */
  function applyLoaded(json: string) {
    try {
      const arr = JSON.parse(json)
      if (Array.isArray(arr)) promptOrderList.value = arr
    } catch {}
  }
  function bind() { return deps.onBackendEvent('promptOrderLoaded', applyLoaded) }

  /** 스튜디오 도구 '순서' 버튼 — 모달을 열고 현재 순서를 받는다. */
  function openOrderManager() {
    showOrderManager.value = true
    loadPromptOrder()
  }
  function closeOrderManager() { showOrderManager.value = false }

  return {
    showOrderManager, promptOrderList,
    loadPromptOrder, moveOrderUp, moveOrderDown, saveOrderAndClose, resetPromptOrder,
    applyLoaded, bind, openOrderManager, closeOrderManager,
  }
}

export type PromptOrder = ReturnType<typeof createPromptOrder>

let _app: PromptOrder | null = null
export function usePromptOrder(): PromptOrder {
  if (!_app) _app = createPromptOrder(appManagerDeps())
  return _app
}
