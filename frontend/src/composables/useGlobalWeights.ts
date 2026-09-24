import { reactive, ref } from 'vue'
import { applyGlobalWeights, createRowKeyer, type GlobalWeight } from '../utils/globalWeights'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

/**
 * 글로벌 태그 가중치 — App.vue 에서 추출(App.vue 분할 ④).
 * 모달(components/managers/WeightManagerModal.vue)은 표시만 하고, 목록은 여기 있다: 생성 경로
 * (`applyToPrompt`)가 모달이 닫혀 있어도 이 목록을 쓰고, 부팅 때 오는 globalWeightsLoaded 도
 * 모달과 무관하게 받아야 한다(App 의 onMounted 가 `bind()` 를 부른다).
 */
export function createGlobalWeights(deps: Pick<ManagerDeps, 'onBackendEvent' | 'requestAction'>) {
  const showWeightManager = ref(false)
  const globalWeights = reactive<GlobalWeight[]>([])  // [{tag, weight}]
  // 행 key — 편집 중인 태그 문자열을 key 로 쓰면 글자마다 행이 다시 마운트돼 포커스를 잃었다(utils/globalWeights)
  const weightRowKey = createRowKeyer()

  function saveGlobalWeights() {
    const valid = globalWeights.filter(w => w.tag.trim())
    deps.requestAction('save_global_weights', { weights: valid })
    showWeightManager.value = false
  }
  function addWeightRow() { globalWeights.push({ tag: '', weight: 100 }) }
  function removeWeightRow(index: number) { globalWeights.splice(index, 1) }

  /** globalWeightsLoaded(JSON 배열) — 목록을 통째로 바꾼다. 깨진 페이로드는 무시한다. */
  function applyLoaded(json: string) {
    try {
      const d = JSON.parse(json)
      globalWeights.splice(0)
      d.forEach((w: GlobalWeight) => globalWeights.push(w))
    } catch {}
  }
  function bind() { return deps.onBackendEvent('globalWeightsLoaded', applyLoaded) }

  /**
   * 생성 직전 — 프롬프트에 글로벌 가중치를 적용한 결과. 가중치가 없거나 바뀐 게 없으면 null.
   * 정규식 없이 쉼표·줄바꿈 토큰 단위로 적용한다(utils/globalWeights.applyGlobalWeights). 예외는 호출부가 처리.
   */
  function weightedPrompt(prompt: string): string | null {
    if (globalWeights.length === 0) return null
    const weighted = applyGlobalWeights(prompt, globalWeights)
    return weighted !== prompt ? weighted : null
  }

  function openWeightManager() { showWeightManager.value = true }
  function closeWeightManager() { showWeightManager.value = false }

  return {
    showWeightManager, globalWeights, weightRowKey,
    saveGlobalWeights, addWeightRow, removeWeightRow, applyLoaded, bind, weightedPrompt,
    openWeightManager, closeWeightManager,
  }
}

export type GlobalWeightsManager = ReturnType<typeof createGlobalWeights>

let _app: GlobalWeightsManager | null = null
export function useGlobalWeights(): GlobalWeightsManager {
  if (!_app) _app = createGlobalWeights(appManagerDeps())
  return _app
}
