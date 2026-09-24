import { ref } from 'vue'
import { DEFAULT_RANDOM_RES_H, DEFAULT_RANDOM_RES_W, normalizeRandomRes, type RandomResEntry } from '../utils/randomResolution'
import { widgetFlag } from './widgetFlag'
import type { GetBackendFn, RequestActionFn, WidgetValues } from './managerDeps'

/**
 * 랜덤 해상도 — 파라미터 열의 '랜덤' 토글과 목록 편집기(App.vue 에서 추출, App.vue 분할 ④).
 * 토글은 Python CheckBoxProxy 'random_res_check', 목록은 Python 이 가진다(getRandomResolutions /
 * set_random_resolutions). App 이 부팅 때 `loadRandomResList()` 를 부르고, 카드
 * (components/params/ParamsBasicCard.vue)는 이 객체를 prop 으로 받아 그린다.
 */
export function useRandomResolutions(deps: {
  storeWidgets: WidgetValues
  getBackend: GetBackendFn
  requestAction: RequestActionFn
}) {
  const randomResList = ref<RandomResEntry[]>([])
  const newResW = ref(DEFAULT_RANDOM_RES_W)
  const newResH = ref(DEFAULT_RANDOM_RES_H)

  async function loadRandomResList() {
    const bk = await deps.getBackend()
    if (bk.getRandomResolutions) {
      bk.getRandomResolutions((json: string) => {
        try { randomResList.value = JSON.parse(json) } catch {}
      })
    }
  }
  // 켜는 순간 목록을 새로 받는다
  const randomResEnabled = widgetFlag(deps.storeWidgets, 'random_res_check', (on) => { if (on) void loadRandomResList() })

  function addRandomRes() {
    const entry = normalizeRandomRes(newResW.value, newResH.value)
    if (!entry) return
    randomResList.value.push(entry)
    deps.requestAction('set_random_resolutions', { list: randomResList.value })
  }
  function removeRandomRes(i: number) {
    randomResList.value.splice(i, 1)
    deps.requestAction('set_random_resolutions', { list: randomResList.value })
  }

  return { randomResEnabled, randomResList, newResW, newResH, loadRandomResList, addRandomRes, removeRandomRes }
}

export type RandomResolutions = ReturnType<typeof useRandomResolutions>
