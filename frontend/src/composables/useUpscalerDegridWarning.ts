import { computed, type ComputedRef } from 'vue'
import { useWidgetStore } from '../stores/widgetStore.js'
import { useSamExtraCapabilities } from './useSamExtraCapabilities'
import { COMFY_MODELS_PROPERTY, WIDGET_IDS, upscalerDegridWarning } from '../utils/vaeDegrid'

/**
 * 업스케일러 선택 칸의 DeGrid 경고 — Hires.fix 카드(components/params/HiresFixCard.vue)와 배치 업스케일
 * (views/BatchView.vue)이 같은 규칙을 쓴다(utils/vaeDegrid `upscalerDegridWarning`).
 *
 * 두 백엔드의 DeGrid 모델 목록과 비교한다: Forge 기능 스냅샷 `choices.degrid_models`(Forge 는 models/ESRGAN 의 NAFNet 을
 * 업스케일러 목록에도 올린다)와 ComfyUI 의 `_degrid_model` 위젯 속성 `comfyModels`(upscale_models 의 NAFNet). 표시만 —
 * 고른 값은 막지 않는다. `upscaler` 는 지금 고른 이름을 돌려주는 함수(스토어 위젯·ref 어느 쪽이든).
 */
export function useUpscalerDegridWarning(upscaler: () => unknown): ComputedRef<string> {
  const store = useWidgetStore()
  const getProperty = store.getProperty as (id: string, prop: string, def?: unknown) => unknown
  const { capabilities } = useSamExtraCapabilities()
  return computed(() => upscalerDegridWarning(
    upscaler(), capabilities.value, getProperty(WIDGET_IDS.model, COMFY_MODELS_PROPERTY, null)))
}
