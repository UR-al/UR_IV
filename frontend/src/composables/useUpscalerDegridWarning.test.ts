import { describe, expect, it, vi } from 'vitest'
import { nextTick, reactive, ref } from 'vue'

// 스토어 위젯 속성(ComfyUI `comfyModels`)과 Forge 기능 스냅샷을 대역으로 — 컴포저블이 고른 업스케일러와 두 목록을 실제로 잇는지 본다
const caps = ref<any>(null)
const properties = reactive<Record<string, unknown>>({})
vi.mock('../stores/widgetStore.js', () => ({
  useWidgetStore: () => ({ getProperty: (id: string, prop: string, def?: unknown) => properties[`${id}:${prop}`] ?? def }),
}))
vi.mock('./useSamExtraCapabilities', () => ({ useSamExtraCapabilities: () => ({ capabilities: caps }) }))

const { useUpscalerDegridWarning } = await import('./useUpscalerDegridWarning')
const { COMFY_MODELS_PROPERTY, UPSCALER_DEGRID_WARNING, WIDGET_IDS } = await import('../utils/vaeDegrid')

const comfyKey = `${WIDGET_IDS.model}:${COMFY_MODELS_PROPERTY}`

describe('useUpscalerDegridWarning', () => {
  it('warns for the selected upscaler only when it is a Forge DeGrid model, and follows the selection', async () => {
    caps.value = { choices: { degrid_models: ['qwenVAEDegridNafnet_v11', 'None'] } }
    delete properties[comfyKey]
    const upscaler = ref('4x-UltraSharp')
    const warning = useUpscalerDegridWarning(() => upscaler.value)
    expect(warning.value).toBe('')
    upscaler.value = 'qwenVAEDegridNafnet_v11'
    await nextTick()
    expect(warning.value).toBe(UPSCALER_DEGRID_WARNING)
    upscaler.value = 'None'
    await nextTick()
    expect(warning.value).toBe('')
  })

  it('also checks the ComfyUI model list by stem (folder, extension and case ignored)', async () => {
    caps.value = null
    properties[comfyKey] = ['DeGrid/NAFNet-QwenVAE-DeGrid.safetensors']
    const upscaler = ref('nafnet-qwenvae-degrid')
    const warning = useUpscalerDegridWarning(() => upscaler.value)
    expect(warning.value).toBe(UPSCALER_DEGRID_WARNING)
    properties[comfyKey] = []
    await nextTick()
    expect(warning.value).toBe('')
  })

  it('reacts when the Forge snapshot arrives after mounting', async () => {
    caps.value = null
    delete properties[comfyKey]
    const warning = useUpscalerDegridWarning(() => 'qwenVAEDegridNafnet_v11')
    expect(warning.value).toBe('')
    caps.value = { choices: { degrid_models: ['qwenVAEDegridNafnet_v11'] } }
    await nextTick()
    expect(warning.value).toBe(UPSCALER_DEGRID_WARNING)
  })
})
