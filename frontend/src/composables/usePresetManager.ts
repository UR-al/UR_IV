import { ref } from 'vue'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

/**
 * 프리셋 관리 모달의 상태와 동작 — App.vue 에서 추출(App.vue 분할 ④).
 * 화면은 components/managers/PresetManagerModal.vue. 상태를 여기 두는 이유는 managerDeps.ts 주석.
 */
export function createPresetManager(deps: Pick<ManagerDeps, 'getBackend' | 'requestAction'>) {
  const showPresetManager = ref(false)
  const presetList = ref<string[]>([])
  const selectedPreset = ref('')
  const presetPreview = ref<any>(null)

  async function loadPresetList() {
    const bk = await deps.getBackend()
    if (bk.getPresetList) bk.getPresetList((json: string) => { try { presetList.value = JSON.parse(json) } catch {} })
  }
  async function loadPresetPreview(name: string) {
    const bk = await deps.getBackend()
    if (bk.getPresetData) bk.getPresetData(name, (json: string) => { try { presetPreview.value = JSON.parse(json) } catch {} })
  }
  function selectPreset(name: string) {
    selectedPreset.value = name
    void loadPresetPreview(name)
  }
  function loadSelectedPreset() {
    if (!selectedPreset.value) return
    deps.requestAction('load_preset_by_name', { name: selectedPreset.value })
    showPresetManager.value = false
  }
  function deleteSelectedPreset() {
    if (!selectedPreset.value || !confirm(`"${selectedPreset.value}" 삭제?`)) return
    deps.requestAction('delete_preset', { name: selectedPreset.value })
    presetList.value = presetList.value.filter(p => p !== selectedPreset.value)
    selectedPreset.value = ''; presetPreview.value = null
  }
  function saveNewPreset() {
    const name = prompt('프리셋 이름:')
    if (!name) return
    // 결과 토스트는 백엔드가 낸다(정규화된 실제 이름·실패 사유 포함) — 여기서 성공을 미리 알리지 않는다.
    deps.requestAction('save_preset_by_name', { name })
    // 원본 이름을 낙관적으로 넣지 않음 — 백엔드가 정규화해 저장하므로 실제 목록을 재조회
    setTimeout(loadPresetList, 200)
  }
  /** 스튜디오 도구 '프리셋' 버튼 */
  function openPresetManager() {
    showPresetManager.value = true
    void loadPresetList()
  }
  function closePresetManager() { showPresetManager.value = false }

  return {
    showPresetManager, presetList, selectedPreset, presetPreview,
    loadPresetList, loadPresetPreview, selectPreset, loadSelectedPreset, deleteSelectedPreset, saveNewPreset,
    openPresetManager, closePresetManager,
  }
}

export type PresetManager = ReturnType<typeof createPresetManager>

let _app: PresetManager | null = null
export function usePresetManager(): PresetManager {
  if (!_app) _app = createPresetManager(appManagerDeps())
  return _app
}
