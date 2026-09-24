import { computed, ref } from 'vue'
import { profileFileKey } from '../utils/profileFileName'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

export interface WorkflowProfile { name: string; created_at?: string; model?: string; vae?: string; [k: string]: any }

/**
 * 워크플로우 프로파일(config/profiles/*.json) — App.vue 에서 추출(App.vue 분할 ④).
 * 왼쪽 열의 프로파일 드롭다운과 관리 모달(components/managers/WorkflowProfileModal.vue)이 같은 목록을
 * 쓴다. 목록은 workflowProfilesList 로 오고(App 의 onMounted 가 `bind()`), 모달이 닫혀 있어도 받는다.
 */
export function createWorkflowProfiles(deps: Pick<ManagerDeps, 'onBackendEvent' | 'requestAction'>) {
  const showProfileManager = ref(false)
  const workflowProfiles = ref<WorkflowProfile[]>([])  // [{name, created_at, model, vae}]
  const profileNames = computed(() => workflowProfiles.value.map(p => p.name))

  function loadWorkflowProfilesList() {
    deps.requestAction('workflow_profile_list', {})
  }
  function loadWorkflowProfile(name: string | number) {
    if (!name) return
    deps.requestAction('workflow_profile_load', { name })
  }
  function saveCurrentAsProfile() {
    const name = (window.prompt('프로파일 이름 (예: ANIMA Pony, Flux 표준):', '') || '').trim()
    if (!name) return
    // 이미 있으면 덮어쓰기 확인 — 글자 그대로가 아니라 **같은 파일**인지로 본다. 'Flux.' · 'flux' ·
    // 'Fl/ux' 는 저장 규칙상 모두 Flux.json 이라, 예전엔 확인 없이 기존 'Flux' 를 덮었다(utils/profileFileName).
    const key = profileFileKey(name)
    const existing = workflowProfiles.value.find(p => profileFileKey(String(p.name ?? '')) === key)
    if (existing) {
      const message = existing.name === name
        ? `'${name}' 이미 존재합니다. 덮어쓸까요?`
        : `'${name}' 은(는) 기존 프로파일 '${existing.name}' 과(와) 같은 파일로 저장됩니다. '${existing.name}' 을(를) 덮어쓸까요?`
      if (!window.confirm(message)) return
    }
    // overwrite 는 확인받았을 때만 true — 목록이 낡아 여기서 못 본 충돌은 백엔드가 쓰지 않고 경고한다
    deps.requestAction('workflow_profile_save', { name, overwrite: !!existing })
  }
  function deleteWorkflowProfile(name: string) {
    if (!window.confirm(`'${name}' 삭제할까요?`)) return
    deps.requestAction('workflow_profile_delete', { name })
    workflowProfiles.value = workflowProfiles.value.filter(p => p.name !== name)
  }
  function renameWorkflowProfile(oldName: string) {
    const newName = window.prompt(`'${oldName}' → 새 이름:`, oldName)
    if (!newName || !newName.trim() || newName.trim() === oldName) return
    deps.requestAction('workflow_profile_rename', { old: oldName, new: newName.trim() })
  }

  /** workflowProfilesList(JSON 배열) — 배열이 아니면 무시한다. */
  function applyList(json: string) {
    try {
      const arr = JSON.parse(json)
      if (Array.isArray(arr)) workflowProfiles.value = arr
    } catch {}
  }
  function bind() { return deps.onBackendEvent('workflowProfilesList', applyList) }

  /** 드롭다운 옆 ⚙ — 관리 모달을 열고 목록을 새로 받는다. */
  function openProfileManager() {
    showProfileManager.value = true
    loadWorkflowProfilesList()
  }
  function closeProfileManager() { showProfileManager.value = false }

  return {
    showProfileManager, workflowProfiles, profileNames,
    loadWorkflowProfilesList, loadWorkflowProfile, saveCurrentAsProfile, deleteWorkflowProfile, renameWorkflowProfile,
    applyList, bind, openProfileManager, closeProfileManager,
  }
}

export type WorkflowProfiles = ReturnType<typeof createWorkflowProfiles>

let _app: WorkflowProfiles | null = null
export function useWorkflowProfiles(): WorkflowProfiles {
  if (!_app) _app = createWorkflowProfiles(appManagerDeps())
  return _app
}
