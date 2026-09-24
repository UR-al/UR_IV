import { computed, ref } from 'vue'
import { appManagerDeps, type ManagerDeps } from './managerDeps'

export interface InstantWildcard { name: string; lines: string[]; [k: string]: any }

/** 즉석 와일드카드 이름 규칙 — `$$이름$$` 로 쓰이므로 영숫자 · 밑줄 · 점 · 슬래시만. */
export const INSTANT_WILDCARD_NAME = /^[\w./]+$/

/**
 * 즉석 와일드카드(user_data/instant_wildcards.json, `$$name$$`) — App.vue 에서 추출(App.vue 분할 ④).
 * 화면은 components/managers/InstantWildcardModal.vue. 목록은 instantWildcardsList 로 온다(App 이 `bind()`).
 */
export function createInstantWildcards(deps: Pick<ManagerDeps, 'onBackendEvent' | 'requestAction' | 'addToast'>) {
  const showInstantWcManager = ref(false)
  const instantWildcards = ref<InstantWildcard[]>([])  // [{name, lines}]
  const selectedInstantWc = ref('')
  const selectedInstantWcData = computed(() =>
    instantWildcards.value.find(w => w.name === selectedInstantWc.value) || null,
  )
  const iwEditLines = ref<string[]>([])

  function selectInstantWc(name: string) {
    selectedInstantWc.value = name
    const iw = instantWildcards.value.find(w => w.name === name)
    iwEditLines.value = iw ? [...iw.lines] : []
  }

  function loadInstantWcList() {
    deps.requestAction('instant_wildcards_list', {})
  }

  function createNewInstantWc() {
    const name = window.prompt('새 인스턴트 와일드카드 이름 (영숫자/_/.만):', '')
    if (!name || !INSTANT_WILDCARD_NAME.test(name)) return
    if (instantWildcards.value.find(w => w.name === name)) {
      deps.addToast('error', '이미 존재함')
      return
    }
    instantWildcards.value.push({ name, lines: [''] })
    selectedInstantWc.value = name
    iwEditLines.value = ['']
    deps.requestAction('instant_wildcards_save', { name, lines: [''] })
  }

  function saveCurrentInstantWc() {
    if (!selectedInstantWc.value) return
    const lines = iwEditLines.value.filter(l => l !== undefined)
    deps.requestAction('instant_wildcards_save', { name: selectedInstantWc.value, lines })
    // 로컬 미러
    const iw = instantWildcards.value.find(w => w.name === selectedInstantWc.value)
    if (iw) iw.lines = lines
    deps.addToast('success', `인스턴트 와일드카드 저장: $$${selectedInstantWc.value}$$`)
  }

  function deleteInstantWc(name: string) {
    if (!window.confirm(`'${name}' 삭제할까요?`)) return
    deps.requestAction('instant_wildcards_delete', { name })
    instantWildcards.value = instantWildcards.value.filter(w => w.name !== name)
    if (selectedInstantWc.value === name) {
      selectedInstantWc.value = ''
      iwEditLines.value = []
    }
  }

  function appendIwLine() { iwEditLines.value.push('') }
  function removeIwLine(index: number) { iwEditLines.value.splice(index, 1) }

  /** instantWildcardsList(JSON 배열) — 배열이 아니면 무시한다. */
  function applyList(json: string) {
    try {
      const arr = JSON.parse(json)
      if (Array.isArray(arr)) instantWildcards.value = arr
    } catch {}
  }
  function bind() { return deps.onBackendEvent('instantWildcardsList', applyList) }

  /** 스튜디오 도구 '즉석 WC' 버튼 — 모달을 열고 목록을 받는다. */
  function openInstantWcManager() {
    showInstantWcManager.value = true
    loadInstantWcList()
  }
  function closeInstantWcManager() { showInstantWcManager.value = false }

  return {
    showInstantWcManager, instantWildcards, selectedInstantWc, selectedInstantWcData, iwEditLines,
    selectInstantWc, loadInstantWcList, createNewInstantWc, saveCurrentInstantWc, deleteInstantWc,
    appendIwLine, removeIwLine, applyList, bind, openInstantWcManager, closeInstantWcManager,
  }
}

export type InstantWildcards = ReturnType<typeof createInstantWildcards>

let _app: InstantWildcards | null = null
export function useInstantWildcards(): InstantWildcards {
  if (!_app) _app = createInstantWildcards(appManagerDeps())
  return _app
}
