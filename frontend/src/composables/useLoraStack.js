import { reactive, ref, computed, watch, nextTick } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { fromBridgeLoraEntries, toBridgeLoraEntries } from '../utils/loraUnits'
import { createTrailingDebounce, flushOnPageHide } from '../utils/trailingDebounce'

/** 가중치 슬라이더 드래그 중 ui_prefs.json 저장을 모으는 간격(ms) — Python 동기화는 즉시 (감사 #130) */
export const LORA_SAVE_DEBOUNCE_MS = 300

/**
 * LoRA 스택 상태 + 동작 (App.vue에서 추출 — App.vue 분할 ④).
 * 영속 소스: config/ui_prefs.json 의 loraStack(정수 %). 생성 LoRA 의 단일 소스는 Python
 * _vue_lora_entries(배율) — 부팅 시 ui_prefs 에서 복원되고, 여기 syncLoraStack(set_lora_stack)이
 * ui_prefs 복원 직후·변경마다·생성 직전마다 갱신한다(빈 스택도 전송). 마운트 시점의 localStorage
 * 스택은 보내지 않는다(restoreFromPrefs 주석 참고). 단위 변환은 utils/loraUnits.ts 에서만.
 *
 * @param {object} deps
 * @param {object} deps.storeWidgets  위젯 스토어 reactive (main_prompt_text 등)
 * @param {Function} deps.addToast    (type, msg) 토스트
 * @param {Function} deps.saveUiPrefs (payload) ui_prefs 영속
 */
export function useLoraStack({ storeWidgets, addToast, saveUiPrefs }) {
  /** @type {import('../types/bridge').LoraEntry[]} */
  const loraStack = reactive([])

  // localStorage 복원 (빠른 폴백 — 시작 시 uiPrefs가 override)
  try {
    const saved = JSON.parse(window.localStorage.getItem('loraStack') || '[]')
    if (Array.isArray(saved)) saved.forEach(l => loraStack.push(l))
  } catch {}

  let _loraInitialized = false
  let _loraRestoring = false

  // 파일 저장만 모은다 — 슬라이더 드래그(input 이벤트마다)가 틱마다 GUI 스레드의 ui_prefs.json
  // 읽기-수정-쓰기를 부르지 않게. Python 생성 스택(set_lora_stack)은 syncLoraStack 이 즉시 보낸다.
  const _persistLoraStack = createTrailingDebounce(
    (stack) => saveUiPrefs({ loraStack: stack }),
    LORA_SAVE_DEBOUNCE_MS,
  )
  flushOnPageHide(() => _persistLoraStack.flush())

  function _saveLoraStack() {
    try { window.localStorage.setItem('loraStack', JSON.stringify(loraStack)) } catch {}
    // 초기화 완료 전 빈 배열로 덮어쓰기 방지, 이후에는 빈 배열도 정상 저장
    if (!_loraInitialized && loraStack.length === 0) return
    _loraInitialized = true
    _persistLoraStack(loraStack.map(l => ({ ...l })))
  }

  function syncLoraStack() {
    // 빈 스택·전부 꺼짐도 그대로 보낸다 — Python 이 옛 LoRA 를 계속 붙이지 않게.
    requestAction('set_lora_stack', { entries: toBridgeLoraEntries(loraStack) })
  }

  // LoRA 추가 (LoRA 매니저 모달의 onLoraAdd) — 저장·Python 동기는 아래 deep watch 한 곳에서
  // (예전엔 여기서도 직접 저장해 한 번 추가에 save_ui_prefs 가 두 번 나갔다)
  function addLoraToStack(name, weight, triggerWords = []) {
    const existing = loraStack.find(l => l.name === name)
    if (existing) {
      existing.weight = Math.round(weight * 100); existing.enabled = true
      if (triggerWords.length) existing.triggerWords = triggerWords
    } else {
      loraStack.push({ name, weight: Math.round(weight * 100), enabled: true, triggerWords })
    }
  }

  // 변경 감시 → 자동 저장(디바운스) + Python 동기(즉시) (복원 중에는 무시)
  watch(loraStack, () => {
    if (_loraRestoring) return
    _saveLoraStack()
    syncLoraStack()
  }, { deep: true })

  function insertTriggerWord(tw) {
    const cur = storeWidgets.main_prompt_text || ''
    if (!cur.toLowerCase().includes(tw.toLowerCase())) {
      storeWidgets.main_prompt_text = cur ? cur.replace(/,?\s*$/, '') + ', ' + tw + ', ' : tw + ', '
      addToast('info', `트리거 워드 삽입: ${tw}`)
    } else {
      addToast('info', `이미 포함된 태그: ${tw}`)
    }
  }

  const allLorasOn = computed(() => loraStack.length > 0 && loraStack.every(l => l.enabled))
  function toggleAllLoras(on) { loraStack.forEach(l => { l.enabled = on }) }
  function insertAllTriggers() {
    const tws = []
    for (const l of loraStack) {
      if (l.enabled && Array.isArray(l.triggerWords)) {
        for (const tw of l.triggerWords) if (tw && !tws.includes(tw)) tws.push(tw)
      }
    }
    if (!tws.length) { addToast('info', '활성 LoRA의 트리거 워드가 없습니다'); return }
    const cur = (storeWidgets.main_prompt_text || '').trim()
    const lower = cur.toLowerCase()
    const add = tws.filter(tw => !lower.includes(tw.toLowerCase()))
    if (!add.length) { addToast('info', '트리거가 이미 모두 포함됨'); return }
    storeWidgets.main_prompt_text = cur ? (cur.replace(/,?\s*$/, '') + ', ' + add.join(', ')) : add.join(', ')
    addToast('success', `트리거 ${add.length}개 삽입`)
  }

  // LoRA 매니저(Vue 모달)
  const showLoraModal = ref(false)
  function onLoraAdd(p) {
    addLoraToStack(p.name, typeof p.weight === 'number' ? p.weight : 1.0, p.triggerWords || [])
    addToast('success', `LoRA 추가: ${p.name}`)
  }

  // LoRA 세트 저장/불러오기 (localStorage 'loraSets')
  const loraSets = ref({})
  try { const s = JSON.parse(window.localStorage.getItem('loraSets') || '{}'); if (s && typeof s === 'object') loraSets.value = s } catch {}
  const loraSetName = ref('')
  const loraSetSel = ref('')
  const loraSetNames = computed(() => Object.keys(loraSets.value))
  function _persistLoraSets() { try { window.localStorage.setItem('loraSets', JSON.stringify(loraSets.value)) } catch {} }
  function saveLoraSet() {
    const name = loraSetName.value.trim()
    if (!name || !loraStack.length) return
    loraSets.value = { ...loraSets.value, [name]: loraStack.map(l => ({ ...l })) }
    _persistLoraSets()
    loraSetSel.value = name; loraSetName.value = ''
    addToast('success', `LoRA 세트 저장: ${name} (${loraStack.length}개)`)
  }
  function loadLoraSet() {
    const set = loraSets.value[loraSetSel.value]
    if (!Array.isArray(set)) return
    loraStack.splice(0, loraStack.length, ...set.map(l => ({ ...l })))   // 저장·동기는 deep watch
    addToast('success', `세트 적용: ${loraSetSel.value} (${set.length}개)`)
  }
  function deleteLoraSet() {
    const n = loraSetSel.value
    if (!n) return
    const cp = { ...loraSets.value }; delete cp[n]
    loraSets.value = cp; _persistLoraSets(); loraSetSel.value = ''
    addToast('info', `세트 삭제: ${n}`)
  }

  // 드래그 순서 변경 — 그립(⠿)만 draggable, 삽입 위치(loraDropIdx)를 마커로 표시
  const loraDragIdx = ref(-1)   // 드래그 출발 인덱스
  const loraDropIdx = ref(-1)   // 삽입 위치 (0..length)
  function loraDragStart(i) { loraDragIdx.value = i }
  function loraDragOver(e, i) {
    // 블록 상/하 절반으로 'i 앞' / 'i+1 앞' 삽입 결정
    try {
      const r = e.currentTarget.getBoundingClientRect()
      loraDropIdx.value = (e.clientY - r.top) > r.height / 2 ? i + 1 : i
    } catch { loraDropIdx.value = i }
  }
  function loraDragEnd() { loraDragIdx.value = -1; loraDropIdx.value = -1 }
  function loraDrop() {
    const from = loraDragIdx.value
    let to = loraDropIdx.value
    loraDragIdx.value = -1; loraDropIdx.value = -1
    if (from < 0 || to < 0) return
    if (to > from) to -= 1          // 앞쪽 제거로 인덱스 보정
    if (to === from) return          // 제자리 → 무동작
    const moved = loraStack.splice(from, 1)[0]
    loraStack.splice(to, 0, moved)   // 저장·동기는 deep watch
  }

  // 단일 소스(ui_prefs) 복원 — App.vue uiPrefsLoaded 핸들러에서 호출.
  // 복원 뒤 반드시 Python 에도 보낸다(useRatingFilter.restoreFromPrefs → pushRatingFilter 와 같은 계약).
  // 마운트 시점엔 보내지 않는다 — 웹 모드는 ui_prefs 가 getInitialConfig 응답으로 늦게 오고 Python
  // 호스트는 새 클라이언트마다 _restore_runtime_prefs 를 돌리지 않아서, 이 브라우저의 낡은(또는 빈)
  // localStorage 스택이 공유 _vue_lora_entries 를 덮으면 채팅·XYZ·대기열·자동화가 그 스택으로 생성됐다.
  // ui_prefs 에 loraStack 이 없으면(한 번도 저장된 적 없음) 화면의 localStorage 스택이 곧 사용자가
  // 보는 값이므로 그대로 보낸다.
  function restoreFromPrefs(prefs) {
    if (!prefs || typeof prefs !== 'object') return
    const hasStack = Array.isArray(prefs.loraStack)
    if (hasStack) {
      _loraRestoring = true
      _persistLoraStack.cancel()   // 파일 값이 이긴다 — 복원 전 화면 스택의 늦은 저장이 덮지 않게
      loraStack.splice(0, loraStack.length, ...prefs.loraStack.map(l => ({ ...l })))
      try { window.localStorage.setItem('loraStack', JSON.stringify(prefs.loraStack)) } catch {}
    }
    nextTick(() => {
      if (hasStack) { _loraRestoring = false; _loraInitialized = true }
      syncLoraStack()
    })
  }

  // Python → Vue: loraStackLoaded (워크플로 프로파일 적용 등). 시작 1회 active_loras emit은
  // 단일 소스 통합으로 제거됨 → 평소엔 프로파일 적용 때만 발생. 항목은 배율 단위(Python 이 옛
  // 퍼센트 프로파일까지 정규화해서 보낸다). Python _vue_lora_entries 는 이미 같은 값이라 되돌려
  // 보내지 않고, ui_prefs 에만 영속한다 — 안 하면 재시작 시 옛 스택으로 돌아갔다.
  onBackendEvent('loraStackLoaded', (json) => {
    try {
      const entries = fromBridgeLoraEntries(JSON.parse(json))
      if (!entries) return
      _loraRestoring = true
      loraStack.splice(0, loraStack.length, ...entries)
      _loraInitialized = true   // 프로파일의 빈 스택도 사용자의 선택 — 그대로 저장
      _saveLoraStack()
      nextTick(() => { _loraRestoring = false })
    } catch {}
  })

  return {
    loraStack, syncLoraStack, addLoraToStack,
    allLorasOn, toggleAllLoras, insertTriggerWord, insertAllTriggers,
    showLoraModal, onLoraAdd,
    loraSetName, loraSetSel, loraSetNames, saveLoraSet, loadLoraSet, deleteLoraSet,
    loraDragIdx, loraDropIdx, loraDragStart, loraDragOver, loraDragEnd, loraDrop,
    restoreFromPrefs,
  }
}
