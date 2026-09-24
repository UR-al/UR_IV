/**
 * 크래시 복구(세션 백업) — App.vue 에서 분리 (감사 #160, App.vue 분할 ④).
 *
 * 흐름:
 * 1. 마운트 1.2초 뒤(위젯 초기값이 들어온 뒤) getSession 으로 백업을 읽는다.
 * 2. utils/sessionBackup.shouldOfferSessionRestore 가 참이면 배너를 띄우고, 사용자가 복원/닫기를 고를
 *    때까지 백업을 **덮지 않는다** — 예전엔 곧이어 2.5초·30초 저장이 크래시 직전 편집을 지웠다.
 * 3. 결정(또는 제안할 것이 없음)이 나면 편집 2.5초 뒤·30초마다 백업한다.
 * 응답이 오지 않으면(웹 파사드 오류 등) answerTimeoutMs 뒤 결정된 것으로 보고 백업을 재개한다.
 */
import { ref, watch } from 'vue'
import {
  buildSessionBackup,
  parseSessionBackup,
  shouldOfferSessionRestore,
  type SessionBackup,
} from '../utils/sessionBackup'

export interface SessionRestoreDeps {
  storeWidgets: Record<string, any>
  /** 지금 라우트 이름(탭) */
  currentTab: () => unknown
  /** 복원한 탭으로 이동 */
  goToTab: (tab: string) => void
  getBackend: () => Promise<any>
  addToast: (type: string, msg: string) => void
  checkDelayMs?: number
  saveDelayMs?: number
  periodicMs?: number
  answerTimeoutMs?: number
}

export function useSessionRestore(deps: SessionRestoreDeps) {
  const {
    storeWidgets, currentTab, goToTab, getBackend, addToast,
    checkDelayMs = 1200, saveDelayMs = 2500, periodicMs = 30000, answerTimeoutMs = 5000,
  } = deps
  const sessionRestore = ref<SessionBackup | null>(null)
  let decided = false          // 복구 여부가 정해졌는가 — 그 전엔 백업을 덮지 않는다
  let saveTimer: ReturnType<typeof setTimeout> | null = null

  function currentPrompts() {
    return {
      prompt: storeWidgets.main_prompt_text || '',
      negative: storeWidgets.neg_prompt_text || '',
    }
  }

  function saveNow() {
    if (!decided) return
    getBackend().then((bk) => {
      if (!bk || !bk.saveSession) return
      bk.saveSession(JSON.stringify(buildSessionBackup(currentTab(), currentPrompts())), () => {})
    }).catch(() => {})
  }

  function scheduleSave() {
    if (!decided) return
    if (saveTimer) clearTimeout(saveTimer)
    saveTimer = setTimeout(saveNow, saveDelayMs)
  }

  function decide() {
    decided = true
  }

  async function check() {
    let answered = false
    const finish = (json: unknown) => {
      if (answered) return
      answered = true
      const backup = parseSessionBackup(json)
      if (shouldOfferSessionRestore(backup, currentPrompts())) {
        sessionRestore.value = backup   // 사용자가 고를 때까지 decided=false — 백업 보존
      } else {
        decide()
        saveNow()   // 지금 화면을 새 기준으로 (정상 종료 표시 clean 도 이 저장으로 풀린다)
      }
    }
    try {
      const bk = await getBackend()
      if (!bk || !bk.getSession) { finish(null); return }
      setTimeout(() => finish(null), answerTimeoutMs)
      bk.getSession((json: string) => finish(json))
    } catch {
      finish(null)
    }
  }

  function applySessionRestore() {
    const d = sessionRestore.value
    if (!d) return
    if (typeof d.prompt === 'string' && d.prompt) storeWidgets.main_prompt_text = d.prompt
    if (typeof d.negative === 'string') storeWidgets.neg_prompt_text = d.negative
    if (typeof d.tab === 'string' && d.tab) {
      try { goToTab(d.tab) } catch { /* 없는 탭 — 프롬프트만 복원 */ }
    }
    sessionRestore.value = null
    decide()
    saveNow()
    addToast('success', '이전 세션을 복원했습니다')
  }

  function dismissSessionRestore() {
    sessionRestore.value = null
    decide()
    saveNow()
  }

  /** App.vue onMounted 에서 한 번 — 복구 확인 예약 + 편집 감시 + 주기 백업. */
  function startSessionBackup() {
    setTimeout(() => { void check() }, checkDelayMs)
    watch(() => [storeWidgets.main_prompt_text, storeWidgets.neg_prompt_text, currentTab()], scheduleSave)
    setInterval(saveNow, periodicMs)   // root 는 언마운트되지 않는다
  }

  return { sessionRestore, applySessionRestore, dismissSessionRestore, startSessionBackup }
}
