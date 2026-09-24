import { ref } from 'vue'

/**
 * 전역 토스트 + 알림 기록(🔔) — App.vue 에서 추출(App.vue 분할 ④).
 *
 * 앱 어디서든 같은 목록을 쓰도록 모듈 싱글턴(`useToasts()`)으로 둔다. 화면은
 * components/ToastLayer.vue(토스트 스택)와 components/NotificationCenter.vue(벨·기록 패널)가 그린다.
 * Python 쪽 알림(showNotification)과 composable 들(useLoraStack · useSessionRestore …)이 모두
 * `addToast` 한 곳으로 들어온다.
 */

export interface Toast {
  id: number
  type: string
  msg: string
  count: number
  _ts: number
  _timer?: ReturnType<typeof setTimeout>
  [k: string]: any
}
export interface ToastHistoryItem { id: number; type: string; msg: string; ts: number; [k: string]: any }

/** 화면에 동시에 뜨는 토스트 수 — 넘치면 가장 오래된 것부터 뺀다 */
export const MAX_TOASTS = 5
/** 알림 기록(🔔)에 남기는 수 */
export const MAX_TOAST_HISTORY = 40
/** 토스트가 떠 있는 시간(ms) — 같은 메시지가 다시 오면 다시 센다 */
export const TOAST_TTL_MS = 3000

export function createToasts(opts: { now?: () => number } = {}) {
  const now = opts.now ?? (() => Date.now())
  const toasts = ref<Toast[]>([])
  let toastId = 0

  // 알림(토스트) 히스토리 — 🔔 버튼으로 사라진 토스트 다시 보기
  const toastHistory = ref<ToastHistoryItem[]>([])
  const showNotifPanel = ref(false)
  const unread = ref(0)

  function toggleNotifPanel() {
    showNotifPanel.value = !showNotifPanel.value
    if (showNotifPanel.value) unread.value = 0
  }
  function clearNotifHistory() { toastHistory.value = [] }

  function removeToast(id: number) {
    const t = toasts.value.find(x => x.id === id)
    if (t && t._timer) clearTimeout(t._timer)
    toasts.value = toasts.value.filter(t => t.id !== id)
  }

  function addToast(type: string, msg: string) {
    const id = toastId++
    // 같은 메시지가 직전에 있으면 카운터만 증가 (스팸 방지)
    const last = toasts.value[toasts.value.length - 1]
    if (last && last.type === type && last.msg === msg) {
      last.count = (last.count || 1) + 1
      last._ts = now()  // 타이머 리셋
      // 기존 타이머 취소 + 새로 설정
      if (last._timer) clearTimeout(last._timer)
      last._timer = setTimeout(() => removeToast(last.id), TOAST_TTL_MS)
      return
    }
    const toast: Toast = { id, type, msg, count: 1, _ts: now() }
    toasts.value.push(toast)
    toast._timer = setTimeout(() => removeToast(id), TOAST_TTL_MS)
    // 스택 초과분 — 가장 오래된 것부터 제거 (화면엔 최대 5개)
    while (toasts.value.length > MAX_TOASTS) {
      const oldest = toasts.value.shift()!
      if (oldest._timer) clearTimeout(oldest._timer)
    }
    // 알림 기록에 누적 (사라진 토스트도 🔔에서 다시 볼 수 있게)
    toastHistory.value.unshift({ id, type, msg, ts: now() })
    if (toastHistory.value.length > MAX_TOAST_HISTORY) toastHistory.value.length = MAX_TOAST_HISTORY
    if (!showNotifPanel.value) unread.value++
  }

  function clearAllToasts() {
    toasts.value.forEach(t => { if (t._timer) clearTimeout(t._timer) })
    toasts.value = []
  }

  return {
    toasts, toastHistory, showNotifPanel, unread,
    addToast, removeToast, clearAllToasts, toggleNotifPanel, clearNotifHistory,
  }
}

export type ToastsApi = ReturnType<typeof createToasts>

let _app: ToastsApi | null = null
/** 앱 전체가 쓰는 하나의 토스트 목록. */
export function useToasts(): ToastsApi {
  if (!_app) _app = createToasts()
  return _app
}
