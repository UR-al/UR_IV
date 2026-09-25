import { ref } from 'vue'
import { onBackendBound, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { parseMemoState } from '../utils/memos'
import type { MemoItem, MemoSaveResult, MemoSavePayload, MemoSyncState } from '../types/bridge'
import type { OnBackendEventFn, RequestActionFn } from './managerDeps'

/**
 * 메모장 상태 — 백엔드(user_data/memos.json ↔ Forge sam-extra Notebook 메모)가 보내는 memoState 를
 * 받아 둔다. **모듈 싱글턴**이다: 패널(components/dock/MemoPanel.vue)을 닫아도 목록 · 동기화 상태를
 * 계속 받는다. 편집 초안과 자동 저장은 composables/useMemoDraft.
 *
 * 브리지(tests/test_bridge_contract.py 가 이름을 지킨다): memo_list · memo_save · memo_delete · memo_sync,
 * 시그널 memoState.
 */
export interface MemoDeps {
  onBackendEvent: OnBackendEventFn
  /** 백엔드가 붙을 때마다(첫 연결 · 웹 재접속) — 그때 목록을 당겨 온다 */
  onBackendBound: (cb: () => void) => (() => void) | void
  requestAction: RequestActionFn
}

export function createMemos(deps: MemoDeps) {
  const memos = ref<MemoItem[]>([])
  /** null = memoState 를 아직 못 받음(옛 백엔드이거나 답을 기다리는 중) */
  const sync = ref<MemoSyncState | null>(null)
  /** 마지막 memoState 에 실린 저장 답(memo_save 의 답에만 — 다른 memoState 면 null). revision 과 함께 바뀐다 */
  const saved = ref<MemoSaveResult | null>(null)
  /** memoState 를 받을 때마다 1 씩 — 초안 맞추기가 '새 서버 상태'를 알아채는 신호 */
  const revision = ref(0)

  function applyState(json: string) {
    const state = parseMemoState(json)
    if (!state) return
    memos.value = state.memos
    sync.value = state.sync
    saved.value = state.saved
    revision.value++
  }
  function refresh() { deps.requestAction('memo_list', {}) }
  function syncNow() { deps.requestAction('memo_sync', {}) }
  function save(payload: MemoSavePayload) { deps.requestAction('memo_save', payload) }
  /** 지운 것은 목록에서 바로 뺀다 — 백엔드의 다음 memoState 가 권위 있게 다시 알려 준다. */
  function remove(id: string) {
    memos.value = memos.value.filter(m => m.id !== id)
    deps.requestAction('memo_delete', { id })
  }

  let unbind: Array<() => void> = []
  function bind() {
    if (unbind.length) return
    const off = deps.onBackendEvent('memoState', applyState)
    const offBound = deps.onBackendBound(refresh)
    unbind = [off || (() => {}), offBound || (() => {})]
  }
  function unbindAll() { for (const off of unbind) { try { off() } catch {} } unbind = [] }

  return { memos, sync, saved, revision, applyState, refresh, syncNow, save, remove, bind, unbind: unbindAll }
}

export type MemoStore = ReturnType<typeof createMemos>

let _app: MemoStore | null = null
/** 앱 전체가 함께 쓰는 메모 목록. 처음 부를 때 memoState 를 듣기 시작하고 목록을 당겨 온다. */
export function useMemos(): MemoStore {
  if (!_app) {
    _app = createMemos({ onBackendEvent, onBackendBound, requestAction })
    _app.bind()
  }
  return _app
}
