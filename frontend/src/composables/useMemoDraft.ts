import { computed, onScopeDispose, shallowRef, watch, type Ref } from 'vue'
import {
  ackWaitMs, canSendDraft, draftFromMemo, emptyDraft, newMemoId, reconcileDraft, retryDraft, sendDraft,
  type MemoDraft, type MemoSentSave,
} from '../utils/memos'
import type { MemoItem, MemoSaveResult, MemoSavePayload } from '../types/bridge'

/**
 * 메모장에서 고르고 있는 메모의 초안 + 자동 저장(입력이 멈추고 ~600ms 뒤).
 *
 * 한 메모에 저장은 한 번에 하나만 보낸다 — 앞선 저장의 확인(memoState)이 오기 전에 또 보내면 둘째 저장의
 * base_updated_at 이 옛것이라 백엔드가 충돌로 보고 사본을 만든다. 그래서 다른 메모로 옮기거나 · 새 메모 ·
 * 패널을 닫을 때도 확인 대기 중이면 보내지 않고, 그 초안을 `parked` 에 두었다가 확인이 오면 새 base 로
 * 보낸다. 확인이 끝내 안 오면 목록을 다시 청하고(refresh = memo_list) 그 답을 보고 **같은 저장을 똑같이**
 * 다시 보낸다 — 새 편집을 옛 base 로 보내지 않는다. 페이지가 닫힐 때만은 기다릴 수 없어 그냥 보낸다
 * (드물게 충돌 사본이 생겨도 글은 남는다).
 * 저장마다 이 편집기의 id 와 새 request id 를 싣고, 충돌 사본으로 옮겨 가는 것은 그 저장의 답(saved)으로만 한다.
 * 서버 목록과 맞추는 판단은 utils/memos.reconcileDraft(순수 함수, 테스트 있음).
 */
export interface MemoDraftDeps {
  /** 서버 목록(useMemos().memos) */
  memos: Ref<MemoItem[]>
  /** memoState 를 받을 때마다 바뀌는 값(useMemos().revision) */
  revision: Ref<number>
  /** 그 memoState 에 실린 저장 답(useMemos().saved) — 보낸 저장이 충돌 사본이 됐는지 */
  saved: Ref<MemoSaveResult | null>
  save: (payload: MemoSavePayload) => void
  remove: (id: string) => void
  /** 목록을 다시 청한다(useMemos().refresh = memo_list) — 확인이 늦은 저장을 가려낼 때 */
  refresh?: () => void
  debounceMs?: number
  now?: () => number
  newId?: () => string
  /** 편집기 id · 저장 request id 를 만든다(테스트용) */
  newToken?: () => string
}

export function useMemoDraft(deps: MemoDraftDeps) {
  const debounceMs = deps.debounceMs ?? 600
  const now = deps.now ?? Date.now
  const newId = deps.newId ?? (() => newMemoId())
  const newToken = deps.newToken ?? (() => newMemoId())
  /** 이 화면(편집기)의 id — 다른 창 · 웹 클라이언트의 저장과 백엔드가 가른다 */
  const editor = newToken()
  // 초안은 통째로 바꿔 끼운다(얕은 ref) — 보낸 저장 객체가 프록시로 감싸이지 않아 probed 가 그대로 알아본다
  /** 고르고 있는 메모의 초안 */
  const draft = shallowRef<MemoDraft | null>(null)
  /** 고르지 않은 메모 중 저장이 끝나지 않은 초안 — 확인 대기 중이거나, 확인 뒤 보낼 편집이 있다 */
  const parked = shallowRef<MemoDraft[]>([])
  const selectedId = computed(() => draft.value?.id ?? '')
  let timer: ReturnType<typeof setTimeout> | null = null
  let probeTimer: ReturnType<typeof setTimeout> | null = null
  /** 이미 목록을 다시 청한 저장(다시 보내면 새 객체가 되어 다시 청할 수 있다) */
  const probed = new WeakSet<MemoSentSave>()
  /** 상태를 다 바꾼 뒤에 보낸다 — 백엔드가 같은 턴에 memoState 로 답해도 새 상태로 맞추게 */
  const outbox: MemoSavePayload[] = []

  function clearTimer() { if (timer) { clearTimeout(timer); timer = null } }
  function schedule(ms = debounceMs) {
    clearTimer()
    timer = setTimeout(flush, ms)
  }

  function send(d: MemoDraft): MemoDraft {
    const { payload, draft: next } = sendDraft(d, now(), { editor, request: newToken() })
    outbox.push(payload)
    return next
  }
  function resend(d: MemoDraft): MemoDraft {
    const retried = retryDraft(d, now())
    if (!retried) return d
    outbox.push(retried.payload)
    return retried.draft
  }
  function drain() {
    while (outbox.length) deps.save(outbox.shift()!)
    armProbe()
  }

  /** 확인을 기다리는 저장 중 가장 먼저 기다림이 끝나는 때 목록을 다시 청한다(저장 하나에 한 번). */
  function armProbe() {
    if (probeTimer) { clearTimeout(probeTimer); probeTimer = null }
    let due = Infinity
    for (const d of [draft.value, ...parked.value]) {
      if (d?.sent && !probed.has(d.sent)) due = Math.min(due, d.sent.at + ackWaitMs(d.sent))
    }
    if (!Number.isFinite(due)) return
    probeTimer = setTimeout(probe, Math.max(0, due - now()))
  }
  function probe() {
    probeTimer = null
    const t = now()
    let overdue = false
    for (const d of [draft.value, ...parked.value]) {
      if (!d?.sent || probed.has(d.sent)) continue
      if (t - d.sent.at >= ackWaitMs(d.sent)) { probed.add(d.sent); overdue = true }
    }
    // 답(memoState)이 오면 watch 가 확인하거나 · 사본으로 옮기거나 · 같은 저장을 다시 보낸다
    if (overdue) deps.refresh?.()
    armProbe()
  }

  /** 디바운스가 끝났을 때 · 옮기거나 닫을 때 — 보낼 편집이 있고 확인 대기 중이 아니면 지금 보낸다. */
  function flush() {
    clearTimer()
    const current = draft.value
    if (!current || !canSendDraft(current)) return   // 확인 대기 중이면 확인이 올 때 watch 가 부른다
    draft.value = send(current)
    drain()
  }
  const flushNow = flush

  /** 고른 초안을 내려놓는다 — 보낼 수 있으면 지금 보내고, 저장이 끝나지 않았으면 parked 에 둔다. */
  function park(d: MemoDraft | null) {
    clearTimer()
    if (!d) return
    const next = canSendDraft(d) ? send(d) : d
    const rest = parked.value.filter(p => p.id !== next.id)
    parked.value = next.dirty || next.sent ? [...rest, next] : rest
  }
  /** 메모를 연다 — 저장이 끝나지 않은 초안이 있으면 그것을(서버 목록은 아직 옛 내용이다). */
  function open(memo: MemoItem | undefined, id = memo?.id): MemoDraft | null {
    const waiting = id ? parked.value.find(p => p.id === id) : undefined
    if (waiting) {
      parked.value = parked.value.filter(p => p !== waiting)
      return waiting
    }
    return memo ? draftFromMemo(memo) : null
  }

  function edit(patch: Partial<Pick<MemoDraft, 'title' | 'text'>>) {
    if (!draft.value) return
    draft.value = { ...draft.value, ...patch, dirty: true }
    schedule()
  }
  function select(id: string) {
    if (draft.value?.id === id) return
    park(draft.value)
    draft.value = open(deps.memos.value.find(m => m.id === id), id)
    if (draft.value && canSendDraft(draft.value)) schedule()
    drain()
  }
  /** 새 메모 — 이미 빈 메모가 있으면 그걸 연다(빈 것이 쌓이지 않게). 첫 입력 때 저장된다. */
  function createNew() {
    const current = draft.value
    if (current && !current.title.trim() && !current.text.trim()) return
    park(current)
    const empty = deps.memos.value.find(m => !m.title.trim() && !m.text.trim() && !parked.value.some(p => p.id === m.id))
    draft.value = empty ? draftFromMemo(empty) : emptyDraft(newId())
    drain()
  }
  /** 고르고 있는 메모를 지운다. 한 번도 저장 안 된 새 메모는 초안만 버린다. */
  function removeSelected() {
    const current = draft.value
    if (!current) return
    clearTimer()
    draft.value = null
    if (current.base !== null || current.sent) deps.remove(current.id)
    armProbe()
  }
  function clear() {
    park(draft.value)
    draft.value = null
    drain()
  }
  /** 페이지가 닫힐 때 — 확인을 기다릴 수 없으니 밀린 편집을 전부 지금 보낸다(글을 잃지 않는 쪽). */
  function flushForUnload() {
    clearTimer()
    if (draft.value?.dirty) draft.value = send(draft.value)
    parked.value = parked.value.map(p => (p.dirty ? send(p) : p))
    drain()
  }

  // 서버 상태가 올 때마다 맞춘다 — 보낸 저장의 확인 · 충돌 사본 · 다른 곳(Forge)에서 바뀐 내용 · 삭제
  watch(deps.revision, () => {
    const t = now()
    const memos = deps.memos.value
    const saved = deps.saved.value
    const current = draft.value
    if (current) {
      const { draft: next, change } = reconcileDraft(current, memos, t, saved)
      if (change === 'gone') { clearTimer(); draft.value = null }
      else if (change === 'retry') draft.value = resend(next)
      else {
        draft.value = next
        if ((change === 'acked' || change === 'moved') && canSendDraft(next)) schedule()
      }
    }
    if (parked.value.length) {
      const kept: MemoDraft[] = []
      for (const p of parked.value) {
        const { draft: next, change } = reconcileDraft(p, memos, t, saved)
        if (change === 'gone' || change === 'replaced') continue
        let d = change === 'retry' ? resend(next) : next
        if (canSendDraft(d)) d = send(d)                 // 확인이 왔다 — 밀린 편집을 새 base 로
        if (d.dirty || d.sent) kept.push(d)
      }
      parked.value = kept
    }
    drain()
  }, { flush: 'sync' })

  const onUnload = () => flushForUnload()
  window.addEventListener('beforeunload', onUnload)
  onScopeDispose(() => {
    window.removeEventListener('beforeunload', onUnload)
    flushForUnload()
    clearTimer()
    if (probeTimer) { clearTimeout(probeTimer); probeTimer = null }
  })

  return { draft, parked, selectedId, edit, select, createNew, removeSelected, clear, flush, flushNow }
}
