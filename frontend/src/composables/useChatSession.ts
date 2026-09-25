import { computed, ref, watch } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { applyGenerationEvent, type ChatArtifact, type GenerationRequest, type GenerationState } from '../utils/chatGeneration'
import { createRefCountedSingleton } from '../utils/refCountedSingleton'
import { useChatPreferences } from './useChatPreferences'

/**
 * 대화 목록 · 보내기 · 스트리밍 받기 · 파일 저장.
 *
 * views/ChatView.vue 에서 **위치만 옮겼다** — 우하단 도크의 작은 대화 패널
 * (components/dock/ChatMiniPanel.vue)이 대화 탭과 같은 대화 목록을 쓰고, 한쪽에서 보낸 답이
 * 다른 쪽에도 흐르게 하려고. 상태는 Python 파일(config/chat_threads.json)에 저장한다 —
 * localStorage 는 이미지가 붙는 순간 5MB 를 넘겨 조용히 실패한다. 브리지: chat_load/chat_save/
 * chat_send/chat_stop, 시그널 chatThreads/chatToken/chatDone/chatGenerationEvent
 * (tests/test_bridge_contract.py 가 이름을 지킨다).
 *
 * 화면 일(스크롤 따라가기 · 입력칸 포커스)은 각 화면이 `onChatActivity` 로 받아서 한다 — 두 화면이
 * 동시에 떠 있어도 서로의 포커스를 뺏지 않게, 여기서는 DOM 을 만지지 않는다.
 */

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: number
  images?: string[]
  model?: string
  thinking?: string
  evalCount?: number
  durationMs?: number
  doneReason?: string
  pending?: boolean
  requestId?: string
  error?: string
  structured?: boolean
  artifacts?: ChatArtifact[]
  generationRequest?: GenerationRequest
  generation?: GenerationState
}
export interface ChatThread {
  id: string
  title: string
  model: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
}

/**
 * 화면이 따라 할 일.
 * - `thread-opened`: 저절로(불러오기 · 빈 목록 · 마지막 대화 삭제) 빈 대화가 열렸다 — 대화 탭은 입력칸에 포커스.
 * - `asked`: 새 요청을 보냈다 — 맨 아래를 따라가기 시작.
 * - `stream`: 답이 자랐다(토큰 · 생각 · 생성 진행 · 끝) — 맨 아래를 보고 있었으면 계속 맨 아래.
 */
export type ChatActivity = 'thread-opened' | 'asked' | 'stream'

export const chatUid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8)

function createChatSession() {
  const prefs = useChatPreferences()
  const threads = ref<ChatThread[]>([])
  const activeId = ref('')
  const busyId = ref('')
  const listeners = new Set<(kind: ChatActivity) => void>()
  function notify(kind: ChatActivity) {
    for (const listener of [...listeners]) { try { listener(kind) } catch {} }
  }
  /** 화면이 스크롤·포커스를 맞추려고 듣는다. 돌려받은 함수로 해제. */
  function onChatActivity(listener: (kind: ChatActivity) => void) {
    listeners.add(listener)
    return () => { listeners.delete(listener) }
  }

  const active = computed(() => threads.value.find((t) => t.id === activeId.value) || null)
  /** 최근 것이 위 — 대화 탭 목록(검색 필터 전)과 도크 패널의 대화 고르기가 같은 순서를 쓴다. */
  const sortedThreads = computed(() => [...threads.value].sort((a, b) => b.updatedAt - a.updatedAt))

  // ── 저장 — 파일로, 지연해서 ──
  let saveTimer: ReturnType<typeof setTimeout> | null = null
  // 방금 파일에서 읽어 온 목록을 그대로 다시 쓰지 않게, 불러온 직후의 변경 알림 한 번은 건너뛴다
  let skipLoadedSave = false
  function saveNow() {
    if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }
    const snapshot = threads.value.map((t) => ({
      ...t,
      messages: t.messages.filter((m) => !m.pending).map(({ pending, requestId, ...rest }) => rest),
    }))
    requestAction('chat_save', { threads: snapshot })
  }
  function scheduleSave() {
    if (skipLoadedSave) { skipLoadedSave = false; return }
    if (saveTimer) clearTimeout(saveTimer)
    saveTimer = setTimeout(saveNow, 600)
  }
  /** 대기 중인 저장을 지금 보낸다 — 탭 전환·창 닫기 직전 600ms 안의 변경을 잃지 않게. */
  function flushSave() {
    if (saveTimer) saveNow()
  }
  // 안전망: 9곳 넘는 변경 지점(send·onDone·이름 바꾸기·삭제 …)을 빠짐없이 저장한다 — 유지
  watch(threads, scheduleSave, { deep: true })

  // ── 대화 목록 ──
  /** 빈 대화를 연다(이미 빈 대화가 있으면 그걸 쓴다 — 빈 것이 쌓이지 않게). 포커스는 부르는 화면이 한다. */
  function newThread() {
    const empty = threads.value.find((t) => !t.messages.length)
    if (empty) { activeId.value = empty.id; return }
    const t: ChatThread = { id: chatUid(), title: '', model: prefs.model.value, createdAt: Date.now(), updatedAt: Date.now(), messages: [] }
    threads.value.unshift(t)
    activeId.value = t.id
  }
  function openEmptyThread() { newThread(); notify('thread-opened') }
  function stopIfAnswering(thread: ChatThread | undefined) {
    // The target is captured when the dialog opens; another thread stays untouched.
    if (busyId.value && thread?.messages.some(m => m.requestId === busyId.value)) stop()
  }
  /** 대화 내용 비우기(확인은 화면이 받는다). */
  function clearThread(id: string) {
    const thread = threads.value.find(t => t.id === id)
    stopIfAnswering(thread)
    if (thread) { thread.messages = []; thread.title = ''; thread.updatedAt = Date.now() }
  }
  /** 대화 삭제(확인은 화면이 받는다). 마지막 대화였으면 빈 대화를 새로 연다. */
  function removeThread(id: string) {
    const thread = threads.value.find(t => t.id === id)
    stopIfAnswering(thread)
    if (!thread) return
    threads.value = threads.value.filter(t => t.id !== thread.id)
    if (activeId.value === thread.id) activeId.value = threads.value[0]?.id || ''
    if (!threads.value.length) openEmptyThread()
  }

  // ── 보내기 · 받기 ──
  /** 사용자 메시지를 붙이고 답을 요청한다. 스키마 검사 · 입력칸 비우기는 부르는 화면이 한다. */
  function sendMessage(thread: ChatThread, text: string, images: string[], request: GenerationRequest) {
    const user: ChatMessage = { id: chatUid(), role: 'user', content: text, createdAt: Date.now() }
    user.generationRequest = { ...request, hadImage: images.length > 0 }
    if (images.length) user.images = [...images]
    thread.messages.push(user)
    if (!thread.title) thread.title = (text || '이미지').replace(/\s+/g, ' ').slice(0, 36)
    ask(thread)
  }
  function ask(thread: ChatThread, retryRequest?: GenerationRequest) {
    const requestId = chatUid()
    const latestUser = [...thread.messages].reverse().find(m => m.role === 'user')
    const request = { ...(retryRequest || latestUser?.generationRequest || prefs.generationRequest.value) }
    const model = prefs.model.value
    const assistant: ChatMessage = { id: chatUid(), role: 'assistant', content: '', createdAt: Date.now(), pending: true, requestId, model }
    assistant.generationRequest = request
    assistant.structured = prefs.usesStructuredOutput(request)
    thread.messages.push(assistant)
    thread.model = model
    thread.updatedAt = Date.now()
    busyId.value = requestId
    notify('asked')
    requestAction('chat_send', {
      id: requestId,
      ...prefs.requestSettings(request),
      generation: request,
      messages: thread.messages
        .filter((m) => !m.pending && !m.error)
        .map((m) => ({ role: m.role, content: m.content, images: m.images })),
    })
  }
  function stop() {
    if (!busyId.value) return
    requestAction('chat_stop', { id: busyId.value })
  }
  /**
   * 마지막 답을 같은 요청으로 다시 받는다. `check` 는 그 요청을 보내도 되는지(스키마 초안 등)를
   * 화면이 판단한다 — 안 되면 false 를 돌려주고 스스로 알린다.
   */
  function regenerate(check: (request: GenerationRequest) => boolean = () => true) {
    const thread = active.value
    if (!thread || busyId.value) return
    const last = thread.messages[thread.messages.length - 1]
    const request = last?.generationRequest || [...thread.messages].reverse().find(m => m.role === 'user')?.generationRequest || prefs.generationRequest.value
    if (!check(request)) return
    if (last?.role === 'assistant') thread.messages.pop()
    ask(thread, request)
  }
  function findPending(requestId: string): ChatMessage | null {
    for (const t of threads.value) {
      const m = t.messages.find((x) => x.requestId === requestId)
      if (m) return m
    }
    return null
  }
  function onToken(json: string) {
    try {
      const { id, text, thinking } = JSON.parse(json)
      const m = findPending(id)
      if (!m) return
      if (thinking) m.thinking = (m.thinking || '') + thinking
      if (!text) { if (thinking) notify('stream'); return }
      m.content += text
      notify('stream')
    } catch {}
  }
  function onDone(json: string) {
    try {
      const d = JSON.parse(json)
      const m = findPending(d.id)
      if (m) {
        if (typeof d.content === 'string' && d.content.length >= m.content.length) m.content = d.content
        if (!d.ok) m.error = d.error || '응답을 받지 못했습니다'
        else if (d.stopped) m.error = m.content ? '' : '중지됨'
        if (typeof d.evalCount === 'number') m.evalCount = d.evalCount
        if (typeof d.durationMs === 'number') m.durationMs = d.durationMs
        if (typeof d.doneReason === 'string' && d.doneReason) m.doneReason = d.doneReason
        m.pending = false
        delete m.requestId
        const t = threads.value.find((x) => x.messages.includes(m))
        if (t) t.updatedAt = Date.now()
        // 끝나면 액션 줄·메타가 생기고 생각 블록이 접혀 높이가 바뀐다 — 보고 있던 맨 아래를 지킨다
        notify('stream')
      }
      if (busyId.value === d.id) busyId.value = ''
    } catch { busyId.value = '' }
  }
  function onGeneration(json: string) {
    try {
      const event = JSON.parse(json)
      const message = findPending(event.id)
      if (!message || !applyGenerationEvent(message, event)) return
      // Auto mode may route to media after sending: this is not a JSON response.
      message.structured = false
      if (event.model) message.model = String(event.model)
      const thread = threads.value.find(t => t.messages.includes(message))
      if (thread) thread.updatedAt = Date.now()
      if (event.done && busyId.value === event.id) busyId.value = ''
      notify('stream')
    } catch { /* unrelated or malformed events do not release another request */ }
  }

  // ── 시작 — 예전 ChatView onMounted 의 대화 부분 ──
  const unsubs: Array<() => void> = []
  unsubs.push(onBackendEvent('chatThreads', (json: string) => {
    try {
      const list = JSON.parse(json)
      if (Array.isArray(list)) {
        // 파일에서 막 읽은 목록 — 이 대입으로 생기는 저장 한 번은 건너뛴다(같은 내용 재기록 방지).
        // 불러오기가 목록을 통째로 바꾸므로 그 전에 걸린 저장도 의미가 없다.
        if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }
        skipLoadedSave = true
        threads.value = list
      }
    } catch {}
    // GemmaStudio 처럼 열 때마다 빈 새 대화에서 시작한다 — 기존 기록은 목록에 남는다
    if (!activeId.value) openEmptyThread()
  }))
  unsubs.push(onBackendEvent('chatToken', onToken))
  unsubs.push(onBackendEvent('chatDone', onDone))
  unsubs.push(onBackendEvent('chatGenerationEvent', onGeneration))
  window.addEventListener('beforeunload', flushSave)
  requestAction('chat_load')
  // 백엔드가 목록을 안 돌려줘도(웹 모드·개발 서버) 빈 대화 하나는 있어야 입력이 된다
  const emptyFallback = setTimeout(() => { if (!threads.value.length) openEmptyThread() }, 1500)

  function dispose() {
    clearTimeout(emptyFallback)
    unsubs.forEach((u) => { try { u() } catch {} })
    window.removeEventListener('beforeunload', flushSave)
    flushSave()   // 예전엔 타이머만 지워 마지막 600ms 안의 변경이 사라졌다
    listeners.clear()
  }

  return {
    api: {
      threads, activeId, busyId, active, sortedThreads,
      onChatActivity, newThread, clearThread, removeThread,
      sendMessage, stop, regenerate, flushSave,
    },
    dispose,
  }
}

export type ChatSession = ReturnType<typeof createChatSession>['api']

/** 대화 목록 — 대화 탭과 도크 대화 패널이 같은 것을 쓴다(쓰는 곳이 있는 동안만 산다). */
export const useChatSession = createRefCountedSingleton(createChatSession)
