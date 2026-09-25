/**
 * 메모장(우하단 도크 · components/dock/MemoPanel.vue)의 순수 로직 — Vue 없이 테스트한다.
 *
 * 메모는 앱(user_data/memos.json)에 저장되고, 메모 라우트가 있는 Forge(sam-extra Notebook)에 닿으면
 * 백엔드가 합친다. 화면이 하는 일은 셋: memoState 를 안전하게 읽기, 목록 정렬 · 표시 문구,
 * 그리고 편집 중인 초안과 서버 저장본 맞추기(reconcileDraft) — 내가 보낸 저장의 확인인지,
 * 그 저장이 충돌 사본이 됐는지, 다른 곳(Forge)에서 바뀐 것인지, 지워진 것인지.
 *
 * 저장 규칙(core/memo_store.MemoStore.save 와 짝): base_updated_at 이 저장본보다 낡았고 내용이 다르면
 * 백엔드는 저장본을 그대로 두고 보낸 글을 새 id 의 "<제목> (충돌 사본)" 으로 남긴다. 그래서
 *   - 한 메모에 저장은 한 번에 하나만 보낸다. 확인 전에 또 보내면 둘째 저장의 base 가 옛것이라 사본이 된다.
 *   - 확인이 늦으면 **같은 저장을 똑같이** 다시 보낸다 — 앞선 것이 늦게 처리돼도 둘째는 '같은 내용'이라
 *     사본이 생기지 않는다. 그 사이의 새 편집은 확인 뒤에 보낸다.
 *   - 보낸 저장이 충돌 사본이 됐으면 초안이 그 사본으로 옮겨 간다 — 이후 저장은 사본을 고친다
 *     (안 옮기면 저장마다 옛 base 로 나가 사본이 끝없이 늘었다). 어느 사본인지는 백엔드가 그 저장의 답
 *     (memoState.saved — 저장마다 붙인 request 로 짝짓는다)으로만 알려 준다. 본문 · 제목 꼬리로 짐작하면 본문이
 *     같은 남의 메모 사본(제목만 있는 메모 둘 등)으로 옮겨 가 이후 편집이 다른 메모에 저장됐다.
 *   - 저장마다 편집기(editor, 화면 인스턴스)를 싣는다 — 백엔드는 같은 편집기의 이어 친 글만 그 사본에 모으고
 *     다른 편집기(웹 클라이언트 둘 · 탭 둘)의 글은 서로의 사본을 덮지 않게 따로 남긴다.
 */
import type { MemoItem, MemoSaveResult, MemoSavePayload, MemoStateEvent, MemoSyncState } from '../types/bridge'
import { relativeTimeKo } from './relativeTime'

export const MEMO_TITLE_MAX = 120
export const MEMO_TEXT_MAX = 100_000
export const MEMO_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,79}$/
/** 백엔드가 양쪽이 바뀐 충돌에서 로컬 쪽 사본에 붙이는 꼬리. */
export const MEMO_CONFLICT_SUFFIX = '(충돌 사본)'
/** 보낸 저장의 확인(memoState)을 이만큼 기다린다. 넘으면 목록을 다시 청하고 같은 저장을 다시 보낸다. */
export const MEMO_ACK_TIMEOUT_MS = 5000
/** 다시 보낼수록 기다림을 두 배로(5 → 10 → 20 → 40초에서 멈춤) — 저장소가 계속 실패해도 두드리지 않게. */
const MEMO_ACK_BACKOFF_MAX_STEPS = 3

/** memoState 를 아직 한 번도 못 받았을 때 — 백엔드가 메모를 모르거나(옛 버전) 아직 답이 없다. */
export const UNKNOWN_SYNC: MemoSyncState = { available: false, target: 'local', syncing: false, last_synced_at: null, error: null }

/** 새 메모 id — 계약의 id 모양(영숫자로 시작, 영숫자 · . _ : - 최대 80자)을 따른다. */
export function newMemoId(now: number = Date.now(), random: () => number = Math.random): string {
  const tail = Math.floor(random() * 36 ** 6).toString(36).padStart(6, '0')
  return `m${now.toString(36)}-${tail}`
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

/**
 * 계약 길이로 자른다 — 백엔드(파이썬)처럼 **코드 포인트** 수로. UTF-16 으로 자르면 이모지가 든
 * 제목에서 '(충돌 사본)' 꼬리가 잘려 사본을 못 알아보고, 긴 본문은 끝이 잘린 채 다시 저장된다.
 */
function clampCodePoints(value: string, max: number): string {
  if (value.length <= max) return value          // UTF-16 길이 ≥ 코드 포인트 수
  return Array.from(value).slice(0, max).join('')
}

/** 메모 하나 — 모양이 틀리면 null(목록에서 뺀다). 제목 · 본문은 계약 길이로 자른다. */
export function normalizeMemo(raw: unknown): MemoItem | null {
  if (!raw || typeof raw !== 'object') return null
  const item = raw as Record<string, unknown>
  const id = text(item.id)
  if (!MEMO_ID_PATTERN.test(id) || item.deleted === true) return null
  const updated = text(item.updated_at)
  return {
    id,
    title: clampCodePoints(text(item.title), MEMO_TITLE_MAX),
    text: clampCodePoints(text(item.text), MEMO_TEXT_MAX),
    created_at: text(item.created_at) || updated,
    updated_at: updated,
  }
}

function timeOf(iso: string): number {
  const t = Date.parse(iso)
  return Number.isFinite(t) ? t : 0
}

/** 최근에 고친 것이 위. 같으면 최근에 만든 것, 그다음 id — 순서가 흔들리지 않게. */
export function sortMemos(list: readonly MemoItem[]): MemoItem[] {
  return [...list].sort((a, b) =>
    timeOf(b.updated_at) - timeOf(a.updated_at)
    || timeOf(b.created_at) - timeOf(a.created_at)
    || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
}

function normalizeSync(raw: unknown): MemoSyncState {
  const s = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  return {
    available: s.available === true,
    target: s.target === 'forge' ? 'forge' : 'local',
    syncing: s.syncing === true,
    last_synced_at: typeof s.last_synced_at === 'string' && s.last_synced_at ? s.last_synced_at : null,
    error: typeof s.error === 'string' && s.error ? s.error : null,
  }
}

function normalizeSaved(raw: unknown): MemoSaveResult | null {
  if (!raw || typeof raw !== 'object') return null
  const s = raw as Record<string, unknown>
  const request = text(s.request)
  const id = text(s.id)
  if (!request || !MEMO_ID_PATTERN.test(id)) return null
  return { request, id, conflict_of: text(s.conflict_of) }
}

/** memoState JSON — 깨졌으면 null. 모양이 틀린 메모 · 중복 id 는 빼고, 정렬해서 돌려준다. */
export function parseMemoState(json: string): MemoStateEvent | null {
  let data: unknown
  try { data = JSON.parse(json) } catch { return null }
  if (!data || typeof data !== 'object' || !Array.isArray((data as { memos?: unknown }).memos)) return null
  const seen = new Set<string>()
  const memos: MemoItem[] = []
  for (const raw of (data as { memos: unknown[] }).memos) {
    const memo = normalizeMemo(raw)
    if (!memo || seen.has(memo.id)) continue
    seen.add(memo.id)
    memos.push(memo)
  }
  return {
    memos: sortMemos(memos),
    sync: normalizeSync((data as { sync?: unknown }).sync),
    saved: normalizeSaved((data as { saved?: unknown }).saved),
  }
}

/** 충돌에서 남긴 로컬 사본인가 — 목록에서 따로 표시한다. */
export function isConflictCopy(title: string): boolean {
  return title.trim().endsWith(MEMO_CONFLICT_SUFFIX)
}

/** 목록에 보일 이름 — 제목, 없으면 본문 첫 줄, 둘 다 없으면 '빈 메모'. */
export function memoDisplayTitle(memo: Pick<MemoItem, 'title' | 'text'>): string {
  const title = memo.title.trim()
  if (title) return title
  const firstLine = memo.text.split('\n').map(line => line.trim()).find(Boolean) || ''
  return firstLine ? firstLine.slice(0, 40) : '빈 메모'
}

export type MemoSyncTone = 'ok' | 'busy' | 'warn' | 'muted'

/** 패널 머리의 동기화 상태 한 줄. sync 가 null 이면 memoState 를 아직 못 받은 것. */
export function memoSyncLabel(sync: MemoSyncState | null, now: number = Date.now()): { text: string; tone: MemoSyncTone; detail: string } {
  if (!sync) return { text: '메모 저장소 확인 중…', tone: 'muted', detail: '앱 백엔드의 답을 기다리는 중입니다' }
  if (sync.syncing) return { text: 'Forge 와 동기화 중…', tone: 'busy', detail: '' }
  if (!sync.available || sync.target !== 'forge') {
    return { text: '로컬에만 저장됨', tone: 'muted',
      detail: sync.error || 'Forge 에 메모 기능(sam-extra Notebook)이 없거나 연결되지 않았습니다. 연결되면 합칩니다.' }
  }
  if (sync.error) return { text: '동기화 오류 · 로컬에 저장됨', tone: 'warn', detail: sync.error }
  const at = sync.last_synced_at ? Date.parse(sync.last_synced_at) : NaN
  return { text: Number.isFinite(at) ? `Forge 와 동기화됨 · ${relativeTimeKo(at, Math.max(now, at))}` : 'Forge 와 동기화됨', tone: 'ok', detail: '' }
}

// ── 편집 초안 ────────────────────────────────────────────────────────────────

/** 보냈지만 memoState 로 확인받지 못한 저장. 다시 보낼 때는 이것을 **그대로** 보낸다. */
export interface MemoSentSave {
  title: string
  text: string
  /** 보낼 때 실은 base_updated_at */
  base: string | null
  /** 보낸(다시 보낸) 시각 — 확인을 기다리는 기준 */
  at: number
  /** 확인 없이 다시 보낸 횟수 */
  tries: number
  /** 이 저장의 식별자 — 백엔드가 memoState.saved.request 로 돌려준다(다시 보내도 같은 값) */
  request: string
  /** 보낸 편집기(화면 인스턴스) — 다시 보낼 때도 똑같이 싣는다 */
  editor: string
}

/** 저장에 붙이는 식별자 — 편집기 하나 · 저장 하나마다. */
export interface MemoSaveTags {
  editor: string
  request: string
}

export interface MemoDraft {
  id: string
  title: string
  text: string
  /** 이 편집이 기대는 저장본의 updated_at(확인받은 것) — 한 번도 저장 안 된 새 메모는 null */
  base: string | null
  /** 아직 보내지 않은 편집이 있다 */
  dirty: boolean
  /** 확인 대기 중인 저장 — 있는 동안은 다음 저장을 보내지 않는다 */
  sent: MemoSentSave | null
}

export function draftFromMemo(memo: MemoItem): MemoDraft {
  return { id: memo.id, title: memo.title, text: memo.text, base: memo.updated_at, dirty: false, sent: null }
}

export function emptyDraft(id: string): MemoDraft {
  return { id, title: '', text: '', base: null, dirty: false, sent: null }
}

function sameContent(a: { title: string; text: string }, b: { title: string; text: string }) {
  return a.title.trim() === b.title.trim() && a.text === b.text
}

/** 확인을 기다리는 시간 — 다시 보낼수록 길게. */
export function ackWaitMs(sent: Pick<MemoSentSave, 'tries'>): number {
  return MEMO_ACK_TIMEOUT_MS * 2 ** Math.min(Math.max(sent.tries, 0), MEMO_ACK_BACKOFF_MAX_STEPS)
}

/** 보낸 저장의 확인을 기다린 시간이 넘었나. */
export function isSaveOverdue(sent: MemoSentSave, now: number): boolean {
  return now - sent.at >= ackWaitMs(sent)
}

/**
 * 지금 저장을 보내도 되나 — 보낼 편집이 있고, 앞선 저장의 확인을 기다리는 중이 아닐 때만.
 * 시간이 지났다고 새 편집을 옛 base 로 보내지 않는다(앞선 저장이 늦게 처리되면 사본이 된다).
 */
export function canSendDraft(draft: MemoDraft): boolean {
  return draft.dirty && !draft.sent
}

/**
 * 보낼 페이로드와 '보냄' 표시가 붙은 초안. 제목 · 본문은 계약 길이로 자른다.
 * `tags` = 편집기 id(화면 인스턴스마다 하나)와 이 저장의 request id(저장마다 새로).
 */
export function sendDraft(draft: MemoDraft, now: number, tags: MemoSaveTags): { payload: MemoSavePayload; draft: MemoDraft } {
  const title = clampCodePoints(draft.title, MEMO_TITLE_MAX)
  const body = clampCodePoints(draft.text, MEMO_TEXT_MAX)
  const { editor, request } = tags
  return {
    payload: { id: draft.id, title, text: body, base_updated_at: draft.base, editor, request },
    draft: { ...draft, dirty: false, sent: { title, text: body, base: draft.base, at: now, tries: 0, request, editor } },
  }
}

/**
 * 확인이 끝내 안 온 저장을 **똑같이** 다시 보낸다(같은 내용 · 같은 base). 앞선 것이 늦게라도 처리됐으면
 * 백엔드는 같은 내용이라 아무것도 안 하고, 잃어버렸으면 이것이 대신 저장된다. 그 사이의 새 편집(dirty)은
 * 확인 뒤에 보낸다.
 */
export function retryDraft(draft: MemoDraft, now: number): { payload: MemoSavePayload; draft: MemoDraft } | null {
  const sent = draft.sent
  if (!sent) return null
  return {
    payload: { id: draft.id, title: sent.title, text: sent.text, base_updated_at: sent.base, editor: sent.editor, request: sent.request },
    draft: { ...draft, sent: { ...sent, at: now, tries: sent.tries + 1 } },
  }
}

/**
 * 보낸 저장이 만든 충돌 사본 — 백엔드가 **그 저장의 답**(saved.request = sent.request)으로 알려 준 사본만.
 * 본문이 같다고 남의 사본을 제 것으로 보지 않는다.
 */
export function conflictCopyOf(memos: readonly MemoItem[], draftId: string, sent: MemoSentSave, saved: MemoSaveResult | null): MemoItem | undefined {
  if (!saved || saved.request !== sent.request || saved.conflict_of !== draftId || saved.id === draftId) return undefined
  return memos.find(m => m.id === saved.id)
}

/** 초안을 충돌 사본으로 옮긴다 — 보낸 뒤에 더 친 글은 그대로 두고(다음 저장이 사본을 고친다). */
function moveToCopy(draft: MemoDraft, sent: MemoSentSave, copy: MemoItem): MemoDraft {
  // 보낸 뒤 제목을 안 고쳤으면 사본의 제목('… (충돌 사본)')을 쓴다 — 다음 저장이 꼬리를 떼지 않게
  const title = draft.title === sent.title ? copy.title : draft.title
  const text = draft.text
  return { id: copy.id, title, text, base: copy.updated_at, dirty: title !== copy.title || text !== copy.text, sent: null }
}

export type DraftChange = 'none' | 'acked' | 'moved' | 'retry' | 'replaced' | 'gone'

/**
 * 서버 목록(memoState 의 memos)과 초안을 맞춘다. `saved` = 같은 memoState 에 실린 저장 답(없으면 null).
 * - acked: 보낸 저장이 반영됐다 — 다음 저장은 새 updated_at 에 기댄다.
 * - moved: 보낸 저장이 base 가 낡아 충돌 사본이 됐다(그 저장의 답이 알려 줬다) — 초안이 그 사본(새 id)으로 옮겨 간다.
 * - retry: 확인을 기다린 시간이 넘었는데 반영된 흔적이 없다 — 같은 저장을 다시 보내야 한다(retryDraft).
 * - replaced: 편집 중이 아니었는데 저장본이 바뀌었다(Forge 등 다른 곳) — 초안을 저장본으로 바꾼다.
 * - gone: 저장된 적 있는 메모가 목록에서 사라졌다(다른 곳에서 삭제) — 편집 중이 아니면 화면이 선택을 푼다.
 * - none: 그대로(편집 중 · 확인 대기 중 · 아직 저장 전).
 * 보내지 않은 편집(dirty)이 있으면 저장본이 달라도 초안을 지킨다 — 다음 저장의 base 가 옛것이라 백엔드가
 * 충돌로 보고 둘 다 남기고, 그 memoState 에서 초안이 사본으로 옮겨 간다(moved). 사본은 하나만 생긴다.
 */
export function reconcileDraft(draft: MemoDraft, memos: readonly MemoItem[], now: number, saved: MemoSaveResult | null = null): { draft: MemoDraft; change: DraftChange } {
  const server = memos.find(m => m.id === draft.id)
  const sent = draft.sent
  if (sent) {
    const copy = conflictCopyOf(memos, draft.id, sent, saved)
    if (copy) return { draft: moveToCopy(draft, sent, copy), change: 'moved' }
    if (server && sameContent(server, sent)) {
      return { draft: { ...draft, base: server.updated_at, sent: null }, change: 'acked' }
    }
    return { draft, change: isSaveOverdue(sent, now) ? 'retry' : 'none' }
  }
  if (!server) return { draft, change: draft.base !== null && !draft.dirty ? 'gone' : 'none' }
  if (draft.dirty) return { draft, change: 'none' }
  if (sameContent(server, draft)) {
    if (server.updated_at === draft.base) return { draft, change: 'none' }
    return { draft: { ...draft, base: server.updated_at }, change: 'acked' }
  }
  return { draft: draftFromMemo(server), change: 'replaced' }
}

/**
 * 목록에 초안들(고른 것 + 저장이 끝나지 않은 것)을 겹친다 — 제목을 고치면 목록 이름도 바로 바뀌고,
 * 아직 한 번도 저장 안 된 새 메모도 맨 위에 보인다(서버 목록에는 첫 저장이 확인돼야 들어온다).
 */
export function mergeDraftIntoList(memos: readonly MemoItem[], drafts: MemoDraft | null | ReadonlyArray<MemoDraft | null>): MemoItem[] {
  const all = (drafts === null ? [] : 'id' in drafts ? [drafts] : drafts).filter((d): d is MemoDraft => !!d)
  if (!all.length) return [...memos]
  const byId = new Map(all.map(d => [d.id, d] as const))
  const seen = new Set<string>()
  const list = memos.map(m => {
    const d = byId.get(m.id)
    if (!d) return m
    seen.add(m.id)
    return { ...m, title: d.title, text: d.text }
  })
  const unsaved = all.filter(d => !seen.has(d.id))
    .map(d => ({ id: d.id, title: d.title, text: d.text, created_at: '', updated_at: '' }))
  return [...unsaved, ...list]
}
