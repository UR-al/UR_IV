import { describe, expect, it } from 'vitest'
import {
  MEMO_ACK_TIMEOUT_MS, MEMO_ID_PATTERN, MEMO_TITLE_MAX, ackWaitMs, canSendDraft, conflictCopyOf, draftFromMemo, emptyDraft,
  isConflictCopy, memoDisplayTitle, memoSyncLabel, mergeDraftIntoList, newMemoId, parseMemoState,
  reconcileDraft, retryDraft, sendDraft, sortMemos,
} from './memos'
import type { MemoItem, MemoSyncState } from '../types/bridge'

const memo = (id: string, updated: string, extra: Partial<MemoItem> = {}): MemoItem =>
  ({ id, title: id, text: `${id} body`, created_at: '2026-09-01T00:00:00Z', updated_at: updated, ...extra })

describe('memo ids', () => {
  it('new ids follow the shared contract pattern and differ', () => {
    const a = newMemoId(1_700_000_000_000, () => 0)
    const b = newMemoId(1_700_000_000_000, () => 0.999999)
    expect(a).toMatch(MEMO_ID_PATTERN)
    expect(b).toMatch(MEMO_ID_PATTERN)
    expect(a).not.toBe(b)
    expect(newMemoId()).toMatch(MEMO_ID_PATTERN)
  })
})

describe('parseMemoState', () => {
  it('sorts newest-updated first, drops malformed/duplicate/tombstoned entries and clamps titles', () => {
    const state = parseMemoState(JSON.stringify({
      memos: [
        memo('old', '2026-09-01T10:00:00Z'),
        memo('new', '2026-09-03T10:00:00Z'),
        memo('new', '2026-09-04T10:00:00Z'),                 // 중복 id — 처음 것만
        { id: '-bad', title: 'x', text: '', created_at: '', updated_at: '' },   // id 모양 위반
        { ...memo('dead', '2026-09-05T10:00:00Z'), deleted: true },
        memo('long', '2026-09-02T10:00:00Z', { title: 'x'.repeat(500) }),
        'garbage',
      ],
      sync: { available: true, target: 'forge', syncing: false, last_synced_at: '2026-09-03T10:00:00Z', error: null },
    }))!
    expect(state.memos.map(m => m.id)).toEqual(['new', 'long', 'old'])
    expect(state.memos[1].title).toHaveLength(MEMO_TITLE_MAX)
    expect(state.sync).toEqual({ available: true, target: 'forge', syncing: false, last_synced_at: '2026-09-03T10:00:00Z', error: null })
  })

  it('rejects broken payloads and fills a missing sync block as local-only', () => {
    expect(parseMemoState('{')).toBeNull()
    expect(parseMemoState('{"memos": 3}')).toBeNull()
    expect(parseMemoState('{"memos": []}')!.sync).toEqual({ available: false, target: 'local', syncing: false, last_synced_at: null, error: null })
  })

  it('reads the per-save answer only when it is well formed', () => {
    const answer = { request: 'rq1', id: 'c1', conflict_of: 'm1' }
    expect(parseMemoState(JSON.stringify({ memos: [], saved: answer }))!.saved).toEqual(answer)
    expect(parseMemoState(JSON.stringify({ memos: [], saved: { ...answer, conflict_of: undefined } }))!.saved)
      .toEqual({ request: 'rq1', id: 'c1', conflict_of: '' })
    expect(parseMemoState('{"memos": []}')!.saved).toBeNull()
    expect(parseMemoState(JSON.stringify({ memos: [], saved: { id: 'c1' } }))!.saved).toBeNull()        // request 없음
    expect(parseMemoState(JSON.stringify({ memos: [], saved: { request: 'r', id: '-bad' } }))!.saved).toBeNull()
  })

  it('clamps by code points like the backend, so an emoji title keeps its conflict suffix', () => {
    const title = `${'🙂'.repeat(100)}${' x'.repeat(4)} (충돌 사본)`   // 116 코드 포인트, UTF-16 로는 216
    const state = parseMemoState(JSON.stringify({ memos: [memo('c', '2026-09-01T00:00:00Z', { title })] }))!
    expect(state.memos[0].title).toBe(title)
    expect(isConflictCopy(state.memos[0].title)).toBe(true)
  })

  it('keeps ordering stable for equal timestamps', () => {
    const same = '2026-09-01T00:00:00Z'
    expect(sortMemos([memo('b', same), memo('a', same)]).map(m => m.id)).toEqual(['a', 'b'])
  })
})

describe('display helpers', () => {
  it('names a memo by its title, else its first line, else a placeholder', () => {
    expect(memoDisplayTitle({ title: '  할 일 ', text: 'x' })).toBe('할 일')
    expect(memoDisplayTitle({ title: '', text: '\n\n  첫 줄\n둘째' })).toBe('첫 줄')
    expect(memoDisplayTitle({ title: '', text: '' })).toBe('빈 메모')
  })

  it('recognizes conflict copies made by the sync merge', () => {
    expect(isConflictCopy('장보기 (충돌 사본)')).toBe(true)
    expect(isConflictCopy('장보기')).toBe(false)
  })

  it('labels sync state — local only when Forge has no memo routes', () => {
    const base: MemoSyncState = { available: false, target: 'local', syncing: false, last_synced_at: null, error: null }
    expect(memoSyncLabel(null).tone).toBe('muted')
    expect(memoSyncLabel(base).text).toBe('로컬에만 저장됨')
    expect(memoSyncLabel({ ...base, available: true, target: 'forge', syncing: true }).tone).toBe('busy')
    expect(memoSyncLabel({ ...base, available: true, target: 'forge', error: '연결 거부' })).toMatchObject({ tone: 'warn', detail: '연결 거부' })
    const now = Date.parse('2026-09-25T10:05:00Z')
    expect(memoSyncLabel({ ...base, available: true, target: 'forge', last_synced_at: '2026-09-25T10:00:00Z' }, now))
      .toMatchObject({ text: 'Forge 와 동기화됨 · 5분 전', tone: 'ok' })
  })

  it('overlays the draft on the list and shows a never-saved memo on top', () => {
    const list = [memo('a', '2026-09-02T00:00:00Z'), memo('b', '2026-09-01T00:00:00Z')]
    const editing = { ...draftFromMemo(list[1]), title: '고친 제목', dirty: true }
    expect(mergeDraftIntoList(list, editing).map(m => m.title)).toEqual(['a', '고친 제목'])
    const fresh = { ...emptyDraft('fresh'), text: '새 글' }
    expect(mergeDraftIntoList(list, fresh).map(m => m.id)).toEqual(['fresh', 'a', 'b'])
    expect(mergeDraftIntoList(list, null)).toEqual(list)
  })

  it('overlays every live draft — the selected one and ones still waiting for a save', () => {
    const list = [memo('a', '2026-09-02T00:00:00Z'), memo('b', '2026-09-01T00:00:00Z')]
    const selected = { ...emptyDraft('fresh'), text: '새 글' }
    const waiting = { ...draftFromMemo(list[0]), title: '저장 대기 중', dirty: true }
    expect(mergeDraftIntoList(list, [selected, waiting, null]).map(m => [m.id, m.title]))
      .toEqual([['fresh', ''], ['a', '저장 대기 중'], ['b', 'b']])
  })
})

describe('draft reconciliation', () => {
  const T0 = 1_000_000
  const stored = memo('m1', '2026-09-01T00:00:00Z', { title: '제목', text: '본문' })
  const tags = (request: string) => ({ editor: 'ed1', request })

  it('sends one save at a time and adopts the acknowledged updated_at as the next base', () => {
    let draft = { ...draftFromMemo(stored), text: '본문 추가', dirty: true }
    expect(canSendDraft(draft)).toBe(true)
    const sent = sendDraft(draft, T0, tags('rq1'))
    expect(sent.payload).toEqual({ id: 'm1', title: '제목', text: '본문 추가', base_updated_at: '2026-09-01T00:00:00Z', editor: 'ed1', request: 'rq1' })
    draft = { ...sent.draft, text: '본문 추가 더', dirty: true }
    expect(canSendDraft(draft)).toBe(false)                     // 확인 전엔 다음 저장을 미룬다 — 시간이 지나도
    const ack = reconcileDraft(draft, [{ ...stored, text: '본문 추가', updated_at: '2026-09-01T00:00:05Z' }], T0 + 200)
    expect(ack.change).toBe('acked')
    expect(ack.draft).toMatchObject({ base: '2026-09-01T00:00:05Z', sent: null, dirty: true, text: '본문 추가 더' })
    expect(canSendDraft(ack.draft)).toBe(true)
  })

  it('replaces an idle draft with a newer stored version (changed in Forge)', () => {
    const idle = draftFromMemo(stored)
    const remote = { ...stored, text: 'Forge 에서 고침', updated_at: '2026-09-02T00:00:00Z' }
    const result = reconcileDraft(idle, [remote], T0)
    expect(result.change).toBe('replaced')
    expect(result.draft).toMatchObject({ text: 'Forge 에서 고침', base: '2026-09-02T00:00:00Z', dirty: false })
  })

  it('keeps unsent edits over a remote change so the backend can keep both as a conflict copy', () => {
    const editing = { ...draftFromMemo(stored), text: '내 편집', dirty: true }
    const remote = { ...stored, text: 'Forge 에서 고침', updated_at: '2026-09-02T00:00:00Z' }
    const result = reconcileDraft(editing, [remote], T0)
    expect(result.change).toBe('none')
    expect(result.draft.text).toBe('내 편집')
    expect(result.draft.base).toBe('2026-09-01T00:00:00Z')                 // 옛 base 로 보내야 충돌로 판정된다
  })

  it('moves the draft onto the conflict copy its save produced, keeping what was typed since', () => {
    const sent = sendDraft({ ...draftFromMemo(stored), text: '내 편집', dirty: true }, T0, tags('rq1')).draft
    const typing = { ...sent, text: '내 편집 계속', dirty: true }
    const remote = { ...stored, text: 'Forge 에서 고침', updated_at: '2026-09-02T00:00:00Z' }
    const copy = memo('c1', '2026-09-02T00:00:01Z', { title: '제목 (충돌 사본)', text: '내 편집', created_at: '2026-09-02T00:00:01Z' })
    const answer = { request: 'rq1', id: 'c1', conflict_of: 'm1' }
    const result = reconcileDraft(typing, [copy, remote], T0 + 50, answer)
    expect(result.change).toBe('moved')
    expect(result.draft).toEqual({
      id: 'c1', title: '제목 (충돌 사본)', text: '내 편집 계속', base: '2026-09-02T00:00:01Z', dirty: true, sent: null,
    })
    // 보낸 뒤 더 친 글이 없으면 사본 그대로 — 보낼 것이 없다
    expect(reconcileDraft(sent, [copy, remote], T0 + 50, answer).draft).toMatchObject({ id: 'c1', dirty: false })
    // 답이 없으면(다른 memoState) 사본이 보여도 옮기지 않는다 — 기다리다 같은 저장을 다시 보내 답을 받는다
    expect(reconcileDraft(sent, [copy, remote], T0 + 50).change).toBe('none')
  })

  it("only the answer to this very save moves the draft — never another memo's copy with the same body", () => {
    const b = memo('B', '2026-09-01T00:00:00Z', { title: 'b', text: '' })
    const sent = sendDraft({ ...draftFromMemo(b), title: 'b2', dirty: true }, T0, tags('rqB')).draft.sent!
    const otherCopy = memo('cA', '2026-09-02T00:00:02Z', { title: 'a2 (충돌 사본)', text: '' })
    const list = [otherCopy, b]
    expect(conflictCopyOf(list, 'B', sent, null)).toBeUndefined()
    expect(conflictCopyOf(list, 'B', sent, { request: 'rqA', id: 'cA', conflict_of: 'A' })).toBeUndefined()
    expect(conflictCopyOf(list, 'B', sent, { request: 'rqB', id: 'B', conflict_of: '' })).toBeUndefined()   // 제자리 저장
    const mine = memo('cB', '2026-09-02T00:00:03Z', { title: 'b2 (충돌 사본)', text: '' })
    expect(conflictCopyOf([...list, mine], 'B', sent, { request: 'rqB', id: 'cB', conflict_of: 'B' })?.id).toBe('cB')
  })

  it('never discards an unconfirmed save: after the wait it asks to resend the very same save, backing off', () => {
    const first = sendDraft({ ...draftFromMemo(stored), text: '보낸 글', dirty: true }, T0, tags('rq1'))
    const typing = { ...first.draft, text: '보낸 글 그리고 더', dirty: true }
    const other = { ...stored, text: '다른 글', updated_at: '2026-09-02T00:00:00Z' }
    expect(reconcileDraft(typing, [other], T0 + 10).change).toBe('none')
    const late = reconcileDraft(typing, [other], T0 + MEMO_ACK_TIMEOUT_MS)
    expect(late.change).toBe('retry')
    expect(late.draft.text).toBe('보낸 글 그리고 더')                         // 사용자의 글은 그대로
    const retried = retryDraft(late.draft, T0 + MEMO_ACK_TIMEOUT_MS)!
    expect(retried.payload).toEqual(first.payload)                          // 새 편집이 아니라 같은 저장을 같은 base 로
    expect(retried.draft.sent).toMatchObject({ tries: 1, at: T0 + MEMO_ACK_TIMEOUT_MS })
    expect(retried.draft.dirty).toBe(true)                                  // 새 편집은 확인 뒤에
    expect(ackWaitMs({ tries: 0 })).toBe(MEMO_ACK_TIMEOUT_MS)
    expect(ackWaitMs({ tries: 1 })).toBe(MEMO_ACK_TIMEOUT_MS * 2)
    expect(ackWaitMs({ tries: 9 })).toBe(MEMO_ACK_TIMEOUT_MS * 8)
    expect(retryDraft(draftFromMemo(stored), T0)).toBeNull()
  })

  it('treats a vanished saved memo as deleted elsewhere, but not a new unsaved one', () => {
    expect(reconcileDraft(draftFromMemo(stored), [], T0).change).toBe('gone')
    expect(reconcileDraft({ ...emptyDraft('new1'), text: 'x', dirty: true }, [], T0).change).toBe('none')
    expect(reconcileDraft({ ...draftFromMemo(stored), dirty: true }, [], T0).change).toBe('none')
  })

  it('acknowledges a first save of a new memo', () => {
    const sent = sendDraft({ ...emptyDraft('new1'), title: '새 메모', dirty: true }, T0, tags('rq1'))
    expect(sent.payload.base_updated_at).toBeNull()
    const created = memo('new1', '2026-09-03T00:00:00Z', { title: '새 메모', text: '' })
    expect(reconcileDraft(sent.draft, [created], T0 + 5)).toMatchObject({ change: 'acked', draft: { base: '2026-09-03T00:00:00Z', sent: null } })
  })
})
