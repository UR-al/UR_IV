import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { effectScope, ref, type EffectScope } from 'vue'
import { useMemoDraft } from './useMemoDraft'
import type { MemoItem, MemoSaveResult, MemoSavePayload } from '../types/bridge'

/** 자동 저장(600ms 디바운스) · 한 번에 하나만 보내기 · memoState 로 확인받기 — 실제 타이머 대신 가짜 시계. */
describe('useMemoDraft', () => {
  let scope: EffectScope
  let clock = 0
  const saves: MemoSavePayload[] = []
  const removed: string[] = []
  const memos = ref<MemoItem[]>([])
  const revision = ref(0)
  const saved = ref<MemoSaveResult | null>(null)
  let token = 0

  function serverState(list: MemoItem[]) { memos.value = list; saved.value = null; revision.value++ }
  function setup() {
    token = 0
    return scope.run(() => useMemoDraft({
      memos, revision, saved, save: p => saves.push(p), remove: id => removed.push(id),
      now: () => clock, newId: () => 'fresh1', newToken: () => `tok${token++}`,
    }))!
  }

  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {} })
    scope = effectScope()
    clock = 1000
    saves.length = 0; removed.length = 0
    memos.value = [{ id: 'm1', title: '제목', text: '본문', created_at: 't0', updated_at: '2026-09-01T00:00:00Z' }]
    revision.value = 0
  })
  afterEach(() => { scope.stop(); vi.useRealTimers(); vi.unstubAllGlobals() })

  function tick(ms: number) { clock += ms; vi.advanceTimersByTime(ms) }

  it('debounces typing into one save carrying the stored updated_at as base', () => {
    const d = setup()
    d.select('m1')
    d.edit({ text: '본문 1' }); tick(300)
    d.edit({ text: '본문 12' }); tick(599)
    expect(saves).toHaveLength(0)
    tick(1)
    expect(saves).toEqual([{ id: 'm1', title: '제목', text: '본문 12', base_updated_at: '2026-09-01T00:00:00Z', editor: 'tok0', request: 'tok1' }])
  })

  it('holds the next save until the previous one is acknowledged, then sends it on the new base', () => {
    const d = setup()
    d.select('m1')
    d.edit({ text: 'A' }); tick(600)
    d.edit({ text: 'AB' }); tick(600)
    expect(saves).toHaveLength(1)                 // 확인 전 — 둘째 저장을 보내지 않는다
    serverState([{ ...memos.value[0], text: 'A', updated_at: '2026-09-01T00:00:01Z' }])
    tick(600)
    expect(saves).toHaveLength(2)
    expect(saves[1]).toMatchObject({ text: 'AB', base_updated_at: '2026-09-01T00:00:01Z' })
  })

  it('flushes immediately when switching memos, and a new memo reuses an empty draft', () => {
    const d = setup()
    d.select('m1')
    d.edit({ title: '바뀐 제목' })
    d.createNew()
    expect(saves).toEqual([expect.objectContaining({ id: 'm1', title: '바뀐 제목', text: '본문', base_updated_at: '2026-09-01T00:00:00Z' })])
    expect(d.draft.value).toMatchObject({ id: 'fresh1', base: null })
    d.createNew()                                  // 빈 새 메모에서 또 누르면 그대로
    expect(d.draft.value!.id).toBe('fresh1')
    d.edit({ text: '첫 줄' }); tick(600)
    expect(saves[1]).toEqual({ id: 'fresh1', title: '', text: '첫 줄', base_updated_at: null, editor: 'tok0', request: 'tok2' })
  })

  it('loads a remote change into an idle draft and drops a memo deleted elsewhere', () => {
    const d = setup()
    d.select('m1')
    serverState([{ ...memos.value[0], text: 'Forge 에서 고침', updated_at: '2026-09-02T00:00:00Z' }])
    expect(d.draft.value).toMatchObject({ text: 'Forge 에서 고침', base: '2026-09-02T00:00:00Z' })
    serverState([])
    expect(d.draft.value).toBeNull()
  })

  it('deleting an unsaved new memo only discards the draft; a saved one asks the backend', () => {
    const d = setup()
    d.createNew()
    d.removeSelected()
    expect(removed).toEqual([])
    d.select('m1')
    d.removeSelected()
    expect(removed).toEqual(['m1'])
  })

  it('flushes pending edits when the owner goes away', () => {
    const d = setup()
    d.select('m1')
    d.edit({ text: '닫기 직전' })
    scope.stop()
    expect(saves).toEqual([expect.objectContaining({ id: 'm1', title: '제목', text: '닫기 직전', base_updated_at: '2026-09-01T00:00:00Z' })])
  })
})

/**
 * 백엔드가 core/memo_store.MemoStore.save 처럼 판정할 때 — base 가 낡았고 내용이 다르면 저장본은 두고
 * 보낸 글을 새 id 의 '(충돌 사본)' 으로 남긴다. 액션은 백엔드에서 순서대로 처리된다(queue).
 */
describe('useMemoDraft against MemoStore.save semantics', () => {
  let scope: EffectScope
  let clock = 0
  let seq = 0
  let store: MemoItem[] = []
  let queue: Array<() => void> = []
  let refreshes = 0
  let immediate = false
  const sent: MemoSavePayload[] = []
  const memos = ref<MemoItem[]>([])
  const revision = ref(0)
  const saved = ref<MemoSaveResult | null>(null)
  let token = 0
  const stamp = () => new Date(Date.UTC(2026, 8, 1) + clock).toISOString()

  function emit(result: MemoSaveResult | null = null) { memos.value = store.map(m => ({ ...m })); saved.value = result; revision.value++ }
  /** 백엔드처럼 저장 답(saved)을 그 저장의 request 로 돌려준다 */
  function processSave(p: MemoSavePayload) {
    const s = store.find(m => m.id === p.id)
    const at = stamp()
    const answer = (id: string, of = '') => ({ request: p.request!, id, conflict_of: of })
    if (!s) { store.push({ id: p.id, title: p.title, text: p.text, created_at: at, updated_at: at }); emit(answer(p.id)); return }
    const same = s.title === p.title && s.text === p.text
    const stale = p.base_updated_at !== null && s.updated_at !== p.base_updated_at && s.updated_at > p.base_updated_at
    if (stale && !same) {
      const id = `copy${++seq}`
      store.push({ id, title: `${p.title} (충돌 사본)`, text: p.text, created_at: at, updated_at: at })
      emit(answer(id, p.id))
      return
    }
    if (!same) Object.assign(s, { title: p.title, text: p.text, updated_at: at })
    emit(answer(p.id))
  }
  function act(job: () => void) { if (immediate) job(); else queue.push(job) }
  function processAll() { while (queue.length) queue.shift()!() }
  function setup(sync = false) {
    immediate = sync
    return scope.run(() => useMemoDraft({
      memos, revision, saved, now: () => clock, newId: () => 'fresh1', newToken: () => `tok${token++}`,
      save: p => { sent.push(p); act(() => processSave(p)) },
      remove: () => {},
      refresh: () => { refreshes++; act(() => emit()) },
    }))!
  }
  const copies = () => store.filter(m => m.id.startsWith('copy'))
  function tick(ms: number) { clock += ms; vi.advanceTimersByTime(ms) }
  function remoteEdit(id: string, text: string) {   // Forge 동기화가 다른 곳의 편집을 들였다
    Object.assign(store.find(m => m.id === id)!, { text, updated_at: stamp() })
    emit()
  }

  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {} })
    scope = effectScope()
    clock = 1000; seq = 0; refreshes = 0; queue = []; sent.length = 0; token = 0
    store = [
      { id: 'A', title: 't', text: 'orig', created_at: stamp(), updated_at: stamp() },
      { id: 'B', title: 'b', text: 'bb', created_at: stamp(), updated_at: stamp() },
    ]
    emit()
  })
  afterEach(() => { scope.stop(); vi.useRealTimers(); vi.unstubAllGlobals() })

  it('a remote change while typing leaves exactly one conflict copy, and later typing edits that copy', () => {
    const d = setup(true)
    d.select('A')
    d.edit({ text: 'orig x' }); tick(100)
    remoteEdit('A', 'REMOTE')
    let text = 'orig x'
    for (let burst = 0; burst < 30; burst++) {            // 30초 동안 끊어 치기 — 저장이 여러 번 나간다
      for (let k = 0; k < 5; k++) { text += 'y'; d.edit({ text }); tick(100) }
      tick(1000)
      emit()                                              // 저장 뒤 동기화도 memoState 를 보낸다
    }
    tick(10_000)
    expect(copies()).toHaveLength(1)
    expect(copies()[0]).toMatchObject({ title: 't (충돌 사본)', text })
    expect(store.find(m => m.id === 'A')!.text).toBe('REMOTE')
    expect(d.draft.value).toMatchObject({ id: copies()[0].id, text, dirty: false, sent: null })
  })

  it('switching memo, a new memo or closing never sends a second save before the first is confirmed', () => {
    const d = setup()
    d.select('A')
    d.edit({ text: 'one' }); tick(600)                    // 첫 저장 — 아직 처리 전
    d.edit({ text: 'one two' })
    d.select('B')
    d.createNew()
    d.clear()
    expect(sent).toHaveLength(1)
    expect(d.parked.value.map(p => p.id)).toEqual(['A'])
    processAll()                                          // 첫 저장 확인 → 밀린 편집을 확인된 base 로
    expect(sent).toHaveLength(2)
    expect(sent[1]).toMatchObject({ id: 'A', text: 'one two', base_updated_at: store.find(m => m.id === 'A')!.updated_at })
    processAll()
    expect(copies()).toHaveLength(0)
    expect(store.find(m => m.id === 'A')!.text).toBe('one two')
    expect(d.parked.value).toEqual([])
    d.select('A')
    expect(d.draft.value).toMatchObject({ text: 'one two', dirty: false, sent: null })
  })

  it('reopening a memo whose save is still unconfirmed shows the unsaved text, not the old stored one', () => {
    const d = setup()
    d.select('A')
    d.edit({ text: 'draft text' }); tick(600)
    d.select('B')
    d.select('A')
    expect(d.draft.value).toMatchObject({ text: 'draft text' })
    expect(d.draft.value!.sent).not.toBeNull()
  })

  it('a slow backend (confirmation later than the wait) makes no copy — it probes instead of resending', () => {
    const d = setup()
    d.select('A')
    d.edit({ text: 'first' }); tick(600)
    d.edit({ text: 'first second' }); tick(12_000)        // 확인이 한참 안 온다(백엔드가 바쁘다)
    expect(sent).toHaveLength(1)                          // 새 편집을 옛 base 로 보내지 않는다
    expect(refreshes).toBe(1)                             // 대신 목록을 다시 청했다(memo_list)
    processAll(); tick(600); processAll()
    expect(copies()).toHaveLength(0)
    expect(store.find(m => m.id === 'A')!.text).toBe('first second')
  })

  it("never retargets a draft onto another memo's conflict copy that has the same body", () => {
    // 제목만 있는 두 메모. A 의 저장이 충돌 사본이 되는 동안 B 의 저장은 아직 확인 전이다 — 본문('')이 같아도
    // A 의 사본은 B 의 것이 아니다(예전엔 본문 · 꼬리만 보고 B 의 초안이 A 의 사본으로 옮겨 갔다)
    store = [
      { id: 'A', title: 'a', text: '', created_at: stamp(), updated_at: stamp() },
      { id: 'B', title: 'b', text: '', created_at: stamp(), updated_at: stamp() },
    ]
    emit()
    const d = setup()
    d.select('A')
    d.edit({ title: 'a2' }); tick(600)                    // A 저장 — 처리 전
    tick(10); remoteEdit('A', 'REMOTE')                   // 그 사이 다른 곳에서 A 가 바뀌었다 → A 저장은 사본이 된다
    d.select('B')
    d.edit({ title: 'b2' }); tick(600)                    // B 저장 — 처리 전
    processAll()
    expect(copies()).toHaveLength(1)
    expect(copies()[0]).toMatchObject({ title: 'a2 (충돌 사본)', text: '' })
    expect(d.draft.value).toMatchObject({ id: 'B', title: 'b2', sent: null })
    d.edit({ title: 'b3' }); tick(600); processAll()
    expect(store.find(m => m.id === 'B')!.title).toBe('b3')
    expect(copies()[0].title).toBe('a2 (충돌 사본)')     // 남의 사본은 그대로
  })

  it('a save that was lost is resent unchanged, then the newer edits follow on the confirmed base', () => {
    const d = setup()
    d.select('A')
    d.edit({ text: 'lost' }); tick(600)
    queue = []                                            // 저장이 사라졌다(연결 끊김 등)
    d.edit({ text: 'lost and more' }); tick(5000)
    expect(refreshes).toBe(1)
    processAll()                                          // 목록 답 → 확인이 없다 → 같은 저장을 다시
    expect(sent.slice(1)).toEqual([sent[0]])
    processAll(); tick(600); processAll()
    expect(sent[2]).toMatchObject({ text: 'lost and more' })
    expect(copies()).toHaveLength(0)
    expect(store.find(m => m.id === 'A')!.text).toBe('lost and more')
  })
})
