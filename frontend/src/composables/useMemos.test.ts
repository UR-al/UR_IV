import { describe, expect, it, vi } from 'vitest'
import { createMemos } from './useMemos'

describe('createMemos', () => {
  function fixture() {
    const handlers = new Map<string, (json: string) => void>()
    const bound: Array<() => void> = []
    const requestAction = vi.fn()
    const store = createMemos({
      onBackendEvent: (name, cb) => { handlers.set(name, cb); return () => handlers.delete(name) },
      onBackendBound: cb => { bound.push(cb); return () => {} },
      requestAction,
    })
    return { store, handlers, bound, requestAction }
  }

  it('listens to memoState, pulls the list whenever the backend binds, and binds once', () => {
    const { store, handlers, bound, requestAction } = fixture()
    store.bind(); store.bind()
    expect(bound).toHaveLength(1)
    bound[0]()
    expect(requestAction).toHaveBeenCalledWith('memo_list', {})
    handlers.get('memoState')!(JSON.stringify({
      memos: [{ id: 'a', title: 'A', text: '', created_at: 'x', updated_at: '2026-09-01T00:00:00Z' }],
      sync: { available: false, target: 'local', syncing: false, last_synced_at: null, error: null },
    }))
    expect(store.memos.value.map(m => m.id)).toEqual(['a'])
    expect(store.sync.value?.available).toBe(false)
    expect(store.revision.value).toBe(1)
    expect(store.saved.value).toBeNull()
    handlers.get('memoState')!(JSON.stringify({ memos: [], saved: { request: 'rq1', id: 'c1', conflict_of: 'a' } }))
    expect(store.saved.value).toEqual({ request: 'rq1', id: 'c1', conflict_of: 'a' })   // 그 저장의 답
    handlers.get('memoState')!(JSON.stringify({ memos: [] }))
    expect(store.saved.value).toBeNull()                    // 다음 memoState 에는 없다 — 한 번만 본다
    expect(store.revision.value).toBe(3)
    handlers.get('memoState')!('not json')                 // 깨진 페이로드는 무시한다
    expect(store.revision.value).toBe(3)
    store.unbind()
    expect(handlers.has('memoState')).toBe(false)
  })

  it('sends the contract actions and removes a deleted memo from the list right away', () => {
    const { store, requestAction } = fixture()
    store.applyState(JSON.stringify({ memos: [
      { id: 'a', title: 'A', text: '', created_at: '', updated_at: '2026-09-01T00:00:00Z' },
      { id: 'b', title: 'B', text: '', created_at: '', updated_at: '2026-09-02T00:00:00Z' },
    ], sync: {} }))
    store.save({ id: 'a', title: 'A2', text: 't', base_updated_at: '2026-09-01T00:00:00Z' })
    store.remove('b')
    store.syncNow()
    expect(requestAction.mock.calls).toEqual([
      ['memo_save', { id: 'a', title: 'A2', text: 't', base_updated_at: '2026-09-01T00:00:00Z' }],
      ['memo_delete', { id: 'b' }],
      ['memo_sync', {}],
    ])
    expect(store.memos.value.map(m => m.id)).toEqual(['a'])
  })
})
