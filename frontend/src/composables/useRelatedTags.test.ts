import { describe, expect, it, vi } from 'vitest'
import { nextTick, reactive, ref } from 'vue'
import { BUILD_QUIET_MS, RELATED_TAGS_TIMEOUT_MESSAGE, createRelatedTags } from './useRelatedTags'

function fakeTimers() {
  let next = 1
  const pending = new Map<number, () => void>()
  return {
    set: (fn: () => void) => { const id = next++; pending.set(id, fn); return id },
    clear: (id: unknown) => { pending.delete(id as number) },
    run() { const fns = [...pending.values()]; pending.clear(); fns.forEach(fn => fn()) },
    get size() { return pending.size },
  }
}

function fixture() {
  const handlers = new Map<string, (raw: unknown) => void>()
  const requestAction = vi.fn()
  const persistUiPrefs = vi.fn()
  const widgets = reactive<Record<string, unknown>>({ main_prompt_text: 'armor', neg_prompt_text: '' })
  const debounce = fakeTimers()
  const timeout = fakeTimers()
  let clock = 1000
  let id = 0
  const revision = ref(0)
  const store = createRelatedTags({
    onBackendEvent: (name, cb) => { handlers.set(name, cb); return () => handlers.delete(name) },
    requestAction, widgets, persistUiPrefs, debounceMs: 500, timers: debounce,
    newId: () => `rtb-${++id}`, now: () => clock, ratingRevision: revision,
    request: { setTimer: timeout.set, clearTimer: timeout.clear, randomSuffix: () => 'x' },
  })
  const emit = (event: object) => handlers.get('relatedTagsEvent')!(JSON.stringify(event))
  const lastQuery = () => [...requestAction.mock.calls].reverse().find(c => c[0] === 'related_tags_query')
  const reply = (extra: object = {}) => emit({
    action: 'related_tags_query', requestId: lastQuery()![1].requestId, state: 'ready', tags: [], context: [],
    ratings: ['g', 's'], missing: [], stale: [], unavailable: [], estimateSec: 0, build: null, ...extra,
  })
  return {
    store, handlers, requestAction, persistUiPrefs, widgets, debounce, timeout, emit, reply, lastQuery, revision,
    advance: (ms: number) => { clock += ms },
  }
}

const SUGGESTION = { tag: 'holding_sword', ko: '검 들기', count: 10, via: ['armor'] }
const flush = () => new Promise(resolve => setTimeout(resolve, 0))

describe('createRelatedTags', () => {
  it('stays quiet while closed and queries once when opened, persisting the open state', async () => {
    const f = fixture()
    expect(f.handlers.has('relatedTagsEvent')).toBe(true)
    f.widgets.main_prompt_text = 'armor, sword'
    await nextTick()
    expect(f.debounce.size).toBe(0)
    expect(f.requestAction).not.toHaveBeenCalled()
    f.store.setOpen(true)
    expect(f.persistUiPrefs).toHaveBeenCalledWith({ relatedTagsOpen: true })
    expect(f.lastQuery()![1]).toMatchObject({ fields: { main: 'armor, sword' }, limit: 16 })
    expect(f.store.loading.value).toBe(true)
    f.reply({ tags: [SUGGESTION], context: ['armor'] })
    await flush()
    expect(f.store.loading.value).toBe(false)
    expect(f.store.tags.value).toEqual([SUGGESTION])
    expect(f.store.context.value).toEqual(['armor'])
  })

  it('debounces prompt edits, skips unchanged prompts and ignores stale replies', async () => {
    const f = fixture()
    f.store.setOpen(true)
    const first = f.lastQuery()![1].requestId
    f.widgets.main_prompt_text = 'armor, cape'
    await nextTick()
    f.widgets.main_prompt_text = 'armor, cape, sword'
    await nextTick()
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_query')).toHaveLength(1)
    f.debounce.run()
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_query')).toHaveLength(2)
    expect(f.lastQuery()![1].fields.main).toBe('armor, cape, sword')
    f.emit({ action: 'related_tags_query', requestId: first, state: 'ready', tags: [SUGGESTION], context: [],
      ratings: ['g'], missing: [], stale: [], unavailable: [], estimateSec: 0, build: null })
    await flush()
    expect(f.store.tags.value).toEqual([])          // 옛 요청의 답은 버린다
    f.reply()
    await flush()
    f.widgets.main_prompt_text = 'armor, cape, sword'
    await nextTick()
    f.debounce.run()
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_query')).toHaveLength(2)
  })

  it('asks again with the same prompt after the rating filter is sent to Python', async () => {
    const f = fixture()
    f.store.setOpen(true)
    f.reply()
    await flush()
    f.revision.value += 1
    await nextTick()
    f.debounce.run()
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_query')).toHaveLength(2)
  })

  it('closing cancels the pending query; a timeout shows an error and allows a retry', async () => {
    const f = fixture()
    f.store.setOpen(true)
    f.store.setOpen(false)
    await flush()
    expect(f.store.loading.value).toBe(false)
    expect(f.persistUiPrefs).toHaveBeenLastCalledWith({ relatedTagsOpen: false })
    f.store.setOpen(true)
    f.timeout.run()
    await flush()
    expect(f.store.state.value).toBe('error')
    expect(f.store.error.value).toBe(RELATED_TAGS_TIMEOUT_MESSAGE)
    void f.store.refresh()                           // 같은 프롬프트라도 다시 묻는다
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_query')).toHaveLength(3)
  })

  it('restores the open state once without saving it again, unless the user already toggled', () => {
    const f = fixture()
    f.store.restoreOpen('not json')
    f.store.restoreOpen(JSON.stringify({ relatedTagsOpen: true }))
    expect(f.store.open.value).toBe(true)
    expect(f.persistUiPrefs).not.toHaveBeenCalled()
    expect(f.lastQuery()).toBeTruthy()
    const g = fixture()
    g.store.setOpen(true)
    g.store.setOpen(false)
    g.store.restoreOpen({ relatedTagsOpen: true })
    expect(g.store.open.value).toBe(false)
  })

  it('builds the missing shards, follows broadcast progress and re-queries when done', async () => {
    const f = fixture()
    f.store.setOpen(true)
    f.reply({ state: 'missing', missing: ['g', 's'], estimateSec: 140 })
    await flush()
    expect(f.store.state.value).toBe('missing')
    f.store.startBuild()
    expect(f.requestAction).toHaveBeenLastCalledWith('related_tags_build', { requestId: 'rtb-1', shards: ['g', 's'] })
    f.store.startBuild()                           // 도는 동안은 다시 보내지 않는다
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_build')).toHaveLength(1)
    f.emit({ action: 'related_tags_build', requestId: 'rtb-other', state: 'building',
      progress: { shard: 's', index: 1, count: 2, progress: 0.5, overall: 0.75 } })
    expect(f.store.build.value).toMatchObject({ requestId: 'rtb-other', shard: 's', overall: 0.75 })
    f.store.cancelBuild()
    f.store.cancelBuild()
    expect(f.requestAction.mock.calls.filter(c => c[0] === 'related_tags_cancel')).toEqual([
      ['related_tags_cancel', { requestId: 'rtb-other' }]])
    f.emit({ action: 'related_tags_build', requestId: 'rtb-other', state: 'canceled', shards: ['g'] })
    expect(f.store.build.value).toBeNull()
    expect(f.store.feedback.value).toBe('표 만들기를 취소했습니다.')
    expect(f.lastQuery()).toBeTruthy()
    f.reply({ missing: ['s'], tags: [SUGGESTION] })
    await flush()
    expect(f.store.missing.value).toEqual(['s'])
    f.store.startBuild()
    expect(f.requestAction).toHaveBeenLastCalledWith('related_tags_build', { requestId: 'rtb-2', shards: ['s'] })
    f.emit({ action: 'related_tags_build', requestId: 'rtb-2', state: 'done', shards: ['s'] })
    expect(f.store.feedback.value).toBe('연관 태그 표를 만들었습니다.')
    f.store.startBuild(['g'])
    f.emit({ action: 'related_tags_build', requestId: 'rtb-3', state: 'error', error: '실패 이유' })
    expect(f.store.feedback.value).toBe('실패 이유')
  })

  it('a reply computed before the build started does not hide fresh progress', async () => {
    const f = fixture()
    f.store.setOpen(true)
    f.store.startBuild(['g'])
    f.reply({ state: 'missing', missing: ['g'] })
    await flush()
    expect(f.store.build.value?.requestId).toBe('rtb-1')
    f.advance(BUILD_QUIET_MS + 1)
    void f.store.refresh(true)
    f.reply({ state: 'missing', missing: ['g'] })
    await flush()
    expect(f.store.build.value).toBeNull()
    f.store.startBuild(['g'])
    void f.store.refresh(true)
    f.reply({ state: 'building', build: { requestId: 'rtb-2', shard: 'g', index: 0, count: 1, progress: 0.2, overall: 0.2 } })
    await flush()
    expect(f.store.build.value?.progress).toBe(0.2)
  })

  it('drops a clicked suggestion, ignores junk events and stops everything on dispose', async () => {
    const f = fixture()
    f.store.setOpen(true)
    f.reply({ tags: [SUGGESTION, { ...SUGGESTION, tag: 'cape' }] })
    await flush()
    f.store.dropSuggestion('cape')
    expect(f.store.tags.value.map(t => t.tag)).toEqual(['holding_sword'])
    f.handlers.get('relatedTagsEvent')!('garbage')
    f.store.dispose()
    expect(f.handlers.has('relatedTagsEvent')).toBe(false)
    f.widgets.main_prompt_text = 'changed'
    await nextTick()
    expect(f.debounce.size).toBe(0)
    f.store.setOpen(false)
    expect(f.store.open.value).toBe(true)
  })
})
