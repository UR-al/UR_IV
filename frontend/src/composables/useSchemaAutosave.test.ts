import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createSchemaAutosave } from './useSchemaAutosave'

const bridge = vi.hoisted(() => ({ saveChatSchemaDraft: vi.fn() }))
vi.mock('../bridge.js', () => ({ getBackend: async () => bridge }))

beforeEach(() => { vi.useFakeTimers(); bridge.saveChatSchemaDraft.mockReset() })
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers() })
function acknowledge(index: number, ok = true) {
  const [schemaText, callback] = bridge.saveChatSchemaDraft.mock.calls[index]
  callback(JSON.stringify({ ok, schemaText, error: ok ? undefined : '파일 쓰기 실패' }))
}

it('debounces typing, accepts unfinished JSON, and confirms saved only after the disk acknowledgement', async () => {
  const save = createSchemaAutosave()
  save.queue('{')
  await vi.advanceTimersByTimeAsync(200)
  save.queue('{"type":')
  await vi.advanceTimersByTimeAsync(399)
  expect(bridge.saveChatSchemaDraft).not.toHaveBeenCalled()
  await vi.advanceTimersByTimeAsync(1)
  expect(bridge.saveChatSchemaDraft.mock.calls[0][0]).toBe('{"type":')
  expect(save.state.value).toBe('saving')
  acknowledge(0)
  expect(save.state.value).toBe('saved')
})

it('serializes rapid edits and never acknowledges a newer unsaved draft as saved', async () => {
  const save = createSchemaAutosave()
  save.queue('first'); await vi.advanceTimersByTimeAsync(400)
  save.queue('second'); await vi.advanceTimersByTimeAsync(400)
  save.queue('latest')
  expect(bridge.saveChatSchemaDraft).toHaveBeenCalledTimes(1)
  expect(save.state.value).toBe('pending')
  acknowledge(0)
  expect(bridge.saveChatSchemaDraft).toHaveBeenCalledTimes(2)
  expect(bridge.saveChatSchemaDraft.mock.calls[1][0]).toBe('latest')
  expect(save.state.value).toBe('saving')
  acknowledge(1)
  await vi.advanceTimersByTimeAsync(400)
  expect(save.state.value).toBe('saved')
  expect(bridge.saveChatSchemaDraft).toHaveBeenCalledTimes(2)
})

it('retains a failed draft for explicit retry without an automatic retry loop', async () => {
  const save = createSchemaAutosave()
  save.queue(''); await vi.advanceTimersByTimeAsync(400)
  acknowledge(0, false)
  expect(save.state.value).toBe('error')
  expect(save.error.value).toBe('파일 쓰기 실패')
  await vi.advanceTimersByTimeAsync(20000)
  expect(bridge.saveChatSchemaDraft).toHaveBeenCalledTimes(1)
  save.flush()
  expect(bridge.saveChatSchemaDraft.mock.calls[1][0]).toBe('')
  acknowledge(1)
  expect(save.state.value).toBe('saved')
  expect(save.error.value).toBe('')
})

it('ignores timed-out callbacks and retries only the newest pending text', async () => {
  const save = createSchemaAutosave()
  save.queue('old'); await vi.advanceTimersByTimeAsync(400)
  save.queue('new')
  await vi.advanceTimersByTimeAsync(12000)
  expect(bridge.saveChatSchemaDraft).toHaveBeenCalledTimes(2)
  expect(save.state.value).toBe('saving')
  acknowledge(0)
  expect(save.state.value).toBe('saving')
  acknowledge(1)
  expect(save.state.value).toBe('saved')
})

it('flushes on close and drains a newer draft after the in-flight write', async () => {
  const save = createSchemaAutosave()
  save.queue('first'); save.flush(); await Promise.resolve()
  save.queue('final'); save.close()
  save.queue('ignored after close')
  acknowledge(0)
  expect(bridge.saveChatSchemaDraft.mock.calls[1][0]).toBe('final')
  acknowledge(1)
  expect(save.state.value).toBe('saved')
})

it('reports mismatched acknowledgements and missing bridge methods as failures', async () => {
  const save = createSchemaAutosave()
  save.queue('draft'); await vi.advanceTimersByTimeAsync(400)
  bridge.saveChatSchemaDraft.mock.calls[0][1](JSON.stringify({ ok: true, schemaText: 'wrong' }))
  expect(save.state.value).toBe('error')
  expect(save.error.value).toContain('확인하지 못했습니다')
  bridge.saveChatSchemaDraft.mockImplementation(() => { throw Error('연결 종료') })
  save.flush()
  expect(save.state.value).toBe('error')
  expect(save.error.value).toBe('연결 종료')
})
