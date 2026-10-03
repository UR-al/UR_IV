import { ref } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useHistoryTrash, type HistoryDeleteRequest, type HistoryRestoreRequest } from './useHistoryTrash'

const a = 'C:/out/a.png', b = 'C:/out/b.png', c = 'C:/out/c.png'
const disposers: Array<() => void> = []
afterEach(() => {
  disposers.splice(0).forEach(dispose => dispose())
  vi.useRealTimers()
})

function harness(paths = [a, b, c], selected = b, perPage = 2, timeoutMs?: number) {
  const history = ref([...paths]), current = ref(selected), page = ref(0)
  const deletes: HistoryDeleteRequest[] = [], restores: HistoryRestoreRequest[] = []
  const onSelection = vi.fn(), onChanged = vi.fn(), onTimeout = vi.fn()
  const sendDelete = vi.fn((request: HistoryDeleteRequest) => { deletes.push(request) })
  const sendRestore = vi.fn((request: HistoryRestoreRequest) => { restores.push(request) })
  const api = useHistoryTrash({ history, current, page, perPage, sendDelete, sendRestore, onSelection, onChanged, onTimeout, timeoutMs })
  disposers.push(api.dispose)
  function deleted(overrides: Record<string, unknown> = {}, request = deletes[deletes.length - 1]) {
    api.onDeleteResult(JSON.stringify({ ...request, ok: true, removed: true, undo_token: `undo-${request.path}`,
      level: 'info', message: '휴지통으로 이동됨', ...overrides }))
  }
  function restored(overrides: Record<string, unknown> = {}, request = restores[restores.length - 1]) {
    api.onRestoreResult({ ...request, path: request.undo_token.replace('undo-', ''), ok: true, restored: true,
      level: 'info', message: '복구됨', ...overrides })
  }
  return { api, history, current, page, deletes, restores, sendDelete, sendRestore, onSelection, onChanged, onTimeout, deleted, restored }
}

describe('useHistoryTrash', () => {
  it('waits for successful disk deletion and rejects repeated requests while pending', () => {
    const h = harness()
    expect(h.api.deleteImage(b)).toBe(true)
    expect(h.deletes[0]).toMatchObject({ path: b, undoable: true, request_id: expect.any(String) })
    expect(h.api.pending.value).toBe(true)
    expect(h.history.value).toEqual([a, b, c])
    expect(h.current.value).toBe(b)
    expect(h.api.deleteImage(b)).toBe(false)
    expect(h.api.deleteImage(c)).toBe(false)
    expect(h.api.undo()).toBe(false)
    expect(h.onChanged).not.toHaveBeenCalled()
    h.deleted()
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(true)
    expect(h.history.value).toEqual([a, c])
    expect(h.current.value).toBe(a)
    expect(h.onSelection).toHaveBeenCalledExactlyOnceWith(a)
    expect(h.onChanged).toHaveBeenCalledOnce()
  })

  it('ignores empty and non-history paths instead of deleting an unrelated image', () => {
    const h = harness()
    expect(h.api.deleteImage('')).toBe(false)
    expect(h.api.deleteImage('   ')).toBe(false)
    expect(h.api.deleteImage('C:/other.png')).toBe(false)
    expect(h.deletes).toEqual([])
  })

  it('leaves the list and selection intact when a trash request fails', () => {
    const h = harness()
    h.api.deleteImage(b)
    h.deleted({ ok: false, removed: false, undo_token: '' })
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(false)
    expect(h.api.undo()).toBe(false)
    expect(h.history.value).toEqual([a, b, c])
    expect(h.current.value).toBe(b)
    expect(h.onSelection).not.toHaveBeenCalled()
    expect(h.onChanged).not.toHaveBeenCalled()
  })

  it('drops already-missing files without creating an undo entry', () => {
    const h = harness([a], a)
    h.page.value = 4
    h.api.deleteImage(a)
    h.deleted({ ok: false, removed: true, undo_token: '' })
    expect(h.history.value).toEqual([])
    expect(h.current.value).toBe('')
    expect(h.page.value).toBe(0)
    expect(h.onSelection).toHaveBeenCalledExactlyOnceWith('')
    expect(h.api.canUndo.value).toBe(false)
  })

  it('applies deletes from another surface without claiming their undo or releasing its own pending request', () => {
    const h = harness()
    h.api.deleteImage(b)
    h.deleted({ path: a, request_id: 'gallery-request', undo_token: 'gallery-token' })
    expect(h.history.value).toEqual([b, c])
    expect(h.api.pending.value).toBe(true)
    h.deleted({ ok: false, removed: false })
    expect(h.api.canUndo.value).toBe(false)
    expect(h.current.value).toBe(b)
    expect(h.onSelection).not.toHaveBeenCalled()
  })

  it('supports legacy delete broadcasts without a request id', () => {
    const h = harness()
    h.api.onDeleteResult({ path: b, ok: true, removed: true })
    expect(h.history.value).toEqual([a, c])
    expect(h.api.canUndo.value).toBe(false)
  })

  it('rejects malformed, mismatched-path, and wrong-kind replies without unlocking the pending request', () => {
    const h = harness()
    h.api.deleteImage(b)
    const request = h.deletes[0]
    for (const result of [null, [], '{', '{}', { ...request, removed: true },
      { ...request, ok: true, removed: 'true' }, { ...request, path: c, ok: true, removed: true }]) {
      h.api.onDeleteResult(result)
      expect(h.api.pending.value).toBe(true)
      expect(h.history.value).toEqual([a, b, c])
    }
    h.api.onRestoreResult({ ...request, ok: true, restored: true, undo_token: 'x' })
    expect(h.api.pending.value).toBe(true)
    h.deleted()
    expect(h.api.pending.value).toBe(false)
  })

  it('restores in original order and reselects the restored image with its EXIF and page', () => {
    const h = harness([a, b, c], c)
    h.page.value = 1
    h.api.deleteImage(c)
    h.deleted()
    expect(h.page.value).toBe(0)
    expect(h.api.undo()).toBe(true)
    expect(h.restores[0].undo_token).toBe(`undo-${c}`)
    expect(h.restores[0].request_id).not.toBe(h.deletes[0].request_id)
    expect(h.history.value).toEqual([a, b])
    expect(h.api.pending.value).toBe(true)
    expect(h.api.canUndo.value).toBe(false)
    expect(h.api.deleteImage(a)).toBe(false)
    expect(h.api.undo()).toBe(false)
    h.restored()
    expect(h.history.value).toEqual([a, b, c])
    expect(h.current.value).toBe(c)
    expect(h.page.value).toBe(1)
    expect(h.onSelection).toHaveBeenLastCalledWith(c)
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(false)
  })

  it('anchors restoration among neighbors after new generations are prepended', () => {
    const h = harness()
    h.api.deleteImage(b)
    h.deleted()
    h.history.value.unshift('C:/out/new.png')
    h.api.undo()
    h.restored()
    expect(h.history.value).toEqual(['C:/out/new.png', a, b, c])
    expect(h.page.value).toBe(1)
  })

  it('uses the next neighbor when the first item was deleted, and clamps the index after a list refresh', () => {
    const h = harness()
    h.api.deleteImage(a)
    h.deleted()
    h.history.value.unshift('C:/out/new.png')
    h.api.undo()
    h.restored()
    expect(h.history.value).toEqual(['C:/out/new.png', a, b, c])
    h.api.deleteImage(c)
    h.deleted()
    h.history.value = []
    h.api.undo()
    h.restored()
    expect(h.history.value).toEqual([c])
    expect(h.page.value).toBe(0)
  })

  it('does not duplicate paths already reloaded with another equivalent Windows spelling', () => {
    const h = harness()
    h.api.deleteImage('file:///C:/out/b.png')
    expect(h.deletes[0].path).toBe(b)
    h.deleted({ path: 'c:\\OUT\\B.png' })
    h.history.value = [a, 'c:\\OUT\\B.png', c]
    h.api.undo()
    h.restored({ path: 'file:///C:/out/b.png' })
    expect(h.history.value).toEqual([a, 'c:\\OUT\\B.png', c])
    expect(h.current.value).toBe('c:\\OUT\\B.png')
    expect(h.onSelection).toHaveBeenLastCalledWith('c:\\OUT\\B.png')
  })

  it('keeps retryable conflicts in the undo stack and drops terminally unavailable tokens', () => {
    const h = harness()
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    h.restored({ ok: false, restored: false, retryable: true })
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(true)
    expect(h.history.value).toEqual([a, c])
    h.api.undo()
    expect(h.restores[1].undo_token).toBe(h.restores[0].undo_token)
    expect(h.restores[1].request_id).not.toBe(h.restores[0].request_id)
    h.restored({ ok: false, restored: false, retryable: false })
    expect(h.api.canUndo.value).toBe(false)
  })

  it('ignores stale delete responses after undo and does not release a different pending operation', () => {
    const h = harness()
    h.api.deleteImage(b)
    const oldDelete = h.deletes[0]
    h.deleted()
    h.api.undo()
    h.restored()
    h.api.deleteImage(c)
    h.deleted({}, oldDelete)
    expect(h.api.pending.value).toBe(true)
    expect(h.history.value).toEqual([a, b, c])
    h.deleted()
    expect(h.history.value).toEqual([a, b])
    h.api.undo()
    const request = h.restores[h.restores.length - 1]
    for (const overrides of [{ request_id: 'old-request' }, { path: a }, { undo_token: 'wrong' }, { restored: 1 }, { ok: 'true' }]) {
      h.api.onRestoreResult({ ...request, path: c, ok: true, restored: true, ...overrides })
      expect(h.api.pending.value).toBe(true)
      expect(h.history.value).toEqual([a, b])
    }
    h.restored()
    expect(h.api.pending.value).toBe(false)
  })

  it('unlocks matching failed restores even when an expired token or exception has no known path', () => {
    const h = harness()
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    h.restored({ path: '', ok: false, restored: false, retryable: true })
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(true)
    h.api.undo()
    h.restored({ path: '', ok: false, restored: false, retryable: false })
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(false)
    expect(h.history.value).toEqual([a, c])
  })

  it('undoes several deletions in reverse order, keeping no more than 30 entries', () => {
    const paths = Array.from({ length: 32 }, (_, i) => `C:/out/${i}.png`)
    const h = harness(paths, paths[0])
    for (const path of paths) {
      expect(h.api.deleteImage(path)).toBe(true)
      h.deleted()
    }
    for (let index = 31; index >= 2; index--) {
      expect(h.api.undo()).toBe(true)
      expect(h.restores[h.restores.length - 1].undo_token).toBe(`undo-${paths[index]}`)
      h.restored()
    }
    expect(h.api.undo()).toBe(false)
    expect(h.history.value).toEqual(paths.slice(2))
    expect(new Set([...h.deletes, ...h.restores].map(request => request.request_id)).size).toBe(62)
  })

  it('clears pending after synchronous sender errors without losing images or undo entries', () => {
    const h = harness()
    h.sendDelete.mockImplementationOnce(() => { throw new Error('offline') })
    expect(() => h.api.deleteImage(b)).toThrow('offline')
    expect(h.api.pending.value).toBe(false)
    expect(h.history.value).toEqual([a, b, c])
    h.api.deleteImage(b)
    h.deleted()
    h.sendRestore.mockImplementationOnce(() => { throw new Error('offline') })
    expect(() => h.api.undo()).toThrow('offline')
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(true)
    expect(h.history.value).toEqual([a, c])
  })

  it('unlocks a missing reply after 30 seconds without pretending deletion succeeded', () => {
    vi.useFakeTimers()
    const h = harness()
    h.api.deleteImage(b)
    vi.advanceTimersByTime(29_999)
    expect(h.api.pending.value).toBe(true)
    vi.advanceTimersByTime(1)
    expect(h.api.pending.value).toBe(false)
    expect(h.history.value).toEqual([a, b, c])
    expect(h.api.canUndo.value).toBe(false)
    expect(h.onTimeout).toHaveBeenCalledOnce()
    expect(h.api.deleteImage(c)).toBe(true)
    h.deleted()
    vi.advanceTimersByTime(30_000)
    expect(h.onTimeout).toHaveBeenCalledOnce()
  })

  it('accepts a late delete success and its undo token without unlocking a newer request', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    const delayed = h.deletes[0]
    vi.advanceTimersByTime(100)
    h.api.deleteImage(c)
    h.deleted({}, delayed)
    expect(h.history.value).toEqual([a, c])
    expect(h.api.pending.value).toBe(true)
    vi.advanceTimersByTime(99)
    expect(h.api.pending.value).toBe(true)
    h.deleted({ ok: false, removed: false })
    expect(h.api.pending.value).toBe(false)
    expect(h.api.undo()).toBe(true)
    expect(h.restores[0].undo_token).toBe(`undo-${b}`)
    h.restored()
    expect(h.history.value).toEqual([a, b, c])
  })

  it('orders late delete acknowledgments by the user requests, not network arrival order', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    const delayed = h.deletes[0]
    vi.advanceTimersByTime(100)
    h.api.deleteImage(c)
    h.deleted()
    h.deleted({}, delayed)
    expect(h.history.value).toEqual([a])
    h.api.undo()
    expect(h.restores[0].undo_token).toBe(`undo-${c}`)
    h.restored()
    h.api.undo()
    expect(h.restores[1].undo_token).toBe(`undo-${b}`)
    h.restored()
    expect(h.history.value).toEqual([a, b, c])
  })

  it('removes only a late failed restore token, preserving newer undo entries and pending requests', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    const delayed = h.restores[0]
    vi.advanceTimersByTime(100)
    h.api.deleteImage(c)
    h.deleted()
    h.api.undo()
    h.restored({ path: '', ok: false, restored: false, retryable: false }, delayed)
    expect(h.api.pending.value).toBe(true)
    h.restored({ path: '', ok: false, restored: false, retryable: true })
    expect(h.api.canUndo.value).toBe(true)
    h.api.undo()
    expect(h.restores[2].undo_token).toBe(`undo-${c}`)
    h.restored()
    expect(h.api.canUndo.value).toBe(false)
    expect(h.history.value).toEqual([a, c])
  })

  it('restores a late success while another delete remains pending', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    const delayed = h.restores[0]
    vi.advanceTimersByTime(100)
    h.api.deleteImage(c)
    h.restored({}, delayed)
    expect(h.history.value).toEqual([a, b, c])
    expect(h.current.value).toBe(b)
    expect(h.api.pending.value).toBe(true)
    h.deleted()
    expect(h.history.value).toEqual([a, b])
    expect(h.api.pending.value).toBe(false)
    h.api.undo()
    expect(h.restores[1].undo_token).toBe(`undo-${c}`)
    h.restored()
  })

  it('deduplicates concurrent retry restore successes without interrupting newer correlation', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    const delayed = h.restores[0]
    vi.advanceTimersByTime(100)
    h.api.undo()
    h.restored({}, delayed)
    expect(h.api.pending.value).toBe(true)
    expect(h.history.value).toEqual([a, b, c])
    const changes = h.onChanged.mock.calls.length
    h.restored()
    expect(h.api.pending.value).toBe(false)
    expect(h.onChanged).toHaveBeenCalledTimes(changes)
    expect(h.history.value).toEqual([a, b, c])
    expect(h.api.canUndo.value).toBe(false)
  })

  it('bounds the unresolved correlation map to the latest 60 requests', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    for (let index = 0; index < 61; index++) {
      expect(h.api.deleteImage(b)).toBe(true)
      vi.advanceTimersByTime(100)
    }
    h.deleted({}, h.deletes[0])
    expect(h.history.value).toEqual([a, b, c])
    expect(h.api.canUndo.value).toBe(false)
    h.deleted({}, h.deletes[1])
    expect(h.history.value).toEqual([a, c])
    expect(h.api.canUndo.value).toBe(true)
  })

  it('disposes timers and state, rejecting late replies and new requests after unmount', () => {
    vi.useFakeTimers()
    const h = harness([a, b, c], b, 2, 100)
    h.api.deleteImage(b)
    h.deleted()
    h.api.undo()
    h.api.dispose()
    vi.advanceTimersByTime(200)
    h.restored()
    expect(h.onTimeout).not.toHaveBeenCalled()
    expect(h.api.pending.value).toBe(false)
    expect(h.api.canUndo.value).toBe(false)
    expect(h.history.value).toEqual([a, c])
    expect(h.api.deleteImage(a)).toBe(false)
    expect(h.api.undo()).toBe(false)
    expect(vi.getTimerCount()).toBe(0)
  })
})
