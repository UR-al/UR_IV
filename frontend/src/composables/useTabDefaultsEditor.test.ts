import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { useTabDefaultsEditor } from './useTabDefaultsEditor'

function setup(file: Record<string, unknown>) {
  const calls: Array<[string, any]> = []
  const state = { file: { ...file } }
  const backend = { getTabDefaults: (cb: (json: string) => void) => cb(JSON.stringify(state.file)) }
  const editor = useTabDefaultsEditor({
    sendAction: (name: string, payload?: any) => { calls.push([name, payload]) },
    getBackend: async () => backend,
    debounceMs: 100,
  })
  return { editor, calls, state }
}

/**
 * 응답을 붙잡아 두는 가짜 호스트 — getTabDefaults 는 호출 시점의 파일을 읽고(동기 슬롯) 콜백만 늦게
 * 보낸다. save_tab_defaults 는 파일에 합친다. QWebChannel 은 FIFO 라 읽기 뒤에 온 저장은 그 응답에 없다.
 */
function deferredSetup(file: Record<string, unknown>) {
  const calls: Array<[string, any]> = []
  const state = { file: { ...file } as Record<string, unknown> }
  const replies: Array<() => void> = []
  const backend = {
    getTabDefaults: (cb: (json: string) => void) => {
      const snapshot = JSON.stringify(state.file)
      replies.push(() => cb(snapshot))
    },
  }
  const editor = useTabDefaultsEditor({
    sendAction: (name: string, payload?: any) => {
      calls.push([name, payload])
      if (name === 'save_tab_defaults') Object.assign(state.file, payload)
    },
    getBackend: async () => backend,
    debounceMs: 100,
  })
  const settle = async () => { for (let i = 0; i < 5; i++) await Promise.resolve() }
  /** 가장 오래된 응답 하나를 도착시키고 reload 가 끝날 때까지 흘린다 */
  const answer = async () => { replies.shift()?.(); await settle(); await nextTick(); await settle() }
  return { editor, calls, state, replies, answer, settle }
}

describe('useTabDefaultsEditor — edits while a reload is in flight', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('keeps a field typed during the wait and saves it; untouched fields follow the file', async () => {
    const h = deferredSetup({ steps: 30, denoising: 0.4 })
    const first = h.editor.reload(); await h.settle(); await h.answer(); await first
    h.state.file.denoising = 0.6                  // 다른 경로(전역 저장)가 파일을 갱신
    const again = h.editor.reload(); await h.settle()
    h.editor.defaults.steps = 42                  // 응답을 기다리는 사이 입력
    await nextTick()
    await h.answer(); await again
    expect(h.editor.defaults.steps).toBe(42)
    expect(h.editor.defaults.denoising).toBe(0.6)
    vi.advanceTimersByTime(150)
    expect(h.calls).toEqual([['save_tab_defaults', { steps: 42 }]])
    expect(h.state.file.steps).toBe(42)
  })

  it('does not snap back (or resave) when the debounce already saved before the reply', async () => {
    const h = deferredSetup({ steps: 30 })
    const first = h.editor.reload(); await h.settle(); await h.answer(); await first
    const again = h.editor.reload(); await h.settle()
    h.editor.defaults.steps = 42
    await nextTick()
    vi.advanceTimersByTime(150)                   // 응답 전에 디바운스 저장
    expect(h.state.file.steps).toBe(42)
    await h.answer(); await again                 // 응답은 저장 전에 읽은 30
    expect(h.editor.defaults.steps).toBe(42)
    vi.advanceTimersByTime(500)
    expect(h.calls).toEqual([['save_tab_defaults', { steps: 42 }]])
  })

  it('overlapping reloads apply only the newest answer and save nothing', async () => {
    const h = deferredSetup({ steps: 30 })
    const a = h.editor.reload(); await h.settle()
    h.state.file.steps = 44
    const b = h.editor.reload(); await h.settle()
    await h.answer(); await h.answer(); await a; await b
    expect(h.editor.defaults.steps).toBe(44)
    vi.advanceTimersByTime(500)
    expect(h.calls).toEqual([])
  })
})

describe('useTabDefaultsEditor', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('loading values does not save or toast', async () => {
    const { editor, calls } = setup({ steps: 30, denoising: 0.4 })
    await editor.reload()
    await nextTick()
    vi.advanceTimersByTime(500)
    expect(editor.defaults.steps).toBe(30)
    expect(editor.defaults.denoising).toBe(0.4)
    expect(calls).toEqual([])
  })

  it('saves only the changed key after the debounce', async () => {
    const { editor, calls } = setup({ steps: 30 })
    await editor.reload()
    editor.defaults.snapRadius = 20
    await nextTick()
    vi.advanceTimersByTime(150)
    expect(calls).toEqual([['save_tab_defaults', { snapRadius: 20 }]])
  })

  it('re-reads fresh T2I values after the global save instead of overwriting them', async () => {
    const { editor, calls, state } = setup({ steps: 30 })
    await editor.reload()
    state.file.steps = 44            // '전역 저장'이 파일을 갱신
    await editor.reload()             // Settings 재활성
    editor.defaults.denoising = 0.5
    await nextTick()
    vi.advanceTimersByTime(150)
    expect(editor.defaults.steps).toBe(44)
    expect(calls).toEqual([['save_tab_defaults', { denoising: 0.5 }]])
  })

  it('reload flushes a pending edit first', async () => {
    const { editor, calls } = setup({})
    await editor.reload()
    editor.defaults.brushSize = 33
    await nextTick()
    await editor.reload()
    expect(calls[0]).toEqual(['save_tab_defaults', { brushSize: 33 }])
  })

  it('save button reports when nothing changed', async () => {
    const { editor, calls } = setup({})
    await editor.reload()
    editor.saveNow()
    expect(calls[0][0]).toBe('show_toast')
  })
})
