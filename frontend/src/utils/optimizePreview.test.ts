import { describe, expect, it } from 'vitest'
import {
  OPTIMIZE_CONTEXT_KEYS, isOptimizePreviewStale, optimizeContextPayload, snapshotOptimizeInputs,
} from './optimizePreview'

const widgets = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  main_prompt_text: 'red hair, solo',
  char_count_input: '1girl',
  character_input: 'hatsune miku',
  copyright_input: '',
  artist_input: '',
  prefix_prompt_text: 'red hair',
  suffix_prompt_text: 'masterpiece',
  unrelated_widget: 'x',
  ...over,
})

describe('optimize preview inputs', () => {
  it('sends the same context list as before — blank fields dropped', () => {
    const snap = snapshotOptimizeInputs(widgets())
    expect(snap.main).toBe('red hair, solo')
    expect(optimizeContextPayload(snap)).toEqual(['1girl', 'hatsune miku', 'red hair', 'masterpiece'])
    // 공백만 있는 칸도 빠진다
    expect(optimizeContextPayload(snapshotOptimizeInputs(widgets({ artist_input: '   ' })))).not.toContain('   ')
  })

  it('is not stale while nothing changed', () => {
    const w = widgets()
    expect(isOptimizePreviewStale(snapshotOptimizeInputs(w), w)).toBe(false)
    // 최적화에 쓰지 않는 칸이 바뀐 것은 상관없다
    expect(isOptimizePreviewStale(snapshotOptimizeInputs(w), { ...w, unrelated_widget: 'y' })).toBe(false)
  })

  it('is stale when the main tags changed', () => {
    const snap = snapshotOptimizeInputs(widgets())
    expect(isOptimizePreviewStale(snap, widgets({ main_prompt_text: 'red hair, solo, smile' }))).toBe(true)
  })

  it('is stale when a context field was cleared (the tag would vanish from the prompt)', () => {
    // 미리보기: 접두 'red hair' 와 겹쳐 메인에서 'red hair' 를 뺐다 → 접두를 비우고 적용하면 태그가 사라진다
    const snap = snapshotOptimizeInputs(widgets())
    expect(isOptimizePreviewStale(snap, widgets({ prefix_prompt_text: '' }))).toBe(true)
  })

  it('is stale when the character was switched or a tag was added to a context field', () => {
    const snap = snapshotOptimizeInputs(widgets())
    expect(isOptimizePreviewStale(snap, widgets({ character_input: 'kagamine rin' }))).toBe(true)
    expect(isOptimizePreviewStale(snap, widgets({ suffix_prompt_text: 'masterpiece, solo' }))).toBe(true)
  })

  it('watches every context field', () => {
    const snap = snapshotOptimizeInputs(widgets())
    expect(OPTIMIZE_CONTEXT_KEYS).toHaveLength(6)
    for (const key of OPTIMIZE_CONTEXT_KEYS) {
      expect(isOptimizePreviewStale(snap, widgets({ [key]: 'changed tag' })), key).toBe(true)
    }
  })

  it('treats a missing widget value like an empty field — no false positives', () => {
    const w = widgets({ copyright_input: undefined, artist_input: null })
    const snap = snapshotOptimizeInputs(w)
    expect(isOptimizePreviewStale(snap, widgets({ copyright_input: '', artist_input: '' }))).toBe(false)
    const noMain = widgets()
    delete noMain.main_prompt_text
    expect(isOptimizePreviewStale(snapshotOptimizeInputs(noMain), widgets({ main_prompt_text: '' }))).toBe(false)
  })

  it('the snapshot is a copy — later widget edits do not rewrite it', () => {
    const w = widgets()
    const snap = snapshotOptimizeInputs(w)
    w.prefix_prompt_text = 'blue hair'
    expect(snap.context.prefix_prompt_text).toBe('red hair')
    expect(isOptimizePreviewStale(snap, w)).toBe(true)
  })
})
