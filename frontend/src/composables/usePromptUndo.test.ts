import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'
import { usePromptUndo } from './usePromptUndo'
import { PROMPT_UNDO_KEYS } from '../utils/promptUndoKeys'

/**
 * 프롬프트 패널 스냅숏 Undo 기록(#3). 위젯 값이 바뀌면 watch → 500ms 뒤 스냅숏.
 * Undo/Redo 는 대기 중인 변경을 먼저 확정해야 '방금 바뀌기 직전' 상태로 돌아간다.
 */
function setup(initial: Record<string, string> = {}) {
  const widgets = reactive<Record<string, any>>({ main_prompt_text: 'a', char_count_input: '1girl', ...initial })
  const undo = usePromptUndo(widgets)
  undo.resetBaseline()
  return { widgets, undo }
}

/** 값을 바꾸고 watch 가 debounce 타이머를 걸 때까지 기다린다(타이머는 흘리지 않는다) */
async function edit(widgets: Record<string, any>, key: string, value: string) {
  widgets[key] = value
  await nextTick()
}

/** 값을 바꾸고 입력이 멎은 것처럼 debounce 를 흘려 스냅숏을 쌓는다 */
async function settle(widgets: Record<string, any>, key: string, value: string) {
  await edit(widgets, key, value)
  vi.advanceTimersByTime(500)
}

describe('usePromptUndo', () => {
  let undoRef: ReturnType<typeof usePromptUndo> | null = null
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { undoRef?.dispose(); undoRef = null; vi.useRealTimers() })

  it('snapshots after typing settles and undoes/redoes it', async () => {
    const { widgets, undo } = setup(); undoRef = undo
    await settle(widgets, 'main_prompt_text', 'a, b')
    expect(undo.undoStack.value.map(s => s.main_prompt_text)).toEqual(['a', 'a, b'])
    undo.undo()
    expect(widgets.main_prompt_text).toBe('a')
    await nextTick()
    vi.advanceTimersByTime(1000)
    // 되쓴 값이 다시 스냅숏으로 쌓이지 않는다
    expect(undo.undoStack.value.length).toBe(1)
    undo.redo()
    expect(widgets.main_prompt_text).toBe('a, b')
  })

  it('undo right after a change (inside the debounce) returns to the state just before it', async () => {
    const { widgets, undo } = setup(); undoRef = undo
    await settle(widgets, 'main_prompt_text', 'a, b')
    // '모두 비우기' 직후 500ms 안에 Ctrl+Z — 예전엔 'a, b' 를 건너뛰고 'a' 로 갔다
    await edit(widgets, 'main_prompt_text', '')
    undo.undo()
    expect(widgets.main_prompt_text).toBe('a, b')
    undo.redo()
    expect(widgets.main_prompt_text).toBe('')
    undo.undo()
    undo.undo()
    expect(widgets.main_prompt_text).toBe('a')
  })

  it('tracks the char count field (block-mode final prompt edits write it)', async () => {
    const { widgets, undo } = setup(); undoRef = undo
    expect(PROMPT_UNDO_KEYS).toContain('char_count_input')
    await edit(widgets, 'char_count_input', '')
    undo.undo()
    expect(widgets.char_count_input).toBe('1girl')
    expect(widgets.main_prompt_text).toBe('a')
  })

  it('a new edit after undo drops the redo branch', async () => {
    const { widgets, undo } = setup(); undoRef = undo
    await settle(widgets, 'main_prompt_text', 'a, b')
    undo.undo()
    await nextTick()
    await edit(widgets, 'main_prompt_text', 'a, c')
    undo.redo()                                     // 대기 중 변경을 확정하며 Redo 가 무효가 된다
    expect(widgets.main_prompt_text).toBe('a, c')
    expect(undo.redoStack.value).toEqual([])
    undo.undo()
    expect(widgets.main_prompt_text).toBe('a')
  })

  it('commit records programmatic edits immediately and ignores duplicates', async () => {
    const { widgets, undo } = setup(); undoRef = undo
    undo.commit()
    expect(undo.undoStack.value.length).toBe(1)     // 맨 위와 같으면 쌓지 않는다
    undo.commit()
    widgets.main_prompt_text = 'optimized'
    undo.commit()
    expect(undo.undoStack.value.map(s => s.main_prompt_text)).toEqual(['a', 'optimized'])
    await nextTick()
    vi.advanceTimersByTime(500)                     // 뒤늦은 watch 타이머도 같은 값이라 무시
    expect(undo.undoStack.value.length).toBe(2)
    undo.undo()
    expect(widgets.main_prompt_text).toBe('a')
  })

  it('does not undo past the baseline and caps the stack', async () => {
    const widgets = reactive<Record<string, any>>({ main_prompt_text: '' })
    const undo = usePromptUndo(widgets, { max: 3 }); undoRef = undo
    undo.resetBaseline()
    undo.undo()
    expect(widgets.main_prompt_text).toBe('')
    for (const v of ['1', '2', '3', '4']) await settle(widgets, 'main_prompt_text', v)
    expect(undo.undoStack.value.map(s => s.main_prompt_text)).toEqual(['2', '3', '4'])
  })

  it('resetBaseline drops a pending snapshot and dispose stops watching', async () => {
    const { widgets, undo } = setup()
    await edit(widgets, 'main_prompt_text', 'loaded from backend')
    undo.resetBaseline()
    vi.advanceTimersByTime(500)
    expect(undo.undoStack.value.map(s => s.main_prompt_text)).toEqual(['loaded from backend'])
    undo.dispose()
    await settle(widgets, 'main_prompt_text', 'after dispose')
    expect(undo.undoStack.value.length).toBe(1)
  })
})
