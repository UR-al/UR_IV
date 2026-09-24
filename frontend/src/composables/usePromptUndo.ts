import { nextTick, ref, watch } from 'vue'
import { PROMPT_UNDO_KEYS } from '../utils/promptUndoKeys'

/**
 * 프롬프트 패널 스냅숏 Undo/Redo 기록 (#3). PromptPanel 에서 떼어 냈다.
 *
 * 추적 키(PROMPT_UNDO_KEYS)의 값이 바뀌면 입력이 멎은 뒤(debounce) 스냅숏을 쌓는다.
 * Undo/Redo 는 스냅숏을 위젯에 되쓰고, 그 값은 v-model 과 같은 길로 Python 까지 간다.
 *
 * 예전엔 Undo 가 **아직 쌓이지 않은 변경**(debounce 대기 중)을 모른 채 맨 위 스냅숏을 꺼내
 * 그 앞 상태로 갔다 — '모두 비우기' 직후나 타이핑 직후 500ms 안에 Ctrl+Z 하면 방금 바뀌기
 * **직전** 상태를 건너뛰고 한 단계 더 옛 상태가 됐다. 이제 Undo/Redo 전에 대기 중인 변경을
 * 먼저 스냅숏으로 확정(commit)한다.
 */

export type PromptSnapshot = Record<string, string>

export interface PromptUndoOptions {
  /** 추적 키 — 기본 PROMPT_UNDO_KEYS */
  keys?: readonly string[]
  /** 스택 최대 길이 */
  max?: number
  /** 빠른 타이핑을 한 스냅숏으로 묶는 대기(ms) */
  debounceMs?: number
}

export function usePromptUndo(widgets: Record<string, any>, options: PromptUndoOptions = {}) {
  const keys = options.keys ?? PROMPT_UNDO_KEYS
  const max = options.max ?? 50
  const debounceMs = options.debounceMs ?? 500
  const undoStack = ref<PromptSnapshot[]>([])    // 과거 스냅숏 — 맨 위가 '지금'
  const redoStack = ref<PromptSnapshot[]>([])    // 미래 (Ctrl+Y)
  let applying = false                             // 되쓰는 동안 watch 가 다시 쌓지 않게
  let timer: ReturnType<typeof setTimeout> | null = null

  function snapshot(): PromptSnapshot {
    const s: PromptSnapshot = {}
    for (const k of keys) s[k] = widgets[k] || ''
    return s
  }
  function same(a: PromptSnapshot, b: PromptSnapshot): boolean {
    for (const k of keys) if ((a[k] || '') !== (b[k] || '')) return false
    return true
  }
  function cancelPending() {
    if (timer !== null) { clearTimeout(timer); timer = null }
  }

  /** 지금 상태를 스냅숏으로 쌓는다(맨 위와 같으면 무시). 새 변경이라 Redo 는 비운다. */
  function push() {
    const cur = snapshot()
    const last = undoStack.value[undoStack.value.length - 1]
    if (last && same(last, cur)) return
    undoStack.value.push(cur)
    if (undoStack.value.length > max) undoStack.value.shift()
    redoStack.value = []
  }

  /** debounce 를 기다리지 않고 지금 상태를 확정한다 — 프로그램 편집 직전·직후, Undo/Redo 직전 */
  function commit() {
    cancelPending()
    push()
  }

  /** 초기값이 들어온 지금 상태를 Undo 의 출발점으로 — 그 전(빈 칸)으로는 되돌아가지 않는다 */
  function resetBaseline() {
    cancelPending()
    undoStack.value = [snapshot()]
    redoStack.value = []
  }

  function apply(snap: PromptSnapshot) {
    applying = true
    for (const k of keys) {
      if (widgets[k] !== snap[k]) widgets[k] = snap[k]
    }
    nextTick(() => { applying = false })
  }

  function undo() {
    commit()                                        // 대기 중인 변경부터 확정
    if (undoStack.value.length < 2) return          // 첫 스냅(현재) 외에 없으면 안 함
    const cur = undoStack.value.pop() as PromptSnapshot
    redoStack.value.push(cur)
    apply(undoStack.value[undoStack.value.length - 1])
  }

  function redo() {
    commit()                                        // 되돌린 뒤 새로 고쳤으면 Redo 는 무효(push 가 비운다)
    if (redoStack.value.length === 0) return
    const next = redoStack.value.pop() as PromptSnapshot
    undoStack.value.push(next)
    apply(next)
  }

  const stops = keys.map(k => watch(() => widgets[k], () => {
    if (applying) return
    cancelPending()
    timer = setTimeout(() => { timer = null; push() }, debounceMs)
  }))

  /** 타이머와 watch 해제 — 컴포넌트 onUnmounted 에서 */
  function dispose() {
    cancelPending()
    for (const stop of stops.splice(0)) {
      try { stop() } catch { /* 이미 해제됨 */ }
    }
  }

  return { undoStack, redoStack, commit, resetBaseline, undo, redo, dispose }
}
