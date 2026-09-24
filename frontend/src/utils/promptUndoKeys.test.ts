import { describe, expect, it } from 'vitest'
import {
  PROMPT_UNDO_KEYS, isEditableTarget, isPanelOnTop, isUnfocusedTarget, promptUndoCommand, promptUndoScope,
  undoCommandForKey, type PromptUndoContext,
} from './promptUndoKeys'

/** closest/contains 만 흉내 내는 가짜 DOM — 조상 사슬과 속성/클래스로 선택자를 판정한다. */
interface FakeNode {
  name: string
  attrs?: string[]
  parent?: FakeNode | null
  tagName?: string
  type?: string
  isContentEditable?: boolean
}

function matches(node: FakeNode, selector: string): boolean {
  return selector.split(',').map(s => s.trim()).some(sel => {
    if (sel === '[data-prompt-undo]') return !!node.attrs?.includes('data-prompt-undo')
    if (sel === '[role="dialog"]') return !!node.attrs?.includes('role=dialog')
    if (sel.startsWith('.')) return !!node.attrs?.includes(sel)
    return false
  })
}

type FakeEl = FakeNode & { closest: (s: string) => FakeNode | null; contains: (n: any) => boolean }

function el(name: string, parent: FakeNode | null, attrs: string[] = [], extra: Partial<FakeNode> = {}): FakeEl {
  const node: FakeEl = {
    name, attrs, parent, tagName: 'DIV', ...extra,
    closest(selector: string) {
      let cur: FakeNode | null | undefined = node
      while (cur) { if (matches(cur, selector)) return cur; cur = cur.parent }
      return null
    },
    contains(other: any) {
      let cur: FakeNode | null | undefined = other
      while (cur) { if (cur === node) return true; cur = cur.parent }
      return false
    },
  }
  return node
}

const body = el('body', null, [], { tagName: 'BODY' })
const panel = el('prompt-panel', body)
const mainGroup = el('main-group', panel, ['data-prompt-undo'])
const mainTextarea = el('textarea', mainGroup, [], { tagName: 'TEXTAREA' })
const blockClearAll = el('tbf-clear-all', mainGroup, [], { tagName: 'BUTTON' })     // TagBlockField 모두 비우기
const charCountInput = el('char-count', el('char-count-group', panel, ['data-prompt-undo']), [], { tagName: 'INPUT', type: 'text' })
const nlInput = el('nl-input', panel, [], { tagName: 'INPUT', type: 'text' })      // 자연어 AI 입력 — 프롬프트 필드 아님
const totalTextarea = el('total-prompt', panel, [], { tagName: 'TEXTAREA' })         // 텍스트 모드 최종 프롬프트 — 추적 안 함
const optimizeApply = el('opt-apply', panel, [], { tagName: 'BUTTON' })              // 최적화 [적용]
const undoButton = el('undo-btn', el('summary', panel, [], { tagName: 'SUMMARY' }), [], { tagName: 'BUTTON' })
const sepCheckbox = el('sep-check', panel, [], { tagName: 'INPUT', type: 'checkbox' })
const editable = el('ce', panel, [], { isContentEditable: true })
const modal = el('em-overlay', panel, ['.em-overlay', 'data-prompt-undo'])
const modalSearch = el('em-search', modal, [], { tagName: 'INPUT', type: 'text' })
const modalButton = el('em-close', modal, [], { tagName: 'BUTTON' })
const chatInput = el('chat-input', el('chat-view', body), [], { tagName: 'TEXTAREA' })
const headerButton = el('generate-btn', body, [], { tagName: 'BUTTON' })             // 패널 밖 버튼

const ctrlZ = { key: 'z', ctrlKey: true }
const ctx = (target: unknown, extra: Partial<PromptUndoContext> = {}): PromptUndoContext =>
  ({ target, root: panel, visible: true, ...extra })

describe('undoCommandForKey', () => {
  it('maps Ctrl/Cmd+Z, Ctrl+Y and Ctrl+Shift+Z', () => {
    expect(undoCommandForKey(ctrlZ)).toBe('undo')
    expect(undoCommandForKey({ key: 'Z', metaKey: true })).toBe('undo')
    expect(undoCommandForKey({ key: 'y', ctrlKey: true })).toBe('redo')
    expect(undoCommandForKey({ key: 'Z', ctrlKey: true, shiftKey: true })).toBe('redo')
    expect(undoCommandForKey({ key: 'z' })).toBeNull()
    expect(undoCommandForKey({ key: 'z', ctrlKey: true, altKey: true })).toBeNull()
    expect(undoCommandForKey({ key: 'g', ctrlKey: true })).toBeNull()
  })
})

describe('promptUndoScope', () => {
  it('classifies prompt fields, panel controls and no-focus', () => {
    expect(promptUndoScope(mainTextarea, panel)).toBe('field')
    expect(promptUndoScope(charCountInput, panel)).toBe('field')
    // 필드 안의 버튼(블록 모두 비우기)도 필드 범위
    expect(promptUndoScope(blockClearAll, panel)).toBe('field')
    // 패널 버튼 — 누른 뒤 곧바로 Ctrl+Z 로 되돌릴 수 있어야 한다
    expect(promptUndoScope(optimizeApply, panel)).toBe('panel')
    expect(promptUndoScope(undoButton, panel)).toBe('panel')
    expect(promptUndoScope(sepCheckbox, panel)).toBe('panel')
    // 포커스가 없음(누른 버튼이 v-if 로 사라짐)
    expect(promptUndoScope(body, panel)).toBe('unfocused')
    expect(promptUndoScope(null, panel)).toBe('unfocused')
    expect(promptUndoScope({ nodeType: 9 }, panel)).toBe('unfocused')
  })

  it('leaves untracked inputs, dialogs and everything outside the panel alone', () => {
    expect(promptUndoScope(nlInput, panel)).toBeNull()
    expect(promptUndoScope(totalTextarea, panel)).toBeNull()
    expect(promptUndoScope(editable, panel)).toBeNull()
    expect(promptUndoScope(modalSearch, panel)).toBeNull()
    expect(promptUndoScope(modalButton, panel)).toBeNull()
    expect(promptUndoScope(chatInput, panel)).toBeNull()
    expect(promptUndoScope(headerButton, panel)).toBeNull()
  })

  it('is defensive about missing DOM APIs', () => {
    expect(promptUndoScope(mainTextarea, null)).toBeNull()
    expect(promptUndoScope({}, panel)).toBeNull()
    expect(promptUndoScope('x', panel)).toBeNull()
  })
})

describe('promptUndoCommand', () => {
  it('handles Ctrl+Z/Y inside a prompt field of a visible panel (char count included)', () => {
    expect(promptUndoCommand(ctrlZ, ctx(mainTextarea))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(charCountInput))).toBe('undo')
    expect(promptUndoCommand({ key: 'y', ctrlKey: true }, ctx(mainTextarea))).toBe('redo')
  })

  it('keeps keyboard undo after panel buttons — focused button or focus dropped to body', () => {
    expect(promptUndoCommand(ctrlZ, ctx(optimizeApply))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(undoButton))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(blockClearAll))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(body, { unobscured: () => true }))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(body))).toBe('undo')          // 가림 판정이 없으면 보인다고 본다
  })

  it('does not revert a prompt hidden behind a modal when nothing is focused', () => {
    let asked = 0
    const covered = () => { asked++; return false }
    expect(promptUndoCommand(ctrlZ, ctx(body, { unobscured: covered }))).toBeNull()
    expect(asked).toBe(1)
    // 필드·패널 버튼 포커스에서는 가림 판정을 묻지 않는다(방금 그 요소를 눌렀다)
    expect(promptUndoCommand(ctrlZ, ctx(mainTextarea, { unobscured: covered }))).toBe('undo')
    expect(promptUndoCommand(ctrlZ, ctx(optimizeApply, { unobscured: covered }))).toBe('undo')
    expect(asked).toBe(1)
  })

  it('leaves native undo alone everywhere else', () => {
    // 채팅·설정 등 다른 화면의 입력칸, 패널 밖 버튼
    expect(promptUndoCommand(ctrlZ, ctx(chatInput))).toBeNull()
    expect(promptUndoCommand(ctrlZ, ctx(headerButton))).toBeNull()
    // 패널 안이지만 추적하지 않는 입력(자연어 AI · 텍스트 모드 최종 프롬프트)
    expect(promptUndoCommand(ctrlZ, ctx(nlInput))).toBeNull()
    expect(promptUndoCommand(ctrlZ, ctx(totalTextarea))).toBeNull()
    // 패널 안 모달(제외어 관리)
    expect(promptUndoCommand(ctrlZ, ctx(modalSearch))).toBeNull()
  })

  it('does nothing while hidden, composing or already handled', () => {
    expect(promptUndoCommand(ctrlZ, ctx(mainTextarea, { visible: false }))).toBeNull()
    expect(promptUndoCommand(ctrlZ, ctx(body, { visible: false }))).toBeNull()
    expect(promptUndoCommand({ ...ctrlZ, isComposing: true }, ctx(mainTextarea))).toBeNull()
    expect(promptUndoCommand({ ...ctrlZ, defaultPrevented: true }, ctx(mainTextarea))).toBeNull()
  })
})

describe('isEditableTarget / isUnfocusedTarget', () => {
  it('treats text inputs, textareas and contenteditable as editable', () => {
    expect(isEditableTarget({ tagName: 'TEXTAREA' })).toBe(true)
    expect(isEditableTarget({ tagName: 'INPUT' })).toBe(true)
    expect(isEditableTarget({ tagName: 'input', type: 'search' })).toBe(true)
    expect(isEditableTarget({ tagName: 'DIV', isContentEditable: true })).toBe(true)
    expect(isEditableTarget({ tagName: 'INPUT', type: 'checkbox' })).toBe(false)
    expect(isEditableTarget({ tagName: 'BUTTON' })).toBe(false)
    expect(isEditableTarget(null)).toBe(false)
  })

  it('recognises body/html/document as no focus', () => {
    expect(isUnfocusedTarget({ tagName: 'BODY' })).toBe(true)
    expect(isUnfocusedTarget({ tagName: 'HTML' })).toBe(true)
    expect(isUnfocusedTarget({ nodeType: 9 })).toBe(true)
    expect(isUnfocusedTarget(undefined)).toBe(true)
    expect(isUnfocusedTarget({ tagName: 'BUTTON' })).toBe(false)
  })
})

describe('isPanelOnTop', () => {
  const viewport = { width: 1600, height: 900 }
  const rect = { left: 60, top: 40, right: 420, bottom: 2400 }       // 스크롤로 아래가 화면 밖
  const panelNode = el('prompt-panel', body)
  const inside = el('card', panelNode)
  const withRect = (r: typeof rect) => Object.assign(panelNode, { getBoundingClientRect: () => r })

  it('is true when a probe inside the visible part of the panel hits the panel', () => {
    const probes: Array<[number, number]> = []
    const doc = { elementFromPoint: (x: number, y: number) => { probes.push([x, y]); return inside } }
    expect(isPanelOnTop(withRect(rect), doc, viewport)).toBe(true)
    // 가로 가운데, 세로는 화면에 보이는 구간(40~900)의 가운데부터
    expect(probes[0]).toEqual([240, 470])
  })

  it('is false when a modal covers every probe', () => {
    const overlay = el('pm-overlay', body)
    const probes: number[] = []
    const doc = { elementFromPoint: (_x: number, y: number) => { probes.push(y); return overlay } }
    expect(isPanelOnTop(withRect(rect), doc, viewport)).toBe(false)
    expect(probes.length).toBe(5)
    for (const y of probes) { expect(y).toBeGreaterThanOrEqual(40); expect(y).toBeLessThanOrEqual(900) }
  })

  it('treats the in-panel exclude manager modal as covering the panel', () => {
    const emOverlay = el('em-overlay', panelNode, ['.em-overlay'])
    const emList = el('em-list', emOverlay)
    const doc = { elementFromPoint: () => emList }
    expect(isPanelOnTop(withRect(rect), doc, viewport)).toBe(false)
  })

  it('survives a partial cover (toast) by probing more than one point', () => {
    const toast = el('toast', body)
    let n = 0
    const doc = { elementFromPoint: () => (n++ === 0 ? toast : inside) }
    expect(isPanelOnTop(withRect(rect), doc, viewport)).toBe(true)
  })

  it('is false when the panel is scrolled out of view or APIs are missing', () => {
    const doc = { elementFromPoint: () => inside }
    expect(isPanelOnTop(withRect({ left: 60, top: -900, right: 420, bottom: -10 }), doc, viewport)).toBe(false)
    expect(isPanelOnTop(withRect(rect), {}, viewport)).toBe(false)
    expect(isPanelOnTop(null, doc, viewport)).toBe(false)
    expect(isPanelOnTop(el('no-rect', body), doc, viewport)).toBe(false)
  })
})

describe('PROMPT_UNDO_KEYS', () => {
  it('tracks the char count field with the other prompt fields, without duplicates', () => {
    expect(PROMPT_UNDO_KEYS).toContain('char_count_input')
    expect(new Set(PROMPT_UNDO_KEYS).size).toBe(PROMPT_UNDO_KEYS.length)
  })
})
