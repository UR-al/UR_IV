/**
 * 프롬프트 패널 Undo/Redo 단축키 판정 (#3).
 *
 * PromptPanel 은 v-show 안에서 늘 마운트돼 있고, 예전 window keydown 핸들러는 포커스를 보지
 * 않고 모든 Ctrl/Cmd+Z·Y 를 가로챘다 — 채팅 입력·설정·갤러리 검색·게이트 URL·모달 입력칸의
 * 네이티브 실행 취소가 죽고, 대신 **보이지 않는 T2I 프롬프트**가 되돌아가 다음 Ctrl+G 생성에
 * 쓰였다. 이제 패널이 **보이고** 포커스가 다음 중 하나일 때만 패널의 스냅숏 Undo/Redo 를 쓴다.
 *
 *  - ``field``     프롬프트 필드(``[data-prompt-undo]`` 로 표시한 영역) 안
 *  - ``panel``     패널 안의 편집 불가 요소(버튼·칩·summary) — 최적화 [적용]·분류·구도 추가·
 *                  Undo 버튼처럼 패널 버튼이 프롬프트를 바꾼 직후 Ctrl+Z 로 되돌릴 수 있어야 한다
 *  - ``unfocused`` 포커스가 어디에도 없음(body) — '모두 비우기'·최적화 [적용]처럼 누른 버튼이
 *                  v-if 로 사라지면 포커스가 body 로 떨어진다. 단 패널이 모달 등에 **가려져 있으면**
 *                  (``unobscured`` 가 거짓) 보이지 않는 프롬프트를 되돌리지 않는다.
 *
 * 그 밖 — 패널 밖 요소, 프롬프트 필드가 아닌 입력칸(자연어 AI·텍스트 모드 최종 프롬프트 등),
 * 모달 안 — 은 브라우저 기본 동작 그대로다.
 *
 * DOM 을 직접 모르는 순수 함수라 vitest(node 환경)에서 가짜 요소로 검사한다.
 */

import { isImeComposing } from './imeComposition'

export const PROMPT_UNDO_SCOPE_ATTR = 'data-prompt-undo'
const SCOPE_SELECTOR = `[${PROMPT_UNDO_SCOPE_ATTR}]`
/** 패널 안이어도 제외하는 곳 — 모달(제외어 관리 등) */
const EXCLUDED_SELECTOR = '[role="dialog"], .em-overlay'

/**
 * 스냅숏 Undo 가 추적하는 위젯 키. ``[data-prompt-undo]`` 로 표시한 필드는 전부 이 키 중 하나에
 * 쓰거나(v-model), 이 키들로만 되쓰는 경로(블록 모드 최종 프롬프트 → onTotalBlockChange)여야 한다 —
 * 표시만 하고 추적하지 않으면 Ctrl+Z 가 그 칸의 네이티브 실행 취소를 막고 **다른 칸**을 되돌린다.
 * (PromptPanel.undo.test.ts 가 템플릿과 이 목록의 짝을 지킨다.)
 */
export const PROMPT_UNDO_KEYS: readonly string[] = [
  'char_count_input', 'character_input', 'copyright_input', 'artist_input',
  'main_prompt_text', 'prefix_prompt_text', 'suffix_prompt_text',
  'neg_prompt_text',
  'exclude_prompt_local_input',
]

export type PromptUndoCommand = 'undo' | 'redo' | null
export type PromptUndoScope = 'field' | 'panel' | 'unfocused' | null

export interface UndoKeyEventLike {
  key: string
  ctrlKey?: boolean
  metaKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  isComposing?: boolean
  keyCode?: number
  defaultPrevented?: boolean
}

interface ClosestLike {
  closest?: (selector: string) => unknown
}

interface ContainsLike {
  contains?: (node: any) => boolean
}

interface ElementLike extends ClosestLike {
  tagName?: string
  type?: string
  nodeType?: number
  isContentEditable?: boolean
}

interface RectLike { left: number; top: number; right: number; bottom: number }

interface PanelLike extends ContainsLike {
  getBoundingClientRect?: () => RectLike
}

interface HitTestLike {
  elementFromPoint?: (x: number, y: number) => unknown
}

/** 글자를 치는 input 이 아닌 type — 체크박스·버튼 등은 네이티브 실행 취소할 게 없다 */
const NON_TEXT_INPUT_TYPES = new Set([
  'button', 'checkbox', 'radio', 'range', 'color', 'file', 'submit', 'reset', 'image', 'hidden',
])

/** 네이티브 실행 취소가 의미 있는 편집 요소인가 (textarea · 글자 input · contenteditable) */
export function isEditableTarget(target: unknown): boolean {
  const el = target as ElementLike | null
  if (!el || typeof el !== 'object') return false
  if (el.isContentEditable) return true
  const tag = String(el.tagName || '').toUpperCase()
  if (tag === 'TEXTAREA') return true
  if (tag === 'INPUT') return !NON_TEXT_INPUT_TYPES.has(String(el.type || 'text').toLowerCase())
  return false
}

/** 포커스가 어디에도 없는가 — keydown 대상이 body/html/document(또는 없음) */
export function isUnfocusedTarget(target: unknown): boolean {
  if (target == null) return true
  const el = target as ElementLike
  if (typeof el !== 'object') return false
  if (el.nodeType === 9) return true            // document
  const tag = String(el.tagName || '').toUpperCase()
  return tag === 'BODY' || tag === 'HTML'
}

/** 키 조합만 본다 — Ctrl/Cmd+Z = undo, Ctrl/Cmd+Y · Ctrl/Cmd+Shift+Z = redo. */
export function undoCommandForKey(e: UndoKeyEventLike): PromptUndoCommand {
  if (!e || !(e.ctrlKey || e.metaKey) || e.altKey) return null
  const key = String(e.key || '').toLowerCase()
  if (key === 'z') return e.shiftKey ? 'redo' : 'undo'
  if (key === 'y' && !e.shiftKey) return 'redo'
  return null
}

/**
 * 포커스(이벤트 대상)가 패널 Undo 의 어느 범위인가 — null 이면 범위 밖(브라우저 기본 동작).
 * 가시성·가림 여부는 여기서 보지 않는다(promptUndoCommand 몫).
 */
export function promptUndoScope(target: unknown, root: unknown): PromptUndoScope {
  const panel = root as ContainsLike | null
  if (!panel || typeof panel.contains !== 'function') return null
  if (isUnfocusedTarget(target)) return 'unfocused'
  const el = target as ElementLike
  if (typeof el !== 'object' || typeof el.closest !== 'function') return null
  if (!panel.contains(el)) return null
  if (el.closest(EXCLUDED_SELECTOR)) return null
  if (el.closest(SCOPE_SELECTOR)) return 'field'
  // 패널 안이지만 추적하지 않는 입력칸(자연어 AI·텍스트 모드 최종 프롬프트·선택 상자 검색 등)
  if (isEditableTarget(el)) return null
  return 'panel'
}

/** 세로로 찍어 볼 위치(보이는 구간 안의 비율) — 가운데부터 */
const PROBE_FRACTIONS = [0.5, 0.25, 0.75, 0.1, 0.9]

/**
 * 패널이 모달·오버레이에 가려지지 않았는가. 화면에 보이는 패널 구간의 세로 중심선 몇 곳을
 * ``elementFromPoint`` 로 찍어, 하나라도 패널 자신(또는 그 자손)이 맨 위면 참이다.
 * 전체 화면 모달이 떠 있거나 패널이 스크롤로 화면 밖에 있으면 거짓. 패널 **안**에 든 모달
 * (제외어 관리 .em-overlay)에 찍힌 것도 가려진 것으로 본다.
 */
export function isPanelOnTop(root: unknown, doc: unknown, viewport: { width: number; height: number }): boolean {
  const panel = root as PanelLike | null
  const hit = doc as HitTestLike | null
  if (!panel || typeof panel.contains !== 'function' || typeof panel.getBoundingClientRect !== 'function') return false
  if (!hit || typeof hit.elementFromPoint !== 'function') return false
  const rect = panel.getBoundingClientRect()
  const x0 = Math.max(rect.left, 0)
  const x1 = Math.min(rect.right, viewport.width)
  const y0 = Math.max(rect.top, 0)
  const y1 = Math.min(rect.bottom, viewport.height)
  if (!(x1 > x0) || !(y1 > y0)) return false
  const x = (x0 + x1) / 2
  for (const f of PROBE_FRACTIONS) {
    const node = hit.elementFromPoint(x, y0 + (y1 - y0) * f) as ClosestLike | null
    if (!node || !panel.contains(node)) continue
    if (typeof node.closest === 'function' && node.closest(EXCLUDED_SELECTOR)) continue
    return true
  }
  return false
}

export interface PromptUndoContext {
  /** 이벤트 대상(보통 포커스된 요소, 포커스가 없으면 body) */
  target: unknown
  /** PromptPanel 루트 요소 */
  root: unknown
  /** 패널이 화면에 보이는가 (v-show 로 숨겨지면 offsetParent 가 null) */
  visible: boolean
  /**
   * 포커스가 없을 때(``unfocused``)만 묻는다: 패널이 모달 등에 가려지지 않았는가.
   * 생략하면 가려지지 않은 것으로 본다.
   */
  unobscured?: () => boolean
}

/**
 * 이 keydown 을 패널 Undo/Redo 로 처리할지. null 이면 건드리지 않는다(브라우저 기본 동작).
 * 이미 다른 핸들러(에디터 등)가 처리했거나 IME 조합 중이면 손대지 않는다.
 */
export function promptUndoCommand(e: UndoKeyEventLike, ctx: PromptUndoContext): PromptUndoCommand {
  const command = undoCommandForKey(e)
  if (!command) return null
  if (e.defaultPrevented) return null
  if (isImeComposing(e)) return null
  if (!ctx.visible) return null
  const scope = promptUndoScope(ctx.target, ctx.root)
  if (!scope) return null
  if (scope === 'unfocused' && ctx.unobscured && !ctx.unobscured()) return null
  return command
}
