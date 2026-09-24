/**
 * App 의 전역 단축키(document keydown) — App.vue 에서 떼어낸 판단 로직.
 *
 * - 게이트(백엔드 선택)가 떠 있는 동안은 전부 죽는다. 백엔드를 아직 안 골랐는데 Ctrl+G 로 생성이
 *   나가거나 F5 로 히스토리가 도는 건 보이지 않는 곳의 오작동이다(게이트 안의 ESC·Tab 은 게이트가
 *   직접 stopPropagation 으로 막는다).
 * - Ctrl+G 생성 · Ctrl+S 설정 저장 · F5 히스토리 새로고침 · Ctrl+Tab / Ctrl+Shift+Tab 탭 순환.
 * - ESC: 모달이 하나라도 떠 있으면 맨 위 층만(utils/modalStack) — 그다음에야 파라미터 열 → 프롬프트.
 * - ↑/↓: 입력 칸에 포커스가 없고 **모달이 하나도 없을 때만** 히스토리 이동(보조키 + 화살표 = 끝으로 점프).
 *   예전엔 여기 모달 플래그 목록을 따로 적어 두었는데 LoRA · 조건부 · A/B · 캐릭터 프리셋 · override 가
 *   빠져, 그 모달을 띄운 채 ↑/↓ 를 누르면 뒤의 히스토리가 넘어갔다(감사 #186). 이제 "모달이 떠 있나"는
 *   스택에만 묻는다 — 모달은 열릴 때 스스로 올라간다(composables/useModalLayer).
 * - ESC · ↑/↓ 는 포커스된 위젯이 이미 처리한 키(defaultPrevented)면 건드리지 않는다. 이 핸들러는
 *   document 의 버블 단계라 위젯이 먼저 받는다 — CustomSelect(드롭다운 열기·항목 이동·닫기),
 *   CompositionControl 궤도(고도 조절), ChatView 의 ESC(생성 중지)가 preventDefault 만 하고 올려 보내,
 *   예전엔 같은 키가 히스토리까지 넘기고 파라미터 열·모달까지 닫았다. Ctrl+G · Ctrl+S · F5 · Ctrl+Tab 은
 *   그대로 둔다 — 에디터가 제 document 리스너에서 Ctrl+S 를 preventDefault 하는데, 여기서 막으면 리스너
 *   등록 순서에 따라 설정 저장이 됐다 안 됐다 한다.
 *   그래서 위젯은 키보드로 다루는 동안만 포커스를 쥐어야 한다 — CustomSelect 는 마우스로 고르거나 닫으면
 *   포커스를 놓는다(안 그러면 마우스로 샘플러를 고른 뒤의 ↓ 가 목록만 다시 열고 히스토리는 안 넘겼다).
 *
 * App.vue 는 ref 읽기와 동작 함수만 넘긴다. 판단은 여기 있고 appShortcuts.test.ts 가 지킨다.
 */
import { appModalStack, type ModalStack } from './modalStack'

/** 핸들러가 읽는 KeyboardEvent 의 부분 — 테스트는 이 모양의 평범한 객체를 넘긴다. */
export type AppShortcutEvent = Pick<KeyboardEvent, 'key' | 'ctrlKey' | 'shiftKey' | 'altKey' | 'metaKey' | 'preventDefault' | 'defaultPrevented'>

/** 포커스된 요소에서 보는 것 — `document.activeElement` 의 부분. */
export interface FocusedElementLike {
  tagName?: string
  isContentEditable?: boolean
}

export interface AppShortcutDeps {
  isGateOpen: () => boolean
  generate: () => void
  saveSettings: () => void
  reloadHistory: () => void
  /** Ctrl+Tab = 1(다음), Ctrl+Shift+Tab = -1(이전) */
  navigateTabs: (direction: 1 | -1) => void
  /** 왼쪽 열이 파라미터 모드인가 — ESC 는 모달이 없을 때 이걸 프롬프트로 되돌린다 */
  isParamsPanelOpen: () => boolean
  closeParamsPanel: () => void
  hasHistory: () => boolean
  navigateHistory: (direction: 1 | -1) => void
  navigateHistoryEdge: (edge: 'top' | 'bottom') => void
  /** 지금 포커스된 요소 — 입력 칸이면 ↑/↓ 는 그 칸의 커서 이동이다 */
  activeElement: () => FocusedElementLike | null
  /** 끝으로 점프하는 보조키의 이벤트 필드 이름(설정 'historyJumpModifier') — 비어 있으면 'shiftKey' */
  jumpModifier: () => string | null | undefined
  /** 기본은 앱 전체 스택(utils/modalStack 의 appModalStack) — 테스트만 바꾼다 */
  modalStack?: ModalStack
}

/** ↑/↓ 를 그 요소가 써야 하는가(입력 · 텍스트 영역 · 셀렉트 · contenteditable). */
export function isEditableElement(el: FocusedElementLike | null | undefined): boolean {
  const tag = (el?.tagName || '').toLowerCase()
  return tag === 'input' || tag === 'textarea' || tag === 'select' || !!el?.isContentEditable
}

export function createAppKeydownHandler(deps: AppShortcutDeps): (e: AppShortcutEvent) => void {
  const stack = deps.modalStack ?? appModalStack
  return (e) => {
    if (deps.isGateOpen()) return
    if (e.ctrlKey && e.key === 'g') { e.preventDefault(); deps.generate() }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); deps.saveSettings() }
    if (e.key === 'F5') { e.preventDefault(); deps.reloadHistory() }
    if (e.ctrlKey && e.key === 'Tab') { e.preventDefault(); deps.navigateTabs(e.shiftKey ? -1 : 1) }

    // ESC — 열려 있는 오버레이/모달을 최상위 우선순위로 닫는다
    if (e.key === 'Escape') {
      // 0) 포커스된 위젯이 이미 쓴 ESC(열린 드롭다운 닫기 · 생성 중지)는 그 위젯의 몫 — 모달·파라미터 열을
      //    같이 닫지 않는다
      if (e.defaultPrevented) return
      // 1) 모달이 떠 있으면 맨 위 것부터(z-index 2000). 제 ESC 를 가진 모달(LoRA · 조건부 …)은 window
      //    capture 에서 stopPropagation 하므로 보통 여기까지 오지 않고, 와도 스택이 "먹기만" 한다.
      if (stack.closeTop()) return
      // 2) 파라미터 열 → 프롬프트로
      if (deps.isParamsPanelOpen()) { deps.closeParamsPanel(); return }
    }

    // 히스토리 ↑/↓ — 입력 칸 포커스 중도, 모달이 떠 있는 동안도 아닐 때만
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      // 포커스된 위젯이 이미 쓴 화살표(CustomSelect 항목 이동 · 궤도 고도) — 히스토리까지 넘기지 않는다
      if (e.defaultPrevented) return
      if (isEditableElement(deps.activeElement())) return
      if (stack.isAnyOpen()) return
      if (!deps.hasHistory()) return
      e.preventDefault()
      const down = e.key === 'ArrowDown'
      // 설정된 보조키(기본 Shift) + 화살표 → 최상단/최하단으로 바로 점프
      const modifier = deps.jumpModifier() || 'shiftKey'
      if ((e as unknown as Record<string, unknown>)[modifier]) deps.navigateHistoryEdge(down ? 'bottom' : 'top')
      else deps.navigateHistory(down ? 1 : -1)
    }
  }
}
