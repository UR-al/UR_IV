/**
 * 도크 패널(components/dock/ChatMiniPanel.vue · MemoPanel.vue) 안에서 눌린 키.
 *
 * 두 패널은 모달이 아니라 화면을 덮지 않고 떠 있는 카드다(dockPanel.css) — 히스토리 · 프롬프트를 보면서
 * 쓰라고 둔 것이다. 그래서 앱 모달 스택(utils/modalStack)에 오르지 않는다: 올라가면 패널이 열려 있는 내내
 * 바깥에 포커스가 있어도 전역 ↑/↓ 히스토리 이동이 막히고, ESC 가 파라미터 열보다 패널을 먼저 닫는다.
 * 대신 패널 루트의 keydown(버블)이 패널 **안에서** 눌린 키만 다룬다:
 * - Esc: 패널을 닫는다. 위젯이 이미 쓴 Esc(대화 입력칸의 '답 받는 중 중지')와 IME 조합 중은 건드리지 않는다.
 * - ↑/↓: 패널 것 — 앱의 히스토리 이동(document keydown)까지 올려 보내지 않는다. 목록 스크롤 같은
 *   기본 동작은 그대로 둔다.
 * 바깥에 포커스가 있으면 이 핸들러는 불리지 않고 앱의 단축키(utils/appShortcuts)가 그대로 돈다.
 */
import { isImeComposing } from './imeComposition'

export type DockPanelKeyEvent = Pick<KeyboardEvent, 'key' | 'defaultPrevented' | 'preventDefault' | 'stopPropagation'>
  & { isComposing?: boolean; keyCode?: number }

export type DockPanelKeyAction = 'close' | 'contain' | null

export function dockPanelKeyAction(e: Pick<DockPanelKeyEvent, 'key' | 'defaultPrevented' | 'isComposing' | 'keyCode'>): DockPanelKeyAction {
  if (e.key === 'Escape') return e.defaultPrevented || isImeComposing(e) ? null : 'close'
  if (e.key === 'ArrowUp' || e.key === 'ArrowDown') return 'contain'
  return null
}

/** 패널 루트의 @keydown 에 건다. */
export function handleDockPanelKeydown(e: DockPanelKeyEvent, close: () => void): void {
  const action = dockPanelKeyAction(e)
  if (!action) return
  e.stopPropagation()
  if (action === 'close') {
    e.preventDefault()
    close()
  }
}
