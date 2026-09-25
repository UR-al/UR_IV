import { describe, expect, it, vi } from 'vitest'
import { dockPanelKeyAction, handleDockPanelKeydown, type DockPanelKeyEvent } from './dockPanelKeys'
import { createAppKeydownHandler, type AppShortcutDeps } from './appShortcuts'
import { createModalStack } from './modalStack'
import memoPanelSource from '../components/dock/MemoPanel.vue?raw'
import chatPanelSource from '../components/dock/ChatMiniPanel.vue?raw'

function keyEvent(key: string, extra: Partial<DockPanelKeyEvent> = {}) {
  let prevented = !!extra.defaultPrevented
  let stopped = false
  const e = {
    key, isComposing: extra.isComposing, keyCode: extra.keyCode,
    ctrlKey: false, shiftKey: false, altKey: false, metaKey: false,
    get defaultPrevented() { return prevented },
    preventDefault() { prevented = true },
    stopPropagation() { stopped = true },
  }
  return { e, stopped: () => stopped }
}

describe('dock panel keys', () => {
  it('closes on Esc pressed inside the panel, unless a widget used it or IME is composing', () => {
    expect(dockPanelKeyAction({ key: 'Escape', defaultPrevented: false })).toBe('close')
    expect(dockPanelKeyAction({ key: 'Escape', defaultPrevented: true })).toBeNull()      // 대화 입력칸의 '중지'
    expect(dockPanelKeyAction({ key: 'Escape', defaultPrevented: false, isComposing: true })).toBeNull()
    expect(dockPanelKeyAction({ key: 'Escape', defaultPrevented: false, keyCode: 229 })).toBeNull()
    expect(dockPanelKeyAction({ key: 'ArrowDown', defaultPrevented: false })).toBe('contain')
    expect(dockPanelKeyAction({ key: 'Enter', defaultPrevented: false })).toBeNull()
  })

  it('keeps its keys away from the app (no history move, no params-panel close) but leaves other shortcuts alone', () => {
    const close = vi.fn()
    const esc = keyEvent('Escape')
    handleDockPanelKeydown(esc.e, close)
    expect(close).toHaveBeenCalledOnce()
    expect(esc.stopped()).toBe(true)
    expect(esc.e.defaultPrevented).toBe(true)

    const down = keyEvent('ArrowDown')
    handleDockPanelKeydown(down.e, close)
    expect(down.stopped()).toBe(true)
    expect(down.e.defaultPrevented).toBe(false)                     // 목록 스크롤 등 기본 동작은 둔다

    const save = keyEvent('s')
    handleDockPanelKeydown(save.e, close)
    expect(save.stopped()).toBe(false)                              // Ctrl+S 같은 전역 단축키는 그대로 올라간다
    expect(close).toHaveBeenCalledOnce()
  })

  it('with focus outside an open dock panel, ↑/↓ still moves history and Esc closes the params panel', () => {
    // 패널은 모달 스택에 오르지 않는다 — 스택이 비어 있으면 앱 단축키가 예전처럼 돈다
    const stack = createModalStack()
    const deps: AppShortcutDeps = {
      isGateOpen: () => false, generate() {}, saveSettings() {}, reloadHistory() {}, navigateTabs() {},
      isParamsPanelOpen: () => true, closeParamsPanel: vi.fn(), hasHistory: () => true,
      navigateHistory: vi.fn(), navigateHistoryEdge() {}, activeElement: () => ({ tagName: 'IMG' }),
      jumpModifier: () => 'shiftKey', modalStack: stack,
    }
    const onKey = createAppKeydownHandler(deps)
    onKey(keyEvent('ArrowDown').e)
    expect(deps.navigateHistory).toHaveBeenCalledWith(1)
    onKey(keyEvent('Escape').e)
    expect(deps.closeParamsPanel).toHaveBeenCalledOnce()
  })

  it.each([['MemoPanel', memoPanelSource], ['ChatMiniPanel', chatPanelSource]])(
    '%s handles its own keys on the panel root instead of joining the app modal stack', (_name, source) => {
      expect(source).not.toMatch(/useModalLayer\s*\(/)
      expect(source).toMatch(/<aside\s[^>]*class="dock-panel[^"]*"[^>]*@keydown="onPanelKey"/)
      expect(source).toMatch(/handleDockPanelKeydown\(/)
    })
})
