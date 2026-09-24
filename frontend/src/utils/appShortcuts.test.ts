import { describe, expect, it, vi } from 'vitest'
import { createAppKeydownHandler, isEditableElement, type AppShortcutDeps, type AppShortcutEvent } from './appShortcuts'
import { createModalStack, type ModalStack } from './modalStack'

function setup(overrides: Partial<AppShortcutDeps> = {}) {
  const stack: ModalStack = overrides.modalStack ?? createModalStack()
  const deps = {
    isGateOpen: () => false,
    generate: vi.fn(),
    saveSettings: vi.fn(),
    reloadHistory: vi.fn(),
    navigateTabs: vi.fn(),
    isParamsPanelOpen: () => false,
    closeParamsPanel: vi.fn(),
    hasHistory: () => true,
    navigateHistory: vi.fn(),
    navigateHistoryEdge: vi.fn(),
    activeElement: () => ({ tagName: 'BODY', isContentEditable: false }),
    jumpModifier: () => null,
    ...overrides,
    modalStack: stack,
  }
  return { deps, stack, handle: createAppKeydownHandler(deps) }
}

/** preventDefault 가 defaultPrevented 를 세우는 — 실제 KeyboardEvent 처럼 — 평범한 객체. */
function key(k: string, mods: Partial<Pick<AppShortcutEvent, 'ctrlKey' | 'shiftKey' | 'altKey' | 'metaKey' | 'defaultPrevented'>> = {}) {
  const ev = { key: k, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, defaultPrevented: false, preventDefault: vi.fn(), ...mods }
  ev.preventDefault = vi.fn(() => { ev.defaultPrevented = true })
  return ev
}

/** 포커스된 위젯(CustomSelect · 궤도 · 채팅 입력)이 먼저 받아 preventDefault 한 뒤 document 로 올라온 키. */
function handled(k: string, mods: Partial<Pick<AppShortcutEvent, 'ctrlKey' | 'shiftKey' | 'altKey' | 'metaKey'>> = {}) {
  const ev = key(k, mods)
  ev.preventDefault()
  ev.preventDefault.mockClear()
  return ev
}

describe('history ↑/↓ behind a modal (audit #186)', () => {
  it('moves the history when nothing is open', () => {
    const { deps, handle } = setup()
    const down = key('ArrowDown')
    handle(down)
    expect(deps.navigateHistory).toHaveBeenCalledWith(1)
    expect(down.preventDefault).toHaveBeenCalled()
    handle(key('ArrowUp'))
    expect(deps.navigateHistory).toHaveBeenLastCalledWith(-1)
  })

  it('does nothing while any modal layer is on the stack — self-ESC modals (LoRA · 조건부 · A/B …) included', () => {
    const { deps, stack, handle } = setup()
    const release = stack.open({})          // useModalLayer() 무인자 — 제 ESC 를 가진 모달
    const up = key('ArrowUp')
    handle(up)
    handle(key('ArrowDown', { shiftKey: true }))
    expect(deps.navigateHistory).not.toHaveBeenCalled()
    expect(deps.navigateHistoryEdge).not.toHaveBeenCalled()
    expect(up.preventDefault).not.toHaveBeenCalled()   // 모달 안의 목록 스크롤 등은 그대로 산다

    release()                                // 모달이 닫히면 바로 다시 움직인다
    handle(key('ArrowUp'))
    expect(deps.navigateHistory).toHaveBeenCalledWith(-1)
  })

  it('leaves the arrows to a focused input, textarea, select or contenteditable', () => {
    for (const el of [{ tagName: 'INPUT' }, { tagName: 'textarea' }, { tagName: 'SELECT' }, { tagName: 'DIV', isContentEditable: true }]) {
      const { deps, handle } = setup({ activeElement: () => el })
      const e = key('ArrowDown')
      handle(e)
      expect(deps.navigateHistory).not.toHaveBeenCalled()
      expect(e.preventDefault).not.toHaveBeenCalled()
    }
  })

  it('does not swallow the arrows when the history is empty', () => {
    const { deps, handle } = setup({ hasHistory: () => false })
    const e = key('ArrowDown')
    handle(e)
    expect(deps.navigateHistory).not.toHaveBeenCalled()
    expect(e.preventDefault).not.toHaveBeenCalled()
  })

  it('jumps to the ends with the configured modifier (default Shift)', () => {
    const { deps, handle } = setup()
    handle(key('ArrowDown', { shiftKey: true }))
    expect(deps.navigateHistoryEdge).toHaveBeenCalledWith('bottom')
    handle(key('ArrowUp', { shiftKey: true }))
    expect(deps.navigateHistoryEdge).toHaveBeenLastCalledWith('top')
    expect(deps.navigateHistory).not.toHaveBeenCalled()

    const alt = setup({ jumpModifier: () => 'altKey' })
    alt.handle(key('ArrowDown', { shiftKey: true }))       // 설정이 Alt 면 Shift 는 한 칸 이동
    expect(alt.deps.navigateHistory).toHaveBeenCalledWith(1)
    alt.handle(key('ArrowUp', { altKey: true }))
    expect(alt.deps.navigateHistoryEdge).toHaveBeenCalledWith('top')
  })
})

describe('ESC', () => {
  it('closes only the top modal and leaves the params column alone', () => {
    const closeUnder = vi.fn()
    const closeTop = vi.fn()
    const { deps, stack, handle } = setup({ isParamsPanelOpen: () => true })
    stack.open({ close: closeUnder })
    stack.open({ close: closeTop })
    handle(key('Escape'))
    expect(closeTop).toHaveBeenCalledTimes(1)
    expect(closeUnder).not.toHaveBeenCalled()
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
  })

  it('a self-ESC modal on top eats the key instead of letting it close the params column behind', () => {
    const { deps, stack, handle } = setup({ isParamsPanelOpen: () => true })
    stack.open({})
    handle(key('Escape'))
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
  })

  it('returns the params column to the prompt when no modal is open', () => {
    const { deps, handle } = setup({ isParamsPanelOpen: () => true })
    handle(key('Escape'))
    expect(deps.closeParamsPanel).toHaveBeenCalledTimes(1)
    const closed = setup({ isParamsPanelOpen: () => false })
    closed.handle(key('Escape'))
    expect(closed.deps.closeParamsPanel).not.toHaveBeenCalled()
  })
})

// 파라미터 열의 샘플러 CustomSelect 에서 ↓ 를 누르면 드롭다운이 열리면서 히스토리도 넘어갔고, 열린
// 드롭다운을 ESC 로 닫으면 파라미터 열(또는 와일드카드 관리 모달)까지 닫혔다 — 위젯이 preventDefault 만
// 하고 올려 보낸 키를 여기서 한 번 더 처리했다.
describe('keys a focused widget already handled', () => {
  it('↑/↓ (with or without the jump modifier) leave the history alone', () => {
    const { deps, handle } = setup()
    for (const e of [handled('ArrowDown'), handled('ArrowUp'), handled('ArrowDown', { shiftKey: true }), handled('ArrowUp', { shiftKey: true })]) {
      handle(e)
      expect(e.preventDefault).not.toHaveBeenCalled()
    }
    expect(deps.navigateHistory).not.toHaveBeenCalled()
    expect(deps.navigateHistoryEdge).not.toHaveBeenCalled()
    handle(key('ArrowDown'))                  // 위젯이 안 쓴 화살표는 그대로 히스토리
    expect(deps.navigateHistory).toHaveBeenCalledWith(1)
  })

  it('ESC does not also close the params column', () => {
    const { deps, handle } = setup({ isParamsPanelOpen: () => true })
    handle(handled('Escape'))
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
    handle(key('Escape'))
    expect(deps.closeParamsPanel).toHaveBeenCalledTimes(1)
  })

  it('ESC does not also close the modal the widget sits in, and the layer stays on the stack', () => {
    const closeModal = vi.fn()
    const { stack, handle } = setup({ isParamsPanelOpen: () => true })
    stack.open({ close: closeModal })
    handle(handled('Escape'))
    expect(closeModal).not.toHaveBeenCalled()
    expect(stack.isAnyOpen()).toBe(true)
    handle(key('Escape'))                     // 드롭다운이 닫힌 뒤의 ESC 는 모달을 닫는다
    expect(closeModal).toHaveBeenCalledTimes(1)
  })

  it('Ctrl+G / Ctrl+S / F5 / Ctrl+Tab still run — the editor preventDefaults Ctrl+S on its own document listener', () => {
    const { deps, handle } = setup()
    handle(handled('g', { ctrlKey: true }))
    handle(handled('s', { ctrlKey: true }))
    handle(handled('F5'))
    handle(handled('Tab', { ctrlKey: true }))
    expect(deps.generate).toHaveBeenCalledTimes(1)
    expect(deps.saveSettings).toHaveBeenCalledTimes(1)
    expect(deps.reloadHistory).toHaveBeenCalledTimes(1)
    expect(deps.navigateTabs).toHaveBeenCalledWith(1)
  })
})

describe('other shortcuts', () => {
  it('Ctrl+G generates, Ctrl+S saves, F5 reloads the history, Ctrl+(Shift+)Tab cycles tabs', () => {
    const { deps, handle } = setup()
    const g = key('g', { ctrlKey: true })
    handle(g)
    handle(key('s', { ctrlKey: true }))
    handle(key('F5'))
    handle(key('Tab', { ctrlKey: true }))
    handle(key('Tab', { ctrlKey: true, shiftKey: true }))
    expect(g.preventDefault).toHaveBeenCalled()
    expect(deps.generate).toHaveBeenCalledTimes(1)
    expect(deps.saveSettings).toHaveBeenCalledTimes(1)
    expect(deps.reloadHistory).toHaveBeenCalledTimes(1)
    expect(deps.navigateTabs).toHaveBeenCalledTimes(2)
    expect(deps.navigateTabs).toHaveBeenNthCalledWith(1, 1)
    expect(deps.navigateTabs).toHaveBeenNthCalledWith(2, -1)
  })

  it('plain g / s / Tab are left alone', () => {
    const { deps, handle } = setup()
    const tab = key('Tab')
    handle(key('g'))
    handle(key('s'))
    handle(tab)
    expect(deps.generate).not.toHaveBeenCalled()
    expect(deps.saveSettings).not.toHaveBeenCalled()
    expect(deps.navigateTabs).not.toHaveBeenCalled()
    expect(tab.preventDefault).not.toHaveBeenCalled()
  })

  it('everything is dead while the backend gate is open', () => {
    const closeModal = vi.fn()
    const { deps, stack, handle } = setup({ isGateOpen: () => true, isParamsPanelOpen: () => true })
    stack.open({ close: closeModal })
    for (const e of [key('g', { ctrlKey: true }), key('s', { ctrlKey: true }), key('F5'), key('Tab', { ctrlKey: true }), key('Escape')]) {
      handle(e)
      expect(e.preventDefault).not.toHaveBeenCalled()
    }
    const bare = setup({ isGateOpen: () => true })
    bare.handle(key('ArrowDown'))
    expect(bare.deps.navigateHistory).not.toHaveBeenCalled()
    expect(deps.generate).not.toHaveBeenCalled()
    expect(deps.saveSettings).not.toHaveBeenCalled()
    expect(deps.reloadHistory).not.toHaveBeenCalled()
    expect(deps.navigateTabs).not.toHaveBeenCalled()
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
    expect(closeModal).not.toHaveBeenCalled()
  })
})

it('isEditableElement', () => {
  expect(isEditableElement(null)).toBe(false)
  expect(isEditableElement({ tagName: 'BUTTON' })).toBe(false)
  expect(isEditableElement({ tagName: 'Input' })).toBe(true)
  expect(isEditableElement({ tagName: 'DIV', isContentEditable: true })).toBe(true)
})
