import { describe, expect, it, vi } from 'vitest'
import { createHistoryTrashKeydownHandler, type HistoryTrashKeyEvent } from './historyTrashShortcuts'

function harness() {
  const deps = {
    isEnabled: vi.fn(() => true), hasModal: vi.fn(() => false),
    activeElement: vi.fn<() => { tagName?: string; isContentEditable?: boolean } | null>(() => ({ tagName: 'ASIDE' })),
    deleteSelected: vi.fn(() => true), undo: vi.fn(() => true),
  }
  return { deps, handle: createHistoryTrashKeydownHandler(deps) }
}

function key(overrides: Partial<HistoryTrashKeyEvent> = {}): HistoryTrashKeyEvent {
  return { key: 'Delete', ctrlKey: false, metaKey: false, shiftKey: false, altKey: false,
    defaultPrevented: false, repeat: false, isComposing: false, preventDefault: vi.fn(), stopPropagation: vi.fn(), ...overrides }
}

describe('history trash shortcuts', () => {
  it('handles only a bare Delete and consumes the successfully dispatched event', () => {
    const h = harness(), event = key()
    h.handle(event)
    expect(h.deps.deleteSelected).toHaveBeenCalledOnce()
    expect(h.deps.undo).not.toHaveBeenCalled()
    expect(event.preventDefault).toHaveBeenCalledOnce()
    expect(event.stopPropagation).toHaveBeenCalledOnce()
    for (const modifier of ['ctrlKey', 'metaKey', 'shiftKey', 'altKey']) {
      h.handle(key({ [modifier]: true }))
    }
    expect(h.deps.deleteSelected).toHaveBeenCalledOnce()
  })

  it('supports Ctrl+Z and Cmd+Z but leaves redo and unrelated shortcuts alone', () => {
    const h = harness()
    for (const modifiers of [{ ctrlKey: true }, { metaKey: true }]) {
      const event = key({ key: 'Z', ...modifiers })
      h.handle(event)
      expect(event.preventDefault).toHaveBeenCalledOnce()
      expect(event.stopPropagation).toHaveBeenCalledOnce()
    }
    expect(h.deps.undo).toHaveBeenCalledTimes(2)
    for (const data of [{ key: 'z' }, { key: 'z', ctrlKey: true, shiftKey: true },
      { key: 'z', ctrlKey: true, altKey: true }, { key: 'y', ctrlKey: true }, { key: 'Backspace' }]) {
      const event = key(data)
      h.handle(event)
      expect(event.preventDefault).not.toHaveBeenCalled()
    }
    expect(h.deps.undo).toHaveBeenCalledTimes(2)
  })

  it('does not consume a key when there is no selection, no undo, or a pending request', () => {
    const h = harness()
    h.deps.deleteSelected.mockReturnValue(false)
    h.deps.undo.mockReturnValue(false)
    for (const event of [key(), key({ key: 'z', ctrlKey: true })]) {
      h.handle(event)
      expect(event.preventDefault).not.toHaveBeenCalled()
      expect(event.stopPropagation).not.toHaveBeenCalled()
    }
  })

  it('respects disabled history, open modals, IME, repeat, and already handled events', () => {
    for (const override of [{ defaultPrevented: true }, { repeat: true }, { isComposing: true }, { keyCode: 229 }]) {
      const h = harness(), event = key(override)
      h.handle(event)
      expect(h.deps.deleteSelected).not.toHaveBeenCalled()
      expect(event.preventDefault).not.toHaveBeenCalled()
    }
    for (const name of ['isEnabled', 'hasModal'] as const) {
      const h = harness()
      h.deps[name].mockReturnValue(name === 'hasModal')
      h.handle(key())
      expect(h.deps.deleteSelected).not.toHaveBeenCalled()
    }
  })

  it('leaves native text undo and deletion alone for both active and event target editable elements', () => {
    for (const element of [{ tagName: 'INPUT' }, { tagName: 'textarea' }, { tagName: 'SELECT' }, { isContentEditable: true }]) {
      for (const targetOnly of [false, true]) {
        const h = harness()
        if (!targetOnly) h.deps.activeElement.mockReturnValue(element)
        for (const event of [key({ target: targetOnly ? element : null }), key({ key: 'z', ctrlKey: true, target: targetOnly ? element : null })]) {
          h.handle(event)
          expect(h.deps.deleteSelected).not.toHaveBeenCalled()
          expect(h.deps.undo).not.toHaveBeenCalled()
          expect(event.preventDefault).not.toHaveBeenCalled()
        }
      }
    }
  })
})
