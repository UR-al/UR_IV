import { describe, expect, it } from 'vitest'
import { createQuickDock, useQuickDock } from './useQuickDock'

describe('quick dock state', () => {
  it('opening a panel collapses the icon row and replaces any other open panel', () => {
    const dock = createQuickDock()
    dock.toggleExpanded()
    expect(dock.expanded.value).toBe(true)
    dock.openPanel('memo')
    expect(dock.expanded.value).toBe(false)
    expect(dock.activePanel.value).toBe('memo')
    dock.openPanel('queue')
    expect(dock.isOpen('queue')).toBe(true)
    expect(dock.isOpen('memo')).toBe(false)
  })

  it('toggling the same icon closes its panel', () => {
    const dock = createQuickDock()
    dock.togglePanel('chat')
    expect(dock.activePanel.value).toBe('chat')
    dock.toggleExpanded()
    dock.togglePanel('chat')
    expect(dock.activePanel.value).toBeNull()
    expect(dock.expanded.value).toBe(false)
  })

  it('a late close for another panel does not close the one that is open now', () => {
    const dock = createQuickDock()
    dock.openPanel('chat')
    dock.closePanel('queue')
    expect(dock.activePanel.value).toBe('chat')
    dock.closePanel()
    expect(dock.activePanel.value).toBeNull()
  })

  it('the app shares one dock', () => {
    expect(useQuickDock()).toBe(useQuickDock())
  })
})
