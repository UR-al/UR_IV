import { describe, expect, it } from 'vitest'
import app from './App.vue?raw'
import dock from './components/dock/QuickDock.vue?raw'

describe('history trash keyboard wiring', () => {
  it('scopes deletion to the focusable history panel, not the global document handler', () => {
    expect(app).toMatch(/<aside ref="historyPanel"[^>]*tabindex="-1"[^>]*@keydown="onHistoryTrashKeydown"/s)
    expect(app).toContain('@click="pickHistoryImage(img)"')
    expect(app).toContain('historyPanel.value?.focus({ preventScroll: true })')
    expect(app).toContain("onBackendEvent('imageRestoreResult', historyTrash.onRestoreResult)")
    expect(app).toContain('onUnmounted(() => historyTrash.dispose())')
    const globalHandler = app.split("document.addEventListener('keydown', createAppKeydownHandler(")[1]?.split('}))')[0]
    expect(globalHandler).toBeDefined()
    expect(globalHandler).not.toContain('historyTrash')
  })

  it('keeps the trash menu above the queue pin so the pin cannot intercept its click', () => {
    expect(app).toMatch(/<Teleport to="body">\s*<transition name="pop">\s*<div v-if="ctxMenu.show"/)
    const menuZ = Number(app.match(/\.modern-ctx-menu\s*\{[^}]*z-index:\s*(\d+)/)?.[1])
    const dockZ = [...dock.matchAll(/z-index:\s*(\d+)/g)].map(match => Number(match[1]))
    expect(menuZ).toBeGreaterThan(Math.max(...dockZ))
  })
})
