import { describe, expect, it } from 'vitest'
import app from './App.vue?raw'
import dock from './components/dock/QuickDock.vue?raw'

function resolveZ(raw: string | undefined, styleCss: string): number {
  if (!raw) return NaN
  const trimmed = raw.trim()
  if (/^\d+$/.test(trimmed)) return Number(trimmed)
  const varMatch = trimmed.match(/^var\((--[\w-]+)\)$/)
  if (!varMatch) return NaN
  const def = styleCss.match(new RegExp(`${varMatch[1]}\\s*:\\s*(\\d+)`))
  return def ? Number(def[1]) : NaN
}

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

  it('keeps the trash menu above the queue pin so the pin cannot intercept its click', async () => {
    const fs = await import('node:fs' as string)
    const styleCss: string = fs.readFileSync(new URL('./style.css', import.meta.url), 'utf8')
    expect(app).toMatch(/<Teleport to="body">\s*<transition name="pop">\s*<div v-if="ctxMenu.show"/)
    const menuZ = resolveZ(app.match(/\.modern-ctx-menu\s*\{[^}]*z-index:\s*([^;]+)/)?.[1], styleCss)
    const dockZ = [...dock.matchAll(/z-index:\s*([^;]+)/g)].map(match => resolveZ(match[1], styleCss)).filter(Number.isFinite)
    expect(dockZ.length).toBeGreaterThan(0)
    expect(menuZ).toBeGreaterThan(Math.max(...dockZ))
  })
})
