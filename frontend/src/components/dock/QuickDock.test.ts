import { afterEach, expect, it, vi } from 'vitest'
import { createSSRApp } from 'vue'
import { renderToString } from '@vue/server-renderer'
import QuickDock from './QuickDock.vue'
import { useQuickDock } from '../../composables/useQuickDock'

// 대기열 처리와 모달 등록은 별도 테스트 대상 — 실제 도크의 표시 계약만 확인한다.
vi.mock('../QueuePanel.vue', () => ({ default: { render: () => null } }))
vi.mock('../../composables/useModalLayer', () => ({ useModalLayer: () => {} }))

const dock = useQuickDock()
afterEach(() => { dock.collapse(); dock.closePanel() })

async function renderDock() {
  const app = createSSRApp(QuickDock)
  app.component('Icon', { render: () => null })
  return renderToString(app)
}

it('does not show a hover tooltip on the queue pin while keeping its accessible name', async () => {
  const html = await renderDock()
  const pin = html.match(/<button\b[^>]*class="queue-pin[^>]*>/)?.[0]
  expect(pin).toBeDefined()
  expect(pin).not.toMatch(/\stitle=/)
  expect(pin).toContain('aria-label="대기열 · 대화 · 메모장"')
  expect(pin).toContain('aria-expanded="false"')
  expect(html).not.toContain('id="quick-dock-menu"')
})

it('keeps the queue, chat and memo choices available when the pin is activated', async () => {
  dock.toggleExpanded()
  const html = await renderDock()
  expect(html).toContain('aria-expanded="true"')
  expect(html).toContain('id="quick-dock-menu"')
  for (const label of ['대기열', '대화', '메모장']) {
    expect(html).toContain(`aria-label="${label}"`)
  }
})
