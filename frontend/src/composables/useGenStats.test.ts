import { expect, it, vi } from 'vitest'
import { createGenStats } from './useGenStats'

it('opening the modal shows it and merges a fresh stats payload', async () => {
  const bk = { getGenStats: vi.fn((cb: (json: string) => void) => cb(JSON.stringify({ total: 3, success_rate: 66.7, recent: [{ timestamp: 't' }] }))) }
  const stats = createGenStats({ getBackend: async () => bk })
  stats.openStatsModal()
  expect(stats.showStatsModal.value).toBe(true)
  await Promise.resolve(); await Promise.resolve()
  expect(stats.genStats.total).toBe(3)
  expect(stats.genStats.success_rate).toBe(66.7)
  expect(stats.genStats.daily).toEqual([])          // 없는 키는 기본값 그대로
  stats.closeStatsModal()
  expect(stats.showStatsModal.value).toBe(false)
})

it('ignores a broken payload and a backend without the slot', async () => {
  const broken = createGenStats({ getBackend: async () => ({ getGenStats: (cb: (json: string) => void) => cb('nope') }) })
  await broken.loadGenStats()
  expect(broken.genStats.total).toBe(0)
  const none = createGenStats({ getBackend: async () => ({}) })
  await expect(none.loadGenStats()).resolves.toBeUndefined()
})
