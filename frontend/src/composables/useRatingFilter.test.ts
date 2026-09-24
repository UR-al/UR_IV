import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// rating 필터는 마운트 때 Python 으로 보내지 않는다 — ui_prefs 복원 직후에만 보낸다(감사 #107).
const mocks = vi.hoisted(() => ({ actionSpy: vi.fn() }))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))

import { useRatingFilter } from './useRatingFilter.js'

let storage: Map<string, string>
const pushes = () => mocks.actionSpy.mock.calls.filter(([n]) => n === 'set_rating_filter').map(([, p]) => p.ratings)

beforeEach(() => {
  mocks.actionSpy.mockReset()
  storage = new Map([['ratingFilter', JSON.stringify([true, true, true, false])]])
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (k: string) => storage.get(k) ?? null,
      setItem: (k: string, v: string) => { storage.set(k, String(v)) },
    },
  })
})
afterEach(() => { vi.unstubAllGlobals() })

it('shows the cached filter but does not push it at setup', () => {
  const { ratingFilters } = useRatingFilter({ saveUiPrefs: vi.fn() })
  expect(ratingFilters.map(r => r.on)).toEqual([true, true, true, false])
  expect(pushes()).toEqual([])
})

it('file value wins and is pushed once', () => {
  const saveUiPrefs = vi.fn()
  const { ratingFilters, restoreFromPrefs } = useRatingFilter({ saveUiPrefs })
  restoreFromPrefs({ ratingFilter: [true, false, false, false] })
  expect(ratingFilters.map(r => r.on)).toEqual([true, false, false, false])
  expect(pushes()).toEqual([['g']])
  expect(saveUiPrefs).not.toHaveBeenCalled()
  expect(storage.get('ratingFilter')).toBe('[true,false,false,false]')
})

it('when ui_prefs never stored a filter, the on-screen value is migrated and pushed', () => {
  const saveUiPrefs = vi.fn()
  const { restoreFromPrefs } = useRatingFilter({ saveUiPrefs })
  restoreFromPrefs({ theme: 'dark' })
  expect(saveUiPrefs).toHaveBeenCalledWith({ ratingFilter: [true, true, true, false] })
  expect(pushes()).toEqual([['g', 's', 'q']])
})
