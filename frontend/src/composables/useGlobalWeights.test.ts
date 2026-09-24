import { expect, it, vi } from 'vitest'
import { createGlobalWeights } from './useGlobalWeights'

function setup() {
  const handlers = new Map<string, (...args: any[]) => void>()
  const requestAction = vi.fn()
  const mgr = createGlobalWeights({
    requestAction,
    onBackendEvent: (name, cb) => { handlers.set(name, cb); return () => handlers.delete(name) },
  })
  return { mgr, handlers, requestAction }
}

it('receives the list from globalWeightsLoaded only after bind(), replacing it wholesale', () => {
  const { mgr, handlers } = setup()
  expect(handlers.size).toBe(0)
  mgr.bind()
  handlers.get('globalWeightsLoaded')!(JSON.stringify([{ tag: 'a', weight: 120 }]))
  handlers.get('globalWeightsLoaded')!(JSON.stringify([{ tag: 'b', weight: 80 }]))
  expect(mgr.globalWeights.map(w => w.tag)).toEqual(['b'])
  handlers.get('globalWeightsLoaded')!('not json')
  expect(mgr.globalWeights.map(w => w.tag)).toEqual(['b'])
})

it('saves only rows with a tag and closes the manager', () => {
  const { mgr, requestAction } = setup()
  mgr.openWeightManager()
  mgr.addWeightRow()
  mgr.globalWeights[0].tag = 'long hair'
  mgr.addWeightRow()                         // 빈 행
  mgr.saveGlobalWeights()
  expect(requestAction).toHaveBeenCalledWith('save_global_weights', { weights: [{ tag: 'long hair', weight: 100 }] })
  expect(mgr.showWeightManager.value).toBe(false)
})

it('row keys stay stable while the tag is edited', () => {
  const { mgr } = setup()
  mgr.addWeightRow()
  const row = mgr.globalWeights[0]
  const key = mgr.weightRowKey(row)
  row.tag = 'x'
  expect(mgr.weightRowKey(row)).toBe(key)
  mgr.removeWeightRow(0)
  expect(mgr.globalWeights).toHaveLength(0)
})

it('weightedPrompt is null when nothing applies and the weighted text otherwise', () => {
  const { mgr } = setup()
  expect(mgr.weightedPrompt('1girl, long hair')).toBeNull()      // 가중치 없음
  mgr.applyLoaded(JSON.stringify([{ tag: 'long hair', weight: 120 }]))
  expect(mgr.weightedPrompt('1girl, smile')).toBeNull()          // 바뀐 게 없음
  expect(mgr.weightedPrompt('1girl, long hair')).toBe('1girl, (long hair:1.20)')
})
