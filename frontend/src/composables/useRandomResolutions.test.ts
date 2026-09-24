import { expect, it, vi } from 'vitest'
import { reactive } from 'vue'
import { useRandomResolutions } from './useRandomResolutions'

function setup() {
  const storeWidgets = reactive<Record<string, any>>({ random_res_check: 'false' })
  const requestAction = vi.fn()
  const bk = { getRandomResolutions: vi.fn((cb: (json: string) => void) => cb(JSON.stringify([[1024, 1024, 'square']]))) }
  const rr = useRandomResolutions({ storeWidgets, getBackend: async () => bk, requestAction })
  return { rr, storeWidgets, requestAction, bk }
}

const flush = async () => { await Promise.resolve(); await Promise.resolve() }

it('turning the toggle on writes the widget and fetches the list; off does not fetch', async () => {
  const { rr, storeWidgets, bk } = setup()
  rr.randomResEnabled.value = true
  expect(storeWidgets.random_res_check).toBe('true')
  await flush()
  expect(rr.randomResList.value).toEqual([[1024, 1024, 'square']])
  rr.randomResEnabled.value = false
  expect(storeWidgets.random_res_check).toBe('false')
  await flush()
  expect(bk.getRandomResolutions).toHaveBeenCalledTimes(1)
})

it('adds a rounded entry and sends the whole list; too small is ignored', async () => {
  const { rr, requestAction } = setup()
  await rr.loadRandomResList()
  rr.newResW.value = 900
  rr.newResH.value = 1100
  rr.addRandomRes()
  expect(rr.randomResList.value).toEqual([[1024, 1024, 'square'], [904, 1104, '904x1104']])
  expect(requestAction).toHaveBeenLastCalledWith('set_random_resolutions', { list: rr.randomResList.value })
  rr.newResW.value = 100
  rr.addRandomRes()
  expect(requestAction).toHaveBeenCalledTimes(1)
})

it('removes an entry and sends the rest', async () => {
  const { rr, requestAction } = setup()
  await rr.loadRandomResList()
  rr.removeRandomRes(0)
  expect(rr.randomResList.value).toEqual([])
  expect(requestAction).toHaveBeenCalledWith('set_random_resolutions', { list: [] })
})
