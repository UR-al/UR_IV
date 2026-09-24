import { expect, it } from 'vitest'
import { reactive } from 'vue'
import { useParamItems } from './useParamItems'

it('reads each dropdown from its widget items and falls back to the old defaults', () => {
  const props = reactive<Record<string, Record<string, any>>>({})
  const getProperty = (id: string, prop: string) => props[id]?.[prop] ?? ''
  const items = useParamItems(getProperty)
  expect(items.samplerItems.value).toEqual([])
  expect(items.sam3CheckpointItems.value).toEqual(['sam3.pt'])
  expect(items.adCheckpointItems.value).toEqual(['Use same checkpoint'])
  expect(items.adVaeItems.value).toEqual(['Use same VAE'])
  expect(items.hiresCheckpointItems.value).toEqual(['Use same checkpoint'])
  expect(items.hiresSamplerItems.value).toEqual(['Use same sampler'])
  expect(items.hiresSchedulerItems.value).toEqual(['Use same scheduler'])

  props.sampler_combo = { items: ['Euler a', 'DPM++ 2M'] }
  props._ad_s1_vae = { items: ['Use same VAE', 'sdxl_vae'] }
  expect(items.samplerItems.value).toEqual(['Euler a', 'DPM++ 2M'])
  expect(items.adVaeItems.value).toEqual(['Use same VAE', 'sdxl_vae'])
})
