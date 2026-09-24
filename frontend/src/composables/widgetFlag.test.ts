import { expect, it, vi } from 'vitest'
import { reactive } from 'vue'
import { widgetFlag } from './widgetFlag'

it("reads only the exact string 'true' as on", () => {
  const widgets = reactive<Record<string, any>>({ a: 'true', b: 'false', c: '', d: true })
  expect(widgetFlag(widgets, 'a').value).toBe(true)
  expect(widgetFlag(widgets, 'b').value).toBe(false)
  expect(widgetFlag(widgets, 'c').value).toBe(false)
  expect(widgetFlag(widgets, 'd').value).toBe(false)       // boolean true 는 프록시 형식이 아니다
  expect(widgetFlag(widgets, 'missing').value).toBe(false)
})

it("writes 'true'/'false' strings and then calls onSet", () => {
  const widgets = reactive<Record<string, any>>({})
  const seen: Array<[boolean, string]> = []
  const flag = widgetFlag(widgets, 'random_res_check', (v) => seen.push([v, widgets.random_res_check]))
  flag.value = true
  flag.value = false
  expect(seen).toEqual([[true, 'true'], [false, 'false']])
})

it('follows later store changes (the value lives in the store, not the flag)', () => {
  const widgets = reactive<Record<string, any>>({ hires_options_group: 'false' })
  const flag = widgetFlag(widgets, 'hires_options_group')
  widgets.hires_options_group = 'true'
  expect(flag.value).toBe(true)
  const spy = vi.fn()
  widgetFlag(widgets, 'x', spy).value = true
  expect(spy).toHaveBeenCalledWith(true)
})
