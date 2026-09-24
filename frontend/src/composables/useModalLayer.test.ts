import { afterEach, expect, it, vi } from 'vitest'
import { defineComponent, h, nextTick, ref } from 'vue'
import { createModalStack } from '../utils/modalStack'
import { useModalLayer, type ModalLayerOptions } from './useModalLayer'
import { mountFake, type Mounted } from '../testing/fakeDomRenderer'

let mounted: Mounted | null = null
afterEach(() => { mounted?.unmount(); mounted = null })

function layerComponent(options: ModalLayerOptions) {
  return defineComponent({ setup() { useModalLayer(options); return () => h('div') } })
}

it('a v-if modal is on the stack exactly while it is mounted', () => {
  const stack = createModalStack()
  mounted = mountFake(layerComponent({ stack }))
  expect(stack.isAnyOpen()).toBe(true)
  // 제 ESC 를 가진 모달 — 전역 ESC 는 먹기만 하고 아무것도 닫지 않는다
  expect(stack.closeTop()).toBe(true)
  mounted.unmount(); mounted = null
  expect(stack.isAnyOpen()).toBe(false)
})

it('leaves the stack the moment its flag turns off, even while a leave transition keeps it mounted', async () => {
  const stack = createModalStack()
  const show = ref(true)
  const close = vi.fn(() => { show.value = false })
  mounted = mountFake(layerComponent({ stack, isOpen: () => show.value, close }))
  expect(stack.size()).toBe(1)

  expect(stack.closeTop()).toBe(true)   // 전역 ESC
  expect(close).toHaveBeenCalledTimes(1)
  expect(stack.isAnyOpen()).toBe(false)  // 동기 — 다음 ↑/↓ 부터 바로 히스토리가 움직인다

  show.value = true                      // 다시 열림(같은 인스턴스)
  await nextTick()
  expect(stack.size()).toBe(1)
  mounted.unmount(); mounted = null
  expect(stack.isAnyOpen()).toBe(false)
})

it('does not register a modal whose flag is already off when it mounts', () => {
  const stack = createModalStack()
  mounted = mountFake(layerComponent({ stack, isOpen: () => false }))
  expect(stack.isAnyOpen()).toBe(false)
})
