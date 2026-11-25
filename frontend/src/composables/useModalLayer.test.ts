import { afterEach, expect, it, vi } from 'vitest'
import { KeepAlive, defineComponent, h, nextTick, ref } from 'vue'
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

it('releases its stack layer when deactivated inside KeepAlive and re-registers on reactivation (A-3)', async () => {
  const stack = createModalStack()
  const modalOpen = ref(true)
  const close = vi.fn(() => { modalOpen.value = false })
  const activeTab = ref<'gallery' | 'other'>('gallery')

  const GalleryLike = defineComponent({
    name: 'GalleryLike',
    setup() {
      useModalLayer({ stack, isOpen: () => modalOpen.value, close })
      return () => h('div', 'gallery')
    },
  })
  const OtherTab = defineComponent({
    name: 'OtherTab',
    setup: () => () => h('div', 'other'),
  })
  const Host = defineComponent({
    setup: () => () => h(KeepAlive, null, {
      default: () => (activeTab.value === 'gallery'
        ? h(GalleryLike, { key: 'gallery' })
        : h(OtherTab, { key: 'other' })),
    }),
  })

  mounted = mountFake(Host)
  expect(stack.isAnyOpen()).toBe(true)

  // 다른 탭으로 이동(KeepAlive 비활성화) — 뷰어가 열린 채여도 다른 탭의 단축키를 막으면 안 된다
  activeTab.value = 'other'
  await nextTick()
  expect(stack.isAnyOpen()).toBe(false)

  // 비활성화 중 플래그가 토글돼도 스택에 다시 올라가면 안 된다
  modalOpen.value = false
  await nextTick()
  modalOpen.value = true
  await nextTick()
  expect(stack.isAnyOpen()).toBe(false)

  // 다시 갤러리 탭으로 복귀 — 뷰어가 여전히 열려 있으면 스택에 재등록된다
  activeTab.value = 'gallery'
  await nextTick()
  expect(stack.isAnyOpen()).toBe(true)

  // 비활성화 중 닫힌 상태로 복귀하면 스택에 올라가지 않는다
  activeTab.value = 'other'
  await nextTick()
  expect(stack.isAnyOpen()).toBe(false)
  modalOpen.value = false
  await nextTick()
  activeTab.value = 'gallery'
  await nextTick()
  expect(stack.isAnyOpen()).toBe(false)
})
