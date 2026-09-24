import { afterEach, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'
import presetSource from './PresetManagerModal.vue?raw'
import wildcardSource from './WildcardManagerModal.vue?raw'
import { byClass, byTag, mountFake, type Mounted } from '../../testing/fakeDomRenderer'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import * as modalLayer from '../../composables/useModalLayer'
import * as wildcardFile from '../../utils/wildcardFile'
import { appModalStack } from '../../utils/modalStack'
import { createPresetManager } from '../../composables/usePresetManager'
import { createWildcardManager } from '../../composables/useWildcardManager'

// 매니저 모달은 표시만 한다 — 상태는 모듈 싱글턴 composable 에 있고, 모달은 열려 있는 동안 앱 모달
// 스택에 올라가 전역 ESC(App.vue)가 닫고 ↑/↓ 히스토리 이동이 막힌다.

let mounted: Mounted | null = null
afterEach(() => { mounted?.unmount(); mounted = null })

function presetModal() {
  const bk = {
    getPresetList: (cb: (json: string) => void) => cb(JSON.stringify(['a', 'b'])),
    getPresetData: (name: string, cb: (json: string) => void) => cb(JSON.stringify({ name, steps: 20 })),
  }
  const mgr = createPresetManager({ getBackend: async () => bk, requestAction: vi.fn() })
  const component = compileSfc(presetSource, 'preset-modal-test', {
    '../../composables/usePresetManager': { usePresetManager: () => mgr },
    '../../composables/useModalLayer': modalLayer,
  })
  return { mgr, component }
}

it('an open manager sits on the app modal stack; the global ESC closes it and it leaves at once', async () => {
  const { mgr, component } = presetModal()
  mgr.openPresetManager()
  mounted = mountFake(component)
  await vi.waitFor(() => expect(byClass(mounted!.root, 'pm-item')).toHaveLength(2))
  expect(appModalStack.isAnyOpen()).toBe(true)

  expect(appModalStack.closeTop()).toBe(true)          // App.vue 의 전역 ESC 가 부르는 것
  expect(mgr.showPresetManager.value).toBe(false)
  expect(appModalStack.isAnyOpen()).toBe(false)        // 나가는 fade 동안에도 ↑/↓ 는 바로 풀린다
})

it('clicking a preset selects it and shows its preview through the composable', async () => {
  const { mgr, component } = presetModal()
  mgr.openPresetManager()
  mounted = mountFake(component)
  await vi.waitFor(() => expect(byClass(mounted!.root, 'pm-item')).toHaveLength(2))
  byClass(mounted.root, 'pm-item')[1].props.onClick()
  await vi.waitFor(() => expect(byClass(mounted!.root, 'pm-key').map(n => n.textContent)).toEqual(['name', 'steps']))
  expect(mgr.selectedPreset.value).toBe('b')
  expect(byClass(mounted.root, 'pm-item')[1].props.class.split(' ')).toContain('active')
})

it('double-clicking the wildcard title swaps in a focused rename input (template ref wiring)', async () => {
  const bk = {
    renameWildcard: vi.fn(),
  }
  const mgr = createWildcardManager({ getBackend: async () => bk, addToast: vi.fn(), storeWidgets: reactive({}), writeClipboard: vi.fn() })
  mgr.wildcards.value = [{ name: 'hair', file: 'hair.txt', tags: ['red hair'], lines: ['# c', 'red hair'] }]
  mgr.openWildcardByName('hair')
  const component = compileSfc(wildcardSource, 'wildcard-modal-test', {
    '../CustomSelect.vue': stubComponentModule('csel'),
    '../ToggleSwitch.vue': stubComponentModule('toggle'),
    '../../composables/useWildcardManager': { useWildcardManager: () => mgr },
    '../../composables/useModalLayer': modalLayer,
    '../../utils/wildcardFile': wildcardFile,
  }, { ...(await import('vue')), vModelText: {} })
  mounted = mountFake(component)
  await nextTick()
  expect(byClass(mounted.root, 'wc-block-input')).toHaveLength(2)
  expect(byClass(mounted.root, 'wc-block-row')[0].props.class.split(' ')).toContain('comment')   // # 주석 줄

  byTag(mounted.root, 'h4')[0].props.onDblclick()
  await nextTick(); await nextTick()
  const input = byClass(mounted.root, 'wc-rename-input')[0]
  expect(input).toBeTruthy()
  expect(input.focused).toBe(true)
  expect(mgr.wcNewName.value).toBe('hair')

  expect(appModalStack.isAnyOpen()).toBe(true)
  appModalStack.closeTop()
  expect(mgr.showWcManager.value).toBe(false)
})
