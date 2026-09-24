import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import source from './CustomSelect.vue?raw'
import * as dropdownPlacement from '../utils/dropdownPlacement'
import { compileSfc } from '../testing/compileSfc'
import { byClass, byTag, mountFake, type FakeNode, type Mounted } from '../testing/fakeDomRenderer'
import { createAppKeydownHandler, type AppShortcutEvent } from '../utils/appShortcuts'
import { createModalStack, type ModalStack } from '../utils/modalStack'

/**
 * 실제 CustomSelect → App 의 전역 keydown(utils/appShortcuts) — 같은 KeyboardEvent 가 포커스된 트리거
 * 버튼에서 document 로 버블링되는 순서 그대로.
 *
 * 버그: 파라미터 열의 샘플러·스케줄러 드롭다운에서 ↑/↓ 를 누르면 목록이 열리고 움직이면서 뒤의 생성
 * 히스토리도 한 칸씩 넘어갔고, 열린 목록을 ESC 로 닫으면 파라미터 열(와일드카드 관리 모달 안이면 그
 * 모달)까지 닫혔다. CustomSelect 는 쓴 키를 preventDefault 만 하고 올려 보낸다 — 전역 핸들러가 그걸
 * 보고 비켜야 한다. 판단 로직 자체는 utils/appShortcuts.test.ts 가 지킨다; 여기는 CustomSelect 가 쓴
 * 키에 실제로 표시를 남기는지(그래서 두 쪽이 맞물리는지)를 본다.
 */

// Teleport 는 가짜 렌더러에 'body' 가 없다 — 목록을 제자리(Fragment)에 그려 활성 항목까지 본다
const vueForTest = { ...Vue, Teleport: Vue.Fragment }
const CustomSelect = compileSfc(source, 'custom-select-shortcuts', { '../utils/dropdownPlacement': dropdownPlacement }, vueForTest)

/** 실제 KeyboardEvent 처럼 preventDefault 가 defaultPrevented 를 세운다(그래서 읽기 전용이 아니다). */
type KeyEvent = Omit<AppShortcutEvent, 'defaultPrevented'> & { defaultPrevented: boolean; stopPropagation: () => void }

function keyEvent(key: string): KeyEvent {
  const ev: KeyEvent = {
    key, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, defaultPrevented: false,
    preventDefault: () => { ev.defaultPrevented = true },
    stopPropagation: () => {},
  }
  return ev
}

let mounted: Mounted | null = null

beforeEach(() => {
  vi.stubGlobal('document', {
    addEventListener() {}, removeEventListener() {},
    documentElement: { clientWidth: 1200 },
    getElementById: () => null,
  })
  vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {}, innerHeight: 900 })
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => {})
})
afterEach(() => {
  mounted?.unmount(); mounted = null
  vi.unstubAllGlobals()
})

function setup(opts: { paramsPanelOpen?: boolean; stack?: ModalStack } = {}) {
  const value = Vue.ref('Euler')
  mounted = mountFake(CustomSelect, () => ({
    modelValue: value.value,
    options: ['Euler', 'DPM++ 2M', 'DDIM'],
    'onUpdate:modelValue': (v: string) => { value.value = v },
  }))
  const trigger = byTag(mounted.root, 'button')[0] as FakeNode & { getBoundingClientRect?: () => object }
  trigger.getBoundingClientRect = () => ({ left: 10, top: 100, bottom: 130, width: 200 })
  // 키보드 사용자가 Tab 으로 트리거에 온 상태에서 시작한다(목록은 aria-activedescendant 방식이라
  // 열려도 포커스는 트리거에 남는다)
  trigger.focus()
  const deps = {
    isGateOpen: () => false,
    generate: vi.fn(), saveSettings: vi.fn(), reloadHistory: vi.fn(), navigateTabs: vi.fn(),
    isParamsPanelOpen: () => opts.paramsPanelOpen ?? true,
    closeParamsPanel: vi.fn(),
    hasHistory: () => true,
    navigateHistory: vi.fn(),
    navigateHistoryEdge: vi.fn(),
    activeElement: () => ({ tagName: trigger.focused ? 'BUTTON' : 'BODY' }),
    jumpModifier: () => null,
    modalStack: opts.stack ?? createModalStack(),
  }
  const appKeydown = createAppKeydownHandler(deps)
  const root = mounted.root
  /** (트리거에 포커스가 있으면) 버튼의 @keydown → document 의 App keydown — 포커스가 없으면 키는 바로 document 로 */
  const press = async (key: string) => {
    const ev = keyEvent(key)
    if (trigger.focused) trigger.props.onKeydown(ev)
    appKeydown(ev)
    await Vue.nextTick()
    return ev
  }
  /** 마우스로 트리거 클릭 — 브라우저는 mousedown 에서 버튼에 포커스를 준다 */
  const clickTrigger = async () => {
    trigger.focus()
    trigger.props.onClick({ detail: 1 })
    await Vue.nextTick()
  }
  /** 마우스로 항목 클릭. 실제 브라우저는 옵션(포커스 불가) mousedown 에서 트리거 포커스를 풀기도 하지만,
   *  여기선 일부러 흉내 내지 않는다 — 포커스를 놓는 건 컴포넌트가 스스로 보장해야 한다. */
  const clickOption = async (label: string) => {
    const option = byClass(root, 'csel-option').find(n => n.textContent === label)
    expect(option, label).toBeTruthy()
    option!.props.onClick({ detail: 1 })
    await Vue.nextTick()
    await Vue.nextTick()
  }
  const isOpen = () => trigger.props['aria-expanded'] === true
  const activeOption = () => byClass(root, 'active').map(n => n.textContent)
  return { deps, press, clickTrigger, clickOption, isOpen, activeOption, trigger, value }
}

describe('CustomSelect keys do not leak into the global shortcuts', () => {
  it('↑/↓ open and move the dropdown without moving the history behind it', async () => {
    const { deps, press, isOpen, activeOption } = setup()
    await press('ArrowDown')
    expect(isOpen()).toBe(true)
    expect(activeOption()).toEqual(['Euler'])
    await press('ArrowDown')
    expect(activeOption()).toEqual(['DPM++ 2M'])
    await press('ArrowUp')
    expect(activeOption()).toEqual(['Euler'])
    await press('ArrowDown')
    await press('ArrowDown')
    expect(activeOption()).toEqual(['DDIM'])
    expect(deps.navigateHistory).not.toHaveBeenCalled()
    expect(deps.navigateHistoryEdge).not.toHaveBeenCalled()
  })

  it('ESC closes the open dropdown only — the params column stays; the next ESC closes the column', async () => {
    const { deps, press, isOpen } = setup({ paramsPanelOpen: true })
    await press('ArrowDown')
    expect(isOpen()).toBe(true)
    await press('Escape')
    expect(isOpen()).toBe(false)
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
    await press('Escape')                                   // 목록이 닫혀 있으면 CustomSelect 는 ESC 를 안 쓴다
    expect(deps.closeParamsPanel).toHaveBeenCalledTimes(1)
  })

  it('inside a closable modal (wildcard manager) ESC closes the dropdown, not the modal', async () => {
    const stack = createModalStack()
    const closeModal = vi.fn()
    stack.open({ close: closeModal })
    const { press, isOpen } = setup({ stack })
    await press('ArrowDown')
    await press('Escape')
    expect(isOpen()).toBe(false)
    expect(closeModal).not.toHaveBeenCalled()
    expect(stack.isAnyOpen()).toBe(true)
    await press('Escape')
    expect(closeModal).toHaveBeenCalledTimes(1)
  })
})

// S2-appvue-split#2-a — select() 가 마우스로 고른 뒤에도 트리거에 포커스를 되돌려, 샘플러를 마우스로 고르고
// ↓ 로 히스토리를 넘기면 목록만 다시 열리고 히스토리는 그대로였다.
describe('after the mouse finishes with the dropdown, ↑/↓ belong to the history again', () => {
  it('a mouse pick lets go of focus — the next ↓ moves the history, not the list', async () => {
    const { deps, press, clickTrigger, clickOption, isOpen, trigger, value } = setup()
    await clickTrigger()
    expect(isOpen()).toBe(true)
    await clickOption('DDIM')
    expect(value.value).toBe('DDIM')
    expect(isOpen()).toBe(false)
    expect(trigger.focused).toBe(false)
    await press('ArrowDown')
    expect(isOpen()).toBe(false)
    expect(deps.navigateHistory).toHaveBeenCalledTimes(1)
    expect(deps.navigateHistory).toHaveBeenLastCalledWith(1)
    await press('ArrowUp')
    expect(deps.navigateHistory).toHaveBeenLastCalledWith(-1)
  })

  it('closing the list with a second click on the trigger also lets go of focus', async () => {
    const { deps, press, clickTrigger, isOpen, trigger, value } = setup()
    await clickTrigger()
    await clickTrigger()
    expect(isOpen()).toBe(false)
    expect(trigger.focused).toBe(false)
    expect(value.value).toBe('Euler')
    await press('ArrowDown')
    expect(isOpen()).toBe(false)
    expect(deps.navigateHistory).toHaveBeenCalledTimes(1)
  })

  it('a list opened with the mouse still takes ↑/↓ while it is open', async () => {
    const { deps, press, clickTrigger, isOpen, activeOption } = setup()
    await clickTrigger()
    await press('ArrowDown')
    expect(isOpen()).toBe(true)
    expect(activeOption()).toEqual(['DPM++ 2M'])
    expect(deps.navigateHistory).not.toHaveBeenCalled()
  })

  it('a keyboard pick keeps focus on the trigger — ↓ reopens the list for the keyboard user', async () => {
    const { deps, press, isOpen, trigger, value } = setup()
    await press('ArrowDown')                                // 열기(현재 값 Euler 에서)
    await press('ArrowDown')                                // DPM++ 2M
    await press('Enter')
    expect(value.value).toBe('DPM++ 2M')
    expect(isOpen()).toBe(false)
    expect(trigger.focused).toBe(true)
    await press('ArrowDown')
    expect(isOpen()).toBe(true)
    expect(deps.navigateHistory).not.toHaveBeenCalled()
  })

  it('a click the keyboard synthesized (detail 0) closes without dropping focus', async () => {
    const { press, isOpen, trigger } = setup()
    await press('ArrowDown')
    expect(isOpen()).toBe(true)
    trigger.props.onClick({ detail: 0 })
    await Vue.nextTick()
    expect(isOpen()).toBe(false)
    expect(trigger.focused).toBe(true)
  })
})
