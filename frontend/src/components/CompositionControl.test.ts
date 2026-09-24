import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createRenderer, nextTick, ref, h, type App } from 'vue'
import * as Vue from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './CompositionControl.vue?raw'
import * as hostBridge from '../bridge.js'
import * as widgetStore from '../stores/widgetStore.js'
import * as compositionUtils from '../utils/compositionPrompt'
import { DEFAULT_COMPOSITION } from '../utils/compositionPrompt'
import { createAppKeydownHandler } from '../utils/appShortcuts'
import { createModalStack } from '../utils/modalStack'
import panelSource from './PromptPanel.vue?raw'

const host = vi.hoisted(() => ({ get: vi.fn(), action: vi.fn(), off: vi.fn(), event: vi.fn() }))
vi.mock('../bridge.js', () => ({
  getBackend: async () => ({ getUiPrefs: host.get }),
  onBackendEvent: (_name: string, callback: unknown) => { host.event(callback); return host.off },
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: host.action }))

// Compile the browser template for the host-neutral renderer; the normal Vite
// node transform produces SSR-only output, which cannot exercise event handlers.
const script = compileScript(parse(source).descriptor, { id: 'composition-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../bridge.js') return hostBridge
  if (name === '../stores/widgetStore.js') return widgetStore
  if (name === '../utils/compositionPrompt') return compositionUtils
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)
const CompositionControl = compiled.default

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  constructor(public tag: string, public text = '') {}
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
}
function insert(node: Node, parent: Node, anchor: Node | null = null) {
  if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1)
  node.parent = parent
  const index = anchor ? parent.children.indexOf(anchor) : -1
  parent.children.splice(index < 0 ? parent.children.length : index, 0, node)
}
const renderer = createRenderer<Node, Node>({
  createElement: tag => new Node(tag), createText: text => new Node('#text', text), createComment: () => new Node('#comment'),
  insert, remove: node => { if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1) },
  setText: (node, text) => { node.text = text }, setElementText: (node, text) => { node.text = text; node.children = [] },
  parentNode: node => node.parent, nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
  patchProp: (node, key, _old, value) => { node.props[key] = value }, setScopeId: () => {},
  insertStaticContent: (text, parent, anchor) => { const node = new Node('#static', text); insert(node, parent, anchor); return [node, node] },
})
function all(root: Node, predicate: (node: Node) => boolean): Node[] {
  return [...(predicate(root) ? [root] : []), ...root.children.flatMap(child => all(child, predicate))]
}
function button(root: Node, label: string) { return all(root, node => node.tag === 'button' && node.textContent.includes(label))[0]! }
function range(root: Node, key: string) { return all(root, node => node.tag === 'input' && node.props.id.endsWith(`-${key}`))[0]! }
function orbit(root: Node) { return all(root, node => node.props['aria-label'] === '카메라 구도 조작')[0]! }
let app: App | undefined
async function mount(initial = 'forest', otherPrompts = '') {
  const root = new Node('root')
  const prompt = ref(initial)
  const append = vi.fn((value: string) => { prompt.value = value })
  app = renderer.createApp({ setup: () => () => h(CompositionControl, { modelValue: prompt.value, otherPrompts, onAppend: append }) })
  app.mount(root)
  await nextTick()
  return { root, append, prompt }
}
beforeEach(() => {
  host.get.mockReset(); host.action.mockReset(); host.event.mockReset(); host.off.mockReset()
  host.get.mockImplementation(callback => callback('{}'))
})
afterEach(() => { app?.unmount(); app = undefined })

it('renders labeled native controls and never changes the prompt on preset or range changes', async () => {
  const { root, append, prompt } = await mount()
  expect(all(root, node => node.tag === 'input')).toHaveLength(5)
  expect(orbit(root).props.tabindex).toBe('0')
  expect(root.textContent).toContain('실제 3D 카메라 제어가 아니며')
  button(root, '낮은 시점').props.onClick()
  range(root, 'elevation').props.onInput({ target: { value: '60' } })
  await nextTick()
  expect(prompt.value).toBe('forest'); expect(append).not.toHaveBeenCalled()
  expect(root.textContent).toContain('from above')
  expect(host.action).toHaveBeenCalledWith('save_ui_prefs', expect.objectContaining({ compositionControl: expect.objectContaining({ elevation: -30 }) }))
})

it('appends explicitly through main tags and repeated clicks cannot duplicate tags', async () => {
  const { root, append, prompt } = await mount('forest', 'upper_body')
  button(root, '메인 태그에 추가').props.onClick()
  await nextTick()
  expect(prompt.value).toBe('forest, facing viewer, centered composition')
  expect(button(root, '이미 포함된').props.disabled).toBe(true)
  button(root, '이미 포함된').props.onClick()
  expect(append).toHaveBeenCalledTimes(1)
  expect(root.textContent).toContain('2개를 추가했습니다')
})

it('exposes conflict-aware append without deleting the previous composition', async () => {
  const { root, prompt } = await mount('from below, full body')
  button(root, '위에서').props.onClick()
  await nextTick()
  expect(root.textContent).toContain('기존 태그는 삭제하지 않습니다')
  button(root, '기존 구도 유지하고 추가').props.onClick()
  await nextTick()
  expect(prompt.value.startsWith('from below, full body, ')).toBe(true)
})

it('restores preferences once but does not overwrite a user change with a late response', async () => {
  host.get.mockImplementation(() => {})
  const { root } = await mount()
  host.event.mock.calls[0]![0](JSON.stringify({ compositionControl: { ...DEFAULT_COMPOSITION, azimuth: -90 } }))
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(-90)
  range(root, 'azimuth').props.onInput({ target: { value: '45' } })
  host.get.mock.calls[0]![0](JSON.stringify({ compositionControl: { ...DEFAULT_COMPOSITION, azimuth: 90 } }))
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(45)
  app!.unmount(); app = undefined
  expect(host.off).toHaveBeenCalledOnce()
})

it('handles keyboard, touch cancellation and repeated drags without leftover pointer state', async () => {
  const { root, append } = await mount()
  const area = orbit(root)
  const preventDefault = vi.fn()
  area.props.onKeydown({ key: 'ArrowRight', shiftKey: true, preventDefault })
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(15)
  expect(preventDefault).toHaveBeenCalledOnce()
  const target = { focus: vi.fn(), setPointerCapture: vi.fn(), hasPointerCapture: () => true, releasePointerCapture: vi.fn() }
  const pointer = { pointerId: 7, isPrimary: true, button: 0, clientX: 0, clientY: 0, currentTarget: target }
  area.props.onPointerdown(pointer)
  area.props.onPointermove({ ...pointer, clientX: 100 })
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(85)
  area.props.onPointercancel(pointer)
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(15)
  area.props.onPointermove({ ...pointer, clientX: 300 })
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(15)
  area.props.onPointerdown(pointer)
  area.props.onPointerup({ ...pointer, clientX: 10 })
  area.props.onLostpointercapture(pointer)
  await nextTick()
  expect(range(root, 'azimuth').props.value).toBe(22)
  expect(target.releasePointerCapture).toHaveBeenCalledWith(7)
  expect(append).not.toHaveBeenCalled()
})

// 궤도는 PromptPanel 안의 포커스 가능한 div 라 입력 칸이 아니다 — 예전엔 ↑/↓ 가 고도를 바꾸면서 버블링된
// 같은 키로 App 의 전역 단축키가 생성 히스토리까지 넘겼다(utils/appShortcuts 는 이제 쓴 키를 비켜 간다).
it('orbit ↑/↓ change the elevation only — the same key does not also move the history', async () => {
  const { root } = await mount()
  const navigateHistory = vi.fn()
  const appKeydown = createAppKeydownHandler({
    isGateOpen: () => false, generate: vi.fn(), saveSettings: vi.fn(), reloadHistory: vi.fn(), navigateTabs: vi.fn(),
    isParamsPanelOpen: () => false, closeParamsPanel: vi.fn(), hasHistory: () => true,
    navigateHistory, navigateHistoryEdge: vi.fn(),
    activeElement: () => ({ tagName: 'DIV', isContentEditable: false }), jumpModifier: () => null,
    modalStack: createModalStack(),
  })
  const press = async (key: string) => {
    const ev = { key, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, defaultPrevented: false, preventDefault() { ev.defaultPrevented = true } }
    orbit(root).props.onKeydown(ev)      // 포커스된 궤도가 먼저, 그다음 document 의 App keydown
    appKeydown(ev)
    await nextTick()
  }
  await press('ArrowUp')
  expect(range(root, 'elevation').props.value).toBe(5)
  await press('ArrowDown')
  await press('ArrowDown')
  expect(range(root, 'elevation').props.value).toBe(-5)
  expect(navigateHistory).not.toHaveBeenCalled()
})

it('connects the panel to the same main_prompt_text proxy used by text and block inputs', () => {
  expect(panelSource).toContain('<CompositionControl :model-value="widgets.main_prompt_text')
  expect(panelSource).toContain('@append="appendComposition"')
  expect(panelSource).toMatch(/function appendComposition\(text: string\)[\s\S]*?widgets\.main_prompt_text = text/)
})
