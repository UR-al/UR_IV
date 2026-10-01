import { afterEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './VaeDegridCard.vue?raw'
import * as degrid from '../../utils/vaeDegrid'
import * as overridesTable from '../../utils/forgeOptionOverrides'
import * as anima38 from '../../utils/anima38Card'
import * as family from '../../utils/generationFamily'
import { widgetFlag } from '../../composables/widgetFlag'

const refresh = vi.fn(), readPrefs = vi.fn()
const caps = Vue.ref<any>(null)
const widgets = Vue.reactive<Record<string, any>>({})
const properties = Vue.reactive<Record<string, Record<string, unknown>>>({})
const compiled = compileScript(parse(source).descriptor, { id: 'degrid-card-test', inlineTemplate: true })
const code = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
// CustomSelect·ToggleSwitch 대역 — 고른 값을 onPick 으로 흉내 낸다
const SelectStub = Vue.defineComponent({
  props: { modelValue: { type: [String, Number], default: '' }, options: { type: Array, default: () => [] } },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    return () => Vue.h('csel', { value: props.modelValue, options: props.options, onPick: (label: string) => emit('update:modelValue', label) })
  },
})
const ToggleStub = Vue.defineComponent({
  props: { modelValue: { type: Boolean, default: false } },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    return () => Vue.h('toggle', { value: props.modelValue, onFlip: () => emit('update:modelValue', !props.modelValue) })
  },
})
const IconStub = Vue.defineComponent({ props: { name: String }, setup: props => () => Vue.h('icon', { name: props.name }) })
const output: { default?: any } = {}
new Function('require', 'exports', code)((id: string) => {
  if (id === 'vue') return Vue
  if (id === '../CustomSelect.vue') return { __esModule: true, default: SelectStub }
  if (id === '../ToggleSwitch.vue') return { __esModule: true, default: ToggleStub }
  if (id === '../../bridge.js') return { getBackend: async () => ({ getUiPrefs: readPrefs }) }
  if (id === '../../stores/widgetStore.js') {
    return { useWidgetStore: () => ({ widgets, getProperty: (wid: string, prop: string, def: unknown = '') => properties[wid]?.[prop] ?? def }) }
  }
  if (id === '../../composables/widgetFlag') return { widgetFlag }
  if (id === '../../composables/useSamExtraCapabilities') return { useSamExtraCapabilities: () => ({ capabilities: caps, refresh }) }
  if (id === '../../utils/generationFamily') return family
  if (id === '../../utils/anima38Card') return anima38
  if (id === '../../utils/forgeOptionOverrides') return overridesTable
  if (id === '../../utils/vaeDegrid') return degrid
  throw Error(id)
}, output)

// 숫자 칸의 v-model(vModelText)은 요소에 리스너를 달고 갱신 때 document.activeElement 를 본다 — 최소 대역
class Node {
  children: Node[] = []; parent?: Node; props: Record<string, any> = {}; value: unknown = ''
  constructor(public tag: string, public text = '') {}
  addEventListener() {}
  removeEventListener() {}
  getRootNode() { return { activeElement: null } }
}
const g = globalThis as { document?: unknown; Document?: unknown; ShadowRoot?: unknown }
if (!g.document) g.document = { activeElement: null }
if (!g.Document) g.Document = class {}
if (!g.ShadowRoot) g.ShadowRoot = class {}
const node = (tag: string, text = ''): Node => new Node(tag, text)
const renderer = Vue.createRenderer<Node, Node>({
  createElement: tag => node(tag), createText: text => node('#text', text), createComment: text => node('#comment', text),
  insert(child, parent, anchor) { child.parent = parent; const i = anchor ? parent.children.indexOf(anchor) : -1; parent.children.splice(i < 0 ? parent.children.length : i, 0, child) },
  remove(child) { const parent = child.parent; if (parent) parent.children.splice(parent.children.indexOf(child), 1) },
  setText: (n, text) => { n.text = text }, setElementText: (n, text) => { n.text = text; n.children = [] },
  parentNode: n => n.parent ?? null, nextSibling: () => null,
  patchProp: (n, key, _old, value) => { n.props[key] = value }, setScopeId: () => {},
  insertStaticContent(text, parent) { const child = node('#static', text); child.parent = parent; parent.children.push(child); return [child, child] },
})
let app: Vue.App | undefined
const all = (root: Node, tag: string): Node[] => [...(root.tag === tag ? [root] : []), ...root.children.flatMap(c => all(c, tag))]
const text = (root: Node): string => root.text + root.children.map(text).join('')
async function mount() {
  const root = node('root')
  app = renderer.createApp(output.default)
  app.component('Icon', IconStub)
  app.mount(root)
  await Vue.nextTick()
  return root
}
afterEach(() => {
  app?.unmount(); app = undefined; caps.value = null
  for (const key of Object.keys(widgets)) delete widgets[key]
  for (const key of Object.keys(properties)) delete properties[key]
  refresh.mockReset(); readPrefs.mockReset()
})

const V11 = 'qwenVAEDegridNafnet_v11'
const selects = (root: Node) => all(root, 'csel')   // [모델, 모드]

it('default card is off with an empty summary and writes nothing', async () => {
  const root = await mount()
  expect(Object.keys(widgets)).toEqual([])
  expect(text(root)).toContain('꺼짐')
  expect(selects(root)[0]!.props.value).toBe(degrid.AUTO_LABEL)
  expect(selects(root)[1]!.props.value).toBe('Full (전체)')
})

it('model select stores the Forge name (auto = ""), mode stores the key, refresh asks the snapshot', async () => {
  caps.value = { status: 'ok', known: true, features: { degrid: true }, choices: { degrid_models: [V11, 'None'] } }
  widgets[degrid.WIDGET_IDS.enabled] = 'true'
  const root = await mount()
  const [model, mode] = selects(root)
  expect(model!.props.options).toEqual([`자동 (가장 높은 버전 — 지금: ${V11})`, V11])
  model!.props.onPick(V11)
  mode!.props.onPick('Dark Pixels Mainly (어두운 점 위주)')
  await Vue.nextTick()
  expect(widgets[degrid.WIDGET_IDS.model]).toBe(V11)
  expect(widgets[degrid.WIDGET_IDS.mode]).toBe('dark')
  expect(text(root)).toContain(`Dark 1 · 512 · ${V11}`)
  selects(root)[0]!.props.onPick(`자동 (가장 높은 버전 — 지금: ${V11})`)
  await Vue.nextTick()
  expect(widgets[degrid.WIDGET_IDS.model]).toBe('')
  all(root, 'button').find(b => b.props['aria-label'] === '모델 목록 다시 확인')!.props.onClick()
  expect(refresh).toHaveBeenCalledOnce()
})

it('ComfyUI list comes from the comfyModels property; missing pack is an alert line', async () => {
  caps.value = { status: 'not_applicable', known: false }
  widgets[degrid.WIDGET_IDS.enabled] = 'true'
  const root = await mount()
  expect(text(root)).toContain('1.5.0')
  properties[degrid.WIDGET_IDS.model] = { [degrid.COMFY_MODELS_PROPERTY]: [V11] }
  await Vue.nextTick()
  expect(selects(root)[0]!.props.options).toEqual([`자동 (가장 높은 버전 — 지금: ${V11})`, V11])
  expect(text(root)).toContain('ComfyUI 노드로 적용')
  expect(text(root)).toContain('ComfyUI 는 Forge 설정을 쓰지 않습니다')
})

it('refresh button names the active backend and sends the same refresh action on both', async () => {
  caps.value = { status: 'ok', known: true, features: { degrid: true }, choices: { degrid_models: [V11] } }
  const root = await mount()
  const button = () => all(root, 'button').find(b => b.props['aria-label'] === '모델 목록 다시 확인')!
  expect(button().props.title).toContain('Forge')
  expect(button().props.title).not.toContain('ComfyUI')
  caps.value = { status: 'not_applicable', known: false }   // ComfyUI 로 전환 — 파이썬이 object_info 를 다시 받는다
  await Vue.nextTick()
  expect(button().props.title).toContain('ComfyUI')
  expect(button().props.title).not.toContain('Forge')
  button().props.onClick()
  expect(refresh).toHaveBeenCalledOnce()
})

it('opening the device/memory section re-reads the P10 overrides (read-only)', async () => {
  caps.value = { status: 'ok', known: true, features: { degrid: true }, options_known: true,
    options: { sam3_degrid_device: 'auto', sam3_degrid_gpu_precision: 'fp32', sam3_degrid_keep_loaded: false } }
  readPrefs.mockImplementation((cb: (raw: string) => void) => cb(JSON.stringify({ forgeOptionOverrides: { sam3_degrid_device: 'cpu' } })))
  const root = await mount()
  expect(text(root)).toContain('Forge 설정: GPU (Forge 장치)')
  const sub = all(root, 'details').find(d => d.props.class === 'ext-sub')!
  sub.props.onToggle({ target: { open: true } })
  await Vue.nextTick(); await Vue.nextTick(); await Vue.nextTick()
  expect(readPrefs).toHaveBeenCalledOnce()
  expect(text(root)).toContain('앱 설정: CPU (VRAM 안 씀, 느림)')
  sub.props.onToggle({ target: { open: false } })
  await Vue.nextTick()
  expect(readPrefs).toHaveBeenCalledOnce()
})

it('reset writes the app defaults', async () => {
  widgets[degrid.WIDGET_IDS.enabled] = 'true'
  widgets[degrid.WIDGET_IDS.tile] = '0'
  const root = await mount()
  all(root, 'button').find(b => b.props.class === 'degrid-reset')!.props.onClick()
  await Vue.nextTick()
  for (const [key, value] of Object.entries(degrid.DEFAULTS)) {
    expect(widgets[degrid.WIDGET_IDS[key as keyof typeof degrid.WIDGET_IDS]]).toBe(value)
  }
})
