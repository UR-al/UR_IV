import { afterEach, expect, it } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './AnimaSpeedCard.vue?raw'
import * as speed from '../../utils/animaSpeed'
import * as family from '../../utils/generationFamily'
import { widgetFlag } from '../../composables/widgetFlag'

const caps = Vue.ref<any>(null)
const widgets = Vue.reactive<Record<string, any>>({})
const compiled = compileScript(parse(source).descriptor, { id: 'speed-card-test', inlineTemplate: true })
const code = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
// CustomSelect·ToggleSwitch 대역 — 고른 값을 onPick, 토글을 onFlip 으로 흉내 낸다
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
const output: { default?: any } = {}
new Function('require', 'exports', code)((id: string) => {
  if (id === 'vue') return Vue
  if (id === '../CustomSelect.vue') return { __esModule: true, default: SelectStub }
  if (id === '../ToggleSwitch.vue') return { __esModule: true, default: ToggleStub }
  if (id === '../../stores/widgetStore.js') return { useWidgetStore: () => ({ widgets }) }
  if (id === '../../composables/widgetFlag') return { widgetFlag }
  if (id === '../../composables/useSamExtraCapabilities') return { useSamExtraCapabilities: () => ({ capabilities: caps }) }
  if (id === '../../utils/generationFamily') return family
  if (id === '../../utils/animaSpeed') return speed
  throw Error(id)
}, output)

// 텍스트·숫자 칸의 v-model(vModelText)은 요소에 리스너를 달고 갱신 때 document.activeElement 를 본다 — 최소 대역
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
  // v-if 로 나타나는 칸이 제자리에 들어가게 형제를 돌려준다(템플릿 순서 = 트리 순서)
  parentNode: n => n.parent ?? null, nextSibling: n => n.parent?.children[n.parent.children.indexOf(n) + 1] ?? null,
  patchProp: (n, key, _old, value) => { n.props[key] = value }, setScopeId: () => {},
  insertStaticContent(text, parent) { const child = node('#static', text); child.parent = parent; parent.children.push(child); return [child, child] },
})
let app: Vue.App | undefined
const all = (root: Node, tag: string): Node[] => [...(root.tag === tag ? [root] : []), ...root.children.flatMap(c => all(c, tag))]
const text = (root: Node): string => root.text + root.children.map(text).join('')
async function mount() {
  const root = node('root')
  app = renderer.createApp(output.default)
  app.mount(root)
  await Vue.nextTick()
  return root
}
afterEach(() => {
  app?.unmount(); app = undefined; caps.value = null
  for (const key of Object.keys(widgets)) delete widgets[key]
})

const selects = (root: Node) => all(root, 'csel')
const toggles = (root: Node) => all(root, 'toggle')
const inputs = (root: Node) => all(root, 'input')
const label = (options: readonly speed.SpeedOption[], key: string) => options.find(o => o.key === key)!.label

it('default card is off with an empty badge, shows the neo_shift fields and writes nothing', async () => {
  const root = await mount()
  expect(Object.keys(widgets)).toEqual([])
  expect(all(root, 'span').find(s => s.props.class === 'speed-summary')).toBeUndefined()
  const [mode, threshold, preset, transform] = selects(root)
  expect(mode!.props.value).toBe(label(speed.MODE_OPTIONS, 'transition'))
  expect(threshold!.props.value).toBe(label(speed.THRESHOLD_OPTIONS, 'neo_shift'))
  expect([preset!.props.value, preset!.props.options]).toEqual(['anima', [...speed.PRESET_NAMES]])
  expect(transform!.props.value).toBe(label(speed.TRANSFORM_OPTIONS, 'dct'))
  expect(inputs(root)).toHaveLength(4)                                    // 해상도 비율·δ·σ 나눔값·시드
  expect(toggles(root).map(t => t.props.value)).toEqual([false, true, false, false])   // 사용·적응형(기본 켬)·Hires·I2I
  expect(text(root)).toContain(speed.LEAD_NOTE)
  expect(text(root)).toContain(speed.FORGE_NOTE)
  expect(text(root)).not.toContain(speed.INVALID_NOTE)
})

it('selects store keys; manual hides preset, δ, divisor and adaptive; custom shows A and β', async () => {
  const root = await mount()
  selects(root)[1]!.props.onPick(label(speed.THRESHOLD_OPTIONS, 'manual'))
  selects(root)[0]!.props.onPick(label(speed.MODE_OPTIONS, 'respace'))
  await Vue.nextTick()
  expect([widgets[speed.WIDGET_IDS.threshold], widgets[speed.WIDGET_IDS.mode]]).toEqual(['manual', 'respace'])
  expect(selects(root)).toHaveLength(3)                                   // 프리셋 없음
  expect(inputs(root)).toHaveLength(3)                                    // 해상도 비율·수동 σ·시드
  expect(toggles(root)).toHaveLength(3)                                   // 적응형 없음
  widgets[speed.WIDGET_IDS.threshold] = 'delta_optimal'
  await Vue.nextTick()
  selects(root)[2]!.props.onPick('custom')
  await Vue.nextTick()
  expect(widgets[speed.WIDGET_IDS.preset]).toBe('custom')
  expect(inputs(root)).toHaveLength(5)                                    // 비율·δ·A·β·시드(나눔값은 neo_shift 만)
  selects(root)[3]!.props.onPick(label(speed.TRANSFORM_OPTIONS, 'fft'))
  await Vue.nextTick()
  expect(widgets[speed.WIDGET_IDS.transform]).toBe('fft')
})

it('badge and validation lines follow utils/animaSpeed', async () => {
  widgets[speed.WIDGET_IDS.enabled] = 'true'
  const root = await mount()
  expect(text(root)).toContain('transition · neo_shift · 0.5→1')
  widgets[speed.WIDGET_IDS.scales] = '0.5'
  widgets[speed.WIDGET_IDS.delta] = '0'
  await Vue.nextTick()
  expect(text(root)).toContain('⚠ 설정 오류')
  expect(text(root)).toContain(speed.VALIDATION_MESSAGES.scales)
  expect(text(root)).toContain(speed.VALIDATION_MESSAGES.delta)
  expect(text(root)).toContain(speed.INVALID_NOTE)
  expect(all(root, 'div').find(d => d.props.role === 'alert')).toBeDefined()
})

it('toggles write true/false strings', async () => {
  const root = await mount()
  const [enabled, adaptive] = toggles(root)
  enabled!.props.onFlip()
  adaptive!.props.onFlip()
  await Vue.nextTick()
  expect(widgets[speed.WIDGET_IDS.enabled]).toBe('true')
  expect(widgets[speed.WIDGET_IDS.adaptive]).toBe('false')
  toggles(root)[3]!.props.onFlip()
  await Vue.nextTick()
  expect(widgets[speed.WIDGET_IDS.apply_img2img]).toBe('true')
})

it('backend line: Forge, old Forge, ComfyUI pending, Krea2', async () => {
  caps.value = { status: 'ok', known: true, features: { speed: true } }
  const root = await mount()
  expect(text(root)).toContain(speed.FORGE_NOTE)
  caps.value = { status: 'ok', known: true, features: { speed: false } }
  await Vue.nextTick()
  expect(text(root)).toContain(speed.OLD_FORGE_NOTE)
  caps.value = { status: 'not_applicable', known: false }
  await Vue.nextTick()
  expect(text(root)).toContain(speed.COMFY_PENDING_NOTE)
  widgets.generation_family_combo = 'Krea2'
  await Vue.nextTick()
  expect(text(root)).toContain(speed.KREA2_NOTE)
})

it('reset writes the app defaults', async () => {
  widgets[speed.WIDGET_IDS.enabled] = 'true'
  widgets[speed.WIDGET_IDS.seed] = '7'
  const root = await mount()
  all(root, 'button').find(b => String(b.props.class).includes('speed-reset'))!.props.onClick()
  await Vue.nextTick()
  for (const [key, value] of Object.entries(speed.DEFAULTS)) {
    expect(widgets[speed.WIDGET_IDS[key as speed.SpeedKey]]).toBe(value)
  }
})
