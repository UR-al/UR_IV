import { afterEach, expect, it } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './ExtraSamplersPanel.vue?raw'
import * as xsa from '../../utils/extraSamplers'
import * as family from '../../utils/generationFamily'

const caps = Vue.ref<any>(null)
const widgets = Vue.reactive<Record<string, any>>({})
const compiled = compileScript(parse(source).descriptor, { id: 'xsamp-panel-test', inlineTemplate: true })
const code = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
// ToggleSwitch 대역 — 토글을 onFlip 으로 흉내 낸다
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
  if (id === '../ToggleSwitch.vue') return { __esModule: true, default: ToggleStub }
  if (id === '../../stores/widgetStore.js') return { useWidgetStore: () => ({ widgets }) }
  if (id === '../../composables/useSamExtraCapabilities') return { useSamExtraCapabilities: () => ({ capabilities: caps }) }
  if (id === '../../utils/generationFamily') return family
  if (id === '../../utils/extraSamplers') return xsa
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
  // v-if 로 오가는 뿌리(조각) — 자리 표시(주석)의 다음 형제를 알아야 같은 자리에 다시 넣는다
  parentNode: n => n.parent ?? null,
  nextSibling: n => (n.parent ? n.parent.children[n.parent.children.indexOf(n) + 1] ?? null : null),
  patchProp: (n, key, _old, value) => { n.props[key] = value }, setScopeId: () => {},
  insertStaticContent(text, parent) { const child = node('#static', text); child.parent = parent; parent.children.push(child); return [child, child] },
})
let app: Vue.App | undefined
const all = (root: Node, tag: string): Node[] => [...(root.tag === tag ? [root] : []), ...root.children.flatMap(c => all(c, tag))]
const text = (root: Node): string => (root.tag === '#comment' ? '' : root.text) + root.children.map(text).join('')
const inputs = (root: Node) => all(root, 'input')
const labels = (root: Node) => all(root, 'label').map(text)
const stages = (root: Node) => all(root, 'button').filter(b => b.props.role === 'radio')
const toggle = (root: Node) => all(root, 'toggle')[0]!
const resets = (root: Node) => all(root, 'button').filter(b => b.props.class === 'xsa-reset')
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

it('hidden for samplers that read no values, and writes nothing', async () => {
  widgets.sampler_combo = 'Euler a'
  const root = await mount()
  expect(all(root, 'section')).toEqual([])
  expect(text(root)).toBe('')
  expect(Object.keys(widgets)).toEqual(['sampler_combo'])
})

it('Reverse-time shows stage, η and the noise window row with the extension ranges', async () => {
  widgets.sampler_combo = 'ER SDE (Reverse-time)'
  const root = await mount()
  expect(text(root)).toContain('ER SDE 설정 (sam-extra Extra Samplers)')
  expect(text(root)).toContain('사용: 메인')
  expect(labels(root)).toEqual(['풀이 차수 (max stage)', 'η (eta)', '잡음 구간', '시작', '끝'])
  const [eta, start, end] = inputs(root)
  expect([eta!.props.min, eta!.props.max, eta!.props.step, eta!.props.disabled]).toEqual([0, 10, 0.01, false])
  for (const input of [start!, end!]) {
    expect([input.props.min, input.props.max, input.props.step, input.props.disabled]).toEqual([0, 1, 0.01, false])
  }
  expect(stages(root).map(b => [text(b), b.props['aria-checked']])).toEqual([['1', false], ['2', false], ['3', true]])
  expect(text(root)).toContain(xsa.ETA_HELP)
  expect(text(root)).toContain(xsa.WINDOW_HELP)
  expect(text(root)).toContain(xsa.FORGE_PARAMS_NOTE)
  expect(text(root)).not.toContain(xsa.ODE_NOTE)
  stages(root)[1]!.props.onClick()
  await Vue.nextTick()
  expect(widgets[xsa.WIDGET_IDS.max_stage]).toBe('2')
  expect(stages(root).map(b => b.props['aria-checked'])).toEqual([false, true, false])
})

it('ODE only turns η and the window row off; a Tunable pass turns them on again', async () => {
  widgets.sampler_combo = 'er_sde_ode'
  const root = await mount()
  expect(inputs(root).map(i => i.props.disabled)).toEqual([true, true, true])
  expect(text(root)).toContain(xsa.ODE_NOTE)
  expect(text(root)).not.toContain(xsa.ETA_HELP)
  toggle(root).props.onFlip()
  await Vue.nextTick()
  expect(widgets[xsa.WIDGET_IDS.noise_window]).toBeUndefined()
  widgets.hires_options_group = 'true'
  widgets.hires_sampler_combo = 'ER SDE (Tunable)'
  await Vue.nextTick()
  expect(text(root)).toContain('사용: 메인 · Hires')
  expect(inputs(root).map(i => i.props.disabled)).toEqual([false, false, false])
  expect(text(root)).toContain(xsa.TUNABLE_HINT)
  toggle(root).props.onFlip()
  await Vue.nextTick()
  expect(widgets[xsa.WIDGET_IDS.noise_window]).toBe('1')
  expect(toggle(root).props.value).toBe(true)
})

it('warn line when the window is on and start ≥ end, not when it is off', async () => {
  widgets.sampler_combo = 'ER SDE (Tunable)'
  widgets[xsa.WIDGET_IDS.noise_start] = 0.8
  widgets[xsa.WIDGET_IDS.noise_end] = '0.5'
  const root = await mount()
  expect(text(root)).not.toContain(xsa.WINDOW_EMPTY_WARNING)
  widgets[xsa.WIDGET_IDS.noise_window] = '1'
  await Vue.nextTick()
  expect(text(root)).toContain(xsa.WINDOW_EMPTY_WARNING)
  widgets[xsa.WIDGET_IDS.noise_end] = 0.9
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.WINDOW_EMPTY_WARNING)
  widgets[xsa.WIDGET_IDS.noise_end] = 0.8   // start = end is empty too
  await Vue.nextTick()
  expect(text(root)).toContain(xsa.WINDOW_EMPTY_WARNING)
})

it('X10 note: a 2-argument sam-extra drops the window when it is on', async () => {
  widgets.sampler_combo = 'ER SDE (Reverse-time)'
  widgets[xsa.WIDGET_IDS.noise_window] = '1'
  const snapshot = (argc: number | null) => ({ status: 'ok', known: true, features: { extra_samplers: true },
    scripts: { [xsa.SCRIPT_TITLE]: { present: true, img2img: true, live_argc: argc } } })
  caps.value = snapshot(2)
  const root = await mount()
  expect(text(root)).toContain(xsa.WINDOW_UNSUPPORTED_TEXT)
  caps.value = snapshot(5)
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.WINDOW_UNSUPPORTED_TEXT)
  caps.value = snapshot(2)
  widgets[xsa.WIDGET_IDS.noise_window] = '0'
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.WINDOW_UNSUPPORTED_TEXT)
  widgets[xsa.WIDGET_IDS.noise_window] = '1'
  widgets.sampler_combo = 'ER SDE (ODE)'   // the window is not read by ODE
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.WINDOW_UNSUPPORTED_TEXT)
})

it('reset writes the extension defaults and is disabled at defaults', async () => {
  widgets.sampler_combo = 'ER SDE (Reverse-time)'
  widgets[xsa.WIDGET_IDS.max_stage] = '1'
  widgets[xsa.WIDGET_IDS.eta] = 0.5
  widgets[xsa.WIDGET_IDS.noise_window] = '1'
  const root = await mount()
  const [reset] = resets(root)
  expect(reset!.props.disabled).toBe(false)
  reset!.props.onClick()
  await Vue.nextTick()
  for (const [key, value] of Object.entries(xsa.DEFAULTS)) {
    expect(widgets[xsa.WIDGET_IDS[key as xsa.XsampKey]]).toBe(value)
  }
  expect(resets(root)[0]!.props.disabled).toBe(true)
})

it('non-default values stay visible as one line when no pass uses ER SDE', async () => {
  widgets.sampler_combo = 'Euler'
  widgets[xsa.WIDGET_IDS.eta] = '0.5'
  const root = await mount()
  expect(all(root, 'section')).toEqual([])
  expect(text(root)).toContain('ER SDE 값(차수 3 · η 0.5)은 ER SDE (Reverse-time)·(ODE)·(Tunable) 을 고를 때만 쓰입니다.')
  resets(root)[0]!.props.onClick()
  await Vue.nextTick()
  expect(widgets[xsa.WIDGET_IDS.eta]).toBe('1')
  expect(text(root)).toBe('')
})

it('ADetailer and SAM3 passes show the box too', async () => {
  widgets.sampler_combo = 'Euler'
  widgets.sam3_group = 'true'
  widgets._sam3_mode = 'Inpaint'
  widgets._sam3_use_sampler = 'true'
  widgets._sam3_sampler = 'er_sde_reverse_time'
  const root = await mount()
  expect(text(root)).toContain('사용: SAM3')
  widgets._sam3_mode = 'Mask only'
  widgets.adetailer_group = 'true'
  widgets.ad_slot2_group = 'true'
  widgets._ad_s2_use_sampler = 'true'
  widgets._ad_s2_sampler = 'ER SDE (ODE)'
  await Vue.nextTick()
  expect(text(root)).toContain('사용: ADetailer 2')
})

it('sampler hints show without the box; CFG++ warns above CFG 2', async () => {
  widgets.sampler_combo = 'CFG++ UD10 AB'
  widgets.cfg_input = 2.5
  const root = await mount()
  expect(all(root, 'section')).toEqual([])
  expect(text(root)).toContain(xsa.UD10_HINT)
  expect(text(root)).toContain(xsa.cfgWarning(2.5))
  widgets.cfg_input = 2
  widgets.hires_options_group = 'true'
  widgets.hires_sampler_combo = 'Restart (flow)'
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.cfgWarning(2.5))
  expect(text(root)).toContain(xsa.RESTART_HINT)
})

it('status line: old Forge, Krea2 replace the Forge parameters note; ComfyUI gets its own note (no warning)', async () => {
  widgets.sampler_combo = 'ER SDE (Reverse-time)'
  caps.value = { status: 'ok', known: true, features: { extra_samplers: false } }
  const root = await mount()
  expect(text(root)).toContain(xsa.OLD_FORGE_NOTE)
  expect(text(root)).not.toContain(xsa.FORGE_PARAMS_NOTE)
  caps.value = { status: 'ok', known: true, features: { extra_samplers: true } }
  await Vue.nextTick()
  expect(text(root)).not.toContain(xsa.OLD_FORGE_NOTE)
  expect(text(root)).toContain(xsa.FORGE_PARAMS_NOTE)
  caps.value = { status: 'not_applicable', known: false, features: {} }
  await Vue.nextTick()
  expect(text(root)).toContain(xsa.COMFY_PARAMS_NOTE)                         // 계획 C6
  expect(text(root)).not.toContain(xsa.FORGE_PARAMS_NOTE)
  expect(all(root, 'p').filter(p => String(p.props.class ?? '').includes('xsa-warnline'))).toEqual([])
  widgets.generation_family_combo = 'Krea2'
  await Vue.nextTick()
  expect(text(root)).toContain(xsa.KREA2_NOTE)
})
