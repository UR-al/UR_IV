import { afterEach, expect, it } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './ExtraSchedulerFields.vue?raw'
import * as xs from '../../utils/extraSchedulers'
import * as family from '../../utils/generationFamily'

const caps = Vue.ref<any>(null)
const widgets = Vue.reactive<Record<string, any>>({})
const properties = Vue.reactive<Record<string, Record<string, unknown>>>({})
const compiled = compileScript(parse(source).descriptor, { id: 'xsched-fields-test', inlineTemplate: true })
const code = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
const output: { default?: any } = {}
new Function('require', 'exports', code)((id: string) => {
  if (id === 'vue') return Vue
  if (id === '../../stores/widgetStore.js') {
    return { useWidgetStore: () => ({ widgets, getProperty: (wid: string, prop: string, def: unknown = '') => properties[wid]?.[prop] ?? def }) }
  }
  if (id === '../../composables/useSamExtraCapabilities') return { useSamExtraCapabilities: () => ({ capabilities: caps }) }
  if (id === '../../utils/generationFamily') return family
  if (id === '../../utils/extraSchedulers') return xs
  throw Error(id)
}, output)

// 글자·숫자 칸의 v-model(vModelText)은 요소에 리스너를 달고 갱신 때 document.activeElement 를 본다 — 최소 대역
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
  parentNode: n => n.parent ?? null,
  // v-if 로 오가는 두 뿌리(조각) — 자리 표시(주석)의 다음 형제를 알아야 같은 자리에 다시 넣는다
  nextSibling: n => (n.parent ? n.parent.children[n.parent.children.indexOf(n) + 1] ?? null : null),
  patchProp: (n, key, _old, value) => { n.props[key] = value }, setScopeId: () => {},
  insertStaticContent(text, parent) { const child = node('#static', text); child.parent = parent; parent.children.push(child); return [child, child] },
})
let app: Vue.App | undefined
const all = (root: Node, tag: string): Node[] => [...(root.tag === tag ? [root] : []), ...root.children.flatMap(c => all(c, tag))]
const text = (root: Node): string => (root.tag === '#comment' ? '' : root.text) + root.children.map(text).join('')
const inputs = (root: Node) => all(root, 'input')
const labels = (root: Node) => all(root, 'label').map(text)
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
  for (const key of Object.keys(properties)) delete properties[key]
})

it('hidden for schedulers that read no values, and writes nothing', async () => {
  widgets.scheduler_combo = 'simple'
  const root = await mount()
  expect(all(root, 'section')).toEqual([])
  expect(text(root)).toBe('')
  expect(Object.keys(widgets)).toEqual(['scheduler_combo'])
})

it('Laplace shows only the μ/β row with the extension ranges; the subtitle names the passes', async () => {
  widgets.scheduler_combo = 'Laplace'
  widgets.hires_options_group = 'true'
  widgets.hires_scheduler_combo = 'Use same scheduler'
  const root = await mount()
  expect(text(root)).toContain('사용: 메인 · Hires')
  expect(labels(root)).toEqual(['Laplace μ', 'Laplace β'])
  const [mu, beta] = inputs(root)
  expect([mu!.props.min, mu!.props.max, mu!.props.step]).toEqual([-10, 10, 0.1])
  expect([beta!.props.min, beta!.props.max, beta!.props.step]).toEqual([0, 10, 0.1])
  expect(text(root)).not.toContain('custom 식')
})

it('custom shows the mode switch; the mode swaps the expression and sigma list inputs', async () => {
  widgets.scheduler_combo = 'karras'
  widgets.sam3_group = 'true'
  widgets._sam3_mode = 'Inpaint'
  widgets._sam3_use_scheduler = 'true'
  widgets._sam3_scheduler = 'custom'
  const root = await mount()
  expect(text(root)).toContain('사용: SAM3')
  expect(labels(root)).toEqual(['custom 방식', 'custom 식'])
  expect(inputs(root)[0]!.props.placeholder).toBe('m + (M - m) * (1 - x) ** 3')
  const [expression, list] = all(root, 'button').filter(b => b.props.role === 'radio')
  expect([expression!.props['aria-checked'], list!.props['aria-checked']]).toEqual([true, false])
  list!.props.onClick()
  await Vue.nextTick()
  expect(widgets[xs.WIDGET_IDS.custom_mode]).toBe('sigmas')
  expect(labels(root)).toEqual(['custom 방식', 'custom 시그마 목록'])
  expect(inputs(root)[0]!.props.placeholder).toBe('[1.0, 0.6, 0.25, 0.1, 0.0]')
})

it('error property from python is shown under the input it belongs to', async () => {
  widgets.scheduler_combo = 'custom'
  properties[xs.WIDGET_IDS.custom_expression] = { [xs.ERROR_PROPERTY]: "name 'q' is not allowed" }
  const root = await mount()
  expect(text(root)).toContain("식을 쓸 수 없습니다: name 'q' is not allowed")
  widgets[xs.WIDGET_IDS.custom_mode] = 'sigmas'
  await Vue.nextTick()
  expect(text(root)).not.toContain('식을 쓸 수 없습니다')
  properties[xs.WIDGET_IDS.custom_sigmas] = { [xs.ERROR_PROPERTY]: 'no numbers' }
  await Vue.nextTick()
  expect(text(root)).toContain('시그마 목록을 쓸 수 없습니다: no numbers')
})

it('reset writes the extension defaults and is disabled at defaults', async () => {
  widgets.scheduler_combo = 'laplace'
  widgets[xs.WIDGET_IDS.laplace_mu] = '2.5'
  widgets[xs.WIDGET_IDS.custom_mode] = 'sigmas'
  const root = await mount()
  const reset = () => all(root, 'button').find(b => b.props.class === 'xs-reset')!
  expect(reset().props.disabled).toBe(false)
  reset().props.onClick()
  await Vue.nextTick()
  for (const [key, value] of Object.entries(xs.DEFAULTS)) {
    expect(widgets[xs.WIDGET_IDS[key as keyof typeof xs.WIDGET_IDS]]).toBe(value)
  }
  expect(reset().props.disabled).toBe(true)
})

it('flow-unsuitable and Karras Dynamic hints show without the fields', async () => {
  widgets.scheduler_combo = 'Karras Dynamic'
  const root = await mount()
  expect(all(root, 'section')).toEqual([])
  expect(text(root)).toContain(xs.FLOW_HINT)
  expect(text(root)).toContain(xs.KARRAS_DYNAMIC_HINT)
})

it('status line: old Forge, ComfyUI pending, Krea2', async () => {
  widgets.scheduler_combo = 'laplace'
  caps.value = { status: 'ok', known: true, features: { extra_schedulers: false } }
  const root = await mount()
  expect(text(root)).toContain(xs.OLD_FORGE_NOTE)
  caps.value = { status: 'ok', known: true, features: { extra_schedulers: true } }
  await Vue.nextTick()
  expect(text(root)).not.toContain(xs.OLD_FORGE_NOTE)
  caps.value = { status: 'not_applicable', known: false, features: {} }
  await Vue.nextTick()
  expect(text(root)).toContain(xs.COMFY_PENDING_NOTE)
  widgets.generation_family_combo = 'Krea2'
  await Vue.nextTick()
  expect(text(root)).toContain(xs.KREA2_NOTE)
})
