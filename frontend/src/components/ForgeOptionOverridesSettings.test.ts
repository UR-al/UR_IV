import { afterEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './ForgeOptionOverridesSettings.vue?raw'
import * as table from '../utils/forgeOptionOverrides'

const action = vi.fn(), read = vi.fn(), off = vi.fn()
let prefsEvent: ((raw: string) => void) | undefined
const caps = Vue.ref<any>(null)
const compiled = compileScript(parse(source).descriptor, { id: 'forge-options-test', inlineTemplate: true })
const code = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
// CustomSelect 대역 — 고른 라벨을 onPick 으로 흉내 낸다
const SelectStub = Vue.defineComponent({
  props: { modelValue: { type: [String, Number], default: '' }, options: { type: Array, default: () => [] } },
  emits: ['update:modelValue'],
  setup(props, { emit }) {
    return () => Vue.h('csel', { value: props.modelValue, options: props.options, onPick: (label: string) => emit('update:modelValue', label) })
  },
})
const output: { default?: any } = {}
new Function('require', 'exports', code)((id: string) => {
  if (id === 'vue') return Vue
  if (id === './CustomSelect.vue') return { __esModule: true, default: SelectStub }
  if (id === '../bridge.js') return { getBackend: async () => ({ getUiPrefs: read }), onBackendEvent: (_event: string, receive: (raw: string) => void) => { prefsEvent = receive; return off } }
  if (id === '../stores/widgetStore.js') return { requestAction: action }
  if (id === '../composables/useSamExtraCapabilities') return { useSamExtraCapabilities: () => ({ capabilities: caps }) }
  if (id === '../utils/forgeOptionOverrides') return table
  throw Error(id)
}, output)

class Node {
  children: Node[] = []; parent?: Node; props: Record<string, any> = {}
  constructor(public tag: string, public text = '') {}
}
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
async function mount(prefs = '{}') {
  prefsEvent = undefined
  action.mockReset(); read.mockReset(); off.mockReset()
  read.mockImplementation((callback: (raw: string) => void) => callback(prefs))
  const root = node('root'); app = renderer.createApp(output.default); app.mount(root)
  await Vue.nextTick(); await Vue.nextTick(); return root
}
const selectFor = (root: Node, key: string) => all(root, 'csel')[table.SPECS.findIndex(spec => spec.key === key)]!
afterEach(() => { app?.unmount(); app = undefined; caps.value = null })

const DEDUP = 'sam3_guidance_pag_prefix_dedup', DAVE = 'sam3_guidance_dave_pre_dd_sigma'
const DEVICE = 'sam3_degrid_device', PRECISION = 'sam3_degrid_gpu_precision'

it('mount reads prefs and never saves (default = Forge 설정 따름, nothing sent)', async () => {
  const root = await mount()
  expect(all(root, 'csel')).toHaveLength(13)
  expect(all(root, 'csel').every(sel => sel.props.value === 'Forge 설정 따름')).toBe(true)
  expect(action).not.toHaveBeenCalled()
  expect(prefsEvent).toBeTypeOf('function')
  expect(text(root)).toContain('모두 Forge 설정 따름')
})

it('shows saved values and writes the whole dict; follow removes the key', async () => {
  const root = await mount(JSON.stringify({ forgeOptionOverrides: { [DEDUP]: false, junk: true } }))
  expect(selectFor(root, DEDUP).props.value).toBe('끔')
  selectFor(root, DAVE).props.onPick('켬')
  await Vue.nextTick()
  expect(action).toHaveBeenLastCalledWith('save_ui_prefs', { forgeOptionOverrides: { [DEDUP]: false, [DAVE]: true } })
  selectFor(root, DEDUP).props.onPick('Forge 설정 따름')
  await Vue.nextTick()
  expect(action).toHaveBeenLastCalledWith('save_ui_prefs', { forgeOptionOverrides: { [DAVE]: true } })
  // 늦게 온 시작 값은 사용자가 고친 값을 덮지 않는다
  prefsEvent!(JSON.stringify({ forgeOptionOverrides: {} }))
  await Vue.nextTick()
  expect(selectFor(root, DAVE).props.value).toBe('켬')
})

it('radio rows list their own choices and save the choice string', async () => {
  const root = await mount(JSON.stringify({ forgeOptionOverrides: { [PRECISION]: 'fp16', [DEVICE]: true } }))
  expect(selectFor(root, PRECISION).props.value).toBe('fp16 autocast')
  expect(selectFor(root, DEVICE).props.value).toBe('Forge 설정 따름')       // bool 은 라디오 값이 아니다(따름)
  expect(selectFor(root, DEVICE).props.options).toEqual(['Forge 설정 따름', 'GPU (Forge 장치)', 'CPU (VRAM 안 씀, 느림)'])
  expect(selectFor(root, DEDUP).props.options).toEqual(['Forge 설정 따름', '켬', '끔'])
  selectFor(root, DEVICE).props.onPick('CPU (VRAM 안 씀, 느림)')
  await Vue.nextTick()
  expect(action).toHaveBeenLastCalledWith('save_ui_prefs', { forgeOptionOverrides: { [DEVICE]: 'cpu', [PRECISION]: 'fp16' } })
  expect(text(root)).toContain('6.5초')
  selectFor(root, PRECISION).props.onPick('Forge 설정 따름')
  await Vue.nextTick()
  expect(action).toHaveBeenLastCalledWith('save_ui_prefs', { forgeOptionOverrides: { [DEVICE]: 'cpu' } })
})

it('status line follows the capability snapshot; ComfyUI shows the not-applicable note', async () => {
  caps.value = { status: 'ok', known: true, options_known: true, options: { [DEDUP]: true } }
  const root = await mount(JSON.stringify({ forgeOptionOverrides: { [DAVE]: false } }))
  const body = text(root)
  expect(body).toContain('Forge 값: 켬 (Forge 시작 시점)')
  expect(body).toContain('이 Forge 의 sam-extra 에 없음')
  expect(body).toContain('무너집니다')
  caps.value = { status: 'not_applicable', known: false }
  await Vue.nextTick()
  expect(text(root)).toContain('ComfyUI 백엔드에는 해당 없음')
})
