import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, reactive, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './Sam3ControlNetPanel.vue?raw'
import * as sam3ControlNet from '../utils/sam3ControlNet'
import { sam3CnDefaults, sam3CnId } from '../utils/sam3ControlNet'

/**
 * SAM3 ControlNet 패널의 모델 칸 (P3 보완).
 *
 * 라이브 모델 목록은 **연결 때** 받은 Forge `controlnet_names` 다. 확장은 SAM3 패스마다 models/sam3 를 다시
 * 스캔해 `controlnet_filename_dict` 에만 더한다(sam3ext/inpaint_core.inject_controlnet_unit) — 그래서 목록에 없는
 * 모델도 생성 때 찾을 수 있다. 목록을 알아도 직접 입력할 수 있어야 하고, 경고는 '실패'로 단정하지 않으며,
 * 스냅샷을 다시 받는 버튼이 있어야 한다.
 */

const caps = ref<Record<string, unknown> | null>(null)
const refresh = vi.fn()
const composable = {
  useSamExtraCapabilities: () => ({ capabilities: caps, refresh, mayUse: () => true }),
}
const widgetStore = { getProperty: () => '' }

const script = compileScript(parse(source).descriptor, { id: 'sam3-cn-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
const toggleStub = { __esModule: true, default: { render: () => h('toggle') } }
const selectStub = {
  __esModule: true,
  default: {
    props: ['modelValue', 'options', 'placeholder'],
    render(this: any) { return h('csel', { options: this.options, value: this.modelValue }) },
  },
}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../stores/widgetStore.js') return widgetStore
  if (name === '../composables/useSamExtraCapabilities') return composable
  if (name === '../utils/sam3ControlNet') return sam3ControlNet
  if (name === './ToggleSwitch.vue') return toggleStub
  if (name === './CustomSelect.vue') return selectStub
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  value = ''
  type = 'text'
  constructor(public tag: string, public text = '') {}
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
  addEventListener() {}   // v-model 지시자용
  removeEventListener() {}
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

function all(node: Node, match: (n: Node) => boolean, out: Node[] = []): Node[] {
  if (match(node)) out.push(node)
  for (const child of node.children) all(child, match, out)
  return out
}
const modelField = (root: Node) => all(root, n => n.props.class === 'cn-model-row')[0]
const buttons = (root: Node) => all(modelField(root), n => n.tag === 'button')
const warnings = (root: Node) => all(root, n => n.props.class === 'cn-warn').map(n => n.textContent)

// 녹화된 Forge classic(sam-extra 0.30.0) 목록의 일부 (전체는 tests/fixtures/sam_extra_live_cn_*.json)
const LIVE_MODULES = ['None', 'inpaint_noobai', 'inpaint_only', 'inpaint_only+lama']
const LIVE_MODELS = ['None', 'anima-lllite-inpainting-v2', 'animaTileRepair_v10']
const known = (models = LIVE_MODELS) => ({
  status: 'ok', known: true, choices: { controlnet_modules: LIVE_MODULES, controlnet_models: models },
})

let app: App | undefined
function mount(values: Record<string, string> = {}) {
  const widgets = reactive({ ...sam3CnDefaults(), [sam3CnId('cn_enable')]: 'true', ...values })
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compiled.default, { widgets }) })
  app.component('Icon', { props: ['name'], render(this: any) { return h('icon', { name: this.name }) } })
  app.mount(root)
  return { root, widgets }
}
beforeEach(() => { caps.value = null; refresh.mockReset() })
afterEach(() => { app?.unmount(); app = undefined })

it('without a snapshot the model is free text and the list can be fetched again', async () => {
  const { root } = mount()
  await nextTick()
  const field = modelField(root)
  expect(all(field, n => n.tag === 'input')).toHaveLength(1)
  expect(all(field, n => n.tag === 'csel')).toHaveLength(0)
  // 연필(직접 입력 전환)은 목록이 있을 때만 — 지금 남는 것은 ↻ 하나
  expect(buttons(root)).toHaveLength(1)
  buttons(root)[0].props.onClick()
  expect(refresh).toHaveBeenCalledTimes(1)
  expect(warnings(root)).toEqual([])
})

it('a live list keeps a name Forge still resolves (models/sam3 LLLite added after start) and does not claim failure', async () => {
  caps.value = known()
  const { root, widgets } = mount({ [sam3CnId('cn_model')]: 'new-lllite' })
  await nextTick()
  const select = all(modelField(root), n => n.tag === 'csel')
  expect(select).toHaveLength(1)
  expect(select[0].props.options).toEqual([...LIVE_MODELS, 'new-lllite'])
  expect(widgets[sam3CnId('cn_model')]).toBe('new-lllite')          // 지우거나 바꿔치지 않는다
  const [modelWarning] = warnings(root)
  expect(modelWarning).toContain('models/sam3')
  expect(modelWarning).toContain('다시 찾지만')
})

it('the pencil switches a live-list model field to free text and back', async () => {
  caps.value = known()
  const { root } = mount()
  await nextTick()
  const [pencil, reload] = buttons(root)
  expect(all(pencil, n => n.tag === 'icon')[0].props.name).toBe('pencil')
  pencil.props.onClick()
  await nextTick()
  expect(all(modelField(root), n => n.tag === 'input')).toHaveLength(1)
  expect(all(modelField(root), n => n.tag === 'csel')).toHaveLength(0)
  expect(buttons(root)[0].props['aria-pressed']).toBe(true)
  buttons(root)[0].props.onClick()
  await nextTick()
  expect(all(modelField(root), n => n.tag === 'csel')).toHaveLength(1)
  reload.props.onClick()
  expect(refresh).toHaveBeenCalledTimes(1)
})

it('a listed model is quiet; an unknown preprocessor still warns about the KeyError', async () => {
  caps.value = known()
  const { root } = mount({ [sam3CnId('cn_model')]: 'anima-lllite-inpainting-v2', [sam3CnId('cn_module')]: 'not_a_module' })
  await nextTick()
  const got = warnings(root)
  expect(got).toHaveLength(1)                     // 전처리기는 Forge 시작 때 정해진다 — 목록 밖이면 실패
  expect(got[0]).toContain('전처리기')
})

it('ComfyUI (not_applicable) has no Forge list to fetch', async () => {
  caps.value = { status: 'not_applicable', known: false, choices: {} }
  const { root } = mount()
  await nextTick()
  expect(buttons(root)).toHaveLength(0)
})

// Anima LLLite 전처리기 가드 (plan §7.2 #7·#14) — 원본(kohya sd-scripts·ComfyUI-Anima-LLLite)은 사용자가 준 제어 이미지
const moduleSelect = (root: Node) => all(root, n => n.tag === 'csel' && n.props.options?.includes('None'))
  .find(n => !n.props.options.some((o: string) => o.startsWith('anima')))
const infos = (root: Node) => all(root, n => n.props.class === 'cn-info').map(n => n.textContent)

it('a saved inpaint_only with a Tile & Repair LLLite becomes None and the list is only None', async () => {
  caps.value = known([...LIVE_MODELS, 'animaTileRepair_v20'])
  const { root, widgets } = mount({ [sam3CnId('cn_model')]: 'animaTileRepair_v20' })
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('None')           // 기본값·저장값 inpaint_only 보정
  expect(moduleSelect(root)?.props.options).toEqual(['None'])
  expect(infos(root)).toHaveLength(1)
  expect(infos(root)[0]).toContain('Tile &')
  expect(warnings(root)).toEqual([])
  widgets[sam3CnId('cn_module')] = 'tile_resample'              // 다른 경로로 들어와도 다시 None
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('None')
})

it('a 3-channel lineart Anima LLLite keeps lineart_anime; only inpaint_* becomes None', async () => {
  const modules = ['None', 'canny', 'inpaint_only', 'lineart_anime']
  caps.value = { status: 'ok', known: true,
    choices: { controlnet_modules: modules, controlnet_models: [...LIVE_MODELS, 'anima_lllite_lineart_v1'] } }
  const { root, widgets } = mount({ [sam3CnId('cn_model')]: 'anima_lllite_lineart_v1', [sam3CnId('cn_module')]: 'lineart_anime' })
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('lineart_anime')     // 전처리기가 제어 맵을 만든다 — 그대로
  expect(moduleSelect(root)?.props.options).toEqual(['None', 'canny', 'lineart_anime'])
  expect(infos(root)).toHaveLength(1)
  expect(infos(root)[0]).toContain('3채널 Anima LLLite')
  expect(infos(root)[0]).not.toContain('Tile')
  widgets[sam3CnId('cn_module')] = 'inpaint_only'                 // 원본 3채널은 마스크를 쓰지 않는다
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('None')
  widgets[sam3CnId('cn_module')] = 'canny'
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('canny')
})

it('a 4-channel inpaint LLLite drops inpaint_* preprocessors only', async () => {
  caps.value = known()
  const { root, widgets } = mount({ [sam3CnId('cn_model')]: 'anima-lllite-inpainting-v2' })
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('None')
  expect(moduleSelect(root)?.props.options).toEqual(['None'])   // 이 라이브 목록의 나머지는 모두 inpaint_*
  expect(infos(root)[0]).toContain('4채널')
})

it('other models keep the preprocessor and show no LLLite note', async () => {
  caps.value = known()
  const { root, widgets } = mount()
  await nextTick()
  expect(widgets[sam3CnId('cn_module')]).toBe('inpaint_only')
  expect(infos(root)).toEqual([])
})
