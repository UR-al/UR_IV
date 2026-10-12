import { afterEach, beforeEach, expect, it } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, reactive, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './Sam3FaceCropFields.vue?raw'
import * as sam3FaceCrop from '../utils/sam3FaceCrop'
import { COMFY_SUPPORT_PROPERTY, sam3FaceCropDefaults, sam3FaceCropId } from '../utils/sam3FaceCrop'

/**
 * SAM3 '얼굴 안에서만 찾기' 줄의 ComfyUI 게이트 (앱 노드 팩 1.8.0).
 *
 * ComfyUI 는 팩이 sam-extra 와 같은 얼굴 crop 을 하지만 얼굴 검출에 ultralytics 가 있어야 한다. Python 이 /object_info 를
 * 받을 때마다 판단해 `_sam3_face_crop` 의 `comfySupport` 속성으로 보낸다. 검출기가 없으면 켜기를 막고 이유를 보인다 — 이미
 * 켜진 설정은 끌 수 있어야 한다(켠 채 생성하면 컴파일러가 이유와 함께 멈춘다). 옛 팩은 경고만 한다(다음 생성이 팩을 설치한다).
 */

const caps = ref<Record<string, unknown> | null>(null)
const comfy = ref<unknown>(null)
let refreshes = 0
const composable = { useSamExtraCapabilities: () => ({ capabilities: caps, refresh: () => { refreshes += 1 } }) }
const widgetStore = {
  getProperty: (id: string, prop: string) => (id === sam3FaceCropId('face_crop') && prop === COMFY_SUPPORT_PROPERTY
    ? comfy.value : undefined),
}

const script = compileScript(parse(source).descriptor, { id: 'sam3-face-crop-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
const toggleStub = { __esModule: true, default: { render: () => h('toggle') } }
const selectStub = {
  __esModule: true,
  default: { props: ['modelValue', 'options', 'placeholder'], render(this: any) { return h('csel', { options: this.options }) } },
}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../stores/widgetStore.js') return widgetStore
  if (name === '../composables/useSamExtraCapabilities') return composable
  if (name === '../utils/sam3FaceCrop') return sam3FaceCrop
  if (name === './ToggleSwitch.vue') return toggleStub
  if (name === './CustomSelect.vue') return selectStub
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  constructor(public tag: string, public text = '') {}
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
  addEventListener() {}
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
const toggle = (root: Node) => all(root, n => n.tag === 'toggle')[0]
const recheck = (root: Node) => all(root, n => n.tag === 'button' && n.props.class === 'fc-recheck')
const lines = (root: Node, cls: string) => all(root, n => n.tag === 'p' && n.props.class === cls).map(n => n.textContent)
const row = (root: Node) => all(root, n => n.tag === 'label' && String(n.props.class ?? '').includes('ext-check-row'))[0]

let app: App | undefined
function mount(values: Record<string, string> = {}) {
  const widgets = reactive({ ...sam3FaceCropDefaults(), ...values })
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compiled.default, { widgets }) })
  app.mount(root)
  return { root, widgets }
}
const ENABLE = sam3FaceCropId('face_crop')
beforeEach(() => { caps.value = { status: 'not_applicable', known: false }; comfy.value = null; refreshes = 0 })
afterEach(() => { app?.unmount(); app = undefined })

it('ComfyUI without ultralytics: switching on is refused and the reason is shown with a re-check', async () => {
  comfy.value = 'no_detector'
  const { root, widgets } = mount()
  await nextTick()
  expect(lines(root, 'fc-warn')).toHaveLength(1)
  expect(lines(root, 'fc-warn')[0]).toContain('ultralytics')
  expect(String(row(root).props.class)).toContain('fc-blocked')
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('false')
  expect(recheck(root)).toHaveLength(1)
  recheck(root)[0].props.onClick()
  expect(refreshes).toBe(1)
})

it('a setting already on can still be switched off on a blocked ComfyUI, and says generation would stop', async () => {
  comfy.value = 'no_detector'
  const { root, widgets } = mount({ [ENABLE]: 'true' })
  await nextTick()
  expect(lines(root, 'fc-warn')[0]).toContain('켜 둔 채 생성하면 이 이유로 멈춥니다')
  expect(String(row(root).props.class)).not.toContain('fc-blocked')
  toggle(root).props['onUpdate:modelValue'](false)
  await nextTick()
  expect(widgets[ENABLE]).toBe('false')
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('false')
})

it('an old pack only warns: the next generation installs the bundled pack, and the status can be re-checked', async () => {
  comfy.value = 'old_pack'
  const { root, widgets } = mount()
  await nextTick()
  expect(lines(root, 'fc-warn')[0]).toContain('1.8.0')
  expect(String(row(root).props.class)).not.toContain('fc-blocked')
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('true')
  expect(recheck(root)).toHaveLength(1)
  comfy.value = 'no_detector'   // the pushed status after the install: now it is blocked
  await nextTick()
  expect(lines(root, 'fc-warn')[0]).toContain('ultralytics')
})

it('ComfyUI with pack 1.8.0 and ultralytics switches on and says how it runs there', async () => {
  comfy.value = 'ready'
  const { root, widgets } = mount()
  await nextTick()
  expect(lines(root, 'fc-warn')).toEqual([])
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('true')
  expect(lines(root, 'fc-note').some(text => text.includes('ComfyUI: 앱 노드 팩'))).toBe(true)
  expect(lines(root, 'fc-warn')).toEqual([])
  expect(recheck(root)).toEqual([])
  const notes = lines(root, 'fc-note').join(' ')
  expect(notes).toContain('crop 을 키워 다시 찾습니다')
  expect(notes).not.toContain('잘립니다')
  expect(notes).not.toContain('sam-extra 와 같은 방식')
})

it('Forge says hair is cut on sam-extra 0.34.0 and regrown from 0.34.1', async () => {
  caps.value = { status: 'ok', known: true, features: { sam3: true }, sam3_keys: { missing_in_extension: [] },
    version: { extension_version: '0.34.0' } }
  const first = mount({ [ENABLE]: 'true' })
  await nextTick()
  expect(lines(first.root, 'fc-note').join(' ')).toContain('잘립니다')
  app?.unmount()
  caps.value = { ...caps.value, version: { extension_version: '0.34.1' } }
  const second = mount({ [ENABLE]: 'true' })
  await nextTick()
  const notes = lines(second.root, 'fc-note').join(' ')
  expect(notes).toContain('crop 을 키워 다시 찾습니다')
  expect(notes).not.toContain('잘립니다')
})

it('ComfyUI before the node info arrives is not blocked and warns only once switched on', async () => {
  const { root, widgets } = mount()
  await nextTick()
  expect(lines(root, 'fc-warn')).toEqual([])
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('true')
  expect(lines(root, 'fc-warn')[0]).toContain('아직 확인하지 못했습니다')
})

it('Forge is never blocked by a stale ComfyUI property', async () => {
  caps.value = { status: 'ok', known: true, features: { sam3: true }, sam3_keys: { missing_in_extension: [] } }
  comfy.value = 'no_detector'
  const { root, widgets } = mount()
  await nextTick()
  toggle(root).props['onUpdate:modelValue'](true)
  await nextTick()
  expect(widgets[ENABLE]).toBe('true')
  expect(lines(root, 'fc-warn')).toEqual([])
})
