import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './RefinePanel.vue?raw'
import i2iSource from '../views/I2IView.vue?raw'
import * as hostBridge from '../bridge.js'
import * as widgetStore from '../stores/widgetStore.js'
import * as media from '../utils/media.js'
import * as sam3ControlNet from '../utils/sam3ControlNet'
import { SAM3_CN_FIELDS } from '../utils/sam3ControlNet'

// 구독 목록을 bridge.js 처럼 흉내 낸다 — onBackendEvent 는 해제 함수를 돌려준다.
const host = vi.hoisted(() => ({ subscribers: new Set<(raw: string) => void>(), action: vi.fn() }))
vi.mock('../bridge.js', () => ({
  onBackendEvent: (name: string, callback: (raw: string) => void) => {
    if (name !== 'refineResult') throw Error(`unexpected event ${name}`)
    host.subscribers.add(callback)
    return () => { host.subscribers.delete(callback) }
  },
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: host.action }))
vi.mock('../utils/media.js', () => ({ mediaUrl: (path: string) => `media://${path}` }))

const script = compileScript(parse(source).descriptor, { id: 'refine-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
const stub = { __esModule: true, default: { render: () => null } }
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../bridge.js') return hostBridge
  if (name === '../stores/widgetStore.js') return widgetStore
  if (name === '../utils/media.js') return media
  if (name === '../utils/sam3ControlNet') return sam3ControlNet
  if (name === './CustomSelect.vue' || name === './ToggleSwitch.vue' || name === './Sam3ControlNetPanel.vue') return stub
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  value = ''
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

let app: App | undefined
beforeEach(() => { host.subscribers.clear(); host.action.mockReset() })
afterEach(() => { app?.unmount(); app = undefined })

it('switching the I2I sub-tab away and back never accumulates refineResult subscribers', async () => {
  const shown = ref(true)
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => (shown.value ? h(compiled.default, { imagePath: 'C:/in.png' }) : null) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await nextTick()
  expect(host.subscribers.size).toBe(1)
  for (let i = 0; i < 5; i++) {   // img2img ↔ refine 전환
    shown.value = false; await nextTick()
    expect(host.subscribers.size).toBe(0)
    shown.value = true; await nextTick()
    expect(host.subscribers.size).toBe(1)
  }
  const [listener] = [...host.subscribers]
  listener!(JSON.stringify({ error: '마스크가 비었습니다' }))
  await nextTick()
  expect(root.textContent).toContain('마스크가 비었습니다')
})

function findNode(node: Node, match: (candidate: Node) => boolean): Node | undefined {
  if (match(node)) return node
  for (const child of node.children) {
    const found = findNode(child, match)
    if (found) return found
  }
  return undefined
}

it('run_refine sends all 13 SAM3 ControlNet fields with extension defaults', async () => {
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compiled.default, { imagePath: 'C:/in.png' }) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await nextTick()
  const button = findNode(root, (node) => node.tag === 'button' && String(node.props.class || '').includes('btn-generate'))
  expect(button).toBeDefined()
  button!.props.onClick()
  expect(host.action).toHaveBeenCalledTimes(1)
  const [name, payload] = host.action.mock.calls[0]
  expect(name).toBe('run_refine')
  const cnKeys = Object.keys(payload.settings).filter((key) => key.startsWith('sam3_cn_'))
  expect(cnKeys.sort()).toEqual(SAM3_CN_FIELDS.map((field) => `sam3_${field.key}`).sort())
  expect(payload.settings.sam3_cn_enable).toBe(false)
  expect(payload.settings.sam3_cn_override_external).toBe(false)
  expect(payload.settings.sam3_cn_threshold_a).toBe(-1)
  expect(payload.settings.sam3_cn_threshold_b).toBe(-1)
  expect(payload.settings.sam3_cn_module).toBe('inpaint_only')
})

it('RefinePanel no longer carries its own copy of the ControlNet module list', () => {
  expect(source).not.toMatch(/'inpaint_only\+lama'/)
  expect(source).toMatch(/<Sam3ControlNetPanel\b/)
})

it('I2IView still mounts RefinePanel with v-if, which is why cleanup is required', () => {
  expect(i2iSource).toMatch(/<RefinePanel v-if="subTab === 'refine'"/)
})
