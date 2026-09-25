import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './QueuePanel.vue?raw'
import * as queueLocks from '../utils/queueLocks'

/**
 * 대기열 패널 — 실제 bridge.js · widgetStore.js 로, App 과 같은 순서(패널 마운트 → initBridge)에서.
 *
 * 회귀: 패널이 onMounted 에서 곧바로 sync_queue_state 를 보냈는데, 자식 onMounted 는 App 의
 * onMounted(await initBridge())보다 먼저 돌아 스토어에 백엔드가 없었다 — 요청이 버려져 시작 때 복구된
 * 대기열이 Vue 에 보이지 않았다('시작' 버튼도 숨은 채).
 */

const script = compileScript(parse(source).descriptor, { id: 'queue-panel-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText

// 드로어 전환 효과(runtime-dom Transition)는 classList·requestAnimationFrame 이 필요하다 — 여기선 통과만
const vueForTest = { ...Vue, Transition: (_props: unknown, { slots }: any) => slots.default?.() }

function compile(bridge: unknown, store: unknown) {
  const compiled: { default?: any } = {}
  new Function('require', 'exports', code)((name: string) => {
    if (name === 'vue') return vueForTest
    if (name === '../bridge.js') return bridge
    if (name === '../stores/widgetStore.js') return store
    if (name === '../utils/queueLocks') return queueLocks
    throw Error(`Unexpected component dependency: ${name}`)
  }, compiled)
  return compiled.default
}

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
function findAll(node: Node, match: (candidate: Node) => boolean, out: Node[] = []): Node[] {
  if (match(node)) out.push(node)
  for (const child of node.children) findAll(child, match, out)
  return out
}
const byClass = (root: Node, cls: string) =>
  findAll(root, node => String(node.props.class || '').split(/\s+/).includes(cls))

type Listener = (...args: any[]) => void
function signal() {
  const listeners = new Set<Listener>()
  return {
    connect: (fn: Listener) => { listeners.add(fn) },
    disconnect: (fn: Listener) => { listeners.delete(fn) },
    emit: (...args: any[]) => { for (const fn of [...listeners]) fn(...args) },
  }
}

/** Qt 임베드 모드 백엔드 대역 — 받은 액션을 기록하고, 시그널을 테스트가 쏠 수 있다. */
function fakeQtBackend() {
  const actions: Array<[string, any]> = []
  const backend = {
    widgetValueChanged: signal(), widgetPropertyChanged: signal(), batchUpdate: signal(),
    queueUpdated: signal(), queueItemAdded: signal(), queueCompleted: signal(), automationStatus: signal(),
    onWidgetChanged: () => {},
    onAction: (name: string, payload: string) => { actions.push([name, JSON.parse(payload || '{}')]) },
    getAllWidgetValues: (cb: (json: string) => void) => cb('{}'),
  }
  return { backend, actions }
}

let app: App | undefined
beforeEach(() => {
  vi.spyOn(console, 'log').mockImplementation(() => {})
})
afterEach(() => {
  app?.unmount()
  app = undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function mountBeforeBridge() {
  const { backend, actions } = fakeQtBackend()
  vi.stubGlobal('window', {
    QWebChannel: class { constructor(_transport: unknown, ready: (channel: any) => void) { ready({ objects: { backend } }) } },
    qt: { webChannelTransport: {} },
    addEventListener: () => {}, removeEventListener: () => {},
  })
  vi.resetModules()
  const bridge = await import('../bridge.js')
  const store = await import('../stores/widgetStore.js')
  const root = new Node('root')
  // 드로어는 우하단 도크(dock/QuickDock.vue)가 v-model:open 으로 연다 — 여기선 그 자리를 대신한다
  const open = ref(false)
  const panel = compile(bridge, store)
  app = renderer.createApp({ setup: () => () => h(panel, { open: open.value, 'onUpdate:open': (v: boolean) => { open.value = v } }) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)                                   // App 의 onMounted(initBridge)보다 먼저 — 아직 백엔드 없음
  openers.set(root, open)
  return { bridge, backend, actions, root }
}

const openers = new WeakMap<Node, { value: boolean }>()
async function openDrawer(root: Node) {
  openers.get(root)!.value = true
  await nextTick()
}

const QUEUE_STATE = (extra: Record<string, unknown> = {}) => JSON.stringify({
  items: [{ id: 'a', prompt: 'restored one' }, { id: 'b', prompt: 'restored two' }],
  running: false, paused: false, current_index: -1, completed: 0, processing_index: -1, ...extra,
})

it('asks for the restored queue only once the bridge is bound, and shows the answer', async () => {
  const { bridge, backend, actions, root } = await mountBeforeBridge()
  expect(actions).toEqual([])
  await bridge.initBridge()
  expect(actions.map(([name]) => name)).toEqual(['sync_queue_state'])   // 버려지지 않았다

  backend.queueUpdated.emit(QUEUE_STATE())                               // 백엔드의 답
  await nextTick()
  await openDrawer(root)
  expect(byClass(root, 'count-badge')[0].textContent).toBe('2')
  const start = byClass(root, 'btn').find(node => node.textContent.includes('시작'))
  expect(start).toBeDefined()                                            // 복구된 대기열을 시작할 수 있다
  expect(start!.props.disabled).toBe(false)
})

it('disables Start while automation drains the queue and follows automationStatus', async () => {
  const { bridge, backend, actions, root } = await mountBeforeBridge()
  await bridge.initBridge()
  backend.queueUpdated.emit(QUEUE_STATE({ automation: true }))
  await openDrawer(root)
  const start = () => byClass(root, 'btn').find(node => node.textContent.includes('시작'))!
  expect(start().props.disabled).toBe(true)
  start().props.onClick()
  expect(actions.map(([name]) => name)).not.toContain('start_queue')

  backend.automationStatus.emit(JSON.stringify({ running: false }))
  await nextTick()
  expect(start().props.disabled).toBe(false)
  start().props.onClick()
  expect(actions.map(([name]) => name)).toContain('start_queue')
})

it('clearing everything leaves the generating row out of the count', async () => {
  const { bridge, backend, actions, root } = await mountBeforeBridge()
  await bridge.initBridge()
  const confirmSpy = vi.fn(() => true)
  vi.stubGlobal('confirm', confirmSpy)
  backend.queueUpdated.emit(QUEUE_STATE({ processing_index: 0 }))
  await openDrawer(root)
  const clear = () => byClass(root, 'btn').find(node => node.textContent.includes('전체'))!
  clear().props.onClick()
  expect(confirmSpy).toHaveBeenCalledWith('생성 중인 1개를 뺀 1개 항목을 삭제할까요?')
  expect(actions.map(([name]) => name)).toContain('clear_queue')

  // 생성 중인 행뿐이면 묻지도 지우지도 않고 알린다
  actions.length = 0
  confirmSpy.mockClear()
  backend.queueUpdated.emit(JSON.stringify({
    items: [{ id: 'a', prompt: 'generating' }], running: false, paused: false,
    current_index: -1, completed: 0, processing_index: 0,
  }))
  await nextTick()
  clear().props.onClick()
  expect(confirmSpy).not.toHaveBeenCalled()
  expect(actions).toEqual([['show_toast', { type: 'warning', msg: queueLocks.ONLY_RUNNING_ROW_NOTICE }]])
})
