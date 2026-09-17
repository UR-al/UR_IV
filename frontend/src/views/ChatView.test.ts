import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import chatSource from './ChatView.vue?raw'
import selectSource from '../components/CustomSelect.vue?raw'
import presetSource from '../components/InstructionPresets.vue?raw'
import * as chatBridge from '../bridge.js'
import * as widgetStore from '../stores/widgetStore.js'
import * as media from '../utils/media.js'
import * as chatMarkdown from '../utils/chatMarkdown'
import * as clipboard from '../utils/clipboard'
import * as chatSettings from '../utils/chatSettings'
import * as chatStructuredOutput from '../utils/chatStructuredOutput'
import * as chatGeneration from '../utils/chatGeneration'
import * as dropdownPlacement from '../utils/dropdownPlacement'

const bridge = vi.hoisted(() => ({
  handlers: new Map<string, (raw: string) => void>(),
  action: vi.fn(),
  backend: null as any,
}))
vi.mock('../bridge.js', () => ({
  getBackend: async () => bridge.backend,
  onBackendEvent: (name: string, callback: (raw: string) => void) => {
    bridge.handlers.set(name, callback)
    return () => bridge.handlers.delete(name)
  },
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: bridge.action }))

// Vitest's node environment normally compiles SFCs for SSR, which cannot mount.
// Compile their current source for this renderer, using the installed compiler.
function compileClient(source: string, modules: Record<string, unknown>) {
  const script = compileScript(parse(source).descriptor, { id: 'chat-regression', inlineTemplate: true })
  const code = ts.transpileModule(script.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText
  const exports: { default?: any } = {}
  new Function('require', 'exports', code)((id: string) => {
    if (!(id in modules)) throw Error(`Unknown component dependency: ${id}`)
    return modules[id]
  }, exports)
  return exports.default
}
const CustomSelect = compileClient(selectSource, { vue: Vue, '../utils/dropdownPlacement': dropdownPlacement })
const InstructionPresets = compileClient(presetSource, { vue: Vue, '../bridge.js': chatBridge })
const ChatView = compileClient(chatSource, {
  vue: Vue, '../bridge.js': chatBridge, '../stores/widgetStore.js': widgetStore,
  '../utils/media.js': media, '../utils/chatMarkdown': chatMarkdown, '../utils/clipboard': clipboard,
  '../utils/chatSettings': chatSettings, '../utils/chatGeneration': chatGeneration,
  '../components/CustomSelect.vue': { __esModule: true, default: CustomSelect },
  '../utils/chatStructuredOutput': chatStructuredOutput,
  '../components/AiAssistInstructionsSettings.vue': { __esModule: true, default: { render: () => null } },
  '../components/InstructionPresets.vue': { __esModule: true, default: { render: () => null } },
})

// Vue's public renderer boundary supplies a tiny in-memory DOM. No browser,
// Ollama process, real clipboard or user localStorage is touched by this test.
class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  style: Record<string, unknown> = {}
  value = ''
  constructor(public tag: string, public text = '') {}
  get tagName() { return this.tag.toUpperCase() }
  get options() { return this.children.filter(child => child.tag === 'option') }
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
  addEventListener() {}
  removeEventListener() {}
  setAttribute(name: string, value: string) { this.props[name] = value }
  removeAttribute(name: string) { delete this.props[name] }
  focus() {}
  scrollTo() {}
  getRootNode() { return { activeElement: null } }
}
const body = new Node('body')
function insert(node: Node, parent: Node, anchor: Node | null = null) {
  if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1)
  node.parent = parent
  const index = anchor ? parent.children.indexOf(anchor) : -1
  parent.children.splice(index < 0 ? parent.children.length : index, 0, node)
}
const renderer = createRenderer<Node, Node>({
  createElement: tag => new Node(tag), createText: text => new Node('#text', text),
  createComment: () => new Node('#comment'),
  insert, remove: node => { if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1) },
  setText: (node, text) => { node.text = text },
  setElementText: (node, text) => { node.text = text; node.children = [] },
  parentNode: node => node.parent,
  nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
  patchProp: (node, key, _old, value) => { node.props[key] = value; if (key === 'value') node.value = value },
  querySelector: () => body, setScopeId: () => {},
  insertStaticContent: (text, parent, anchor) => { const node = new Node('#static', text); insert(node, parent, anchor); return [node, node] },
})
function find(root: Node, predicate: (node: Node) => boolean): Node | undefined {
  if (predicate(root)) return root
  for (const child of root.children) { const found = find(child, predicate); if (found) return found }
}
let app: App | undefined
beforeEach(() => {
  vi.useFakeTimers()
  bridge.handlers.clear()
  bridge.action.mockClear()
  bridge.backend = null
  body.children = []
  const storage = new Map([['ollamaModel', 'selected-model']])
  vi.stubGlobal('localStorage', { getItem: (key: string) => storage.get(key) ?? null, setItem: (key: string, value: string) => storage.set(key, value) })
  vi.stubGlobal('window', { addEventListener() {}, removeEventListener() {} })
  vi.stubGlobal('document', { addEventListener() {}, removeEventListener() {} })
  vi.stubGlobal('Document', class {})
  vi.stubGlobal('ShadowRoot', class {})
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} })
  vi.stubGlobal('cancelAnimationFrame', () => {})
})
afterEach(() => { app?.unmount(); app = undefined; vi.clearAllTimers(); vi.useRealTimers(); vi.unstubAllGlobals() })

it('replaces a timed-out model-info warning with matching late success and keeps stale replies out', async () => {
  const root = new Node('root')
  app = renderer.createApp(ChatView)
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await nextTick()
  find(root, node => String(node.props.title || '').startsWith('대화 설정'))!.props.onClick()
  await nextTick()
  const request = bridge.action.mock.calls.find(call => call[0] === 'chat_model_info')![1]
  await vi.advanceTimersByTimeAsync(15000)
  expect(root.textContent).toContain('모델 정보를 받지 못했습니다')

  bridge.handlers.get('chatModelInfo')!(JSON.stringify({ id: 'old-request', model: 'selected-model', ok: true, info: { architecture: 'stale-model' } }))
  await nextTick()
  expect(root.textContent).not.toContain('stale-model')

  bridge.handlers.get('chatModelInfo')!(JSON.stringify({ id: request.id, model: 'selected-model', ok: true,
    info: { architecture: 'qwen3', thinkingMode: 'boolean', moe: false, vision: false } }))
  await nextTick()
  expect(root.textContent).toContain('qwen3')
  expect(root.textContent).toContain('켜기/끄기 지원')
  expect(root.textContent).not.toContain('모델 정보를 받지 못했습니다')
  expect(root.textContent).not.toContain('추론 설정은 모델 기본값으로 전달합니다')
})

async function mountChat() {
  const root = new Node('root')
  app = renderer.createApp(ChatView)
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await nextTick()
  bridge.handlers.get('chatThreads')!('[]')
  await nextTick()
  return root
}
function button(root: Node, text: string) { return find(root, node => node.tag === 'button' && node.textContent === text)! }

it('sends JSON schema only when enabled and leaves an invalid draft unsent', async () => {
  const root = await mountChat()
  const schema = find(root, n => n.props.id === 'chat-json-schema')!
  const toggle = find(root, n => n.tag === 'input' && n.props.type === 'checkbox')!
  toggle.props['onUpdate:modelValue'](true); toggle.props.onChange()
  schema.props['onUpdate:modelValue']('{broken')
  const composer = find(root, n => n.props.class === 'cmp-input')!
  composer.props['onUpdate:modelValue']('태그를 작성해 줘')
  await nextTick()
  const send = () => find(root, n => n.props.title === '보내기 (Enter)')!.props.onClick()
  send()
  expect(bridge.action.mock.calls.some(call => call[0] === 'chat_send')).toBe(false)
  expect(composer.value).toBe('태그를 작성해 줘')
  schema.props['onUpdate:modelValue']('{"type":"object","properties":{"answer":{"type":"string"}}}')
  await nextTick()
  send()
  const sent = bridge.action.mock.calls.find(call => call[0] === 'chat_send')![1]
  expect(sent.schema.type).toBe('object')
  expect(sent.generation.mode).toBe('chat')
  expect(sent.provider).toBe('ollama')
  expect(sent.system).not.toContain('AI 어시스트')
  bridge.handlers.get('chatDone')!(JSON.stringify({ id: sent.id, ok: false, content: '{partial', error: '형식 오류' }))
  await nextTick()
  expect(root.textContent).toContain('{partial')
  expect(root.textContent).toContain('형식 오류')
})

it('LM Studio selection uses a separate model list and does not overwrite assist Ollama preferences', async () => {
  const root = await mountChat()
  const provider = find(root, n => n.tag === 'select' && n.children.some(c => c.props.value === 'lmstudio'))!
  provider.props['onUpdate:modelValue']('lmstudio'); provider.props.onChange()
  await nextTick()
  const requests = bridge.action.mock.calls.filter(call => call[0] === 'chat_models')
  const req = requests[requests.length - 1]![1]
  bridge.handlers.get('ollamaModelsReady')!('["wrong-ollama"]')
  bridge.handlers.get('chatModelsReady')!(JSON.stringify({ id: 'stale', ok: true, models: ['wrong-lm'] }))
  bridge.handlers.get('chatModelsReady')!(JSON.stringify({ id: req.id, ok: true, models: ['lm-fixture'] }))
  await nextTick()
  expect(root.textContent).toContain('lm-fixture')
  expect(root.textContent).not.toContain('wrong-lm')
  expect(localStorage.getItem('ollamaModel')).toBe('selected-model')
  expect(localStorage.getItem('chatLmStudioModel')).toBe('lm-fixture')
  expect(bridge.action.mock.calls.filter(call => call[0] === 'save_ui_prefs').every(call => !('ollamaModel' in call[1]))).toBe(true)
})

it('named presets save explicitly, apply separately and delete only after confirmation', async () => {
  let items: any[] = []
  const apply = vi.fn()
  const reply = (preset?: any) => JSON.stringify({ ok: true, scope: 'chat', presets: items, preset })
  bridge.backend = {
    getInstructionPresets: (_scope: string, cb: Function) => cb(reply()),
    saveInstructionPreset: vi.fn((raw: string, cb: Function) => {
      const value = { ...JSON.parse(raw), id: 'one' }; items = [value]; cb(reply(value))
    }),
    deleteInstructionPreset: vi.fn((_raw: string, cb: Function) => { items = []; cb(reply()) }),
  }
  const root = new Node('root')
  app = renderer.createApp(InstructionPresets, { scope: 'chat', instructions: '내 지침', onApply: apply })
  app.mount(root); await nextTick()
  find(root, n => n.props['aria-label'] === '대화 지침 새 프리셋 이름')!.props['onUpdate:modelValue']('태그 작업')
  await nextTick()
  await button(root, '새 프리셋 저장').props.onClick(); await nextTick()
  expect(apply).not.toHaveBeenCalled()
  expect(JSON.parse(bridge.backend.saveInstructionPreset.mock.calls[0][0])).toMatchObject({ scope: 'chat', instructions: '내 지침', name: '태그 작업' })
  button(root, '불러오기').props.onClick()
  expect(apply).toHaveBeenCalledWith('내 지침')
  button(root, '삭제').props.onClick(); await nextTick()
  expect(bridge.backend.deleteInstructionPreset).not.toHaveBeenCalled()
  await button(root, '삭제 확인').props.onClick(); await nextTick()
  expect(bridge.backend.deleteInstructionPreset).toHaveBeenCalledTimes(1)
  expect(apply).toHaveBeenCalledTimes(1)
  expect(root.textContent).toContain('현재 지침은 유지됩니다')
})
