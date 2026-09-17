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
import * as schemaAutosave from '../composables/useSchemaAutosave'
import * as dropdownPlacement from '../utils/dropdownPlacement'

const bridge = vi.hoisted(() => ({
  handlers: new Map<string, (raw: string) => void>(),
  listeners: new Map<string, Set<(raw: string) => void>>(),
  action: vi.fn(),
  backend: null as any,
}))
vi.mock('../bridge.js', () => ({
  getBackend: async () => bridge.backend,
  onBackendEvent: (name: string, callback: (raw: string) => void) => {
    const listeners = bridge.listeners.get(name) || new Set()
    listeners.add(callback); bridge.listeners.set(name, listeners)
    bridge.handlers.set(name, raw => listeners.forEach(listener => listener(raw)))
    return () => { listeners.delete(callback); if (!listeners.size) { bridge.listeners.delete(name); bridge.handlers.delete(name) } }
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
  '../composables/useSchemaAutosave': schemaAutosave,
  '../components/CustomSelect.vue': { __esModule: true, default: CustomSelect },
  '../utils/chatStructuredOutput': chatStructuredOutput,
  '../components/AiAssistInstructionsSettings.vue': { __esModule: true, default: { render: () => null } },
  '../components/InstructionPresets.vue': { __esModule: true, default: InstructionPresets },
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
  bridge.listeners.clear()
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
  find(root, n => n.props['aria-label'] === '대화 지침 프리셋 내용')!.props['onUpdate:modelValue']('프리셋만의 다른 지침')
  await nextTick()
  await button(root, '새 프리셋 저장').props.onClick(); await nextTick()
  expect(apply).not.toHaveBeenCalled()
  expect(JSON.parse(bridge.backend.saveInstructionPreset.mock.calls[0][0])).toMatchObject({ scope: 'chat', instructions: '프리셋만의 다른 지침', name: '태그 작업' })
  button(root, '불러오기').props.onClick()
  expect(apply).toHaveBeenCalledWith('프리셋만의 다른 지침')
  button(root, '삭제').props.onClick(); await nextTick()
  expect(bridge.backend.deleteInstructionPreset).not.toHaveBeenCalled()
  await button(root, '삭제 확인').props.onClick(); await nextTick()
  expect(bridge.backend.deleteInstructionPreset).toHaveBeenCalledTimes(1)
  expect(apply).toHaveBeenCalledTimes(1)
  expect(root.textContent).toContain('현재 지침은 유지됩니다')
})

function presetBackend() {
  let items: any[] = []
  const reply = (scope: string, preset?: any) => JSON.stringify({ ok: true, scope, presets: items.filter(item => item.scope === scope), preset })
  return {
    getInstructionPresets: vi.fn((scope: string, cb: Function) => cb(reply(scope))),
    saveInstructionPreset: vi.fn((raw: string, cb: Function) => {
      const value = JSON.parse(raw); value.id ||= `preset-${items.length + 1}`
      items = [...items.filter(item => item.id !== value.id), value]
      bridge.handlers.get('instructionPresetsChanged')?.(reply(value.scope, value))
      cb(reply(value.scope, value))
    }),
    deleteInstructionPreset: vi.fn((raw: string, cb: Function) => {
      const value = JSON.parse(raw); items = items.filter(item => item.id !== value.id)
      bridge.handlers.get('instructionPresetsChanged')?.(reply(value.scope))
      cb(reply(value.scope))
    }),
    saveChatSchemaDraft: vi.fn((schemaText: string, cb: Function) => cb(JSON.stringify({ ok: true, schemaText }))),
  }
}

it('refreshes the top preset list after save, applies explicitly, and renames/deletes the same preset', async () => {
  bridge.backend = presetBackend()
  const root = await mountChat()
  const section = find(root, n => n.props['aria-label'] === '대화 지침 사용자 프리셋')!
  const prompt = find(root, n => n.props.id === 'chat-system-prompt')!
  const original = prompt.value
  const name = find(section, n => n.tag === 'input')!
  const contents = find(section, n => n.tag === 'textarea')!
  name.props['onUpdate:modelValue']('내 태그 지침')
  contents.props['onUpdate:modelValue']('테스트 전용 새 내용')
  await nextTick()
  await button(section, '새 프리셋 저장').props.onClick(); await nextTick()
  const select = find(root, n => n.props['aria-label'] === '지침 프리셋')!
  expect(select.textContent).toContain('내 태그 지침 · 내 프리셋')
  expect(find(section, n => n.tag === 'select')).toBeUndefined()
  expect(prompt.value).toBe(original)
  button(root, '선택한 지침 적용').props.onClick(); await nextTick()
  expect(prompt.value).toBe('테스트 전용 새 내용')
  name.props['onUpdate:modelValue']('이름 수정')
  contents.props['onUpdate:modelValue']('수정된 프리셋 내용')
  await nextTick()
  button(section, '선택 프리셋 이름·내용 수정').props.onClick(); await nextTick()
  await button(section, '덮어쓰기 확인').props.onClick(); await nextTick()
  expect(select.textContent).toContain('이름 수정 · 내 프리셋')
  expect(select.textContent).not.toContain('내 태그 지침 · 내 프리셋')
  expect(prompt.value).toBe('테스트 전용 새 내용')
  const row = find(root, n => n.props.class === 'cm-preset-row')!
  await button(row, '프리셋 새로고침').props.onClick(); await nextTick()
  expect(bridge.backend.getInstructionPresets.mock.calls.filter((c: any) => c[0] === 'chat')).toHaveLength(2)
  button(section, '삭제').props.onClick(); await nextTick()
  await button(section, '삭제 확인').props.onClick(); await nextTick()
  expect(select.textContent).not.toContain('이름 수정 · 내 프리셋')
  expect(prompt.value).toBe('테스트 전용 새 내용')
})

it('autosaves unfinished schema without blur, restores named schema buttons, and keeps the toggle unchanged', async () => {
  bridge.backend = presetBackend()
  const root = await mountChat()
  const schema = find(root, n => n.props.id === 'chat-json-schema')!
  schema.props['onUpdate:modelValue']('{"type":')
  await vi.advanceTimersByTimeAsync(400)
  expect(bridge.backend.saveChatSchemaDraft.mock.calls[0][0]).toBe('{"type":')
  expect(root.textContent).toContain('입력 내용 자동 저장됨')
  expect(localStorage.getItem('chatJsonSchemaPending')).toBe('0')
  const valid = '{"type":"object","properties":{"tags":{"type":"string"}}}'
  schema.props['onUpdate:modelValue'](valid)
  const section = find(root, n => n.props['aria-label'] === '구조화된 출력 사용자 프리셋')!
  find(section, n => n.tag === 'input')!.props['onUpdate:modelValue']('태그 JSON')
  await nextTick()
  await button(section, '새 프리셋 저장').props.onClick(); await nextTick()
  expect(JSON.parse(bridge.backend.saveInstructionPreset.mock.calls[0][0])).toMatchObject({ scope: 'schema', name: '태그 JSON', instructions: valid })
  schema.props['onUpdate:modelValue']('{"type":"string"}')
  await nextTick()
  button(section, '태그 JSON').props.onClick(); await nextTick()
  expect(schema.value).toBe(valid)
  expect(button(section, '태그 JSON').props['aria-pressed']).toBe(true)
  await vi.advanceTimersByTimeAsync(400)
  const calls = bridge.backend.saveChatSchemaDraft.mock.calls
  expect(calls[calls.length - 1][0]).toBe(valid)
  expect(find(root, n => n.tag === 'input' && n.props.type === 'checkbox')!.props['onUpdate:modelValue']).toBeDefined()
  expect(localStorage.getItem('chatStructuredEnabled')).not.toBe('1')
  expect(find(root, n => n.props['aria-label'] === '지침 프리셋')!.textContent).not.toContain('태그 JSON')
})

it('recovers an unacknowledged schema draft even if a stale disk read arrives after the new save', async () => {
  localStorage.setItem('chatJsonSchema', '{recover me')
  localStorage.setItem('chatJsonSchemaPending', '1')
  let restore: (raw: string) => void = () => {}
  bridge.backend = { ...presetBackend(), getUiPrefs: (cb: typeof restore) => { restore = cb } }
  const root = await mountChat()
  await vi.advanceTimersByTimeAsync(400)
  expect(bridge.backend.saveChatSchemaDraft.mock.calls[0][0]).toBe('{recover me')
  restore(JSON.stringify({ chatSettingsV2: { schemaText: '{"type":"string"}', systemPrompt: '복구된 지침', provider: 'lmstudio' } }))
  await nextTick()
  expect(find(root, n => n.props.id === 'chat-json-schema')!.value).toBe('{recover me')
  expect(find(root, n => n.props.id === 'chat-system-prompt')!.value).toBe('복구된 지침')
})
