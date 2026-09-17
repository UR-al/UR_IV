import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, nextTick, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './AiAssistInstructionsSettings.vue?raw'
import * as hostBridge from '../bridge.js'

const bridge = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn(), connect: vi.fn(), handlers: new Set<(raw: string) => void>() }))
vi.mock('../bridge.js', () => ({ getBackend: () => bridge.connect(),
  onBackendEvent: (_: string, fn: (raw: string) => void) => { bridge.handlers.add(fn); return () => bridge.handlers.delete(fn) },
}))

const script = compileScript(parse(source).descriptor, { id: 'assist-settings-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const exports: { default?: any } = {}
new Function('require', 'exports', code)((id: string) => {
  if (id === 'vue') return Vue
  if (id === '../bridge.js') return hostBridge
  if (id === './InstructionPresets.vue') return { __esModule: true, default: { render: () => null } }
  throw Error(`Unknown component dependency: ${id}`)
}, exports)

// Exercise the rendered controls and host callback contract, without a DOM
// library, browser process, user settings or real Ollama requests.
class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  value = ''
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
  setText: (node, text) => { node.text = text },
  setElementText: (node, text) => { node.text = text; node.children = [] },
  parentNode: node => node.parent, nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
  patchProp: (node, key, _old, value) => { node.props[key] = value; if (key === 'value') node.value = value },
  setScopeId: () => {},
  insertStaticContent: (text, parent, anchor) => { const node = new Node('#static', text); insert(node, parent, anchor); return [node, node] },
})
function all(root: Node, predicate: (node: Node) => boolean): Node[] {
  return [...(predicate(root) ? [root] : []), ...root.children.flatMap(child => all(child, predicate))]
}
function control(root: Node, id: string) { return all(root, node => node.props.id === id)[0]! }
function button(root: Node, text: string) { return all(root, node => node.tag === 'button' && node.textContent.includes(text))[0]! }
function instructions(common = '') {
  return { common, features: { expand: '', suggest: '', nl2tags: '', nl_caption: '', nl_scene: '', translate: '', creative: '', negative: '', auto_nl: '' } }
}
let app: App | undefined
async function mount() {
  const root = new Node('root')
  app = renderer.createApp(exports.default)
  app.mount(root)
  await nextTick()
  return root
}
beforeEach(() => {
  vi.useFakeTimers()
  bridge.get.mockReset(); bridge.save.mockReset(); bridge.connect.mockReset()
  bridge.handlers.clear()
  bridge.connect.mockResolvedValue({ getAiAssistInstructions: bridge.get, saveAiAssistInstructions: bridge.save })
})
afterEach(() => { app?.unmount(); app = undefined; vi.clearAllTimers(); vi.useRealTimers() })

it('loads all ten fields and reports saved only after the explicit host acknowledgement', async () => {
  const root = await mount()
  expect(control(root, 'ai-assist-common').props.disabled).toBe(true)
  bridge.get.mock.calls[0]![0](JSON.stringify({ ok: true, instructions: instructions('keep common') }))
  await nextTick()
  expect(all(root, node => node.tag === 'textarea')).toHaveLength(10)
  expect(control(root, 'ai-assist-common').value).toBe('keep common')
  control(root, 'ai-assist-expand').props.onInput({ target: { value: 'visible details only' } })
  await nextTick()
  expect(root.textContent).toContain('저장하지 않은 변경사항')
  expect(bridge.save).not.toHaveBeenCalled()
  await button(root, '지침 저장').props.onClick()
  await nextTick()
  expect(control(root, 'ai-assist-expand').props.disabled).toBe(true)
  expect(root.textContent).not.toContain('지침을 저장했습니다')
  const payload = JSON.parse(bridge.save.mock.calls[0]![0])
  expect(payload.common).toBe('keep common')
  expect(payload.features.expand).toBe('visible details only')
  bridge.save.mock.calls[0]![1](JSON.stringify({ ok: true, instructions: payload }))
  await nextTick()
  expect(root.textContent).toContain('지침을 저장했습니다')
  expect(root.textContent).not.toContain('저장하지 않은 변경사항')
  expect(control(root, 'ai-assist-expand').props.disabled).toBe(false)
})

it('synchronizes clean editors but keeps dirty drafts until a conflict is explicitly resolved', async () => {
  bridge.get.mockImplementation(callback => callback(JSON.stringify({ ok: true, instructions: instructions('initial') })))
  const root = await mount()
  for (const handler of bridge.handlers) handler(JSON.stringify({ ok: true, instructions: instructions('remote one') }))
  await nextTick()
  expect(control(root, 'ai-assist-common').value).toBe('remote one')
  control(root, 'ai-assist-common').props.onInput({ target: { value: 'local draft' } })
  for (const handler of bridge.handlers) handler(JSON.stringify({ ok: true, instructions: instructions('remote two') }))
  await nextTick()
  expect(control(root, 'ai-assist-common').value).toBe('local draft')
  expect(root.textContent).toContain('다른 화면에서 지침을 저장했습니다')
  expect(button(root, '지침 저장').props.disabled).toBe(true)
  button(root, '새 저장본 불러오기').props.onClick()
  await nextTick()
  expect(control(root, 'ai-assist-common').value).toBe('remote two')
  expect(bridge.save).not.toHaveBeenCalled()
})

it('keeps failed saves dirty and allows a successful explicit retry', async () => {
  bridge.get.mockImplementation(callback => callback(JSON.stringify({ ok: true, instructions: instructions() })))
  const root = await mount()
  control(root, 'ai-assist-common').props.onInput({ target: { value: 'new instructions' } })
  await nextTick()
  await button(root, '지침 저장').props.onClick()
  await nextTick()
  bridge.save.mock.calls[0]![1](JSON.stringify({ ok: false, error: '저장 폴더에 쓸 수 없습니다' }))
  await nextTick()
  expect(root.textContent).toContain('저장 폴더에 쓸 수 없습니다')
  expect(root.textContent).toContain('저장하지 않은 변경사항')
  expect(control(root, 'ai-assist-common').value).toBe('new instructions')
  await button(root, '지침 저장').props.onClick()
  await nextTick()
  const [payload, callback] = bridge.save.mock.calls[1]!
  callback(JSON.stringify({ ok: true, instructions: JSON.parse(payload) }))
  await nextTick()
  expect(root.textContent).toContain('지침을 저장했습니다')
  expect(root.textContent).not.toContain('저장 폴더에 쓸 수 없습니다')
})

it('locks initial input and ignores an obsolete load response after timeout and retry', async () => {
  const root = await mount()
  control(root, 'ai-assist-common').props.onInput({ target: { value: 'must stay locked' } })
  expect(control(root, 'ai-assist-common').value).toBe('')
  const oldReply = bridge.get.mock.calls[0]![0]
  await vi.advanceTimersByTimeAsync(12000)
  expect(root.textContent).toContain('불러오는 시간이 초과')
  expect(control(root, 'ai-assist-common').props.disabled).toBe(true)
  await button(root, '연결 다시 시도').props.onClick()
  await nextTick()
  bridge.get.mock.calls[1]![0](JSON.stringify({ ok: true, instructions: instructions('newest saved') }))
  await nextTick()
  control(root, 'ai-assist-common').props.onInput({ target: { value: 'newer draft' } })
  await nextTick()
  oldReply(JSON.stringify({ ok: true, instructions: instructions('obsolete saved') }))
  await nextTick()
  expect(control(root, 'ai-assist-common').value).toBe('newer draft')
  expect(root.textContent).toContain('저장하지 않은 변경사항')
})

it('preserves edits after a save timeout and never lets late success overwrite them', async () => {
  bridge.get.mockImplementation(callback => callback(JSON.stringify({ ok: true, instructions: instructions() })))
  const root = await mount()
  control(root, 'ai-assist-nl_caption').props.onInput({ target: { value: 'first caption rule' } })
  await nextTick()
  await button(root, '지침 저장').props.onClick()
  await nextTick()
  const [payload, oldReply] = bridge.save.mock.calls[0]!
  control(root, 'ai-assist-nl_caption').props.onInput({ target: { value: 'blocked during save' } })
  await nextTick()
  expect(control(root, 'ai-assist-nl_caption').value).toBe('first caption rule')
  await vi.advanceTimersByTimeAsync(12000)
  expect(root.textContent).toContain('저장 응답을 확인하지 못했습니다')
  control(root, 'ai-assist-nl_caption').props.onInput({ target: { value: 'new caption rule' } })
  await nextTick()
  oldReply(JSON.stringify({ ok: true, instructions: JSON.parse(payload) }))
  await nextTick()
  expect(control(root, 'ai-assist-nl_caption').value).toBe('new caption rule')
  expect(root.textContent).toContain('저장하지 않은 변경사항')
  expect(root.textContent).not.toContain('지침을 저장했습니다')
})

it('shows malformed or unsupported connection errors without enabling blank overwrites', async () => {
  bridge.get.mockImplementation(callback => callback('{not-json'))
  const root = await mount()
  expect(all(root, node => node.props.role === 'alert').length).toBe(1)
  expect(control(root, 'ai-assist-common').props.disabled).toBe(true)
  expect(button(root, '지침 저장').props.disabled).toBe(true)
  bridge.connect.mockResolvedValue({})
  await button(root, '연결 다시 시도').props.onClick()
  await nextTick()
  expect(root.textContent).toContain('지원하지 않습니다')
  expect(bridge.save).not.toHaveBeenCalled()
})

it('enforces per-field limits and displays the special automatic-caption inheritance', async () => {
  bridge.get.mockImplementation(callback => callback(JSON.stringify({ ok: true, instructions: instructions() })))
  const root = await mount()
  expect(root.textContent).toContain('공통 → 자연어 캡션 → 생성 전 자동 자연어 변환')
  expect(root.textContent).toContain('여기서 작성한 지침은 Chat 대화와 Batch/Upscale 이미지 캡션에 적용되지 않습니다')
  expect(all(root, node => node.tag === 'textarea').every(node => node.props.maxlength === 16000)).toBe(true)
  control(root, 'ai-assist-negative').props.onInput({ target: { value: 'n'.repeat(8005) } })
  await nextTick()
  expect(control(root, 'ai-assist-negative').value.length).toBe(8000)
})

it('counts Unicode code points like the host and clamps the actual textarea without splitting emoji', async () => {
  const existing = '😀'.repeat(8000)
  bridge.get.mockImplementation(callback => callback(JSON.stringify({ ok: true, instructions: instructions(existing) })))
  const root = await mount()
  expect(control(root, 'ai-assist-common').value).toBe(existing)
  expect(root.textContent).not.toContain('16,000 / 8,000')
  const target = { value: 'a'.repeat(7999) + '😀x' }
  control(root, 'ai-assist-common').props.onInput({ target })
  await nextTick()
  expect(target.value).toBe('a'.repeat(7999) + '😀')
  expect([...control(root, 'ai-assist-common').value]).toHaveLength(8000)
  // The model may be unchanged at its limit; the native textarea still needs
  // its rejected extra character removed immediately.
  target.value += 'y'
  control(root, 'ai-assist-common').props.onInput({ target })
  await nextTick()
  expect(target.value).toBe('a'.repeat(7999) + '😀')
})

it('cleans up timeouts and ignores pending host callbacks after unmount', async () => {
  const root = await mount()
  const lateReply = bridge.get.mock.calls[0]![0]
  app!.unmount(); app = undefined
  expect(vi.getTimerCount()).toBe(0)
  lateReply(JSON.stringify({ ok: true, instructions: instructions('late private content') }))
  await vi.advanceTimersByTimeAsync(12000)
  expect(root.textContent).not.toContain('late private content')
  expect(bridge.save).not.toHaveBeenCalled()
})

it('times out a missing bridge connection and does not issue requests when it connects after unmount', async () => {
  let resolveConnection!: (value: unknown) => void
  bridge.connect.mockReturnValue(new Promise(resolve => { resolveConnection = resolve }))
  const root = await mount()
  await vi.advanceTimersByTimeAsync(12000)
  expect(root.textContent).toContain('불러오는 시간이 초과')
  expect(control(root, 'ai-assist-common').props.disabled).toBe(true)
  app!.unmount(); app = undefined
  resolveConnection({ getAiAssistInstructions: bridge.get, saveAiAssistInstructions: bridge.save })
  await nextTick()
  expect(bridge.get).not.toHaveBeenCalled()
  expect(bridge.save).not.toHaveBeenCalled()
})
