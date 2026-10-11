import { afterEach, beforeEach, expect, it, vi, type Mock } from 'vitest'
import { createRenderer, nextTick, reactive, ref, h, type App } from 'vue'
import * as Vue from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './RelatedTagsControl.vue?raw'
import panelSource from './PromptPanel.vue?raw'
import * as hostBridge from '../bridge.js'
import * as relatedUtils from '../utils/relatedTags'
import { createRelatedTags, type RelatedTagsStore } from '../composables/useRelatedTags'

const host = vi.hoisted(() => ({ get: vi.fn(), off: vi.fn(), event: vi.fn() }))
vi.mock('../bridge.js', () => ({
  getBackend: async () => ({ getUiPrefs: host.get }),
  onBackendEvent: (name: string, callback: unknown) => { host.event(name, callback); return host.off },
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: vi.fn(), useWidgetStore: () => ({ widgets: {} }) }))

// 브라우저 템플릿으로 컴파일해 호스트 중립 렌더러로 그린다(CompositionControl.test.ts 와 같은 방식) —
// Vite 의 node 변환은 SSR 전용 출력이라 이벤트 핸들러를 부를 수 없다.
const script = compileScript(parse(source).descriptor, { id: 'related-tags-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
let store: RelatedTagsStore
const compiled: { default?: any } = {}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../bridge.js') return hostBridge
  if (name === '../utils/relatedTags') return relatedUtils
  if (name === '../composables/useRelatedTags') return { useRelatedTags: () => store }
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)
const RelatedTagsControl = compiled.default

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
function all(root: Node, predicate: (node: Node) => boolean): Node[] {
  return [...(predicate(root) ? [root] : []), ...root.children.flatMap(child => all(child, predicate))]
}
function button(root: Node, label: string) { return all(root, node => node.tag === 'button' && node.textContent.includes(label))[0] }
function chips(root: Node) { return all(root, node => node.tag === 'button' && node.props.class === 'related-chip') }
const flush = () => new Promise(resolve => setTimeout(resolve, 0))

let app: App | undefined
let actions: Mock<(...args: any[]) => void>
let prefs: Mock<(payload: Record<string, unknown>) => void>
let handlers: Map<string, (raw: unknown) => void>
async function mount(initial = 'armor', blockMode = true, otherPrompts = '') {
  handlers = new Map()
  actions = vi.fn()
  prefs = vi.fn()
  const widgets = reactive<Record<string, unknown>>({ main_prompt_text: initial })
  store = createRelatedTags({
    onBackendEvent: (name, cb) => { handlers.set(name, cb); return () => handlers.delete(name) },
    requestAction: actions, widgets, persistUiPrefs: prefs, newId: () => 'rtb-1',
  })
  const root = new Node('root')
  const prompt = ref(initial)
  const append = vi.fn((value: string) => { prompt.value = value; widgets.main_prompt_text = value })
  app = renderer.createApp({ setup: () => () => h(RelatedTagsControl, {
    modelValue: prompt.value, otherPrompts, blockMode, onAppend: append }) })
  app.mount(root)
  await flush()
  return { root, append, prompt, details: all(root, node => node.tag === 'details')[0]! }
}
function reply(extra: object = {}) {
  const query = [...actions.mock.calls].reverse().find(call => call[0] === 'related_tags_query')!
  handlers.get('relatedTagsEvent')!(JSON.stringify({
    action: 'related_tags_query', requestId: query[1].requestId, state: 'ready', tags: [], context: [],
    ratings: ['g', 's'], missing: [], stale: [], unavailable: [], estimateSec: 0, build: null, ...extra,
  }))
}
const TAGS = [
  { tag: 'holding_sword', ko: '검 들기', count: 1200, via: ['armor'] },
  { tag: 'cape', ko: '케이프', count: 900, via: ['armor'] },
]

beforeEach(() => {
  host.get.mockReset(); host.off.mockReset(); host.event.mockReset()
  host.get.mockImplementation(callback => callback('{}'))
})
afterEach(() => { app?.unmount(); app = undefined; store?.dispose() })

it('is a closed sibling panel with Korean copy, and asks only after opening', async () => {
  const { root, details } = await mount()
  expect(root.textContent).toContain('연관 태그')
  expect(root.textContent).toContain('함께 자주 쓰인 태그 추천')
  expect(root.textContent).toContain('누르면 메인 태그 끝에 추가합니다')
  expect(details.props.open).toBe(false)
  expect(actions).not.toHaveBeenCalled()
  details.props.onToggle({ target: { open: true } })
  await nextTick()
  expect(prefs).toHaveBeenCalledWith({ relatedTagsOpen: true })
  expect(actions).toHaveBeenCalledWith('related_tags_query', expect.objectContaining({
    fields: expect.objectContaining({ main: 'armor' }) }))
  expect(root.textContent).toContain('불러오는 중')
})

it('shows chips with Korean labels and appends a clicked tag in the current mode', async () => {
  const { root, details, append, prompt } = await mount('armor', true)
  details.props.onToggle({ target: { open: true } })
  reply({ tags: TAGS, context: ['armor'] })
  await flush()
  expect(root.textContent).toContain('등급: G · S (등급 필터를 따름) · 기준 태그: armor')
  expect(chips(root).map(chip => chip.textContent)).toEqual(['holding sword검 들기', 'cape케이프'])
  expect(chips(root)[0]!.props.title).toContain('게시물 1,200개')
  chips(root)[0]!.props.onClick()
  await nextTick()
  expect(append).toHaveBeenCalledWith('armor, holding sword')
  expect(prompt.value).toBe('armor, holding sword')
  expect(chips(root).map(chip => chip.textContent)).toEqual(['cape케이프'])
  expect(root.textContent).toContain('메인 태그에 추가했습니다: holding sword')
})

it('text mode inserts underscores and never duplicates a tag that is already present', async () => {
  const { root, details, append } = await mount('armor, cape', false, 'holding_sword')
  details.props.onToggle({ target: { open: true } })
  reply({ tags: [...TAGS, { tag: 'shoulder_armor', ko: '', count: 5, via: [] }] })
  await flush()
  chips(root)[0]!.props.onClick()          // holding_sword — 다른 칸(otherPrompts)에 있다
  await nextTick()
  chips(root)[0]!.props.onClick()          // cape — 메인에 있다
  await nextTick()
  expect(append).not.toHaveBeenCalled()
  expect(root.textContent).toContain('이미 프롬프트에 있는 태그입니다.')
  expect(chips(root).map(chip => chip.textContent)).toEqual(['shoulder armor'])
  chips(root)[0]!.props.onClick()
  expect(append).toHaveBeenCalledWith('armor, cape, shoulder_armor')
})

it('offers the first build with an estimate, shows progress and can cancel', async () => {
  const { root, details } = await mount()
  details.props.onToggle({ target: { open: true } })
  reply({ state: 'missing', missing: ['g', 's'], estimateSec: 140 })
  await flush()
  expect(root.textContent).toContain('처음 한 번 연관 태그 표를 만들어야 합니다 (로컬 데이터 · 약 3분, 그동안 앱은 그대로 쓸 수 있습니다).')
  button(root, '표 만들기')!.props.onClick()
  expect(actions).toHaveBeenLastCalledWith('related_tags_build', { requestId: 'rtb-1', shards: ['g', 's'] })
  handlers.get('relatedTagsEvent')!(JSON.stringify({ action: 'related_tags_build', requestId: 'rtb-1', state: 'building',
    progress: { shard: 's', index: 1, count: 2, progress: 0.42, overall: 0.71 } }))
  await nextTick()
  expect(root.textContent).toContain('표 만드는 중… S 42% (2/2)')
  expect(button(root, '표 만들기')).toBeUndefined()
  button(root, '취소')!.props.onClick()
  expect(actions).toHaveBeenLastCalledWith('related_tags_cancel', { requestId: 'rtb-1' })
  handlers.get('relatedTagsEvent')!(JSON.stringify({ action: 'related_tags_build', requestId: 'rtb-1', state: 'canceled', shards: [] }))
  await nextTick()
  expect(root.textContent).toContain('표 만들기를 취소했습니다.')
})

it('explains partial, stale, unavailable and empty results', async () => {
  const { root, details } = await mount()
  details.props.onToggle({ target: { open: true } })
  reply({ ratings: ['g', 'q'], missing: ['q'], stale: ['g'], tags: TAGS })
  await flush()
  expect(root.textContent).toContain('Q 표가 없어 G 만 반영했습니다.')
  expect(button(root, 'Q 표 만들기')).toBeDefined()
  expect(root.textContent).toContain('Danbooru 데이터가 바뀌었습니다 — 다시 만들면 더 정확합니다.')
  button(root, '다시 만들기')!.props.onClick()
  expect(actions).toHaveBeenLastCalledWith('related_tags_build', { requestId: 'rtb-1', shards: ['g'] })
  store.dispose()
  const second = await mount()
  second.details.props.onToggle({ target: { open: true } })
  reply({ state: 'unavailable', ratings: ['e'], unavailable: ['e'] })
  await flush()
  expect(second.root.textContent).toContain('Danbooru 데이터가 없어 연관 태그를 쓸 수 없습니다.')
  void store.refresh(true)
  reply({ context: [] })
  await flush()
  expect(second.root.textContent).toContain('기준이 될 태그가 없습니다. 메인 태그를 입력해 보세요.')
})

it('restores the saved open state and stops listening on unmount', async () => {
  host.get.mockImplementation(() => {})
  const { details } = await mount()
  const restore = host.event.mock.calls.find(call => call[0] === 'uiPrefsLoaded')![1]
  restore(JSON.stringify({ relatedTagsOpen: true }))
  await nextTick()
  expect(details.props.open).toBe(true)
  expect(prefs).not.toHaveBeenCalled()
  app!.unmount(); app = undefined
  expect(host.off).toHaveBeenCalledOnce()
})

it('is wired under the composition panel to the same main_prompt_text append path', () => {
  expect(panelSource).toContain('<RelatedTagsControl :model-value="widgets.main_prompt_text || \'\'"')
  expect(panelSource).toMatch(/<CompositionControl [^\n]*\/>\r?\n\s*<RelatedTagsControl [^\n]*:block-mode="tagBlockMode" @append="appendComposition" \/>/)
})
