import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './CharacterPresetModal.vue?raw'
import * as bridgeRequest from '../utils/bridgeRequest'
import * as charPresetState from '../utils/charPresetState'
import * as imeComposition from '../utils/imeComposition'
import * as modalLayer from '../composables/useModalLayer'

/**
 * 캐릭터 프리셋 모달의 per-캐릭터 작업 상태(cpmCharState.v2) 저장.
 *
 * 회귀: 특징 조회(getCharacterFeatures)가 실패하면 저장된 작업 상태를 칩에 올리지 못한 빈 화면이
 * 남는데, 그 화면에서 닫거나(onUnmounted) 무엇이든 편집하면(deep watch 자동 저장) diffCharState 가
 * 빈 커스텀 목록으로 diff 를 만들어 저장된 커스텀 태그(사용자 태그 · 끈 프리셋 태그)를 조용히 지웠다.
 * 이제 저장값을 칩에 올린 캐릭터에서만 작업 상태를 저장한다.
 */

const script = compileScript(parse(source).descriptor, { id: 'character-preset-modal-state-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText

// v-model 지시자(runtime-dom)는 실제 DOM 이 필요하다 — 입력은 onUpdate:modelValue 로 직접 넣는다
const noopDirective = {}
const vueForTest = { ...Vue, vModelText: noopDirective, vModelSelect: noopDirective, vModelCheckbox: noopDirective }

function compile(bridge: unknown, store: unknown) {
  const compiled: { default?: any } = {}
  new Function('require', 'exports', code)((name: string) => {
    if (name === 'vue') return vueForTest
    if (name === '../bridge.js') return bridge
    if (name === '../stores/widgetStore.js') return store
    if (name === '../utils/bridgeRequest') return bridgeRequest
    if (name === '../utils/charPresetState') return charPresetState
    if (name === '../utils/imeComposition') return imeComposition
    if (name === '../composables/useModalLayer') return modalLayer
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
  focus() {}
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
function findAll(node: Node, match: (candidate: Node) => boolean, out: Node[] = []): Node[] {
  if (match(node)) out.push(node)
  for (const child of node.children) findAll(child, match, out)
  return out
}
const byClass = (root: Node, cls: string) =>
  findAll(root, node => String(node.props.class || '').split(/\s+/).includes(cls))

async function settle() {
  for (let i = 0; i < 20; i++) await Promise.resolve()
  await nextTick()
}

function memStorage(init: Record<string, string> = {}) {
  const data: Record<string, string> = { ...init }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => { data[k] = v },
    removeItem: (k: string) => { delete data[k] },
  }
}

const CHAR = 'hatsune_miku'
const OK_REPLY = JSON.stringify({ count: 5, core: [{ tag: 'aqua hair' }], costume: [], etc: [], aux: [], custom: [] })
const ERROR_REPLY = JSON.stringify({ error: 'boom' })
const NOT_JSON_REPLY = '<<not json>>'
const STORED = JSON.stringify({
  [CHAR]: { checked: { 'aqua hair': false }, custom: [{ tag: 'my custom tag', checked: true }, { tag: 'preset off', checked: false }] },
})

let app: App | undefined
let storage: ReturnType<typeof memStorage>
beforeEach(() => {
  vi.stubGlobal('window', { addEventListener: () => {}, removeEventListener: () => {} })
  storage = memStorage()
  vi.stubGlobal('localStorage', storage)
})
afterEach(() => {
  app?.unmount()
  app = undefined
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

/** 모달을 열고 검색 결과의 캐릭터를 고른다 — getCharacterFeatures 는 replies 를 차례로 돌려준다. */
async function openAndSelect(replies: string[]) {
  const backend = {
    getCharGlobalPrefs: (cb: (json: string) => void) => cb('{}'),
    getWidgetValue: (_id: string, cb: (value: string) => void) => cb('hatsune miku'),
    searchCharacters: (_q: string, cb: (json: string) => void) =>
      cb(JSON.stringify([{ key: CHAR, count: 5, hasPreset: false }])),
    getCharacterFeatures: (_key: string, cb: (json: string) => void) => cb(replies.shift() ?? OK_REPLY),
  }
  const bridge = { getBackend: async () => backend, onBackendEvent: () => () => {} }
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compile(bridge, { requestAction: vi.fn() })) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await settle()
  const select = async () => { byClass(root, 'cpm-item')[0].props.onClick(); await settle() }
  await select()
  return { root, select }
}

async function close() {
  app!.unmount()                         // Esc·닫기·적용 → 부모의 v-if 가 모달을 내린다 → onUnmounted 저장
  app = undefined
  await settle()
}

const customChips = (root: Node) => byClass(root, 'cust').map(n => n.textContent.trim())
const stored = () => storage.data[charPresetState.CHAR_STATE_KEY]

it('a failed feature lookup followed by close keeps the stored working state byte for byte', async () => {
  storage.data[charPresetState.CHAR_STATE_KEY] = STORED
  const { root } = await openAndSelect([ERROR_REPLY])
  expect(byClass(root, 'cpm-foot-status')[0].textContent).toContain('특징 조회 실패')
  expect(customChips(root)).toEqual([])  // 저장값을 올리지 못한 빈 화면
  await close()
  expect(stored()).toBe(STORED)          // 예전: custom 이 비고, custom 만 있던 항목은 통째로 지워졌다
})

it('a reply that is not JSON is treated the same (nothing is overwritten)', async () => {
  storage.data[charPresetState.CHAR_STATE_KEY] = STORED
  await openAndSelect([NOT_JSON_REPLY])
  await close()
  expect(stored()).toBe(STORED)
})

it('editing after a failed lookup does not autosave over the stored customs', async () => {
  storage.data[charPresetState.CHAR_STATE_KEY] = STORED
  const { root } = await openAndSelect([ERROR_REPLY])
  byClass(root, 'cpm-addinput')[0].props['onUpdate:modelValue']('new tag')
  byClass(root, 'cpm-add')[0].props.onClick()
  await settle()
  expect(customChips(root)).toEqual(['new tag'])
  expect(stored()).toBe(STORED)          // 예전: 자동 저장이 {custom:[new tag]} 로 덮었다
  await close()
  expect(stored()).toBe(STORED)
})

it('customs migrated from the legacy key survive a later failed open and close', async () => {
  storage.data.cpmCharState = JSON.stringify({ [CHAR]: { checked: { 'aqua hair': true }, custom: [{ tag: 'mine', checked: true }] } })
  const first = await openAndSelect([OK_REPLY])
  expect(customChips(first.root)).toEqual(['mine'])
  await close()
  expect(storage.data.cpmCharState).toBeUndefined()   // 옛 키는 정리됐다
  const cleaned = stored()
  expect(JSON.parse(cleaned)[CHAR].custom).toEqual([{ tag: 'mine', checked: true }])

  await openAndSelect([ERROR_REPLY])
  await close()
  expect(stored()).toBe(cleaned)
})

it('a successful load still saves edits (autosave and save on close)', async () => {
  storage.data[charPresetState.CHAR_STATE_KEY] = STORED
  const { root } = await openAndSelect([OK_REPLY])
  expect(customChips(root)).toEqual(['my custom tag', 'preset off'])
  byClass(root, 'cust')[0].props.onClick()            // 'my custom tag' 끄기
  await settle()
  const expected = { checked: { 'aqua hair': false }, custom: [{ tag: 'my custom tag', checked: false }, { tag: 'preset off', checked: false }] }
  expect(JSON.parse(stored())[CHAR]).toEqual(expected)
  await close()
  expect(JSON.parse(stored())[CHAR]).toEqual(expected)
})

it('re-selecting after a failure restores the chips and saves later edits', async () => {
  storage.data[charPresetState.CHAR_STATE_KEY] = STORED
  const { root, select } = await openAndSelect([ERROR_REPLY, OK_REPLY])
  expect(customChips(root)).toEqual([])
  await select()
  expect(customChips(root)).toEqual(['my custom tag', 'preset off'])
  expect(byClass(root, 'cpm-foot-status')[0].textContent).not.toContain('특징 조회 실패')
  byClass(root, 'cpm-addinput')[0].props['onUpdate:modelValue']('new tag')
  byClass(root, 'cpm-add')[0].props.onClick()
  await settle()
  await close()
  expect(JSON.parse(stored())[CHAR].custom).toEqual([
    { tag: 'my custom tag', checked: true }, { tag: 'preset off', checked: false }, { tag: 'new tag', checked: true },
  ])
})
