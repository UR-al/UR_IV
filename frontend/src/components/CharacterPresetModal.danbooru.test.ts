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
 * 캐릭터 프리셋 모달의 danbooru 조회(requestCharacterTagsOnline → characterTagsOnlineReady).
 *
 * 회귀: 조회가 도는 사이 모달을 닫으면(Esc·적용 → v-if 언마운트) onUnmounted 의 cancel() 이 시간
 * 초과와 똑같이 null 로 끝나, 사용자가 버린 조회에 대해 'danbooru 조회 실패: 응답 없음' 오류 토스트가
 * 떴다. 이제 취소는 조용히 끝나고, 진짜 시간 초과만 실패로 알린다.
 */

const script = compileScript(parse(source).descriptor, { id: 'character-preset-modal-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText

// v-model 지시자(runtime-dom)는 실제 DOM 이벤트·select.options 가 필요하다 — 이 테스트는 입력을 쓰지 않는다
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

let app: App | undefined
beforeEach(() => {
  vi.stubGlobal('window', { addEventListener: () => {}, removeEventListener: () => {} })
})
afterEach(() => {
  app?.unmount()
  app = undefined
  vi.useRealTimers()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function openAndStartLookup() {
  const lookups: Array<[string, string]> = []
  const listeners: Record<string, (json: string) => void> = {}
  const backend = {
    getCharGlobalPrefs: (cb: (json: string) => void) => cb('{}'),
    getWidgetValue: (_id: string, cb: (value: string) => void) => cb('hatsune miku'),
    searchCharacters: (_q: string, cb: (json: string) => void) =>
      cb(JSON.stringify([{ key: 'hatsune_miku', count: 5, hasPreset: false }])),
    getCharacterFeatures: (_key: string, cb: (json: string) => void) =>
      cb(JSON.stringify({ count: 5, core: [{ tag: 'aqua hair' }], costume: [], etc: [], aux: [], custom: [] })),
    requestCharacterTagsOnline: (name: string, id: string) => { lookups.push([name, id]) },
  }
  const bridge = {
    getBackend: async () => backend,
    onBackendEvent: (name: string, cb: (json: string) => void) => {
      listeners[name] = cb
      return () => { delete listeners[name] }
    },
  }
  const requestAction = vi.fn()
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compile(bridge, { requestAction })) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  await settle()                                           // 전역 설정 → 프롬프트 캐릭터로 검색
  byClass(root, 'cpm-item')[0].props.onClick()             // 결과 선택 → 특징 로드
  await settle()
  byClass(root, 'db')[0].props.onClick()                   // 'danbooru' 버튼
  await settle()
  expect(lookups).toHaveLength(1)
  expect(lookups[0][0]).toBe('hatsune_miku')
  const toasts = () => requestAction.mock.calls.filter(([name]) => name === 'show_toast').map(([, payload]) => payload)
  return { root, lookups, listeners, toasts }
}

it('closing the modal mid-lookup does not report a failure for the abandoned lookup', async () => {
  const { listeners, toasts } = await openAndStartLookup()
  app!.unmount()                                            // Esc·적용 → 부모의 v-if 가 모달을 내린다
  app = undefined
  await settle()
  expect(toasts()).toEqual([])                              // 예전: 'danbooru 조회 실패: 응답 없음'
  expect(listeners.characterTagsOnlineReady).toBeUndefined()
})

it('a real timeout is still reported as no response', async () => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  const { toasts } = await openAndStartLookup()
  await vi.advanceTimersByTimeAsync(40_000)
  await settle()
  expect(toasts()).toEqual([{ type: 'error', msg: 'danbooru 조회 실패: 응답 없음' }])
})

it('the reply for the current lookup replaces the chips', async () => {
  const { root, lookups, listeners, toasts } = await openAndStartLookup()
  listeners.characterTagsOnlineReady(JSON.stringify({
    requestId: lookups[0][1], name: 'hatsune_miku', tags: ['twintails', 'aqua eyes'], sampled: 20,
  }))
  await settle()
  expect(toasts()).toEqual([{ type: 'success', msg: "danbooru 2개 태그 로드 — 검토 후 '프리셋 저장'" }])
  const text = root.textContent
  expect(text).toContain('twintails')
  expect(text).not.toContain('aqua hair')
})
