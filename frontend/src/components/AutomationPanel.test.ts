import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, reactive, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './AutomationPanel.vue?raw'
import * as promptEdit from '../utils/automationPromptEdit'
import * as imeComposition from '../utils/imeComposition'

// 색 분류(classifyTags)는 이 테스트의 관심사가 아니다 — 백엔드 없음으로 둔다.
const bridge = { getBackend: async () => null }

const script = compileScript(parse(source).descriptor, { id: 'automation-panel-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
const stub = { __esModule: true, default: { render: () => null } }
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../bridge.js') return bridge
  if (name === '../utils/automationPromptEdit') return promptEdit
  if (name === '../utils/imeComposition') return imeComposition
  if (name === './CustomSelect.vue' || name === './ToggleSwitch.vue') return stub
  throw Error(`Unexpected component dependency: ${name}`)
}, compiled)

class Node {
  children: Node[] = []
  parent: Node | null = null
  props: Record<string, any> = {}
  value = ''
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

function findAll(node: Node, match: (candidate: Node) => boolean, out: Node[] = []): Node[] {
  if (match(node)) out.push(node)
  for (const child of node.children) findAll(child, match, out)
  return out
}

let app: App | undefined
beforeEach(() => { vi.useFakeTimers() })
afterEach(() => { app?.unmount(); app = undefined; vi.useRealTimers() })

function mount(nextPrompt: string) {
  const sent: string[] = []
  const state = reactive({
    settings: { mode: 'count', limit: 10, delay: 1, repeat: 1, maxRetries: 2, allowDupes: false, autoResetDeck: false, cleanupEveryN: 0 },
    running: true, paused: false, count: 0, waiting: true, waitRemainingMs: 0, waitTotalMs: 0,
    deckTotal: 0, deckRemaining: 0, deckUsed: 0, deckAllowDup: false, nextPrompt, promptIsNext: true,
  })
  const root = new Node('root')
  app = renderer.createApp({ setup: () => () => h(compiled.default, { ...state, onOverride: (text: string) => sent.push(text) }) })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  const tags = () => findAll(root, node => node.tag === 'button' && String(node.props.class || '').includes('ap-tag'))
  return { state, sent, tags, root }
}

it('keeps the edit base when the backend echoes the override back', async () => {
  const { state, sent, tags } = mount('a, b, c')
  await nextTick()
  tags()[1].props.onClick()                 // b 를 뺀다
  vi.advanceTimersByTime(400)
  expect(sent).toEqual(['a, c'])

  state.nextPrompt = 'a, c'                 // 백엔드 메아리(automationStatus.prompt)
  await nextTick()
  const shown = tags()
  expect(shown.map(t => t.textContent)).toEqual(['a', 'b', 'c'])     // 기준은 원래 프롬프트
  expect(shown.map(t => String(t.props.class).includes('off'))).toEqual([false, true, false])

  shown[0].props.onClick()                  // a 도 뺀다 → 'c' 만 나가야 한다
  vi.advanceTimersByTime(400)
  expect(sent).toEqual(['a, c', 'c'])       // 예전: 메아리를 기준으로 인덱스를 다시 적용해 엉뚱한 문자열
})

it('a new deck prompt resets the strike-throughs and the base', async () => {
  const { state, sent, tags } = mount('a, b')
  await nextTick()
  tags()[0].props.onClick()
  vi.advanceTimersByTime(400)
  state.nextPrompt = 'b'                    // 메아리
  await nextTick()
  state.nextPrompt = 'x, y, z'              // 다음 장의 새 프롬프트
  await nextTick()
  expect(tags().map(t => t.textContent)).toEqual(['x', 'y', 'z'])
  expect(tags().some(t => String(t.props.class).includes('off'))).toBe(false)
  expect(sent).toEqual(['b'])
})

it('keeps a pending repeat edit when the image count moves on with the echo', async () => {
  // repeat≥2: 생성 도중 건 편집은 다음 반복 장용으로 남는다 — 장이 끝나 count 가 올라도 백엔드는
  // 그 덮어쓰기를 그대로 되돌려 준다. 예전엔 count 감시가 초기화해 기준이 편집본이 됐고,
  // 태그 하나를 껐다 켜면 '' 가 나가 덮어쓰기가 취소되며 뺀 b 가 몰래 돌아왔다.
  const { state, sent, tags } = mount('a, b, c')
  await nextTick()
  tags()[1].props.onClick()                 // b 를 뺀다
  vi.advanceTimersByTime(400)
  state.nextPrompt = 'a, c'                 // 메아리
  await nextTick()
  state.count = 1                           // 이 장이 끝났다 — 덮어쓰기는 아직 안 쓰였다
  await nextTick()
  const shown = tags()
  expect(shown.map(t => t.textContent)).toEqual(['a', 'b', 'c'])
  expect(shown.map(t => String(t.props.class).includes('off'))).toEqual([false, true, false])

  shown[2].props.onClick()                  // c 를 껐다가
  vi.advanceTimersByTime(400)
  tags()[2].props.onClick()                 // 다시 켠다
  vi.advanceTimersByTime(400)
  expect(sent).toEqual(['a, c', 'a', 'a, c'])   // '' (덮어쓰기 취소)가 아니다

  state.nextPrompt = 'a, b, c'              // 덮어쓰기를 쓴 장이 끝나 원래 프롬프트가 돌아왔다
  state.count = 2
  await nextTick()
  expect(tags().some(t => String(t.props.class).includes('off'))).toBe(false)
})

it('does not take edits while the shown prompt is the one being generated', async () => {
  const { state, sent, tags, root } = mount('a, b, c')
  state.promptIsNext = false                // 마지막 반복 장이 생성 중 — 다음 장은 새로 뽑는다
  await nextTick()
  tags()[1].props.onClick()
  vi.advanceTimersByTime(400)
  expect(sent).toEqual([])
  expect(tags().every(t => t.props.disabled === true)).toBe(true)
  expect(root.textContent).toContain('지금 장 프롬프트')
  expect(findAll(root, n => n.tag === 'input')).toEqual([])   // 태그 추가 칸도 없다

  state.promptIsNext = true                 // 일시정지 → 다음 프롬프트를 뽑아 보여 준 뒤 섰다
  state.nextPrompt = 'x, y'
  await nextTick()
  expect(root.textContent).toContain('다음 프롬프트')
  tags()[0].props.onClick()
  vi.advanceTimersByTime(400)
  expect(sent).toEqual(['y'])
})

it('drops an edit that generation overtook before it was sent, and says so', async () => {
  const { state, sent, tags, root } = mount('a, b, c')
  await nextTick()
  tags()[1].props.onClick()                 // 모으는 400ms 안에
  state.promptIsNext = false                // 그 장 생성이 먼저 시작됐다
  await nextTick()
  vi.advanceTimersByTime(400)
  expect(sent).toEqual([])                  // 버려질 덮어쓰기를 보내지 않는다
  expect(tags().some(t => String(t.props.class).includes('off'))).toBe(false)
  expect(root.textContent).toContain('들어가지 않았다')

  state.promptIsNext = true
  state.nextPrompt = 'x, y'
  state.count = 1
  await nextTick()
  expect(root.textContent).not.toContain('들어가지 않았다')
})
