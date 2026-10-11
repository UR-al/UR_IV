import { afterEach, beforeAll, afterAll, describe, expect, it } from 'vitest'
import * as Vue from 'vue'
import { createRenderer, h, nextTick, ref, type App } from 'vue'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'
import source from './TagBlockField.vue?raw'
import * as imeComposition from '../utils/imeComposition'
import * as promptComments from '../utils/promptComments'
import { rewriteExcludeRules, splitExcludeRules } from '../utils/excludeRules'

/**
 * 블록 칸의 join prop — 제외 규칙 칸(블록 모드)에서 블록을 지우고·고치고·더하고·옮겨도
 * 한 줄에 한 규칙·카테고리별 줄 배치가 쉼표 한 줄로 뭉개지지 않는다(utils/excludeRules.rewriteExcludeRules).
 * join 이 없는 다른 칸(프롬프트 태그)은 예전처럼 ', ' 로 잇는다.
 */

// 자동완성(브리지)은 대부분 이 테스트의 관심사가 아니다 — 기본은 후보 없음 대역(요청한 질의만 적어 둔다).
// 후보 수락 표기를 볼 때만 suggestions 에 후보를 넣는다(테스트가 끝나면 비운다).
const requested: string[] = []
const suggestions: { tag: string }[] = []
const autocomplete = {
  useTagAutocomplete: () => ({
    items: ref([...suggestions]), index: ref(0), queryHangul: ref(false),
    close: () => {}, request: (query: string) => { requested.push(query) }, move: () => {},
    selected: () => suggestions[0],
  }),
}

const script = compileScript(parse(source).descriptor, { id: 'tag-block-field-join-test', inlineTemplate: true })
const code = ts.transpileModule(script.content, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText
const compiled: { default?: any } = {}
new Function('require', 'exports', code)((name: string) => {
  if (name === 'vue') return Vue
  if (name === '../composables/useTagAutocomplete') return autocomplete
  if (name === '../utils/imeComposition') return imeComposition
  if (name === '../utils/promptComments') return promptComments
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
  focus() {}
  getRootNode(): Node { return this.parent ? this.parent.getRootNode() : this }
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
const hasClass = (name: string) => (node: Node) => String(node.props.class ?? '').split(/\s+/).includes(name)

// v-model(input) 은 값이 바뀌면 `el.getRootNode() instanceof Document|ShadowRoot` 로 포커스를 본다 —
// DOM 없이 도는 테스트용 빈 대역(우리 Node 는 어느 쪽도 아니라 값이 그대로 들어간다)
const DOM_GLOBALS = ['Document', 'ShadowRoot'] as const
const missing = DOM_GLOBALS.filter(name => !(name in globalThis))
beforeAll(() => { for (const name of missing) (globalThis as any)[name] = class {} })
afterAll(() => { for (const name of missing) delete (globalThis as any)[name] })

/** 같은 이벤트에 수식어 핸들러가 여럿이면(@keydown.enter + @keydown.escape) 배열로 붙는다 */
function fire(node: Node, prop: string, e: unknown) {
  const handler = node.props[prop]
  for (const fn of Array.isArray(handler) ? handler : [handler]) fn(e)
}

let app: App | undefined
afterEach(() => { app?.unmount(); app = undefined; suggestions.length = 0 })

const event = (extra: Record<string, unknown> = {}) => ({
  preventDefault() {}, stopPropagation() {}, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, ...extra,
})
const enter = () => event({ key: 'Enter', keyCode: 13, isComposing: false })

function mount(initial: string, withJoin = true, excludeField = true) {
  const model = ref(initial)
  const root = new Node('root')
  app = renderer.createApp({
    setup: () => () => h(compiled.default, {
      modelValue: model.value,
      'onUpdate:modelValue': (value: string) => { model.value = value },
      split: excludeField ? splitExcludeRules : undefined,
      join: excludeField && withJoin ? rewriteExcludeRules : undefined,
    }),
  })
  app.component('Icon', { render: () => h('span') })
  app.mount(root)
  const blocks = () => findAll(root, hasClass('tbf-block'))
  const texts = () => blocks().map(b => b.textContent)
  const typeInto = (input: Node, text: string) => { input.value = text; input.props['onUpdate:modelValue'](text) }
  return { model, root, blocks, texts, typeInto }
}

const PASTED = 'watermark, *qr code\nborder, *2koma'

describe('TagBlockField join prop (제외 규칙 칸)', () => {
  it('블록 삭제(우클릭) — 리뷰 사례: 두 줄이 한 줄로 뭉개지지 않는다', async () => {
    const { model, blocks, texts } = mount(PASTED)
    expect(texts()).toEqual(['watermark', '*qr code', 'border', '*2koma'])
    blocks()[2].props.onContextmenu(event())
    await nextTick()
    expect(model.value).toBe('watermark, *qr code\n*2koma')
    expect(texts()).toEqual(['watermark', '*qr code', '*2koma'])
  })

  it('블록 편집 — 그 규칙 자리만 바뀐다', async () => {
    const { model, root, blocks, typeInto } = mount(PASTED)
    blocks()[2].props.onClick(event())
    await nextTick()
    const [edit] = findAll(root, hasClass('tbf-edit'))
    typeInto(edit, 'frame')
    fire(edit, 'onKeydown', enter())
    await nextTick()
    expect(model.value).toBe('watermark, *qr code\nframe, *2koma')
  })

  it('블록 추가 — 마지막 규칙 뒤에 잇고 꼬리 줄바꿈은 둔다', async () => {
    const { model, root, typeInto } = mount(PASTED + '\n')
    const [add] = findAll(root, hasClass('tbf-add'))
    typeInto(add, '~solo')
    fire(add, 'onKeydown', enter())
    await nextTick()
    expect(model.value).toBe('watermark, *qr code\nborder, *2koma, ~solo\n')
  })

  it('끌어 옮기기 — 구분자 자리는 그대로, 규칙만 옮겨진다', async () => {
    const { model, root, blocks, texts } = mount('a, b\nc, d')
    const [area] = findAll(root, hasClass('tbf-blocks'))
    // 한 줄에 네 블록(폭 100) — 맨 오른쪽 너머로 끌어 놓으면 끝으로 간다
    const rects = [0, 1, 2, 3].map(i => ({ getBoundingClientRect: () => ({ top: 0, bottom: 20, left: i * 100, width: 100 }) }))
    blocks()[0].props.onDragstart(event())
    area.props.onDragover(event({ currentTarget: { querySelectorAll: () => rects }, clientX: 390, clientY: 10 }))
    area.props.onDrop(event())
    await nextTick()
    expect(model.value).toBe('b, c\nd, a')
    expect(texts()).toEqual(['b', 'c', 'd', 'a'])
  })

  it('모두 비우기 — 빈 칸', async () => {
    const { model, root } = mount(PASTED)
    const [clear] = findAll(root, hasClass('tbf-clear-all'))
    clear.props.onClick(event())
    await nextTick()
    expect(model.value).toBe('')
  })

  it('join 이 없으면 예전처럼 ", " 로 잇는다 (프롬프트 태그 칸)', async () => {
    const { model, blocks } = mount(PASTED, false)
    blocks()[2].props.onContextmenu(event())
    await nextTick()
    expect(model.value).toBe('watermark, *qr code, *2koma')
  })
})

/**
 * 프롬프트 칸(split·join 없음)의 주석(utils/promptComments) — 블록을 지우고·고치고·더하고·옮겨도 줄 주석 뒤
 * 줄바꿈이 남아 다음 블록이 주석에 먹히지 않는다. 주석만 있는 블록은 흐리게(comment), 닫히지 않은 /* 는 open.
 */
describe('TagBlockField 프롬프트 주석', () => {
  const COMMENTED = 'masterpiece, // quality\nbest quality, 1girl /* a, b */ solo'

  it('줄 주석 뒤 줄바꿈이 블록 경계이고, 주석 안 쉼표로 나누지 않는다', () => {
    const { texts, blocks } = mount(COMMENTED, false, false)
    expect(texts()).toEqual(['masterpiece', '// quality', 'best quality', '1girl /* a, b */ solo'])
    const classes = blocks().map(b => String(b.props.class))
    expect(classes[1]).toContain('comment')
    expect(classes[0]).not.toContain('comment')
    expect(classes[3]).not.toContain('comment')   // 주석이 섞였을 뿐 태그가 있다
  })

  it('자동완성 수락 — 밑줄→공백이 주석을 여는 태그는 밑줄 그대로', async () => {
    const zato = 'z.a.t.o._//_i_love_the_world_and_everything_in_it'
    suggestions.push({ tag: zato })
    const { model, root } = mount('1girl', false, false)
    const [add] = findAll(root, hasClass('tbf-add'))
    fire(add, 'onKeydown', event({ key: 'Tab' }))
    await nextTick()
    expect(model.value).toBe(`1girl, ${zato}`)
    expect(promptComments.hasPromptComment(model.value)).toBe(false)
    app?.unmount(); app = undefined
    suggestions.splice(0, 1, { tag: 'long_hair' })
    const second = mount('1girl', false, false)
    fire(findAll(second.root, hasClass('tbf-add'))[0]!, 'onKeydown', event({ key: 'Tab' }))
    await nextTick()
    expect(second.model.value).toBe('1girl, long hair')   // 보통 태그는 예전처럼 공백
  })

  it('블록 삭제 — 다음 블록이 줄 주석에 먹히지 않는다', async () => {
    const { model, blocks } = mount('a // first\nb, c', false, false)
    blocks()[1].props.onContextmenu(event())
    await nextTick()
    expect(model.value).toBe('a // first\nc')
    expect(promptComments.stripPromptComments(model.value)).toBe('a,\nc')   // 블록 경계 = 보내는 쉼표
  })

  it('블록 편집·추가 — 줄 주석으로 끝나는 블록 뒤는 줄바꿈', async () => {
    const { model, root, blocks, typeInto } = mount('a, b', false, false)
    blocks()[1].props.onClick(event())
    await nextTick()
    const [edit] = findAll(root, hasClass('tbf-edit'))
    typeInto(edit, 'b // 메모')
    fire(edit, 'onKeydown', enter())
    await nextTick()
    expect(model.value).toBe('a, b // 메모')
    const [add] = findAll(root, hasClass('tbf-add'))
    typeInto(add, 'c')
    fire(add, 'onKeydown', enter())
    await nextTick()
    expect(model.value).toBe('a, b // 메모\nc')
  })

  it('끌어 옮기기 — 주석 블록을 끝으로 옮겨도 태그를 삼키지 않는다', async () => {
    const { model, root, blocks } = mount('x // note\ny, z', false, false)
    const [area] = findAll(root, hasClass('tbf-blocks'))
    const rects = [0, 1, 2].map(i => ({ getBoundingClientRect: () => ({ top: 0, bottom: 20, left: i * 100, width: 100 }) }))
    blocks()[0].props.onDragstart(event())
    area.props.onDragover(event({ currentTarget: { querySelectorAll: () => rects }, clientX: 290, clientY: 10 }))
    area.props.onDrop(event())
    await nextTick()
    expect(model.value).toBe('y, z, x // note')
  })

  it('닫히지 않은 /* 는 open 표시, 주석을 쓰는 중에는 자동완성을 요청하지 않는다', () => {
    const { blocks, root } = mount('a /* open', false, false)
    expect(String(blocks()[0].props.class)).toContain('open')
    const [add] = findAll(root, hasClass('tbf-add'))
    requested.length = 0
    add.props.onInput(event({ target: { value: 'blue // why' } }))
    add.props.onInput(event({ target: { value: 'blue' } }))
    expect(requested).toEqual(['blue'])
  })

  it('제외 규칙 칸(split 있음)은 주석 문법이 없다', () => {
    const { texts, blocks, root } = mount('a, // b', true, true)
    expect(texts()).toEqual(['a', '// b'])
    expect(String(blocks()[1].props.class)).not.toContain('comment')
    const [add] = findAll(root, hasClass('tbf-add'))
    requested.length = 0
    add.props.onInput(event({ target: { value: '// x' } }))
    expect(requested).toEqual(['// x'])
  })
})
