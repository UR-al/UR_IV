/**
 * 테스트 전용 — DOM 없이(vitest environment: 'node') Vue 컴포넌트를 마운트하는 작은 렌더러.
 *
 * 요소는 `FakeNode` 이고, 속성·이벤트 핸들러는 `props` 에 그대로 담긴다(`props.onClick()` 으로 누른다).
 * runtime-dom 의 v-model 지시자가 만지는 필드(value · checked · options)와 focus · addEventListener 를
 * 흉내만 낸다. 런타임 코드는 이 파일을 import 하지 않는다.
 */
import { createRenderer, h, type App, type Component } from 'vue'

export class FakeNode {
  children: FakeNode[] = []
  parent: FakeNode | null = null
  props: Record<string, any> = {}
  listeners: Record<string, Array<(e: any) => void>> = {}
  value: any = ''
  checked = false
  multiple = false
  options: any[] = []
  style: Record<string, string> = {}
  focused = false
  constructor(public tag: string, public text = '') {}
  get textContent(): string { return this.text + this.children.map(child => child.textContent).join('') }
  focus() { this.focused = true }
  blur() { this.focused = false }
  addEventListener(name: string, cb: (e: any) => void) { (this.listeners[name] ||= []).push(cb) }
  removeEventListener(name: string, cb: (e: any) => void) {
    this.listeners[name] = (this.listeners[name] || []).filter(fn => fn !== cb)
  }
  setAttribute(name: string, value: string) { this.props[name] = value }
  removeAttribute(name: string) { delete this.props[name] }
}

function insert(node: FakeNode, parent: FakeNode, anchor: FakeNode | null = null) {
  if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1)
  node.parent = parent
  const index = anchor ? parent.children.indexOf(anchor) : -1
  parent.children.splice(index < 0 ? parent.children.length : index, 0, node)
}

export const fakeRenderer = createRenderer<FakeNode, FakeNode>({
  createElement: tag => new FakeNode(tag),
  createText: text => new FakeNode('#text', text),
  createComment: () => new FakeNode('#comment'),
  insert,
  remove: node => { if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1) },
  setText: (node, text) => { node.text = text },
  setElementText: (node, text) => { node.text = text; node.children = [] },
  parentNode: node => node.parent,
  nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] ?? null,
  patchProp: (node, key, _old, value) => { node.props[key] = value },
  setScopeId: () => {},
  insertStaticContent: (text, parent, anchor) => {
    const node = new FakeNode('#static', text)
    insert(node, parent, anchor)
    return [node, node]
  },
})

export function findAll(node: FakeNode, match: (candidate: FakeNode) => boolean, out: FakeNode[] = []): FakeNode[] {
  if (match(node)) out.push(node)
  for (const child of node.children) findAll(child, match, out)
  return out
}

export function byClass(root: FakeNode, cls: string): FakeNode[] {
  return findAll(root, node => String(node.props.class || '').split(/\s+/).includes(cls))
}

export function byTag(root: FakeNode, tag: string): FakeNode[] {
  return findAll(root, node => node.tag === tag)
}

export interface Mounted {
  app: App<FakeNode>
  root: FakeNode
  unmount(): void
}

/**
 * `component` 을 가짜 루트에 마운트한다. `Icon` 처럼 main.js 가 전역 등록하는 컴포넌트는
 * 빈 span 으로 대신 등록한다. `stubs` 로 더 넘길 수 있다.
 */
export function mountFake(
  component: Component,
  props: Record<string, any> | (() => Record<string, any>) = {},
  stubs: Record<string, Component> = {},
): Mounted {
  const root = new FakeNode('root')
  const app = fakeRenderer.createApp({
    setup: () => () => h(component, typeof props === 'function' ? props() : props),
  })
  app.component('Icon', { render: () => h('span') })
  for (const [name, stub] of Object.entries(stubs)) app.component(name, stub)
  app.mount(root)
  return { app, root, unmount: () => app.unmount() }
}
