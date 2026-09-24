import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as Vue from 'vue'
import { compileScript, parse } from '@vue/compiler-sfc'
import appSource from './App.vue?raw'
import condSource from './components/CondPromptModal.vue?raw'
import loraSource from './components/LoraManagerModal.vue?raw'
import abSource from './components/ABTestModal.vue?raw'
import charPresetSource from './components/CharacterPresetModal.vue?raw'
import charOverrideSource from './components/CharFeatureOverrideModal.vue?raw'
import * as modalLayer from './composables/useModalLayer'
import * as bridgeRequest from './utils/bridgeRequest'
import * as charPresetState from './utils/charPresetState'
import * as imeComposition from './utils/imeComposition'
import { appModalStack } from './utils/modalStack'
import { createAppKeydownHandler, type AppShortcutEvent } from './utils/appShortcuts'
import { compileSfc, stubComponentModule } from './testing/compileSfc'
import { mountFake, type Mounted } from './testing/fakeDomRenderer'

/**
 * 모달 뒤의 ↑/↓ 히스토리 이동 · ESC — App 과 모달 사이 배선의 회귀 가드(감사 #186).
 *
 * 버그: LoRA · 조건부 · A/B · 캐릭터 프리셋 · override 모달을 띄운 채 ↑/↓ 를 누르면 뒤의 히스토리가
 * 넘어갔다 — App 이 모달 플래그 목록을 따로 들고 있었는데 그 다섯이 빠져 있었다. 지금은 모달이 열릴 때
 * 스스로 앱 모달 스택에 오르고(composables/useModalLayer), App 의 전역 keydown(utils/appShortcuts)은
 * 스택에만 묻는다. 그러니 되살아나는 길은 둘이다 — 모달에서 `useModalLayer()` 한 줄이 빠지거나, App 의
 * keydown 이 스택이 아닌 다른 것을 보거나. 판단 로직 자체는 utils/appShortcuts.test.ts 가 지킨다.
 */

// ── 정적 가드 ────────────────────────────────────────────────────────────────────────────────

/** src 아래 모든 `*Modal.vue` — App 이 그리는 것 말고도 새로 생기는 모달까지. */
const MODAL_SOURCES = import.meta.glob<string>('./**/*Modal.vue', { query: '?raw', import: 'default', eager: true })

type AstNode = { type: string; [key: string]: unknown }

function isAstNode(value: unknown): value is AstNode {
  return !!value && typeof value === 'object' && typeof (value as { type?: unknown }).type === 'string'
}

/** babel AST 를 깊이 우선으로 훑는다(위치 정보 · 주석 필드는 건너뛴다). */
function walk(node: unknown, visit: (n: AstNode) => void) {
  if (Array.isArray(node)) { for (const child of node) walk(child, visit); return }
  if (!isAstNode(node)) return
  visit(node)
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc' || key === 'leadingComments' || key === 'trailingComments' || key === 'innerComments') continue
    if (value && typeof value === 'object') walk(value, visit)
  }
}

function setupAst(source: string, filename: string): AstNode[] {
  const { descriptor, errors } = parse(source, { filename })
  if (errors.length) throw new Error(`${filename}: ${errors.map(e => String(e)).join('; ')}`)
  if (!descriptor.scriptSetup) throw new Error(`${filename}: <script setup> 이 없다`)
  const script = compileScript(descriptor, { id: 'modal-layer-guard' })
  return (script.scriptSetupAst ?? []) as unknown as AstNode[]
}

/** `import { local } from 'source'` / `import local from 'source'` — 지역 이름 → 모듈 경로. */
function importsOf(ast: AstNode[]): Map<string, string> {
  const map = new Map<string, string>()
  for (const statement of ast) {
    if (statement.type !== 'ImportDeclaration') continue
    const source = (statement.source as { value: string }).value
    for (const spec of statement.specifiers as Array<{ local: { name: string } }>) map.set(spec.local.name, source)
  }
  return map
}

const isIdentifier = (node: unknown, name: string) =>
  isAstNode(node) && node.type === 'Identifier' && (node as { name?: string }).name === name

/** setup 최상위에서 조건 없이 부르는 `useModalLayer(...)` 문의 개수. */
function topLevelModalLayerCalls(ast: AstNode[]): number {
  return ast.filter(statement =>
    statement.type === 'ExpressionStatement'
    && isAstNode(statement.expression)
    && statement.expression.type === 'CallExpression'
    && isIdentifier(statement.expression.callee, 'useModalLayer'),
  ).length
}

/** App 템플릿이 그리는 `<XxxModal>` 태그 이름. */
function modalTagsInTemplate(source: string): string[] {
  const { descriptor } = parse(source, { filename: 'App.vue' })
  const tags = new Set<string>()
  const visit = (node: { type: number; tag?: string; tagType?: number; children?: unknown[] }) => {
    if (node.type === 1 && node.tagType === 1 && node.tag && /Modal$/.test(node.tag)) tags.add(node.tag)
    for (const child of (node.children ?? []) as Array<typeof node>) visit(child)
  }
  if (descriptor.template?.ast) visit(descriptor.template.ast as unknown as Parameters<typeof visit>[0])
  return [...tags].sort()
}

describe('modal layer wiring guard (static)', () => {
  it('every *Modal.vue climbs onto the app modal stack from its setup, unconditionally', () => {
    const offenders: Record<string, string> = {}
    for (const [path, source] of Object.entries(MODAL_SOURCES)) {
      const ast = setupAst(source, path)
      const from = importsOf(ast).get('useModalLayer')
      if (!from || !/(^|\/)composables\/useModalLayer$/.test(from)) offenders[path] = `useModalLayer import 없음 (${from ?? '-'})`
      else if (topLevelModalLayerCalls(ast) !== 1) offenders[path] = `setup 최상위 useModalLayer() 호출 ${topLevelModalLayerCalls(ast)}개`
    }
    expect(Object.keys(MODAL_SOURCES).length).toBeGreaterThanOrEqual(12)
    expect(offenders).toEqual({})
  })

  it('the checker itself notices a modal whose useModalLayer() line was deleted or made conditional', () => {
    const header = ['<template><div /></template>', '<script setup lang="ts">', "import { useModalLayer } from '../composables/useModalLayer'"]
    const deleted = [...header, 'const x = 1', '</script>'].join('\n')
    const conditional = [...header, 'if (Math.random() > 2) useModalLayer()', '</script>'].join('\n')
    const ok = [...header, 'useModalLayer()', '</script>'].join('\n')
    expect(topLevelModalLayerCalls(setupAst(deleted, 'Deleted.vue'))).toBe(0)
    expect(topLevelModalLayerCalls(setupAst(conditional, 'Conditional.vue'))).toBe(0)
    expect(topLevelModalLayerCalls(setupAst(ok, 'Ok.vue'))).toBe(1)
  })

  it('every modal App.vue renders is one of those checked files — the five from the bug included', () => {
    const imports = importsOf(setupAst(appSource, 'App.vue'))
    const tags = modalTagsInTemplate(appSource)
    for (const name of ['LoraManagerModal', 'CondPromptModal', 'ABTestModal', 'CharacterPresetModal', 'CharFeatureOverrideModal']) {
      expect(tags).toContain(name)
    }
    expect(tags.length).toBeGreaterThanOrEqual(12)
    const unresolved = tags.filter(tag => !(imports.get(tag) ?? '') || !(imports.get(tag)! in MODAL_SOURCES))
    expect(unresolved).toEqual([])
  })

  it("App's only keydown listener is createAppKeydownHandler over appModalStack", () => {
    const ast = setupAst(appSource, 'App.vue')
    const imports = importsOf(ast)
    expect(imports.get('createAppKeydownHandler')).toBe('./utils/appShortcuts')
    expect(imports.get('appModalStack')).toBe('./utils/modalStack')

    const keydownListeners: AstNode[] = []
    walk(ast, node => {
      if (node.type !== 'CallExpression') return
      const callee = node.callee as AstNode
      const args = node.arguments as AstNode[]
      const isAddListener = callee.type === 'MemberExpression' && isIdentifier(callee.property, 'addEventListener')
      if (isAddListener && args[0]?.type === 'StringLiteral' && (args[0] as { value?: string }).value === 'keydown') {
        expect(isIdentifier(callee.object, 'document')).toBe(true)
        keydownListeners.push(args[1])
      }
    })
    expect(keydownListeners).toHaveLength(1)
    const handler = keydownListeners[0]
    expect(handler.type).toBe('CallExpression')
    expect(isIdentifier(handler.callee, 'createAppKeydownHandler')).toBe(true)
    const options = (handler.arguments as AstNode[])[0]
    expect(options.type).toBe('ObjectExpression')
    const stackProp = (options.properties as AstNode[]).find(p => p.type === 'ObjectProperty' && isIdentifier(p.key, 'modalStack'))
    // 모달이 오르는 바로 그 스택이어야 한다 — 다른 스택(또는 빈 목록)을 넘기면 가드가 허공을 본다
    expect(stackProp && isIdentifier(stackProp.value, 'appModalStack')).toBe(true)
  })
})

// ── 실제 모달 마운트 → 실제 appModalStack → App 의 keydown 판단 ───────────────────────────────

const noop = {}
const vueForTest = { ...Vue, vModelText: noop, vModelSelect: noop, vModelCheckbox: noop, vModelRadio: noop, vModelDynamic: noop }

/** 백엔드는 영영 안 온다 — 마운트 훅이 브리지 호출 앞에서 멈춰, 이 테스트는 스택만 본다. */
const bridgeStub = { getBackend: () => new Promise(() => {}), onBackendEvent: () => () => {} }

const SELF_ESC_MODALS: Record<string, string> = {
  LoraManagerModal: loraSource,
  CondPromptModal: condSource,
  ABTestModal: abSource,
  CharacterPresetModal: charPresetSource,
  CharFeatureOverrideModal: charOverrideSource,
}

function compileModal(name: string) {
  const condEnabled = Vue.ref(true)
  return compileSfc(SELF_ESC_MODALS[name], `${name}-layer-test`, {
    '../bridge.js': bridgeStub,
    '../stores/widgetStore.js': { requestAction: vi.fn() },
    '../utils/bridgeRequest': bridgeRequest,
    '../utils/charPresetState': charPresetState,
    '../utils/imeComposition': imeComposition,
    '../composables/useModalLayer': modalLayer,
    '../composables/condRules.js': {
      condPositive: Vue.reactive([]), condNegative: Vue.reactive([]), condEnabled,
      addCondRule: vi.fn(), removeCondRule: vi.fn(), saveCondRules: vi.fn(), loadCondRules: vi.fn(),
    },
    './ToggleSwitch.vue': stubComponentModule('toggle'),
  }, vueForTest)
}

/** window capture 리스너(모달의 제 ESC) → 막히지 않았으면 document 의 App keydown — 실제 DOM 순서. */
type KeyListener = (e: unknown) => void
let windowListeners: KeyListener[] = []
let mounted: Mounted | null = null

beforeEach(() => {
  windowListeners = []
  vi.stubGlobal('window', {
    addEventListener: (name: string, cb: KeyListener) => { if (name === 'keydown') windowListeners.push(cb) },
    removeEventListener: (name: string, cb: KeyListener) => { windowListeners = windowListeners.filter(fn => fn !== cb) },
  })
})
afterEach(() => {
  mounted?.unmount(); mounted = null
  vi.unstubAllGlobals()
})

function appHandler() {
  const deps = {
    isGateOpen: () => false,
    generate: vi.fn(), saveSettings: vi.fn(), reloadHistory: vi.fn(), navigateTabs: vi.fn(),
    isParamsPanelOpen: () => true,
    closeParamsPanel: vi.fn(),
    hasHistory: () => true,
    navigateHistory: vi.fn(),
    navigateHistoryEdge: vi.fn(),
    activeElement: () => ({ tagName: 'BODY' }),
    jumpModifier: () => null,
    modalStack: appModalStack,   // App.vue 와 같은 스택(위 정적 가드가 확인)
  }
  const handle = createAppKeydownHandler(deps)
  const press = (key: string) => {
    let stopped = false
    const event = {
      key, ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, defaultPrevented: false,
      preventDefault: vi.fn(() => { event.defaultPrevented = true }), stopPropagation: () => { stopped = true },
    }
    for (const listener of [...windowListeners]) listener(event)
    if (!stopped) handle(event as AppShortcutEvent)
    return event
  }
  return { deps, press }
}

describe.each(Object.keys(SELF_ESC_MODALS))('%s behind the history', (name) => {
  it('blocks ↑/↓ history navigation exactly while it is mounted', () => {
    const { deps, press } = appHandler()
    const onClose = vi.fn()
    expect(appModalStack.isAnyOpen()).toBe(false)
    mounted = mountFake(compileModal(name), { onClose })
    expect(appModalStack.isAnyOpen()).toBe(true)

    press('ArrowUp'); press('ArrowDown')
    expect(deps.navigateHistory).not.toHaveBeenCalled()

    mounted.unmount(); mounted = null                        // 부모의 v-if 가 내린다
    expect(appModalStack.isAnyOpen()).toBe(false)
    press('ArrowDown')
    expect(deps.navigateHistory).toHaveBeenCalledWith(1)
  })

  it('ESC closes the modal itself, not the params column behind it', () => {
    const { deps, press } = appHandler()
    const onClose = vi.fn()
    mounted = mountFake(compileModal(name), { onClose })
    press('Escape')
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(deps.closeParamsPanel).not.toHaveBeenCalled()
  })
})
