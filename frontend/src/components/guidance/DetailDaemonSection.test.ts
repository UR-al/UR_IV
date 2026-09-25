import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createSSRApp, nextTick, reactive, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import type { SamExtraCapabilitiesEvent } from '../../types/bridge'
import * as ddInputs from '../../utils/detailDaemonInputs'
import { DD_FLOAT_INPUTS } from '../../utils/detailDaemonInputs'
import * as guidanceWidgets from './guidanceWidgets'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import { byTag, mountFake, type FakeNode, type Mounted } from '../../testing/fakeDomRenderer'
import sectionSource from './DetailDaemonSection.vue?raw'

const caps = ref<Partial<SamExtraCapabilitiesEvent> | null>(null)
vi.mock('../../composables/useSamExtraCapabilities', () => ({
  useSamExtraCapabilities: () => ({ capabilities: caps, refresh: () => {}, mayUse: () => true }),
}))

import DetailDaemonSection from './DetailDaemonSection.vue'

/**
 * Detail Daemon 칸 = 원본 노드 입력(Jonseed/ComfyUI-Detail-Daemon DetailDaemonSamplerNode) + Forge 전용 Hires Pass.
 * 값은 노드 단위 그대로 저장·전송된다(core/anima_guidance.py) — 여기서는 입력 범위·단위와 없어진 칸을 지킨다.
 *
 * origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:324-356 (이름: [default, min, max, step])
 */
const NODE_INPUTS: Record<string, [number, number, number, number]> = {
  dd_amount: [0.1, -5.0, 5.0, 0.01],
  dd_start: [0.2, 0.0, 1.0, 0.01],
  dd_end: [0.8, 0.0, 1.0, 0.01],
  dd_bias: [0.5, 0.0, 1.0, 0.01],
  dd_exponent: [1.0, 0.0, 10.0, 0.05],
  dd_start_offset: [0.0, -1.0, 1.0, 0.01],
  dd_end_offset: [0.0, -1.0, 1.0, 0.01],
  dd_fade: [0.0, 0.0, 1.0, 0.05],
}

async function render(values: Record<string, string>) {
  const widgets = reactive<Record<string, any>>(values)
  return renderToString(createSSRApp(DetailDaemonSection, { widgets }))
}

function inputFor(html: string, value: string): string {
  const tag = html.match(new RegExp(`<input[^>]*value="${value.replace('.', '\\.')}"[^>]*>`))
  expect(tag, value).not.toBeNull()
  return tag![0]
}

function attrOf(tag: string, name: string): string {
  const m = tag.match(new RegExp(`\\s${name}="([^"]*)"`))
  expect(m, `${name} in ${tag}`).not.toBeNull()
  return m![1]
}

describe('DetailDaemonSection', () => {
  beforeEach(() => { caps.value = null })

  it('uses the node input table (defaults, ranges, steps)', () => {
    for (const [key, [def, min, max, step]] of Object.entries(NODE_INPUTS)) {
      const spec = DD_FLOAT_INPUTS[key as keyof typeof DD_FLOAT_INPUTS]
      expect([spec.default, spec.min, spec.max, spec.step], key).toEqual([def, min, max, step])
    }
    expect(Object.keys(DD_FLOAT_INPUTS).sort()).toEqual(Object.keys(NODE_INPUTS).sort())
  })

  it('renders each node input with the node range, step and default as placeholder', async () => {
    // 칸마다 다른 표식 값을 넣고 렌더된 <input> 을 그 값으로 찾는다 — 세부 스케줄(<details>)이 닫혀 있어도 DOM 에 있다
    const marks = Object.fromEntries(Object.keys(NODE_INPUTS).map((key, i) => [key, `0.0${i + 1}7`]))
    const html = await render({
      _dd_enabled: 'true',
      ...Object.fromEntries(Object.entries(marks).map(([key, mark]) => [`_${key}`, mark])),
    })
    for (const [key, [def, min, max, step]] of Object.entries(NODE_INPUTS)) {
      const tag = inputFor(html, marks[key])
      expect(attrOf(tag, 'type'), key).toBe('number')
      expect([attrOf(tag, 'min'), attrOf(tag, 'max'), attrOf(tag, 'step')].map(Number), key).toEqual([min, max, step])
      expect(Number(attrOf(tag, 'placeholder')), key).toBe(def)
    }
    expect(inputFor(html, marks.dd_amount)).toMatch(/placeholder="0\.10"/)
  })

  it('shows the original labels in the original order', async () => {
    // origin: muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:103-121 (라벨·배치 순서)
    const html = await render({ _dd_enabled: 'true' })
    const order = ['Enable Detail Daemon', 'Hires Pass', 'Detail Amount', '>Start<', '>End<', '>Start Offset<',
      '>End Offset<', '>Bias<', '>Exponent<', '>Fade<', 'Smooth']
    const positions = order.map(text => html.indexOf(text))
    order.forEach((text, i) => expect(positions[i], text).toBeGreaterThanOrEqual(0))
    expect(positions).toEqual([...positions].sort((a, b) => a - b))
    expect(html).not.toMatch(/Detail amount|Start offset|End offset/)
  })

  it('wires every number input to commit(<its key>) on change and a default placeholder (template text)', () => {
    // 확정 동작 자체는 아래 'mounted' 묶음이 이벤트를 실제로 불러 확인한다
    const inputs = [...sectionSource.matchAll(/<input\b[^>]*>/g)].map(m => m[0])
    expect(inputs).toHaveLength(Object.keys(NODE_INPUTS).length)
    for (const tag of inputs) {
      const key = tag.match(/v-model="w\._(dd_\w+)"/)?.[1]
      expect(key, tag).toBeDefined()
      expect(tag, key).toContain(`:placeholder="placeholder.${key}"`)
      expect(tag, key).toContain(`@change="commit('${key}')"`)
    }
  })

  it('shows the saved amount as is (node units, no ×10 or preset conversion)', async () => {
    const html = await render({ _dd_enabled: 'true', _dd_amount: '0.1', _dd_start_offset: '0.05' })
    expect(inputFor(html, '0.1')).toMatch(/min="-5" max="5"/)
    expect(inputFor(html, '0.05')).toMatch(/min="-1" max="1"/)
    expect(html).not.toMatch(/실효|슬라이더 값|÷|× ?10/)
  })

  it('has no preset, multiplier or CFG-couple controls (not in the original)', async () => {
    const html = await render({ _dd_enabled: 'true' })
    expect(sectionSource).not.toMatch(/_dd_(preset|multiplier|cfg_couple)|detailDaemonScale/)
    expect(html).not.toMatch(/Preset|Multiplier|Couple to CFG/)
  })

  it('has a Hires Pass toggle bound to dd_hires', async () => {
    expect(sectionSource).toMatch(/b\('dd_hires'\)/)
    expect(sectionSource).toMatch(/setB\('dd_hires', \$event\)/)
    expect(await render({ _dd_enabled: 'true' })).toContain('Hires Pass')
    expect(await render({ _dd_enabled: 'false' })).not.toContain('Hires Pass')
  })

  it('warns only when Hires Pass is on and the connected sam-extra has no Hires Pass slot', async () => {
    const on = { _dd_enabled: 'true', _dd_hires: 'true' }
    // ComfyUI(not_applicable)는 컴파일러가 Hires Pass 를 따른다(DD-C) — 예전 'ComfyUI 는 아직' 경고는 없어졌다
    for (const snapshot of [null, { status: 'ok', known: true, detail_daemon_hires: true },
      { status: 'unknown', known: false, detail_daemon_hires: null },
      { status: 'not_applicable', known: false, detail_daemon_hires: null }] as const) {
      caps.value = snapshot
      const html = await render(on)
      expect(html, snapshot?.status ?? 'null').not.toContain('모든 패스에 적용')
      expect(html, snapshot?.status ?? 'null').not.toContain('Hires Pass 를 따르지')
    }
    caps.value = { status: 'ok', known: true, detail_daemon_hires: false }
    expect(await render(on)).toContain('모든 패스에 적용')
    expect(await render({ ...on, _dd_hires: 'false' })).not.toContain('모든 패스에 적용')
    expect(sectionSource).not.toContain('ComfyUI 는 아직')
  })
})

/**
 * 마운트해서 칸 확정(change)을 실제로 부른다 — SSR 렌더는 이벤트를 돌리지 않는다.
 * vitest 의 .vue import 는 SSR 용이라 원문을 브라우저용으로 컴파일해(testing/compileSfc) 가짜 DOM 에 올린다.
 */
describe('DetailDaemonSection (mounted)', () => {
  const Section = compileSfc(sectionSource, 'dd-section-test', {
    '../ToggleSwitch.vue': stubComponentModule('toggle'),
    '../../composables/useSamExtraCapabilities': {
      useSamExtraCapabilities: () => ({ capabilities: caps, refresh: () => {}, mayUse: () => true }),
    },
    './guidanceWidgets': guidanceWidgets,
    '../../utils/detailDaemonInputs': ddInputs,
  })
  let mounted: Mounted | null = null

  beforeEach(() => {
    caps.value = null
    // runtime-dom v-model 이 값이 바뀐 칸을 다시 그릴 때 보는 DOM 타입 — 가짜 노드는 어느 문서에도 없다
    vi.stubGlobal('Document', class {})
    vi.stubGlobal('ShadowRoot', class {})
  })
  afterEach(() => { mounted?.unmount(); mounted = null; vi.unstubAllGlobals() })

  function mountSection(values: Record<string, unknown>) {
    const widgets = reactive<Record<string, any>>({ _dd_enabled: 'true', ...values })
    mounted = mountFake(Section, { widgets })
    const inputs = byTag(mounted.root, 'input')
    const root = mounted.root
    // runtime-dom vModelText.beforeUpdate 가 부른다 — 가짜 루트를 돌려주면(Document 아님) 새 값을 칸에 쓴다
    for (const input of inputs) Object.assign(input, { getRootNode: () => root })
    return { widgets, inputs }
  }

  /** 사용자 입력 — 값을 넣고 input(v-model) → change(v-model 캐스트, 그다음 @change) 순서로 알린다. */
  function type(input: FakeNode, text: string) {
    input.value = text
    for (const cb of input.listeners.input || []) cb({ target: input })
    for (const cb of input.listeners.change || []) cb({ target: input })
    input.props.onChange?.({ target: input })
  }

  /** 칸마다 다른 표식 값(범위 안) — 렌더된 칸을 그 값으로 찾는다. */
  const MARKS = Object.fromEntries(Object.keys(NODE_INPUTS).map((key, i) => [`_${key}`, `0.0${i + 1}7`]))

  it('binds one number input per node input, each to its own widget', () => {
    const { inputs } = mountSection(MARKS)
    expect(inputs).toHaveLength(Object.keys(NODE_INPUTS).length)
    expect(inputs.map(n => n.value).sort()).toEqual(Object.values(MARKS).sort())
    for (const input of inputs) expect(input.props.type).toBe('number')
  })

  it('clamps an out-of-range value to the node range on change — every input, only its own key', async () => {
    for (const [key, [, min, max]] of Object.entries(NODE_INPUTS)) {
      for (const [text, want] of [[String(max + 3), max], [String(min - 3), min]] as const) {
        const { widgets, inputs } = mountSection(MARKS)
        const input = inputs.find(n => n.value === MARKS[`_${key}`])
        expect(input, key).toBeDefined()
        type(input!, text)
        expect(widgets[`_${key}`], `${key} ← ${text}`).toBe(want)
        for (const [id, mark] of Object.entries(MARKS)) if (id !== `_${key}`) expect(widgets[id], `${key}: ${id}`).toBe(mark)
        await nextTick()
        expect(input!.value, `${key} shown`).toBe(want)
        mounted!.unmount(); mounted = null
      }
    }
  })

  it('keeps in-range and empty values as typed on change', async () => {
    const { widgets, inputs } = mountSection(MARKS)
    const amount = inputs.find(n => n.value === MARKS._dd_amount)!
    for (const [text, want] of [['0.123', 0.123], ['-4.99', -4.99], ['5', 5], ['-5', -5], ['', '']] as const) {
      type(amount, text)
      expect(widgets._dd_amount, text).toBe(want)
      await nextTick()
      expect(amount.value, text).toBe(want)
    }
  })

  it('empties an unreadable saved value on change (the backend sends the default)', () => {
    const { widgets, inputs } = mountSection({ ...MARKS, _dd_exponent: 'abc' })
    const exponent = inputs.find(n => n.value === 'abc')!
    exponent.props.onChange({ target: exponent })
    expect(widgets._dd_exponent).toBe('')
  })
})
