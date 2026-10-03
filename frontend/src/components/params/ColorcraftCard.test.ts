import { afterEach, expect, it } from 'vitest'
import * as Vue from 'vue'
import * as colorcraft from '../../utils/colorcraft'
import * as family from '../../utils/generationFamily'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import { FakeNode, byClass, byTag, mountFake, type Mounted } from '../../testing/fakeDomRenderer'
import source from './ColorcraftCard.vue?raw'

// textarea 의 v-model(vModelText)은 갱신 때 getRootNode().activeElement 를 본다 — 가짜 노드에 최소 대역
;(FakeNode.prototype as any).getRootNode ??= () => ({ activeElement: null })
const g = globalThis as { document?: unknown; Document?: unknown; ShadowRoot?: unknown }
if (!g.document) g.document = { activeElement: null }
if (!g.Document) g.Document = class {}
if (!g.ShadowRoot) g.ShadowRoot = class {}
const caps = Vue.ref<any>(null)
const imports: Array<[string, string | undefined]> = []
const widgets = Vue.reactive<Record<string, any>>({})
// 수정자 편집기 대역 — 받은 속성을 드러내고, 카드가 넣는 미리보기 슬롯을 그 수정자로 그린다
const EditorStub = Vue.defineComponent({
  inheritAttrs: false,
  setup(_props, { attrs, slots }) {
    return () => {
      const config = attrs.config as colorcraft.ColorcraftConfig
      const modifier = config.modifiers[colorcraft.MODIFIER_TAGS.indexOf(String(attrs.tag))]
      return Vue.h('modeditor', attrs, slots.schedule?.({ modifier }))
    }
  },
})
const Card = compileSfc(source, 'cc-card-test', {
  '../ToggleSwitch.vue': stubComponentModule('toggle'),
  '../colorcraft/ColorcraftNodePicker.vue': stubComponentModule('picker'),
  '../colorcraft/ColorcraftModifierEditor.vue': { __esModule: true, default: EditorStub },
  '../colorcraft/ColorcraftMaskEditor.vue': stubComponentModule('maskeditor'),
  '../colorcraft/ColorcraftSchedulePreview.vue': stubComponentModule('preview'),
  '../../stores/widgetStore.js': { useWidgetStore: () => ({ widgets }) },
  '../../composables/useSamExtraCapabilities': { useSamExtraCapabilities: () => ({ capabilities: caps }) },
  '../../utils/generationFamily': family,
  '../../utils/samExtraImport': {
    importSamExtra: (text: string, only?: string) => { if (!text.trim()) return false; imports.push([text, only]); return true },
  },
  '../../utils/colorcraft': colorcraft,
})
let mounted: Mounted | undefined
afterEach(() => {
  mounted?.unmount(); mounted = undefined; caps.value = null; imports.length = 0
  for (const key of Object.keys(widgets)) delete widgets[key]
})

async function mount(): Promise<FakeNode> {
  mounted = mountFake(Card)
  await Vue.nextTick()
  return mounted.root
}
const one = (root: FakeNode, tag: string) => byTag(root, tag)[0]!
const status = (root: FakeNode) => byClass(root, 'cc-status')[0]!.textContent
const ID = colorcraft.WIDGET_IDS

it('default card is off, writes nothing and shows the extension defaults', async () => {
  const root = await mount()
  expect(Object.keys(widgets)).toEqual([])
  expect(byClass(root, 'cc-summary')).toEqual([])
  expect(status(root)).toBe(colorcraft.OFF_NOTE)
  expect(byTag(root, 'toggle').map(t => t.props.modelValue)).toEqual([false, false, false])   // 사용·마스크·I2I
  const picker = one(root, 'picker')
  expect(picker.props.selected).toBe('I')
  expect(picker.props.marks[0]).toEqual({ label: 'I ●', tag: 'I', mark: '●' })
  expect(byTag(root, 'maskeditor')).toEqual([])                                 // 마스크 끔
  expect(byClass(root, 'cc-reset-all')[0]!.props.class).toContain('cc-muted')
  expect(root.textContent).toContain('한눈에: 꺼짐 — 켠 수정자: I Advanced')
})

it('switches and editor updates write the four widget strings', async () => {
  const root = await mount()
  byTag(root, 'toggle')[0]!.props['onUpdate:modelValue'](true)
  await Vue.nextTick()
  expect(widgets[ID.enabled]).toBe('true')
  expect(status(root)).toBe(colorcraft.NO_ENTRIES_NOTE)                         // 켰지만 모두 0
  one(root, 'modeditor').props.onUpdate('exposure', '0.30')
  await Vue.nextTick()
  expect(widgets[ID.state]).toBe('{"v":1,"I.exposure":0.3}')
  expect(status(root)).toBe('Forge 로 보냅니다 — 수정자 1개(Base 1·Hires 0)')
  expect(byClass(root, 'cc-summary')[0]!.textContent).toBe('I Advanced')
  expect(byClass(root, 'cc-infotext')[0]!.props.value).toBe('v1;mods=I;I.exposure=0.3')
  one(root, 'picker').props.onSelect('III')
  await Vue.nextTick()
  expect(one(root, 'modeditor').props.tag).toBe('III')
  one(root, 'modeditor').props.onUpdate('active', true)
  one(root, 'modeditor').props.onUpdate('kind', 'Punch')
  await Vue.nextTick()
  expect(widgets[ID.state]).toBe('{"v":1,"I.exposure":0.3,"III.active":true,"III.kind":"Punch"}')
  one(root, 'modeditor').props.onReset()
  await Vue.nextTick()
  expect(widgets[ID.state]).toBe('{"v":1,"I.exposure":0.3}')
  byTag(root, 'toggle')[2]!.props['onUpdate:modelValue'](true)
  await Vue.nextTick()
  expect(widgets[ID.img2img]).toBe('true')
})

it('masking shows the mask picker and editor; a mask nobody uses is pointed out', async () => {
  widgets[ID.enabled] = 'true'
  widgets[ID.masking] = 'true'
  widgets[ID.state] = '{"v":1,"I.exposure":0.3}'
  const root = await mount()
  const pickers = byTag(root, 'picker')
  expect(pickers).toHaveLength(2)
  expect(pickers[1]!.props.dividerBefore).toBe('C1')
  expect(root.textContent).toContain('마스크를 켰지만 마스크를 고른 켜진 수정자가 없습니다')
  pickers[1]!.props.onSelect('C2')
  await Vue.nextTick()
  expect(one(root, 'maskeditor').props.tag).toBe('C2')
  one(root, 'maskeditor').props.onUpdate('mask_a', 'M3')
  one(root, 'modeditor').props.onUpdate('mask', 'C2')
  await Vue.nextTick()
  expect(widgets[ID.state]).toBe('{"v":1,"I.mask":"C2","I.exposure":0.3,"C2.mask_a":"M3"}')
  expect(root.textContent).not.toContain('마스크를 켰지만')
  one(root, 'maskeditor').props.onReset()
  await Vue.nextTick()
  expect(widgets[ID.state]).toBe('{"v":1,"I.mask":"C2","I.exposure":0.3}')
  widgets[ID.state] = '{"v":1,"I.kind":"Basic","I.mask":"M1","I.contrast":0.5}'
  await Vue.nextTick()
  expect(root.textContent).toContain('마스크를 켰지만')                       // Basic 은 고른 마스크를 받지 않는다
})

it('hires switch, steps and an unreadable saved state reach the editor, preview and status', async () => {
  widgets.hires_options_group = 'true'
  widgets.steps_input = '30'
  widgets[ID.enabled] = 'true'
  widgets[ID.state] = '{broken'
  const root = await mount()
  expect(one(root, 'modeditor').props.hiresOn).toBe(true)
  expect(one(root, 'preview').props.steps).toBe(30)
  expect(one(root, 'preview').props.modifier.exposure).toBe(0)
  expect(root.textContent).toContain('저장된 Colorcraft 상태를 읽지 못해')
  widgets.hires_options_group = 'false'
  widgets.steps_input = 'x'
  await Vue.nextTick()
  expect(one(root, 'modeditor').props.hiresOn).toBe(false)
  expect(one(root, 'preview').props.steps).toBe(0)
})

it('backend line: Forge, old Forge, ComfyUI pending, Krea2; reset all', async () => {
  widgets[ID.enabled] = 'true'
  widgets[ID.state] = '{"v":1,"I.exposure":0.3}'
  caps.value = { status: 'ok', known: true, features: { colorcraft: true } }
  const root = await mount()
  expect(status(root)).toContain('Forge 로 보냅니다')
  caps.value = { status: 'ok', known: true, features: { colorcraft: false } }
  await Vue.nextTick()
  expect(status(root)).toBe(colorcraft.OLD_FORGE_NOTE)
  caps.value = { status: 'not_applicable', known: false }
  await Vue.nextTick()
  expect(status(root)).toBe(colorcraft.COMFY_PENDING_NOTE)
  widgets.generation_family_combo = 'Krea2'
  await Vue.nextTick()
  expect(status(root)).toBe(colorcraft.KREA2_NOTE)
  one(root, 'picker').props.onSelect('IV')
  byClass(root, 'cc-reset-all')[0]!.props.onClick()
  await Vue.nextTick()
  expect([widgets[ID.enabled], widgets[ID.masking], widgets[ID.img2img], widgets[ID.state]]).toEqual(['false', 'false', 'false', ''])
  expect(one(root, 'modeditor').props.tag).toBe('I')
})

it('paste box imports Colorcraft only through the generic action and clears itself', async () => {
  const root = await mount()
  const area = byTag(root, 'textarea')[0]!
  const button = byClass(root, 'cc-btn').find(b => b.textContent === '가져오기')!
  expect(button.props.disabled).toBe(true)
  area.value = 'Colorcraft: MOD(I:none,1.0)'
  for (const listener of area.listeners.input ?? []) listener({ target: area })
  await Vue.nextTick()
  expect(button.props.disabled).toBe(false)
  button.props.onClick()
  await Vue.nextTick()
  expect(imports).toEqual([['Colorcraft: MOD(I:none,1.0)', 'colorcraft']])
  expect(byClass(root, 'cc-btn').find(b => b.textContent === '가져오기')!.props.disabled).toBe(true)   // 비웠다
})
