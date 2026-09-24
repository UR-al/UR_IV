import { afterEach, expect, it } from 'vitest'
import { nextTick, reactive } from 'vue'
import source from './AdSlotFields.vue?raw'
import { byTag, mountFake, type FakeNode, type Mounted } from '../../testing/fakeDomRenderer'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import * as adSlotKeys from '../../utils/adSlotKeys'
import * as paramItems from '../../composables/useParamItems'
import { AD_SLOT_FIELDS, adSlotKey, type AdSlotNo } from '../../utils/adSlotKeys'

// ADetailer 슬롯 하나의 필드가 정확히 `_ad_s{n}_*` 위젯에 묶이는지 — 슬롯 1·2 를 한 컴포넌트로 합치며
// 키 하나라도 잘못 이으면(오타·슬롯 번호) 예외 없이 그 칸만 조용히 딴 값을 쓴다.
let widgets: Record<string, any> = {}
const store = { useWidgetStore: () => ({ widgets, getProperty: () => '' }) }
// 자식 컴포넌트는 받은 속성을 그대로 드러내는 빈 요소로 — v-model 은 modelValue / onUpdate:modelValue 로 보인다
const AdSlotFields = compileSfc(source, 'ad-slot-fields-test', {
  '../CustomSelect.vue': stubComponentModule('csel'),
  '../ToggleSwitch.vue': stubComponentModule('toggle'),
  '../../stores/widgetStore.js': store,
  '../../composables/useParamItems': paramItems,
  '../../utils/adSlotKeys': adSlotKeys,
})

const FLAG_FIELDS = AD_SLOT_FIELDS.filter(f => f.startsWith('use_'))
const VALUE_FIELDS = AD_SLOT_FIELDS.filter(f => !f.startsWith('use_'))

let mounted: Mounted | null = null
afterEach(() => { mounted?.unmount(); mounted = null })

function mountSlot(slotNo: AdSlotNo, flagsOn: boolean) {
  widgets = reactive<Record<string, any>>({})
  for (const f of VALUE_FIELDS) widgets[adSlotKey(slotNo, f)] = `v:${f}`
  for (const f of FLAG_FIELDS) widgets[adSlotKey(slotNo, f)] = flagsOn ? 'true' : 'false'
  widgets[`ad_slot${slotNo}_group`] = 'false'
  mounted = mountFake(AdSlotFields, { slotNo, modelItems: ['face_yolov8n.pt'], promptPlaceholder: `slot ${slotNo} prompt` })
  return { widgets, root: mounted.root }
}

const boundValues = (root: FakeNode) => [
  ...byTag(root, 'input').map(n => n.value),
  ...byTag(root, 'csel').map(n => n.props.modelValue),
].sort()

it.each([1, 2] as const)('slot %i binds every value field to its own widget key', async (slotNo) => {
  const { root } = mountSlot(slotNo, true)
  await nextTick()
  expect(boundValues(root)).toEqual(VALUE_FIELDS.map(f => `v:${f}`).sort())
  const prompt = byTag(root, 'input').find(n => n.value === 'v:prompt')!
  expect(prompt.props.placeholder).toBe(`slot ${slotNo} prompt`)
  const model = byTag(root, 'csel').find(n => n.props.modelValue === 'v:model')!
  expect(model.props.options).toEqual(['face_yolov8n.pt'])
})

it('optional rows stay hidden until their "use" flag is on', async () => {
  const { root } = mountSlot(2, false)
  await nextTick()
  const always = ['model', 'prompt', 'neg', 'confidence', 'denoise', 'mask_blur', 'padding', 'dilate_erode', 'mask_merge']
  expect(boundValues(root)).toEqual(always.map(f => `v:${f}`).sort())
})

it('toggles write "true"/"false" strings to the slot keys (and the enable box to ad_slot{n}_group)', async () => {
  const { root, widgets } = mountSlot(2, false)
  await nextTick()
  const toggles = byTag(root, 'toggle')
  expect(toggles).toHaveLength(1 + FLAG_FIELDS.length)
  toggles[0].props['onUpdate:modelValue'](true)
  expect(widgets.ad_slot2_group).toBe('true')
  for (let i = 0; i < FLAG_FIELDS.length; i++) byTag(root, 'toggle')[i + 1].props['onUpdate:modelValue'](true)
  for (const f of FLAG_FIELDS) expect(widgets[adSlotKey(2, f)]).toBe('true')
  expect(Object.keys(widgets).some(k => k.startsWith('_ad_s1_'))).toBe(false)   // 다른 슬롯은 건드리지 않는다
  await nextTick()
  // 스텁은 props 를 선언하지 않아 `:model-value` 가 케밥 그대로 온다(진짜 ToggleSwitch 는 modelValue 로 받는다)
  const shown = byTag(root, 'toggle').map(t => t.props.modelValue ?? t.props['model-value'])
  expect(shown).toEqual([true, true, true, true, true, true, true])
})

it('typing into a field writes the store under the slot key', async () => {
  const { root, widgets } = mountSlot(1, false)
  await nextTick()
  const neg = byTag(root, 'input').find(n => n.value === 'v:neg')!
  neg.value = 'bad hands'
  for (const cb of neg.listeners.input || []) cb({ target: neg })
  expect(widgets._ad_s1_neg).toBe('bad hands')
})
