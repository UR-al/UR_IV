import { afterEach, expect, it } from 'vitest'
import * as colorcraft from '../../utils/colorcraft'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import { byClass, byTag, findAll, mountFake, type Mounted } from '../../testing/fakeDomRenderer'
import source from './ColorcraftMaskEditor.vue?raw'
import previewSource from './ColorcraftSchedulePreview.vue?raw'

const MaskEditor = compileSfc(source, 'cc-mask-test', {
  '../CustomSelect.vue': stubComponentModule('csel'),
  '../ToggleSwitch.vue': stubComponentModule('toggle'),
  '../../utils/colorcraft': colorcraft,
})
const Preview = compileSfc(previewSource, 'cc-preview-test', { '../../utils/colorcraft': colorcraft })
let mounted: Mounted | undefined
afterEach(() => { mounted?.unmount(); mounted = undefined })

function mount(state: string, tag: string) {
  const updates: Array<[string, unknown]> = []
  let resets = 0
  mounted = mountFake(MaskEditor, { config: colorcraft.configFromState(true, true, state), tag,
    onUpdate: (name: string, value: unknown) => updates.push([name, value]), onReset: () => { resets += 1 } })
  return { root: mounted.root, updates, resets: () => resets }
}

it('a leaf shows its ten fields with the extension choices', () => {
  const { root, updates } = mount('{"v":1,"M2.mask_mode":"lows","M2.blur":80}', 'M2')
  const selects = byTag(root, 'csel')
  expect(selects.map(s => s.props.modelValue)).toEqual(['exposure', 'lows 낮은 쪽'])
  expect(selects[0]!.props.options).toEqual([...colorcraft.SCHEMA.leafFields[0]!.choices])
  expect(findAll(root, n => n.tag === 'input')).toHaveLength(7)               // 강도·폭·중심·단단함·흐림·넓힘·대비
  expect(byTag(root, 'toggle')).toHaveLength(1)                              // 정규화
  expect(root.textContent).toContain(`흐림(이미지 px): ${colorcraft.OUT_OF_RANGE_NOTE}`)
  selects[1]!.props['onUpdate:modelValue']('protect range 범위 보호')
  findAll(root, n => n.tag === 'input' && n.props.title === 'Hardness')[0]!.props.onChange({ target: { value: '2' } })
  expect(updates).toEqual([['mask_mode', 'protect range'], ['mask_hardness', '2']])
})

it('a combo offers leaves and earlier combos only and says when it is incomplete', () => {
  const c1 = mount('{"v":1,"C1.mask_a":"M1"}', 'C1')
  const [a, b] = byTag(c1.root, 'csel')
  expect(a!.props.options).toEqual(['없음', ...colorcraft.MASK_TAGS])
  expect(a!.props.modelValue).toBe('M1')
  expect(b!.props.modelValue).toBe('없음')
  expect(c1.root.textContent).toContain('(미완성) — A·B 를 모두 골라야')
  expect(c1.root.textContent).toContain('C1 은 앞선 조합이 없어')
  mounted!.unmount()
  const c3 = mount('{"v":1,"C1.mask_a":"M1","C1.mask_b":"M2","C3.mask_a":"C1","C3.mask_b":"M4"}', 'C3')
  expect(byTag(c3.root, 'csel')[0]!.props.options).toEqual(['없음', ...colorcraft.MASK_TAGS, 'C1', 'C2'])
  expect(c3.root.textContent).not.toContain('(미완성)')
  byTag(c3.root, 'csel')[1]!.props['onUpdate:modelValue']('C2')
  byTag(c3.root, 'csel')[1]!.props['onUpdate:modelValue']('C4')              // 뒤 조합은 목록에 없다
  byTag(c3.root, 'csel')[2]!.props['onUpdate:modelValue']('xor 한쪽만')
  byClass(c3.root, 'cc-reset')[0]!.props.onClick()
  expect(c3.updates).toEqual([['mask_b', 'C2'], ['operation', 'xor']])
  expect(c3.resets()).toBe(1)
})

it('schedule preview draws make_schedule for the card steps, or says why not', () => {
  const node = colorcraft.configFromState(true, false, '{"v":1,"I.start":0,"I.end":1}').modifiers[0]!
  mounted = mountFake(Preview, { modifier: node, steps: 5 })
  const line = byClass(mounted.root, 'cc-spark-line')[0]!
  const values = colorcraft.scheduleValues(node, 5)!
  expect(String(line.props.points).split(' ')).toHaveLength(5)
  expect(String(line.props.points).split(' ')[0]).toBe(`0.00,${(12 - values[0]! * 11).toFixed(2)}`)
  expect(mounted.root.textContent).toContain('5 스텝 · 가장 큰 값 1.00')
  mounted.unmount()
  mounted = mountFake(Preview, { modifier: { ...node, start: 1.5 }, steps: 5 })
  expect(byClass(mounted.root, 'cc-spark-line')).toEqual([])
  expect(mounted.root.textContent).toContain('미리보기 없음')
})
