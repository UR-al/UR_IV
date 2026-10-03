import { afterEach, expect, it } from 'vitest'
import { nextTick } from 'vue'
import * as colorcraft from '../../utils/colorcraft'
import { compileSfc } from '../../testing/compileSfc'
import { byClass, byTag, mountFake, type Mounted } from '../../testing/fakeDomRenderer'
import source from './ColorcraftNodePicker.vue?raw'

const Picker = compileSfc(source, 'cc-picker-test', { '../../utils/colorcraft': colorcraft })
let mounted: Mounted | undefined
afterEach(() => { mounted?.unmount(); mounted = undefined })

function mount(config: colorcraft.ColorcraftConfig, selected: string, which: 'modifierMarks' | 'maskMarks') {
  const picked: string[] = []
  const marks = colorcraft.overview(config)[which]
  mounted = mountFake(Picker, { marks, selected, label: '수정자', dividerBefore: which === 'maskMarks' ? 'C1' : '',
    onSelect: (tag: string) => picked.push(tag) })
  return { root: mounted.root, picked }
}

it('shows the extension marks and which node is open', () => {
  const config = colorcraft.configFromState(true, false, '{"v":1,"II.exposure":0.4,"III.active":true}')
  const { root } = mount(config, 'III', 'modifierMarks')
  const chips = byTag(root, 'button')
  expect(chips.map(c => c.textContent)).toEqual(['I ●', 'II ○', 'III ●', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'])
  expect(chips.map(c => c.props['aria-pressed'])).toEqual(['false', 'false', 'true', ...Array(7).fill('false')])
  expect(byClass(root, 'cc-chip-on').map(c => c.textContent)).toEqual(['I ●', 'III ●'])
  expect(byClass(root, 'cc-chip-edited').map(c => c.textContent)).toEqual(['II ○'])
  expect(chips[1]!.props.title).toContain('바꿨지만 꺼짐')
  expect(byTag(root, 'div')[0]!.props.role).toBe('toolbar')
})

it('click and arrow keys select (wrapping at both ends)', async () => {
  const { root, picked } = mount(colorcraft.defaultConfig(), 'I', 'modifierMarks')
  byTag(root, 'button')[4]!.props.onClick()
  const toolbar = byTag(root, 'div')[0]!
  let prevented = 0
  const key = (name: string) => toolbar.props.onKeydown({ key: name, preventDefault: () => { prevented += 1 } })
  key('ArrowLeft')
  key('ArrowRight')
  key('Enter')
  await nextTick()
  expect(picked).toEqual(['V', 'X', 'II'])
  expect(prevented).toBe(2)
})

it('mask picker has the leaf │ combo divider and marks masks in use', () => {
  const config = colorcraft.configFromState(true, true, '{"v":1,"I.mask":"C1","C1.mask_a":"M1","C1.mask_b":"M2","M5.blur":3}')
  const { root } = mount(config, 'M1', 'maskMarks')
  expect(byClass(root, 'cc-picker-divider')).toHaveLength(1)
  const labels = byTag(root, 'button').map(c => c.textContent)
  expect(labels.slice(0, 5)).toEqual(['M1 ●', 'M2 ●', 'M3', 'M4', 'M5 ○'])
  expect(labels[10]).toBe('C1 ●')
  expect(labels).toHaveLength(15)
})
