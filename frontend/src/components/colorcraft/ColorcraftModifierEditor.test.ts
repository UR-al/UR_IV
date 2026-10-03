import { afterEach, expect, it } from 'vitest'
import { nextTick, reactive } from 'vue'
import * as colorcraft from '../../utils/colorcraft'
import { compileSfc, stubComponentModule } from '../../testing/compileSfc'
import { byClass, byTag, findAll, mountFake, type FakeNode, type Mounted } from '../../testing/fakeDomRenderer'
import source from './ColorcraftModifierEditor.vue?raw'

const Editor = compileSfc(source, 'cc-modifier-test', {
  '../CustomSelect.vue': stubComponentModule('csel'),
  '../ToggleSwitch.vue': stubComponentModule('toggle'),
  '../../utils/colorcraft': colorcraft,
})
let mounted: Mounted | undefined
afterEach(() => { mounted?.unmount(); mounted = undefined })

function mount(state: string, { masking = false, tag = 'I', hiresOn = true } = {}) {
  const updates: Array<[string, unknown]> = []
  let resets = 0
  const props = reactive({ config: colorcraft.configFromState(true, masking, state), tag, hiresOn,
    onUpdate: (name: string, value: unknown) => updates.push([name, value]), onReset: () => { resets += 1 } })
  mounted = mountFake(Editor, () => ({ ...props }))
  return { root: mounted.root, updates, props, resets: () => resets }
}

const sections = (root: FakeNode) => byClass(root, 'cc-group').map(s => s.props['data-group'])
const selects = (root: FakeNode) => byTag(root, 'csel')
const numberInput = (root: FakeNode, title: string) =>
  findAll(root, n => n.tag === 'input' && n.props.title === title)[0]!

it('Advanced shows every group with its two gates; the mask row needs masking', () => {
  const { root } = mount('')
  expect(sections(root)).toEqual(['luma', 'contrast', 'detail', 'chroma', 'chroma_plus', 'color_shift', 'dev'])
  // 종류·패스·색 이동 방식·색 평면(마스크 칸은 마스크 사용일 때만)
  expect(selects(root).map(s => s.props.modelValue)).toEqual(['Advanced (전부)', 'Base (1차·I2I)', '기본', '색온도·틴트'])
  expect(byClass(root, 'cc-gate').map(g => g.textContent)).toEqual(['Chroma Plus 적용', '색 이동 적용'])
  expect(selects(root)[0]!.props.options).toEqual(['Advanced (전부)', 'Basic (대비·색 이동)', 'Luma (밝기)', 'Chroma (색)',
    'Chroma Plus (색 축 6개)', 'Punch (대비·디테일)', 'Shift (색 이동)'])
  const masked = mount('', { masking: true })
  expect(selects(masked.root).map(s => s.props.modelValue).slice(0, 3)).toEqual(['Advanced (전부)', 'Base (1차·I2I)', '없음'])
})

it('the Type decides the groups; Basic takes no mask; Shift has no gate', () => {
  expect(sections(mount('{"v":1,"I.kind":"Punch"}').root)).toEqual(['contrast', 'detail'])
  const basic = mount('{"v":1,"I.kind":"Basic","I.mask":"M1"}', { masking: true })
  expect(sections(basic.root)).toEqual(['contrast', 'color_shift'])
  expect(selects(basic.root).map(s => s.props.modelValue)).not.toContain('M1')
  const shift = mount('{"v":1,"I.kind":"Shift"}')
  expect(byClass(shift.root, 'cc-gate')).toEqual([])
})

it('emits path names with typed text, picked values and toggles', async () => {
  const { root, updates } = mount('')
  numberInput(root, 'Exposure').props.onChange({ target: { value: '0.35' } })
  selects(root)[0]!.props['onUpdate:modelValue']('Punch (대비·디테일)')
  selects(root)[1]!.props['onUpdate:modelValue']('Hires (Hires 패스)')
  selects(root)[1]!.props['onUpdate:modelValue']('nonsense')                  // 모르는 라벨은 보내지 않는다
  byTag(root, 'toggle')[0]!.props['onUpdate:modelValue'](false)
  await nextTick()
  expect(updates).toEqual([['exposure', '0.35'], ['kind', 'Punch'], ['pass', 'Hires'], ['active', false]])
})

it('gates, developer overrides and advanced schedule lock what the extension ignores', () => {
  const { root } = mount('{"v":1,"I.more_colors":false,"I.recenter_override":true}')
  const notes = byClass(root, 'ext-note').map(n => n.textContent)
  expect(notes).toContain('스위치가 꺼져 있어 이 묶음 값은 쓰이지 않습니다')
  expect(numberInput(root, 'Recenter').props.disabled).toBe(false)
  expect(numberInput(root, 'Max Chroma').props.disabled).toBe(true)
  expect(numberInput(root, 'Bias').props.disabled).toBe(true)                  // 고급 스케줄 끔
  expect(mount('{"v":1,"I.advanced":true}').root && numberInput(mounted!.root, 'Bias').props.disabled).toBe(false)
})

it('hints: Hires pass with Hires.fix off, incomplete combo, out-of-range values; reset', () => {
  const idle = mount('{"v":1,"I.pass":"Hires"}', { hiresOn: false })
  expect(idle.root.textContent).toContain('Hires.fix 가 꺼져 있어')
  const combo = mount('{"v":1,"I.mask":"C1","C1.mask_a":"M1"}', { masking: true })
  expect(combo.root.textContent).toContain('C1(미완성) — 마스크 없이 적용합니다')
  const fine = mount('{"v":1,"I.mask":"C1","C1.mask_a":"M1","C1.mask_b":"M2"}', { masking: true })
  expect(fine.root.textContent).not.toContain('(미완성)')
  const wide = mount('{"v":1,"I.exposure":12.5,"I.strength":-3}')
  expect(wide.root.textContent).toContain(`강도 · 노출: ${colorcraft.OUT_OF_RANGE_NOTE}`)
  expect(byClass(wide.root, 'cc-out')).toHaveLength(2)
  byClass(wide.root, 'cc-reset')[0]!.props.onClick()
  expect(wide.resets()).toBe(1)
})
