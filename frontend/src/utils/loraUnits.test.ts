import { describe, expect, it } from 'vitest'
import { fromBridgeLoraEntries, toBridgeLoraEntries } from './loraUnits'

describe('LoRA weight units (Vue percent <-> bridge multiplier)', () => {
  it('sends percent stack as multipliers, including an empty stack', () => {
    expect(toBridgeLoraEntries([
      { name: 'a', weight: 85, enabled: true, triggerWords: ['tw'] },
      { name: 'b', weight: 5, enabled: false, triggerWords: [] },
    ])).toEqual([
      { name: 'a', weight: 0.85, enabled: true, triggerWords: ['tw'] },
      { name: 'b', weight: 0.05, enabled: false, triggerWords: [] },
    ])
    // 전부 지운 스택도 '빈 목록'으로 보내야 Python 이 옛 LoRA 를 버린다
    expect(toBridgeLoraEntries([])).toEqual([])
    expect(toBridgeLoraEntries(null)).toEqual([])
  })

  it('falls back to 0.8x for missing or non-numeric weights', () => {
    const [entry] = toBridgeLoraEntries([{ name: 'x', weight: Number.NaN }])
    expect(entry).toEqual({ name: 'x', weight: 0.8, enabled: true, triggerWords: [] })
  })

  it('converts profile/bridge multipliers back to integer percent once (no 100x)', () => {
    expect(fromBridgeLoraEntries([
      { name: 'a', weight: 0.95, enabled: true, triggerWords: [] },
      { name: 'b', weight: -0.4 },
    ])).toEqual([
      { name: 'a', weight: 95, enabled: true, triggerWords: [] },
      { name: 'b', weight: -40, enabled: true, triggerWords: [] },
    ])
    expect(fromBridgeLoraEntries('nope')).toBeNull()
  })

  it('round-trips without drift', () => {
    const stack = [{ name: 'a', weight: 65, enabled: true, triggerWords: [] }]
    expect(fromBridgeLoraEntries(toBridgeLoraEntries(stack))).toEqual(stack)
  })
})
