import { describe, expect, it } from 'vitest'
import { diffParameters } from './paramDiff'

describe('diffParameters', () => {
  it('compares parsed values, so quoted commas never create fake rows', () => {
    const before = { Steps: 20, 'ADetailer prompt': 'smile, blue eyes', 'Lora hashes': 'a: 1, b: 2' }
    const after = { Steps: 28, 'ADetailer prompt': 'smile, blue eyes', 'Lora hashes': 'a: 1, b: 2' }
    const rows = diffParameters(before, after)
    expect(rows.map(r => r.key)).toEqual(['Steps', 'ADetailer prompt', 'Lora hashes'])
    expect(rows[0]).toEqual({ key: 'Steps', before: '20', after: '28', changed: true })
    expect(rows[1]).toEqual({ key: 'ADetailer prompt', before: 'smile, blue eyes', after: 'smile, blue eyes', changed: false })
  })

  it('shows keys present on only one side and tolerates missing dicts', () => {
    expect(diffParameters({ Seed: 1 }, { Size: '512x512' })).toEqual([
      { key: 'Seed', before: '1', after: '', changed: true },
      { key: 'Size', before: '', after: '512x512', changed: true },
    ])
    expect(diffParameters(undefined, null)).toEqual([])
  })
})
