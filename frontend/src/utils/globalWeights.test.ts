import { describe, expect, it } from 'vitest'
import { applyGlobalWeights, createRowKeyer, formatWeight, normalizeTagKey, weightMap } from './globalWeights'

const w = (tag: string, weight: number) => ({ tag, weight })

describe('applyGlobalWeights (#109)', () => {
  it('never throws on regex-special emoticon tags and weights the exact token only', () => {
    expect(() => applyGlobalWeights('1girl, :(, smile', [w(':(', 120)])).not.toThrow()
    expect(applyGlobalWeights('1girl, :(, smile', [w(':(', 120)])).toBe('1girl, (:\\(:1.20), smile')
    expect(applyGlobalWeights('1girl, +_+, smile', [w('+_+', 120)])).toBe('1girl, (+_+:1.20), smile')
    expect(applyGlobalWeights('^^^, 1girl', [w('^^^', 130)])).toBe('(^^^:1.30), 1girl')
  })

  it('does not treat dots as wildcards', () => {
    expect(applyGlobalWeights('cxcx, c.c.', [w('c.c.', 120)])).toBe('cxcx, (c.c.:1.20)')
    expect(applyGlobalWeights('cxcx', [w('c.c.', 120)])).toBe('cxcx')
  })

  it('matches parenthesised tags, escaped or not, and keeps the original spelling', () => {
    expect(applyGlobalWeights('hatsune miku \\(cosplay\\), smile', [w('hatsune miku (cosplay)', 110)]))
      .toBe('(hatsune miku \\(cosplay\\):1.10), smile')
    expect(applyGlobalWeights('Long_Hair, smile', [w('long hair', 120)])).toBe('(Long_Hair:1.20), smile')
  })

  it('replaces an existing weight instead of nesting, and is idempotent', () => {
    const once = applyGlobalWeights('a, (long hair:1.05), b', [w('long_hair', 120)])
    expect(once).toBe('a, (long hair:1.20), b')
    expect(applyGlobalWeights(once, [w('long_hair', 120)])).toBe(once)
    const emo = applyGlobalWeights('x, :(', [w(':(', 120)])
    expect(applyGlobalWeights(emo, [w(':(', 150)])).toBe('x, (:\\(:1.50)')
  })

  it('leaves groups, LoRA, BREAK and 1.00 weights alone; keeps separators', () => {
    const text = '<lora:foo:0.8>, (smile, blush:1.1),\nBREAK, smile, 1girl'
    expect(applyGlobalWeights(text, [w('smile', 120), w('BREAK', 150), w('1girl', 100)]))
      .toBe('<lora:foo:0.8>, (smile, blush:1.1),\nBREAK, (smile:1.20), 1girl')
    expect(applyGlobalWeights('smile', [])).toBe('smile')
    expect(applyGlobalWeights('', [w('smile', 120)])).toBe('')
  })

  it('last duplicate entry wins; blank tags are ignored', () => {
    expect(weightMap([w('smile', 120), w('Smile', 140), w('  ', 150)])).toEqual(new Map([['smile', '1.40']]))
    expect(weightMap([w('smile', 120), w('smile', 100)]).size).toBe(0)
  })

  it('formats and normalises', () => {
    expect(formatWeight(120)).toBe('1.20')
    expect(formatWeight(NaN)).toBe('1.00')
    expect(normalizeTagKey('  Hatsune_Miku  \\(Cosplay\\) ')).toBe('hatsune miku (cosplay)')
  })
})

describe('createRowKeyer', () => {
  it('gives each row object a stable key regardless of its tag text', () => {
    const key = createRowKeyer()
    const a = { tag: '', weight: 100 }
    const b = { tag: '', weight: 100 }
    const ka = key(a)
    a.tag = 'bl'
    expect(key(a)).toBe(ka)          // 입력 중 key 가 바뀌지 않는다 → 재마운트·포커스 유실 없음
    expect(key(b)).not.toBe(ka)      // 같은 태그/빈 태그 두 줄도 key 가 겹치지 않는다
  })
})
