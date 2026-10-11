import { describe, expect, it } from 'vitest'
import {
  COMPOSITION_PRESETS, DEFAULT_COMPOSITION, compositionCameraPoint, compositionTags,
  dragComposition, normalizeComposition, planCompositionAppend, splitCompositionPrompt,
} from './compositionPrompt'

describe('independent composition prompt steering', () => {
  it('normalizes corrupt or obsolete preferences without accepting executable values', () => {
    expect(normalizeComposition(null)).toEqual(DEFAULT_COMPOSITION)
    expect(normalizeComposition({ azimuth: Infinity, elevation: 100, distance: -4, roll: '24', framing: NaN }))
      .toEqual({ azimuth: 0, elevation: 75, distance: 0, roll: 0, framing: 0 })
  })
  it('uses neutral, unweighted descriptive defaults', () => {
    expect(compositionTags({ ...DEFAULT_COMPOSITION })).toEqual(['facing viewer', 'upper body', 'centered composition'])
    for (const preset of COMPOSITION_PRESETS) expect(compositionTags(preset.state).join(',')).not.toMatch(/:\d|masterpiece|1girl/)
  })
  it.each([
    [-180, 'from behind'], [-150, "rear three-quarter view from the subject's left"],
    [-90, 'profile'], [-40, "three-quarter view from the subject's left"],
    [0, 'facing viewer'], [40, "three-quarter view from the subject's right"], [180, 'from behind'],
  ])('maps azimuth %s to descriptive prompt %s', (azimuth, expected) => {
    expect(compositionTags({ ...DEFAULT_COMPOSITION, azimuth })).toContain(expected)
  })
  it('keeps contradictory elevation and crop descriptions mutually exclusive', () => {
    const high = compositionTags({ ...DEFAULT_COMPOSITION, elevation: 70, distance: 100, roll: -12, framing: -50 })
    expect(high).toEqual(expect.arrayContaining(['from above', "bird's eye view", 'wide shot', 'dutch angle', 'frame tilted counterclockwise', 'subject on the left side of the frame']))
    expect(high).not.toContain('from below')
    expect(high).not.toContain('close-up')
    expect(compositionTags({ ...DEFAULT_COMPOSITION, elevation: -40, distance: 5 })).toContain('from below')
    expect(compositionTags({ ...DEFAULT_COMPOSITION, roll: 5 })).not.toContain('dutch angle')
  })
  it.each([[0, 'close-up'], [20, 'portrait'], [45, 'upper body'], [60, 'cowboy shot'], [75, 'full body'], [100, 'wide shot']])
    ('maps distance %s to exactly one crop %s', (distance, expected) => {
      expect(compositionTags({ ...DEFAULT_COMPOSITION, distance })).toContain(expected)
    })
  it('splits only top-level commas without changing schedules, weights or escaped literals', () => {
    expect(splitCompositionPrompt('a, (b, c:1.1), [d:e:0.5], {f|g,h}, <lora:a,b:1>, escaped\\,comma'))
      .toEqual(['a', '(b, c:1.1)', '[d:e:0.5]', '{f|g,h}', '<lora:a,b:1>', 'escaped\\,comma'])
  })
  it('preserves existing text byte-for-byte and deduplicates normalized complete tags across sections', () => {
    const original = 'rain, (from_above:1.15), (city, trees:1.2), \\(sign\\),  '
    const result = planCompositionAppend(original, ['from above', 'full body', 'from above', 'wide shot'], '((full_body)), wide_shot')
    expect(result.additions).toEqual([])
    expect(result.text).toBe(original)
    expect(planCompositionAppend('cityscape', ['city']).text).toBe('cityscape, city')
    expect(planCompositionAppend(original, ['dutch angle']).text).toBe(original + 'dutch angle')
  })
  it('warns about opposed prior tags instead of silently removing or replacing them', () => {
    const result = planCompositionAppend('from below, close-up', ['from above', 'full body'], 'from_behind')
    expect(result.conflicts).toEqual(['from below ↔ from above', 'close-up ↔ full body'])
    expect(result.text).toBe('from below, close-up, from above, full body')
  })
  it('does not treat part of a weighted multi-tag group as a removable duplicate', () => {
    const result = planCompositionAppend('(from above, blue sky:1.1)', ['from above'])
    expect(result.text).toBe('(from above, blue sky:1.1), from above')
  })
  it('ignores tags inside prompt comments and never appends into a line comment', () => {
    const result = planCompositionAppend('forest, // from above', ['from above'])
    expect(result.additions).toEqual(['from above'])
    expect(result.text).toBe('forest, // from above\nfrom above')
    expect(planCompositionAppend('forest /* full body */', ['full body']).text).toBe('forest /* full body */, full body')
    expect(planCompositionAppend('forest', ['from above'], '/* from above */').additions).toEqual(['from above'])
  })
  it('is idempotent on repeated apply including a trailing comma', () => {
    const tags = compositionTags({ ...DEFAULT_COMPOSITION })
    const first = planCompositionAppend('forest,', tags)
    expect(first.text).toBe('forest, facing viewer, upper body, centered composition')
    expect(planCompositionAppend(first.text, tags).additions).toEqual([])
    expect(planCompositionAppend('', tags).text.startsWith(',')).toBe(false)
  })
  it('computes gestures from the starting state and clamps rapid out-of-bounds input', () => {
    const initial = { ...DEFAULT_COMPOSITION }
    expect(dragComposition(initial, 100, -50)).toMatchObject({ azimuth: 70, elevation: 30 })
    expect(dragComposition(initial, 0, 0)).toEqual(initial)
    expect(dragComposition(initial, 9999, -9999)).toMatchObject({ azimuth: 180, elevation: 75 })
    expect(initial).toEqual(DEFAULT_COMPOSITION)
  })
  it('keeps camera diagram geometry in bounds across extreme controls', () => {
    for (const azimuth of [-180, -90, 0, 90, 180]) {
      for (const elevation of [-75, 0, 75]) {
        for (const distance of [0, 100]) {
          const point = compositionCameraPoint({ ...DEFAULT_COMPOSITION, azimuth, elevation, distance })
          expect(point.x).toBeGreaterThan(15); expect(point.x).toBeLessThan(285)
          expect(point.y).toBeGreaterThan(15); expect(point.y).toBeLessThan(155)
        }
      }
    }
  })
})
