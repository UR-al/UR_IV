import { describe, expect, it } from 'vitest'
import { applyEventCarry, classifyEventTag, effectiveStepPrompt } from './eventCarry'

const ALL_ON = { appearance: true, costume: true, background: true }
const ALL_OFF = { appearance: false, costume: false, background: false }

describe('eventCarry', () => {
  it('classifies tags by keyword with appearance > costume > background priority', () => {
    expect(classifyEventTag('long hair')).toBe('appearance')
    expect(classifyEventTag('Blue Eyes')).toBe('appearance')
    expect(classifyEventTag('school uniform')).toBe('costume')
    expect(classifyEventTag('outdoors')).toBe('background')
    expect(classifyEventTag('smile')).toBe('other')
    // 'hair' 가 'hairclip' 보다 먼저 걸린다 — 기존 분류 순서 보존
    expect(classifyEventTag('hairclip')).toBe('appearance')
  })

  it('leaves the parent step untouched and carries enabled categories into children', () => {
    const steps = [
      { prompt: '1girl, long hair, school uniform, outdoors, smile', type: 'parent' },
      { prompt: '1girl, crying', type: 'child' },
    ]
    const out = applyEventCarry(steps, ALL_ON)
    expect(out[0].displayPrompt).toBe(steps[0].prompt)
    expect(out[1].displayPrompt).toBe('1girl, crying, long hair, school uniform, outdoors')
    // 원본 필드는 보존
    expect(out[1].prompt).toBe('1girl, crying')
    expect(out[1].type).toBe('child')

    const onlyCostume = applyEventCarry(steps, { ...ALL_OFF, costume: true })
    expect(onlyCostume[1].displayPrompt).toBe('1girl, crying, school uniform')

    const none = applyEventCarry(steps, ALL_OFF)
    expect(none[1].displayPrompt).toBe('1girl, crying')
  })

  it('does not duplicate tags the child already has (case-insensitive) or parent duplicates', () => {
    const steps = [
      { prompt: 'Long Hair, long hair, dress' },
      { prompt: 'long hair, sad' },
    ]
    expect(applyEventCarry(steps, ALL_ON)[1].displayPrompt).toBe('long hair, sad, dress')
  })

  it('returns an empty list for no steps and tolerates missing prompts', () => {
    expect(applyEventCarry([], ALL_ON)).toEqual([])
    const out = applyEventCarry([{}, { prompt: '' }], ALL_ON)
    expect(out[0].displayPrompt).toBe('')
    expect(out[1].displayPrompt).toBe('')
  })

  it('effectiveStepPrompt prefers the carried prompt — T2I send must match the card', () => {
    const steps = [
      { prompt: '1girl, twintails, kimono' },
      { prompt: '1girl, running' },
    ]
    const shown = applyEventCarry(steps, ALL_ON)
    // 카드 표시·큐·T2I 전송이 같은 함수로 같은 문자열을 얻는다
    expect(effectiveStepPrompt(shown[1])).toBe('1girl, running, twintails, kimono')
    expect(effectiveStepPrompt({ prompt: 'raw only' })).toBe('raw only')
    expect(effectiveStepPrompt({ prompt: 'raw', displayPrompt: '' })).toBe('raw')
    expect(effectiveStepPrompt(null)).toBe('')
    expect(effectiveStepPrompt(undefined)).toBe('')
  })
})
