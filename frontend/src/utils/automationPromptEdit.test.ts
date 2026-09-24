import { describe, expect, it } from 'vitest'
import { composeEditedPrompt, isOverrideEcho, splitPrompt } from './automationPromptEdit'

/** AutomationPanel 의 편집 상태 전이를 그대로 흉내 낸다(basePrompt 는 메아리로 안 바뀐다). */
function panel(initial: string) {
  let basePrompt = initial
  let removed: number[] = []
  let added: string[] = []
  let lastSent = ''
  const reset = (next: string) => { basePrompt = next; removed = []; added = []; lastSent = '' }
  return {
    get final() { return composeEditedPrompt(splitPrompt(basePrompt), removed, added) },
    get base() { return basePrompt },
    remove(i: number) { removed = [...removed, i] },
    restore(i: number) { removed = removed.filter(x => x !== i) },
    add(t: string) { added = [...added, t] },
    send() {
      const edits = removed.length + added.length > 0
      lastSent = edits ? this.final : ''
      return lastSent
    },
    receive(next: string) { if (!isOverrideEcho(next, lastSent)) reset(next) },
  }
}

describe('automation prompt edit', () => {
  it('splits only on top-level commas', () => {
    expect(splitPrompt('a, (b, c:1.2), <lora:x:1>, d')).toEqual(['a', '(b, c:1.2)', '<lora:x:1>', 'd'])
    expect(splitPrompt('')).toEqual([])
  })

  it('does not apply edits twice when the backend echoes the override', () => {
    const p = panel('a, b, c')
    p.remove(1)
    p.add('d')
    expect(p.send()).toBe('a, c, d')
    p.receive('a, c, d')                    // 백엔드 메아리
    expect(p.base).toBe('a, b, c')         // 기준은 그대로
    expect(p.final).toBe('a, c, d')        // 예전: 'a, d, d'
    p.add('x')
    expect(p.send()).toBe('a, c, d, x')    // 예전: 'x, x' 처럼 중복
  })

  it('restoring every edit cancels the override without resurrecting other tags', () => {
    const p = panel('a, b, c')
    p.remove(1)
    p.send()
    p.receive('a, c')
    p.restore(1)
    expect(p.send()).toBe('')               // 편집 없음 → 덮어쓰기 취소
    p.receive('a, b, c')                    // 백엔드가 원래 프롬프트를 다시 보냄
    expect(p.final).toBe('a, b, c')
  })

  it('a genuinely new prompt resets the edits', () => {
    const p = panel('a, b')
    p.remove(0)
    p.send()
    p.receive('b')
    p.receive('x, y')                       // 다음 덱 프롬프트
    expect(p.base).toBe('x, y')
    expect(p.final).toBe('x, y')
  })

  it('an empty prompt is not mistaken for an echo when nothing was sent', () => {
    expect(isOverrideEcho('', '')).toBe(false)
    expect(isOverrideEcho('a', 'a')).toBe(true)
    expect(isOverrideEcho('a', 'b')).toBe(false)
  })
})
