import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useTagAutocomplete } from './useTagAutocomplete'
import type { TagSuggestion } from '../utils/tagSuggest'

/**
 * 자동완성 요청 수명주기(#39). 가짜 fetch 는 응답을 쥐고 있다가 원하는 순서로 도착시킨다 —
 * 실제 QWebChannel 응답이 비동기라 '닫은 뒤 도착' · '지운 뒤 옛 질의 도착'이 실제로 일어난다.
 */
function harness(delay = 250) {
  const calls: string[] = []
  const pending: Array<{ prefix: string; cb: (items: TagSuggestion[]) => void }> = []
  const ac = useTagAutocomplete({
    delay,
    fetch: (prefix, cb) => { calls.push(prefix); pending.push({ prefix, cb }) },
  })
  const answer = (prefix: string, tags: string[]) => {
    const i = pending.findIndex(p => p.prefix === prefix)
    const [p] = pending.splice(i, 1)
    p.cb(tags.map(tag => ({ tag })))
  }
  return { ac, calls, answer, pending }
}

describe('useTagAutocomplete', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('debounces input and shows the answer for the owning field', () => {
    const { ac, calls, answer } = harness()
    ac.request('lo', 'main')
    ac.request('lon', 'main')
    vi.advanceTimersByTime(249)
    expect(calls).toEqual([])
    vi.advanceTimersByTime(1)
    expect(calls).toEqual(['lon'])
    answer('lon', ['long_hair', 'long_sleeves'])
    expect(ac.items.value.map(i => i.tag)).toEqual(['long_hair', 'long_sleeves'])
    expect(ac.isOpenFor('main')).toBe(true)
    expect(ac.isOpenFor('character_input')).toBe(false)
  })

  it('closing cancels the pending debounce — no popup resurrects after blur/Enter', () => {
    const { ac, calls } = harness()
    ac.request('long', 'main')
    ac.close()                                // blur · 블록 추가 · Escape
    vi.advanceTimersByTime(1000)
    expect(calls).toEqual([])
    expect(ac.items.value).toEqual([])
  })

  it('drops an in-flight answer that arrives after close', () => {
    const { ac, answer } = harness()
    ac.request('long', '')
    vi.advanceTimersByTime(250)
    ac.close()
    answer('long', ['long_hair'])
    expect(ac.items.value).toEqual([])
    expect(ac.owner.value).toBe('')
  })

  it('going below the minimum length closes and invalidates the older query', () => {
    const { ac, calls, answer } = harness()
    ac.request('lo', 'main')
    vi.advanceTimersByTime(250)
    ac.request('l', 'main')                   // 'lo' → 'l' 로 지움: 최소 길이 미만
    answer('lo', ['long_hair'])
    expect(ac.items.value).toEqual([])        // 옛 'lo' 후보가 뜨지 않는다
    expect(calls).toEqual(['lo'])
  })

  it('a newer request wins over a slower older one', () => {
    const { ac, answer } = harness()
    ac.request('bl', 'main')
    vi.advanceTimersByTime(250)
    ac.request('blu', 'main')
    vi.advanceTimersByTime(250)
    answer('blu', ['blue_eyes'])
    answer('bl', ['black_hair'])              // 늦게 온 옛 응답
    expect(ac.items.value.map(i => i.tag)).toEqual(['blue_eyes'])
  })

  it('another field typing takes the popup over; stale answers for the first field are ignored', () => {
    const { ac, answer } = harness()
    ac.request('sm', 'main_prompt_text')
    vi.advanceTimersByTime(250)
    ac.request('ha', 'character_input')
    answer('sm', ['smile'])
    expect(ac.items.value).toEqual([])
    vi.advanceTimersByTime(250)
    answer('ha', ['hatsune_miku'])
    expect(ac.isOpenFor('character_input')).toBe(true)
    expect(ac.isOpenFor('main_prompt_text')).toBe(false)
  })

  // 'bl' 후보가 떠 있는 채로 'blue' 를 치고 새 응답 전에 Enter/Tab — 옛 후보가 들어가면 안 된다
  it('a new query hides shown candidates that do not match it until its own answer arrives', () => {
    const { ac, answer } = harness()
    ac.request('bl', 'main_prompt_text')
    vi.advanceTimersByTime(250)
    answer('bl', ['black_hair', 'blonde_hair'])
    expect(ac.isOpenFor('main_prompt_text')).toBe(true)
    ac.request('blu', 'main_prompt_text')
    ac.request('blue', 'main_prompt_text')
    expect(ac.isOpenFor('main_prompt_text')).toBe(false)
    expect(ac.items.value).toEqual([])
    expect(ac.selected()).toBeUndefined()
    vi.advanceTimersByTime(250)                // 디바운스가 끝나 요청이 날아가는 중에도 닫혀 있다
    expect(ac.items.value).toEqual([])
    answer('blue', ['blue_eyes'])
    expect(ac.items.value.map(i => i.tag)).toEqual(['blue_eyes'])
    expect(ac.isOpenFor('main_prompt_text')).toBe(true)
  })

  it('block mode (no owner) also drops the stale list — Enter then adds the typed text', () => {
    const { ac, answer } = harness()
    ac.request('bl')
    vi.advanceTimersByTime(250)
    answer('bl', ['black_hair'])
    ac.request('blu')
    ac.request('blue')
    vi.advanceTimersByTime(50)
    expect(ac.items.value.length).toBe(0)
    expect(ac.queryHangul.value).toBe(false)
  })

  it('keeps shown candidates that still match the longer query (spelling-insensitive)', () => {
    const { ac, answer } = harness()
    ac.request('lo', 'main')
    vi.advanceTimersByTime(250)
    answer('lo', ['looking_at_viewer', 'long_hair', 'long_sleeves'])
    ac.move(2)                                   // long_sleeves 선택
    ac.request('long s', 'main')
    expect(ac.items.value.map(i => i.tag)).toEqual(['long_sleeves'])
    expect(ac.selected()?.tag).toBe('long_sleeves')
    ac.request('long', 'main')                   // 지워서 짧아져도 여전히 맞는 후보만 남는다
    expect(ac.items.value.map(i => i.tag)).toEqual(['long_sleeves'])
  })

  it('repeating the same query in the same field keeps the shown answer', () => {
    const { ac, answer } = harness()
    ac.request('bl', 'main')
    vi.advanceTimersByTime(250)
    answer('bl', ['black_hair'])
    ac.request('bl ', 'main')                    // 뒤 공백 · IME 무변화 입력
    expect(ac.selected()?.tag).toBe('black_hair')
  })

  it('typing in another field hides the first field\'s list right away', () => {
    const { ac, answer } = harness()
    ac.request('sm', 'main_prompt_text')
    vi.advanceTimersByTime(250)
    answer('sm', ['smile'])
    ac.request('sm', 'character_input')
    expect(ac.isOpenFor('main_prompt_text')).toBe(false)
    expect(ac.isOpenFor('character_input')).toBe(false)
  })

  it('a hangul query never keeps latin candidates of an older query', () => {
    const { ac, answer } = harness()
    ac.request('lo', 'main')
    vi.advanceTimersByTime(250)
    answer('lo', ['long_hair'])
    ac.request('긴', 'main')
    expect(ac.items.value).toEqual([])
  })

  it('hangul starts at one syllable and flags the query as hangul', () => {
    const { ac, calls, answer } = harness()
    ac.request('장', '')
    vi.advanceTimersByTime(250)
    expect(calls).toEqual(['장'])
    answer('장', ['long_hair'])
    expect(ac.queryHangul.value).toBe(true)
  })

  it('moves the selection within bounds and survives a throwing or rejecting fetch', async () => {
    const { ac, answer } = harness()
    ac.request('lo', '')
    vi.advanceTimersByTime(250)
    answer('lo', ['a', 'b', 'c'])
    ac.move(5)
    expect(ac.selected()?.tag).toBe('c')
    ac.move(-9)
    expect(ac.selected()?.tag).toBe('a')

    const throwing = useTagAutocomplete({ delay: 10, fetch: () => { throw new Error('boom') } })
    throwing.request('lo', '')
    vi.advanceTimersByTime(10)
    expect(throwing.items.value).toEqual([])

    const rejecting = useTagAutocomplete({ delay: 10, fetch: () => Promise.reject(new Error('boom')) })
    rejecting.request('lo', '')
    vi.advanceTimersByTime(10)
    await Promise.resolve(); await Promise.resolve()
    expect(rejecting.items.value).toEqual([])
  })
})
