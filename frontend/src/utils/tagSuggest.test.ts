import { describe, expect, it } from 'vitest'
import {
  fetchTagSuggestions, hasHangul, isCaretMoveKey, isSameTagQuery, minSuggestPrefix, normalizeSuggestions,
  replaceTagToken, tagQueryAt, tagTokenAt,
} from './tagSuggest'

describe('tagSuggest', () => {
  it('detects hangul and lowers the prefix threshold for it', () => {
    expect(hasHangul('장발')).toBe(true)
    expect(hasHangul('long hair')).toBe(false)
    expect(minSuggestPrefix('장')).toBe(1)
    expect(minSuggestPrefix('l')).toBe(2)
  })

  it('normalises rich objects, plain strings and JSON text; dedupes and caps', () => {
    const rich = JSON.stringify([
      { tag: 'long hair', ko: '긴 머리', category: '패션 > 헤어스타일', count: 4800000 },
      { tag: 'long hair', ko: 'dup' },
      { tag: 'solo' },
      'highres',
      { nope: true },
      42,
    ])
    expect(normalizeSuggestions(rich)).toEqual([
      { tag: 'long hair', ko: '긴 머리', category: '패션 > 헤어스타일', desc: '', count: 4800000 },
      { tag: 'solo', ko: '', category: '', desc: '', count: 0 },
      { tag: 'highres' },
    ])
    expect(normalizeSuggestions(['a', 'b', 'c'], 2)).toHaveLength(2)
    expect(normalizeSuggestions('{not json')).toEqual([])
    expect(normalizeSuggestions(null)).toEqual([])
  })

  it('asks only the rich slot and never throws', () => {
    const calls: string[] = []
    const rich = {
      getTagSuggestionsRich: (p: string, cb: (j: string) => void) => { calls.push('rich:' + p); cb(JSON.stringify([{ tag: 'long hair', ko: '긴 머리' }])) },
    }
    let got: any = null
    fetchTagSuggestions(rich, 'lon', (items) => { got = items })
    expect(calls).toEqual(['rich:lon'])
    expect(got[0]).toMatchObject({ tag: 'long hair', ko: '긴 머리' })

    // 옛 문자열 전용 슬롯(getTagSuggestions)은 제거됐다 — 그것만 있는 백엔드는 후보 없음
    const legacyOnly = { getTagSuggestions: (_p: string, cb: (j: string) => void) => cb('["solo","highres"]') }
    got = null
    fetchTagSuggestions(legacyOnly, 'so', (items) => { got = items })
    expect(got).toEqual([])

    fetchTagSuggestions({}, 'x', (items) => { got = items })
    expect(got).toEqual([])
    fetchTagSuggestions({ getTagSuggestionsRich: () => { throw new Error('boom') } }, 'x', (items) => { got = items })
    expect(got).toEqual([])
  })
})

describe('tagTokenAt — 커서가 놓인 태그 조각', () => {
  it('uses the fragment under the caret, not the one after the last comma', () => {
    const text = '1girl, smile, o, blue sky'
    const caret = text.indexOf('o,') + 1
    expect(tagTokenAt(text, caret)).toEqual({ start: 14, end: 15, query: 'o' })
  })

  it('queries up to the caret but replaces the whole fragment', () => {
    const text = 'a, long_hair, b'
    const caret = text.indexOf('_h') + 2          // 'long_h|air'
    const token = tagTokenAt(text, caret)
    expect(token.query).toBe('long_h')
    expect(text.slice(token.start, token.end)).toBe('long_hair')
  })

  it('treats newlines as separators and ignores padding', () => {
    const text = 'a,\n  sm  , b'
    const token = tagTokenAt(text, text.indexOf('sm') + 2)
    expect(token.query).toBe('sm')
    expect(text.slice(token.start, token.end)).toBe('sm')
  })

  it('defaults to the end and returns an empty query right after a separator', () => {
    expect(tagTokenAt('a, lo').query).toBe('lo')
    expect(tagTokenAt('a, lo', 999).query).toBe('lo')
    expect(tagTokenAt('a, ', 3).query).toBe('')
    expect(tagTokenAt('', 0)).toEqual({ start: 0, end: 0, query: '' })
    // 쉼표로 끝나는 텍스트에서도 중간 편집은 자동완성이 된다
    const text = '1girl, sm, blue sky, '
    expect(tagTokenAt(text, text.indexOf('sm') + 2).query).toBe('sm')
  })
})

describe('replaceTagToken', () => {
  it('replaces a middle fragment without adding a duplicate comma', () => {
    const text = '1girl, smile, o, blue sky'
    const token = tagTokenAt(text, text.indexOf('o,') + 1)
    const out = replaceTagToken(text, token, 'orange_hair')
    expect(out.text).toBe('1girl, smile, orange_hair, blue sky')
    expect(out.caret).toBe('1girl, smile, orange_hair'.length)
  })

  it('appends ", " at the end of the text (previous behaviour)', () => {
    const text = 'a, lo  '
    const out = replaceTagToken(text, tagTokenAt(text, 5), 'long_hair')
    expect(out.text).toBe('a, long_hair, ')
    expect(out.caret).toBe(out.text.length)
    expect(replaceTagToken('lo', tagTokenAt('lo'), 'long_hair').text).toBe('long_hair, ')
  })

  it('inserts a space after a bare comma and keeps a following newline', () => {
    expect(replaceTagToken('a,lo', tagTokenAt('a,lo'), 'long_hair').text).toBe('a, long_hair, ')
    const text = 'a, lo\nb'
    const out = replaceTagToken(text, tagTokenAt(text, 5), 'long_hair')
    expect(out.text).toBe('a, long_hair\nb')
    expect(out.caret).toBe('a, long_hair'.length)
  })
})

describe('tagSuggest — 요청한 조각 그대로일 때만 수락 (#39 커서)', () => {
  const text = '1girl, smile, lon'

  it('records the token the suggestions were asked for', () => {
    expect(tagQueryAt(text, text.length)).toEqual({ text, start: 14, query: 'lon' })
    expect(tagQueryAt(text)).toEqual({ text, start: 14, query: 'lon' })
  })

  it('accepts while the caret is still at the queried token', () => {
    const q = tagQueryAt(text, text.length)
    expect(isSameTagQuery(q, text, text.length)).toBe(true)
  })

  it('rejects after the caret moved by keyboard (Home) while the answer was in flight', () => {
    // 'lon' 을 치고 300ms 안에 Home → 도착한 'lon' 후보로 Enter 하면 '1girl' 이 바뀌던 경로
    const q = tagQueryAt(text, text.length)
    expect(isSameTagQuery(q, text, 0)).toBe(false)
    expect(isSameTagQuery(q, text, 9)).toBe(false)               // 'smile' 안
    expect(isSameTagQuery(q, text, text.length - 1)).toBe(false) // 같은 조각이지만 질의가 'lo' 로 줄었다
  })

  it('rejects when the text changed underneath (undo, backend update)', () => {
    const q = tagQueryAt(text, text.length)
    expect(isSameTagQuery(q, '1girl, smile, long', 18)).toBe(false)
    expect(isSameTagQuery(q, '1girl, lon', 10)).toBe(false)
    expect(isSameTagQuery(null, text, text.length)).toBe(false)
  })

  it('knows which keys only move the caret', () => {
    for (const key of ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']) {
      expect(isCaretMoveKey(key), key).toBe(true)
    }
    for (const key of ['a', 'Enter', 'Tab', 'Escape', 'Backspace', 'Process', '', undefined]) {
      expect(isCaretMoveKey(key), String(key)).toBe(false)
    }
  })
})
