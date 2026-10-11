import { describe, expect, it } from 'vitest'
import cases from './promptComments.cases.json'
import {
  appendPromptText,
  commentSegments,
  commentSpans,
  endsInLineComment,
  hasPromptComment,
  isCaretInComment,
  isCommentOnly,
  joinPromptBlocks,
  splitPromptBlocks,
  stripPromptComments,
} from './promptComments'

/**
 * 프롬프트 주석은 파이썬(core/prompt_comments.py)이 요청에서 떼고, 여기서는 같은 문법으로 표시·자동완성 끄기·
 * 블록 나누기를 한다. 아래 골든 사례는 tests/test_prompt_comments.py 도 읽는다 — 두 구현이 갈라지면 한쪽이 실패한다.
 */
describe('골든 사례 — 파이썬과 같은 결과', () => {
  it('주석 구간', () => {
    expect(cases.strip.length).toBeGreaterThan(20)
    for (const c of cases.strip) {
      expect(commentSpans(c.text).map(s => [s.start, s.end, s.kind]), JSON.stringify(c.text)).toEqual(c.spans)
    }
  })

  it('주석 떼기', () => {
    for (const c of cases.strip) expect(stripPromptComments(c.text), JSON.stringify(c.text)).toBe(c.stripped)
  })

  it('줄 주석으로 끝나는가', () => {
    for (const c of cases.endsInLineComment) expect(endsInLineComment(c.text), JSON.stringify(c.text)).toBe(c.ends)
  })
})

describe('성질', () => {
  it('멱등 · 주석 없음 · 길어지지 않음 · / 가 없으면 그대로', () => {
    let seed = 20261011
    const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 }
    const alphabet = 'a /*,\n:\\()\r\t'
    for (let k = 0; k < 2000; k++) {
      let text = ''
      const len = Math.floor(rand() * 25)
      for (let j = 0; j < len; j++) text += alphabet[Math.floor(rand() * alphabet.length)]
      const once = stripPromptComments(text)
      expect(stripPromptComments(once), JSON.stringify(text)).toBe(once)
      expect(commentSpans(once), JSON.stringify(text)).toEqual([])
      expect(once.length).toBeLessThanOrEqual(text.length)
      if (!text.includes('/')) expect(once).toBe(text)
    }
  })

  it('URL·LoRA 경로·와일드카드는 주석이 아니다', () => {
    for (const t of ['http://e.com/a', 'a//b', '>//<', '__x/*__', '~/x/~', '<lora:d/n:1>', 'file:///p']) {
      expect(hasPromptComment(t), t).toBe(false)
    }
  })
})

describe('편집기 도우미', () => {
  it('커서가 주석 안인가 — 자동완성 끄기', () => {
    const t = 'a // note\nb'
    expect(isCaretInComment(t, 1)).toBe(false)
    expect(isCaretInComment(t, 2)).toBe(false)     // '//' 바로 앞
    expect(isCaretInComment(t, 3)).toBe(true)
    expect(isCaretInComment(t, 9)).toBe(true)      // 줄 끝 자리(입력 중인 끝)
    expect(isCaretInComment(t, 11)).toBe(false)    // 다음 줄
    const b = 'x /* c */ y'
    expect(isCaretInComment(b, 5)).toBe(true)
    expect(isCaretInComment(b, 9)).toBe(false)     // '*/' 뒤
    expect(isCaretInComment('a /* open', 9)).toBe(true)
    expect(isCaretInComment('http://x', 8)).toBe(false)
    expect(isCaretInComment('', 0)).toBe(false)
  })

  it('표시 조각과 주석만 있는 블록', () => {
    expect(commentSegments('smile // why')).toEqual([
      { text: 'smile ', comment: false, open: false },
      { text: '// why', comment: true, open: false },
    ])
    expect(commentSegments('a /* x')).toEqual([
      { text: 'a ', comment: false, open: false },
      { text: '/* x', comment: true, open: true },
    ])
    expect(commentSegments('plain')).toEqual([{ text: 'plain', comment: false, open: false }])
    expect(isCommentOnly('// only')).toBe(true)
    expect(isCommentOnly('/* a */ /* b */')).toBe(true)
    expect(isCommentOnly('a // b')).toBe(false)
    expect(isCommentOnly('plain')).toBe(false)
  })

  it('뒤에 붙이기 — 예전 규칙, 줄 주석 뒤만 줄바꿈', () => {
    expect(appendPromptText('a, ', 'b')).toBe('a, b')
    expect(appendPromptText('a', 'b')).toBe('a, b')
    expect(appendPromptText('', 'b')).toBe('b')
    expect(appendPromptText('a', '')).toBe('a')
    expect(appendPromptText('a // note', 'b')).toBe('a // note\nb')
    expect(appendPromptText('a // note, ', 'b')).toBe('a // note\nb')   // 쉼표가 주석 안에 있었다
    expect(stripPromptComments(appendPromptText('a // note', 'b'))).toBe('a\nb')
  })
})

describe('블록 나누기·잇기', () => {
  it('주석이 없으면 예전 규칙 그대로', () => {
    expect(splitPromptBlocks('a, {b, c}, d')).toEqual(['a', '{b, c}', 'd'])
    expect(splitPromptBlocks('a\nb, c')).toEqual(['a\nb', 'c'])
    expect(splitPromptBlocks('__a,b__, (x, y:1.2)')).toEqual(['__a,b__', '(x, y:1.2)'])
    expect(splitPromptBlocks('')).toEqual([])
    expect(joinPromptBlocks(['a', 'b'])).toBe('a, b')
  })

  it('줄 주석 뒤 줄바꿈은 블록 경계, 주석 안 쉼표·괄호는 세지 않는다', () => {
    const text = 'masterpiece, // quality, tags (x\nbest quality, 1girl /* a, b */ solo'
    const blocks = splitPromptBlocks(text)
    expect(blocks).toEqual(['masterpiece', '// quality, tags (x', 'best quality', '1girl /* a, b */ solo'])
    expect(joinPromptBlocks(blocks)).toBe('masterpiece, // quality, tags (x\nbest quality, 1girl /* a, b */ solo')
    expect(stripPromptComments(joinPromptBlocks(blocks))).toBe(stripPromptComments(text))
    expect(splitPromptBlocks('1girl // main\r\nsolo')).toEqual(['1girl // main', 'solo'])
    expect(splitPromptBlocks('(a // x\n, b), c')).toEqual(['(a // x\n, b)', 'c'])
  })

  it('블록을 지우거나 옮겨도 다음 블록이 주석에 먹히지 않는다', () => {
    const blocks = splitPromptBlocks('a // first\nb, c // third\nd')
    expect(blocks).toEqual(['a // first', 'b', 'c // third', 'd'])
    const moved = [blocks[2]!, blocks[0]!, blocks[3]!]
    expect(stripPromptComments(joinPromptBlocks(moved))).toBe('c\na\nd')
  })
})
