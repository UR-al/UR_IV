import { describe, expect, it } from 'vitest'
import {
  createdWildcard,
  dunderWildcardOk,
  editableLines,
  insertWildcardSyntax,
  isWildcardComment,
  savedEntryLines,
  savedWildcardName,
  wildcardContent,
  wildcardFailureMessage,
  wildcardReplyOk,
  wildcardSyntax,
  wildcardSyntaxForms,
} from './wildcardFile'

describe('wildcard file rules', () => {
  it('treats blank and #-prefixed lines as comments (leading spaces ignored)', () => {
    expect(isWildcardComment('# note')).toBe(true)
    expect(isWildcardComment('   # indented note')).toBe(true)
    expect(isWildcardComment('   ')).toBe(true)
    expect(isWildcardComment('red hair, blue hair')).toBe(false)
  })

  it('edits the raw lines so comments survive a save', () => {
    const entry = { name: 'hair', file: 'hair.txt', tags: ['red hair'], lines: ['# colors', 'red hair'] }
    const lines = editableLines(entry)
    expect(lines).toEqual(['# colors', 'red hair'])
    lines.push('blue hair')
    expect(entry.lines).toEqual(['# colors', 'red hair'])   // 복사본을 고친다
    expect(wildcardContent(lines)).toBe('# colors\nred hair\nblue hair')
  })

  it('falls back to tags when an older backend sends no raw lines', () => {
    expect(editableLines({ name: 'x', file: 'x.txt', tags: ['a', 'b'] })).toEqual(['a', 'b'])
    expect(editableLines(null)).toEqual([])
  })

  it('keeps interior blank lines but trims trailing ones', () => {
    expect(wildcardContent(['a', '', '# c', 'b', '', ''])).toBe('a\n\n# c\nb')
    expect(savedEntryLines(['a', '', '# c', 'b', ''])).toEqual({
      lines: ['a', '', '# c', 'b'],
      tags: ['a', 'b'],
    })
    expect(savedEntryLines(['', ''])).toEqual({ lines: [], tags: [] })
  })

  it('inserts the dunder syntax the Python resolver understands', () => {
    expect(wildcardSyntax('hairstyle')).toBe('__hairstyle__')
    expect(wildcardSyntax('hair_style')).toBe('__hair_style__')      // 가운데 밑줄 하나는 된다
    expect(wildcardSyntaxForms('hair_style')).toEqual(['__hair_style__', '~/hair_style/~'])
  })

  it('falls back to the tilde syntax for names the dunder form cannot express', () => {
    // 앞의 네 이름은 tests/test_file_wildcard.py test_names_dunder_cannot_express_resolve_with_the_tilde_form
    // 이 파이썬 해석기로 고정한다(~/이름/~ 은 풀리고 __이름__ 은 안 풀린다)
    for (const name of ['_base', 'base_', 'a,b', 'a__b', ' lead', 'trail ']) {
      expect(dunderWildcardOk(name)).toBe(false)
      expect(wildcardSyntax(name)).toBe(`~/${name}/~`)
      expect(wildcardSyntaxForms(name)).toEqual([`~/${name}/~`])
    }
    expect(dunderWildcardOk('')).toBe(false)
    expect(dunderWildcardOk('a b')).toBe(true)
  })

  it('reads the create-only response and never treats an existing file as new', () => {
    expect(createdWildcard('{"ok":true,"name":"mood","created":true}')).toEqual({ name: 'mood', created: true })
    expect(createdWildcard('{"ok":true,"name":"hairstyle","created":false}')).toEqual({ name: 'hairstyle', created: false })
    expect(createdWildcard('{"error":"bad name"}')).toBeNull()
    expect(createdWildcard('{"ok":true}')).toBeNull()
    expect(createdWildcard('nope')).toBeNull()
  })

  it('reads the sanitized name the backend actually saved', () => {
    expect(savedWildcardName('{"ok":true,"name":"hair"}', 'hair?')).toBe('hair')
    expect(savedWildcardName('{"ok":true}', 'hair')).toBe('hair')
    expect(savedWildcardName('{"error":"exists"}', 'hair')).toBeNull()
    expect(savedWildcardName('not json', 'hair')).toBeNull()
  })

  it('reads a name-less reply (delete) as ok only for {ok:true} without an error', () => {
    expect(wildcardReplyOk('{"ok":true}')).toBe(true)
    expect(wildcardReplyOk('{"ok":true,"error":"x"}')).toBe(false)
    expect(wildcardReplyOk('{"error":"[WinError 5] denied"}')).toBe(false)
    for (const bad of ['', 'not json', 'null', '{}', '[]', '"ok"', 'true']) expect(wildcardReplyOk(bad)).toBe(false)
    expect(wildcardReplyOk(undefined as unknown as string)).toBe(false)   // 응답 없는 콜백
  })

  it('builds the failure toast with the backend error when there is one', () => {
    expect(wildcardFailureMessage('{"error":"[WinError 32] in use"}', '와일드카드 삭제 실패'))
      .toBe('와일드카드 삭제 실패: [WinError 32] in use')
    expect(wildcardFailureMessage('{"ok":false}', '실패')).toBe('실패')
    expect(wildcardFailureMessage('not json', '실패')).toBe('실패')
    expect(wildcardFailureMessage('', '실패')).toBe('실패')
    expect(wildcardFailureMessage('null', '실패')).toBe('실패')
  })
})

describe('insertWildcardSyntax', () => {
  it('appends ", syntax, " after tidying a trailing comma/space, or starts an empty field', () => {
    expect(insertWildcardSyntax('', '__hair__')).toBe('__hair__, ')
    expect(insertWildcardSyntax('1girl', '__hair__')).toBe('1girl, __hair__, ')
    expect(insertWildcardSyntax('1girl, ', '__hair__')).toBe('1girl, __hair__, ')
    expect(insertWildcardSyntax('1girl,   ', '~/_x/~')).toBe('1girl, ~/_x/~, ')
    // 쉼표는 하나만 정리한다 — 사용자가 남긴 빈 태그 자리는 건드리지 않는다
    expect(insertWildcardSyntax('1girl,, ', '__a__')).toBe('1girl,, __a__, ')
  })
})
