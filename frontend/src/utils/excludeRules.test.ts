import { describe, expect, it } from 'vitest'
import cases from './excludeRules.cases.json'
import {
  EXCLUDE_RULE_FORMS,
  addExactExcludeRule,
  appendExcludeRule,
  buildExcludeKeepIndex,
  excludeRuleColorClass,
  excludeRuleSpans,
  keepToggleHint,
  normalizeExcludeTag,
  parseExcludeRule,
  pyStrip,
  rewriteExcludeRules,
  splitExcludeRules,
  toggleKeepExact,
} from './excludeRules'

/**
 * 제외 규칙은 파이썬(core/exclude_rules.py)이 적용하고, 관리 창·블록 칸은 여기서 나누고 가린다.
 * 아래 골든 사례는 tests/test_exclude_rules.py 도 읽는다 — 두 파서가 갈라지면 한쪽이 실패한다.
 */
describe('골든 사례 — 파이썬 파서와 같은 결과', () => {
  it('나누기', () => {
    expect(cases.split.length).toBeGreaterThan(5)
    for (const c of cases.split) expect(splitExcludeRules(c.text), JSON.stringify(c.text)).toEqual(c.rules)
  })

  it('분류 (9종 모두)', () => {
    const seen = new Set<string>()
    for (const c of cases.parse) {
      const parsed = parseExcludeRule(c.rule)
      if (c.form === null) {
        expect(parsed, JSON.stringify(c.rule)).toBeNull()
        continue
      }
      expect(parsed, JSON.stringify(c.rule)).not.toBeNull()
      expect(parsed!.form, c.rule).toBe(c.form)
      expect(parsed!.keyword, c.rule).toBe((c as { keyword?: string }).keyword)
      expect(parsed!.text).toBe(pyStrip(c.rule))
      seen.add(parsed!.form)
    }
    expect([...seen].sort()).toEqual(Object.keys(EXCLUDE_RULE_FORMS).sort())
  })

  it('예외 판정 — 관리 창 표시가 적용(ExcludeRuleSet)과 같은 태그를 유지로 본다', () => {
    expect(cases.kept.length).toBeGreaterThan(5)
    for (const c of cases.kept) {
      expect(buildExcludeKeepIndex(c.text).keptBy(c.tag) !== null, `${JSON.stringify(c.text)} / ${c.tag}`).toBe(c.kept)
    }
  })
})

describe('splitExcludeRules', () => {
  it('공백으로 나누지 않고 밑줄을 바꾸지 않는다 (쉼표 없는 규칙 하나)', () => {
    expect(splitExcludeRules('_short')).toEqual(['_short'])
    expect(splitExcludeRules('~_tank_top')).toEqual(['~_tank_top'])
    expect(splitExcludeRules('long hair')).toEqual(['long hair'])
  })

  it('줄바꿈도 구분자다 — 참고 목록 여러 줄을 붙여 넣어도 줄 끝 규칙이 합쳐지지 않는다', () => {
    expect(splitExcludeRules('watermark, *qr code\nborder, *2koma')).toEqual(['watermark', '*qr code', 'border', '*2koma'])
  })

  it('문자열이 아니면 빈 목록', () => {
    expect(splitExcludeRules(undefined)).toEqual([])
    expect(splitExcludeRules(null)).toEqual([])
  })

  it('줄 끝에 표시만 있는 조각은 쉼표 전의 다음 줄과 한 규칙 — 유지 규칙이 포함 제외로 뒤집히지 않는다', () => {
    expect(splitExcludeRules('x, ~\nsolo')).toEqual(['x', '~\nsolo'])
    expect(parseExcludeRule('~\nsolo')).toMatchObject({ form: 'keep_exact', keyword: 'solo' })
    expect(splitExcludeRules('x, ~,\nsolo')).toEqual(['x', '~', 'solo'])   // 쉼표 뒤로는 잇지 않는다
    expect(splitExcludeRules('solo\n_')).toEqual(['solo', '_'])            // 앞 줄로도 잇지 않는다
  })
})

describe('excludeRuleSpans', () => {
  it('규칙 자리는 원문 위치이고 앞뒤 공백을 뺀다', () => {
    const text = '  a ,\n~\n solo  , b'
    const spans = excludeRuleSpans(text)
    expect(spans.map(s => text.slice(s.start, s.end))).toEqual(['a', '~\n solo', 'b'])
    expect(spans[0]).toEqual({ start: 2, end: 3 })
    expect(excludeRuleSpans(42)).toEqual([])
  })
})

describe('parseExcludeRule', () => {
  it('비교 방식과 유지 여부', () => {
    expect(parseExcludeRule('_short')).toEqual({ text: '_short', form: 'suffix', match: 'suffix', keep: false, keyword: 'short' })
    expect(parseExcludeRule('~_x_')).toMatchObject({ form: 'keep_contains', match: 'contains', keep: true, keyword: 'x' })
    expect(parseExcludeRule('_tank_top_')).toMatchObject({ form: 'contains_explicit', match: 'contains', keep: false })
    expect(parseExcludeRule(42)).toBeNull()
  })

  it('파이썬 strip 과 같은 공백 집합', () => {
    // JS trim 은 BOM 을 자르지만 파이썬 strip 은 남긴다 — 규칙 수가 두 쪽에서 같아야 한다
    expect(pyStrip('﻿short ')).toBe('﻿short')
    expect(pyStrip('\x1cshort\x85')).toBe('short')
    expect(splitExcludeRules('﻿, a')).toEqual(['﻿', 'a'])
  })
})

describe('normalizeExcludeTag', () => {
  it('밑줄→공백, 앞뒤 공백 제거, 소문자', () => {
    expect(normalizeExcludeTag('  Blue_Hair ')).toBe('blue hair')
    expect(normalizeExcludeTag('_hair_')).toBe('hair')
  })
})

describe('관리 창 — ~유지 / *완전일치 규칙', () => {
  it('buildExcludeKeepIndex 는 표기 차이는 같게, 부분 문자열은 다르게, 패턴 예외도 유지로 본다', () => {
    expect(buildExcludeKeepIndex('short, ~Long Hair').keptBy('long_hair')).toMatchObject({ form: 'keep_exact', text: '~Long Hair' })
    expect(buildExcludeKeepIndex('~long_hair').keptBy('Long_Hair')).toMatchObject({ form: 'keep_exact' })
    // 옛 includes('~long_hair') 검사는 이것을 유지로 착각했다 — 'long hair ornament' 완전 일치 유지다
    expect(buildExcludeKeepIndex('~long_hair_ornament').keptBy('long_hair')).toBeNull()
    // 접두 유지 규칙도 long hair 를 지킨다 — 적용이 유지하는 태그는 초록으로 보인다
    expect(buildExcludeKeepIndex('~long_hair_').keptBy('long_hair')).toMatchObject({ form: 'keep_prefix' })
    expect(buildExcludeKeepIndex('~_hair').keptBy('long_hair')).toMatchObject({ form: 'keep_suffix' })
    expect(buildExcludeKeepIndex('~_ng ha_').keptBy('long_hair')).toMatchObject({ form: 'keep_contains' })
    // ~완전일치가 패턴보다 먼저 — 클릭(해제)이 무엇을 빼는지와 같은 규칙을 보여 준다
    expect(buildExcludeKeepIndex('~long_, ~long hair').keptBy('long_hair')).toMatchObject({ form: 'keep_exact' })
    // 제외 규칙·빈 키워드·빈 태그는 유지가 아니다
    expect(buildExcludeKeepIndex('long hair, *long hair, ~, ~_').keptBy('long_hair')).toBeNull()
    expect(buildExcludeKeepIndex('~_hair').keptBy('  ')).toBeNull()
    expect(buildExcludeKeepIndex('').keptBy('long_hair')).toBeNull()
    expect(buildExcludeKeepIndex(undefined).keptBy('long_hair')).toBeNull()
  })

  it('buildExcludeKeepIndex 는 텍스트를 한 번만 나눈다 — 태그마다 규칙 전체를 다시 나누지 않는다', () => {
    // 옛 isExcepted → hasKeepExact 는 태그마다 칸 전체를 split·parse 했다(규칙 수 × 매칭 태그 수)
    const rules = Array.from({ length: 600 }, (_, i) => (i % 3 === 0 ? `~keep ${i}` : `rule ${i}`)).join(', ')
    const index = buildExcludeKeepIndex(rules)
    const split = String.prototype.split
    let splits = 0
    String.prototype.split = function (this: string, ...args: any[]) { splits++; return (split as any).apply(this, args) } as any
    try {
      for (let i = 0; i < 5000; i++) index.keptBy(`keep_${i}`)
    } finally {
      String.prototype.split = split
    }
    expect(splits).toBe(0)
    expect(index.keptBy('keep_3')).toMatchObject({ form: 'keep_exact' })
    expect(index.keptBy('keep_4')).toBeNull()
  })

  it('toggleKeepExact 는 공백형 ~규칙을 더하고, 같은 태그의 ~완전일치 규칙을 모두 뺀다 (줄 배치는 그대로)', () => {
    expect(toggleKeepExact('', 'long_hair')).toBe('~long hair')
    expect(toggleKeepExact('short', 'long_hair')).toBe('short, ~long hair')
    expect(toggleKeepExact('short, ~Long Hair\n~long_hair, ~long_hair_ornament', 'long_hair'))
      .toBe('short\n~long_hair_ornament')
    expect(toggleKeepExact('bad, error\n~long hair\nborder, koma', 'long_hair')).toBe('bad, error\nborder, koma')
  })

  it('toggleKeepExact 는 패턴 예외로만 유지되는 태그에 중복 ~태그 를 더하지 않는다', () => {
    // 옛 includes 검사도 이 경우 규칙을 바꾸지 않았다 — 표시는 초록, 클릭은 그대로
    expect(toggleKeepExact('hair, ~long_hair_', 'long_hair')).toBe('hair, ~long_hair_')
    expect(toggleKeepExact('hair, ~_hair', 'long_hair')).toBe('hair, ~_hair')
    // ~완전일치와 패턴이 같이 있으면 클릭은 ~완전일치만 뺀다 (패턴이 남아 여전히 유지)
    const next = toggleKeepExact('hair, ~_hair, ~long hair', 'long_hair')
    expect(next).toBe('hair, ~_hair')
    expect(buildExcludeKeepIndex(next).keptBy('long_hair')).toMatchObject({ form: 'keep_suffix' })
  })

  it('keepToggleHint 는 클릭이 하는 일을 알려 준다', () => {
    expect(keepToggleHint(null)).toContain('예외(~) 추가')
    expect(keepToggleHint(parseExcludeRule('~Long Hair'))).toBe("예외 '~Long Hair' — 클릭하면 해제")
    expect(keepToggleHint(parseExcludeRule('~long_hair_'))).toContain('클릭으로는 풀리지 않는다')
    expect(keepToggleHint(parseExcludeRule('~\nsolo'))).toBe("예외 '~ solo' — 클릭하면 해제")
  })

  it('태그 앞뒤 밑줄이 접두·접미 표시로 읽히지 않는다', () => {
    const text = toggleKeepExact('', 'hair_')
    expect(parseExcludeRule(text)).toMatchObject({ form: 'keep_exact', keyword: 'hair' })
    expect(parseExcludeRule(addExactExcludeRule('', '_hair'))).toMatchObject({ form: 'exact', keyword: 'hair' })
  })

  it('addExactExcludeRule 은 같은 태그의 완전 일치 규칙이 있을 때만 건너뛴다', () => {
    expect(addExactExcludeRule('', 'long_hair')).toBe('*long hair')
    expect(addExactExcludeRule('*Long_Hair', 'long_hair')).toBe('*Long_Hair')
    // 옛 includes('*long_hair') 검사는 이 경우 규칙을 더하지 않았다
    expect(addExactExcludeRule('*long_hair_ornament', 'long_hair')).toBe('*long_hair_ornament, *long hair')
    expect(addExactExcludeRule('long_hair', 'long_hair')).toBe('long_hair, *long hair')
  })

  it('addExactExcludeRule 는 마지막 규칙 뒤에 잇는다 — 꼬리 줄바꿈은 그대로', () => {
    expect(addExactExcludeRule('a, b\nc\n', 'long_hair')).toBe('a, b\nc, *long hair\n')
  })
})

describe('rewriteExcludeRules — 칸을 다시 써도 줄 배치가 남는다', () => {
  const PASTED = 'watermark, *qr code\nborder, *2koma'
  const without = (text: string, i: number) => {
    const rules = splitExcludeRules(text)
    rules.splice(i, 1)
    return rewriteExcludeRules(text, rules)
  }
  const replaced = (text: string, i: number, rule: string) => {
    const rules = splitExcludeRules(text)
    rules[i] = rule
    return rewriteExcludeRules(text, rules)
  }

  it('삭제 — 리뷰 사례: 규칙 하나를 지워도 두 줄이 한 줄로 뭉개지지 않는다', () => {
    expect(without(PASTED, 2)).toBe('watermark, *qr code\n*2koma')
    expect(without(PASTED, 1)).toBe('watermark\nborder, *2koma')
    expect(without(PASTED, 0)).toBe('*qr code\nborder, *2koma')
    expect(without(PASTED, 3)).toBe('watermark, *qr code\nborder')
  })

  it('삭제 — 둘레 구분자 중 줄바꿈이 가장 많은 것을 남긴다 (카테고리 사이 빈 줄 유지)', () => {
    expect(without('a\nb\n\nc\nd', 2)).toBe('a\nb\n\nd')
    expect(without('a\nb\n\nc\nd', 1)).toBe('a\n\nc\nd')
    expect(without('a, b, c', 1)).toBe('a, c')
    expect(without('a\r\nb\r\nc', 1)).toBe('a\r\nc')
    // 머리·꼬리(앞뒤 공백, 끝 쉼표·줄바꿈)는 그대로
    expect(without('  a, b  \n', 0)).toBe('  b  \n')
    expect(without('a, b,', 1)).toBe('a,')
    // 여러 개가 한 번에 빠져도
    expect(rewriteExcludeRules('a, b\nc, d', ['a', 'd'])).toBe('a\nd')
  })

  it('편집 — 규칙만 제자리에서 바뀐다', () => {
    expect(replaced(PASTED, 2, 'frame')).toBe('watermark, *qr code\nframe, *2koma')
    expect(replaced('a\nb', 1, 'x, y')).toBe('a\nx, y')   // 쉼표를 넣으면 그 자리에서 두 규칙
    expect(rewriteExcludeRules('a, b\nc, d', ['a', 'x', 'd'])).toBe('a, x\nd')
    expect(rewriteExcludeRules('a\nb\nc', ['a', 'x', 'y', 'c'])).toBe('a\nx, y\nc')
  })

  it('끌어 옮기기 — 구분자 자리는 그대로, 규칙만 옮겨진다', () => {
    expect(rewriteExcludeRules('a, b\nc, d', ['d', 'a', 'b', 'c'])).toBe('d, a\nb, c')
    expect(rewriteExcludeRules('a, b\nc, d', ['a', 'c', 'b', 'd'])).toBe('a, c\nb, d')
  })

  it('추가 — 마지막 규칙 뒤(또는 그 자리 앞)에 ", " 로', () => {
    expect(rewriteExcludeRules('a, b\n', ['a', 'b', 'c'])).toBe('a, b, c\n')
    expect(rewriteExcludeRules('a\nc', ['a', 'b', 'c'])).toBe('a\nb, c')
    expect(rewriteExcludeRules('a\nb', ['x', 'a', 'b'])).toBe('x, a\nb')
    expect(appendExcludeRule('', 'c')).toBe('c')
    expect(appendExcludeRule('  \n , ', 'c')).toBe('c')
    expect(appendExcludeRule('a', 'a')).toBe('a, a')
    expect(appendExcludeRule('a\n', '   ')).toBe('a\n')
  })

  it('그대로면 원문 그대로, 모두 빠지면 빈 칸, 빈 항목은 버린다', () => {
    expect(rewriteExcludeRules(PASTED + '\n', splitExcludeRules(PASTED))).toBe(PASTED + '\n')
    expect(rewriteExcludeRules(PASTED, [])).toBe('')
    expect(rewriteExcludeRules(PASTED, ['  ', ''])).toBe('')
    expect(rewriteExcludeRules(undefined, ['a', 'b'])).toBe('a, b')
    expect(rewriteExcludeRules('a\nb', ['a', ' ', 'b'])).toBe('a\nb')
  })

  it('배치를 살리다 규칙이 달라지면 쉼표 한 줄로 쓴다 — 표시만 있는 조각이 다음 줄과 이어지는 경우', () => {
    // b 를 지우고 줄바꿈을 남기면 'a, ~\nc' = '~c'(유지 규칙)가 된다
    expect(without('a, ~, b\nc', 2)).toBe('a, ~, c')
    expect(splitExcludeRules(without('a, ~, b\nc', 2))).toEqual(['a', '~', 'c'])
  })

  it('다시 나누면 넘긴 목록과 같다 (여러 모양 무작위)', () => {
    let seed = 7
    const rand = (n: number) => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed % n }
    const pool = ['a', 'b', '*c', '_d', 'e_', '~f', '~_g_', 'long hair', '~']
    const seps = [', ', ',', '\n', ',\n', '\n\n', ' , ', '\r\n']
    for (let round = 0; round < 400; round++) {
      const count = 1 + rand(6)
      let text = ''
      for (let i = 0; i < count; i++) text += (i ? seps[rand(seps.length)] : '') + pool[rand(pool.length)]
      const rules = splitExcludeRules(text)
      const next = rules.filter(() => rand(3) !== 0)
      if (rand(2)) next.splice(rand(next.length + 1), 0, pool[rand(pool.length - 1)])
      if (next.length > 1 && rand(2)) next.push(next.shift()!)
      const out = rewriteExcludeRules(text, next)
      const label = `${JSON.stringify(text)} → ${JSON.stringify(next)}`
      expect(splitExcludeRules(out), label).toEqual(splitExcludeRules(next.join(', ')))
      expect(out, label).not.toContain('undefined')
      if (next.length === 0) expect(out).toBe('')
    }
  })
})

describe('excludeRuleColorClass', () => {
  it('종류별 색 — 파서와 같은 분류', () => {
    expect(excludeRuleColorClass('~_x')).toBe('bc-action')
    expect(excludeRuleColorClass('*x')).toBe('bc-expression')
    expect(excludeRuleColorClass('_x_')).toBe('bc-nsfw')
    expect(excludeRuleColorClass('x')).toBe('bc-nsfw')
    expect(excludeRuleColorClass('_x')).toBe('bc-body')
    expect(excludeRuleColorClass('x_')).toBe('bc-clothing')
    // 키워드가 빈 규칙은 적용 쪽이 버린다 — 색도 없다
    expect(excludeRuleColorClass('__')).toBe('')
    expect(excludeRuleColorClass('~')).toBe('')
  })
})
