import { describe, expect, it } from 'vitest'

import { cachedTagSplitter, splitDanbooruTags } from './searchTags'


describe('splitDanbooruTags', () => {
  it('splits space-separated danbooru tags and shows underscores as spaces', () => {
    expect(splitDanbooruTags('hatsune_miku kagamine_rin')).toEqual(['hatsune miku', 'kagamine rin'])
  })

  it('keeps spaces inside parentheses as part of one tag', () => {
    expect(splitDanbooruTags('kafka_(honkai: star_rail) original')).toEqual([
      'kafka (honkai: star rail)',
      'original',
    ])
  })

  it('reads the legacy comma format by commas only', () => {
    expect(splitDanbooruTags('aqua gloves, long_hair')).toEqual(['aqua gloves', 'long hair'])
  })

  it('treats empty and missing values as no tags', () => {
    expect(splitDanbooruTags('')).toEqual([])
    expect(splitDanbooruTags('   ')).toEqual([])
    expect(splitDanbooruTags(null)).toEqual([])
    expect(splitDanbooruTags(undefined)).toEqual([])
  })
})


describe('cachedTagSplitter', () => {
  it('returns the same frozen result for the same text in one scope', () => {
    const scope: object[] = []
    const split = cachedTagSplitter(scope)
    const first = split('hatsune_miku')
    expect(first).toEqual(['hatsune miku'])
    expect(split('hatsune_miku')).toBe(first)
    // 같은 scope 로 다시 만든 splitter 도 같은 캐시를 쓴다
    expect(cachedTagSplitter(scope)('hatsune_miku')).toBe(first)
    expect(Object.isFrozen(first)).toBe(true)
  })

  it('keeps separate caches per scope', () => {
    const a = cachedTagSplitter([])
    const b = cachedTagSplitter([])
    expect(a('original')).toEqual(b('original'))
    expect(a('original')).not.toBe(b('original'))
  })

  it('matches the uncached splitter for every input', () => {
    const split = cachedTagSplitter([])
    for (const raw of ['', 'a_b c', 'x, y_z', 'kafka_(honkai: star_rail)', null]) {
      expect([...split(raw)]).toEqual(splitDanbooruTags(raw))
    }
  })
})
