import { describe, expect, it } from 'vitest'
import {
  RELATED_TAG_FIELDS, buildRelatedTagsQuery, formatBuildProgress, formatEstimate, formatShards, parseRelatedTagsEvent,
  planRelatedTagAppend, querySignature, relatedInsertText, relatedTagLabel, relatedTagTitle,
} from './relatedTags'

const WIDGETS = {
  main_prompt_text: 'armor, sword', character_input: 'hatsune miku', copyright_input: 'vocaloid',
  char_count_input: '1girl', prefix_prompt_text: '', suffix_prompt_text: null, neg_prompt_text: 'lowres',
  exclude_prompt_local_input: '_armor', artist_input: 'someone',
}

const QUERY_EVENT = {
  action: 'related_tags_query', requestId: 'rtq-1', state: 'ready',
  tags: [{ tag: 'holding_sword', ko: '검 들기', count: 1234, via: ['armor', 'sword', 'x', 'y'] }],
  context: ['armor', 'sword'], ratings: ['g', 's'], missing: [], stale: ['s'], unavailable: [],
  estimateSec: 0, build: null,
}

describe('relatedTags query payload', () => {
  it('maps every field to the widget the Python proxy owns and ignores the artist field', () => {
    expect(RELATED_TAG_FIELDS).toEqual({
      main: 'main_prompt_text', character: 'character_input', copyright: 'copyright_input',
      count: 'char_count_input', prefix: 'prefix_prompt_text', suffix: 'suffix_prompt_text',
    })
    expect(buildRelatedTagsQuery(WIDGETS, 'rtq-1')).toEqual({
      requestId: 'rtq-1',
      fields: { main: 'armor, sword', character: 'hatsune miku', copyright: 'vocaloid', count: '1girl', prefix: '', suffix: '' },
      negative: 'lowres', exclude: '_armor', limit: 16,
    })
  })

  it('signature ignores the request id but sees every prompt change', () => {
    const a = querySignature(buildRelatedTagsQuery(WIDGETS, 'a'))
    expect(querySignature(buildRelatedTagsQuery(WIDGETS, 'b'))).toBe(a)
    expect(querySignature(buildRelatedTagsQuery({ ...WIDGETS, neg_prompt_text: 'x' }, 'a'))).not.toBe(a)
    expect(querySignature(buildRelatedTagsQuery({ ...WIDGETS, artist_input: 'other' }, 'a'))).toBe(a)
  })
})

describe('relatedTags insertion', () => {
  it('uses spaces in block mode and underscores in text mode, like each mode’s autocomplete', () => {
    expect(relatedInsertText('holding_sword', true)).toBe('holding sword')
    expect(relatedInsertText('holding_sword', false)).toBe('holding_sword')
    expect(relatedInsertText('star_(sky)', true)).toBe('star (sky)')
    // 밑줄→공백이 '//' 를 주석으로 만드는 태그는 블록 모드에서도 밑줄 그대로
    const zato = 'z.a.t.o._//_i_love_the_world_and_everything_in_it'
    expect(relatedInsertText(zato, true)).toBe(zato)
  })

  it('appends to the end without rewriting the prompt and skips tags already anywhere', () => {
    expect(planRelatedTagAppend('armor, (sword:1.2)', 'holding_sword', false)).toEqual({
      text: 'armor, (sword:1.2), holding_sword', added: true,
    })
    expect(planRelatedTagAppend('', 'cape', true)).toEqual({ text: 'cape', added: true })
    expect(planRelatedTagAppend('armor,', 'cape', true).text).toBe('armor, cape')
    expect(planRelatedTagAppend('armor, holding sword', 'holding_sword', false)).toEqual({
      text: 'armor, holding sword', added: false,
    })
    expect(planRelatedTagAppend('armor', 'cape', true, 'red cape, cape').added).toBe(false)
  })

  it('treats prompt comments as absent and never appends into a line comment', () => {
    // 주석 속 태그는 '이미 있음'이 아니고, 줄 주석으로 끝나면 줄바꿈으로 잇는다(utils/promptComments)
    expect(planRelatedTagAppend('armor // cape', 'cape', true)).toEqual({ text: 'armor // cape\ncape', added: true })
    expect(planRelatedTagAppend('armor, /* holding sword */', 'holding_sword', true).added).toBe(true)
  })
})

describe('parseRelatedTagsEvent', () => {
  it('accepts a valid query event (string or object) and caps via at three', () => {
    const parsed = parseRelatedTagsEvent(JSON.stringify(QUERY_EVENT))
    expect(parsed).toMatchObject({ action: 'related_tags_query', state: 'ready', stale: ['s'] })
    expect(parsed && parsed.action === 'related_tags_query' && parsed.tags[0]!.via).toEqual(['armor', 'sword', 'x'])
    expect(parseRelatedTagsEvent({ ...QUERY_EVENT, unavailable: undefined })).not.toBeNull()
    const building = parseRelatedTagsEvent({ ...QUERY_EVENT, state: 'building',
      build: { requestId: 'rtb-1', shard: 'g', index: 0, count: 2, progress: 0.4, overall: 0.2 } })
    expect(building && building.action === 'related_tags_query' && building.build?.requestId).toBe('rtb-1')
  })

  it('accepts build events and rejects anything malformed', () => {
    expect(parseRelatedTagsEvent({ action: 'related_tags_build', requestId: 'b', state: 'done', shards: ['g'] }))
      .toEqual({ action: 'related_tags_build', requestId: 'b', state: 'done', shards: ['g'] })
    for (const bad of [
      'not json', null, [], { ...QUERY_EVENT, state: 'weird' }, { ...QUERY_EVENT, ratings: ['x'] },
      { ...QUERY_EVENT, tags: [{ tag: 'a', ko: '', count: 'many', via: [] }] }, { ...QUERY_EVENT, requestId: 3 },
      { ...QUERY_EVENT, build: { shard: 'g', index: 0, count: 1, progress: 0, overall: 0 } },
      { ...QUERY_EVENT, action: 'other' }, { action: 'related_tags_build', requestId: 'b', state: 'building',
        progress: { shard: 'z', index: 0, count: 1, progress: 0, overall: 0 } },
      { action: 'related_tags_build', requestId: 'b', state: 'done', shards: 'g' },
    ]) {
      expect(parseRelatedTagsEvent(bad)).toBeNull()
    }
  })
})

describe('relatedTags labels', () => {
  it('formats chips, ratings, estimates and progress in Korean', () => {
    expect(relatedTagLabel('holding_sword')).toBe('holding sword')
    expect(relatedTagTitle({ tag: 'holding_sword', ko: '검 들기', count: 12345, via: ['armor'] }))
      .toBe('holding sword · 게시물 12,345개\n함께 쓰인 태그: armor')
    expect(formatShards(['g', 's'])).toBe('G · S')
    expect(formatEstimate(0)).toBe('')
    expect(formatEstimate(3)).toBe('약 10초')
    expect(formatEstimate(26)).toBe('약 30초')
    expect(formatEstimate(140)).toBe('약 3분')
    expect(formatBuildProgress({ shard: 's', index: 1, count: 2, progress: 0.427, overall: 0.7 }))
      .toBe('표 만드는 중… S 42% (2/2)')
  })
})
