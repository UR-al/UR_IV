import { describe, expect, it } from 'vitest'
import { createExcludeMatchRequests, parseExcludeMatches } from './excludeMatches'

describe('excludeMatches', () => {
  it('asks the backend once per rule: cached or in-flight rules are skipped', () => {
    const requests = createExcludeMatchRequests()
    const cache: Record<string, string[]> = {}

    expect(requests.begin('hair', cache, 0)).toBe(true)
    expect(requests.begin('hair', cache, 10)).toBe(false) // 조회 중 — 더블클릭이 두 번 보내지 않는다
    expect(requests.begin('eyes', cache, 10)).toBe(true) // 다른 규칙은 따로

    requests.end('hair')
    cache.hair = ['long_hair']
    expect(requests.begin('hair', cache, 20)).toBe(false) // 캐시에 있으면 다시 안 부른다

    // 빈 결과도 결과다 — 매칭 0개인 규칙을 누를 때마다 다시 훑지 않는다
    requests.end('eyes')
    cache.eyes = []
    expect(requests.begin('eyes', cache, 30)).toBe(false)
    expect(requests.begin('', cache, 30)).toBe(false)
  })

  it('failed lookups are retried, and a lost callback is released after the TTL', () => {
    const requests = createExcludeMatchRequests(1000)
    const cache: Record<string, string[]> = {}

    expect(requests.begin('*solo', cache, 0)).toBe(true)
    requests.end('*solo') // 오류 응답 — 캐시하지 않았으니 다음 클릭에 다시 조회
    expect(requests.begin('*solo', cache, 5)).toBe(true)

    expect(requests.isPending('*solo', 500)).toBe(true)
    expect(requests.begin('*solo', cache, 999)).toBe(false)
    expect(requests.begin('*solo', cache, 2000)).toBe(true) // 콜백이 안 와도 영영 막히지 않는다
  })

  it('parses only string arrays; error objects and broken JSON are not cacheable', () => {
    expect(parseExcludeMatches('["long_hair","short_hair"]')).toEqual(['long_hair', 'short_hair'])
    expect(parseExcludeMatches('[]')).toEqual([])
    expect(parseExcludeMatches(['a', 3, null, 'b'])).toEqual(['a', 'b'])
    expect(parseExcludeMatches('{"error":"boom"}')).toBeNull()
    expect(parseExcludeMatches('{not json')).toBeNull()
    expect(parseExcludeMatches(null)).toBeNull()
  })
})
