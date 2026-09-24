import { afterEach, describe, expect, it, vi } from 'vitest'
import { computed } from 'vue'
import {
  MAX_LISTED_VERSIONS,
  bumpMediaVersion,
  composeMediaVersion,
  listedVersionEntries,
  mediaVersion,
  recordListedMediaVersions,
  resetMediaVersionsForTest,
} from './mediaVersions'

afterEach(() => {
  resetMediaVersionsForTest()
  vi.useRealTimers()
})

describe('mediaVersions', () => {
  it('composes the listed signature and the saved counter', () => {
    expect(composeMediaVersion('18a-3f', 1700000000000)).toBe('18a-3f.1700000000000')
    expect(composeMediaVersion('18a-3f', undefined)).toBe('18a-3f')
    expect(composeMediaVersion(undefined, 5)).toBe('5')
    expect(composeMediaVersion(null, null)).toBe('')
  })

  it('reads aligned files/versions and skips malformed items', () => {
    expect(listedVersionEntries(['C:\\A\\x.png', 3, 'c:/a/y.png', ''], ['v1', 'v2', 7, 'v4'])).toEqual([
      ['c:/a/x.png', 'v1'],
      ['c:/a/y.png', '7'],
    ])
    expect(listedVersionEntries(['a.png'], undefined)).toEqual([])   // 옛 페이로드(versions 없음)
    expect(listedVersionEntries('nope', ['v'])).toEqual([])
  })

  it('changes the card version when the listing reports a new signature (overwritten outside the app)', () => {
    recordListedMediaVersions(['C:/out/a.png', 'C:/out/b.png'], ['1-10', '2-20'])
    expect(mediaVersion('C:/out/a.png')).toBe('1-10')
    // 구분자·드라이브 대소문자가 달라도 같은 파일
    expect(mediaVersion('c:\\out\\a.png')).toBe('1-10')
    recordListedMediaVersions(['C:/out/a.png'], ['9-10'])
    expect(mediaVersion('C:/out/a.png')).toBe('9-10')
    // 다른 목록(다른 폴더·히스토리)이 앞 목록을 지우지 않는다
    expect(mediaVersion('C:/out/b.png')).toBe('2-20')
    expect(mediaVersion('C:/elsewhere/z.png')).toBe('')
    expect(mediaVersion('')).toBe('')
  })

  it('a save bump survives a stale listing that arrives later', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_700_000_000_000)
    recordListedMediaVersions(['C:/out/a.png'], ['1-10'])
    const beforeSave = mediaVersion('C:/out/a.png')
    bumpMediaVersion('C:\\out\\a.png')
    const afterSave = mediaVersion('C:/out/a.png')
    expect(afterSave).not.toBe(beforeSave)
    // 저장 전에 만든 목록이 늦게 와도 저장 전 URL 로 돌아가지 않는다
    recordListedMediaVersions(['C:/out/a.png'], ['1-10'])
    expect(mediaVersion('C:/out/a.png')).toBe(afterSave)
    // 같은 밀리초에 두 번 덮어써도 버전이 오른다
    bumpMediaVersion('C:/out/a.png')
    expect(mediaVersion('C:/out/a.png')).not.toBe(afterSave)
    bumpMediaVersion('')
    bumpMediaVersion(null)
  })

  it('is reactive so rendered cards pick up a bump or a new listing', () => {
    const url = computed(() => mediaVersion('C:/out/a.png'))
    expect(url.value).toBe('')
    recordListedMediaVersions(['C:/out/a.png'], ['1-10'])
    expect(url.value).toBe('1-10')
    bumpMediaVersion('C:/out/a.png')
    expect(url.value).toMatch(/^1-10\.\d+$/)
  })

  it('drops old listed versions instead of growing without bound', () => {
    const many = Array.from({ length: MAX_LISTED_VERSIONS }, (_, i) => `C:/f/${i}.png`)
    recordListedMediaVersions(many, many.map((_, i) => `v${i}`))
    expect(mediaVersion('C:/f/0.png')).toBe('v0')
    recordListedMediaVersions(['C:/g/new.png'], ['n'])
    expect(mediaVersion('C:/g/new.png')).toBe('n')
    expect(mediaVersion('C:/f/0.png')).toBe('')
  })
})
