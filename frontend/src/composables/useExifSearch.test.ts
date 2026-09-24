import { describe, expect, it } from 'vitest'
import { ref } from 'vue'
import { useExifSearch } from './useExifSearch'
import { isImage, mediaLabel, filenameOf } from '../utils/mediaKind'

function harness(texts: Record<string, string>, opts: { chunkSize?: number } = {}) {
  const images = ref<string[]>(['a.png', 'b.png', 'clip.mp4'])
  const calls: string[][] = []
  let changes = 0
  let gate: (() => void) | null = null
  let hold = false
  const search = useExifSearch({
    source: () => images.value,
    isImage,
    labelFor: path => `${filenameOf(path)} ${mediaLabel(path)}`,
    fetchTexts: async (paths) => {
      calls.push([...paths])
      // 백엔드는 요청을 받은 때의 파일을 읽는다 — 붙잡아 둔 사이 texts 를 바꿔도 이 응답은 그대로다
      const answer = Object.fromEntries(paths.filter(p => p in texts).map(p => [p, texts[p]]))
      if (hold) await new Promise<void>(resolve => { gate = resolve })
      return answer
    },
    onChange: () => { changes++ },
    chunkSize: opts.chunkSize,
  })
  return {
    images, calls, search,
    changes: () => changes,
    holdNext: () => { hold = true },
    release: () => { hold = false; gate?.() },
  }
}

describe('useExifSearch', () => {
  it('filters images by fetched text and media by name, never sending media to the backend', async () => {
    const h = harness({ 'a.png': 'a blue cat', 'b.png': 'red dog' })
    h.search.query.value = 'CAT'
    expect(await h.search.run()).toBe(true)
    expect(h.calls).toEqual([['a.png', 'b.png']])
    expect(h.search.results.value).toEqual(['a.png'])
    h.search.query.value = 'video'
    await h.search.run()
    expect(h.search.results.value).toEqual(['clip.mp4'])
    expect(h.calls.length).toBe(1)          // 캐시 — 다시 묻지 않는다
    expect(h.changes()).toBe(2)
  })

  it('results follow the current list (deleted paths disappear)', async () => {
    const h = harness({ 'a.png': 'cat', 'b.png': 'cat' })
    h.search.query.value = 'cat'
    await h.search.run()
    expect(h.search.results.value).toEqual(['a.png', 'b.png'])
    h.images.value = ['b.png']
    expect(h.search.results.value).toEqual(['b.png'])
  })

  it('ignores re-entrant runs and drops results when the query is cleared mid-search', async () => {
    const h = harness({ 'a.png': 'cat', 'b.png': 'dog' }, { chunkSize: 1 })
    h.holdNext()
    h.search.query.value = 'cat'
    const first = h.search.run()
    expect(await h.search.run()).toBe(false)  // Enter 연타
    h.search.query.value = ''
    h.release()
    expect(await first).toBe(false)
    expect(h.search.filtered.value).toBe(false)
    expect(h.search.searching.value).toBe(false)
  })

  it('cancel() makes a pending search a no-op', async () => {
    const h = harness({ 'a.png': 'cat' })
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.search.cancel()
    h.release()
    expect(await pending).toBe(false)
    expect(h.search.filtered.value).toBe(false)
  })

  it('does not cache paths the backend did not answer, and invalidates after an edit', async () => {
    const texts: Record<string, string> = { 'a.png': 'old words' }
    const h = harness(texts)
    h.search.query.value = 'new'
    await h.search.run()
    expect(h.search.results.value).toEqual([])
    texts['a.png'] = 'new words'
    texts['b.png'] = 'nothing'
    h.search.invalidate('a.png')
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    expect(h.calls[1]).toEqual(['a.png', 'b.png'])   // b 는 처음에 답이 없어 다시 묻는다
    expect(h.search.results.value).toEqual(['a.png'])
  })

  it('moves cache and matches on rename', async () => {
    const h = harness({ 'a.png': 'cat' })
    h.search.query.value = 'cat'
    await h.search.run()
    h.images.value = ['z.png', 'b.png', 'clip.mp4']
    h.search.rename('a.png', 'z.png')
    expect(h.search.results.value).toEqual(['z.png'])
  })

  // 영상·오디오의 검색 텍스트는 파일 이름이다 — 이름을 바꾸면 새 이름으로 찾혀야 한다
  it('rebuilds a video label on rename instead of carrying the old name', async () => {
    const h = harness({})
    h.search.query.value = 'clip'
    await h.search.run()
    expect(h.search.results.value).toEqual(['clip.mp4'])
    h.images.value = ['a.png', 'b.png', 'movie.mp4']
    h.search.rename('clip.mp4', 'movie.mp4')
    expect(h.search.results.value).toEqual([])      // 걸린 필터('clip')에 새 이름이 안 맞는다
    h.search.query.value = 'movie'
    await h.search.run()
    expect(h.search.results.value).toEqual(['movie.mp4'])
    h.search.query.value = 'clip'
    await h.search.run()
    expect(h.search.results.value).toEqual([])
    expect(h.calls.flat().some(p => p.endsWith('.mp4'))).toBe(false)   // 영상은 백엔드로 보내지 않는다
  })

  it('rebuilds an audio label on rename too, and keeps it matched when the new name still fits', async () => {
    const h = harness({})
    h.images.value = ['a.png', 'song.mp3', 'take.mp4']
    h.search.query.value = 'song'
    await h.search.run()
    expect(h.search.results.value).toEqual(['song.mp3'])
    h.images.value = ['a.png', 'track2.mp3', 'take.mp4']
    h.search.rename('song.mp3', 'track2.mp3')
    h.search.query.value = 'track2'
    await h.search.run()
    expect(h.search.results.value).toEqual(['track2.mp3'])
    h.search.query.value = 'song'
    await h.search.run()
    expect(h.search.results.value).toEqual([])

    h.search.query.value = 'take'
    await h.search.run()
    h.images.value = ['a.png', 'track2.mp3', 'take_2.mp4']
    h.search.rename('take.mp4', 'take_2.mp4')
    expect(h.search.results.value).toEqual(['take_2.mp4'])   // 새 이름도 'take' 에 맞는다
  })

  it('a media file renamed while a search is in flight is matched by its new name', async () => {
    const h = harness({ 'a.png': 'movie night' })
    h.holdNext()
    h.search.query.value = 'movie'
    const pending = h.search.run()
    h.images.value = ['a.png', 'b.png', 'movie.mp4']
    h.search.rename('clip.mp4', 'movie.mp4')
    h.release()
    expect(await pending).toBe(true)
    expect(h.search.results.value).toEqual(['a.png', 'movie.mp4'])
  })

  // 확대 뷰에서 이름을 바꾸는 사이 그 이미지의 메타데이터를 묻고 있었다 — 응답은 옛 이름으로 온다.
  // 예전엔 옛 이름으로 캐시해, 메타데이터가 맞아도 그 검색 결과에서 빠지고 옛 이름 캐시만 남았다.
  it('an image renamed while its metadata is in flight is matched under its new name', async () => {
    const texts: Record<string, string> = { 'a.png': 'blue cat', 'b.png': 'dog' }
    const h = harness(texts)
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.images.value = ['z.png', 'b.png', 'clip.mp4']
    h.search.rename('a.png', 'z.png')
    h.release()
    expect(await pending).toBe(true)
    expect(h.search.results.value).toEqual(['z.png'])
    // 캐시도 새 이름으로 — 다시 묻지 않고, 같은 옛 이름의 새 파일은 옛 텍스트로 맞추지 않는다
    texts['a.png'] = 'other'
    h.images.value = ['a.png', 'z.png', 'b.png']
    h.search.query.value = 'blue'
    await h.search.run()
    expect(h.calls).toEqual([['a.png', 'b.png'], ['a.png']])
    expect(h.search.results.value).toEqual(['z.png'])
  })

  // 백엔드 스레드가 그 파일에 닿기 전에 이름이 바뀌면 옛 경로는 이미 없다 — 없는 경로의 답은 ''
  // (core/metadata_search.search_texts_for). 예전엔 그 ''를 새 이름에 캐시해, 메타데이터가 맞아도
  // 이번 검색에서 빠지고 캐시 때문에 다음 검색에서도 다시 묻지 않아 세션 내내 안 찾혔다.
  it('an image renamed before the backend read its old name is asked again by its new name', async () => {
    const texts: Record<string, string> = { 'a.png': '', 'z.png': 'blue cat', 'b.png': 'dog' }
    const h = harness(texts)
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.images.value = ['z.png', 'b.png', 'clip.mp4']
    h.search.rename('a.png', 'z.png')
    h.release()
    expect(await pending).toBe(true)
    expect(h.calls).toEqual([['a.png', 'b.png'], ['z.png']])
    expect(h.search.results.value).toEqual(['z.png'])
    h.search.query.value = 'blue'           // 새 이름의 진짜 텍스트가 캐시됐다 — 다시 묻지 않는다
    await h.search.run()
    expect(h.calls.length).toBe(2)
    expect(h.search.results.value).toEqual(['z.png'])
  })

  it('a renamed image that really has no metadata is asked once more, then cached as empty', async () => {
    const texts: Record<string, string> = { 'a.png': '', 'z.png': '', 'b.png': 'dog' }
    const h = harness(texts)
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.images.value = ['z.png', 'b.png', 'clip.mp4']
    h.search.rename('a.png', 'z.png')
    h.release()
    expect(await pending).toBe(true)
    expect(h.calls).toEqual([['a.png', 'b.png'], ['z.png']])   // 한 번만 더 — 되묻기가 돌지 않는다
    expect(h.search.results.value).toEqual([])
    h.search.query.value = 'dog'
    await h.search.run()
    expect(h.calls.length).toBe(2)
    expect(h.search.results.value).toEqual(['b.png'])
  })

  it('an empty answer for an image that was not renamed is cached as is (no extra request)', async () => {
    const h = harness({ 'a.png': '', 'b.png': 'dog' })
    h.search.query.value = 'dog'
    await h.search.run()
    h.search.query.value = 'cat'
    await h.search.run()
    expect(h.calls).toEqual([['a.png', 'b.png']])
    expect(h.search.results.value).toEqual([])
  })

  it('a batch not sent yet asks for a renamed image by its new name', async () => {
    const h = harness({ 'a.png': 'dog', 'y.png': 'cat' }, { chunkSize: 1 })   // 이름을 바꾼 b.png 는 이제 y.png
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.images.value = ['a.png', 'y.png', 'clip.mp4']
    h.search.rename('b.png', 'y.png')
    h.release()
    expect(await pending).toBe(true)
    expect(h.calls).toEqual([['a.png'], ['y.png']])
    expect(h.search.results.value).toEqual(['y.png'])
  })

  it('an EXIF save during the search drops the possibly stale answer and asks again in the same search', async () => {
    const texts: Record<string, string> = { 'a.png': 'old words' }
    const h = harness(texts)
    h.holdNext()
    h.search.query.value = 'new'
    const pending = h.search.run()
    texts['a.png'] = 'new words'
    h.search.invalidate('a.png')
    h.release()
    expect(await pending).toBe(true)
    expect(h.calls).toEqual([['a.png', 'b.png'], ['a.png']])
    expect(h.search.results.value).toEqual(['a.png'])
  })

  it('a path deleted during the search is not cached or asked for', async () => {
    const texts: Record<string, string> = { 'a.png': 'cat', 'b.png': 'cat' }
    const h = harness(texts, { chunkSize: 1 })
    h.holdNext()
    h.search.query.value = 'cat'
    const pending = h.search.run()
    h.images.value = ['clip.mp4']
    h.search.forget('a.png')      // 묻는 중
    h.search.forget('b.png')      // 아직 안 물었다
    h.release()
    expect(await pending).toBe(true)
    expect(h.calls).toEqual([['a.png']])
    expect(h.search.results.value).toEqual([])
    texts['a.png'] = 'fresh'      // 같은 이름의 새 파일
    h.images.value = ['a.png', 'clip.mp4']
    h.search.query.value = 'fresh'
    await h.search.run()
    expect(h.calls).toEqual([['a.png'], ['a.png']])
    expect(h.search.results.value).toEqual(['a.png'])
  })

  it('an empty query clears the filter', async () => {
    const h = harness({})
    h.search.query.value = '   '
    expect(await h.search.run()).toBe(false)
    expect(h.search.filtered.value).toBe(false)
    expect(h.changes()).toBe(1)
  })
})
