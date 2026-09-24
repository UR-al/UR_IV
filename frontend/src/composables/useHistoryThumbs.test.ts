import { describe, expect, it } from 'vitest'
import { normalizePreviewThumbWidth, useHistoryThumbs, versionedThumbUrl } from './useHistoryThumbs'

function harness(opts: { withSlot?: boolean; chunk?: number } = {}) {
  const calls: Array<{ paths: string[]; width: number }> = []
  const backend: any = opts.withSlot === false ? {} : {
    generateThumbnails: (json: string, width: number) => calls.push({ paths: JSON.parse(json), width }),
  }
  let handler: ((json: string) => void) | null = null
  let unbound = 0
  const h = useHistoryThumbs({
    getBackend: async () => backend,
    onBackendEvent: (_name, cb) => { handler = cb; return () => { unbound++ } },
    mediaUrl: (p: string) => (p.startsWith('file:///') ? p : 'file:///' + p.replace(/\\/g, '/')),
    chunk: opts.chunk,
  })
  return { h, calls, fire: (json: string) => handler && handler(json), unboundCount: () => unbound }
}

describe('useHistoryThumbs', () => {
  it('normalises the preview width to a preset', () => {
    expect(normalizePreviewThumbWidth(384)).toBe(384)
    expect(normalizePreviewThumbWidth('512')).toBe(512)
    expect(normalizePreviewThumbWidth(999)).toBe(256)
    expect(normalizePreviewThumbWidth(undefined)).toBe(256)
  })

  it('falls back to the original until a thumbnail arrives, then uses it', async () => {
    const { h, calls, fire } = harness()
    h.bind()
    expect(h.srcFor('C:\\out\\a.png')).toBe('file:///C:/out/a.png')
    await h.ensure(['C:\\out\\a.png'])
    expect(calls).toEqual([{ paths: ['C:\\out\\a.png'], width: 256 }])
    fire(JSON.stringify({ path: 'C:\\out\\a.png', thumb: 'file:///C:/cache/ab/hash.jpg' }))
    expect(h.srcFor('C:\\out\\a.png')).toBe('file:///C:/cache/ab/hash.jpg')
    // 편집으로 버전이 오르면 썸네일 대신 원본(캐시 무효화)
    expect(h.srcFor('C:\\out\\a.png', 1234)).toBe('file:///C:/out/a.png?t=1234')
  })

  it('requests each path once per width, in chunks, and ignores foreign or stale replies', async () => {
    const { h, calls, fire } = harness({ chunk: 2 })
    h.bind()
    const list = ['a.png', 'b.png', 'c.png']
    expect(await h.ensure(list)).toBe(3)
    expect(calls.map(c => c.paths)).toEqual([['a.png', 'b.png'], ['c.png']])
    expect(await h.ensure(list)).toBe(0)                      // 다시 요청하지 않는다
    fire(JSON.stringify({ path: 'zzz.png', thumb: 'file:///x.jpg' }))   // 즐겨찾기 등 다른 화면의 응답
    expect(h.thumbs['zzz.png']).toBeUndefined()
    // 폭 변경 → 캐시 비우고 새 폭으로 재요청; 옛 폭 응답은 버린다
    expect(h.setWidth(384, list)).toBe(true)
    expect(h.setWidth(384, list)).toBe(false)
    await Promise.resolve()
    expect(calls[calls.length - 1]?.width).toBe(384)
    fire(JSON.stringify({ path: 'a.png', thumb: 'file:///old.jpg' }))
    expect(h.thumbs['a.png']).toBe('file:///old.jpg')          // 같은 경로를 새 폭으로 다시 요청했으므로 수락
    expect(h.width.value).toBe(384)
  })

  it('accepts chunked payloads for the current width and drops other widths', async () => {
    const { h, fire } = harness()
    h.bind()
    await h.ensure(['a.png', 'b.png'])
    // 다른 폭(예: 폭 변경 전 요청)의 청크 응답은 통째로 버린다
    fire(JSON.stringify({ width: 384, items: [{ path: 'a.png', thumb: 'file:///384/a.jpg' }] }))
    expect(h.thumbs['a.png']).toBeUndefined()
    // 캐시 적중 청크 — 한 번에 여러 장
    fire(JSON.stringify({ width: 256, items: [
      { path: 'a.png', thumb: 'file:///256/a.jpg' },
      { path: 'b.png', thumb: '' },                        // 생성 실패 → 원본 폴백
      { path: 'zzz.png', thumb: 'file:///x.jpg' },         // 요청 안 한 경로
      { nope: 1 },
    ] }))
    expect(h.thumbs['a.png']).toBe('file:///256/a.jpg')
    expect(h.thumbs['b.png']).toBe('')
    expect(h.srcFor('b.png')).toBe('file:///b.png')
    expect(h.thumbs['zzz.png']).toBeUndefined()
    h.onReady('not json')
    h.onReady('null')
  })

  it('invalidates an overwritten image, re-requests it and busts the unchanged thumb URL', async () => {
    const { h, calls, fire } = harness()
    h.bind()
    await h.ensure(['a.png'])
    fire(JSON.stringify({ width: 256, items: [{ path: 'a.png', thumb: 'file:///c/a.jpg' }] }))
    // 같은 경로에 덮어쓴 새 이미지 — 새 썸네일이 올 때까지 원본(버전 쿼리)
    h.invalidate('a.png')
    expect(h.srcFor('a.png', 77)).toBe('file:///a.png?t=77')
    expect(await h.ensure(['a.png'])).toBe(1)
    expect(calls).toHaveLength(2)
    fire(JSON.stringify({ width: 256, items: [{ path: 'a.png', thumb: 'file:///c/a.jpg' }] }))
    expect(h.srcFor('a.png', 77)).toBe('file:///c/a.jpg?t=77')
    expect(h.srcFor('a.png')).toBe('file:///c/a.jpg')
  })

  it('adds the thumbnail file version so a rebuilt thumbnail at the same path is reloaded', async () => {
    const { h, fire } = harness()
    h.bind()
    await h.ensure(['a.png'])
    h.invalidate('a.png')
    await h.ensure(['a.png'])
    // 무효화 뒤 첫 응답이 덮어쓰기 전 렌더였고(v=100), 다시 돌린 응답이 새 렌더다(v=200) — 같은 캐시 경로
    fire(JSON.stringify({ width: 256, items: [{ path: 'a.png', thumb: 'file:///c/a.jpg', v: '100' }] }))
    const first = h.srcFor('a.png', 77)
    fire(JSON.stringify({ width: 256, items: [{ path: 'a.png', thumb: 'file:///c/a.jpg', v: '200' }] }))
    const second = h.srcFor('a.png', 77)
    expect(first).toBe('file:///c/a.jpg?v=100&t=77')
    expect(second).toBe('file:///c/a.jpg?v=200&t=77')
    expect(h.srcFor('a.png')).toBe('file:///c/a.jpg?v=200')
    expect(versionedThumbUrl('/file?path=x', '9')).toBe('/file?path=x&v=9')
    expect(versionedThumbUrl('file:///x.jpg', '')).toBe('file:///x.jpg')
  })

  it('keeps working when the backend has no thumbnail slot and unbinds cleanly', async () => {
    const { h, unboundCount } = harness({ withSlot: false })
    h.bind()
    expect(await h.ensure(['a.png'])).toBe(0)
    expect(h.srcFor('a.png')).toBe('file:///a.png')
    h.unbind(); h.unbind()
    expect(unboundCount()).toBe(1)
  })
})
