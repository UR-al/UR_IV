import { afterEach, describe, expect, it } from 'vitest'
import { GALLERY_THUMB_BUCKETS, bucketThumbWidth, mediaUrl, thumbnailUrl, withUrlVersion } from './media.js'

const g = globalThis as unknown as { window?: Record<string, unknown> }

afterEach(() => { delete g.window })

describe('thumbnail urls', () => {
  it('quantises widths to the shared gallery buckets', () => {
    expect(GALLERY_THUMB_BUCKETS).toEqual([192, 256, 384, 512, 768])
    expect(bucketThumbWidth(100)).toBe(192)
    expect(bucketThumbWidth(200 * 1.25)).toBe(256)
    expect(bucketThumbWidth(380 * 2)).toBe(768)
    expect(bucketThumbWidth(5000)).toBe(768)
    expect(bucketThumbWidth(undefined)).toBe(384)
    expect(bucketThumbWidth('junk')).toBe(384)
  })

  it('uses the aithumb: scheme in the Qt app (same shape core/thumb_cache.parse_thumb_url reads)', () => {
    expect(thumbnailUrl('C:/img/a b+1.png', 380))
      .toBe('aithumb:thumb?path=C%3A%2Fimg%2Fa%20b%2B1.png&width=384')
    expect(thumbnailUrl('file:///C:/img/x.png', 200)).toBe('aithumb:thumb?path=C%3A%2Fimg%2Fx.png&width=256')
    expect(mediaUrl('C:/img/x.png')).toBe('file:///C:/img/x.png')
  })

  it('escapes "#" in Qt file urls so folders like "art#1" load (and ?t= stays a real query)', () => {
    expect(mediaUrl('C:/a#1/x.png')).toBe('file:///C:/a%231/x.png')
    expect(mediaUrl('file:///D:/#x/y.webm')).toBe('file:///D:/%23x/y.webm')
    const busted = mediaUrl('C:/a#1/x.png', true)
    expect(busted).toMatch(/^file:\/\/\/C:\/a%231\/x\.png\?t=\d+$/)
    const parsed = new URL(busted)
    expect(decodeURIComponent(parsed.pathname)).toBe('/C:/a#1/x.png')
    expect(parsed.hash).toBe('')
    // 이미 인코딩된 file URL 은 다시 인코딩하지 않는다
    expect(mediaUrl('file:///C:/a%231/x.png')).toBe('file:///C:/a%231/x.png')
    g.window = { __AISTUDIO_WS_PORT__: 7801 }
    expect(mediaUrl('C:/a#1/x.png')).toBe('/file?path=C%3A%2Fa%231%2Fx.png')
  })

  it('keeps a literal "%" in raw Windows paths (Qt would otherwise decode %XX into another file)', () => {
    // 예전: `C:/art%231/a.png` → `file:///C:/art%231/a.png` → Chromium 이 `C:/art#1/a.png` 를 열었다
    const art = mediaUrl('C:/art%231/a.png')
    expect(art).toBe('file:///C:/art%25231/a.png')
    expect(decodeURIComponent(new URL(art).pathname)).toBe('/C:/art%231/a.png')
    expect(mediaUrl('C:/dl/image%20(1).png')).toBe('file:///C:/dl/image%2520(1).png')
    expect(mediaUrl('C:/x/100%.png')).toBe('file:///C:/x/100%25.png')
    // `%` 다음에 `#` — 순서가 바뀌면 `%23` 의 `%` 가 다시 `%25` 로 이중 인코딩된다
    expect(mediaUrl('C:/50% #1/a.png')).toBe('file:///C:/50%25 %231/a.png')
    const busted = mediaUrl('C:/dl/a%2Bb.png', true)
    expect(busted).toMatch(/^file:\/\/\/C:\/dl\/a%252Bb\.png\?t=\d+$/)
    expect(decodeURIComponent(new URL(busted).pathname)).toBe('/C:/dl/a%2Bb.png')
    // 이미 인코딩된 file URL 은 그대로(대소문자 무관한 스킴도)
    expect(mediaUrl('file:///C:/dl/a%20b.png')).toBe('file:///C:/dl/a%20b.png')
    expect(mediaUrl('FILE:///C:/dl/a%20b.png')).toBe('file:///C:/dl/a%20b.png')
  })

  it('sends the server a literal path in web mode — file URLs are decoded here, raw paths are not', () => {
    g.window = { __AISTUDIO_WS_PORT__: 7801 }
    expect(mediaUrl('C:/art%231/a.png')).toBe('/file?path=C%3A%2Fart%25231%2Fa.png')
    expect(mediaUrl('file:///C:/a%20b.png')).toBe('/file?path=C%3A%2Fa%20b.png')
    // 깨진 시퀀스를 담은 file URL 은 원문 그대로 보낸다(예외로 멈추지 않는다)
    expect(mediaUrl('file:///C:/x/100%.png')).toBe('/file?path=C%3A%2Fx%2F100%25.png')
    expect(thumbnailUrl('file:///C:/img/a%20b.png', 300)).toBe('/thumbnail?path=C%3A%2Fimg%2Fa%20b.png&width=384')
    expect(thumbnailUrl('C:/img/a%20b.png', 300)).toBe('/thumbnail?path=C%3A%2Fimg%2Fa%2520b.png&width=384')
  })

  it('thumbnail urls carry a literal path in the Qt app too (parse_thumb_url decodes exactly once)', () => {
    expect(thumbnailUrl('file:///C:/img/a%20b.png', 200)).toBe('aithumb:thumb?path=C%3A%2Fimg%2Fa%20b.png&width=256')
    expect(thumbnailUrl('C:/img/a%20b.png', 200)).toBe('aithumb:thumb?path=C%3A%2Fimg%2Fa%2520b.png&width=256')
    const round = new URLSearchParams(thumbnailUrl('C:/img/a%20b.png', 200).split('?')[1]).get('path')
    expect(round).toBe('C:/img/a%20b.png')
  })

  it('uses the HTTP endpoint in web mode and leaves ready urls alone', () => {
    g.window = { __AISTUDIO_WS_PORT__: 7801 }
    expect(thumbnailUrl('C:/img/x.png', 300)).toBe('/thumbnail?path=C%3A%2Fimg%2Fx.png&width=384')
    expect(thumbnailUrl('data:image/png;base64,AA', 300)).toBe('data:image/png;base64,AA')
    expect(thumbnailUrl('https://e.x/a.png', 300)).toBe('https://e.x/a.png')
    expect(thumbnailUrl('', 300)).toBe('')
  })

  it('adds the content version so an overwritten file gets a new card url (Qt and web)', () => {
    // 같은 경로·폭이라도 버전이 다르면 URL 이 다르다 — 브라우저가 옛 썸네일을 재사용하지 않는다
    expect(thumbnailUrl('C:/img/x.png', 200, '18a-3f'))
      .toBe('aithumb:thumb?path=C%3A%2Fimg%2Fx.png&width=256&v=18a-3f')
    expect(thumbnailUrl('C:/img/x.png', 200, '')).toBe('aithumb:thumb?path=C%3A%2Fimg%2Fx.png&width=256')
    g.window = { __AISTUDIO_WS_PORT__: 7801 }
    expect(thumbnailUrl('C:/img/x.png', 300, '18a-3f.1700000000000'))
      .toBe('/thumbnail?path=C%3A%2Fimg%2Fx.png&width=384&v=18a-3f.1700000000000')
    expect(thumbnailUrl('data:image/png;base64,AA', 300, '1')).toBe('data:image/png;base64,AA')
  })

  it('withUrlVersion appends v= only when there is a version', () => {
    expect(withUrlVersion('file:///C:/a%231/x.png', '9')).toBe('file:///C:/a%231/x.png?v=9')
    expect(withUrlVersion('/file?path=x', 7)).toBe('/file?path=x&v=7')
    expect(withUrlVersion('/file?path=x', 'a b')).toBe('/file?path=x&v=a%20b')
    expect(withUrlVersion('/file?path=x', '')).toBe('/file?path=x')
    expect(withUrlVersion('/file?path=x', undefined)).toBe('/file?path=x')
    expect(withUrlVersion('blob:abc', '1')).toBe('blob:abc')
    expect(withUrlVersion('', '1')).toBe('')
  })
})
