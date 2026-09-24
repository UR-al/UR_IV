import { describe, expect, it } from 'vitest'
import {
  filenameOf, isAnimated, isImage, isPng, mediaExtension, mediaKind, mediaLabel,
  replaceMediaPath, sortMediaPaths,
} from './mediaKind'

describe('mediaKind', () => {
  it('classifies by extension, ignoring query strings and case', () => {
    expect(mediaExtension('C:/a/b.PNG?t=1')).toBe('png')
    expect(mediaKind('C:\\out\\clip.MP4')).toBe('video')
    expect(mediaKind('song.flac')).toBe('audio')
    expect(mediaKind('noext')).toBe('image')
    expect(isImage('a.webp')).toBe(true)
    expect(isAnimated('a.webp')).toBe(true)
    expect(isPng('a.PNG')).toBe(true)
    expect(isPng('a.jpg')).toBe(false)
  })

  // '#' 는 Windows 파일·폴더 이름에 쓸 수 있다 — 로컬 경로에선 URL 프래그먼트가 아니다
  it('keeps "#" in local paths but still strips real URL fragments', () => {
    expect(mediaExtension('C:/media/artist#1/clip.mp4')).toBe('mp4')
    expect(mediaKind('C:/media/artist#1/clip.mp4')).toBe('video')
    expect(mediaKind('C:\\m\\#tag\\song.flac')).toBe('audio')
    expect(mediaKind('C:/media/plain/clip#2.mp4')).toBe('video')
    expect(isPng('D:/#aiart/out.png')).toBe(true)
    expect(isAnimated('C:/a#1/anim.webp')).toBe(true)
    expect(mediaLabel('C:/a#1/x.mov')).toBe('VIDEO')
    expect(mediaKind('file:///D:/#x/y.webm')).toBe('video')
    expect(mediaKind('\\\\nas\\#share\\s.flac')).toBe('audio')
    expect(mediaExtension('file:///C:/a%231/x.png?t=12')).toBe('png')
    expect(mediaExtension('https://e.x/a.mp4#t=3')).toBe('mp4')
    expect(mediaExtension('https://e.x/a.mp4?x=1#t=3')).toBe('mp4')
  })

  it('labels and names media', () => {
    expect(mediaLabel('a.mov')).toBe('VIDEO')
    expect(mediaLabel('a.opus')).toBe('AUDIO')
    expect(mediaLabel('a.gif')).toBe('ANIMATED')
    expect(mediaLabel('a.png')).toBe('IMAGE')
    expect(filenameOf('C:\\out\\sub\\x.png')).toBe('x.png')
  })

  it('derives the sort order without mutating the source list', () => {
    const list = ['C:/o/b.png', 'C:/o/a.png', 'C:/o/c.png']
    const byName = sortMediaPaths(list, 'name')
    expect(byName).toEqual(['C:/o/a.png', 'C:/o/b.png', 'C:/o/c.png'])
    expect(list).toEqual(['C:/o/b.png', 'C:/o/a.png', 'C:/o/c.png'])
    expect(sortMediaPaths(list, 'date')).toBe(list)
  })

  it('replaces a renamed path regardless of slash style', () => {
    expect(replaceMediaPath(['C:/o/a.png', 'C:/o/b.png'], 'C:\\o\\A.png', 'C:/o/z.png')).toEqual(['C:/o/z.png', 'C:/o/b.png'])
    expect(replaceMediaPath(['C:/o/a.png'], 'C:/o/q.png', 'C:/o/z.png')).toBeNull()
  })
})
