import { describe, expect, it } from 'vitest'
import {
  applyDeleteToHistory,
  imagePathKey,
  isSameImagePath,
  parseImageDeleteResult,
  withoutImagePath,
} from './imageDeleteResult'

const moved = { path: 'D:/out/a.png', ok: true, removed: true, level: 'info', message: '휴지통으로 이동됨' }
const failed = { path: 'D:/out/a.png', ok: false, removed: false, level: 'error', message: '휴지통으로 옮기지 못했습니다: denied' }
const gone = { path: 'D:/out/a.png', ok: false, removed: true, level: 'warning', message: '파일이 이미 없어 휴지통으로 옮길 것이 없습니다' }

describe('parseImageDeleteResult', () => {
  it('reads the JSON signal payload', () => {
    expect(parseImageDeleteResult(JSON.stringify(moved))).toEqual(moved)
    expect(parseImageDeleteResult(failed)).toEqual(failed)
  })

  it('rejects malformed payloads so lists are never touched by accident', () => {
    for (const raw of ['', '{', 'null', '[]', '{"ok":true,"removed":true}', JSON.stringify({ path: '  ', removed: true }), 5, null]) {
      expect(parseImageDeleteResult(raw)).toBeNull()
    }
  })

  it('only a literal true counts as removed/ok, unknown levels become error', () => {
    const r = parseImageDeleteResult({ path: 'x.png', ok: 'true', removed: 1, level: 'fatal' })!
    expect(r.ok).toBe(false)
    expect(r.removed).toBe(false)
    expect(r.level).toBe('error')
    expect(r.message).toBe('')
  })
})

describe('path keys', () => {
  it('treats slash, file:// and Windows case differences as the same file', () => {
    expect(imagePathKey('D:\\Out\\A.png')).toBe(imagePathKey('d:/out/a.png'))
    expect(imagePathKey('file:///D:/out/a.png')).toBe(imagePathKey('D:\\out\\a.png'))
    expect(imagePathKey('D:/out//a.png')).toBe(imagePathKey('D:/out/a.png'))
    expect(isSameImagePath('\\\\NAS\\Share\\x.png', '//nas/share/x.png')).toBe(true)
  })

  it('decodes file URLs but keeps a raw path\'s % as part of the name (backend strip_file_url rule)', () => {
    expect(imagePathKey('file:///D:/out/a%20b.png')).toBe(imagePathKey('D:\\out\\a b.png'))
    expect(imagePathKey('file:///D:/%ED%95%9C%EA%B8%80/x.png')).toBe(imagePathKey('D:/한글/x.png'))
    // 원시 이름의 '%20' 은 글자 그대로 — 'a b.png' 와 다른 파일, file URL 에선 %2520
    expect(isSameImagePath('D:/out/a%20b.png', 'D:/out/a b.png')).toBe(false)
    expect(isSameImagePath('file:///D:/out/a%20b.png', 'D:/out/a%20b.png')).toBe(false)
    expect(isSameImagePath('file:///D:/out/a%2520b.png', 'D:/out/a%20b.png')).toBe(true)
    const r = applyDeleteToHistory(['D:/out/a%20b.png', 'D:/out/a b.png'], 'D:/out/a b.png',
      parseImageDeleteResult({ ...moved, path: 'file:///D:/out/a%20b.png' }))!
    expect(r.history).toEqual(['D:/out/a%20b.png'])
    expect(r.current).toBe('D:/out/a%20b.png')
  })

  it('keeps POSIX case and the UNC double slash', () => {
    expect(isSameImagePath('/home/me/A.png', '/home/me/a.png')).toBe(false)
    expect(imagePathKey('\\\\nas\\share\\x.png').startsWith('//')).toBe(true)
    expect(isSameImagePath('', 'a.png')).toBe(false)
  })

  it('withoutImagePath returns null when nothing matched', () => {
    const list = ['D:/out/a.png', 'D:/out/b.png']
    expect(withoutImagePath(list, 'D:/out/c.png')).toBeNull()
    expect(withoutImagePath(list, 'D:\\OUT\\A.png')).toEqual(['D:/out/b.png'])
    expect(list).toEqual(['D:/out/a.png', 'D:/out/b.png'])   // 원본은 그대로
  })
})

describe('applyDeleteToHistory', () => {
  const history = ['D:/out/c.png', 'D:\\out\\a.png', 'D:/out/b.png']

  it('a failed trash move leaves history and the current image alone', () => {
    expect(applyDeleteToHistory(history, 'D:\\out\\a.png', parseImageDeleteResult(failed))).toBeNull()
    expect(applyDeleteToHistory(history, 'D:\\out\\a.png', null)).toBeNull()
  })

  it('a successful move removes the entry and moves off the deleted current image', () => {
    const r = applyDeleteToHistory(history, 'D:\\out\\a.png', parseImageDeleteResult(moved))!
    expect(r.history).toEqual(['D:/out/c.png', 'D:/out/b.png'])
    expect(r.current).toBe('D:/out/c.png')
  })

  it('an already-missing file is still dropped (nothing left to trash)', () => {
    const r = applyDeleteToHistory(history, 'D:/out/b.png', parseImageDeleteResult(gone))!
    expect(r.history).toEqual(['D:/out/c.png', 'D:/out/b.png'])
    expect(r.current).toBe('D:/out/b.png')   // 다른 그림을 보고 있으면 그대로
  })

  it('clears the viewer when the last image is removed and ignores unrelated paths', () => {
    expect(applyDeleteToHistory(['D:/out/a.png'], 'D:/out/a.png', parseImageDeleteResult(moved)))
      .toEqual({ history: [], current: '' })
    expect(applyDeleteToHistory(['D:/out/z.png'], 'D:/out/z.png', parseImageDeleteResult(moved))).toBeNull()
  })
})
