import { describe, expect, it } from 'vitest'
import { stripFileUrl } from './fileUrl'

// tests/test_path_safety.py 의 strip_file_url 케이스와 같은 기대값 — 두 구현이 갈라지지 않게.
describe('stripFileUrl', () => {
  it('file URL 은 스킴을 떼고 퍼센트 인코딩을 푼다', () => {
    expect(stripFileUrl('file:///C:/dl/a%20b.png')).toBe('C:/dl/a b.png')
    expect(stripFileUrl('FILE:///C:/dl/a%20b.png')).toBe('C:/dl/a b.png')
    expect(stripFileUrl('file://C:/dl/a%20b.png')).toBe('C:/dl/a b.png')
    expect(stripFileUrl('file://server/share/a%20b.png')).toBe('server/share/a b.png')
    expect(stripFileUrl('file:///D:/%ED%95%9C%EA%B8%80/x.webp')).toBe('D:/한글/x.webp')
    expect(stripFileUrl('file:///C:/a%231/x.png')).toBe('C:/a#1/x.png')
  })

  it('원시 경로의 % 는 이름의 일부 — 디코드하지 않는다', () => {
    expect(stripFileUrl('C:/dl/image%20(1).png')).toBe('C:/dl/image%20(1).png')
    expect(stripFileUrl('C:\\x\\a%2520b.png')).toBe('C:\\x\\a%2520b.png')
    expect(stripFileUrl('/home/me/100%.png')).toBe('/home/me/100%.png')
  })

  it('이중 인코딩은 한 번만 푼다(원시 이름의 %20 이 file URL 에선 %2520)', () => {
    expect(stripFileUrl('file:///C:/dl/image%2520(1).png')).toBe('C:/dl/image%20(1).png')
  })

  it('깨진 시퀀스는 그 자리만 그대로 두고 나머지는 푼다 (Python unquote 처럼)', () => {
    expect(stripFileUrl('file:///C:/x/100%.png')).toBe('C:/x/100%.png')
    expect(stripFileUrl('file:///C:/a%20b/100%.png')).toBe('C:/a b/100%.png')
    expect(stripFileUrl('file:///C:/x/%zz%20y.png')).toBe('C:/x/%zz y.png')
  })

  it('빈 값·null 은 빈 문자열', () => {
    expect(stripFileUrl('')).toBe('')
    expect(stripFileUrl(null)).toBe('')
    expect(stripFileUrl(undefined)).toBe('')
  })
})
