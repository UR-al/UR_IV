import { describe, expect, it } from 'vitest'
import { droppedImagePaths, newPaths } from './dropPaths'

describe('droppedImagePaths', () => {
  it('QtWebEngine File 에는 path 가 없다 — undefined 를 넣지 않고 건너뛴 수를 센다', () => {
    const result = droppedImagePaths({
      files: [{ name: 'a.png', type: 'image/png' }, { name: 'b.jpg', type: 'image/jpeg' }],
      getData: () => '',
    })
    expect(result.paths).toEqual([])
    expect(result.unresolved).toBe(2)
  })

  it('path 가 있는 환경(Electron 류)이면 그 경로를 슬래시로 정규화해 쓴다', () => {
    const result = droppedImagePaths({
      files: [{ name: 'a.png', type: 'image/png', path: 'C:\\img\\a.png' }, { name: 'note.txt', type: 'text/plain', path: 'C:\\n.txt' }],
    })
    expect(result).toEqual({ paths: ['C:/img/a.png'], unresolved: 0 })
  })

  it('앱 안 히스토리·갤러리 카드의 text/plain 경로를 받는다 (file:/// 접두사 포함)', () => {
    expect(droppedImagePaths({ files: [], getData: () => 'C:/out/generated_1.png' }).paths)
      .toEqual(['C:/out/generated_1.png'])
    expect(droppedImagePaths({ files: [], getData: () => 'file:///D:/%ED%95%9C%EA%B8%80/x.webp' }).paths)
      .toEqual(['D:/한글/x.webp'])
  })

  it('이미지 경로가 아닌 텍스트는 무시하고 중복은 한 번만', () => {
    const text = 'hello world\nC:/a.png\nC:/a.png\nC:/doc.txt'
    expect(droppedImagePaths({ files: [], getData: () => text })).toEqual({ paths: ['C:/a.png'], unresolved: 0 })
  })

  it('dataTransfer 가 없거나 getData 가 던져도 빈 결과', () => {
    expect(droppedImagePaths(null)).toEqual({ paths: [], unresolved: 0 })
    expect(droppedImagePaths({ files: [], getData: () => { throw new Error('denied') } }))
      .toEqual({ paths: [], unresolved: 0 })
  })
})

describe('newPaths', () => {
  it('이미 목록에 있는 경로와 빈 값을 뺀다', () => {
    expect(newPaths(['C:/a.png'], ['C:/a.png', '', 'C:/b.png', 'C:/b.png'])).toEqual(['C:/b.png'])
  })
})
