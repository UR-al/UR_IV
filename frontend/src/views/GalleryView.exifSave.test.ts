import { describe, expect, it } from 'vitest'
import source from './GalleryView.vue?raw'

/**
 * 갤러리 'EXIF 저장' 응답 처리 — 판단 로직은 utils/exifSaveResponse(단위 테스트)에 있고,
 * 여기서는 뷰가 그것을 실제로 쓰는지만 소스로 고정한다(되돌려도 빌드는 통과한다).
 */
const script = source.slice(source.indexOf('<script setup'), source.indexOf('</script>'))
const template = source.slice(0, source.indexOf('<script'))

function body(name: string): string {
  const start = script.search(new RegExp(`(?:function ${name}\\(|const ${name} = )`))
  expect(start, name).toBeGreaterThanOrEqual(0)
  const rest = script.slice(start + 1)
  const end = rest.search(/\n(?:async function |function |const |let )/)
  return script.slice(start, end > 0 ? start + 1 + end : undefined)
}

describe('GalleryView EXIF 저장 응답', () => {
  it('응답은 보낸 상태와 비교해 반영한다 — 무조건 dirty 를 끄고 덮어쓰지 않는다', () => {
    const save = body('saveExif')
    expect(save).toContain('resolveExifSaveResponse(')
    expect(save).toMatch(/viewGen: gen/)
    expect(save).not.toMatch(/exifDirty\.value = false/)
    expect(save).toMatch(/if \(out\.dirty !== null\) exifDirty\.value = out\.dirty/)
    expect(save).toMatch(/if \(gen === exifViewGen\) exifSaving\.value = false/)
  })

  it('이미지를 열거나 닫으면 뷰 세대를 올리고 저장 중 표시를 푼다', () => {
    for (const name of ['viewImage', 'closeLargeView']) {
      const b = body(name)
      expect(b, name).toContain('exifViewGen++')
      expect(b, name).toContain('exifSaving.value = false')
    }
  })

  it('저장 중에는 프롬프트/네거티브를 고칠 수 없다', () => {
    const editable = template.match(/:contenteditable="[^"]*"/g) || []
    expect(editable.length).toBeGreaterThanOrEqual(2)
    for (const attr of editable) expect(attr).toContain('!exifSaving')
  })
})
