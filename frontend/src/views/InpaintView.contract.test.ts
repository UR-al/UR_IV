import { describe, expect, it } from 'vitest'
import source from './InpaintView.vue?raw'

/**
 * InpaintView ↔ core/inpaint_payload.py 계약 (#33, #64).
 * 캔버스를 마운트하지 않고 소스로 고정한다 — 규칙이 조용히 되돌아가지 않게.
 */
const script = source.slice(source.indexOf('<script setup'), source.indexOf('</script>'))

function body(name: string): string {
  const start = script.indexOf(`function ${name}(`)
  expect(start, `function ${name}`).toBeGreaterThanOrEqual(0)
  const next = script.indexOf('\nfunction ', start + 1)
  const nextAsync = script.indexOf('\nasync function ', start + 1)
  const ends = [next, nextAsync].filter((i) => i > 0)
  return script.slice(start, ends.length ? Math.min(...ends) : undefined)
}

describe('InpaintView 이미지 로드', () => {
  it('파일 선택·드롭은 경로를 늘 다시 정한다 — 옛 이미지 경로를 남기지 않는다', () => {
    const loadFile = body('loadFile')
    expect(loadFile).toMatch(/imagePath\.value = typeof nativePath === 'string' && nativePath \? .* : ''/)
    expect(body('handleDrop')).not.toMatch(/imagePath\.value =/)
  })

  it('경로로 받은 이미지는 mediaUrl 로 표시한다 (웹 모드 /file?path=)', () => {
    const loadFromPath = script.slice(script.indexOf('async function loadFromPath'), script.indexOf('function beginImageLoad'))
    expect(loadFromPath).toContain('mediaUrl(normalized)')
    expect(loadFromPath).not.toContain("'file:///' +")
  })

  it('이미지를 못 읽으면 안내하고 경로·마스크를 비운다', () => {
    const initCanvas = body('initCanvas')
    expect(initCanvas).toMatch(/img\.onerror = \(\) => \{[^}]*failImageLoad\(\)/)
    const fail = body('failImageLoad')
    expect(fail).toContain('resetAfterLoadError()')
    expect(fail).toContain("'show_toast'")
    const reset = body('resetAfterLoadError')
    expect(reset).toContain("imagePath.value = ''")
    expect(reset).toContain("imageSrc.value = ''")
    expect(reset).toContain('maskData = null')
  })

  it('파일을 읽는 동안(FileReader 비동기)에도 generate 가 막힌다 — 읽기 전에 준비 상태를 내린다', () => {
    // 예전엔 initCanvas(=FileReader 완료 뒤)에서야 handImageReady 를 내려, 그 사이 generate 가
    // 옛 imageSrc 를 image_path 없이·옛 마스크와 함께 보냈다.
    const loadFile = body('loadFile')
    const begin = loadFile.indexOf('beginImageLoad()')
    expect(begin).toBeGreaterThan(-1)
    expect(begin).toBeLessThan(loadFile.indexOf('readAsDataURL('))
    expect(loadFile).toMatch(/r\.onerror = \(\) => \{[^}]*failImageLoad\(\)/)
    // 늦게 끝난 읽기가 더 최근 이미지를 덮지 않는다
    expect(loadFile).toMatch(/r\.onload = \(\) => \{\s*if \(loadRevision !== imageLoadRevision\) return/)
    expect(loadFile).toContain('initCanvas(src, loadRevision)')

    const beginBody = body('beginImageLoad')
    expect(beginBody).toContain('++imageLoadRevision')
    expect(beginBody).toContain('handSourceRevision.value++')
    expect(beginBody).toContain('handImageReady.value = false')
  })

  it('모든 로드 경로가 같은 준비 규칙을 쓴다 — initCanvas 는 받은 revision 만 확인한다', () => {
    const calls = script.match(/initCanvas\([^\n]*/g) || []
    const callers = calls.filter((call) => !call.startsWith('initCanvas(src: string'))
    expect(callers.length).toBeGreaterThanOrEqual(3)
    for (const call of callers) expect(call).toMatch(/initCanvas\([^,]+, (loadRevision|beginImageLoad\(\))\)/)
    const initCanvas = body('initCanvas')
    expect(initCanvas).not.toContain('++imageLoadRevision')
    expect(initCanvas).toMatch(/^[^{]*\{\s*if \(loadRevision !== imageLoadRevision\) return/)
  })
})

describe('InpaintView generate_inpaint 페이로드', () => {
  it('마스크 초기값·인페인트 범위·샘플링 값을 모두 보낸다', () => {
    const generate = body('generate')
    for (const key of ['mask_content', 'inpaint_area', 'negative_prompt', 'mask_blur', 'padding', 'steps', 'cfg', 'seed', 'denoising']) {
      expect(generate).toContain(`${key}:`)
    }
  })

  it('화면 기본값은 그동안 실제로 나가던 값(원본 유지 · 마스크 영역만)', () => {
    expect(script).toMatch(/const maskContent = ref\(1\)/)
    expect(script).toMatch(/const inpaintArea = ref\(1\)/)
  })

  it('이미지를 읽는 중이거나 마스크가 없으면 보내지 않는다', () => {
    const generate = body('generate')
    expect(generate).toMatch(/if \(!handImageReady\.value\)/)
    expect(generate).toMatch(/if \(!hasMask\.value\)/)
  })
})
