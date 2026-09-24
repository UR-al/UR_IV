import { describe, expect, it } from 'vitest'
import ts from 'typescript'
import source from './EditorCanvas.vue?raw'
import { PristineSource, restoreStrokeMode } from '../../utils/pristineSnapshot'

/**
 * EditorCanvas 의 비동기 디코드 두 가지를 **실제 함수 본문**으로 돌려 본다(캔버스·Image 는 스텁).
 *
 * 1) 자동 감지 마스크(loadMaskFromBase64): 결과가 문서 세대 게이트를 통과한 뒤 data URL 디코드가
 *    끝나기 전에 다른 문서를 열면, onload 가 새 문서 위에 옛 문서의 마스크를 썼다.
 * 2) 모자이크 지우개의 '적용 전' 그림: 예전에는 로드한 이미지로 스냅숏을 뜨고 효과 직후 한 번만
 *    유지해, 효과를 두 번 적용하면 화면(원본)과 커밋(pristinePath = 효과 1 결과)이 갈렸다.
 *    이제 스냅숏은 부모의 pristinePath 파일(props.pristineSrc → loadPristine)에서만 뜬다.
 */

/** `function name(...) {...}` 를 소스에서 그대로 잘라 낸다(최상위 함수). */
function extract(name: string): string {
  const start = source.indexOf(`\nfunction ${name}(`)
  if (start < 0) throw new Error(`EditorCanvas.vue 에 function ${name} 이 없다`)
  let p = start, depth = 0
  for (; p < source.length; p++) {
    const c = source[p]
    if (c === '(') depth++
    else if (c === ')') depth--
    else if (c === '{' && depth === 0) break
  }
  let d = 0, end = p
  for (; end < source.length; end++) {
    if (source[end] === '{') d++
    else if (source[end] === '}') { d--; if (d === 0) break }
  }
  return source.slice(start, end + 1)
}

const FUNCTIONS = [
  'loadNewImage', 'loadMaskFromBase64', 'cancelMaskLoad', 'clearSelection', 'initMask', 'resetMaskBounds',
  'saveMaskState', 'syncMaskHistoryCounts', 'recomputeBounds', 'getSelection', 'emitMaskBounds',
  'loadPristine', 'resetPristine', 'clearRestoreMask', 'ensureRestoreMask', 'restoreLine',
]
const js = ts.transpileModule(FUNCTIONS.map(extract).join('\n\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2020 },
}).outputText

// ── 스텁: 그림은 회색조 픽셀 배열(src → {w,h,px}), 디코드는 테스트가 순서를 정해 끝낸다 ──
interface Decoded { w: number; h: number; px: number[] }
class FakeImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  naturalWidth = 0
  naturalHeight = 0
  px: number[] = []
  private _src = ''
  constructor(private world: World) {}
  set src(v: string) { this._src = v; this.world.pending.push(this) }
  get src() { return this._src }
}
class FakeCanvas {
  width = 0
  height = 0
  data = new Uint8ClampedArray(0)
  getContext() { return this }
  private fit() { if (this.data.length !== this.width * this.height * 4) this.data = new Uint8ClampedArray(this.width * this.height * 4) }
  drawImage(img: FakeImage, _x: number, _y: number, dw = img.naturalWidth, dh = img.naturalHeight) {
    this.fit()
    for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) {
      const v = img.px[Math.floor(y * img.naturalHeight / dh) * img.naturalWidth + Math.floor(x * img.naturalWidth / dw)]
      const i = (y * this.width + x) * 4
      this.data[i] = v; this.data[i + 1] = v; this.data[i + 2] = v; this.data[i + 3] = 255
    }
  }
  getImageData(x0: number, y0: number, w: number, h: number) {
    this.fit()
    const data = new Uint8ClampedArray(w * h * 4)
    for (let y = 0; y < h; y++) data.set(this.data.subarray(((y0 + y) * this.width + x0) * 4, ((y0 + y) * this.width + x0 + w) * 4), y * w * 4)
    return { data, width: w, height: h }
  }
  putImageData(img: { data: Uint8ClampedArray; width: number; height: number }, x0: number, y0: number) {
    for (let y = 0; y < img.height; y++) this.data.set(img.data.subarray(y * img.width * 4, (y + 1) * img.width * 4), ((y0 + y) * this.width + x0) * 4)
  }
  gray(): number[] { return Array.from({ length: this.width * this.height }, (_, i) => this.data[i * 4]) }
}

/**
 * 컴포넌트의 모듈 상태(let sourceImg·maskData·토큰 …)를 함수 스코프 변수로 두고 추출한 함수를 그 안에
 * 정의한다. 반환한 `run` 은 같은 스코프의 직접 eval 이라 상태를 읽고 함수를 부를 수 있다.
 */
const RUNTIME = `
  const { Image, document, PristineSource, restoreStrokeMode, maskHistory, emit } = deps
  const ctx = deps.screen
  const pristineSource = new PristineSource()
  let sourceImg = null, maskData = null, maskImageData = null, lassoPoints = []
  let boundsMinX = Infinity, boundsMinY = Infinity, boundsMaxX = -Infinity, boundsMaxY = -Infinity
  let boundsDirty = false, maskPixelCount = 0
  const hasMask = { value: false }, imgWidth = { value: 0 }, imgHeight = { value: 0 }
  const zoom = { value: 1 }, rotation = { value: 0 }, panX = { value: 0 }, panY = { value: 0 }
  const maskUndoCount = { value: 0 }, maskRedoCount = { value: 0 }
  let savedZoom = 1, savedRotation = 0, savedPanX = 0, savedPanY = 0
  let imageLoadToken = 0, maskLoadToken = 0, requestedImageSrc = '', baseLoadPending = false, hidePreviewOnBaseLoad = false
  let restoreMask = null, restoreDirty = false, pristineImg = null, pristineCtx = null
  function markDirtyAll() {}
  function flushMaskOverlay() {}
  function hidePreviewNow() {}
  // 화면 캔버스 = 지금 원본을 그린 것
  function drawAll() {
    if (!sourceImg) return
    ctx.width = sourceImg.naturalWidth; ctx.height = sourceImg.naturalHeight
    ctx.drawImage(sourceImg, 0, 0)
  }
  ${js}
  return (code) => eval(code)
`

class World {
  pending: FakeImage[] = []
  files = new Map<string, Decoded>()
  screen = new FakeCanvas()
  emitted: unknown[] = []
  history: Uint8Array[] = []
  private evaluate: (code: string) => unknown

  constructor() {
    const world = this
    const deps = {
      Image: class extends FakeImage { constructor() { super(world) } },
      document: { createElement: () => new FakeCanvas() },
      PristineSource, restoreStrokeMode,
      screen: this.screen,
      maskHistory: {
        save: (m: Uint8Array) => { world.history.push(m.slice()) },
        clear: () => { world.history.length = 0 },
        get undoCount() { return world.history.length },
        redoCount: 0,
      },
      emit: (name: string, value: unknown) => { world.emitted.push([name, value]) },
    }
    this.evaluate = new Function('deps', RUNTIME)(deps) as (code: string) => unknown
  }

  run<T = unknown>(code: string): T { return this.evaluate(code) as T }
  /** 다음에 시작된 디코드를 꺼낸다 */
  take(): FakeImage { const img = this.pending.shift(); if (!img) throw new Error('디코드 대기 중인 이미지가 없다'); return img }
  decode(img: FakeImage) {
    const f = this.files.get(img.src)
    if (!f) { img.onerror?.(); return }
    img.naturalWidth = f.w; img.naturalHeight = f.h; img.px = f.px
    img.onload?.()
  }
  /** EditorCanvas 의 watch(() => props.imageSrc) */
  setImageSrc(src: string) { this.run(`loadNewImage(${JSON.stringify(src)}, sourceImg !== null)`) }
  /** EditorCanvas 의 watch(() => props.pristineSrc) */
  setPristineSrc(src: string) { this.run(`loadPristine(${JSON.stringify(src)})`) }
  maskPixels(): number { return this.run<Uint8Array | null>('maskData')?.reduce((n, v) => n + (v ? 1 : 0), 0) ?? 0 }
  /** 열고 디코드까지 끝낸다 */
  open(src: string) { this.setImageSrc(src); this.decode(this.take()) }
}

describe('late auto-detect mask decode', () => {
  const W = 8, H = 8
  function setup(bW: number, bH: number) {
    const world = new World()
    world.files.set('A.png', { w: W, h: H, px: new Array(W * H).fill(50) })
    world.files.set('B.png', { w: bW, h: bH, px: new Array(bW * bH).fill(60) })
    world.files.set('data:mask/A', { w: W, h: H, px: Array.from({ length: W * H }, (_, i) => (i < W * 2 ? 255 : 0)) })
    world.open('A.png')
    world.run(`loadMaskFromBase64('data:mask/A')`)   // A 의 auto_detect 결과 — 게이트 통과
    const mask = world.take()
    return { world, mask }
  }
  /**
   * EditorView.loadImage 순서: _resetDocTransients(cancelMaskLoad) → imageDisplay=B → clearSelection(true)
   * → (다음 flush 에) 캔버스의 imageSrc 감시가 loadNewImage(B).
   */
  function openB(world: World) {
    world.run('cancelMaskLoad()')
    world.run('clearSelection(true)')
    world.setImageSrc('B.png')
    return world.take()
  }

  for (const [label, bW, bH] of [['same size', W, H], ['different size', 16, 16]] as const) {
    it(`does not land on B (${label}) when it decodes before B loads`, () => {
      const { world, mask } = setup(bW, bH)
      const b = openB(world)
      world.decode(mask)
      world.decode(b)
      expect(world.maskPixels()).toBe(0)
      expect(world.emitted).toEqual([])
      expect(world.run('getSelection()')).toBeNull()
    })
    it(`does not land on B (${label}) when it decodes after B loads`, () => {
      const { world, mask } = setup(bW, bH)
      const b = openB(world)
      world.decode(b)
      world.decode(mask)
      expect(world.maskPixels()).toBe(0)
      expect(world.emitted).toEqual([])
      expect(world.run('hasMask')).toEqual({ value: false })
    })
  }

  it('is dropped by Esc (clearSelection) during the decode and by an image swap of the same document', () => {
    const { world, mask } = setup(W, H)
    world.run('clearSelection()')
    world.decode(mask)
    expect(world.maskPixels()).toBe(0)

    world.run(`loadMaskFromBase64('data:mask/A')`)
    const again = world.take()
    world.files.set('A2.png', { w: W, h: H, px: new Array(W * H).fill(70) })
    world.open('A2.png')                               // undo·작업 결과로 원본이 바뀌었다
    world.decode(again)
    expect(world.maskPixels()).toBe(0)
  })

  it('still applies when nothing changed in between', () => {
    const { world, mask } = setup(W, H)
    world.decode(mask)
    expect(world.maskPixels()).toBe(W * 2)
    expect(world.emitted).toEqual([['selection-changed', { x: 0, y: 0, w: W, h: 2 }]])
  })
})

describe("mosaic eraser paints the parent's pristinePath", () => {
  function setup() {
    const world = new World()
    world.files.set('orig.png', { w: 4, h: 1, px: [10, 11, 12, 13] })
    world.files.set('job1.png', { w: 4, h: 1, px: [99, 11, 12, 13] })   // 효과 1: 0번 모자이크
    world.files.set('job2.png', { w: 4, h: 1, px: [99, 11, 99, 13] })   // 효과 2: 2번 모자이크
    world.open('orig.png')
    return world
  }
  /** EditorView.applyEffect: pristinePath = imagePath → 결과가 오면 pushState */
  function effect(world: World, before: string, result: string) {
    world.setPristineSrc(before)
    world.decode(world.take())
    world.setImageSrc(result)
    world.decode(world.take())
  }

  it('after two effects the eraser shows what the restore commit will write', () => {
    const world = setup()
    effect(world, 'orig.png', 'job1.png')
    effect(world, 'job1.png', 'job2.png')
    world.run('restoreLine(0, 0, 0, 0, 1)')            // 효과 1 자리를 지운다
    // 커밋은 pristinePath(job1)에서 0번을 가져온다 = 99. 예전 화면은 원본(10)을 보여 줬다가 커밋 뒤 되살아났다.
    expect(world.screen.gray()).toEqual([99, 11, 99, 13])
    expect(world.run('restoreDirty')).toBe(true)
  })

  it("never paints document A's late pristine decode onto document B", () => {
    const world = setup()
    world.setPristineSrc('orig.png')                   // A 에 효과 적용 — 스냅숏 디코드 중
    const late = world.take()
    world.run('resetPristine()')                       // B 를 연다(_resetDocTransients)
    world.setPristineSrc('')
    world.files.set('B.png', { w: 4, h: 1, px: [20, 21, 22, 23] })
    world.open('B.png')
    world.decode(late)
    expect(world.run('pristineImg')).toBeNull()
    world.run('restoreLine(0, 0, 0, 0, 1)')            // 되돌릴 그림 없음 — 칠하지 않고 영역만 기록
    expect(world.screen.gray()).toEqual([20, 21, 22, 23])
    expect(world.run('restoreDirty')).toBe(true)      // 부모가 '되돌릴 이전 상태가 없습니다'로 거절한다
  })

  it('paints nothing while the new pristine image is still decoding', () => {
    const world = setup()
    effect(world, 'orig.png', 'job1.png')
    world.setPristineSrc('job1.png')                   // 효과 2 — 새 출처 디코드 중
    world.setImageSrc('job2.png')
    world.decode(world.pending.pop() as FakeImage)     // 결과는 먼저 왔다
    world.run('restoreLine(0, 0, 0, 0, 1)')
    expect(world.screen.gray()).toEqual([99, 11, 99, 13])
    expect(world.run('restoreDirty')).toBe(false)     // 커밋도 보내지 않는다
  })

  it('image loads alone never replace the snapshot', () => {
    const world = setup()
    effect(world, 'orig.png', 'job1.png')
    const snap = world.run('pristineImg')
    world.setImageSrc('job2.png')
    world.decode(world.take())
    expect(world.run('pristineImg')).toBe(snap)
  })

  it('a pristinePath that fails to decode still sends the commit (the backend reports it) without painting', () => {
    const world = setup()
    world.setPristineSrc('gone.png')                   // 연 채로 지워진 '적용 전' 파일
    world.decode(world.take())                         // img.onerror
    expect(world.run('pristineImg')).toBeNull()
    world.run('restoreLine(0, 0, 0, 0, 1)')
    expect(world.screen.gray()).toEqual([10, 11, 12, 13])   // 칠하지 않았다 — 화면 == 파일
    // 예전: 'skip' 이 영원히 이어져 영역도 커밋도 없었다(지우개가 조용히 멈춤). 이제 onMouseUp 이
    // restore-ready 를 보내고, 백엔드가 '복원 원본 이미지를 찾을 수 없습니다'를 알린다.
    expect(world.run('restoreDirty')).toBe(true)
    expect(Array.from(world.run<Uint8Array>('restoreMask'))).toEqual([255, 0, 0, 0])   // 문지른 자리만
  })

  it("an older source's late decode failure does not affect the current pristine source", () => {
    const world = setup()
    world.setPristineSrc('gone.png')
    const old = world.take()
    world.setPristineSrc('job1.png')                   // 새 출처 — 디코드 중
    const current = world.take()
    world.decode(old)                                  // 옛 출처가 늦게 실패했다
    world.run('restoreLine(0, 0, 0, 0, 1)')
    expect(world.run('restoreDirty')).toBe(false)     // 새 출처는 아직 디코드 중 — 'skip'
    world.decode(current)
    world.run('restoreLine(0, 0, 0, 0, 1)')
    expect(world.screen.gray()).toEqual([99, 11, 12, 13])
    expect(world.run('restoreDirty')).toBe(true)
  })
})
