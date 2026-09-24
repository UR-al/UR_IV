import { describe, expect, it } from 'vitest'
import { PristineSource, restoreStrokeMode } from './pristineSnapshot'

describe('PristineSource', () => {
  it('accepts only the decode of the latest requested source', () => {
    const source = new PristineSource()
    const first = source.request('file:///C:/tmp/job1.png')
    const second = source.request('file:///C:/tmp/job2.png')
    expect(source.accepts(first)).toBe(false)         // 늦게 끝난 옛 출처 — 버린다
    expect(source.accepts(second)).toBe(true)
    expect(source.current).toBe('file:///C:/tmp/job2.png')
  })

  it("drops document A's in-flight decode after a document switch", () => {
    const source = new PristineSource()
    const forA = source.request('file:///C:/a/pre-effect.png')
    source.reset()                                    // EditorView._resetDocTransients → resetPristine
    expect(source.accepts(forA)).toBe(false)
    expect(source.requested).toBe(false)
    // 부모의 pristinePath 가 '' 로 바뀐 것이 뒤늦게 prop 으로 와도 여전히 없음이다
    const empty = source.request('')
    expect(source.accepts(empty)).toBe(false)
    expect(source.requested).toBe(false)
  })

  it('an empty source means there is nothing to restore', () => {
    const source = new PristineSource()
    expect(source.requested).toBe(false)
    expect(source.accepts(source.request(''))).toBe(false)
  })

  it('records a decode failure only for the current source', () => {
    const source = new PristineSource()
    const old = source.request('file:///C:/tmp/job1.png')
    const cur = source.request('file:///C:/tmp/job2.png')
    source.fail(old)                                  // 옛 출처의 늦은 실패 — 새 출처와 무관
    expect(source.failed).toBe(false)
    source.fail(cur)
    expect(source.failed).toBe(true)
    expect(source.requested).toBe(true)               // 출처는 그대로 — 커밋은 이 경로로 간다
    source.request('file:///C:/tmp/job3.png')         // 새 출처 — 다시 디코드를 기다린다
    expect(source.failed).toBe(false)
  })

  it('a document switch clears the failure, and an empty source never fails', () => {
    const source = new PristineSource()
    source.fail(source.request('file:///C:/a/gone.png'))
    expect(source.failed).toBe(true)
    source.reset()
    expect(source.failed).toBe(false)
    source.fail(source.request(''))
    expect(source.failed).toBe(false)
  })
})

describe('restoreStrokeMode', () => {
  const img = { width: 64, height: 32 }
  it('paints only from a decoded snapshot of the same size', () => {
    expect(restoreStrokeMode(true, { width: 64, height: 32 }, img)).toBe('paint')
    expect(restoreStrokeMode(true, null, img)).toBe('skip')                         // 디코드 중
    expect(restoreStrokeMode(true, { width: 32, height: 64 }, img)).toBe('skip')    // 회전·자르기 뒤
  })
  it('marks the area without painting when nothing can be restored (the parent refuses with a toast)', () => {
    expect(restoreStrokeMode(false, null, img)).toBe('mark')
  })
  it('marks (commits without painting) when the pristinePath decode failed instead of skipping forever', () => {
    expect(restoreStrokeMode(true, null, img, true)).toBe('mark')
    expect(restoreStrokeMode(true, null, img, false)).toBe('skip')                  // 아직 디코드 중
  })
})

/**
 * EditorView(pristinePath·undo/redo·commitRestore) + EditorCanvas(스냅숏 디코드·지우개) + 백엔드
 * ('restore' = pristinePath 파일에서 마스크 자리 픽셀을 가져옴)를 픽셀 배열로 흉내 낸다.
 * 불변식: 드래그 중 화면 == 커밋 뒤 파일. 예전(캔버스가 로드한 이미지로 스냅숏을 뜨고 효과 직후
 * 한 번만 유지)에는 효과×2, 효과→조정, 효과×2→undo→redo, 효과→복원→효과에서 둘이 갈렸다.
 */
class EditorModel {
  private files = new Map<string, number[]>()
  private seq = 0
  private undoStack: string[] = []
  private redoStack: string[] = []
  private source = new PristineSource()
  private pending: Array<{ token: number; path: string }> = []
  imagePath = ''
  pristinePath = ''
  snapshot: number[] | null = null

  private write(px: number[]): string {
    const path = `C:/tmp/job${++this.seq}.png`
    this.files.set(path, [...px])
    return path
  }
  file(path = this.imagePath): number[] { return [...(this.files.get(path) || [])] }
  private push(path: string) { this.undoStack.push(path); this.redoStack = []; this.imagePath = path }
  /** 부모의 pristinePath 가 바뀌었다 → :pristine-src → 캔버스 loadPristine */
  private setPristine(path: string) {
    this.pristinePath = path
    const token = this.source.request(path)
    this.snapshot = null
    if (path) this.pending.push({ token, path })
  }
  /** 캔버스의 디코드가 (요청 순서와 상관없이) 끝난다 */
  decode(order: 'fifo' | 'lifo' = 'fifo') {
    const jobs = order === 'fifo' ? this.pending : [...this.pending].reverse()
    for (const job of jobs) if (this.source.accepts(job.token)) this.snapshot = this.file(job.path)
    this.pending = []
  }
  /** 캔버스의 디코드가 실패한다(EditorCanvas img.onerror — 연 채 파일이 지워짐·잠김) */
  failDecode() {
    for (const job of this.pending) this.source.fail(job.token)
    this.pending = []
  }
  /** 문서를 연 채 파일이 사라진다(임시 폴더 정리 등) */
  remove(path: string) { this.files.delete(path) }
  open(px: number[]) {
    this.source.reset(); this.snapshot = null
    this.setPristine('')
    const path = this.write(px)
    this.undoStack = [path]; this.redoStack = []; this.imagePath = path
  }
  effect(indices: number[]) {
    this.setPristine(this.imagePath)                  // applyEffect: pristinePath = imagePath
    const out = this.file()
    for (const i of indices) out[i] = 99
    this.push(this.write(out))
  }
  adjust(delta: number) { this.push(this.write(this.file().map(v => (v === 99 ? v : v + delta)))) }
  undo() { this.redoStack.push(this.undoStack.pop() as string); this.imagePath = this.undoStack[this.undoStack.length - 1] }
  redo() { const p = this.redoStack.pop() as string; this.undoStack.push(p); this.imagePath = p }
  /** 지우개로 indices 를 문지르고 뗀다 — 드래그 중 화면과 커밋 뒤 파일(error = 백엔드 거절 토스트) */
  erase(indices: number[]): { screen: number[]; file: number[]; committed: boolean; error?: string } {
    const cur = this.file()
    const shape = (a: number[]) => ({ width: a.length, height: 1 })
    const mode = restoreStrokeMode(this.source.requested, this.snapshot && shape(this.snapshot), shape(cur),
      this.source.failed)
    const screen = [...cur]
    if (mode === 'paint') for (const i of indices) screen[i] = (this.snapshot as number[])[i]
    if (mode === 'skip' || !this.pristinePath) return { screen, file: cur, committed: false }   // 커밋 안 함 / 부모가 거절
    if (!this.files.has(this.pristinePath)) {
      // 백엔드 'restore' 가 원본을 못 찾았다 — 토스트 + drawAll(파일과 같은 그림으로 다시 그림)
      return { screen: cur, file: cur, committed: true, error: '복원 원본 이미지를 찾을 수 없습니다' }
    }
    const src = this.file(this.pristinePath)
    const out = [...cur]
    for (const i of indices) out[i] = src[i]
    this.push(this.write(out))                        // commitRestore → doOp('restore') → pushState
    return { screen, file: this.file(), committed: true }
  }
}

describe('eraser screen matches the committed file', () => {
  const ORIG = [10, 11, 12, 13]

  it('after two effects in a row (the reported case)', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0]); m.decode()
    m.effect([2]); m.decode()
    expect(m.snapshot).toEqual([99, 11, 12, 13])      // 효과 1 결과 = 지금 pristinePath — 원본이 아니다
    const r = m.erase([0])
    expect(r.screen).toEqual(r.file)                  // 예전: 화면 [10,11,99,13] → 커밋 뒤 [99,11,99,13]
    expect(r.file).toEqual([99, 11, 99, 13])
    const r2 = m.erase([2])
    expect(r2.screen).toEqual(r2.file)
    expect(r2.file).toEqual([99, 11, 12, 13])
  })

  it('after an effect followed by a colour adjustment', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0]); m.decode()
    m.adjust(1)
    const r = m.erase([1])
    expect(r.screen).toEqual(r.file)                  // 예전: 화면은 그대로, 파일만 되돌아갔다
  })

  it('after two effects, undo and redo', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0]); m.decode()
    m.effect([2]); m.decode()
    m.undo(); m.redo()
    const r = m.erase([2])
    expect(r.screen).toEqual(r.file)
  })

  it('after effect → restore commit → effect', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0, 1]); m.decode()
    m.erase([1])
    m.effect([3]); m.decode()
    const r = m.erase([0])
    expect(r.screen).toEqual(r.file)
  })

  it('a late decode of an older pristinePath never becomes the snapshot', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0])
    m.effect([2])
    m.decode('lifo')                                  // 효과 2 직전 그림이 먼저, 원본이 나중에 끝났다
    expect(m.snapshot).toEqual([99, 11, 12, 13])
    const r = m.erase([0])
    expect(r.screen).toEqual(r.file)
  })

  it('before the snapshot has decoded, nothing is painted and nothing is committed', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0])                                     // pristinePath 디코드 중
    const before = m.file()
    const r = m.erase([0])
    expect(r.screen).toEqual(before)
    expect(r.file).toEqual(before)
    expect(r.committed).toBe(false)
  })

  it('a pristinePath deleted while open: nothing is painted, the commit is sent and the backend reports it', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0])
    m.remove(m.pristinePath)                          // 연 채로 '적용 전' 파일이 지워졌다
    m.failDecode()                                    // 캔버스 디코드 실패(img.onerror)
    const before = m.file()
    const r = m.erase([0])
    // 예전: 'skip' 이 계속돼 칠하지도, 커밋하지도, 알리지도 않았다(지우개가 조용히 멈춤)
    expect(r.committed).toBe(true)
    expect(r.error).toBe('복원 원본 이미지를 찾을 수 없습니다')
    expect(r.screen).toEqual(before)                  // 칠하지 않았다 — 화면 == 파일
    expect(r.file).toEqual(before)
  })

  it('a failed decode of a still-readable pristinePath still restores through the backend', () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0])
    m.failDecode()                                    // 브라우저 쪽만 못 읽었다(잠김 등) — 파일은 있다
    const r = m.erase([0])
    expect(r.committed).toBe(true)
    expect(r.error).toBeUndefined()
    expect(r.screen).toEqual([99, 11, 12, 13])        // 드래그 중에는 칠하지 않는다
    expect(r.file).toEqual(ORIG)                      // 커밋 결과가 화면에 온다(pushState)
  })

  it("document A's pre-effect image is not painted onto document B", () => {
    const m = new EditorModel()
    m.open(ORIG)
    m.effect([0])                                     // A 의 pristinePath 디코드 중에
    m.open([20, 21, 22, 23])                          // 같은 크기의 B 를 연다
    m.decode()
    expect(m.snapshot).toBeNull()
    const r = m.erase([1])                            // 효과 전 — 칠하지 않고 부모가 거절한다
    expect(r.screen).toEqual([20, 21, 22, 23])
    expect(r.file).toEqual([20, 21, 22, 23])
  })
})
