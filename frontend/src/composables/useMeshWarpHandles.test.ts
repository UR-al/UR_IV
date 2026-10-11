import { describe, expect, it } from 'vitest'
import { useMeshWarpHandles } from './useMeshWarpHandles'

/** 마지막 원소 — tsconfig lib 이 Array.prototype.at 이전이라 직접 */
const last = <T>(arr: readonly T[]): T | undefined => arr[arr.length - 1]

function setup(size: { w: number; h: number } | null = { w: 300, h: 300 }) {
  const calls = { redraw: 0, redrawAll: 0, changes: [] as number[][][], pins: [] as boolean[] }
  const handles = useMeshWarpHandles({
    imageSize: () => size,
    imagePerScreenPx: () => 1,
    redraw: () => { calls.redraw++ },
    redrawAll: () => { calls.redrawAll++ },
    onChange: (dst) => { calls.changes.push(dst) },
    onPinEdgesRestored: (on) => { calls.pins.push(on) },
  })
  return { handles, calls }
}

/** 그린 것만 세는 가짜 2D 컨텍스트 */
function fakeCtx() {
  const log: string[] = []
  const ctx: Record<string, unknown> = {}
  for (const m of ['save', 'restore', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc', 'fill', 'strokeRect']) {
    ctx[m] = () => { log.push(m) }
  }
  return { ctx: ctx as unknown as CanvasRenderingContext2D, log }
}

describe('useMeshWarpHandles', () => {
  it('does nothing until begun, and cannot begin without an image', () => {
    const { handles } = setup(null)
    expect(handles.isActive()).toBe(false)
    expect(handles.pointerDown({ x: 100, y: 100 })).toBe(false)
    expect(handles.begin(true)).toBe(false)
    expect(handles.end()).toBeNull()
  })

  it('drags an inner handle and reports normalized points on move and release', () => {
    const { handles, calls } = setup()
    expect(handles.begin(true)).toBe(true)
    expect(calls.redrawAll).toBe(1)
    expect(handles.pointerDown({ x: 105, y: 96 })).toBe(true)   // 점 5 (100, 100) 근처
    expect(handles.pointerMove({ x: 130, y: 120 })).toBe(true)
    expect(last(calls.changes)?.[5]).toEqual([0.43333, 0.4])
    expect(handles.pointerUp()).toBe(true)
    const dst = handles.end()
    expect(dst?.[5]).toEqual([0.43333, 0.4])
    expect(handles.isActive()).toBe(false)
  })

  it('ignores clicks away from handles and on pinned border points', () => {
    const { handles, calls } = setup()
    handles.begin(true)
    expect(handles.pointerDown({ x: 150, y: 150 })).toBe(false)
    expect(handles.pointerDown({ x: 0, y: 0 })).toBe(false)
    expect(handles.pointerMove({ x: 10, y: 10 })).toBe(false)
    expect(calls.changes).toHaveLength(0)
  })

  it('undoStep and reset report the new points', () => {
    const { handles, calls } = setup()
    handles.begin(true)
    handles.pointerDown({ x: 200, y: 200 })
    handles.pointerMove({ x: 220, y: 230 })
    handles.pointerUp()
    const before = calls.changes.length
    expect(handles.undoStep()).toBe(true)
    expect(calls.changes.length).toBe(before + 1)
    expect(last(calls.changes)?.[10]).toEqual([0.66667, 0.66667])
    expect(handles.undoStep()).toBe(false)
    expect(handles.reset()).toBe(false)
  })

  it('unpinning lets border points move; pinning again snaps them back', () => {
    const { handles, calls } = setup()
    handles.begin(false)
    expect(handles.pointerDown({ x: 100, y: 2 })).toBe(true)
    handles.pointerMove({ x: 110, y: 40 })
    handles.pointerUp()
    handles.setPinEdges(true)
    expect(last(calls.changes)?.[1]).toEqual([0.33333, 0])
  })

  it('undo that also restores the pin tells the parent so the checkbox follows', () => {
    const { handles, calls } = setup()
    handles.begin(false)
    expect(handles.pointerDown({ x: 100, y: 0 })).toBe(true)     // 테두리 점 1
    handles.pointerMove({ x: 120, y: 30 })
    handles.pointerUp()
    handles.setPinEdges(true)
    expect(last(calls.changes)?.[1]).toEqual([0.33333, 0])
    expect(handles.undoStep()).toBe(true)
    expect(calls.pins).toEqual([false])
    expect(last(calls.changes)?.[1]).toEqual([0.4, 0.1])
    expect(handles.undoStep()).toBe(true)                          // 점 이동만 되돌린 칸 — 고정은 그대로
    expect(calls.pins).toEqual([false])
  })

  it('draws the grid while active only', () => {
    const { handles } = setup()
    const { ctx, log } = fakeCtx()
    handles.draw(ctx)
    expect(log).toHaveLength(0)
    handles.begin(true)
    handles.draw(ctx)
    expect(log.filter(c => c === 'arc')).toHaveLength(4)          // 움직이는 안쪽 점 4개
    expect(log.filter(c => c === 'strokeRect')).toHaveLength(12)  // 고정된 테두리 12개
  })

  it('cancel stops without reporting', () => {
    const { handles, calls } = setup()
    handles.begin(true)
    handles.cancel()
    expect(handles.isActive()).toBe(false)
    expect(calls.redrawAll).toBe(2)
    expect(calls.changes).toHaveLength(0)
  })
})
