import { describe, expect, it } from 'vitest'
import {
  MESH_HISTORY_MAX, MESH_PIN_STORAGE_KEY, MeshSession, clampMeshPoint, hitMeshHandle, isEdgeIndex,
  isIdentityMesh, isIdentityNormalized, loadMeshPinEdges, meshGrid, saveMeshPinEdges, snapEdgesToGrid,
  toNormalized,
} from './meshWarp'

function memoryStorage(init: Record<string, string> = {}) {
  const data = { ...init }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => { data[k] = String(v) },
  }
}

describe('meshGrid', () => {
  it('is 4×4 row-major from the top-left corner to the bottom-right corner', () => {
    const g = meshGrid(300, 150)
    expect(g).toHaveLength(16)
    expect(g[0]).toEqual({ x: 0, y: 0 })
    expect(g[1]).toEqual({ x: 100, y: 0 })
    expect(g[4]).toEqual({ x: 0, y: 50 })
    expect(g[5]).toEqual({ x: 100, y: 50 })
    expect(g[15]).toEqual({ x: 300, y: 150 })
  })

  it('marks the twelve border points', () => {
    const edges = Array.from({ length: 16 }, (_, i) => i).filter(isEdgeIndex)
    expect(edges).toHaveLength(12)
    expect([5, 6, 9, 10].some(isEdgeIndex)).toBe(false)
  })
})

describe('hitMeshHandle', () => {
  const g = meshGrid(300, 300)
  it('picks the nearest handle inside the radius', () => {
    expect(hitMeshHandle(g, 103, 98, 10, false)).toBe(5)
    expect(hitMeshHandle(g, 150, 150, 10, false)).toBe(-1)
  })
  it('skips pinned border points', () => {
    expect(hitMeshHandle(g, 2, 2, 10, true)).toBe(-1)
    expect(hitMeshHandle(g, 2, 2, 10, false)).toBe(0)
  })
})

describe('clampMeshPoint', () => {
  const w = 300, h = 300
  it('keeps the point inside the image', () => {
    const g = meshGrid(w, h)
    expect(clampMeshPoint(g, 0, { x: -20, y: -5 }, w, h)).toEqual({ x: 0, y: 0 })
    expect(clampMeshPoint(g, 15, { x: 999, y: 999 }, w, h)).toEqual({ x: 300, y: 300 })
  })

  it('stays strictly between the left/right and up/down neighbours with a 2% gap', () => {
    const g = meshGrid(w, h)
    // 안쪽 점 5 를 오른쪽 이웃(200) 너머로 끌면 200 − 6 에서 멈춘다
    expect(clampMeshPoint(g, 5, { x: 260, y: 100 }, w, h)).toEqual({ x: 194, y: 100 })
    // 위 이웃(0) 쪽으로 끌면 0 + 6
    expect(clampMeshPoint(g, 5, { x: 100, y: -50 }, w, h)).toEqual({ x: 100, y: 6 })
  })

  it('keeps the current coordinate when the neighbours leave no room', () => {
    const g = meshGrid(w, h)
    g[4] = { x: 190, y: 100 }
    g[6] = { x: 200, y: 100 }
    expect(clampMeshPoint(g, 5, { x: 120, y: 120 }, w, h)).toEqual({ x: 100, y: 120 })
  })
})

describe('normalized payload', () => {
  it('divides by the image size and rounds to 1e-5', () => {
    const g = meshGrid(3000, 3000)
    g[5] = { x: 1234.5678, y: 999.99999 }
    const dst = toNormalized(g, 3000, 3000)
    expect(dst[5]).toEqual([0.41152, 0.33333])
    expect(dst[15]).toEqual([1, 1])
  })

  it('recognises the untouched grid', () => {
    expect(isIdentityMesh(meshGrid(640, 480), 640, 480)).toBe(true)
    const g = meshGrid(640, 480)
    g[10].x += 3
    expect(isIdentityMesh(g, 640, 480)).toBe(false)
    expect(isIdentityNormalized([[0, 0]])).toBe(false)
  })

  it('snaps only the border back to the grid', () => {
    const g = meshGrid(300, 300)
    g[1] = { x: 120, y: 30 }
    g[5] = { x: 130, y: 110 }
    const s = snapEdgesToGrid(g, 300, 300)
    expect(s[1]).toEqual({ x: 100, y: 0 })
    expect(s[5]).toEqual({ x: 130, y: 110 })
  })
})

describe('MeshSession', () => {
  it('records one undo step per drag, not per move', () => {
    const s = new MeshSession()
    s.begin(300, 300, true)
    expect(s.startDrag(5)).toBe(true)
    expect(s.dragTo({ x: 110, y: 100 })).toBe(true)
    expect(s.dragTo({ x: 120, y: 105 })).toBe(true)
    expect(s.endDrag()).toBe(true)
    expect(s.points[5]).toEqual({ x: 120, y: 105 })
    expect(s.canUndo).toBe(true)
    expect(s.undo()).toBe(true)
    expect(s.points[5]).toEqual({ x: 100, y: 100 })
    expect(s.undo()).toBe(false)
  })

  it('does not record a drag that did not move anything', () => {
    const s = new MeshSession()
    s.begin(300, 300, true)
    s.startDrag(5)
    expect(s.dragTo({ x: 100, y: 100 })).toBe(false)
    expect(s.endDrag()).toBe(false)
    expect(s.canUndo).toBe(false)
  })

  it('refuses to drag pinned border points', () => {
    const s = new MeshSession()
    s.begin(300, 300, true)
    expect(s.startDrag(0)).toBe(false)
    s.setPinEdges(false)
    expect(s.startDrag(0)).toBe(true)
  })

  it('reset returns to the grid as one undoable step', () => {
    const s = new MeshSession()
    s.begin(300, 300, true)
    expect(s.reset()).toBe(false)
    s.startDrag(10); s.dragTo({ x: 180, y: 220 }); s.endDrag()
    expect(s.reset()).toBe(true)
    expect(s.isIdentity()).toBe(true)
    expect(s.undo()).toBe(true)
    expect(s.points[10]).toEqual({ x: 180, y: 220 })
  })

  it('pinning the edges snaps moved border points back (undoable)', () => {
    const s = new MeshSession()
    s.begin(300, 300, false)
    s.startDrag(1); s.dragTo({ x: 120, y: 30 }); s.endDrag()
    expect(s.setPinEdges(true)).toBe(true)
    expect(s.points[1]).toEqual({ x: 100, y: 0 })
    expect(s.undo()).toBe(true)
    expect(s.points[1]).toEqual({ x: 120, y: 30 })
  })

  it('keeps at most MESH_HISTORY_MAX steps', () => {
    const s = new MeshSession()
    s.begin(1000, 1000, true)
    for (let i = 0; i < MESH_HISTORY_MAX + 10; i++) {
      s.startDrag(5); s.dragTo({ x: 300 + (i % 2) * 10, y: 333 }); s.endDrag()
    }
    let n = 0
    while (s.undo()) n++
    expect(n).toBe(MESH_HISTORY_MAX)
  })

  it('normalized() is the backend payload', () => {
    const s = new MeshSession()
    s.begin(200, 100, true)
    s.startDrag(6); s.dragTo({ x: 150, y: 40 }); s.endDrag()
    expect(s.normalized()[6]).toEqual([0.75, 0.4])
    s.clear()
    expect(s.active).toBe(false)
  })
})

describe('pin-edges preference', () => {
  it('defaults to on and round-trips', () => {
    const st = memoryStorage()
    expect(loadMeshPinEdges(st)).toBe(true)
    saveMeshPinEdges(false, st)
    expect(st.data[MESH_PIN_STORAGE_KEY]).toBe('0')
    expect(loadMeshPinEdges(st)).toBe(false)
  })

  it('survives a blocked storage', () => {
    const broken = {
      getItem: () => { throw new Error('blocked') },
      setItem: () => { throw new Error('blocked') },
    }
    expect(loadMeshPinEdges(broken)).toBe(true)
    expect(() => saveMeshPinEdges(true, broken)).not.toThrow()
    expect(loadMeshPinEdges(null)).toBe(true)
  })
})
