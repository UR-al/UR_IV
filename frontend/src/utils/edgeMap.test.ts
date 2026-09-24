import { describe, expect, it, vi } from 'vitest'
import { EdgeMapCache, edgeMapFromRgba, snapToEdge, type EdgeMap } from './edgeMap'

function edgeWithPixels(w: number, h: number, on: Array<[number, number]>): EdgeMap {
  const data = new Uint8Array(w * h)
  for (const [x, y] of on) data[y * w + x] = 255
  return { data, w, h }
}

describe('edgeMapFromRgba', () => {
  it('keeps the red channel of the decoded PNG', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 128, 1, 2, 255])
    expect(edgeMapFromRgba(rgba, 3, 1)).toEqual({ data: new Uint8Array([0, 255, 128]), w: 3, h: 1 })
  })
})

describe('snapToEdge', () => {
  it('snaps to the nearest edge pixel inside the radius', () => {
    const edge = edgeWithPixels(40, 40, [[20, 20], [25, 20]])
    expect(snapToEdge(edge, 22, 21, 12)).toEqual({ x: 20, y: 20 })
    expect(snapToEdge(edge, 24.4, 20, 12)).toEqual({ x: 25, y: 20 })
  })

  it('leaves the point alone without an edge map or a nearby edge', () => {
    expect(snapToEdge(null, 3.5, 4.5)).toEqual({ x: 3.5, y: 4.5 })
    const edge = edgeWithPixels(40, 40, [[39, 39]])
    expect(snapToEdge(edge, 2, 2, 5)).toEqual({ x: 2, y: 2 })
  })

  it('ignores weak (<=127) pixels and clips at the borders', () => {
    const edge = edgeWithPixels(10, 10, [])
    edge.data[0] = 127
    expect(snapToEdge(edge, 1, 1, 3)).toEqual({ x: 1, y: 1 })
    edge.data[0] = 200
    expect(snapToEdge(edge, 1, 1, 3)).toEqual({ x: 0, y: 0 })
  })
})

describe('EdgeMapCache', () => {
  it('fetches once per image path and serves repeats from the cache', async () => {
    const fetcher = vi.fn(async (path: string) => `edges:${path}`)
    const cache = new EdgeMapCache(fetcher)
    const [a, b] = await Promise.all([cache.get('a.png'), cache.get('a.png')])
    expect(a).toBe('edges:a.png')
    expect(b).toBe('edges:a.png')
    expect(await cache.get('a.png')).toBe('edges:a.png')
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(cache.peek('a.png')).toBe('edges:a.png')
    expect(cache.peek('b.png')).toBe('')
  })

  it('keeps a single entry and refetches after clear()', async () => {
    const fetcher = vi.fn(async (path: string) => `edges:${path}`)
    const cache = new EdgeMapCache(fetcher)
    await cache.get('a.png')
    await cache.get('b.png')
    expect(cache.peek('a.png')).toBe('')
    cache.clear()
    await cache.get('b.png')
    expect(fetcher).toHaveBeenCalledTimes(3)
  })

  it('does not cache a late answer that arrives after clear() (image changed meanwhile)', async () => {
    let release: (v: string) => void = () => {}
    const fetcher = vi.fn(() => new Promise<string>((resolve) => { release = resolve }))
    const cache = new EdgeMapCache(fetcher)
    const pending = cache.get('old.png')
    cache.clear()
    release('edges:old')
    expect(await pending).toBe('edges:old')
    expect(cache.peek('old.png')).toBe('')
  })

  it('treats empty or failed answers as a miss', async () => {
    const cache = new EdgeMapCache(vi.fn(async () => ''))
    expect(await cache.get('x.png')).toBe('')
    expect(cache.peek('x.png')).toBe('')
    const failing = new EdgeMapCache(vi.fn(async () => { throw new Error('boom') }))
    expect(await failing.get('x.png')).toBe('')
    expect(await failing.get('')).toBe('')
  })
})
