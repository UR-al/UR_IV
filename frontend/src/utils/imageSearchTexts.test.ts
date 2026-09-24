import { describe, expect, it } from 'vitest'
import { createSearchTextFetcher, exifSearchText } from './imageSearchTexts'

function harness(backend: any, timeoutMs = 50) {
  let handler: ((json: string) => void) | null = null
  let unbound = 0
  const fetcher = createSearchTextFetcher({
    getBackend: async () => backend,
    onBackendEvent: (_name, cb) => { handler = cb; return () => { unbound++ } },
    timeoutMs,
    makeToken: (() => { let n = 0; return () => `t${++n}` })(),
  })
  return { fetcher, fire: (json: string) => handler && handler(json), unbound: () => unbound }
}

describe('createSearchTextFetcher', () => {
  it('sends a token and resolves with the matching reply only', async () => {
    const sent: any[] = []
    const { fetcher, fire } = harness({ requestImageSearchTexts: (json: string) => sent.push(JSON.parse(json)) }, 1000)
    const pending = fetcher.fetch(['a.png', 'b.png'])
    await Promise.resolve(); await Promise.resolve()
    expect(sent).toEqual([{ token: 't1', paths: ['a.png', 'b.png'] }])
    fire(JSON.stringify({ token: 'other', texts: { 'a.png': 'wrong' } }))
    fire(JSON.stringify({ token: 't1', texts: { 'a.png': 'cat', 'b.png': '' } }))
    await expect(pending).resolves.toEqual({ 'a.png': 'cat', 'b.png': '' })
  })

  it('gives up after the timeout instead of hanging the search', async () => {
    const { fetcher } = harness({ requestImageSearchTexts: () => {} }, 10)
    await expect(fetcher.fetch(['a.png'])).resolves.toEqual({})
  })

  it('falls back to getImageExif on an older backend', async () => {
    const { fetcher } = harness({
      getImageExif: (path: string, cb: (json: string) => void) =>
        cb(JSON.stringify({ prompt: `Prompt ${path}`, negative: 'Neg', raw: 'RAW' })),
    })
    await expect(fetcher.fetch(['x.png'])).resolves.toEqual({ 'x.png': 'prompt x.png neg raw' })
  })

  it('dispose releases the subscription and pending waiters', async () => {
    const { fetcher, unbound } = harness({ requestImageSearchTexts: () => {} }, 10_000)
    const pending = fetcher.fetch(['a.png'])
    await Promise.resolve(); await Promise.resolve()
    fetcher.dispose()
    await expect(pending).resolves.toEqual({})
    expect(unbound()).toBe(1)
  })

  it('builds search text from getImageExif json', () => {
    expect(exifSearchText('{"error":"x"}')).toBe('')
    expect(exifSearchText('not json')).toBe('')
    expect(exifSearchText('{"prompt":"A","negative":"B","raw":"C"}')).toBe('a b c')
  })
})
