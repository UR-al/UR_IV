/**
 * 'EXIF 검색' 텍스트 가져오기 — 브리지 requestImageSearchTexts ↔ imageSearchTextsReady.
 *
 * 예전 검색은 이미지마다 동기 슬롯 getImageExif(GUI 스레드, 전체 메타 dict)를 불렀다.
 * 이제 경로 묶음을 보내면 백엔드가 백그라운드 스레드에서 검색 텍스트(prompt·negative·원문)만
 * 모아 token 과 함께 돌려준다. 응답이 오지 않으면 timeout 뒤 빈 결과로 끝낸다(검색이 멈추지 않게).
 * 슬롯이 없는 구 백엔드는 getImageExif 로 폴백한다.
 */
export type SearchTexts = Record<string, string>

export interface SearchTextFetcherDeps {
  getBackend: () => Promise<any>
  onBackendEvent: (name: 'imageSearchTextsReady', cb: (json: string) => void) => unknown
  timeoutMs?: number
  makeToken?: () => string
}

export interface SearchTextFetcher {
  fetch: (paths: readonly string[]) => Promise<SearchTexts>
  dispose: () => void
}

let _tokenSeq = 0
const defaultToken = () => `search-${Date.now().toString(36)}-${(++_tokenSeq).toString(36)}`

/** getImageExif 결과 → 검색 텍스트(백엔드 core.metadata_search 와 같은 구성). */
export function exifSearchText(json: string): string {
  try {
    const d = JSON.parse(json)
    if (!d || d.error) return ''
    return `${d.prompt || ''} ${d.negative || ''} ${d.raw || ''}`.toLowerCase()
  } catch { return '' }
}

export function createSearchTextFetcher(deps: SearchTextFetcherDeps): SearchTextFetcher {
  const timeoutMs = deps.timeoutMs ?? 30_000
  const makeToken = deps.makeToken ?? defaultToken
  const pending = new Map<string, (texts: SearchTexts) => void>()
  let off: (() => void) | null = null

  function onReady(json: string) {
    try {
      const d = JSON.parse(json)
      const resolve = d && typeof d.token === 'string' ? pending.get(d.token) : undefined
      if (!resolve) return
      pending.delete(d.token)
      resolve(d.texts && typeof d.texts === 'object' ? d.texts : {})
    } catch { /* 모양이 틀린 응답은 무시 — timeout 이 정리한다 */ }
  }

  function ensureBound() {
    if (off) return
    const r = deps.onBackendEvent('imageSearchTextsReady', onReady)
    off = typeof r === 'function' ? (r as () => void) : () => {}
  }

  async function fetch(paths: readonly string[]): Promise<SearchTexts> {
    if (!paths.length) return {}
    let backend: any = null
    try { backend = await deps.getBackend() } catch { backend = null }
    if (backend && typeof backend.requestImageSearchTexts === 'function') {
      ensureBound()
      const token = makeToken()
      return new Promise<SearchTexts>(resolve => {
        const timer = setTimeout(() => { pending.delete(token); resolve({}) }, timeoutMs)
        pending.set(token, texts => { clearTimeout(timer); resolve(texts) })
        try {
          backend.requestImageSearchTexts(JSON.stringify({ token, paths }))
        } catch {
          clearTimeout(timer)
          pending.delete(token)
          resolve({})
        }
      })
    }
    if (backend && typeof backend.getImageExif === 'function') {
      const out: SearchTexts = {}
      await Promise.all(paths.map(path => new Promise<void>(resolve => {
        try {
          backend.getImageExif(path, (json: string) => { out[path] = exifSearchText(json); resolve() })
        } catch { resolve() }
      })))
      return out
    }
    return {}
  }

  function dispose() {
    if (off) { off(); off = null }
    for (const resolve of pending.values()) resolve({})
    pending.clear()
  }

  return { fetch, dispose }
}
