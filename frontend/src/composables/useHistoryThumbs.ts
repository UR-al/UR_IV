import { reactive, ref } from 'vue'

/**
 * 히스토리 스트립 썸네일 (App.vue 분할 ④).
 *
 * 예전엔 히스토리 카드가 원본 PNG(2048px)를 file:/// 로 직접 읽어 페이지를 위아래로
 * 넘길 때마다 디코딩 지연이 있었다. 이제 백엔드 썸네일 캐시(generateThumbnails →
 * thumbnailReady, image_cache/thumbs_v2/<샤드>/<sha1(path@width)>.jpg)를 목록 전체에 대해 미리
 * 만들어 두고, 카드는 준비된 썸네일을 쓴다(없으면 원본으로 폴백). 폭은 설정
 * '미리보기 품질'(ui_prefs.previewThumbWidth)로 정한다.
 *
 * thumbnailReady 페이로드는 청크 단위 ``{width, items: [{path, thumb, v}]}`` 다(core/thumb_prefetch.py).
 * 폭이 실려 오므로 다른 폭으로 요청한 응답은 버린다. 옛 단건 ``{path, thumb}`` 도 받는다.
 * ``v``(썸네일 파일 버전)는 URL 에 붙인다 — 원본을 덮어써 다시 만든 썸네일도 캐시 경로가 같다.
 */

export const PREVIEW_THUMB_PRESETS = [
  { value: 192, label: '낮음 · 192px (가장 빠름)' },
  { value: 256, label: '보통 · 256px' },
  { value: 384, label: '높음 · 384px' },
  { value: 512, label: '최고 · 512px' },
] as const

export const DEFAULT_PREVIEW_THUMB_WIDTH = 256

export function normalizePreviewThumbWidth(value: unknown): number {
  const n = Number(value)
  return PREVIEW_THUMB_PRESETS.some(p => p.value === n) ? n : DEFAULT_PREVIEW_THUMB_WIDTH
}

export interface HistoryThumbDeps {
  getBackend: () => Promise<any>
  onBackendEvent: (name: 'thumbnailReady', cb: (json: string) => void) => any
  mediaUrl: (path: string) => string
  /**
   * generateThumbnails 한 번에 보내는 경로 수. 백엔드는 청크를 작업자 2개짜리 풀에서
   * 처리하고(청크 사이가 병렬일 수 있다), 캐시 적중은 청크마다 한 번에 알린다.
   */
  chunk?: number
}

/** v = 썸네일 파일 버전(렌더 시각). 캐시 파일 경로는 다시 만들어도 같으므로 URL 에 붙여 새로 읽게 한다. */
interface ThumbReadyItem { path: string; thumb: string; v: string }

/** thumbnailReady 페이로드 → (폭, 항목들). 폭이 없는 옛 단건 형식은 width=null. */
export function parseThumbReady(json: string): { width: number | null; items: ThumbReadyItem[] } | null {
  let d: any
  try { d = JSON.parse(json) } catch { return null }
  if (!d || typeof d !== 'object') return null
  const width = typeof d.width === 'number' && Number.isFinite(d.width) ? d.width : null
  const raw: any[] = Array.isArray(d.items) ? d.items : (typeof d.path === 'string' ? [d] : [])
  const items: ThumbReadyItem[] = []
  for (const it of raw) {
    if (!it || typeof it.path !== 'string' || !it.path) continue
    const v = typeof it.v === 'string' || typeof it.v === 'number' ? String(it.v) : ''
    items.push({ path: it.path, thumb: typeof it.thumb === 'string' ? it.thumb : '', v })
  }
  return { width, items }
}

/** 썸네일 URL 에 파일 버전을 붙인다(버전이 없으면 그대로) — 다시 만든 썸네일을 캐시가 옛 그림으로 주지 않게. */
export function versionedThumbUrl(url: string, version: string): string {
  if (!url || !version) return url
  return url + (url.includes('?') ? '&' : '?') + 'v=' + encodeURIComponent(version)
}

export function useHistoryThumbs(deps: HistoryThumbDeps) {
  const chunkSize = deps.chunk && deps.chunk > 0 ? deps.chunk : 40
  const width = ref(DEFAULT_PREVIEW_THUMB_WIDTH)
  /** path → 표시용 URL ('' = 생성 실패 → 원본 폴백) */
  const thumbs = reactive<Record<string, string>>({})
  /** 요청 보낸 경로 → 요청 당시 폭 (폭이 바뀐 뒤 도착한 응답은 버린다) */
  const requested = new Map<string, number>()
  /** 원본이 바뀌어 무효화한 경로 / 무효화 뒤 새 썸네일을 받은 경로 */
  const invalidated = new Set<string>()
  const refreshed = new Set<string>()
  let off: null | (() => void) = null

  function fullUrl(path: string, version?: number) {
    const base = deps.mediaUrl(path)
    if (!version) return base
    return base + (base.includes('?') ? '&' : '?') + 't=' + version
  }

  /**
   * 카드 src — 썸네일 우선, 없으면 원본.
   * 같은 경로에 덮어써 버전이 오른 이미지는 새 썸네일이 올 때까지 원본(버전 쿼리로 캐시 무효화)을
   * 쓰고, 새 썸네일이 오면 그것도 버전 쿼리를 붙인다 — 썸네일 파일 경로(sha1(path@width))는
   * 그대로라 쿼리가 없으면 WebEngine 메모리 캐시가 옛 그림을 보여 준다.
   */
  function srcFor(path: string, version?: number): string {
    if (!path) return ''
    const thumb = thumbs[path]
    if (version) {
      // 무효화 뒤에 다시 받은 썸네일만 믿는다 — 그 전 것은 덮어쓰기 전 그림이다
      if (!thumb || !refreshed.has(path)) return fullUrl(path, version)
      return thumb + (thumb.includes('?') ? '&' : '?') + 't=' + version
    }
    return thumb || fullUrl(path)
  }

  /** 원본이 바뀐 경로의 썸네일을 버린다 — 다음 ensure 가 다시 요청한다(백엔드는 mtime 으로 재생성). */
  function invalidate(path: string) {
    if (!path) return
    invalidated.add(path)
    refreshed.delete(path)
    delete thumbs[path]
    requested.delete(path)
  }

  /** 목록의 아직 요청 안 한 경로를 청크로 요청 — 페이지 넘김이 즉시 되게 전체를 미리 만든다 */
  async function ensure(list: readonly string[]): Promise<number> {
    const w = width.value
    const need = list.filter(p => p && requested.get(p) !== w)
    if (!need.length) return 0
    need.forEach(p => requested.set(p, w))
    let backend: any = null
    try { backend = await deps.getBackend() } catch { backend = null }
    if (!backend || typeof backend.generateThumbnails !== 'function') {
      need.forEach(p => requested.delete(p))
      return 0
    }
    for (let i = 0; i < need.length; i += chunkSize) {
      backend.generateThumbnails(JSON.stringify(need.slice(i, i + chunkSize)), w)
    }
    return need.length
  }

  /** 설정 변경 — 캐시를 비우고 현재 목록을 새 폭으로 다시 요청 */
  function setWidth(next: unknown, list: readonly string[] = []) {
    const w = normalizePreviewThumbWidth(next)
    if (w === width.value) return false
    width.value = w
    for (const key of Object.keys(thumbs)) delete thumbs[key]
    requested.clear()
    refreshed.clear()   // 새 폭 썸네일은 지금 원본에서 만들어진다 — 도착하면 다시 표시된다
    void ensure(list)
    return true
  }

  function onReady(json: string) {
    const parsed = parseThumbReady(json)
    if (!parsed) return
    // 폭이 실려 오면 지금 폭과 다른 응답(폭 변경 전 요청·다른 화면의 요청)은 통째로 버린다
    if (parsed.width !== null && parsed.width !== width.value) return
    for (const { path, thumb, v } of parsed.items) {
      const asked = requested.get(path)
      if (asked === undefined) continue          // 이 스트립이 요청하지 않은 썸네일(옛 요청·다른 호출자)
      if (asked !== width.value) continue        // 폭이 바뀐 뒤 도착한 옛 응답
      if (invalidated.has(path)) refreshed.add(path)
      // 다시 만든 썸네일은 파일 경로가 같다 — 버전(v)을 붙여야 옛 그림 대신 새로 읽는다
      thumbs[path] = thumb ? versionedThumbUrl(deps.mediaUrl(thumb), v) : ''
    }
  }

  function bind() {
    if (off) return
    const r = deps.onBackendEvent('thumbnailReady', onReady)
    off = typeof r === 'function' ? r : () => {}
  }
  function unbind() { if (off) { off(); off = null } }

  return { thumbs, width, srcFor, ensure, invalidate, setWidth, bind, unbind, onReady }
}
