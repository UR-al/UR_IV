import { computed, ref } from 'vue'
import type { SearchTexts } from '../utils/imageSearchTexts'

/**
 * Gallery·Favorites 'EXIF 검색' (App 분할 ④ — 두 뷰에 복사돼 갈라져 있던 루프를 하나로).
 *
 * 예전 복제본 차이: 비이미지 미디어 필터는 Gallery 에만, 검색어를 지우면 멈추는 로직은
 * Favorites 에만 있었고, Gallery 는 Enter 연타에 재진입 가드가 없었다. 'EXIF 저장' 뒤
 * 캐시를 무효화하는 곳도 없어 고치기 전 텍스트로 매칭됐다. 여기서 모두 한 번에 다룬다.
 *
 * - 비이미지(영상·오디오)는 파일 이름·미디어 종류로만 검색한다(Pillow 로 보내지 않음).
 * - 텍스트는 경로 묶음(chunk)으로 가져오고 세션 동안 캐시한다. 결과는 현재 목록과의 교집합이라
 *   삭제·폴더 새로고침이 곧바로 반영된다.
 * - 검색어를 지우거나 clear()/cancel() 하면 진행 중인 검색은 결과를 적용하지 않는다.
 * - 검색 중의 이름 변경·EXIF 저장·삭제는 기록해 두고, 아직 안 보낸 묶음은 새 이름으로 묻고 이미 보낸
 *   묶음의 응답은 새 이름으로 캐시한다(내용이 바뀐 것은 버리고 다시 묻는다). 옛 이름의 응답이 비어
 *   있으면 이미 옮겨 간 파일을 못 읽은 것일 수 있어 캐시하지 않고 새 이름으로 다시 묻는다.
 */
export interface ExifSearchDeps {
  /** 지금 목록(원본 순서) — computed 안에서 읽혀도 되는 반응형 getter */
  source: () => readonly string[]
  isImage: (path: string) => boolean
  /** 비이미지 검색 텍스트(파일 이름 + 미디어 종류) */
  labelFor: (path: string) => string
  fetchTexts: (paths: readonly string[]) => Promise<SearchTexts>
  /** 결과 적용·해제 뒤(보이는 개수 초기화 등) */
  onChange?: () => void
  chunkSize?: number
}

export function useExifSearch(deps: ExifSearchDeps) {
  const chunkSize = deps.chunkSize && deps.chunkSize > 0 ? deps.chunkSize : 50
  const query = ref('')
  const searching = ref(false)
  const filtered = ref(false)
  const matched = ref<ReadonlySet<string>>(new Set())
  const cache = new Map<string, string>()
  let generation = 0
  /** 지금 적용된 필터의 검색어(소문자) — 이름 변경 때 새 이름이 여전히 맞는지 다시 본다 */
  let appliedNeedle = ''
  /**
   * 검색이 도는 동안의 이름 변경(to = 새 경로)·캐시 무효화(to = null) 기록 — 백엔드에 묻는 사이 바뀐
   * 경로를 맞춘다. 예전엔 응답을 옛 이름으로 캐시해, 검색 중 이름을 바꾼 이미지가 메타데이터가 맞아도
   * 그 검색 결과에서 빠지고 옛 이름의 캐시만 남았다. 검색이 끝나거나 취소되면 비운다.
   */
  let touches: { from: string; to: string | null }[] = []
  /** 이번 검색이 아직 묻지 않은 이미지 — since: 그 이름이 유효해진 touches 위치 */
  let queue: { path: string; since: number }[] = []

  /**
   * touches[from..] 을 따라가 경로의 지금 이름을 찾는다. dropInvalidated 면 무효화(내용 변경·삭제)를
   * 만났을 때 null — 그 전에 읽은 텍스트는 낡았을 수 있다.
   */
  function currentName(path: string, from: number, dropInvalidated: boolean): string | null {
    let cur = path
    for (let i = from; i < touches.length; i++) {
      const t = touches[i]
      if (t.from !== cur) continue
      if (t.to !== null) cur = t.to
      else if (dropInvalidated) return null
    }
    return cur
  }

  /** 비이미지(영상·오디오)의 검색 텍스트 = 파일 이름 + 종류. 캐시에 없으면 만든다. */
  function labelMedia(paths: Iterable<string>) {
    for (const path of paths) {
      if (!deps.isImage(path) && !cache.has(path)) cache.set(path, deps.labelFor(path).toLowerCase())
    }
  }

  /** 필터가 켜져 있으면 현재 목록 중 매칭된 경로(원본 순서), 아니면 [] */
  const results = computed(() => {
    if (!filtered.value) return [] as string[]
    const hits = matched.value
    return deps.source().filter(path => hits.has(path))
  })

  function cancel() {
    generation++
    searching.value = false
    touches = []
    queue = []
  }

  function clear() {
    cancel()
    query.value = ''
    filtered.value = false
    matched.value = new Set()
    appliedNeedle = ''
    deps.onChange?.()
  }

  async function run(): Promise<boolean> {
    const needle = query.value.trim().toLowerCase()
    if (!needle) { clear(); return false }
    if (searching.value) return false
    const mine = ++generation
    searching.value = true
    touches = []
    try {
      const list = [...deps.source()]
      labelMedia(list)
      queue = list.filter(path => deps.isImage(path) && !cache.has(path)).map(path => ({ path, since: 0 }))
      while (queue.length) {
        const mark = touches.length
        // 검색 중에 이름이 바뀐 경로는 새 이름으로 묻는다(옛 이름의 파일은 이제 없다)
        const names = queue.splice(0, chunkSize).map(q => currentName(q.path, q.since, false))
        const batch = [...new Set(names)].filter((path): path is string =>
          path !== null && deps.isImage(path) && !cache.has(path))
        if (!batch.length) continue
        let texts: SearchTexts = {}
        try { texts = await deps.fetchTexts(batch) } catch { texts = {} }
        if (mine !== generation) return false
        // 응답에 없는 경로(타임아웃 등)는 캐시하지 않는다 — 다음 검색에서 다시 묻는다.
        // 기다리는 사이 이름이 바뀌었으면 새 이름으로 캐시하고, 내용이 바뀌었거나 지워졌으면 버린다.
        for (const path of batch) {
          if (!Object.prototype.hasOwnProperty.call(texts, path)) continue
          const now = currentName(path, mark, true)
          if (now === null || !deps.isImage(now)) continue
          const text = String(texts[path] || '')
          // 옛 이름의 빈 응답은 '메타데이터 없음'인지 '백엔드가 읽기 전에 파일이 옮겨 가 없음'인지 모른다
          // (없는 경로도 '' — core/metadata_search.search_texts_for). 그대로 새 이름에 캐시하면 그
          // 이미지는 세션 내내 EXIF 검색에서 빠진다 — 이번 검색에서 새 이름으로 한 번 더 묻는다.
          if (now !== path && !text) { queue.push({ path: now, since: touches.length }); continue }
          cache.set(now, text.toLowerCase())
        }
        if (!query.value.trim()) { clear(); return false }
      }
      if (mine !== generation) return false
      const current = query.value.trim().toLowerCase() || needle
      // 검색 중에 목록이 바뀌었을 수 있다(이름 변경 등) — 지금 목록으로 맞춘다. 캐시에 없는 새 이미지는
      // 다음 검색에서 묻는다(매칭 안 됨), 새 영상·오디오는 이름으로 바로 맞춘다.
      const finalList = [...deps.source()]
      labelMedia(finalList)
      matched.value = new Set(finalList.filter(path => (cache.get(path) || '').includes(current)))
      appliedNeedle = current
      filtered.value = true
      deps.onChange?.()
      return true
    } finally {
      if (mine === generation) {
        searching.value = false
        touches = []
        queue = []
      }
    }
  }

  /**
   * 'EXIF 저장' 등으로 내용이 바뀐 이미지 — 캐시를 버리고, 필터 중이면 다시 맞춰 본다.
   * 검색 중이면 이미 보낸 요청의 응답(저장 전 내용일 수 있다)을 버리고 이번 검색에서 다시 묻는다.
   */
  function invalidate(path: string) {
    cache.delete(path)
    if (searching.value) {
      touches.push({ from: path, to: null })
      queue.push({ path, since: touches.length })
      return
    }
    if (filtered.value && query.value.trim()) void run()
  }

  /**
   * 이름 변경 — 캐시·매칭 집합의 키를 새 경로로 옮긴다.
   * 이미지의 검색 텍스트는 메타데이터라 이름과 무관해 그대로 옮긴다. 영상·오디오는 텍스트가 파일
   * 이름이라 새 이름으로 다시 만들고, 필터 중이면 새 이름이 검색어에 맞는지 다시 본다 — 예전엔 옛
   * 이름을 그대로 옮겨 새 이름으로는 안 찾히고 옛 이름으로 찾혔다.
   * 검색 중이면 기록해 둔다 — 아직 캐시에 없는 이미지의 텍스트는 응답이 온 뒤 새 이름으로 들어간다.
   */
  function rename(oldPath: string, newPath: string) {
    if (searching.value && oldPath !== newPath) touches.push({ from: oldPath, to: newPath })
    const prev = cache.get(oldPath)
    cache.delete(oldPath)
    const wasMatched = matched.value.has(oldPath)
    let hit: boolean
    if (deps.isImage(newPath)) {
      const sameKind = deps.isImage(oldPath)
      if (prev !== undefined && sameKind) cache.set(newPath, prev)
      else cache.delete(newPath)   // 영상 → 이미지 등 — 다음 검색에서 메타데이터를 묻는다
      hit = sameKind && wasMatched
    } else {
      const label = deps.labelFor(newPath).toLowerCase()
      cache.set(newPath, label)
      hit = !!appliedNeedle && label.includes(appliedNeedle)
    }
    if (!filtered.value) return
    if (wasMatched === hit && oldPath === newPath) return
    const next = new Set(matched.value)
    next.delete(oldPath)
    if (hit) next.add(newPath)
    else next.delete(newPath)
    matched.value = next
  }

  /** 목록에서 빠진(삭제·즐겨찾기 해제) 경로 — 검색 중 도착하는 그 경로의 응답도 캐시하지 않는다. */
  function forget(path: string) {
    cache.delete(path)
    if (!searching.value) return
    queue = queue.filter(q => currentName(q.path, q.since, false) !== path)   // 아직 안 물었으면 묻지 않는다
    touches.push({ from: path, to: null })
  }

  return { query, searching, filtered, results, run, clear, cancel, invalidate, rename, forget }
}
