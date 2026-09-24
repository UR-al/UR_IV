import { onMounted, onUnmounted, ref, watch, type Ref, type WatchSource } from 'vue'

/**
 * Gallery·Favorites 카드 그리드 페이징 (두 뷰에 복사돼 있던 fillViewport·스크롤 더 보기).
 *
 * '더 보기'는 스크롤 이벤트로만 발동한다. 첫 페이지가 화면에 다 들어가면(넓은 모니터 + 작은
 * 썸네일) 스크롤이 생기지 않아 "N / M — 스크롤하여 더 보기"만 떠 있는 채 멈췄다. 목록·크기가
 * 바뀔 때마다 컨테이너가 넘칠 때까지 더 보인다.
 */
export interface FillState {
  total: number
  visible: number
  clientWidth: number
  clientHeight: number
  scrollHeight: number
  /** 카드 한 칸의 대략적인 크기(px) — 썸네일 폭 */
  cell: number
  page?: number
}

/** 화면을 채우는 데 필요한 보이는 개수(늘어나기만 한다). */
export function fillTarget(s: FillState): number {
  const page = s.page ?? 30
  if (s.visible >= s.total) return s.visible
  // keep-alive 로 떼어졌거나 아직 레이아웃이 없으면 높이가 0 — 그때 늘리면 전부 펼쳐진다
  if (s.clientHeight <= 0) return s.visible
  // 1) 기하 추정 — 썸네일이 뜨기 전엔 카드 높이가 0 이라 scrollHeight 로는 알 수 없다.
  const cell = Math.max(60, s.cell)
  const need = Math.min(s.total, Math.max(1, Math.floor(s.clientWidth / cell)) * (Math.ceil(s.clientHeight / cell) + 1))
  if (need > s.visible) return need
  // 2) 실측 — 다 떴는데도 안 넘치면(가로로 긴 그림들) 한 페이지 더
  if (s.scrollHeight <= s.clientHeight + 1) return Math.min(s.total, s.visible + page)
  return s.visible
}

export interface ScrollState {
  total: number
  visible: number
  scrollHeight: number
  scrollTop: number
  clientHeight: number
  threshold?: number
  page?: number
}

/** 바닥 근처까지 스크롤했으면 한 페이지 더. */
export function scrollTarget(s: ScrollState): number {
  const threshold = s.threshold ?? 200
  const page = s.page ?? 30
  if (s.visible >= s.total) return s.visible
  if (s.scrollHeight - s.scrollTop - s.clientHeight < threshold) return Math.min(s.visible + page, s.total)
  return s.visible
}

/** 목록이 새로 왔을 때 보이는 개수 — 스크롤 위치를 지키되 최소 한 페이지. */
export function keepVisibleCount(visible: number, total: number, initial = 40): number {
  return Math.min(Math.max(initial, visible), Math.max(initial, total))
}

export interface GridPagingOptions {
  container: Ref<HTMLElement | null>
  total: () => number
  cell: () => number
  /** 바뀌면 다시 채워 볼 값들(목록·필터·썸네일 크기) */
  sources: WatchSource[]
  initial?: number
  page?: number
}

export function useGridPaging(opts: GridPagingOptions) {
  const initial = opts.initial ?? 40
  const page = opts.page ?? 30
  const visibleCount = ref(initial)

  function fillViewport() {
    const el = opts.container.value
    if (!el || !el.isConnected) return
    visibleCount.value = fillTarget({
      total: opts.total(), visible: visibleCount.value, clientWidth: el.clientWidth,
      clientHeight: el.clientHeight, scrollHeight: el.scrollHeight, cell: opts.cell(), page,
    })
  }

  function onScroll(e: Event) {
    const el = e.target as HTMLElement
    visibleCount.value = scrollTarget({
      total: opts.total(), visible: visibleCount.value, scrollHeight: el.scrollHeight,
      scrollTop: el.scrollTop, clientHeight: el.clientHeight, page,
    })
  }

  function reset() { visibleCount.value = initial }
  function keep(total: number) { visibleCount.value = keepVisibleCount(visibleCount.value, total, initial) }

  let observer: ResizeObserver | null = null
  onMounted(() => {
    const el = opts.container.value
    if (typeof ResizeObserver === 'undefined' || !el) return
    // 콜백 안에서 바로 늘리면 같은 프레임에 크기가 또 바뀌어 'ResizeObserver loop' 경고 — 다음 프레임에
    observer = new ResizeObserver(() => { requestAnimationFrame(fillViewport) })
    observer.observe(el)                                              // 창 크기
    if (el.firstElementChild) observer.observe(el.firstElementChild)  // 그리드 — 썸네일이 뜨며 자란다
  })
  onUnmounted(() => { observer?.disconnect(); observer = null })
  watch(opts.sources, () => { fillViewport() }, { flush: 'post' })

  return { visibleCount, fillViewport, onScroll, reset, keep }
}
