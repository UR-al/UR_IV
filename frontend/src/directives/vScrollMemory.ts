import { nextTick, type Directive } from 'vue'

/**
 * `v-scroll-memory="'key'"` — 스크롤 위치 기억(App.vue 에서 추출).
 * 탭 전환/버튼으로 패널이 언마운트→재마운트돼도 위치를 되살린다. 위치는 모듈 스코프 표에
 * key 별로 남는다(세션 동안 유지 — 새로고침하면 처음부터).
 */
const scrollMemory: Record<string, number> = {}

interface ScrollMemoryEl extends HTMLElement {
  __sk?: string
  __onScroll?: () => void
}

/** 콘텐츠가 늦게 채워지는 패널을 위해 한 번 더 복원하는 지연(ms) */
export const SCROLL_RESTORE_RETRY_MS = 80

export const vScrollMemory: Directive<ScrollMemoryEl, string> = {
  mounted(el, binding) {
    const key = binding.value
    el.__sk = key
    el.__onScroll = () => { scrollMemory[key] = el.scrollTop }
    el.addEventListener('scroll', el.__onScroll, { passive: true })
    const restore = () => { if (key in scrollMemory) el.scrollTop = scrollMemory[key] }
    nextTick(restore)
    setTimeout(restore, SCROLL_RESTORE_RETRY_MS)   // 콘텐츠가 늦게 채워지는 경우 한 번 더
  },
  updated(el) {
    // 같은 탭에서 콘텐츠가 갱신돼 scrollTop이 0으로 튀면 복원
    const key = el.__sk
    if (key && (key in scrollMemory) && el.scrollTop === 0 && scrollMemory[key] > 0) {
      nextTick(() => { el.scrollTop = scrollMemory[key] })
    }
  },
  beforeUnmount(el) {
    if (el.__sk) scrollMemory[el.__sk] = el.scrollTop
    if (el.__onScroll) el.removeEventListener('scroll', el.__onScroll)
  },
}
