import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { clampMenuPosition, menuPositionStyle, type MenuSize } from '../utils/ctxMenuPosition'

/**
 * Gallery·Favorites 카드 우클릭 메뉴 — 열기/닫기 + 화면 밖 보정.
 *
 * 처음엔 예상 크기로 당기고, 그려진 뒤 실제 크기를 재서 다시 당긴다(항목 수가 이미지/영상에
 * 따라 달라진다). 문서 아무 곳이나 클릭하면 닫힌다.
 */
export function useContextMenu(estimate: MenuSize) {
  const menu = ref({ show: false, x: 0, y: 0, path: '' })
  const menuEl = ref<HTMLElement | null>(null)
  const measured = ref<MenuSize | null>(null)

  const style = computed(() => {
    const viewport = typeof window === 'undefined'
      ? { width: Number.POSITIVE_INFINITY, height: Number.POSITIVE_INFINITY }
      : { width: window.innerWidth, height: window.innerHeight }
    return menuPositionStyle(clampMenuPosition({ x: menu.value.x, y: menu.value.y }, measured.value || estimate, viewport))
  })

  function open(e: MouseEvent, path: string) {
    measured.value = null
    menu.value = { show: true, x: e.clientX, y: e.clientY, path }
    void nextTick(() => {
      const rect = menuEl.value?.getBoundingClientRect()
      if (rect && rect.width > 0 && rect.height > 0) measured.value = { width: rect.width, height: rect.height }
    })
  }

  function hide() { menu.value.show = false }

  onMounted(() => document.addEventListener('click', hide))
  onUnmounted(() => document.removeEventListener('click', hide))

  return { menu, menuEl, style, open, hide }
}
