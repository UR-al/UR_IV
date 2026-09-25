import { ref } from 'vue'

/**
 * 우하단 빠른 도크(components/dock/QuickDock.vue) — 어느 패널이 열려 있나.
 *
 * 예전 '대기열' 핀을 누르면 이제 대기열 · 대화 · 메모장 세 아이콘이 펼쳐지고(expanded), 그중 하나를
 * 누르면 그 패널이 오른쪽에 열린다(activePanel). 패널은 한 번에 하나 — 대기열 드로어를 열면 대화 ·
 * 메모장 패널은 닫힌다. 모듈 싱글턴이라 도크 밖(단축키 · 미리보기 도구)에서도 같은 상태를 연다.
 */
export type DockPanel = 'queue' | 'chat' | 'memo'

export function createQuickDock() {
  /** 세 아이콘이 펼쳐져 있나 */
  const expanded = ref(false)
  /** 열려 있는 패널(없으면 null) */
  const activePanel = ref<DockPanel | null>(null)

  function toggleExpanded() { expanded.value = !expanded.value }
  function collapse() { expanded.value = false }
  /** 패널을 연다 — 다른 패널은 닫히고 아이콘 줄은 접힌다. */
  function openPanel(panel: DockPanel) { activePanel.value = panel; expanded.value = false }
  /** 아이콘을 다시 누르면 닫는다. */
  function togglePanel(panel: DockPanel) {
    if (activePanel.value === panel) activePanel.value = null
    else activePanel.value = panel
    expanded.value = false
  }
  /** 그 패널이 열려 있을 때만 닫는다(다른 패널이 열린 사이 늦게 온 닫기가 엉뚱한 것을 닫지 않게). */
  function closePanel(panel?: DockPanel) {
    if (!panel || activePanel.value === panel) activePanel.value = null
  }
  function isOpen(panel: DockPanel) { return activePanel.value === panel }

  return { expanded, activePanel, toggleExpanded, collapse, openPanel, togglePanel, closePanel, isOpen }
}

export type QuickDock = ReturnType<typeof createQuickDock>

let _app: QuickDock | null = null
export function useQuickDock(): QuickDock {
  if (!_app) _app = createQuickDock()
  return _app
}
