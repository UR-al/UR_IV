<template>
  <Teleport to="body">
    <Transition name="app-tooltip-fade">
      <div
        v-if="visible"
        ref="tooltipElement"
        class="app-tooltip"
        role="tooltip"
        :style="{ left: `${position.left}px`, top: `${position.top}px` }"
      >{{ text }}</div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { nextTick, onMounted, onUnmounted, reactive, ref } from 'vue'

const visible = ref(false)
const text = ref('')
const tooltipElement = ref<HTMLElement | null>(null)
const position = reactive({ left: 12, top: 12 })

let activeTarget: HTMLElement | null = null
let observer: MutationObserver | null = null

function migrateTitle(element: HTMLElement): void {
  const title = element.getAttribute('title')
  if (!title?.trim()) return

  element.dataset.appTooltip = title
  element.removeAttribute('title')

  const needsAccessibleName = element.matches('button, input, select, textarea, [role="button"]')
  if (needsAccessibleName && !element.hasAttribute('aria-label')) {
    element.setAttribute('aria-label', title.replace(/\s+/g, ' ').trim())
  }
}

function migrateTree(node: Node): void {
  if (!(node instanceof HTMLElement)) return
  migrateTitle(node)
  node.querySelectorAll<HTMLElement>('[title]').forEach(migrateTitle)
}

function findTooltipTarget(node: EventTarget | null): HTMLElement | null {
  if (!(node instanceof Element)) return null
  const target = node.closest<HTMLElement>('[data-app-tooltip], [title]')
  if (!target || target.closest('.app-tooltip')) return null
  migrateTitle(target)
  return target.dataset.appTooltip?.trim() ? target : null
}

function placeTooltip(): void {
  if (!activeTarget || !tooltipElement.value) return

  const gap = 10
  const viewportPadding = 12
  const targetRect = activeTarget.getBoundingClientRect()
  const tooltipRect = tooltipElement.value.getBoundingClientRect()

  const centeredLeft = targetRect.left + (targetRect.width - tooltipRect.width) / 2
  position.left = Math.min(
    window.innerWidth - tooltipRect.width - viewportPadding,
    Math.max(viewportPadding, centeredLeft),
  )

  const below = targetRect.bottom + gap
  const above = targetRect.top - tooltipRect.height - gap
  position.top = below + tooltipRect.height <= window.innerHeight - viewportPadding
    ? below
    : Math.max(viewportPadding, above)
}

async function showTooltip(target: HTMLElement): Promise<void> {
  const tooltipText = target.dataset.appTooltip?.trim()
  if (!tooltipText) return

  activeTarget = target
  text.value = tooltipText
  visible.value = true
  await nextTick()
  placeTooltip()
}

function hideTooltip(): void {
  activeTarget = null
  visible.value = false
}

function onPointerOver(event: PointerEvent): void {
  const target = findTooltipTarget(event.target)
  if (target) void showTooltip(target)
}

function onPointerOut(event: PointerEvent): void {
  if (!activeTarget) return
  if (event.relatedTarget instanceof Node && activeTarget.contains(event.relatedTarget)) return
  hideTooltip()
}

function onFocusIn(event: FocusEvent): void {
  const target = findTooltipTarget(event.target)
  if (target) void showTooltip(target)
}

function onFocusOut(event: FocusEvent): void {
  if (!activeTarget) return
  if (event.relatedTarget instanceof Node && activeTarget.contains(event.relatedTarget)) return
  hideTooltip()
}

onMounted(() => {
  migrateTree(document.body)
  observer = new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === 'attributes') migrateTree(record.target)
      record.addedNodes.forEach(migrateTree)
    }
  })
  observer.observe(document.body, {
    attributes: true,
    attributeFilter: ['title'],
    childList: true,
    subtree: true,
  })

  document.addEventListener('pointerover', onPointerOver, true)
  document.addEventListener('pointerout', onPointerOut, true)
  document.addEventListener('focusin', onFocusIn, true)
  document.addEventListener('focusout', onFocusOut, true)
  window.addEventListener('resize', placeTooltip)
  window.addEventListener('scroll', hideTooltip, true)
})

onUnmounted(() => {
  observer?.disconnect()
  document.removeEventListener('pointerover', onPointerOver, true)
  document.removeEventListener('pointerout', onPointerOut, true)
  document.removeEventListener('focusin', onFocusIn, true)
  document.removeEventListener('focusout', onFocusOut, true)
  window.removeEventListener('resize', placeTooltip)
  window.removeEventListener('scroll', hideTooltip, true)
})
</script>

<style>
.app-tooltip {
  position: fixed;
  z-index: var(--z-tooltip);
  width: max-content;
  max-width: min(460px, calc(100vw - 24px));
  padding: 10px 12px;
  /* 툴팁은 떠 있는 면이라 카드 배경 + 강한 테두리. 배경·글자를 같이 토큰화해야
     한쪽만 바뀌어 라이트 모드에서 글자가 사라지는 일이 없다. */
  border: 1px solid var(--edge);
  border-radius: 7px;
  background: var(--bg-card);
  color: var(--text-primary);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.72);
  font-size: 12px;
  font-weight: var(--fw-bold);
  line-height: 1.55;
  letter-spacing: normal;
  text-align: left;
  text-transform: none;
  white-space: pre-line;
  overflow-wrap: anywhere;
  pointer-events: none;
  user-select: none;
}

.app-tooltip-fade-enter-active,
.app-tooltip-fade-leave-active {
  transition: opacity 0.12s ease, transform 0.12s ease;
}

.app-tooltip-fade-enter-from,
.app-tooltip-fade-leave-to {
  opacity: 0;
  transform: translateY(-3px);
}
</style>
