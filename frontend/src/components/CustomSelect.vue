<template>
  <div class="csel" :class="{ open: isOpen }" ref="root">
    <button
      ref="trigger"
      type="button"
      class="csel-display"
      role="combobox"
      aria-haspopup="listbox"
      :aria-expanded="isOpen"
      :aria-controls="listboxId"
      :aria-activedescendant="activeDescendant"
      @click="onTriggerClick"
      @keydown="onTriggerKeydown"
    >
      <span class="csel-text">{{ displayText }}</span>
      <span class="csel-arrow" aria-hidden="true"><Icon name="chevron-down" /></span>
    </button>
    <Teleport to="body">
    <div ref="dropdown" :id="listboxId" class="csel-dropdown" :style="menuStyle" role="listbox" v-if="isOpen">
      <template v-if="normalizedGroups.length">
        <section v-for="(group, groupIndex) in normalizedGroups"
          :key="`${group.source || group.label}-${groupIndex}`" class="csel-group"
          role="group" :aria-label="group.label">
          <div class="csel-group-header" role="presentation">
            <span>{{ group.label }}</span>
            <span v-if="group.primary" class="csel-group-main">메인</span>
            <span v-else-if="group.source" class="csel-group-source">{{ group.source }}</span>
          </div>
          <div v-for="(opt, optionIndex) in group.options"
            :key="`${group.source || group.label}-${String(opt)}-${optionIndex}`"
            class="csel-option csel-group-option"
            :class="{ selected: modelValue === opt, active: activeIndex === optionFlatIndex(groupIndex, optionIndex) }"
            :id="optionDomId(optionFlatIndex(groupIndex, optionIndex))"
            role="option"
            :aria-selected="modelValue === opt"
            @mouseenter="activeIndex = optionFlatIndex(groupIndex, optionIndex)"
            @click="select(opt, 'pointer')"
          >{{ opt === '' ? placeholder : opt }}</div>
        </section>
      </template>
      <template v-else>
        <div v-for="(opt, optionIndex) in options" :key="`${String(opt)}-${optionIndex}`" class="csel-option"
          :class="{ selected: modelValue === opt, active: activeIndex === optionIndex }"
          :id="optionDomId(optionIndex)"
          role="option"
          :aria-selected="modelValue === opt"
          @mouseenter="activeIndex = optionIndex"
          @click="select(opt, 'pointer')">{{ opt === '' ? placeholder : opt }}</div>
      </template>
      <div v-if="!hasOptions" class="csel-empty">항목 없음</div>
    </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onDeactivated, onUnmounted, useId, watch } from 'vue'
import { dropdownPlacement } from '../utils/dropdownPlacement'

type SelectValue = string | number
interface SelectOptionGroup {
  label: string
  source?: string
  primary?: boolean
  options: SelectValue[]
}

const props = withDefaults(defineProps<{
  modelValue?: SelectValue
  options?: SelectValue[]
  optionGroups?: SelectOptionGroup[]
  placeholder?: string
}>(), {
  modelValue: '',
  options: () => [],
  optionGroups: () => [],
  placeholder: '선택...',
})
const emit = defineEmits<{ 'update:modelValue': [value: SelectValue] }>()

const isOpen = ref(false)
const root = ref<HTMLElement | null>(null)
const trigger = ref<HTMLButtonElement | null>(null)
const dropdown = ref<HTMLElement | null>(null)
const menuStyle = ref<Record<string, string>>({})
let placementFrame = 0
let triggerObserver: ResizeObserver | undefined
const activeIndex = ref(-1)
const listboxId = `csel-listbox-${useId().replace(/[^a-zA-Z0-9_-]/g, '')}`

const displayText = computed(() => props.modelValue || props.placeholder)
const normalizedGroups = computed<SelectOptionGroup[]>(() => props.optionGroups
  .filter(group => group && Array.isArray(group.options) && group.options.length > 0)
  .map(group => ({
    label: String(group.label || group.source || 'Models'),
    source: group.source ? String(group.source).toUpperCase() : '',
    primary: Boolean(group.primary),
    options: group.options,
  })))
const hasOptions = computed(() => normalizedGroups.value.length > 0 || props.options.length > 0)
const flatOptions = computed<SelectValue[]>(() => normalizedGroups.value.length
  ? normalizedGroups.value.flatMap(group => group.options)
  : props.options)
const activeDescendant = computed(() => isOpen.value && activeIndex.value >= 0
  ? optionDomId(activeIndex.value)
  : undefined)

watch(activeIndex, index => {
  if (!isOpen.value || index < 0) return
  void nextTick(() => {
    document.getElementById(optionDomId(index))?.scrollIntoView({ block: 'nearest' })
  })
})

function optionFlatIndex(groupIndex: number, optionIndex: number) {
  return normalizedGroups.value
    .slice(0, groupIndex)
    .reduce((total, group) => total + group.options.length, optionIndex)
}

function optionDomId(index: number) {
  return `${listboxId}-option-${index}`
}

function open(direction: 1 | -1 = 1) {
  if (!flatOptions.value.length) return
  updatePlacement()
  isOpen.value = true
  const selectedIndex = flatOptions.value.findIndex(option => option === props.modelValue)
  activeIndex.value = selectedIndex >= 0
    ? selectedIndex
    : direction > 0 ? 0 : flatOptions.value.length - 1
}

function updatePlacement() {
  if (!trigger.value) return
  const { above: _above, ...position } = dropdownPlacement(trigger.value.getBoundingClientRect(), {
    width: document.documentElement.clientWidth, height: window.innerHeight,
  })
  menuStyle.value = Object.fromEntries(Object.entries(position).map(([key, value]) => [key, `${value}px`]))
}

function schedulePlacement() {
  if (!isOpen.value || placementFrame) return
  placementFrame = requestAnimationFrame(() => { placementFrame = 0; updatePlacement() })
}

/**
 * 포커스는 키보드로 다루는 동안만 트리거에 둔다. 트리거에 포커스가 있으면 ↑/↓ 는 이 목록의 몫이다
 * (onTriggerKeydown 이 preventDefault → utils/appShortcuts 가 히스토리 이동을 비킨다). 마우스로 고르거나
 * 닫은 뒤에도 포커스가 남으면, 이어서 ↓ 로 히스토리를 넘기려 해도 목록만 다시 열렸다(S2-appvue-split#2-a).
 * 그래서 마우스로 끝낸 상호작용은 포커스를 놓는다 — 브라우저가 옵션 mousedown 에서 이미 놓았어도 무해하다.
 */
function releaseFocus() {
  trigger.value?.blur()
}

function onTriggerClick(event: MouseEvent) {
  if (!isOpen.value) { open(); return }
  isOpen.value = false
  // detail = 클릭 수 — 0 이면 키보드·코드가 만든 클릭(그때는 포커스를 그대로 둔다)
  if (event.detail > 0) releaseFocus()
}

function moveActive(delta: 1 | -1) {
  if (!isOpen.value) {
    open(delta)
    return
  }
  const length = flatOptions.value.length
  if (!length) return
  activeIndex.value = (activeIndex.value + delta + length) % length
}

function onTriggerKeydown(event: KeyboardEvent) {
  if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
    event.preventDefault()
    moveActive(event.key === 'ArrowDown' ? 1 : -1)
    return
  }
  if (event.key === 'Home' || event.key === 'End') {
    if (!isOpen.value) return
    event.preventDefault()
    activeIndex.value = event.key === 'Home' ? 0 : flatOptions.value.length - 1
    return
  }
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    if (!isOpen.value) open()
    else if (activeIndex.value >= 0) select(flatOptions.value[activeIndex.value], 'keyboard')
    return
  }
  if (event.key === 'Escape' && isOpen.value) {
    event.preventDefault()
    isOpen.value = false
  } else if (event.key === 'Tab') {
    isOpen.value = false
  }
}

/** 항목 고르기 — 키보드(트리거의 Enter/Space)로 골랐으면 포커스를 트리거에 두고(이어서 ↑/↓ 로 다시 연다),
 *  마우스로 골랐으면 놓는다(이어지는 ↑/↓ 는 히스토리 — releaseFocus 참고). */
function select(opt: SelectValue, via: 'keyboard' | 'pointer') {
  emit('update:modelValue', opt)
  isOpen.value = false
  if (via === 'keyboard') void nextTick(() => trigger.value?.focus())
  else releaseFocus()
}

function onClickOutside(e: MouseEvent) {
  if (root.value && !root.value.contains(e.target as Node) && !dropdown.value?.contains(e.target as Node)) isOpen.value = false
}

onMounted(() => {
  document.addEventListener('click', onClickOutside)
  window.addEventListener('scroll', schedulePlacement, true)
  window.addEventListener('resize', schedulePlacement)
  triggerObserver = new ResizeObserver(schedulePlacement)
  if (trigger.value) triggerObserver.observe(trigger.value)
})
// Teleported menus must not remain above another keep-alive tab.
onDeactivated(() => {
  isOpen.value = false
  cancelAnimationFrame(placementFrame)
  placementFrame = 0
})
onUnmounted(() => {
  document.removeEventListener('click', onClickOutside)
  window.removeEventListener('scroll', schedulePlacement, true)
  window.removeEventListener('resize', schedulePlacement)
  triggerObserver?.disconnect()
  cancelAnimationFrame(placementFrame)
})
</script>

<style scoped>
.csel { position: relative; width: 100%; }
.csel-display {
  width: 100%; text-align: left;
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px; background: var(--bg-input); border: 1px solid var(--border);
  border-radius: var(--radius-base); color: var(--text-primary); font-size: 14px;
  cursor: pointer; transition: var(--transition);
}
.csel.open .csel-display { border-color: var(--accent); }
.csel-display:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.csel-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.csel-arrow { color: var(--text-muted); font-size: 12px; flex-shrink: 0; }
.csel-dropdown {
  position: fixed; z-index: 20000; box-sizing: border-box;
  max-height: 240px; overflow-y: auto; overscroll-behavior: contain;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 12px 32px rgba(0,0,0,0.2);
}
.csel-option {
  overflow-wrap: anywhere; white-space: normal;
  padding: 8px 14px; color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: background 0.1s;
}
.csel-option:hover { background: var(--accent-dim); color: var(--accent); }
.csel-option.active:not(.selected) { background: var(--accent-dim); color: var(--accent); }
/* 선택된 항목은 글자를 얹는 면이라 --accent-fill + --on-accent */
.csel-option.selected { background: var(--accent-fill); color: var(--on-accent); font-weight: var(--fw-bold); }
.csel-group + .csel-group { border-top: 1px solid var(--border); }
.csel-group-header {
  position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: 6px;
  padding: 7px 11px; background: var(--bg-secondary); color: var(--text-muted);
  font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; cursor: default;
}
.csel-group-main, .csel-group-source {
  padding: 2px 5px; border: 1px solid rgba(96,165,250,.35); border-radius: 7px;
  background: rgba(96,165,250,.1); color: var(--state-info-fg); font-size: 7px; letter-spacing: 0;
}
.csel-group-source { border-color: var(--border); background: var(--bg-input); color: var(--text-muted); }
.csel-group-option { padding-left: 17px; }
.csel-empty { padding: 12px; color: var(--text-muted); text-align: center; font-size: 12px; }
</style>
