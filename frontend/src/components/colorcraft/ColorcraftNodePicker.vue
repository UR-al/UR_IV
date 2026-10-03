<template>
  <div class="cc-picker" role="toolbar" :aria-label="label" @keydown="onKey">
    <template v-for="item in marks" :key="item.tag">
      <span v-if="dividerBefore && item.tag === dividerBefore" class="cc-picker-divider" aria-hidden="true">│</span>
      <button type="button" class="cc-chip"
        :class="{ 'cc-chip-on': item.mark === MARK_ON, 'cc-chip-edited': item.mark === MARK_EDITED,
                  'cc-chip-selected': item.tag === selected }"
        :aria-pressed="item.tag === selected ? 'true' : 'false'" :title="titleOf(item)"
        @click="emit('select', item.tag)">{{ item.label }}</button>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * Colorcraft 노드 고르개 — 수정자(I … X) 또는 마스크(M1 … M10 │ C1 … C5) 칩. 표시는 확장 공유 편집기와 같은 ●(켜짐·쓰는
 * 마스크)/○(바꿨지만 꺼짐) — utils/colorcraft.overview 가 만든 표를 받아 그린다. ←/→ 로 옆 노드를 고른다(끝에서 돈다).
 */
import { MARK_EDITED, MARK_ON, type PickerMark } from '../../utils/colorcraft'

const props = withDefaults(defineProps<{
  marks: PickerMark[]
  selected: string
  label: string
  dividerBefore?: string
}>(), { dividerBefore: '' })
const emit = defineEmits<{ select: [tag: string] }>()

function titleOf(item: PickerMark): string {
  if (item.mark === MARK_ON) return `${item.tag} — 켜짐(쓰는 중)`
  if (item.mark === MARK_EDITED) return `${item.tag} — 바꿨지만 꺼짐`
  return `${item.tag} — 기본값`
}

function onKey(event: KeyboardEvent) {
  if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return
  const tags = props.marks.map(m => m.tag)
  if (!tags.length) return
  const at = Math.max(0, tags.indexOf(props.selected))
  const step = event.key === 'ArrowRight' ? 1 : -1
  event.preventDefault()
  emit('select', tags[(at + step + tags.length) % tags.length]!)
}
</script>

<style scoped>
.cc-picker { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; margin: 4px 0 6px; }
.cc-picker-divider { color: var(--text-muted); padding: 0 2px; }
.cc-chip {
  min-width: 32px; height: 24px; padding: 0 6px; border-radius: 4px; cursor: pointer; font-size: var(--fs-meta);
  background: transparent; border: 1px solid var(--border); color: var(--text-muted);
}
.cc-chip:hover { border-color: var(--text-muted); color: var(--text-primary); }
.cc-chip-on { color: var(--state-info-fg); border-color: var(--state-info-fg); }
.cc-chip-edited { color: var(--text-primary); border-style: dashed; }
.cc-chip-selected { background: var(--bg-button); color: var(--text-primary); font-weight: var(--fw-medium); }
</style>
