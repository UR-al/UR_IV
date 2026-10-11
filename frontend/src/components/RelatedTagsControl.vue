<template>
  <details class="related-tags" :open="open" @toggle="onToggle">
    <summary>연관 태그 <span>함께 자주 쓰인 태그 추천</span></summary>
    <div class="related-body">
      <p class="related-help">Danbooru 게시물에서 지금 프롬프트의 태그와 자주 함께 쓰인 태그입니다. 누르면 메인 태그 끝에 추가합니다.</p>
      <p v-if="ratings.length" class="related-meta">등급: {{ formatShards(ratings) }} (등급 필터를 따름)<template v-if="context.length"> · 기준 태그: {{ contextText }}</template></p>
      <div v-if="build" class="related-note" role="status">
        <span>{{ formatBuildProgress(build) }}</span>
        <button type="button" :disabled="cancelling" @click="cancelBuild()">취소</button>
      </div>
      <template v-else>
        <div v-if="state === 'missing'" class="related-note">
          <span>처음 한 번 연관 태그 표를 만들어야 합니다 ({{ estimateText }}, 그동안 앱은 그대로 쓸 수 있습니다).</span>
          <button type="button" class="related-primary" @click="startBuild()">표 만들기</button>
        </div>
        <div v-else-if="state === 'ready' && missing.length" class="related-note">
          <span>{{ formatShards(missing) }} 표가 없어 {{ formatShards(usable) }} 만 반영했습니다.</span>
          <button type="button" @click="startBuild()">{{ formatShards(missing) }} 표 만들기</button>
        </div>
        <p v-else-if="state === 'ready' && unavailable.length" class="related-help">{{ formatShards(unavailable) }} 데이터가 없어 {{ formatShards(usable) }} 만 반영했습니다.</p>
        <div v-if="state === 'ready' && stale.length" class="related-note">
          <span>Danbooru 데이터가 바뀌었습니다 — 다시 만들면 더 정확합니다.</span>
          <button type="button" @click="startBuild(stale)">다시 만들기</button>
        </div>
      </template>
      <p v-if="state === 'unavailable'" class="related-help">Danbooru 데이터가 없어 연관 태그를 쓸 수 없습니다. 검색 데이터를 받은 뒤 다시 열어 주세요.</p>
      <p v-if="state === 'error' && error" class="related-warning" role="alert">{{ error }}</p>
      <div v-if="tags.length" class="related-chips" aria-label="연관 태그">
        <button v-for="item in tags" :key="item.tag" type="button" class="related-chip" :title="relatedTagTitle(item)"
          @click="add(item)">{{ relatedTagLabel(item.tag) }}<span v-if="item.ko" class="rt-ko">{{ item.ko }}</span></button>
      </div>
      <p v-else-if="state === 'ready' && !loading" class="related-help">{{ context.length ? '함께 쓰인 태그를 찾지 못했습니다.' : '기준이 될 태그가 없습니다. 메인 태그를 입력해 보세요.' }}</p>
      <span class="related-feedback" role="status" aria-live="polite">{{ feedback || (loading && !tags.length ? '불러오는 중…' : '') }}</span>
    </div>
  </details>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { useRelatedTags } from '../composables/useRelatedTags'
import type { RelatedTagSuggestion } from '../types/bridge'
import {
  formatBuildProgress, formatEstimate, formatShards, planRelatedTagAppend, relatedTagLabel, relatedTagTitle,
} from '../utils/relatedTags'

const props = withDefaults(defineProps<{ modelValue: string; otherPrompts?: string; blockMode?: boolean }>(),
  { otherPrompts: '', blockMode: false })
const emit = defineEmits<{ append: [text: string] }>()
const store = useRelatedTags()
const {
  open, loading, state, tags, context, ratings, missing, stale, unavailable, estimateSec, build, cancelling,
  error, feedback, startBuild, cancelBuild,
} = store
const usable = computed(() => ratings.value.filter(s => !missing.value.includes(s) && !unavailable.value.includes(s)))
const contextText = computed(() => context.value.map(relatedTagLabel).join(', '))
const estimateText = computed(() => {
  const estimate = formatEstimate(estimateSec.value)
  return estimate ? `로컬 데이터 · ${estimate}` : '로컬 데이터'
})
let disposed = false
let disconnect: (() => void) | undefined

function onToggle(event: Event) {
  store.setOpen(!!(event.target as HTMLDetailsElement | null)?.open)
}
function add(item: RelatedTagSuggestion) {
  const plan = planRelatedTagAppend(props.modelValue, item.tag, props.blockMode, props.otherPrompts)
  store.dropSuggestion(item.tag)
  if (!plan.added) {
    feedback.value = '이미 프롬프트에 있는 태그입니다.'
    return
  }
  emit('append', plan.text)
  feedback.value = `메인 태그에 추가했습니다: ${relatedTagLabel(item.tag)}`
}
onMounted(async () => {
  disconnect = onBackendEvent('uiPrefsLoaded', store.restoreOpen) || undefined
  const backend = await getBackend()
  if (!disposed && backend?.getUiPrefs) backend.getUiPrefs(store.restoreOpen)
})
onUnmounted(() => { disposed = true; disconnect?.() })
</script>

<style scoped>
.related-tags { margin: 8px 0 12px; min-width: 0; border: 1px solid var(--border); border-radius: var(--radius-base, 8px); background: var(--bg-secondary); }
.related-tags summary { min-height: 36px; display: flex; align-items: center; gap: 8px; padding: 4px 10px; flex-wrap: wrap; cursor: pointer; color: var(--text-primary); font-size: var(--fs-label, 12px); font-weight: var(--fw-bold, 600); }
.related-tags summary::before { content: '›'; font-size: 16px; }
.related-tags[open] summary::before { content: '⌄'; }
.related-tags summary span { color: var(--text-muted); font-size: 11px; font-weight: normal; }
.related-body { padding: 0 10px 8px; min-width: 0; }
.related-help, .related-meta { margin: 5px 0 8px; color: var(--text-muted); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.related-meta { color: var(--text-secondary); }
.related-note { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 8px 0; color: var(--text-secondary); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.related-note span { flex: 1 1 180px; min-width: 0; }
.related-tags button { min-height: 32px; padding: 5px 8px; color: var(--text-secondary); background: var(--bg-button); border: 1px solid var(--border); border-radius: 5px; cursor: pointer; font: inherit; font-size: 11px; white-space: normal; }
.related-tags .related-primary { background: var(--accent-fill); color: var(--on-accent); border-color: var(--accent-fill); }
.related-tags :is(button, summary):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.related-tags button:disabled { opacity: 0.5; cursor: default; }
.related-warning { margin: 8px 0; color: var(--state-warn-fg); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
/* 분류 없는 추천이라 태그 6색 중 중립을 쓴다(공식 태그 칩 char-tag-chip 과 같은 면·테두리 비율) */
.related-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.related-tags .related-chip { min-height: 28px; padding: 2px 8px; background: color-mix(in srgb, var(--tag-neutral) 8%, transparent); border: 1px solid color-mix(in srgb, var(--tag-neutral) 20%, transparent); border-radius: 4px; color: var(--tag-neutral); font-size: var(--fs-label); }
.rt-ko { margin-left: 6px; font-size: var(--fs-label); color: var(--text-muted); }
.related-feedback { display: block; color: var(--text-muted); font-size: 11px; margin-top: 5px; overflow-wrap: anywhere; }
@media (hover: hover) and (pointer: fine) {
  .related-tags button:not(:disabled):hover { border-color: var(--accent); }
  .related-tags .related-chip:hover { background: color-mix(in srgb, var(--tag-neutral) 15%, transparent); border-color: var(--tag-neutral); }
}
@media (pointer: coarse) { .related-tags button { min-height: 44px; } }
</style>
