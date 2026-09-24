<template>
  <details class="ext-card" open>
    <summary class="ext-title">프롬프트 필터</summary>
    <!-- Rating 토글 -->
    <div class="ext-sub-title">등급 필터</div>
    <div class="rating-toggle-row">
      <button v-for="r in ratingFilters" :key="r.key" class="rating-toggle"
        :class="{ active: r.on }" @click="r.on = !r.on; saveRatingFilter()">{{ r.label }}</button>
    </div>
    <div class="ext-toggle-grid">
      <label class="ext-check-row"><ToggleSwitch v-model="removeCharacter" size="sm" /><span>캐릭터 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeCharacterFeatures" size="sm" /><span>캐릭터 특징 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeCopyright" size="sm" /><span>작품 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeArtist" size="sm" /><span>작가 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeMeta" size="sm" /><span>메타 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeCensorship" size="sm" /><span>검열 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="removeText" size="sm" /><span>텍스트 제거</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="promptFocus" size="sm" /><span title="muscular male이 있으면 muscular처럼, 더 구체적인 태그에 포함되는 광범위 태그를 제거해 집중 태그만 남김">프롬프트 집중</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="autoCharFeatures" size="sm" /><span>특징 자동 추가</span></label>
      <label class="ext-check-row"><ToggleSwitch v-model="autoRemoveCharFeatures" size="sm" /><span title="closed eyes 있으면 눈색 특징 생략, 머리 길이 충돌 시 생략">특징 auto remove</span></label>
    </div>
    <div v-if="autoRemoveCharFeatures" class="char-ovr-row">
      <button class="char-ovr-btn" @click="openCharOverrideModal()"><Icon name="settings" /> override 설정 (머리길이 / 눈색)</button>
    </div>
  </details>
</template>

<script setup lang="ts">
/**
 * 파라미터 열 — 프롬프트 필터 카드(등급 g/s/q/e + 태그 제거 토글 10개). App.vue 에서 추출(App.vue 분할 ④).
 * 등급 필터 상태의 주인은 App(uiPrefsLoaded 에서 복원)이라 useRatingFilter 결과를 prop 으로 받는다.
 */
import type { PropType } from 'vue'
import ToggleSwitch from '../ToggleSwitch.vue'
import { useWidgetStore } from '../../stores/widgetStore.js'
import { widgetFlag } from '../../composables/widgetFlag'
import { openCharOverrideModal } from '../../composables/uiModals.js'
import type { useRatingFilter } from '../../composables/useRatingFilter.js'

const props = defineProps({
  rating: { type: Object as PropType<ReturnType<typeof useRatingFilter>>, required: true },
})

const storeWidgets = useWidgetStore().widgets
const { ratingFilters, saveRatingFilter } = props.rating

const removeArtist = widgetFlag(storeWidgets, 'chk_remove_artist')
const removeCopyright = widgetFlag(storeWidgets, 'chk_remove_copyright')
const removeCharacter = widgetFlag(storeWidgets, 'chk_remove_character')
const removeCharacterFeatures = widgetFlag(storeWidgets, 'chk_remove_character_features')
const removeMeta = widgetFlag(storeWidgets, 'chk_remove_meta')
const removeCensorship = widgetFlag(storeWidgets, 'chk_remove_censorship')
const removeText = widgetFlag(storeWidgets, 'chk_remove_text')
const promptFocus = widgetFlag(storeWidgets, 'chk_prompt_focus')
const autoCharFeatures = widgetFlag(storeWidgets, 'chk_auto_char_features')
const autoRemoveCharFeatures = widgetFlag(storeWidgets, 'chk_auto_remove_char_features')
</script>

<style scoped>
.rating-toggle-row { display: flex; gap: 4px; margin-bottom: 8px; }
.rating-toggle {
  flex: 1; padding: 4px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold);
  cursor: pointer; text-align: center; transition: var(--transition);
}
.rating-toggle.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.char-ovr-row { margin-top: 6px; }
.char-ovr-btn { width: 100%; padding: 6px 10px; background: var(--accent-dim); border: 1px solid var(--accent); border-radius: var(--radius-base); color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.char-ovr-btn:hover { background: rgba(250,204,21,0.18); }
</style>
