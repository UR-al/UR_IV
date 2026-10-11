<template>
  <div class="color-match-panel">
    <div class="mode-grid">
      <button class="preset-btn" :class="{ active: mode === 'reference' }" @click="mode = 'reference'">참조 이미지</button>
      <button class="preset-btn" :class="{ active: mode === 'ring' }" @click="mode = 'ring'">마스크 주변</button>
    </div>

    <template v-if="mode === 'reference'">
      <button class="file-btn" v-host-dialog="'editor_load_color_reference'" @click="$emit('load-reference')">참조 이미지 불러오기</button>
      <div class="file-label" :title="referencePath || undefined">{{ referenceName }}</div>
      <label class="checkbox-row">
        <input type="checkbox" v-model="maskOnly" />
        <span>칠한 영역만</span>
      </label>
      <p class="hint-text">참조 이미지의 색감(평균·대비)을 옮깁니다</p>
    </template>
    <template v-else>
      <div class="slider-group">
        <label class="slider-label">주변 폭 px</label>
        <input type="range" :min="RING_PX_MIN" :max="RING_PX_MAX" v-model.number="ringPx" class="slider" />
        <span class="slider-value">{{ ringPx }}</span>
      </div>
      <p class="hint-text">칠한 영역의 색을 바로 바깥 테두리에 맞춥니다 (인페인트 경계 보정)</p>
    </template>
    <p v-if="needsMask && !hasMask" class="need-hint">먼저 브러시·올가미로 영역을 칠하세요</p>

    <div class="slider-group">
      <label class="slider-label">강도 %</label>
      <input type="range" :min="STRENGTH_MIN" :max="STRENGTH_MAX" v-model.number="strength" class="slider" />
      <span class="slider-value">{{ strength }}</span>
    </div>

    <div class="btn-row">
      <button class="accent-btn flex-2" @click="$emit('apply', config())">색 맞추기 적용</button>
      <button class="reset-btn flex-1" @click="$emit('cancel')">미리보기 해제</button>
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * 보정 탭 '참조 색 맞추기' — 참조 이미지의 색감(Lab 평균·표준편차)을 옮기거나, 칠한 영역을 바로 바깥
 * 테두리 색에 맞춘다(인페인트·손 보정 경계). 계산은 백엔드(core/editor_color_match), 무엇을 보낼지는
 * 부모의 useColorMatch(utils/colorMatch.colorMatchPlan). 여기는 설정 표시와 저장만 한다.
 *
 * 설정을 바꾸면 'preview' — 부모가 디바운스 프리뷰를 건다(보낼 수 없는 설정이면 걷는다).
 * 참조 이미지를 새로 불러와도 다시 'preview'. 경로는 부모가 들고 있고 저장하지 않는다.
 */
import { computed, ref, watch } from 'vue'
import { vHostDialog } from '../../utils/hostDialogs'
import {
  RING_PX_MAX, RING_PX_MIN, STRENGTH_MAX, STRENGTH_MIN, colorMatchNeedsMask, loadColorMatchPrefs,
  referenceLabel, saveColorMatchPrefs, type ColorMatchConfig, type ColorMatchMode,
} from '../../utils/colorMatch'

const props = withDefaults(defineProps<{
  /** 불러온 참조 이미지 경로 — 비어 있으면 '불러온 이미지 없음' */
  referencePath?: string
  /** 캔버스에 칠한 영역이 있는지 — 마스크가 필요한 설정에서 안내를 띄운다 */
  hasMask?: boolean
}>(), { referencePath: '', hasMask: false })

const emit = defineEmits<{
  'load-reference': []
  preview: [config: ColorMatchConfig]
  apply: [config: ColorMatchConfig]
  cancel: []
}>()

const saved = loadColorMatchPrefs()
const mode = ref<ColorMatchMode>(saved.mode)
const strength = ref(saved.strength)
const ringPx = ref(saved.ringPx)
const maskOnly = ref(saved.maskOnly)

function config(): ColorMatchConfig {
  return { mode: mode.value, strength: strength.value, ringPx: ringPx.value, maskOnly: maskOnly.value }
}

const needsMask = computed(() => colorMatchNeedsMask(config()))
const referenceName = computed(() => referenceLabel(props.referencePath))

watch([mode, strength, ringPx, maskOnly], () => {
  saveColorMatchPrefs(config())
  emit('preview', config())
})
watch(() => props.referencePath, (path) => {
  if (path && mode.value === 'reference') emit('preview', config())
})
</script>

<style scoped>
.color-match-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px;
  color: var(--text-primary);
  font-size: 13px;
}

.mode-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 6px;
}

.preset-btn {
  height: 35px;
  background-color: var(--bg-button);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  font-size: 12px;
  font-weight: var(--fw-bold);
  cursor: pointer;
  word-break: keep-all;
  line-height: 1.15;
}
.preset-btn:hover {
  background-color: var(--bg-button-hover);
  border-color: var(--edge);
}
.preset-btn.active {
  background-color: var(--accent-fill);
  color: var(--on-accent);
  border-color: var(--accent-fill);
}

.file-btn {
  height: 35px;
  background-color: var(--bg-button);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
  width: 100%;
}
.file-btn:hover {
  background-color: var(--bg-button-hover);
}

.file-label {
  color: var(--text-muted);
  font-size: 11px;
  min-height: 20px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}
.checkbox-row input[type="checkbox"] {
  accent-color: var(--accent);
}

.hint-text {
  margin: 0;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.slider-label {
  color: var(--text-secondary);
  font-size: 12px;
  min-width: 70px;
  white-space: nowrap;
}

.slider {
  flex: 1;
  accent-color: var(--accent);
  height: 4px;
  background: var(--rule);
  border-radius: 2px;
}

.slider-value {
  color: var(--text-primary);
  font-size: 12px;
  min-width: 30px;
  text-align: right;
}

.btn-row {
  display: flex;
  gap: 6px;
}

.accent-btn {
  height: 35px;
  background-color: var(--accent-fill);
  color: var(--on-accent);
  border: none;
  border-radius: 4px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
.accent-btn:hover {
  background-color: var(--accent-fill-hover);
}

.reset-btn {
  height: 35px;
  background-color: var(--bg-button);
  color: var(--text-secondary);
  border: none;
  border-radius: 4px;
  font-size: 13px;
  cursor: pointer;
}
.reset-btn:hover {
  background-color: var(--bg-button-hover);
}

.flex-1 { flex: 1; }
.flex-2 { flex: 2; }
</style>
