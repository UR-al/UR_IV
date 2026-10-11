<template>
  <div class="color-panel">
    <!-- Brightness / Contrast / Saturation Sliders -->
    <div class="slider-group">
      <label class="slider-label">밝기</label>
      <input type="range" :min="-100" :max="100" v-model.number="brightness" class="slider" />
      <span class="slider-value">{{ brightness }}</span>
    </div>

    <div class="slider-group">
      <label class="slider-label">대비</label>
      <input type="range" :min="-100" :max="100" v-model.number="contrast" class="slider" />
      <span class="slider-value">{{ contrast }}</span>
    </div>

    <div class="slider-group">
      <label class="slider-label">채도</label>
      <input type="range" :min="-100" :max="100" v-model.number="saturation" class="slider" />
      <span class="slider-value">{{ saturation }}</span>
    </div>

    <!-- Apply / Reset -->
    <div class="btn-row">
      <button class="accent-btn flex-2" @click="onApply">조정 적용</button>
      <button class="reset-btn flex-1" @click="onReset">초기화</button>
    </div>

    <!-- Auto Correction -->
    <button class="auto-correct-btn" @click="$emit('auto-correct')">자동 보정</button>

    <!-- Filter Presets -->
    <div class="section-label">필터 프리셋</div>

    <div class="preset-grid">
      <button
        v-for="preset in presets"
        :key="preset.name"
        class="preset-btn"
        :class="{ active: activeFilter === preset.name }"
        @click="onFilterSelect(preset.name)"
      >
        {{ preset.label }}
      </button>
    </div>

    <!-- 픽셀아트 옵션 — 고른 동안에만. 값은 localStorage(utils/pixelArt)에 남는다 -->
    <template v-if="activeFilter === 'pixelart'">
      <div class="slider-group">
        <label class="slider-label">픽셀 크기</label>
        <input type="range" :min="PIXEL_SIZE_MIN" :max="PIXEL_SIZE_MAX" v-model.number="pixelSize" class="slider" />
        <span class="slider-value">{{ pixelSizeLabel(pixelSize) }}</span>
      </div>
      <div class="slider-group">
        <label class="slider-label">색 수</label>
        <input type="range" :min="COLORS_MIN" :max="COLORS_MAX" v-model.number="pixelColors" class="slider" />
        <span class="slider-value">{{ colorsLabel(pixelColors) }}</span>
      </div>
    </template>

    <!-- Filter Strength -->
    <div class="slider-group">
      <label class="slider-label">필터 강도 %</label>
      <input type="range" :min="1" :max="100" v-model.number="filterStrength" class="slider" />
      <span class="slider-value">{{ filterStrength }}</span>
    </div>

    <!-- Filter Apply / Cancel -->
    <div class="btn-row">
      <button class="accent-btn flex-2" @click="onFilterApply">필터 적용</button>
      <button class="filter-cancel-btn flex-1" @click="onFilterCancel">필터 해제</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import {
  COLORS_MAX, COLORS_MIN, PIXEL_SIZE_MAX, PIXEL_SIZE_MIN, colorsLabel, loadPixelArtPrefs, pixelSizeLabel,
  savePixelArtPrefs, type PixelArtOptions,
} from '../../utils/pixelArt'

interface Adjustment {
  brightness: number
  contrast: number
  saturation: number
}

interface FilterPayload {
  filter: string
  strength: number
  /** 옵션이 있는 필터(픽셀아트)만 — 백엔드 editor_ops.FILTER_OPTIONS */
  options?: PixelArtOptions
}

interface Preset {
  label: string
  name: string
}

const emit = defineEmits<{
  'adjustment-changed': [payload: Adjustment]
  apply: [payload: Adjustment]
  reset: []
  'auto-correct': []
  'filter-preview': [payload: FilterPayload]
  'filter-apply': [payload: FilterPayload]
  'filter-cancel': []
}>()

const brightness = ref(0)
const contrast = ref(0)
const saturation = ref(0)
const filterStrength = ref(100)
const activeFilter = ref<string | null>(null)

const presets: Preset[] = [
  { label: '흑백', name: 'grayscale' },
  { label: '세피아', name: 'sepia' },
  { label: '선명하게', name: 'sharpen' },
  { label: '따뜻하게', name: 'warm' },
  { label: '차갑게', name: 'cool' },
  { label: '소프트', name: 'soft' },
  { label: '반전', name: 'invert' },
  { label: '엠보스', name: 'emboss' },
  { label: '스케치', name: 'sketch' },
  { label: '포스터', name: 'posterize' },
  { label: '비네트', name: 'vignette' },
  { label: '노이즈 제거', name: 'denoise' },
  { label: '픽셀아트', name: 'pixelart' },
]

const savedPixelArt = loadPixelArtPrefs()
const pixelSize = ref(savedPixelArt.pixel_size)
const pixelColors = ref(savedPixelArt.colors)

/** 필터 페이로드 — 픽셀아트면 옵션을 함께 싣는다 */
function filterPayload(name: string): FilterPayload {
  const payload: FilterPayload = { filter: name, strength: filterStrength.value }
  if (name === 'pixelart') payload.options = { pixel_size: pixelSize.value, colors: pixelColors.value }
  return payload
}

// Live preview on slider change
watch([brightness, contrast, saturation], () => {
  emit('adjustment-changed', {
    brightness: brightness.value,
    contrast: contrast.value,
    saturation: saturation.value,
  })
})

watch(filterStrength, () => {
  if (activeFilter.value) {
    emit('filter-preview', filterPayload(activeFilter.value))
  }
})

watch([pixelSize, pixelColors], () => {
  savePixelArtPrefs({ pixel_size: pixelSize.value, colors: pixelColors.value })
  if (activeFilter.value === 'pixelart') emit('filter-preview', filterPayload('pixelart'))
})

function onApply() {
  if (brightness.value === 0 && contrast.value === 0 && saturation.value === 0) return
  emit('apply', {
    brightness: brightness.value,
    contrast: contrast.value,
    saturation: saturation.value,
  })
  resetSliders()
}

function onReset() {
  resetSliders()
  emit('reset')
}

function resetSliders() {
  brightness.value = 0
  contrast.value = 0
  saturation.value = 0
}

function onFilterSelect(name: string) {
  if (activeFilter.value === name) {
    onFilterCancel()
    return
  }
  activeFilter.value = name
  emit('filter-preview', filterPayload(name))
}

function onFilterApply() {
  if (!activeFilter.value) return
  emit('filter-apply', filterPayload(activeFilter.value))
  clearFilterSelection()
}

function onFilterCancel() {
  clearFilterSelection()
  emit('filter-cancel')
}

function clearFilterSelection() {
  activeFilter.value = null
}
</script>

<style scoped>
.color-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px;
  color: var(--text-primary);
  font-size: 13px;
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

.auto-correct-btn {
  height: 35px;
  background-color: var(--state-ok);
  /* 상태 채움색은 세 프리셋 모두 흰 글자 기준으로 잡은 값이라 흰색을 유지한다 */
  color: #FFFFFF;
  border: none;
  border-radius: 4px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
/* 상태색에는 hover 파생 토큰이 없다 — 사용자가 state-ok 를 바꿔도 따라오게 밝기로 민다 */
.auto-correct-btn:hover {
  filter: brightness(1.15);
}

.section-label {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: var(--fw-bold);
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
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
  /* 좁은 4열 격자라 긴 이름은 두 줄이 된다 — 글자 중간('노이즈제|거')이 아니라 단어 사이에서 끊는다 */
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

.filter-cancel-btn {
  height: 35px;
  background-color: var(--state-alert);
  /* 상태 채움색은 세 프리셋 모두 흰 글자 기준으로 잡은 값이라 흰색을 유지한다 */
  color: #FFFFFF;
  border: none;
  border-radius: 4px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
/* 상태색에는 hover 파생 토큰이 없다 — 사용자가 state-alert 를 바꿔도 따라오게 밝기로 민다 */
.filter-cancel-btn:hover {
  filter: brightness(1.15);
}

.flex-1 { flex: 1; }
.flex-2 { flex: 2; }
</style>
