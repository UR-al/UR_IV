<template>
  <div class="watermark-panel">
    <!-- Text Watermark -->
    <fieldset class="group-box">
      <legend>텍스트 워터마크</legend>

      <input
        v-model="textValue"
        type="text"
        placeholder="워터마크 텍스트 입력..."
        class="text-input"
      />

      <div class="font-color-row">
        <CustomSelect v-model="fontFamily" :options="fonts" placeholder="Font" class="font-select" />
        <button
          class="color-btn"
          :style="{ backgroundColor: textColor }"
          @click="$emit('pick-text-color')"
        >
          색상
        </button>
      </div>

      <div class="slider-group">
        <label class="slider-label">크기</label>
        <input type="range" :min="8" :max="200" v-model.number="fontSize" class="slider" />
        <span class="slider-value">{{ fontSize }}</span>
      </div>

      <!-- Position Presets -->
      <div class="preset-row">
        <button
          v-for="pos in positionPresets"
          :key="pos.name"
          class="preset-pos-btn"
          @click="setTextPosition(pos.x, pos.y)"
        >
          {{ pos.name }}
        </button>
      </div>

      <div class="slider-group">
        <label class="slider-label">X 위치 (%)</label>
        <input type="range" :min="0" :max="100" v-model.number="textX" class="slider" />
        <span class="slider-value">{{ textX }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">Y 위치 (%)</label>
        <input type="range" :min="0" :max="100" v-model.number="textY" class="slider" />
        <span class="slider-value">{{ textY }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">투명도</label>
        <input type="range" :min="0" :max="100" v-model.number="textOpacity" class="slider" />
        <span class="slider-value">{{ textOpacity }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">회전</label>
        <input type="range" :min="-180" :max="180" v-model.number="textRotation" class="slider" />
        <span class="slider-value">{{ textRotation }}°</span>
      </div>

      <label class="checkbox-row">
        <input type="checkbox" v-model="tileRepeat" />
        <span>타일 반복</span>
      </label>

      <button class="accent-btn" @click="onApplyText">텍스트 워터마크 적용</button>
    </fieldset>

    <!-- Image Watermark -->
    <fieldset class="group-box">
      <legend>이미지 워터마크</legend>

      <button class="file-btn" v-host-dialog="'editor_load_watermark_image'" @click="$emit('load-watermark-image')">이미지 불러오기</button>
      <div class="file-label" :title="imagePath || undefined">{{ imageLabel }}</div>

      <!-- Position Presets -->
      <div class="preset-row">
        <button
          v-for="pos in positionPresets"
          :key="pos.name"
          class="preset-pos-btn"
          @click="setImagePosition(pos.x, pos.y)"
        >
          {{ pos.name }}
        </button>
      </div>

      <div class="slider-group">
        <label class="slider-label">X 위치 (%)</label>
        <input type="range" :min="0" :max="100" v-model.number="imgX" class="slider" />
        <span class="slider-value">{{ imgX }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">Y 위치 (%)</label>
        <input type="range" :min="0" :max="100" v-model.number="imgY" class="slider" />
        <span class="slider-value">{{ imgY }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">투명도</label>
        <input type="range" :min="0" :max="100" v-model.number="imgOpacity" class="slider" />
        <span class="slider-value">{{ imgOpacity }}</span>
      </div>

      <div class="slider-group">
        <label class="slider-label">크기 (%)</label>
        <input type="range" :min="10" :max="500" v-model.number="imgScale" class="slider" />
        <span class="slider-value">{{ imgScale }}</span>
      </div>

      <button class="accent-btn" @click="onApplyImage">이미지 워터마크 적용</button>
    </fieldset>

    <!-- Common Options -->
    <label class="checkbox-row bold">
      <input type="checkbox" v-model="clampToImage" />
      <span>이미지 영역 내 제한</span>
    </label>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { vHostDialog } from '../../utils/hostDialogs'
import {
  NO_WATERMARK_IMAGE, WATERMARK_FONTS, watermarkFileLabel, watermarkPreviewToRefresh,
  type WatermarkKind,
} from '../../utils/watermark'
import CustomSelect from '../CustomSelect.vue'

interface PositionPreset {
  name: string
  x: number
  y: number
}

interface TextWatermarkConfig {
  type: 'text'
  text: string
  fontFamily: string
  fontSize: number
  xPct: number
  yPct: number
  opacity: number
  rotation: number
  tile: boolean
}

interface ImageWatermarkConfig {
  type: 'image'
  xPct: number
  yPct: number
  opacity: number
  /** 원래 크기 대비 % (슬라이더 '크기 (%)' 그대로). 백엔드도 %로 받는다. */
  scale: number
}

const emit = defineEmits<{
  'apply-text': [config: TextWatermarkConfig]
  'apply-image': [config: ImageWatermarkConfig]
  'preview': [config: TextWatermarkConfig | ImageWatermarkConfig]
  'preview-clear': []
  'pick-text-color': []
  'load-watermark-image': []
  'clamp-changed': [val: boolean]
}>()

const props = withDefaults(
  defineProps<{
    textColor?: string
    /** 불러온 워터마크 이미지 경로 — 파일 이름 표시와, 바뀌면 이미지 프리뷰 다시 그리기 */
    imagePath?: string
    /** 이미지가 없을 때 보일 문구 */
    imageFileName?: string
    fonts?: string[]
  }>(),
  {
    textColor: '#FFFFFF',
    imagePath: '',
    imageFileName: NO_WATERMARK_IMAGE,
    fonts: () => [...WATERMARK_FONTS],
  }
)

const imageLabel = computed(() => watermarkFileLabel(props.imagePath, props.imageFileName))

const positionPresets: PositionPreset[] = [
  { name: '좌상', x: 5, y: 5 },
  { name: '우상', x: 95, y: 5 },
  { name: '중앙', x: 50, y: 50 },
  { name: '좌하', x: 5, y: 95 },
  { name: '우하', x: 95, y: 95 },
]

const textValue = ref('')
const fontFamily = ref('Arial')
const fontSize = ref(36)
const textX = ref(95)
const textY = ref(95)
const textOpacity = ref(50)
const textRotation = ref(0)
const tileRepeat = ref(false)

const imgX = ref(95)
const imgY = ref(95)
const imgOpacity = ref(50)
const imgScale = ref(100)

const clampToImage = ref(true)

function setTextPosition(x: number, y: number) {
  textX.value = x
  textY.value = y
}

function setImagePosition(x: number, y: number) {
  imgX.value = x
  imgY.value = y
}

// 마지막으로 보여 준 프리뷰 종류 — 공용 옵션(영역 제한·글자 색)이 바뀌면 그걸 다시 그린다
let lastPreview: WatermarkKind | null = null
function previewText() { lastPreview = 'text'; emit('preview', buildTextConfig()) }
function previewImage() { lastPreview = 'image'; emit('preview', buildImageConfig()) }
function clearPreviewFromText() { lastPreview = null; emit('preview-clear') }

// Emit preview on relevant changes — 글꼴도 결과를 바꾸므로 목록에 있어야 한다(예전엔 빠져 있었다)
watch(
  [textValue, fontFamily, fontSize, textX, textY, textOpacity, textRotation, tileRepeat],
  () => {
    if (textValue.value.trim()) previewText()
    else clearPreviewFromText()
  }
)

// 글자 색은 부모(EditorView)가 고른다 — 바뀌면 텍스트 프리뷰를 다시 그린다
watch(() => props.textColor, () => {
  if (textValue.value.trim()) previewText()
})

watch([imgX, imgY, imgOpacity, imgScale], () => {
  previewImage()
})

// 워터마크 이미지를 (다시) 불러오면 슬라이더를 건드리지 않아도 바로 보여 준다
watch(() => props.imagePath, (path) => {
  if (path) previewImage()
})

watch(clampToImage, (val) => {
  // 부모가 새 값을 먼저 받아야 다시 그린 프리뷰에 반영된다
  emit('clamp-changed', val)
  const kind = watermarkPreviewToRefresh(lastPreview, !!textValue.value.trim(), !!props.imagePath)
  if (kind === 'text') previewText()
  else if (kind === 'image') previewImage()
})

function buildTextConfig(): TextWatermarkConfig {
  return {
    type: 'text',
    text: textValue.value.trim(),
    fontFamily: fontFamily.value,
    fontSize: fontSize.value,
    xPct: textX.value,
    yPct: textY.value,
    opacity: textOpacity.value / 100.0,
    rotation: textRotation.value,
    tile: tileRepeat.value,
  }
}

function buildImageConfig(): ImageWatermarkConfig {
  return {
    type: 'image',
    xPct: imgX.value,
    yPct: imgY.value,
    opacity: imgOpacity.value / 100.0,
    // % 그대로 — 예전 `/ 100` 은 백엔드가 다시 100 으로 나눠 100% 가 1% 로 줄었다
    scale: imgScale.value,
  }
}

function onApplyText() {
  if (!textValue.value.trim()) return
  lastPreview = null
  emit('preview-clear')
  emit('apply-text', buildTextConfig())
}

function onApplyImage() {
  lastPreview = null
  emit('preview-clear')
  emit('apply-image', buildImageConfig())
}
</script>

<style scoped>
.watermark-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 10px;
  color: var(--text-primary);
  font-size: 13px;
}

.group-box {
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 15px 8px 8px;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.group-box legend {
  color: var(--text-muted);
  font-weight: var(--fw-bold);
  font-size: 13px;
  padding: 0 4px;
}

.text-input {
  background-color: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 6px;
  font-size: 13px;
  width: 100%;
  box-sizing: border-box;
}
.text-input::placeholder {
  color: var(--text-muted);
}

.font-color-row {
  display: flex;
  gap: 6px;
}

.font-select {
  flex: 2;
  background-color: var(--bg-input);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  padding: 4px;
  font-size: 12px;
}

.color-btn {
  flex: 1;
  height: 35px;
  border: 1px solid var(--rule);
  border-radius: 4px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
  /* 배경이 사용자가 고른 워터마크 색(인라인 스타일)이라 테마와 무관하다 —
     밝은 색 위에서 글자가 읽히게 검정 고정. 토큰화하면 흰 색을 고른 순간 안 보인다. */
  color: #000;
}

.preset-row {
  display: flex;
  gap: 3px;
}

.preset-pos-btn {
  flex: 1;
  height: 26px;
  background-color: var(--bg-button);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 3px;
  font-size: 11px;
  cursor: pointer;
  padding: 3px 6px;
}
.preset-pos-btn:hover {
  background-color: var(--bg-button-hover);
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

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}
.checkbox-row.bold {
  font-weight: var(--fw-bold);
}
.checkbox-row input[type="checkbox"] {
  accent-color: var(--accent);
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
  width: 100%;
}
.accent-btn:hover {
  background-color: var(--accent-fill-hover);
}
</style>
