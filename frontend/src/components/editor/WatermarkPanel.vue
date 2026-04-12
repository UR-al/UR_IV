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

      <button class="file-btn" @click="$emit('load-watermark-image')">이미지 불러오기</button>
      <div class="file-label">{{ imageFileName }}</div>

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

<script setup>
import { ref, watch } from 'vue'
import CustomSelect from '../CustomSelect.vue'

const emit = defineEmits([
  'apply-text',
  'apply-image',
  'preview',
  'preview-clear',
  'pick-text-color',
  'load-watermark-image',
  'clamp-changed',
])

const props = defineProps({
  textColor: { type: String, default: '#FFFFFF' },
  imageFileName: { type: String, default: '이미지 없음' },
  fonts: { type: Array, default: () => ['Arial', 'Times New Roman', 'Courier New', 'Verdana', 'Georgia'] },
})

const positionPresets = [
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

function setTextPosition(x, y) {
  textX.value = x
  textY.value = y
}

function setImagePosition(x, y) {
  imgX.value = x
  imgY.value = y
}

// Emit preview on relevant changes
watch(
  [textValue, fontSize, textX, textY, textOpacity, textRotation, tileRepeat],
  () => {
    if (textValue.value.trim()) {
      emit('preview', buildTextConfig())
    } else {
      emit('preview-clear')
    }
  }
)

watch([imgX, imgY, imgOpacity, imgScale], () => {
  emit('preview', buildImageConfig())
})

watch(clampToImage, (val) => {
  emit('clamp-changed', val)
})

function buildTextConfig() {
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

function buildImageConfig() {
  return {
    type: 'image',
    xPct: imgX.value,
    yPct: imgY.value,
    opacity: imgOpacity.value / 100.0,
    scale: imgScale.value / 100.0,
  }
}

function onApplyText() {
  if (!textValue.value.trim()) return
  emit('preview-clear')
  emit('apply-text', buildTextConfig())
}

function onApplyImage() {
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
  color: #E8E8E8;
  font-size: 13px;
}

.group-box {
  border: 1px solid #2A2A2A;
  border-radius: 6px;
  padding: 15px 8px 8px;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.group-box legend {
  color: #585858;
  font-weight: bold;
  font-size: 13px;
  padding: 0 4px;
}

.text-input {
  background-color: #1A1A1A;
  color: #E8E8E8;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  padding: 6px;
  font-size: 13px;
  width: 100%;
  box-sizing: border-box;
}
.text-input::placeholder {
  color: #585858;
}

.font-color-row {
  display: flex;
  gap: 6px;
}

.font-select {
  flex: 2;
  background-color: #1A1A1A;
  color: #E8E8E8;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  padding: 4px;
  font-size: 12px;
}

.color-btn {
  flex: 1;
  height: 35px;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  font-size: 13px;
  font-weight: bold;
  cursor: pointer;
  color: #000;
}

.preset-row {
  display: flex;
  gap: 3px;
}

.preset-pos-btn {
  flex: 1;
  height: 26px;
  background-color: #222;
  color: #E8E8E8;
  border: 1px solid #2A2A2A;
  border-radius: 3px;
  font-size: 11px;
  cursor: pointer;
  padding: 3px 6px;
}
.preset-pos-btn:hover {
  background-color: #2A2A2A;
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.slider-label {
  color: #B0B0B0;
  font-size: 12px;
  min-width: 70px;
  white-space: nowrap;
}

.slider {
  flex: 1;
  accent-color: #E2B340;
  height: 4px;
  background: #2A2A2A;
  border-radius: 2px;
}

.slider-value {
  color: #E8E8E8;
  font-size: 12px;
  min-width: 30px;
  text-align: right;
}

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #B0B0B0;
  font-size: 12px;
  cursor: pointer;
}
.checkbox-row.bold {
  font-weight: bold;
}
.checkbox-row input[type="checkbox"] {
  accent-color: #E2B340;
}

.file-btn {
  height: 35px;
  background-color: #1A1A1A;
  color: #E8E8E8;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  font-size: 13px;
  font-weight: bold;
  cursor: pointer;
  width: 100%;
}
.file-btn:hover {
  background-color: #222;
}

.file-label {
  color: #585858;
  font-size: 11px;
  min-height: 20px;
}

.accent-btn {
  height: 35px;
  background-color: #E2B340;
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 13px;
  font-weight: bold;
  cursor: pointer;
  width: 100%;
}
.accent-btn:hover {
  background-color: #c9a038;
}
</style>
