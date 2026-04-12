<template>
  <div class="move-panel">
    <!-- Header -->
    <div class="section-header">영역 이동</div>

    <!-- Status -->
    <div class="status-label">{{ statusText }}</div>

    <div class="divider" />

    <!-- Fill Color -->
    <div class="small-header">구멍 채우기 색</div>
    <CustomSelect v-model="fillColor" :options="['black', 'white']" placeholder="색상" />

    <!-- Rotation / Scale Sliders -->
    <div class="slider-group">
      <label class="slider-label">회전 (°)</label>
      <input type="range" :min="-180" :max="180" v-model.number="rotation" class="slider" />
      <span class="slider-value">{{ rotation }}</span>
    </div>

    <div class="slider-group">
      <label class="slider-label">크기 (%)</label>
      <input type="range" :min="10" :max="500" v-model.number="scale" class="slider" />
      <span class="slider-value">{{ scale }}</span>
    </div>

    <!-- Undo Move -->
    <button
      class="action-btn"
      :disabled="!canUndo"
      @click="$emit('undo-move')"
    >
      이동 되돌리기
    </button>

    <!-- Start Move -->
    <button
      class="accent-btn"
      :disabled="isMoving"
      @click="onStartMove"
    >
      이동 시작
    </button>

    <!-- Confirm / Cancel -->
    <div class="btn-row">
      <button
        class="action-btn"
        :disabled="!isMoving"
        @click="onConfirm"
      >
        확정
      </button>
      <button
        class="action-btn"
        :disabled="!isMoving"
        @click="onCancel"
      >
        취소
      </button>
    </div>

    <div class="divider" />

    <!-- Inpaint -->
    <button
      class="inpaint-btn"
      :disabled="!canInpaint"
      @click="onSendInpaint"
    >
      인페인트
    </button>

    <!-- Prompt -->
    <div class="small-header">Prompt</div>
    <textarea
      v-model="prompt"
      class="prompt-input"
      placeholder="인페인트할 내용 (비우면 메인 프롬프트 사용)"
      rows="2"
    />

    <div class="small-header">Negative Prompt</div>
    <textarea
      v-model="negPrompt"
      class="prompt-input small"
      placeholder="네거티브 (비우면 메인 네거티브 사용)"
      rows="1"
    />
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import CustomSelect from '../CustomSelect.vue'

const emit = defineEmits([
  'start-move',
  'confirm-move',
  'cancel-move',
  'undo-move',
  'send-inpaint',
  'rotation-changed',
  'scale-changed',
])

const props = defineProps({
  statusText: { type: String, default: '마스킹을 먼저 해주세요' },
  canUndo: { type: Boolean, default: false },
  canInpaint: { type: Boolean, default: false },
})

const isMoving = ref(false)
const fillColor = ref('black')
const rotation = ref(0)
const scale = ref(100)
const prompt = ref('')
const negPrompt = ref('')

watch(rotation, (val) => {
  emit('rotation-changed', val)
})

watch(scale, (val) => {
  emit('scale-changed', val)
})

function onStartMove() {
  isMoving.value = true
  emit('start-move', {
    fillColor: fillColor.value,
    rotation: rotation.value,
    scale: scale.value,
  })
}

function onConfirm() {
  isMoving.value = false
  emit('confirm-move', {
    rotation: rotation.value,
    scale: scale.value,
  })
}

function onCancel() {
  isMoving.value = false
  emit('cancel-move')
}

function onSendInpaint() {
  emit('send-inpaint', {
    prompt: prompt.value,
    negPrompt: negPrompt.value,
  })
}

/** Called by parent to update moving state */
function setMovingState(moving) {
  isMoving.value = moving
}

defineExpose({ setMovingState })
</script>

<style scoped>
.move-panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  color: #E8E8E8;
  font-size: 13px;
}

.section-header {
  color: #585858;
  font-size: 18px;
  font-weight: bold;
  padding: 2px;
}

.small-header {
  color: #585858;
  font-size: 12px;
  font-weight: bold;
}

.status-label {
  color: #B0B0B0;
  font-size: 12px;
  padding: 6px;
  background-color: #111;
  border-radius: 4px;
  word-wrap: break-word;
}

.divider {
  height: 1px;
  background-color: #2A2A2A;
  margin: 4px 0;
}

.select-input {
  height: 34px;
  background-color: #1A1A1A;
  color: #B0B0B0;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  padding: 4px;
  font-size: 13px;
  width: 100%;
  box-sizing: border-box;
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.slider-label {
  color: #B0B0B0;
  font-size: 12px;
  min-width: 60px;
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

.btn-row {
  display: flex;
  gap: 6px;
}

.action-btn {
  flex: 1;
  height: 40px;
  background-color: #1A1A1A;
  border: 1px solid #2A2A2A;
  border-radius: 6px;
  color: #B0B0B0;
  font-size: 13px;
  font-weight: bold;
  padding: 8px 12px;
  cursor: pointer;
}
.action-btn:hover:not(:disabled) {
  border-color: #585858;
  background-color: #222;
}
.action-btn:disabled {
  color: #2A2A2A;
  background-color: #111;
  cursor: default;
}

.accent-btn {
  height: 40px;
  background-color: #E2B340;
  border: 1px solid #E2B340;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  font-weight: bold;
  padding: 8px 12px;
  cursor: pointer;
  width: 100%;
}
.accent-btn:hover:not(:disabled) {
  background-color: #c9a038;
}
.accent-btn:disabled {
  background-color: #333;
  color: #585858;
  border-color: #333;
  cursor: default;
}

.inpaint-btn {
  height: 40px;
  background-color: #e67e22;
  border: 1px solid #e67e22;
  border-radius: 6px;
  color: #fff;
  font-size: 13px;
  font-weight: bold;
  padding: 8px 12px;
  cursor: pointer;
  width: 100%;
}
.inpaint-btn:hover:not(:disabled) {
  background-color: #f39c12;
}
.inpaint-btn:disabled {
  background-color: #5C4A2C;
  color: #777;
  cursor: default;
}

.prompt-input {
  background-color: #111;
  color: #B0B0B0;
  border: 1px solid #2A2A2A;
  border-radius: 4px;
  padding: 4px;
  font-size: 12px;
  resize: vertical;
  width: 100%;
  box-sizing: border-box;
  font-family: inherit;
}
.prompt-input::placeholder {
  color: #585858;
}
.prompt-input.small {
  min-height: 35px;
}
</style>
