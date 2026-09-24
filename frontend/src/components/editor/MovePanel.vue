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

    <!-- Start Move — 옮길 영역이 없으면 눌러도 아무 일이 없다. 막고 이유를 적는다. -->
    <button
      class="accent-btn"
      :disabled="isMoving || !hasSelection"
      :title="hasSelection ? '선택한 영역을 잘라 옮깁니다' : '먼저 캔버스에서 옮길 영역을 선택하세요'"
      @click="onStartMove"
    >
      이동 시작
    </button>
    <p v-if="!hasSelection && !isMoving" class="need-hint">
      선택 도구(M · L · B)로 옮길 영역을 먼저 지정하세요
    </p>

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

    <!-- Inpaint — 이미지를 Inpaint 탭으로 넘긴다. 프롬프트는 그 탭에서 쓴다.
         (여기 있던 Prompt/Negative 입력칸은 어디로도 전달되지 않아 지웠다) -->
    <button
      class="inpaint-btn"
      :disabled="!canInpaint"
      @click="$emit('send-inpaint')"
    >
      인페인트
    </button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import CustomSelect from '../CustomSelect.vue'

interface MoveStartPayload {
  fillColor: string
  rotation: number
  scale: number
}

/** 회전·크기는 확정할 때 이 페이로드로만 넘긴다 (슬라이더마다 따로 올리던 emit 은 받는 쪽이 안 썼다) */
interface MoveConfirmPayload {
  rotation: number
  scale: number
}

const emit = defineEmits<{
  'start-move': [payload: MoveStartPayload]
  'confirm-move': [payload: MoveConfirmPayload]
  'cancel-move': []
  'undo-move': []
  'send-inpaint': []
}>()

const props = withDefaults(defineProps<{
  statusText?: string
  canUndo?: boolean
  canInpaint?: boolean
  /** 캔버스에 옮길 영역이 잡혀 있는지 */
  hasSelection?: boolean
}>(), {
  statusText: '마스킹을 먼저 해주세요',
  canUndo: false,
  canInpaint: false,
  hasSelection: false,
})

const isMoving = ref(false)
const fillColor = ref('black')
const rotation = ref(0)
const scale = ref(100)

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

/** Called by parent to update moving state */
function setMovingState(moving: boolean) {
  isMoving.value = moving
}

defineExpose({ setMovingState })
</script>

<style scoped>
/* 비활성 버튼만 두면 "고장났나" 로 읽힌다 — 무엇을 먼저 해야 하는지 붙여 준다 */
.need-hint {
  margin: 0;
  color: var(--text-muted);
  font-size: var(--fs-label);
  line-height: 1.45;
}
button:disabled { opacity: 0.45; cursor: not-allowed; }

.move-panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  color: var(--text-primary);
  font-size: 13px;
}

.section-header {
  color: var(--text-muted);
  font-size: 18px;
  font-weight: var(--fw-bold);
  padding: 2px;
}

.small-header {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: var(--fw-bold);
}

.status-label {
  color: var(--text-secondary);
  font-size: 12px;
  padding: 6px;
  background-color: var(--bg-secondary);
  border-radius: 4px;
  word-wrap: break-word;
}

.divider {
  height: 1px;
  background-color: var(--rule);
  margin: 4px 0;
}

.select-input {
  height: 34px;
  background-color: var(--bg-input);
  color: var(--text-secondary);
  border: 1px solid var(--rule);
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
  color: var(--text-secondary);
  font-size: 12px;
  min-width: 60px;
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

.action-btn {
  flex: 1;
  height: 40px;
  background-color: var(--bg-button);
  border: 1px solid var(--rule);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: var(--fw-bold);
  padding: 8px 12px;
  cursor: pointer;
}
.action-btn:hover:not(:disabled) {
  border-color: var(--edge);
  background-color: var(--bg-button-hover);
}
.action-btn:disabled {
  color: var(--text-muted);
  background-color: var(--bg-secondary);
  cursor: default;
}

.accent-btn {
  height: 40px;
  background-color: var(--accent-fill);
  border: 1px solid var(--accent-fill);
  border-radius: 6px;
  color: var(--on-accent);
  font-size: 13px;
  font-weight: var(--fw-bold);
  padding: 8px 12px;
  cursor: pointer;
  width: 100%;
}
.accent-btn:hover:not(:disabled) {
  background-color: var(--accent-fill-hover);
}
.accent-btn:disabled {
  background-color: var(--bg-button);
  color: var(--text-muted);
  border-color: var(--border);
  cursor: default;
}

.inpaint-btn {
  height: 40px;
  background-color: var(--state-warn);
  border: 1px solid var(--state-warn);
  border-radius: 6px;
  /* 상태 채움색은 세 프리셋 모두 흰 글자 기준으로 잡은 값이라 흰색을 유지한다 */
  color: #FFFFFF;
  font-size: 13px;
  font-weight: var(--fw-bold);
  padding: 8px 12px;
  cursor: pointer;
  width: 100%;
}
/* 상태색에는 hover 파생 토큰이 없다 — 사용자가 색을 바꿔도 따라오게 밝기로 민다 */
.inpaint-btn:hover:not(:disabled) {
  filter: brightness(1.15);
}
/* 상태색에는 dim 파생 토큰이 없다 — 색은 활성과 같게 두고 흐림은 위의
   button:disabled(opacity .45)가 준다 */
.inpaint-btn:disabled {
  background-color: var(--state-warn);
  color: var(--text-muted);
  cursor: default;
}

</style>
