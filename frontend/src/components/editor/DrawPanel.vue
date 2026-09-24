<template>
  <div class="draw-panel">
    <!-- 도구 목록은 캔버스 옆 세로 툴바가 맡는다. 여기 두면 전폭 버튼 10개가
         세로 360px 를 먹고, 도구를 고르려고 이 탭까지 와야 했다. -->
    <div class="dp-head">
      <span class="dp-title">{{ currentToolLabel }}</span>
      <span class="dp-key">{{ currentToolKey }}</span>
    </div>

    <!-- Color Palette -->
    <div class="section-subheader">색상</div>

    <div class="palette-grid">
      <button
        v-for="color in paletteColors"
        :key="color"
        class="palette-btn"
        :style="{ backgroundColor: color }"
        @click="onPaletteClick(color)"
      />
    </div>

    <div class="color-row">
      <div
        class="color-preview"
        :style="{ backgroundColor: currentColor }"
      />
      <button class="secondary-btn" @click="$emit('pick-custom-color')">+ 색상 선택</button>
    </div>

    <!-- Gradient End Color (visible only for gradient tool) -->
    <div v-if="currentToolName === 'gradient'" class="gradient-row">
      <span class="small-label">끝 색상:</span>
      <div
        class="color-preview small"
        :style="{ backgroundColor: gradientEndColor }"
      />
      <button class="small-btn" @click="$emit('pick-gradient-end-color')">선택</button>
    </div>

    <!-- Heal Apply Button (visible only for heal tool) -->
    <button
      v-if="currentToolName === 'heal'"
      class="heal-apply-btn"
      @click="$emit('heal-apply')"
    >
      복원 적용
    </button>

    <div class="divider" />

    <!-- Size / Opacity Sliders -->
    <div class="slider-group">
      <label class="slider-label">크기</label>
      <input type="range" :min="1" :max="100" v-model.number="brushSize" class="slider" />
      <span class="slider-value">{{ brushSize }}</span>
    </div>

    <div class="slider-group">
      <label class="slider-label">투명도</label>
      <input type="range" :min="1" :max="100" v-model.number="brushOpacity" class="slider" />
      <span class="slider-value">{{ brushOpacity }}</span>
    </div>

    <!-- Fill Toggle -->
    <button
      class="fill-btn"
      :class="{ active: isFilled }"
      @click="isFilled = !isFilled"
    >
      <Icon name="square" /> {{ isFilled ? '채우기 ON' : '채우기 OFF' }}
    </button>

    <!-- 레이어 조작은 '지금 그리는 설정' 이 아니라 다 그린 뒤에 하는 일이다.
         펼쳐 두면 도구 옵션이 270px 를 먹어 아래 탭 내용이 눌린다 — 기본 접힘. -->
    <PanelSection title="레이어" storage-key="drawLayer" :default-open="false">
      <div class="slider-group">
        <label class="slider-label">투명도</label>
        <input type="range" :min="0" :max="100" v-model.number="layerOpacity" class="slider" />
        <span class="slider-value">{{ layerOpacity }}</span>
      </div>
      <!-- 레이어는 병합 전까지 원본을 건드리지 않는다. 그래서 되돌리기·지우기가
           이미지 undo 스택과 별개로 필요하다 — 없으면 획 하나 잘못 그었을 때
           이미지를 다시 여는 것 말고는 방법이 없다. -->
      <div class="layer-btn-row">
        <button class="secondary-btn flex-1" @click="$emit('undo-stroke')"><Icon name="undo" /> 획 되돌리기</button>
        <button class="secondary-btn flex-1" @click="$emit('clear-layer')">레이어 지우기</button>
      </div>
      <button class="secondary-btn full-width" @click="$emit('flatten-layer')">레이어 병합</button>
    </PanelSection>
  </div>
</template>

<script setup lang="ts">
/**
 * 그리기 도구 옵션 — **표시만** 한다(제어 컴포넌트).
 *
 * 예전에는 도구·색·크기·투명도를 이 패널이 로컬 ref 로 들고 있었다. 마스크 도구에서
 * 그리기 도구로 넘어갈 때마다 v-if 로 새로 마운트되는데, 부모가 같은 tick 에 부른
 * `setTool` 은 ref 가 아직 null 이라 무시됐다. 그래서 J(복원)·D(그라디언트)로 들어가도
 * 패널은 '펜'으로 떠서 '복원 적용' 버튼과 끝 색 행이 숨었고, 색을 누르면 tool:'pen' 이
 * 실려 가 도구가 펜으로 바뀌었다. 표시(#000000/3)와 실제 값(#ffffff/10)도 달랐다.
 *
 * 이제 값은 전부 부모(EditorView)가 props 로 내려주고, 패널은 바뀐 값만 올려 보낸다.
 * 도구는 올려 보내지 않는다 — 도구의 주인은 세로 툴바다.
 */
import { computed } from 'vue'
import { toolById } from '../../utils/editorTools'
import PanelSection from './PanelSection.vue'

interface ToolParams {
  color: string
  size: number
  /** 0~1 */
  opacity: number
  filled: boolean
}

const emit = defineEmits<{
  'params-changed': [params: Partial<ToolParams>]
  'pick-custom-color': []
  'pick-gradient-end-color': []
  'heal-apply': []
  'flatten-layer': []
  'undo-stroke': []
  'clear-layer': []
  'layer-opacity-changed': [value: number]
}>()

const props = withDefaults(defineProps<{
  /** 현재 그리기 도구 id (세로 툴바가 고른 것) */
  tool?: string
  /** 현재 그리기 파라미터 — 부모가 단일 출처다 */
  params?: ToolParams
  /** 드로잉 레이어 표시 투명도 0~100 */
  layerOpacity?: number
  gradientEndColor?: string
}>(), {
  tool: 'pen',
  params: () => ({ color: '#ffffff', size: 10, opacity: 1, filled: false }),
  layerOpacity: 100,
  gradientEndColor: '#000000',
})

/** 그리기 팔레트 — 이미지에 실제로 칠해지는 사용자 콘텐츠 색이라 테마 토큰이 아니다.
 *  테마가 바뀌어도 빨강은 빨강이어야 한다. */
const paletteColors: string[] = [
  '#000000', '#FFFFFF', '#FF0000', '#00FF00',
  '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
  '#FF8800', '#8800FF', '#888888', '#FF4488',
]

const currentToolName = computed(() => props.tool || 'pen')
const currentToolLabel = computed(() => toolById(currentToolName.value)?.label ?? '그리기')
const currentToolKey = computed(() => toolById(currentToolName.value)?.shortcut ?? '')

const currentColor = computed(() => props.params.color)

// 슬라이더·토글은 부모 값을 보여 주고, 바뀐 필드만 올려 보낸다.
const brushSize = computed({
  get: () => props.params.size,
  set: (size: number) => emit('params-changed', { size }),
})
/** 화면은 1~100, 부모는 0~1 */
const brushOpacity = computed({
  get: () => Math.round((props.params.opacity ?? 1) * 100),
  set: (value: number) => emit('params-changed', { opacity: value / 100 }),
})
const isFilled = computed({
  get: () => !!props.params.filled,
  set: (filled: boolean) => emit('params-changed', { filled }),
})
const layerOpacity = computed({
  get: () => props.layerOpacity,
  set: (value: number) => emit('layer-opacity-changed', value),
})

function onPaletteClick(color: string) {
  emit('params-changed', { color })
}
</script>

<style scoped>
.draw-panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 8px;
  color: var(--text-primary);
  font-size: 13px;
}

.dp-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--sp-2);
  padding: 2px 0 var(--sp-1);
}
.dp-title {
  color: var(--text-primary);
  font-size: var(--fs-body);
  font-weight: var(--fw-medium);
}
.dp-key {
  min-width: 18px;
  padding: 1px 5px;
  background: var(--bg-button);
  border: 1px solid var(--rule);
  border-radius: 3px;
  color: var(--text-muted);
  font-size: var(--fs-label);
  text-align: center;
}

.section-subheader {
  color: var(--text-muted);
  font-size: 14px;
  font-weight: var(--fw-bold);
  padding: 2px;
}

.tool-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tool-btn {
  background-color: var(--bg-button);
  border: 1px solid var(--rule);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: var(--fw-bold);
  text-align: left;
  padding: 8px 12px;
  cursor: pointer;
  height: 36px;
  display: flex;
  align-items: center;
}
.tool-btn:hover {
  border-color: var(--edge);
  background-color: var(--bg-button-hover);
}
.tool-btn.active {
  background-color: var(--accent-fill);
  color: var(--on-accent);
  border-color: var(--accent-fill);
}

.divider {
  height: 1px;
  background-color: var(--rule);
  margin: 4px 0;
}

.palette-grid {
  display: grid;
  grid-template-columns: repeat(6, 32px);
  gap: 4px;
}

.palette-btn {
  width: 32px;
  height: 32px;
  border: 2px solid var(--rule);
  border-radius: 4px;
  cursor: pointer;
  padding: 0;
}
.palette-btn:hover {
  border-color: var(--edge);
}

.color-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.color-preview {
  width: 36px;
  height: 36px;
  border: 2px solid var(--edge);
  border-radius: 4px;
  flex-shrink: 0;
}
.color-preview.small {
  width: 32px;
  height: 32px;
}

.gradient-row {
  display: flex;
  align-items: center;
  gap: 6px;
}

.small-label {
  color: var(--text-muted);
  font-size: 12px;
}

.small-btn {
  height: 32px;
  background: var(--bg-button);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  padding: 0 8px;
}

.secondary-btn {
  flex: 1;
  height: 36px;
  background-color: var(--bg-button);
  color: var(--text-primary);
  border: 1px solid var(--rule);
  border-radius: 4px;
  font-size: 12px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
.secondary-btn:hover {
  background-color: var(--bg-button-hover);
}
.secondary-btn.full-width {
  width: 100%;
}

.layer-btn-row {
  display: flex;
  gap: 6px;
}
.layer-btn-row .flex-1 {
  flex: 1;
  min-width: 0;
}

.heal-apply-btn {
  height: 36px;
  background-color: var(--state-ok);
  /* 상태 채움색은 세 프리셋 모두 흰 글자 기준(4.5:1)으로 잡은 값이라 흰색을 유지한다 */
  color: #FFFFFF;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
/* 상태색에는 hover 파생 토큰이 없다 — 사용자가 state-ok 를 바꿔도 따라오게 밝기로 민다 */
.heal-apply-btn:hover {
  filter: brightness(1.15);
}

.slider-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.slider-label {
  color: var(--text-secondary);
  font-size: 12px;
  min-width: 80px;
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

.fill-btn {
  height: 34px;
  background-color: var(--bg-button);
  color: var(--text-secondary);
  border: 1px solid var(--rule);
  border-radius: 6px;
  font-size: 13px;
  font-weight: var(--fw-bold);
  cursor: pointer;
}
.fill-btn:hover {
  border-color: var(--edge);
}
.fill-btn.active {
  background-color: var(--accent-fill);
  color: var(--on-accent);
  border-color: var(--accent-fill);
}
</style>
