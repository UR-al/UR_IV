<template>
  <div class="editor-panel tool-options">
    <div class="to-head">
      <span class="to-title">{{ toolLabel }}</span>
      <span class="to-key">{{ toolKey }}</span>
    </div>

    <div class="slider-box">
      <div class="slider-header"><span>크기</span><span>{{ toolSize }}px</span></div>
      <input type="range" min="1" max="300" v-model.number="toolSize" class="modern-slider" />
    </div>

    <!-- 올가미 -->
    <div class="control-group" v-if="tool === 'lasso'">
      <label>올가미 방식</label>
      <div class="chip-grid-2">
        <button class="chip-btn" :class="{ active: !magnetic }"
          @click="emit('magnetic-changed', false)"><Icon name="loop" /> 자유</button>
        <button class="chip-btn magnet" :class="{ active: magnetic }"
          @click="emit('magnetic-changed', true)"><Icon name="magnet" /> 자석</button>
      </div>
    </div>

    <!-- 스탬프 -->
    <template v-if="tool === 'stamp'">
      <div class="control-group">
        <label>도장 모양</label>
        <div class="chip-grid-3">
          <button class="chip-btn" :class="{ active: stampShape === 'circle' }"
            @click="emitParams({ stampShape: 'circle' })"><Icon name="circle" /> 원</button>
          <button class="chip-btn" :class="{ active: stampShape === 'bar' }"
            @click="emitParams({ stampShape: 'bar' })"><Icon name="bar" /> 띠</button>
          <button class="chip-btn" :class="{ active: stampShape === 'rect' }"
            @click="emitParams({ stampShape: 'rect' })"><Icon name="square" /> 사각</button>
        </div>
      </div>
      <div class="slider-box">
        <div class="slider-header"><span>간격</span><span>{{ stampSpacing }}px</span></div>
        <input type="range" min="5" max="200" v-model.number="stampSpacing" class="modern-slider" />
      </div>
      <template v-if="stampShape === 'bar'">
        <div class="slider-box">
          <div class="slider-header"><span>띠 너비</span><span>{{ barW }}px</span></div>
          <input type="range" min="5" max="200" v-model.number="barW" class="modern-slider" />
        </div>
        <div class="slider-box">
          <div class="slider-header"><span>띠 높이</span><span>{{ barH }}px</span></div>
          <input type="range" min="3" max="100" v-model.number="barH" class="modern-slider" />
        </div>
      </template>
    </template>

    <!-- 지우개 -->
    <template v-if="tool === 'eraser'">
      <div class="control-group">
        <label>지우개 종류</label>
        <div class="chip-grid-2">
          <button class="chip-btn" :class="{ active: !eraserRestore }"
            @click="emit('eraser-restore-changed', false)">
            <Icon name="wand" /> 마스크
          </button>
          <button class="chip-btn restore" :class="{ active: eraserRestore }"
            @click="emit('eraser-restore-changed', true)">
            <Icon name="sparkles" /> 모자이크
          </button>
        </div>
      </div>
      <div class="control-group" v-if="!eraserRestore">
        <label>지우는 모양</label>
        <div class="chip-grid-3">
          <button class="chip-btn" :class="{ active: eraserMode === 'brush' }" @click="emit('eraser-mode-changed', 'brush')">브러시</button>
          <button class="chip-btn" :class="{ active: eraserMode === 'box' }" @click="emit('eraser-mode-changed', 'box')">사각</button>
          <button class="chip-btn" :class="{ active: eraserMode === 'lasso' }" @click="emit('eraser-mode-changed', 'lasso')">올가미</button>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
/**
 * 지금 고른 마스크 도구의 옵션.
 *
 * 예전에는 이 설정들이 `MosaicPanel` 안에 효과·검열·변형과 함께 세로로 쌓여 있었다
 * (그 패널이 1077px = 화면의 135%였던 이유 중 하나). 도구를 세로 툴바로 뺐으니
 * 옵션도 도구를 따라오게 패널 맨 위로 올린다 — 고른 도구의 것만 보인다.
 *
 * 그리기 도구의 옵션은 `DrawPanel` 이 맡는다.
 *
 * 값은 전부 부모(EditorView)가 props 로 내려준다 — 이 패널은 표시하고 바뀐 필드만 올린다.
 * 예전에는 로컬 ref 가 주인이라, 그리기 도구에 다녀오며 다시 마운트될 때마다
 * 크기·도장 모양이 기본값(20/원)으로 부모 값을 덮어썼고, 지우개 종류·자석 올가미
 * 표시는 실제 값과 어긋났다.
 */
import { computed } from 'vue'
import { toolById } from '../../utils/editorTools'

interface MaskToolParams {
  toolSize: number; stampSpacing: number; stampShape: string; barW: number; barH: number
}

const props = withDefaults(defineProps<{
  /** 현재 도구 id (box/lasso/brush/eraser/stamp) */
  tool?: string
  toolSize?: number
  stampSpacing?: number
  /** 사용자가 고른 도장 모양 (circle/bar/rect) */
  stampShape?: string
  barW?: number
  barH?: number
  eraserMode?: string
  eraserRestore?: boolean
  magnetic?: boolean
}>(), {
  tool: 'box', toolSize: 20, stampSpacing: 30, stampShape: 'circle', barW: 40, barH: 15,
  eraserMode: 'brush', eraserRestore: false, magnetic: false,
})

const emit = defineEmits<{
  'params-changed': [payload: Partial<MaskToolParams>]
  'eraser-mode-changed': [mode: string]
  'eraser-restore-changed': [val: boolean]
  'magnetic-changed': [val: boolean]
}>()

const toolLabel = computed(() => toolById(props.tool)?.label ?? '도구')
const toolKey = computed(() => toolById(props.tool)?.shortcut ?? '')

function emitParams(patch: Partial<MaskToolParams>) {
  emit('params-changed', patch)
}

/** 슬라이더용 — 부모 값을 보여 주고 바뀐 필드만 올린다 */
function field<K extends keyof MaskToolParams>(key: K) {
  return computed({
    get: () => props[key] as MaskToolParams[K],
    set: (value: MaskToolParams[K]) => emitParams({ [key]: value } as Partial<MaskToolParams>),
  })
}
const toolSize = field('toolSize')
const stampSpacing = field('stampSpacing')
const barW = field('barW')
const barH = field('barH')
</script>

<style scoped>
.tool-options {
  gap: var(--sp-2);
  padding: var(--sp-2) var(--sp-3) var(--sp-3);
  border-bottom: 1px solid var(--rule);
}

.to-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--sp-2);
}

.to-title {
  color: var(--text-primary);
  font-size: var(--fs-body);
  font-weight: var(--fw-medium);
}

.to-key {
  min-width: 18px;
  padding: 1px 5px;
  background: var(--bg-button);
  border: 1px solid var(--rule);
  border-radius: 3px;
  color: var(--text-muted);
  font-size: var(--fs-label);
  text-align: center;
}
</style>
