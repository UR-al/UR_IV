<template>
  <div class="editor-view" @dragover.prevent="isDragging = true" @dragleave="isDragging = false" @drop.prevent="onDrop">
    <template v-if="imagePath">
      <!-- 상단 도구바 (개선: 넓고 깔끔) -->
      <div class="top-bar">
        <div class="bar-group">
          <button class="bar-btn accent" @click="openFile">📂 파일 열기</button>
          <button class="bar-btn save" @click="saveImage">💾 저장</button>
        </div>
        <div class="bar-group center">
          <button class="bar-btn" @click="doUndo" :disabled="undoStack.length <= 1">↩ Undo</button>
          <button class="bar-btn" @click="doRedo" :disabled="redoStack.length === 0">↪ Redo</button>
          <span class="bar-sep">|</span>
          <span class="bar-info">{{ imgWidth }}×{{ imgHeight }}</span>
        </div>
        <div class="bar-group">
          <button class="bar-btn danger" @click="resetEditor">✕ 닫기</button>
        </div>
      </div>

      <div class="editor-body">
        <!-- 좌측: 서브탭 패널 -->
        <div class="side-panel">
          <div class="tab-buttons">
            <button v-for="(tab, i) in tabs" :key="i"
              class="tab-btn" :class="{ active: activeTab === i }"
              @click="activeTab = i"
            >{{ tab.icon }} {{ tab.label }}</button>
          </div>
          <div class="tab-content">
            <MosaicPanel v-show="activeTab === 0"
              :model-label="modelLabel"
              :detect-status="detectStatus"
              @tool-changed="onToolChanged"
              @effect-apply="applyEffect"
              @add-model="openModelDialog"
              @clear-models="clearModels"
              @auto-censor="runAutoCensor"
              @auto-detect="runAutoDetect"
              @cancel-selection="canvasRef?.clearSelection()"
              @crop="doCrop"
              @resize="doResize"
              @perspective="doOp('perspective')"
              @rotate="op => doOp('rotate_' + op)"
              @flip="op => doOp('flip_' + (op === 'horizontal' ? 'h' : 'v'))"
              @remove-bg="params => doOp('remove_bg', params)"
              @params-changed="onParamsChanged"
              @eraser-mode-changed="m => eraserMode = m"
              @eraser-restore-changed="v => eraserRestore = v"
              @magnetic-changed="onMagneticChanged"
            />
            <ColorPanel v-show="activeTab === 1"
              @adjustment-changed="previewAdj" @apply="applyAdj"
              @reset="resetAdj" @filter-apply="applyFilter"
              @auto-correct="doOp('auto_correct')"
            />
            <AdvancedColorPanel v-show="activeTab === 2"
              @preview="previewAdvAdj" @apply="applyAdvAdj" @reset="resetAdj"
            />
            <WatermarkPanel v-show="activeTab === 3"
              @apply-text="applyTextWm" @apply-image="applyImageWm"
              @load-watermark-image="loadWatermarkImage"
            />
            <DrawPanel v-show="activeTab === 4" ref="drawPanelRef"
              @tool-changed="currentTool = $event"
              @flatten="doOp('flatten')"
            />
            <MovePanel v-show="activeTab === 5"
              @start-move="doOp('start_move')"
              @confirm="doOp('confirm_move')" @cancel="doOp('cancel_move')"
            />
          </div>
        </div>

        <!-- 중앙: 캔버스 -->
        <EditorCanvas ref="canvasRef"
          :image-src="imageDisplay"
          :tool="currentTool"
          :brush-size="brushSize"
          :eraser-mode="eraserMode"
          :eraser-restore="eraserRestore"
          :magnetic-lasso="magneticLasso"
          :stamp-spacing="stampSpacing"
          :stamp-shape="stampShape"
          :bar-width="barWidth"
          :bar-height="barHeight"
          @selection-changed="onSelectionChanged"
        />
      </div>
    </template>

    <template v-else>
      <div class="drop-area" :class="{ dragging: isDragging }">
        <div class="drop-icon">🎨</div>
        <h2>Image Editor</h2>
        <p>이미지를 드래그앤드롭하거나 파일을 선택하세요</p>
        <button class="open-btn" @click="openFile">파일 선택</button>
        <div class="feature-list">
          <span>모자이크/블러</span><span>색감 조절</span><span>고급 색감</span>
          <span>워터마크</span><span>그리기</span><span>이동/변환</span>
          <span>크롭/리사이즈</span><span>회전/반전</span><span>배경 제거</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { getBackend, onBackendEvent } from '../bridge.js'
import EditorCanvas from '../components/editor/EditorCanvas.vue'
import MosaicPanel from '../components/editor/MosaicPanel.vue'
import ColorPanel from '../components/editor/ColorPanel.vue'
import AdvancedColorPanel from '../components/editor/AdvancedColorPanel.vue'
import WatermarkPanel from '../components/editor/WatermarkPanel.vue'
import DrawPanel from '../components/editor/DrawPanel.vue'
import MovePanel from '../components/editor/MovePanel.vue'

const isDragging = ref(false)
const imagePath = ref('')
const imageDisplay = ref('')
const imgWidth = ref(0)
const imgHeight = ref(0)
const activeTab = ref(0)
const currentTool = ref('box')
const brushSize = ref(20)
const eraserMode = ref('brush')
const eraserRestore = ref(false)
const magneticLasso = ref(false)
const stampSpacing = ref(30)
const stampShape = ref('circle')
const barWidth = ref(40)
const barHeight = ref(15)
const canvasRef = ref(null)
const drawPanelRef = ref(null)
const selection = ref(null)
const modelLabel = ref('No Model Loaded')
const detectStatus = ref('')

const undoStack = ref([])
const redoStack = ref([])

const tabs = [
  { icon: '🔲', label: '모자이크' },
  { icon: '🎨', label: '색감' },
  { icon: '🔧', label: '고급색감' },
  { icon: '💧', label: '워터마크' },
  { icon: '✏️', label: '그리기' },
  { icon: '✂️', label: '이동' },
]

function loadImage(path) {
  if (!path) return
  undoStack.value = [path]
  redoStack.value = []
  imagePath.value = path
  imageDisplay.value = 'file:///' + path + '?t=' + Date.now()
  const img = new Image()
  img.onload = () => { imgWidth.value = img.naturalWidth; imgHeight.value = img.naturalHeight }
  img.src = 'file:///' + path
}

const MAX_UNDO = 5

function pushState(path, clearMask = true) {
  undoStack.value.push(path)
  // undo 5회 제한
  while (undoStack.value.length > MAX_UNDO + 1) undoStack.value.shift()
  redoStack.value = []
  imagePath.value = path
  // 타임스탬프 없이 경로만 변경 → watch에서 zoom/rotation 유지됨
  imageDisplay.value = 'file:///' + path + '?t=' + Date.now()
  const img = new Image()
  img.onload = () => { imgWidth.value = img.naturalWidth; imgHeight.value = img.naturalHeight }
  img.src = 'file:///' + path
  if (clearMask) canvasRef.value?.clearSelection()
}

function doUndo() {
  if (undoStack.value.length <= 1) return
  redoStack.value.push(undoStack.value.pop())
  const path = undoStack.value[undoStack.value.length - 1]
  imagePath.value = path
  imageDisplay.value = 'file:///' + path + '?t=' + Date.now()
}
function doRedo() {
  if (redoStack.value.length === 0) return
  const path = redoStack.value.pop()
  undoStack.value.push(path)
  imagePath.value = path
  imageDisplay.value = 'file:///' + path + '?t=' + Date.now()
}

async function doOp(operation, params = {}) {
  if (!imagePath.value) return
  const backend = await getBackend()
  const cleanPath = imagePath.value.replace('file:///', '')
  backend.editorProcess(cleanPath, operation, JSON.stringify(params), (json) => {
    try {
      const result = JSON.parse(json)
      if (result.path) pushState(result.path)
      else if (result.error) console.error('[Editor] error:', result.error)
    } catch (e) { console.error('[Editor] parse error:', e) }
  })
}

// 마스크 기반 효과 적용 (base64 마스크 전송)
async function doOpWithMask(operation, params = {}) {
  if (!imagePath.value) return
  const maskB64 = canvasRef.value?.getMaskBase64()
  if (!maskB64) {
    // 마스크 없으면 선택 영역(rect)으로 fallback
    doOp(operation, params)
    return
  }
  const backend = await getBackend()
  const cleanPath = imagePath.value.replace('file:///', '')
  const fullParams = { ...params, mask_base64: maskB64 }
  backend.editorProcess(cleanPath, operation, JSON.stringify(fullParams), (json) => {
    try {
      const result = JSON.parse(json)
      if (result.path) pushState(result.path)
      else if (result.error) console.error('[Editor] error:', result.error)
    } catch (e) { console.error('[Editor] parse error:', e) }
  })
}

function onToolChanged(data) {
  const id = typeof data === 'object' ? data.tool : data
  const toolMap = { 0: 'box', 1: 'lasso', 2: 'brush', 3: 'eraser', 4: 'stamp' }
  currentTool.value = toolMap[id] ?? 'box'
  if (typeof data === 'object' && data.size) brushSize.value = data.size
}

async function onMagneticChanged(enabled) {
  magneticLasso.value = enabled
  if (enabled && imagePath.value) {
    // Canny edge map을 Python에서 생성하여 로드
    const backend = await getBackend()
    if (backend.getEdgeMap) {
      const cleanPath = imagePath.value.replace('file:///', '')
      backend.getEdgeMap(cleanPath, 50, 150, (b64) => {
        if (b64) canvasRef.value?.loadEdgeMap(b64)
      })
    }
  }
}

function onParamsChanged(params) {
  if (params.toolSize) brushSize.value = params.toolSize
  if (params.stampSpacing) stampSpacing.value = params.stampSpacing
  if (params.stampShape) stampShape.value = params.stampShape
  if (params.barW) barWidth.value = params.barW
  if (params.barH) barHeight.value = params.barH
}

function applyEffect(effectData) {
  const sel = canvasRef.value?.getSelection()
  const effectMap = { 0: 'mosaic', 1: 'censor_bar', 2: 'blur' }
  const op = effectMap[effectData.effect] ?? 'mosaic'
  if (sel) {
    // 마스크 기반 처리
    doOpWithMask(op, { ...effectData, selection: sel })
  }
}

async function openModelDialog() {
  // 먼저 자동 감지 새로고침
  const backend = await getBackend()
  if (backend.refreshYoloModels) {
    backend.refreshYoloModels((label) => { if (label) modelLabel.value = label })
  }
  // 새 모델 추가도 가능
  requestAction('editor_add_yolo_model')
}
function clearModels() { requestAction('editor_clear_yolo_models') }

async function runAutoCensor(params) {
  if (!imagePath.value) return
  detectStatus.value = '감지 중...'
  const backend = await getBackend()
  const cleanPath = imagePath.value.replace('file:///', '')
  backend.editorProcess(cleanPath, 'auto_censor', JSON.stringify({
    confidence: (params?.confidence || 25) / 100,
    sam_model: params?.samModel || 'auto',
  }), (json) => {
    try {
      const result = JSON.parse(json)
      if (result.path) { pushState(result.path); detectStatus.value = '완료' }
      else { detectStatus.value = result.error || '실패' }
    } catch { detectStatus.value = '오류' }
  })
}

async function runAutoDetect(params) {
  if (!imagePath.value) return
  detectStatus.value = '감지 중...'
  const backend = await getBackend()
  const cleanPath = imagePath.value.replace('file:///', '')
  backend.editorProcess(cleanPath, 'auto_detect', JSON.stringify({
    confidence: (params?.confidence || 25) / 100,
    sam_model: params?.samModel || 'auto',
  }), (json) => {
    try {
      const result = JSON.parse(json)
      if (result.mask_base64) {
        // 마스크를 캔버스에 로드
        canvasRef.value?.loadMaskFromBase64(result.mask_base64)
        detectStatus.value = `${result.detect_count || 0}개 감지됨`
      } else if (result.error) {
        detectStatus.value = result.error
      }
    } catch { detectStatus.value = '오류' }
  })
}

function doCrop() {
  const sel = canvasRef.value?.getSelection()
  if (sel) doOp('crop', { selection: sel })
}
function doResize(params) { doOp('resize', params) }
function applyAdj(adj) { doOp('color_adjust', adj) }
function previewAdj() {}
function resetAdj() {}
function applyFilter(filter) { doOp(filter.name || filter.type, filter) }
function previewAdvAdj() {}
function applyAdvAdj(adj) { doOp('adv_color', adj) }
function applyTextWm(params) { doOp('text_watermark', params) }
function applyImageWm(params) { doOp('image_watermark', params) }
function loadWatermarkImage() { requestAction('editor_load_watermark_image') }
function onSelectionChanged(sel) { selection.value = sel }

function onDrop(e) {
  isDragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file?.path) loadImage(file.path.replace(/\\/g, '/'))
}

function openFile() { requestAction('editor_open_file') }
function saveImage() { requestAction('editor_save', { path: imagePath.value }) }
function resetEditor() {
  imagePath.value = ''; imageDisplay.value = ''; undoStack.value = []; redoStack.value = []
}

// 앱 시작 시 YOLO 라벨 로드
async function refreshYoloLabel() {
  const backend = await getBackend()
  if (backend.getYoloModelLabel) {
    backend.getYoloModelLabel((label) => { if (label) modelLabel.value = label })
  }
}

onMounted(() => {
  onBackendEvent('editorImageLoaded', (path) => loadImage(path))
  onBackendEvent('yoloModelUpdated', (label) => { modelLabel.value = label })
  // 앱 시작 시 YOLO 모델 자동 감지
  refreshYoloLabel()

  document.addEventListener('keydown', (e) => {
    if (!imagePath.value) return
    if (e.ctrlKey && e.shiftKey && e.key === 'z') { e.preventDefault(); canvasRef.value?.redoMask() }
    else if (e.ctrlKey && e.key === 'z') { e.preventDefault(); canvasRef.value?.undoMask() || doUndo() }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); canvasRef.value?.redoMask() || doRedo() }
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveImage() }
    if (e.key === 'Escape') canvasRef.value?.clearSelection()
  })
})
</script>

<style scoped>
.editor-view { width: 100%; height: 100%; display: flex; flex-direction: column; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; background: #0D0D0D; flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
.bar-group { display: flex; align-items: center; gap: 6px; }
.bar-group.center { flex: 1; justify-content: center; }
.bar-btn {
  padding: 8px 16px; background: #181818; border: 1px solid var(--border); border-radius: 6px;
  color: #909090; font-size: 12px; font-weight: 600; cursor: pointer; white-space: nowrap;
  transition: var(--transition);
}
.bar-btn:hover { background: #222; color: #E8E8E8; border-color: #333; }
.bar-btn:disabled { opacity: 0.3; }
.bar-btn.accent { border-color: var(--accent-dim); color: var(--accent); }
.bar-btn.save { background: var(--accent); color: #000; border: none; font-weight: 800; }
.bar-btn.save:hover { background: var(--accent-hover); }
.bar-btn.danger { color: #f87171; border-color: rgba(248,113,113,0.2); }
.bar-sep { color: #333; margin: 0 4px; }
.bar-info { color: #585858; font-size: 11px; font-family: 'Consolas', monospace; }

.editor-body { flex: 1; display: flex; overflow: hidden; }

.side-panel {
  width: 280px; flex-shrink: 0; background: #0D0D0D;
  display: flex; flex-direction: column; overflow: hidden;
}
.tab-buttons { display: flex; flex-wrap: wrap; gap: 2px; padding: 4px; background: #0A0A0A; flex-shrink: 0; }
.tab-btn {
  padding: 5px 8px; background: #131313; border: none; border-radius: 4px;
  color: #585858; font-size: 10px; cursor: pointer; white-space: nowrap;
}
.tab-btn:hover { background: #1A1A1A; color: #E8E8E8; }
.tab-btn.active { background: #1A1A1A; color: #E2B340; }
.tab-content { flex: 1; overflow-y: auto; overflow-x: hidden; }

.drop-area {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px;
}
.drop-area.dragging { background: #111; }
.drop-icon { font-size: 48px; opacity: 0.3; }
.drop-area h2 { color: #787878; font-size: 20px; }
.drop-area p { color: #484848; font-size: 13px; }
.open-btn {
  padding: 10px 24px; background: #E2B340; border: none; border-radius: 8px;
  color: #000; font-weight: 700; font-size: 14px; cursor: pointer;
}
.feature-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; justify-content: center; }
.feature-list span { padding: 5px 12px; background: #131313; border-radius: 6px; color: #585858; font-size: 11px; }
</style>
