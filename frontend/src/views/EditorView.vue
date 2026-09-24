<template>
  <div class="editor-view" @dragover.prevent="isDragging = true" @dragleave="isDragging = false" @drop.prevent="onDrop">
    <template v-if="imagePath">
      <!-- 상단 도구바 -->
      <div class="top-bar">
        <div class="bar-group">
          <button class="bar-btn accent" v-host-dialog="'editor_open_file'" @click="openFile" title="Ctrl+O"><Icon name="folder-open" /> 열기</button>
          <button class="bar-btn save" @click="saveImage" title="Ctrl+S"><Icon name="save" /> 저장</button>
          <button class="bar-btn" v-host-dialog="'editor_save_as'" @click="saveAsImage" title="Ctrl+Shift+S"><Icon name="save" /> 다른 이름</button>
          <button class="bar-btn" @click="pasteFromClipboard" title="Ctrl+V"><Icon name="clipboard" /> 붙여넣기</button>
        </div>
        <div class="bar-group center">
          <button class="bar-btn" @click="onUndo" :disabled="undoStack.length <= 1 && !canvasRef?.maskUndoCount" title="Ctrl+Z (마스킹 우선)">
            <Icon name="undo" /> Undo <span class="bar-counter">({{ Math.max(0, undoStack.length - 1) }}/{{ MAX_UNDO }})</span>
          </button>
          <button class="bar-btn" @click="onRedo" :disabled="redoStack.length === 0 && !canvasRef?.maskRedoCount" title="Ctrl+Y (마스킹 우선)">
            <Icon name="redo" /> Redo <span class="bar-counter">({{ redoStack.length }})</span>
          </button>
          <span class="bar-sep">|</span>
          <span class="bar-filename" :title="sourcePath || imagePath">
            <span v-if="isDirty" class="dirty-mark">●</span>{{ baseName }}
          </span>
          <span class="bar-info">{{ imgWidth }}×{{ imgHeight }}{{ fileInfoExtra }}</span>
          <span v-if="autoSaveAgoText" class="bar-info autosave" :title="`마지막 자동저장: ${new Date(lastAutoSaveAt).toLocaleTimeString()}`"><Icon name="save" /> {{ autoSaveAgoText }}
          </span>
        </div>
        <div class="bar-group">
          <button class="bar-btn danger" @click="confirmClose"><Icon name="close" /> 닫기</button>
        </div>
      </div>

      <div class="editor-body">
        <!-- 캔버스 도구 툴바 — 도구를 고르려고 탭을 옮기지 않아도 되게 -->
        <EditorToolbar :model-value="currentTool" @select="selectTool" />

        <!-- 좌측: 패널 — 너비는 localStorage 영속 -->
        <div class="side-panel" :style="{ width: sidePanelWidth + 'px' }">
          <!-- 도구 옵션 — 툴바에서 고른 도구를 따라간다. 탭과 무관하게 항상 위에 있어서
               도구를 바꿔도 설정을 찾아다닐 필요가 없다. -->
          <!-- 두 패널 모두 값은 여기(부모)가 주인이다 — 도구를 오가며 다시 마운트돼도
               표시가 실제 값과 어긋나거나 기본값으로 덮어쓰지 않는다. -->
          <MaskToolOptions v-if="toolKind === 'mask'"
            :tool="currentTool"
            :tool-size="brushSize"
            :stamp-spacing="stampSpacing"
            :stamp-shape="stampShape"
            :bar-w="barWidth"
            :bar-h="barHeight"
            :eraser-mode="eraserMode"
            :eraser-restore="eraserRestore"
            :magnetic="magneticLasso"
            @params-changed="onParamsChanged"
            @eraser-mode-changed="m => eraserMode = m"
            @eraser-restore-changed="v => eraserRestore = v"
            @magnetic-changed="onMagneticChanged"
          />
          <DrawPanel v-else-if="toolKind === 'draw'"
            :tool="currentTool"
            :params="drawParams"
            :layer-opacity="drawLayerOpacity"
            :gradient-end-color="drawGradientEnd"
            @params-changed="onDrawParamsChanged"
            @pick-custom-color="() => pickColor('draw')"
            @pick-gradient-end-color="() => pickColor('gradient')"
            @layer-opacity-changed="v => drawLayerOpacity = v"
            @heal-apply="applyHeal"
            @flatten-layer="applyFlatten"
            @undo-stroke="undoDrawStroke"
            @clear-layer="clearDrawLayer"
          />

          <div class="tab-buttons">
            <button v-for="(tab, i) in tabs" :key="i"
              class="tab-btn" :class="{ active: activeTab === i }"
              @click="switchTab(i)"
            >{{ tab.label }}</button>
          </div>

          <div class="tab-content">
            <!-- 보정 -->
            <div v-show="activeTab === 0" class="editor-panel tab-stack">
              <PanelSection title="밝기 · 대비 · 필터" storage-key="basicColor">
                <ColorPanel
                  @adjustment-changed="previewAdj" @apply="applyAdj"
                  @reset="resetAdj" @filter-apply="applyFilter"
                  @filter-preview="previewFilter" @filter-cancel="clearPreview"
                  @auto-correct="doOp('auto_correct')"
                />
              </PanelSection>
              <PanelSection title="히스토그램 · 레벨 · 커브" storage-key="advColor" :default-open="false">
                <AdvancedColorPanel
                  :src="canvasSrc" :active="activeTab === 0"
                  @preview="previewAdvAdj" @apply="applyAdvAdj" @reset="resetAdj"
                />
              </PanelSection>
            </div>

            <!-- 효과 -->
            <div v-show="activeTab === 1" class="tab-stack">
              <EffectPanel
                :model-label="modelLabel"
                :detect-status="detectStatus"
                :default-strength="defaultEffectStrength"
                :default-detect-conf="defaultDetectConf"
                @effect-apply="applyEffect"
                @effect-preview="previewEffect"
                @cancel-selection="canvasRef?.clearSelection()"
                @add-model="openModelDialog"
                @clear-models="clearModels"
                @auto-censor="runAutoCensor"
                @auto-detect="runAutoDetect"
                @remove-bg="params => doOp('remove_bg', params)"
              />
              <div class="editor-panel">
                <PanelSection title="워터마크" storage-key="watermark" :default-open="false">
                  <WatermarkPanel
                    @apply-text="applyTextWm" @apply-image="applyImageWm"
                    @load-watermark-image="loadWatermarkImage"
                    :text-color="wmTextColor"
                    :image-path="wmImagePath"
                    @preview="previewWatermark" @preview-clear="clearPreview"
                    @pick-text-color="() => pickColor('wmText')"
                    @clamp-changed="v => wmClamp = v"
                  />
                </PanelSection>
              </div>
            </div>

            <!-- 변형 -->
            <div v-show="activeTab === 2" class="tab-stack">
              <TransformPanel
                :img-width="imgWidth" :img-height="imgHeight"
                :crop-pending="cropPending"
                :perspective-active="perspectiveActive"
                :has-selection="hasSelection"
                @crop="doCrop" @crop-confirm="confirmCrop" @crop-cancel="cancelCrop"
                @resize="doResize"
                @perspective-start="onStartPerspective"
                @perspective-confirm="onConfirmPerspective"
                @perspective-cancel="onCancelPerspective"
                @rotate="op => doOp('rotate_' + op)"
                @flip="op => doOp('flip_' + (op === 'horizontal' ? 'h' : 'v'))"
              />
              <div class="editor-panel">
                <PanelSection title="선택 영역 이동" storage-key="move" :default-open="false">
                  <MovePanel ref="movePanelRef"
                    :status-text="moveStatusText"
                    :can-inpaint="canInpaint"
                    :has-selection="hasSelection"
                    @send-inpaint="onSendInpaint"
                    :can-undo="undoStack.length > 1"
                    @start-move="onStartMove"
                    @confirm-move="onConfirmMove"
                    @cancel-move="onCancelMove"
                    @undo-move="onUndo"
                  />
                </PanelSection>
              </div>
            </div>
          </div>
        </div>

        <!-- 중앙: 캔버스 -->
        <!-- 확정 이미지와 프리뷰를 따로 넘긴다. 프리뷰(축소본)를 image-src 로 넘기면
             캔버스가 새 원본으로 받아 마스크·드로잉 레이어·복원 스냅숏을 초기화했다.
             pristine-src: 모자이크 지우개가 칠하는 '적용 전' 그림 = 복원 커밋의 source_path 파일. -->
        <EditorCanvas ref="canvasRef"
          :image-src="imageDisplay"
          :preview-src="previewSrc"
          :pristine-src="pristineDisplay"
          :tool="currentTool"
          :brush-size="brushSize"
          :eraser-mode="eraserMode"
          :eraser-restore="eraserRestore"
          :magnetic-lasso="magneticLasso"
          :snap-radius="snapRadius"
          :stamp-spacing="stampSpacing"
          :stamp-shape="canvasStampShape"
          :bar-width="barWidth"
          :bar-height="barHeight"
          :draw-params="canvasDrawParams"
          :layer-opacity="drawLayerOpacity"
          @selection-changed="onSelectionChanged"
          @restore-ready="commitRestore"
          @color-picked="onEyedropperColor"
        />
      </div>
    </template>

    <template v-else>
      <div class="drop-area" :class="{ dragging: isDragging }">
        <div class="drop-icon"><Icon name="palette" /></div>
        <h2>이미지 편집</h2>
        <p>이미지를 드래그앤드롭하거나 파일을 선택하세요</p>
        <div class="drop-actions">
          <button class="open-btn" v-host-dialog="'editor_open_file'" @click="openFile"><Icon name="folder-open" /> 파일 선택</button>
          <button class="open-btn secondary" @click="pasteFromClipboard"><Icon name="clipboard" /> 클립보드</button>
        </div>
        <div class="drop-shortcuts">
          <kbd>Ctrl+O</kbd> 열기 &nbsp; <kbd>Ctrl+V</kbd> 붙여넣기
        </div>
        <!-- 최근 파일 -->
        <div v-if="recentFiles.length > 0" class="recent-files">
          <div class="recent-label">최근 편집</div>
          <div class="recent-list">
            <button v-for="path in recentFiles" :key="path"
              class="recent-item" :title="path"
              @click="loadImage(path)">
              <span class="recent-name">{{ path.replace(/\\/g, '/').split('/').pop() }}</span>
            </button>
          </div>
        </div>
        <div class="feature-list">
          <span>모자이크/블러</span><span>색감 조절</span><span>고급 색감</span>
          <span>워터마크</span><span>그리기</span><span>이동/변환</span>
          <span>크롭/리사이즈</span><span>회전/반전</span><span>배경 제거</span>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, onActivated, onDeactivated } from 'vue'
import { useTabDefaultsFollower } from '../composables/useTabDefaultsFollower'
import { editorDefaultsFrom, followDefault } from '../utils/tabDefaults'
import { requestAction } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import { getBackend, onBackendEvent } from '../bridge.js'
import { mediaUrl } from '../utils/media.js'
import { stripFileUrl } from '../utils/fileUrl'
import { isIdentity } from '../utils/curves'
import EditorCanvas from '../components/editor/EditorCanvas.vue'
import ColorPanel from '../components/editor/ColorPanel.vue'
import AdvancedColorPanel from '../components/editor/AdvancedColorPanel.vue'
import WatermarkPanel from '../components/editor/WatermarkPanel.vue'
import DrawPanel from '../components/editor/DrawPanel.vue'
import MovePanel from '../components/editor/MovePanel.vue'
import MaskToolOptions from '../components/editor/MaskToolOptions.vue'
import EffectPanel from '../components/editor/EffectPanel.vue'
import TransformPanel from '../components/editor/TransformPanel.vue'
import PanelSection from '../components/editor/PanelSection.vue'
import EditorToolbar from '../components/editor/EditorToolbar.vue'
import { toolById, toolByKey } from '../utils/editorTools'
import {
  ImageSizeCache, aliasesFromSaveResult, editorResultForDoc, hasPendingSaveFor, initialDocGen,
  isEditorDirty, parseSaveResult, referencesAutosaveFile, replacePath, saveResultAction, saveToastMessage,
  writtenPathFromSaveResult, type PendingSave, type SavedMarker,
} from '../utils/editorDocument'
import { bumpMediaVersion } from '../utils/mediaVersions'
import { PREVIEW_MAX_EDGE, PreviewGate } from '../utils/editorPreview'
import { acceptsTextPaste, pasteClipboardImage } from '../utils/clipboardImage'
import { EdgeMapCache } from '../utils/edgeMap'
import { createLatestRequest, wasAbandoned } from '../utils/bridgeRequest'
import type { EditorAutoSaveReadyPayload } from '../types/bridge'

const isDragging = ref(false)
const imagePath = ref('')
const imageDisplay = ref('')
// 프리뷰가 살아있는 동안 캔버스는 이 base64 를 원본 위에 겹쳐 보여준다(EditorCanvas 프리뷰 레이어).
// 파일도 undo 도, 캔버스의 원본·마스크·드로잉 레이어도 건드리지 않는다.
const previewSrc = ref('')
// 히스토그램·커브는 지금 보이는 것(프리뷰가 있으면 프리뷰)의 분포를 보여준다.
const canvasSrc = computed(() => previewSrc.value || imageDisplay.value)
// 프리뷰 세대 — 걷거나 확정 작업을 보낸 뒤 늦게 도착한 옛 프리뷰를 버린다.
const previewGate = new PreviewGate()
// 확정 이미지가 바뀌면 프리뷰는 무조건 무효다. 결과 핸들러 안에서 인라인으로
// 지우면 도착 순서(늦게 온 프리뷰, keep-alive 로 살아있는 옛 리스너)에 흔들린다.
watch(imageDisplay, () => { previewSrc.value = '' })
const imgWidth = ref(0)
const imgHeight = ref(0)
// 마지막 사용 탭 영속화 (localStorage)
// 탭이 6개였을 때 저장된 값(0~5)이 남아 있을 수 있다 — 범위 밖이면 첫 탭으로
const activeTab = ref(Math.min(2, Math.max(0, parseInt(window.localStorage.getItem('editorActiveTab') || '0') || 0)))
// 파일 정보 (포맷/용량)
const fileSize = ref(0)
const fileFormat = ref('')
// ── 문서(저장) 상태 ──
// sourcePath: 이 문서의 파일. loadImage 로 연 원본이고, 저장에 성공하면 저장된 사본으로 바뀐다.
//   (imagePath 는 편집할 때마다 editor_temp 로 바뀌고, undoStack[0] 은 30회 뒤 밀려난다)
// _sourceOwned: sourcePath 가 이 문서가 저장한 사본인지 — 그때만 다음 저장이 그 파일을
//   덮어쓴다(백엔드도 자기가 쓴 파일인지 다시 확인한다). 연 원본은 절대 덮어쓰지 않는다.
// savedMarker: 마지막으로 디스크와 일치했던 상태 — isDirty 는 이것과 비교해 계산한다.
// _docGen: 문서 세대. 다른 이미지를 열거나 닫으면 오른다 — 저장·편집 중에 문서가 바뀌면
//   늦게 온 저장 결과(saveResultAction)·편집 결과(editorResultForDoc)를 새 문서에 적용하지 않는다.
//   창마다 다른 값에서 시작한다(웹 모드에서 editorResult 는 모든 탭에 방송된다).
const sourcePath = ref('')
let _sourceOwned = false
let _docGen = initialDocGen()
const savedMarker = ref<SavedMarker | null>(null)
// 경로별 이미지 크기 — undo/redo 때 상단바·변형 패널 크기를 되돌린다
const sizeCache = new ImageSizeCache()
// 사이드 패널 너비 (localStorage 영속, 200~500px)
const sidePanelWidth = ref(parseInt(window.localStorage.getItem('editorSidePanelWidth') || '280'))
watch(sidePanelWidth, (v) => {
  window.localStorage.setItem('editorSidePanelWidth', String(v))
})

// editorSidePanelWidth가 Settings에서 바뀐 경우 동기화
function _syncSidePanelWidthFromStorage() {
  const v = parseInt(window.localStorage.getItem('editorSidePanelWidth') || '280')
  if (v !== sidePanelWidth.value) sidePanelWidth.value = v
}
// 현재 도구 id(문자열). 세로 툴바·단축키·원근/이동 모드만 바꾼다 — 옵션 패널은 바꾸지 않는다.
const currentTool = ref<string>('box')
// 마스크 도구 옵션 — MaskToolOptions 는 이 값들을 props 로 표시만 한다(단일 출처)
const brushSize = ref(20)
// Settings 'EDITOR 기본값' — 브러시·효과 세기·YOLO 신뢰도·스냅 반경의 처음 값(audit #140, 예전엔 읽는 곳이 없었다).
// 사용자가 이 화면에서 바꾸지 않은 동안만 기본값을 따른다(followDefault).
const snapRadius = ref(12)
const defaultEffectStrength = ref<number | undefined>(undefined)
const defaultDetectConf = ref<number | undefined>(undefined)
useTabDefaultsFollower((next, prev) => {
  const e = editorDefaultsFrom(next)
  const p = editorDefaultsFrom(prev)
  brushSize.value = followDefault(brushSize.value, p.brushSize, e.brushSize)
  snapRadius.value = followDefault(snapRadius.value, p.snapRadius, e.snapRadius)
  defaultEffectStrength.value = e.effectStrength   // EffectPanel 이 손대지 않은 값만 따른다
  defaultDetectConf.value = e.detectConf
})
const eraserMode = ref('brush')
const eraserRestore = ref(false)
const magneticLasso = ref(false)
const stampSpacing = ref(30)
const stampShape = ref('circle')
const barWidth = ref(40)
const barHeight = ref(15)
// 도장 모양은 스탬프 도구일 때만 뜻이 있다 — 다른 도구에서 'bar' 가 새어 나가면
// 캔버스가 엉뚱한 커서를 그린다. 사용자가 고른 모양(stampShape)은 그대로 기억한다.
const canvasStampShape = computed(() => (currentTool.value === 'stamp' ? stampShape.value : 'circle'))
const canvasRef = ref<any>(null)
const movePanelRef = ref<any>(null)
const selection = ref<any>(null)
const modelLabel = ref('No Model Loaded')
const detectStatus = ref('')

/**
 * 저장하지 않은 변경 — 확정 이미지가 저장 때와 다르거나, 병합 안 한 드로잉 레이어가
 * 저장 뒤에 바뀌었으면 true. undo 로 저장 시점까지 되돌아가면 다시 false 가 된다.
 */
// 드로잉 레이어 불투명도 (병합·저장·자동저장 합성에 사용 — 바꾸면 저장본과 달라진다)
const drawLayerOpacity = ref(100)
const isDirty = computed(() => isEditorDirty({
  imagePath: imagePath.value,
  drawRevision: Number(canvasRef.value?.drawRevision ?? 0),
  drawHasContent: !!canvasRef.value?.hasDrawContent,
  drawOpacity: drawLayerOpacity.value,
}, savedMarker.value))

// 마스크 영역 이동(MovePanel) 상태 — 드래그 미리보기는 캔버스가, 확정은 백엔드가 한다
const moveStatusText = ref('마스킹을 먼저 해주세요')
const moveFillColor = ref('black')
// 그리기 파라미터 — DrawPanel 은 이 값을 props 로 표시만 하고 바뀐 필드만 올린다.
// 도구는 여기 없다: 도구의 주인은 currentTool 이고, 캔버스로 갈 때 합친다.
// 아래 펜·워터마크·그라디언트의 기본 색은 테마 토큰으로 바꾸지 않는다 —
// 이미지에 실제로 칠해지는 사용자 콘텐츠지 UI 크롬이 아니라서, 테마를 바꿨다고
// 이미 그린 그림의 색이 따라 변하면 안 된다.
const drawParams = ref<{ color: string; size: number; opacity: number; filled: boolean }>({
  color: '#ffffff', size: 10, opacity: 1, filled: false,
})
const wmClamp = ref(true)          // 워터마크 '이미지 영역 내 제한'
const wmImagePath = ref('')        // 이미지 워터마크로 고른 파일 경로
const wmTextColor = ref('#FFFFFF') // 텍스트 워터마크 색 (WatermarkPanel의 textColor prop)
const drawGradientEnd = ref('#000000')  // 그라디언트 끝 색 (DrawPanel의 gradientEndColor prop)
// 캔버스로 내려보내는 최종 그리기 파라미터 — 끝 색은 따로 관리되므로 여기서 합친다
/** 지금 고른 도구가 마스크 계열인지 그리기 계열인지 — 패널 위쪽 도구 옵션을 가른다. */
const toolKind = computed(() => toolById(currentTool.value)?.kind ?? 'mask')
// 캔버스(드로잉 레이어)가 보는 도구는 currentTool 그 자체다 — 패널과 캔버스가 다른 도구를
// 가리킬 일이 없다(예전엔 drawParams.tool 과 currentTool 을 따로 맞춰야 했다).
const canvasDrawParams = computed(() => ({
  ...drawParams.value, tool: currentTool.value, gradientEnd: drawGradientEnd.value,
}))

const undoStack = ref<string[]>([])
const redoStack = ref<string[]>([])

// 탭 6개(모자이크·색감·고급색감·워터마크·그리기·이동)를 3개로 합쳤다.
// 도구는 세로 툴바로 빠졌고, 남은 것은 '값을 넣고 적용하는 것' 뿐이라
// 무엇에 적용하는지로 묶인다: 색이냐 · 선택 영역이냐 · 이미지 전체냐.
const tabs = [
  { label: '보정' },
  { label: '효과' },
  { label: '변형' },
]

// 파일명 / 확장자 / 정보 표시용 computed
// 문서 이름은 원본(저장 대상) 기준 — 편집할 때마다 바뀌는 editor_temp 이름(edited_<uuid>)이 아니라
const baseName = computed(() => {
  const full = sourcePath.value || imagePath.value
  if (!full) return ''
  const p = full.replace(/\\/g, '/')
  return p.substring(p.lastIndexOf('/') + 1) || full
})
function _formatSize(bytes: number) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`
}
const fileInfoExtra = computed(() => {
  const parts = []
  if (fileFormat.value) parts.push(fileFormat.value)
  if (fileSize.value) parts.push(_formatSize(fileSize.value))
  return parts.length > 0 ? ' · ' + parts.join(' · ') : ''
})

// 마지막 탭 영속
watch(activeTab, (v) => {
  window.localStorage.setItem('editorActiveTab', String(v))
})

// 최근 파일 관리 (드롭존용)
const recentFiles = ref<string[]>([])
function _loadRecentFiles() {
  try {
    const saved = JSON.parse(window.localStorage.getItem('editorRecentFiles') || '[]')
    if (Array.isArray(saved)) recentFiles.value = saved.slice(0, 6)
  } catch {}
}
function _pushRecentFile(path: string) {
  if (!path) return
  const arr = recentFiles.value.filter(p => p !== path)
  arr.unshift(path)
  recentFiles.value = arr.slice(0, 6)
  window.localStorage.setItem('editorRecentFiles', JSON.stringify(recentFiles.value))
}

/**
 * 문서가 바뀔 때(열기·닫기) 옛 문서에 매인 일시 상태를 비운다.
 *  - 프리뷰: 옛 이미지 기준이다(보낸 것도 세대 무효화로 버린다).
 *  - 감지 상태: 옛 문서의 감지 결과는 editorResultForDoc 가 버린다 — '감지 중...' 이 남지 않게.
 *  - pristinePath: 남으면 새 문서의 모자이크 지우개가 옛 이미지의 '적용 전' 픽셀을 가져온다
 *    (백엔드는 크기만 맞으면 복원한다 — 같은 해상도의 생성 이미지끼리 흔하다).
 *  - 캔버스의 '적용 전' 스냅숏: 화면의 지우개는 캔버스 스냅숏(pristinePath 를 디코드한 것)에서 칠한다.
 *    디코드 중인 옛 문서의 pristinePath 가 새 문서 위에 앉지 않게 캔버스 쪽도 비운다.
 *  - 자동 감지 마스크: 결과가 문서 세대 게이트를 통과한 뒤 디코드가 끝나기 전에 문서가 바뀌면
 *    옛 문서의 마스크가 새 문서에 앉았다 — 디코드 중인 것을 버린다.
 */
function _resetDocTransients() {
  clearPreview()
  detectStatus.value = ''
  pristinePath.value = ''
  canvasRef.value?.resetPristine?.()
  canvasRef.value?.cancelMaskLoad?.()
}

function loadImage(path: string) {
  if (!path) return
  // 새 문서 — 진행 중인 저장·편집(editorProcess)의 결과가 늦게 와도 이 문서에는 적용하지 않는다
  _docGen++
  _resetDocTransients()
  undoStack.value = [path]
  redoStack.value = []
  imagePath.value = path
  sourcePath.value = path
  _sourceOwned = false   // 연 파일은 원본이다 — 첫 저장은 옆에 사본을 만든다
  // 같은 파일을 다시 열면 imagePath 감시가 안 돌아 레이어가 남는다 — 여기서 직접 비운다.
  canvasRef.value?.clearDrawLayer?.()
  // 새로 연 파일은 디스크와 같다. (감시가 레이어를 한 번 더 비워 리비전이 올라도,
  // 비어 있으면 변경이 아니다 — isEditorDirty 참고)
  savedMarker.value = {
    imagePath: path,
    drawRevision: Number(canvasRef.value?.drawRevision ?? 0),
    drawHadContent: false,
    drawOpacity: drawLayerOpacity.value,
  }
  sizeCache.clear()
  _pathSwaps.clear()
  imageDisplay.value = mediaUrl(path, true)
  canvasRef.value?.clearSelection(true)   // 새 이미지: stale 마스크 + undo/redo 스택 초기화
  _pushRecentFile(path)
  _applyImageSize(path, imageDisplay.value)
  // 파일 정보 조회 (포맷/용량)
  _loadFileInfo(path)
}

/**
 * 상단바·변형 패널의 이미지 크기를 `path` 기준으로 맞춘다. 편집 결과가 알려 준 크기가
 * 캐시에 있으면 그대로 쓰고, 없을 때만(첫 로드·옛 항목) 헤더를 읽는다.
 *
 * 프로브는 캔버스가 받는 것과 같은 **캐시 무력화 URL**(`displayUrl` = imageDisplay)로 읽는다.
 * 무력화 없는 `mediaUrl(path)` 는 한 문서 안에서 Blink 가 예전에 읽은 그림을 재사용해서,
 * 같은 경로의 파일이 저장으로 바뀐 뒤 다시 열면 옛 크기를 보고했다(비율 유지 리사이즈가
 * 옛 비율로 계산됨). 같은 URL 이라 캔버스의 로드와 한 번의 요청으로 합쳐진다.
 * 늦게 끝난 프로브가 그새 바뀐 이미지(또는 새 문서)의 크기를 덮지 않도록 경로·세대를 확인한다.
 */
function _applyImageSize(path: string, displayUrl?: string) {
  const known = sizeCache.get(path)
  if (known) {
    imgWidth.value = known.w
    imgHeight.value = known.h
    return
  }
  const gen = _docGen
  const img = new Image()
  img.onload = () => {
    if (gen !== _docGen) return   // 그사이 다른 문서를 열었다 — 옛 파일 크기일 수 있다
    sizeCache.set(path, img.naturalWidth, img.naturalHeight)
    if (imagePath.value === path) {
      imgWidth.value = img.naturalWidth
      imgHeight.value = img.naturalHeight
    }
  }
  img.src = displayUrl || mediaUrl(path, true)
}

async function _loadFileInfo(path: string) {
  fileFormat.value = ''
  fileSize.value = 0
  try {
    const backend: any = await getBackend()
    if (backend.getFileInfo) {
      backend.getFileInfo(path, (json: string) => {
        try {
          const d = JSON.parse(json)
          if (d.size) fileSize.value = d.size
          if (d.format) fileFormat.value = d.format
        } catch {}
      })
    } else {
      // 폴백: 확장자만 표시
      const m = path.match(/\.([a-zA-Z0-9]+)$/)
      if (m) fileFormat.value = m[1].toUpperCase()
    }
  } catch {}
}

const MAX_UNDO = 30

/** @param size 편집 결과가 알려 준 크기 — 있으면 헤더를 다시 읽지 않는다 */
function pushState(path: string, clearMask = true, size?: { width?: number; height?: number }) {
  undoStack.value.push(path)
  // undo 한도 (MAX_UNDO + 초기 상태 1개)
  while (undoStack.value.length > MAX_UNDO + 1) undoStack.value.shift()
  redoStack.value = []
  imagePath.value = path
  // 타임스탬프 없이 경로만 변경 → watch에서 zoom/rotation 유지됨
  imageDisplay.value = mediaUrl(path, true)
  if (size?.width && size?.height) sizeCache.set(path, size.width, size.height)
  _applyImageSize(path, imageDisplay.value)
  if (clearMask) canvasRef.value?.clearSelection(true)   // 이미지 작업 후: 마스크 히스토리도 리셋
}

// undo/redo 는 크기도 함께 되돌린다 — 예전에는 회전·자르기·리사이즈를 되돌려도 상단바와
// 변형 패널에 옛 크기가 남아, 비율 유지 리사이즈가 뒤집힌 비율로 계산됐다.
// (저장 여부는 isDirty computed 가 savedMarker 와 비교해 알아서 다시 계산한다)
function doUndo() {
  if (undoStack.value.length <= 1) return
  redoStack.value.push(undoStack.value.pop() as string)
  const path = undoStack.value[undoStack.value.length - 1]
  imagePath.value = path
  imageDisplay.value = mediaUrl(path, true)
  _applyImageSize(path, imageDisplay.value)
}
function doRedo() {
  if (redoStack.value.length === 0) return
  const path = redoStack.value.pop() as string
  undoStack.value.push(path)
  imagePath.value = path
  imageDisplay.value = mediaUrl(path, true)
  _applyImageSize(path, imageDisplay.value)
}
// 통합 undo/redo — 마스킹이 있으면 마스킹 먼저, 없으면 이미지 작업. (버튼·Ctrl+Z/Y 공용)
function onUndo() { if (canvasRef.value?.undoMask()) return; doUndo() }
function onRedo() { if (canvasRef.value?.redoMask()) return; doRedo() }

// 에디터 작업 순서 보장 — editorProcess는 클릭마다 백그라운드 스레드를 띄우고 job_id를
// 반환한다. 느린 작업 A 뒤에 빠른 B를 실행하면 A 결과가 *나중*에 도착해 B를 덮을 수 있다.
// 시작 시 받은 job_id 중 최대값(_latestEditorJob)만 유효로 보고, 더 낮은 job 결과는 버린다.
let _latestEditorJob = 0
function _captureJob(result: any) {
  if (result && typeof result.job_id === 'number' && result.job_id > _latestEditorJob) {
    _latestEditorJob = result.job_id
  }
}

async function doOp(operation: string, params: any = {}) {
  if (!imagePath.value) return
  // 이 요청의 문서 세대·경로 — 백엔드가 doc_gen 을 결과에 돌려주고, 그사이 다른 문서를 열거나
  // 닫았으면 onEditorResult 가 버린다(editorResultForDoc).
  const gen = _docGen
  const cleanPath = stripFileUrl(imagePath.value)
  if (params.preview) {
    // 이 요청의 세대 — 그사이 프리뷰를 걷으면 결과가 와도 버린다
    params = { ...params, preview_token: previewGate.current(), doc_gen: gen }
  } else {
    // 확정 작업이면 예약된 프리뷰를 먼저 취소하고, 이미 보낸 프리뷰도 무효로 한다(결과 역전 방지).
    cancelPendingPreview()
    previewGate.invalidate()
    params = { ...params, doc_gen: gen }
  }
  const backend: any = await getBackend()
  if (gen !== _docGen) return   // 백엔드를 기다리는 사이 문서가 바뀌었다 — 옛 문서 작업은 보내지 않는다
  // 처리는 비동기 — 결과는 editorResult 이벤트로 도착. 콜백은 즉시 거절(경로/파라미터 오류)만 + job_id 캡처.
  backend.editorProcess(cleanPath, operation, JSON.stringify(params), (json: string) => {
    try {
      const result = JSON.parse(json)
      _captureJob(result)
      if (result.error) console.error('[Editor] error:', result.error)
    } catch (e) { console.error('[Editor] parse error:', e) }
  })
}

// 마스크 기반 효과 적용 (base64 마스크 전송)
async function doOpWithMask(operation: string, params: any = {}) {
  if (!imagePath.value) return
  // 프리뷰는 백엔드가 긴 변 PREVIEW_MAX_EDGE 로 줄여 처리한다 — 마스크도 그 크기로 줄여 보낸다
  const maskB64 = canvasRef.value?.getMaskBase64?.(params.preview ? { maxEdge: PREVIEW_MAX_EDGE } : {})
  if (!maskB64) {
    // 칠한 곳이 없다. 예전 폴백(마스크 없이 doOp)은 선택도 없으면 백엔드가 '이미지 전체'에
    // 효과를 적용했다 — 프리뷰는 걷고, 확정은 알린다.
    if (params.preview) { clearPreview(); return }
    requestAction('show_toast', {
      type: 'warning', msg: '적용할 영역이 없습니다 — 브러시/올가미/박스로 먼저 마스킹하세요',
    })
    return
  }
  doOp(operation, { ...params, mask_base64: maskB64 })
}

// editorResult — 백그라운드 처리 완료 수신 (onMounted에서 연결)
function onEditorResult(json: string) {
  try {
    const result = JSON.parse(json)
    // 다른 문서(열기·붙여넣기·닫기 전의 문서, 웹 모드의 다른 탭)의 결과 — 이미지·마스크·오류·프리뷰
    // 모두 버린다. job_id 가드는 새 문서가 아직 작업을 안 했으면 옛 결과를 통과시킨다.
    if (!editorResultForDoc(result, _docGen)) return
    // 순서 역전 차단 — 더 새 작업(job_id↑)이 이미 시작됐으면 늦게 온 옛 결과는 버림
    if (typeof result.job_id === 'number' && result.job_id < _latestEditorJob) {
      return
    }
    if (result.preview || result.preview_request) {
      // 그사이 프리뷰를 걷었거나(탭 전환·리셋·도구 변경) 확정 작업을 보냈다 — 낡은 결과다.
      if (!previewGate.accepts(result.preview_token)) return
      // 원본이 그새 교체됐으면(적용/회전 등) 이 프리뷰는 이미 낡았다 — 버린다.
      if (_previewForPath && _previewForPath !== imagePath.value) return
      if (!result.preview) {
        // 프리뷰 실패는 슬라이더 틱마다 토스트를 띄울 일이 아니다 — 콘솔에만 남기고 걷는다
        if (result.error) console.warn('[Editor] preview error:', result.error)
        previewSrc.value = ''
        return
      }
      // 프리뷰는 화면에만 반영한다 — 파일 교체도 undo 푸시도 하지 않는다.
      previewSrc.value = result.image_base64 || ''
      return
    }
    if (result.path) {
      previewSrc.value = ''   // 확정 결과가 왔으니 프리뷰는 걷는다
      pushState(result.path, true, { width: result.width, height: result.height })
      if (result.operation === 'auto_censor') detectStatus.value = '완료'
    } else if (result.mask_base64) {
      canvasRef.value?.loadMaskFromBase64(result.mask_base64)
      detectStatus.value = `${result.detect_count || 0}개 감지됨`
    } else if (result.error) {
      // 콘솔에만 찍으면 사용자는 '무반응'으로 느낀다 — 토스트로 올린다
      console.error('[Editor] error:', result.error)
      requestAction('show_toast', { type: 'error', msg: result.error })
      if (result.operation === 'auto_censor' || result.operation === 'auto_detect') {
        detectStatus.value = result.error
      } else {
        // 영역 이동 미리보기·모자이크 지우개처럼 화면에만 먼저 그린 것이 남지 않게
        // 원본(파일과 같은 그림)으로 다시 그린다.
        canvasRef.value?.drawAll?.()
      }
    }
  } catch (e) { console.error('[Editor] parse error:', e) }
}


// ── 세로 툴바 ──
// 도구를 고르면 캔버스 모드를 바꾼다. 옵션은 탭 위 '도구 옵션' 자리에 있으므로
// 탭을 갈아탈 필요가 없다 — 그게 툴바를 뺀 이유다.
function selectTool(id: string) {
  const tool = toolById(id)
  if (!tool) return
  clearPreview()
  // 도구의 단일 출처. 옵션 패널(DrawPanel/MaskToolOptions)은 :tool 로, 캔버스는
  // canvasDrawParams.tool 로 같은 값을 본다 — 패널이 새로 마운트돼도 어긋나지 않는다.
  currentTool.value = id
}

// Edge map — utils/edgeMap 의 EdgeMapCache(InpaintView 와 공용). 같은 이미지에서 magnetic 을
// 다시 켜도 Canny 를 다시 돌리지 않는다(4K+ 에서 200~500ms). 이미지가 바뀌면 캔버스의 엣지맵도
// 버리고, 자석 올가미를 쓰는 중이면 새 이미지로 다시 받는다 — 예전엔 캐시만 비우고 캔버스에는
// 옛 엣지맵이 남아, 회전·자르기 뒤에도 옛 이미지의 윤곽에 붙었다.
const edgeMapCache = new EdgeMapCache()
let _edgeMapFor = ''   // 캔버스에 올라간 엣지맵의 이미지 경로
function _edgeMapPath() { return stripFileUrl(imagePath.value) }
async function ensureEdgeMap() {
  if (!magneticLasso.value || !imagePath.value) return
  const path = _edgeMapPath()
  if (_edgeMapFor === path) return
  const b64 = await edgeMapCache.get(path)
  // 기다리는 사이 이미지가 바뀌었거나 자석을 껐으면 버린다
  const canvas = canvasRef.value
  if (!b64 || !canvas || !magneticLasso.value || _edgeMapPath() !== path) return
  _edgeMapFor = path
  canvas.loadEdgeMap(b64)
}
function onMagneticChanged(enabled: boolean) {
  magneticLasso.value = enabled
  if (enabled) void ensureEdgeMap()
}
watch(imagePath, () => {
  edgeMapCache.clear()
  _edgeMapFor = ''
  canvasRef.value?.clearEdgeMap?.()
  if (magneticLasso.value && currentTool.value === 'lasso') void ensureEdgeMap()
})
// 자석을 켜 둔 채 다른 도구를 쓰다가 올가미로 돌아오면 지금 이미지의 엣지맵을 챙긴다
watch(currentTool, (tool) => { if (tool === 'lasso') void ensureEdgeMap() })

/** MaskToolOptions 는 바뀐 필드만 올린다 — 나머지는 건드리지 않는다. */
function onParamsChanged(params: {
  toolSize?: number; stampSpacing?: number; stampShape?: string; barW?: number; barH?: number
}) {
  if (!params) return
  if (typeof params.toolSize === 'number' && params.toolSize > 0) brushSize.value = params.toolSize
  if (typeof params.stampSpacing === 'number' && params.stampSpacing > 0) stampSpacing.value = params.stampSpacing
  if (typeof params.stampShape === 'string' && params.stampShape) stampShape.value = params.stampShape
  if (typeof params.barW === 'number' && params.barW > 0) barWidth.value = params.barW
  if (typeof params.barH === 'number' && params.barH > 0) barHeight.value = params.barH
}

// ── 모자이크 지우개 커밋 ──
// 지우개는 화면 캔버스만 되돌린다(저장은 파일 경로 기반이라 그대로 두면 결과가 사라짐).
// 효과 적용 직전 이미지 경로를 pristinePath로 들고 있다가, 그 파일에서 픽셀을 되가져온다.
// 캔버스의 지우개도 같은 파일을 디코드한 그림(pristineDisplay → :pristine-src)을 칠한다 — 화면과
// 커밋이 한 출처를 본다. 예전에는 캔버스가 로드한 이미지로 따로 스냅숏을 떠, 효과를 두 번 적용한 뒤
// (또는 효과→색 조정, 효과×2→undo→redo 뒤) 화면에서 지운 자리가 커밋 뒤 되살아나는 등 둘이 갈렸다.
const pristinePath = ref('')
// 캐시 무력화 URL — 같은 경로가 저장으로 바뀌었어도 옛 그림을 재사용하지 않게(computed 라 경로가 바뀔 때만 새로 만든다)
const pristineDisplay = computed(() => (pristinePath.value ? mediaUrl(pristinePath.value, true) : ''))
async function commitRestore() {
  const maskB64 = canvasRef.value?.getRestoreMaskBase64?.()
  if (!maskB64) return
  if (!pristinePath.value) {
    // 되돌릴 '적용 전' 이미지가 없다 — 캔버스는 칠하지 않고 영역만 기록해 보냈다. 사용자에게 알린다
    requestAction('show_toast', {
      type: 'info', msg: '되돌릴 이전 상태가 없습니다 (효과를 먼저 적용하세요)',
    })
    canvasRef.value?.clearRestoreMask?.()
    // 지우개가 화면에 칠한 픽셀은 파일에 반영되지 않는다 — 파일과 같은 그림으로 다시 그린다
    canvasRef.value?.drawAll?.()
    return
  }
  // pristinePath 는 그대로다 — 복원 결과가 로드된 뒤에도 지우개는 같은 '적용 전' 그림을 칠한다
  doOp('restore', {
    mask_base64: maskB64,
    source_path: stripFileUrl(pristinePath.value),
  })
  canvasRef.value?.clearRestoreMask?.()
}

const EFFECT_OPS: Record<string, string> = { 0: 'mosaic', 1: 'censor_bar', 2: 'blur' }

/**
 * 효과 프리뷰 — 칠한 마스크가 있어야 보여줄 게 있다.
 *
 * 가드는 getSelection(증분 추적한 경계 상자, O(1))으로 본다 — applyEffect 와 같은 기준.
 * 예전에는 getMaskBase64() 를 가드로 썼는데, 이미지를 열기만 해도 빈 마스크 PNG 가 나와
 * 늘 참이었고, 슬라이더 틱마다 원본 해상도 w×h 루프 + PNG 인코딩이 디바운스 전에 돌았다.
 * 인코딩은 디바운스된 doOpWithMask 에서 축소본으로 한 번만 한다.
 */
function previewEffect(effectData: any) {
  if (!canvasRef.value?.getSelection?.()) { clearPreview(); return }
  const op = EFFECT_OPS[effectData?.effect] ?? 'mosaic'
  scheduleMaskPreview(op, effectData)
}

function applyEffect(effectData: any) {
  const sel = canvasRef.value?.getSelection()
  const effectMap: Record<string, string> = { 0: 'mosaic', 1: 'censor_bar', 2: 'blur' }
  const op = effectMap[effectData.effect] ?? 'mosaic'
  if (!sel) {
    // 예전엔 조용히 return 해서 "APPLY를 눌렀는데 아무 일도 안 남"이었다
    requestAction('show_toast', {
      type: 'warning',
      msg: '적용할 영역이 없습니다 — 브러시/올가미/박스로 먼저 마스킹하세요',
    })
    return
  }
  // 효과 적용 전 상태를 기억해 둔다 — 모자이크 지우개가 이 픽셀을 되살린다(화면·커밋 모두 이 파일).
  pristinePath.value = imagePath.value
  doOpWithMask(op, { ...effectData, selection: sel })
}

async function openModelDialog() {
  // 먼저 자동 감지 새로고침
  const backend: any = await getBackend()
  if (backend.refreshYoloModels) {
    backend.refreshYoloModels((label: string) => { if (label) modelLabel.value = label })
  }
  // 새 모델 추가도 가능
  requestAction('editor_add_yolo_model')
}
function clearModels() { requestAction('editor_clear_yolo_models') }

async function runAutoCensor(params: any) {
  if (!imagePath.value) return
  // 문서 세대·경로는 요청한 순간의 것 — 결과가 오기 전에 다른 문서를 열면 버린다(editorResultForDoc)
  const gen = _docGen
  const cleanPath = stripFileUrl(imagePath.value)
  detectStatus.value = '감지 중...'
  const backend: any = await getBackend()
  if (gen !== _docGen) return
  const samModel = params?.samModel || 'auto'
  const payload: Record<string, any> = {
    confidence: (params?.confidence || 25) / 100,
    sam_model: samModel,
    doc_gen: gen,
  }
  // SAM3일 때만 exclude_prompt 전달 (다른 SAM 모델은 텍스트 프롬프트를 받지 않음)
  if (samModel === 'sam3' && params?.excludePrompt && String(params.excludePrompt).trim()) {
    payload.exclude_prompt = String(params.excludePrompt).trim()
  }
  // SAM3는 텍스트 기반 세그멘터 — detect prompt가 있으면 YOLO 없이도 단독 실행된다
  if (samModel === 'sam3' && params?.detectPrompt && String(params.detectPrompt).trim()) {
    payload.detect_prompt = String(params.detectPrompt).trim()
  }
  // 결과는 editorResult 이벤트로 도착 — 콜백은 즉시 거절만 처리 + job_id 캡처
  backend.editorProcess(cleanPath, 'auto_censor', JSON.stringify(payload), (json: string) => {
    try {
      const result = JSON.parse(json)
      _captureJob(result)
      if (result.error && gen === _docGen) detectStatus.value = result.error
    } catch { if (gen === _docGen) detectStatus.value = '오류' }
  })
}

async function runAutoDetect(params: any) {
  if (!imagePath.value) return
  const gen = _docGen
  const cleanPath = stripFileUrl(imagePath.value)
  detectStatus.value = '감지 중...'
  const backend: any = await getBackend()
  if (gen !== _docGen) return
  const samModel = params?.samModel || 'auto'
  const payload: Record<string, any> = {
    confidence: (params?.confidence || 25) / 100,
    sam_model: samModel,
    doc_gen: gen,
  }
  if (samModel === 'sam3' && params?.excludePrompt && String(params.excludePrompt).trim()) {
    payload.exclude_prompt = String(params.excludePrompt).trim()
  }
  // SAM3는 텍스트 기반 세그멘터 — detect prompt가 있으면 YOLO 없이도 단독 실행된다
  if (samModel === 'sam3' && params?.detectPrompt && String(params.detectPrompt).trim()) {
    payload.detect_prompt = String(params.detectPrompt).trim()
  }
  // 결과(mask_base64)는 editorResult 이벤트로 도착 — 콜백은 즉시 거절만 처리 + job_id 캡처
  backend.editorProcess(cleanPath, 'auto_detect', JSON.stringify(payload), (json: string) => {
    try {
      const result = JSON.parse(json)
      _captureJob(result)
      if (result.error && gen === _docGen) detectStatus.value = result.error
    } catch { if (gen === _docGen) detectStatus.value = '오류' }
  })
}

// 예전에는 마스크 bbox 로 확인 없이 즉시 잘랐고, 선택이 없으면 아무 말 없이 무시했다.
const cropSel = ref<any>(null)
const cropPending = computed(() => {
  const s = cropSel.value
  if (!s) return ''
  // EditorCanvas.getSelection() 은 {x, y, w, h} 를 준다
  return `${Math.max(0, Math.round(s.w))} × ${Math.max(0, Math.round(s.h))}`
})
function doCrop() {
  const sel = canvasRef.value?.getSelection()
  if (!sel) {
    requestAction('show_toast', { type: 'warning', msg: '먼저 자를 영역을 선택하세요' })
    return
  }
  cropSel.value = sel
}
function confirmCrop() {
  if (!cropSel.value) return
  doOp('crop', { selection: cropSel.value })
  cropSel.value = null
}
function cancelCrop() { cropSel.value = null }

// ── 원근 보정 ──
// 꼭짓점 4개를 드래그해 '원본에서 직사각형이어야 할 영역'을 지정하면
// 백엔드(core/editor_ops.perspective)가 그 사다리꼴을 정직사각형으로 편다.
// 출력 크기는 넘기지 않는다 — 백엔드가 대변 길이 최댓값으로 추론한다.
const perspectiveActive = ref(false)
function onStartPerspective() {
  if (!imagePath.value) return
  clearPreview()   // 프리뷰가 떠 있으면 원본 캔버스가 가려져 꼭짓점을 맞출 그림이 안 보인다
  perspectiveActive.value = true
  currentTool.value = 'perspective'
  canvasRef.value?.beginPerspective?.()
  requestAction('show_toast', {
    type: 'info', msg: '꼭짓점 4개를 펴고 싶은 사각형 모서리에 맞춘 뒤 "적용"을 누르세요',
  })
}
function onConfirmPerspective() {
  const corners = canvasRef.value?.endPerspective?.()
  perspectiveActive.value = false
  currentTool.value = 'box'
  if (!corners) {
    requestAction('show_toast', { type: 'warning', msg: '꼭짓점 정보가 없습니다' })
    return
  }
  doOp('perspective', { corners })
}
function onCancelPerspective() {
  canvasRef.value?.cancelPerspective?.()
  perspectiveActive.value = false
  currentTool.value = 'box'
}
function doResize(params?: any) { doOp('resize', params) }
function applyAdj(adj: any) { doOp('color_adjust', adj) }
// ── 실시간 프리뷰 ────────────────────────────────────────────────────────
// 예전에는 이 두 함수가 빈 TODO 였다. 패널은 열심히 emit 하는데 받는 쪽이
// 아무것도 안 해서, 슬라이더를 움직여도 화면이 그대로였다(= '적용해야 결과를 앎').
// 백엔드가 축소본으로 같은 연산을 돌려 base64 로 돌려준다 — CSS 필터 근사와 달리
// 12종 필터·HSV 채도·레벨까지 실제 결과와 일치한다.
let _previewTimer: ReturnType<typeof setTimeout> | null = null
// 프리뷰를 쏠 때의 원본 경로. 확정 작업이 끼어들어 이미지가 교체되면, 뒤늦게
// 도착하는 프리뷰는 '옛 이미지 기준' 결과라 화면에 올리면 안 된다.
// (job_id 가드는 이걸 못 잡는다 — 늦게 쏜 프리뷰가 job_id 는 더 크기 때문)
let _previewForPath = ''
function schedulePreview(operation: string, params: any) {
  if (!imagePath.value) return
  if (_previewTimer) clearTimeout(_previewTimer)
  _previewTimer = setTimeout(() => {
    _previewForPath = imagePath.value
    doOp(operation, { ...params, preview: true })
  }, 120)
}
// 대기 중인 프리뷰만 취소한다(화면은 그대로) — 확정 작업을 보낼 때 쓴다.
// 이걸 안 하면: 적용 클릭 → 패널이 adjustment-changed 도 함께 emit → 120ms 뒤
// 프리뷰가 한 번 더 나가고, 그게 적용보다 늦게 도착해 job_id 가 더 커서 가드도
// 통과하며 화면을 축소본으로 되돌린다(실제로 있었던 증상).
/**
 * 마스크가 필요한 작업의 프리뷰. `schedulePreview` 와 갈라놓은 이유는 이쪽만
 * 마스크 base64 를 함께 실어야 하기 때문이다(그쪽은 `doOp`, 이쪽은 `doOpWithMask`).
 */
function scheduleMaskPreview(operation: string, params: any) {
  if (!imagePath.value) return
  if (_previewTimer) clearTimeout(_previewTimer)
  _previewTimer = setTimeout(() => {
    _previewForPath = imagePath.value
    doOpWithMask(operation, { ...params, preview: true })
  }, 120)
}

// 대기 중인 프리뷰만 취소한다(화면은 그대로) — 확정 작업을 보낼 때 쓴다.
function cancelPendingPreview() {
  if (_previewTimer) { clearTimeout(_previewTimer); _previewTimer = null }
}
/** 프리뷰를 걷는다 — 예약된 것은 취소하고, 이미 보낸 것은 도착해도 버린다(세대 무효화). */
function clearPreview() {
  cancelPendingPreview()
  previewGate.invalidate()
  previewSrc.value = ''
}
// 조정값이 전부 중립이면 보여줄 게 없다 — 프리뷰를 요청하지 말고 걷는다.
// (ColorPanel.onApply 는 적용 후 resetSliders() 를 부르고, 그 watch 가
//  adjustment-changed {0,0,0} 을 다시 쏜다. 그걸 그대로 프리뷰로 만들면
//  방금 확정한 전체 해상도 결과를 축소본이 덮어쓴다.)
function previewAdj(adj: any) {
  if (!adj || (!adj.brightness && !adj.contrast && !adj.saturation)) { clearPreview(); return }
  schedulePreview('color_adjust', adj)
}
function previewAdvAdj(adj: any) {
  // 커브도 함께 봐야 한다 — 슬라이더가 전부 중립이어도 커브만 건드린 경우가 있다.
  const neutral = !adj || (!adj.blackPoint && (adj.whitePoint ?? 255) === 255
    && Math.abs((adj.gamma ?? 1) - 1) < 1e-6 && !adj.temperature && !adj.tint
    && isIdentity(adj.curves))
  if (neutral) { clearPreview(); return }
  schedulePreview('adv_color', adj)
}
function previewFilter(payload: any) {
  if (!payload || !payload.filter || !payload.strength) { clearPreview(); return }
  schedulePreview('filter', payload)
}
// WatermarkPanel 은 텍스트/이미지 두 설정을 같은 'preview' 로 보낸다 — text 유무로 가른다.
function previewWatermark(cfg: any) {
  if (!cfg) return
  if (typeof cfg.text === 'string') {
    schedulePreview('text_watermark', { ...cfg, clamp: wmClamp.value, color: wmTextColor.value })
  } else if (wmImagePath.value) {
    schedulePreview('image_watermark', { ...cfg, clamp: wmClamp.value, watermark_path: wmImagePath.value })
  }
}
function switchTab(i: number) { clearPreview(); activeTab.value = i }
function resetAdj() { clearPreview() }
// ColorPanel은 { filter, strength }를 보낸다. 예전 `filter.name || filter.type`은
// 둘 다 없어서 operation이 undefined로 나갔고, 백엔드는 모든 분기를 통과해
// 원본을 그대로 재저장했다 (= 필터 프리셋 전부 무반응).
function applyFilter(payload: any) {
  if (!payload?.filter) return
  doOp('filter', { filter: payload.filter, strength: payload.strength ?? 100 })
}
function applyAdvAdj(adj: any) { doOp('adv_color', adj) }

// ── 드로잉 레이어 병합 ──
// DrawPanel이 emit하는 이름은 'flatten-layer'인데 예전에는 '@flatten'에 물려 있어
// 버튼이 아예 아무것도 안 했다. 오버레이 캔버스를 base64로 실어 보낸다.
async function applyFlatten() {
  const overlay = canvasRef.value?.getDrawOverlayBase64?.()
  if (!overlay) {
    requestAction('show_toast', { type: 'info', msg: '병합할 드로잉이 없습니다' })
    return
  }
  doOp('flatten', { overlay_base64: overlay, opacity: drawLayerOpacity.value })
  // 레이어 비우기는 imagePath 감시가 맡는다 — 병합 결과가 새 경로로 돌아오면 지워진다.
  // 여기서 미리 지우면 백엔드가 실패했을 때 그린 게 통째로 날아간다.
}

// 확정 이미지가 바뀌면(병합·회전·자르기·undo 등) 드로잉 레이어는 더 이상 맞지 않는다.
// 예: 회전 후에도 레이어가 남아 있으면 안 돌아간 그림이 돌아간 이미지 위에 얹힌다.
// 단, 저장이 원본을 덮어써서 경로만 '덮어쓰기 전 사본'으로 바꾼 경우는 같은 그림이다 —
// 그때 레이어를 비우면 병합 안 한 그림이 저장 직후 사라진다.
const _pathSwaps = new Map<string, string>()
function _consumePathSwap(from: string, to: string): boolean {
  if (!from || _pathSwaps.get(from) !== to) return false
  _pathSwaps.delete(from)
  return true
}
watch(imagePath, (p, prev) => { if (!_consumePathSwap(prev, p)) canvasRef.value?.clearDrawLayer?.() })

// ── 마스크 영역 이동 (MovePanel) ──
// MovePanel은 'confirm-move'/'cancel-move'를 emit하는데 예전에는 '@confirm'/'@cancel'에
// 물려 있어 전부 죽어 있었다. 게다가 백엔드에 start/confirm/cancel_move 핸들러가 없었다.
// 이제 이동은 캔버스에서 드래그로 미리보기하고, 확정할 때 한 번만 백엔드 move_region을 부른다.
function onStartMove(payload: any) {
  if (!canvasRef.value?.getSelection()) {
    requestAction('show_toast', { type: 'warning', msg: '이동할 영역을 먼저 마스킹하세요' })
    movePanelRef.value?.setMovingState?.(false)
    return
  }
  // 프리뷰가 떠 있으면 원본 캔버스가 가려져 이동 미리보기가 보이지 않는다
  clearPreview()
  moveFillColor.value = payload?.fillColor || 'black'
  currentTool.value = 'move'
  // 구멍은 확정 결과(move_region)와 같은 색으로 미리 보여 준다
  canvasRef.value?.beginMove?.(moveFillColor.value)
  moveStatusText.value = '영역을 드래그해 옮긴 뒤 "확정"을 누르세요'
}

/** 회전·크기는 MovePanel 이 확정 페이로드로 준다 — 그 값이 유일한 출처다. */
function onConfirmMove(payload: { rotation?: number; scale?: number } | undefined) {
  const offset = canvasRef.value?.endMove?.() || { dx: 0, dy: 0 }
  currentTool.value = 'box'
  moveStatusText.value = '마스킹을 먼저 해주세요'
  doOpWithMask('move_region', {
    dx: offset.dx, dy: offset.dy,
    rotation: Number(payload?.rotation ?? 0),
    scale: Number(payload?.scale ?? 100),
    fillColor: moveFillColor.value,
  })
}

function onCancelMove() {
  canvasRef.value?.cancelMove?.()
  currentTool.value = 'box'
  moveStatusText.value = '마스킹을 먼저 해주세요'
}
function applyTextWm(params: any) { doOp('text_watermark', { ...params, clamp: wmClamp.value, color: wmTextColor.value }) }
function applyImageWm(params: any) {
  if (!wmImagePath.value) { requestAction('show_toast', { type: 'warning', msg: '먼저 워터마크 이미지를 불러오세요' }); return }
  doOp('image_watermark', { ...params, clamp: wmClamp.value, watermark_path: wmImagePath.value })
}
function loadWatermarkImage() { requestAction('editor_load_watermark_image') }

// DrawPanel 은 바뀐 필드({color|size|opacity|filled})만 보낸다. 도구는 보내지 않는다 —
// 예전에는 패널 로컬 상태의 tool 까지 실려 와 currentTool 을 덮어써서, 색 하나 눌러도
// 도구가 펜으로 바뀌었다(패널이 새로 마운트되며 '펜'으로 초기화돼 있었으므로).
function onDrawParamsChanged(p: Partial<{ color: string; size: number; opacity: number; filled: boolean }>) {
  if (!p) return
  const next = { ...drawParams.value }
  if (typeof p.color === 'string' && p.color) next.color = p.color
  if (typeof p.size === 'number' && p.size > 0) next.size = p.size
  if (typeof p.opacity === 'number') next.opacity = Math.max(0, Math.min(1, p.opacity))
  if (typeof p.filled === 'boolean') next.filled = p.filled
  drawParams.value = next
}

// 앱에 색상 선택 다이얼로그가 없다(QColorDialog 미사용). 브리지 왕복 없이
// 네이티브 컬러 입력을 띄운다. 실패하면 패널의 12색 팔레트가 그대로 대안이다.
function pickColor(target: 'draw' | 'gradient' | 'wmText') {
  const el = document.createElement('input')
  el.type = 'color'
  // 지금 쓰는 색에서 시작한다 — 예전엔 워터마크 글자 색을 고를 때마다 흰색부터 다시 골라야 했다
  el.value = target === 'draw' ? drawParams.value.color
    : target === 'wmText' ? wmTextColor.value : drawGradientEnd.value
  el.style.position = 'fixed'; el.style.left = '-9999px'
  document.body.appendChild(el)
  el.addEventListener('change', () => {
    const v = el.value
    // 패널은 drawParams 를 props 로 보므로 여기만 바꾸면 표시도 따라온다
    if (target === 'draw') { drawParams.value = { ...drawParams.value, color: v } }
    else if (target === 'gradient') { drawGradientEnd.value = v }
    else { wmTextColor.value = v }
    el.remove()
  })
  el.addEventListener('cancel', () => el.remove())
  el.click()
}

// 지금 이미지를 인페인트로 넘긴다. 백엔드에 이미 send_to_inpaint 액션이 있다
// (generator_main.py: tabChanged → inpaintImageLoaded). 마스킹·프롬프트는 그 탭에서 한다.
function onSendInpaint() {
  if (!imagePath.value) return
  requestAction('send_to_inpaint', { path: imagePath.value })
}

// 복원 브러시 — 칠한 자리를 주변 픽셀로 메운다(백엔드 inpaint).
// 레이어에 그리는 다른 도구와 달리 원본을 고치는 작업이라 확정 연산으로 나간다.
function applyHeal() {
  const mask = canvasRef.value?.getHealMaskBase64?.()
  if (!mask) {
    requestAction('show_toast', { type: 'info', msg: '복원할 자리를 먼저 칠하세요' })
    return
  }
  doOp('heal', { mask_base64: mask, radius: Math.max(1, Math.round(drawParams.value.size / 2)) })
  canvasRef.value?.clearHealMask?.()
}

/** 스포이트가 집은 색을 현재 색으로 되돌린다 — 패널은 props 로 같은 값을 본다. */
function onEyedropperColor(hex: string) {
  drawParams.value = { ...drawParams.value, color: hex }
}

function undoDrawStroke() {
  if (!canvasRef.value?.undoDrawStroke?.()) {
    requestAction('show_toast', { type: 'info', msg: '되돌릴 획이 없습니다' })
  }
}

function clearDrawLayer() {
  canvasRef.value?.clearDrawLayer?.()
}
function onSelectionChanged(sel: any) { selection.value = sel }
/** 자르기·영역 이동은 선택 영역이 있어야 성립한다 — 버튼을 막는 조건. */
const hasSelection = computed(() => !!selection.value)

// MovePanel 인페인트 버튼 활성 조건. canInpaint prop 자체가 전달되지 않아
// 기본값 false 로 영구 비활성이었다. 이 버튼은 이미지를 Inpaint 탭으로 넘기는
// 동작이므로(마스킹은 그 탭에서 한다) 이미지가 열려 있으면 충분하다.
const canInpaint = computed(() => !!imagePath.value)

function onDrop(e: DragEvent) {
  isDragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file && (file as any).path) loadImage((file as any).path.replace(/\\/g, '/'))
}

function openFile() { requestAction('editor_open_file') }

// ── 저장 ────────────────────────────────────────────────────────────────
// 예전 '저장'은 파일을 쓰지 않고 성공 토스트만 띄운 뒤 곧바로 isDirty=false 로 바꿨다.
// 이제는 백엔드(ui/editor_save_actions.py)가 실제로 쓰고 editorSaveResult 로 답한 뒤에만
// '저장됨'으로 바꾼다. 저장은 비파괴다 — 연 원본은 덮어쓰지 않고 원본 옆(임시 원본이면
// 기본 출력 폴더)에 <이름>_edited[_N] 사본을 만들며(메타데이터 보존), 이 문서의 다음 저장은
// 그 사본만 갱신한다. 병합 안 한 드로잉 레이어도 저장본에 합성된다.
// 요청별로 들고 있다 — 저장 중에 다른 문서를 열어도 옛 응답이 새 문서를 건드리지 않고,
// 새 문서는 옛 저장이 끝나기를 기다리지 않고 저장할 수 있다.
const _pendingSaves = new Map<number, PendingSave>()
let _saveSeq = 0
// 응답이 영영 안 오면(백엔드 예외 등) 저장 버튼이 잠기지 않게 — 대화상자가 열려 있을 시간은 넉넉히
const SAVE_STALE_MS = 120_000

/** 에디터 히스토리가 참조하는 경로 — 저장이 이 중 하나를 덮어쓰면 백엔드가 사본을 떠 준다. */
function _historyPaths(): string[] {
  const paths = new Set<string>([...undoStack.value, ...redoStack.value])
  if (imagePath.value) paths.add(imagePath.value)
  if (pristinePath.value) paths.add(pristinePath.value)
  return [...paths]
}

function _requestSave(saveAs: boolean) {
  if (!imagePath.value) return
  const now = Date.now()
  // 응답이 끝내 안 온 옛 요청은 정리한다(백엔드 예외 등) — 맵이 계속 자라지 않게
  for (const [id, p] of _pendingSaves) if (now - p.startedAt >= SAVE_STALE_MS) _pendingSaves.delete(id)
  if (hasPendingSaveFor(_pendingSaves.values(), _docGen, now, SAVE_STALE_MS)) {
    requestAction('show_toast', { type: 'info', msg: '저장 중입니다 — 잠시 후 다시 시도하세요' })
    return
  }
  const overlay: string | null = canvasRef.value?.getDrawOverlayBase64?.() || null
  // 보낸 값과 '저장됨'으로 기록할 값이 어긋나지 않게 한 번만 읽는다
  const opacity = drawLayerOpacity.value
  // 창(웹 모드의 여러 탭)마다 겹치지 않게 시각을 섞는다
  const id = now * 100 + (++_saveSeq % 100)
  _pendingSaves.set(id, {
    id,
    docGen: _docGen,
    imagePath: imagePath.value,
    drawRevision: Number(canvasRef.value?.drawRevision ?? 0),
    drawHadContent: !!overlay,
    drawOpacity: opacity,
    startedAt: now,
  })
  const payload = {
    request_id: id,
    path: imagePath.value,
    source_path: sourcePath.value,
    // 이 문서가 앞서 저장한 사본이면 그 파일을 갱신한다. 아니면(연 원본) 옆에 새 사본을 만든다.
    overwrite_source: _sourceOwned,
    overlay_base64: overlay,
    overlay_opacity: opacity,
    protect_paths: _historyPaths(),
  }
  if (saveAs) requestAction('editor_save_as', payload)
  else requestAction('editor_save', payload)
}
function saveImage() { _requestSave(false) }
function saveAsImage() { _requestSave(true) }

/**
 * 저장이 덮어쓴 파일을 히스토리가 가리키고 있으면, 그 자리를 '덮어쓰기 전 사본'으로 바꾼다.
 * 그래야 원본을 열고 바로 저장한 뒤 undo 해도 저장된 결과가 아니라 저장 전 그림이 나온다.
 */
function _swapHistoryPath(from: string, to: string) {
  undoStack.value = replacePath(undoStack.value, from, to)
  redoStack.value = replacePath(redoStack.value, from, to)
  // 지우개의 '적용 전' 그림도 사본을 가리킨다(같은 그림 — 캔버스는 pristineDisplay 가 바뀌어 사본을 다시 읽는다)
  if (pristinePath.value === from) pristinePath.value = to
  if (_previewForPath === from) _previewForPath = to
  sizeCache.rename(from, to)
  if (imagePath.value === from) {
    // 같은 그림이다 — 드로잉 레이어를 지킨다(_pathSwaps)
    _pathSwaps.set(from, to)
    imagePath.value = to
    imageDisplay.value = mediaUrl(to, true)
  }
}

async function _clearAutoSaveRecovery() {
  // 저장했으니 크래시 복구본은 낡았다 — 다음 시작 때 '복구할까요?'를 묻지 않게 지운다
  lastAutoSaveAt.value = 0
  // 쓰는 중인 자동저장(저장 전 상태)의 응답이 늦게 와 '자동저장 N초 전'을 되살리지 않게 버린다.
  // 백엔드도 폐기 뒤에 끝난 그 쓰기의 파일을 지운다(core/editor_autosave.py).
  autoSaveRequest.cancel()
  // 단, 이 문서가 복구 파일 그 자체를 참조하면(직접 연 경우) 지우면 편집 중인 그림이 사라진다
  // — 편집·undo·다음 저장이 전부 '경로가 없다'로 깨진다. (복구는 작업 사본으로 연다)
  if (referencesAutosaveFile([..._historyPaths(), sourcePath.value])) return
  try {
    const backend: any = await getBackend()
    backend?.editorClearAutoSave?.(() => {})
  } catch {}
}

function onEditorSaveResult(json: string) {
  const result = parseSaveResult(json)
  if (!result) return
  // 파일이 (덮어)쓰였으면 갤러리·즐겨찾기 카드가 같은 URL 의 옛 그림을 쓰지 않게 — 어느 창의 저장이든
  const written = writtenPathFromSaveResult(result)
  if (written) bumpMediaVersion(written)
  const pending = typeof result.request_id === 'number' ? _pendingSaves.get(result.request_id) : undefined
  const action = saveResultAction(pending, result, _docGen)
  // 다른 창(웹 모드의 다른 탭)의 저장 응답이면 무시
  if (action === 'ignore' || !pending) return
  _pendingSaves.delete(pending.id)
  if (action === 'cancelled') return
  if (action === 'failed') {
    requestAction('show_toast', { type: 'error', msg: `저장 실패: ${result.error || '알 수 없는 오류'}` })
    return
  }
  if (action === 'stale') {
    // 저장은 끝났지만 그사이 다른 문서를 열었거나 닫았다. 파일이 어디 저장됐는지만 알리고
    // 지금 문서(sourcePath·저장 표시·파일 정보·히스토리·복구본)는 건드리지 않는다.
    requestAction('show_toast', { type: 'success', msg: saveToastMessage(result) })
    return
  }
  let savedImagePath = pending.imagePath
  for (const [from, to] of aliasesFromSaveResult(result)) {
    _swapHistoryPath(from, to)
    if (savedImagePath === from) savedImagePath = to
  }
  // 이 문서는 이제 방금 쓴 파일이다 — 다음 Ctrl+S 는 그 사본만 갱신한다(owned).
  // '변경 없음'으로 원본을 그대로 돌려받았으면 owned=false 라 다음 저장도 원본을 건드리지 않는다.
  if (result.path) {
    sourcePath.value = result.path
    _sourceOwned = result.owned === true
    _loadFileInfo(result.path)
  }
  // 요청 시점의 상태가 저장됐다 — 응답을 기다리는 사이 더 편집했으면 여전히 dirty 다
  savedMarker.value = {
    imagePath: savedImagePath,
    drawRevision: pending.drawRevision,
    drawHadContent: pending.drawHadContent,
    drawOpacity: pending.drawOpacity,
  }
  _clearAutoSaveRecovery()
  requestAction('show_toast', { type: 'success', msg: saveToastMessage(result) })
}

// 클립보드에서 이미지 붙여넣기 — 데스크톱은 Qt 시스템 클립보드, 웹은 브라우저 클립보드 API.
// 형식 화이트리스트·청크 base64·응답 해석은 utils/clipboardImage.ts (예전 스프레드 btoa 는
// ~120KB 를 넘는 이미지에서 RangeError 로 전부 실패했다).
async function pasteFromClipboard() {
  try {
    const outcome = await pasteClipboardImage()
    if (outcome.kind === 'path') {
      loadImage(outcome.path)
      requestAction('show_toast', { type: 'success', msg: '클립보드 이미지 로드 완료' })
    } else {
      requestAction('show_toast', { type: outcome.kind, msg: outcome.msg })
    }
  } catch (e: any) {
    requestAction('show_toast', { type: 'error', msg: `클립보드 접근 실패: ${e?.message || e}` })
  }
}

function confirmClose() {
  if (isDirty.value) {
    if (!window.confirm('저장하지 않은 변경 사항이 있습니다.\n정말 닫으시겠습니까?')) return
  }
  resetEditor()
}
function resetEditor() {
  _resetDocTransients()
  imagePath.value = ''; imageDisplay.value = ''
  undoStack.value = []; redoStack.value = []
  sourcePath.value = ''; savedMarker.value = null
  _sourceOwned = false
  sizeCache.clear(); _pathSwaps.clear()
  // 문서 세대를 올린다 — 닫기 전에 보낸 저장의 응답은 '저장됨' 알림만 띄우고 상태는 건드리지 않고,
  // 닫기 전에 시작한 편집의 결과는 버린다(늦은 결과가 닫은 에디터를 다시 열지 않게)
  _docGen++
}

// 앱 시작 시 YOLO 라벨 로드
async function refreshYoloLabel() {
  const backend: any = await getBackend()
  if (backend.getYoloModelLabel) {
    backend.getYoloModelLabel((label: string) => { if (label) modelLabel.value = label })
  }
}

// Ctrl 빠른 두 번 누름 감지 — 변환(zoom/rotation/pan) 초기화
let _lastCtrlTime = 0
const CTRL_DOUBLE_TAP_MS = 300

// keep-alive로 에디터 뷰가 항상 mount된 상태(탭 워밍)라도, 전역 keydown 핸들러는
// 에디터 탭이 *활성*일 때만 동작해야 한다. (안 그러면 t2i 등에서 Ctrl+V/Z/Y/S/O를
// 에디터가 가로채 일반 붙여넣기/실행취소가 막힘 — 워밍 도입 후 회귀)
let _editorActive = false
onActivated(() => { _editorActive = true })
onDeactivated(() => { _editorActive = false })

function onEditorKeyDown(e: KeyboardEvent) {
  if (!_editorActive) return   // 에디터 탭이 아닐 땐 단축키 가로채지 않음
  // ── Ctrl 단독 두 번 빠르게 → 변환 reset (모자이크/그리기는 유지)
  // (다른 Ctrl 조합은 _lastCtrlTime 갱신 안 함 — Ctrl+Z 등과 충돌 회피)
  if (e.key === 'Control' && !e.altKey && !e.shiftKey) {
    const now = Date.now()
    if (now - _lastCtrlTime < CTRL_DOUBLE_TAP_MS) {
      _lastCtrlTime = 0
      if (imagePath.value && canvasRef.value?.resetTransform) {
        canvasRef.value.resetTransform()
        requestAction('show_toast', { type: 'info', msg: '확대/회전/위치 초기화' })
      }
    } else {
      _lastCtrlTime = now
    }
    return
  }

  // Ctrl+O / Ctrl+V는 이미지 없어도 동작 (열기/붙여넣기)
  if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 'o') {
    e.preventDefault(); openFile(); return
  }
  if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 'v') {
    // 입력칸(워터마크 텍스트 등)에서의 Ctrl+V 는 글자 붙여넣기다 — 가로채면 글자가 안 들어가고
    // 엉뚱하게 '클립보드에 이미지가 없습니다'가 떴다.
    // (슬라이더·체크박스에 포커스가 남아 있을 때는 여전히 이미지 붙여넣기다)
    if (acceptsTextPaste(document.activeElement)) return
    e.preventDefault(); pasteFromClipboard(); return
  }
  // 이미지가 없으면 그 외 단축키는 무시
  if (!imagePath.value) return
  // 저장
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 's') { e.preventDefault(); saveAsImage(); return }
  if (e.ctrlKey && !e.shiftKey && e.key.toLowerCase() === 's') { e.preventDefault(); saveImage(); return }
  // Undo / Redo (마스크 우선, 그 다음 작업)
  // stopImmediatePropagation으로 다른 핸들러(PromptPanel 등)가 같은 키를 가로채지 못하게
  // — Editor 탭에서는 Editor undo가 우선권을 가짐 (사용자 명시 요구사항)
  if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'z') {
    e.preventDefault(); e.stopImmediatePropagation()
    onRedo(); return
  }
  if (e.ctrlKey && e.key.toLowerCase() === 'z') {
    e.preventDefault(); e.stopImmediatePropagation()
    onUndo(); return
  }
  if (e.ctrlKey && e.key.toLowerCase() === 'y') {
    e.preventDefault(); e.stopImmediatePropagation()
    onRedo(); return
  }
  if (e.key === 'Escape') {
    // 원근 보정 중이면 그것부터 취소 (마스크를 날리지 않게)
    if (perspectiveActive.value) { onCancelPerspective(); return }
    canvasRef.value?.clearSelection()
  }

  // 도구 단축키 (B=브러시, P=펜 …). 글자 하나짜리라 입력 중에는 절대 가로채면 안 된다
  // — 프롬프트나 텍스트 도구에 'b' 를 치는 순간 도구가 바뀌면 못 쓴다.
  const el = document.activeElement as HTMLElement | null
  const typing = !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)
  // 원근 보정·영역 이동 중에는 도구를 갈아타면 진행 중인 조작이 날아간다
  if (!typing && !perspectiveActive.value && currentTool.value !== 'move') {
    const tool = toolByKey(e.key, { ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey })
    if (tool) {
      e.preventDefault()
      selectTool(tool.id)
      requestAction('show_toast', { type: 'info', msg: `${tool.label} (${tool.shortcut})` })
      return
    }
  }
  // 원근 보정 중 Enter = 적용
  if (e.key === 'Enter' && perspectiveActive.value) {
    e.preventDefault()
    onConfirmPerspective()
  }
}

// 자동 저장 — 5분마다 변경 있으면 임시본 기록
let _autoSaveTimer: ReturnType<typeof setInterval> | null = null
const AUTO_SAVE_INTERVAL_MS = 5 * 60 * 1000
// 마지막 자동저장 시각 — 상태바에 "마지막 저장: N분 전" 표시
const lastAutoSaveAt = ref(0)
const _nowTick = ref(0)  // 1분마다 증가 — autoSaveAgoText 재계산 트리거
const autoSaveAgoText = computed(() => {
  // 의존성: lastAutoSaveAt + _nowTick
  void _nowTick.value
  if (!lastAutoSaveAt.value) return ''
  const sec = Math.floor((Date.now() - lastAutoSaveAt.value) / 1000)
  if (sec < 60) return `${sec}초 전`
  if (sec < 3600) return `${Math.floor(sec / 60)}분 전`
  return `${Math.floor(sec / 3600)}시간 전`
})
// 표시 부드럽게 갱신 — 매 분
let _autoSaveTickTimer: ReturnType<typeof setInterval> | null = null
// 복구본은 백엔드 워커가 쓴다(requestEditorAutoSave → editorAutoSaveReady). 레이어 합성(디코드·합성·
// PNG 인코딩·fsync)이 2048² 에 0.6초, 4K 에 1초가 넘어, 예전 동기 슬롯은 5분마다 창을(웹 모드는 모든
// 클라이언트를) 그리는 도중에도 멈췄다. 마지막 요청만 유효 — 응답은 요청 id 로 짝을 맞춘다.
const autoSaveRequest = createLatestRequest<EditorAutoSaveReadyPayload>({ timeoutMs: 120_000, prefix: 'autosave' })
async function _tryAutoSave() {
  if (!isDirty.value || !imagePath.value) return
  if (autoSaveRequest.pending) return   // 앞선 자동저장이 아직 쓰는 중 — 겹쳐 보내지 않는다
  // 확정 이미지만 복사하면 병합 안 한 드로잉(펜·도형·텍스트·그라디언트…)이 복구본에서 빠진다 —
  // 수동 저장처럼 레이어와 불투명도를 함께 보내 백엔드가 합성한다(없으면 빈 문자열 = 그대로 복사).
  const path = stripFileUrl(imagePath.value)
  const overlay: string = canvasRef.value?.getDrawOverlayBase64?.() || ''
  const opacity = Number(drawLayerOpacity.value)
  const gen = _docGen
  const { id, done, outcome } = autoSaveRequest.begin()
  try {
    const backend: any = await getBackend()
    // 백엔드를 기다리는 사이 저장이 끝나 복구본을 폐기했다 — 저장 전 상태를 다시 쓰지 않는다
    if (wasAbandoned(outcome())) return
    if (!backend?.requestEditorAutoSave) { autoSaveRequest.cancel(); return }
    backend.requestEditorAutoSave(path, overlay, Number.isFinite(opacity) ? opacity : 100, id)
  } catch {
    autoSaveRequest.cancel()
    return
  }
  const r = await done
  if (!r) {
    if (outcome() === 'timeout') console.warn('[Editor] auto-save: 응답 없음')
    return   // 폐기(저장 성공)·화면 닫힘은 조용히 끝낸다
  }
  if (r.path) {
    console.log('[Editor] auto-saved →', r.path)
    // 쓰는 사이 다른 문서를 열었으면 그 문서의 '자동저장 N분 전'이 아니다
    if (gen === _docGen) lastAutoSaveAt.value = Date.now()
  } else if (r.error) {
    console.warn('[Editor] auto-save failed:', r.error)
  }
}

/**
 * 복구본을 **작업 사본**(image_cache/editor_temp/recovered_*)으로 연다.
 * 복구 파일(_autosave_session.png)을 그대로 열면 그 파일이 undo 바닥·편집 중 이미지가 되어,
 * 저장 뒤 복구본 정리가 편집 중인 그림을 지우고(이후 편집·저장이 전부 실패) 5분 자동저장이
 * 그 파일을 새 편집으로 덮어써 undo 바닥이 바뀌었다.
 */
function _openRecoveredWork(backend: any) {
  if (!backend?.editorRecoverAutoSave) return
  backend.editorRecoverAutoSave((json: string) => {
    try {
      const r = JSON.parse(json)
      if (!r.path) {
        requestAction('show_toast', { type: 'error', msg: r.error || '복구 실패' })
        return
      }
      loadImage(r.path)
      // 복구한 작업은 아직 어디에도 저장되지 않았다 — 닫기 경고·자동저장이 계속 지켜야 한다
      savedMarker.value = null
      requestAction('show_toast', { type: 'success', msg: '이전 작업 복구됨' })
    } catch {}
  })
}

async function _checkAutoSaveRecovery() {
  try {
    const backend: any = await getBackend()
    if (backend.editorCheckAutoSave) {
      backend.editorCheckAutoSave((json: string) => {
        try {
          const r = JSON.parse(json)
          if (r.path && r.exists) {
            if (window.confirm(
              `이전 세션에 저장되지 않은 작업이 있습니다.\n` +
              `(${r.basename || ''}, ${r.age_minutes || '?'}분 전)\n\n` +
              `복구할까요?`
            )) {
              _openRecoveredWork(backend)
            } else if (backend.editorClearAutoSave) {
              backend.editorClearAutoSave(() => {})
            }
          }
        } catch {}
      })
    }
  } catch {}
}

onMounted(() => {
  onBackendEvent('editorImageLoaded', (path: string) => loadImage(path))
  onBackendEvent('editorWatermarkImageLoaded', (path: string) => { wmImagePath.value = path })
  onBackendEvent('yoloModelUpdated', (label: string) => { modelLabel.value = label })
  onBackendEvent('editorResult', onEditorResult)
  onBackendEvent('editorSaveResult', onEditorSaveResult)
  // 자동저장 결과 — 다른 요청(옛 요청·웹 모드의 다른 탭)의 응답은 요청 id 로 걸러진다
  onBackendEvent('editorAutoSaveReady', (json: string) => { autoSaveRequest.receive(json) })
  // 앱 시작 시 YOLO 모델 자동 감지 + 최근 파일 로드 + 크래시 복구 확인
  refreshYoloLabel()
  _loadRecentFiles()
  _checkAutoSaveRecovery()
  document.addEventListener('keydown', onEditorKeyDown)
  // 5분마다 자동 저장 시도
  _autoSaveTimer = setInterval(_tryAutoSave, AUTO_SAVE_INTERVAL_MS)
  // 60초마다 "N분 전" 표시 갱신 (computed 의존성 _nowTick)
  _autoSaveTickTimer = setInterval(() => { _nowTick.value++ }, 60_000)
  // 사이드 패널 폭이 Settings에서 변경되면 동기화
  window.addEventListener('storage', _syncSidePanelWidthFromStorage)
  window.addEventListener('editorSidePanelWidthChanged', _syncSidePanelWidthFromStorage)
})
onUnmounted(() => {
  document.removeEventListener('keydown', onEditorKeyDown)
  if (_autoSaveTimer) clearInterval(_autoSaveTimer)
  if (_autoSaveTickTimer) clearInterval(_autoSaveTickTimer)
  autoSaveRequest.cancel()
  window.removeEventListener('storage', _syncSidePanelWidthFromStorage)
  window.removeEventListener('editorSidePanelWidthChanged', _syncSidePanelWidthFromStorage)
})
</script>

<style scoped>
.editor-view { width: 100%; height: 100%; display: flex; flex-direction: column; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 12px; background: var(--bg-secondary); flex-shrink: 0;
  border-bottom: 1px solid var(--border);
}
.bar-group { display: flex; align-items: center; gap: 6px; }
.bar-group.center { flex: 1; justify-content: center; }
.bar-btn {
  padding: 8px 16px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-muted); font-size: 12px; font-weight: var(--fw-bold); cursor: pointer; white-space: nowrap;
  transition: var(--transition);
}
.bar-btn:hover { background: var(--bg-button-hover); color: var(--text-primary); border-color: var(--border); }
.bar-btn:disabled { opacity: 0.3; }
.bar-btn.accent { border-color: var(--accent-dim); color: var(--accent); }
/* 주 버튼이라 면은 --accent-fill(글자가 읽히게 민 값), 글자는 --on-accent.
   hover 도 같은 계열이어야 해서 --accent-hover 가 아니라 --accent-fill-hover. */
.bar-btn.save { background: var(--accent-fill); color: var(--on-accent); border: none; font-weight: var(--fw-bold); }
.bar-btn.save:hover { background: var(--accent-fill-hover); }
.bar-btn.danger { color: var(--state-alert-fg); border-color: rgba(248,113,113,0.2); }
.bar-sep { color: var(--border); margin: 0 4px; }
.bar-info { color: var(--text-muted); font-size: 11px; font-family: 'Consolas', monospace; }
.bar-info.autosave { color: var(--state-ok-fg); opacity: 0.75; }
.bar-info.autosave:hover { opacity: 1; cursor: help; }
.bar-filename {
  color: var(--text-secondary); font-size: 12px; font-weight: var(--fw-bold);
  max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  padding: 0 8px;
}
/* 저장 안 된 표시는 오류가 아니라 '주의'라 warn 계열 */
.dirty-mark { color: var(--state-warn-fg); margin-right: 4px; font-size: 14px; vertical-align: middle; }
.bar-counter { color: var(--text-muted); font-size: var(--fs-label); font-family: 'Consolas', monospace; margin-left: 4px; }

.editor-body { flex: 1; display: flex; overflow: hidden; }

.side-panel {
  width: 280px; flex-shrink: 0; background: var(--bg-secondary);
  display: flex; flex-direction: column; overflow: hidden;
}
/* 탭이 6개에서 3개가 되어 한 줄에 들어간다. 글자 10px 은 진단에서 지적된 크기라
   12px 로 올리고 높이도 타격 가능한 32px 로 준다. */
.tab-buttons { display: flex; gap: var(--sp-1); padding: var(--sp-2); background: var(--bg-primary); flex-shrink: 0; }
.tab-btn {
  flex: 1; height: 32px; background: var(--bg-button); border: none; border-radius: var(--radius-base);
  color: var(--text-muted); font-size: var(--fs-meta); font-weight: var(--fw-medium);
  cursor: pointer; white-space: nowrap;
}
.tab-btn:hover { background: var(--bg-button-hover); color: var(--text-primary); }
.tab-btn.active { background: var(--bg-button-hover); color: var(--accent); }
.tab-content { flex: 1; overflow-y: auto; overflow-x: hidden; }
.tab-stack { display: flex; flex-direction: column; }

.drop-area {
  flex: 1; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px;
}
.drop-area.dragging { background: var(--bg-secondary); }
.drop-icon { font-size: 48px; opacity: 0.3; }
/* 제목이 본문보다 밝아야 위계가 산다 — 둘 다 --text-muted 로 눕히지 않는다 */
.drop-area h2 { color: var(--text-secondary); font-size: 20px; }
.drop-area p { color: var(--text-muted); font-size: 13px; }
.drop-actions { display: flex; gap: 8px; }
.open-btn {
  padding: 10px 24px; background: var(--accent-fill); border: none; border-radius: 8px;
  color: var(--on-accent); font-weight: var(--fw-bold); font-size: 14px; cursor: pointer;
}
/* 보조 버튼은 강조색을 '테두리·글자'로만 쓴다 — 면이 아니라서 --accent 그대로 */
.open-btn.secondary { background: var(--bg-button); color: var(--accent); border: 1px solid var(--accent); }
.drop-shortcuts { color: var(--text-muted); font-size: 11px; margin-top: 4px; }
.drop-shortcuts kbd {
  background: var(--bg-button); color: var(--accent); padding: 1px 6px; border-radius: 3px;
  font-family: Consolas, monospace; font-size: var(--fs-label); margin: 0 2px;
}
.recent-files { width: 100%; max-width: 540px; margin-top: 14px; }
.recent-label { color: var(--text-muted); font-size: var(--fs-label); letter-spacing: 0;
  font-weight: var(--fw-bold); padding: 0 4px 6px; }
.recent-list { display: flex; flex-wrap: wrap; gap: 6px; }
.recent-item {
  padding: 6px 12px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-secondary); font-size: 11px; cursor: pointer;
  max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.recent-item:hover { background: var(--bg-button-hover); border-color: var(--accent); color: var(--accent); }
.recent-name { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
.feature-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; justify-content: center; }
.feature-list span { padding: 5px 12px; background: var(--bg-secondary); border-radius: 6px; color: var(--text-muted); font-size: 11px; }
</style>
