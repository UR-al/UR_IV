<template>
  <div class="canvas-container" ref="containerRef"
    @wheel.prevent="onWheel"
    @pointerdown="onMouseDown" @pointermove="onMouseMoveWrap" @pointerup="onMouseUp"
    @pointerleave="onMouseUp" @pointercancel="onMouseUp" @contextmenu.prevent
    @dblclick="onDblClick"
  >
    <canvas ref="canvasEl" :style="baseCanvasStyle" />
    <!-- 실시간 프리뷰 표시 전용. 백엔드 축소본(긴 변 1024)을 원본 크기로 늘려 그린다.
         예전에는 프리뷰를 이미지 자체로 갈아 끼워서, 큰 이미지에서 마스크·드로잉 레이어·
         복원 스냅숏이 전부 초기화됐다. 원본(base) 캔버스는 프리뷰 동안에도 그대로다. -->
    <canvas v-show="previewShown" ref="previewCanvasEl" :style="previewLayerStyle" class="preview-layer" />
    <canvas ref="drawCanvasEl" :style="drawLayerStyle" class="draw-layer" />
    <canvas ref="maskCanvasEl" :style="canvasStyle" class="mask-overlay" />

    <!-- 텍스트 도구: 클릭한 자리에서 바로 입력받는다. 별도 다이얼로그를 띄우면
         어디에 찍힐지 보이지 않아 위치를 짐작해야 한다. -->
    <div v-if="textAnchor" class="text-entry" :style="textEntryStyle">
      <input
        ref="textInputEl"
        v-model="textDraft"
        class="text-entry-input"
        placeholder="텍스트 입력 후 Enter"
        @keydown.enter.stop.prevent="confirmText"
        @keydown.esc.stop.prevent="dismissText"
        @blur="confirmText"
      />
    </div>

    <div class="canvas-info">
      {{ imgWidth }} × {{ imgHeight }}
      <template v-if="hasMask"> | 마스크 활성</template>
      | {{ Math.round(zoom * 100) }}% | {{ Math.round(rotation) }}°
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue'
import { useDrawLayer, type DrawParams } from '../../composables/useDrawLayer'
import { isDrawTool } from '../../utils/drawTools'
import { downscaleMaskNearest, previewDims, shouldDeferPreviewHide } from '../../utils/editorPreview'
import { buildMoveBackground, composeMoveFrame, holeColorFor, type PixelRect } from '../../utils/movePreview'
// 마스크 픽셀 연산·undo 기록·엣지맵은 InpaintView 와 같은 코드를 쓴다(예전엔 복사본이 갈라졌다)
import {
  MaskHistory, appendLassoPoint, dragRect, encodeMaskPng, fillPolygon, fillRect, paintMaskOverlay,
  rectIsApplicable, snapshotsOnPress, stampCircle, strokeLine, type MaskEdit, type Point,
} from '../../utils/maskOps'
import { decodeEdgeMap, snapToEdge as snapToEdgeMap, type EdgeMap } from '../../utils/edgeMap'
import { PristineSource, restoreStrokeMode } from '../../utils/pristineSnapshot'

interface SelectionBounds { x: number; y: number; w: number; h: number }

const props = withDefaults(defineProps<{
  imageSrc?: string
  /** 실시간 프리뷰(data URL). 원본을 바꾸지 않고 그 위에 겹쳐 보여 준다. 빈 문자열이면 숨긴다. */
  previewSrc?: string
  /**
   * 모자이크 지우개의 '적용 전' 그림 — 부모의 pristinePath(복원 커밋이 픽셀을 가져오는 파일) URL.
   * 스냅숏은 이 파일에서만 뜬다(화면과 커밋이 같은 그림을 본다). 빈 문자열이면 되돌릴 그림이 없다.
   */
  pristineSrc?: string
  tool?: string
  brushSize?: number
  eraserMode?: string
  eraserRestore?: boolean
  stampSpacing?: number
  stampShape?: string // 'circle' or 'bar'
  barWidth?: number
  barHeight?: number
  magneticLasso?: boolean
  snapRadius?: number
  /** DrawPanel 이 보내는 그리기 파라미터 (도구·색·크기·투명도·채우기) */
  drawParams?: DrawParams
  /** 드로잉 레이어 표시 투명도 0~100. 병합 전까지는 화면에만 반영된다. */
  layerOpacity?: number
}>(), {
  imageSrc: '',
  previewSrc: '',
  pristineSrc: '',
  tool: 'box',
  brushSize: 20,
  eraserMode: 'brush',
  eraserRestore: false,
  stampSpacing: 30,
  stampShape: 'circle',
  barWidth: 40,
  barHeight: 15,
  magneticLasso: false,
  snapRadius: 12,
  drawParams: () => ({
    tool: 'pen', color: '#000000', size: 3, opacity: 1, filled: false, gradientEnd: '#000000',
  }),
  layerOpacity: 100,
})

const emit = defineEmits<{
  'selection-changed': [bounds: SelectionBounds]
  'mask-changed': [arg: any]
  // 모자이크 지우개 스트로크가 끝났다 — 부모가 백엔드에 커밋해야 파일에 반영된다
  'restore-ready': []
  // 스포이트가 집은 색 — 부모가 DrawPanel 의 현재 색으로 되돌려 준다
  'color-picked': [hex: string]
}>()

const containerRef = ref<HTMLDivElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)
const previewCanvasEl = ref<HTMLCanvasElement | null>(null)
const drawCanvasEl = ref<HTMLCanvasElement | null>(null)
const maskCanvasEl = ref<HTMLCanvasElement | null>(null)
/** 프리뷰 레이어가 보이는지 — 프리뷰 이미지가 실제로 그려진 뒤에만 true */
const previewShown = ref(false)
const textInputEl = ref<HTMLInputElement | null>(null)
const textDraft = ref('')
const imgWidth = ref(0)
const imgHeight = ref(0)
const zoom = ref(1)
const rotation = ref(0)
const panX = ref(0)
const panY = ref(0)
const hasMask = ref(false)

let ctx: CanvasRenderingContext2D | null = null
let maskCtx: CanvasRenderingContext2D | null = null
let sourceImg: HTMLImageElement | null = null
let drawing = false
let panning = false
let startX = 0, startY = 0
let panStartX = 0, panStartY = 0
let lastBrushX = -1, lastBrushY = -1
let lassoPoints: Point[] = []
let maskData: Uint8Array | null = null
let stampAccum = 0
const MAX_MASK_UNDO = 10
const maskHistory = new MaskHistory(MAX_MASK_UNDO)
const maskUndoCount = ref(0)   // 버튼 disabled 반응형 (마스크 undo/redo 가능 여부)
const maskRedoCount = ref(0)
let savedZoom = 1, savedRotation = 0, savedPanX = 0, savedPanY = 0
let pristineImg: HTMLCanvasElement | null = null  // '적용 전' 그림 (모자이크 지우개용, props.pristineSrc 에서 뜬다)
let pristineCtx: CanvasRenderingContext2D | null = null   // getContext 반복 호출 방지
let edgeMap: EdgeMap | null = null   // Canny edge map (자석 올가미용)
let edgeMapToken = 0                 // 늦게 디코드된 옛 엣지맵이 새것(또는 비우기)을 덮지 못하게
// 늦게 디코드된 자동 감지 마스크가 새 문서(또는 비운 뒤·교체된 이미지)에 앉지 못하게
let maskLoadToken = 0

// ── 마스크 오버레이 렌더링 상태 (성능 핵심) ──────────────────────────────────
// 예전에는 pointermove 마다 createImageData(w*h*4) 를 새로 할당하고 maskData 전체를
// 순회했다. 4K(3840×2160)면 이벤트 1건당 830만 회 루프 + 33MB 할당이고, 펜/고주사율
// 마우스는 초당 수백 건을 쏘므로 이게 에디터 버벅임의 주원인이었다.
//
// 바꾼 방식:
//   1) ImageData 를 이미지당 1개만 만들어 재사용 (할당 0)
//   2) 변경된 사각형(dirty rect)만 다시 칠함
//   3) 실제 캔버스 반영은 requestAnimationFrame 으로 1프레임당 1회로 합침
let maskImageData: ImageData | null = null
let maskPixels: Uint32Array | null = null   // maskImageData.data 의 32bit 뷰 (픽셀당 1회 대입)
let dirtyMinX = Infinity, dirtyMinY = Infinity, dirtyMaxX = -Infinity, dirtyMaxY = -Infinity
let overlayFrame = 0            // rAF 핸들
let overlayFullRedraw = false   // 마스크 전체가 갈아엎힌 경우(undo/자동감지 등)
let overlayGuideOnly = false    // 마스크는 그대로, 위에 얹는 가이드만 갱신
let cursorNeedsDraw = false     // 큰 브러시 커서를 오버레이에 직접 그려야 하는지
let cursorX = 0, cursorY = 0

// 마스크 픽셀 색 (RGBA little-endian → ABGR 로 패킹)
// r=226, g=179, b=64, a=80
const MASK_RGBA32 = (80 << 24) | (64 << 16) | (179 << 8) | 226

// 마스크 경계 상자를 증분 추적 — 예전엔 mouseup 마다 w*h 이중 루프로 다시 계산했다
let boundsMinX = Infinity, boundsMinY = Infinity, boundsMaxX = -Infinity, boundsMaxY = -Infinity
let boundsDirty = true          // 지우개로 줄어들 수 있으므로 필요 시 전체 재계산
let maskPixelCount = 0

// ── 영역 이동(MovePanel) 상태 ──
// 드래그 중에는 마스크·스냅숏이 바뀌지 않으므로, 배경과 출력 버퍼는 시작 때 한 번만 만들고
// 프레임마다 바뀐 사각형만 다시 칠한다 (utils/movePreview.ts).
let moveActive = false
let moveDX = 0, moveDY = 0
let moveStartX = 0, moveStartY = 0
let moveSnapshot: ImageData | null = null   // 이동 시작 시점의 화면 픽셀
let moveSrc32: Uint32Array | null = null    // moveSnapshot 의 32비트 뷰
let moveBg32: Uint32Array | null = null     // 원래 자리를 구멍으로 비운 배경
let moveOut: ImageData | null = null        // 재사용하는 출력 버퍼
let moveOut32: Uint32Array | null = null
let moveBBox: PixelRect | null = null       // 옮길 조각의 경계 상자
let moveDest: PixelRect | null = null       // 직전 프레임에 조각이 놓인 자리
let moveNeedsFullPut = false                // 첫 프레임은 구멍까지 통째로 올려야 한다
let moveLastDX = NaN, moveLastDY = NaN

// ── 원근 보정 상태 ──
// 꼭짓점 4개를 드래그해 '원본에서 직사각형이어야 할 영역'을 지정하면
// 백엔드(core/editor_ops.perspective)가 그 사다리꼴을 정직사각형으로 편다.
// 순서는 백엔드 기대값과 동일: 좌상 → 우상 → 우하 → 좌하
let perspectiveActive = false
let perspectivePoints: Point[] = []
let perspectiveDragIdx = -1
const PERSPECTIVE_HANDLE_PX = 9   // 화면 기준 반경 (줌과 무관하게 일정하게 보이도록)
const PERSPECTIVE_LABELS = ['1', '2', '3', '4']

const canvasStyle = computed(() => {
  let cursor = 'crosshair'
  if (panning) cursor = 'grabbing'
  else if (props.tool === 'perspective') cursor = 'move'
  else if (props.tool === 'brush' || props.tool === 'eraser' || props.tool === 'stamp') {
    const rawSize = Math.round(props.brushSize * zoom.value * 2)
    if (rawSize <= 120) {
      // 작은 크기: SVG 커서
      const displaySize = Math.max(6, rawSize)
      const half = displaySize / 2
      const color = props.tool === 'eraser' ? '%23f87171' : props.tool === 'stamp' ? '%2360a5fa' : '%23E2B340'
      const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='${displaySize}' height='${displaySize}'><circle cx='${half}' cy='${half}' r='${half-1}' fill='none' stroke='${color}' stroke-width='1.5'/><line x1='${half}' y1='${half-3}' x2='${half}' y2='${half+3}' stroke='${color}' stroke-width='0.8'/><line x1='${half-3}' y1='${half}' x2='${half+3}' y2='${half}' stroke='${color}' stroke-width='0.8'/></svg>`
      cursor = `url("data:image/svg+xml,${svg}") ${half} ${half}, crosshair`
    } else {
      // 큰 크기: 커서 숨기고 캔버스에 직접 그림 (onMouseMove에서 처리)
      cursor = 'none'
    }
  }
  return {
    transform: `translate(${panX.value}px, ${panY.value}px) scale(${zoom.value}) rotate(${rotation.value}deg)`,
    transformOrigin: 'center center',
    cursor,
  }
})

// ── 드로잉 레이어 ──────────────────────────────────────────────────────────
// 원본 위에 겹치는 별도 캔버스. 병합(flatten)하기 전까지 원본 픽셀은 그대로다.
const drawLayer = useDrawLayer({
  baseCanvas: () => canvasEl.value,
  visibleCanvas: () => drawCanvasEl.value,
  params: () => props.drawParams,
  onColorPicked: (hex) => emit('color-picked', hex),
})
const { textAnchor } = drawLayer

const drawLayerStyle = computed(() => ({
  ...canvasStyle.value,
  opacity: String(Math.max(0, Math.min(100, props.layerOpacity)) / 100),
  // 포인터는 컨테이너가 받는다 — 레이어가 가로채면 마스크 도구가 죽는다
  pointerEvents: 'none' as const,
}))

/** 원본 캔버스 — 프리뷰가 떠 있는 동안에는 가린다. 반투명 이미지에서 프리뷰 밑으로
 *  원본이 비쳐 두 번 겹쳐 보이지 않게. (visibility 라 레이아웃·좌표 변환은 그대로다) */
const baseCanvasStyle = computed(() => ({
  ...canvasStyle.value,
  visibility: previewShown.value ? 'hidden' as const : 'visible' as const,
}))

const previewLayerStyle = computed(() => ({
  ...canvasStyle.value,
  pointerEvents: 'none' as const,
}))

// ── 실시간 프리뷰 레이어 ──────────────────────────────────────────────────────
// 프리뷰는 '보여주기'일 뿐이다. 원본(sourceImg·base 캔버스·마스크·드로잉 레이어·
// pristine 스냅숏)은 건드리지 않고, 축소본을 원본 크기 캔버스에 늘려 그린다.
let previewImg: HTMLImageElement | null = null
let previewToken = 0   // 늦게 디코드된 옛 프리뷰가 새 프리뷰나 '걷기'를 덮지 못하게

function releasePreviewCanvas() {
  const pc = previewCanvasEl.value
  // 4K 캔버스 한 장이 33MB — 숨길 때는 버퍼를 놓아 준다
  if (pc && (pc.width > 1 || pc.height > 1)) { pc.width = 1; pc.height = 1 }
}

function paintPreview() {
  const pc = previewCanvasEl.value
  if (!pc || !sourceImg || !previewImg) { previewShown.value = false; return }
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  if (pc.width !== w || pc.height !== h) { pc.width = w; pc.height = h }
  const pctx = pc.getContext('2d')
  if (!pctx) { previewShown.value = false; return }
  pctx.clearRect(0, 0, w, h)
  pctx.imageSmoothingEnabled = true
  pctx.imageSmoothingQuality = 'high'
  pctx.drawImage(previewImg, 0, 0, w, h)
  previewShown.value = true
}

// 원본 로드 상태 — 프리뷰를 걷을 때 '새 원본이 오는 중인가'를 판단한다(loadNewImage 가 갱신)
let requestedImageSrc = ''     // 마지막으로 로드를 시작한 원본 src
let baseLoadPending = false    // 그 원본이 아직 디코드 중인지
let hidePreviewOnBaseLoad = false   // 걷기를 새 원본이 그려질 때까지 미뤄 둔 상태

function hidePreviewNow() {
  hidePreviewOnBaseLoad = false
  previewImg = null
  previewShown.value = false
  releasePreviewCanvas()
}

function showPreview(src: string) {
  const token = ++previewToken   // 디코드 중인 옛 프리뷰는 어느 쪽이든 버린다
  if (!src) {
    // 확정 결과로 원본이 바뀌는 중이면 새 원본이 그려질 때까지 프리뷰(= 결과와 거의 같은
    // 그림)를 남긴다. 바로 숨기면 base 캔버스의 '적용 전' 원본이 잠깐 비친다.
    if (shouldDeferPreviewHide({
      previewShown: previewShown.value, baseLoadPending,
      imageSrc: props.imageSrc || '', requestedImageSrc,
    })) {
      hidePreviewOnBaseLoad = true
      return
    }
    hidePreviewNow()
    return
  }
  hidePreviewOnBaseLoad = false
  const img = new Image()
  img.onload = () => {
    if (token !== previewToken) return   // 그사이 더 새 프리뷰가 왔거나 걷혔다
    previewImg = img
    paintPreview()
  }
  img.onerror = () => {
    if (token !== previewToken) return
    previewImg = null
    previewShown.value = false
  }
  img.src = src
}

watch(() => props.previewSrc, (src: string) => showPreview(src || ''))

/** 이미지 좌표 → 컨테이너 안의 화면 좌표. `getImagePos` 의 역변환. */
function imageToContainer(x: number, y: number): Point {
  const c = canvasEl.value
  const container = containerRef.value
  if (!c || !container) return { x: 0, y: 0 }
  const rect = c.getBoundingClientRect()
  const box = container.getBoundingClientRect()
  const baseScale = (c.clientWidth || 1) / (c.width || 1)
  const total = baseScale * (zoom.value || 1)
  const lx = x - c.width / 2
  const ly = y - c.height / 2
  const phi = rotation.value * Math.PI / 180
  const cos = Math.cos(phi), sin = Math.sin(phi)
  const sx = (lx * cos - ly * sin) * total
  const sy = (lx * sin + ly * cos) * total
  return {
    x: rect.left + rect.width / 2 + sx - box.left,
    y: rect.top + rect.height / 2 + sy - box.top,
  }
}

const textEntryStyle = computed(() => {
  const anchor = textAnchor.value
  if (!anchor) return { display: 'none' }
  const p = imageToContainer(anchor.x, anchor.y)
  return { left: `${Math.round(p.x)}px`, top: `${Math.round(p.y)}px` }
})

watch(textAnchor, (anchor) => {
  if (anchor) textDraft.value = ''
})

/** 텍스트 입력칸에 포커스를 준다. 호출 시점이 중요해서 함수로 뺐다 — 아래 설명 참고. */
function focusTextEntry() {
  nextTick(() => textInputEl.value?.focus())
}

function confirmText() {
  if (!textAnchor.value) return
  drawLayer.commitText(textDraft.value)
  textDraft.value = ''
}

function dismissText() {
  drawLayer.cancelText()
  textDraft.value = ''
}

// 레이어 투명도는 CSS 로만 반영한다 — 픽셀을 다시 그릴 필요가 없다.
watch(() => props.layerOpacity, () => drawLayer.render())

// ── 이미지 로드 (zoom/rotation 보존 옵션) ──
function loadNewImage(src: string, preserveTransform = false) {
  if (!src) return
  // 교체 전 이미지에 대해 요청한 자동 감지 마스크가 아직 디코드 중이면 버린다(새 이미지에 앉지 않게)
  maskLoadToken++
  if (!preserveTransform) {
    savedZoom = 1; savedRotation = 0; savedPanX = 0; savedPanY = 0
  } else {
    savedZoom = zoom.value; savedRotation = rotation.value
    savedPanX = panX.value; savedPanY = panY.value
  }
  const img = new Image()
  const token = ++imageLoadToken
  requestedImageSrc = src
  baseLoadPending = true
  img.onerror = () => {
    if (token !== imageLoadToken) return
    baseLoadPending = false
    // 원본을 못 읽었다 — 미뤄 둔 프리뷰 걷기를 더 기다릴 이유가 없다
    if (hidePreviewOnBaseLoad) hidePreviewNow()
  }
  img.onload = () => {
    // undo 를 빠르게 두 번 누르는 식으로 교체가 겹치면, 먼저 요청한 쪽이 늦게 디코드돼
    // 나중 이미지를 덮을 수 있다 — 마지막 요청만 반영한다.
    if (token !== imageLoadToken) return
    baseLoadPending = false
    // 확정 결과가 그려지는 바로 이 순간에 미뤄 둔 프리뷰를 걷는다(같은 작업 안이라 번쩍임 없음).
    // drawAll 이 옛 프리뷰를 새 크기로 다시 늘려 그리지 않게 먼저 걷는다.
    if (hidePreviewOnBaseLoad) hidePreviewNow()
    const prevW = sourceImg?.naturalWidth ?? 0
    const prevH = sourceImg?.naturalHeight ?? 0
    sourceImg = img
    imgWidth.value = img.naturalWidth
    imgHeight.value = img.naturalHeight
    zoom.value = savedZoom
    rotation.value = savedRotation
    panX.value = savedPanX
    panY.value = savedPanY

    // 크기가 바뀌었으면(회전/크롭/리사이즈, 또는 다른 이미지) 마스크 버퍼를 새로 잡는다.
    // 예전에는 preserveTransform=true 경로에서 이걸 건너뛰어 stale 마스크가 남았다.
    const sizeChanged = prevW !== img.naturalWidth || prevH !== img.naturalHeight
    if (!preserveTransform || sizeChanged) {
      maskData = new Uint8Array(img.naturalWidth * img.naturalHeight)
      maskImageData = null
      resetMaskBounds()
      hasMask.value = false
      restoreMask = null
      restoreDirty = false
    }

    // pristine('적용 전') 스냅숏은 여기서 뜨지 않는다 — 복원 커밋이 픽셀을 가져오는 pristinePath
    // 파일(props.pristineSrc)에서만 뜬다(loadPristine). 예전에는 로드한 이미지에서 뜨고 효과 직후
    // 한 번만 유지해, 효과를 두 번 적용하면 화면(첫 효과 전 원본)과 커밋(첫 효과 결과)이 갈렸다.

    drawAll()
  }
  img.src = src
}

let imageLoadToken = 0

// ── 모자이크 지우개의 '적용 전' 그림 ──
// 부모의 pristinePath(복원 커밋의 source_path) 파일을 그대로 디코드해 스냅숏으로 쓴다.
// 순서 판정(늦게 끝난 옛 디코드·문서 전환 뒤의 디코드 버리기)은 utils/pristineSnapshot.
const pristineSource = new PristineSource()

function loadPristine(src: string) {
  const token = pristineSource.request(src)
  // 출처가 바뀌었다 — 새 그림이 디코드될 때까지 옛 스냅숏으로 칠하지 않는다(그사이 커밋은 새 출처로 간다)
  pristineImg = null
  pristineCtx = null
  if (!src) return
  const img = new Image()
  img.onload = () => {
    if (!pristineSource.accepts(token)) return   // 더 새 출처가 왔거나 문서가 바뀌었다
    const pc = document.createElement('canvas')
    pc.width = img.naturalWidth; pc.height = img.naturalHeight
    const pctx = pc.getContext('2d', { willReadFrequently: true })
    if (!pctx) { pristineSource.fail(token); return }
    pctx.drawImage(img, 0, 0)
    pristineImg = pc
    pristineCtx = pctx
  }
  // 못 읽으면(문서를 연 채 파일이 지워짐·잠김) 실패로 기록한다 — 지우개는 칠하지 않고 영역만 기록해
  // 커밋을 보내고, 백엔드가 파일을 직접 읽어 복원하거나 '찾을 수 없습니다'를 알린다(restoreStrokeMode 'mark').
  // 예전엔 아무것도 하지 않아 스냅숏을 영영 기다리며('skip') 지우개가 조용히 멈췄다.
  img.onerror = () => { pristineSource.fail(token) }
  img.src = src
}

watch(() => props.pristineSrc, (src: string) => loadPristine(src || ''))

/**
 * 문서가 바뀌었다(열기·닫기) — 옛 문서의 '적용 전' 스냅숏·디코드 중인 출처·커밋 안 한 복원 영역을
 * 버린다. 부모(EditorView._resetDocTransients)가 부른다(부모의 pristinePath 도 함께 비워진다).
 * A 의 pristinePath 디코드가 B 를 연 뒤에 끝나도 받아들이지 않는다 — B 위에 A 의 픽셀을 칠하지 않게.
 */
function resetPristine() {
  pristineSource.reset()
  pristineImg = null
  pristineCtx = null
  clearRestoreMask()
}

watch(() => props.imageSrc, (src: string) => {
  // 효과 적용 후 이미지 교체 시 transform 유지
  const preserve = sourceImg !== null
  loadNewImage(src, preserve)
})

function initMask() {
  if (!sourceImg) return
  const need = sourceImg.naturalWidth * sourceImg.naturalHeight
  if (!maskData || maskData.length !== need) {
    maskData = new Uint8Array(need)
    resetMaskBounds()
    maskImageData = null   // 크기가 바뀌었으니 오버레이 버퍼도 새로 만든다
    markDirtyAll()
  }
}

function resetMaskBounds() {
  boundsMinX = Infinity; boundsMinY = Infinity
  boundsMaxX = -Infinity; boundsMaxY = -Infinity
  boundsDirty = false
  maskPixelCount = 0
}

/** 페인트할 때마다 경계 상자를 넓힌다 (mouseup 마다 전체 스캔하지 않기 위해) */
function growBounds(x1: number, y1: number, x2: number, y2: number) {
  if (x1 < boundsMinX) boundsMinX = x1
  if (y1 < boundsMinY) boundsMinY = y1
  if (x2 > boundsMaxX) boundsMaxX = x2
  if (y2 > boundsMaxY) boundsMaxY = y2
}

/** 변경된 사각형을 dirty 영역에 합친다 */
function markDirty(x1: number, y1: number, x2: number, y2: number) {
  if (x1 < dirtyMinX) dirtyMinX = x1
  if (y1 < dirtyMinY) dirtyMinY = y1
  if (x2 > dirtyMaxX) dirtyMaxX = x2
  if (y2 > dirtyMaxY) dirtyMaxY = y2
  scheduleOverlay()
}

function markDirtyAll() {
  overlayFullRedraw = true
  if (sourceImg) markDirty(0, 0, sourceImg.naturalWidth, sourceImg.naturalHeight)
  else scheduleOverlay()
}

/** 마스크 픽셀은 그대로고 위에 얹는 가이드(원근 사각형/올가미 경로 등)만 바뀐 경우.
 *  maskData→픽셀 변환 루프를 건너뛰고 기존 ImageData를 다시 올리기만 한다.
 *  (4K에서 드래그마다 830만 회 루프를 도는 것을 피하려는 것 — dirty-rect와 같은 이유) */
function markGuideDirty() {
  overlayGuideOnly = true
  scheduleOverlay()
}

/** 실제 캔버스 반영은 프레임당 1회로 합친다 — 포인터 이벤트가 초당 수백 건 와도 안전 */
function scheduleOverlay() {
  if (overlayFrame) return
  overlayFrame = requestAnimationFrame(() => {
    overlayFrame = 0
    flushMaskOverlay()
  })
}

function ensureMaskBuffers(w: number, h: number) {
  if (!maskImageData || maskImageData.width !== w || maskImageData.height !== h) {
    maskImageData = new ImageData(w, h)
    maskPixels = new Uint32Array(maskImageData.data.buffer)
    overlayFullRedraw = true
  }
}

/** dirty 사각형만 다시 칠하고 putImageData 로 그 영역만 올린다 */
function flushMaskOverlay() {
  if (!maskCtx || !maskData || !sourceImg) return
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  ensureMaskBuffers(w, h)
  if (!maskPixels || !maskImageData) return

  // 가이드만 바뀐 프레임 — 픽셀 변환 루프를 건너뛰고 기존 버퍼를 다시 올린다
  if (overlayGuideOnly && !overlayFullRedraw && dirtyMaxX < dirtyMinX) {
    overlayGuideOnly = false
    maskCtx.putImageData(maskImageData, 0, 0)
    drawTransientOverlay()
    return
  }
  overlayGuideOnly = false

  let x1 = Math.max(0, Math.floor(dirtyMinX))
  let y1 = Math.max(0, Math.floor(dirtyMinY))
  let x2 = Math.min(w, Math.ceil(dirtyMaxX))
  let y2 = Math.min(h, Math.ceil(dirtyMaxY))
  if (overlayFullRedraw) { x1 = 0; y1 = 0; x2 = w; y2 = h; overlayFullRedraw = false }

  const hasDirty = x2 > x1 && y2 > y1
  if (hasDirty) {
    paintMaskOverlay(maskPixels, maskData, w, x1, y1, x2, y2, MASK_RGBA32)
    // 마스크 레이어는 전체를 다시 올리지 않고 변경 영역만 갱신
    maskCtx.putImageData(maskImageData, 0, 0, x1, y1, x2 - x1, y2 - y1)
  }
  dirtyMinX = Infinity; dirtyMinY = Infinity
  dirtyMaxX = -Infinity; dirtyMaxY = -Infinity

  hasMask.value = maskPixelCount > 0 || boundsDirty ? computeHasMask() : false

  // 오버레이 위에 임시로 그리는 것들(선택 박스, 올가미 경로, 큰 브러시 커서)
  drawTransientOverlay()
}

/** hasMask 는 화면 표시용이라 정확도보다 비용이 중요 — 카운터로 판단하고,
 *  지우개로 0이 될 수 있는 경우에만 실제로 확인한다. */
function computeHasMask(): boolean {
  if (maskPixelCount > 0) return true
  if (!boundsDirty || !maskData) return false
  return recomputeBounds()
}

/** 지우개로 마스크가 줄어든 뒤 정확한 경계가 필요할 때만 전체 스캔 (mouseup 1회) */
function recomputeBounds(): boolean {
  if (!maskData || !sourceImg) return false
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  let minX = w, minY = h, maxX = -1, maxY = -1, count = 0
  for (let y = 0; y < h; y++) {
    const row = y * w
    let rowHit = false
    for (let x = 0; x < w; x++) {
      if (maskData[row + x] > 0) {
        count++
        rowHit = true
        if (x < minX) minX = x
        if (x > maxX) maxX = x
      }
    }
    if (rowHit) {
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }
  maskPixelCount = count
  boundsDirty = false
  if (maxX < 0) {
    boundsMinX = Infinity; boundsMinY = Infinity
    boundsMaxX = -Infinity; boundsMaxY = -Infinity
    return false
  }
  boundsMinX = minX; boundsMinY = minY
  boundsMaxX = maxX + 1; boundsMaxY = maxY + 1
  return true
}

function drawAll() {
  if (!canvasEl.value || !sourceImg) return
  const c = canvasEl.value
  c.width = sourceImg.naturalWidth
  c.height = sourceImg.naturalHeight
  // willReadFrequently — 모자이크 지우개가 getImageData 를 반복 호출하므로
  // 이 힌트가 없으면 Chromium 이 GPU 경로로 두고 매번 느린 리드백을 한다
  ctx = c.getContext('2d', { willReadFrequently: true })!
  ctx.clearRect(0, 0, c.width, c.height)
  ctx.drawImage(sourceImg, 0, 0)
  const mc = maskCanvasEl.value
  if (mc) {
    mc.width = c.width; mc.height = c.height
    maskCtx = mc.getContext('2d')
    maskImageData = null
    markDirtyAll()
    flushMaskOverlay()
  }
  // 이미지 크기가 바뀌면(회전·자르기·리사이즈·다른 이미지) 레이어도 다시 잡는다.
  // 크기가 같으면 그린 것을 지키기 위해 건드리지 않는다. 실시간 프리뷰는 여기를 지나가지
  // 않는다 — 별도 프리뷰 레이어에만 그려진다(예전에는 축소 프리뷰가 이 경로로 들어와
  // 레이어·마스크·pristine 을 초기화했다).
  if (drawCanvasEl.value
      && (drawCanvasEl.value.width !== c.width || drawCanvasEl.value.height !== c.height)) {
    drawLayer.resize(c.width, c.height)
  } else {
    drawLayer.render()
  }
  // 프리뷰가 떠 있는 채로 원본이 바뀌면 새 크기에 맞춰 다시 늘려 그린다
  if (previewImg) paintPreview()
}

/** 화면 1px이 이미지 좌표로 몇 px인지 — 핸들/선 두께를 줌과 무관하게 유지 */
function imagePerScreenPx(): number {
  const c = canvasEl.value
  if (!c) return 1
  const baseScale = (c.clientWidth || 1) / (c.width || 1)
  const total = baseScale * (zoom.value || 1)
  return total > 0 ? 1 / total : 1
}

/** 매 프레임 오버레이 위에 다시 그려야 하는 일회성 요소들.
 *
 * 여기 hex 는 마스크 오버레이 색이라 **테마 토큰으로 바꾸지 않는다** — 임의의 이미지
 * 위에 얹히는 표시라 배경색과 무관해야 하고, `strokeLasso` 가 색으로 선택/지우기를
 * 구분(`color === '#E2B340'`)하므로 `var(...)` 문자열을 넣으면 비교가 깨진다. */
function drawTransientOverlay() {
  if (!maskCtx) return
  if (perspectiveActive) { drawPerspectiveGuide(); return }
  if (drawing && props.tool === 'box') {
    strokeDashRect(startX, startY, lastBrushX - startX, lastBrushY - startY, '#E2B340')
  } else if (drawing && props.tool === 'eraser' && props.eraserMode === 'box') {
    strokeDashRect(startX, startY, lastBrushX - startX, lastBrushY - startY, '#f87171')
  } else if (drawing && (props.tool === 'lasso'
              || (props.tool === 'eraser' && props.eraserMode === 'lasso'))) {
    strokeLasso(props.tool === 'lasso' ? '#E2B340' : '#f87171')
  }
  if (cursorNeedsDraw) drawBigCursor()
}

function strokeDashRect(x: number, y: number, w: number, h: number, color: string) {
  if (!maskCtx) return
  maskCtx.strokeStyle = color
  maskCtx.lineWidth = 2 / zoom.value
  maskCtx.setLineDash([6 / zoom.value, 4 / zoom.value])
  maskCtx.strokeRect(x, y, w, h)
  maskCtx.setLineDash([])
}

function strokeLasso(color: string) {
  if (!maskCtx || lassoPoints.length < 2) return
  maskCtx.strokeStyle = color
  maskCtx.lineWidth = 2 / zoom.value
  maskCtx.setLineDash([4 / zoom.value, 3 / zoom.value])
  maskCtx.beginPath()
  maskCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y)
  for (let i = 1; i < lassoPoints.length; i++) maskCtx.lineTo(lassoPoints[i].x, lassoPoints[i].y)
  maskCtx.closePath()
  maskCtx.stroke()
  maskCtx.setLineDash([])
  if (color === '#E2B340') {
    maskCtx.fillStyle = 'rgba(226, 179, 64, 0.1)'
    maskCtx.fill()
  }
}

function drawBigCursor() {
  if (!maskCtx) return
  const color = props.tool === 'eraser' ? 'rgba(248,113,113,0.5)'
    : props.tool === 'stamp' ? 'rgba(96,165,250,0.5)' : 'rgba(226,179,64,0.5)'
  maskCtx.strokeStyle = color
  maskCtx.lineWidth = 2
  if (props.tool === 'stamp' && props.stampShape === 'bar') {
    maskCtx.strokeRect(cursorX - props.barWidth / 2, cursorY - props.barHeight / 2,
                       props.barWidth, props.barHeight)
  } else {
    maskCtx.beginPath()
    maskCtx.arc(cursorX, cursorY, props.brushSize, 0, Math.PI * 2)
    maskCtx.stroke()
  }
  maskCtx.lineWidth = 1
  maskCtx.beginPath(); maskCtx.moveTo(cursorX - 5, cursorY); maskCtx.lineTo(cursorX + 5, cursorY); maskCtx.stroke()
  maskCtx.beginPath(); maskCtx.moveTo(cursorX, cursorY - 5); maskCtx.lineTo(cursorX, cursorY + 5); maskCtx.stroke()
}

/** 원근 보정 가이드 — 사각형 + 꼭짓점 핸들 + 3분할 그리드 */
function drawPerspectiveGuide() {
  if (!maskCtx || perspectivePoints.length !== 4) return
  const px = imagePerScreenPx()
  const ctx2 = maskCtx

  // 바깥 영역을 어둡게 — 어디가 펴질 부분인지 한눈에
  if (sourceImg) {
    ctx2.save()
    ctx2.beginPath()
    ctx2.rect(0, 0, sourceImg.naturalWidth, sourceImg.naturalHeight)
    ctx2.moveTo(perspectivePoints[0].x, perspectivePoints[0].y)
    for (let i = 3; i >= 1; i--) ctx2.lineTo(perspectivePoints[i].x, perspectivePoints[i].y)
    ctx2.closePath()
    ctx2.fillStyle = 'rgba(0, 0, 0, 0.45)'
    ctx2.fill('evenodd')
    ctx2.restore()
  }

  // 사각형 외곽선
  ctx2.strokeStyle = '#E2B340'
  ctx2.lineWidth = 2 * px
  ctx2.beginPath()
  ctx2.moveTo(perspectivePoints[0].x, perspectivePoints[0].y)
  for (let i = 1; i < 4; i++) ctx2.lineTo(perspectivePoints[i].x, perspectivePoints[i].y)
  ctx2.closePath()
  ctx2.stroke()

  // 3분할 그리드 — 변을 따라 보간해서 기울기를 눈으로 확인
  ctx2.strokeStyle = 'rgba(226, 179, 64, 0.35)'
  ctx2.lineWidth = 1 * px
  const lerp = (a: Point, b: Point, t: number): Point => ({
    x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t,
  })
  for (const t of [1 / 3, 2 / 3]) {
    const top = lerp(perspectivePoints[0], perspectivePoints[1], t)
    const bottom = lerp(perspectivePoints[3], perspectivePoints[2], t)
    ctx2.beginPath(); ctx2.moveTo(top.x, top.y); ctx2.lineTo(bottom.x, bottom.y); ctx2.stroke()
    const left = lerp(perspectivePoints[0], perspectivePoints[3], t)
    const right = lerp(perspectivePoints[1], perspectivePoints[2], t)
    ctx2.beginPath(); ctx2.moveTo(left.x, left.y); ctx2.lineTo(right.x, right.y); ctx2.stroke()
  }

  // 꼭짓점 핸들
  const r = PERSPECTIVE_HANDLE_PX * px
  for (let i = 0; i < 4; i++) {
    const p = perspectivePoints[i]
    const active = i === perspectiveDragIdx
    ctx2.beginPath()
    ctx2.arc(p.x, p.y, r, 0, Math.PI * 2)
    ctx2.fillStyle = active ? '#E2B340' : 'rgba(20, 20, 20, 0.85)'
    ctx2.fill()
    ctx2.strokeStyle = '#E2B340'
    ctx2.lineWidth = 2 * px
    ctx2.stroke()
    // 순서 번호 (좌상 1 → 시계방향)
    ctx2.fillStyle = active ? '#000' : '#E2B340'
    ctx2.font = `${Math.round(11 * px)}px sans-serif`
    ctx2.textAlign = 'center'
    ctx2.textBaseline = 'middle'
    ctx2.fillText(PERSPECTIVE_LABELS[i], p.x, p.y)
  }
  ctx2.textAlign = 'start'
  ctx2.textBaseline = 'alphabetic'
}

/** 클릭 지점에서 가장 가까운 꼭짓점 인덱스 (히트 반경 안일 때만) */
function hitPerspectiveHandle(x: number, y: number): number {
  const r = PERSPECTIVE_HANDLE_PX * imagePerScreenPx() * 1.8   // 넉넉하게
  let best = -1
  let bestDist = r * r
  for (let i = 0; i < perspectivePoints.length; i++) {
    const dx = perspectivePoints[i].x - x
    const dy = perspectivePoints[i].y - y
    const d = dx * dx + dy * dy
    if (d <= bestDist) { bestDist = d; best = i }
  }
  return best
}

/** 원근 보정 시작 — 이미지 모서리에서 5% 안쪽으로 꼭짓점 4개 배치
 *  (PyQt판 tabs/editor/perspective_dialog.py(은퇴해 삭제됨)와 동일한 초기값) */
function beginPerspective() {
  if (!sourceImg) return
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  const m = 0.05
  perspectivePoints = [
    { x: w * m, y: h * m },
    { x: w * (1 - m), y: h * m },
    { x: w * (1 - m), y: h * (1 - m) },
    { x: w * m, y: h * (1 - m) },
  ]
  perspectiveDragIdx = -1
  perspectiveActive = true
  markDirtyAll()
  flushMaskOverlay()
}

/** 확정 — 꼭짓점 4개를 [[x,y] x4] 로 반환 (백엔드 corners 형식) */
function endPerspective(): number[][] | null {
  if (!perspectiveActive || perspectivePoints.length !== 4) return null
  const corners = perspectivePoints.map(p => [Math.round(p.x), Math.round(p.y)])
  perspectiveActive = false
  perspectivePoints = []
  perspectiveDragIdx = -1
  markDirtyAll()
  flushMaskOverlay()
  return corners
}

function cancelPerspective() {
  perspectiveActive = false
  perspectivePoints = []
  perspectiveDragIdx = -1
  markDirtyAll()
  flushMaskOverlay()
}

/** 하위 호환 alias — 외부/기존 호출부가 쓰던 이름 */
function renderMaskOverlay() {
  markDirtyAll()
  flushMaskOverlay()
}

// ── 좌표 변환: 화면 → 이미지 ──
// FIX 1 (이전): 회전 미반영 — getBoundingClientRect의 AABB로 비율 계산
// FIX 2 (이번): CSS max-width로 캔버스가 축소 표시되는 경우, zoom으로 나누면
//              실제 표시 배율과 안 맞아 클릭 위치가 좌측 위로 어긋남.
//              base scale = clientWidth/canvas.width 로 정확히 계산.
function getImagePos(e: MouseEvent | PointerEvent): Point {
  if (!canvasEl.value) return { x: 0, y: 0 }
  const c = canvasEl.value
  const rect = c.getBoundingClientRect()
  const cx = rect.left + rect.width / 2
  const cy = rect.top + rect.height / 2
  // 1) 화면 좌표 → AABB 중심 기준 상대 벡터
  const dx = e.clientX - cx
  const dy = e.clientY - cy
  // 2) CSS 레이아웃 배율 (max-width 등 반영) + zoom 적용 = 총 화면-내부 배율
  //    clientWidth/Height는 transform 적용 전 CSS 크기 → 안정적
  const baseScale = (c.clientWidth || 1) / (c.width || 1)
  const totalScale = baseScale * (zoom.value || 1)
  if (totalScale === 0) return { x: c.width / 2, y: c.height / 2 }
  const sx = dx / totalScale
  const sy = dy / totalScale
  // 3) 역 회전 (-rotation rad)
  const theta = -rotation.value * Math.PI / 180
  const cos = Math.cos(theta), sin = Math.sin(theta)
  const lx = sx * cos - sy * sin
  const ly = sx * sin + sy * cos
  // 4) 중심 기준 → 캔버스 좌상단 기준 (내부 좌표)
  return {
    x: lx + c.width / 2,
    y: ly + c.height / 2,
  }
}

// ── 변환 초기화 (확대/회전/이동 전부 reset) ──
function resetTransform() {
  zoom.value = 1
  rotation.value = 0
  panX.value = 0
  panY.value = 0
}

// ── Alt 더블클릭: 위치 복귀 ──
function onDblClick(e: MouseEvent) {
  if (e.altKey) resetTransform()
}

function syncMaskHistoryCounts() {
  maskUndoCount.value = maskHistory.undoCount
  maskRedoCount.value = maskHistory.redoCount
}
function saveMaskState() {
  if (maskData) {
    maskHistory.save(maskData)
    syncMaskHistoryCounts()
  }
}
// 마스크를 한 단계 되돌림. 되돌렸으면 true(이미지 undo로 안 넘어가도록), 없으면 false.
// 크기가 다른 스냅샷(이미지가 바뀐 뒤 남은 stale 항목)은 MaskHistory 가 버린다.
function undoMask(): boolean {
  if (!maskData) return false
  const applied = maskHistory.undo(maskData)
  syncMaskHistoryCounts()
  if (!applied) return false
  boundsDirty = true
  recomputeBounds()
  markDirtyAll(); flushMaskOverlay(); emitMaskBounds()
  return true
}
function redoMask(): boolean {
  if (!maskData) return false
  const applied = maskHistory.redo(maskData)
  syncMaskHistoryCounts()
  if (!applied) return false
  boundsDirty = true
  recomputeBounds()
  markDirtyAll(); flushMaskOverlay(); emitMaskBounds()
  return true
}

function onMouseDown(e: PointerEvent) {
  // 포인터 캡처 — 이게 없으면 빠른 드래그가 컨테이너 밖으로 나가는 순간
  // pointerleave 가 떠서 스트로크가 그 자리에서 끊겼다
  try { (e.currentTarget as Element)?.setPointerCapture?.(e.pointerId) } catch {}

  // 알트+드래그는 화면 이동이지만, 클론 스탬프는 알트+클릭으로 복제 원점을 잡는다.
  // 그리기 도구를 쓰는 동안에는 도구가 먼저다 (가운데 버튼 이동은 그대로 둔다).
  if ((e.altKey && !isDrawTool(props.tool)) || e.button === 1) {
    panning = true
    panStartX = e.clientX - panX.value
    panStartY = e.clientY - panY.value
    return
  }

  // ── 드로잉 레이어 도구 ──
  if (isDrawTool(props.tool) && e.button === 0) {
    const pos = getImagePos(e)
    drawLayer.begin(pos.x, pos.y, e)
    return
  }
  // 원근 보정 모드 — 꼭짓점만 잡고 마스킹은 하지 않는다
  if (perspectiveActive) {
    const p = getImagePos(e)
    perspectiveDragIdx = hitPerspectiveHandle(p.x, p.y)
    drawing = perspectiveDragIdx >= 0
    if (drawing) { markGuideDirty(); scheduleOverlay() }
    return
  }

  initMask()
  const pos = getImagePos(e)
  drawing = true
  startX = pos.x; startY = pos.y
  lastBrushX = pos.x; lastBrushY = pos.y
  stampAccum = 0

  // 영역 이동 모드에서는 마스킹 대신 드래그로 옮긴다
  if (moveActive) {
    moveStartX = pos.x - moveDX
    moveStartY = pos.y - moveDY
    return
  }

  // 압력 감응 — 펜 입력일 때만 적용 (마우스는 항상 0.5로 고정되어 의미 없음)
  // 펜 압력 0~1 → 0.3~1.2 배율로 매핑 (최소 30% 보장, 최대 120%)
  const sizeFor = (base: number) => {
    if (e.pointerType === 'pen' && typeof e.pressure === 'number' && e.pressure > 0) {
      return Math.max(2, base * (0.3 + 0.9 * Math.min(1, e.pressure)))
    }
    return base
  }
  const brushR = sizeFor(props.brushSize)

  // maskData가 즉시 바뀌는 도구만 undo 스냅샷 저장 (box/lasso는 mouseup 적용 시 저장 —
  //  빈 클릭/미세 드래그가 빈 undo 단계로 쌓이거나 redo를 날리는 것 방지) — InpaintView 와 같은 정책
  if (snapshotsOnPress(props.tool, props.eraserMode, props.eraserRestore)) {
    saveMaskState()
  }
  if (props.tool === 'lasso') {
    const sp = props.magneticLasso ? snapToEdge(pos.x, pos.y) : pos
    lassoPoints = [{ x: sp.x, y: sp.y }]
  } else if (props.tool === 'brush') {
    paintMaskCircle(pos.x, pos.y, brushR)
    renderMaskOverlay()
  } else if (props.tool === 'stamp') {
    paintStamp(pos.x, pos.y)
    renderMaskOverlay()
  } else if (props.tool === 'eraser') {
    if (props.eraserRestore) {
      restoreCircle(pos.x, pos.y, brushR)
    } else if (props.eraserMode === 'brush') {
      eraseMaskCircle(pos.x, pos.y, brushR)
      renderMaskOverlay()
    } else {
      lassoPoints = props.eraserMode === 'lasso' ? [{ x: pos.x, y: pos.y }] : []
    }
  }
}

function onMouseMoveWrap(e: PointerEvent) {
  // 큰 브러시일 땐 커서를 오버레이에 직접 그린다(네이티브 커서가 120px를 넘으면 잘림).
  // 예전에는 여기서 renderMaskOverlay()를 한 번 더 불러 프레임당 전체 스캔이 2회 돌았다.
  // 이제는 플래그만 세우고 실제 그리기는 rAF 한 번에서 처리한다.
  const rawSize = Math.round(props.brushSize * zoom.value * 2)
  const wantCursor = rawSize > 120
    && (props.tool === 'brush' || props.tool === 'eraser' || props.tool === 'stamp')
  if (wantCursor) {
    const pos = getImagePos(e)
    cursorX = pos.x; cursorY = pos.y
  }
  // 커서를 껐다 켤 때 잔상이 남지 않게 해당 프레임은 전체를 다시 칠한다
  if (cursorNeedsDraw || wantCursor) {
    cursorNeedsDraw = wantCursor
    markGuideDirty()
  }
  onMouseMove(e)
  if (wantCursor) scheduleOverlay()
}

function onMouseMove(e: PointerEvent) {
  // 그리기 도구가 잡고 있는 동안에는 마스크/선택 처리로 내려보내지 않는다.
  // (가운데 버튼 화면 이동은 위에서 이미 panning 을 세웠으므로 그대로 통과한다)
  if (!panning && isDrawTool(props.tool)) {
    const pos = getImagePos(e)
    drawLayer.move(pos.x, pos.y)
    return
  }
  if (panning) {
    panX.value = e.clientX - panStartX
    panY.value = e.clientY - panStartY
    return
  }
  if (perspectiveActive) {
    if (drawing && perspectiveDragIdx >= 0 && sourceImg) {
      const p = getImagePos(e)
      // 이미지 밖으로 나가지 않게 클램프 — warpPerspective가 빈 영역을 만들지 않도록
      perspectivePoints[perspectiveDragIdx] = {
        x: Math.max(0, Math.min(sourceImg.naturalWidth, p.x)),
        y: Math.max(0, Math.min(sourceImg.naturalHeight, p.y)),
      }
      markGuideDirty()
      scheduleOverlay()
    }
    return
  }
  if (moveActive && drawing) {
    const pos = getImagePos(e)
    moveDX = pos.x - moveStartX
    moveDY = pos.y - moveStartY
    renderMovePreview()
    return
  }
  if (!drawing) return
  const pos = getImagePos(e)

  if (props.tool === 'box') {
    // 점선 사각형은 매 프레임 drawTransientOverlay()가 그린다 —
    // 여기서는 좌표만 갱신하고 전체 스캔은 하지 않는다
    lastBrushX = pos.x; lastBrushY = pos.y
    markGuideDirty()
  } else if (props.tool === 'lasso') {
    // 직전 점과 1px 미만이면 버린다 — 제자리 이벤트(특히 같은 엣지에 붙는 자석)가 꼭짓점을 불리지 않게
    const sp = props.magneticLasso ? snapToEdge(pos.x, pos.y) : pos
    if (appendLassoPoint(lassoPoints, sp)) markGuideDirty()
  } else if (props.tool === 'brush') {
    // 펜 압력에 따라 브러시 반경 동적 조정 (마우스는 props.brushSize 그대로)
    let brushR = props.brushSize
    if (e.pointerType === 'pen' && typeof e.pressure === 'number' && e.pressure > 0) {
      brushR = Math.max(2, props.brushSize * (0.3 + 0.9 * Math.min(1, e.pressure)))
    }
    paintMaskLine(lastBrushX, lastBrushY, pos.x, pos.y, brushR)
    lastBrushX = pos.x; lastBrushY = pos.y
  } else if (props.tool === 'stamp') {
    // STAMP: 일정 간격마다 원형 마스킹
    const dx = pos.x - lastBrushX, dy = pos.y - lastBrushY
    const dist = Math.sqrt(dx * dx + dy * dy)
    stampAccum += dist
    if (stampAccum >= props.stampSpacing) {
      paintStamp(pos.x, pos.y)
      stampAccum = 0
    }
    lastBrushX = pos.x; lastBrushY = pos.y
  } else if (props.tool === 'eraser') {
    if (props.eraserRestore) {
      restoreLine(lastBrushX, lastBrushY, pos.x, pos.y, props.brushSize)
      lastBrushX = pos.x; lastBrushY = pos.y
    } else if (props.eraserMode === 'brush') {
      eraseMaskLine(lastBrushX, lastBrushY, pos.x, pos.y, props.brushSize)
      lastBrushX = pos.x; lastBrushY = pos.y
    } else if (props.eraserMode === 'box') {
      lastBrushX = pos.x; lastBrushY = pos.y
      markGuideDirty()
    } else if (props.eraserMode === 'lasso') {
      if (appendLassoPoint(lassoPoints, pos)) markGuideDirty()
    }
  }
}

function onMouseUp(e: PointerEvent) {
  try { (e.currentTarget as Element)?.releasePointerCapture?.(e.pointerId) } catch {}
  if (panning) { panning = false; return }
  if (isDrawTool(props.tool)) {
    drawLayer.end()
    // 텍스트 입력칸 포커스는 포인터 조작이 **끝난 뒤**에 준다. pointerdown 직후에 주면
    // 이어지는 pointerup 이 포커스를 body 로 걷어가고, 그 blur 가 입력칸을 즉시 닫는다.
    if (textAnchor.value) focusTextEntry()
    return
  }
  if (perspectiveActive) {
    drawing = false
    perspectiveDragIdx = -1
    markGuideDirty(); scheduleOverlay()
    return
  }
  if (!drawing) return
  drawing = false
  if (moveActive) return   // 이동 미리보기는 확정/취소 때 정리한다

  // 모자이크 지우개는 화면 캔버스만 바꾼다 — 파일에 남기려면 부모가 커밋해야 한다
  if (props.tool === 'eraser' && props.eraserRestore && restoreDirty) {
    emit('restore-ready')
    return
  }
  const pos = getImagePos(e)

  if (props.tool === 'box') {
    const r = dragRect(startX, startY, pos.x, pos.y)
    if (rectIsApplicable(r)) { saveMaskState(); fillMaskRect(r.x1, r.y1, r.x2, r.y2) }
  } else if (props.tool === 'lasso') {
    if (lassoPoints.length > 2) { saveMaskState(); fillMaskPolygon(lassoPoints) }
    lassoPoints = []
  } else if (props.tool === 'eraser') {
    if (props.eraserMode === 'box') {
      const r = dragRect(startX, startY, pos.x, pos.y)
      if (rectIsApplicable(r)) { saveMaskState(); eraseMaskRect(r.x1, r.y1, r.x2, r.y2) }
    } else if (props.eraserMode === 'lasso') {
      if (lassoPoints.length > 2) { saveMaskState(); eraseMaskPolygon(lassoPoints) }
      lassoPoints = []
    }
  }
  // 지운 뒤에는 경계가 부정확할 수 있으니 여기서 한 번만 정확히 다시 잡는다
  if (boundsDirty) recomputeBounds()
  markGuideDirty()
  scheduleOverlay()
  emitMaskBounds()
}

// ── 마스크 조작 ──
// 픽셀 연산은 utils/maskOps(InpaintView 와 공용)가 하고, 여기서는 결과(건드린 사각형·켜진 픽셀 수
// 변화)로 경계 상자·dirty 영역·픽셀 카운터만 맞춘다.
function maskBuffer() {
  return maskData && sourceImg ? { data: maskData, w: sourceImg.naturalWidth, h: sourceImg.naturalHeight } : null
}
/** 칠한 결과 반영 — 켜기는 경계 상자를 넓힌다 */
function commitPaint(edit: MaskEdit) {
  maskPixelCount += edit.delta
  growBounds(edit.x1, edit.y1, edit.x2, edit.y2)
  markDirty(edit.x1, edit.y1, edit.x2, edit.y2)
}
/** 지운 결과 반영 — 경계가 줄어들 수 있으니 정확한 경계는 mouseup 때 한 번만 재계산 */
function commitErase(edit: MaskEdit) {
  maskPixelCount += edit.delta
  boundsDirty = true
  markDirty(edit.x1, edit.y1, edit.x2, edit.y2)
}
function paintMaskBar(cx: number, cy: number, bw: number, bh: number) {
  const mask = maskBuffer(); if (!mask) return
  commitPaint(fillRect(mask, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, true))
}

function paintStamp(cx: number, cy: number) {
  if (props.stampShape === 'bar') paintMaskBar(cx, cy, props.barWidth, props.barHeight)
  else if (props.stampShape === 'rect') paintMaskBar(cx, cy, props.brushSize * 2, props.brushSize * 2)
  else paintMaskCircle(cx, cy, props.brushSize)
}

function paintMaskCircle(cx: number, cy: number, r: number) {
  const mask = maskBuffer(); if (!mask) return
  commitPaint(stampCircle(mask, cx, cy, r, true))
}
function paintMaskLine(x0: number, y0: number, x1: number, y1: number, r: number) {
  const mask = maskBuffer(); if (!mask) return
  commitPaint(strokeLine(mask, x0, y0, x1, y1, r, true))
}
function eraseMaskCircle(cx: number, cy: number, r: number) {
  const mask = maskBuffer(); if (!mask) return
  commitErase(stampCircle(mask, cx, cy, r, false))
}
function eraseMaskLine(x0: number, y0: number, x1: number, y1: number, r: number) {
  const mask = maskBuffer(); if (!mask) return
  commitErase(strokeLine(mask, x0, y0, x1, y1, r, false))
}
function fillMaskRect(x1: number, y1: number, x2: number, y2: number) {
  const mask = maskBuffer(); if (!mask) return
  commitPaint(fillRect(mask, x1, y1, x2, y2, true))
}
function eraseMaskRect(x1: number, y1: number, x2: number, y2: number) {
  const mask = maskBuffer(); if (!mask) return
  commitErase(fillRect(mask, x1, y1, x2, y2, false))
}
// 올가미 채우기는 스캔라인(maskOps.fillPolygon) — 예전 bbox 전 픽셀 × 꼭짓점 point-in-polygon 은
// 긴 올가미(수백 점)·큰 이미지에서 mouseup 한 번에 수 초씩 멈췄다. 결과는 픽셀 단위로 같다.
function fillMaskPolygon(pts: Point[]) {
  const mask = maskBuffer(); if (!mask || pts.length < 3) return
  commitPaint(fillPolygon(mask, pts, true))
}
function eraseMaskPolygon(pts: Point[]) {
  const mask = maskBuffer(); if (!mask || pts.length < 3) return
  commitErase(fillPolygon(mask, pts, false))
}

// ── 모자이크 지우개 (원본 복원) ──────────────────────────────────────────────
// 복원한 픽셀은 화면 캔버스에만 있으면 저장에 반영되지 않는다(저장은 파일 경로 기반).
// 그래서 복원 영역을 restoreMask 에 기록해 두고, 지우개를 뗄 때 부모가
// getRestoreMaskBase64() 로 가져가 백엔드에 'restore' 로 커밋한다.
let restoreMask: Uint8Array | null = null
let restoreDirty = false

function ensureRestoreMask() {
  if (!sourceImg) return
  const need = sourceImg.naturalWidth * sourceImg.naturalHeight
  if (!restoreMask || restoreMask.length !== need) restoreMask = new Uint8Array(need)
}

/** 한 스트로크 구간을 한 번의 getImageData/putImageData 로 처리.
 *  예전에는 보간 스텝마다(수십 회) 리드백을 해서 지우개가 가장 느린 도구였다. */
function restoreLine(x0: number, y0: number, x1: number, y1: number, r: number) {
  if (!ctx || !sourceImg) return
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  const radius = Math.max(1, r)
  // 'paint': pristinePath 의 그림을 칠한다. 'mark': 칠하지 않고 영역만 기록해 커밋을 부모에 맡긴다 —
  // 되돌릴 그림이 아예 없으면 부모가 '되돌릴 이전 상태가 없습니다'를, pristinePath 디코드가 실패했으면
  // 백엔드가 직접 읽어 복원하거나 '찾을 수 없습니다'를 알린다. 'skip': 스냅숏 디코드 중이거나 크기가
  // 다르다(회전/크롭 후 — 백엔드도 거절한다).
  const mode = restoreStrokeMode(pristineSource.requested, pristineImg, { width: w, height: h },
    pristineSource.failed)
  if (mode === 'skip') return

  const bx1 = Math.max(0, Math.floor(Math.min(x0, x1) - radius))
  const by1 = Math.max(0, Math.floor(Math.min(y0, y1) - radius))
  const bx2 = Math.min(w, Math.ceil(Math.max(x0, x1) + radius))
  const by2 = Math.min(h, Math.ceil(Math.max(y0, y1) + radius))
  const sw = bx2 - bx1, sh = by2 - by1
  if (sw <= 0 || sh <= 0) return

  ensureRestoreMask()
  const src = mode === 'paint' && pristineCtx ? pristineCtx.getImageData(bx1, by1, sw, sh) : null
  const dst = src ? ctx.getImageData(bx1, by1, sw, sh) : null

  const dist = Math.hypot(x1 - x0, y1 - y0)
  const steps = Math.max(1, Math.ceil(dist / Math.max(1, radius * 0.3)))
  const rr = radius * radius

  for (let s = 0; s <= steps; s++) {
    const t = steps === 0 ? 0 : s / steps
    const cx = x0 + (x1 - x0) * t
    const cy = y0 + (y1 - y0) * t
    const px1 = Math.max(bx1, Math.floor(cx - radius))
    const py1 = Math.max(by1, Math.floor(cy - radius))
    const px2 = Math.min(bx2, Math.ceil(cx + radius))
    const py2 = Math.min(by2, Math.ceil(cy + radius))
    for (let py = py1; py < py2; py++) {
      const dy = py - cy
      const dy2 = dy * dy
      const rowOff = (py - by1) * sw
      for (let px = px1; px < px2; px++) {
        const dx = px - cx
        if (dx * dx + dy2 > rr) continue
        if (src && dst) {
          const i = (rowOff + (px - bx1)) * 4
          dst.data[i] = src.data[i]
          dst.data[i + 1] = src.data[i + 1]
          dst.data[i + 2] = src.data[i + 2]
          dst.data[i + 3] = src.data[i + 3]
        }
        if (restoreMask) restoreMask[py * w + px] = 255
      }
    }
  }
  if (dst) ctx.putImageData(dst, bx1, by1)
  restoreDirty = true
}

function restoreCircle(cx: number, cy: number, r: number) {
  restoreLine(cx, cy, cx, cy, r)
}

/** 복원 영역을 흑백 PNG 마스크로 — 백엔드가 pristine 픽셀을 되돌리는 데 쓴다 */
function getRestoreMaskBase64(): string | null {
  if (!restoreDirty || !restoreMask || !sourceImg) return null
  return encodeMaskPng(restoreMask, sourceImg.naturalWidth, sourceImg.naturalHeight)
}

function hasPendingRestore(): boolean { return restoreDirty }

function clearRestoreMask() {
  if (restoreMask) restoreMask.fill(0)
  restoreDirty = false
}

// ── 자석 올가미: edge map 로드 + snap (디코드·스냅은 utils/edgeMap — InpaintView 와 공용) ──
function loadEdgeMap(b64: string) {
  if (!b64) return
  const token = ++edgeMapToken
  void decodeEdgeMap(b64).then((decoded) => {
    if (token === edgeMapToken && decoded) edgeMap = decoded
  })
}

/** 엣지맵을 버린다 — 이미지가 바뀌면 부모가 부른다(옛 이미지 윤곽에 붙지 않게). */
function clearEdgeMap() {
  edgeMapToken++
  edgeMap = null
}

function snapToEdge(x: number, y: number): Point {
  return snapToEdgeMap(props.magneticLasso ? edgeMap : null, x, y, props.snapRadius)
}

function emitMaskBounds() {
  const sel = getSelection()
  if (sel) emit('selection-changed', sel)
  hasMask.value = sel !== null
}

function onWheel(e: WheelEvent) {
  if (e.shiftKey) { rotation.value += e.deltaY > 0 ? 5 : -5 }
  else { zoom.value = Math.max(0.1, Math.min(10, zoom.value * (e.deltaY > 0 ? 0.9 : 1.1))) }
}

function clearSelection(resetHistory = false) {
  // 디코드 중인 자동 감지 마스크도 버린다 — 비운 직후(새 문서·이미지 작업·Esc) 늦게 앉지 않게
  maskLoadToken++
  if (maskData) maskData.fill(0)
  resetMaskBounds()
  hasMask.value = false; lassoPoints = []
  // 이미지 작업/새 이미지 로드 후에만(resetHistory=true) 마스크 히스토리 리셋 — stale 마스크
  // undo가 이미지 undo를 가리지 않게. Esc/취소(기본 false)는 보존 → Ctrl+Z로 마스크 복구 가능.
  if (resetHistory) {
    maskHistory.clear()
    syncMaskHistoryCounts()
  }
  markDirtyAll()
  flushMaskOverlay()
}

/** 마스크 경계 상자. 증분 추적한 값을 쓰고, 지우개 이후에만 한 번 재계산한다.
 *  (예전에는 호출할 때마다 w*h 이중 루프를 돌았다 — 4K에서 830만 회) */
function getSelection(): SelectionBounds | null {
  if (!maskData || !sourceImg) return null
  if (boundsDirty) recomputeBounds()
  if (boundsMaxX <= boundsMinX || boundsMaxY <= boundsMinY) return null
  return {
    x: boundsMinX, y: boundsMinY,
    w: boundsMaxX - boundsMinX, h: boundsMaxY - boundsMinY,
  }
}

/**
 * 마스크를 흑백 PNG 로. 칠한 곳이 없으면 null — 예전에는 이미지를 열기만 해도 maskData 가
 * 있어서 전면 검은 PNG 를 돌려줬고, '마스크 있음' 가드가 늘 참이었다.
 *
 * `maxEdge` 를 주면 백엔드 프리뷰와 같은 크기로 먼저 줄인다(최근접). 프리뷰는 백엔드가
 * 어차피 그 크기로 줄여 쓰므로, 원본 해상도 PNG 인코딩(4K 에서 틱마다 수십 ms)이 낭비다.
 */
function getMaskBase64(opts: { maxEdge?: number } = {}): string | null {
  if (!maskData || !sourceImg) return null
  if (!getSelection()) return null
  let w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  let data = maskData
  if (opts.maxEdge) {
    const d = previewDims(w, h, opts.maxEdge)
    if (d.w !== w || d.h !== h) {
      data = downscaleMaskNearest(maskData, w, h, d.w, d.h)
      w = d.w; h = d.h
    }
  }
  // 32bit 뷰로 픽셀당 1회 대입 (0xAABBGGRR, little-endian) — InpaintView 와 같은 인코더
  return encodeMaskPng(data, w, h)
}

// 외부에서 마스크 로드 (YOLO/SAM3 auto-detect 결과)
// data URL 디코드는 비동기다. 결과가 도착한 뒤(부모의 문서 세대 게이트를 통과한 뒤) 디코드가 끝나기
// 전에 다른 문서를 열거나 이미지가 바뀌면, 예전에는 onload 가 그 새 이미지 위에 옛 문서의 마스크를
// 썼다(같은 크기면 그대로, 다르면 늘려서). 요청 시점의 토큰·원본과 다르면 버린다.
function loadMaskFromBase64(b64: string) {
  if (!sourceImg) return
  const token = ++maskLoadToken          // 더 새 마스크 요청·비우기·이미지 교체가 오면 무효
  const imgToken = imageLoadToken        // 이 뒤에 시작한 원본 로드(새 문서·작업 결과·undo)가 있으면 무효
  const forImg = sourceImg               // 원본 자체가 바뀌었으면 무효
  const stale = () => token !== maskLoadToken || imgToken !== imageLoadToken || sourceImg !== forImg
  const img = new Image()
  img.onload = () => {
    if (stale() || !sourceImg) return
    const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
    const tc = document.createElement('canvas'); tc.width = w; tc.height = h
    const tctx = tc.getContext('2d', { willReadFrequently: true })!
    tctx.drawImage(img, 0, 0, w, h)
    const id = tctx.getImageData(0, 0, w, h)
    initMask()
    if (!maskData) return
    saveMaskState()   // 자동 감지 마스크도 undo 한 단계로 (통합 undo에서 마스크 우선 되돌림)
    // maskData.length 로 돌면 캔버스 크기와 어긋날 수 있어 w*h 기준으로 순회
    const n = Math.min(maskData.length, w * h)
    let count = 0
    for (let i = 0; i < n; i++) {
      const on = id.data[i * 4] > 127
      maskData[i] = on ? 255 : 0
      if (on) count++
    }
    maskPixelCount = count
    boundsDirty = true
    recomputeBounds()
    markDirtyAll()
    flushMaskOverlay()
    emitMaskBounds()
  }
  img.src = b64
}

/** 디코드 중인 자동 감지 마스크를 버린다 — 문서가 바뀔 때(열기·닫기) 부모가 부른다. */
function cancelMaskLoad() {
  maskLoadToken++
}

// ── 영역 이동 미리보기 ──────────────────────────────────────────────────────
// 확정 전까지는 화면에서만 옮겨 보여주고, 실제 픽셀 연산은 백엔드 move_region이 한다.
// 배경(구멍 뚫린 스냅숏)과 출력 버퍼는 여기서 한 번만 만든다 — 드래그 중엔 안 바뀐다.
/** @param fillColor 구멍 채우기 색('black'|'white') — 확정 결과(move_region)와 같은 색으로 보여 준다 */
function beginMove(fillColor = 'black') {
  if (!ctx || !sourceImg || !maskData) return
  const sel = getSelection()
  if (!sel) return
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  moveActive = true
  moveDX = 0; moveDY = 0
  moveSnapshot = ctx.getImageData(0, 0, w, h)
  moveSrc32 = new Uint32Array(moveSnapshot.data.buffer)
  moveBBox = { x: sel.x, y: sel.y, w: sel.w, h: sel.h }
  moveBg32 = buildMoveBackground(moveSrc32, maskData, w, h, moveBBox, holeColorFor(fillColor))
  moveOut = new ImageData(w, h)
  moveOut32 = new Uint32Array(moveOut.data.buffer)
  moveOut32.set(moveBg32)
  moveDest = null
  moveNeedsFullPut = true
  moveLastDX = NaN; moveLastDY = NaN
}

function renderMovePreview() {
  if (!ctx || !moveOut || !moveOut32 || !moveBg32 || !moveSrc32 || !moveBBox || !maskData || !sourceImg) return
  const w = sourceImg.naturalWidth, h = sourceImg.naturalHeight
  const dx = Math.round(moveDX), dy = Math.round(moveDY)
  if (!moveNeedsFullPut && dx === moveLastDX && dy === moveLastDY) return   // 같은 자리 — 할 일 없음
  const frame = composeMoveFrame(moveOut32, moveBg32, moveSrc32, maskData, w, h, moveBBox, moveDest, dx, dy)
  moveDest = frame.dest
  moveLastDX = dx; moveLastDY = dy
  if (moveNeedsFullPut) {
    // 첫 프레임 — 원래 자리의 구멍까지 캔버스에 올라가야 한다
    ctx.putImageData(moveOut, 0, 0)
    moveNeedsFullPut = false
    return
  }
  // 바뀐 사각형(직전 조각 자리 + 새 조각 자리)만 올린다
  for (const r of frame.dirty) ctx.putImageData(moveOut, 0, 0, r.x, r.y, r.w, r.h)
}

function resetMoveBuffers() {
  moveActive = false
  moveSnapshot = null
  moveSrc32 = null; moveBg32 = null
  moveOut = null; moveOut32 = null
  moveBBox = null; moveDest = null
  moveNeedsFullPut = false
  moveDX = 0; moveDY = 0
}

function endMove(): { dx: number; dy: number } {
  const result = { dx: Math.round(moveDX), dy: Math.round(moveDY) }
  resetMoveBuffers()
  return result
}

function cancelMove() {
  if (ctx && moveSnapshot) ctx.putImageData(moveSnapshot, 0, 0)
  resetMoveBuffers()
}

// zoom/rotation 초기화
function resetView() { resetTransform() }  // 하위 호환 alias

defineExpose({
  clearSelection, getSelection, getMaskBase64, loadMaskFromBase64, cancelMaskLoad, loadEdgeMap, clearEdgeMap,
  drawAll, resetView, resetTransform, undoMask, redoMask, maskUndoCount, maskRedoCount,
  // 모자이크 지우개 커밋용 — 화면에만 있던 복원을 백엔드에 반영하기 위해
  getRestoreMaskBase64, hasPendingRestore, clearRestoreMask, resetPristine,
  // 영역 이동
  beginMove, endMove, cancelMove,
  // 원근 보정
  beginPerspective, endPerspective, cancelPerspective,
  // 드로잉 레이어 — 병합·복원 브러시·되돌리기
  getDrawOverlayBase64: drawLayer.getOverlayBase64,
  getHealMaskBase64: drawLayer.getHealMaskBase64,
  clearDrawLayer: drawLayer.clear,
  clearHealMask: drawLayer.clearHeal,
  undoDrawStroke: drawLayer.undo,
  drawUndoCount: drawLayer.undoCount,
  hasDrawContent: drawLayer.hasContent,
  hasHealMask: drawLayer.hasHeal,
  // 레이어 내용 리비전 — 부모가 '저장 뒤 레이어가 바뀌었는지'(미저장 변경)를 판단한다
  drawRevision: drawLayer.revision,
})

onMounted(() => {
  if (props.imageSrc) loadNewImage(props.imageSrc, false)
  if (props.pristineSrc) loadPristine(props.pristineSrc)
})

onBeforeUnmount(() => {
  // rAF 핸들 정리 — 탭을 떠난 뒤에도 프레임이 돌면 누수가 된다
  if (overlayFrame) { cancelAnimationFrame(overlayFrame); overlayFrame = 0 }
})
</script>

<style scoped>
.canvas-container {
  width: 100%; height: 100%; position: relative;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden; background: var(--bg-secondary);
}
/* 90% 제한은 고정폭 사이드패널과 겹쳐 이미지가 창의 약 59%만 쓰게 했다.
   baseScale(=clientWidth/width)이 실측값을 읽으므로 좌표 변환은 그대로 성립한다. */
canvas { max-width: 100%; max-height: 100%; position: absolute; }
.mask-overlay { pointer-events: none; }
.preview-layer { pointer-events: none; }
.draw-layer { pointer-events: none; }
/* 텍스트 도구 입력칸 — 찍힐 자리에 그대로 뜬다 */
.text-entry {
  position: absolute;
  z-index: 3;
  transform: translateY(-2px);
}
.text-entry-input {
  min-width: 160px;
  padding: 4px 8px;
  background: rgba(0, 0, 0, 0.85);
  /* 바탕이 이미지 위에 얹히는 고정 검정 오버레이라 테마를 따라가면 안 된다 —
     라이트에서 글자만 어두워지면 검은 알약 위에서 안 읽힌다. */
  color: #fff;
  border: 1px solid var(--edge, #666);
  border-radius: 4px;
  font-size: 13px;
  outline: none;
}
.canvas-info {
  position: absolute; bottom: 8px; right: 12px;
  /* 위와 같은 이유 — 고정 검정 알약 위의 글자라 테마와 무관하다 */
  color: #585858; font-size: 11px;
  background: rgba(0,0,0,0.6); padding: 2px 8px; border-radius: 4px;
  pointer-events: none; z-index: 2;
}
</style>
