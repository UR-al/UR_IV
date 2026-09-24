<template>
  <div class="inpaint-workspace">
    <!-- 마스크 도구 — 에디터와 같은 세로 툴바를 쓴다. 두 탭에서 같은 도구가
         다르게 생기거나 다른 키로 잡히면 손이 헷갈린다. -->
    <EditorToolbar :model-value="currentTool" :tools="INPAINT_TOOLS" @select="currentTool = $event" />

    <!-- Left Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <div class="glass-card">
          <label>원본 이미지</label>
          <div class="source-thumb" @click="triggerFileInput">
            <img v-if="imageSrc" :src="imageSrc" />
            <div v-else class="upload-hint">끌어 놓거나 클릭</div>
          </div>
        </div>

        <!-- 고른 도구의 이름 — 아이콘만 있는 툴바를 보완한다 -->
        <div class="tool-head">
          <span class="tool-name">{{ currentToolLabel }}</span>
          <span class="tool-key">{{ currentToolKey }}</span>
        </div>

        <!-- 올가미 모드 -->
        <div class="glass-card" v-if="currentTool === 'lasso'">
          <label>올가미 모드</label>
          <div class="tool-grid small">
            <button class="tool-chip" :class="{ active: !magneticLasso }" @click="magneticLasso = false"><Icon name="loop" /> 자유</button>
            <button class="tool-chip magnet" :class="{ active: magneticLasso }" @click="enableMagnetic"><Icon name="magnet" /> 자석</button>
          </div>
        </div>

        <!-- 지우개 모드 -->
        <div class="glass-card" v-if="currentTool === 'eraser'">
          <label>지우개 모양</label>
          <div class="tool-grid small">
            <button class="tool-chip" :class="{ active: eraserMode === 'brush' }" @click="eraserMode = 'brush'">브러시</button>
            <button class="tool-chip" :class="{ active: eraserMode === 'box' }" @click="eraserMode = 'box'">사각형</button>
            <button class="tool-chip" :class="{ active: eraserMode === 'lasso' }" @click="eraserMode = 'lasso'">올가미</button>
          </div>
        </div>

        <!-- 브러시 크기 -->
        <div class="glass-card">
          <label>브러시 크기</label>
          <div class="slider-row">
            <input type="range" min="3" max="200" v-model.number="brushSize" />
            <span class="slider-val">{{ brushSize }}px</span>
          </div>
        </div>

        <div class="glass-card">
          <label>Denoising</label>
          <div class="slider-row">
            <input type="range" min="0" max="1" step="0.01" v-model.number="denoising" />
            <span class="slider-val">{{ denoising.toFixed(2) }}</span>
          </div>
        </div>

        <div class="glass-card">
          <label>프롬프트 덮어쓰기</label>
          <textarea v-model="prompt" rows="3" placeholder="Describe the change..."></textarea>
          <label class="mt-6">네거티브 덮어쓰기</label>
          <textarea v-model="negPrompt" rows="2" placeholder="비워두면 T2I 네거티브 사용"></textarea>
        </div>

        <div class="glass-card">
          <label>마스크 설정</label>
          <CustomSelect v-model="maskContentLabel" :options="maskContents" placeholder="마스크 영역 초기값" />
          <CustomSelect v-model="inpaintAreaLabel" :options="inpaintAreas" placeholder="인페인트 범위" class="mt-6" />
          <label class="mt-6">마스크 블러</label>
          <div class="slider-row">
            <input type="range" min="0" max="64" v-model.number="maskBlur" />
            <span class="slider-val">{{ maskBlur }}px</span>
          </div>
          <template v-if="inpaintArea === 1">
            <label class="mt-6">마스크 영역 여백</label>
            <div class="slider-row">
              <input type="range" min="0" max="256" step="4" v-model.number="padding" />
              <span class="slider-val">{{ padding }}px</span>
            </div>
          </template>
        </div>

        <details class="glass-card">
          <summary class="card-header">고급 설정</summary>
          <label class="mt-6">스텝</label>
          <div class="slider-row">
            <input type="range" min="1" max="150" v-model.number="steps" />
            <span class="slider-val">{{ steps }}</span>
          </div>
          <label class="mt-6">CFG</label>
          <div class="slider-row">
            <input type="range" min="1" max="30" step="0.5" v-model.number="cfg" />
            <span class="slider-val">{{ cfg }}</span>
          </div>
          <label class="mt-6">Seed (−1 = 랜덤)</label>
          <div class="seed-row">
            <input v-model="seed" type="text" class="seed-input" placeholder="-1" />
            <button class="act-btn seed-btn" @click="seed = '-1'" title="랜덤으로 초기화"><Icon name="dice" /></button>
          </div>
        </details>

        <HandReconstructionPanel :source-revision="handSourceRevision" :has-image="handImageReady" :has-mask="hasMask" :get-input="getHandReconstructionInput" />
      </div>

      <div class="sidebar-footer">
        <div class="mask-actions">
          <button class="act-btn" @click="clearMask">비우기</button>
          <button class="act-btn" @click="undoMask">실행 취소</button>
          <button class="act-btn" @click="redoMask">다시 실행</button>
        </div>
        <button class="btn-gen" @click="generate" :disabled="!imageSrc">인페인트 시작</button>
      </div>
    </aside>

    <!-- Canvas -->
    <section class="canvas-area">
      <div class="canvas-wrap" @dragover.prevent="isDragging = true" @dragleave="isDragging = false"
        @drop.prevent="handleDrop" :class="{ dragging: isDragging }">
        <div v-if="!imageSrc" class="drop-empty" @click="triggerFileInput">
          <div class="drop-icon"><Icon name="pencil" /></div>
          <h2>MASK EDITOR</h2>
          <p>이미지를 끌어 놓거나 클릭하세요</p>
        </div>
        <template v-else>
          <canvas ref="imgRef" class="cv" :style="cvStyle"></canvas>
          <canvas ref="maskRef" class="cv mask" :style="cvStyle"></canvas>
          <canvas ref="overlayRef" class="cv overlay" :style="cvStyle"
            @mousedown="onDown" @mousemove="onMove" @mouseup="onUp"
            @mouseleave="onUp" @wheel.prevent="onWheel" @dblclick="onDblClick"
            @contextmenu.prevent></canvas>
          <div class="cv-info">{{ imgW }}×{{ imgH }} | {{ Math.round(zoom*100) }}%<template v-if="hasMask"> | MASK</template></div>
        </template>
      </div>
      <input ref="fileInput" type="file" accept="image/*" hidden @change="handleFileSelect" />
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import EditorToolbar from '../components/editor/EditorToolbar.vue'
import { INPAINT_TOOLS, toolById, toolByKey } from '../utils/editorTools'
import { requestAction } from '../stores/widgetStore.js'
import { getBackend, onBackendEvent } from '../bridge.js'
import { mediaUrl } from '../utils/media.js'
import CustomSelect from '../components/CustomSelect.vue'
import HandReconstructionPanel from '../components/HandReconstructionPanel.vue'
// 마스크 픽셀 연산·undo 정책·엣지맵은 에디터(EditorCanvas)와 같은 코드를 쓴다 — 예전엔 복사본이라
// undo 시점·엣지맵 캐시 같은 개선이 에디터에만 들어갔다.
import {
  MaskHistory, appendLassoPoint, dragRect, encodeMaskPng, fillPolygon, fillRect as fillMaskRect,
  paintMaskOverlay, rectIsApplicable, rgba32, snapshotsOnPress, stampCircle, strokeLine,
  type MaskEdit, type Point,
} from '../utils/maskOps'
import {
  EDGE_SNAP_RADIUS, EdgeMapCache, decodeEdgeMap, snapToEdge as snapToEdgeMap, type EdgeMap,
} from '../utils/edgeMap'

interface DirtyRect { x1: number; y1: number; x2: number; y2: number }
/** 인페인트 마스크 오버레이 색 (226,179,64, 알파 100) */
const INPAINT_MASK_RGBA32 = rgba32(226, 179, 64, 100)

// ── State ──
const isDragging = ref(false)
const imageSrc = ref('')
const imagePath = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const imgRef = ref<HTMLCanvasElement | null>(null)
const maskRef = ref<HTMLCanvasElement | null>(null)
const overlayRef = ref<HTMLCanvasElement | null>(null)
const brushSize = ref(40)
const prompt = ref('')
const negPrompt = ref('')
const denoising = ref(0.75)
// 기본값은 그동안 실제로 나가던 값(숨은 레거시 탭의 원본 유지 · 마스크 영역만)과 같게 —
// 이 두 옵션이 백엔드에 연결되면서 결과가 조용히 바뀌지 않도록 (core/inpaint_payload.py).
const maskContent = ref(1)
const inpaintArea = ref(1)
const maskBlur = ref(4)
const padding = ref(32)
const steps = ref(20)
const cfg = ref(7)
const seed = ref('-1')
const currentTool = ref('brush')
const eraserMode = ref('brush')
const magneticLasso = ref(false)
// 자석 올가미 엣지맵 — 한 장짜리 캐시(같은 이미지에서 '자석'을 다시 켜도 Canny 를 다시 돌리지 않는다)
const edgeMapCache = new EdgeMapCache()
let edgeMap: EdgeMap | null = null
let edgeMapFor = ''     // edgeMap 이 어느 이미지 경로의 것인지
let edgeMapToken = 0    // 늦게 온 옛 요청이 새 이미지의 엣지맵을 덮지 못하게
const maskContents = ['채우기', '원본 유지', 'Latent 노이즈', 'Latent 없음']
const inpaintAreas = ['전체 이미지', '마스크 영역만']
const maskContentLabel = computed({
  get: () => maskContents[maskContent.value] || maskContents[0],
  set: (v: string) => { maskContent.value = maskContents.indexOf(v) }
})
const inpaintAreaLabel = computed({
  get: () => inpaintAreas[inpaintArea.value] || inpaintAreas[0],
  set: (v: string) => { inpaintArea.value = inpaintAreas.indexOf(v) }
})
const currentToolLabel = computed(() => toolById(currentTool.value)?.label ?? '도구')
const currentToolKey = computed(() => toolById(currentTool.value)?.shortcut ?? '')

/**
 * 도구 단축키 (M·L·B·E). 에디터와 같은 키를 쓴다.
 * 입력 중에는 절대 가로채지 않는다 — 프롬프트에 'b' 를 치는 순간 도구가 바뀌면 못 쓴다.
 */
function onInpaintKeyDown(e: KeyboardEvent) {
  const el = document.activeElement as HTMLElement | null
  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return
  const tool = toolByKey(e.key, { ctrl: e.ctrlKey, alt: e.altKey, meta: e.metaKey })
  if (!tool || !INPAINT_TOOLS.some((t) => t.id === tool.id)) return
  e.preventDefault()
  currentTool.value = tool.id
}
onMounted(() => document.addEventListener('keydown', onInpaintKeyDown))
onUnmounted(() => document.removeEventListener('keydown', onInpaintKeyDown))

const imgW = ref(0), imgH = ref(0)
const zoom = ref(1), panX = ref(0), panY = ref(0)
const hasMask = ref(false)
const handSourceRevision = ref(0)
const handImageReady = ref(false)
let imageLoadRevision = 0

let iCtx: CanvasRenderingContext2D | null = null
let mCtx: CanvasRenderingContext2D | null = null
let oCtx: CanvasRenderingContext2D | null = null
let srcImg: HTMLImageElement | null = null
let maskData: Uint8Array | null = null
let maskImageData: ImageData | null = null
let maskPixels: Uint32Array | null = null   // maskImageData 의 32비트 뷰 (픽셀당 1회 대입)
let drawing = false, panning = false
let startX = 0, startY = 0, lastX = -1, lastY = -1
let panSX = 0, panSY = 0
let lassoPoints: Point[] = []
const maskHistory = new MaskHistory(10)
let overlayFrame = 0    // 올가미 경로 다시 그리기 rAF — pointermove 마다 전체 경로를 그리지 않게

const cvStyle = computed(() => ({
  transform: `translate(${panX.value}px,${panY.value}px) scale(${zoom.value})`,
  transformOrigin: 'center center',
  cursor: panning ? 'grabbing' : (currentTool.value === 'brush' || currentTool.value === 'eraser') ? 'crosshair' : 'crosshair',
}))

// ── 이미지 로드 ──
function triggerFileInput() { fileInput.value?.click() }
function handleFileSelect(e: Event) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) loadFile(f) }
function handleDrop(e: DragEvent) {
  isDragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) { loadFile(f); return }
  const p = e.dataTransfer?.getData('text/plain')
  if (p && p.includes('/')) loadFromPath(p)
}
function loadFile(file: File) {
  // 파일 선택·드롭은 경로를 모른다(QtWebEngine File 에는 .path 가 없다) — 이전 경로를 반드시
  // 비운다. 안 비우면 갤러리에서 보냈던 **옛 이미지 경로**와 새 마스크가 함께 전송되고,
  // 자석 올가미 엣지맵도 옛 이미지로 계산된다 (I2IView.loadFile 과 같은 규칙).
  const nativePath = (file as any).path
  imagePath.value = typeof nativePath === 'string' && nativePath ? nativePath.replace(/\\/g, '/') : ''
  resetEdgeMap()
  // FileReader 가 끝날 때까지 imageSrc·마스크는 옛 이미지 것이다 — 읽기 **전에** 준비 상태를
  // 내려야 그 사이 generate 가 옛 이미지(경로 없이)·옛 마스크를 보내지 않는다.
  const loadRevision = beginImageLoad()
  const r = new FileReader()
  r.onload = () => {
    if (loadRevision !== imageLoadRevision) return   // 그새 다른 이미지를 받았다
    const src = typeof r.result === 'string' ? r.result : ''
    if (!src) { failImageLoad(); return }
    imageSrc.value = src
    initCanvas(src, loadRevision)
  }
  r.onerror = () => {
    if (loadRevision !== imageLoadRevision) return
    failImageLoad()
  }
  r.readAsDataURL(file)
}
async function loadFromPath(path: string) {
  resetEdgeMap()
  if (/^data:image\//i.test(path) || path.startsWith('blob:')) {
    imagePath.value = ''
    imageSrc.value = path
    initCanvas(path, beginImageLoad())
    return
  }
  const normalized = path.replace(/\\/g, '/')
  imagePath.value = normalized
  // 표시용 src 는 mediaUrl 로 — Qt 는 file:///, 웹 모드는 /file?path= (http 페이지는
  // file:/// 을 못 읽어 onload 가 영영 오지 않았다). 이미 file:/// 이 붙은 드롭 경로도 정리된다.
  const displayUrl = mediaUrl(normalized)
  imageSrc.value = displayUrl
  initCanvas(displayUrl, beginImageLoad())
}

/** 새 이미지 읽기를 시작한다. 이 순간부터 generate·손 재구성은 막히고(handImageReady=false),
 *  먼저 시작한 읽기(FileReader·<img>)의 늦은 결과는 revision 이 달라 버려진다. */
function beginImageLoad(): number {
  const loadRevision = ++imageLoadRevision
  handSourceRevision.value++
  handImageReady.value = false
  return loadRevision
}

/** 파일·이미지를 못 읽었다 — 캔버스·경로를 비우고 알린다. */
function failImageLoad() {
  resetAfterLoadError()
  requestAction('show_toast', { type: 'error', msg: '인페인트 이미지를 불러오지 못했습니다' })
}

/** 이미지를 못 읽었으면 캔버스·마스크·경로를 전부 비운다 — 이전 이미지의 마스크가
 *  새 경로와 함께 전송되는 일이 없게. */
function resetAfterLoadError() {
  imageSrc.value = ''
  imagePath.value = ''
  srcImg = null
  maskData = null
  maskImageData = null
  maskPixels = null
  hasMask.value = false
  maskHistory.clear()
  resetEdgeMap()
  imgW.value = 0; imgH.value = 0
}

/** `loadRevision` 은 beginImageLoad() 가 준 값 — 그새 다른 이미지를 받았으면 아무것도 안 한다. */
function initCanvas(src: string, loadRevision: number) {
  if (loadRevision !== imageLoadRevision) return
  const img = new Image()
  img.onerror = () => {
    if (loadRevision !== imageLoadRevision) return
    failImageLoad()
  }
  img.onload = () => {
    if (loadRevision !== imageLoadRevision) return
    srcImg = img; imgW.value = img.naturalWidth; imgH.value = img.naturalHeight
    zoom.value = 1; panX.value = 0; panY.value = 0
    const ic = imgRef.value; if (!ic) return
    ic.width = img.naturalWidth; ic.height = img.naturalHeight
    iCtx = ic.getContext('2d'); iCtx!.drawImage(img, 0, 0)
    const mc = maskRef.value; if (!mc) return
    mc.width = img.naturalWidth; mc.height = img.naturalHeight
    mCtx = mc.getContext('2d'); mCtx!.clearRect(0, 0, mc.width, mc.height)
    const oc = overlayRef.value; if (!oc) return
    oc.width = img.naturalWidth; oc.height = img.naturalHeight
    oCtx = oc.getContext('2d'); oCtx!.clearRect(0, 0, oc.width, oc.height)
    maskData = new Uint8Array(img.naturalWidth * img.naturalHeight)
    maskImageData = mCtx!.createImageData(img.naturalWidth, img.naturalHeight)
    maskPixels = new Uint32Array(maskImageData.data.buffer)
    hasMask.value = false; maskHistory.clear()
    handImageReady.value = true
    // 자석 올가미를 켜 둔 채 이미지를 바꿨으면 새 이미지로 엣지맵을 다시 만든다.
    // (옛 엣지맵은 loadFile/loadFromPath 가 이미 버렸다 — 옛 이미지 윤곽에 붙지 않게)
    if (magneticLasso.value) void ensureEdgeMap()
  }
  img.src = src
}

// ── 좌표 ──
function getPos(e: MouseEvent): Point {
  if (!overlayRef.value) return { x: 0, y: 0 }
  const r = overlayRef.value.getBoundingClientRect()
  return { x: (e.clientX - r.left) / r.width * overlayRef.value.width, y: (e.clientY - r.top) / r.height * overlayRef.value.height }
}

// ── 마우스 이벤트 ──
function onDblClick(e: MouseEvent) { if (e.altKey) { zoom.value = 1; panX.value = 0; panY.value = 0 } }

function onDown(e: MouseEvent) {
  if (e.altKey || e.button === 1) { panning = true; panSX = e.clientX - panX.value; panSY = e.clientY - panY.value; return }
  if (!maskData) return
  // 누르는 순간 마스크가 바뀌는 도구(브러시·브러시 지우개)만 여기서 undo 스냅숏을 뜬다.
  // 사각형·올가미는 실제로 적용될 때(onUp) 뜬다 — 예전엔 누를 때마다 떠서 빈 클릭도
  // 마스크 전체 스냅숏을 쌓고 redo 를 날렸다(에디터와 같은 정책: maskOps.snapshotsOnPress).
  if (snapshotsOnPress(currentTool.value, eraserMode.value)) saveUndo()
  drawing = true
  const p = getPos(e); startX = p.x; startY = p.y; lastX = p.x; lastY = p.y

  if (currentTool.value === 'lasso') { const sp = magneticLasso.value ? snapToEdge(p.x, p.y) : p; lassoPoints = [{ x: sp.x, y: sp.y }] }
  else if (currentTool.value === 'brush') { renderDirty(paintCircle(p.x, p.y)) }
  else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'brush') { renderDirty(eraseCircle(p.x, p.y)) }
    else { lassoPoints = eraserMode.value === 'lasso' ? [{ x: p.x, y: p.y }] : [] }
  }
}

function onMove(e: MouseEvent) {
  if (panning) { panX.value = e.clientX - panSX; panY.value = e.clientY - panSY; return }
  const p = getPos(e)
  if (!drawing) {
    clearOverlay()
    if (oCtx && (currentTool.value === 'brush' || currentTool.value === 'eraser')) {
      const col = currentTool.value === 'eraser' ? 'rgba(248,113,113,0.5)' : 'rgba(226,179,64,0.5)'
      oCtx.strokeStyle = col; oCtx.lineWidth = 2
      oCtx.beginPath(); oCtx.arc(p.x, p.y, brushSize.value, 0, Math.PI * 2); oCtx.stroke()
    }
    return
  }

  // 아래 오버레이 캔버스의 선 색들은 토큰화하지 않는다 — 이미지 위에 얹히는 선택 표시라
  // UI 크롬이 아니고, 배경(사용자 이미지)이 테마와 무관해서 테마색을 따라가면 오히려 안 보인다.
  if (currentTool.value === 'box') {
    clearOverlay()
    if (oCtx) { oCtx.strokeStyle = '#E2B340'; oCtx.lineWidth = 2; oCtx.setLineDash([6,4]); oCtx.strokeRect(startX, startY, p.x-startX, p.y-startY); oCtx.setLineDash([]) }
  } else if (currentTool.value === 'lasso') {
    // 직전 점과 1px 미만이면 버린다. 경로는 프레임당 한 번만 다시 그린다(예전엔 move 마다 전체 경로).
    const sp = magneticLasso.value ? snapToEdge(p.x, p.y) : p
    if (appendLassoPoint(lassoPoints, sp)) scheduleLassoOverlay()
  } else if (currentTool.value === 'brush') {
    renderDirty(paintLine(lastX, lastY, p.x, p.y)); lastX = p.x; lastY = p.y
  } else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'brush') { renderDirty(eraseLine(lastX, lastY, p.x, p.y)); lastX = p.x; lastY = p.y }
    else if (eraserMode.value === 'box') { clearOverlay(); if (oCtx) { oCtx.strokeStyle = '#f87171'; oCtx.lineWidth = 2; oCtx.setLineDash([6,4]); oCtx.strokeRect(startX, startY, p.x-startX, p.y-startY); oCtx.setLineDash([]) } }
    else if (eraserMode.value === 'lasso') { if (appendLassoPoint(lassoPoints, p)) scheduleLassoOverlay() }
  }
}

/** 올가미 경로 다시 그리기를 다음 프레임으로 미룬다(한 프레임에 한 번). */
function scheduleLassoOverlay() {
  if (overlayFrame) return
  overlayFrame = requestAnimationFrame(() => { overlayFrame = 0; drawLassoOverlay() })
}
function cancelLassoOverlay() {
  if (overlayFrame) { cancelAnimationFrame(overlayFrame); overlayFrame = 0 }
}
function drawLassoOverlay() {
  if (!drawing) return
  clearOverlay()
  if (!oCtx || lassoPoints.length < 2) return
  const erasing = currentTool.value === 'eraser'
  oCtx.strokeStyle = erasing ? '#f87171' : (magneticLasso.value ? '#60a5fa' : '#E2B340')
  oCtx.lineWidth = 2
  if (!erasing) oCtx.setLineDash([4, 3])
  oCtx.beginPath(); oCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y)
  for (let i = 1; i < lassoPoints.length; i++) oCtx.lineTo(lassoPoints[i].x, lassoPoints[i].y)
  oCtx.closePath(); oCtx.stroke(); oCtx.setLineDash([])
  if (!erasing) { oCtx.fillStyle = 'rgba(226,179,64,0.1)'; oCtx.fill() }
}

function onUp(e: MouseEvent) {
  if (panning) { panning = false; return }
  if (!drawing) return
  // FIX: drawing=false 를 좌표/도구 완성 처리 후로 이동 — 그래야 box/lasso 완성 시점에
  // 다른 핸들러가 짧게 끼어들어 좌표를 더럽히는 것을 막을 수 있음.
  const p = getPos(e)
  cancelLassoOverlay()
  // 사각형·올가미는 실제로 적용할 때만 undo 스냅숏 — 3px 이하 드래그(클릭)는 무시한다
  let dirty: DirtyRect | null = null
  if (currentTool.value === 'box') {
    const r = dragRect(startX, startY, p.x, p.y)
    if (rectIsApplicable(r)) { saveUndo(); dirty = fillRect(r.x1, r.y1, r.x2, r.y2) }
  } else if (currentTool.value === 'lasso') {
    if (lassoPoints.length > 2) { saveUndo(); dirty = fillPoly(lassoPoints) }
    lassoPoints = []
  } else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'box') {
      const r = dragRect(startX, startY, p.x, p.y)
      if (rectIsApplicable(r)) { saveUndo(); dirty = eraseRect(r.x1, r.y1, r.x2, r.y2) }
    } else if (eraserMode.value === 'lasso') {
      if (lassoPoints.length > 2) { saveUndo(); dirty = erasePoly(lassoPoints) }
      lassoPoints = []
    }
  }
  if (dirty) renderDirty(dirty)
  clearOverlay(); updateHasMask()
  drawing = false  // 모든 처리 완료 후에만 false
}

function onWheel(e: WheelEvent) { zoom.value = Math.max(0.2, Math.min(5, zoom.value * (e.deltaY > 0 ? 0.9 : 1.1))) }

// ── 마스크 조작 ──
// 픽셀 연산은 utils/maskOps(에디터와 공용). 각 함수는 건드린 사각형을 돌려주고, 그 영역만 다시 그린다.
function maskBuffer() {
  return maskData && srcImg ? { data: maskData, w: srcImg.naturalWidth, h: srcImg.naturalHeight } : null
}
function paintCircle(cx: number, cy: number): MaskEdit | null {
  const m = maskBuffer(); return m ? stampCircle(m, cx, cy, brushSize.value, true) : null
}
function paintLine(x0: number, y0: number, x1: number, y1: number): MaskEdit | null {
  const m = maskBuffer(); return m ? strokeLine(m, x0, y0, x1, y1, brushSize.value, true) : null
}
function eraseCircle(cx: number, cy: number): MaskEdit | null {
  const m = maskBuffer(); return m ? stampCircle(m, cx, cy, brushSize.value, false) : null
}
function eraseLine(x0: number, y0: number, x1: number, y1: number): MaskEdit | null {
  const m = maskBuffer(); return m ? strokeLine(m, x0, y0, x1, y1, brushSize.value, false) : null
}
function fillRect(x1: number, y1: number, x2: number, y2: number): MaskEdit | null {
  const m = maskBuffer(); return m ? fillMaskRect(m, x1, y1, x2, y2, true) : null
}
function eraseRect(x1: number, y1: number, x2: number, y2: number): MaskEdit | null {
  const m = maskBuffer(); return m ? fillMaskRect(m, x1, y1, x2, y2, false) : null
}
// 올가미는 스캔라인 채우기 — 예전 bbox 전 픽셀 × 꼭짓점 point-in-polygon 은 긴 올가미에서 수 초 멈췄다
function fillPoly(pts: Point[]): MaskEdit | null {
  const m = maskBuffer(); return m && pts.length >= 3 ? fillPolygon(m, pts, true) : null
}
function erasePoly(pts: Point[]): MaskEdit | null {
  const m = maskBuffer(); return m && pts.length >= 3 ? fillPolygon(m, pts, false) : null
}

function clearOverlay() {
  if (oCtx && srcImg) oCtx.clearRect(0, 0, srcImg.naturalWidth, srcImg.naturalHeight)
}
function renderDirty(rect?: DirtyRect | null) {
  if (!mCtx || !maskData || !maskImageData || !maskPixels || !srcImg) return
  handSourceRevision.value++
  const w = srcImg.naturalWidth, h = srcImg.naturalHeight
  const x1 = Math.max(0, Math.floor(rect?.x1 ?? 0)), y1 = Math.max(0, Math.floor(rect?.y1 ?? 0))
  const x2 = Math.min(w, Math.ceil(rect?.x2 ?? w)), y2 = Math.min(h, Math.ceil(rect?.y2 ?? h))
  if (x2 <= x1 || y2 <= y1) return
  paintMaskOverlay(maskPixels, maskData, w, x1, y1, x2, y2, INPAINT_MASK_RGBA32)
  mCtx.putImageData(maskImageData, 0, 0, x1, y1, x2-x1, y2-y1)
}

// ── 자석 올가미 ── (디코드·스냅·캐시는 utils/edgeMap — 에디터와 공용)
// 예전엔 캐시가 없어 '자석'을 누를 때마다 동기 슬롯 getEdgeMap 이 Canny 를 다시 돌려 GUI 스레드를
// 막았고, 새 이미지를 열어도 옛 엣지맵이 남아 이전 이미지 윤곽에 붙었다.
function resetEdgeMap() {
  edgeMapToken++
  edgeMap = null
  edgeMapFor = ''
  edgeMapCache.clear()
}
async function ensureEdgeMap() {
  const path = imagePath.value
  if (!magneticLasso.value || !path || edgeMapFor === path) return
  const token = ++edgeMapToken
  const b64 = await edgeMapCache.get(path)
  // 그 사이 다른 이미지로 바뀌었으면 옛 이미지의 엣지맵을 쓰지 않는다
  if (!b64 || token !== edgeMapToken || path !== imagePath.value) return
  const decoded = await decodeEdgeMap(b64)
  if (!decoded || token !== edgeMapToken || path !== imagePath.value) return
  edgeMap = decoded
  edgeMapFor = path
}
function enableMagnetic() {
  magneticLasso.value = true
  return ensureEdgeMap()
}
function snapToEdge(x: number, y: number): Point {
  return snapToEdgeMap(magneticLasso.value ? edgeMap : null, x, y, EDGE_SNAP_RADIUS)
}
// 자석을 켜 둔 채 다른 도구를 쓰다 올가미로 돌아오면(그사이 이미지가 바뀌었을 수 있다) 엣지맵을 챙긴다
watch(currentTool, (tool) => { if (tool === 'lasso' && magneticLasso.value) void ensureEdgeMap() })
onUnmounted(cancelLassoOverlay)

function updateHasMask() { hasMask.value = maskData ? maskData.some(v => v > 0) : false }
function saveUndo() { if (maskData) maskHistory.save(maskData) }
// 이미 비어 있으면 undo 단계를 쌓지 않는다(빈 '비우기'가 redo 를 날리지 않게)
function clearMask() { if (maskData && maskData.some(v => v > 0)) { saveUndo(); maskData.fill(0) }; hasMask.value = false; renderDirty() }
function undoMask() { if (!maskData || !maskHistory.undo(maskData)) return; updateHasMask(); renderDirty() }
function redoMask() { if (!maskData || !maskHistory.redo(maskData)) return; updateHasMask(); renderDirty() }

function getMaskBase64() {
  if (!maskData || !srcImg) return ''
  return encodeMaskPng(maskData, srcImg.naturalWidth, srcImg.naturalHeight)
}

const HAND_SOURCE_MAX_BYTES = 64 * 1024 * 1024

function validatedHandSourceUrl(source: string): URL {
  const url = new URL(source)
  const decodedPath = decodeURIComponent(url.pathname).replace(/\\/g, '/')
  if (!(url.protocol === 'blob:' || (url.protocol === 'file:' && !url.hostname && !decodedPath.startsWith('//')))) {
    throw Error('외부 이미지 URL은 읽지 않습니다. 원본 PNG/JPEG/WebP를 직접 올려주세요.')
  }
  return url
}

function handRasterMime(head: Uint8Array): string {
  const ascii = (start: number, end: number) => String.fromCharCode(...head.slice(start, end))
  return head[0] === 137 && ascii(1, 4) === 'PNG' && head[4] === 13 && head[5] === 10 && head[6] === 26 && head[7] === 10 ? 'image/png'
    : head[0] === 255 && head[1] === 216 && head[2] === 255 ? 'image/jpeg'
      : ascii(0, 4) === 'RIFF' && ascii(8, 12) === 'WEBP' ? 'image/webp' : ''
}

function readHandNativeSource(source: string): Promise<{ image: string; sourceKind: 'original' | 'canvas' }> {
  return new Promise((resolve, reject) => {
    let settled = false
    const fail = () => {
      if (settled) return
      settled = true
      clearTimeout(timeout)
      reject(Error('현재 로컬 원본 파일을 읽지 못했습니다. 64 MB 이하 PNG/JPEG/WebP 파일을 직접 올려주세요. 캔버스로 자동 대체하지 않습니다.'))
    }
    const timeout = setTimeout(fail, 30000)
    void getBackend().then((backend: any) => {
      if (settled) return
      if (typeof backend.loadImageBase64 !== 'function') { fail(); return }
      backend.loadImageBase64(source, (data: unknown) => {
        if (settled) return
        if (typeof data !== 'string' || data.length > Math.ceil(HAND_SOURCE_MAX_BYTES * 4 / 3) + 128) { fail(); return }
        const match = /^data:image\/(?:png|jpeg|webp);base64,([A-Za-z0-9+/]+={0,2})$/.exec(data)
        if (!match || match[1].length % 4 || match[1].length / 4 * 3 - (match[1].endsWith('==') ? 2 : match[1].endsWith('=') ? 1 : 0) > HAND_SOURCE_MAX_BYTES) { fail(); return }
        let head: Uint8Array
        try { head = Uint8Array.from(atob(match[1].slice(0, 64)), character => character.charCodeAt(0)) } catch { fail(); return }
        const mime = handRasterMime(head)
        const textHead = String.fromCharCode(...head)
        // The legacy reader labels unknown extensions PNG. Check bytes before
        // accepting that label; supported original bytes never pass a canvas.
        const convertible = textHead.startsWith('BM') || /^GIF8[79]a/.test(textHead) || (/\.svg$/i.test(new URL(source).pathname) && /^\s*(?:<svg[\s>]|<\?xml[\s>])/i.test(textHead))
        if (!mime && !convertible) { fail(); return }
        settled = true
        clearTimeout(timeout)
        resolve(mime ? { image: `data:${mime};base64,${match[1]}`, sourceKind: 'original' } : { image: '', sourceKind: 'canvas' })
      })
    }).catch(fail)
  })
}

async function readHandSourceBlob(source: string): Promise<Blob> {
  validatedHandSourceUrl(source)
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 30000)
  try {
    const response = await fetch(source, { redirect: 'error', credentials: 'omit', signal: controller.signal })
    if (!response.ok || !response.body || Number(response.headers.get('content-length')) > HAND_SOURCE_MAX_BYTES) throw Error('Source unavailable or too large')
    const reader = response.body.getReader()
    const chunks: ArrayBuffer[] = []
    let length = 0
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        length += value.byteLength
        if (length > HAND_SOURCE_MAX_BYTES) { await reader.cancel(); throw Error('Source too large') }
        const copy = new Uint8Array(value.byteLength)
        copy.set(value)
        chunks.push(copy.buffer)
      }
    } finally { reader.releaseLock() }
    if (length === 0) throw Error('Empty source')
    return new Blob(chunks, { type: response.headers.get('content-type')?.split(';')[0] || '' })
  } catch {
    controller.abort()
    throw Error('현재 로컬 원본 파일을 읽지 못했습니다. 64 MB 이하 PNG/JPEG/WebP 파일을 직접 올려주세요. 캔버스로 자동 대체하지 않습니다.')
  } finally { clearTimeout(timeout) }
}

async function handBlobData(blob: Blob): Promise<{ image: string; sourceKind: 'original' | 'canvas' }> {
  const head = new Uint8Array(await blob.slice(0, 16).arrayBuffer())
  const ascii = (start: number, end: number) => String.fromCharCode(...head.slice(start, end))
  const mime = handRasterMime(head)
  if (!mime) {
    // Only images that the canvas has already decoded may take the disclosed
    // conversion path. Never silently convert a failed original file read.
    if (/^image\/(?:svg\+xml|bmp|x-ms-bmp|gif|tiff|avif)$/i.test(blob.type) || ascii(0, 2) === 'BM' || /^GIF8[79]a$/.test(ascii(0, 6))) return { image: '', sourceKind: 'canvas' }
    throw Error('원본 형식을 확인할 수 없습니다. PNG/JPEG/WebP 파일을 직접 올려주세요.')
  }
  const image = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(Error('원본 파일을 읽지 못했습니다. PNG/JPEG/WebP 파일을 직접 올려주세요.'))
    reader.readAsDataURL(new Blob([blob], { type: mime }))
  })
  return { image, sourceKind: 'original' }
}

async function getHandReconstructionInput() {
  if (!handImageReady.value || !imgRef.value || !hasMask.value || drawing) throw Error('원본을 올린 뒤 손 마스크 그리기를 끝내주세요.')
  const source = imageSrc.value
  const revision = handSourceRevision.value
  const mask = getMaskBase64()
  let input: { image: string; sourceKind: 'original' | 'canvas' }
  if (/^data:image\/(?:png|jpeg|webp);base64,/i.test(source)) {
    if (source.length > Math.ceil(HAND_SOURCE_MAX_BYTES * 4 / 3) + 128) throw Error('64 MB 이하의 원본 PNG/JPEG/WebP를 사용하세요.')
    input = { image: source, sourceKind: 'original' }
  } else if (source.startsWith('file:///') || source.startsWith('blob:')) {
    const url = validatedHandSourceUrl(source)
    if (url.protocol === 'file:' && typeof window !== 'undefined' && (window as Window & { qt?: { webChannelTransport?: unknown } }).qt?.webChannelTransport) input = await readHandNativeSource(source)
    else input = await handBlobData(await readHandSourceBlob(source))
  } else if (/^data:image\/(?:svg\+xml|bmp|x-ms-bmp|gif|tiff|avif)[;,]/i.test(source)) {
    input = { image: '', sourceKind: 'canvas' }
  } else throw Error('외부 이미지 URL은 읽지 않습니다. 원본 PNG/JPEG/WebP를 직접 올려주세요.')
  if (source !== imageSrc.value || revision !== handSourceRevision.value || !handImageReady.value || !hasMask.value || drawing) throw Error('원본 또는 마스크가 바뀌었습니다. 현재 이미지에서 다시 실행하세요.')
  try {
    // Supported raster bytes retain metadata and transparent RGB. Only unsupported
    // but already-displayed formats use an explicitly disclosed canvas snapshot.
    return { image: input.sourceKind === 'canvas' ? imgRef.value.toDataURL('image/png') : input.image, mask, sourceKind: input.sourceKind }
  } catch {
    throw Error('현재 이미지 픽셀을 읽을 수 없습니다. 원본 파일을 직접 올린 뒤 다시 시도하세요.')
  }
}

function generate() {
  // 새 이미지를 읽는 중이면 캔버스의 마스크는 아직 이전 이미지 것이다.
  if (!handImageReady.value) {
    requestAction('show_toast', { type: 'warning', msg: '이미지를 불러오는 중입니다 — 잠시 후 다시 시도하세요' })
    return
  }
  if (!hasMask.value) {
    requestAction('show_toast', { type: 'warning', msg: 'Inpaint: 마스크를 그려주세요' })
    return
  }
  // 백엔드는 이 값만 본다 (core/inpaint_payload.py) — 숨은 레거시 탭 값은 쓰지 않는다.
  requestAction('generate_inpaint', {
    image: imagePath.value ? '' : imageSrc.value,
    image_path: imagePath.value,
    mask: getMaskBase64(),
    prompt: prompt.value,
    negative_prompt: negPrompt.value,
    denoising: denoising.value,
    mask_content: maskContent.value,
    inpaint_area: inpaintArea.value,
    mask_blur: maskBlur.value,
    padding: padding.value,
    steps: steps.value,
    cfg: cfg.value,
    seed: seed.value,
  })
}

onMounted(() => { onBackendEvent('inpaintImageLoaded', (path: string) => loadFromPath(path)) })
</script>

<style scoped>
.inpaint-workspace { height: 100%; display: flex; background: var(--bg-primary); }
.sidebar { width: 280px; display: flex; flex-direction: column; background: var(--bg-secondary); border-right: 1px solid var(--border); }
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 0 10px; display: flex; flex-direction: column; }
.sidebar-footer { padding: 10px; background: var(--bg-card); border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
/* 아이콘만 있는 툴바를 보완하는 이름표 — 지금 무슨 도구인지 글자로도 알려준다 */
.tool-head { display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2); padding: 2px 2px 4px; }
.tool-name { color: var(--text-primary); font-size: var(--fs-body); font-weight: var(--fw-medium); }
.tool-key {
  min-width: 18px; padding: 1px 5px; text-align: center;
  background: var(--bg-button); border: 1px solid var(--rule); border-radius: 3px;
  color: var(--text-muted); font-size: var(--fs-label);
}

.source-thumb { height: 100px; border-radius: 6px; overflow: hidden; cursor: pointer; background: var(--bg-input); display: flex; align-items: center; justify-content: center; }
.source-thumb img { width: 100%; height: 100%; object-fit: contain; }
.upload-hint { color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); }
.tool-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; }
.tool-grid.small { grid-template-columns: repeat(3, 1fr); }
.tool-chip { min-height: 30px; padding: 6px 8px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; text-align: center; display: flex; align-items: center; justify-content: center; }
.tool-chip:hover { border-color: var(--text-muted); }
.tool-chip.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
/* 테두리도 선/글자 역할이라 채움용 --state-info 가 아니라 --state-info-fg 를 쓴다 */
.tool-chip.magnet.active { background: rgba(96,165,250,0.1); border-color: var(--state-info-fg); color: var(--state-info-fg); }
.slider-row { display: flex; align-items: center; gap: 6px; }
.slider-row input[type="range"] { flex: 1; accent-color: var(--accent); }
.slider-val { font-size: var(--fs-label); color: var(--accent); min-width: 36px; text-align: right; font-family: monospace; }
.mt-6 { margin-top: 6px; }
.seed-row { display: flex; gap: 4px; align-items: center; }
.seed-input { flex: 1; min-width: 0; }
.act-btn.seed-btn { flex: 0 0 32px; padding: 0; display: flex; align-items: center; justify-content: center; }
.mask-actions { display: flex; gap: 3px; }
.act-btn { flex: 1; height: 30px; padding: 0 8px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; }
.btn-gen { width: 100%; height: 42px; background: var(--accent-fill); border: none; border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 12px; cursor: pointer; }
.btn-gen:disabled { opacity: 0.4; }
/* 캔버스 '뒤' 여백은 UI 면이다 — 이미지에 칠해지는 색이 아니라 토큰으로 간다 */
.canvas-area { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; background: var(--bg-primary); position: relative; }
.canvas-wrap { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; position: relative; }
.canvas-wrap.dragging { background: rgba(250,204,21,0.05); }
.drop-empty { text-align: center; cursor: pointer; }
.drop-icon { font-size: 48px; opacity: 0.3; }
.drop-empty h2 { letter-spacing: 0.08em; color: var(--text-muted); letter-spacing: 0; }
.drop-empty p { color: var(--text-muted); font-size: 12px; }
.cv { position: absolute; max-width: 85%; max-height: 85%; }
.cv.mask { pointer-events: none; }
.cv.overlay { pointer-events: auto; }
/* 글자색을 토큰으로 못 바꾼다: 바탕이 테마를 안 타는 rgba(0,0,0,0.6) 칩이라
   --text-muted 로 두면 라이트에서 어두운 글자가 어두운 칩 위에 얹혀 안 보인다.
   제대로 고치려면 칩 자체에 라이트 모드 규칙이 필요하다(이번 범위 밖). */
.cv-info { position: absolute; bottom: 8px; right: 12px; color: #585858; font-size: var(--fs-label); background: rgba(0,0,0,0.6); padding: 2px 8px; border-radius: 4px; pointer-events: none; }
</style>
