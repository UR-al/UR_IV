<template>
  <div class="inpaint-workspace">
    <!-- Left Sidebar -->
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <div class="glass-card">
          <label>Source Image</label>
          <div class="source-thumb" @click="triggerFileInput">
            <img v-if="imageSrc" :src="imageSrc" />
            <div v-else class="upload-hint">DROP OR CLICK</div>
          </div>
        </div>

        <!-- 도구 선택 -->
        <div class="glass-card">
          <label>Selection Tool</label>
          <div class="tool-grid">
            <button v-for="t in tools" :key="t.id" class="tool-chip"
              :class="{ active: currentTool === t.id }" @click="currentTool = t.id">
              {{ t.icon }} {{ t.label }}
            </button>
          </div>
        </div>

        <!-- 올가미 모드 -->
        <div class="glass-card" v-if="currentTool === 'lasso'">
          <label>Lasso Mode</label>
          <div class="tool-grid small">
            <button class="tool-chip" :class="{ active: !magneticLasso }" @click="magneticLasso = false">➰ FREE</button>
            <button class="tool-chip magnet" :class="{ active: magneticLasso }" @click="enableMagnetic">🧲 MAGNETIC</button>
          </div>
        </div>

        <!-- 지우개 모드 -->
        <div class="glass-card" v-if="currentTool === 'eraser'">
          <label>Eraser Shape</label>
          <div class="tool-grid small">
            <button class="tool-chip" :class="{ active: eraserMode === 'brush' }" @click="eraserMode = 'brush'">BRUSH</button>
            <button class="tool-chip" :class="{ active: eraserMode === 'box' }" @click="eraserMode = 'box'">RECT</button>
            <button class="tool-chip" :class="{ active: eraserMode === 'lasso' }" @click="eraserMode = 'lasso'">LASSO</button>
          </div>
        </div>

        <!-- 브러시 크기 -->
        <div class="glass-card">
          <label>Brush Size</label>
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
          <label>Override Prompt</label>
          <textarea v-model="prompt" rows="3" placeholder="Describe the change..."></textarea>
        </div>

        <div class="glass-card">
          <label>Mask Settings</label>
          <CustomSelect v-model="maskContentLabel" :options="maskContents" placeholder="Mask Content" />
          <CustomSelect v-model="inpaintAreaLabel" :options="inpaintAreas" placeholder="Inpaint Area" class="mt-6" />
        </div>
      </div>

      <div class="sidebar-footer">
        <div class="mask-actions">
          <button class="act-btn" @click="clearMask">CLEAR</button>
          <button class="act-btn" @click="undoMask">UNDO</button>
          <button class="act-btn" @click="redoMask">REDO</button>
        </div>
        <button class="btn-gen" @click="generate" :disabled="!imageSrc">START INPAINTING</button>
      </div>
    </aside>

    <!-- Canvas -->
    <section class="canvas-area">
      <div class="canvas-wrap" @dragover.prevent="isDragging = true" @dragleave="isDragging = false"
        @drop.prevent="handleDrop" :class="{ dragging: isDragging }">
        <div v-if="!imageSrc" class="drop-empty" @click="triggerFileInput">
          <div class="drop-icon">✎</div>
          <h2>MASK EDITOR</h2>
          <p>Drop image or click to start</p>
        </div>
        <template v-else>
          <canvas ref="imgRef" class="cv" :style="cvStyle"></canvas>
          <canvas ref="maskRef" class="cv mask" :style="cvStyle"
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

<script setup>
import { ref, computed, onMounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { getBackend, onBackendEvent } from '../bridge.js'
import CustomSelect from '../components/CustomSelect.vue'

// ── State ──
const isDragging = ref(false)
const imageSrc = ref('')
const imagePath = ref('')
const fileInput = ref(null)
const imgRef = ref(null)
const maskRef = ref(null)
const brushSize = ref(40)
const prompt = ref('')
const denoising = ref(0.75)
const maskContent = ref(0)
const inpaintArea = ref(0)
const currentTool = ref('brush')
const eraserMode = ref('brush')
const magneticLasso = ref(false)
let edgeMapData = null, edgeMapW = 0, edgeMapH = 0
const maskContents = ['FILL', 'ORIGINAL', 'LATENT NOISE', 'LATENT NOTHING']
const inpaintAreas = ['WHOLE IMAGE', 'ONLY MASKED']
const maskContentLabel = computed({
  get: () => maskContents[maskContent.value] || maskContents[0],
  set: v => { maskContent.value = maskContents.indexOf(v) }
})
const inpaintAreaLabel = computed({
  get: () => inpaintAreas[inpaintArea.value] || inpaintAreas[0],
  set: v => { inpaintArea.value = inpaintAreas.indexOf(v) }
})
const tools = [
  { id: 'box', icon: '⬚', label: 'RECT' },
  { id: 'lasso', icon: '➰', label: 'LASSO' },
  { id: 'brush', icon: '🖌', label: 'BRUSH' },
  { id: 'eraser', icon: '⌫', label: 'ERASER' },
]

const imgW = ref(0), imgH = ref(0)
const zoom = ref(1), panX = ref(0), panY = ref(0)
const hasMask = ref(false)

let iCtx = null, mCtx = null, srcImg = null
let maskData = null
let drawing = false, panning = false
let startX = 0, startY = 0, lastX = -1, lastY = -1
let panSX = 0, panSY = 0
let lassoPoints = []
let undoStack = [], redoStack = []

const cvStyle = computed(() => ({
  transform: `translate(${panX.value}px,${panY.value}px) scale(${zoom.value})`,
  transformOrigin: 'center center',
  cursor: panning ? 'grabbing' : (currentTool.value === 'brush' || currentTool.value === 'eraser') ? 'crosshair' : 'crosshair',
}))

// ── 이미지 로드 ──
function triggerFileInput() { fileInput.value?.click() }
function handleFileSelect(e) { const f = e.target.files?.[0]; if (f) loadFile(f) }
function handleDrop(e) {
  isDragging.value = false
  const f = e.dataTransfer?.files?.[0]
  if (f) { imagePath.value = f.path || ''; loadFile(f); return }
  const p = e.dataTransfer?.getData('text/plain')
  if (p && p.includes('/')) loadFromPath(p)
}
function loadFile(file) {
  const r = new FileReader()
  r.onload = (ev) => { imageSrc.value = ev.target.result; initCanvas(ev.target.result) }
  r.readAsDataURL(file)
  if (file.path) imagePath.value = file.path.replace(/\\/g, '/')
}
async function loadFromPath(path) {
  imagePath.value = path
  const bk = await getBackend()
  if (bk.loadImageBase64) bk.loadImageBase64(path, (b64) => { if (b64) { imageSrc.value = b64; initCanvas(b64) } })
}

function initCanvas(src) {
  const img = new Image()
  img.onload = () => {
    srcImg = img; imgW.value = img.naturalWidth; imgH.value = img.naturalHeight
    zoom.value = 1; panX.value = 0; panY.value = 0
    const ic = imgRef.value; if (!ic) return
    ic.width = img.naturalWidth; ic.height = img.naturalHeight
    iCtx = ic.getContext('2d'); iCtx.drawImage(img, 0, 0)
    const mc = maskRef.value; if (!mc) return
    mc.width = img.naturalWidth; mc.height = img.naturalHeight
    mCtx = mc.getContext('2d'); mCtx.clearRect(0, 0, mc.width, mc.height)
    maskData = new Uint8Array(img.naturalWidth * img.naturalHeight)
    hasMask.value = false; undoStack = []; redoStack = []
  }
  img.src = src
}

// ── 좌표 ──
function getPos(e) {
  if (!maskRef.value) return { x: 0, y: 0 }
  const r = maskRef.value.getBoundingClientRect()
  return { x: (e.clientX - r.left) / r.width * maskRef.value.width, y: (e.clientY - r.top) / r.height * maskRef.value.height }
}

// ── 마우스 이벤트 ──
function onDblClick(e) { if (e.altKey) { zoom.value = 1; panX.value = 0; panY.value = 0 } }

function onDown(e) {
  if (e.altKey || e.button === 1) { panning = true; panSX = e.clientX - panX.value; panSY = e.clientY - panY.value; return }
  if (!maskData) return
  saveUndo(); drawing = true
  const p = getPos(e); startX = p.x; startY = p.y; lastX = p.x; lastY = p.y

  if (currentTool.value === 'lasso') { const sp = magneticLasso.value ? snapToEdge(p.x, p.y) : p; lassoPoints = [{ x: sp.x, y: sp.y }] }
  else if (currentTool.value === 'brush') { paintCircle(p.x, p.y); render() }
  else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'brush') { eraseCircle(p.x, p.y); render() }
    else { lassoPoints = eraserMode.value === 'lasso' ? [{ x: p.x, y: p.y }] : [] }
  }
}

function onMove(e) {
  if (panning) { panX.value = e.clientX - panSX; panY.value = e.clientY - panSY; return }
  const p = getPos(e)
  // 커서 표시
  if (!drawing && mCtx && (currentTool.value === 'brush' || currentTool.value === 'eraser')) {
    render()
    const col = currentTool.value === 'eraser' ? 'rgba(248,113,113,0.5)' : 'rgba(226,179,64,0.5)'
    mCtx.strokeStyle = col; mCtx.lineWidth = 2
    mCtx.beginPath(); mCtx.arc(p.x, p.y, brushSize.value, 0, Math.PI * 2); mCtx.stroke()
  }
  if (!drawing) return

  if (currentTool.value === 'box') {
    render()
    if (mCtx) { mCtx.strokeStyle = '#E2B340'; mCtx.lineWidth = 2; mCtx.setLineDash([6,4]); mCtx.strokeRect(startX, startY, p.x-startX, p.y-startY); mCtx.setLineDash([]) }
  } else if (currentTool.value === 'lasso') {
    const sp = magneticLasso.value ? snapToEdge(p.x, p.y) : p
    lassoPoints.push({ x: sp.x, y: sp.y }); render()
    if (mCtx && lassoPoints.length > 1) {
      mCtx.strokeStyle = magneticLasso.value ? '#60a5fa' : '#E2B340'; mCtx.lineWidth = 2; mCtx.setLineDash([4,3])
      mCtx.beginPath(); mCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y)
      for (let i = 1; i < lassoPoints.length; i++) mCtx.lineTo(lassoPoints[i].x, lassoPoints[i].y)
      mCtx.closePath(); mCtx.stroke(); mCtx.setLineDash([])
      mCtx.fillStyle = 'rgba(226,179,64,0.1)'; mCtx.fill()
    }
  } else if (currentTool.value === 'brush') {
    paintLine(lastX, lastY, p.x, p.y); lastX = p.x; lastY = p.y; render()
  } else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'brush') { eraseLine(lastX, lastY, p.x, p.y); lastX = p.x; lastY = p.y; render() }
    else if (eraserMode.value === 'box') { render(); if (mCtx) { mCtx.strokeStyle = '#f87171'; mCtx.lineWidth = 2; mCtx.setLineDash([6,4]); mCtx.strokeRect(startX, startY, p.x-startX, p.y-startY); mCtx.setLineDash([]) } }
    else if (eraserMode.value === 'lasso') { lassoPoints.push({ x: p.x, y: p.y }); render(); if (mCtx && lassoPoints.length > 1) { mCtx.strokeStyle = '#f87171'; mCtx.lineWidth = 2; mCtx.beginPath(); mCtx.moveTo(lassoPoints[0].x, lassoPoints[0].y); for (let i = 1; i < lassoPoints.length; i++) mCtx.lineTo(lassoPoints[i].x, lassoPoints[i].y); mCtx.closePath(); mCtx.stroke() } }
  }
}

function onUp(e) {
  if (panning) { panning = false; return }
  if (!drawing) return; drawing = false
  const p = getPos(e)
  if (currentTool.value === 'box') { fillRect(Math.min(startX,p.x), Math.min(startY,p.y), Math.max(startX,p.x), Math.max(startY,p.y)) }
  else if (currentTool.value === 'lasso') { if (lassoPoints.length > 2) fillPoly(lassoPoints); lassoPoints = [] }
  else if (currentTool.value === 'eraser') {
    if (eraserMode.value === 'box') { eraseRect(Math.min(startX,p.x), Math.min(startY,p.y), Math.max(startX,p.x), Math.max(startY,p.y)) }
    else if (eraserMode.value === 'lasso') { if (lassoPoints.length > 2) erasePoly(lassoPoints); lassoPoints = [] }
  }
  render(); updateHasMask()
}

function onWheel(e) { zoom.value = Math.max(0.2, Math.min(5, zoom.value * (e.deltaY > 0 ? 0.9 : 1.1))) }

// ── 마스크 조작 ──
function paintCircle(cx, cy) { if (!maskData || !srcImg) return; const w = srcImg.naturalWidth, h = srcImg.naturalHeight, r = brushSize.value; for (let y = Math.max(0,Math.floor(cy-r)); y < Math.min(h,Math.ceil(cy+r)); y++) for (let x = Math.max(0,Math.floor(cx-r)); x < Math.min(w,Math.ceil(cx+r)); x++) if ((x-cx)**2+(y-cy)**2<=r**2) maskData[y*w+x]=255 }
function paintLine(x0,y0,x1,y1) { const d=Math.hypot(x1-x0,y1-y0),s=Math.max(1,Math.ceil(d/Math.max(1,brushSize.value*0.3))); for(let i=0;i<=s;i++){const t=i/s;paintCircle(x0+(x1-x0)*t,y0+(y1-y0)*t)} }
function eraseCircle(cx,cy) { if(!maskData||!srcImg)return;const w=srcImg.naturalWidth,h=srcImg.naturalHeight,r=brushSize.value;for(let y=Math.max(0,Math.floor(cy-r));y<Math.min(h,Math.ceil(cy+r));y++)for(let x=Math.max(0,Math.floor(cx-r));x<Math.min(w,Math.ceil(cx+r));x++)if((x-cx)**2+(y-cy)**2<=r**2)maskData[y*w+x]=0 }
function eraseLine(x0,y0,x1,y1) { const d=Math.hypot(x1-x0,y1-y0),s=Math.max(1,Math.ceil(d/Math.max(1,brushSize.value*0.3))); for(let i=0;i<=s;i++){const t=i/s;eraseCircle(x0+(x1-x0)*t,y0+(y1-y0)*t)} }
function fillRect(x1,y1,x2,y2) { if(!maskData||!srcImg)return;const w=srcImg.naturalWidth,h=srcImg.naturalHeight;for(let y=Math.max(0,Math.round(y1));y<Math.min(h,Math.round(y2));y++)for(let x=Math.max(0,Math.round(x1));x<Math.min(w,Math.round(x2));x++)maskData[y*w+x]=255 }
function eraseRect(x1,y1,x2,y2) { if(!maskData||!srcImg)return;const w=srcImg.naturalWidth,h=srcImg.naturalHeight;for(let y=Math.max(0,Math.round(y1));y<Math.min(h,Math.round(y2));y++)for(let x=Math.max(0,Math.round(x1));x<Math.min(w,Math.round(x2));x++)maskData[y*w+x]=0 }
function fillPoly(pts) { if(!maskData||!srcImg||pts.length<3)return;const w=srcImg.naturalWidth,h=srcImg.naturalHeight;let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of pts){minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y)};for(let y=Math.max(0,Math.floor(minY));y<Math.min(h,Math.ceil(maxY));y++)for(let x=Math.max(0,Math.floor(minX));x<Math.min(w,Math.ceil(maxX));x++)if(pip(x,y,pts))maskData[y*w+x]=255 }
function erasePoly(pts) { if(!maskData||!srcImg||pts.length<3)return;const w=srcImg.naturalWidth,h=srcImg.naturalHeight;let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of pts){minX=Math.min(minX,p.x);minY=Math.min(minY,p.y);maxX=Math.max(maxX,p.x);maxY=Math.max(maxY,p.y)};for(let y=Math.max(0,Math.floor(minY));y<Math.min(h,Math.ceil(maxY));y++)for(let x=Math.max(0,Math.floor(minX));x<Math.min(w,Math.ceil(maxX));x++)if(pip(x,y,pts))maskData[y*w+x]=0 }
function pip(x,y,poly){let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const xi=poly[i].x,yi=poly[i].y,xj=poly[j].x,yj=poly[j].y;if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)inside=!inside};return inside}

function render() {
  if (!mCtx || !maskData || !srcImg) return
  const w = srcImg.naturalWidth, h = srcImg.naturalHeight
  mCtx.clearRect(0, 0, w, h)
  const id = mCtx.createImageData(w, h)
  for (let i = 0; i < maskData.length; i++) { if (maskData[i] > 0) { id.data[i*4]=226;id.data[i*4+1]=179;id.data[i*4+2]=64;id.data[i*4+3]=100 } }
  mCtx.putImageData(id, 0, 0)
}

// 자석 올가미
async function enableMagnetic() {
  magneticLasso.value = true
  if (!imagePath.value) return
  const bk = await getBackend()
  if (bk.getEdgeMap) {
    bk.getEdgeMap(imagePath.value, 50, 150, (b64) => {
      if (!b64) return
      const img = new Image()
      img.onload = () => {
        const tc = document.createElement('canvas'); tc.width = img.naturalWidth; tc.height = img.naturalHeight
        const tctx = tc.getContext('2d'); tctx.drawImage(img, 0, 0)
        const id = tctx.getImageData(0, 0, tc.width, tc.height)
        edgeMapW = tc.width; edgeMapH = tc.height
        edgeMapData = new Uint8Array(edgeMapW * edgeMapH)
        for (let i = 0; i < edgeMapData.length; i++) edgeMapData[i] = id.data[i * 4]
      }
      img.src = b64
    })
  }
}
function snapToEdge(x, y) {
  if (!edgeMapData || !magneticLasso.value) return { x, y }
  const r = 12; let best = Infinity, bx = x, by = y
  for (let py = Math.max(0, Math.floor(y-r)); py < Math.min(edgeMapH, Math.ceil(y+r)); py++)
    for (let px = Math.max(0, Math.floor(x-r)); px < Math.min(edgeMapW, Math.ceil(x+r)); px++)
      if (edgeMapData[py * edgeMapW + px] > 127) { const d = (px-x)**2+(py-y)**2; if (d < best) { best=d; bx=px; by=py } }
  return { x: bx, y: by }
}

function updateHasMask() { hasMask.value = maskData ? maskData.some(v => v > 0) : false }
function saveUndo() { if (maskData) { undoStack.push(new Uint8Array(maskData)); if (undoStack.length > 10) undoStack.shift(); redoStack = [] } }
function clearMask() { if (maskData) { saveUndo(); maskData.fill(0) }; hasMask.value = false; render() }
function undoMask() { if (!undoStack.length || !maskData) return; redoStack.push(new Uint8Array(maskData)); maskData.set(undoStack.pop()); updateHasMask(); render() }
function redoMask() { if (!redoStack.length || !maskData) return; undoStack.push(new Uint8Array(maskData)); maskData.set(redoStack.pop()); updateHasMask(); render() }

function getMaskBase64() {
  if (!maskData || !srcImg) return ''
  const w = srcImg.naturalWidth, h = srcImg.naturalHeight
  const tc = document.createElement('canvas'); tc.width = w; tc.height = h
  const tctx = tc.getContext('2d'); const id = tctx.createImageData(w, h)
  for (let i = 0; i < maskData.length; i++) { id.data[i*4]=id.data[i*4+1]=id.data[i*4+2]=maskData[i]; id.data[i*4+3]=255 }
  tctx.putImageData(id, 0, 0); return tc.toDataURL('image/png')
}

function generate() {
  requestAction('generate_inpaint', { image: imageSrc.value, image_path: imagePath.value, mask: getMaskBase64(), prompt: prompt.value, denoising: denoising.value, mask_content: maskContent.value, inpaint_area: inpaintArea.value })
}

onMounted(() => { onBackendEvent('inpaintImageLoaded', (path) => loadFromPath(path)) })
</script>

<style scoped>
.inpaint-workspace { height: 100%; display: flex; background: var(--bg-primary); }
.sidebar { width: 280px; display: flex; flex-direction: column; background: var(--bg-secondary); border-right: 1px solid var(--border); }
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
.sidebar-footer { padding: 10px; background: var(--bg-card); border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
.glass-card { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 10px; }
.source-thumb { height: 100px; border-radius: 6px; overflow: hidden; cursor: pointer; background: var(--bg-input); display: flex; align-items: center; justify-content: center; }
.source-thumb img { width: 100%; height: 100%; object-fit: contain; }
.upload-hint { color: var(--text-muted); font-size: 10px; font-weight: 700; }
.tool-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; }
.tool-grid.small { grid-template-columns: repeat(3, 1fr); }
.tool-chip { padding: 6px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: 9px; font-weight: 800; cursor: pointer; text-align: center; }
.tool-chip:hover { border-color: var(--text-muted); }
.tool-chip.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
.tool-chip.magnet.active { background: rgba(96,165,250,0.1); border-color: #60a5fa; color: #60a5fa; }
.slider-row { display: flex; align-items: center; gap: 6px; }
.slider-row input[type="range"] { flex: 1; accent-color: var(--accent); }
.slider-val { font-size: 10px; color: var(--accent); min-width: 36px; text-align: right; font-family: monospace; }
.mt-6 { margin-top: 6px; }
.mask-actions { display: flex; gap: 3px; }
.act-btn { flex: 1; padding: 5px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-secondary); font-size: 9px; font-weight: 700; cursor: pointer; }
.btn-gen { width: 100%; height: 42px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-weight: 800; font-size: 12px; cursor: pointer; }
.btn-gen:disabled { opacity: 0.4; }
.canvas-area { flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; background: #080808; position: relative; }
.canvas-wrap { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; position: relative; }
.canvas-wrap.dragging { background: rgba(250,204,21,0.05); }
.drop-empty { text-align: center; cursor: pointer; }
.drop-icon { font-size: 48px; opacity: 0.3; }
.drop-empty h2 { color: var(--text-muted); letter-spacing: 4px; }
.drop-empty p { color: #484848; font-size: 12px; }
.cv { position: absolute; max-width: 85%; max-height: 85%; }
.cv.mask { pointer-events: auto; }
.cv-info { position: absolute; bottom: 8px; right: 12px; color: #585858; font-size: 10px; background: rgba(0,0,0,0.6); padding: 2px 8px; border-radius: 4px; pointer-events: none; }
</style>
