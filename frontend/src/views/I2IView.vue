<template>
  <div class="i2i-workspace">
    <!-- Left Sidebar: Settings -->
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <!-- Input Card -->
        <div class="glass-card">
          <label>Source Image</label>
          <div class="source-thumb" @click="triggerFileInput">
            <img v-if="imageSrc" :src="imageSrc" />
            <div v-else class="upload-hint">EMPTY</div>
            <div class="edit-overlay">CHANGE IMAGE</div>
          </div>
        </div>

        <!-- Prompt Card -->
        <div class="glass-card">
          <label>Override Prompt</label>
          <textarea v-model="prompt" rows="3" placeholder="Leave empty to use T2I prompt..."></textarea>
          <label class="mt-12 danger">Negative</label>
          <textarea v-model="negPrompt" rows="2" placeholder="Override negative..."></textarea>
        </div>

        <!-- Generation Params Card -->
        <div class="glass-card">
          <label>Denoising Strength</label>
          <div class="premium-slider">
            <input type="range" min="0" max="1" step="0.01" v-model.number="denoising" />
            <div class="slider-display">
              <span class="val">{{ denoising.toFixed(2) }}</span>
              <span class="label">Intensity</span>
            </div>
          </div>

          <label class="mt-12">Resize Mode</label>
          <CustomSelect v-model="resizeModeLabel" :options="resizeModeOptions" placeholder="Resize Mode" />

          <div class="grid-2 mt-12">
            <div class="input-unit">
              <label>Width</label>
              <input v-model="width" type="number" />
            </div>
            <div class="input-unit">
              <label>Height</label>
              <input v-model="height" type="number" />
            </div>
          </div>
        </div>

        <!-- Advanced Card -->
        <details class="glass-card">
          <summary class="card-header">ADVANCED SETTINGS</summary>
          <div class="input-group mt-12">
            <label>Steps</label>
            <input type="range" min="1" max="100" v-model.number="steps" class="modern-slider" />
            <div class="val-tag">{{ steps }}</div>
          </div>
          <div class="input-group mt-12">
            <label>CFG Scale</label>
            <input type="range" min="1" max="20" step="0.5" v-model.number="cfg" class="modern-slider" />
            <div class="val-tag">{{ cfg }}</div>
          </div>
        </details>
      </div>

      <div class="sidebar-footer">
        <button class="btn-generate primary" @click="generate" :disabled="!imageSrc">
          {{ !imageSrc ? 'UPLOAD IMAGE FIRST' : 'START I2I GENERATION' }}
        </button>
      </div>
    </aside>

    <!-- Main Content: Canvas -->
    <section class="canvas-area">
      <div class="drop-zone" :class="{ 'drag-over': isDragging }"
        @dragover.prevent="isDragging = true" @dragleave="isDragging = false"
        @drop.prevent="handleDrop" @click="triggerFileInput"
      >
        <transition name="scale">
          <div v-if="!imageSrc" class="drop-empty">
            <div class="icon">⤓</div>
            <h2>DRAG & DROP SOURCE</h2>
            <p>Or click to browse your local storage</p>
          </div>
          <div v-else class="preview-container">
            <img :src="imageSrc" class="main-preview" />
          </div>
        </transition>
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

const isDragging = ref(false)
const imageSrc = ref('')
const imagePath = ref('')
const fileInput = ref(null)
const prompt = ref('')
const negPrompt = ref('')
const denoising = ref(0.75)
const resizeMode = ref('0')
const resizeModeOptions = ['JUST RESIZE', 'CROP AND RESIZE', 'RESIZE AND FILL', 'LATENT RESIZE']
const resizeModeLabel = computed({
  get: () => resizeModeOptions[parseInt(resizeMode.value)] || resizeModeOptions[0],
  set: v => { resizeMode.value = String(resizeModeOptions.indexOf(v)) }
})
const width = ref('1024')
const height = ref('1024')
const steps = ref(20)
const cfg = ref(7)
const seed = ref('-1')

function triggerFileInput() { fileInput.value?.click() }
function handleFileSelect(e) { const f = e.target.files?.[0]; if (f) loadFile(f) }
function handleDrop(e) {
  isDragging.value = false
  // 파일 드롭
  const f = e.dataTransfer?.files?.[0]
  if (f) { imagePath.value = f.path || ''; loadFile(f); return }
  // History에서 경로 텍스트 드롭
  const path = e.dataTransfer?.getData('text/plain')
  if (path && path.includes('/')) loadFromPath(path)
}
function loadFile(file) {
  const reader = new FileReader()
  reader.onload = (ev) => { imageSrc.value = ev.target.result }
  reader.readAsDataURL(file)
  if (file.path) imagePath.value = file.path.replace(/\\/g, '/')
}

// 경로로 직접 이미지 로드 (History/Gallery에서 전송 시)
async function loadFromPath(path) {
  imagePath.value = path
  const backend = await getBackend()
  if (backend.loadImageBase64) {
    backend.loadImageBase64(path, (b64) => {
      if (b64) imageSrc.value = b64
    })
  } else {
    imageSrc.value = 'file:///' + path
  }
}

onMounted(() => {
  // History/Gallery에서 send_to_i2i 시 이미지 로드
  onBackendEvent('i2iImageLoaded', (path) => loadFromPath(path))
})

function generate() {
  requestAction('generate_i2i', {
    image: imageSrc.value,
    image_path: imagePath.value,
    prompt: prompt.value,
    negative_prompt: negPrompt.value,
    denoising: denoising.value,
    resize_mode: parseInt(resizeMode.value),
    width: parseInt(width.value),
    height: parseInt(height.value),
    steps: steps.value,
    cfg: cfg.value,
    seed: seed.value,
  })
}
</script>

<style scoped>
.i2i-workspace { height: 100%; display: flex; background: var(--bg-primary); }

/* Sidebar */
.sidebar {
  width: 340px; display: flex; flex-direction: column;
  background: var(--bg-secondary); border-right: 1px solid var(--border);
}
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.sidebar-footer { padding: 16px; background: var(--bg-card); border-top: 1px solid var(--border); }

.glass-card {
  background: rgba(255,255,255,0.02); border: 1px solid var(--border);
  border-radius: var(--radius-card); padding: 14px;
}

.source-thumb {
  width: 100%; aspect-ratio: 16/9; background: var(--bg-input); border-radius: var(--radius-base);
  margin-top: 8px; overflow: hidden; position: relative; cursor: pointer; border: 1px solid var(--border);
}
.source-thumb img { width: 100%; height: 100%; object-fit: cover; }
.upload-hint { height: 100%; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 10px; font-weight: 800; letter-spacing: 2px; }
.edit-overlay {
  position: absolute; inset: 0; background: rgba(0,0,0,0.6); color: var(--accent);
  display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 800;
  opacity: 0; transition: var(--transition);
}
.source-thumb:hover .edit-overlay { opacity: 1; }

/* Premium Slider */
.premium-slider { margin-top: 8px; }
.slider-display { display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px; }
.slider-display .val { font-size: 20px; font-weight: 900; color: var(--accent); font-family: 'Consolas', monospace; }
.slider-display .label { font-size: 9px; font-weight: 800; color: var(--text-muted); text-transform: uppercase; }

.modern-slider { appearance: none; width: 100%; height: 4px; background: var(--bg-input); border-radius: 2px; outline: none; accent-color: var(--accent); }
.val-tag { font-size: 11px; font-weight: 800; color: var(--text-secondary); text-align: right; margin-top: 4px; }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mt-12 { margin-top: 12px; }
.danger { color: #f87171; }

.btn-generate {
  width: 100%; height: 46px; background: var(--accent); border: none;
  border-radius: var(--radius-pill); color: #000; font-weight: 900;
  font-size: 12px; letter-spacing: 1px; cursor: pointer; transition: var(--transition);
}
.btn-generate:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250, 204, 21, 0.3); }

/* Canvas Area */
.canvas-area { flex: 1; padding: 24px; display: flex; align-items: center; justify-content: center; }
.drop-zone {
  width: 100%; height: 100%; max-width: 1200px;
  background: var(--bg-card); border: 2px dashed var(--border); border-radius: 20px;
  display: flex; align-items: center; justify-content: center; cursor: pointer;
  transition: var(--transition); position: relative;
}
.drop-zone.drag-over { border-color: var(--accent); background: var(--accent-dim); }
.drop-zone:hover { border-color: var(--text-muted); }

.drop-empty { text-align: center; }
.drop-empty .icon { font-size: 64px; color: var(--text-muted); margin-bottom: 16px; }
.drop-empty h2 { font-size: 18px; letter-spacing: 4px; color: var(--text-secondary); margin-bottom: 8px; }
.drop-empty p { font-size: 13px; color: var(--text-muted); }

.preview-container { width: 100%; height: 100%; padding: 20px; display: flex; align-items: center; justify-content: center; }
.main-preview { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }

/* Animation */
.scale-enter-active, .scale-leave-active { transition: all 0.3s ease; }
.scale-enter-from, .scale-leave-to { opacity: 0; transform: scale(0.95); }
</style>
