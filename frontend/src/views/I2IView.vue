<template>
  <div class="i2i-root">
    <RefinePanel v-if="subTab === 'refine'" ref="refineRef"
      :image-path="imagePath"
      :sampler-options="samplerOptions"
      :scheduler-options="schedulerOptions"
      :checkpoint-options="checkpointOptions"
      @pick-image="triggerFileInput"
      @image-changed="p => imagePath = p"
    />

  <div class="i2i-workspace" v-show="subTab === 'i2i'">
    <!-- Left Sidebar: Settings -->
    <aside class="sidebar">
      <div class="sidebar-scroll">
        <!-- Input Card -->
        <div class="glass-card">
          <label>원본 이미지</label>
          <div class="source-thumb" @click="triggerFileInput">
            <img v-if="imageSrc" :src="imageSrc" />
            <div v-else class="upload-hint">비어 있음</div>
            <div class="edit-overlay">이미지 변경</div>
          </div>
        </div>

        <RelightPanel :image-src="imageSrc" @apply="applyRelight" />
        <button v-if="relightOriginal" type="button" class="clear-reference" @click="restoreRelightOriginal">조명 적용 전 원본 복원</button>

        <div v-if="isKrea2" class="glass-card krea-card">
          <label>아이덴티티 참조 <span class="optional">선택</span></label>
          <div class="source-thumb identity-thumb" @click="triggerReferenceInput">
            <img v-if="referenceSrc" :src="referenceSrc" />
            <div v-else class="upload-hint">원본만 편집</div>
            <div class="edit-overlay">{{ referenceSrc ? 'CHANGE REFERENCE' : 'ADD REFERENCE' }}</div>
          </div>
          <button v-if="referenceSrc" class="clear-reference" @click.stop="clearReference">참조 제거</button>
          <input ref="referenceFileInput" type="file" accept="image/*" hidden @change="handleReferenceSelect" />
        </div>

        <!-- Prompt Card -->
        <div class="glass-card">
          <label>프롬프트 덮어쓰기</label>
          <textarea v-model="prompt" rows="3" placeholder="Leave empty to use T2I prompt..."></textarea>
          <template v-if="!isKrea2">
            <label class="mt-12 danger">네거티브</label>
            <textarea v-model="negPrompt" rows="2" placeholder="Override negative..."></textarea>
          </template>
          <p v-else class="krea-help">Krea2 Identity Edit는 입력 이미지를 사용한 grounded unconditional 경로를 사용하므로 Negative는 적용하지 않습니다.</p>
        </div>

        <!-- Generation Params Card -->
        <div class="glass-card">
          <template v-if="isKrea2">
            <label>아이덴티티 유지도</label>
            <div class="premium-slider krea-slider">
              <input type="range" min="0.5" max="12" step="0.1" v-model.number="fidelity" />
              <div class="slider-display">
                <span class="val">{{ fidelity.toFixed(1) }}</span>
                <span class="label">참조 강도</span>
              </div>
            </div>
            <p class="krea-help">높을수록 원본/identity reference 보존을 강하게 요구합니다. 기본값 4.0.</p>
          </template>
          <template v-else>
            <label>디노이즈 강도</label>
            <div class="premium-slider">
              <input type="range" min="0" max="1" step="0.01" v-model.number="denoising" />
              <div class="slider-display">
                <span class="val">{{ denoising.toFixed(2) }}</span>
                <span class="label">강도</span>
              </div>
            </div>
          </template>

          <template v-if="!isKrea2">
            <label class="mt-12">크기 조정</label>
            <CustomSelect v-model="resizeModeLabel" :options="resizeModeOptions" placeholder="크기 조정" />
          </template>

          <div class="grid-2 mt-12">
            <div class="input-unit">
              <label>너비</label>
              <input v-model="width" type="number" />
            </div>
            <div class="input-unit">
              <label>높이</label>
              <input v-model="height" type="number" />
            </div>
          </div>
        </div>

        <!-- Advanced Card -->
        <details class="glass-card">
          <summary class="card-header">고급 설정</summary>
          <div class="input-group mt-12">
            <label>스텝</label>
            <input type="range" min="1" :max="isKrea2 ? 80 : 100" v-model.number="activeSteps" class="modern-slider" />
            <div class="val-tag">{{ activeSteps }}</div>
          </div>
          <div class="input-group mt-12">
            <label>CFG</label>
            <input type="range" min="1" :max="isKrea2 ? 10 : 20" step="0.5" v-model.number="activeCfg" class="modern-slider" />
            <div class="val-tag">{{ activeCfg }}</div>
          </div>
          <div class="input-group mt-12">
            <label>Seed (−1 = 랜덤)</label>
            <div class="seed-row">
              <input v-model="seed" type="text" class="seed-input" placeholder="-1" />
              <button class="seed-btn" @click="seed = '-1'" title="랜덤으로 초기화"><Icon name="dice" /></button>
            </div>
          </div>
        </details>
      </div>

      <div class="sidebar-footer">
        <button class="btn-generate primary" @click="generate" :disabled="!imageSrc">
          {{ !imageSrc ? '이미지를 먼저 올리세요' : isKrea2 ? 'Krea2 아이덴티티 편집 시작' : 'I2I 생성 시작' }}
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
            <h2>원본을 끌어 놓으세요</h2>
            <p>또는 클릭해서 파일 찾기</p>
          </div>
          <div v-else class="preview-container">
            <img :src="imageSrc" class="main-preview" />
          </div>
        </transition>
      </div>
      <input ref="fileInput" type="file" accept="image/*" hidden @change="handleFileSelect" />
    </section>
  </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { requestAction, getProperty, getValue } from '../stores/widgetStore.js'
import { onBackendEvent } from '../bridge.js'
import { mediaUrl } from '../utils/media.js'
import CustomSelect from '../components/CustomSelect.vue'
import RefinePanel from '../components/RefinePanel.vue'
import RelightPanel from '../components/RelightPanel.vue'
import { useViewMode } from '../composables/useViewMode'
import { useTabDefaultsFollower } from '../composables/useTabDefaultsFollower'
import { followDefault } from '../utils/tabDefaults'

/** 하위 탭(img2img / SAM3 정밀화)은 왼쪽 레일의 서랍이 정한다 — `useViewMode` 참조. */
const { mode: subTab } = useViewMode('i2i')
const refineRef = ref<any>(null)

// Refine 패널 드롭다운 — t2i가 쓰는 것과 같은 위젯 property를 읽는다.
// (전용 이벤트를 새로 만들면 브리지 계약이 어긋난다 — tests/test_bridge_contract.py)
const samplerOptions = computed(
  () => ['Use same sampler', ...(getProperty('sampler_combo', 'items') || [])])
const schedulerOptions = computed(
  () => ['Use same scheduler', ...(getProperty('scheduler_combo', 'items') || [])])
const checkpointOptions = computed(
  () => getProperty('_sam3_checkpoint', 'items') || ['sam3.pt'])

const isDragging = ref(false)
const imageSrc = ref('')
const imagePath = ref('')
const relightOriginal = ref<{ src: string; path: string } | null>(null)
function applyRelight(result: { image: string; width: number; height: number }) {
  if (!relightOriginal.value) relightOriginal.value = { src: imageSrc.value, path: imagePath.value }
  imagePath.value = ''
  imageSrc.value = result.image
}
function restoreRelightOriginal() {
  const original = relightOriginal.value
  if (!original) return
  imageSrc.value = original.src; imagePath.value = original.path
  relightOriginal.value = null
}
const fileInput = ref<HTMLInputElement | null>(null)
const referenceSrc = ref('')
const referencePath = ref('')
const referenceFileInput = ref<HTMLInputElement | null>(null)
const prompt = ref('')
const negPrompt = ref('')
const denoising = ref(0.75)
// Settings '기본값' 의 Denoising(I2I) — 예전엔 저장만 되고 여기서는 늘 0.75 로 시작했다(audit #140).
// 사용자가 이 화면에서 바꾸지 않은 동안만 기본값을 따른다.
useTabDefaultsFollower((next, prev) => {
  denoising.value = followDefault(denoising.value, prev.denoising, next.denoising)
})
const fidelity = ref(4)
const resizeMode = ref('0')
const resizeModeOptions = ['그대로 늘리기', '잘라서 맞추기', '여백 채우기', 'Latent 리사이즈']
const resizeModeLabel = computed({
  get: () => resizeModeOptions[parseInt(resizeMode.value)] || resizeModeOptions[0],
  set: (v: string) => { resizeMode.value = String(resizeModeOptions.indexOf(v)) }
})
const width = ref('1024')
const height = ref('1024')
const steps = ref(20)
const cfg = ref(7)
const kreaSteps = ref(15)
const kreaCfg = ref(1)
const seed = ref('-1')

const generationFamily = computed(() => String(getValue('generation_family_combo') || 'STANDARD').toUpperCase())
const isKrea2 = computed(() => generationFamily.value === 'KREA2')
const activeSteps = computed({
  get: () => isKrea2.value ? kreaSteps.value : steps.value,
  set: (value: number) => { if (isKrea2.value) kreaSteps.value = value; else steps.value = value },
})
const activeCfg = computed({
  get: () => isKrea2.value ? kreaCfg.value : cfg.value,
  set: (value: number) => { if (isKrea2.value) kreaCfg.value = value; else cfg.value = value },
})

function triggerFileInput() { fileInput.value?.click() }
function handleFileSelect(e: Event) { const f = (e.target as HTMLInputElement).files?.[0]; if (f) loadFile(f) }
function triggerReferenceInput() { referenceFileInput.value?.click() }
function handleReferenceSelect(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = (ev: ProgressEvent<FileReader>) => { referenceSrc.value = ev.target?.result as string }
  reader.readAsDataURL(file)
  referencePath.value = ((file as any).path || '').replace(/\\/g, '/')
}
function clearReference() {
  referenceSrc.value = ''
  referencePath.value = ''
  if (referenceFileInput.value) referenceFileInput.value.value = ''
}
function handleDrop(e: DragEvent) {
  isDragging.value = false
  // 파일 드롭
  const f = e.dataTransfer?.files?.[0]
  if (f) { imagePath.value = (f as any).path || ''; loadFile(f); return }
  // History에서 경로 텍스트 드롭
  const path = e.dataTransfer?.getData('text/plain')
  if (path && path.includes('/')) loadFromPath(path)
}
function loadFile(file: File) {
  relightOriginal.value = null
  const reader = new FileReader()
  reader.onload = (ev: ProgressEvent<FileReader>) => { imageSrc.value = ev.target?.result as string }
  reader.readAsDataURL(file)
  imagePath.value = ((file as any).path || '').replace(/\\/g, '/')
}

// 경로로 직접 이미지 로드 (History/Gallery에서 전송 시)
async function loadFromPath(path: string) {
  relightOriginal.value = null
  const normalized = path.replace(/\\/g, '/')
  imagePath.value = normalized
  imageSrc.value = mediaUrl(normalized)
}

onMounted(() => {
  // History/Gallery에서 send_to_i2i 시 이미지 로드 — 현재 하위 탭 쪽으로 보낸다
  onBackendEvent('i2iImageLoaded', (path: string) => {
    loadFromPath(path)
    if (subTab.value === 'refine') refineRef.value?.setImage?.(path)
  })
})

// Refine 탭으로 넘어갈 때 현재 이미지를 물려준다
watch(subTab, (v) => {
  if (v === 'refine' && imagePath.value) refineRef.value?.setImage?.(imagePath.value)
})

function generate() {
  requestAction('generate_i2i', {
    generation_family: isKrea2.value ? 'krea2' : 'standard',
    image: imagePath.value ? '' : imageSrc.value,
    image_path: imagePath.value,
    reference_image: isKrea2.value && !referencePath.value ? referenceSrc.value : '',
    reference_path: isKrea2.value ? referencePath.value : '',
    prompt: prompt.value,
    negative_prompt: negPrompt.value,
    denoising: denoising.value,
    fidelity: fidelity.value,
    resize_mode: parseInt(resizeMode.value),
    width: parseInt(width.value),
    height: parseInt(height.value),
    steps: activeSteps.value,
    cfg: activeCfg.value,
    seed: seed.value,
  })
}
</script>

<style scoped>
.i2i-root { height: 100%; display: flex; flex-direction: column; background: var(--bg-primary); min-height: 0; }
.i2i-root > * { flex: 1; min-height: 0; }

.i2i-workspace { height: 100%; display: flex; background: var(--bg-primary); }

/* Sidebar */
.sidebar {
  width: 340px; display: flex; flex-direction: column;
  background: var(--bg-secondary); border-right: 1px solid var(--border);
}
.sidebar-scroll { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 16px; }
.sidebar-footer { padding: 16px; background: var(--bg-card); border-top: 1px solid var(--border); }

.krea-card { border-color: rgba(167,139,250,0.35); background: rgba(124,58,237,0.06); }
/* 보라색이었지만 태그 분류가 아니라 '선택 항목' 표시라 정보색으로 간다 */
.optional { margin-left: 5px; font-size: var(--fs-label); color: var(--state-info-fg); letter-spacing: 0; }
.identity-thumb { aspect-ratio: 2/1; }
.clear-reference {
  width: 100%; margin-top: 7px; padding: 6px; background: transparent;
  border: 1px solid rgba(248,113,113,0.3); border-radius: 5px;
  color: var(--state-alert-fg); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer;
}
.krea-help { margin: 8px 0 0; color: var(--text-muted); font-size: var(--fs-label); line-height: 1.45; }
.krea-slider .val { color: var(--state-info-fg); }

.source-thumb {
  width: 100%; aspect-ratio: 16/9; background: var(--bg-input); border-radius: var(--radius-base);
  margin-top: 8px; overflow: hidden; position: relative; cursor: pointer; border: 1px solid var(--border);
}
.source-thumb img { width: 100%; height: 100%; object-fit: cover; }
.upload-hint { height: 100%; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.edit-overlay {
  position: absolute; inset: 0; background: rgba(0,0,0,0.6); color: var(--accent);
  display: flex; align-items: center; justify-content: center; font-size: var(--fs-label); font-weight: var(--fw-bold);
  opacity: 0; transition: var(--transition);
}
.source-thumb:hover .edit-overlay { opacity: 1; }

/* Premium Slider */
.premium-slider { margin-top: 8px; }
.slider-display { display: flex; justify-content: space-between; align-items: baseline; margin-top: 4px; }
.slider-display .val { font-size: 20px; font-weight: var(--fw-bold); color: var(--accent); font-family: 'Consolas', monospace; }
.slider-display .label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); }

.modern-slider { appearance: none; width: 100%; height: 4px; background: var(--bg-input); border-radius: 2px; outline: none; accent-color: var(--accent); }
.val-tag { font-size: 11px; font-weight: var(--fw-bold); color: var(--text-secondary); text-align: right; margin-top: 4px; }
.seed-row { display: flex; gap: 6px; align-items: center; }
.seed-input {
  flex: 1; padding: 8px 10px; background: var(--bg-input); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-primary); font-size: 12px; outline: none;
  font-family: 'Consolas', monospace;
}
.seed-input:focus { border-color: var(--accent); }
.seed-btn {
  padding: 8px 12px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--accent); font-size: 14px; cursor: pointer;
}
.seed-btn:hover { background: var(--bg-input); border-color: var(--accent); }

.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.mt-12 { margin-top: 12px; }
.danger { color: var(--state-alert-fg); }

/* 주 버튼: 면은 --accent 가 아니라 글자가 읽히게 민 --accent-fill */
.btn-generate {
  width: 100%; height: 46px; background: var(--accent-fill); border: none;
  border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold);
  font-size: 12px; letter-spacing: 0; cursor: pointer; transition: var(--transition);
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
.drop-empty h2 { font-size: 18px; letter-spacing: 0; color: var(--text-secondary); margin-bottom: 8px; }
.drop-empty p { font-size: 13px; color: var(--text-muted); }

.preview-container { width: 100%; height: 100%; padding: 20px; display: flex; align-items: center; justify-content: center; }
.main-preview { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 8px; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }

/* Animation */
.scale-enter-active, .scale-leave-active { transition: all 0.3s ease; }
.scale-enter-from, .scale-leave-to { opacity: 0; transform: scale(0.95); }
</style>
