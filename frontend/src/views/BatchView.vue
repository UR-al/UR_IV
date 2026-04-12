<template>
  <div class="batch-view">
    <!-- 서브탭 -->
    <div class="sub-tabs">
      <button class="sub-tab" :class="{ active: subTab === 'batch' }" @click="subTab = 'batch'">BATCH</button>
      <button class="sub-tab" :class="{ active: subTab === 'upscale' }" @click="subTab = 'upscale'">UPSCALE</button>
      <button class="sub-tab" :class="{ active: subTab === 'adetailer' }" @click="subTab = 'adetailer'">ADETAILER</button>
    </div>

    <!-- Batch 탭 -->
    <div v-if="subTab === 'batch'" class="tab-body">
      <div class="panel">
        <h3>BATCH PROCESSING</h3>
        <div class="file-drop" @dragover.prevent @drop.prevent="onDropBatch">
          <div v-if="batchFiles.length === 0" class="drop-hint">
            이미지 드래그 또는 <button class="link-btn" @click="action('open_batch_files')">파일 선택</button>
          </div>
          <div v-else class="file-list">
            <div v-for="(f, i) in batchFiles" :key="i" class="file-item">
              <span>{{ basename(f) }}</span>
              <button class="rm-btn" @click="batchFiles.splice(i, 1)">×</button>
            </div>
          </div>
        </div>
        <label class="s-label">작업</label>
        <CustomSelect v-model="batchOp" :options="['resize', 'format']" placeholder="작업 선택" />
        <div v-if="batchOp === 'resize'" class="op-settings">
          <div class="row"><input class="s-input" v-model="resizeW" placeholder="W" /><span>×</span><input class="s-input" v-model="resizeH" placeholder="H" /></div>
        </div>
        <div v-if="batchOp === 'format'" class="op-settings">
          <CustomSelect v-model="formatType" :options="['PNG', 'JPEG', 'WEBP']" placeholder="포맷" />
        </div>
        <button class="btn-start" @click="startBatch" :disabled="batchFiles.length === 0">
          배치 시작 ({{ batchFiles.length }}파일)
        </button>
      </div>
    </div>

    <!-- Upscale 탭 -->
    <div v-if="subTab === 'upscale'" class="tab-body">
      <div class="panel">
        <h3>UPSCALE</h3>
        <div class="file-drop" @dragover.prevent @drop.prevent="onDropUpscale">
          <div v-if="upscaleFiles.length === 0" class="drop-hint">
            이미지 드래그 또는 <button class="link-btn" @click="action('open_upscale_files')">파일 선택</button>
          </div>
          <div v-else class="file-list">
            <div v-for="(f, i) in upscaleFiles" :key="i" class="file-item">
              <span>{{ basename(f) }}</span>
              <button class="rm-btn" @click="upscaleFiles.splice(i, 1)">×</button>
            </div>
          </div>
        </div>
        <label class="s-label">업스케일러</label>
        <CustomSelect v-model="upscaler" :options="upscalers" placeholder="업스케일러 선택..." />
        <label class="s-label">배율</label>
        <div class="slider-row">
          <input type="range" min="1" max="4" step="0.5" v-model.number="scaleFactor" />
          <span>{{ scaleFactor }}x</span>
        </div>
        <button class="btn-start" @click="startUpscale" :disabled="upscaleFiles.length === 0">
          업스케일 시작 ({{ upscaleFiles.length }}파일)
        </button>
      </div>
    </div>

    <!-- ADetailer 탭 -->
    <div v-if="subTab === 'adetailer'" class="tab-body ad-layout">
      <!-- 좌측: 설정 -->
      <div class="ad-settings">
        <h3>ADETAILER</h3>
        <div class="file-drop" @dragover.prevent @drop.prevent="onDropAd">
          <div v-if="adFiles.length === 0" class="drop-hint">
            이미지 드래그 또는
            <button class="link-btn" @click="action('open_ad_files')">파일 선택</button>
            /
            <button class="link-btn" @click="action('open_ad_folder')">폴더 선택</button>
          </div>
          <div v-else class="file-list">
            <div v-for="(f, i) in adFiles" :key="i" class="file-item"
              :class="{ active: adCurrentIdx === i, done: adResults[i] }"
              @click="previewAdFile(i)">
              <span>{{ basename(f) }}</span>
              <span v-if="adResults[i]" class="done-badge">✓</span>
              <button class="rm-btn" @click.stop="adFiles.splice(i, 1)">×</button>
            </div>
          </div>
        </div>
        <div class="file-count" v-if="adFiles.length">{{ adFiles.length }}개 파일</div>

        <label class="s-label">AD Model</label>
        <CustomSelect v-model="adModel" :options="adModelItems" placeholder="AD Model..." />
        <div class="ad-params">
          <div class="ad-param">
            <label>Confidence</label>
            <input type="number" v-model.number="adConfidence" step="0.05" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Denoise</label>
            <input type="number" v-model.number="adDenoise" step="0.05" min="0" max="1" />
          </div>
          <div class="ad-param">
            <label>Prompt</label>
            <label class="ad-toggle"><input type="checkbox" v-model="adUseExifPrompt" /><span>EXIF 프롬프트 사용</span></label>
            <input v-if="!adUseExifPrompt" type="text" v-model="adPrompt" placeholder="(선택사항)" />
            <div v-else class="exif-prompt-hint">각 이미지의 EXIF에서 Positive/Negative를 자동으로 읽어 사용합니다</div>
          </div>
        </div>

        <!-- 프로그레스 -->
        <div class="ad-progress" v-if="adProcessing">
          <div class="progress-bar">
            <div class="progress-fill" :style="{ width: adProgressPct + '%' }"></div>
          </div>
          <span>{{ adProgressCur }}/{{ adProgressTotal }}</span>
        </div>

        <div class="ad-actions">
          <button class="btn-start" @click="runAdSingle" :disabled="adFiles.length === 0 || adProcessing">
            현재 이미지 적용
          </button>
          <button class="btn-start batch" @click="runAdBatch" :disabled="adFiles.length === 0 || adProcessing">
            전체 배치 ({{ adFiles.length }}장)
          </button>
          <button class="btn-stop" v-if="adProcessing" @click="action('stop_adetailer_batch')">
            중지
          </button>
        </div>
      </div>

      <!-- 우측: Before/After 비교 -->
      <div class="ad-compare">
        <div v-if="adBefore && adAfter" class="compare-split">
          <div class="compare-col">
            <div class="compare-label">BEFORE</div>
            <img :src="'file:///' + adBefore" />
          </div>
          <div class="compare-col">
            <div class="compare-label">AFTER</div>
            <img :src="'file:///' + adAfter" />
          </div>
        </div>
        <div v-else-if="adPreview" class="preview-single">
          <img :src="'file:///' + adPreview" />
        </div>
        <div v-else class="compare-empty">
          좌측에서 이미지를 선택하면 미리보기가 표시됩니다
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import CustomSelect from '../components/CustomSelect.vue'

const subTab = ref('batch')
const action = (name, payload = {}) => requestAction(name, payload)
const basename = (p) => typeof p === 'string' ? p.split('/').pop().split('\\').pop() : p.name || p

// ── Batch ──
const batchFiles = ref([])
const batchOp = ref('resize')
const resizeW = ref('1024')
const resizeH = ref('1024')
const formatType = ref('PNG')

function onDropBatch(e) {
  const files = Array.from(e.dataTransfer?.files || [])
  batchFiles.value.push(...files.filter(f => f.type.startsWith('image/')).map(f => f.path))
}
function startBatch() {
  action('start_batch', {
    files: batchFiles.value,
    operation: batchOp.value,
    settings: { width: resizeW.value, height: resizeH.value, format: formatType.value },
  })
}

// ── Upscale ──
const upscaleFiles = ref([])
const upscaler = ref('')
const upscalers = ref(['R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B'])
const scaleFactor = ref(2)

function onDropUpscale(e) {
  const files = Array.from(e.dataTransfer?.files || [])
  upscaleFiles.value.push(...files.filter(f => f.type.startsWith('image/')).map(f => f.path))
}
function startUpscale() {
  action('start_upscale', {
    files: upscaleFiles.value,
    upscaler: upscaler.value,
    scale: scaleFactor.value,
  })
}

// ── ADetailer ──
const adFiles = ref([])
const adModel = ref('face_yolov8n.pt')
const adModelItems = ref([])
const adConfidence = ref(0.3)
const adDenoise = ref(0.4)
const adPrompt = ref('')
const adUseExifPrompt = ref(false)
const adCurrentIdx = ref(-1)
const adPreview = ref('')
const adBefore = ref('')
const adAfter = ref('')
const adResults = ref({})  // index → true
const adProcessing = ref(false)
const adProgressCur = ref(0)
const adProgressTotal = ref(0)
const adProgressPct = computed(() => adProgressTotal.value ? Math.round(adProgressCur.value / adProgressTotal.value * 100) : 0)

function onDropAd(e) {
  const files = Array.from(e.dataTransfer?.files || [])
  adFiles.value.push(...files.filter(f => f.type.startsWith('image/')).map(f => f.path))
}

function previewAdFile(i) {
  adCurrentIdx.value = i
  adPreview.value = adFiles.value[i]
  adBefore.value = ''
  adAfter.value = ''
}

function _adSettings() {
  return {
    ad_model: adModel.value,
    ad_confidence: adConfidence.value,
    ad_denoise: adDenoise.value,
    ad_prompt: adUseExifPrompt.value ? '' : adPrompt.value,
    use_exif_prompt: adUseExifPrompt.value,
  }
}

function runAdSingle() {
  const idx = adCurrentIdx.value >= 0 ? adCurrentIdx.value : 0
  const path = adFiles.value[idx]
  if (!path) return
  adProcessing.value = true
  adBefore.value = path
  adAfter.value = ''
  action('run_adetailer_single', { path, settings: _adSettings() })
}

function runAdBatch() {
  if (!adFiles.value.length) return
  adProcessing.value = true
  adResults.value = {}
  adProgressCur.value = 0
  adProgressTotal.value = adFiles.value.length
  action('run_adetailer_batch', { paths: adFiles.value, settings: _adSettings() })
}

// 외부에서 이미지 수신 (History/Gallery 우클릭 → "ADetailer 적용")
const props = defineProps({ initialAdPath: { type: String, default: '' } })

onMounted(async () => {
  const backend = await getBackend()

  // 업스케일러 로드
  if (backend.getUpscalers) {
    backend.getUpscalers((json) => {
      try {
        const list = JSON.parse(json)
        if (list.length) { upscalers.value = list; upscaler.value = list[0] }
      } catch {}
    })
  }

  // AD 모델 로드
  if (backend.getADetailerModels) {
    backend.getADetailerModels((json) => {
      try {
        const models = JSON.parse(json)
        if (models.length) { adModelItems.value = models; adModel.value = models[0] }
      } catch {}
    })
  }

  // 파일 선택 이벤트
  onBackendEvent('batchFilesSelected', (json) => {
    try {
      const paths = JSON.parse(json)
      if (subTab.value === 'adetailer') {
        adFiles.value.push(...paths)
      } else if (subTab.value === 'upscale') {
        upscaleFiles.value.push(...paths)
      } else {
        batchFiles.value.push(...paths)
      }
    } catch {}
  })

  // ADetailer 결과 수신
  onBackendEvent('adetailerResult', (json) => {
    try {
      const d = JSON.parse(json)
      if (d.error) {
        requestAction('show_toast', { type: 'error', msg: `AD 오류: ${d.error}` })
        adProcessing.value = false
        return
      }
      adBefore.value = d.before
      adAfter.value = d.after
      if (typeof d.index === 'number') {
        adResults.value[d.index] = true
        adCurrentIdx.value = d.index
      }
      // 단일 처리 완료
      if (typeof d.index !== 'number') {
        adProcessing.value = false
        requestAction('show_toast', { type: 'success', msg: 'ADetailer 완료' })
      }
    } catch {}
  })

  // 배치 진행률
  onBackendEvent('adetailerProgress', (cur, total) => {
    adProgressCur.value = cur
    adProgressTotal.value = total
    if (cur >= total) adProcessing.value = false
  })
})
</script>

<style scoped>
.batch-view { width: 100%; height: 100%; display: flex; flex-direction: column; }
.sub-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.sub-tab {
  flex: 1; padding: 8px; background: transparent; border: none; border-bottom: 2px solid transparent;
  color: var(--text-muted); font-size: 11px; font-weight: 800; cursor: pointer; text-align: center;
  letter-spacing: 1px;
}
.sub-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-body { flex: 1; overflow-y: auto; padding: 20px; }

.panel { max-width: 500px; margin: 0 auto; display: flex; flex-direction: column; gap: 10px; }
.panel h3 { color: var(--text-primary); font-size: 13px; font-weight: 900; letter-spacing: 2px; margin: 0; }

.file-drop {
  border: 2px dashed var(--border); border-radius: 8px; min-height: 100px;
  display: flex; align-items: center; justify-content: center; padding: 12px;
}
.drop-hint { color: var(--text-muted); font-size: 12px; text-align: center; }
.link-btn { background: none; border: none; color: var(--accent); cursor: pointer; text-decoration: underline; font-size: 12px; }
.file-list { width: 100%; max-height: 200px; overflow-y: auto; }
.file-item {
  display: flex; align-items: center; gap: 6px;
  padding: 4px 8px; font-size: 11px; color: var(--text-secondary); cursor: pointer;
  border-radius: 4px;
}
.file-item span { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-item.active { background: var(--accent-dim); color: var(--accent); }
.file-item.done { opacity: 0.6; }
.done-badge { color: #4ade80; font-weight: 900; flex: 0 !important; }
.rm-btn { background: none; border: none; color: #f87171; cursor: pointer; font-size: 14px; flex-shrink: 0; }
.file-count { font-size: 10px; color: var(--text-muted); }

.s-label { color: var(--text-muted); font-size: 10px; font-weight: 700; letter-spacing: 1px; }
.s-select, .s-input {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 10px; color: var(--text-primary); font-size: 12px; outline: none;
}
.s-select:focus, .s-input:focus { border-color: var(--accent); }
.row { display: flex; align-items: center; gap: 6px; }
.row span { color: var(--text-muted); }
.slider-row { display: flex; align-items: center; gap: 8px; }
.slider-row input { flex: 1; accent-color: var(--accent); }
.slider-row span { color: var(--text-secondary); font-size: 12px; min-width: 30px; font-family: monospace; }
.op-settings { display: flex; flex-direction: column; gap: 4px; }
.btn-start {
  padding: 10px; background: var(--accent); border: none; border-radius: 8px;
  color: #000; font-weight: 800; font-size: 11px; cursor: pointer; letter-spacing: 1px;
}
.btn-start:disabled { opacity: 0.3; cursor: not-allowed; }
.btn-start.batch { background: var(--bg-button); color: var(--accent); border: 1px solid var(--accent-dim); }
.btn-stop { padding: 10px; background: #f87171; border: none; border-radius: 8px; color: #000; font-weight: 800; font-size: 11px; cursor: pointer; }

/* ADetailer Layout */
.ad-layout { display: flex; gap: 0; padding: 0 !important; }
.ad-settings { width: 320px; flex-shrink: 0; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; border-right: 1px solid var(--border); }
.ad-settings h3 { color: var(--text-primary); font-size: 13px; font-weight: 900; letter-spacing: 2px; margin: 0; }
.ad-params { display: flex; flex-direction: column; gap: 8px; }
.ad-param { display: flex; flex-direction: column; gap: 2px; }
.ad-param label { font-size: 10px; color: var(--text-muted); font-weight: 700; }
.ad-param input { padding: 6px 8px; font-size: 12px; }
.ad-toggle { display: flex; align-items: center; gap: 4px; font-size: 10px; color: var(--text-muted); cursor: pointer; margin-bottom: 4px; }
.ad-toggle input { width: 14px; height: 14px; accent-color: var(--accent); }
.exif-prompt-hint { font-size: 10px; color: var(--accent); background: var(--accent-dim); padding: 6px 8px; border-radius: 4px; }
.ad-actions { display: flex; flex-direction: column; gap: 6px; margin-top: auto; }

.ad-progress { display: flex; align-items: center; gap: 8px; }
.progress-bar { flex: 1; height: 6px; background: var(--bg-button); border-radius: 3px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.ad-progress span { font-size: 10px; color: var(--text-muted); font-family: monospace; }

/* Compare */
.ad-compare { flex: 1; display: flex; align-items: center; justify-content: center; padding: 20px; overflow: hidden; }
.compare-split { display: flex; gap: 12px; width: 100%; height: 100%; }
.compare-col { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; min-width: 0; }
.compare-label { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 2px; }
.compare-col img { max-width: 100%; max-height: calc(100% - 24px); object-fit: contain; border-radius: 6px; }
.preview-single { display: flex; align-items: center; justify-content: center; }
.preview-single img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 6px; }
.compare-empty { color: var(--text-muted); font-size: 13px; }
</style>
