<template>
  <div class="pnginfo-view">
    <!-- 상단 서브탭 -->
    <div class="sub-tabs">
      <button class="sub-tab" :class="{ active: subTab === 'info' }" @click="subTab = 'info'">📄 PNG Info</button>
      <button class="sub-tab" :class="{ active: subTab === 'compare' }" @click="subTab = 'compare'">🔍 Compare</button>
    </div>

    <!-- PNG Info 탭 -->
    <div v-if="subTab === 'info'" class="tab-content info-layout">
      <div class="image-area" @dragover.prevent @drop.prevent="onDrop" @dblclick="openFile">
        <img v-if="imagePath" :src="'file:///' + imagePath" class="preview-img" />
        <div v-else class="drop-hint">
          <div class="icon">📄</div>
          <div>이미지를 드래그하거나 더블클릭</div>
        </div>
      </div>
      <div class="info-panel">
        <div class="info-header">
          <h3>PNG Info</h3>
          <button class="btn" @click="openFile">📂 열기</button>
          <button class="btn" @click="copyAll" v-if="exif.raw">📋 복사</button>
          <button class="btn" @click="sendToCompare" v-if="imagePath">🔍 비교로</button>
        </div>
        <div class="info-body" v-if="exif.raw">
          <div v-if="exif.prompt" class="section"><label>Prompt</label><pre>{{ exif.prompt }}</pre></div>
          <div v-if="exif.negative" class="section"><label>Negative</label><pre>{{ exif.negative }}</pre></div>
          <div v-if="exif.params" class="section params-section">
            <label>Parameters</label>
            <div class="params-grid">
              <div class="param-line" v-if="exif.params.generation"><span class="pl">GEN</span><span>{{ exif.params.generation }}</span></div>
              <div class="param-line" v-if="exif.params.core"><span class="pl">CORE</span><span>{{ exif.params.core }}</span></div>
              <div class="param-line" v-if="exif.params.model"><span class="pl">MODEL</span><span>{{ exif.params.model }}</span></div>
              <div class="param-line" v-if="exif.params.hires"><span class="pl">HIRES</span><span>{{ exif.params.hires }}</span></div>
              <div class="param-line" v-if="exif.params.extensions"><span class="pl">EXT</span><span>{{ exif.params.extensions }}</span></div>
              <div class="param-line" v-if="exif.params.other"><span class="pl">ETC</span><span>{{ exif.params.other }}</span></div>
            </div>
          </div>
          <div v-else-if="exif.params_line" class="section"><label>Parameters</label><pre>{{ exif.params_line }}</pre></div>
          <div v-if="!exif.prompt" class="section"><label>Raw</label><pre>{{ exif.raw }}</pre></div>
          <div class="action-bar">
            <button class="btn accent" @click="sendPrompt">📤 T2I 전송</button>
            <button class="btn" @click="sendGenerate">⚡ 즉시 생성</button>
            <button class="btn" @click="action('send_to_i2i', { path: imagePath })">I2I</button>
            <button class="btn" @click="action('send_to_inpaint', { path: imagePath })">Inpaint</button>
            <button class="btn" @click="action('send_to_editor', { path: imagePath })">Editor</button>
            <button class="btn" @click="action('add_favorite', { path: imagePath })">⭐</button>
          </div>
        </div>
        <div v-else class="info-empty">이미지를 선택하면 메타데이터가 표시됩니다</div>
      </div>
    </div>

    <!-- Compare 탭 -->
    <div v-if="subTab === 'compare'" class="tab-content compare-layout">
      <div class="compare-controls">
        <div class="cmp-slot">
          <span class="cmp-label">BEFORE</span>
          <button class="btn" @click="loadCompareImage('before')">📂 열기</button>
          <span class="cmp-name">{{ beforeName || '없음' }}</span>
        </div>
        <div class="cmp-slot">
          <span class="cmp-label">AFTER</span>
          <button class="btn" @click="loadCompareImage('after')">📂 열기</button>
          <span class="cmp-name">{{ afterName || '없음' }}</span>
        </div>
      </div>
      <div class="compare-area">
        <CompareSlider v-if="compareBefore && compareAfter"
          :before-src="'file:///' + compareBefore"
          :after-src="'file:///' + compareAfter"
        />
        <div v-else class="compare-hint">
          <div class="icon">🔍</div>
          <p>두 이미지를 선택하면 비교 슬라이더가 표시됩니다</p>
          <p class="sub">우클릭 메뉴의 "비교로 보내기"로도 이미지를 추가할 수 있습니다</p>
        </div>
      </div>
      <!-- EXIF 비교 -->
      <div class="exif-diff" v-if="compareBefore && compareAfter && (beforeExif.raw || afterExif.raw)">
        <div class="diff-header">
          <h4>PARAMETER DIFF</h4>
          <button class="btn" @click="showDiffOnly = !showDiffOnly">{{ showDiffOnly ? '전체 보기' : '차이만 보기' }}</button>
        </div>
        <div class="diff-table">
          <div class="diff-row header">
            <span class="diff-key">Parameter</span>
            <span class="diff-val">BEFORE</span>
            <span class="diff-val">AFTER</span>
          </div>
          <template v-for="row in paramDiffRows" :key="row.key">
            <div class="diff-row" :class="{ changed: row.changed, same: !row.changed }" v-if="!showDiffOnly || row.changed">
              <span class="diff-key">{{ row.key }}</span>
              <span class="diff-val" :class="{ highlight: row.changed }">{{ row.before }}</span>
              <span class="diff-val" :class="{ highlight: row.changed }">{{ row.after }}</span>
            </div>
          </template>
        </div>
        <!-- 프롬프트 비교 -->
        <div class="diff-prompts" v-if="beforeExif.prompt || afterExif.prompt">
          <div class="diff-prompt-section">
            <label>PROMPT DIFF</label>
            <div class="diff-tags">
              <span v-for="t in promptDiff.same" :key="'s'+t" class="dtag same">{{ t }}</span>
              <span v-for="t in promptDiff.removed" :key="'r'+t" class="dtag removed">- {{ t }}</span>
              <span v-for="t in promptDiff.added" :key="'a'+t" class="dtag added">+ {{ t }}</span>
            </div>
          </div>
        </div>
      </div>
      <!-- GIF 내보내기 -->
      <div class="gif-bar" v-if="compareBefore && compareAfter">
        <span class="gif-label">GIF Export</span>
        <label>Speed</label>
        <CustomSelect v-model="gifDurationStr" :options="['Fast', 'Normal', 'Slow']" placeholder="Speed" />
        <button class="gif-btn" @click="exportGif" :disabled="gifExporting">
          {{ gifExporting ? 'Exporting...' : '🎬 Export GIF' }}
        </button>
        <a v-if="gifResult" :href="'file:///' + gifResult" class="gif-link" target="_blank">📥 {{ gifResult.split('/').pop() }}</a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import CompareSlider from '../components/CompareSlider.vue'
import CustomSelect from '../components/CustomSelect.vue'

const subTab = ref('info')
const imagePath = ref('')
const exif = ref({})

// Compare
const compareBefore = ref('')
const compareAfter = ref('')
const beforeName = ref('')
const afterName = ref('')
const beforeExif = ref({})
const afterExif = ref({})
const showDiffOnly = ref(true)

// 비교 이미지 로드 시 EXIF도 함께 로드
async function loadCompareExif(path, target) {
  const backend = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json) => {
      try {
        const d = JSON.parse(json)
        if (target === 'before') beforeExif.value = d
        else afterExif.value = d
      } catch {}
    })
  }
}

// 파라미터 라인을 key:value 파싱
function _parseParamsLine(raw) {
  if (!raw) return {}
  const params = {}
  // "Steps: 28, Sampler: Euler, ..." 형식
  const matches = raw.matchAll(/([A-Za-z][A-Za-z0-9_ ]*?):\s*([^,]+?)(?:,\s*|$)/g)
  for (const m of matches) params[m[1].trim()] = m[2].trim()
  return params
}

// 파라미터 diff 계산
const paramDiffRows = computed(() => {
  const bParams = _parseParamsLine(beforeExif.value.params_line || '')
  const aParams = _parseParamsLine(afterExif.value.params_line || '')
  const allKeys = new Set([...Object.keys(bParams), ...Object.keys(aParams)])
  const rows = []
  for (const key of allKeys) {
    const bv = bParams[key] || ''
    const av = aParams[key] || ''
    rows.push({ key, before: bv, after: av, changed: bv !== av })
  }
  // 변경된 것을 상단에
  rows.sort((a, b) => (b.changed ? 1 : 0) - (a.changed ? 1 : 0))
  return rows
})

// 프롬프트 diff (태그 단위)
const promptDiff = computed(() => {
  const bTags = new Set((beforeExif.value.prompt || '').split(',').map(t => t.trim()).filter(Boolean))
  const aTags = new Set((afterExif.value.prompt || '').split(',').map(t => t.trim()).filter(Boolean))
  const same = [], added = [], removed = []
  for (const t of bTags) { if (aTags.has(t)) same.push(t); else removed.push(t) }
  for (const t of aTags) { if (!bTags.has(t)) added.push(t) }
  return { same, added, removed }
})

async function loadImage(path) {
  imagePath.value = path
  const backend = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json) => { try { exif.value = JSON.parse(json) } catch {} })
  }
}

function onDrop(e) {
  const file = e.dataTransfer?.files?.[0]
  if (file?.path) loadImage(file.path.replace(/\\/g, '/'))
  else {
    const path = e.dataTransfer?.getData('text/plain')
    if (path && path.includes('/')) loadImage(path)
  }
}

function openFile() { requestAction('open_png_info_file') }
function copyAll() { navigator.clipboard?.writeText(exif.value.raw || '') }
function sendPrompt() { requestAction('pnginfo_send_prompt', { prompt: exif.value.prompt || '', negative: exif.value.negative || '' }) }
function sendGenerate() { requestAction('pnginfo_generate', exif.value) }
function action(name, payload = {}) { requestAction(name, payload) }

function sendToCompare() {
  if (!compareBefore.value) {
    compareBefore.value = imagePath.value
    beforeName.value = imagePath.value.split('/').pop()
    loadCompareExif(imagePath.value, 'before')
  } else {
    compareAfter.value = imagePath.value
    afterName.value = imagePath.value.split('/').pop()
    loadCompareExif(imagePath.value, 'after')
  }
  subTab.value = 'compare'
}

async function loadCompareImage(slot) {
  requestAction('open_compare_image', { slot })
}

// GIF 내보내기
const gifDuration = ref(400)
const gifSpeedMap = { 'Fast': 200, 'Normal': 400, 'Slow': 700 }
const gifDurationStr = computed({
  get: () => Object.entries(gifSpeedMap).find(([, v]) => v === gifDuration.value)?.[0] || 'Normal',
  set: v => { gifDuration.value = gifSpeedMap[v] || 400 }
})
const gifExporting = ref(false)
const gifResult = ref('')

async function exportGif() {
  if (!compareBefore.value || !compareAfter.value) return
  gifExporting.value = true; gifResult.value = ''
  const backend = await getBackend()
  if (backend.exportCompareGif) {
    backend.exportCompareGif(compareBefore.value, compareAfter.value, gifDuration.value, 0, (json) => {
      try {
        const d = JSON.parse(json)
        if (d.path) { gifResult.value = d.path; requestAction('show_toast', { type: 'success', msg: `GIF 생성 완료 (${d.frames} frames)` }) }
        else if (d.error) requestAction('show_toast', { type: 'error', msg: d.error })
      } catch {}
      gifExporting.value = false
    })
  }
}

onMounted(() => {
  onBackendEvent('inpaintImageLoaded', (path) => loadImage(path))
  // 비교 이미지 수신 (우클릭 메뉴 "비교로 보내기" 또는 파일 다이얼로그)
  onBackendEvent('compareImageLoaded', (json) => {
    try {
      const d = JSON.parse(json)
      if (d.slot === 'before') {
        compareBefore.value = d.path; beforeName.value = d.path.split('/').pop()
        loadCompareExif(d.path, 'before')
      } else {
        compareAfter.value = d.path; afterName.value = d.path.split('/').pop()
        loadCompareExif(d.path, 'after')
      }
      subTab.value = 'compare'
    } catch {}
  })
})
</script>

<style scoped>
.pnginfo-view { width: 100%; height: 100%; display: flex; flex-direction: column; }
.sub-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.sub-tab {
  flex: 1; padding: 8px; background: transparent; border: none; border-bottom: 2px solid transparent;
  color: var(--text-muted); font-size: 11px; font-weight: 700; cursor: pointer; text-align: center;
}
.sub-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { flex: 1; overflow: hidden; }

/* Info Layout */
.info-layout { display: flex; }
.image-area { flex: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; min-width: 300px; padding: 16px; }
.preview-img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }
.drop-hint { text-align: center; color: #484848; user-select: none; }
.drop-hint .icon { font-size: 48px; opacity: 0.3; margin-bottom: 12px; }
.info-panel { width: 400px; flex-shrink: 0; display: flex; flex-direction: column; overflow: hidden; }
.info-header { display: flex; align-items: center; gap: 6px; padding: 8px 12px; flex-shrink: 0; }
.info-header h3 { color: #E8E8E8; font-size: 14px; margin: 0; flex: 1; }
.btn { padding: 5px 10px; background: #181818; border: none; border-radius: 4px; color: #787878; font-size: 10px; cursor: pointer; }
.btn:hover { background: #222; color: #E8E8E8; }
.btn.accent { background: #E2B340; color: #000; }
.info-body { flex: 1; overflow-y: auto; padding: 8px 12px; }
.section { margin-bottom: 10px; }
.section label { color: #E2B340; font-size: 11px; font-weight: 600; display: block; margin-bottom: 2px; }
.section pre { color: #B0B0B0; font-size: 11px; white-space: pre-wrap; word-break: break-all; background: #111; padding: 6px 8px; border-radius: 4px; margin: 0; max-height: 150px; overflow-y: auto; }
.action-bar { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 12px; }
.info-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: #484848; font-size: 14px; }

.params-grid { background: #111; border-radius: 4px; padding: 6px 8px; }
.param-line { display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: 11px; color: #B0B0B0; border-bottom: 1px solid #1a1a1a; }
.param-line:last-child { border-bottom: none; }
.pl { font-size: 9px; font-weight: 900; color: #E2B340; letter-spacing: 1px; min-width: 45px; flex-shrink: 0; }

/* Compare Layout */
.compare-layout { display: flex; flex-direction: column; }
.compare-controls { display: flex; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.cmp-slot { display: flex; align-items: center; gap: 6px; flex: 1; }
.cmp-label { font-size: 10px; font-weight: 900; color: var(--accent); letter-spacing: 1px; }
.cmp-name { font-size: 10px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.compare-area { flex: 1; position: relative; }
.compare-hint { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; }
.compare-hint .icon { font-size: 48px; opacity: 0.3; margin-bottom: 12px; }
.compare-hint p { color: #484848; font-size: 13px; }
.compare-hint .sub { font-size: 11px; color: #383838; margin-top: 4px; }

/* GIF Export */
.gif-bar { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-top: 1px solid var(--border); flex-shrink: 0; }
.gif-label { font-size: 10px; font-weight: 800; color: var(--text-muted); letter-spacing: 1px; }
.gif-bar label { font-size: 10px; color: var(--text-muted); }
.gif-bar select { padding: 3px 8px; font-size: 10px; }
.gif-btn { padding: 5px 14px; background: var(--accent); border: none; border-radius: 6px; color: #000; font-size: 10px; font-weight: 800; cursor: pointer; }
.gif-btn:disabled { opacity: 0.4; }
.gif-link { font-size: 10px; color: #60a5fa; text-decoration: none; margin-left: auto; }

/* EXIF Diff */
.exif-diff { padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; max-height: 300px; overflow-y: auto; }
.diff-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.diff-header h4 { font-size: 11px; font-weight: 900; color: var(--text-muted); letter-spacing: 1.5px; }
.diff-table { display: flex; flex-direction: column; gap: 1px; font-family: 'Consolas', monospace; }
.diff-row { display: grid; grid-template-columns: 140px 1fr 1fr; gap: 8px; padding: 4px 8px; border-radius: 4px; font-size: 10px; }
.diff-row.header { font-weight: 900; color: var(--text-muted); font-size: 9px; letter-spacing: 1px; border-bottom: 1px solid var(--border); }
.diff-row.changed { background: rgba(248, 113, 113, 0.06); }
.diff-row.same { opacity: 0.5; }
.diff-key { color: var(--text-secondary); font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.diff-val { color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.diff-val.highlight { color: #f87171; font-weight: 700; }

.diff-prompts { margin-top: 12px; }
.diff-prompt-section label { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 1.5px; margin-bottom: 6px; display: block; }
.diff-tags { display: flex; flex-wrap: wrap; gap: 3px; }
.dtag { padding: 2px 8px; border-radius: 4px; font-size: 9px; }
.dtag.same { background: rgba(255,255,255,0.03); color: var(--text-muted); }
.dtag.added { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.2); }
.dtag.removed { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.2); text-decoration: line-through; }
</style>
