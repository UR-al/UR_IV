<template>
  <div class="settings-workspace">
    <!-- Left: Sub-navigation -->
    <aside class="settings-nav">
      <div class="nav-header">PREFERENCES</div>
      <button v-for="tab in subTabs" :key="tab.id"
        class="nav-item" :class="{ active: currentTab === tab.id }"
        @click="currentTab = tab.id"
      >
        <span class="icon">{{ tab.icon }}</span>
        <span class="label">{{ tab.label }}</span>
      </button>
    </aside>

    <!-- Right: Content Area -->
    <main class="settings-body">
      <div class="settings-content">
        <!-- 1. General & Backend -->
        <div v-show="currentTab === 'general'" class="section-fade">
          <div class="glass-card">
            <label>SYSTEM STATUS</label>
            <div class="info-row">
              <span class="desc">CORE VERSION</span>
              <span class="val-badge">v2.0.0 PRO</span>
            </div>
            <div class="info-row mt-12">
              <span class="desc">API ORCHESTRATOR</span>
              <button class="btn-pill" @click="act('show_api_manager')">MANAGE BACKENDS</button>
            </div>
          </div>
        </div>

        <!-- 2. Network & API -->
        <div v-show="currentTab === 'api'" class="section-fade">
          <div class="glass-card">
            <label>API CONFIGURATION</label>
            <div class="input-stack">
              <div class="input-unit">
                <span class="unit-label">STABLE DIFFUSION WEBUI</span>
                <input v-model="webuiUrl" placeholder="http://127.0.0.1:7860" />
              </div>
              <div class="input-unit mt-12">
                <span class="unit-label">COMFYUI (NODE-BASED)</span>
                <input v-model="comfyUrl" placeholder="http://127.0.0.1:8188" />
              </div>
            </div>
            <button class="btn-pill primary mt-16" @click="act('show_api_manager')">TEST CONNECTIVITY</button>
          </div>
        </div>

        <!-- 3. Prompt Logic -->
        <div v-show="currentTab === 'prompt'" class="section-fade">
          <div class="glass-card">
            <label>PROMPT AUTOMATION</label>
            <div class="toggle-grid">
              <label class="toggle-row">
                <span>Auto-clean Duplicates</span>
                <input type="checkbox" v-model="cleanDuplicates" />
              </label>
              <label class="toggle-row">
                <span>Sanitize Whitespaces</span>
                <input type="checkbox" v-model="cleanSpaces" />
              </label>
              <label class="toggle-row">
                <span>Convert Underscores</span>
                <input type="checkbox" v-model="cleanUnderscore" />
              </label>
              <label class="toggle-row">
                <span>Tag Block Mode</span>
                <input type="checkbox" v-model="defaultBlockMode" @change="setBlockMode" />
              </label>
              <label class="toggle-row">
                <span>Gallery Metadata Panel</span>
                <input type="checkbox" v-model="galleryMetadata" @change="window.localStorage.setItem('galleryShowMetadata', String(galleryMetadata))" />
              </label>
            </div>
          </div>
          <div class="glass-card mt-16">
            <label>DATA PERSISTENCE</label>
            <div class="btn-row-2">
              <button class="btn-pill" @click="act('save_settings')">SAVE GLOBAL</button>
              <button class="btn-pill" @click="act('show_prompt_history')">OPEN HISTORY</button>
            </div>
          </div>
        </div>

        <!-- 4. Tab Layout (Drag & Drop) -->
        <div v-show="currentTab === 'tabs'" class="section-fade">
          <div class="glass-card">
            <label>CUSTOM WORKSPACE ORDER</label>
            <div class="drag-list">
              <div v-for="(tab, i) in tabOrder" :key="tab"
                class="drag-item"
                draggable="true"
                @dragstart="dragStart(i)" @dragover.prevent @drop="dragDrop(i)"
              >
                <span class="handle">⠿</span>
                <span class="name">{{ tab }}</span>
              </div>
            </div>
            <div class="btn-row-2 mt-16">
              <button class="btn-pill primary" @click="applyTabOrder">APPLY LAYOUT</button>
              <button class="btn-pill" @click="resetTabOrder">RESET DEFAULT</button>
            </div>
          </div>
        </div>

        <!-- 5. Shortcuts -->
        <div v-show="currentTab === 'shortcuts'" class="section-fade">
          <div class="glass-card">
            <label>HOTKEYS</label>
            <div class="shortcut-grid">
              <div class="s-row"><span>Undo Action</span><kbd>Ctrl + Z</kbd></div>
              <div class="s-row"><span>Redo Action</span><kbd>Ctrl + Y</kbd></div>
              <div class="s-row"><span>Force Save</span><kbd>Ctrl + S</kbd></div>
              <div class="s-row"><span>Generate</span><kbd>Ctrl + G</kbd></div>
              <div class="s-row"><span>Close Modal</span><kbd>Esc</kbd></div>
              <div class="s-row"><span>Refresh</span><kbd>F5</kbd></div>
            </div>
          </div>
        </div>

        <!-- 6. 기본값 설정 (탭별) -->
        <div v-show="currentTab === 'defaults'" class="section-fade">
          <div class="glass-card">
            <label>T2I 기본값 <span class="sync-badge" v-if="t2iSynced">SYNCED</span></label>
            <div class="defaults-grid">
              <div class="def-field"><span>Steps</span><input type="number" v-model.number="defaults.steps" /></div>
              <div class="def-field"><span>CFG Scale</span><input type="number" v-model.number="defaults.cfg" step="0.5" /></div>
              <div class="def-field"><span>Width</span><input type="number" v-model.number="defaults.width" /></div>
              <div class="def-field"><span>Height</span><input type="number" v-model.number="defaults.height" /></div>
              <div class="def-field"><span>Seed</span><input type="text" v-model="defaults.seed" /></div>
              <div class="def-field"><span>Denoising (I2I)</span><input type="number" v-model.number="defaults.denoising" step="0.05" /></div>
              <div class="def-field"><span>Sampler</span>
                <CustomSelect v-model="defaults.sampler" :options="['', ...samplerList]" placeholder="Auto" />
              </div>
              <div class="def-field"><span>Scheduler</span>
                <CustomSelect v-model="defaults.scheduler" :options="['', ...schedulerList]" placeholder="Auto" />
              </div>
            </div>
            <button class="btn-pill mt-12" @click="syncFromT2I">SYNC FROM T2I</button>
          </div>

          <div class="glass-card mt-16">
            <label>EDITOR 기본값</label>
            <div class="defaults-grid">
              <div class="def-field"><span>Brush Size</span><input type="number" v-model.number="defaults.brushSize" /></div>
              <div class="def-field"><span>Effect Strength</span><input type="number" v-model.number="defaults.effectStrength" /></div>
              <div class="def-field"><span>YOLO Confidence</span><input type="number" v-model.number="defaults.yoloConf" step="0.05" /></div>
              <div class="def-field"><span>Snap Radius</span><input type="number" v-model.number="defaults.snapRadius" /></div>
            </div>
          </div>

          <div class="glass-card mt-16">
            <label>SEARCH 기본값</label>
            <div class="defaults-grid">
              <div class="def-field"><span>Default Rating</span>
                <CustomSelect v-model="defaults.defaultRating" :options="['g', 's', 'q', 'e']" placeholder="Rating" />
              </div>
            </div>
          </div>

          <div class="glass-card mt-16">
            <label>확장 기본값</label>
            <div class="toggle-grid">
              <label class="toggle-row"><input type="checkbox" v-model="defaults.hires_enabled" /><span>Hires.fix 기본 활성화</span></label>
              <label class="toggle-row"><input type="checkbox" v-model="defaults.ad_enabled" /><span>ADetailer 기본 활성화</span></label>
              <label class="toggle-row"><input type="checkbox" v-model="defaults.negpip_enabled" /><span>NegPiP 기본 활성화</span></label>
            </div>
          </div>
          <div class="btn-row-2 mt-16">
            <button class="btn-pill primary" @click="saveDefaults">SAVE DEFAULTS</button>
            <button class="btn-pill" @click="resetDefaults">RESET TO FACTORY</button>
          </div>
        </div>

        <!-- 7. AI Assist (Ollama) -->
        <div v-show="currentTab === 'ollama'" class="section-fade">
          <div class="glass-card">
            <label>OLLAMA CONFIGURATION</label>
            <div class="input-stack">
              <div class="input-unit">
                <span class="unit-label">SERVER URL</span>
                <input v-model="ollamaUrl" @change="saveOllamaSettings" placeholder="http://localhost:11434" />
              </div>
              <div class="input-unit mt-12">
                <span class="unit-label">MODEL</span>
                <CustomSelect v-if="ollamaModels.length" v-model="ollamaModel" :options="ollamaModels" placeholder="모델 선택..." @update:modelValue="saveOllamaSettings" />
                <input v-else v-model="ollamaModel" @change="saveOllamaSettings" placeholder="llama3.1, gemma3 등" />
              </div>
            </div>
            <div class="btn-row-2 mt-16">
              <button class="btn-pill primary" @click="testOllama">TEST CONNECTION</button>
              <button class="btn-pill" @click="loadOllamaModels">REFRESH MODELS</button>
            </div>
            <div class="info-row mt-12" v-if="ollamaModels.length">
              <span class="desc">AVAILABLE MODELS ({{ ollamaModels.length }})</span>
              <span class="val-badge">{{ ollamaModel }}</span>
            </div>
          </div>
          <div class="glass-card mt-16">
            <label>RECOMMENDED MODELS</label>
            <div class="recommend-grid">
              <div class="rec-item best">
                <span class="rec-name">gemma3:4b</span>
                <span class="rec-desc">가장 추천 — 빠르고 태그 품질 우수, VRAM 3GB</span>
              </div>
              <div class="rec-item">
                <span class="rec-name">llama3.1:8b</span>
                <span class="rec-desc">범용 고품질, 영어 태그 강점, VRAM 5GB</span>
              </div>
              <div class="rec-item">
                <span class="rec-name">mistral:7b</span>
                <span class="rec-desc">빠른 응답, 창의적 태그 변형에 강함, VRAM 5GB</span>
              </div>
              <div class="rec-item">
                <span class="rec-name">phi4-mini:3.8b</span>
                <span class="rec-desc">초경량, VRAM 부족 시 대안, VRAM 2.5GB</span>
              </div>
              <div class="rec-item">
                <span class="rec-name">qwen3:8b</span>
                <span class="rec-desc">다국어+태그 강점, thinking 모드, VRAM 5GB</span>
              </div>
              <div class="rec-item">
                <span class="rec-name">gemma3:12b</span>
                <span class="rec-desc">최고 품질, 여유 VRAM 시 추천, VRAM 8GB</span>
              </div>
            </div>
            <div class="rec-note mt-12">
              SD 이미지 생성과 동시 사용 시 VRAM을 공유하므로 4b 이하 경량 모델 권장.<br/>
              <code>ollama pull gemma3:4b</code> 로 설치
            </div>
          </div>
          <div class="glass-card mt-16">
            <label>USAGE</label>
            <div class="shortcut-grid">
              <div class="s-row"><span>✨ Expand Tags</span><span>기존 태그를 고품질 태그로 확장</span></div>
              <div class="s-row"><span>💬 Natural Language</span><span>자연어 설명을 태그로 변환</span></div>
              <div class="s-row"><span>🔄 Suggest Similar</span><span>유사하지만 다른 태그 추천</span></div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { requestAction, useWidgetStore } from '../stores/widgetStore.js'
import CustomSelect from '../components/CustomSelect.vue'

const subTabs = [
  { id: 'general', label: 'GENERAL', icon: '⚙️' },
  { id: 'api', label: 'NETWORK', icon: '🌐' },
  { id: 'prompt', label: 'LOGIC', icon: '📝' },
  { id: 'tabs', label: 'WORKSPACE', icon: '🗂️' },
  { id: 'shortcuts', label: 'HOTKEYS', icon: '⌨️' },
  { id: 'defaults', label: 'DEFAULTS', icon: '🎛️' },
  { id: 'ollama', label: 'AI ASSIST', icon: '🧠' },
]
const currentTab = ref('general')
const webuiUrl = ref('http://127.0.0.1:7860')
const comfyUrl = ref('http://127.0.0.1:8188')
const cleanDuplicates = ref(true)
const cleanSpaces = ref(true)
const cleanUnderscore = ref(true)
const defaultBlockMode = ref(window.localStorage.getItem('tagBlockMode') === 'true')
const galleryMetadata = ref(window.localStorage.getItem('galleryShowMetadata') !== 'false')

// API에서 sampler/scheduler 목록 가져오기
const wStore = useWidgetStore()
const samplerList = computed(() => wStore.getProperty('sampler_combo', 'items') || [])
const schedulerList = computed(() => wStore.getProperty('scheduler_combo', 'items') || [])

// UI prefs 로드 시 동기화
import { onBackendEvent, getBackend } from '../bridge.js'
onMounted(async () => {
  // defaults 로드
  const bk = await getBackend()
  if (bk.getTabDefaults) {
    bk.getTabDefaults((json) => {
      try { const d = JSON.parse(json); Object.assign(defaults, d) } catch {}
    })
  }
  onBackendEvent('uiPrefsLoaded', (json) => {
    try {
      const prefs = JSON.parse(json)
      if (typeof prefs.tagBlockMode === 'boolean') { defaultBlockMode.value = prefs.tagBlockMode; window.localStorage.setItem('tagBlockMode', String(prefs.tagBlockMode)) }
      if (typeof prefs.cleanDuplicates === 'boolean') cleanDuplicates.value = prefs.cleanDuplicates
      if (typeof prefs.cleanSpaces === 'boolean') cleanSpaces.value = prefs.cleanSpaces
      if (typeof prefs.cleanUnderscore === 'boolean') cleanUnderscore.value = prefs.cleanUnderscore
      if (typeof prefs.galleryShowMetadata === 'boolean') { galleryMetadata.value = prefs.galleryShowMetadata; window.localStorage.setItem('galleryShowMetadata', String(prefs.galleryShowMetadata)) }
      if (Array.isArray(prefs.tabOrder) && prefs.tabOrder.length > 0) {
        tabOrder.value = [...prefs.tabOrder]
        window.localStorage.setItem('tabOrder', JSON.stringify(prefs.tabOrder))
      }
      // Ollama 설정 복원
      if (prefs.ollamaUrl) { ollamaUrl.value = prefs.ollamaUrl; window.localStorage.setItem('ollamaUrl', prefs.ollamaUrl) }
      if (prefs.ollamaModel) { ollamaModel.value = prefs.ollamaModel; window.localStorage.setItem('ollamaModel', prefs.ollamaModel) }
    } catch {}
  })
})
function setBlockMode() {
  window.localStorage.setItem('tagBlockMode', String(defaultBlockMode.value))
  console.log('[Settings] Block mode set to:', defaultBlockMode.value)
}

const defaultOrder = ['T2I','I2I','Inpaint','Event Gen','Search','Batch / Upscale','Gallery','XYZ Plot','PNG Info','Favorites','Settings']
function _loadTabOrder() {
  try {
    const saved = JSON.parse(window.localStorage.getItem('tabOrder') || '[]')
    if (saved.length > 0) return saved
  } catch {}
  return [...defaultOrder]
}
const tabOrder = ref(_loadTabOrder())
let dragIdx = -1

function dragStart(i) { dragIdx = i }
function persistTabOrder() {
  window.localStorage.setItem('tabOrder', JSON.stringify(tabOrder.value))
  requestAction('save_ui_prefs', { tabOrder: tabOrder.value })
  requestAction('set_tab_order', { order: tabOrder.value })
}
function dragDrop(i) {
  if (dragIdx < 0) return
  const item = tabOrder.value.splice(dragIdx, 1)[0]
  tabOrder.value.splice(i, 0, item)
  dragIdx = -1
  persistTabOrder()
}
const applyTabOrder = () => {
  persistTabOrder()
  requestAction('show_toast', { type: 'success', msg: '탭 순서가 적용되었습니다' })
}
const resetTabOrder = () => {
  tabOrder.value = [...defaultOrder]
  persistTabOrder()
}
const act = (name) => {
  // SAVE GLOBAL 시 localStorage 설정도 함께 저장
  if (name === 'save_settings') {
    const stWidgets = useWidgetStore().widgets
    requestAction('save_ui_prefs', {
      tagBlockMode: defaultBlockMode.value,
      cleanDuplicates: cleanDuplicates.value,
      cleanSpaces: cleanSpaces.value,
      cleanUnderscore: cleanUnderscore.value,
      // Hires/ADetailer/NegPiP 상태
      hires_enabled: stWidgets.hires_options_group === 'true',
      ad_enabled: stWidgets.adetailer_group === 'true',
      ad_s1_enabled: stWidgets.ad_slot1_group === 'true',
      ad_s2_enabled: stWidgets.ad_slot2_group === 'true',
      negpip_enabled: stWidgets.negpip_group === 'true',
      galleryShowMetadata: galleryMetadata.value,
      tabOrder: tabOrder.value,
      // Ollama
      ollamaUrl: ollamaUrl.value,
      ollamaModel: ollamaModel.value,
    })
  }
  requestAction(name)
}

// 기본값 설정
const FACTORY_DEFAULTS = { steps: 20, cfg: 7, width: 1024, height: 1024, seed: '-1', denoising: 0.75, sampler: '', scheduler: '', brushSize: 20, effectStrength: 15, yoloConf: 0.25, snapRadius: 12, defaultRating: 'g', hires_enabled: false, ad_enabled: false, negpip_enabled: false }
const defaults = reactive({ ...FACTORY_DEFAULTS })

function saveDefaults() {
  requestAction('save_tab_defaults', { ...defaults })
}

// defaults 변경 감시 → 자동 저장 알림
let defaultsTimer = null
watch(defaults, () => {
  clearTimeout(defaultsTimer)
  defaultsTimer = setTimeout(() => {
    requestAction('save_tab_defaults', { ...defaults })
  }, 1500)
}, { deep: true })
function resetDefaults() { Object.assign(defaults, FACTORY_DEFAULTS) }

const t2iSynced = ref(false)
async function syncFromT2I() {
  const { useWidgetStore } = await import('../stores/widgetStore.js')
  const store = useWidgetStore()
  const w = store.widgets
  defaults.steps = parseInt(w.steps_input) || defaults.steps
  defaults.cfg = parseFloat(w.cfg_input) || defaults.cfg
  defaults.width = parseInt(w.width_input) || defaults.width
  defaults.height = parseInt(w.height_input) || defaults.height
  defaults.seed = w.seed_input || defaults.seed
  defaults.sampler = w.sampler_combo || defaults.sampler
  defaults.scheduler = w.scheduler_combo || defaults.scheduler
  t2iSynced.value = true
  setTimeout(() => { t2iSynced.value = false }, 3000)
}

// Ollama
const ollamaUrl = ref(window.localStorage.getItem('ollamaUrl') || 'http://localhost:11434')
const ollamaModel = ref(window.localStorage.getItem('ollamaModel') || 'gemma3:4b')
const ollamaModels = ref([])

function saveOllamaSettings() {
  window.localStorage.setItem('ollamaUrl', ollamaUrl.value)
  window.localStorage.setItem('ollamaModel', ollamaModel.value)
}

async function testOllama() {
  const { getBackend } = await import('../bridge.js')
  const backend = await getBackend()
  if (backend.ollamaListModels) {
    backend.ollamaListModels(ollamaUrl.value, (json) => {
      try {
        const models = JSON.parse(json)
        ollamaModels.value = models
        if (models.length > 0) {
          requestAction('show_toast', { type: 'success', msg: `Ollama 연결 성공! ${models.length}개 모델 발견` })
          // 현재 선택 모델이 목록에 없으면 첫번째로 설정
          if (!models.includes(ollamaModel.value)) {
            ollamaModel.value = models[0]
            saveOllamaSettings()
          }
        } else {
          requestAction('show_toast', { type: 'info', msg: 'Ollama 연결됨 — 설치된 모델 없음' })
        }
      } catch {
        requestAction('show_toast', { type: 'error', msg: 'Ollama 연결 실패' })
      }
    })
  }
}
function loadOllamaModels() { testOllama() }
</script>

<style scoped>
.settings-workspace { height: 100%; display: flex; background: var(--bg-primary); }

/* Navigation */
.settings-nav {
  width: 240px; background: var(--bg-secondary); border-right: 1px solid var(--border);
  padding: 24px 12px; display: flex; flex-direction: column; gap: 4px;
}
.nav-header { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 2px; padding: 0 12px 12px; }
.nav-item {
  height: 44px; padding: 0 16px; border: none; background: transparent;
  border-radius: var(--radius-base); display: flex; align-items: center; gap: 12px;
  cursor: pointer; transition: var(--transition);
}
.nav-item .icon { font-size: 16px; opacity: 0.5; transition: var(--transition); }
.nav-item .label { font-size: 11px; font-weight: 800; color: var(--text-muted); letter-spacing: 1px; }
.nav-item:hover { background: var(--bg-input); }
.nav-item.active { background: var(--accent-dim); }
.nav-item.active .icon { opacity: 1; }
.nav-item.active .label { color: var(--accent); }

/* Content Area */
.settings-body { flex: 1; overflow-y: auto; padding: 40px; }
.settings-content { max-width: 700px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }

.section-fade { animation: fadeIn 0.3s ease-out; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.glass-card { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 24px; }

.info-row { display: flex; justify-content: space-between; align-items: center; }
.desc { font-size: 12px; font-weight: 700; color: var(--text-secondary); }
.val-badge { background: var(--border); padding: 4px 12px; border-radius: var(--radius-pill); font-size: 10px; font-weight: 900; color: var(--accent); }

.input-stack { display: flex; flex-direction: column; gap: 16px; }
.input-unit { position: relative; }
.unit-label { position: absolute; left: 12px; top: -8px; background: var(--bg-primary); padding: 0 6px; font-size: 9px; font-weight: 900; color: var(--text-muted); letter-spacing: 1px; }

.toggle-grid { display: flex; flex-direction: column; gap: 8px; }
.toggle-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 16px; background: var(--bg-input); border-radius: var(--radius-base);
  cursor: pointer; transition: var(--transition);
}
.toggle-row:hover { background: var(--bg-button); }
.toggle-row span { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
.toggle-row input { width: 40px; height: 20px; accent-color: var(--accent); }

.btn-row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.mt-16 { margin-top: 16px; }

.drag-list { display: flex; flex-direction: column; gap: 6px; }
.drag-item {
  display: flex; align-items: center; gap: 12px; padding: 12px 16px;
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 8px;
  cursor: grab; transition: var(--transition);
}
.drag-item:active { cursor: grabbing; background: var(--bg-button); scale: 0.98; }
.drag-item .handle { color: var(--text-muted); }
.drag-item .name { font-size: 12px; font-weight: 800; letter-spacing: 1px; color: var(--text-primary); }

.shortcut-grid { display: flex; flex-direction: column; gap: 12px; }
.s-row { display: flex; justify-content: space-between; align-items: center; }
.s-row span { font-size: 13px; color: var(--text-secondary); }
kbd { background: var(--bg-button); color: var(--accent); padding: 4px 10px; border-radius: 6px; font-family: 'Consolas', monospace; font-size: 11px; border: 1px solid var(--border); }

/* Defaults */
.defaults-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.def-field { display: flex; flex-direction: column; gap: 3px; }
.def-field span { font-size: 10px; font-weight: 700; color: var(--text-muted); }
.sync-badge { background: #4ade80; color: #000; padding: 1px 6px; border-radius: 4px; font-size: 8px; font-weight: 900; margin-left: 8px; }
.def-field input, .def-field select { padding: 8px 10px; font-size: 12px; }

/* Ollama */
.model-select { width: 100%; padding: 10px 12px; background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-primary); font-size: 13px; font-weight: 600; }
.model-select:focus { border-color: var(--accent); outline: none; }

.recommend-grid { display: flex; flex-direction: column; gap: 8px; }
.rec-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 14px; background: var(--bg-input); border: 1px solid var(--border);
  border-radius: var(--radius-base); transition: var(--transition);
}
.rec-item:hover { border-color: #444; }
.rec-item.best { border-color: var(--accent-dim); background: rgba(250, 204, 21, 0.03); }
.rec-name { font-size: 13px; font-weight: 800; color: var(--text-primary); min-width: 120px; font-family: 'Consolas', monospace; }
.rec-item.best .rec-name { color: var(--accent); }
.rec-desc { font-size: 11px; color: var(--text-muted); text-align: right; }
.rec-note { font-size: 11px; color: var(--text-muted); line-height: 1.6; }
.rec-note code { background: var(--bg-button); padding: 2px 8px; border-radius: 4px; font-size: 11px; color: var(--accent); }
</style>
