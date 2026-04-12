<template>
  <div class="eg-view">
    <!-- 검색 전: 중앙 검색 폼 -->
    <div v-if="events.length === 0 && !searching" class="welcome">
      <div class="ws-header">
        <div class="ws-icon">🎬</div>
        <h1>EVENT GENERATOR</h1>
        <p>Danbooru 데이터베이스에서 이벤트 시퀀스를 검색합니다</p>
      </div>

      <div class="search-form">
        <div class="form-section">
          <div class="form-row">
            <div class="form-field wide">
              <label>Character</label>
              <input v-model="character" placeholder="e.g. hatsune_miku, raiden_shogun" @keydown.enter="searchEvents" />
            </div>
            <div class="form-field">
              <label>Copyright</label>
              <input v-model="copyright" placeholder="e.g. genshin_impact" @keydown.enter="searchEvents" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-field wide">
              <label>General Tags</label>
              <input v-model="prompt" placeholder="e.g. 1girl, blue_hair, sword" @keydown.enter="searchEvents" />
            </div>
            <div class="form-field">
              <label>Artist</label>
              <input v-model="artist" placeholder="Artist name..." @keydown.enter="searchEvents" />
            </div>
          </div>
        </div>

        <!-- Exclude -->
        <details class="form-section exclude-section" open>
          <summary class="exclude-toggle">Exclude Tags ▾</summary>
          <div class="form-row">
            <div class="form-field"><label class="danger">Tags</label><input v-model="excludeTags" placeholder="제외 태그..." @keydown.enter="searchEvents" /></div>
          </div>
        </details>

        <!-- Event Length -->
        <div class="form-section">
          <div class="form-row">
            <div class="form-field">
              <label>Min Steps</label>
              <input type="number" v-model.number="minSteps" min="1" max="50" />
            </div>
            <div class="form-field">
              <label>Max Steps</label>
              <input type="number" v-model.number="maxSteps" min="1" max="100" />
            </div>
          </div>
        </div>

        <!-- Rating + Options + Go -->
        <div class="form-footer">
          <div class="rating-row">
            <button v-for="r in ratings" :key="r.key" class="rating-chip"
              :class="{ active: r.checked }" @click="r.checked = !r.checked">{{ r.label }}</button>
          </div>
          <div class="opt-row-inline">
            <label><input type="checkbox" v-model="limitResults" />상위 100개</label>
            <label><input type="checkbox" v-model="fixSeed" />시드 고정</label>
            <label><input type="checkbox" v-model="useT2ISettings" />T2I 설정</label>
          </div>
          <button class="go-btn" @click="searchEvents" :disabled="searching">🚀 RUN SEARCH</button>
          <div class="io-row">
            <button class="io-btn" @click="importEvents">📥 IMPORT .parquet</button>
          </div>
        </div>
      </div>

      <div class="hints">
        <span>쉼표(,) = AND</span><span>[A|B] = OR</span><span>이벤트 = 연속 태그 변화 시퀀스</span>
      </div>
    </div>

    <!-- 검색 중 -->
    <div v-else-if="events.length === 0 && searching" class="loading">
      <div class="spinner"></div>
      <p>{{ loadingMsg || '이벤트 검색 중...' }}</p>
      <div class="search-progress" v-if="searchTotal > 0">
        <div class="sp-bar">
          <div class="sp-fill" :style="{ width: searchPct + '%' }"></div>
        </div>
        <span class="sp-text">{{ searchCur.toLocaleString() }} / {{ searchTotal.toLocaleString() }} ({{ searchPct }}%)</span>
      </div>
    </div>

    <!-- 결과 -->
    <template v-else>
      <!-- 상단 바 -->
      <div class="result-bar">
        <button class="bar-btn back" @click="events = []; steps = []; selectedIdx = -1">← BACK</button>
        <span class="bar-count">{{ events.length }} EVENTS</span>
        <div class="bar-spacer"></div>
        <button class="bar-btn" @click="exportEvents">📤 EXPORT</button>
        <button class="bar-btn" @click="importEvents">📥 IMPORT</button>
      </div>

      <div class="result-body">
        <!-- 좌측: 이벤트 목록 -->
        <div class="eg-list">
          <div v-for="(ev, i) in events" :key="i" class="eg-list-item"
            :class="{ active: selectedIdx === i }" @click="selectEvent(i)">
            <span class="eg-idx">#{{ i + 1 }}</span>
            <span class="eg-desc">{{ ev.character || ev.copyright || 'Event' }}</span>
            <span class="eg-cnt">{{ ev.children_count || '?' }}</span>
          </div>
        </div>

        <!-- 우측: 스텝 상세 -->
        <section class="eg-main">
          <div v-if="steps.length === 0" class="eg-empty">
            <div class="eg-empty-icon">🎬</div>
            <h2>EVENT SEQUENCE</h2>
            <p>이벤트를 선택하면 스텝이 표시됩니다</p>
          </div>
          <template v-else>
            <!-- 이벤트 정보 -->
            <div class="eg-event-info" v-if="currentEvent">
              <span class="ei-char" v-if="currentEvent.character">{{ currentEvent.character }}</span>
              <span class="ei-copy" v-if="currentEvent.copyright">{{ currentEvent.copyright }}</span>
              <span class="ei-sim" v-if="currentEvent.similarity">유사도: {{ (currentEvent.similarity * 100).toFixed(0) }}%</span>
            </div>

            <!-- 캐리 옵션 -->
            <div class="eg-carry">
              <div class="carry-opt">
                <label><input type="checkbox" v-model="carryAppearance" />외모 유지</label>
                <span class="carry-desc">머리색, 눈색 등 외모 태그를 전 스텝에 유지</span>
              </div>
              <div class="carry-opt">
                <label><input type="checkbox" v-model="carryCostume" />의상 유지</label>
                <span class="carry-desc">의상/복장 태그를 Parent 기준으로 유지</span>
              </div>
              <div class="carry-opt">
                <label><input type="checkbox" v-model="carryBackground" />배경 유지</label>
                <span class="carry-desc">배경/장소 태그를 변경하지 않음</span>
              </div>
            </div>

            <!-- 전체/해제 -->
            <div class="step-toolbar">
              <button class="stb" @click="selectAllSteps">전체 선택</button>
              <button class="stb" @click="clearAllSteps">전체 해제</button>
              <span class="stb-count">{{ selectedCount }}개 선택</span>
            </div>

            <div class="eg-steps">
              <div v-for="(step, i) in displaySteps" :key="i" class="eg-step"
                :class="{ 'step-selected': selectedSteps[i] }"
                @click="toggleStep(i)">
                <div class="step-head">
                  <input type="checkbox" :checked="selectedSteps[i]" @click.stop="toggleStep(i)" />
                  <span class="step-no">Step {{ i + 1 }}</span>
                  <span class="step-badge" :class="i === 0 ? 'parent' : 'child'">{{ i === 0 ? 'PARENT' : 'CHILD' }}</span>
                  <span class="step-char" v-if="currentEvent?.character && i === 0">{{ currentEvent.character }}</span>
                  <button class="step-send" @click.stop="sendStepToT2I(step)" title="T2I로 전송">📤</button>
                </div>
                <div class="step-diff" v-if="i > 0 && (step.added?.length || step.removed?.length)">
                  <span v-for="t in (step.added || [])" :key="'a'+t" class="diff-tag add">+ {{ t }}</span>
                  <span v-for="t in (step.removed || [])" :key="'r'+t" class="diff-tag rm">- {{ t }}</span>
                </div>
                <div class="step-prompt">{{ step.displayPrompt || step.prompt || '' }}</div>
              </div>
            </div>

            <div class="eg-actions">
              <div class="eg-row">
                <span class="eg-mini">Repeat</span>
                <input type="number" v-model.number="repeatCount" min="1" max="100" />
              </div>
              <button class="eg-btn" @click="addToQueue" :disabled="selectedCount === 0">
                ADD TO QUEUE ({{ selectedCount }})
              </button>
              <button class="eg-btn primary" @click="generateNow" :disabled="selectedCount === 0">
                GENERATE NOW ({{ selectedCount }})
              </button>
            </div>
          </template>
        </section>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { onBackendEvent } from '../bridge.js'

const ratings = reactive([
  { key: 'g', label: 'GEN', checked: true },
  { key: 's', label: 'SENS', checked: false },
  { key: 'q', label: 'QUES', checked: false },
  { key: 'e', label: 'EXPL', checked: false },
])
const character = ref('')
const copyright = ref('')
const artist = ref('')
const prompt = ref('')
const excludeTags = ref('')
const minSteps = ref(1)
const maxSteps = ref(50)
const limitResults = ref(true)
const fixSeed = ref(false)
const useT2ISettings = ref(true)
const repeatCount = ref(1)
const searching = ref(false)
const loadingMsg = ref('')
const searchCur = ref(0)
const searchTotal = ref(0)
const searchPct = computed(() => searchTotal.value ? Math.round(searchCur.value / searchTotal.value * 100) : 0)
const events = ref([])
const steps = ref([])
const selectedIdx = ref(-1)
const selectedSteps = reactive({})  // { index: true }
const selectedCount = computed(() => Object.values(selectedSteps).filter(Boolean).length)
const carryAppearance = ref(true)
const carryCostume = ref(true)
const carryBackground = ref(true)
const currentEvent = computed(() => selectedIdx.value >= 0 ? events.value[selectedIdx.value] : null)

function _saveEventFields() {
  try {
    window.localStorage.setItem('lastEventFields', JSON.stringify({
      character: character.value, copyright: copyright.value,
      artist: artist.value, prompt: prompt.value,
      excludeTags: excludeTags.value,
      minSteps: minSteps.value, maxSteps: maxSteps.value,
      ratings: ratings.map(r => ({ key: r.key, checked: r.checked })),
    }))
  } catch {}
}

function searchEvents() {
  _saveEventFields()
  searching.value = true
  loadingMsg.value = '데이터 로딩 중...'
  searchCur.value = 0
  searchTotal.value = 0
  requestAction('search_events', {
    character: character.value,
    copyright: copyright.value,
    artist: artist.value,
    ratings: ratings.filter(r => r.checked).map(r => r.key),
    prompt: prompt.value,
    min_steps: minSteps.value,
    max_steps: maxSteps.value,
    exclude_tags: excludeTags.value,
    limit: limitResults.value,
  })
}

// 외모/의상/배경 태그 분류용 키워드
const APPEARANCE_KEYS = ['hair', 'eyes', 'skin', 'ears', 'horns', 'tail', 'wings', 'fang', 'mole', 'scar', 'freckle', 'eyelash', 'pupil', 'iris', 'ahoge', 'bangs', 'sidelocks', 'ponytail', 'twintails', 'braid', 'bun', 'bob', 'short hair', 'long hair', 'medium hair']
const COSTUME_KEYS = ['dress', 'shirt', 'skirt', 'pants', 'uniform', 'armor', 'suit', 'coat', 'jacket', 'hat', 'ribbon', 'bow', 'gloves', 'boots', 'shoes', 'socks', 'stockings', 'thighhighs', 'pantyhose', 'bikini', 'swimsuit', 'cape', 'scarf', 'necktie', 'collar', 'headband', 'hairclip', 'earrings', 'necklace', 'bracelet', 'belt', 'glasses', 'mask', 'hood', 'apron', 'maid', 'school uniform', 'sailor', 'kimono', 'yukata']
const BG_KEYS = ['background', 'outdoors', 'indoors', 'sky', 'cloud', 'tree', 'grass', 'water', 'ocean', 'beach', 'mountain', 'city', 'room', 'bed', 'floor', 'wall', 'window', 'night', 'day', 'sunset', 'sunrise', 'rain', 'snow', 'forest', 'garden', 'street', 'school', 'classroom', 'library', 'kitchen', 'bathroom', 'rooftop', 'bridge', 'castle', 'temple', 'church']

function classifyTag(tag) {
  const t = tag.toLowerCase()
  if (APPEARANCE_KEYS.some(k => t.includes(k))) return 'appearance'
  if (COSTUME_KEYS.some(k => t.includes(k))) return 'costume'
  if (BG_KEYS.some(k => t.includes(k))) return 'background'
  return 'other'
}

// carry 옵션 적용된 스텝 표시
const displaySteps = computed(() => {
  if (steps.value.length === 0) return []
  const parentTags = steps.value[0]?.prompt?.split(',').map(t => t.trim()).filter(Boolean) || []
  const parentByType = {}
  parentTags.forEach(t => {
    const cls = classifyTag(t)
    if (!parentByType[cls]) parentByType[cls] = []
    parentByType[cls].push(t)
  })

  return steps.value.map((step, i) => {
    if (i === 0) return { ...step, displayPrompt: step.prompt }
    let tags = step.prompt?.split(',').map(t => t.trim()).filter(Boolean) || []
    const tagSet = new Set(tags.map(t => t.toLowerCase()))

    // carry: Parent의 해당 카테고리 태그를 추가
    if (carryAppearance.value) {
      (parentByType.appearance || []).forEach(t => { if (!tagSet.has(t.toLowerCase())) tags.push(t) })
    }
    if (carryCostume.value) {
      (parentByType.costume || []).forEach(t => { if (!tagSet.has(t.toLowerCase())) tags.push(t) })
    }
    if (carryBackground.value) {
      (parentByType.background || []).forEach(t => { if (!tagSet.has(t.toLowerCase())) tags.push(t) })
    }
    return { ...step, displayPrompt: tags.join(', ') }
  })
})

function selectEvent(i) {
  selectedIdx.value = i
  const ev = events.value[i]
  // 스텝 초기화
  if (ev.steps) { steps.value = ev.steps }
  else {
    steps.value = [{ prompt: ev.parent_tags || ev.general || '', type: 'parent' }]
    if (ev.children) ev.children.forEach(c => steps.value.push({ ...c, type: 'child' }))
  }
  // 전체 선택 초기화
  Object.keys(selectedSteps).forEach(k => delete selectedSteps[k])
  steps.value.forEach((_, idx) => { selectedSteps[idx] = true })
  requestAction('select_event', { index: i })
}

function toggleStep(i) { selectedSteps[i] = !selectedSteps[i] }
function selectAllSteps() { steps.value.forEach((_, i) => { selectedSteps[i] = true }) }
function clearAllSteps() { Object.keys(selectedSteps).forEach(k => { selectedSteps[k] = false }) }

function _buildScenarios() {
  // 선택된 스텝만 시나리오로 변환
  const scenarios = []
  const dSteps = displaySteps.value
  for (const [idx, checked] of Object.entries(selectedSteps)) {
    if (!checked) continue
    const step = dSteps[parseInt(idx)]
    if (!step) continue
    const prompt = step.displayPrompt || step.prompt || ''
    if (!prompt) continue
    for (let r = 0; r < repeatCount.value; r++) {
      scenarios.push({ payload: { prompt, negative_prompt: '' } })
    }
  }
  return scenarios
}

function addToQueue() {
  const scenarios = _buildScenarios()
  if (!scenarios.length) { requestAction('show_toast', { type: 'info', msg: '스텝을 선택하세요' }); return }
  requestAction('event_add_to_queue', { scenarios })
  requestAction('show_toast', { type: 'success', msg: `${scenarios.length}개 큐에 추가됨` })
}
function generateNow() {
  const scenarios = _buildScenarios()
  if (!scenarios.length) { requestAction('show_toast', { type: 'info', msg: '스텝을 선택하세요' }); return }
  requestAction('event_generate_now', { scenarios })
}

function sendStepToT2I(step) {
  const prompt = step.prompt || ''
  if (!prompt) return
  requestAction('pnginfo_send_prompt', { prompt, negative: '' })
  requestAction('show_toast', { type: 'success', msg: 'T2I로 전송됨' })
}

function exportEvents() {
  requestAction('export_event_results', { events: events.value })
}
function importEvents() {
  requestAction('import_event_results')
}

onMounted(() => {
  // 이전 입력 필드 복원
  try {
    const sf = window.localStorage.getItem('lastEventFields')
    if (sf) {
      const d = JSON.parse(sf)
      if (d.character) character.value = d.character
      if (d.copyright) copyright.value = d.copyright
      if (d.artist) artist.value = d.artist
      if (d.prompt) prompt.value = d.prompt
      if (d.excludeTags) excludeTags.value = d.excludeTags
      if (d.minSteps) minSteps.value = d.minSteps
      if (d.maxSteps) maxSteps.value = d.maxSteps
      if (d.ratings) d.ratings.forEach(r => { const found = ratings.find(rt => rt.key === r.key); if (found) found.checked = r.checked })
    }
  } catch {}
  // 이전 검색 결과 복원
  try {
    const saved = window.localStorage.getItem('lastEventResults')
    if (saved) {
      const data = JSON.parse(saved)
      if (Array.isArray(data) && data.length > 0) events.value = data
    }
  } catch {}
  onBackendEvent('searchStatus', (msg) => {
    loadingMsg.value = msg
  })
  onBackendEvent('eventSearchProgress', (cur, total) => {
    searchCur.value = cur
    searchTotal.value = total
    loadingMsg.value = `유사도 검색 중... (${cur.toLocaleString()} / ${total.toLocaleString()})`
  })
  onBackendEvent('eventSearchResults', (json) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        events.value = data
        if (data.length) {
          requestAction('show_toast', { type: 'success', msg: `${data.length}개 이벤트 발견` })
          try { window.localStorage.setItem('lastEventResults', JSON.stringify(data.slice(0, 200))) } catch {}
        }
        else requestAction('show_toast', { type: 'info', msg: '검색 결과가 없습니다' })
      }
      else if (data.error) { requestAction('show_toast', { type: 'error', msg: data.error }) }
    } catch {}
    searching.value = false
    loadingMsg.value = ''
  })
  // import 결과 수신
  onBackendEvent('eventImportResults', (json) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        events.value = data
        requestAction('show_toast', { type: 'success', msg: `${data.length}개 이벤트 불러옴` })
      }
    } catch {}
  })
})
</script>

<style scoped>
.eg-view { height: 100%; display: flex; flex-direction: column; background: var(--bg-primary); overflow: hidden; }

/* ── Welcome / Search Form ── */
.welcome { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; overflow-y: auto; }
.ws-header { text-align: center; margin-bottom: 32px; }
.ws-icon { font-size: 48px; opacity: 0.6; margin-bottom: 8px; }
.ws-header h1 { font-size: 18px; letter-spacing: 6px; color: var(--text-primary); margin-bottom: 6px; }
.ws-header p { font-size: 12px; color: var(--text-muted); }

.search-form { width: 100%; max-width: 700px; display: flex; flex-direction: column; gap: 12px; }
.form-section { display: flex; flex-direction: column; gap: 8px; }
.form-row { display: flex; gap: 10px; }
.form-field { flex: 1; display: flex; flex-direction: column; gap: 3px; }
.form-field.wide { flex: 2; }
.form-field label { font-size: 10px; font-weight: 800; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 0; }
.form-field label.danger { color: #f87171; }
.form-field input { padding: 10px 12px; font-size: 13px; }

.exclude-section { border: 1px solid rgba(248,113,113,0.1); border-radius: var(--radius-card); padding: 12px; }
.exclude-toggle { font-size: 11px; font-weight: 800; color: #f87171; cursor: pointer; letter-spacing: 0.5px; list-style: none; }
.exclude-toggle::-webkit-details-marker { display: none; }

.form-footer { display: flex; flex-direction: column; gap: 10px; align-items: center; margin-top: 4px; }
.rating-row { display: flex; gap: 6px; }
.rating-chip {
  padding: 6px 16px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: var(--radius-pill); color: var(--text-muted); font-size: 10px; font-weight: 800;
  cursor: pointer; letter-spacing: 1px; transition: var(--transition);
}
.rating-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

.opt-row-inline { display: flex; gap: 16px; }
.opt-row-inline label { display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--text-secondary); cursor: pointer; white-space: nowrap; }

.go-btn {
  width: 100%; max-width: 400px; height: 44px; background: var(--accent); border: none;
  border-radius: var(--radius-pill); color: #000; font-weight: 900; font-size: 12px;
  letter-spacing: 2px; cursor: pointer; transition: var(--transition);
}
.go-btn:hover { background: var(--accent-hover); transform: translateY(-1px); box-shadow: 0 4px 16px rgba(250,204,21,0.2); }
.go-btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
.io-row { display: flex; gap: 8px; justify-content: center; }
.io-btn { padding: 6px 16px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-muted); font-size: 10px; font-weight: 700; cursor: pointer; transition: var(--transition); }
.io-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }

.hints { display: flex; gap: 16px; margin-top: 20px; }
.hints span { font-size: 10px; color: var(--text-muted); }

/* ── Loading ── */
.loading { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }
.spinner { width: 32px; height: 32px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading p { color: var(--text-muted); font-size: 12px; }
.search-progress { width: 300px; display: flex; flex-direction: column; align-items: center; gap: 6px; margin-top: 12px; }
.sp-bar { width: 100%; height: 6px; background: var(--bg-button); border-radius: 3px; overflow: hidden; }
.sp-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
.sp-text { font-size: 11px; color: var(--text-muted); font-family: monospace; }

/* ── Result Bar ── */
.result-bar { display: flex; align-items: center; gap: 12px; padding: 8px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.bar-btn { padding: 5px 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; }
.bar-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }
.bar-count { font-size: 10px; color: var(--text-muted); font-weight: 800; letter-spacing: 1px; }
.bar-spacer { flex: 1; }

/* ── Result Body ── */
.result-body { flex: 1; display: flex; overflow: hidden; }

/* Event List */
.eg-list { width: 240px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid var(--border); }
.eg-list-item { display: flex; align-items: center; gap: 6px; padding: 10px 12px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.02); font-size: 11px; transition: background 0.15s; }
.eg-list-item:hover { background: rgba(255,255,255,0.02); }
.eg-list-item.active { background: var(--accent-dim); border-left: 3px solid var(--accent); }
.eg-idx { color: var(--text-muted); font-family: monospace; font-size: 10px; min-width: 24px; }
.eg-desc { flex: 1; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.eg-cnt { font-size: 9px; color: var(--accent); font-weight: 800; }

/* Main */
.eg-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.eg-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; opacity: 0.2; }
.eg-empty-icon { font-size: 48px; margin-bottom: 12px; }
.eg-empty h2 { letter-spacing: 4px; }
/* Event Info */
.eg-event-info { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; background: rgba(250,204,21,0.03); }
.ei-char { font-size: 12px; font-weight: 800; color: var(--accent); }
.ei-copy { font-size: 11px; color: var(--text-muted); }
.ei-sim { font-size: 10px; color: #4ade80; margin-left: auto; font-family: monospace; }

/* Carry Options */
.eg-carry { display: flex; flex-direction: column; gap: 4px; padding: 10px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.carry-opt { display: flex; align-items: center; gap: 8px; }
.carry-opt label { font-size: 11px; color: var(--text-secondary); min-width: 80px; }
.carry-desc { font-size: 9px; color: var(--text-muted); }

/* Step Toolbar */
.step-toolbar { display: flex; align-items: center; gap: 6px; padding: 6px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.stb { padding: 3px 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: 9px; font-weight: 700; cursor: pointer; }
.stb:hover { color: var(--text-primary); border-color: var(--text-muted); }
.stb-count { font-size: 10px; color: var(--accent); margin-left: auto; font-weight: 800; }

.step-char { font-size: 9px; color: var(--accent); background: var(--accent-dim); padding: 1px 6px; border-radius: 3px; margin-left: auto; }
.eg-steps { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.eg-step { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 12px; cursor: pointer; transition: border-color 0.15s; }
.eg-step:hover { border-color: #333; }
.eg-step.step-selected { border-color: var(--accent); background: var(--accent-dim); }
.step-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.step-send { background: none; border: none; cursor: pointer; font-size: 12px; margin-left: auto; opacity: 0.4; transition: opacity 0.15s; }
.step-send:hover { opacity: 1; }
.step-no { font-size: 12px; font-weight: 800; color: var(--text-primary); }
.step-badge { padding: 2px 8px; border-radius: 4px; font-size: 8px; font-weight: 900; letter-spacing: 1px; }
.step-badge.parent { background: var(--accent-dim); color: var(--accent); }
.step-badge.child { background: rgba(96,165,250,0.1); color: #60a5fa; }
.step-diff { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.diff-tag { padding: 2px 8px; border-radius: 4px; font-size: 9px; }
.diff-tag.add { background: rgba(74,222,128,0.1); color: #4ade80; border: 1px solid rgba(74,222,128,0.2); }
.diff-tag.rm { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.2); text-decoration: line-through; }
.step-prompt { font-size: 11px; color: var(--text-secondary); line-height: 1.5; max-height: 80px; overflow-y: auto; }
.eg-actions { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; }
.eg-row { display: flex; align-items: center; gap: 6px; }
.eg-row input { width: 60px; }
.eg-mini { font-size: 9px; color: var(--text-muted); font-weight: 700; }
.eg-btn { padding: 8px 16px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-size: 11px; font-weight: 700; cursor: pointer; }
.eg-btn.primary { background: var(--accent); color: #000; border: none; }
</style>
