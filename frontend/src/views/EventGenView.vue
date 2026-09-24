<template>
  <div class="eg-view">
    <!-- 질의 열 — 결과가 나와도 사라지지 않는다.
         예전엔 검색하면 폼이 통째로 결과로 바뀌어서, 태그를 고치려면 되돌아가야 했다.
         결과를 보며 고쳐 바로 다시 돌리는 게 이 화면이 하는 일이다. -->
    <aside class="eg-query">
      <div class="q-head">
        <span class="q-kicker">Danbooru</span>
        <h3>이벤트 생성</h3>
      </div>

      <div class="q-scroll">
        <div class="form-row">
          <div class="form-field wide">
            <label>캐릭터</label>
            <input v-model="character" placeholder="hatsune_miku" @keydown.enter="searchEvents" />
          </div>
          <div class="form-field">
            <label>작품</label>
            <input v-model="copyright" placeholder="genshin_impact" @keydown.enter="searchEvents" />
          </div>
        </div>

        <div class="form-row">
          <div class="form-field wide">
            <label>일반 태그</label>
            <input v-model="prompt" placeholder="1girl, blue_hair, sword" @keydown.enter="searchEvents" />
          </div>
          <div class="form-field">
            <label>작가</label>
            <input v-model="artist" placeholder="작가명" @keydown.enter="searchEvents" />
          </div>
        </div>

        <details class="exclude-section" open>
          <summary class="exclude-toggle">제외 태그 <Icon name="chevron-down" size="12" /></summary>
          <div class="form-field">
            <input v-model="excludeTags" placeholder="제외할 태그..." @keydown.enter="searchEvents" />
          </div>
        </details>

        <div class="form-row">
          <div class="form-field">
            <label>최소 스텝</label>
            <input type="number" v-model.number="minSteps" min="1" max="50" @keydown.enter="searchEvents" />
          </div>
          <div class="form-field">
            <label>최대 스텝</label>
            <input type="number" v-model.number="maxSteps" min="1" max="100" @keydown.enter="searchEvents" />
          </div>
        </div>

        <div class="form-field">
          <label>등급</label>
          <div class="rating-row">
            <button v-for="r in ratings" :key="r.key" class="rating-chip"
              :class="{ active: r.checked }" @click="r.checked = !r.checked">{{ r.label }}</button>
          </div>
        </div>

        <label class="opt-line"><input type="checkbox" v-model="limitResults" />상위 100개만</label>
      </div>

      <div class="q-foot">
        <button class="go-btn" @click="searchEvents" :disabled="searching">
          <Icon name="rocket" /> {{ searching ? '검색 중…' : '검색' }}
        </button>
        <button class="io-btn" v-host-dialog="'import_event_results'" @click="importEvents"><Icon name="download" /> .parquet 가져오기</button>
      </div>
    </aside>

    <!-- 무대 — 검색 중 · 아직 없음 · 결과. 질의 열이 늘 떠 있으므로 이 셋은
         '어느 화면인가'가 아니라 '무대에 무엇이 있는가'다. -->
    <section class="eg-stage">
      <div v-if="searching" class="loading">
        <div class="spinner"></div>
        <p>{{ loadingMsg || '이벤트 검색 중...' }}</p>
        <div class="search-progress" v-if="searchTotal > 0">
          <div class="sp-bar">
            <div class="sp-fill" :style="{ width: searchPct + '%' }"></div>
          </div>
          <span class="sp-text">{{ searchCur.toLocaleString() }} / {{ searchTotal.toLocaleString() }} ({{ searchPct }}%)</span>
        </div>
      </div>

      <div v-else-if="events.length === 0" class="eg-blank">
        <div class="eg-empty-icon"><Icon name="video" /></div>
        <h2>이벤트를 검색하세요</h2>
        <p>왼쪽에 캐릭터나 태그를 넣고 검색하면 연속 태그 변화 시퀀스를 찾습니다</p>
        <div class="hints">
          <span>쉼표(,) = AND</span><span>[A|B] = OR</span><span>이벤트 = 연속 태그 변화 시퀀스</span>
        </div>
      </div>

      <template v-else>
        <div class="result-bar">
          <span class="bar-count">이벤트 {{ events.length }}개</span>
          <div class="bar-spacer"></div>
          <button class="bar-btn" v-host-dialog="'export_event_results'" @click="exportEvents"><Icon name="upload" /> 내보내기</button>
          <button class="bar-btn" @click="clearResults">결과 지우기</button>
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
              <div class="eg-empty-icon"><Icon name="video" /></div>
              <h2>이벤트 시퀀스</h2>
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
                    <button class="step-send" @click.stop="sendStepToT2I(step)" title="T2I로 전송"><Icon name="upload" /></button>
                  </div>
                  <div class="step-diff" v-if="i > 0 && (step.added?.length || step.removed?.length)">
                    <span v-for="t in (step.added || [])" :key="'a'+t" class="diff-tag add">+ {{ t }}</span>
                    <span v-for="t in (step.removed || [])" :key="'r'+t" class="diff-tag rm">- {{ t }}</span>
                  </div>
                  <div class="step-prompt">{{ effectiveStepPrompt(step) }}</div>
                </div>
              </div>

              <div class="eg-actions">
                <div class="eg-row">
                  <span class="eg-mini">반복</span>
                  <input type="number" v-model.number="repeatCount" min="1" max="100" />
                </div>
                <button class="eg-btn" @click="addToQueue" :disabled="selectedCount === 0">
                  큐에 추가 ({{ selectedCount }})
                </button>
                <button class="eg-btn primary" @click="generateNow" :disabled="selectedCount === 0">
                  지금 생성 ({{ selectedCount }})
                </button>
              </div>
            </template>
          </section>
        </div>
      </template>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import { applyEventCarry, effectiveStepPrompt } from '../utils/eventCarry'
import { onBackendEvent } from '../bridge.js'

interface Rating { key: string; label: string; checked: boolean }
interface EventStep { prompt?: string; displayPrompt?: string; type?: string; added?: string[]; removed?: string[]; [k: string]: any }
interface EventItem {
  character?: string; copyright?: string; similarity?: number;
  children_count?: number; parent_tags?: string; general?: string;
  steps?: EventStep[]; children?: EventStep[]; error?: string; [k: string]: any
}

const ratings = reactive<Rating[]>([
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
const repeatCount = ref(1)
const searching = ref(false)
const loadingMsg = ref('')
const searchCur = ref(0)
const searchTotal = ref(0)
const searchPct = computed(() => searchTotal.value ? Math.round(searchCur.value / searchTotal.value * 100) : 0)
const events = ref<EventItem[]>([])
const steps = ref<EventStep[]>([])
const selectedIdx = ref(-1)
const selectedSteps = reactive<Record<string, boolean>>({})  // { index: true }
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
      limitResults: limitResults.value,
      ratings: ratings.map(r => ({ key: r.key, checked: r.checked })),
    }))
  } catch {}
}

function searchEvents() {
  // 입력칸의 Enter 는 버튼의 :disabled 가드를 거치지 않는다 — 검색 중 재진입을 여기서 막는다.
  // (막지 않으면 워커가 하나 더 떠 늦게 끝난 옛 결과가 새 결과를 덮어쓴다)
  if (searching.value) return
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

// carry 옵션 적용된 스텝 — 카드 표시·큐·T2I 전송이 모두 이 결과(effectiveStepPrompt)를 쓴다.
const displaySteps = computed(() => applyEventCarry(steps.value, {
  appearance: carryAppearance.value,
  costume: carryCostume.value,
  background: carryBackground.value,
}))

/**
 * 결과 목록 갈아끼우기.
 *
 * 선택 상태를 **같이** 비워야 한다. 안 그러면 새 검색 뒤에도 `selectedSteps` 가
 * 남아 `_buildScenarios` 가 이전 검색의 스텝을 큐에 넣는다. 예전엔 결과가 나오면
 * 화면이 통째로 바뀌어 눈에 잘 안 띄었지만, 질의 열이 늘 떠 있는 지금은
 * 같은 자리에서 바로 다시 검색하게 되므로 훨씬 자주 밟힌다.
 */
function _setEvents(list: EventItem[]) {
  events.value = list
  steps.value = []
  selectedIdx.value = -1
  Object.keys(selectedSteps).forEach((k) => delete selectedSteps[k])
}

/** 무대만 비운다 — 질의는 그대로 두어 바로 고쳐 다시 돌릴 수 있게. */
function clearResults() {
  _setEvents([])
  loadingMsg.value = ''
  searchCur.value = 0
  searchTotal.value = 0
  // 캐시를 안 지우면 앱을 다시 켰을 때 지운 결과가 되살아난다.
  try { window.localStorage.removeItem('lastEventResults') } catch {}
}

function selectEvent(i: number) {
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
}

function toggleStep(i: number) { selectedSteps[i] = !selectedSteps[i] }
function selectAllSteps() { steps.value.forEach((_, i) => { selectedSteps[i] = true }) }
function clearAllSteps() { Object.keys(selectedSteps).forEach(k => { selectedSteps[k] = false }) }

function _buildScenarios() {
  // 선택된 스텝만 시나리오로 변환
  const scenarios: any[] = []
  const dSteps = displaySteps.value
  for (const [idx, checked] of Object.entries(selectedSteps)) {
    if (!checked) continue
    const step = dSteps[parseInt(idx)]
    if (!step) continue
    const prompt = effectiveStepPrompt(step)
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

function sendStepToT2I(step: EventStep) {
  // 카드에 보이는 carry 적용 프롬프트를 그대로 보낸다 (원본 step.prompt 가 아니라).
  const prompt = effectiveStepPrompt(step)
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
      if (typeof d.limitResults === 'boolean') limitResults.value = d.limitResults
      if (d.ratings) d.ratings.forEach((r: any) => { const found = ratings.find(rt => rt.key === r.key); if (found) found.checked = r.checked })
    }
  } catch {}
  // 이전 검색 결과 복원
  try {
    const saved = window.localStorage.getItem('lastEventResults')
    if (saved) {
      const data = JSON.parse(saved)
      if (Array.isArray(data) && data.length > 0) _setEvents(data)
    }
  } catch {}
  // Event 데이터 적재 문구 전용 채널. Search 탭의 searchStatus 를 같이 들으면
  // 두 탭이 keep-alive 로 살아 있는 동안 서로의 로딩 문구를 덮어쓴다.
  onBackendEvent('eventLoadStatus', (msg: string) => {
    if (searching.value) loadingMsg.value = msg
  })
  onBackendEvent('eventSearchProgress', (cur: number, total: number) => {
    searchCur.value = cur
    searchTotal.value = total
    loadingMsg.value = `유사도 검색 중... (${cur.toLocaleString()} / ${total.toLocaleString()})`
  })
  onBackendEvent('eventSearchResults', (json: string) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        _setEvents(data)
        if (data.length) {
          requestAction('show_toast', { type: 'success', msg: `${data.length}개 이벤트 발견` })
          try { window.localStorage.setItem('lastEventResults', JSON.stringify(data.slice(0, 200))) } catch {}
        }
        else {
          requestAction('show_toast', { type: 'info', msg: '검색 결과가 없습니다' })
          // 0건인데 캐시를 남기면, 다음 실행 때 **새 질의 옆에 옛 결과**가 뜬다.
          // 질의 열이 늘 보이므로 둘이 한 쌍처럼 읽혀 더 나쁘다.
          try { window.localStorage.removeItem('lastEventResults') } catch {}
        }
      }
      else if (data.error) { requestAction('show_toast', { type: 'error', msg: data.error }) }
    } catch {}
    searching.value = false
    loadingMsg.value = ''
  })
  // import 결과 수신
  onBackendEvent('eventImportResults', (json: string) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        _setEvents(data)
        requestAction('show_toast', { type: 'success', msg: `${data.length}개 이벤트 불러옴` })
      }
    } catch {}
    // 가져오기는 검색을 대신할 수 있다 — 스피너가 남아 무대를 가리면 안 된다.
    searching.value = false
  })
})
</script>

<style scoped>
/* 질의 열 + 무대. 세로가 아니라 **가로**다 — 질의가 결과에 자리를 내주지 않는다. */
.eg-view { height: 100%; display: flex; background: var(--bg-primary); overflow: hidden; }

/* ── 질의 열 ── */
/* 360 은 다른 탭의 설정 열과 같은 값이다. 탭마다 다르면 탭을 옮길 때마다
   눈이 기준선을 다시 찾는다. */
.eg-query { width: 360px; flex-shrink: 0; display: flex; flex-direction: column; border-right: 1px solid var(--border); background: var(--bg-secondary); }
.q-head { padding: 14px 16px 10px; flex-shrink: 0; }
.q-kicker { display: block; color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.q-head h3 { margin: 2px 0 0; font-size: 15px; font-weight: var(--fw-bold); color: var(--text-primary); letter-spacing: 0; }
.q-scroll { flex: 1; min-height: 0; overflow-y: auto; padding: 0 16px 14px; display: flex; flex-direction: column; gap: 12px; }
/* 검색 버튼은 늘 손 닿는 곳에 — 폼이 길어져도 스크롤 밖으로 나가지 않는다. */
.q-foot { flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); }

.form-row { display: flex; gap: 10px; }
.form-field { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.form-field.wide { flex: 2; }
.form-field label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-bottom: 0; }
.form-field input { width: 100%; padding: 8px 10px; font-size: 12px; }

.exclude-section { border: 1px solid rgba(248,113,113,0.1); border-radius: var(--radius-card); padding: 10px; }
.exclude-toggle { font-size: 11px; font-weight: var(--fw-bold); color: var(--state-alert-fg); cursor: pointer; letter-spacing: 0; list-style: none; margin-bottom: 6px; }
.exclude-toggle::-webkit-details-marker { display: none; }

.rating-row { display: flex; gap: 5px; }
.rating-chip {
  flex: 1; height: 28px; padding: 0 6px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: var(--radius-pill); color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold);
  cursor: pointer; letter-spacing: 0; transition: var(--transition);
}
.rating-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }

.opt-line { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text-secondary); cursor: pointer; }

.go-btn {
  width: 100%; height: 40px; display: flex; align-items: center; justify-content: center; gap: 6px;
  background: var(--accent-fill); border: none;
  border-radius: var(--radius-pill); color: var(--on-accent); font-weight: var(--fw-bold); font-size: 12px;
  letter-spacing: 0; cursor: pointer; transition: var(--transition);
}
.go-btn:hover { background: var(--accent-hover); }
.go-btn:disabled { opacity: 0.3; cursor: not-allowed; }
.io-btn { width: 100%; height: 30px; display: flex; align-items: center; justify-content: center; gap: 6px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; transition: var(--transition); }
.io-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }

/* ── 무대 ── */
.eg-stage { flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
.eg-blank { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center; padding: 24px; }
.eg-blank .eg-empty-icon { font-size: 48px; opacity: 0.2; margin-bottom: 4px; }
.eg-blank h2 { font-size: 16px; color: var(--text-secondary); letter-spacing: 0; }
.eg-blank p { font-size: 12px; color: var(--text-muted); }
.hints { display: flex; gap: 16px; margin-top: 12px; }
.hints span { font-size: var(--fs-label); color: var(--text-muted); }

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
/* 오른쪽 여백은 알림 종 자리다 (style.css --notif-gutter) — 없으면 마지막 버튼이 안 눌린다 */
.result-bar { display: flex; align-items: center; gap: 12px; padding: 8px var(--notif-gutter) 8px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.bar-btn { padding: 5px 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.bar-btn:hover { color: var(--text-primary); border-color: var(--text-muted); }
.bar-count { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-bold); letter-spacing: 0; }
.bar-spacer { flex: 1; }

/* ── Result Body ── */
.result-body { flex: 1; display: flex; overflow: hidden; }

/* Event List */
.eg-list { width: 220px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid var(--border); }
.eg-list-item { display: flex; align-items: center; gap: 6px; padding: 10px 12px; cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.02); font-size: 11px; transition: background 0.15s; }
.eg-list-item:hover { background: rgba(255,255,255,0.02); }
.eg-list-item.active { background: var(--accent-dim); border-left: 3px solid var(--accent); }
.eg-idx { color: var(--text-muted); font-family: monospace; font-size: var(--fs-label); min-width: 24px; }
.eg-desc { flex: 1; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.eg-cnt { font-size: var(--fs-label); color: var(--accent); font-weight: var(--fw-bold); }

/* Main */
.eg-main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.eg-empty { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; opacity: 0.2; }
.eg-empty-icon { font-size: 48px; margin-bottom: 12px; }
.eg-empty h2 { letter-spacing: 0; }
/* Event Info */
.eg-event-info { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; background: rgba(250,204,21,0.03); }
.ei-char { font-size: 12px; font-weight: var(--fw-bold); color: var(--accent); }
.ei-copy { font-size: 11px; color: var(--text-muted); }
.ei-sim { font-size: var(--fs-label); color: var(--state-ok-fg); margin-left: auto; font-family: monospace; }

/* Carry Options */
.eg-carry { display: flex; flex-direction: column; gap: 4px; padding: 10px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.carry-opt { display: flex; align-items: center; gap: 8px; }
.carry-opt label { font-size: 11px; color: var(--text-secondary); min-width: 80px; }
.carry-desc { font-size: var(--fs-label); color: var(--text-muted); }

/* Step Toolbar */
.step-toolbar { display: flex; align-items: center; gap: 6px; padding: 6px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.stb { padding: 3px 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.stb:hover { color: var(--text-primary); border-color: var(--text-muted); }
.stb-count { font-size: var(--fs-label); color: var(--accent); margin-left: auto; font-weight: var(--fw-bold); }

.step-char { font-size: var(--fs-label); color: var(--accent); background: var(--accent-dim); padding: 1px 6px; border-radius: 3px; margin-left: auto; }
.eg-steps { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
.eg-step { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; padding: 12px; cursor: pointer; transition: border-color 0.15s; }
.eg-step:hover { border-color: var(--border); }
.eg-step.step-selected { border-color: var(--accent); background: var(--accent-dim); }
.step-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.step-send { background: none; border: none; cursor: pointer; font-size: 12px; margin-left: auto; opacity: 0.4; transition: opacity 0.15s; }
.step-send:hover { opacity: 1; }
.step-no { font-size: 12px; font-weight: var(--fw-bold); color: var(--text-primary); }
.step-badge { padding: 2px 8px; border-radius: 4px; font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.step-badge.parent { background: var(--accent-dim); color: var(--accent); }
.step-badge.child { background: rgba(96,165,250,0.1); color: var(--state-info-fg); }
.step-diff { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
.diff-tag { padding: 2px 8px; border-radius: 4px; font-size: var(--fs-label); }
.diff-tag.add { background: rgba(74,222,128,0.1); color: var(--state-ok-fg); border: 1px solid rgba(74,222,128,0.2); }
.diff-tag.rm { background: rgba(248,113,113,0.1); color: var(--state-alert-fg); border: 1px solid rgba(248,113,113,0.2); text-decoration: line-through; }
.step-prompt { font-size: 11px; color: var(--text-secondary); line-height: 1.5; max-height: 80px; overflow-y: auto; }
.eg-actions { display: flex; align-items: center; gap: 8px; padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; }
.eg-row { display: flex; align-items: center; gap: 6px; }
.eg-row input { width: 60px; }
.eg-mini { font-size: var(--fs-label); color: var(--text-muted); font-weight: var(--fw-bold); }
.eg-btn { padding: 8px 16px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-size: 11px; font-weight: var(--fw-bold); cursor: pointer; }
.eg-btn.primary { background: var(--accent-fill); color: var(--on-accent); border: none; }

/* ── 좁은 창 ───────────────────────────────────────────────────────────
   이 화면은 질의(360) · 이벤트 목록(220) · 스텝 상세, 세 열이다. 창이 줄면
   맨 오른쪽 상세가 먼저 짓눌린다 — 정작 일하는 자리가 거기다.
   창은 1600 으로 열리지만 최소 크기 제한이 없어 사용자가 얼마든지 줄일 수 있다. */
@media (max-width: 1200px) {
  .eg-query { width: 300px; }
  .eg-list { width: 180px; }
}
@media (max-width: 900px) {
  /* 셋을 한 줄에 못 넣는다 — 목록을 상세 위로 눕혀 가로를 되찾는다 */
  .result-body { flex-direction: column; }
  .eg-list { width: auto; max-height: 132px; flex-shrink: 0; border-right: 0; border-bottom: 1px solid var(--border); }
}
</style>
