<template>
  <div class="search-view">
    <!-- 검색 전: 중앙 검색 폼 -->
    <div v-if="results.length === 0 && !searching" class="welcome">
      <div class="ws-header">
        <div class="ws-icon">🔍</div>
        <h1>TAG EXPLORER</h1>
        <p>Danbooru 데이터베이스에서 프롬프트를 검색합니다</p>
      </div>

      <div class="search-form">
        <!-- Include -->
        <div class="form-section">
          <div class="form-row">
            <div class="form-field wide">
              <label>Character</label>
              <input v-model="fields[1].include" placeholder="e.g. hatsune_miku, raiden_shogun" @keydown.enter="search" />
            </div>
            <div class="form-field">
              <label>Copyright</label>
              <input v-model="fields[0].include" placeholder="e.g. genshin_impact" @keydown.enter="search" />
            </div>
          </div>
          <div class="form-row">
            <div class="form-field wide">
              <label>General Tags</label>
              <input v-model="fields[3].include" placeholder="e.g. 1girl, blue_hair, sword, outdoors" @keydown.enter="search" />
            </div>
            <div class="form-field">
              <label>Artist</label>
              <input v-model="fields[2].include" placeholder="Artist name..." @keydown.enter="search" />
            </div>
          </div>
        </div>

        <!-- Exclude (접이식) -->
        <details class="form-section exclude-section" open>
          <summary class="exclude-toggle">Exclude Tags ▾</summary>
          <div class="form-row">
            <div class="form-field"><label class="danger">Character</label><input v-model="fields[1].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">Copyright</label><input v-model="fields[0].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">Tags</label><input v-model="fields[3].exclude" placeholder="제외..." @keydown.enter="search" /></div>
            <div class="form-field"><label class="danger">Artist</label><input v-model="fields[2].exclude" placeholder="제외..." @keydown.enter="search" /></div>
          </div>
        </details>

        <!-- Rating + Go -->
        <div class="form-footer">
          <div class="rating-row">
            <button v-for="r in ratings" :key="r.key" class="rating-chip"
              :class="{ active: r.checked }" @click="r.checked = !r.checked">{{ r.label }}</button>
          </div>
          <div class="io-row">
            <button class="io-btn" @click="importResults">📤 IMPORT .parquet</button>
          </div>
          <button class="go-btn" @click="search" :disabled="searching">🚀 RUN SEARCH</button>
        </div>
      </div>

      <div class="hints">
        <span>쉼표(,) = AND</span><span>[A|B] = OR</span><span>Exclude로 제외 조건 설정</span>
      </div>

      <!-- 이전 검색 결과 있으면 바로 보기/랜덤 뽑기 -->
      <div class="prev-results" v-if="lastResults.length > 0">
        <span class="prev-label">이전 검색 결과 {{ lastResults.length }}건</span>
        <button class="prev-btn" @click="restoreLastResults">📋 목록 보기</button>
        <button class="prev-btn gold" @click="restoreAndRandom">🎲 랜덤 뽑기</button>
      </div>
    </div>

    <!-- 검색 중 -->
    <div v-else-if="results.length === 0 && searching" class="loading">
      <div class="spinner"></div>
      <h2>SEARCHING...</h2>
      <div class="progress-bar"><div class="progress-fill" :style="{ width: searchProgress + '%' }"></div></div>
      <p>{{ statusText }}</p>
    </div>

    <!-- 검색 후: 결과 -->
    <template v-else>
      <!-- 상단 바 -->
      <div class="result-bar">
        <div class="bar-left">
          <button class="bar-btn" @click="viewMode = 'single'" :class="{ active: viewMode === 'single' }">📄 Single</button>
          <button class="bar-btn" @click="viewMode = 'list'" :class="{ active: viewMode === 'list' }">📋 List</button>
          <span class="bar-sep">|</span>
          <button class="bar-btn" @click="prevResult" :disabled="previewIdx <= 0">◀</button>
          <span class="bar-idx"><b>{{ previewIdx + 1 }}</b>/{{ filteredResults.length }}</span>
          <button class="bar-btn" @click="nextResult" :disabled="previewIdx >= filteredResults.length - 1">▶</button>
          <button class="bar-btn gold" @click="randomResult">🎲 Random</button>
        </div>
        <div class="bar-right">
          <button class="bar-btn parquet" @click="exportResults">📥 EXPORT .parquet</button>
          <button class="bar-btn" @click="newSearch">🔍 새 검색</button>
        </div>
      </div>
      <!-- 심층검색 바 + 분기 -->
      <div class="deep-bar">
        <span class="deep-label">DEEP SEARCH</span>
        <input v-model="deepInclude" placeholder="포함 태그 (쉼표 구분)" @keydown.enter="applyDeepSearch" class="deep-input" />
        <input v-model="deepExclude" placeholder="제외 태그 (쉼표 구분)" @keydown.enter="applyDeepSearch" class="deep-input neg" />
        <button class="bar-btn" @click="applyDeepSearch">FILTER</button>
        <template v-if="filterHistory.length > 0">
          <span class="bar-sep">|</span>
          <span class="branch-label">분기:</span>
          <button v-for="(fh, fi) in filterHistory" :key="fi" class="branch-btn"
            @click="restoreBranch(fi)">{{ fh.label }} ({{ fh.count }})</button>
        </template>
        <button class="bar-btn" @click="resetDeepSearch" v-if="isFiltered">✕ RESET</button>
        <button class="bar-btn filter-btn" :class="{ active: hasActiveFilters }" @click="showFilterManager = true">
          🔽 FILTER{{ hasActiveFilters ? ` (${activeFilterCount})` : '' }}
        </button>
        <button v-if="hasActiveFilters" class="bar-btn" @click="clearAllFilters">✕ 필터 해제</button>
        <span class="bar-count">{{ filteredResults.length }} / {{ results.length }}</span>
      </div>

      <!-- Single View -->
      <div v-if="viewMode === 'single'" class="single-view">
        <div class="detail-card" v-if="currentResult">
          <div class="detail-meta">
            <div class="meta-pill"><span class="ml">PROJECT</span>{{ currentResult.copyright || 'ORIGINAL' }}</div>
            <div class="meta-pill artist"><span class="ml">ARTIST</span>{{ currentResult.artist || 'UNKNOWN' }}</div>
            <div class="meta-pill character"><span class="ml">CHAR</span>{{ currentResult.character || 'GENERIC' }}</div>
            <div class="meta-pill mini"><span class="ml">RATING</span>{{ currentResult.rating || '?' }}</div>
            <div class="meta-pill mini"><span class="ml">TAGS</span>{{ currentTags.length }}</div>
          </div>
          <div class="tag-section">
            <label>General Tags ({{ currentTags.length }})</label>
            <div class="tag-cloud">
              <span v-for="tag in currentTags" :key="tag" class="tag" :class="tagColor(tag)">{{ tag }}</span>
            </div>
          </div>
          <!-- 태그 색상 범례 -->
          <div class="tag-legend">
            <span class="legend-item count">인물수</span>
            <span class="legend-item clothing">의상</span>
            <span class="legend-item body">신체</span>
            <span class="legend-item nsfw">NSFW</span>
            <span class="legend-item action">포즈</span>
            <span class="legend-item expression">표정</span>
            <span class="legend-item bg">배경</span>
            <span class="legend-item objects">사물</span>
            <span class="legend-item effect">효과</span>
            <span class="legend-item color-tag">색상</span>
            <span class="legend-item trait">특성</span>
          </div>
          <div class="detail-actions">
            <button class="primary-btn" @click="applyResult">USE AS PROMPT</button>
            <button class="secondary-btn" @click="addToQueue">ADD TO QUEUE</button>
            <button class="secondary-btn" @click="randomResult">🎲 NEXT RANDOM</button>
          </div>
        </div>
      </div>

      <!-- List View -->
      <div v-else class="list-view">
        <div class="list-header">
          <span class="lh idx">#</span><span class="lh char">Character</span>
          <span class="lh copy">Copyright</span><span class="lh artist">Artist</span>
          <span class="lh tags">Tags</span><span class="lh act"></span>
        </div>
        <div class="list-scroll">
          <div v-for="(r, i) in filteredResults" :key="i" class="list-row"
            :class="{ active: previewIdx === i }"
            @click="previewIdx = i; viewMode = 'single'">
            <span class="lr idx">{{ i + 1 }}</span>
            <span class="lr char">{{ r.character || '-' }}</span>
            <span class="lr copy">{{ r.copyright || '-' }}</span>
            <span class="lr artist">{{ r.artist || '-' }}</span>
            <span class="lr tags">{{ (r.general || '').substring(0, 80) }}</span>
            <span class="lr act"><button class="use-btn" @click.stop="previewIdx = i; applyResult()">USE</button></span>
          </div>
        </div>
      </div>
      <!-- Filter Manager Modal -->
      <transition name="fade">
        <div v-if="showFilterManager" class="fm-overlay" @click.self="showFilterManager = false">
          <div class="fm-modal">
            <div class="fm-header">
              <h3>FILTER MANAGER</h3>
              <button class="close-btn" @click="showFilterManager = false">✕</button>
            </div>
            <div class="fm-body">
              <!-- Rating -->
              <div class="fm-section">
                <div class="fm-label">RATING</div>
                <div class="fm-chips">
                  <button v-for="r in filterOptions.ratings" :key="r" class="fm-chip"
                    :class="{ active: activeFilters.ratings.has(r) }"
                    @click="toggleFilter('ratings', r)">{{ r.toUpperCase() }}</button>
                </div>
              </div>
              <!-- Character -->
              <div class="fm-section" v-if="filterOptions.characters.length">
                <div class="fm-label">CHARACTER ({{ filterOptions.characters.length }})</div>
                <input class="fm-search" v-model="fmCharSearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="c in filteredFmChars" :key="c" class="fm-chip"
                    :class="{ active: activeFilters.characters.has(c) }"
                    @click="toggleFilter('characters', c)">{{ c }}</button>
                </div>
              </div>
              <!-- Copyright -->
              <div class="fm-section" v-if="filterOptions.copyrights.length">
                <div class="fm-label">COPYRIGHT ({{ filterOptions.copyrights.length }})</div>
                <input class="fm-search" v-model="fmCopySearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="c in filteredFmCopyrights" :key="c" class="fm-chip"
                    :class="{ active: activeFilters.copyrights.has(c) }"
                    @click="toggleFilter('copyrights', c)">{{ c }}</button>
                </div>
              </div>
              <!-- Artist -->
              <div class="fm-section" v-if="filterOptions.artists.length">
                <div class="fm-label">ARTIST ({{ filterOptions.artists.length }})</div>
                <input class="fm-search" v-model="fmArtistSearch" placeholder="검색..." />
                <div class="fm-chip-scroll">
                  <button v-for="a in filteredFmArtists" :key="a" class="fm-chip"
                    :class="{ active: activeFilters.artists.has(a) }"
                    @click="toggleFilter('artists', a)">{{ a }}</button>
                </div>
              </div>
            </div>
            <div class="fm-footer">
              <span class="fm-count">{{ filteredByManager.length }} / {{ results.length }} 결과</span>
              <button class="fm-apply" @click="applyFilterManager">APPLY FILTER</button>
            </div>
          </div>
        </div>
      </transition>

      <!-- 조건부 프롬프트 (항상 표시) -->
      <div class="cond-section" v-if="results.length > 0 || condPositive.length > 0 || condNegative.length > 0">
        <details class="cond-card">
          <summary class="cond-title positive">CONDITIONAL POSITIVE</summary>
          <p class="cond-desc">태그가 존재하면 자동으로 다른 태그를 추가/제거합니다</p>
          <div v-for="(rule, ri) in condPositive" :key="'p'+ri" class="cond-rule-block">
            <div class="cond-row1">
              <input type="checkbox" v-model="rule.enabled" />
              <span class="cond-kw">IF</span>
              <input v-model="rule.condition" placeholder="조건 태그" class="cond-input" />
              <select v-model="rule.exists" class="cond-sel"><option :value="true">있으면</option><option :value="false">없으면</option></select>
              <button class="cond-rm" @click="condPositive.splice(ri, 1)">✕</button>
            </div>
            <div class="cond-row2">
              <span class="cond-kw">→</span>
              <input v-model="rule.target" placeholder="대상 태그" class="cond-input" />
              <select v-model="rule.action" class="cond-sel"><option value="add">추가</option><option value="remove">제거</option><option value="replace">대체</option></select>
              <select v-model="rule.location" class="cond-sel"><option value="main">main</option><option value="prefix">prefix</option><option value="suffix">suffix</option></select>
            </div>
          </div>
          <button class="cond-add" @click="condPositive.push({enabled:true,condition:'',exists:true,target:'',action:'add',location:'main'})">+ 규칙 추가</button>
        </details>

        <details class="cond-card neg">
          <summary class="cond-title negative">CONDITIONAL NEGATIVE</summary>
          <p class="cond-desc">태그가 존재하면 네거티브 프롬프트에 자동 추가/제거합니다</p>
          <div v-for="(rule, ri) in condNegative" :key="'n'+ri" class="cond-rule-block">
            <div class="cond-row1">
              <input type="checkbox" v-model="rule.enabled" />
              <span class="cond-kw">IF</span>
              <input v-model="rule.condition" placeholder="조건 태그" class="cond-input" />
              <select v-model="rule.exists" class="cond-sel"><option :value="true">있으면</option><option :value="false">없으면</option></select>
              <button class="cond-rm" @click="condNegative.splice(ri, 1)">✕</button>
            </div>
            <div class="cond-row2">
              <span class="cond-kw">→</span>
              <input v-model="rule.target" placeholder="네거티브 태그" class="cond-input neg" />
              <select v-model="rule.action" class="cond-sel"><option value="add">추가</option><option value="remove">제거</option></select>
            </div>
          </div>
          <button class="cond-add" @click="condNegative.push({enabled:true,condition:'',exists:true,target:'',action:'add'})">+ 규칙 추가</button>
        </details>
        <button class="cond-save-btn" @click="saveCondRules">💾 조건식 저장</button>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import CustomSelect from '../components/CustomSelect.vue'

const ratings = reactive([
  { key: 'g', label: 'GEN', checked: true },
  { key: 's', label: 'SENS', checked: false },
  { key: 'q', label: 'QUES', checked: false },
  { key: 'e', label: 'EXPL', checked: false },
])
const fields = reactive([
  { key: 'copyright', label: 'PROJECT', placeholder: 'e.g. genshin_impact', include: '', exclude: '' },
  { key: 'character', label: 'CHAR', placeholder: 'e.g. raiden_shogun', include: '', exclude: '' },
  { key: 'artist', label: 'ARTIST', placeholder: 'Artist name...', include: '', exclude: '' },
  { key: 'general', label: 'TAGS', placeholder: '1girl, blue_hair...', include: '', exclude: '' },
])

const results = ref([])
const filteredResults = ref([])
const lastResults = ref([])  // 검색 폼으로 돌아가도 보존
const previewIdx = ref(0)
const searching = ref(false)
const statusText = ref('READY')
const searchProgress = ref(0)
const viewMode = ref('single')
const deepInclude = ref('')
const deepExclude = ref('')
const isFiltered = ref(false)
const filterHistory = ref([])

// 조건부 프롬프트
const condPositive = reactive([])
const condNegative = reactive([])
let progressTimer = null

const currentResult = computed(() => filteredResults.value[previewIdx.value] || null)
const currentTags = computed(() => (currentResult.value?.general || '').split(',').map(t => t.trim()).filter(Boolean).map(t => t.replace(/_/g, ' ')))

function persistSearchFields() {
  const payload = {
    fields: fields.map(f => ({ include: f.include, exclude: f.exclude })),
    ratings: ratings.map(r => ({ key: r.key, checked: r.checked })),
  }
  try {
    window.localStorage.setItem('lastSearchFields', JSON.stringify(payload))
  } catch {}
  requestAction('save_ui_prefs', { searchState: payload })
}

async function search() {
  searching.value = true; statusText.value = 'EXPLORING...'
  searchProgress.value = 0
  progressTimer = setInterval(() => { if (searchProgress.value < 90) searchProgress.value += Math.random() * 12 }, 500)
  const backend = await getBackend()
  const query = {
    ratings: ratings.filter(r => r.checked).map(r => r.key),
    queries: Object.fromEntries(fields.map(f => [f.key, f.include])),
    excludes: Object.fromEntries(fields.map(f => [f.key, f.exclude])),
  }
  if (backend.searchDanbooru) backend.searchDanbooru(JSON.stringify(query))
}

function newSearch() {
  results.value = []; filteredResults.value = []; previewIdx.value = 0
  deepInclude.value = ''; deepExclude.value = ''; isFiltered.value = false
  // lastResults는 보존 — 검색 폼에서 다시 볼 수 있음
}

function restoreLastResults() {
  results.value = lastResults.value
  filteredResults.value = lastResults.value
  previewIdx.value = 0
  viewMode.value = 'list'
  statusText.value = `${lastResults.value.length} MATCHES (복원)`
}

function restoreAndRandom() {
  results.value = lastResults.value
  filteredResults.value = lastResults.value
  previewIdx.value = Math.floor(Math.random() * lastResults.value.length)
  viewMode.value = 'single'
  statusText.value = `${lastResults.value.length} MATCHES (랜덤)`
}

onMounted(() => {
  // 이전 검색 입력 필드 복원
  try {
    const sf = window.localStorage.getItem('lastSearchFields')
    if (sf) {
      const d = JSON.parse(sf)
      if (d.fields) d.fields.forEach((f, i) => { if (fields[i]) { fields[i].include = f.include || ''; fields[i].exclude = f.exclude || '' } })
      if (d.ratings) d.ratings.forEach(r => { const found = ratings.find(rt => rt.key === r.key); if (found) found.checked = r.checked })
    }
  } catch {}
  // 이전 검색 결과 복원
  try {
    const saved = window.localStorage.getItem('lastSearchResults')
    if (saved) {
      const data = JSON.parse(saved)
      if (Array.isArray(data) && data.length > 0) {
        results.value = data; filteredResults.value = data; lastResults.value = data
        statusText.value = `${data.length} MATCHES (복원)`
      }
    }
  } catch {}

  onBackendEvent('searchResultsReady', (json) => {
    try {
      const data = JSON.parse(json)
      if (Array.isArray(data)) {
        results.value = data; filteredResults.value = data; previewIdx.value = 0
        lastResults.value = data
        statusText.value = `${data.length} MATCHES`
        // 자동 저장 (재시작 시 복원용)
        try { window.localStorage.setItem('lastSearchResults', JSON.stringify(data.slice(0, 500))) } catch {}
        persistSearchFields()
      } else statusText.value = 'FAILED'
    } catch { statusText.value = 'PARSE ERROR' }
    searching.value = false; searchProgress.value = 100
    if (progressTimer) { clearInterval(progressTimer); progressTimer = null }
  })
  onBackendEvent('searchStatus', (msg) => { statusText.value = msg.toUpperCase() })
  onBackendEvent('uiPrefsLoaded', (json) => {
    try {
      const prefs = JSON.parse(json)
      const state = prefs.searchState
      if (!state) return
      if (Array.isArray(state.fields)) {
        state.fields.forEach((f, i) => {
          if (fields[i]) {
            fields[i].include = f.include || ''
            fields[i].exclude = f.exclude || ''
          }
        })
      }
      if (Array.isArray(state.ratings)) {
        state.ratings.forEach(r => {
          const found = ratings.find(rt => rt.key === r.key)
          if (found) found.checked = !!r.checked
        })
      }
    } catch {}
  })

  // 조건부 프롬프트 로드
  onBackendEvent('condRulesLoaded', (json) => {
    try {
      const d = JSON.parse(json)
      if (d.positive) { condPositive.splice(0); d.positive.forEach(r => condPositive.push(r)) }
      if (d.negative) { condNegative.splice(0); d.negative.forEach(r => condNegative.push(r)) }
    } catch {}
  })
})

watch(fields, persistSearchFields, { deep: true })
watch(ratings, persistSearchFields, { deep: true })

function applyDeepSearch() {
  const inc = deepInclude.value.toLowerCase().trim()
  const exc = deepExclude.value.toLowerCase().trim()
  if (!inc && !exc) return
  // 현재 상태를 분기로 저장
  const label = [inc ? `+${inc.substring(0,15)}` : '', exc ? `-${exc.substring(0,15)}` : ''].filter(Boolean).join(' ')
  filterHistory.value.push({ label, count: filteredResults.value.length, data: [...filteredResults.value] })
  // 현재 filteredResults 기준 누적 필터
  filteredResults.value = filteredResults.value.filter(r => {
    const all = `${r.copyright} ${r.character} ${r.artist} ${r.general}`.toLowerCase()
    if (inc) { for (const t of inc.split(',')) { if (t.trim() && !all.includes(t.trim())) return false } }
    if (exc) { for (const t of exc.split(',')) { if (t.trim() && all.includes(t.trim())) return false } }
    return true
  })
  previewIdx.value = 0; isFiltered.value = true
  deepInclude.value = ''; deepExclude.value = ''
  statusText.value = `DEEP: ${filteredResults.value.length} / ${results.value.length}`
}
function restoreBranch(idx) {
  filteredResults.value = [...filterHistory.value[idx].data]
  filterHistory.value = filterHistory.value.slice(0, idx)
  previewIdx.value = 0
  isFiltered.value = filterHistory.value.length > 0
  statusText.value = `BRANCH: ${filteredResults.value.length}`
}
function resetDeepSearch() {
  deepInclude.value = ''; deepExclude.value = ''
  clearAllFilters()
  isFiltered.value = false; filterHistory.value = []
  statusText.value = `${results.value.length} MATCHES`
}

// 태그 색상 분류 — Python TagClassifier (tags_db 기반)
const tagCategoryCache = ref({})  // tag → category

// 카테고리 → CSS 클래스 매핑 (세분화)
const catToColor = {
  'sexual': 'nsfw', 'body_parts': 'body', 'clothing': 'clothing',
  'pose': 'action', 'expression': 'expression', 'background': 'bg',
  'composition': 'composition', 'effect': 'effect', 'objects': 'objects',
  'character': 'char', 'copyright': 'copy', 'artist': 'artist-tag',
  'color': 'color-tag', 'character_trait': 'trait', 'animals': 'objects',
  'art_style': 'style', 'general': '',
}

// 인물수 태그는 프론트에서 직접 판단 (빠름)
const countPattern = /^(\d+)?(girl|boy|other)s?$|^solo$|^multiple_/

function tagColor(tag) {
  const t = tag.trim().toLowerCase().replace(/ /g, '_')
  if (countPattern.test(t)) return 'count'
  // 캐시 검색 (원본 + underscore 변환 둘 다)
  const cached = tagCategoryCache.value[t] || tagCategoryCache.value[tag.trim()]
  if (cached) return catToColor[cached] || ''
  return ''
}

// 현재 보고 있는 결과의 태그를 배치로 분류 요청
async function classifyCurrentTags() {
  if (!currentResult.value) return
  const tags = currentTags.value.map(t => t.trim().replace(/ /g, '_'))
  // 캐시에 없는 태그만 요청
  const uncached = tags.filter(t => !(t in tagCategoryCache.value))
  if (uncached.length === 0) return
  const backend = await getBackend()
  if (backend.classifyTags) {
    backend.classifyTags(JSON.stringify(uncached), (json) => {
      try {
        const result = JSON.parse(json)
        if (!result.error) {
          tagCategoryCache.value = { ...tagCategoryCache.value, ...result }
        }
      } catch {}
    })
  }
}

// previewIdx 변경 시 + 결과 로드 시 자동 분류
watch(previewIdx, () => { classifyCurrentTags() })
watch(() => filteredResults.value.length, () => { if (filteredResults.value.length > 0) classifyCurrentTags() })

// ── Filter Manager ──
const showFilterManager = ref(false)
const fmCharSearch = ref('')
const fmCopySearch = ref('')
const fmArtistSearch = ref('')
const activeFilters = reactive({
  ratings: new Set(),
  characters: new Set(),
  copyrights: new Set(),
  artists: new Set(),
})

function _splitDanbooruTags(raw) {
  const text = String(raw || '').trim()
  if (!text) return []
  if (text.includes(',')) return text.split(',').map(v => v.trim().replace(/_/g, ' ')).filter(Boolean)
  const tags = []; let current = ''; let depth = 0
  for (const ch of text) {
    if (ch === '(') { depth++; current += ch }
    else if (ch === ')') { depth--; current += ch }
    else if (ch === ' ' && depth === 0) { if (current.trim()) tags.push(current.trim().replace(/_/g, ' ')); current = '' }
    else { current += ch }
  }
  if (current.trim()) tags.push(current.trim().replace(/_/g, ' '))
  return tags
}

// 검색 결과에서 필터 옵션 추출
const filterOptions = computed(() => {
  const chars = new Set(), copys = new Set(), arts = new Set(), rats = new Set()
  for (const row of results.value) {
    if (row.rating) rats.add(row.rating)
    for (const c of _splitDanbooruTags(row.character)) chars.add(c)
    for (const c of _splitDanbooruTags(row.copyright)) copys.add(c)
    for (const a of _splitDanbooruTags(row.artist)) arts.add(a)
  }
  return {
    ratings: [...rats].sort(),
    characters: [...chars].sort(),
    copyrights: [...copys].sort(),
    artists: [...arts].sort(),
  }
})

const filteredFmChars = computed(() => {
  const q = fmCharSearch.value.toLowerCase()
  return q ? filterOptions.value.characters.filter(c => c.toLowerCase().includes(q)) : filterOptions.value.characters
})
const filteredFmCopyrights = computed(() => {
  const q = fmCopySearch.value.toLowerCase()
  return q ? filterOptions.value.copyrights.filter(c => c.toLowerCase().includes(q)) : filterOptions.value.copyrights
})
const filteredFmArtists = computed(() => {
  const q = fmArtistSearch.value.toLowerCase()
  return q ? filterOptions.value.artists.filter(a => a.toLowerCase().includes(q)) : filterOptions.value.artists
})

function toggleFilter(category, value) {
  const s = activeFilters[category]
  if (s.has(value)) s.delete(value)
  else s.add(value)
}

const hasActiveFilters = computed(() =>
  activeFilters.ratings.size + activeFilters.characters.size + activeFilters.copyrights.size + activeFilters.artists.size > 0
)
const activeFilterCount = computed(() =>
  activeFilters.ratings.size + activeFilters.characters.size + activeFilters.copyrights.size + activeFilters.artists.size
)

// 필터 적용 미리보기
const filteredByManager = computed(() => {
  let next = results.value
  if (activeFilters.ratings.size) next = next.filter(r => activeFilters.ratings.has(r.rating))
  if (activeFilters.characters.size) next = next.filter(r => _splitDanbooruTags(r.character).some(c => activeFilters.characters.has(c)))
  if (activeFilters.copyrights.size) next = next.filter(r => _splitDanbooruTags(r.copyright).some(c => activeFilters.copyrights.has(c)))
  if (activeFilters.artists.size) next = next.filter(r => _splitDanbooruTags(r.artist).some(a => activeFilters.artists.has(a)))
  return next
})

function applyFilterManager() {
  filteredResults.value = [...filteredByManager.value]
  previewIdx.value = 0
  showFilterManager.value = false
  requestAction('update_prompt_deck', { results: filteredResults.value })
  requestAction('show_toast', { type: 'success', msg: `필터 적용: ${filteredResults.value.length}건` })
}

function clearAllFilters() {
  activeFilters.ratings.clear()
  activeFilters.characters.clear()
  activeFilters.copyrights.clear()
  activeFilters.artists.clear()
  filteredResults.value = [...results.value]
  previewIdx.value = 0
  requestAction('update_prompt_deck', { results: results.value })
  requestAction('show_toast', { type: 'info', msg: '필터 해제됨' })
}

function clearSortFilter() {
  clearAllFilters()
}

function prevResult() { if (previewIdx.value > 0) previewIdx.value-- }
function nextResult() { if (previewIdx.value < filteredResults.value.length - 1) previewIdx.value++ }
function randomResult() { if (filteredResults.value.length > 1) previewIdx.value = Math.floor(Math.random() * filteredResults.value.length) }
function applyResult() {
  if (!currentResult.value) return
  // 조건부 프롬프트 규칙도 함께 전달
  const payload = {
    ...currentResult.value,
    cond_positive: condPositive.filter(r => r.enabled && r.condition && r.target),
    cond_negative: condNegative.filter(r => r.enabled && r.condition && r.target),
  }
  requestAction('apply_search_result', payload)
}
function addToQueue() {
  if (!currentResult.value) return
  const payload = {
    ...currentResult.value,
    cond_positive: condPositive.filter(r => r.enabled && r.condition && r.target),
    cond_negative: condNegative.filter(r => r.enabled && r.condition && r.target),
  }
  requestAction('add_search_to_queue', payload)
}
function saveCondRules() {
  requestAction('save_cond_rules', {
    positive: condPositive.filter(r => r.condition || r.target),
    negative: condNegative.filter(r => r.condition || r.target),
  })
}

function exportResults() { requestAction('export_search_results', { count: filteredResults.value.length, data: filteredResults.value }) }
function importResults() { requestAction('import_search_results') }
</script>

<style scoped>
.search-view { height: 100%; display: flex; flex-direction: column; background: var(--bg-primary); overflow: hidden; }

/* ═══ Welcome (검색 전) ═══ */
.welcome { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px; gap: 32px; }
.ws-header { text-align: center; }
.ws-icon { font-size: 48px; opacity: 0.5; margin-bottom: 10px; }
.ws-header h1 { font-size: 22px; letter-spacing: 8px; color: var(--text-secondary); font-weight: 900; }
.ws-header p { font-size: 13px; color: var(--text-muted); margin-top: 6px; }

.search-form { width: 100%; max-width: 720px; display: flex; flex-direction: column; gap: 12px; }
.form-section { display: flex; flex-direction: column; gap: 10px; }
.form-row { display: flex; gap: 10px; }
.form-field { flex: 1; display: flex; flex-direction: column; gap: 3px; }
.form-field.wide { flex: 2; }
.form-field label { font-size: 10px; font-weight: 800; color: var(--text-muted); letter-spacing: 0.5px; }
.form-field input { padding: 11px 14px; font-size: 13px; }

.exclude-section { border: 1px solid rgba(248,113,113,0.1); border-radius: var(--radius-card); padding: 12px; }
.exclude-toggle { font-size: 11px; font-weight: 800; color: #f87171; cursor: pointer; letter-spacing: 0.5px; list-style: none; }
.exclude-toggle::-webkit-details-marker { display: none; }

.form-footer { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 4px; }
.rating-row { display: flex; gap: 4px; }
.rating-chip { padding: 7px 18px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-muted); font-size: 11px; font-weight: 800; cursor: pointer; }
.rating-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.go-btn { padding: 12px 40px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-weight: 900; font-size: 14px; cursor: pointer; letter-spacing: 1px; }
.go-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(250,204,21,0.3); }
.io-row { display: flex; gap: 4px; }
.io-btn { padding: 6px 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; }
.prev-results {
  display: flex; align-items: center; gap: 10px; padding: 12px 20px;
  background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-card);
}
.prev-label { font-size: 11px; color: var(--text-muted); font-weight: 700; flex: 1; }
.prev-btn {
  padding: 7px 16px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 6px; color: var(--text-secondary); font-size: 11px; font-weight: 700; cursor: pointer;
}
.prev-btn:hover { border-color: var(--text-muted); color: var(--text-primary); }
.prev-btn.gold { border-color: var(--accent-dim); color: var(--accent); }

.hints { display: flex; gap: 14px; }
.hints span { font-size: 10px; color: var(--text-muted); background: var(--bg-card); padding: 4px 12px; border-radius: 4px; }

/* ═══ Loading ═══ */
.loading { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }
.spinner { width: 40px; height: 40px; border: 3px solid #222; border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.loading h2 { color: var(--text-muted); letter-spacing: 4px; }
.progress-bar { width: 200px; height: 3px; background: var(--bg-input); border-radius: 2px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--accent); transition: width 0.4s; }

/* ═══ Result Bar ═══ */
.result-bar { display: flex; align-items: center; padding: 6px 12px; background: var(--bg-secondary); border-bottom: 1px solid var(--border); gap: 8px; flex-shrink: 0; }
.bar-left, .bar-right { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
.bar-center { flex: 1; display: flex; align-items: center; gap: 4px; justify-content: center; }
.bar-btn { padding: 4px 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: 10px; font-weight: 700; cursor: pointer; }
.bar-btn:hover:not(:disabled) { color: var(--text-primary); border-color: var(--text-muted); }
.bar-btn.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
.bar-btn.gold { color: var(--accent); border-color: var(--accent-dim); }
.bar-btn:disabled { opacity: 0.3; }
.bar-sep { color: #333; }
.bar-idx { font-family: monospace; font-size: 11px; color: var(--text-secondary); }
.bar-idx b { color: var(--accent); font-size: 14px; }
.bar-count { font-size: 9px; color: var(--text-muted); font-weight: 700; letter-spacing: 0.5px; }
.filter-btn.active { color: var(--accent); border-color: var(--accent-dim); }

/* Filter Manager Modal */
.fm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 2000; display: flex; align-items: center; justify-content: center; }
.fm-modal { background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 16px; width: 560px; max-height: 80vh; display: flex; flex-direction: column; }
.fm-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
.fm-header h3 { font-size: 13px; font-weight: 900; letter-spacing: 2px; color: var(--text-primary); }
.fm-body { flex: 1; overflow-y: auto; padding: 16px 20px; display: flex; flex-direction: column; gap: 16px; }
.fm-section { display: flex; flex-direction: column; gap: 6px; }
.fm-label { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 1.5px; }
.fm-search { padding: 6px 10px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; color: var(--text-primary); }
.fm-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.fm-chip-scroll { display: flex; flex-wrap: wrap; gap: 4px; max-height: 120px; overflow-y: auto; }
.fm-chip {
  padding: 4px 10px; background: var(--bg-button); border: 1px solid var(--border);
  border-radius: var(--radius-pill); color: var(--text-muted); font-size: 10px; font-weight: 700;
  cursor: pointer; transition: var(--transition); white-space: nowrap;
}
.fm-chip:hover { border-color: var(--text-muted); color: var(--text-primary); }
.fm-chip.active { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.fm-footer { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-top: 1px solid var(--border); }
.fm-count { font-size: 11px; color: var(--text-muted); font-family: monospace; }
.fm-apply { padding: 8px 24px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-weight: 800; font-size: 11px; cursor: pointer; letter-spacing: 1px; }
.bar-btn.parquet { background: rgba(96,165,250,0.1); border-color: rgba(96,165,250,0.3); color: #60a5fa; }

/* Deep bar */
.deep-bar {
  display: flex; align-items: center; gap: 6px; padding: 5px 12px;
  background: rgba(250,204,21,0.02); border-bottom: 1px solid var(--border); flex-shrink: 0; flex-wrap: wrap;
}
.deep-label { font-size: 9px; font-weight: 900; color: var(--accent); letter-spacing: 1px; }
.deep-input { width: 160px; padding: 4px 8px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); font-size: 10px; }
.deep-input.neg { border-color: rgba(248,113,113,0.2); }
.branch-label { font-size: 9px; color: var(--text-muted); font-weight: 700; }
.branch-btn { padding: 3px 8px; background: var(--bg-button); border: 1px solid var(--accent-dim); border-radius: 4px; color: var(--accent); font-size: 9px; font-weight: 700; cursor: pointer; }
.branch-btn:hover { background: var(--accent-dim); }

/* ═══ Single View ═══ */
.single-view { flex: 1; overflow-y: auto; padding: 24px; display: flex; justify-content: center; }
.detail-card { max-width: 720px; width: 100%; display: flex; flex-direction: column; gap: 20px; }
.detail-meta { display: flex; gap: 10px; flex-wrap: wrap; }
.meta-pill { padding: 12px 16px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 10px; font-size: 14px; font-weight: 800; color: var(--text-primary); display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 140px; }
.meta-pill.artist { color: var(--accent); }
.meta-pill.character { color: #4ade80; }
.meta-pill.mini { min-width: 60px; flex: 0; }
.ml { font-size: 9px; font-weight: 900; color: var(--text-muted); letter-spacing: 1px; }

.tag-section { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.tag-section label { font-size: 10px; font-weight: 900; color: var(--text-muted); letter-spacing: 1px; margin-bottom: 10px; display: block; }
.tag-cloud { display: flex; flex-wrap: wrap; gap: 5px; max-height: 300px; overflow-y: auto; }
.tag { padding: 4px 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: #787878; font-size: 10px; }
.tag.count { color: #60a5fa; border-color: rgba(96,165,250,0.3); background: rgba(96,165,250,0.05); }
.tag.clothing { color: #a78bfa; border-color: rgba(167,139,250,0.3); background: rgba(167,139,250,0.05); }
.tag.body { color: #fb923c; border-color: rgba(251,146,60,0.3); background: rgba(251,146,60,0.05); }
.tag.nsfw { color: #f87171; border-color: rgba(248,113,113,0.3); background: rgba(248,113,113,0.05); }
.tag.action { color: #4ade80; border-color: rgba(74,222,128,0.3); background: rgba(74,222,128,0.05); }
.tag.expression { color: #fbbf24; border-color: rgba(251,191,36,0.3); background: rgba(251,191,36,0.05); }
.tag.bg { color: #38bdf8; border-color: rgba(56,189,248,0.3); background: rgba(56,189,248,0.05); }
.tag.objects { color: #94a3b8; border-color: rgba(148,163,184,0.3); background: rgba(148,163,184,0.05); }
.tag.effect { color: #c084fc; border-color: rgba(192,132,252,0.3); background: rgba(192,132,252,0.05); }
.tag.color-tag { color: #f472b6; border-color: rgba(244,114,182,0.3); background: rgba(244,114,182,0.05); }
.tag.trait { color: #34d399; border-color: rgba(52,211,153,0.3); background: rgba(52,211,153,0.05); }
.tag.composition { color: #818cf8; border-color: rgba(129,140,248,0.3); background: rgba(129,140,248,0.05); }
.tag.style { color: #e879f9; border-color: rgba(232,121,249,0.3); background: rgba(232,121,249,0.05); }
.tag.char { color: #2dd4bf; border-color: rgba(45,212,191,0.3); background: rgba(45,212,191,0.05); }
.tag.copy { color: #22d3ee; border-color: rgba(34,211,238,0.3); background: rgba(34,211,238,0.05); }
.tag.artist-tag { color: #facc15; border-color: rgba(250,204,21,0.3); background: rgba(250,204,21,0.05); }

.tag-legend { display: flex; gap: 6px; flex-wrap: wrap; }
.legend-item { font-size: 8px; font-weight: 800; padding: 2px 6px; border-radius: 3px; }
.legend-item.count { color: #60a5fa; background: rgba(96,165,250,0.1); }
.legend-item.clothing { color: #a78bfa; background: rgba(167,139,250,0.1); }
.legend-item.body { color: #fb923c; background: rgba(251,146,60,0.1); }
.legend-item.nsfw { color: #f87171; background: rgba(248,113,113,0.1); }
.legend-item.action { color: #4ade80; background: rgba(74,222,128,0.1); }
.legend-item.expression { color: #fbbf24; background: rgba(251,191,36,0.1); }
.legend-item.bg { color: #38bdf8; background: rgba(56,189,248,0.1); }
.legend-item.objects { color: #94a3b8; background: rgba(148,163,184,0.1); }
.legend-item.effect { color: #c084fc; background: rgba(192,132,252,0.1); }
.legend-item.color-tag { color: #f472b6; background: rgba(244,114,182,0.1); }
.legend-item.trait { color: #34d399; background: rgba(52,211,153,0.1); }

.detail-actions { display: flex; gap: 10px; }
.primary-btn { flex: 2; height: 48px; background: var(--accent); border: none; border-radius: var(--radius-pill); color: #000; font-weight: 900; font-size: 13px; letter-spacing: 1px; cursor: pointer; }
.secondary-btn { flex: 1; height: 48px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-pill); color: var(--text-secondary); font-weight: 800; font-size: 11px; cursor: pointer; }
.primary-btn:hover, .secondary-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.3); }

/* ═══ List View ═══ */
.list-view { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.list-header { display: flex; padding: 6px 16px; background: var(--bg-secondary); border-bottom: 1px solid var(--border); font-size: 9px; font-weight: 900; color: var(--text-muted); letter-spacing: 1px; }
.lh.idx { width: 40px; } .lh.char { width: 160px; } .lh.copy { width: 120px; } .lh.artist { width: 100px; } .lh.tags { flex: 1; } .lh.act { width: 44px; }
.list-scroll { flex: 1; overflow-y: auto; }
.list-row { display: flex; align-items: center; padding: 6px 16px; font-size: 11px; border-bottom: 1px solid rgba(255,255,255,0.02); cursor: pointer; }
.list-row:hover { background: rgba(255,255,255,0.02); }
.list-row.active { background: var(--accent-dim); }
.lr.idx { width: 40px; color: var(--text-muted); font-family: monospace; font-size: 10px; }
.lr.char { width: 160px; color: #4ade80; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.copy { width: 120px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.artist { width: 100px; color: var(--accent); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lr.tags { flex: 1; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 10px; }
.lr.act { width: 44px; text-align: center; }
.use-btn { padding: 2px 8px; background: var(--accent); border: none; border-radius: 3px; color: #000; font-size: 9px; font-weight: 800; cursor: pointer; }
.list-pager { display: flex; align-items: center; justify-content: center; gap: 12px; padding: 6px; border-top: 1px solid var(--border); }
.pager-info { font-size: 11px; color: var(--text-muted); font-family: monospace; }

label.danger { color: #f87171; }

/* 조건부 프롬프트 */
.cond-section { padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; display: flex; flex-direction: column; gap: 8px; }
.cond-card { background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
.cond-card.neg { border-color: rgba(248,113,113,0.15); }
.cond-title { font-size: 10px; font-weight: 900; letter-spacing: 1px; cursor: pointer; list-style: none; }
.cond-title::-webkit-details-marker { display: none; }
.cond-title.positive { color: #4ade80; }
.cond-title.negative { color: #f87171; }
.cond-desc { font-size: 9px; color: var(--text-muted); margin: 4px 0 8px; }
.cond-rule { display: flex; align-items: center; gap: 4px; margin-bottom: 4px; flex-wrap: wrap; }
.cond-rule-block { border: 1px solid var(--border); border-radius: 6px; padding: 6px; margin-bottom: 4px; }
.cond-row1, .cond-row2 { display: flex; align-items: center; gap: 4px; }
.cond-row2 { margin-top: 3px; }
.cond-check input { accent-color: var(--accent); }
.cond-kw { font-size: 9px; font-weight: 900; color: var(--accent); }
.cond-input { flex: 1; min-width: 80px; padding: 4px 8px; font-size: 11px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; color: var(--text-primary); }
.cond-input.neg { border-color: rgba(248,113,113,0.2); }
.cond-sel { padding: 3px 4px; font-size: 9px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; color: var(--text-secondary); }
.cond-sel.sm { width: 55px; }
.cond-rm { background: none; border: none; color: #f87171; cursor: pointer; font-size: 12px; padding: 0 2px; }
.cond-save-btn { width: 100%; padding: 8px; background: var(--accent); border: none; border-radius: 8px; color: #000; font-size: 10px; font-weight: 800; cursor: pointer; }
.cond-add { width: 100%; padding: 5px; background: var(--bg-button); border: 1px dashed var(--border); border-radius: 4px; color: var(--text-muted); font-size: 9px; font-weight: 700; cursor: pointer; margin-top: 4px; }
</style>
