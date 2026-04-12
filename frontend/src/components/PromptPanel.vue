<template>
  <div class="prompt-panel">
    <!-- 1. FINAL OUTPUT PROMPT -->
    <div class="glass-card highlight">
      <div class="card-header">
        FINAL OUTPUT PROMPT
        <span class="token-info">{{ tokenCount }} tokens</span>
      </div>
      <TagBlockField v-if="tagBlockMode" :model-value="widgets.total_prompt_display" :color-fn="blockColorClass" placeholder=""
        @update:model-value="onTotalBlockChange" @open-wildcard="(n) => emit('open-wildcard', n)" />
      <textarea v-else ref="totalPromptRef" v-model="widgets.total_prompt_display"
        class="total-prompt auto-grow" placeholder="최종 프롬프트" @input="autoGrow($event.target)"></textarea>
      <div class="prompt-actions">
        <button class="optimize-btn" @click="optimizePrompt">🧹 OPTIMIZE</button>
        <span class="opt-result" v-if="optResult">{{ optResult }}</span>
      </div>
      <div class="conflicts" v-if="promptConflicts.length > 0">
        <div v-for="c in promptConflicts" :key="c.group" class="conflict-item">⚠ {{ c.group }}: {{ c.tags.join(', ') }}</div>
      </div>
      <details class="neg-section">
        <summary class="danger-label neg-toggle">
          NEGATIVE ▾
          <button class="ai-btn neg-ai" @click.prevent.stop="runSmartNegative()" :disabled="ollamaLoading" title="AI 네거티브 자동 생성">🤖</button>
        </summary>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.neg_prompt_text" :color-fn="() => 'neg'" class="neg" placeholder="네거티브 추가..." />
        <textarea v-else ref="negRef" v-model="widgets.neg_prompt_text" class="neg-prompt auto-grow" placeholder="Negative prompt..." @input="autoGrow($event.target)"></textarea>
      </details>
    </div>

    <!-- 2. CHARACTER & MODEL -->
    <div class="glass-card">
      <div class="card-header">CHARACTER & MODEL</div>
      <div class="input-group">
        <label>Char Count</label>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.char_count_input" :color-fn="() => 'bc-count'" placeholder="인물수..." />
        <input v-else type="text" v-model="widgets.char_count_input" placeholder="e.g. 1girl, 2girls..." />
      </div>
      <div class="input-group autocomplete-wrap">
        <div class="row label-row">
          <label>Character</label>
          <button class="small-btn" @click="requestAction('open_character_preset'); loadCharTags()">PRESET</button>
        </div>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.character_input" :color-fn="() => 'bc-count'" placeholder="캐릭터..." @open-wildcard="(n) => emit('open-wildcard', n)" />
        <input v-else type="text" v-model="widgets.character_input" placeholder="e.g. hatsune miku"
          @input="onFieldInput($event, 'character_input')" @keydown="onFieldKey($event, 'character_input')" @blur="loadCharTags" />
        <div class="char-insight" v-if="charInsight.tags.length > 0">
          <div class="insight-header">
            <span class="insight-label">📖 Official Tags</span>
            <button class="insight-apply" @click="applyOfficialTags">APPLY ALL</button>
          </div>
          <div class="insight-tags">
            <button v-for="tag in charInsight.tags" :key="tag" class="char-tag-chip" @click="insertCharTag(tag)">{{ tag.replace(/_/g, ' ') }}</button>
          </div>
        </div>
        <div class="ac-popup" v-if="!tagBlockMode && fieldAcTarget === 'character_input' && acItems.length > 0">
          <div v-for="(tag, i) in acItems" :key="tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptFieldSuggestion(tag, 'character_input')">{{ tag }}</div>
        </div>
      </div>
      <div class="input-group autocomplete-wrap">
        <label>Copyright</label>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.copyright_input" :color-fn="() => ''" placeholder="작품..." />
        <input v-else type="text" v-model="widgets.copyright_input" placeholder="Copyright / Series..."
          @input="onFieldInput($event, 'copyright_input')" @keydown="onFieldKey($event, 'copyright_input')" />
        <div class="ac-popup" v-if="!tagBlockMode && fieldAcTarget === 'copyright_input' && acItems.length > 0">
          <div v-for="(tag, i) in acItems" :key="tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptFieldSuggestion(tag, 'copyright_input')">{{ tag }}</div>
        </div>
      </div>
      <div class="input-group">
        <div class="row label-row">
          <label>Artist</label>
          <button class="lock-btn" :class="{ locked: artistLocked }" @click="toggleArtistLock">{{ artistLocked ? '🔒' : '🔓' }}</button>
        </div>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.artist_input" :color-fn="() => ''" placeholder="작가..." />
        <textarea v-else ref="artistRef" v-model="widgets.artist_input" class="auto-grow" placeholder="Artist tags..." @input="autoGrow($event.target)"></textarea>
      </div>
      <div class="input-group">
        <label>Checkpoint</label>
        <CustomSelect v-model="widgets.model_combo" :options="modelItems" placeholder="Select model..." />
      </div>
    </div>

    <!-- 3. PROMPT BLOCKS -->
    <details class="glass-card" open>
      <summary class="card-header">PROMPT BLOCKS</summary>
      <div class="input-group autocomplete-wrap">
        <div class="row label-row">
          <label>Main Tags</label>
          <div class="ai-btns">
            <button class="ai-btn" @click="ollamaMode = 'expand'; runOllama()" :disabled="ollamaLoading" title="태그 확장">✨</button>
            <button class="ai-btn" @click="showNlInput = !showNlInput" title="자연어→태그">💬</button>
            <button class="ai-btn" @click="ollamaMode = 'suggest'; runOllama()" :disabled="ollamaLoading" title="유사 태그">🔄</button>
          </div>
        </div>
        <div class="nl-input-row" v-if="showNlInput">
          <input v-model="nlPrompt" placeholder="이미지를 자연어로 설명하세요..." @keydown.enter="ollamaMode = 'nl2tags'; runOllama()" class="nl-input" />
          <button class="ai-btn go" @click="ollamaMode = 'nl2tags'; runOllama()" :disabled="ollamaLoading">GO</button>
        </div>
        <div class="ai-loading" v-if="ollamaLoading">🤖 AI 처리 중...</div>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.main_prompt_text" :color-fn="blockColorClass" placeholder="태그 추가..." @open-wildcard="(n) => emit('open-wildcard', n)" />
        <textarea v-else ref="mainRef" v-model="widgets.main_prompt_text" class="auto-grow" placeholder="메인 태그..."
          @input="onMainInput($event)" @keydown="onAutoKey($event)" rows="3"></textarea>
        <div class="ac-popup" v-if="!tagBlockMode && fieldAcTarget === 'main_prompt_text' && acItems.length > 0">
          <div v-for="(tag, i) in acItems" :key="tag" class="ac-item" :class="{ selected: acIdx === i }" @mousedown.prevent="acceptSuggestion(tag)">{{ tag }}</div>
        </div>
      </div>
      <div class="input-group">
        <label>Prefix</label>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.prefix_prompt_text" :color-fn="blockColorClass" placeholder="선행 추가..." @open-wildcard="(n) => emit('open-wildcard', n)" />
        <textarea v-else ref="prefixRef" v-model="widgets.prefix_prompt_text" class="auto-grow" placeholder="선행..." @input="autoGrow($event.target)"></textarea>
      </div>
      <div class="input-group">
        <label>Suffix</label>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.suffix_prompt_text" :color-fn="blockColorClass" placeholder="후행 추가..." @open-wildcard="(n) => emit('open-wildcard', n)" />
        <textarea v-else ref="suffixRef" v-model="widgets.suffix_prompt_text" class="auto-grow" placeholder="후행..." @input="autoGrow($event.target)"></textarea>
      </div>
      <details class="input-group exclude-section">
        <summary class="exclude-toggle">EXCLUDE (LOCAL) ▾ <button class="excl-mgr-btn" @click.prevent.stop="showExcludeManager = true">🔍 MANAGER</button></summary>
        <div class="exclude-help">
          <span>단어 → 포함하는 모든 태그 제외 (short → short hair, very short hair)</span>
          <span>*단어 → 완전 일치만 제외 (*blue hair → blue hair만)</span>
          <span>_단어 → 앞에 붙는 태그 제외 (_short → very short, too short)</span>
          <span>단어_ → 뒤에 붙는 태그 제외 (short_ → short hair, short pants)</span>
          <span>_단어_ → 포함 (단어와 동일, 명시적)</span>
          <span>~단어 → 예외 완전일치 유지</span>
          <span>~_단어 → 예외 접미 유지 (~_tank top → tank top 유지)</span>
          <span>~단어_ → 예외 접두 유지 (~tank_ → tank top 유지)</span>
          <span>~_단어_ → 예외 포함 유지 (~_tank top_ → blue tank top 등 유지)</span>
        </div>
        <TagBlockField v-if="tagBlockMode" v-model="widgets.exclude_prompt_local_input" :color-fn="excludeColorFn" placeholder="제외 규칙 추가..." />
        <textarea v-else v-model="widgets.exclude_prompt_local_input" class="auto-grow exclude-textarea" placeholder="제외 규칙 (쉼표 구분)..." rows="2"></textarea>
      </details>
    </details>

    <!-- Exclude Manager Modal -->
    <transition name="fade">
      <div v-if="showExcludeManager" class="em-overlay" @click.self="showExcludeManager = false">
        <div class="em-modal">
          <div class="em-header">
            <h3>EXCLUDE MANAGER</h3>
            <span class="em-desc">제외 규칙별 매칭 태그 미리보기</span>
            <button class="close-btn" @click="showExcludeManager = false">✕</button>
          </div>
          <div class="em-body">
            <!-- 좌측: 규칙 목록 + 추가 -->
            <div class="em-rules">
              <div v-for="(rule, ri) in excludeRules" :key="ri" class="em-rule-item"
                :class="[excludeColorFn(rule), { active: selectedExRule === ri }]"
                @click="selectedExRule = ri; loadExcludeMatches(rule)">
                <!-- 편집 모드 -->
                <input v-if="editingExRule === ri" class="em-rule-edit" v-model="editExRuleText"
                  @blur="finishEditExRule(ri)" @keydown.enter="finishEditExRule(ri)" @keydown.escape="editingExRule = -1"
                  @click.stop ref="exRuleEditRef" />
                <!-- 표시 모드 -->
                <span v-else class="em-rule-text" @dblclick.stop="startEditExRule(ri)">{{ rule }}</span>
                <span class="em-match-count">{{ excludeMatches[ri]?.length || '...' }}</span>
                <button class="em-rule-rm" @click.stop="removeExcludeRule(ri)">✕</button>
              </div>
              <div v-if="excludeRules.length === 0" class="em-empty-sm">제외 규칙 없음</div>
              <div class="em-add-row">
                <input v-model="newExcludeRule" placeholder="규칙 추가..." class="em-add-input" @keydown.enter="addExcludeRule" />
                <button class="em-add-btn" @click="addExcludeRule">+</button>
              </div>
            </div>
            <!-- 우측: 매칭 태그 목록 -->
            <div class="em-matches">
              <template v-if="selectedExRule >= 0 && currentExMatches.length > 0">
                <div class="em-match-header">
                  "{{ excludeRules[selectedExRule] }}" — {{ currentExMatches.length }}개 매칭
                </div>
                <div class="em-match-list">
                  <button v-for="tag in currentExMatches" :key="tag" class="em-tag"
                    :class="{ excepted: isExcepted(tag) }"
                    @click="toggleException(tag)"
                    @contextmenu.prevent="addExactExclude(tag)">
                    {{ tag.replace(/_/g, ' ') }}
                  </button>
                </div>
              </template>
              <div v-else class="em-empty">좌측에서 규칙을 선택하세요</div>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useWidgetStore, requestAction } from '../stores/widgetStore.js'
import { getBackend, onBackendEvent } from '../bridge.js'
import CustomSelect from './CustomSelect.vue'
import TagBlockField from './TagBlockField.vue'

const emit = defineEmits(['toggle-extend', 'open-wildcard'])
const store = useWidgetStore()
const widgets = store.widgets

// 블록 모드 (Settings에서 제어)
const tagBlockMode = ref(window.localStorage.getItem('tagBlockMode') === 'true')
watch(tagBlockMode, v => window.localStorage.setItem('tagBlockMode', String(v)))

// 같은 SPA 내 변경 감지 (keep-alive 환경)
let _blockModeTimer = null
onMounted(() => {
  _blockModeTimer = setInterval(() => {
    const stored = window.localStorage.getItem('tagBlockMode') === 'true'
    if (stored !== tagBlockMode.value) {
      tagBlockMode.value = stored
      console.log('[PromptPanel] Block mode synced:', stored)
    }
  }, 300)
})
onUnmounted(() => { if (_blockModeTimer) clearInterval(_blockModeTimer) })

const artistLocked = computed({
  get: () => widgets.btn_lock_artist === 'true',
  set: (v) => { widgets.btn_lock_artist = v ? 'true' : 'false' }
})
function toggleArtistLock() { artistLocked.value = !artistLocked.value; requestAction('set_artist_locked', { locked: artistLocked.value }) }

const totalPromptRef = ref(null)
const negRef = ref(null)
const artistRef = ref(null)
const prefixRef = ref(null)
const mainRef = ref(null)
const suffixRef = ref(null)

const modelItems = computed(() => store.getProperty('model_combo', 'items') || [])
const tokenCount = computed(() => { const t = widgets.total_prompt_display; return t ? t.split(',').filter(s => s.trim()).length : 0 })

// 블록 색상 분류
const countPattern = /^(\d+)?(girl|boy|other)s?$|^solo$|^multiple_/
const blockColorCache = ref({})

// ── Exclude Manager ──
const showExcludeManager = ref(false)
const selectedExRule = ref(-1)
const excludeMatches = ref({})  // {ruleIdx: [tags]}

const excludeRules = computed(() => {
  const text = widgets.exclude_prompt_local_input || ''
  return text.split(',').map(t => t.trim()).filter(Boolean)
})

const currentExMatches = computed(() => {
  return excludeMatches.value[selectedExRule.value] || []
})

async function loadExcludeMatches(rule) {
  const backend = await getBackend()
  if (!backend.getExcludeMatches) return
  backend.getExcludeMatches(rule, (json) => {
    try {
      const tags = JSON.parse(json)
      if (Array.isArray(tags)) {
        excludeMatches.value = { ...excludeMatches.value, [selectedExRule.value]: tags }
      }
    } catch {}
  })
}

const newExcludeRule = ref('')
const editingExRule = ref(-1)
const editExRuleText = ref('')
const exRuleEditRef = ref(null)

function startEditExRule(idx) {
  editingExRule.value = idx
  editExRuleText.value = excludeRules.value[idx]
  nextTick(() => { if (exRuleEditRef.value?.[0]) exRuleEditRef.value[0].focus() })
}

function finishEditExRule(idx) {
  if (editingExRule.value !== idx) return
  const newText = editExRuleText.value.trim()
  editingExRule.value = -1
  if (!newText) { removeExcludeRule(idx); return }
  const rules = excludeRules.value.slice()
  rules[idx] = newText
  widgets.exclude_prompt_local_input = rules.join(', ')
  // 매칭 갱신
  selectedExRule.value = idx
  loadExcludeMatches(newText)
}

function addExcludeRule() {
  const rule = newExcludeRule.value.trim()
  if (!rule) return
  const cur = widgets.exclude_prompt_local_input || ''
  widgets.exclude_prompt_local_input = cur ? cur + ', ' + rule : rule
  newExcludeRule.value = ''
}

function removeExcludeRule(idx) {
  const rules = excludeRules.value.slice()
  rules.splice(idx, 1)
  widgets.exclude_prompt_local_input = rules.join(', ')
  if (selectedExRule.value >= rules.length) selectedExRule.value = -1
}

function isExcepted(tag) {
  const cur = (widgets.exclude_prompt_local_input || '').toLowerCase()
  return cur.includes('~' + tag.toLowerCase())
}

function addExactExclude(tag) {
  // 우클릭: *완전일치 제외 규칙 추가
  const cur = widgets.exclude_prompt_local_input || ''
  const rule = '*' + tag
  if (!cur.includes(rule)) {
    widgets.exclude_prompt_local_input = cur ? cur + ', ' + rule : rule
  }
}

function toggleException(tag) {
  const cur = widgets.exclude_prompt_local_input || ''
  const exception = '~' + tag
  if (isExcepted(tag)) {
    // 예외 해제 — ~tag 제거
    const rules = cur.split(',').map(r => r.trim()).filter(r => r.toLowerCase() !== exception.toLowerCase())
    widgets.exclude_prompt_local_input = rules.join(', ')
  } else {
    // 예외 추가
    widgets.exclude_prompt_local_input = cur ? cur + ', ' + exception : exception
  }
}

// 최종 프롬프트 블록 변경 시 → 원본 필드에서 태그 제거
function onTotalBlockChange(newVal) {
  const newTags = new Set(newVal.split(',').map(t => t.trim().toLowerCase()).filter(Boolean))
  // 각 필드에서 없어진 태그 제거
  for (const key of ['main_prompt_text', 'prefix_prompt_text', 'suffix_prompt_text']) {
    const cur = widgets[key] || ''
    const filtered = cur.split(',').map(t => t.trim()).filter(t => {
      return t && newTags.has(t.toLowerCase())
    })
    const result = filtered.join(', ')
    if (result !== cur.trim()) widgets[key] = result
  }
}

function excludeColorFn(text) {
  const t = text.trim()
  if (t.startsWith('~')) return 'bc-action'       // 예외 계열 (초록)
  if (t.startsWith('*')) return 'bc-expression'   // 완전 일치 (노랑)
  if (t.startsWith('_') && t.endsWith('_')) return 'bc-nsfw'
  if (t.startsWith('_')) return 'bc-body'         // 끝 매치 (주황)
  if (t.endsWith('_')) return 'bc-clothing'       // 시작 매치 (보라)
  return 'bc-nsfw'
}

function blockColorClass(text) {
  if (text.includes('__') && /__(.+?)__/.test(text)) return 'wc-block'
  let t = text.trim().toLowerCase().replace(/ /g, '_').replace(/^\(+/, '').replace(/[\):.\d]+$/, '').trim()
  if (countPattern.test(t)) return 'bc-count'
  return blockColorCache.value[t] ? 'bc-' + blockColorCache.value[t] : ''
}

// 태그 분류 요청
async function classifyVisibleTags() {
  const allTags = new Set()
  for (const key of ['main_prompt_text', 'prefix_prompt_text', 'suffix_prompt_text', 'total_prompt_display']) {
    const text = widgets[key] || ''
    for (const t of text.split(',')) {
      let tag = t.trim().replace(/ /g, '_').replace(/^\(+/, '').replace(/[\):.\d]+$/, '').trim()
      if (tag && !countPattern.test(tag.toLowerCase()) && !blockColorCache.value[tag.toLowerCase()] && !/__(.+?)__/.test(tag)) allTags.add(tag)
    }
  }
  if (allTags.size === 0) return
  const backend = await getBackend()
  if (backend.classifyTags) {
    backend.classifyTags(JSON.stringify([...allTags]), (json) => {
      try {
        const r = JSON.parse(json)
        if (!r.error) {
          const m = { sexual:'nsfw', body_parts:'body', clothing:'clothing', pose:'action', expression:'expression', background:'bg', composition:'bg', effect:'effect', objects:'objects', color:'color', character_trait:'trait' }
          for (const [tag, cat] of Object.entries(r)) blockColorCache.value[tag.toLowerCase()] = m[cat] || ''
        }
      } catch {}
    })
  }
}
watch(tagBlockMode, v => { if (v) setTimeout(classifyVisibleTags, 200) })

// 딥 프롬프트 클리너
const optResult = ref('')
const promptConflicts = ref([])
async function optimizePrompt() {
  const backend = await getBackend()
  if (!backend.deepCleanPrompt) return
  backend.deepCleanPrompt(JSON.stringify({ prompt: widgets.total_prompt_display || '' }), (json) => {
    try {
      const d = JSON.parse(json)
      if (d.error) { optResult.value = d.error; return }
      if (d.optimized) widgets.main_prompt_text = d.optimized
      promptConflicts.value = d.conflicts || []
      optResult.value = `${d.removed}개 중복 제거, ${d.tag_count}개 태그`
      setTimeout(() => { optResult.value = '' }, 5000)
    } catch {}
  })
}

// 캐릭터 인사이트
const charInsight = ref({ tags: [], raw: '' })
async function loadCharTags() {
  const char = widgets.character_input
  if (!char || char.length < 2) { charInsight.value = { tags: [], raw: '' }; return }
  const backend = await getBackend()
  if (backend.getCharacterInsight) {
    backend.getCharacterInsight(char, (json) => {
      try { const d = JSON.parse(json); charInsight.value = { tags: d.tags || [], raw: d.raw || '' } } catch { charInsight.value = { tags: [], raw: '' } }
    })
  }
}
function insertCharTag(tag) {
  const cur = widgets.main_prompt_text || ''
  if (!cur.toLowerCase().includes(tag.toLowerCase())) widgets.main_prompt_text = cur ? cur.replace(/,?\s*$/, '') + ', ' + tag + ', ' : tag + ', '
}
function applyOfficialTags() { if (charInsight.value.raw) { widgets.main_prompt_text = charInsight.value.raw; nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) }) } }

// Ollama
const ollamaLoading = ref(false)
const ollamaMode = ref('expand')
const showNlInput = ref(false)
const nlPrompt = ref('')
let ollamaTimer = null
async function runOllama() {
  const tags = widgets.main_prompt_text || ''
  if (!tags.trim() && ollamaMode.value !== 'nl2tags') {
    requestAction('show_toast', { type: 'info', msg: '프롬프트를 먼저 입력하세요' })
    return
  }
  if (ollamaMode.value === 'nl2tags' && !nlPrompt.value.trim()) {
    requestAction('show_toast', { type: 'info', msg: '자연어 설명을 입력하세요' })
    return
  }
  ollamaLoading.value = true
  // 60초 타임아웃 안전장치
  clearTimeout(ollamaTimer)
  ollamaTimer = setTimeout(() => {
    if (ollamaLoading.value) {
      ollamaLoading.value = false
      requestAction('show_toast', { type: 'error', msg: 'AI 응답 시간 초과 (60초) — Ollama 서버 상태를 확인하세요' })
    }
  }, 65000)
  const backend = await getBackend()
  if (!backend.ollamaEnhance) { ollamaLoading.value = false; clearTimeout(ollamaTimer); return }
  const url = window.localStorage.getItem('ollamaUrl') || 'http://localhost:11434'
  const model = window.localStorage.getItem('ollamaModel') || 'gemma3:4b'
  backend.ollamaEnhance(tags, ollamaMode.value, JSON.stringify({ prompt: nlPrompt.value, url, model }))
}

async function runSmartNegative() {
  const positivePrompt = widgets.total_prompt_display || widgets.main_prompt_text || ''
  if (!positivePrompt.trim()) {
    requestAction('show_toast', { type: 'info', msg: '포지티브 프롬프트를 먼저 입력하세요' })
    return
  }
  ollamaLoading.value = true
  clearTimeout(ollamaTimer)
  ollamaTimer = setTimeout(() => {
    if (ollamaLoading.value) {
      ollamaLoading.value = false
      requestAction('show_toast', { type: 'error', msg: 'AI 응답 시간 초과' })
    }
  }, 65000)
  const backend = await getBackend()
  if (!backend.ollamaEnhance) { ollamaLoading.value = false; clearTimeout(ollamaTimer); return }
  const url = window.localStorage.getItem('ollamaUrl') || 'http://localhost:11434'
  const model = window.localStorage.getItem('ollamaModel') || 'gemma3:4b'
  backend.ollamaEnhance(positivePrompt, 'negative', JSON.stringify({ url, model }))
}

// 자동완성
const acItems = ref([])
const acIdx = ref(0)
const fieldAcTarget = ref('')
let acTimer = null

function onFieldInput(e, fieldId) {
  fieldAcTarget.value = fieldId
  const text = e.target.value; const lastComma = text.lastIndexOf(',')
  const prefix = (lastComma >= 0 ? text.substring(lastComma + 1) : text).trim()
  if (prefix.length < 2) { acItems.value = []; return }
  clearTimeout(acTimer)
  acTimer = setTimeout(async () => {
    const backend = await getBackend()
    if (backend.getTagSuggestions) backend.getTagSuggestions(prefix, (json) => { try { acItems.value = JSON.parse(json).slice(0, 10); acIdx.value = 0 } catch { acItems.value = [] } })
  }, 300)
}
function onFieldKey(e, fieldId) {
  if (fieldAcTarget.value !== fieldId || !acItems.value.length) return
  if (e.key === 'ArrowDown') { e.preventDefault(); acIdx.value = Math.min(acIdx.value + 1, acItems.value.length - 1) }
  else if (e.key === 'ArrowUp') { e.preventDefault(); acIdx.value = Math.max(0, acIdx.value - 1) }
  else if (e.key === 'Tab' || e.key === 'Enter') { e.preventDefault(); acceptFieldSuggestion(acItems.value[acIdx.value], fieldId) }
  else if (e.key === 'Escape') { acItems.value = []; fieldAcTarget.value = '' }
}
function acceptFieldSuggestion(tag, fieldId) {
  const text = widgets[fieldId] || ''; const lastComma = text.lastIndexOf(',')
  widgets[fieldId] = (lastComma >= 0 ? text.substring(0, lastComma + 1) + ' ' : '') + tag + ', '
  acItems.value = []; fieldAcTarget.value = ''
}
function onMainInput(e) { autoGrow(e.target); fieldAcTarget.value = 'main_prompt_text'; onFieldInput(e, 'main_prompt_text') }
function onAutoKey(e) { onFieldKey(e, 'main_prompt_text') }
function acceptSuggestion(tag) { acceptFieldSuggestion(tag, 'main_prompt_text'); nextTick(() => { if (mainRef.value) { mainRef.value.focus(); autoGrow(mainRef.value) } }) }

function autoGrow(el) { if (!el) return; el.style.height = 'auto'; el.style.height = el.scrollHeight + 'px' }
function growAll() { nextTick(() => { ;[totalPromptRef, negRef, artistRef, prefixRef, mainRef, suffixRef].forEach(r => { if (r.value) autoGrow(r.value) }) }) }

onMounted(() => {
  setTimeout(growAll, 500); setTimeout(growAll, 1500)
  // UI prefs 로드 (재시작 시 블록 모드 복원)
  onBackendEvent('uiPrefsLoaded', (json) => {
    try {
      const prefs = JSON.parse(json)
      if (typeof prefs.tagBlockMode === 'boolean') {
        window.localStorage.setItem('tagBlockMode', String(prefs.tagBlockMode))
        tagBlockMode.value = prefs.tagBlockMode
      }
      // Hires/ADetailer/NegPiP 복원
      if (typeof prefs.hires_enabled === 'boolean') widgets.hires_options_group = prefs.hires_enabled ? 'true' : 'false'
      if (typeof prefs.ad_enabled === 'boolean') widgets.adetailer_group = prefs.ad_enabled ? 'true' : 'false'
      if (typeof prefs.ad_s1_enabled === 'boolean') widgets.ad_slot1_group = prefs.ad_s1_enabled ? 'true' : 'false'
      if (typeof prefs.ad_s2_enabled === 'boolean') widgets.ad_slot2_group = prefs.ad_s2_enabled ? 'true' : 'false'
      if (typeof prefs.negpip_enabled === 'boolean') widgets.negpip_group = prefs.negpip_enabled ? 'true' : 'false'
      if (typeof prefs.galleryShowMetadata === 'boolean') window.localStorage.setItem('galleryShowMetadata', String(prefs.galleryShowMetadata))
    } catch {}
  })
  onBackendEvent('ollamaResult', (json) => {
    ollamaLoading.value = false
    clearTimeout(ollamaTimer)
    try {
      const d = JSON.parse(json)
      if (d.error) {
        const msg = d.error.includes('연결') ? 'Ollama 서버에 연결할 수 없습니다' :
                    d.error.includes('시간') || d.error.includes('Timeout') ? 'AI 응답 시간 초과' :
                    `AI 오류: ${d.error}`
        requestAction('show_toast', { type: 'error', msg })
        return
      }
      if (d.tags) {
        if (d.mode === 'negative') {
          const existing = (widgets.neg_prompt_text || '').trim()
          widgets.neg_prompt_text = existing ? existing.replace(/,?\s*$/, '') + ', ' + d.tags : d.tags
          requestAction('show_toast', { type: 'success', msg: 'AI 네거티브 생성 완료' })
        } else {
          widgets.main_prompt_text = d.tags
          showNlInput.value = false
          nlPrompt.value = ''
          requestAction('show_toast', { type: 'success', msg: `AI ${d.mode === 'nl2tags' ? '변환' : d.mode === 'suggest' ? '추천' : '확장'} 완료` })
        }
      }
    } catch {}
  })
})

watch(() => widgets.total_prompt_display, () => nextTick(() => { if (totalPromptRef.value) autoGrow(totalPromptRef.value) }))
watch(() => widgets.neg_prompt_text, () => nextTick(() => { if (negRef.value) autoGrow(negRef.value) }))
watch(() => widgets.artist_input, () => nextTick(() => { if (artistRef.value) autoGrow(artistRef.value) }))
watch(() => widgets.main_prompt_text, () => { nextTick(() => { if (mainRef.value) autoGrow(mainRef.value) }); if (tagBlockMode.value) classifyVisibleTags() })
</script>

<style scoped>
.prompt-panel { display: flex; flex-direction: column; gap: 12px; }
.glass-card { background: rgba(20, 20, 20, 0.6); border: 1px solid var(--border); border-radius: var(--radius-card); padding: 14px; }
.glass-card.highlight { border-color: var(--accent-dim); background: rgba(250, 204, 21, 0.02); }
.glass-card:hover { border-color: #333; }
.card-header { font-size: 10px; font-weight: 800; color: var(--text-muted); letter-spacing: 1.5px; margin-bottom: 12px; cursor: pointer; display: flex; align-items: center; justify-content: space-between; }
summary { list-style: none; outline: none; }
summary::-webkit-details-marker { display: none; }
.input-group { margin-bottom: 10px; }
.row { display: flex; gap: 6px; }
.label-row { align-items: center; margin-bottom: 4px; }
.small-btn { padding: 0 12px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: 10px; font-weight: 700; cursor: pointer; white-space: nowrap; }
.lock-btn { background: none; border: none; cursor: pointer; font-size: 14px; padding: 0 4px; opacity: 0.5; }
.lock-btn.locked { opacity: 1; }
.total-prompt { min-height: 60px; font-family: 'Consolas', monospace; font-size: 12px; line-height: 1.5; color: var(--accent); border-color: var(--accent-dim); }
.neg-section { margin-top: 8px; }
.neg-toggle { cursor: pointer; list-style: none; display: flex; align-items: center; justify-content: space-between; }
.neg-toggle::-webkit-details-marker { display: none; }
.neg-ai { font-size: 12px !important; padding: 2px 6px !important; opacity: 0.6; }
.neg-ai:hover { opacity: 1; }
.danger-label { color: #f87171; font-size: 9px; font-weight: 800; letter-spacing: 1px; }
.neg-prompt { min-height: 30px; color: #f87171; border-color: rgba(248,113,113,0.2); }
.auto-grow { resize: none; overflow: hidden; min-height: 32px; }
.token-info { font-size: 9px; color: var(--text-muted); }
.prompt-actions { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.optimize-btn { padding: 3px 10px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-secondary); font-size: 9px; font-weight: 700; cursor: pointer; }
.optimize-btn:hover { border-color: var(--accent); color: var(--accent); }
.opt-result { font-size: 9px; color: #4ade80; }
.conflicts { margin-top: 4px; }
.conflict-item { font-size: 9px; color: #fbbf24; padding: 2px 0; }
.char-insight { margin-top: 6px; background: rgba(45,212,191,0.03); border: 1px solid rgba(45,212,191,0.1); border-radius: 6px; padding: 8px; }
.insight-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.insight-label { font-size: 9px; font-weight: 800; color: #2dd4bf; }
.insight-apply { padding: 2px 8px; background: #2dd4bf; border: none; border-radius: 3px; color: #000; font-size: 9px; font-weight: 800; cursor: pointer; }
.insight-tags { display: flex; flex-wrap: wrap; gap: 3px; max-height: 80px; overflow-y: auto; }
.char-tag-chip { padding: 2px 8px; background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.2); border-radius: 4px; color: #2dd4bf; font-size: 9px; cursor: pointer; }
.char-tag-chip:hover { background: rgba(45,212,191,0.15); border-color: #2dd4bf; }
.ai-btns { display: flex; gap: 3px; }
.ai-btn { width: 26px; height: 22px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 4px; color: var(--text-muted); font-size: 11px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
.ai-btn:hover { border-color: var(--accent); color: var(--accent); }
.ai-btn:disabled { opacity: 0.3; }
.ai-btn.go { width: auto; padding: 0 8px; background: var(--accent); color: #000; border: none; font-weight: 700; font-size: 10px; }
.nl-input-row { display: flex; gap: 4px; margin-bottom: 6px; }
.nl-input { flex: 1; padding: 6px 10px; font-size: 11px; }
.ai-loading { font-size: 10px; color: var(--accent); margin-bottom: 4px; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.autocomplete-wrap { position: relative; }
.ac-popup { position: absolute; left: 0; right: 0; top: 100%; z-index: 100; background: #1A1A1A; border: 1px solid var(--border); border-radius: 6px; max-height: 200px; overflow-y: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.6); }
.ac-item { padding: 6px 12px; font-size: 11px; color: var(--text-secondary); cursor: pointer; border-bottom: 1px solid rgba(255,255,255,0.03); }
.ac-item:hover, .ac-item.selected { background: var(--accent-dim); color: var(--accent); }
label.danger { color: #f87171; }
.exclude-section { margin-bottom: 0; }
.exclude-toggle { font-size: 10px; font-weight: 800; color: #f87171; letter-spacing: 1px; cursor: pointer; list-style: none; }
.exclude-toggle::-webkit-details-marker { display: none; }
.exclude-help { display: flex; flex-direction: column; gap: 2px; margin: 6px 0; padding: 6px 8px; background: rgba(248,113,113,0.03); border: 1px solid rgba(248,113,113,0.1); border-radius: 4px; }
.exclude-help span { font-size: 9px; color: var(--text-muted); font-family: 'Consolas', monospace; }
.exclude-textarea { color: #f87171; border-color: rgba(248,113,113,0.2); font-size: 11px; }
.excl-mgr-btn { padding: 1px 8px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 3px; color: var(--text-muted); font-size: 8px; cursor: pointer; margin-left: 8px; }
.excl-mgr-btn:hover { border-color: #f87171; color: #f87171; }

/* Exclude Manager Modal */
.em-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); z-index: 3000; display: flex; align-items: center; justify-content: center; }
.em-modal { width: 700px; height: 500px; background: var(--bg-secondary); border: 1px solid var(--border); border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
.em-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
.em-header h3 { font-size: 12px; letter-spacing: 2px; color: #f87171; }
.em-desc { font-size: 9px; color: var(--text-muted); flex: 1; }
.em-body { flex: 1; display: flex; overflow: hidden; }
.em-rules { width: 200px; overflow-y: auto; border-right: 1px solid var(--border); padding: 8px; }
.em-rule-item { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; font-size: 11px; cursor: pointer; border-radius: 4px; margin-bottom: 2px; border: 1px solid transparent; }
.em-rule-item:hover { background: var(--bg-input); }
.em-rule-item.active { border-color: #f87171; background: rgba(248,113,113,0.05); }
.em-rule-text { font-family: 'Consolas', monospace; }
.em-match-count { font-size: 9px; color: var(--text-muted); background: var(--bg-button); padding: 1px 6px; border-radius: 8px; }
.em-matches { flex: 1; overflow-y: auto; padding: 12px; }
.em-match-header { font-size: 10px; color: var(--text-muted); margin-bottom: 8px; }
.em-match-list { display: flex; flex-wrap: wrap; gap: 4px; }
.em-tag { padding: 3px 10px; background: rgba(248,113,113,0.05); border: 1px solid rgba(248,113,113,0.2); border-radius: 4px; color: #f87171; font-size: 10px; cursor: pointer; transition: all 0.12s; }
.em-tag:hover { border-color: rgba(248,113,113,0.5); }
.em-tag.excepted { background: rgba(74,222,128,0.1); border-color: rgba(74,222,128,0.3); color: #4ade80; text-decoration: line-through; opacity: 0.6; }
.em-tag.excepted:hover { opacity: 1; }
.em-rule-rm { background: none; border: none; color: #f87171; cursor: pointer; font-size: 11px; flex-shrink: 0; }
.em-rule-edit { flex: 1; padding: 2px 6px; font-size: 11px; background: var(--bg-card); border: 1px solid var(--accent); border-radius: 3px; color: var(--text-primary); font-family: 'Consolas', monospace; }
.em-rule-text { cursor: default; }
.em-add-row { display: flex; gap: 4px; margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border); }
.em-add-input { flex: 1; padding: 4px 8px; font-size: 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 3px; color: var(--text-primary); }
.em-add-btn { width: 28px; background: var(--bg-button); border: 1px solid var(--border); border-radius: 3px; color: var(--accent); font-weight: 800; cursor: pointer; }
.em-empty-sm { padding: 8px; text-align: center; color: var(--text-muted); font-size: 10px; }
.em-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 12px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
input[type="number"] { -moz-appearance: textfield; }
input::-webkit-outer-spin-button, input::-webkit-inner-spin-button { -webkit-appearance: none; }
/* neg block field */
.neg :deep(.tbf) { border-color: rgba(248,113,113,0.15); }
.neg :deep(.tbf-block) { border-color: rgba(248,113,113,0.2); color: #f87171; font-size: 10px; }
</style>
