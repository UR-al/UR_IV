<template>
  <section class="assist-instructions" :aria-labelledby="`${idPrefix}-instructions-title`" :aria-busy="busy">
    <header>
      <h2 :id="`${idPrefix}-instructions-title`">AI 어시스트 사용자 지침</h2>
      <p>공통 지침과 기능별 추가 지침을 설정합니다. 빈칸은 기존 동작을 유지합니다. 여기서 작성한 지침은 Chat 대화와 Batch/Upscale 이미지 캡션에 적용되지 않습니다.</p>
    </header>
    <InstructionPresets scope="assist" :instructions="draft" :disabled="locked" @apply="applyPreset" />
    <div class="instruction-guide" :id="`${idPrefix}-instructions-guide`">
      <p>일반 기능은 <strong>공통 지침 → 선택한 기능의 추가 지침</strong> 순서로 적용합니다. 기능별 지침으로 공통 지침을 구체화하세요.</p>
      <p>자동 자연어 변환은 <strong>공통 → 자연어 캡션 → 생성 전 자동 자연어 변환</strong> 순서로 이어받습니다.</p>
      <p>태그·문장 등의 출력 형식은 앱이 관리합니다. 원하는 표현, 관찰 기준, 제외할 내용 중심으로 작성하세요.</p>
    </div>

    <div class="instruction-field common-field">
      <div class="field-heading">
        <label :for="`${idPrefix}-common`">공통 지침</label>
        <small aria-hidden="true">{{ characterCount(draft.common).toLocaleString() }} / 8,000</small>
      </div>
      <p :id="`${idPrefix}-common-help`">아래 9개 기능에 함께 적용합니다. 각 입력란은 최대 8,000자이며, 모두 비워 저장하면 추가 지침 없이 기존 기능을 사용합니다.</p>
      <textarea :id="`${idPrefix}-common`" :value="draft.common" :disabled="locked" :maxlength="LIMIT * 2" rows="4"
        :aria-describedby="`${idPrefix}-common-help ${idPrefix}-instructions-guide`"
        placeholder="예: 입력에 없는 인물 관계나 감정을 추측하지 말고, 확인할 수 있는 외형과 행동을 중심으로 표현하세요."
        @input="edit('common', $event)" />
    </div>

    <div class="feature-grid">
      <div v-for="feature in features" :key="feature.id" class="instruction-field">
        <div class="field-heading">
          <label :for="`${idPrefix}-${feature.id}`">{{ feature.label }} 추가 지침</label>
          <small aria-hidden="true">{{ characterCount(draft.features[feature.id]).toLocaleString() }} / 8,000</small>
        </div>
        <p :id="`${idPrefix}-${feature.id}-help`">{{ feature.description }}</p>
        <textarea :id="`${idPrefix}-${feature.id}`" :value="draft.features[feature.id]" :disabled="locked" :maxlength="LIMIT * 2" rows="3"
          :aria-describedby="`${idPrefix}-${feature.id}-help`" :aria-label="`${feature.label} 추가 지침, 최대 8000자`"
          :placeholder="feature.placeholder" @input="edit(feature.id, $event)" />
      </div>
    </div>

    <footer>
      <div v-if="remoteChange" role="alert">
        <p>다른 화면에서 지침을 저장했습니다. 이 화면의 편집 내용은 유지됩니다. 사용할 내용을 선택하세요.</p>
        <button type="button" :disabled="busy" @click="useRemote">새 저장본 불러오기</button>
        <button type="button" :disabled="busy" @click="overwriteRemote">내 편집으로 덮어쓰기</button>
      </div>
      <div class="save-state" role="status" aria-live="polite">
        <span v-if="phase === 'loading'">저장된 지침을 불러오는 중…</span>
        <span v-else-if="phase === 'saving'">지침 저장 중…</span>
        <span v-else-if="dirty" class="unsaved">저장하지 않은 변경사항이 있습니다.</span>
        <span v-else-if="notice">{{ notice }}</span>
        <span v-else-if="loaded">저장된 지침과 같습니다.</span>
      </div>
      <div class="instruction-actions">
        <button v-if="!loaded && !busy" type="button" @click="load">연결 다시 시도</button>
        <button type="button" class="save-button" :disabled="!loaded || busy || !dirty || !!remoteChange" @click="save">
          {{ phase === 'saving' ? '저장 중…' : '지침 저장' }}
        </button>
      </div>
      <p class="save-help">이 지침은 위 전용 ‘지침 저장’ 버튼으로 저장합니다. 모델이나 Chat의 개인 지침은 변경하지 않습니다.</p>
      <p v-if="error" class="instruction-error" role="alert">{{ error }}</p>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import InstructionPresets from './InstructionPresets.vue'
import type { AiAssistFeature, AiAssistInstructions, AiAssistInstructionsResult } from '../types/bridge'
withDefaults(defineProps<{ idPrefix?: string }>(), { idPrefix: 'ai-assist' })
const remoteChange = ref('')
let unsubscribe: (() => void) | undefined

const LIMIT = 8000
const RESPONSE_TIMEOUT = 12000
const features: Array<{ id: AiAssistFeature; label: string; description: string; placeholder: string }> = [
  { id: 'expand', label: '태그 확장', description: '입력한 태그를 바탕으로 관련 세부 태그를 보완합니다.',
    placeholder: '예: 구도와 조명 태그를 보완하되, 원래 의상과 배경은 바꾸지 마세요.' },
  { id: 'suggest', label: '유사 태그 추천', description: '기존 태그와 비슷하거나 함께 사용할 만한 태그를 추천합니다.',
    placeholder: '예: 비슷한 포즈와 카메라 시점의 태그를 우선 추천하세요.' },
  { id: 'nl2tags', label: '자연어 → 태그', description: '자연어 설명을 이미지 생성에 사용할 영어 태그로 변환합니다.',
    placeholder: '예: 설명에 명시된 색상, 의상, 행동을 빠뜨리지 마세요.' },
  { id: 'nl_caption', label: '자연어 캡션', description: '프롬프트와 태그를 바탕으로 자연어 캡션을 작성합니다. 자동 자연어 변환도 이 지침을 이어받습니다.',
    placeholder: '예: 인물 수는 문장에 명시하고, 외형·행동·배경을 각각 구체적으로 묘사하세요.' },
  { id: 'nl_scene', label: '영문 장면 묘사', description: '생성할 장면을 자연스러운 영어 문장으로 묘사합니다.',
    placeholder: '예: 전경과 배경의 위치 관계를 분명히 하고, 모호한 감정 표현은 피하세요.' },
  { id: 'translate', label: '한↔영 번역', description: '한국어와 영어 사이에서 의미를 유지하며 번역합니다.',
    placeholder: '예: 고유명사와 캐릭터 이름은 유지하고, 의상 용어는 일반적인 표현을 사용하세요.' },
  { id: 'creative', label: '창의 생성', description: '주어진 주제를 바탕으로 새로운 장면과 프롬프트 아이디어를 만듭니다.',
    placeholder: '예: 색상 대비와 배경 아이디어에 변화를 주되, 지정된 인물과 시대는 유지하세요.' },
  { id: 'negative', label: '네거티브 추천', description: '원하는 결과에서 피할 요소를 네거티브 프롬프트로 추천합니다.',
    placeholder: '예: 손과 얼굴의 형태 오류, 텍스트와 워터마크 관련 요소를 우선 고려하세요.' },
  { id: 'auto_nl', label: '생성 전 자동 자연어 변환', description: '이미지 생성 직전 실행되는 자동 변환입니다. 공통 + 자연어 캡션 + 이 추가 지침 순서로 적용합니다.',
    placeholder: '예: 생성 직전에는 새로운 요소를 추가하지 말고, 기존 태그의 시각적 정보를 간결하게 정리하세요.' },
]
function emptyInstructions(): AiAssistInstructions {
  return { common: '', features: Object.fromEntries(features.map(feature => [feature.id, ''])) as Record<AiAssistFeature, string> }
}
function characterCount(text: string) { return Array.from(text).length }
const draft = ref<AiAssistInstructions>(emptyInstructions())
const saved = ref('')
const loaded = ref(false)
const phase = ref<'loading' | 'saving' | 'idle'>('loading')
const error = ref('')
const notice = ref('')
const busy = computed(() => phase.value !== 'idle')
const locked = computed(() => busy.value || !loaded.value)
const dirty = computed(() => loaded.value && JSON.stringify(draft.value) !== saved.value)
let disposed = false
let serial = 0
let timer: ReturnType<typeof setTimeout> | undefined

function current(token: number) { return !disposed && token === serial }
function begin(operation: 'loading' | 'saving') {
  const token = ++serial
  clearTimeout(timer)
  phase.value = operation
  error.value = ''; notice.value = ''
  timer = setTimeout(() => {
    if (!current(token)) return
    ++serial // An old response must never overwrite a retry or a newer edit.
    phase.value = 'idle'
    error.value = operation === 'saving'
      ? '저장 응답을 확인하지 못했습니다. 입력한 내용은 유지됩니다. 연결을 확인한 뒤 다시 저장하세요.'
      : '지침을 불러오는 시간이 초과되었습니다. 앱 연결을 확인한 뒤 다시 시도하세요.'
  }, RESPONSE_TIMEOUT)
  return token
}
function fail(token: number, message: string) {
  if (!current(token)) return
  clearTimeout(timer)
  ++serial
  phase.value = 'idle'
  error.value = message
}
function accept(token: number, raw: string, operation: 'loading' | 'saving') {
  if (!current(token)) return
  try {
    const reply = JSON.parse(raw)
    if (!reply || reply.ok !== true) throw new Error(typeof reply?.error === 'string' ? reply.error : '지침 요청을 처리하지 못했습니다.')
    const value = reply.instructions
    if (!value || typeof value.common !== 'string' || !value.features || typeof value.features !== 'object') {
      throw new Error('지침 응답 형식이 올바르지 않습니다. 앱을 업데이트하거나 다시 연결하세요.')
    }
    const clean = emptyInstructions()
    for (const id of ['common', ...features.map(feature => feature.id)] as const) {
      const text = id === 'common' ? value.common : value.features[id]
      if (typeof text !== 'string' || characterCount(text) > LIMIT) throw new Error('저장된 지침의 형식 또는 8,000자 제한을 확인하세요.')
      if (id === 'common') clean.common = text
      else clean.features[id] = text
    }
    clearTimeout(timer)
    ++serial
    draft.value = clean
    saved.value = JSON.stringify(clean)
    remoteChange.value = ''
    loaded.value = true
    phase.value = 'idle'
    notice.value = operation === 'saving' ? '지침을 저장했습니다.' : ''
  } catch (problem) {
    fail(token, problem instanceof SyntaxError
      ? '지침 응답을 읽지 못했습니다. 연결을 확인하고 다시 시도하세요.'
      : problem instanceof Error ? problem.message : '지침 응답을 읽지 못했습니다.')
  }
}
function edit(id: 'common' | AiAssistFeature, event: Event) {
  if (locked.value) return
  const textarea = event.target as HTMLTextAreaElement
  // Match Python's code-point limit without splitting surrogate pairs. The
  // native DOM also needs clamping when the reactive value stays unchanged.
  const text = Array.from(textarea.value).slice(0, LIMIT).join('')
  if (textarea.value !== text) textarea.value = text
  if (id === 'common') draft.value.common = text
  else draft.value.features[id] = text
  notice.value = ''
}
async function load() {
  if (disposed || loaded.value) return
  const token = begin('loading')
  try {
    const backend = await getBackend()
    if (!current(token)) return
    if (!backend?.getAiAssistInstructions || !backend?.saveAiAssistInstructions) {
      throw new Error('이 앱 연결은 사용자 지침 설정을 지원하지 않습니다. 최신 앱에 다시 연결하세요.')
    }
    backend.getAiAssistInstructions((raw: string) => accept(token, raw, 'loading'))
  } catch (problem) {
    fail(token, problem instanceof Error ? problem.message : '앱 연결에 실패했습니다. 다시 시도하세요.')
  }
}
async function save() {
  if (disposed || locked.value || !dirty.value || remoteChange.value) return
  const payload = JSON.stringify(draft.value)
  const token = begin('saving')
  try {
    const backend = await getBackend()
    if (!current(token)) return
    if (!backend?.saveAiAssistInstructions) throw new Error('앱 연결이 끊어졌습니다. 연결 후 다시 저장하세요.')
    backend.saveAiAssistInstructions(payload, (raw: string) => accept(token, raw, 'saving'))
  } catch (problem) {
    fail(token, problem instanceof Error ? problem.message : '지침을 저장하지 못했습니다. 다시 시도하세요.')
  }
}
function applyPreset(value: string | AiAssistInstructions) {
  if (locked.value || typeof value === 'string') return
  draft.value = JSON.parse(JSON.stringify(value))
  notice.value = '프리셋을 불러왔습니다. 지침 저장을 눌러 적용하세요.'
}
function useRemote() {
  const raw = remoteChange.value
  if (raw) accept(++serial, raw, 'loading')
}
function overwriteRemote() { remoteChange.value = ''; save() }
onMounted(() => {
  unsubscribe = onBackendEvent('aiAssistInstructionsChanged', (raw: string) => {
    if (phase.value === 'saving') return // The save callback owns this acknowledgement.
    try {
      // 이벤트 모양은 bridge.d.ts 계약(AiAssistInstructionsResult). 깊은 검증은 accept() 가 한다.
      const reply = JSON.parse(raw) as AiAssistInstructionsResult | null
      if (!reply?.ok || !reply.instructions) return
      const snapshot = JSON.stringify(reply.instructions)
      if (snapshot === saved.value) return
      if (dirty.value && snapshot !== JSON.stringify(draft.value)) remoteChange.value = raw
      else accept(++serial, raw, 'loading')
    } catch { /* keep draft on malformed events */ }
  })
  load()
})
onUnmounted(() => { disposed = true; ++serial; clearTimeout(timer); unsubscribe?.() })
</script>

<style scoped>
.assist-instructions { margin-top: 20px; padding: 20px; border: 1px solid var(--border); border-radius: 16px; background: var(--bg-card); color: var(--text-primary); min-width: 0; }
h2 { margin: 0 0 8px; font-size: 15px; }
p { margin: 8px 0; color: var(--text-muted); font-size: 12px; line-height: 1.65; overflow-wrap: anywhere; }
.instruction-guide { margin: 16px 0; padding: 8px 12px; border-radius: 10px; background: var(--bg-input); }
strong { font-weight: var(--fw-bold); color: var(--text-secondary); }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 16px; margin-top: 20px; }
.instruction-field { min-width: 0; }
.field-heading { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; }
label { font-size: 13px; font-weight: var(--fw-bold); color: var(--text-primary); }
small { font-size: 11px; color: var(--text-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
textarea { display: block; width: 100%; box-sizing: border-box; min-height: 88px; resize: vertical; padding: 10px 12px; border: 1px solid var(--border); border-radius: 9px; background: var(--bg-input); color: var(--text-primary); font: inherit; font-size: 12px; line-height: 1.6; }
textarea::placeholder { color: var(--text-muted); opacity: .8; }
button:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
textarea:disabled, button:disabled { opacity: .55; cursor: not-allowed; }
footer { margin-top: 20px; padding-top: 16px; border-top: 1px solid var(--border); }
.save-state { min-height: 20px; font-size: 12px; color: var(--text-muted); }
.unsaved { color: var(--text-primary); }
.instruction-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
button { padding: 8px 14px; min-height: 36px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-button); color: var(--text-primary); font: inherit; font-size: 12px; cursor: pointer; }
.save-button { border-color: var(--accent); }
.save-help { font-size: 11px; }
.instruction-error { color: var(--state-alert-fg); }
@media (max-width: 640px) { .assist-instructions { padding: 14px; } .feature-grid { grid-template-columns: minmax(0, 1fr); } }
</style>
