<template>
  <section class="instruction-presets" :aria-label="`${label} 사용자 프리셋`" :aria-busy="busy">
    <strong>{{ label }} 사용자 프리셋</strong>
    <div class="preset-controls">
      <select v-model="selected" :disabled="busy || disabled" :aria-label="`${label} 저장된 프리셋`" @change="confirmAction = ''">
        <option value="">저장된 프리셋 선택…</option>
        <option v-for="item in presets" :key="item.id" :value="item.id">{{ item.name }}</option>
      </select>
      <button type="button" :disabled="busy || disabled || !chosen" @click="apply">불러오기</button>
      <button type="button" :disabled="busy || disabled || !chosen" @click="confirmAction = 'delete'">삭제</button>
      <button type="button" :disabled="busy" @click="load">목록 새로고침</button>
    </div>
    <div class="preset-controls">
      <input v-model="name" :disabled="busy || disabled" maxlength="80" :aria-label="`${label} 새 프리셋 이름`" placeholder="프리셋 이름 (최대 80자)" @keydown.enter.prevent="save(false)" />
      <button type="button" :disabled="busy || disabled || !name.trim()" @click="save(false)">새 프리셋 저장</button>
      <button type="button" :disabled="busy || disabled || !chosen" @click="confirmAction = 'replace'">선택 프리셋 덮어쓰기</button>
    </div>
    <div v-if="confirmAction && chosen" class="preset-confirm" role="group" aria-label="프리셋 변경 확인">
      <span>“{{ chosen.name }}”{{ confirmAction === 'delete' ? ' 프리셋을 삭제할까요? 현재 작성 중인 지침은 유지됩니다.' : '에 현재 작성 중인 지침을 덮어쓸까요?' }}</span>
      <button type="button" :disabled="busy" @click="confirmAction === 'delete' ? remove() : save(true)">{{ confirmAction === 'delete' ? '삭제 확인' : '덮어쓰기 확인' }}</button>
      <button type="button" :disabled="busy" @click="confirmAction = ''">취소</button>
    </div>
    <small>프리셋 저장·삭제는 현재 적용된 지침을 바꾸지 않습니다. 불러오기는 편집기에 적용합니다.</small>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else role="status" aria-live="polite">{{ busy ? '처리 중…' : notice }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import type { AiAssistInstructions } from '../types/bridge'

type Instructions = string | AiAssistInstructions
type Preset = { id: string; name: string; scope: string; instructions: Instructions }
const props = defineProps<{ scope: 'chat' | 'assist'; instructions: Instructions; disabled?: boolean }>()
const emit = defineEmits<{ apply: [instructions: Instructions] }>()
const label = computed(() => props.scope === 'chat' ? '대화 지침' : 'AI 어시스트 지침')
const presets = ref<Preset[]>([])
const selected = ref('')
const name = ref('')
const busy = ref(false)
const error = ref('')
const notice = ref('')
const confirmAction = ref<'' | 'delete' | 'replace'>('')
const chosen = computed(() => presets.value.find(item => item.id === selected.value))
let disposed = false
let serial = 0
let timer: ReturnType<typeof setTimeout> | undefined
let unsubscribe: (() => void) | undefined

function acceptList(reply: any) {
  if (reply.scope !== props.scope || !Array.isArray(reply.presets)) return
  presets.value = reply.presets
  if (!chosen.value) { selected.value = ''; confirmAction.value = '' }
}
async function request(method: string, payload: string, success: string) {
  if (busy.value || disposed) return
  const token = ++serial
  busy.value = true; error.value = ''; notice.value = ''
  const current = () => !disposed && token === serial
  timer = setTimeout(() => {
    if (!current()) return
    ++serial; busy.value = false
    error.value = '응답 시간이 초과되었습니다. 저장 여부는 목록 새로고침으로 확인하세요.'
  }, 12000)
  try {
    const backend = await getBackend()
    if (!current()) return
    if (typeof backend?.[method] !== 'function') throw Error('이 연결은 지침 프리셋을 지원하지 않습니다. 앱을 다시 시작하세요.')
    backend[method](payload, (raw: string) => {
      if (!current()) return
      clearTimeout(timer); busy.value = false
      try {
        const reply = JSON.parse(raw)
        if (!reply.ok) throw Error(reply.error || '프리셋 요청에 실패했습니다')
        acceptList(reply)
        if (reply.preset?.id) selected.value = reply.preset.id
        confirmAction.value = ''; notice.value = success
      } catch (problem) { error.value = problem instanceof Error ? problem.message : '프리셋 응답 오류' }
    })
  } catch (problem) {
    if (!current()) return
    clearTimeout(timer); busy.value = false
    error.value = problem instanceof Error ? problem.message : '프리셋 연결 오류'
  }
}
function load() { return request('getInstructionPresets', props.scope, '') }
function save(replace: boolean) {
  if (props.disabled || (replace ? !chosen.value : !name.value.trim())) return
  return request('saveInstructionPreset', JSON.stringify({ scope: props.scope,
    id: replace ? chosen.value?.id : undefined, name: replace ? chosen.value?.name : name.value.trim(),
    instructions: props.instructions }), '프리셋을 저장했습니다.')
}
function remove() {
  if (!chosen.value || props.disabled) return
  return request('deleteInstructionPreset', JSON.stringify({ scope: props.scope, id: chosen.value.id }), '프리셋을 삭제했습니다. 현재 지침은 유지됩니다.')
}
function apply() {
  if (!chosen.value || busy.value || props.disabled) return
  emit('apply', JSON.parse(JSON.stringify(chosen.value.instructions)))
  notice.value = '편집기에 불러왔습니다.'; error.value = ''
}
onMounted(() => {
  unsubscribe = onBackendEvent('instructionPresetsChanged', (raw: string) => {
    try { const reply = JSON.parse(raw); if (reply.ok) acceptList(reply) } catch { /* malformed events are ignored */ }
  })
  load()
})
onUnmounted(() => { disposed = true; ++serial; clearTimeout(timer); unsubscribe?.() })
</script>

<style scoped>
.instruction-presets { min-width: 0; padding: 12px; margin: 10px 0; border: 1px solid var(--border); border-radius: var(--radius-base, 8px); background: var(--bg-card); color: var(--text-primary); font-size: 12px; }
.preset-controls, .preset-confirm { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0; }
input, select { flex: 1 1 180px; min-width: 0; width: 100%; }
input, select, button { font: inherit; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; }
button { cursor: pointer; background: var(--bg-button); }
:disabled { opacity: .55; cursor: not-allowed; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
small, p { display: block; margin: 6px 0 0; color: var(--text-muted); overflow-wrap: anywhere; line-height: 1.6; }
[role=alert] { color: var(--state-alert-fg); }
.preset-confirm span { flex-basis: 100%; overflow-wrap: anywhere; }
</style>
