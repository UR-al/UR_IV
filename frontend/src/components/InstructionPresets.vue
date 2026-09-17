<template>
  <section class="instruction-presets" :aria-label="`${label} 사용자 프리셋`" :aria-busy="busy">
    <strong>{{ label }} 사용자 프리셋</strong>
    <div v-if="scope === 'schema'" class="preset-buttons" role="group" aria-label="구조화된 출력 프리셋 선택">
      <button v-for="item in presets" :key="item.id" type="button" :disabled="busy || disabled"
        :aria-pressed="selected === item.id && item.instructions === instructions" @click="chooseSchema(item)">{{ item.name }}</button>
      <span v-if="!presets.length">저장된 스키마 프리셋이 없습니다.</span>
      <button type="button" :disabled="busy" @click="load">프리셋 새로고침</button>
    </div>
    <div v-else-if="showPicker !== false" class="preset-controls">
      <select v-model="selected" :disabled="busy || disabled" :aria-label="`${label} 저장된 프리셋`" @change="chooseForEditing">
        <option value="">저장된 프리셋 선택…</option>
        <option v-for="item in presets" :key="item.id" :value="item.id">{{ item.name }}</option>
      </select>
      <button type="button" :disabled="busy || disabled || !chosen" @click="apply">불러오기</button>
      <button type="button" :disabled="busy" @click="load">목록 새로고침</button>
    </div>
    <label class="preset-field">프리셋 이름
      <input v-model="name" :disabled="busy || disabled" maxlength="80" :aria-label="`${label} 새 프리셋 이름`" placeholder="프리셋 이름 (최대 80자)" @input="editorDirty = true" @keydown.enter.prevent="save(false)" />
    </label>
    <label v-if="scope === 'chat'" class="preset-field">프리셋 내용
      <textarea v-model="draftContent" :disabled="busy || disabled" :aria-label="`${label} 프리셋 내용`" rows="5" maxlength="64000"
        placeholder="이 프리셋에 저장할 지침을 작성하세요. 현재 대화 지침과 별도로 편집됩니다." @input="editorDirty = true" />
    </label>
    <small v-else>프리셋 내용: {{ scope === 'schema' ? '위 JSON 스키마 편집기의 내용' : '공통 지침과 9개 기능별 지침 전체' }}</small>
    <div class="preset-controls">
      <button type="button" :disabled="busy || disabled || !name.trim()" @click="save(false)">새 프리셋 저장</button>
      <button type="button" :disabled="busy || disabled || !chosen || !name.trim()" @click="confirmAction = 'replace'">선택 프리셋 이름·내용 수정</button>
      <button type="button" :disabled="busy || disabled || !chosen" @click="confirmAction = 'delete'">삭제</button>
    </div>
    <div v-if="confirmAction && chosen" class="preset-confirm" role="group" aria-label="프리셋 변경 확인">
      <span>“{{ chosen.name }}”{{ confirmAction === 'delete' ? ' 프리셋을 삭제할까요? 현재 작성 중인 내용은 유지됩니다.' : `의 이름과 내용을 현재 편집본 “${name}”으로 바꿀까요?` }}</span>
      <button type="button" :disabled="busy" @click="confirmAction === 'delete' ? remove() : save(true)">{{ confirmAction === 'delete' ? '삭제 확인' : '덮어쓰기 확인' }}</button>
      <button type="button" :disabled="busy" @click="confirmAction = ''">취소</button>
    </div>
    <small>{{ scope === 'chat' ? '저장하면 위 지침 프리셋 목록이 즉시 갱신됩니다. 선택한 지침 적용을 눌러야 대화에 사용됩니다.' : scope === 'schema' ? '이름 버튼을 누르면 스키마를 불러와 자동 저장합니다. ON/OFF는 변경하지 않습니다. 편집 중 자동 저장과 이름별 프리셋 저장은 별개입니다.' : '프리셋 저장·삭제는 현재 적용된 지침을 바꾸지 않습니다. 불러오기는 편집기에 적용합니다.' }}</small>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else role="status" aria-live="polite">{{ busy ? '처리 중…' : notice }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import type { AiAssistInstructions, InstructionPreset } from '../types/bridge'

type Instructions = string | AiAssistInstructions
const props = withDefaults(defineProps<{ scope: 'chat' | 'assist' | 'schema'; instructions: Instructions; disabled?: boolean; showPicker?: boolean; selectedId?: string }>(), { showPicker: true })
const emit = defineEmits<{ apply: [instructions: Instructions]; listChanged: [presets: InstructionPreset[]]; selected: [id: string] }>()
const label = computed(() => props.scope === 'chat' ? '대화 지침' : props.scope === 'schema' ? '구조화된 출력' : 'AI 어시스트 지침')
const presets = ref<InstructionPreset[]>([])
const selected = ref('')
const name = ref('')
const draftContent = ref(typeof props.instructions === 'string' ? props.instructions : '')
const editorDirty = ref(false)
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
  emit('listChanged', presets.value)
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
        if (reply.preset?.id) {
          selected.value = reply.preset.id
          editorDirty.value = false
          emit('selected', selected.value)
        }
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
defineExpose({ refresh: load, busy })
function save(replace: boolean) {
  if (props.disabled || !name.value.trim() || (replace && !chosen.value)) return
  return request('saveInstructionPreset', JSON.stringify({ scope: props.scope,
    id: replace ? chosen.value?.id : undefined, name: name.value.trim(),
    instructions: props.scope === 'chat' ? draftContent.value : props.instructions }), '프리셋을 저장했습니다.')
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
function chooseForEditing() {
  confirmAction.value = ''
  if (chosen.value) {
    name.value = chosen.value.name
    if (props.scope === 'chat' && typeof chosen.value.instructions === 'string') draftContent.value = chosen.value.instructions
    editorDirty.value = false
  }
  emit('selected', selected.value)
}
function chooseSchema(item: InstructionPreset) {
  if (busy.value || props.disabled) return
  selected.value = item.id; chooseForEditing(); apply()
}
watch(() => props.selectedId, id => {
  if (id === undefined || id === selected.value) return
  selected.value = id; chooseForEditing()
})
watch(() => props.instructions, value => {
  if (props.scope === 'chat' && !editorDirty.value && !selected.value && typeof value === 'string') draftContent.value = value
})
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
.preset-buttons { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 10px 0; }
.preset-buttons button { max-width: 100%; white-space: normal; overflow-wrap: anywhere; }
.preset-buttons [aria-pressed=true] { border-color: var(--accent); background: var(--accent-dim); }
.preset-field { display: flex; flex-direction: column; gap: 6px; margin: 12px 0; color: var(--text-primary); }
.preset-field input { flex: none; box-sizing: border-box; }
.preset-field textarea { width: 100%; box-sizing: border-box; resize: vertical; min-height: 100px; padding: 9px; font: inherit; line-height: 1.6; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; }
input, select { flex: 1 1 180px; min-width: 0; width: 100%; }
input, select, button { font: inherit; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; }
button { cursor: pointer; background: var(--bg-button); }
:disabled { opacity: .55; cursor: not-allowed; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
small, p { display: block; margin: 6px 0 0; color: var(--text-muted); overflow-wrap: anywhere; line-height: 1.6; }
[role=alert] { color: var(--state-alert-fg); }
.preset-confirm span { flex-basis: 100%; overflow-wrap: anywhere; }
</style>
