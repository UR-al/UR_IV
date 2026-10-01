<template>
  <section class="workflow-controls" aria-label="ComfyUI 워크플로 상세 설정">
    <h3>ComfyUI 워크플로 도구</h3>
    <p class="hint">현재 선택한 ComfyUI 서버를 사용합니다. 노드 연결과 원본 JSON은 바꾸지 않습니다.</p>
    <p v-if="!ready" class="hint" role="status">워크플로 파일·프리셋 관리는 로컬 데스크톱 앱에서 사용할 수 있습니다.</p>
    <fieldset class="quality" :disabled="busy || !ready">
      <legend>빠름 / 정밀 프리셋 · T2I</legend>
      <p class="hint">빠름은 Hires·ADetailer·SAM3를 끕니다. 정밀은 Hires 1.5×와 선택한 SAM3 영역 보정을 켭니다. 모델·해상도·기본 스텝·업스케일러는 유지합니다.</p>
      <label class="inline">정밀 보정 대상
        <select v-model="targets"><option value="none">없음 · Hires만</option><option value="face">얼굴</option><option value="eyes">눈</option><option value="face_then_eyes">얼굴 → 눈 · 순차 2패스</option></select>
      </label>
      <div class="buttons">
        <button type="button" @click="applyPreset('fast')">빠름 적용</button>
        <button type="button" @click="applyPreset('detail')">정밀 적용</button>
        <button type="button" @click="applyPreset('restore')">이전 사용자 설정 복원</button>
        <button type="button" @click="send('comfy_feature_preflight', { mode: 'txt2img' })">현재 기능 사전 검증</button>
      </div>
      <p class="hint">처음 적용하기 전의 설정을 config에 보관합니다. 프리셋을 바꿔도 최초 복원본은 유지됩니다. 얼굴 → 눈은 앞 패스의 결과를 다시 보정하며 적용한 앱 창·서버에서만 사용합니다. SAM3의 대상·모드를 직접 바꾸거나 끄면 추가 눈 패스는 해제됩니다. 다시 사용하거나 앱을 재시작한 뒤에는 프리셋을 다시 적용하세요. 워크플로 상세 입력은 큐 등록 당시 값을 유지합니다.</p>
    </fieldset>
    <div v-if="preflight" class="preflight" role="status">
      <strong>{{ preflight.ok ? '생성 그래프 검증 완료' : '생성 전 해결할 항목이 있습니다' }}</strong>
      <ul><li v-for="feature in preflight.features" :key="feature.id">{{ feature.label }} — {{ stateLabel(feature.state) }}<span v-if="feature.reason" class="hint"> ({{ feature.reason }})</span></li></ul>
      <p v-if="preflight.error" class="error">{{ preflight.error }}</p><p class="hint">{{ preflight.note }}</p>
    </div>
    <details>
      <summary>사용자 워크플로 노드별 상세 설정</summary>
      <p class="hint">API 설정에 지정한 파일에서 입력을 읽습니다. 적용할 입력만 체크하세요. 모델·프롬프트 등 앱에서 관리하는 입력은 여기서 수정하지 않습니다.</p>
      <div class="buttons">
        <label class="inline">워크플로<select v-model="mode" :disabled="busy" @change="resetSchema"><option value="txt2img">T2I</option><option value="img2img">I2I / Inpaint</option></select></label>
        <button type="button" :disabled="busy || !ready" @click="send('comfy_controls_inspect')">입력 새로 읽기</button>
        <button type="button" :disabled="busy || !ready" @click="send('comfy_controls_clear')">이 워크플로의 저장 설정 해제</button>
      </div>
      <p v-if="warning" class="error" role="alert">{{ warning }} 기존 설정이 자동 적용되지 않습니다. 새 설정을 저장하거나 해제하세요.</p>
      <template v-if="schema">
        <label class="search">노드·입력 검색<input v-model="search" type="search" placeholder="노드 이름, 입력 이름, ID" /></label>
        <div class="nodes">
          <details v-for="node in filteredNodes" :key="node.id" class="node">
            <summary>{{ node.title }} <span class="hint">#{{ node.id }} · {{ node.classType }}</span></summary>
            <div v-for="field in node.fields" :key="fieldKey(field)" class="input-row">
              <label class="field-name"><input type="checkbox" v-model="entries[fieldKey(field)]!.enabled" :disabled="busy || field.managed" :aria-label="`${field.name} 상세 설정 적용`" />{{ field.name }}</label>
              <span v-if="field.managed" class="hint">기본 생성 UI에서 관리</span>
              <template v-else>
                <input v-if="field.type === 'boolean'" type="checkbox" v-model="entries[fieldKey(field)]!.value" :disabled="busy || !entries[fieldKey(field)]!.enabled" :aria-label="field.name" />
                <select v-else-if="field.type === 'enum'" v-model="entries[fieldKey(field)]!.value" :disabled="busy || !entries[fieldKey(field)]!.enabled" :aria-label="field.name"><option v-for="(choice, index) in field.choices" :key="index" :value="choice">{{ choice }}</option></select>
                <textarea v-else-if="field.type === 'string' && field.multiline" :value="String(entries[fieldKey(field)]!.value)" @input="entries[fieldKey(field)]!.value = ($event.target as HTMLTextAreaElement).value" :disabled="busy || !entries[fieldKey(field)]!.enabled" :aria-label="field.name" rows="3" />
                <input v-else :type="field.type === 'float' ? 'number' : 'text'" :inputmode="field.type === 'int' ? 'numeric' : undefined" v-model="entries[fieldKey(field)]!.value" :min="field.min" :max="field.max" :step="field.step || (field.type === 'float' ? 'any' : undefined)" :disabled="busy || !entries[fieldKey(field)]!.enabled" :aria-label="field.name" />
                <span v-if="field.min !== undefined || field.max !== undefined" class="hint range">{{ field.min ?? '−∞' }} ~ {{ field.max ?? '∞' }}</span>
              </template>
            </div>
          </details>
          <p v-if="!filteredNodes.length" class="hint">표시할 스칼라 입력이 없습니다. 링크·이미지·컨테이너 입력은 편집하지 않습니다.</p>
        </div>
        <p v-if="validationError" class="error" role="alert">{{ validationError }}</p>
        <button type="button" :disabled="busy || !!validationError" @click="save">선택한 입력 저장 · 다음 생성부터 적용</button>
      </template>
    </details>
    <p v-if="busy" class="hint" role="status">현재 ComfyUI 설정을 확인하는 중…</p>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="notice" class="hint" role="status">{{ notice }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { buildWorkflowBinding, fieldKey, type WorkflowBinding, type WorkflowSchema } from '../utils/comfyWorkflowControls'
import type { ActionName } from '../types/bridge'

const ready = ref(false), busy = ref(false), error = ref(''), warning = ref(''), notice = ref('')
const mode = ref('txt2img'), targets = ref('face'), search = ref('')
const schema = ref<WorkflowSchema | null>(null)
const entries = reactive<Record<string, { enabled: boolean; value: string | number | boolean }>>({})
const preflight = ref<{ ok: boolean; features: { id: string; label: string; state: string; reason?: string }[]; error?: string; note?: string } | null>(null)
let requestId = '', disposed = false, disconnect: (() => void) | undefined, timer: ReturnType<typeof setTimeout> | undefined
const filteredNodes = computed(() => {
  const query = search.value.trim().toLocaleLowerCase()
  return (schema.value?.nodes || []).map(node => {
    const nodeMatches = `${node.id} ${node.classType} ${node.title}`.toLocaleLowerCase().includes(query)
    return { ...node, fields: node.fields.filter(field => nodeMatches || field.name.toLocaleLowerCase().includes(query)) }
  }).filter(node => node.fields.length)
})
const validationError = computed(() => {
  if (!schema.value) return ''
  try { buildWorkflowBinding(schema.value, entries); return '' } catch (problem) { return String((problem as Error).message) }
})
// skipped: 생성을 막지 않고 빼는 기능(VAE DeGrid — 노드·모델·출력 자리가 없을 때, core/comfy_workflow_controls.feature_preflight)
const stateLabel = (state: string) => ({ ready: '그래프에 적용됨', off: '꺼짐', blocked: '검증 실패', missing: '그래프에서 누락', skipped: '건너뜀(생성은 진행)' }[state] || state)
function resetSchema() { schema.value = null; warning.value = ''; for (const key of Object.keys(entries)) delete entries[key] }
function acceptSchema(next: WorkflowSchema, binding?: WorkflowBinding | null) {
  resetSchema(); schema.value = next
  const overrides = new Map((binding?.overrides || []).map(item => [fieldKey(item), item.value]))
  for (const node of next.nodes) for (const field of node.fields) {
    const key = fieldKey(field), value = overrides.has(key) ? overrides.get(key)! : field.value
    entries[key] = { enabled: overrides.has(key), value: field.type === 'int' ? String(value) : value }
  }
}
function send(action: ActionName, payload: Record<string, unknown> = {}) {
  if (!ready.value || busy.value || disposed) return
  error.value = ''; notice.value = ''; busy.value = true
  requestId = `comfy-controls-${crypto.randomUUID()}`
  requestAction(action, { requestId, mode: mode.value, ...payload })
  timer = setTimeout(() => { busy.value = false; requestId = ''; error.value = '응답이 지연됩니다. 연결과 저장 상태를 다시 확인하세요.' }, 65000)
}
function save() {
  if (!schema.value) return
  try { send('comfy_controls_save', { binding: buildWorkflowBinding(schema.value, entries) }) } catch (problem) { error.value = (problem as Error).message }
}
function applyPreset(preset: string) { preflight.value = null; send('comfy_quality_preset', { preset, targets: targets.value }) }
onMounted(() => {
  disconnect = onBackendEvent('comfyWorkflowEvent', (raw: string) => {
    let event: any
    try { event = JSON.parse(raw) } catch { return }
    if (disposed || event.requestId !== requestId) return
    clearTimeout(timer); requestId = ''; busy.value = false
    if (!event.ok) { error.value = event.error || '워크플로 설정을 확인하지 못했습니다.'; return }
    if (event.schema) { acceptSchema(event.schema, event.binding); warning.value = event.warning || '' }
    if (event.saved) notice.value = '상세 설정을 저장했습니다. 원본 워크플로 파일은 유지됩니다.'
    if (event.cleared) { resetSchema(); notice.value = '저장된 상세 설정 적용을 해제했습니다. 원본 JSON은 유지됩니다.' }
    if (event.preflight) preflight.value = event.preflight
    if (event.preset) { notice.value = `${event.preset.name === 'custom' ? '이전 사용자 설정 복원' : '프리셋 적용'} 완료. ${event.preset.note}`; send('comfy_feature_preflight', { mode: 'txt2img' }) }
  })
  getBackend().then(backend => { if (!disposed) ready.value = Boolean(backend.comfyWorkflowEvent) })
})
onUnmounted(() => { disposed = true; requestId = ''; clearTimeout(timer); disconnect?.() })
</script>

<style scoped>
.workflow-controls { display: flex; flex-direction: column; gap: var(--sp-3); color: var(--text-primary); min-width: 0; }
h3 { font-size: var(--fs-body); margin: 0; }
p { margin: 0; line-height: 1.6; overflow-wrap: anywhere; }
.hint { color: var(--text-secondary); font-size: var(--fs-meta); }
.error { color: var(--state-alert-fg); font-size: var(--fs-meta); }
.quality, .preflight { border: 1px solid var(--border); border-radius: var(--r-md, 8px); padding: var(--sp-3); display: flex; flex-direction: column; gap: var(--sp-3); min-width: 0; }
legend, summary { font-size: var(--fs-meta); font-weight: var(--fw-bold); }
summary { cursor: pointer; padding: var(--sp-2) 0; overflow-wrap: anywhere; }
.buttons { display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center; margin: var(--sp-2) 0; }
.inline, .search { display: flex; flex-wrap: wrap; gap: var(--sp-2); align-items: center; font-size: var(--fs-meta); }
.search { margin: var(--sp-3) 0; }
.search input { flex: 1; }
button { background: var(--bg-secondary); color: var(--text-primary); border: 1px solid var(--border); border-radius: var(--r-sm, 4px); padding: var(--sp-2) var(--sp-3); font-size: var(--fs-meta); cursor: pointer; white-space: normal; }
button:hover:not(:disabled) { background: var(--bg-button-hover); }
button:disabled { opacity: .55; cursor: default; }
input, textarea, select { min-width: 0; max-width: 100%; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--r-sm, 4px); padding: var(--sp-2); font-size: var(--fs-meta); }
input[type="checkbox"] { flex: 0 0 auto; width: 16px; height: 16px; }
button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.nodes { max-height: 420px; overflow: auto; margin-bottom: var(--sp-3); }
.node { border-bottom: 1px solid var(--border); padding-bottom: var(--sp-2); }
.input-row { display: grid; grid-template-columns: minmax(100px, 1fr) minmax(0, 1.5fr); gap: var(--sp-2); align-items: center; margin: var(--sp-2) 0; }
.field-name { display: flex; align-items: center; gap: var(--sp-2); font-size: var(--fs-meta); overflow-wrap: anywhere; }
.range { grid-column: 2; }
ul { padding-left: 1.4em; margin: 0; font-size: var(--fs-meta); }
@media (max-width: 520px) { .input-row { grid-template-columns: minmax(0, 1fr); } .range { grid-column: 1; } }
</style>
