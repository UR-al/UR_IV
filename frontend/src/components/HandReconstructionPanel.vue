<template>
  <details class="glass-card hand-panel">
    <summary>실험 · 손 형태 재구성</summary>
    <p class="hand-help">잘못된 손 형태를 먼저 지운 다음 마스크 안을 새로 생성합니다. 손가락 수를 자동 판정하거나 정상 손을 보장하는 기능은 아닙니다.</p>
    <p v-if="webMode" class="hand-warning">로컬 앱 전용입니다. 웹 모드에서는 생성·저장할 수 없습니다.</p>
    <label class="hand-enable"><input type="checkbox" :checked="enabled" :disabled="webMode" @change="setEnabled" /> 실험 기능 사용 (기본 꺼짐)</label>
    <template v-if="enabled && !webMode">
      <p class="hand-help">한 번에 손 하나를 선택하세요. 튀어나온 손가락만이 아니라 잘못된 손 전체와 손가락 뿌리·손목 연결부까지 마스크에 포함하세요. 마스크 밖 픽셀은 원본을 유지합니다.</p>
      <fieldset :disabled="busy" class="hand-controls">
        <legend>재구성 설정</legend>
        <label v-for="control in CONTROLS" :key="control.key" :for="`${id}-${control.key}`">
          <span>{{ control.label }} <output :for="`${id}-${control.key}`">{{ settings[control.key] }}{{ control.unit || '' }}</output></span>
          <input :id="`${id}-${control.key}`" type="range" :min="control.min" :max="control.max" :step="control.step" :value="settings[control.key]" @input="setSetting(control.key, $event)" />
        </label>
        <label :for="`${id}-prompt`">손 동작 추가 지침</label>
        <textarea :id="`${id}-prompt`" :value="prompt" rows="3" maxlength="4000" placeholder="예: relaxed open hand, palm facing camera" @input="setPrompt" />
      </fieldset>
      <p class="hand-help">현재 T2I 모델·샘플러·TE·VAE·LoRA 설정을 사용합니다. Hires·ADetailer·SAM3·품질 후처리는 제외합니다. 기본 Forge/Comfy 경로만 지원하며 커스텀 워크플로와 Krea 경로는 지원하지 않습니다.</p>
      <p class="hand-help">후보를 직접 비교한 뒤 별도 PNG로 저장하세요. 원본 캔버스는 바꾸지 않습니다. 연결된 생성 서버에는 입력·출력 사본이 남을 수 있습니다.</p>
      <p v-if="canvasSource" class="hand-warning">캔버스 스냅샷 기준입니다. 원본 메타데이터는 포함하지 않으며, 마스크 밖 픽셀 보존도 이 스냅샷을 기준으로 합니다.</p>
      <button class="hand-run" type="button" :disabled="busy || !hasImage || !hasMask" @click="generate">{{ busy && pendingAction === 'hand_reconstruction_generate' ? '손 재구성 중…' : '손 재구성 후보 생성' }}</button>
      <button v-if="busy && pendingAction === 'hand_reconstruction_generate'" type="button" :disabled="cancelRequested" @click="cancelRemaining">{{ cancelRequested ? '취소 요청됨' : '후속 후보 취소' }}</button>
      <p v-if="!hasImage || !hasMask" class="hand-help">원본 이미지를 올리고 손 영역에 마스크를 그려주세요.</p>
      <p v-if="error" class="hand-error" role="alert">{{ error }}</p>
      <p class="hand-notice" role="status" aria-live="polite">{{ notice }}</p>
      <div v-if="result" class="hand-results">
        <label :for="`${id}-candidate`">비교할 후보</label>
        <select :id="`${id}-candidate`" :value="selectedIndex" :disabled="busy" @change="selectCandidate">
          <option v-for="candidate in result.candidates" :key="candidate.index" :value="candidate.index">후보 {{ candidate.index + 1 }} · seed {{ candidate.seed }}</option>
        </select>
        <img v-if="selected" class="hand-thumb" :src="selected.image" :alt="`손 재구성 후보 ${selected.index + 1}`" />
        <div class="hand-actions">
          <button type="button" :disabled="!selected" @click="openComparison">크게 전후 비교</button>
          <button type="button" :disabled="busy || !selected" @click="save">선택 후보 별도 PNG 저장</button>
        </div>
        <p class="hand-help">원본·마스크·재구성 설정을 바꾸면 이 결과는 만료됩니다. 보관할 후보는 먼저 저장하세요.</p>
      </div>
    </template>
    <dialog ref="comparisonDialog" class="hand-dialog" :aria-labelledby="`${id}-compare-title`" @close="comparisonOpen = false" @cancel="comparisonOpen = false" @keydown.stop>
      <template v-if="result && selected">
        <header><h2 :id="`${id}-compare-title`">손 형태 재구성 · 후보 {{ selected.index + 1 }}</h2><button type="button" autofocus @click="closeComparison">비교 닫기</button></header>
        <p class="hand-help">원본은 변경되지 않았습니다. 손가락뿐 아니라 손목 연결과 주변 물체도 확인하세요.</p>
        <p v-if="canvasSource" class="hand-warning">캔버스 스냅샷 기준 · 원본 메타데이터 미포함</p>
        <div class="hand-comparison">
          <figure><img :src="result.source" alt="수정하지 않은 원본 이미지" /><figcaption>원본</figcaption></figure>
          <figure><img :src="selected.image" :alt="`재구성 후보 ${selected.index + 1}`" /><figcaption>후보 {{ selected.index + 1 }} · seed {{ selected.seed }}</figcaption></figure>
        </div>
        <details class="hand-diagnostic"><summary>지운 영역 입력 보기 · 생성 전 진단</summary><figure><img :src="result.prepared" alt="기존 손 형태를 제거한 생성 입력 영역" /><figcaption>이 이미지는 완성 결과가 아니라 기존 손 형태를 제거한 입력 영역입니다.</figcaption></figure></details>
        <footer>
          <button v-for="candidate in result.candidates" :key="candidate.index" type="button" :aria-pressed="selectedIndex === candidate.index" :disabled="busy" @click="selectedIndex = candidate.index">후보 {{ candidate.index + 1 }}</button>
          <button type="button" :disabled="busy" @click="save">선택 후보 별도 PNG 저장</button>
        </footer>
        <p class="hand-notice" role="status" aria-live="polite">{{ notice }}</p>
        <p v-if="error" class="hand-error" role="alert">{{ error }}</p>
      </template>
    </dialog>
  </details>
</template>

<script setup lang="ts">
import { computed, nextTick, onDeactivated, onMounted, onUnmounted, reactive, ref, useId, watch } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { isWebMode } from '../utils/media.js'

const props = defineProps<{
  sourceRevision: number
  hasImage: boolean
  hasMask: boolean
  getInput: () => { image: string; mask: string; sourceKind?: 'original' | 'canvas' } | Promise<{ image: string; mask: string; sourceKind?: 'original' | 'canvas' }>
}>()
type SettingKey = 'strength' | 'candidates' | 'padding' | 'resolution' | 'feather'
interface Candidate { index: number; seed: number; image: string }
interface Result { source: string; prepared: string; candidates: Candidate[] }
const CONTROLS: Array<{ key: SettingKey; label: string; min: number; max: number; step: number; unit?: string }> = [
  { key: 'strength', label: '재구성 강도', min: 0.65, max: 1, step: 0.05 },
  { key: 'candidates', label: '후보 수', min: 1, max: 4, step: 1 },
  { key: 'padding', label: '주변 문맥 여백', min: 0, max: 256, step: 16, unit: 'px' },
  { key: 'resolution', label: '생성 영역 해상도', min: 512, max: 1024, step: 256, unit: 'px' },
  { key: 'feather', label: '마스크 안쪽 경계 완화', min: 0, max: 16, step: 1, unit: 'px' },
]
const id = `hand-reconstruction-${useId()}`
const webMode = ref(isWebMode())
const enabled = ref(false)
const settings = reactive<Record<SettingKey, number>>({ strength: 0.9, candidates: 2, padding: 64, resolution: 768, feather: 4 })
const prompt = ref('An anatomically coherent hand matching the existing gesture and wrist alignment. Preserve the interaction with nearby objects and the surrounding style.')
const canvasSource = ref(false)
const busy = ref(false)
const pendingAction = ref('')
const cancelRequested = ref(false)
const error = ref('')
const notice = ref('')
const result = ref<Result | null>(null)
const selectedIndex = ref(0)
const selected = computed(() => result.value?.candidates.find(candidate => candidate.index === selectedIndex.value))
const comparisonDialog = ref<HTMLDialogElement | null>(null)
const comparisonOpen = ref(false)
let pending = ''
let previewId = ''
/** 이 요청에 실제로 보낸 원본 — 비교 화면의 '원본'은 이것이다. 백엔드는 원본을 다시 보내지
 *  않는다(표시 전용 재인코딩이 완료 이벤트를 최대 ~49 MB까지 키웠다). requestId로 짝지어 둔다. */
let sentSource: { requestId: string; image: string } | null = null
let sequence = 0
let disposed = false
let disconnect: (() => void) | undefined
let timer: ReturnType<typeof setTimeout> | undefined
const nextId = () => `hand_${Date.now()}_${++sequence}_${Math.random().toString(36).slice(2, 10)}`
const isRaster = (value: unknown): value is string => typeof value === 'string' && /^data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$/.test(value)

function closeComparison() {
  if (comparisonDialog.value?.open) comparisonDialog.value.close()
  comparisonOpen.value = false
}
async function openComparison() {
  if (!result.value || !selected.value) return
  await nextTick()
  if (!result.value || !comparisonDialog.value || comparisonDialog.value.open) return
  comparisonDialog.value.showModal()
  comparisonOpen.value = true
}
function invalidate() {
  if (pending && pendingAction.value === 'hand_reconstruction_generate') requestAction('hand_reconstruction_cancel', { requestId: pending })
  if (previewId && previewId !== pending) requestAction('hand_reconstruction_cancel', { requestId: previewId })
  clearTimeout(timer)
  closeComparison()
  pending = previewId = ''
  sentSource = null
  pendingAction.value = ''
  result.value = null
  canvasSource.value = false
  busy.value = cancelRequested.value = false
  notice.value = ''
}
function setEnabled(event: Event) {
  enabled.value = (event.target as HTMLInputElement).checked && !isWebMode()
  if (!enabled.value) invalidate()
}
function setSetting(key: SettingKey, event: Event) {
  if (busy.value) return
  const control = CONTROLS.find(item => item.key === key)!
  const value = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(value)) return
  invalidate()
  settings[key] = Number(Math.max(control.min, Math.min(control.max, control.min + Math.round((value - control.min) / control.step) * control.step)).toFixed(2))
}
function setPrompt(event: Event) {
  if (busy.value) return
  invalidate()
  prompt.value = (event.target as HTMLTextAreaElement).value.slice(0, 4000)
}
function selectCandidate(event: Event) {
  if (busy.value) return
  const index = Number((event.target as HTMLSelectElement).value)
  if (result.value?.candidates.some(candidate => candidate.index === index)) selectedIndex.value = index
}
function armTimeout(requestId: string) {
  clearTimeout(timer)
  timer = setTimeout(() => {
    if (pending !== requestId) return
    invalidate()
    error.value = '서버 응답이 지연되어 결과 연결을 해제했습니다. 실행 중인 서버 요청은 끝날 수 있으니 서버 상태를 확인하세요.'
  }, pendingAction.value === 'hand_reconstruction_generate' ? 600000 : 60000)
}
function startRequest(requestId: string, action: string) {
  pending = requestId
  pendingAction.value = action
  busy.value = true
  armTimeout(requestId)
}
async function generate() {
  if (!enabled.value || busy.value || !props.hasImage || !props.hasMask || isWebMode()) return
  invalidate()
  error.value = ''
  const requestId = nextId()
  const revision = props.sourceRevision
  startRequest(requestId, 'hand_reconstruction_generate')
  notice.value = '원본과 마스크를 준비하고 있습니다…'
  try {
    const input = await props.getInput()
    if (disposed || pending !== requestId || props.sourceRevision !== revision) return
    if (cancelRequested.value) {
      invalidate()
      notice.value = '입력 준비 중 취소했습니다. 생성 요청을 보내지 않았습니다.'
      return
    }
    if (!isRaster(input.image) || !isRaster(input.mask)) throw Error('원본과 마스크를 PNG/JPEG/WebP 이미지로 읽지 못했습니다. 원본을 다시 올려주세요.')
    canvasSource.value = input.sourceKind === 'canvas'
    notice.value = '후보 생성 요청을 보냈습니다. 모델 로드에는 시간이 걸릴 수 있습니다.'
    sentSource = { requestId, image: input.image }
    requestAction('hand_reconstruction_generate', { requestId, image: input.image, mask: input.mask, settings: { enabled: true, ...settings }, prompt: prompt.value })
  } catch (exc) {
    if (pending !== requestId) return
    invalidate()
    error.value = String((exc as Error).message || exc)
  }
}
function cancelRemaining() {
  if (!pending || pendingAction.value !== 'hand_reconstruction_generate' || cancelRequested.value) return
  cancelRequested.value = true
  requestAction('hand_reconstruction_cancel', { requestId: pending })
  notice.value = '후속 후보 취소를 요청했습니다. 이미 실행 중인 서버 요청은 끝날 수 있습니다.'
}
function save() {
  if (!result.value || !selected.value || !previewId || busy.value || isWebMode()) return
  error.value = ''
  notice.value = '선택한 후보를 별도 PNG로 저장하고 있습니다…'
  const requestId = nextId()
  startRequest(requestId, 'hand_reconstruction_export')
  requestAction('hand_reconstruction_export', { requestId, previewRequestId: previewId, candidateIndex: selected.value.index })
}
function receive(raw: string) {
  let event: any
  try { event = JSON.parse(raw) } catch { return }
  if (!event || !pending || event.requestId !== pending || event.action !== pendingAction.value) return
  if (event.phase === 'progress' && event.ok) {
    armTimeout(pending)
    if (!cancelRequested.value) {
      const count = Number(event.count)
      const candidate = Number(event.candidate)
      const step = Number(event.step)
      const total = Number(event.total)
      notice.value = `후보 ${Number.isFinite(candidate) ? candidate : '?'} / ${Number.isFinite(count) ? count : settings.candidates} 생성 중${Number.isFinite(step) && total > 0 ? ` · ${step}/${total} 단계` : '…'}`
    }
    return
  }
  if (event.ok && event.phase !== 'complete') return
  const requestId = pending
  // 요청이 끝났으니 보관한 원본(수십 MB 문자열일 수 있음)은 결과로 옮기거나 버린다.
  const source = sentSource?.requestId === requestId ? sentSource.image : ''
  sentSource = null
  clearTimeout(timer)
  pending = ''
  pendingAction.value = ''
  busy.value = false
  if (!event.ok) { error.value = String(event.error || '손 재구성 작업에 실패했습니다.'); notice.value = ''; return }
  if (event.action === 'hand_reconstruction_generate') {
    const candidates = event.candidates
    if (!isRaster(source) || !isRaster(event.prepared) || !Array.isArray(candidates) || candidates.length === 0 || candidates.length > 4 ||
      !candidates.every(candidate => Number.isInteger(candidate.index) && candidate.index >= 0 && candidate.index < 4 && Number.isFinite(candidate.seed) && isRaster(candidate.image)) ||
      new Set(candidates.map(candidate => candidate.index)).size !== candidates.length) {
      error.value = '비교할 수 있는 올바른 후보 결과를 받지 못했습니다.'
      notice.value = ''
      return
    }
    result.value = { source, prepared: event.prepared, candidates }
    selectedIndex.value = candidates[0].index
    previewId = requestId
    notice.value = `${event.canceled ? '후속 생성을 취소했습니다. ' : ''}${candidates.length}개 후보가 준비되었습니다. 원본은 변경하지 않았습니다. 직접 비교하고 저장하세요.`
    if (event.warning) notice.value += ` ${String(event.warning)}`
  } else if (event.action === 'hand_reconstruction_export') notice.value = `별도 PNG로 저장했습니다: ${String(event.path || '')}`
  cancelRequested.value = false
}
watch(() => [props.sourceRevision, props.hasImage, props.hasMask], () => { invalidate(); error.value = '' })
onMounted(() => { webMode.value = isWebMode(); disconnect = onBackendEvent('handReconstructionEvent', receive) })
onDeactivated(closeComparison)
onUnmounted(() => { disposed = true; invalidate(); disconnect?.() })
</script>

<style scoped>
.hand-panel { min-width: 0; }
.hand-panel summary { cursor: pointer; color: var(--text-primary); font-size: 12px; font-weight: var(--fw-bold); }
.hand-panel :is(summary, input, textarea, select, button):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.hand-help, .hand-notice { color: var(--text-muted); font-size: 11px; line-height: 1.6; overflow-wrap: anywhere; margin: 8px 0; }
.hand-enable { display: flex; gap: 7px; align-items: center; color: var(--text-primary); font-size: 11px; }
.hand-enable input { width: auto; accent-color: var(--accent-fill); }
.hand-controls { min-width: 0; margin: 12px 0; padding: 8px; border: 1px solid var(--border); border-radius: 5px; }
.hand-controls legend { color: var(--text-secondary); font-size: 11px; padding: 0 4px; }
.hand-controls label, .hand-results > label { display: block; color: var(--text-secondary); font-size: 11px; }
.hand-controls label > span { display: flex; justify-content: space-between; gap: 5px; }
.hand-controls input { width: 100%; min-height: 28px; padding: 0; accent-color: var(--accent-fill); }
.hand-controls output { color: var(--accent); font-variant-numeric: tabular-nums; }
.hand-panel textarea, .hand-panel select { width: 100%; box-sizing: border-box; min-width: 0; color: var(--text-primary); background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; padding: 7px; font: inherit; font-size: 11px; }
.hand-panel textarea { margin-top: 6px; resize: vertical; }
.hand-panel button { min-height: 34px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg-button); color: var(--text-primary); font-size: 11px; cursor: pointer; white-space: normal; }
.hand-panel button:disabled { opacity: 0.5; cursor: default; }
.hand-panel button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); background: var(--accent-dim); }
.hand-panel .hand-run { width: 100%; background: var(--accent-fill); color: var(--on-accent); margin-bottom: 6px; }
.hand-warning { color: var(--state-warn-fg); font-size: 11px; line-height: 1.6; }
.hand-error { color: var(--state-alert-fg); font-size: 11px; line-height: 1.6; overflow-wrap: anywhere; }
.hand-thumb { display: block; width: 100%; height: 160px; object-fit: contain; background: var(--bg-input); margin: 8px 0; border-radius: 5px; }
.hand-actions { display: flex; flex-wrap: wrap; gap: 6px; }
.hand-actions button { flex: 1; }
.hand-dialog { width: min(1240px, calc(100vw - 40px)); max-width: none; max-height: calc(100dvh - 40px); box-sizing: border-box; padding: 18px; border: 1px solid var(--border); border-radius: 12px; color: var(--text-primary); background: var(--bg-card); overflow: auto; }
.hand-dialog::backdrop { background: rgba(0, 0, 0, 0.6); }
.hand-dialog header { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
.hand-dialog h2 { font-size: 15px; line-height: 1.5; margin: 0; }
.hand-dialog header button { flex-shrink: 0; }
.hand-comparison { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
.hand-dialog figure { min-width: 0; margin: 0; }
.hand-comparison img { display: block; width: 100%; height: min(60vh, 640px); object-fit: contain; background: var(--bg-input); }
.hand-dialog figcaption { font-size: 11px; color: var(--text-secondary); padding: 6px 0; }
.hand-diagnostic { margin: 12px 0; }
.hand-diagnostic img { display: block; max-width: 100%; max-height: 400px; object-fit: contain; margin-top: 10px; background: var(--bg-input); }
.hand-dialog footer { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
@media (max-width: 680px) { .hand-dialog { width: calc(100vw - 16px); padding: 12px; max-height: calc(100dvh - 16px); } .hand-comparison { grid-template-columns: minmax(0, 1fr); } .hand-comparison img { height: 50vh; } }
@media (pointer: coarse) { .hand-panel button, .hand-controls input, .hand-panel select { min-height: 44px; } }
@media (prefers-reduced-motion: reduce) { .hand-panel *, .hand-dialog { animation: none !important; transition: none !important; scroll-behavior: auto; } }
</style>
