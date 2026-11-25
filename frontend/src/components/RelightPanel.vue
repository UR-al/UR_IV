<template>
  <details class="glass-card relight-panel">
    <summary>실험 · 조명 편집</summary>
    <p class="relight-help">원본을 덮어쓰지 않는 CPU 기반 2.5D 후처리입니다. AI 모델 다운로드나 실제 3D 복원은 하지 않습니다. 결과를 확인한 뒤에만 I2I에 적용하거나 별도 PNG로 저장하세요.</p>
    <p v-if="webMode" class="relight-warning">이 실험 기능은 로컬 앱 전용입니다. 웹 모드에서는 실행·저장할 수 없습니다.</p>
    <label class="relight-enable"><input type="checkbox" :checked="enabled" :disabled="webMode" @change="setEnabled" /> 실험 기능 사용 (기본 꺼짐)</label>
    <template v-if="enabled && !webMode">
      <p class="relight-help">{{ geometryLabel }}. 깊이 맵이 있을 때만 투영 그림자를 계산합니다. 피부·선화·기존 그림자가 부자연스러워질 수 있습니다.</p>
      <fieldset class="relight-maps" :disabled="busy">
        <legend>선택 맵 · 원본과 같은 해상도 필요</legend>
        <label v-for="map in MAPS" :key="map.key">
          {{ map.label }} <span>{{ map.help }}</span>
          <input type="file" accept="image/png,image/jpeg,image/webp" @change="loadMap(map.key, $event)" />
          <span v-if="maps[map.key]">{{ mapNames[map.key] }} <button type="button" @click="clearMap(map.key)">제거</button></span>
        </label>
      </fieldset>
      <fieldset class="relight-controls" :disabled="busy">
        <legend>광원 · 반영 강도</legend>
        <label v-for="control in CONTROLS" :key="control.key" :for="`${id}-${control.key}`">
          <span>{{ control.label }} <output :for="`${id}-${control.key}`">{{ settings[control.key] }}</output></span>
          <input :id="`${id}-${control.key}`" type="range" :min="control.min" :max="control.max" :step="control.step"
            :value="settings[control.key]" :disabled="control.shadow && !maps.depth" @input="setSetting(control.key, $event)" />
        </label>
      </fieldset>
      <p class="relight-help">각도는 °, 그림자 길이·부드러움은 원본 픽셀 기준입니다. 기존 그림자 완화는 명암을 평탄화하는 근사이며 그림자를 정확히 제거하지는 않습니다.</p>
      <button class="relight-run" type="button" :disabled="busy || !imageSrc" @click="preview">{{ busy ? '처리 중…' : '미리보기 계산' }}</button>
      <p v-if="!imageSrc" class="relight-help">I2I 원본 이미지를 먼저 올리세요. 최대 16 MP, 정지 PNG/JPEG/WebP만 지원합니다.</p>
      <p v-if="error" class="relight-error" role="alert">{{ error }}</p>
      <p class="relight-notice" role="status" aria-live="polite">{{ notice }}</p>
      <template v-if="result">
        <div class="relight-comparison">
          <figure><img :src="beforeSrc" alt="조명 편집 전 원본" /><figcaption>원본</figcaption></figure>
          <figure><img :src="result.image" alt="조명 편집 미리보기" /><figcaption>편집 결과 · {{ result.width }}×{{ result.height }}</figcaption></figure>
        </div>
        <details class="relight-diagnostics">
          <summary>진단 맵 (보기용 최대 512px)</summary>
          <figure v-for="map in DIAGNOSTICS" :key="map.key">
            <img v-if="result.diagnostics?.[map.key]" :src="result.diagnostics[map.key]" :alt="map.label" />
            <figcaption>{{ map.label }}</figcaption>
          </figure>
        </details>
        <div class="relight-actions">
          <button type="button" :disabled="busy" @click="apply">결과를 I2I 원본으로 사용</button>
          <button type="button" :disabled="busy" @click="save">별도 PNG 저장</button>
        </div>
      </template>
    </template>
  </details>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, reactive, ref, useId, watch } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { isWebMode } from '../utils/media.js'

const props = defineProps<{ imageSrc: string }>()
const emit = defineEmits<{ apply: [result: { image: string; width: number; height: number }] }>()
type MapKey = 'depth' | 'normals' | 'mask'
type SettingKey = 'azimuth' | 'elevation' | 'strength' | 'ambient' | 'depth_scale' | 'shadow_strength' | 'shadow_length' | 'shadow_softness' | 'shadow_cleanup'
interface Result { image: string; width: number; height: number; geometry: string; diagnostics?: Record<string, string> }
const MAPS: Array<{ key: MapKey; label: string; help: string }> = [
  { key: 'depth', label: '깊이 맵', help: '흰색 = 가까움 · 검정 = 멂' },
  { key: 'normals', label: '노멀 맵', help: 'RGB = XYZ · +Y 위 · +Z 카메라 방향' },
  { key: 'mask', label: '적용 마스크', help: '흰색 = 적용 · 검정 = 원본 유지' },
]
const CONTROLS: Array<{ key: SettingKey; label: string; min: number; max: number; step: number; shadow?: boolean }> = [
  { key: 'azimuth', label: '광원 방향', min: -180, max: 180, step: 1 },
  { key: 'elevation', label: '광원 높이', min: 5, max: 85, step: 1 },
  { key: 'strength', label: '반영 강도', min: 0, max: 1, step: 0.05 },
  { key: 'ambient', label: '환경광', min: 0.1, max: 1, step: 0.05 },
  { key: 'depth_scale', label: '깊이 비율', min: 0.1, max: 4, step: 0.1 },
  { key: 'shadow_strength', label: '그림자 강도', min: 0, max: 1, step: 0.05, shadow: true },
  { key: 'shadow_length', label: '그림자 길이', min: 0, max: 128, step: 1, shadow: true },
  { key: 'shadow_softness', label: '그림자 부드러움', min: 0, max: 24, step: 1, shadow: true },
  { key: 'shadow_cleanup', label: '기존 그림자 완화', min: 0, max: 0.7, step: 0.05 },
]
const DIAGNOSTICS = [{ key: 'light', label: '조명 맵' }, { key: 'normals', label: '노멀 시각화' }, { key: 'shadow', label: '투영 그림자 (깊이 맵 필요)' }]
const id = `relight-${useId()}`
const webMode = ref(isWebMode())
const enabled = ref(false)
const settings = reactive<Record<SettingKey, number>>({ azimuth: -35, elevation: 45, strength: 0.5, ambient: 0.6,
  depth_scale: 1, shadow_strength: 0.2, shadow_length: 32, shadow_softness: 3, shadow_cleanup: 0 })
const maps = reactive<Record<MapKey, string>>({ depth: '', normals: '', mask: '' })
const mapNames = reactive<Record<MapKey, string>>({ depth: '', normals: '', mask: '' })
const busy = ref(false)
const error = ref('')
const notice = ref('')
const result = ref<Result | null>(null)
const beforeSrc = ref('')
const geometryLabel = computed(() => maps.normals ? '입력 노멀 기반 조명' : maps.depth ? '입력 깊이 기반 조명' : '명암 기반 근사 (실제 깊이 아님)')
let generation = 0
let pending = ''
let previewId = ''
let disposed = false
let disconnect: (() => void) | undefined
let timer: ReturnType<typeof setTimeout> | undefined
const nextId = () => `relight_${Date.now()}_${++generation}_${Math.random().toString(36).slice(2, 10)}`
function formatRelightError(exc: unknown, fallback: string): string {
  const raw = String((exc as Error)?.message || exc || '').trim()
  if (/[가-힣]/.test(raw)) return raw
  console.error('[RelightPanel]', exc)
  return fallback
}

function invalidate() {
  generation++
  if (pending) requestAction('relight_cancel', { requestId: pending })
  if (previewId && previewId !== pending) requestAction('relight_cancel', { requestId: previewId })
  clearTimeout(timer)
  pending = previewId = ''
  result.value = null
  busy.value = false
  notice.value = ''
}
function setEnabled(event: Event) {
  enabled.value = (event.target as HTMLInputElement).checked && !isWebMode()
  if (!enabled.value) invalidate()
}
function setSetting(key: SettingKey, event: Event) {
  if (busy.value) return
  invalidate()
  settings[key] = Number((event.target as HTMLInputElement).value)
}
function clearMap(key: MapKey) {
  invalidate()
  maps[key] = ''; mapNames[key] = ''
}
async function loadMap(key: MapKey, event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file || busy.value) return
  invalidate()
  error.value = ''
  const version = generation
  busy.value = true
  try {
    if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type) || file.size > 64 * 1024 * 1024) throw Error('64 MB 이하의 정지 PNG/JPEG/WebP 맵을 선택하세요.')
    const data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result || ''))
      reader.onerror = () => reject(Error('맵 파일을 읽지 못했습니다.'))
      reader.readAsDataURL(file)
    })
    if (disposed || version !== generation) return
    maps[key] = data; mapNames[key] = file.name
  } catch (exc) { if (version === generation) error.value = formatRelightError(exc, '맵 파일을 불러오지 못했습니다.') }
  finally { if (version === generation) busy.value = false }
}
async function sourceData(source: string): Promise<string> {
  if (/^data:image\/(png|jpeg|webp);base64,/.test(source)) return source
  // Copy only the currently selected, already-displayed image into a canvas.
  // Do not introduce a native arbitrary-path reader or any HTTP fetch.
  if (source.startsWith('file:///') || source.startsWith('blob:')) {
    return new Promise((resolve, reject) => {
      const image = new Image()
      image.onload = () => {
        if (image.naturalWidth * image.naturalHeight > 16_777_216 || Math.min(image.naturalWidth, image.naturalHeight) < 2) {
          reject(Error('최소 2×2, 최대 16 MP 이미지를 사용하세요.')); return
        }
        try {
          const canvas = document.createElement('canvas')
          canvas.width = image.naturalWidth; canvas.height = image.naturalHeight
          const context = canvas.getContext('2d')
          if (!context) throw Error('Canvas unavailable')
          context.drawImage(image, 0, 0)
          resolve(canvas.toDataURL('image/png'))
        } catch {
          reject(Error('현재 이미지의 픽셀 읽기가 차단되었습니다. 원본 파일을 I2I에 직접 업로드한 뒤 다시 시도하세요.'))
        }
      }
      image.onerror = () => reject(Error('현재 이미지를 읽지 못했습니다. 원본 파일을 I2I에 직접 업로드하세요.'))
      image.src = source
    })
  }
  throw Error('외부 이미지 URL은 읽지 않습니다. 원본 PNG/JPEG/WebP를 I2I에 업로드하세요.')
}
function startWait(requestId: string) {
  pending = requestId
  busy.value = true
  timer = setTimeout(() => {
    if (pending !== requestId) return
    invalidate()
    error.value = '조명 작업 응답이 지연되고 있습니다. 이전 CPU 작업이 끝난 뒤 다시 시도하세요.'
  }, 120000)
}
async function preview() {
  if (!enabled.value || busy.value || !props.imageSrc || isWebMode()) return
  invalidate()
  error.value = ''
  const requestId = nextId()
  const source = props.imageSrc
  startWait(requestId)
  try {
    const image = await sourceData(source)
    if (disposed || pending !== requestId || props.imageSrc !== source) return
    beforeSrc.value = image
    requestAction('relight_preview', { requestId, image, ...maps, settings: { ...settings } })
  } catch (exc) {
    if (pending !== requestId) return
    invalidate()
    error.value = formatRelightError(exc, '조명 미리보기 준비 중 오류가 발생했습니다.')
  }
}
function apply() {
  if (!result.value || busy.value) return
  emit('apply', { image: result.value.image, width: result.value.width, height: result.value.height })
}
function save() {
  if (!result.value || !previewId || busy.value || isWebMode()) return
  error.value = ''; notice.value = ''
  const requestId = nextId()
  startWait(requestId)
  requestAction('relight_export', { requestId, previewRequestId: previewId })
}
function receive(raw: string) {
  let event: any
  try { event = JSON.parse(raw) } catch { return }
  if (!event || !pending || event.requestId !== pending || event.action === 'relight_cancel') return
  const requestId = pending
  clearTimeout(timer)
  pending = ''; busy.value = false
  if (!event.ok) { error.value = String(event.error || '조명 작업에 실패했습니다.'); return }
  if (event.action === 'relight_preview') {
    if (!/^data:image\/png;base64,/.test(String(event.image || ''))) { error.value = '올바르지 않은 조명 결과입니다.'; return }
    result.value = event as Result
    previewId = requestId
    notice.value = '미리보기만 계산했습니다. 원본은 아직 변경하지 않았습니다.'
  } else if (event.action === 'relight_export') notice.value = `별도 PNG로 저장했습니다: ${event.path}`
}
watch(() => props.imageSrc, () => {
  invalidate()
  for (const key of ['depth', 'normals', 'mask'] as const) { maps[key] = ''; mapNames[key] = '' }
  error.value = ''
})
onMounted(() => { webMode.value = isWebMode(); disconnect = onBackendEvent('relightEvent', receive) })
onUnmounted(() => { disposed = true; invalidate(); disconnect?.() })
</script>

<style scoped>
.relight-panel { min-width: 0; }
.relight-panel summary { cursor: pointer; color: var(--text-primary); font-size: 12px; font-weight: var(--fw-bold); }
.relight-panel summary:focus-visible, .relight-panel :is(input, button):focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.relight-help, .relight-notice { font-size: 11px; line-height: 1.5; color: var(--text-muted); overflow-wrap: anywhere; margin: 8px 0; }
.relight-enable { display: flex; gap: 7px; align-items: center; color: var(--text-primary); font-size: 11px; }
.relight-enable input { width: auto; accent-color: var(--accent-fill); }
.relight-panel fieldset { min-width: 0; margin: 12px 0; padding: 8px; border: 1px solid var(--border); border-radius: 5px; }
.relight-panel legend { color: var(--text-secondary); font-size: 11px; padding: 0 4px; }
.relight-maps label { display: block; margin: 4px 0 10px; font-size: 11px; color: var(--text-primary); }
.relight-maps label span { display: block; color: var(--text-muted); overflow-wrap: anywhere; line-height: 1.5; }
.relight-maps input[type="file"] { width: 100%; min-width: 0; box-sizing: border-box; color: var(--text-secondary); font-size: 10px; }
.relight-controls label { display: block; color: var(--text-secondary); font-size: 11px; }
.relight-controls label > span { display: flex; justify-content: space-between; gap: 5px; }
.relight-controls input { width: 100%; min-height: 28px; padding: 0; accent-color: var(--accent-fill); }
.relight-controls output { color: var(--accent); font-variant-numeric: tabular-nums; }
.relight-panel button { min-height: 32px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg-button); color: var(--text-primary); font-size: 11px; cursor: pointer; white-space: normal; }
.relight-panel button:disabled { opacity: 0.5; cursor: default; }
.relight-panel .relight-run { width: 100%; background: var(--accent-fill); color: var(--on-accent); }
.relight-warning { color: var(--state-warn-fg); font-size: 11px; line-height: 1.5; }
.relight-error { color: var(--state-alert-fg); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.relight-comparison { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 6px; }
.relight-panel figure { min-width: 0; margin: 8px 0; }
.relight-panel figure img { width: 100%; height: 140px; object-fit: contain; background: var(--bg-input); border-radius: 4px; }
.relight-panel figcaption { font-size: 10px; color: var(--text-muted); line-height: 1.5; }
.relight-diagnostics { margin: 10px 0; }
.relight-diagnostics summary { font-size: 11px; }
.relight-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.relight-actions button { flex: 1; }
@media (pointer: coarse) { .relight-panel button, .relight-controls input { min-height: 44px; } }
</style>
