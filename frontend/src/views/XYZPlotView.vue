<template>
  <div class="xyz-view">
    <div class="config-panel">
      <h3>XYZ Plot</h3>
      <div class="capability-toolbar">
        <span>{{ backendName || '백엔드 미연결' }}</span>
        <button type="button" class="btn-export" :disabled="loading" @click="refreshCapabilities">{{ loading ? '확인 중…' : '축 새로고침' }}</button>
      </div>
      <p v-if="error" class="axis-error" role="alert">{{ error }}</p>
      <p v-if="!loading && !axisOptions.length && !error" class="axis-note">현재 생성 모드에서 사용할 수 있는 XYZ 축이 없습니다.</p>
      <fieldset :disabled="loading || !capabilityId" class="axes-fieldset">
      <div v-for="(axis, ai) in axes" :key="ai" class="axis-config">
        <label class="axis-label">{{ axis.name }} 축</label>
        <CustomSelect v-model="axis.type" :options="['', ...axisOptions]" placeholder="None" @update:model-value="resetAxis(axis)" />
        <input v-if="definitionFor(axis)?.type === 'replace'" class="s-input" v-model="axis.search" placeholder="찾을 문자열 (S/R 검색어)" :aria-label="`${axis.name} 검색어`" />
        <CustomSelect v-if="definitionFor(axis)?.type === 'choice'" :model-value="''" :options="definitionFor(axis)?.choices || []"
          placeholder="서버에서 사용 가능한 값 추가" @update:model-value="value => addChoice(axis, String(value))" />
        <input v-if="axis.type" class="s-input" v-model="axis.values"
          :placeholder="definitionFor(axis)?.type === 'replace' ? '대체값1, 대체값2 (검색어를 넣으면 원본 유지)' : '값1, 값2 또는 20-40:5'"
          :aria-label="`${axis.name} 축 값`"
        />
        <small v-if="definitionFor(axis)?.min !== undefined" class="axis-note">허용 범위 {{ definitionFor(axis)?.min }}–{{ definitionFor(axis)?.max }}{{ ['Width', 'Height'].includes(axis.type) ? ' · 8 단위' : '' }}</small>
      </div>
      </fieldset>
      <p v-if="validationError" class="axis-error" role="alert">{{ validationError }}</p>
      <div class="combo-info">
        조합 수: <span class="accent">{{ comboCount }}</span> / 256
      </div>
      <button class="btn-gen" @click="startPlot" :disabled="!capabilityId || loading || submitting || comboCount === 0 || comboCount > 256 || !!validationError">
        XYZ Plot 시작
      </button>
      <p v-for="note in notes" :key="note" class="axis-note">{{ note }}</p>
      <details v-if="unsupported.length" class="axis-note"><summary>앱에서 실행하지 않는 서버 확장 축 ({{ unsupported.length }})</summary>{{ unsupported.join(', ') }}</details>
    </div>
    <div class="result-area">
      <div v-if="resultImages.length === 0" class="empty">
        <!-- 빈 상태 일러스트 — 격자 + X/Y 축 + 점들을 SVG로 -->
        <svg class="empty-illustration" viewBox="0 0 320 240" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
          <defs>
            <pattern id="xyz-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1a1a1a" stroke-width="0.5"/>
            </pattern>
            <linearGradient id="xyz-axis" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#FACC15" stop-opacity="0.05"/>
              <stop offset="50%" stop-color="#FACC15" stop-opacity="0.3"/>
              <stop offset="100%" stop-color="#FACC15" stop-opacity="0.05"/>
            </linearGradient>
          </defs>
          <!-- 격자 배경 -->
          <rect width="320" height="240" fill="url(#xyz-grid)"/>
          <!-- 축 -->
          <line x1="40" y1="200" x2="280" y2="200" stroke="url(#xyz-axis)" stroke-width="1.5"/>
          <line x1="40" y1="40" x2="40" y2="200" stroke="url(#xyz-axis)" stroke-width="1.5"/>
          <!-- 축 라벨 -->
          <text x="282" y="204" fill="#FACC15" font-size="9" font-weight="900" opacity="0.4" font-family="Consolas, monospace">X</text>
          <text x="36" y="36" fill="#FACC15" font-size="9" font-weight="900" opacity="0.4" font-family="Consolas, monospace">Y</text>
          <!-- 가상 데이터 포인트 (얇은 grid 노드들) -->
          <g opacity="0.25">
            <circle cx="80"  cy="170" r="3" fill="#FACC15"/>
            <circle cx="120" cy="140" r="3" fill="#FACC15"/>
            <circle cx="160" cy="105" r="3" fill="#FACC15"/>
            <circle cx="200" cy="80"  r="3" fill="#FACC15"/>
            <circle cx="240" cy="60"  r="3" fill="#FACC15"/>
          </g>
          <!-- 연결선 (그리드 다이어그램 느낌) -->
          <polyline points="80,170 120,140 160,105 200,80 240,60"
            fill="none" stroke="#FACC15" stroke-width="1" opacity="0.2"
            stroke-dasharray="3,3"/>
        </svg>
        <div class="empty-text">
          <span class="empty-title">축을 설정하고 시작하세요</span>
          <span class="empty-sub">X·Y·Z 조합이 격자에 펼쳐집니다</span>
        </div>
      </div>
      <div v-else>
        <div class="result-actions">
          <span class="result-count">결과 {{ resultImages.length }}장</span>
          <button class="btn-export" @click="exportCSV" title="축값 + 경로 + 라벨을 CSV로 저장"><Icon name="layers" /> CSV 내보내기
          </button>
        </div>
        <div class="result-grid">
          <div v-for="(img, i) in resultImages" :key="i" class="result-item">
            <img :src="mediaUrl(img.path)" />
            <div class="result-label">{{ img.label }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onActivated, onDeactivated, onUnmounted } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { mediaUrl } from '../utils/media.js'
import CustomSelect from '../components/CustomSelect.vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { acceptCapabilityEvent, parseAxisValues, quoteAxisValue, type XYZAxisDefinition } from '../utils/xyzPlot'

interface Axis {
  name: string
  type: string
  values: string
  search: string
}

interface ResultImage {
  path: string
  label: string
  [k: string]: any
}

const definitions = ref<XYZAxisDefinition[]>([])
const axisOptions = computed(() => definitions.value.map(axis => axis.label))
const capabilityId = ref('')
const backendName = ref('')
const loading = ref(false)
const submitting = ref(false)
const error = ref('')
const notes = ref<string[]>([])
const unsupported = ref<string[]>([])
let queryId = '', plotId = '', bridgeReady = false, active = true, disposed = false
let queryTimer: ReturnType<typeof setTimeout> | undefined
let plotTimer: ReturnType<typeof setTimeout> | undefined
const disconnects: Array<() => void> = []

const axes = reactive<Axis[]>([
  { name: 'X', type: '', values: '', search: '' },
  { name: 'Y', type: '', values: '', search: '' },
  { name: 'Z', type: '', values: '', search: '' },
])
const resultImages = ref<ResultImage[]>([])

const definitionFor = (axis: Axis) => definitions.value.find(item => item.label === axis.type)
function resetAxis(axis: Axis) { axis.values = ''; axis.search = '' }
function addChoice(axis: Axis, value: string) { axis.values += `${axis.values.trim() ? ', ' : ''}${quoteAxisValue(value)}` }
const parsed = computed(() => {
  try {
    const selected = axes.filter(axis => axis.type)
    const used = new Set<string>()
    const data = selected.map(axis => {
      const definition = definitionFor(axis)
      if (!definition || used.has(definition.id)) throw new Error('서로 다른 지원 축을 선택하세요.')
      used.add(definition.id)
      const values = parseAxisValues(axis.values, definition.type)
      if (!values.length) throw new Error(`${axis.name} 축 값을 입력하세요.`)
      if (definition.type === 'replace' && !axis.search) throw new Error(`${axis.name} S/R 검색어를 입력하세요.`)
      for (const value of values) {
        if (definition.type === 'choice' && !definition.choices?.includes(value)) throw new Error(`${axis.name}: 서버 목록에서 값을 선택하세요.`)
        if (definition.type === 'integer' || definition.type === 'number') {
          const number = Number(value)
          if (!Number.isFinite(number) || number < (definition.min ?? -Infinity) || number > (definition.max ?? Infinity)
            || (definition.type === 'integer' && !Number.isInteger(number))
            || (['width', 'height'].includes(definition.id) && number % 8)) throw new Error(`${axis.name}: 허용 범위와 간격을 확인하세요.`)
        }
      }
      return { id: definition.id, search: axis.search, values }
    })
    const count = data.length ? data.reduce((total, axis) => total * axis.values.length, 1) : 0
    if (count > 256) throw new Error('XYZ 조합은 최대 256개입니다.')
    return { data, count, error: '' }
  } catch (problem: any) { return { data: [], count: 0, error: problem.message } }
})
const comboCount = computed(() => parsed.value.count)
const validationError = computed(() => parsed.value.error)

function refreshCapabilities() {
  if (!bridgeReady || disposed) return
  clearTimeout(queryTimer)
  queryId = `xyz-query-${crypto.randomUUID()}`
  capabilityId.value = ''; definitions.value = []; error.value = ''; loading.value = true
  requestAction('get_xyz_capabilities', { requestId: queryId })
  queryTimer = setTimeout(() => { loading.value = false; queryId = ''; error.value = '백엔드 응답이 지연되고 있습니다. 연결을 확인하고 새로고침하세요.' }, 50000)
}
function startPlot() {
  if (validationError.value || !comboCount.value || !capabilityId.value) return
  plotId = `xyz-plot-${crypto.randomUUID()}`
  submitting.value = true; error.value = ''; resultImages.value = []
  requestAction('start_xyz_plot', { requestId: plotId, capabilityId: capabilityId.value, axes: parsed.value.data })
  clearTimeout(plotTimer)
  plotTimer = setTimeout(() => { submitting.value = false; error.value = '대기열 등록 응답이 지연되고 있습니다. 대기열을 확인하세요.' }, 15000)
}
onMounted(() => {
  disconnects.push(onBackendEvent('xyzCapabilitiesReceived', (raw: string) => {
    let event: any
    try { event = JSON.parse(raw) } catch { return }
    if (!acceptCapabilityEvent(queryId, event)) return
    if (event.invalidated) {
      definitions.value = []; capabilityId.value = ''
      axes.forEach(axis => { axis.type = ''; resetAxis(axis) })
      if (active) refreshCapabilities()
      return
    }
    clearTimeout(queryTimer); loading.value = false; queryId = ''
    error.value = event.ok ? '' : event.error || '백엔드 기능을 읽지 못했습니다.'
    definitions.value = event.ok && Array.isArray(event.axes) ? event.axes : []
    capabilityId.value = event.ok ? event.capabilityId : ''
    backendName.value = event.backend === 'comfyui' ? 'ComfyUI' : event.backend === 'webui' ? 'Forge / WebUI' : ''
    notes.value = event.notes || []; unsupported.value = event.unsupported || []
    for (const axis of axes) if (!definitionFor(axis)) { axis.type = ''; resetAxis(axis) }
  }))
  disconnects.push(onBackendEvent('xyzPlotEvent', (raw: string) => {
    let event: any
    try { event = JSON.parse(raw) } catch { return }
    if (!plotId || event.requestId !== plotId) return
    if (event.type === 'result' && event.path) { resultImages.value.push({ path: event.path, label: event.label || '', axes: event.axes }); return }
    clearTimeout(plotTimer); submitting.value = false
    if (!event.ok) error.value = event.error || 'XYZ 대기열 등록에 실패했습니다.'
  }))
  void getBackend().then(() => { bridgeReady = true; refreshCapabilities() })
})
onActivated(() => { active = true; if (bridgeReady) refreshCapabilities() })
onDeactivated(() => { active = false })
onUnmounted(() => { disposed = true; clearTimeout(queryTimer); clearTimeout(plotTimer); disconnects.forEach(disconnect => disconnect()) })

// CSV 내보내기 — 결과 이미지 + 축 메타데이터 (라벨에서 축값 파싱)
function exportCSV() {
  if (!resultImages.value.length) return
  // CSV escape
  const esc = (v: any) => {
    const s = String(v ?? '')
    if (s.includes(',') || s.includes('"') || s.includes('\n')) {
      return '"' + s.replace(/"/g, '""') + '"'
    }
    return s
  }
  const headers = ['index', 'path', 'label']
  const axisHeaders = axes.filter(a => a.type && a.values.trim()).map(a => a.type)
  headers.push(...axisHeaders)
  const lines = [headers.map(esc).join(',')]
  resultImages.value.forEach((img, i) => {
    // 라벨은 보통 "축타입=값, ..." 형식 — 파싱 시도
    const axisValues: Record<string, string> = { ...(img.axes || {}) }
    if (img.label && !img.axes) {
      img.label.split(/\s*[,;]\s*/).forEach((part: string) => {
        const [k, ...vparts] = part.split('=')
        if (k && vparts.length) axisValues[k.trim()] = vparts.join('=').trim()
      })
    }
    const row = [i + 1, img.path || '', img.label || '']
    for (const ah of axisHeaders) row.push(axisValues[ah] || '')
    lines.push(row.map(esc).join(','))
  })
  // BOM + UTF-8로 Excel 호환
  const csv = '﻿' + lines.join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  a.download = `xyz_plot_${ts}.csv`
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  requestAction('show_toast', { type: 'success', msg: `CSV 저장됨 (${resultImages.value.length}건)` })
}
</script>

<style scoped>
.xyz-view { width: 100%; height: 100%; display: flex; }
.config-panel {
  width: 320px; padding: 16px; border-right: 1px solid var(--rule);
  display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
}
.config-panel h3 { color: var(--text-primary); font-size: 14px; margin: 0; }
.axis-config { display: flex; flex-direction: column; gap: 4px; }
.axis-label { color: var(--accent); font-size: 12px; font-weight: var(--fw-bold); }
.axes-fieldset { padding: 0; margin: 0; border: 0; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.axes-fieldset:disabled { opacity: .55; pointer-events: none; }
.capability-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 11px; }
.axis-note { margin: 0; color: var(--text-muted); font-size: 11px; line-height: 1.5; overflow-wrap: anywhere; }
.axis-error { margin: 0; color: var(--state-alert-fg); font-size: 12px; line-height: 1.5; }
@media (max-width: 640px) { .xyz-view { flex-direction: column; overflow-y: auto; } .config-panel { box-sizing: border-box; width: 100%; border-right: 0; border-bottom: 1px solid var(--rule); flex-shrink: 0; } .result-area { min-height: 280px; flex-shrink: 0; } }
.s-select, .s-input {
  background: var(--bg-input); border: 1px solid var(--border); border-radius: 4px; padding: 6px 8px;
  color: var(--text-primary); font-size: 12px; outline: none; caret-color: var(--accent);
}
.s-input:focus { border-color: var(--accent); }
.s-input::selection { background: rgba(226, 179, 64, 0.3); }
.s-select:focus { border-color: var(--accent); }
.combo-info { color: var(--text-muted); font-size: 12px; text-align: center; }
.accent { color: var(--accent); font-weight: var(--fw-bold); }
/* 주 버튼 — 면은 accent-fill, 글자는 on-accent 여야 사용자가 어떤 강조색을 골라도 읽힌다 */
.btn-gen {
  padding: 12px; background: var(--accent-fill); border: none; border-radius: 6px;
  color: var(--on-accent); font-weight: var(--fw-bold); cursor: pointer;
}
.btn-gen:disabled { opacity: 0.35; cursor: not-allowed; }
.result-area { flex: 1; overflow-y: auto; padding: 8px; }
.result-actions {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 4px 12px; gap: 8px;
}
.result-count { font-size: 12px; color: var(--text-muted); font-weight: var(--fw-bold); }
.btn-export {
  padding: 6px 14px; background: var(--bg-button); border: 1px solid var(--rule);
  border-radius: 6px; color: var(--accent); font-size: 11px; font-weight: var(--fw-bold);
  cursor: pointer; transition: all 0.15s;
}
.btn-export:hover { background: var(--bg-button-hover); border-color: var(--accent); }
.empty {
  width: 100%; height: 100%;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 20px;
  color: var(--text-muted); font-size: 14px;
}
.empty-illustration {
  width: 320px; max-width: 60%; height: auto;
  opacity: 0.7; animation: empty-pulse 4s ease-in-out infinite;
}
@keyframes empty-pulse {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 0.85; }
}
.empty-text { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.empty-title { font-size: 14px; color: var(--text-secondary); font-weight: var(--fw-bold); }
.empty-sub { font-size: 11px; color: var(--text-muted); }
.result-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 4px; align-content: start;
}
.result-item { border-radius: 4px; overflow: hidden; border: 1px solid var(--rule); }
.result-item img { width: 100%; aspect-ratio: 1; object-fit: cover; }
.result-label { font-size: var(--fs-label); color: var(--text-muted); padding: 4px 6px; text-align: center; }
</style>
