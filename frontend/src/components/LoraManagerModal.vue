<template>
  <div class="lm-overlay" @mousedown.self="close">
    <div class="lm-modal">
      <div class="lm-header">
        <div>
          <h3>LoRA 매니저</h3>
          <span class="lm-sub">
            {{ extMode ? 'sam-extra 임베드 매니저 (civitai 다운로드 · 메타데이터 · 레시피)'
                       : '설치된 LoRA를 검색해서 스택에 추가합니다' }}
          </span>
        </div>
        <div class="lm-head-actions">
          <button class="lm-modetab" :class="{ active: !extMode }" @click="extMode = false">간편</button>
          <button class="lm-modetab" :class="{ active: extMode }" @click="openExtManager">확장 매니저</button>
          <button class="lm-close" @click="close"><Icon name="close" /></button>
        </div>
      </div>

      <!-- sam-extra 임베드 LoRA Manager (워크플로 4) -->
      <div v-if="extMode" class="lm-ext">
        <div v-if="extLoading" class="lm-empty">LoRA Manager 서버를 여는 중…</div>
        <div v-else-if="extError" class="lm-empty lm-error">
          {{ extError }}
          <button class="lm-refresh mt-8" @click="openExtManager">다시 시도</button>
        </div>
        <iframe v-else-if="extUrl" :src="extUrl" class="lm-iframe"
          referrerpolicy="no-referrer" />
      </div>

      <template v-else>
      <div class="lm-searchbar">
        <input ref="searchEl" v-model="query" class="lm-search" placeholder="LoRA 이름 검색..." />
        <button class="lm-refresh" @click="load('force')" :disabled="loading" title="목록 다시 스캔"><Icon v-if="!loading" name="refresh" /><template v-else>…</template></button>
      </div>

      <div class="lm-list">
        <div v-if="loading" class="lm-empty">로딩 중...</div>
        <div v-else-if="loadError && !loras.length" class="lm-empty lm-error">{{ loadError }}</div>
        <div v-else-if="!filtered.length" class="lm-empty">
          {{ loras.length ? '검색 결과 없음' : '설치된 LoRA가 없습니다 (백엔드 연결 확인)' }}
        </div>
        <!-- 다시 스캔이 실패해도 이전 목록은 남긴다 — 대신 실패를 알린다 -->
        <div v-if="!loading && loadError && loras.length" class="lm-availability-note lm-load-error">{{ loadError }}</div>
        <div v-if="unavailableCount" class="lm-availability-note">
          현재 실행 백엔드에서 보이지 않는 {{ unavailableCount }}개 항목은 확인만 가능하며 추가할 수 없습니다.
        </div>
        <section v-for="section in filteredSections" :key="section.key" class="lm-section">
          <header class="lm-section-header">
            <div>
              <strong>{{ section.label }}</strong>
              <span>{{ section.description }}</span>
            </div>
            <span class="lm-section-count">{{ section.items.length }}</span>
          </header>
          <div v-for="l in section.items" :key="l.id || `${l.source || 'main'}:${l.runtimeName || l.name}`"
            class="lm-item" :class="{ unavailable: l.backendAvailable === false }">
            <div class="lm-item-main">
              <div class="lm-name-row">
                <div class="lm-name" :title="l.runtimeName || l.name">{{ l.label || l.name }}</div>
                <span v-if="isPrimaryLora(l)" class="lm-source-badge main">메인</span>
                <span v-if="l.source" class="lm-source-badge">{{ l.sourceName || sourceLabel(l.source) }}</span>
                <span v-if="l.group && !isPrimaryLora(l)" class="lm-source-badge secondary">{{ groupLabel(l.group) }}</span>
                <span v-if="l.nameConflict" class="lm-source-badge conflict">이름 충돌</span>
              </div>
              <div v-if="l.backendAvailable === false" class="lm-unavailable">
                현재 실행 백엔드에서는 이 LoRA 경로를 사용할 수 없습니다.
              </div>
              <div class="lm-triggers" v-if="l.triggerWords && l.triggerWords.length">
                <span v-for="tw in l.triggerWords.slice(0, 8)" :key="tw" class="lm-tw">{{ tw }}</span>
              </div>
            </div>
            <input type="number" v-model.number="l._w" step="0.05" min="-2" max="3" class="lm-weight"
              title="가중치" :disabled="l.backendAvailable === false" />
            <button class="lm-add" :disabled="l.backendAvailable === false"
              :title="l.backendAvailable === false ? '현재 백엔드에서 사용할 수 없습니다' : `${l.runtimeName || l.name} 추가`"
              @click="add(l)">+ 추가</button>
          </div>
        </section>
      </div>

      <!-- 일괄 붙여넣기 (<lora:name:weight> 텍스트) -->
      <details class="lm-batch">
        <summary>일괄 붙여넣기 (&lt;lora:이름:가중치&gt;)</summary>
        <textarea v-model="batchText" class="lm-batch-text" rows="3"
          placeholder="<lora:my_style:0.8>, <lora:char_a:1.0> ..."></textarea>
        <button class="lm-batch-btn" @click="applyBatch">붙여넣은 LoRA 추가</button>
      </details>

      <div class="lm-footer">
        <span class="lm-count">{{ filtered.length }} / {{ loras.length }} LoRA</span>
        <div class="lm-foot-spacer"></div>
        <button class="lm-done" @click="close">닫기</button>
      </div>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import type { LorasReadyPayload } from '../types/bridge'
import { createLatestRequest, wasAbandoned } from '../utils/bridgeRequest'
import { useModalLayer } from '../composables/useModalLayer'

interface LoraItem {
  name: string
  label?: string
  triggerWords: string[]
  source?: string
  sourceName?: string
  group?: string
  primary?: boolean
  backendAvailable?: boolean
  runtimeName?: string
  nameConflict?: boolean
  _w?: number
  [k: string]: any
}

interface LoraSection {
  key: string
  label: string
  description: string
  items: LoraItem[]
}

const emit = defineEmits<{
  close: []
  add: [payload: { name: string; weight: number; triggerWords: string[] }]
}>()

const loras = ref<LoraItem[]>([])
const query = ref('')
const loading = ref(false)
const loadError = ref('')
const batchText = ref('')
// 목록은 비동기로 받는다(requestLoras → lorasReady). 예전 동기 getLoras 는 첫 열기·다시 스캔마다
// GUI 스레드에서 백엔드 HTTP 와 디스크 카탈로그 병합을 해 앱 전체가 수 초 멈췄다.
// 마지막 요청만 유효 — '다시 스캔'을 연달아 눌러도 늦게 온 옛 목록이 새 목록을 덮지 않는다.
const loraRequest = createLatestRequest<LorasReadyPayload>({ timeoutMs: 120_000, prefix: 'loras' })
let disconnectLorasReady: (() => void) | null = null
const searchEl = ref<HTMLInputElement | null>(null)

// ── sam-extra 임베드 LoRA Manager (워크플로 4) ──────────────────────────────
// 확장이 Forge FastAPI에 등록한 /sam3-lora/spawn 이 aiohttp 서버를 lazy spawn 하고
// URL을 돌려준다. 그 URL을 iframe으로 띄우면 civitai 다운로드·메타데이터 편집·
// 트리거워드·레시피가 그대로 들어온다 — 앱에서 새로 만들 게 없다.
const extMode = ref(false)
const extUrl = ref('')
const extLoading = ref(false)
const extError = ref('')
let disconnectExtUrlReady: (() => void) | null = null

async function openExtManager() {
  extMode.value = true
  if (extUrl.value) return          // 이미 열어둔 서버 재사용
  extLoading.value = true
  extError.value = ''
  try {
    const backend: any = await getBackend()
    if (!backend?.requestLoraManagerUrl) {
      extError.value = 'LoRA Manager 임베드를 지원하지 않는 백엔드입니다'
      extLoading.value = false
      return
    }
    backend.requestLoraManagerUrl()
  } catch (e) {
    extError.value = String(e)
    extLoading.value = false
  }
}

function onExtUrlReady(json: string) {
  extLoading.value = false
  try {
    const r = JSON.parse(json)
    if (r.url) { extUrl.value = r.url; extError.value = '' }
    else extError.value = r.message || 'LoRA Manager를 열 수 없습니다'
  } catch {
    extError.value = 'LoRA Manager 응답을 해석하지 못했습니다'
  }
}

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  return q ? loras.value.filter(l => [l.label, l.name, l.runtimeName, l.source, l.sourceName, l.group, ...(l.triggerWords || [])]
    .some(value => String(value || '').toLowerCase().includes(q))) : loras.value
})

function isPrimaryLora(lora: LoraItem) {
  const group = String(lora.group || '').toLowerCase()
  return Boolean(lora.primary || ['main', 'primary', 'shared'].includes(group))
}

function sourceLabel(source: string) {
  const normalized = source.toLowerCase()
  if (normalized === 'forge') return 'FORGE NEO'
  if (normalized === 'comfyui' || normalized === 'comfy') return 'COMFYUI'
  return source.toUpperCase()
}

function groupLabel(group: string) {
  const normalized = group.toLowerCase().replace(/[_-]+/g, ' ')
  if (normalized.includes('unique') || normalized.includes('secondary')) return 'UNIQUE'
  return group.toUpperCase()
}

const filteredSections = computed<LoraSection[]>(() => {
  const main = filtered.value.filter(isPrimaryLora)
  const secondary = filtered.value.filter(lora => !isPrimaryLora(lora))
  const sections: LoraSection[] = []
  if (main.length) {
    sections.push({
      key: 'main', label: 'MAIN LIBRARY',
      description: '기본 모델 라이브러리에서 공유되는 LoRA', items: main,
    })
  }
  const bySource = new Map<string, LoraItem[]>()
  for (const lora of secondary) {
    const source = String(lora.source || lora.group || 'secondary')
    const entries = bySource.get(source) || []
    entries.push(lora)
    bySource.set(source, entries)
  }
  for (const [source, items] of bySource) {
    const displaySource = items[0]?.sourceName || sourceLabel(source)
    sections.push({
      key: `secondary:${source}`, label: `${displaySource} UNIQUE`,
      description: '메인 라이브러리와 중복되지 않는 백엔드 전용 LoRA', items,
    })
  }
  return sections
})

const unavailableCount = computed(() => filtered.value.filter(lora => lora.backendAvailable === false).length)

async function load(mode = '') {
  loading.value = true
  loadError.value = ''
  const { id, done, outcome } = loraRequest.begin()
  const backend: any = await getBackend()   // QWebChannel 백엔드 — 동적 타입
  if (wasAbandoned(outcome())) return   // 백엔드를 기다리는 사이 모달이 닫혔거나 새 요청이 시작됐다
  if (!backend?.requestLoras) {
    loraRequest.cancel()
    loading.value = false
    loadError.value = 'LoRA 목록을 불러올 수 없습니다 (백엔드 연결 확인)'
    return
  }
  backend.requestLoras(mode, id)
  const reply = await done
  // 더 새 요청이 이어받았거나(그쪽이 마무리한다) 모달이 닫혀 버린 요청 — '응답 없음'으로 보이지 않는다.
  if (wasAbandoned(outcome())) return
  loading.value = false
  if (!reply) { loadError.value = 'LoRA 목록 응답이 없습니다 — 잠시 뒤 다시 스캔해 보세요'; return }
  if (reply.error) { loadError.value = `LoRA 목록을 불러오지 못했습니다: ${reply.error}`; return }
  if (Array.isArray(reply.loras)) loras.value = reply.loras.map((l: any) => ({ ...l, _w: 1.0 }))
}

function add(l: LoraItem) {
  if (l.backendAvailable === false) return
  emit('add', { name: l.runtimeName || l.name, weight: (typeof l._w === 'number' ? l._w : 1.0), triggerWords: l.triggerWords || [] })
}

function applyBatch() {
  const re = /<lora:([^:>]+):([-\d.]+)>/gi
  let m: RegExpExecArray | null = null
  let n = 0
  while ((m = re.exec(batchText.value)) !== null) {
    const name = m[1].trim()
    const w = parseFloat(m[2]) || 1.0
    if (name) { emit('add', { name, weight: w, triggerWords: [] }); n++ }
  }
  if (n) batchText.value = ''
}

function close() { emit('close') }
function onKey(e: KeyboardEvent) { if (e.key === 'Escape') { e.stopPropagation(); close() } }
// 열려 있는 동안 앱 모달 스택에 올라간다 — App 의 ↑/↓ 히스토리 이동이 이 모달 뒤에서 넘어가지 않게.
// ESC 는 위 onKey 가 직접 처리한다(window capture + stopPropagation, utils/modalStack).
useModalLayer()

onMounted(() => {
  window.addEventListener('keydown', onKey, true)
  disconnectExtUrlReady = onBackendEvent('loraManagerUrlReady', onExtUrlReady)
  // 요청을 보내기 전에 붙는다 — 캐시 적중이면 응답이 곧바로 온다.
  disconnectLorasReady = onBackendEvent('lorasReady', (json: string) => { loraRequest.receive(json) })
  load()
  if (searchEl.value) searchEl.value.focus()
})
onUnmounted(() => {
  window.removeEventListener('keydown', onKey, true)
  disconnectExtUrlReady?.()
  disconnectExtUrlReady = null
  disconnectLorasReady?.()
  disconnectLorasReady = null
  loraRequest.cancel()
})
</script>

<style scoped>
.lm-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.72); z-index: 3200; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(4px); }
.lm-modal { width: min(840px, 94vw); height: min(840px, 92vh); background: var(--bg-secondary); border: 1px solid var(--border); border-radius: var(--radius-card); display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 20px 60px rgba(0,0,0,0.6); }
.lm-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; border-bottom: 1px solid var(--border); }
.lm-header h3 { font-size: 17px; font-weight: var(--fw-bold); color: var(--text-primary); }
.lm-sub { font-size: 11px; color: var(--text-muted); }
.lm-close { width: 30px; height: 30px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); cursor: pointer; }
.lm-close:hover { color: var(--text-primary); border-color: var(--accent); }
.lm-head-actions { display: flex; align-items: center; gap: 6px; }
.lm-modetab {
  height: 30px; padding: 0 12px; font-size: var(--fs-label); font-weight: var(--fw-bold);
  background: var(--bg-button); border: 1px solid var(--border);
  border-radius: var(--radius-base); color: var(--text-muted); cursor: pointer;
}
.lm-modetab:hover { color: var(--text-primary); }
.lm-modetab.active { background: var(--accent-dim); border-color: var(--accent); color: var(--accent); }
.lm-ext { flex: 1; display: flex; min-height: 0; }
/* 외부 LoRA 매니저 페이지를 얹는 자리라 우리 테마의 면이 아니다 — 그쪽이 자기 배경을
   그리기 전까지의 받침이므로 토큰화하지 않는다 */
.lm-iframe { flex: 1; width: 100%; height: 100%; border: none; background: #fff; }
.lm-error { color: var(--state-alert-fg); display: flex; flex-direction: column; align-items: center; gap: 8px; }
.lm-availability-note.lm-load-error { color: var(--state-alert-fg); }
.mt-8 { margin-top: 8px; }
.lm-searchbar { display: flex; gap: 8px; padding: 12px 20px; }
.lm-search { flex: 1; background: var(--bg-input); border: 1px solid var(--border); border-radius: var(--radius-base); padding: 9px 12px; color: var(--text-primary); font-size: 13px; }
.lm-search:focus { outline: none; border-color: var(--accent); }
.lm-refresh { width: 38px; background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); cursor: pointer; }
.lm-refresh:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
.lm-list { flex: 1; overflow-y: auto; padding: 4px 16px; }
.lm-empty { padding: 24px; text-align: center; color: var(--text-muted); font-size: 12px; }
.lm-availability-note {
  margin: 4px 4px 10px; padding: 8px 10px; border: 1px solid rgba(251,191,36,.28);
  /* 금색이지만 뜻은 '주의'(모델 일부를 못 쓴다)라 강조색이 아니라 경고 글자색 */
  border-radius: 7px; background: rgba(251,191,36,.06); color: var(--state-warn-fg); font-size: var(--fs-label);
}
.lm-section { margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.lm-section-header {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
  padding: 9px 11px; background: var(--bg-input); border-bottom: 1px solid var(--border);
}
.lm-section-header > div { min-width: 0; display: flex; flex-direction: column; gap: 2px; }
.lm-section-header strong { color: var(--text-primary); font-size: var(--fs-label); letter-spacing: 0; }
.lm-section-header span { color: var(--text-muted); font-size: var(--fs-label); }
.lm-section-count {
  flex-shrink: 0; min-width: 20px; padding: 2px 6px; border-radius: 8px;
  background: var(--bg-button); color: var(--text-secondary) !important; text-align: center; font-weight: var(--fw-bold);
}
.lm-item { display: flex; align-items: center; gap: 10px; padding: 9px 10px; border-bottom: 1px solid var(--border); }
.lm-item:last-child { border-bottom: none; }
.lm-item.unavailable { opacity: .58; }
.lm-item-main { flex: 1; min-width: 0; }
.lm-name-row { display: flex; align-items: center; flex-wrap: wrap; gap: 5px; min-width: 0; }
.lm-name { min-width: 0; flex: 1; font-size: 12px; font-weight: var(--fw-bold); color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lm-source-badge {
  flex-shrink: 0; padding: 2px 5px; border: 1px solid var(--border); border-radius: 7px;
  background: var(--bg-input); color: var(--text-muted); font-size: 7px; font-weight: var(--fw-bold); letter-spacing: 0;
}
/* main/secondary 는 v-if 로 배타 표시라 같은 정보색으로 합쳐도 구분이 죽지 않는다
   (파랑/청록의 색상 차이가 아니라 라벨 글자가 둘을 구분한다) */
.lm-source-badge.main { border-color: rgba(96,165,250,.35); background: rgba(96,165,250,.1); color: var(--state-info-fg); }
.lm-source-badge.secondary { border-color: rgba(34,211,238,.3); background: rgba(34,211,238,.08); color: var(--state-info-fg); }
.lm-source-badge.conflict { border-color: rgba(248,113,113,.35); background: rgba(248,113,113,.1); color: var(--state-alert-fg); }
.lm-unavailable { margin-top: 3px; color: var(--state-warn-fg); font-size: var(--fs-label); }
.lm-triggers { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.lm-tw { font-size: var(--fs-label); padding: 1px 6px; border-radius: 7px; background: var(--bg-button); color: var(--text-muted); }
.lm-weight { width: 64px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 5px; padding: 5px 6px; color: var(--text-primary); font-size: 12px; text-align: center; }
/* 주 버튼: 글자를 얹는 면이라 --accent 가 아니라 --accent-fill + --on-accent */
.lm-add { background: var(--accent-fill); color: var(--on-accent); border: none; border-radius: 6px; font-size: 11px; font-weight: var(--fw-bold); padding: 6px 12px; cursor: pointer; white-space: nowrap; }
.lm-add:hover { background: var(--accent-fill-hover); }
.lm-add:disabled, .lm-weight:disabled { opacity: .5; cursor: not-allowed; }
.lm-batch { margin: 4px 20px; border: 1px solid var(--border); border-radius: var(--radius-base); }
.lm-batch > summary { padding: 8px 12px; font-size: 11px; font-weight: var(--fw-bold); color: var(--text-secondary); cursor: pointer; }
.lm-batch-text { width: calc(100% - 24px); margin: 0 12px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; padding: 7px 9px; color: var(--text-primary); font-size: 11px; resize: vertical; }
.lm-batch-btn { margin: 8px 12px; background: var(--bg-button); border: 1px solid var(--accent); border-radius: 6px; color: var(--accent); font-size: 11px; font-weight: var(--fw-bold); padding: 5px 12px; cursor: pointer; }
.lm-footer { display: flex; align-items: center; gap: 8px; padding: 14px 20px; border-top: 1px solid var(--border); }
.lm-count { font-size: 11px; color: var(--text-muted); }
.lm-foot-spacer { flex: 1; }
.lm-done { background: var(--bg-button); border: 1px solid var(--border); border-radius: var(--radius-base); color: var(--text-secondary); font-size: 12px; font-weight: var(--fw-bold); padding: 9px 16px; cursor: pointer; }
.lm-done:hover { color: var(--text-primary); border-color: var(--accent); }
@media (max-width: 520px) {
  .lm-header { align-items: flex-start; padding: 12px; }
  .lm-head-actions { flex-wrap: wrap; justify-content: flex-end; }
  .lm-searchbar, .lm-footer { padding-left: 12px; padding-right: 12px; }
  .lm-list { padding-left: 8px; padding-right: 8px; }
  .lm-item { align-items: flex-start; flex-wrap: wrap; gap: 7px; }
  .lm-item-main { flex: 0 0 100%; }
  .lm-weight { margin-left: auto; }
  .lm-batch { margin-left: 12px; margin-right: 12px; }
}
</style>
