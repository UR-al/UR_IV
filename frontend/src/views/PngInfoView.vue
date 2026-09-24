<template>
  <div class="pnginfo-view">

    <!-- PNG Info 탭 -->
    <div v-if="subTab === 'info'" class="tab-content info-layout">
      <div class="image-area" @dragover.prevent @drop.prevent="onDrop" @dblclick="openFile">
        <img v-if="imagePath" :src="mediaUrl(imagePath)" class="preview-img" />
        <div v-else class="drop-hint">
          <div class="icon"><Icon name="file" /></div>
          <div>이미지를 드래그하거나 더블클릭</div>
        </div>
      </div>
      <div class="info-panel">
        <div class="info-header">
          <h3>PNG Info</h3>
          <button class="btn" v-host-dialog="'open_png_info_file'" @click="openFile"><Icon name="folder-open" /> 열기</button>
          <button class="btn" @click="copyAll" v-if="exif.raw"><Icon name="clipboard" /> 전체 복사</button>
          <button class="btn" @click="sendToCompare" v-if="imagePath"><Icon name="search" /> 비교로</button>
        </div>
        <div class="info-body" v-if="exif.raw">
          <div class="prompt-copy-actions">
            <button type="button" class="btn" :disabled="!exif.prompt" @click="copySection(exif.prompt || '', 'Prompt')">Prompt만 복사</button>
            <button type="button" class="btn" :disabled="!exif.negative" @click="copySection(exif.negative || '', 'Negative')">Negative만 복사</button>
          </div>
          <div v-if="exif.prompt" class="section">
            <div class="section-head">
              <label>Prompt</label>
              <button class="copy-btn" @click="copySection(exif.prompt, 'Prompt')" title="Prompt 복사" aria-label="Prompt만 복사"><Icon name="clipboard" /></button>
            </div>
            <pre>{{ exif.prompt }}</pre>
          </div>
          <div v-if="exif.negative" class="section">
            <div class="section-head">
              <label>네거티브</label>
              <button class="copy-btn" @click="copySection(exif.negative, 'Negative')" title="Negative 복사" aria-label="Negative만 복사"><Icon name="clipboard" /></button>
            </div>
            <pre>{{ exif.negative }}</pre>
          </div>
          <div v-if="exif.params" class="section params-section">
            <div class="section-head">
              <label>Parameters</label>
              <button class="copy-btn" @click="copySection(exif.params_line || JSON.stringify(exif.params, null, 2), 'Parameters')" title="Parameters 복사"><Icon name="clipboard" /></button>
            </div>
            <div class="params-grid">
              <div class="param-line" v-if="exif.params.generation"><span class="pl">생성</span><span>{{ exif.params.generation }}</span></div>
              <div class="param-line" v-if="exif.params.core"><span class="pl">기본</span><span>{{ exif.params.core }}</span></div>
              <div class="param-line" v-if="exif.params.model"><span class="pl">모델</span><span>{{ exif.params.model }}</span></div>
              <div class="param-line" v-if="exif.params.hires"><span class="pl">고해상도</span><span>{{ exif.params.hires }}</span></div>
              <div class="param-line" v-if="exif.params.extensions"><span class="pl">확장</span><span>{{ exif.params.extensions }}</span></div>
              <div class="param-line" v-if="exif.params.other"><span class="pl">기타</span><span>{{ exif.params.other }}</span></div>
            </div>
          </div>
          <div v-else-if="exif.params_line" class="section">
            <div class="section-head">
              <label>Parameters</label>
              <button class="copy-btn" @click="copySection(exif.params_line, 'Parameters')" title="Parameters 복사"><Icon name="clipboard" /></button>
            </div>
            <pre>{{ exif.params_line }}</pre>
          </div>
          <div v-if="!exif.prompt && !exif.raw_prompt && !exif.raw_workflow" class="section">
            <div class="section-head">
              <label>Raw</label>
              <button class="copy-btn" @click="copySection(exif.raw, 'Raw')" title="Raw 복사"><Icon name="clipboard" /></button>
            </div>
            <pre>{{ exif.raw }}</pre>
          </div>
          <ComfyMetadataDetails :data="exif" />
          <div class="action-section">
            <label class="action-label">보내기</label>
            <div class="send-grid">
              <button class="send-card primary" :disabled="exif.can_apply === false" @click="sendPrompt" title="현재 프롬프트를 T2I 탭으로 전송">
                <span class="send-ico"><Icon name="upload" /></span>
                <span class="send-name">T2I 전송</span>
              </button>
              <button class="send-card" :disabled="exif.can_apply === false" @click="sendGenerate" title="현재 EXIF로 즉시 생성">
                <span class="send-ico"><Icon name="zap" /></span>
                <span class="send-name">즉시 생성</span>
              </button>
              <button class="send-card" @click="action('send_to_i2i', { path: imagePath })" title="I2I 탭으로 전송">
                <span class="send-ico"><Icon name="image" /></span>
                <span class="send-name">I2I</span>
              </button>
              <button class="send-card" @click="action('send_to_inpaint', { path: imagePath })" title="Inpaint 탭으로 전송">
                <span class="send-ico"><Icon name="scissors" /></span>
                <span class="send-name">Inpaint</span>
              </button>
              <button class="send-card" @click="action('send_to_editor', { path: imagePath })" title="Editor 탭으로 전송">
                <span class="send-ico"><Icon name="palette" /></span>
                <span class="send-name">Editor</span>
              </button>
              <button class="send-card star" @click="action('add_favorite', { path: imagePath })" title="즐겨찾기에 추가">
                <span class="send-ico"><Icon name="star" /></span>
                <span class="send-name">즐겨찾기</span>
              </button>
            </div>
          </div>
          <!-- 메타 이식 — 이 이미지의 생성 메타를 다른 이미지(외부 편집으로 메타가 지워진 사본 등)에 박아 새 PNG 로 -->
          <div class="action-section">
            <label class="action-label">메타 이식</label>
            <div class="transplant-row">
              <button class="btn" v-host-dialog="'pnginfo_transplant_meta'" :disabled="!imagePath" @click="transplantMeta"
                title="이 이미지의 프롬프트·파라미터(ComfyUI 워크플로 포함)를 다른 이미지에 넣어 새 PNG 로 저장합니다. 대상 픽셀은 그대로입니다.">
                <Icon name="layers" /> 다른 이미지에 이식…
              </button>
              <label v-if="exif.raw_prompt || exif.raw_workflow" class="transplant-opt">
                <input type="checkbox" v-model="transplantIncludeWorkflow" /> ComfyUI 워크플로 포함
              </label>
            </div>
          </div>
        </div>
        <div v-else class="info-empty">이미지를 선택하면 메타데이터가 표시됩니다</div>
      </div>
    </div>

    <!-- Compare 탭 -->
    <div v-if="subTab === 'compare'" class="tab-content compare-layout">
      <div class="compare-controls">
        <div class="cmp-slot">
          <span class="cmp-label">이전</span>
          <button class="btn" v-host-dialog="'open_compare_image'" @click="loadCompareImage('before')"><Icon name="folder-open" /> 열기</button>
          <span class="cmp-name">{{ beforeName || '없음' }}</span>
        </div>
        <div class="cmp-slot">
          <span class="cmp-label">이후</span>
          <button class="btn" v-host-dialog="'open_compare_image'" @click="loadCompareImage('after')"><Icon name="folder-open" /> 열기</button>
          <span class="cmp-name">{{ afterName || '없음' }}</span>
        </div>
      </div>
      <div class="compare-area">
        <CompareSlider v-if="compareBefore && compareAfter"
          :before-src="mediaUrl(compareBefore)"
          :after-src="mediaUrl(compareAfter)"
        />
        <div v-else class="compare-hint">
          <div class="icon"><Icon name="search" /></div>
          <p>두 이미지를 선택하면 비교 슬라이더가 표시됩니다</p>
          <p class="sub">우클릭 메뉴의 "비교로 보내기"로도 이미지를 추가할 수 있습니다</p>
        </div>
      </div>
      <!-- EXIF 비교 -->
      <div class="exif-diff" v-if="compareBefore && compareAfter && (beforeExif.raw || afterExif.raw)">
        <div class="diff-header">
          <h4>파라미터 차이</h4>
          <button class="btn" @click="showDiffOnly = !showDiffOnly">{{ showDiffOnly ? '전체 보기' : '차이만 보기' }}</button>
        </div>
        <div class="diff-table">
          <div class="diff-row header">
            <span class="diff-key">Parameter</span>
            <span class="diff-val">이전</span>
            <span class="diff-val">이후</span>
          </div>
          <template v-for="row in paramDiffRows" :key="row.key">
            <div class="diff-row" :class="{ changed: row.changed, same: !row.changed }" v-if="!showDiffOnly || row.changed">
              <span class="diff-key">{{ row.key }}</span>
              <span class="diff-val" :class="{ highlight: row.changed }">{{ row.before }}</span>
              <span class="diff-val" :class="{ highlight: row.changed }">{{ row.after }}</span>
            </div>
          </template>
        </div>
        <!-- 프롬프트 비교 -->
        <div class="diff-prompts" v-if="beforeExif.prompt || afterExif.prompt">
          <div class="diff-prompt-section">
            <label>프롬프트 차이</label>
            <div class="diff-tags">
              <span v-for="t in promptDiff.same" :key="'s'+t" class="dtag same">{{ t }}</span>
              <span v-for="t in promptDiff.removed" :key="'r'+t" class="dtag removed">- {{ t }}</span>
              <span v-for="t in promptDiff.added" :key="'a'+t" class="dtag added">+ {{ t }}</span>
            </div>
          </div>
        </div>
      </div>
      <!-- GIF 내보내기 -->
      <div class="gif-bar" v-if="compareBefore && compareAfter">
        <span class="gif-label">GIF 내보내기</span>
        <label>Speed</label>
        <CustomSelect v-model="gifDurationStr" :options="['Fast', 'Normal', 'Slow']" placeholder="Speed" />
        <button class="gif-btn" @click="exportGif" :disabled="gifExporting">
          <Icon v-if="!gifExporting" name="video" /> {{ gifExporting ? 'Exporting...' : 'Export GIF' }}
        </button>
        <a v-if="gifResult" :href="mediaUrl(gifResult)" class="gif-link" target="_blank"><Icon name="download" /> {{ gifResult.split('/').pop() }}</a>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import { useViewMode } from '../composables/useViewMode'
import { mediaUrl } from '../utils/media.js'
import CompareSlider from '../components/CompareSlider.vue'
import CustomSelect from '../components/CustomSelect.vue'
import ComfyMetadataDetails from '../components/ComfyMetadataDetails.vue'
import { copyTextToClipboard } from '../utils/clipboard'
import { diffParameters } from '../utils/paramDiff'
import type { ActionName, ActionPayload, CompareGifReadyPayload } from '../types/bridge'
import { createLatestRequest, wasAbandoned } from '../utils/bridgeRequest'

interface ExifParams {
  generation?: string
  core?: string
  model?: string
  hires?: string
  extensions?: string
  other?: string
  [k: string]: any
}
interface ExifData {
  source?: string
  raw?: string
  prompt?: string
  negative?: string
  /** 표시 그룹 — 파라미터가 없으면 null */
  params?: ExifParams | null
  params_line?: string
  /** core 가 파싱한 원 파라미터(따옴표를 푼 값) — 비교에 쓴다 */
  parameters?: Record<string, unknown>
  [k: string]: any
}

/** 서브탭은 왼쪽 레일의 서랍이 정한다 — 여기선 읽기만 한다. `useViewMode` 참조. */
const { mode: subTab } = useViewMode('png')
const imagePath = ref('')
const exif = ref<ExifData>({})

// Compare
const compareBefore = ref('')
const compareAfter = ref('')
const beforeName = ref('')
const afterName = ref('')
const beforeExif = ref<ExifData>({})
const afterExif = ref<ExifData>({})
const showDiffOnly = ref(true)

// 비교 이미지 로드 시 EXIF도 함께 로드
async function loadCompareExif(path: string, target: string) {
  const backend: any = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json: string) => {
      try {
        const d = JSON.parse(json)
        if (target === 'before') beforeExif.value = d
        else afterExif.value = d
      } catch {}
    })
  }
}

// 파라미터 diff — core 가 따옴표를 풀어 파싱한 parameters dict 를 그대로 비교한다
// (표시용 params_line 을 다시 정규식으로 쪼개지 않는다 — utils/paramDiff.ts)
const paramDiffRows = computed(() => diffParameters(beforeExif.value.parameters, afterExif.value.parameters))

// 프롬프트 diff (태그 단위)
const promptDiff = computed(() => {
  const bTags = new Set((beforeExif.value.prompt || '').split(',').map(t => t.trim()).filter(Boolean))
  const aTags = new Set((afterExif.value.prompt || '').split(',').map(t => t.trim()).filter(Boolean))
  const same: string[] = [], added: string[] = [], removed: string[] = []
  for (const t of bTags) { if (aTags.has(t)) same.push(t); else removed.push(t) }
  for (const t of aTags) { if (!bTags.has(t)) added.push(t) }
  return { same, added, removed }
})

async function loadImage(path: string) {
  imagePath.value = path
  exif.value = {}
  const backend: any = await getBackend()
  if (backend.getImageExif) {
    backend.getImageExif(path, (json: string) => { if (imagePath.value !== path) return; try { exif.value = JSON.parse(json) } catch {} })
  }
}

function onDrop(e: DragEvent) {
  const file: any = e.dataTransfer?.files?.[0]
  if (file?.path) loadImage(file.path.replace(/\\/g, '/'))
  else {
    const path = e.dataTransfer?.getData('text/plain')
    if (path && path.includes('/')) loadImage(path)
  }
}

function openFile() { requestAction('open_png_info_file') }
async function copyAll() {
  const data = exif.value
  const chunks = [data.raw || '']
  if (data.raw_prompt && data.raw_prompt !== data.raw) chunks.push(`prompt:\n${data.raw_prompt}`)
  if (data.raw_workflow && data.raw_workflow !== data.raw) chunks.push(`workflow:\n${data.raw_workflow}`)
  await copySection(chunks.filter(Boolean).join('\n\n'), '전체 메타데이터')
}
// Native Qt and browser-local clipboard share one confirmed-success contract.
async function copySection(text: string, label: string) {
  if (!text) return
  const ok = await copyTextToClipboard(text)
  requestAction('show_toast', { type: ok ? 'success' : 'error', msg: ok ? `${label} 복사됨` : '클립보드에 복사하지 못했습니다' })
}
function sendPrompt() { if (exif.value.can_apply !== false) requestAction('pnginfo_send_prompt', { prompt: exif.value.prompt || '', negative: exif.value.negative || '', source: exif.value.source, can_apply: exif.value.can_apply, path: exif.value.path }) }
function sendGenerate() { if (exif.value.can_apply !== false) requestAction('pnginfo_generate', exif.value) }
function action<K extends ActionName>(name: K, payload?: ActionPayload<K>) { requestAction(name, payload) }

// 메타 이식 — 대상 이미지·저장 위치는 호스트 대화상자(원격 웹 모드에선 v-host-dialog 가 버튼을 끈다)
const transplantIncludeWorkflow = ref(true)
function transplantMeta() {
  if (!imagePath.value) return
  requestAction('pnginfo_transplant_meta', { path: imagePath.value, include_workflow: transplantIncludeWorkflow.value })
}

function sendToCompare() {
  if (!compareBefore.value) {
    compareBefore.value = imagePath.value
    beforeName.value = imagePath.value.split('/').pop() || ''
    loadCompareExif(imagePath.value, 'before')
  } else {
    compareAfter.value = imagePath.value
    afterName.value = imagePath.value.split('/').pop() || ''
    loadCompareExif(imagePath.value, 'after')
  }
  subTab.value = 'compare'
}

async function loadCompareImage(slot: string) {
  requestAction('open_compare_image', { slot })
}

// GIF 내보내기
const gifDuration = ref(400)
const gifSpeedMap: Record<string, number> = { 'Fast': 200, 'Normal': 400, 'Slow': 700 }
const gifDurationStr = computed({
  get: () => Object.entries(gifSpeedMap).find(([, v]) => v === gifDuration.value)?.[0] || 'Normal',
  set: (v: string) => { gifDuration.value = gifSpeedMap[v] || 400 }
})
const gifExporting = ref(false)
const gifResult = ref('')
// GIF 는 워커에서 만든다(requestCompareGif → compareGifReady). 예전 동기 슬롯은 풀해상도 블렌드·
// 저장을 GUI 스레드에서 해 창이 수 초 멈췄다. 마지막 요청만 유효, 1분 안에 답이 없으면 푼다.
const gifRequest = createLatestRequest<CompareGifReadyPayload>({ timeoutMs: 60_000, prefix: 'gif' })

async function exportGif() {
  if (!compareBefore.value || !compareAfter.value) return
  gifExporting.value = true; gifResult.value = ''
  const { id, done, outcome } = gifRequest.begin()
  const backend: any = await getBackend()
  if (wasAbandoned(outcome())) return   // 백엔드를 기다리는 사이 화면이 닫혔거나 새 내보내기가 시작됐다
  if (!backend?.requestCompareGif) {
    gifRequest.cancel()
    gifExporting.value = false
    requestAction('show_toast', { type: 'error', msg: 'GIF 생성 실패: 백엔드 연결 없음' })
    return
  }
  backend.requestCompareGif(compareBefore.value, compareAfter.value, gifDuration.value, 0, id)
  const d = await done
  // 더 새 내보내기가 이어받았거나(그쪽이 마무리한다) 화면이 닫혀 버린 요청 — '응답 없음'을 띄우지 않는다.
  if (wasAbandoned(outcome())) return
  gifExporting.value = false
  if (d?.path) {
    gifResult.value = d.path
    requestAction('show_toast', { type: 'success', msg: `GIF 생성 완료 (${d.frames} frames)` })
  } else {
    requestAction('show_toast', { type: 'error', msg: `GIF 생성 실패: ${d?.error || '응답 없음'}` })
  }
}

// 이벤트 disconnect 핸들 — keep-alive로 재마운트되어도 누적 등록 방지
const _eventUnsubs: Array<() => void> = []
onMounted(() => {
  // '열기' 대화상자 결과 — PNG Info 전용 시그널. 예전엔 inpaintImageLoaded 를 같이 들어서
  // 여기서 파일을 열면 keep-alive 로 살아 있는 인페인트의 원본·마스크·undo 가 날아갔고,
  // 인페인트로 보낼 때마다 이 화면도 바뀌었다.
  _eventUnsubs.push(onBackendEvent('pngInfoImageLoaded', (path: string) => loadImage(path)))
  _eventUnsubs.push(onBackendEvent('compareGifReady', (json: string) => { gifRequest.receive(json) }))
  // 비교 이미지 수신 (우클릭 메뉴 "비교로 보내기" 또는 파일 다이얼로그)
  _eventUnsubs.push(onBackendEvent('compareImageLoaded', (json: string) => {
    try {
      const d = JSON.parse(json)
      if (d.slot === 'before') {
        compareBefore.value = d.path; beforeName.value = d.path.split('/').pop()
        loadCompareExif(d.path, 'before')
      } else {
        compareAfter.value = d.path; afterName.value = d.path.split('/').pop()
        loadCompareExif(d.path, 'after')
      }
      subTab.value = 'compare'
    } catch {}
  }))
})
onUnmounted(() => {
  for (const off of _eventUnsubs) { try { off() } catch {} }
  _eventUnsubs.length = 0
  gifRequest.cancel()
})
</script>

<style scoped>
.pnginfo-view { width: 100%; height: 100%; display: flex; flex-direction: column; }
.tab-content { flex: 1; overflow: hidden; }

/* Info Layout */
.info-layout { display: flex; }
.image-area { flex: 1; display: flex; align-items: center; justify-content: center; cursor: pointer; min-width: 300px; padding: 16px; }
.preview-img { max-width: 100%; max-height: 100%; object-fit: contain; border-radius: 4px; }
.drop-hint { text-align: center; color: var(--text-muted); user-select: none; }
.drop-hint .icon { font-size: 48px; opacity: 0.3; margin-bottom: 12px; }
.info-panel { width: 400px; flex-shrink: 0; display: flex; flex-direction: column; overflow: hidden; }
/* 오른쪽 여백은 알림 종 자리다 (style.css --notif-gutter) — 이 줄은 창 오른쪽 끝까지 찬다 */
.info-header { display: flex; align-items: center; gap: 6px; padding: 8px var(--notif-gutter) 8px 12px; flex-shrink: 0; }
.info-header h3 { color: var(--text-primary); font-size: 14px; margin: 0; flex: 1; }
.btn { height: 28px; padding: 0 12px; background: var(--bg-button); border: none; border-radius: var(--radius-base); color: var(--text-secondary); font-size: var(--fs-meta); cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
.btn:hover { background: var(--bg-button-hover); color: var(--text-primary); }
/* 주 버튼 — 면은 accent-fill, 글자는 on-accent (강조색을 바꿔도 글자가 읽히게) */
.btn.accent { background: var(--accent-fill); color: var(--on-accent); }
.info-body { flex: 1; overflow-y: auto; padding: 8px 12px; }
.section { margin-bottom: 10px; }
.section-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 2px; min-height: 16px;
}
.section-head label { color: var(--accent); font-size: 11px; font-weight: var(--fw-bold); margin: 0; }
/* Keyboard/touch users can discover the section copy actions without hover. */
.copy-btn {
  opacity: .7; transition: opacity 0.15s, background 0.15s;
  background: none; border: 1px solid transparent;
  color: var(--text-muted); font-size: 11px;
  width: 20px; height: 20px; border-radius: 4px;
  cursor: pointer; padding: 0; display: inline-flex;
  align-items: center; justify-content: center;
}
.section:hover .copy-btn { opacity: 0.7; }
.copy-btn:hover { opacity: 1 !important; background: var(--bg-button); border-color: var(--border); color: var(--accent); }
.prompt-copy-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
.btn:disabled, .send-card:disabled { opacity: .45; cursor: not-allowed; }
.copy-btn:focus-visible, .btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; opacity: 1; }
.section pre { color: var(--text-secondary); font-size: 11px; white-space: pre-wrap; word-break: break-all; background: var(--bg-secondary); padding: 6px 8px; border-radius: 4px; margin: 0; max-height: 150px; overflow-y: auto; }
.action-section { margin-top: 16px; }
.action-label {
  display: block; font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--text-muted); letter-spacing: 0;
  margin-bottom: 8px;
}
.send-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}
.transplant-row { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
.transplant-opt { display: inline-flex; align-items: center; gap: 6px; font-size: var(--fs-label); color: var(--text-secondary); cursor: pointer; }
.send-card {
  display: flex; flex-direction: column; align-items: center;
  gap: 4px; padding: 10px 6px;
  background: var(--bg-button); border: 1px solid var(--border);
  border-radius: 8px; cursor: pointer;
  transition: all 0.15s;
}
.send-card:hover {
  background: var(--bg-input); border-color: var(--text-muted);
  transform: translateY(-1px);
}
.send-card.primary {
  background: var(--accent-dim);
  border-color: rgba(250, 204, 21, 0.4);
}
.send-card.primary:hover {
  background: rgba(250, 204, 21, 0.15);
  border-color: var(--accent);
  box-shadow: 0 2px 8px rgba(250, 204, 21, 0.2);
}
.send-card.star:hover { border-color: var(--accent); }
.send-ico { font-size: 18px; line-height: 1; }
.send-name {
  font-size: var(--fs-label); font-weight: var(--fw-bold);
  color: var(--text-secondary); letter-spacing: 0;
}
.send-card.primary .send-name { color: var(--accent); }
.info-empty { flex: 1; display: flex; align-items: center; justify-content: center; color: var(--text-muted); font-size: 14px; }

.params-grid { background: var(--bg-secondary); border-radius: 4px; padding: 6px 8px; }
.param-line { display: flex; align-items: baseline; gap: 8px; padding: 3px 0; font-size: 11px; color: var(--text-secondary); border-bottom: 1px solid var(--rule); }
.param-line:last-child { border-bottom: none; }
.pl { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; min-width: 45px; flex-shrink: 0; }

/* Compare Layout */
.compare-layout { display: flex; flex-direction: column; }
.compare-controls { display: flex; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
.cmp-slot { display: flex; align-items: center; gap: 6px; flex: 1; }
.cmp-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--accent); letter-spacing: 0; }
.cmp-name { font-size: var(--fs-label); color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.compare-area { flex: 1; position: relative; }
.compare-hint { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; text-align: center; }
.compare-hint .icon { font-size: 48px; opacity: 0.3; margin-bottom: 12px; }
.compare-hint p { color: var(--text-muted); font-size: 13px; }
.compare-hint .sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

/* GIF Export */
.gif-bar { display: flex; align-items: center; gap: 8px; padding: 8px 16px; border-top: 1px solid var(--border); flex-shrink: 0; }
.gif-label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.gif-bar label { font-size: var(--fs-label); color: var(--text-muted); }
.gif-bar select { padding: 3px 8px; font-size: var(--fs-label); }
.gif-btn { padding: 5px 14px; background: var(--accent-fill); border: none; border-radius: 6px; color: var(--on-accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.gif-btn:disabled { opacity: 0.4; }
.gif-link { font-size: var(--fs-label); color: var(--state-info-fg); text-decoration: none; margin-left: auto; }

/* EXIF Diff */
.exif-diff { padding: 12px 16px; border-top: 1px solid var(--border); flex-shrink: 0; max-height: 300px; overflow-y: auto; }
.diff-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.diff-header h4 { font-size: 11px; font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; }
.diff-table { display: flex; flex-direction: column; gap: 1px; font-family: 'Consolas', monospace; }
.diff-row { display: grid; grid-template-columns: 140px 1fr 1fr; gap: 8px; padding: 4px 8px; border-radius: 4px; font-size: var(--fs-label); }
.diff-row.header { font-weight: var(--fw-bold); color: var(--text-muted); font-size: var(--fs-label); letter-spacing: 0; border-bottom: 1px solid var(--border); }
.diff-row.changed { background: rgba(248, 113, 113, 0.06); }
.diff-row.same { opacity: 0.5; }
.diff-key { color: var(--text-secondary); font-weight: var(--fw-bold); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.diff-val { color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.diff-val.highlight { color: var(--state-alert-fg); font-weight: var(--fw-bold); }

.diff-prompts { margin-top: 12px; }
.diff-prompt-section label { font-size: var(--fs-label); font-weight: var(--fw-bold); color: var(--text-muted); letter-spacing: 0; margin-bottom: 6px; display: block; }
.diff-tags { display: flex; flex-wrap: wrap; gap: 3px; }
.dtag { padding: 2px 8px; border-radius: 4px; font-size: var(--fs-label); }
.dtag.same { background: rgba(255,255,255,0.03); color: var(--text-muted); }
.dtag.added { background: rgba(74,222,128,0.1); color: var(--state-ok-fg); border: 1px solid rgba(74,222,128,0.2); }
.dtag.removed { background: rgba(248,113,113,0.1); color: var(--state-alert-fg); border: 1px solid rgba(248,113,113,0.2); text-decoration: line-through; }
</style>
