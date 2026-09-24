<template>
  <div class="creator-studio">
    <div v-if="progress.visible" class="global-progress" aria-live="polite">
      <div class="progress-copy">
        <strong>{{ progress.stage || 'Preparing' }}</strong>
        <span>{{ progress.message }}</span>
      </div>
      <div class="progress-track"><div class="progress-fill" :style="{ width: `${progress.percent}%` }" /></div>
      <span class="progress-value">{{ Math.round(progress.percent) }}%</span>
      <button class="danger-button compact" @click="cancelCreator">취소</button>
    </div>

    <main v-if="creatorMode === 'video'" class="creator-body two-column">
      <section class="settings-pane scroll-pane">
        <div class="pane-state" :class="stateClass"><span class="state-dot" />{{ stateLabel }}</div>
        <div class="section-heading">
          <div><span class="section-kicker">MiniMax H3</span><h3>영상 생성</h3></div>
          <div class="segmented">
            <button v-for="mode in videoModes" :key="mode.id" :class="{ active: videoForm.mode === mode.id }"
              @click="videoForm.mode = mode.id">{{ mode.label }}</button>
          </div>
        </div>

        <label class="field wide"><span>프롬프트</span>
          <textarea v-model="videoForm.prompt" rows="5" placeholder="Describe action, camera movement, lighting, timing and atmosphere..." />
        </label>
        <label class="field wide"><span>네거티브 프롬프트</span>
          <textarea v-model="videoForm.negative" rows="2" placeholder="Artifacts or motion to avoid..." />
        </label>

        <div v-if="videoForm.mode !== 't2v'" class="media-slot">
          <div class="slot-preview">
            <video v-if="videoForm.mode === 'v2v' && videoForm.sourcePath" :src="displayMedia(videoForm.sourcePath)" muted />
            <img v-else-if="videoForm.sourcePath" :src="displayMedia(videoForm.sourcePath)" alt="Video source" />
            <span v-else>{{ videoForm.mode === 'i2v' ? 'SOURCE IMAGE' : 'SOURCE VIDEO' }}</span>
          </div>
          <div class="slot-info"><b>{{ videoForm.mode.toUpperCase() }} SOURCE</b><small>{{ videoForm.sourcePath || 'No media selected' }}</small></div>
          <button class="secondary-button" v-host-dialog="'creator_select_media'" @click="pickMedia('video_source')">선택</button>
          <button v-if="videoForm.sourcePath" class="icon-button" title="Clear" @click="videoForm.sourcePath = ''">×</button>
        </div>

        <div v-if="videoForm.mode === 'v2v'" class="media-slot">
          <div class="slot-preview">
            <img v-if="videoForm.identityPath" :src="displayMedia(videoForm.identityPath)" alt="Identity reference" />
            <span v-else>아이덴티티</span>
          </div>
          <div class="slot-info"><b>아이덴티티 참조</b><small>{{ videoForm.identityPath || 'Optional character reference' }}</small></div>
          <button class="secondary-button" v-host-dialog="'creator_select_media'" @click="pickMedia('video_identity')">선택</button>
          <button v-if="videoForm.identityPath" class="icon-button" title="Clear" @click="videoForm.identityPath = ''">×</button>
        </div>

        <div class="field-grid four">
          <label class="field"><span>너비</span><input v-model.number="videoForm.width" type="number" min="256" step="32" /></label>
          <label class="field"><span>높이</span><input v-model.number="videoForm.height" type="number" min="256" step="32" /></label>
          <label class="field"><span>프레임</span><input v-model.number="videoForm.frames" type="number" min="9" step="4" /></label>
          <label class="field"><span>FPS · H3 기본값</span><input v-model.number="videoForm.fps" type="number" min="24" max="24" readonly /></label>
          <label class="field span-two"><span>시드</span><input v-model.number="videoForm.seed" type="number" min="-1" /></label>
          <label class="toggle-field span-two"><input v-model="videoForm.includeAudio" type="checkbox" /><span>오디오 유지 · 생성</span></label>
        </div>
        <template v-if="videoForm.includeAudio">
          <label class="field wide"><span>사운드</span>
            <textarea v-model="videoForm.audioPrompt" rows="2" placeholder="Ambient sound, Foley, music and timing..." />
          </label>
          <label class="field wide"><span>대사</span>
            <textarea v-model="videoForm.dialogue" rows="2" placeholder="Optional spoken dialogue and delivery..." />
          </label>
        </template>

        <div class="action-row sticky-actions">
          <button class="primary-button" :disabled="!canGenerateVideo || progress.visible" @click="generateVideo">
            GENERATE {{ videoForm.mode.toUpperCase() }}
          </button>
          <button v-if="progress.visible" class="danger-button" @click="cancelCreator">취소</button>
        </div>
      </section>

      <section class="output-pane">
        <OutputPreview :result="lastResult || undefined" empty-title="영상 결과"
          empty-copy="생성된 영상 · WebP · 미리보기 프레임이 여기 표시됩니다" />
      </section>
    </main>

    <main v-else-if="creatorMode === 'krea'" class="creator-body two-column">
      <section class="settings-pane scroll-pane">
        <div class="pane-state" :class="stateClass"><span class="state-dot" />{{ stateLabel }}</div>
        <div class="section-heading"><div><span class="section-kicker">Krea2</span><h3>아이덴티티 편집</h3></div></div>
        <label class="field wide"><span>편집 프롬프트</span>
          <textarea v-model="kreaForm.prompt" rows="6" placeholder="Describe the desired edit while preserving identity..." />
        </label>
        <div class="dual-media">
          <button class="media-tile" v-host-dialog="'creator_select_media'" @click="pickMedia('krea_source')">
            <img v-if="kreaForm.sourcePath" :src="displayMedia(kreaForm.sourcePath)" alt="Source" />
            <span v-else class="media-empty">＋<small>원본 이미지</small></span>
            <b>원본</b><small>{{ basename(kreaForm.sourcePath) || 'Select image' }}</small>
          </button>
          <button class="media-tile" v-host-dialog="'creator_select_media'" @click="pickMedia('krea_reference')">
            <img v-if="kreaForm.referencePath" :src="displayMedia(kreaForm.referencePath)" alt="Identity reference" />
            <span v-else class="media-empty">＋<small>아이덴티티 참조</small></span>
            <b>참조</b><small>{{ basename(kreaForm.referencePath) || 'Select image' }}</small>
          </button>
        </div>
        <label class="range-field">
          <span><b>아이덴티티 유지도</b><output>{{ kreaForm.fidelity.toFixed(2) }}</output></span>
          <input v-model.number="kreaForm.fidelity" type="range" min="0.5" max="12" step="0.25" />
          <small>Lower values allow larger edits; higher values preserve the reference identity.</small>
        </label>
        <div class="field-grid two">
          <label class="field"><span>시드</span><input v-model.number="kreaForm.seed" type="number" min="-1" /></label>
          <label class="toggle-field"><input v-model="kreaForm.hires" type="checkbox" /><span>Hires pass</span></label>
          <template v-if="kreaForm.hires">
            <label class="field"><span>고해상도 배율</span><input v-model.number="kreaForm.hiresScale" type="number" min="1" max="4" step="0.25" /></label>
            <label class="field"><span>디노이즈</span><input v-model.number="kreaForm.hiresDenoise" type="number" min="0" max="1" step="0.05" /></label>
          </template>
        </div>
        <div class="action-row sticky-actions">
          <button class="primary-button" :disabled="!canGenerateKrea || progress.visible" @click="generateKrea">Krea2 편집 생성</button>
          <button v-if="progress.visible" class="danger-button" @click="cancelCreator">취소</button>
        </div>
      </section>
      <section class="output-pane">
        <OutputPreview :result="lastResult || undefined" empty-title="Krea2 결과"
          empty-copy="The edited image and hires result appear here." />
      </section>
    </main>

    <main v-else-if="creatorMode === 'comic'" class="creator-body comic-mode">
      <aside class="comic-sidebar scroll-pane">
        <div class="pane-state" :class="stateClass"><span class="state-dot" />{{ stateLabel }}</div>
        <div class="section-heading"><div><span class="section-kicker">스토리보드</span><h3>만화 문서</h3></div></div>
        <label class="field wide"><span>장면</span>
          <textarea v-model="planner.scene" rows="5" placeholder="Describe the scene, characters, conflict and ending..." />
        </label>
        <div class="field-grid two">
          <label class="field"><span>패널</span>
            <select v-model.number="planner.panelCount"><option v-for="n in 6" :key="n" :value="n">{{ n }}</option></select>
          </label>
          <label class="field"><span>스타일</span>
            <select v-model="planner.style"><option v-for="style in comicStyles" :key="style">{{ style }}</option></select>
          </label>
        </div>
        <button class="primary-button" :disabled="!planner.scene.trim() || progress.visible" @click="planComic">AI 스토리보드</button>

        <div class="divider" />
        <label class="field wide"><span>문서 제목</span><input v-model="comicDoc.title" type="text" /></label>
        <div class="toolbar-row">
          <button class="tool-button" :disabled="!canUndo" title="Undo" @click="undo"><Icon name="undo" /> 실행 취소</button>
          <button class="tool-button" :disabled="!canRedo" title="Redo" @click="redo"><Icon name="redo" /> 다시 실행</button>
          <span class="save-state">{{ saveStatus }}</span>
        </div>
        <span class="field-caption">레이아웃</span>
        <div class="layout-presets">
          <button v-for="layout in layouts" :key="layout.id" :class="{ active: comicDoc.layout === layout.id }"
            :title="layout.label" @click="comicDoc.layout = layout.id">
            <span :class="`layout-icon ${layout.id}`"><i v-for="n in Math.min(comicDoc.panels.length || 1, 4)" :key="n" /></span>
            {{ layout.label }}
          </button>
        </div>

        <div class="bubble-heading panel-heading">
          <span class="field-caption">PANELS · {{ comicDoc.panels.length }}/6</span>
          <button class="mini-accent" :disabled="comicDoc.panels.length >= 6" @click="addPanel">＋ PANEL</button>
        </div>
        <div class="panel-list">
          <button v-for="(panel, index) in comicDoc.panels" :key="panel.id" class="panel-list-item"
            :class="{ active: selectedPanelId === panel.id }" @click="selectedPanelId = panel.id">
            <span class="panel-number">{{ String(index + 1).padStart(2, '0') }}</span>
            <span><b>{{ panel.prompt || `Panel ${index + 1}` }}</b><small>{{ panel.imagePath ? basename(panel.imagePath) : 'Image not generated' }}</small></span>
          </button>
        </div>
        <button class="secondary-button full" :disabled="progress.visible || !comicDoc.panels.length" @click="generateAllPanels">전체 패널 생성</button>
        <button class="secondary-button full" :disabled="progress.visible || !allPanelsHaveImages" @click="animateAllPanels">전체 패널 영상화</button>
      </aside>

      <section class="comic-canvas-pane">
        <div class="comic-stage-toolbar">
          <span>{{ comicDoc.panels.length }} PANELS · {{ comicDoc.style }}</span>
          <div>
            <button class="tool-button" @click="exportComic('png')">PNG 내보내기</button>
            <button class="tool-button" @click="exportComic('webp')">WebP 내보내기</button>
            <button class="tool-button" :disabled="!allPanelsHaveVideos" @click="exportLivingComic">리빙 내보내기</button>
          </div>
        </div>
        <div class="page-shell">
          <div class="comic-page" :class="`layout-${comicDoc.layout}`" :style="previewGridStyle">
            <button v-for="(panel, index) in comicDoc.panels" :key="panel.id" class="comic-panel"
              :class="{ selected: selectedPanelId === panel.id, hero: comicDoc.layout === 'hero' && index === 0 }"
              @click="selectedPanelId = panel.id">
              <video v-if="panel.videoPath" :src="displayMedia(panel.videoPath)" muted autoplay loop playsinline />
              <img v-else-if="panel.imagePath" :src="displayMedia(panel.imagePath)" :alt="`Panel ${index + 1}`" />
              <span v-else class="panel-placeholder"><b>{{ index + 1 }}</b><small>{{ panel.prompt || 'Empty panel' }}</small></span>
              <span v-for="bubble in panel.bubbles" :key="bubble.id" class="comic-bubble" :class="bubble.kind"
                :style="bubbleStyle(bubble)">{{ bubble.text || '...' }}</span>
            </button>
          </div>
        </div>
      </section>

      <aside class="inspector scroll-pane">
        <template v-if="selectedPanel">
          <div class="section-heading compact">
            <div><span class="section-kicker">PANEL {{ selectedPanelIndex + 1 }}</span><h3>속성</h3></div>
            <button class="panel-delete" :disabled="comicDoc.panels.length <= 1" @click="deleteSelectedPanel">삭제</button>
          </div>
          <div class="inspector-image">
            <video v-if="selectedPanel.videoPath" :src="displayMedia(selectedPanel.videoPath)" muted autoplay loop controls />
            <img v-else-if="selectedPanel.imagePath" :src="displayMedia(selectedPanel.imagePath)" alt="Selected panel" />
            <span v-else>이미지 없음</span>
            <button v-host-dialog="'creator_select_media'" @click="pickMedia(`comic_panel_${selectedPanel.id}`)">이미지 선택</button>
          </div>
          <label class="field wide"><span>프롬프트</span><textarea v-model="selectedPanel.prompt" rows="5" /></label>
          <label class="field wide"><span>네거티브</span><textarea v-model="selectedPanel.negative" rows="3" /></label>
          <label class="field wide"><span>모션 · 카메라</span><textarea v-model="selectedPanel.motion" rows="3" /></label>
          <button class="secondary-button full" :disabled="progress.visible || !selectedPanel.prompt.trim()" @click="generatePanel(selectedPanel)">패널 생성</button>

          <div class="divider" />
          <div class="bubble-heading"><span class="field-caption">말풍선</span><button class="mini-accent" @click="addBubble">＋ ADD</button></div>
          <div v-if="!selectedPanel.bubbles.length" class="empty-small">대사 · 나레이션 없음</div>
          <div v-for="(bubble, index) in selectedPanel.bubbles" :key="bubble.id" class="bubble-editor">
            <div class="bubble-editor-head">
              <select v-model="bubble.kind"><option value="speech">Speech</option><option value="thought">Thought</option><option value="narration">Narration</option></select>
              <span>#{{ index + 1 }}</span><button title="Delete bubble" @click="deleteBubble(bubble.id)">×</button>
            </div>
            <textarea v-model="bubble.text" rows="3" placeholder="Bubble text..." />
            <div class="bubble-position">
              <label>X<input v-model.number="bubble.x" type="number" min="0" max="100" /></label>
              <label>Y<input v-model.number="bubble.y" type="number" min="0" max="100" /></label>
              <label>W<input v-model.number="bubble.width" type="number" min="10" max="90" /></label>
              <label>H<input v-model.number="bubble.height" type="number" min="5" max="60" /></label>
            </div>
          </div>
        </template>
        <div v-else class="empty-inspector">편집할 패널을 선택하세요</div>
      </aside>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { vHostDialog } from '../utils/hostDialogs'
import {
  createComicRecoveryMirror,
  parseComicRecovery,
  prepareComicDocumentForBackend,
  recoveryDocument,
  saveConflictDecision,
  shouldAcceptAuthoritativeRevision,
  verifyComicRecoveryMirror,
} from '../utils/comicRecovery'
import { mediaUrl } from '../utils/media.js'
import { useViewMode } from '../composables/useViewMode'

type VideoMode = 't2v' | 'i2v' | 'v2v'
type BubbleKind = 'speech' | 'thought' | 'narration'

interface ComicBubble { id: string; text: string; kind: BubbleKind; x: number; y: number; width: number; height: number }
interface ComicPanel { id: string; prompt: string; negative: string; motion: string; imagePath: string; videoPath: string; bubbles: ComicBubble[] }
interface ComicDocument {
  version: number; id: string; title: string; scene: string; style: string; layout: string
  width: number; height: number; panels: ComicPanel[]
  revision: number; contentHash: string; updatedAt: number
  [key: string]: any
}
interface CreatorResult { path?: string; url?: string; mediaType?: string; type?: string; thumbnail?: string; [key: string]: any }

const STORAGE_KEY = 'creatorStudioComicDocument.v1'
const FORM_STORAGE_KEY = 'creatorStudioForms.v1'
const videoModes = [
  { id: 't2v' as VideoMode, label: 'T2V' },
  { id: 'i2v' as VideoMode, label: 'I2V' },
  { id: 'v2v' as VideoMode, label: 'V2V' },
]
const layouts = [
  { id: 'auto', label: 'Auto' },
  { id: 'grid', label: 'Grid' },
  { id: 'vertical', label: 'Vertical' },
  { id: 'horizontal', label: 'Strip' },
  { id: 'hero', label: 'Hero' },
]
const comicStyles = ['Anime', 'Manga', 'Webtoon', 'Cinematic', 'Painterly', 'Graphic Novel']

/** 모드는 왼쪽 레일이 정한다 — 여기선 읽기만 한다. `useViewMode` 참조. */
const { mode: creatorMode } = useViewMode('creator')
const creatorState = ref<any>({ status: 'idle', ready: false })
const lastResult = ref<CreatorResult | null>(null)
const progress = ref({ visible: false, percent: 0, stage: '', message: '' })
const unsubscribers: Array<() => void> = []

const savedForms = loadJson(FORM_STORAGE_KEY, {})
const videoForm = ref({
  mode: (savedForms.video?.mode || 't2v') as VideoMode,
  prompt: savedForms.video?.prompt || '', negative: savedForms.video?.negative || '',
  sourcePath: savedForms.video?.sourcePath || '', identityPath: savedForms.video?.identityPath || '',
  audioPrompt: savedForms.video?.audioPrompt || '', dialogue: savedForms.video?.dialogue || '',
  width: Number(savedForms.video?.width) || 768, height: Number(savedForms.video?.height) || 448,
  frames: Number(savedForms.video?.frames) || 121, fps: 24,
  seed: Number.isFinite(Number(savedForms.video?.seed)) ? Number(savedForms.video.seed) : -1,
  includeAudio: savedForms.video?.includeAudio !== false,
})
const kreaForm = ref({
  prompt: savedForms.krea?.prompt || '', sourcePath: savedForms.krea?.sourcePath || '',
  referencePath: savedForms.krea?.referencePath || '',
  fidelity: Number.isFinite(Number(savedForms.krea?.fidelity)) ? clamp(Number(savedForms.krea.fidelity), 0.5, 12, 4) : 4,
  seed: Number.isFinite(Number(savedForms.krea?.seed)) ? Number(savedForms.krea.seed) : -1,
  hires: !!savedForms.krea?.hires, hiresScale: Number(savedForms.krea?.hiresScale) || 2,
  hiresDenoise: Number.isFinite(Number(savedForms.krea?.hiresDenoise)) ? Number(savedForms.krea.hiresDenoise) : 0.25,
})

const planner = ref({ scene: '', panelCount: 4, style: 'Anime' })
const storedComicRecoveryRaw = window.localStorage.getItem(STORAGE_KEY)
const storedComicRecovery = parseComicRecovery(storedComicRecoveryRaw)
const initialRecoveryDocument = storedComicRecovery?.schema === 2
  ? null
  : recoveryDocument(storedComicRecovery)
const comicDoc = ref<ComicDocument>(normalizeDocument(initialRecoveryDocument))
planner.value.scene = comicDoc.value.scene
planner.value.panelCount = comicDoc.value.panels.length || 4
planner.value.style = comicDoc.value.style
const selectedPanelId = ref(comicDoc.value.panels[0]?.id || '')
const saveStatus = ref('백엔드 확인 중')
const historyEntries = ref<string[]>([snapshot(comicDoc.value)])
const historyIndex = ref(0)
let historyTimer: ReturnType<typeof setTimeout> | null = null
let saveTimer: ReturnType<typeof setTimeout> | null = null
let formsTimer: ReturnType<typeof setTimeout> | null = null
let restoringHistory = false
let applyingBackendDocument = false
let persistenceReady = false
let authoritativeRevision = comicDoc.value.revision
let authoritativeContentHash = comicDoc.value.contentHash
let documentChangeSequence = 0
let mirrorWriteSequence = 0
let saveRequestSequence = 0
let activeSave: { requestId: string; changeSequence: number } | null = null
let pendingSave = false

/** 상태 문자열은 백엔드가 영어 코드로 준다 — 화면에는 한국어로 옮긴다. */
const STATE_LABELS: Record<string, string> = {
  idle: '대기', ready: '준비됨', busy: '작업 중', offline: '연결 안 됨', error: '오류',
}
const stateLabel = computed(() => {
  // 백엔드는 원인을 `error` 키로 보낸다(빈 문자열일 수 있어 ?? 대신 ||).
  if (creatorState.value.status === 'error') {
    return creatorState.value.error || creatorState.value.message || '백엔드 오류'
  }
  if (creatorState.value.busy || progress.value.visible) return '작업 중'
  if (creatorState.value.ready) return '준비됨'
  const raw = String(creatorState.value.status || 'offline').toLowerCase()
  return STATE_LABELS[raw] ?? raw
})
const stateClass = computed(() => creatorState.value.status === 'error' ? 'error' : creatorState.value.ready ? 'ready' : '')
const canGenerateVideo = computed(() => !!videoForm.value.prompt.trim() && (videoForm.value.mode === 't2v' || !!videoForm.value.sourcePath))
const canGenerateKrea = computed(() => !!kreaForm.value.prompt.trim() && !!kreaForm.value.sourcePath && !!kreaForm.value.referencePath)
const allPanelsHaveImages = computed(() => !!comicDoc.value.panels.length && comicDoc.value.panels.every(panel => !!panel.imagePath))
const allPanelsHaveVideos = computed(() => !!comicDoc.value.panels.length && comicDoc.value.panels.every(panel => !!panel.videoPath))
const selectedPanel = computed(() => comicDoc.value.panels.find(p => p.id === selectedPanelId.value) || null)
const selectedPanelIndex = computed(() => Math.max(0, comicDoc.value.panels.findIndex(p => p.id === selectedPanelId.value)))
const canUndo = computed(() => historyIndex.value > 0)
const canRedo = computed(() => historyIndex.value < historyEntries.value.length - 1)
const previewGridStyle = computed(() => {
  const count = Math.max(1, comicDoc.value.panels.length)
  const layout = comicDoc.value.layout
  if (layout === 'horizontal') return { gridTemplateColumns: `repeat(${count}, minmax(0, 1fr))`, gridTemplateRows: '1fr' }
  if (layout === 'vertical') return { gridTemplateColumns: '1fr', gridTemplateRows: `repeat(${count}, minmax(0, 1fr))` }
  if (layout === 'hero') return { gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gridTemplateRows: `repeat(${Math.max(2, Math.ceil((count + 1) / 2))}, minmax(0, 1fr))` }
  const cols = count === 1 ? 1 : count <= 4 ? 2 : 3
  return { gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${Math.ceil(count / cols)}, minmax(0, 1fr))` }
})

const OutputPreview = defineComponent({
  name: 'OutputPreview',
  props: { result: { type: Object, default: null }, emptyTitle: String, emptyCopy: String },
  setup(props) {
    return () => {
      const result: any = props.result
      const path = result?.path || result?.url || ''
      if (!path) return h('div', { class: 'output-empty' }, [h('span', { class: 'output-mark' }, '◇'), h('b', props.emptyTitle), h('p', props.emptyCopy)])
      const kind = String(result.mediaType || result.type || '').toLowerCase()
      const video = kind.includes('video') || /\.(mp4|webm|mov|mkv)$/i.test(path)
      return h('div', { class: 'output-result' }, [
        video ? h('video', { src: mediaUrl(path), controls: true, autoplay: false, loop: true }) : h('img', { src: mediaUrl(path), alt: 'Creator result' }),
        h('div', { class: 'result-meta' }, [h('b', String(result.label || result.mode || 'CREATOR RESULT').toUpperCase()), h('small', path)]),
      ])
    }
  },
})

watch(comicDoc, () => {
  if (applyingBackendDocument) return
  documentChangeSequence += 1
  saveStatus.value = 'SAVING...'
  pendingSave = true
  void persistComicRecovery(true)
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(sendComicSave, 280)
  if (restoringHistory) return
  if (historyTimer) clearTimeout(historyTimer)
  historyTimer = setTimeout(pushHistory, 420)
}, { deep: true })

async function persistComicRecovery(dirty: boolean, document = comicDoc.value) {
  const sequence = ++mirrorWriteSequence
  try {
    const mirror = await createComicRecoveryMirror(backendComicDocument(document), {
      baseRevision: dirty ? authoritativeRevision : document.revision,
      baseContentHash: dirty ? authoritativeContentHash : document.contentHash,
      updatedAt: Date.now() / 1000,
      dirty,
    })
    if (sequence !== mirrorWriteSequence) return
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(mirror))
    if (dirty && !persistenceReady) saveStatus.value = '복구본 보관됨'
  } catch {
    if (sequence === mirrorWriteSequence) saveStatus.value = '복구본 저장 실패'
  }
}

function sendComicSave() {
  saveTimer = null
  if (!persistenceReady || activeSave) return
  const requestId = `comic-save-${++saveRequestSequence}`
  activeSave = { requestId, changeSequence: documentChangeSequence }
  pendingSave = false
  saveStatus.value = '백엔드 저장 중'
  requestAction('comic_save', {
    document: backendComicDocument(),
    expectedRevision: authoritativeRevision,
    requestId,
  })
}

function schedulePendingComicSave() {
  if (!pendingSave || !persistenceReady || activeSave) return
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(sendComicSave, 0)
}

function backendComicDocument(document: ComicDocument = comicDoc.value) {
  return prepareComicDocumentForBackend(JSON.parse(snapshot(document)))
}

watch([videoForm, kreaForm], () => {
  if (formsTimer) clearTimeout(formsTimer)
  formsTimer = setTimeout(() => {
    try { window.localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify({ video: videoForm.value, krea: kreaForm.value })) } catch {}
  }, 300)
}, { deep: true })

watch(() => comicDoc.value.panels, (panels) => {
  if (!panels.some(p => p.id === selectedPanelId.value)) selectedPanelId.value = panels[0]?.id || ''
}, { deep: true })

function uid(prefix = 'id') {
  try { return `${prefix}-${crypto.randomUUID()}` } catch { return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}` }
}
function loadJson(key: string, fallback: any) {
  try { const raw = window.localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback } catch { return fallback }
}
function snapshot(value: any) { return JSON.stringify(value) }
function makePanel(index = 0): ComicPanel {
  return { id: uid('panel'), prompt: '', negative: '', motion: '', imagePath: '', videoPath: '', bubbles: index === 0 ? [] : [] }
}
function normalizeBubble(raw: any): ComicBubble {
  return {
    ...(raw && typeof raw === 'object' ? raw : {}),
    id: String(raw?.id || uid('bubble')), text: String(raw?.text || ''),
    kind: ['speech', 'thought', 'narration'].includes(raw?.kind) ? raw.kind : 'speech',
    x: clamp(Number(raw?.x), 0, 100, 8), y: clamp(Number(raw?.y), 0, 100, 8),
    width: clamp(Number(raw?.width), 10, 90, 38), height: clamp(Number(raw?.height), 5, 60, 18),
  }
}
function normalizePanel(raw: any, index: number): ComicPanel {
  const source = raw || makePanel(index)
  return {
    ...(source && typeof source === 'object' ? source : {}),
    id: String(source.id || uid('panel')), prompt: String(source.prompt || source.description || ''),
    negative: String(source.negative || ''), motion: String(source.motion || source.camera || ''),
    imagePath: String(source.imagePath || source.image_path || source.path || ''),
    videoPath: String(source.videoPath || source.video_path || ''),
    bubbles: Array.isArray(source.bubbles) ? source.bubbles.map(normalizeBubble) : [],
  }
}
function normalizeDocument(raw: any): ComicDocument {
  const panelRaw = Array.isArray(raw?.panels) ? raw.panels.slice(0, 6) : [makePanel(0), makePanel(1), makePanel(2), makePanel(3)]
  if (!panelRaw.length) panelRaw.push(makePanel(0))
  return {
    ...(raw && typeof raw === 'object' ? raw : {}),
    version: 1, id: String(raw?.id || uid('comic')), title: String(raw?.title || 'Untitled Comic'),
    scene: String(raw?.scene || ''), style: String(raw?.style || 'Anime'),
    layout: layouts.some(l => l.id === raw?.layout) ? raw.layout : 'auto',
    width: 1400, height: 2100, panels: panelRaw.map(normalizePanel),
    revision: Math.max(0, Math.trunc(Number(raw?.revision) || 0)),
    contentHash: String(raw?.contentHash || raw?.content_hash || ''),
    updatedAt: Number(raw?.updatedAt || raw?.updated_at) || 0,
  }
}
function clamp(value: number, min: number, max: number, fallback: number) {
  return Number.isFinite(value) ? Math.min(max, Math.max(min, value)) : fallback
}
function decodePayload(raw: any) {
  if (raw && typeof raw === 'object') return raw
  if (typeof raw !== 'string') return raw ?? {}
  try { return JSON.parse(raw) } catch { return { value: raw } }
}
function basename(path: string) { return String(path || '').split(/[\\/]/).pop() || '' }
function displayMedia(path: string) { return mediaUrl(path) }
function toast(type: 'success' | 'error' | 'info', msg: string) { requestAction('show_toast', { type, msg }) }
function pickMedia(slot: string) { requestAction('creator_select_media', { slot }) }
function cancelCreator() { requestAction('creator_cancel', {}); progress.value.visible = false }

function generateVideo() {
  if (!canGenerateVideo.value) return
  const form = videoForm.value
  requestAction('creator_generate', {
    mode: `h3_${form.mode}`, prompt: form.prompt.trim(), negative: form.negative.trim(),
    // 보이는 슬롯만 보낸다 — 원본은 t2v 에서 숨고, 아이덴티티는 v2v 전용(i2v 에 남은 사진이
    // 시작 프레임을 덮지 않게; 백엔드도 mode 가 읽지 않는 입력은 올리지 않는다).
    sourcePath: form.mode === 't2v' ? '' : form.sourcePath,
    identityPath: form.mode === 'v2v' ? form.identityPath : '',
    width: Math.max(64, Math.round(form.width / 32) * 32),
    height: Math.max(64, Math.round(form.height / 32) * 32),
    frames: Math.round(form.frames), fps: Math.round(form.fps), seed: Math.round(form.seed),
    includeAudio: form.includeAudio, audioPrompt: form.audioPrompt.trim(), dialogue: form.dialogue.trim(),
  })
}
function generateKrea() {
  if (!canGenerateKrea.value) return
  const form = kreaForm.value
  requestAction('creator_generate', {
    mode: 'krea2', prompt: form.prompt.trim(), sourcePath: form.sourcePath,
    referencePath: form.referencePath, fidelity: form.fidelity, seed: Math.round(form.seed),
    hires: form.hires, hiresScale: form.hiresScale, hiresDenoise: form.hiresDenoise,
  })
}
function planComic() {
  requestAction('comic_plan', {
    scene: planner.value.scene.trim(),
    panelCount: planner.value.panelCount,
    style: planner.value.style,
    expectedRevision: authoritativeRevision,
  })
}
function generateAllPanels() { requestAction('comic_generate_all', { document: backendComicDocument() }) }
function animateAllPanels() {
  requestAction('comic_animate_all', {
    document: backendComicDocument(),
    videoSettings: { width: 608, height: 352, frames: 124, fps: 24, includeAudio: true },
  })
}
function exportLivingComic() {
  requestAction('comic_export_living', { document: backendComicDocument(), fps: 8, seconds: 4 })
}
function generatePanel(panel: ComicPanel) {
  requestAction('creator_generate', { mode: 'comic_panel', panel: JSON.parse(snapshot(panel)), document: backendComicDocument() })
}
function addBubble() {
  if (!selectedPanel.value) return
  const offset = (selectedPanel.value.bubbles.length * 8) % 45
  selectedPanel.value.bubbles.push({ id: uid('bubble'), text: '', kind: 'speech', x: 8 + offset, y: 8 + offset / 2, width: 38, height: 18 })
}
function addPanel() {
  if (comicDoc.value.panels.length >= 6) return
  const panel = makePanel(comicDoc.value.panels.length)
  comicDoc.value.panels.push(panel)
  selectedPanelId.value = panel.id
}
function deleteSelectedPanel() {
  if (!selectedPanel.value || comicDoc.value.panels.length <= 1) return
  const index = selectedPanelIndex.value
  comicDoc.value.panels.splice(index, 1)
  selectedPanelId.value = comicDoc.value.panels[Math.min(index, comicDoc.value.panels.length - 1)]?.id || ''
}
function deleteBubble(id: string) {
  if (!selectedPanel.value) return
  selectedPanel.value.bubbles = selectedPanel.value.bubbles.filter(b => b.id !== id)
}
function bubbleStyle(bubble: ComicBubble) {
  return { left: `${bubble.x}%`, top: `${bubble.y}%`, width: `${bubble.width}%`, minHeight: `${bubble.height}%` }
}

function pushHistory() {
  historyTimer = null
  const current = snapshot(comicDoc.value)
  if (current === historyEntries.value[historyIndex.value]) return
  const next = historyEntries.value.slice(0, historyIndex.value + 1)
  next.push(current)
  if (next.length > 51) next.shift()
  historyEntries.value = next
  historyIndex.value = next.length - 1
}
async function restoreHistory(index: number) {
  if (index < 0 || index >= historyEntries.value.length) return
  if (historyTimer) { clearTimeout(historyTimer); historyTimer = null }
  restoringHistory = true
  historyIndex.value = index
  comicDoc.value = normalizeDocument(JSON.parse(historyEntries.value[index]))
  await nextTick()
  restoringHistory = false
}
function undo() { restoreHistory(historyIndex.value - 1) }
function redo() { restoreHistory(historyIndex.value + 1) }
function replaceDocument(raw: any) {
  const normalized = normalizeDocument(raw?.document || raw)
  applyingBackendDocument = true
  comicDoc.value = normalized
  planner.value = { scene: normalized.scene, panelCount: normalized.panels.length, style: normalized.style }
  selectedPanelId.value = normalized.panels[0]?.id || ''
  void nextTick(() => {
    applyingBackendDocument = false
    pushHistory()
  })
}

function updateAuthoritativeMetadata(raw: any) {
  const normalized = normalizeDocument(raw)
  authoritativeRevision = normalized.revision
  authoritativeContentHash = normalized.contentHash
  applyingBackendDocument = true
  comicDoc.value.revision = normalized.revision
  comicDoc.value.contentHash = normalized.contentHash
  comicDoc.value.content_hash = normalized.contentHash
  comicDoc.value.updatedAt = normalized.updatedAt
  comicDoc.value.updated_at = normalized.updatedAt
  void nextTick(() => { applyingBackendDocument = false })
}

function handleComicDocumentChanged(raw: any) {
  const data = decodePayload(raw)
  const persistence = data?.persistence || { status: 'saved', source: 'legacy-event' }
  const status = String(persistence.status || 'saved')
  const requestId = String(persistence.requestId || '')
  const authoritative = data?.document || (data?.panels ? data : null)
  const matchingSave = !!activeSave && !!requestId && activeSave.requestId === requestId
  const wasPersistenceReady = persistenceReady

  if (wasPersistenceReady && authoritative) {
    const incomingRevision = normalizeDocument(authoritative).revision
    if (!shouldAcceptAuthoritativeRevision(authoritativeRevision, incomingRevision)) {
      if (matchingSave) activeSave = null
      schedulePendingComicSave()
      return
    }
  }

  if (status === 'error') {
    if (matchingSave) activeSave = null
    persistenceReady = true
    pendingSave = true
    saveStatus.value = '백엔드 저장 실패'
    toast('error', String(persistence.error || 'Comic 문서를 저장하지 못했습니다'))
    return
  }

  persistenceReady = true
  if (matchingSave) {
    const sentSequence = activeSave!.changeSequence
    activeSave = null
    if (status === 'conflict') {
      if (saveConflictDecision(sentSequence, documentChangeSequence) === 'preserve-newer-local') {
        pendingSave = true
        saveStatus.value = '최신 로컬 편집 충돌본 보관 중'
        void persistComicRecovery(true)
        schedulePendingComicSave()
        return
      }
      if (authoritative) {
        const normalized = normalizeDocument(authoritative)
        authoritativeRevision = normalized.revision
        authoritativeContentHash = normalized.contentHash
        replaceDocument(normalized)
        pendingSave = false
        void persistComicRecovery(false, normalized)
      }
      const location = persistence.conflictPath ? ` (${persistence.conflictPath})` : ''
      saveStatus.value = '저장 충돌 · 백엔드 버전 복원'
      toast('error', `동시 편집 충돌본을 별도 보관했습니다${location}`)
      return
    }
    if (authoritative) {
      if (sentSequence === documentChangeSequence) {
        const normalized = normalizeDocument(authoritative)
        authoritativeRevision = normalized.revision
        authoritativeContentHash = normalized.contentHash
        replaceDocument(normalized)
        pendingSave = false
        saveStatus.value = '백엔드 저장됨'
        void persistComicRecovery(false, normalized)
      } else {
        updateAuthoritativeMetadata(authoritative)
        pendingSave = true
        void persistComicRecovery(true)
      }
    }
    schedulePendingComicSave()
    return
  }

  // 시작 응답 또는 생성 작업의 문서 갱신이 도착하기 전에 사용자가 편집했다면,
  // 보이는 초안을 덮지 않는다. 기존 base revision으로 저장을 시도해 백엔드가
  // 안전하게 수락하거나 conflict copy로 보존하도록 한다.
  if ((!wasPersistenceReady && documentChangeSequence > 0) || pendingSave) {
    pendingSave = true
    schedulePendingComicSave()
    return
  }

  if (authoritative) {
    const normalized = normalizeDocument(authoritative)
    authoritativeRevision = normalized.revision
    authoritativeContentHash = normalized.contentHash
    replaceDocument(normalized)
    saveStatus.value = status === 'conflict'
      ? '충돌본 별도 보관됨'
      : status === 'invalid-recovery' ? '손상 복구본 격리 · 백엔드 복원' : '백엔드 저장됨'
    void persistComicRecovery(false, normalized)
    if (status === 'conflict' || status === 'invalid-recovery') {
      const location = persistence.conflictPath ? ` (${persistence.conflictPath})` : ''
      const message = status === 'invalid-recovery'
        ? '손상된 로컬 복구본을 격리하고 백엔드 문서를 복원했습니다'
        : '로컬 복구본과 달라 별도 보관했습니다'
      toast('info', `${message}${location}`)
    }
  } else if (status === 'missing' || status === 'invalid-recovery') {
    pendingSave = true
    saveStatus.value = '새 문서 저장 중'
    schedulePendingComicSave()
  }
}

function getPanelRects(count: number, layout: string, width: number, height: number) {
  const gap = Math.round(width * 0.012)
  const margin = Math.round(width * 0.035)
  const x0 = margin, y0 = margin, usableW = width - margin * 2, usableH = height - margin * 2
  const rects: Array<{ x: number; y: number; w: number; h: number }> = []
  if (layout === 'horizontal') {
    const w = (usableW - gap * (count - 1)) / count
    for (let i = 0; i < count; i++) rects.push({ x: x0 + i * (w + gap), y: y0, w, h: usableH })
    return rects
  }
  if (layout === 'vertical') {
    const h = (usableH - gap * (count - 1)) / count
    for (let i = 0; i < count; i++) rects.push({ x: x0, y: y0 + i * (h + gap), w: usableW, h })
    return rects
  }
  if (layout === 'hero' && count > 1) {
    const heroH = usableH * 0.55
    rects.push({ x: x0, y: y0, w: usableW, h: heroH })
    const rest = count - 1, cols = Math.min(rest, 3), rows = Math.ceil(rest / cols)
    const w = (usableW - gap * (cols - 1)) / cols, h = (usableH - heroH - gap - gap * (rows - 1)) / rows
    for (let i = 0; i < rest; i++) rects.push({ x: x0 + (i % cols) * (w + gap), y: y0 + heroH + gap + Math.floor(i / cols) * (h + gap), w, h })
    return rects
  }
  const cols = count === 1 ? 1 : count <= 4 ? 2 : 3
  const rows = Math.ceil(count / cols), w = (usableW - gap * (cols - 1)) / cols, h = (usableH - gap * (rows - 1)) / rows
  for (let i = 0; i < count; i++) rects.push({ x: x0 + (i % cols) * (w + gap), y: y0 + Math.floor(i / cols) * (h + gap), w, h })
  return rects
}
function loadCanvasImage(path: string): Promise<HTMLImageElement | null> {
  if (!path) return Promise.resolve(null)
  return new Promise(resolve => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = displayMedia(path)
  })
}
function drawImageCover(ctx: CanvasRenderingContext2D, img: HTMLImageElement, x: number, y: number, w: number, h: number) {
  const scale = Math.max(w / img.naturalWidth, h / img.naturalHeight)
  const sw = w / scale, sh = h / scale, sx = (img.naturalWidth - sw) / 2, sy = (img.naturalHeight - sh) / 2
  ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h)
}
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number) {
  const words = String(text || '...').split(/\s+/), lines: string[] = []
  let line = ''
  for (const word of words) {
    const test = line ? `${line} ${word}` : word
    if (ctx.measureText(test).width > maxWidth && line) { lines.push(line); line = word } else line = test
  }
  if (line) lines.push(line)
  return lines.slice(0, 6)
}
async function renderComic(format: 'png' | 'webp') {
  const canvas = window.document.createElement('canvas')
  canvas.width = comicDoc.value.width; canvas.height = comicDoc.value.height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Canvas is unavailable')
  ctx.fillStyle = '#f4f1e8'; ctx.fillRect(0, 0, canvas.width, canvas.height)
  const rects = getPanelRects(comicDoc.value.panels.length, comicDoc.value.layout, canvas.width, canvas.height)
  const images = await Promise.all(comicDoc.value.panels.map(p => loadCanvasImage(p.imagePath)))
  comicDoc.value.panels.forEach((panel, index) => {
    const rect = rects[index], img = images[index]
    ctx.save(); ctx.beginPath(); ctx.rect(rect.x, rect.y, rect.w, rect.h); ctx.clip()
    ctx.fillStyle = '#171717'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h)
    if (img) drawImageCover(ctx, img, rect.x, rect.y, rect.w, rect.h)
    else {
      ctx.fillStyle = '#3a3a3a'; ctx.font = `900 ${Math.round(Math.min(rect.w, rect.h) * 0.18)}px sans-serif`
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(String(index + 1), rect.x + rect.w / 2, rect.y + rect.h / 2)
    }
    ctx.restore(); ctx.strokeStyle = '#080808'; ctx.lineWidth = Math.max(7, canvas.width * 0.006); ctx.strokeRect(rect.x, rect.y, rect.w, rect.h)
    panel.bubbles.forEach(bubble => {
      const bx = rect.x + rect.w * bubble.x / 100, by = rect.y + rect.h * bubble.y / 100
      const bw = rect.w * bubble.width / 100, bh = Math.max(rect.h * bubble.height / 100, 70)
      ctx.save(); ctx.fillStyle = bubble.kind === 'narration' ? '#fff4b8' : '#ffffff'; ctx.strokeStyle = '#111'; ctx.lineWidth = 4
      ctx.beginPath();
      if (bubble.kind === 'narration') ctx.rect(bx, by, bw, bh)
      else ctx.ellipse(bx + bw / 2, by + bh / 2, bw / 2, bh / 2, 0, 0, Math.PI * 2)
      ctx.fill(); ctx.stroke()
      ctx.fillStyle = '#111'; ctx.font = `700 ${Math.max(20, Math.round(Math.min(bw / 8, bh / 3.6)))}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
      const lines = wrapText(ctx, bubble.text, bw * 0.78), lineH = Math.max(25, parseInt(ctx.font) * 1.2)
      const startY = by + bh / 2 - ((lines.length - 1) * lineH) / 2
      lines.forEach((line, lineIndex) => ctx.fillText(line, bx + bw / 2, startY + lineIndex * lineH))
      ctx.restore()
    })
  })
  return canvas.toDataURL(format === 'webp' ? 'image/webp' : 'image/png', 0.94)
}
async function exportComic(format: 'png' | 'webp') {
  try {
    const dataUrl = await renderComic(format)
    requestAction('comic_export_page', { format, dataUrl, document: backendComicDocument() })
  } catch (error: any) { toast('error', `Comic export failed: ${error?.message || error}`) }
}

function handleCreatorState(raw: any) { creatorState.value = { ...creatorState.value, ...decodePayload(raw) } }
function handleProgress(raw: any, total?: number, message?: string) {
  let data = decodePayload(raw)
  if (typeof raw === 'number') data = { current: raw, total, message }
  const percent = Number.isFinite(Number(data.percent)) ? Number(data.percent)
    : Number(data.total) > 0 ? Number(data.current || data.step || 0) / Number(data.total) * 100 : 0
  progress.value = {
    visible: data.done !== true && data.status !== 'complete' && percent < 100,
    percent: clamp(percent, 0, 100, 0), stage: String(data.stage || data.node || ''), message: String(data.message || ''),
  }
}
function handleResult(raw: any) {
  const data = decodePayload(raw)
  if (data.error) { progress.value.visible = false; toast('error', String(data.error)); return }
  progress.value.visible = false; lastResult.value = data
  if (data.panelId && data.path) {
    const panel = comicDoc.value.panels.find(p => p.id === data.panelId)
    if (panel) panel.imagePath = data.path
  }
}
function handleMediaSelected(raw: any) {
  const data = decodePayload(raw), slot = String(data.slot || ''), path = String(data.path || '')
  if (!path) return
  if (slot === 'video_source') videoForm.value.sourcePath = path
  else if (slot === 'video_identity') videoForm.value.identityPath = path
  else if (slot === 'krea_source') kreaForm.value.sourcePath = path
  else if (slot === 'krea_reference') kreaForm.value.referencePath = path
  else if (slot.startsWith('comic_panel_')) {
    const id = slot.slice('comic_panel_'.length), panel = comicDoc.value.panels.find(p => p.id === id)
    if (panel) panel.imagePath = path
  }
}

onMounted(async () => {
  unsubscribers.push(onBackendEvent('creatorStateChanged', handleCreatorState))
  unsubscribers.push(onBackendEvent('creatorProgress', handleProgress))
  unsubscribers.push(onBackendEvent('creatorResult', handleResult))
  unsubscribers.push(onBackendEvent('creatorMediaSelected', handleMediaSelected))
  unsubscribers.push(onBackendEvent('comicStoryboardReady', handleComicDocumentChanged))
  unsubscribers.push(onBackendEvent('comicDocumentChanged', handleComicDocumentChanged))
  let recoveryForBackend = storedComicRecovery
  const malformedStoredRecovery = !!storedComicRecoveryRaw && !storedComicRecovery
  const invalidMirror = storedComicRecovery?.schema === 2
    && !(await verifyComicRecoveryMirror(storedComicRecovery))
  if (malformedStoredRecovery || invalidMirror) {
    try {
      window.localStorage.setItem(`${STORAGE_KEY}.invalid.${Date.now()}`, storedComicRecoveryRaw || '')
      window.localStorage.removeItem(STORAGE_KEY)
    } catch {}
    recoveryForBackend = null
  }
  requestAction('creator_get_state', { comicRecovery: recoveryForBackend })
})
onUnmounted(() => {
  unsubscribers.splice(0).forEach(off => { try { off() } catch {} })
  if (historyTimer) clearTimeout(historyTimer)
  if (saveTimer) clearTimeout(saveTimer)
  if (formsTimer) clearTimeout(formsTimer)
})
</script>

<style scoped>
.creator-studio { width: 100%; height: 100%; min-height: 0; display: flex; flex-direction: column; color: var(--text-primary); background: var(--bg-primary); }
.section-kicker { display: block; color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.section-heading h3 { margin: 2px 0 0; font-size: 15px; font-weight: var(--fw-bold); letter-spacing: 0; }
/* Creator 백엔드 표시등. 아래 StatusStrip 이 보여 주는 이미지 백엔드와 **다른 것**이라
   중복이 아니다. 자체 헤더가 사라졌으므로 각 설정 열 머리로 옮겼다. */
.pane-state { display: flex; align-items: center; justify-content: flex-end; gap: 7px; margin-bottom: 2px; color: var(--text-secondary); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* 표시등 점은 배지가 아니다 — 위에 글자를 얹지 않고 배경 위에서 자기가 읽혀야 하므로
   채움색(--state-ok)이 아니라 글자색(--state-ok-fg)을 쓴다. 채움색을 쓰면 어두운 테마에서 점이 잠긴다. */
.state-dot { width: 7px; height: 7px; flex: 0 0 auto; border-radius: 50%; background: var(--edge); box-shadow: 0 0 0 3px rgba(102,102,102,.12); }
.pane-state.ready { color: var(--state-ok-fg); }.pane-state.ready .state-dot { background: var(--state-ok-fg); box-shadow: 0 0 0 3px rgba(74,222,128,.12); }
.pane-state.error { color: var(--state-alert-fg); }.pane-state.error .state-dot { background: var(--state-alert-fg); }
/* 오른쪽 여백은 알림 종 자리다 (style.css --notif-gutter). 이 줄은 자체 헤더가 사라지면서
   창 맨 위로 올라왔고, 마지막 칸이 취소 버튼이다 — 가리면 작업을 멈출 수 없다. */
.global-progress { display: grid; grid-template-columns: minmax(170px, 260px) 1fr 50px auto; align-items: center; gap: 12px; padding: 8px var(--notif-gutter) 8px 18px; border-bottom: 1px solid var(--accent-dim); background: rgba(250, 204, 21, .045); }
.progress-copy { min-width: 0; display: flex; flex-direction: column; }.progress-copy strong { color: var(--accent); font-size: var(--fs-meta); }.progress-copy span { color: var(--text-muted); font-size: var(--fs-label); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.progress-track { height: 6px; overflow: hidden; border-radius: 6px; background: var(--bg-button); }.progress-fill { height: 100%; border-radius: inherit; background: var(--accent); transition: width .2s ease; }
.progress-value { color: var(--accent); font: 10px monospace; text-align: right; }
.creator-body { flex: 1; min-height: 0; overflow: hidden; }.two-column { display: grid; grid-template-columns: 360px 1fr; }
.scroll-pane { overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
.settings-pane { min-width: 0; padding: 22px; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 14px; }
.section-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 2px; }.section-heading.compact { margin-bottom: 10px; }
/* 검사기는 창 오른쪽 끝에 붙는다 — 첫 줄의 버튼이 알림 종 밑에 깔리지 않게 자리를 비운다 */
.inspector .section-heading { padding-right: calc(var(--notif-gutter) - 18px); }
.segmented { display: flex; padding: 2px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 7px; }
.segmented button { min-height: 28px; padding: 5px 11px; border: 0; border-radius: var(--radius-base); color: var(--text-secondary); background: transparent; font-size: var(--fs-meta); font-weight: var(--fw-bold); cursor: pointer; }.segmented button.active { color: var(--on-accent); background: var(--accent-fill); }
.field { display: flex; flex-direction: column; gap: 5px; min-width: 0; }.field > span, .field-caption { color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }
.field input, .field textarea, .field select, .bubble-editor textarea, .bubble-editor select, .bubble-position input { width: 100%; box-sizing: border-box; padding: 8px 10px; border: 1px solid var(--border); border-radius: 7px; outline: 0; background: var(--bg-input); color: var(--text-primary); font: inherit; font-size: 11px; }
.field textarea, .bubble-editor textarea { resize: vertical; line-height: 1.5; }.field input:focus, .field textarea:focus, .field select:focus, .bubble-editor textarea:focus { border-color: var(--accent); }
.field select, .bubble-editor select { color-scheme: dark; }.field.wide { width: 100%; }
.field-grid { display: grid; gap: 10px; }.field-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }.field-grid.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }.span-two { grid-column: span 2; }
.toggle-field { min-height: 31px; display: flex; align-items: center; gap: 8px; padding-top: 14px; color: var(--text-secondary); font-size: var(--fs-meta); font-weight: var(--fw-medium); }.toggle-field input { width: 16px; height: 16px; accent-color: var(--accent); }
.media-slot { min-height: 58px; display: flex; align-items: center; gap: 10px; padding: 8px; border: 1px solid var(--border); border-radius: 9px; background: var(--bg-card); }.slot-preview { width: 64px; height: 46px; flex: 0 0 auto; display: grid; place-items: center; overflow: hidden; border-radius: 5px; background: var(--bg-primary); color: var(--text-muted); font-size: var(--fs-label); }.slot-preview img, .slot-preview video { width: 100%; height: 100%; object-fit: cover; }.slot-preview.audio { color: var(--accent); font-size: 20px; }
.slot-info { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }.slot-info b { font-size: var(--fs-meta); letter-spacing: 0; }.slot-info small { color: var(--text-muted); font-size: var(--fs-label); overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
button { font-family: inherit; }.primary-button, .secondary-button, .danger-button, .tool-button { min-height: 32px; border-radius: 7px; font-size: var(--fs-meta); font-weight: var(--fw-bold); letter-spacing: 0; cursor: pointer; transition: .15s ease; }.primary-button { min-height: 36px; padding: 9px 15px; border: 0; background: var(--accent-fill); color: var(--on-accent); }.primary-button:hover:not(:disabled) { filter: brightness(1.08); transform: translateY(-1px); }.primary-button:disabled, button:disabled { opacity: .35; cursor: not-allowed; }
.secondary-button { padding: 7px 12px; border: 1px solid var(--border); background: var(--bg-button); color: var(--text-secondary); }.secondary-button:hover { color: var(--accent); border-color: var(--accent); }.secondary-button.full { width: 100%; min-height: 34px; }
.danger-button { padding: 8px 13px; border: 1px solid rgba(248,113,113,.4); color: var(--state-alert-fg); background: rgba(248,113,113,.08); }.danger-button.compact { padding: 5px 9px; }.icon-button { width: 26px; height: 26px; border: 0; border-radius: 50%; color: var(--state-alert-fg); background: transparent; cursor: pointer; }
.action-row { display: flex; gap: 8px; }.action-row .primary-button { flex: 1; }.sticky-actions { position: sticky; bottom: -22px; margin: auto -22px -22px; padding: 14px 22px 20px; background: linear-gradient(transparent, var(--bg-primary) 25%); }
.output-pane { min-width: 0; min-height: 0; display: grid; place-items: center; padding: 24px; background-image: radial-gradient(circle at 1px 1px, var(--border) 1px, transparent 0); background-size: 18px 18px; }
:deep(.output-empty) { max-width: 380px; display: flex; flex-direction: column; align-items: center; text-align: center; color: var(--text-muted); }:deep(.output-empty .output-mark) { color: var(--accent); font-size: 54px; opacity: .4; }:deep(.output-empty b) { color: var(--text-secondary); font-size: 12px; letter-spacing: 0; }:deep(.output-empty p) { font-size: var(--fs-meta); line-height: 1.6; }
:deep(.output-result) { width: 100%; height: 100%; min-height: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }:deep(.output-result img), :deep(.output-result video) { max-width: 100%; max-height: calc(100% - 52px); min-height: 0; object-fit: contain; border-radius: 9px; box-shadow: 0 12px 40px rgba(0,0,0,.45); }:deep(.result-meta) { width: min(720px, 100%); display: flex; align-items: center; gap: 10px; }:deep(.result-meta b) { color: var(--accent); font-size: var(--fs-label); }:deep(.result-meta small) { flex: 1; color: var(--text-muted); font-size: var(--fs-label); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dual-media { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }.media-tile { min-height: 190px; display: grid; grid-template-rows: 1fr auto auto; gap: 3px; padding: 8px; border: 1px solid var(--border); border-radius: 10px; background: var(--bg-card); color: var(--text-primary); cursor: pointer; overflow: hidden; }.media-tile:hover { border-color: var(--accent); }.media-tile img { width: 100%; height: 150px; object-fit: cover; border-radius: 6px; }.media-tile b { font-size: var(--fs-label); }.media-tile > small { color: var(--text-muted); font-size: var(--fs-label); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.media-empty { min-height: 145px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: var(--accent); font-size: 28px; background: var(--bg-input); border-radius: 6px; }.media-empty small { color: var(--text-muted); font-size: var(--fs-label); letter-spacing: 0; }
.range-field { display: flex; flex-direction: column; gap: 7px; }.range-field > span { display: flex; justify-content: space-between; color: var(--text-muted); font-size: var(--fs-label); letter-spacing: 0; }.range-field output { color: var(--accent); font: 11px monospace; }.range-field input { width: 100%; accent-color: var(--accent); }.range-field small { color: var(--text-muted); font-size: var(--fs-label); line-height: 1.5; }
.comic-mode { display: grid; grid-template-columns: 360px minmax(300px, 1fr) 320px; }.comic-sidebar, .inspector { min-width: 0; padding: 18px; border-right: 1px solid var(--border); background: var(--bg-secondary); }.inspector { border-right: 0; border-left: 1px solid var(--border); }
.divider { height: 1px; flex: 0 0 auto; margin: 14px 0; background: var(--border); }.toolbar-row, .bubble-heading, .comic-stage-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 7px; }.tool-button { padding: 6px 9px; border: 1px solid var(--border); background: var(--bg-button); color: var(--text-muted); }.tool-button:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }.save-state { margin-left: auto; color: var(--text-muted); font-size: var(--fs-label); }
.layout-presets { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 4px; margin: 6px 0 12px; }.layout-presets button { min-width: 0; padding: 5px 2px; border: 1px solid var(--border); border-radius: 5px; background: var(--bg-input); color: var(--text-muted); font-size: 7px; cursor: pointer; }.layout-presets button.active { color: var(--accent); border-color: var(--accent); }.layout-icon { width: 22px; height: 18px; margin: 0 auto 3px; display: grid; grid-template-columns: 1fr 1fr; gap: 1px; }.layout-icon i { background: currentColor; opacity: .65; }.layout-icon.vertical { grid-template-columns: 1fr; }.layout-icon.horizontal { grid-template-columns: repeat(4,1fr); }.layout-icon.hero i:first-child { grid-column: 1 / -1; }
.panel-list { display: flex; flex-direction: column; gap: 5px; margin-bottom: 8px; }.panel-list-item { width: 100%; min-height: 45px; display: flex; align-items: center; gap: 8px; padding: 6px 8px; text-align: left; border: 1px solid transparent; border-radius: 6px; background: var(--bg-input); color: var(--text-secondary); cursor: pointer; }.panel-list-item.active { border-color: var(--accent); background: var(--accent-dim); }.panel-number { color: var(--accent); font: 10px monospace; }.panel-list-item > span:last-child { min-width: 0; display: flex; flex-direction: column; gap: 2px; }.panel-list-item b, .panel-list-item small { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }.panel-list-item b { font-size: var(--fs-label); }.panel-list-item small { color: var(--text-muted); font-size: var(--fs-label); }
.comic-canvas-pane { min-width: 0; min-height: 0; display: flex; flex-direction: column; background: var(--bg-primary); }.comic-stage-toolbar { min-height: 40px; padding: 0 12px; border-bottom: 1px solid var(--border); color: var(--text-muted); font-size: var(--fs-label); font-weight: var(--fw-bold); letter-spacing: 0; }.comic-stage-toolbar > div { display: flex; gap: 5px; }.page-shell { flex: 1; min-height: 0; overflow: auto; display: grid; place-items: center; padding: 22px; background-image: radial-gradient(circle at 1px 1px, var(--rule) 1px, transparent 0); background-size: 18px 18px; }
/* 아래 만화 지면·칸·말풍선 색은 UI 크롬이 아니라 작품 자체다. canvas 내보내기
   (renderComic)가 같은 값을 그리므로, 토큰화하면 화면 미리보기와 내보낸
   이미지가 갈라진다. 테마와 무관하게 고정. */
.comic-page { width: min(54vh, 92%); aspect-ratio: 2 / 3; padding: 2.8%; display: grid; gap: 1.2%; box-sizing: border-box; background: #f4f1e8; box-shadow: 0 15px 50px #000; }.comic-panel { position: relative; min-width: 0; min-height: 0; padding: 0; overflow: hidden; border: max(2px, .35vw) solid #080808; background: #252525; color: #fff; cursor: pointer; }.comic-panel.selected { outline: 3px solid var(--accent); outline-offset: -3px; }.comic-panel.hero { grid-column: 1 / -1; grid-row: span 2; }.comic-panel > img, .comic-panel > video { width: 100%; height: 100%; object-fit: cover; }.panel-placeholder { width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 5px; padding: 8px; box-sizing: border-box; text-align: center; }.panel-placeholder b { color: #555; font-size: 32px; }.panel-placeholder small { color: #888; font-size: 7px; line-height: 1.35; overflow: hidden; }
.comic-bubble { position: absolute; z-index: 2; display: flex; align-items: center; justify-content: center; box-sizing: border-box; padding: 4%; border: 1.5px solid #111; border-radius: 50%; background: #fff; color: #111; font-size: clamp(5px, .65vw, 10px); font-weight: var(--fw-bold); line-height: 1.15; text-align: center; overflow: hidden; pointer-events: none; }.comic-bubble.thought { border-style: dotted; }.comic-bubble.narration { border-radius: 2px; background: #fff4b8; }
.inspector-image { height: 135px; position: relative; display: grid; place-items: center; margin-bottom: 12px; overflow: hidden; border: 1px solid var(--border); border-radius: 8px; background: var(--bg-input); color: var(--text-muted); font-size: var(--fs-label); }.inspector-image img, .inspector-image video { width: 100%; height: 100%; object-fit: cover; }.inspector-image button { position: absolute; right: 6px; bottom: 6px; padding: 5px 8px; border: 1px solid var(--accent); border-radius: 5px; background: rgba(0,0,0,.78); color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }
.bubble-heading { margin-bottom: 7px; }.mini-accent { padding: 3px 7px; border: 1px solid var(--accent-dim); border-radius: 4px; background: var(--accent-dim); color: var(--accent); font-size: var(--fs-label); font-weight: var(--fw-bold); cursor: pointer; }.empty-small, .empty-inspector { color: var(--text-muted); font-size: var(--fs-label); text-align: center; padding: 15px 5px; }.empty-inspector { margin-top: 40px; }
.panel-heading { margin-top: 10px; }.panel-delete { align-self: center; padding: 4px 7px; border: 1px solid rgba(248,113,113,.3); border-radius: 4px; background: rgba(248,113,113,.06); color: var(--state-alert-fg); font-size: 7px; font-weight: var(--fw-bold); cursor: pointer; }
.bubble-editor { margin-bottom: 8px; padding: 8px; border: 1px solid var(--border); border-radius: 7px; background: var(--bg-card); }.bubble-editor-head { display: grid; grid-template-columns: 1fr auto auto; align-items: center; gap: 6px; margin-bottom: 5px; }.bubble-editor-head select { padding: 4px 6px; }.bubble-editor-head span { color: var(--text-muted); font-size: var(--fs-label); }.bubble-editor-head button { border: 0; background: transparent; color: var(--state-alert-fg); font-size: 15px; cursor: pointer; }.bubble-position { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 5px; }.bubble-position label { color: var(--text-muted); font-size: 7px; }.bubble-position input { padding: 4px; margin-top: 2px; }

@media (max-width: 1150px) {
  .comic-mode { grid-template-columns: 320px minmax(280px, 1fr) 280px; }
}
@media (max-width: 860px) {
  .two-column { grid-template-columns: 1fr; overflow-y: auto; }.settings-pane { overflow: visible; border-right: 0; border-bottom: 1px solid var(--border); }.output-pane { min-height: 420px; }
  /* 열이 전폭이 되면 오른쪽 끝 정렬인 표시등도 종 밑으로 들어간다 */
  .pane-state { padding-right: calc(var(--notif-gutter) - 18px); }
  .comic-mode { display: flex; flex-direction: column; overflow-y: auto; }.comic-sidebar, .inspector { overflow: visible; border: 0; border-bottom: 1px solid var(--border); }.comic-canvas-pane { min-height: 650px; order: 2; }.inspector { order: 3; }.comic-sidebar { order: 1; }
  .global-progress { grid-template-columns: 1fr 44px auto; }.progress-copy { grid-column: 1 / -1; }
}
</style>
