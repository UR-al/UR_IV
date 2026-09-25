// Durable DEV-ONLY offline fixture. Never import this module from src/main.js.
// HTML installs memory-only storage/network guards before these real app imports.
import { createApp, nextTick } from 'vue'
import '../src/styles/theme-fallback.css'
import '../src/style.css'
import '../src/styles/panels.css'
import '../src/styles/editorPanels.css'
import '../src/styles/galleryShared.css'
import '../src/styles/iconMotion.css'
import '../src/styles/iconMotionClaude.css'
import App from '../src/App.vue'
import Icon from '../src/components/Icon.vue'
import router, { routes } from '../src/router.js'
import { initBridge, onBackendEvent } from '../src/bridge.js'
import { setTheme } from '../src/theme/applyTheme'
import { applyIconAnimationStyle } from '../src/theme/iconAnimationPreference'
import { uiModals } from '../src/composables/uiModals.js'
import { useQuickDock } from '../src/composables/useQuickDock'
import catalog from '../../core/model_download_catalog.json'

const status = document.getElementById('preview-status')
const tools = document.getElementById('preview-tools')
const log = document.querySelector('#preview-log pre')
const entries = []
const offlineMessage = '오프라인 미리보기: 실제 실행·파일 저장·다운로드는 하지 않습니다.'
function record(message) {
  entries.push(message)
  if (entries.length > 30) entries.shift()
  log.textContent = entries.join('\n')
}
function signal() {
  const handlers = new Set()
  return { connect: fn => handlers.add(fn), disconnect: fn => handlers.delete(fn),
    emit: (...args) => { for (const fn of [...handlers]) fn(...args) } }
}
const signalCache = new Map()
function emit(name, ...args) { backend[name].emit(...args) }
const widgetValues = {
  main_prompt_text: 'outdoors, sunlight, blue_dress', neg_prompt_text: 'blurry',
  total_prompt_display: 'outdoors, sunlight, blue_dress',
  model_combo: '오프라인 예시 모델', sampler_combo: 'Euler', scheduler_combo: 'Normal',
  // 파라미터 카드(components/params/ParamsBasicCard.vue)가 읽는 위젯 이름
  steps_input: '28', cfg_input: '5', width_input: '832', height_input: '1216', shift_input: '3', seed_input: '-1',
  i2i_denoising_strength: '0.65', inpaint_denoising_strength: '0.65',
}
let prefs = { theme: 'light', themeOverrides: {}, iconAnimationStyle: 'none',
  ollamaModel: 'offline-preview:8b', ollamaUrl: 'http://offline.invalid:11434',
  uiScale: 1, autoNlGen: false, h3ConditioningCacheEnabled: true }
let instructions = { common: '', features: Object.fromEntries(
  ['expand', 'suggest', 'nl2tags', 'nl_caption', 'nl_scene', 'translate', 'creative', 'negative', 'auto_nl'].map(key => [key, ''])) }
let session = {}
let instructionPresets = []
// 메모장 모의 목록 — 충돌 사본 한 장을 같이 두어 목록 표시를 점검한다(개인 데이터 아님)
const memoSeedAt = new Date(Date.now() - 2 * 3600_000).toISOString()   // 방금 고친 메모가 위로 오게 과거 시각
let memoStore = [
  { id: 'offline-memo-1', title: '오프라인 예시 메모', text: '테마 점검용 메모입니다.\n입력하면 이 페이지 메모리에만 저장됩니다.', created_at: memoSeedAt, updated_at: new Date(Date.now() - 3600_000).toISOString(), deleted: false },
  { id: 'offline-memo-2', title: '프롬프트 아이디어 (충돌 사본)', text: 'Forge 와 앱에서 따로 고친 메모는 이렇게 사본으로 남습니다.', created_at: memoSeedAt, updated_at: memoSeedAt, deleted: false },
]
let memoSync = { available: false, target: 'local', syncing: false, last_synced_at: null, error: null }
function emitMemoState() {
  const memos = memoStore.filter(m => !m.deleted).map(({ deleted, ...m }) => m)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
  emit('memoState', json({ memos, sync: memoSync }))
}
function presetReply(scope, preset = null) { return json({ ok: true, scope, preset, presets: instructionPresets.filter(item => item.scope === scope) }) }
// README 스크린샷 모드(?readme): 문구 없는 합성 그림 여러 장·크기와 시드가 있는 결과·여러 줄의
// 검색 결과·와일드카드 샘플을 쓰고, 모의 동작 알림을 띄우지 않는다. 개인 파일은 여전히 쓰지 않는다.
const README_MODE = new URLSearchParams(location.search).has('readme')
function svgUrl(body, tag) {
  // History appends a cache-busting query; keep it in a fragment, outside SVG bytes.
  return 'data:image/svg+xml,' + encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="832" height="1216" viewBox="0 0 832 1216">${body}</svg>`) + `#${tag}`
}
const sky = (a, b) => `<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${a}"/><stop offset="1" stop-color="${b}"/></linearGradient></defs><rect width="832" height="1216" fill="url(#sky)"/>`
const figure = (dress, hair) => `<path d="M330 520 Q416 420 505 520 L512 640 Q416 590 322 640Z" fill="${hair}"/><circle cx="416" cy="585" r="68" fill="#f3d2bd"/><path d="M336 668 H496 L580 1060 H252Z" fill="${dress}"/>`
const readmeScenes = [
  { tag: 'meadow', seed: 1843021977, prompt: '1girl, solo, blue dress, meadow, sunlight, wind, smile',
    body: sky('#9ecbff', '#f6dcff') + '<circle cx="640" cy="230" r="110" fill="#ffe7a3"/><path d="M0 830 Q220 700 430 800 T832 760 V1216 H0Z" fill="#8cc084"/><path d="M0 940 Q260 850 520 940 T832 910 V1216 H0Z" fill="#5f9f68"/>' + figure('#3f6fb5', '#3b2a24') },
  { tag: 'sunset', seed: 902384411, prompt: '1girl, beach, sunset, white dress, ocean, looking at viewer',
    body: sky('#ff9a8b', '#ffd3a5') + '<circle cx="416" cy="700" r="160" fill="#fff0c2" opacity="0.9"/><rect y="760" width="832" height="456" fill="#35507f"/><path d="M0 820 H832 M60 880 H772 M120 940 H712" stroke="#9fb4dc" stroke-width="6" opacity="0.6"/>' + figure('#f4f1ea', '#5a3a2e') },
  { tag: 'city', seed: 55120387, prompt: 'night, city lights, rain, 1girl, umbrella, reflections',
    body: sky('#0f2027', '#2c5364') + '<circle cx="660" cy="200" r="70" fill="#f5f3ce"/>' +
      [[40, 520, 150], [210, 440, 130], [360, 600, 120], [500, 380, 160], [680, 500, 140]].map(([x, y, w]) =>
        `<rect x="${x}" y="${y}" width="${w}" height="${1216 - y}" fill="#1b2d3a"/>` +
        Array.from({ length: 6 }, (_, i) => `<rect x="${x + 20 + (i % 3) * 40}" y="${y + 40 + Math.floor(i / 3) * 70}" width="18" height="26" fill="#ffd66b" opacity="0.8"/>`).join('')).join('') },
  { tag: 'forest', seed: 3301927745, prompt: 'forest, sunbeam, path, scenery, no humans, lush',
    body: sky('#e3f5c3', '#9ed8a6') + [[80, 0.9], [230, 1], [420, 0.85], [600, 1], [740, 0.9]].map(([x, s]) =>
      `<path d="M${x} ${1060 - 520 * s} L${x + 130 * s} 1060 H${x - 130 * s}Z" fill="#2f6b4f"/><rect x="${x - 14}" y="1060" width="28" height="80" fill="#5b4033"/>`).join('') + '<path d="M300 1216 Q416 1000 520 1216Z" fill="#d9c79a"/>' },
  { tag: 'snow', seed: 72719, prompt: 'snowy mountains, winter, clear sky, scenery, landscape',
    body: sky('#cfe2ff', '#f4f8ff') + '<path d="M0 900 L220 480 L360 700 L520 380 L832 880 V1216 H0Z" fill="#8aa1b8"/><path d="M220 480 L270 575 L180 560Z M520 380 L590 500 L455 490Z" fill="#ffffff"/><rect y="980" width="832" height="236" fill="#f7fbff"/>' },
  { tag: 'sakura', seed: 1200773, prompt: '1girl, cherry blossoms, school uniform, petals, spring',
    body: sky('#fbd3e9', '#bb93d8') + '<rect x="120" y="360" width="46" height="856" fill="#6b4a3c"/><circle cx="143" cy="340" r="190" fill="#f7a8c8" opacity="0.9"/><circle cx="700" cy="260" r="150" fill="#f9bdd6" opacity="0.8"/>' + figure('#2f3b5c', '#241a18') },
]
const readmeImages = readmeScenes.map(scene => svgUrl(scene.body, `readme-${scene.tag}`))
const sampleImage = README_MODE ? readmeImages[0] : 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="768"><rect width="640" height="768" fill="#d7e3e9"/><circle cx="440" cy="130" r="70" fill="#eed996"/><path d="M0 520L160 350 320 500 500 300 640 470V768H0Z" fill="#81998c"/><circle cx="310" cy="280" r="55" fill="#ddb9a1"/><path d="M250 350H370L410 590H210Z" fill="#4c698e"/><text x="26" y="724" font-family="sans-serif" font-size="22" fill="#1b1b19">Offline theme sample · no personal files</text></svg>') + '#offline-preview'
const libraryImages = README_MODE ? readmeImages : [sampleImage]
const sampleRows = README_MODE ? [
  { id: 1, copyright: 'original', character: '', artist: 'fixture_artist', general: '1girl solo smile standing blue_dress blue_eyes long_hair outdoors meadow sunlight', rating: 'g', image_width: 832, image_height: 1216 },
  { id: 2, copyright: 'original', character: '', artist: 'sample_painter', general: '1girl beach sunset white_dress ocean looking_at_viewer', rating: 'g', image_width: 1024, image_height: 1024 },
  { id: 3, copyright: 'original', character: '', artist: 'sample_painter', general: 'night city rain umbrella reflection 1girl', rating: 's', image_width: 896, image_height: 1152 },
] : [{ id: 1, copyright: 'original', character: '', artist: 'fixture_artist',
  general: '1girl smile standing blue_dress blue_eyes outdoors sunlight', rating: 'g', image_width: 640, image_height: 768 }]
const sampleWildcards = README_MODE ? [
  { name: 'hairstyle', file: 'hairstyle.txt', tags: ['long hair', 'short hair', 'twintails', 'ponytail', 'bob cut'], lines: ['# 머리 모양 — 한 줄에서 하나를 고른다', 'long hair, short hair, bob cut', 'twintails, ponytail'] },
  { name: 'outfit', file: 'outfit.txt', tags: ['school uniform', 'white dress', 'hoodie', 'kimono'], lines: ['school uniform, white dress', 'hoodie, kimono'] },
  { name: 'background', file: 'background.txt', tags: ['meadow', 'beach', 'city', 'forest'], lines: ['meadow, beach, city, forest'] },
] : []
const categories = { smile: 'expression', standing: 'pose', blue_dress: 'clothing', blue_eyes: 'character_trait',
  outdoors: 'background', sunlight: 'effect', fixture_artist: 'artist', original: 'copyright' }
const metadata = { source: 'comfyui', path: sampleImage, prompt: 'outdoors, sunlight, blue_dress', negative: 'blurry',
  params_line: 'Steps: 20, Sampler: Euler, CFG scale: 5, Seed: 123',
  raw: 'outdoors, sunlight, blue_dress\nNegative prompt: blurry\nSteps: 20, Sampler: Euler, CFG scale: 5, Seed: 123',
  can_apply: true, metadata_warnings: [], prompt_candidates: [] }
const xyzAxes = [
  { id: 'steps', label: 'Steps', type: 'integer', min: 1, max: 150 },
  { id: 'cfg_scale', label: 'CFG Scale', type: 'number', min: 0, max: 100 },
  { id: 'sampler_name', label: 'Sampler', type: 'choice', choices: ['Euler', 'DPM++ 2M'] },
]
const json = value => JSON.stringify(value)
const reply = value => (...args) => {
  const callback = args.at(-1)
  if (typeof callback === 'function') callback(typeof value === 'function' ? value(...args) : value)
}
const methods = {
  getAllWidgetValues: reply(() => json(widgetValues)),
  getWidgetValue: reply(id => widgetValues[id] ?? ''),
  onWidgetChanged: (id, value) => { widgetValues[id] = value },
  getInitialConfig: reply(() => json({ uiPrefs: prefs, condRules: { positive: [], negative: [] }, globalWeights: [] })),
  getUiPrefs: reply(() => json(prefs)), getSettings: reply('{}'), getTabDefaults: reply('{}'),
  saveChatSchemaDraft: (schemaText, callback) => {
    prefs.chatSettingsV2 = { ...prefs.chatSettingsV2, schemaText }
    callback(json({ ok: true, schemaText })); record('JSON 스키마 초안: 메모리에만 자동 저장')
  },
  getAiAssistInstructions: reply(() => json({ ok: true, instructions })),
  saveAiAssistInstructions: (raw, callback) => { instructions = JSON.parse(raw); const result = json({ ok: true, instructions }); emit('aiAssistInstructionsChanged', result); callback(result); record('AI 어시스트 지침: 메모리에만 저장') },
  getInstructionPresets: (scope, callback) => callback(presetReply(scope)),
  saveInstructionPreset: (raw, callback) => {
    const data = JSON.parse(raw)
    const item = { ...data, id: data.id || crypto.randomUUID() }
    instructionPresets = [...instructionPresets.filter(p => p.id !== item.id), item]
    const result = presetReply(item.scope, item)
    emit('instructionPresetsChanged', result); callback(result); record('지침 프리셋: 메모리에만 저장')
  },
  deleteInstructionPreset: (raw, callback) => {
    const data = JSON.parse(raw)
    instructionPresets = instructionPresets.filter(p => p.id !== data.id || p.scope !== data.scope)
    const result = presetReply(data.scope)
    emit('instructionPresetsChanged', result); callback(result); record('지침 프리셋: 모의 목록에서만 삭제')
  },
  getSession: reply(() => json(session)), saveSession: (raw, callback) => { session = JSON.parse(raw); callback?.('{}') },
  getRandomResolutions: reply('[]'), getPresetList: reply('["오프라인 샘플"]'), getPresetData: reply(json({ prompt: metadata.prompt, negative: metadata.negative })),
  getGenStats: reply('{}'), getWildcardTree: reply(() => json(sampleWildcards)), getCharFeatureOverride: reply('{}'),
  // 비동기 조회(request* → *Ready) — 응답 id 를 되돌려 줘야 화면이 '불러오는 중'에서 풀린다
  requestLoras: (mode, requestId) => emit('lorasReady', json({ requestId, mode, loras: [] })),
  requestCharacterTagsOnline: (name, requestId) => emit('characterTagsOnlineReady', json({ requestId, name, error: offlineMessage })),
  requestCompareGif: (_before, _after, _duration, _loops, requestId) => emit('compareGifReady', json({ requestId, error: offlineMessage })),
  getStatusMessage: reply(() => json({ text: README_MODE ? '' : '오프라인 테마 점검 — 상태줄 예시', level: 'info', timeoutMs: 0, at: Date.now() })),
  getTagSuggestionsRich: reply('[]'), getCharacterInsight: reply('{}'), getExcludeMatches: reply('[]'),
  classifyTags: reply(raw => json(Object.fromEntries(JSON.parse(raw).map(tag => [tag, categories[tag] || 'general'])))),
  getActiveSearchDataset: reply(json({ label: '오프라인 샘플', id: 'offline' })),
  loadLastSearchResults: reply(() => json(sampleRows)), loadFullResults: reply(() => json(sampleRows)),
  searchDanbooru: () => { emit('searchResultsReady', json(sampleRows)); record('Search: 샘플 태그만 표시') },
  getFavorites: reply(json(libraryImages)), getImageExif: reply(json(metadata)),
  getLastGalleryFolder: reply('오프라인 샘플'),
  requestGalleryImages: folder => emit('galleryImagesReady', json({ folder, files: libraryImages })),
  generateThumbnails: (raw, width) => emit('thumbnailReady', json({ width, items: JSON.parse(raw).map(path => ({ path, thumb: README_MODE && path.startsWith('data:') ? path : sampleImage })) })),
  requestOllamaModels: () => emit('ollamaModelsReady', json(['offline-preview:8b'])),
  requestADetailerModels: () => emit('adetailerModelsReady', '[]'),
  requestUpscalers: () => emit('upscalersReady', '["Lanczos"]'),
  editorCheckAutoSave: reply('{}'), getYoloModelLabel: reply('오프라인: 모델 없음'),
  getFileInfo: reply(json({ width: 640, height: 768, size: 0 })),
  copyTextToClipboard: async (value, callback) => { await navigator.clipboard.writeText(value); record('복사: 메모리 클립보드만 사용'); callback?.(true) },
}
// Unknown methods are fail-closed callbacks; unknown signals are safe in-memory signals.
// The hybrid keeps new UI subscriptions from ever falling through to a real backend.
const backend = new Proxy(methods, {
  get(target, key) {
    if (key in target) return target[key]
    if (key === 'then' || typeof key !== 'string') return undefined
    if (!signalCache.has(key)) {
      const stub = (...args) => {
        record(`실행하지 않음: ${key}`)
        const callback = args.at(-1)
        if (typeof callback === 'function') callback(json({ ok: false, error: offlineMessage }))
      }
      Object.assign(stub, signal())
      signalCache.set(key, stub)
    }
    return signalCache.get(key)
  },
})
function savePrefs(patch) {
  prefs = { ...prefs, ...patch }
  emit('uiPrefsLoaded', json(prefs))
  record('설정 변경: 이 페이지 메모리에만 적용 (새로고침 시 초기화)')
}
methods.onAction = (name, raw) => {
  const payload = JSON.parse(raw || '{}')
  if (name.startsWith('hand_reconstruction_')) {
    record(`${name}: 화면 검증용 모의 응답 (생성·저장 없음)`)
    if (name === 'hand_reconstruction_generate') {
      // Copy the input only. This fixture verifies comparison/export UI, NOT
      // Python image preparation, GPU sampling or anatomical repair quality.
      // No 'source': like the real backend, the panel compares against the image it sent.
      emit('handReconstructionEvent', json({ action: name, requestId: payload.requestId, phase: 'complete', ok: true,
        prepared: payload.image,
        candidates: Array.from({ length: payload.settings.candidates }, (_, index) => ({ index, seed: index + 100, image: payload.image })),
        warning: '오프라인 UI 모의 후보입니다. 원본 복제이며 실제 손 재구성·입력 제거·파일 저장은 하지 않습니다.' }))
    } else if (name === 'hand_reconstruction_export') {
      emit('handReconstructionEvent', json({ action: name, requestId: payload.requestId, phase: 'complete', ok: false, error: offlineMessage }))
    }
    return
  }
  if (name === 'save_ui_prefs') { savePrefs(payload); return }
  if (name === 'show_toast') { emit('showNotification', payload.type || 'info', payload.msg || offlineMessage); return }
  if (name === 'vue_tab_switch') return
  if (name === 'get_xyz_capabilities') {
    emit('xyzCapabilitiesReceived', json({ requestId: payload.requestId, ok: true, backend: 'comfyui', capabilityId: 'offline-only', axes: xyzAxes, notes: [offlineMessage], unsupported: [] })); return
  }
  // 메모장(도크) — 메모리 안의 목록만 고친다. Forge 동기화는 하지 않는다(상태 문구 확인용 가짜 상태만).
  if (name === 'memo_list') { emitMemoState(); return }
  if (name === 'memo_save') {
    const at = new Date().toISOString()
    const old = memoStore.find(m => m.id === payload.id)
    const memo = { id: payload.id, title: payload.title || '', text: payload.text || '', created_at: old?.created_at || at, updated_at: at, deleted: false }
    memoStore = [...memoStore.filter(m => m.id !== payload.id), memo]
    record('메모: 이 페이지 메모리에만 저장'); emitMemoState(); return
  }
  if (name === 'memo_delete') {
    const at = new Date().toISOString()
    memoStore = memoStore.map(m => m.id === payload.id ? { ...m, text: '', deleted: true, updated_at: at } : m)
    record('메모: 모의 목록에서만 삭제'); emitMemoState(); return
  }
  if (name === 'memo_sync') { record('메모 동기화: 오프라인 — Forge 에 요청하지 않음'); emitMemoState(); return }
  if (name === 'chat_load') {
    emit('chatThreads', json([{ id: 'offline-chat', title: '오프라인 예시 대화', model: 'offline-preview:8b', messages: [{ id: 'sample-message', role: 'assistant', content: '테마 점검용 예시입니다. 실제 모델 호출은 하지 않습니다.', createdAt: Date.now() }], createdAt: Date.now(), updatedAt: Date.now() }])); return
  }
  if (name === 'chat_model_info') { emit('chatModelInfo', json({ id: payload.id, model: payload.model, ok: true, info: { architecture: 'offline', moe: null, vision: false } })); return }
  if (name === 'chat_models') { emit('chatModelsReady', json({ id: payload.id, ok: true, models: ['offline-lmstudio'] })); return }
  if (name === 'chat_send') {
    record(`모의 대화 요청: ${payload.provider || 'ollama'} · JSON 스키마 ${payload.schema ? 'ON' : 'OFF'} · 실제 모델 호출 없음`)
    const content = payload.schema ? json({ tags: ['outdoors'], caption: 'A scene in daylight. Soft light fills the scene.', explanation_ko: '오프라인 모의 출력이며 실제 모델 결과가 아닙니다.' }) : '오프라인 모의 답변입니다.'
    emit('chatToken', json({ id: payload.id, text: content }))
    emit('chatDone', json({ id: payload.id, ok: true, content, structured: !!payload.schema, doneReason: 'stop' })); return
  }
  if (name === 'model_download_status') {
    emit('modelDownloadEvent', json({ available: true, state: 'idle', busy: false, files: catalog.artifacts.map(file => ({ ...file, status: 'missing' })), packs: catalog.packs.map(pack => ({ ...pack, fileIds: pack.artifact_ids, ready: false, runtimeReady: false, downloadable: false, installedCount: 0, missingCount: pack.artifact_ids.length, blockedReason: offlineMessage })), selectedPackIds: [], message: offlineMessage })); return
  }
  if (name === 'creator_h3_cache_status') { emit('creatorCacheEvent', json({ operation: 'status', requestId: payload.requestId, ok: true, available: false, entries: 0, bytes: 0, message: offlineMessage })); return }
  if (name === 'save_global_weights') { emit('globalWeightsLoaded', json(payload.weights || [])); return }
  if (name === 'chat_save' || name.startsWith('save_') || name.startsWith('set_')) { record(`${name}: 실제 저장 없이 무시`); return }
  record(`실행하지 않음: ${name}`)
  if (!README_MODE) emit('showNotification', 'info', offlineMessage)
  if (/generate|chat_send|start_xyz/.test(name)) {
    emit('generationError', offlineMessage)
    emit('chatDone', json({ id: payload.id, ok: false, error: offlineMessage }))
    emit('xyzPlotEvent', json({ requestId: payload.requestId, ok: false, type: 'error', error: offlineMessage }))
  }
}
window.qt = { webChannelTransport: {} }
window.QWebChannel = function (_transport, callback) { callback({ objects: { backend } }) }
await initBridge()
setTheme({ preset: prefs.theme, overrides: prefs.themeOverrides })
applyIconAnimationStyle(prefs.iconAnimationStyle)
onBackendEvent('uiPrefsLoaded', raw => applyIconAnimationStyle(JSON.parse(raw).iconAnimationStyle))
const app = createApp(App)
app.config.errorHandler = error => {
  status.dataset.state = 'error'
  status.textContent = `화면 오류: ${error?.message || error}`
  record(status.textContent)
  console.error(error)
}
app.use(router).component('Icon', Icon).mount('#app')
await router.isReady()
await nextTick()
emit('uiPrefsLoaded', json(prefs))
emit('condRulesLoaded', json({ positive: [], negative: [] }))
emit('globalWeightsLoaded', '[]')
emit('vramUpdated', json({ used: 0, total: 0, percent: 0 }))

function button(label, callback) {
  const element = document.createElement('button')
  element.type = 'button'
  element.textContent = label
  element.dataset.search = label.toLowerCase()
  element.addEventListener('click', () => Promise.resolve().then(callback).catch(error => {
    status.dataset.state = 'error'
    status.textContent = `미리보기 조작 오류: ${error.message}`
    record(status.textContent)
  }))
  tools.append(element)
}
async function navigate(name) {
  // Click the real navigation control so App sidebar state and router stay in sync.
  const title = routes.find(route => route.name === name)?.meta?.title
  const navButton = [...document.querySelectorAll('nav[aria-label="탭 내비게이션"] button')].find(element => element.getAttribute('aria-label') === title)
  if (!navButton) throw new Error(`탭 버튼을 찾지 못했습니다: ${name}`)
  navButton.click()
  await router.push({ name })
  await nextTick()
}
function changeTheme(preset, overrides = {}) { setTheme({ preset, overrides }, savePrefs) }
for (const preset of ['light', 'default', 'dark']) button(preset, () => changeTheme(preset))
button('Accent Blue', () => changeTheme(prefs.theme, { ...prefs.themeOverrides, accent: '#123456' }))
button('Accent White', () => changeTheme(prefs.theme, { ...prefs.themeOverrides, accent: '#FFFFFF' }))
button('State White', () => changeTheme(prefs.theme, { ...prefs.themeOverrides, 'state-ok': '#FFFFFF', 'state-alert': '#FFFFFF' }))
for (const route of routes) button(route.meta.title, () => navigate(route.name))
for (const [label, key] of [['캐릭터 프리셋', 'charPreset'], ['A/B 모달', 'abTest'], ['캐릭터 Override', 'charOverride']]) button(label, () => { uiModals[key] = true })
for (const [label, text] of [['프리셋 모달', '프리셋'], ['가중치 모달', '가중치'], ['와일드카드 모달', '와일드카드'], ['즉석 WC 모달', '즉석 WC'], ['순서 모달', '순서'], ['통계 모달', '통계'], ['조건부 모달', '조건부']]) {
  button(label, async () => {
    await navigate('t2i')
    const target = [...document.querySelectorAll('#app .tool-btn')].find(element => element.textContent.trim().startsWith(text))
    if (!target) throw new Error(`${text} 버튼을 찾지 못했습니다. 앱의 도구 메뉴를 확인하세요.`)
    target.click()
  })
}
button('샘플 이미지', () => {
  for (const event of ['inpaintImageLoaded', 'pngInfoImageLoaded', 'i2iImageLoaded', 'editorImageLoaded']) emit(event, sampleImage)
  if (README_MODE) {
    // 히스토리가 여러 장으로 차도록 — 마지막(첫 장면)이 맨 위에 온다
    for (const [index, scene] of [...readmeScenes.entries()].reverse()) {
      emit('imageGenerated', json({ path: readmeImages[index], prompt: scene.prompt, negative: 'lowres, blurry', width: 832, height: 1216, seed: scene.seed }))
    }
    return
  }
  emit('imageGenerated', json({ path: sampleImage, prompt: metadata.prompt, negative: metadata.negative }))
})
button('Search 샘플', async () => { await navigate('search'); emit('searchResultsReady', json(sampleRows)) })
button('Favorites 샘플', () => navigate('fav'))
button('Gallery 샘플', async () => { await navigate('gallery'); emit('galleryImagesReady', json({ folder: '오프라인 샘플', files: libraryImages })) })
button('PNG Info 샘플', async () => { await navigate('png'); emit('pngInfoImageLoaded', sampleImage) })
// 우하단 도크 — 펼친 아이콘 줄 · 대기열 드로어 · 작은 대화 패널 · 메모장
const quickDock = useQuickDock()
button('도크 펼치기', () => { quickDock.activePanel.value = null; quickDock.expanded.value = true })
for (const [label, panel] of [['도크 대기열', 'queue'], ['도크 대화', 'chat'], ['도크 메모장', 'memo']]) button(label, () => quickDock.openPanel(panel))
button('메모 동기화 상태', () => {
  // 로컬 전용 → Forge 동기화됨 → 동기화 오류 순으로 돌려 머리 문구 세 가지를 본다(실제 요청 없음)
  memoSync = !memoSync.available ? { available: true, target: 'forge', syncing: false, last_synced_at: new Date().toISOString(), error: null }
    : !memoSync.error ? { ...memoSync, error: '오프라인 미리보기 — 모의 동기화 오류' }
      : { available: false, target: 'local', syncing: false, last_synced_at: null, error: null }
  emitMemoState()
})
button('샘플 알림', () => { for (const type of ['success', 'info', 'error']) emit('showNotification', type, `${type} 테마 점검용 알림`) })
const label = document.createElement('label')
label.htmlFor = 'preview-filter'
label.textContent = '도구 검색 '
const filter = document.createElement('input')
filter.id = 'preview-filter'
filter.type = 'search'
filter.placeholder = 'Search / 모달 / 샘플…'
filter.addEventListener('input', () => {
  for (const element of tools.querySelectorAll('button')) element.hidden = !element.dataset.search.includes(filter.value.toLowerCase())
})
label.append(filter)
tools.prepend(label)
tools.hidden = false
document.getElementById('preview-help').hidden = true
document.getElementById('preview-log').hidden = false
if (status.dataset.state !== 'error') {
  status.dataset.state = 'ready'
  status.textContent = '오프라인 테마 미리보기 준비 완료 · 실제 Vue 화면 · 생성/다운로드/개인 파일 연결 없음 · 설정은 새로고침 시 초기화'
}
record('준비 완료. 왼쪽 실제 앱 탭 또는 위 미리보기 도구를 사용하세요.')
