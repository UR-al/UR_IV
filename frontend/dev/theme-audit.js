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
  steps_slider: '20', cfg_slider: '5', width_spin: '768', height_spin: '1024', seed_input: '-1',
  i2i_denoising_strength: '0.65', inpaint_denoising_strength: '0.65',
}
let prefs = { theme: 'light', themeOverrides: {}, iconAnimationStyle: 'none',
  ollamaModel: 'offline-preview:8b', ollamaUrl: 'http://offline.invalid:11434',
  uiScale: 1, autoNlGen: false, h3ConditioningCacheEnabled: true }
let instructions = { common: '', features: Object.fromEntries(
  ['expand', 'suggest', 'nl2tags', 'nl_caption', 'nl_scene', 'translate', 'creative', 'negative', 'auto_nl'].map(key => [key, ''])) }
let session = {}
let instructionPresets = []
function presetReply(scope, preset = null) { return json({ ok: true, scope, preset, presets: instructionPresets.filter(item => item.scope === scope) }) }
// History appends a cache-busting query; keep it in a fragment, outside SVG bytes.
const sampleImage = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="768"><rect width="640" height="768" fill="#d7e3e9"/><circle cx="440" cy="130" r="70" fill="#eed996"/><path d="M0 520L160 350 320 500 500 300 640 470V768H0Z" fill="#81998c"/><circle cx="310" cy="280" r="55" fill="#ddb9a1"/><path d="M250 350H370L410 590H210Z" fill="#4c698e"/><text x="26" y="724" font-family="sans-serif" font-size="22" fill="#1b1b19">Offline theme sample · no personal files</text></svg>') + '#offline-preview'
const sampleRows = [{ id: 1, copyright: 'original', character: '', artist: 'fixture_artist',
  general: '1girl smile standing blue_dress blue_eyes outdoors sunlight', rating: 'g', image_width: 640, image_height: 768 }]
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
  getGenStats: reply('{}'), getWildcardTree: reply('[]'), getLoras: reply('[]'), getCharFeatureOverride: reply('{}'),
  getTagSuggestions: reply('[]'), getCharacterInsight: reply('{}'), getExcludeMatches: reply('[]'),
  classifyTags: reply(raw => json(Object.fromEntries(JSON.parse(raw).map(tag => [tag, categories[tag] || 'general'])))),
  getActiveSearchDataset: reply(json({ label: '오프라인 샘플', id: 'offline' })),
  loadLastSearchResults: reply(() => json(sampleRows)), loadFullResults: reply(() => json(sampleRows)),
  searchDanbooru: () => { emit('searchResultsReady', json(sampleRows)); record('Search: 샘플 태그만 표시') },
  getFavorites: reply(json([sampleImage])), getImageExif: reply(json(metadata)),
  getLastGalleryFolder: reply('오프라인 샘플'), getGalleryImages: reply(json([sampleImage])),
  requestGalleryImages: folder => emit('galleryImagesReady', json({ folder, files: [sampleImage] })),
  generateThumbnails: raw => JSON.parse(raw).forEach(path => emit('thumbnailReady', json({ path, thumb: sampleImage }))),
  requestOllamaModels: () => emit('ollamaModelsReady', json(['offline-preview:8b'])),
  ollamaListModels: reply('["offline-preview:8b"]'),
  getADetailerModels: reply('[]'), requestADetailerModels: () => emit('adetailerModelsReady', '[]'),
  getUpscalers: reply('["Lanczos"]'), requestUpscalers: () => emit('upscalersReady', '["Lanczos"]'),
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
      emit('handReconstructionEvent', json({ action: name, requestId: payload.requestId, phase: 'complete', ok: true,
        source: payload.image, prepared: payload.image,
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
  emit('showNotification', 'info', offlineMessage)
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
  for (const event of ['inpaintImageLoaded', 'i2iImageLoaded', 'editorImageLoaded']) emit(event, sampleImage)
  emit('imageGenerated', json({ path: sampleImage, prompt: metadata.prompt, negative: metadata.negative }))
})
button('Search 샘플', async () => { await navigate('search'); emit('searchResultsReady', json(sampleRows)) })
button('Favorites 샘플', () => navigate('fav'))
button('Gallery 샘플', async () => { await navigate('gallery'); emit('galleryImagesReady', json({ folder: '오프라인 샘플', files: [sampleImage] })) })
button('PNG Info 샘플', async () => { await navigate('png'); emit('inpaintImageLoaded', sampleImage) })
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
