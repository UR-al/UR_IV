// Development-only, explicit offline bridge fixture. Not a production entry.
// Every action below stays in memory; no model downloads or GPU/API calls occur.
import { createApp, h, KeepAlive, ref } from 'vue'
import { initBridge } from '../src/bridge.js'
import { applyTheme, setTheme } from '../src/theme/applyTheme'
import Icon from '../src/components/Icon.vue'
import SettingsView from '../src/views/SettingsView.vue'
import ChatView from '../src/views/ChatView.vue'
import PngInfoView from '../src/views/PngInfoView.vue'
import XYZPlotView from '../src/views/XYZPlotView.vue'
import catalog from '../../core/model_download_catalog.json'
import '../src/styles/theme-fallback.css'
import '../src/style.css'
import '../src/styles/iconMotion.css'
import '../src/styles/iconMotionClaude.css'

function signal() {
  const callbacks = new Set()
  return { connect: fn => callbacks.add(fn), disconnect: fn => callbacks.delete(fn),
    emit: (...args) => callbacks.forEach(fn => fn(...args)) }
}
const backend = Object.fromEntries([
  'widgetValueChanged', 'widgetPropertyChanged', 'batchUpdate', 'uiPrefsLoaded',
  'modelDownloadEvent', 'creatorCacheEvent', 'chatToken', 'chatDone', 'chatThreads',
  'chatGenerationEvent', 'ollamaModelsReady', 'appUpdateEvent', 'chatModelInfo',
  'inpaintImageLoaded', 'pngInfoImageLoaded', 'xyzCapabilitiesReceived', 'xyzPlotEvent',
].map(name => [name, signal()]))
let prefs = { h3ConditioningCacheEnabled: true, h3ConditioningCacheMaxGB: 8, h3ConditioningCacheMaxEntries: 32 }
const modelNames = ['tinyrick/gemma-4-31B-it-uncensored-heretic-vision-llmfan46:Q6_K_M', 'gpt-oss:20b', 'qwen3:8b']
let threads = [{ id: 'offline-chat', title: '복사 검증', model: modelNames[0], createdAt: Date.now(), updatedAt: Date.now(),
  messages: [{ id: 'sample-message', role: 'assistant', content: '복사 검증용 응답입니다.\ncherry_blossoms, outdoors, sunlight', createdAt: Date.now() }] }]
let activeRequest = null
let cacheEntries = 3
const actions = ref([])
const files = catalog.artifacts.map(file => ({ ...file, path: `C:/offline-models/${file.category}/${file.filename}`, sourceUrl: file.source_url, status: 'missing' }))
function modelState(extra = {}) {
  return { available: true, state: 'idle', busy: false, files,
    packs: catalog.packs.map(pack => ({ ...pack, fileIds: pack.artifact_ids, ready: false, verified: false, runtimeReady: true, downloadable: true, installedCount: 0, missingCount: pack.artifact_ids.length, totalBytes: 100000, requiredBytes: 100000,
      blockedReason: pack.id.includes('turbo') ? 'ComfyUI-MiniMax-H3-Turbo 노드를 먼저 설치하세요. 그리드는 설치된 노드 폴더에만 저장됩니다.' : '' })),
    selectedPackIds: [], message: '오프라인 테스트 데이터 · 실제 파일을 읽지 않습니다', ...extra }
}
backend.getAllWidgetValues = callback => callback('{}')
backend.getUiPrefs = callback => callback(JSON.stringify(prefs))
backend.getAiAssistInstructions = callback => callback(JSON.stringify({ ok: true,
  instructions: prefs.aiAssistInstructions || { common: '', features: Object.fromEntries(
    ['expand', 'suggest', 'nl2tags', 'nl_caption', 'nl_scene', 'translate', 'creative', 'negative', 'auto_nl'].map(id => [id, ''])) } }))
backend.saveAiAssistInstructions = (raw, callback) => {
  prefs = { ...prefs, aiAssistInstructions: JSON.parse(raw) }
  actions.value = [...actions.value.slice(-7), 'AI 어시스트 지침 저장 완료 (오프라인 메모리)']
  callback(JSON.stringify({ ok: true, instructions: prefs.aiAssistInstructions }))
}
backend.getTabDefaults = callback => callback('{}')
backend.getInitialConfig = callback => callback(JSON.stringify({ uiPrefs: prefs }))
backend.onWidgetChanged = () => {}
backend.requestOllamaModels = () => backend.ollamaModelsReady.emit(JSON.stringify(modelNames))
backend.copyTextToClipboard = (value, callback) => {
  actions.value = [...actions.value.slice(-7), `clipboard: ${value}`]
  callback(true)
}
// Inline SVG is fixture-only preview data; metadata is separately mocked below.
const fixtureImage = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#888"/><text x="60" y="256" fill="white" font-size="30">ComfyUI metadata fixture</text></svg>')
backend.getImageExif = (_path, callback) => callback(JSON.stringify({ source: 'comfyui', path: fixtureImage,
  raw: '{"1":{"class_type":"CLIPTextEncode","inputs":{"text":"cherry_blossoms, outdoors"}}}',
  raw_prompt: '{"1":{"class_type":"CLIPTextEncode","inputs":{"text":"cherry_blossoms, outdoors"}}}',
  prompt: 'cherry_blossoms, outdoors', negative: 'blurry', params_line: 'Steps: 20, Sampler: euler, CFG scale: 5, Seed: 123',
  can_apply: true, metadata_warnings: [], prompt_candidates: [] }))
backend.onAction = (name, raw) => {
  const payload = JSON.parse(raw || '{}')
  actions.value = [...actions.value.slice(-7), `${name}: ${JSON.stringify(payload).slice(0, 200)}`]
  if (name === 'save_ui_prefs') { prefs = { ...prefs, ...payload }; backend.uiPrefsLoaded.emit(JSON.stringify(prefs)) }
  if (name.startsWith('model_download_')) {
    const started = name === 'model_download_start' || name === 'model_download_verify'
    backend.modelDownloadEvent.emit(JSON.stringify(modelState(started ? { state: 'verifying', busy: true, jobId: 'offline-job', percent: 30, downloadedBytes: 300, totalBytes: 1000, message: '오프라인 진행 상태 검증' } : {})))
  }
  if (name.startsWith('creator_h3_cache_')) {
    const clear = name.endsWith('clear'), removed = clear ? cacheEntries : 0
    if (clear) cacheEntries = 0
    backend.creatorCacheEvent.emit(JSON.stringify({ operation: clear ? 'clear' : 'status', requestId: payload.requestId, ok: true, available: true, entries: cacheEntries, bytes: cacheEntries * 1048576, removedEntries: removed }))
  }
  if (name === 'chat_load') backend.chatThreads.emit(JSON.stringify(threads))
  if (name === 'chat_model_info') backend.chatModelInfo.emit(JSON.stringify({ id: payload.id, model: payload.model, ok: true,
    info: { architecture: payload.model.includes('gpt-oss') ? 'gptoss' : 'offline', moe: payload.model.includes('gpt-oss') ? true : null,
      experts: payload.model.includes('gpt-oss') ? 32 : null, activeExperts: payload.model.includes('gpt-oss') ? 4 : null,
      thinkingMode: payload.model.includes('gpt-oss') ? 'levels' : 'boolean', vision: true } }))
  if (name === 'open_png_info_file') backend.pngInfoImageLoaded.emit(fixtureImage)
  if (name === 'get_xyz_capabilities') backend.xyzCapabilitiesReceived.emit(JSON.stringify({ requestId: payload.requestId, ok: true,
    backend: 'comfyui', capabilityId: 'offline-capability', axes: [
      { id: 'steps', label: 'Steps', type: 'integer', min: 1, max: 10000 },
      { id: 'cfg_scale', label: 'CFG Scale', type: 'number', min: 0, max: 100 },
      { id: 'sampler_name', label: 'Sampler', type: 'choice', choices: ['euler', 'dpmpp_2m'] },
    ], notes: ['오프라인 ComfyUI 응답 · 실제 백엔드 요청 없음'], unsupported: ['검증용 미지원 확장 축'] }))
  if (name === 'start_xyz_plot') backend.xyzPlotEvent.emit(JSON.stringify({ requestId: payload.requestId, ok: true, type: 'queued' }))
  if (name === 'chat_save') threads = payload.threads || []
  if (name === 'chat_send') {
    activeRequest = { id: payload.id, kind: payload.generation?.mode === 'video' ? 'video' : 'image' }
    backend.chatGenerationEvent.emit(JSON.stringify({ ...activeRequest, phase: 'generating', progress: 42, model: '오프라인 검증', message: '실제 생성 없이 진행 상태만 검증 중' }))
  }
  if (name === 'chat_stop' && activeRequest) {
    backend.chatGenerationEvent.emit(JSON.stringify({ ...activeRequest, phase: 'stopped', done: true, ok: false, stopped: true, artifacts: [] }))
    activeRequest = null
  }
}
window.qt = { webChannelTransport: {} }
window.QWebChannel = function (_transport, callback) { callback({ objects: { backend } }) }
await initBridge()
applyTheme('default')
const app = createApp({
  components: { SettingsView, ChatView, PngInfoView, XYZPlotView },
  setup() {
    const page = ref('settings')
    function complete() {
      if (!activeRequest) return
      backend.chatGenerationEvent.emit(JSON.stringify({ ...activeRequest, phase: 'error', done: true, ok: false, error: '오프라인 실패 피드백 검증', artifacts: [] }))
      activeRequest = null
    }
    const button = (label, onClick) => h('button', { onClick }, label)
    function motionFeedback(event, phase) {
      const element = event.currentTarget
      const report = () => {
        const icons = element.querySelectorAll('.icon, .icon-part')
        const detail = [...icons].map(icon => ({ animation: getComputedStyle(icon).animationName, transform: getComputedStyle(icon).transform }))
        actions.value = [...actions.value.slice(-7), `${element.title} ${phase}: ${JSON.stringify(detail)}`]
      }
      if (phase === 'leave') setTimeout(report, 250)
      else report()
    }
    const theme = preset => setTheme({ preset }, patch => { prefs = { ...prefs, ...patch } })
    return () => h('div', { class: 'offline-fixture' }, [
      h('nav', [h('b', '오프라인 검증 · 다운로드/GPU 실행 없음'),
        button('Settings 화면', () => { page.value = 'settings' }),
        button('Chat 화면', () => { page.value = 'chat' }),
        button('PNG Info 화면', () => { page.value = 'png' }),
        button('XYZ 화면', () => { page.value = 'xyz' }),
        button('모션 검증', () => { page.value = 'motion' }),
        button('검증 다크', () => theme('default')),
        button('검증 라이트', () => theme('light')),
        button('실패 응답 모의', complete)]),
      h('main', [page.value === 'motion' ? h('section', { class: 'motion-fixture' }, [
        ...['none', 'gpt', 'claude'].map(mode => button(`모션 ${mode}`, () => { document.documentElement.dataset.iconAnimation = mode })),
        h('div', { class: 'motion-examples' }, ['search', 'settings', 'refresh', 'trash', 'clipboard', 'bell', 'download', 'sparkles'].map(name => h('button', { 'aria-label': `모션 ${name}`, title: name,
          onPointerdown: event => motionFeedback(event, 'pressed'), onMouseleave: event => motionFeedback(event, 'leave'),
        }, [h(Icon, { name, size: 20 }), name]))),
      ]) : h(KeepAlive, () => h(({ settings: SettingsView, chat: ChatView, png: PngInfoView, xyz: XYZPlotView })[page.value]))]),
      h('details', [h('summary', '오프라인 액션 기록'), h('pre', actions.value.join('\n'))]),
    ])
  },
})
app.component('Icon', Icon).mount('#app')
const style = document.createElement('style')
style.textContent = '.offline-fixture{height:100vh;display:flex;flex-direction:column}.offline-fixture>nav{display:flex;flex-wrap:wrap;gap:8px;padding:10px;background:var(--bg-card);align-items:center}.offline-fixture>nav button{padding:8px;border:1px solid var(--border);color:var(--text-primary);background:var(--bg-button)}.offline-fixture>main{flex:1;min-height:0;overflow:hidden}.offline-fixture>details{padding:6px;font-size:11px;max-height:160px;overflow:auto}.offline-fixture pre{white-space:pre-wrap;overflow-wrap:anywhere}.motion-fixture{padding:24px}.motion-fixture button{padding:14px;margin:8px;background:var(--bg-card);color:var(--text-primary);border:1px solid var(--border);border-radius:10px}.motion-examples{display:flex;flex-wrap:wrap}.motion-examples button{display:flex;align-items:center;gap:12px;min-width:130px}'
document.head.appendChild(style)
