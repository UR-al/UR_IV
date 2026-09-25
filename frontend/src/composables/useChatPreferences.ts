import { computed, ref, watch } from 'vue'
import { getBackend, onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import { DEFAULT_OLLAMA_URL, resolveInstalledModel } from '../utils/ollamaPrefs'
import { CHAT_SYSTEM_PRESETS, thinkingValue, type ChatModelInfo } from '../utils/chatSettings'
import type { GenerationRequest } from '../utils/chatGeneration'
import { PROMPT_JSON_SCHEMA, parseChatSchema } from '../utils/chatStructuredOutput'
import { createSchemaAutosave } from './useSchemaAutosave'
import { createRefCountedSingleton } from '../utils/refCountedSingleton'

/**
 * 대화 설정 — 서버(Ollama/LM Studio) · 모델 · 지침 · 추론 · 생성 옵션 · JSON 스키마.
 *
 * views/ChatView.vue 에서 **위치만 옮겼다**. 대화 탭과 우하단 도크의 작은 대화 패널
 * (components/dock/ChatMiniPanel.vue)이 같은 설정으로 보내야 해서다 — 패널이 먼저 열려도
 * (대화 탭을 한 번도 안 열어도) 같은 값을 읽는다. 상태는 쓰는 곳이 있는 동안만 산다
 * (utils/refCountedSingleton) — 대화 탭은 keep-alive 라 한 번 열면 계속 산다.
 * 설정 화면(지침 프리셋 선택 · 스키마 예제 확인 같은 표시 상태)은 ChatView 에 남았다.
 */

export interface ChatOptions { temperature: number; numPredict: number; numCtx: number }
export const DEFAULT_CHAT_OPTIONS: ChatOptions = { temperature: 0.7, numPredict: -1, numCtx: 8192 }
export const PREDICT_CHOICES = [512, 1024, 2048, 4096, 8192, 16384]
export const CTX_CHOICES = [4096, 8192, 16384, 32768, 65536, 131072]
const DEFAULT_SYSTEM = CHAT_SYSTEM_PRESETS[0].prompt
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8)

function loadChatOptions(): ChatOptions {
  const o = { ...DEFAULT_CHAT_OPTIONS }
  try {
    const raw = JSON.parse(localStorage.getItem('chatOptions.v1') || 'null')
    if (raw && typeof raw === 'object') {
      // 목록에 없는 값(구버전·손으로 고친 것)은 select 가 빈칸으로 보인다 — 기본값으로 되돌린다
      if (typeof raw.temperature === 'number' && raw.temperature >= 0 && raw.temperature <= 2) o.temperature = raw.temperature
      if (raw.numPredict === -1 || PREDICT_CHOICES.includes(raw.numPredict)) o.numPredict = raw.numPredict
      if (raw.numCtx === 0 || CTX_CHOICES.includes(raw.numCtx)) o.numCtx = raw.numCtx
    }
  } catch {}
  return o
}

function createChatPreferences() {
  const generationRequest = ref<GenerationRequest>({ mode: 'auto', family: 'current', duration: 5, denoise: 0.65 })
  const models = ref<string[]>([])
  const provider = ref<'ollama' | 'lmstudio'>(localStorage.getItem('chatProvider') === 'lmstudio' ? 'lmstudio' : 'ollama')
  const model = ref(localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel') || '')
  const url = ref(localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl') || (provider.value === 'lmstudio' ? 'http://localhost:1234' : DEFAULT_OLLAMA_URL))
  const modelsError = ref('')
  const modelsLoading = ref(false)
  let modelsRequestId = ''
  let modelsTimer: ReturnType<typeof setTimeout> | undefined
  const structuredEnabled = ref(localStorage.getItem('chatStructuredEnabled') === '1')
  const schemaText = ref(localStorage.getItem('chatJsonSchema') ?? PROMPT_JSON_SCHEMA)
  const recoveringSchema = localStorage.getItem('chatJsonSchemaPending') === '1'
  const schemaAutosave = createSchemaAutosave()
  const schemaSaveError = schemaAutosave.error
  const schemaSaveStatus = computed(() => ({
    idle: '입력하면 자동 저장됩니다. 작성 중인 JSON도 보관합니다.',
    pending: '입력 내용 저장 대기 중…', saving: '입력 내용 저장 중…',
    saved: '입력 내용 자동 저장됨', error: '자동 저장하지 못했습니다. 입력 내용은 유지됩니다.',
  }[schemaAutosave.state.value]))
  let restoringSchema = false
  watch(schemaText, text => {
    if (restoringSchema) return
    markPreferencesEdited()
    // Keep an unacknowledged draft across a restart or a disconnected bridge.
    try { localStorage.setItem('chatJsonSchema', text); localStorage.setItem('chatJsonSchemaPending', '1') } catch { /* file save still runs */ }
    schemaAutosave.queue(text)
  }, { flush: 'sync' })
  watch(schemaAutosave.state, state => {
    if (state === 'saved') {
      try { localStorage.setItem('chatJsonSchemaPending', '0') } catch { /* saved on disk */ }
    }
  }, { flush: 'sync' })
  const schemaError = computed(() => { try { parseChatSchema(schemaText.value); return '' } catch (error) { return (error as Error).message } })
  let preferencesEdited = false
  function markPreferencesEdited() { preferencesEdited = true }
  let disposed = false
  const systemPrompt = ref(localStorage.getItem('chatSystemPrompt') ?? DEFAULT_SYSTEM)
  const personalSystemPrompt = ref<string | null>(localStorage.getItem('chatPersonalSystemPrompt'))
  const modelInfo = ref<ChatModelInfo | null>(null)
  const modelInfoError = ref('')
  const modelInfoLoading = ref(false)
  let modelInfoRequestId = ''
  let modelInfoTimer: ReturnType<typeof setTimeout> | null = null
  const thinkingLevel = ref(['low', 'medium', 'high'].includes(localStorage.getItem('chatThinkingLevel') || '') ? localStorage.getItem('chatThinkingLevel')! : 'medium')
  function saveThinkingLevel() { localStorage.setItem('chatThinkingLevel', thinkingLevel.value) }
  // 생성 옵션 — Ollama 기본 문맥 창(4096)은 지침 + 이미지 + 24턴 기록이면 금방 찬다. 답이 잘리면 여기서 늘린다.
  const chatOptions = ref<ChatOptions>(loadChatOptions())
  function saveChatOptions() { localStorage.setItem('chatOptions.v1', JSON.stringify(chatOptions.value)); savePreferences() }
  function ollamaOptions(): Record<string, number> {
    const o: Record<string, number> = { temperature: chatOptions.value.temperature }
    // '제한 없음' 도 -1 로 명시한다 — 모델파일이 num_predict 를 박아 둔 모델이 있다
    o.num_predict = chatOptions.value.numPredict > 0 ? chatOptions.value.numPredict : -1
    if (provider.value === 'ollama' && chatOptions.value.numCtx > 0) o.num_ctx = chatOptions.value.numCtx
    return o
  }
  // 깊은 추론 — Gemma 4 / Qwen3.x 같은 thinking 모델은 기본으로 생각부터 하느라 첫 글자가 1분 뒤에 온다.
  // GemmaStudio 처럼 빠른 답변이 기본, 원할 때만 켠다. 켜면 생각이 접힌 블록으로 같이 흐른다.
  const deepThink = ref(localStorage.getItem('chatThink') === '1')
  function toggleThink() { deepThink.value = !deepThink.value; localStorage.setItem('chatThink', deepThink.value ? '1' : '0') }

  // ── 모델 · 설정 ──
  function saveModel() {
    localStorage.setItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel', model.value)
    if (provider.value === 'ollama') requestAction('save_ui_prefs', { ollamaModel: model.value, ollamaUrl: url.value })
    savePreferences()
  }
  function saveSystemPrompt() {
    localStorage.setItem('chatSystemPrompt', systemPrompt.value)
    if (!CHAT_SYSTEM_PRESETS.some(preset => preset.prompt === systemPrompt.value)) {
      personalSystemPrompt.value = systemPrompt.value
      localStorage.setItem('chatPersonalSystemPrompt', systemPrompt.value)
    }
    savePreferences()
  }
  function usesStructuredOutput(request: GenerationRequest) {
    return structuredEnabled.value && request.mode !== 'image' && request.mode !== 'video'
  }
  /** 보내면 안 되는 스키마 초안이면 그 오류 문구, 괜찮으면 ''. 화면 처리(설정 열기 · 토스트)는 부르는 쪽. */
  function schemaProblem(request: GenerationRequest = generationRequest.value) {
    return usesStructuredOutput(request) ? schemaError.value : ''
  }
  function saveStructured() {
    savePreferences()
  }
  function savePreferences() {
    preferencesEdited = true
    localStorage.setItem('chatProvider', provider.value)
    localStorage.setItem('chatStructuredEnabled', structuredEnabled.value ? '1' : '0')
    localStorage.setItem('chatJsonSchema', schemaText.value)
    requestAction('save_ui_prefs', { chatSettingsV2: {
      provider: provider.value, lmStudioUrl: localStorage.getItem('chatLmStudioUrl') || 'http://localhost:1234',
      lmStudioModel: localStorage.getItem('chatLmStudioModel') || '',
      structuredEnabled: structuredEnabled.value, schemaText: schemaText.value,
      systemPrompt: systemPrompt.value, personalSystemPrompt: personalSystemPrompt.value,
      options: chatOptions.value,
    } })
  }
  async function restorePreferences() {
    const backend = await getBackend()
    if (disposed || preferencesEdited || !backend?.getUiPrefs) return
    backend.getUiPrefs((raw: string) => {
      if (disposed || preferencesEdited) return
      try {
        const prefs = JSON.parse(raw), saved = prefs.chatSettingsV2
        if (!saved || typeof saved !== 'object') return
        if (typeof saved.lmStudioUrl === 'string') localStorage.setItem('chatLmStudioUrl', saved.lmStudioUrl)
        if (typeof saved.lmStudioModel === 'string') localStorage.setItem('chatLmStudioModel', saved.lmStudioModel)
        provider.value = saved.provider === 'lmstudio' ? 'lmstudio' : 'ollama'
        localStorage.setItem('chatProvider', provider.value)
        model.value = provider.value === 'lmstudio' ? saved.lmStudioModel || '' : prefs.ollamaModel || model.value
        url.value = provider.value === 'lmstudio' ? saved.lmStudioUrl || 'http://localhost:1234' : prefs.ollamaUrl || url.value
        if (provider.value === 'ollama') {
          localStorage.setItem('ollamaUrl', url.value)
          localStorage.setItem('ollamaModel', model.value)
        }
        if (typeof saved.systemPrompt === 'string') systemPrompt.value = saved.systemPrompt
        if (typeof saved.personalSystemPrompt === 'string') personalSystemPrompt.value = saved.personalSystemPrompt
        if (typeof saved.schemaText === 'string' && !recoveringSchema && localStorage.getItem('chatJsonSchemaPending') !== '1') {
          restoringSchema = true
          schemaText.value = saved.schemaText
          restoringSchema = false
          localStorage.setItem('chatJsonSchema', saved.schemaText)
        }
        if (saved.options && typeof saved.options === 'object') {
          localStorage.setItem('chatOptions.v1', JSON.stringify(saved.options))
          chatOptions.value = loadChatOptions()
        }
        structuredEnabled.value = saved.structuredEnabled === true
        requestModels()
      } catch { /* do not replace local edits on invalid saved settings */ }
    })
  }
  function changeProvider() {
    models.value = []; modelsError.value = ''
    model.value = localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioModel' : 'ollamaModel') || ''
    url.value = localStorage.getItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl') || (provider.value === 'lmstudio' ? 'http://localhost:1234' : DEFAULT_OLLAMA_URL)
    savePreferences(); requestModels()
  }
  function saveConnection() {
    localStorage.setItem(provider.value === 'lmstudio' ? 'chatLmStudioUrl' : 'ollamaUrl', url.value)
    models.value = []; saveModel(); requestModels()
  }
  function requestModelInfo() {
    modelInfo.value = null
    modelInfoError.value = ''
    if (modelInfoTimer) clearTimeout(modelInfoTimer)
    modelInfoRequestId = uid()
    modelInfoLoading.value = !!model.value
    if (!model.value) return
    const id = modelInfoRequestId
    requestAction('chat_model_info', { id, url: url.value, model: model.value, provider: provider.value })
    modelInfoTimer = setTimeout(() => {
      if (modelInfoRequestId === id && modelInfoLoading.value) {
        modelInfoLoading.value = false
        modelInfoError.value = '모델 정보를 받지 못했습니다. 연결 후 다시 확인하세요'
      }
    }, 15000)
  }
  function onModelInfo(raw: string) {
    try {
      const event = JSON.parse(raw)
      if (event.id !== modelInfoRequestId || event.model !== model.value) return
      if (modelInfoTimer) clearTimeout(modelInfoTimer)
      modelInfoLoading.value = false
      if (event.ok && event.info) {
        modelInfo.value = event.info
        modelInfoError.value = ''
      }
      else modelInfoError.value = event.error || '모델 정보를 확인할 수 없습니다'
    } catch { /* stale/malformed metadata never changes the selected model */ }
  }
  watch([model, url, provider], requestModelInfo)
  async function requestModels() {
    modelsRequestId = uid()
    const id = modelsRequestId
    clearTimeout(modelsTimer)
    modelsLoading.value = true; modelsError.value = ''
    modelsTimer = setTimeout(() => {
      if (id !== modelsRequestId) return
      modelsLoading.value = false; modelsError.value = '모델 목록 응답이 없습니다. 서버 실행과 주소를 확인하세요.'
    }, 12000)
    if (provider.value === 'lmstudio') {
      requestAction('chat_models', { id, url: url.value })
      return
    }
    try {
      const bk: any = await getBackend()
      if (disposed || id !== modelsRequestId || provider.value !== 'ollama') return
      url.value = localStorage.getItem('ollamaUrl') || url.value
      if (bk?.requestOllamaModels) bk.requestOllamaModels(url.value)
    } catch {}
  }
  function onModels(json: string) {
    if (provider.value !== 'ollama') return
    try {
      const p = JSON.parse(json)
      const list = Array.isArray(p) ? p : p.models
      if (!Array.isArray(list)) return
      clearTimeout(modelsTimer); modelsLoading.value = false
      models.value = list
      // 백엔드 resolve_model 과 같은 규칙(utils/ollamaPrefs) — 같은 모델 → 같은 계열 태그 → 첫 모델
      if (list.length && !list.includes(model.value)) model.value = resolveInstalledModel(model.value, list)
    } catch {}
  }
  function onChatModels(raw: string) {
    try {
      const event = JSON.parse(raw)
      if (provider.value !== 'lmstudio' || event.id !== modelsRequestId) return
      clearTimeout(modelsTimer); modelsLoading.value = false
      if (!event.ok) { models.value = []; modelsError.value = event.error || '모델 목록을 받지 못했습니다'; return }
      if (!Array.isArray(event.models)) return
      models.value = event.models.filter((item: unknown) => typeof item === 'string')
      if (!models.value.includes(model.value)) {
        model.value = models.value[0] || ''
        localStorage.setItem('chatLmStudioModel', model.value)
      }
      if (!models.value.length) modelsError.value = '서버에 모델이 없습니다. LM Studio에서 모델을 준비하세요.'
    } catch { /* obsolete/malformed model lists do not alter the selection */ }
  }

  /** chat_send 에 실을 설정 부분 — 대화 목록(useChatSession)이 요청마다 읽는다. */
  function requestSettings(request: GenerationRequest) {
    return {
      provider: provider.value,
      schema: usesStructuredOutput(request) ? parseChatSchema(schemaText.value) : undefined,
      url: url.value,
      model: model.value,
      system: systemPrompt.value,
      think: thinkingValue(modelInfo.value, deepThink.value, thinkingLevel.value),
      options: ollamaOptions(),
    }
  }

  // ── 시작 — 예전 ChatView onMounted 의 설정 부분 ──
  const unsubs: Array<() => void> = []
  unsubs.push(onBackendEvent('ollamaModelsReady', onModels))
  unsubs.push(onBackendEvent('chatModelInfo', onModelInfo))
  unsubs.push(onBackendEvent('chatModelsReady', onChatModels))
  window.addEventListener('beforeunload', schemaAutosave.flush)
  if (recoveringSchema) schemaAutosave.queue(schemaText.value)
  requestModels()
  restorePreferences().catch(() => {})
  requestModelInfo()

  function dispose() {
    disposed = true; clearTimeout(modelsTimer)
    unsubs.forEach((u) => { try { u() } catch {} })
    window.removeEventListener('beforeunload', schemaAutosave.flush)
    schemaAutosave.close()
    if (modelInfoTimer) { clearTimeout(modelInfoTimer); modelInfoTimer = null }
  }

  return {
    api: {
      generationRequest, models, provider, model, url, modelsError, modelsLoading,
      structuredEnabled, schemaText, schemaAutosave, schemaSaveError, schemaSaveStatus, schemaError,
      systemPrompt, personalSystemPrompt, modelInfo, modelInfoError, modelInfoLoading,
      thinkingLevel, chatOptions, deepThink,
      markPreferencesEdited, saveThinkingLevel, saveChatOptions, toggleThink, saveModel, saveSystemPrompt,
      usesStructuredOutput, schemaProblem, saveStructured, savePreferences, changeProvider, saveConnection,
      requestModelInfo, requestModels, requestSettings,
    },
    dispose,
  }
}

export type ChatPreferences = ReturnType<typeof createChatPreferences>['api']

/** 대화 설정 — 쓰는 컴포넌트(대화 탭 · 도크 대화 패널)가 있는 동안 하나를 나눠 쓴다. */
export const useChatPreferences = createRefCountedSingleton(createChatPreferences)
