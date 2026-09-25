/**
 * 브리지 계약 타입 (Vue ↔ PyQt) — App.vue 분할/TypeScript 기반(②).
 *
 * 무엇이 어디서 강제되는가(과장 없이):
 *  - 이름: ActionName/BackendEvent 는 Python handler/signal 과 함께 갱신한다.
 *    tests/test_bridge_contract.py 가 양방향(프론트 ↔ Python) 드리프트를 잡는다.
 *  - 액션 페이로드: 아래 ActionPayloads 에 적힌 액션만, TypeScript 호출부(.ts · lang="ts")에서
 *    requestAction/action 래퍼를 거칠 때 `npm run type-check`(vue-tsc)가 모양을 검사한다.
 *    적히지 않은 액션은 `object` 로 느슨하다. `.js` 호출부(checkJs:false)는 검사되지 않는다.
 *  - 이벤트 페이로드: 시그널은 JSON 문자열이라 런타임 검증이 없다. 수신부가
 *    `JSON.parse(json) as XxxEvent` 로 이 파일의 타입에 연결해야 필드 오타가 잡힌다.
 *  Python 쪽 페이로드 키와의 일치는 사람이 맞춘다(자동 검사 없음).
 */

/** Vue가 requestAction()/action()으로 호출하는 백엔드 액션 이름. */
export type ActionName =
  | 'generate' | 'cancel_generation' | 'random_prompt' | 'swap_resolution'
  | 'set_high_res_factor' | 'set_random_resolutions' | 'set_rating_filter' | 'update_prompt_deck'
  | 'set_lora_stack' | 'set_artist_locked'
  | 'save_settings' | 'save_preset_by_name'
  | 'load_preset_by_name' | 'delete_preset'
  | 'settings_export' | 'settings_import' | 'restart_app'
  | 'presets_export' | 'presets_import' | 'character_presets_export' | 'character_presets_import'
  | 'save_ui_prefs' | 'save_global_weights' | 'save_cond_rules' | 'save_tab_defaults' | 'set_tab_order'
  | 'send_to_i2i' | 'send_to_inpaint' | 'send_to_editor' | 'send_to_compare'
  | 'generate_i2i' | 'cancel_i2i' | 'generate_inpaint' | 'start_batch' | 'start_upscale' | 'start_xyz_plot'
  | 'run_adetailer_single' | 'run_adetailer_batch' | 'stop_adetailer_batch'
  | 'run_sam3_single' | 'run_sam3_batch' | 'run_refine' | 'open_ad_files' | 'open_ad_folder'
  | 'open_batch_files' | 'open_upscale_files'
  | 'caption_pick_files' | 'caption_pick_folder' | 'caption_pick_outdir' | 'caption_pick_caformer_dir'
  | 'apply_search_result' | 'add_search_to_queue' | 'export_search_results' | 'import_search_results'
  | 'reset_prompt_deck'
  | 'search_events' | 'event_add_to_queue' | 'event_generate_now'
  | 'export_event_results' | 'import_event_results'
  | 'start_queue' | 'stop_queue' | 'pause_queue' | 'resume_queue' | 'clear_queue'
  | 'remove_queue_items' | 'move_queue_item' | 'update_queue_item' | 'add_image_to_queue'
  | 'sync_queue_state'
  | 'editor_open_file' | 'editor_save' | 'editor_save_as' | 'editor_add_yolo_model'
  | 'editor_clear_yolo_models' | 'editor_load_watermark_image'
  | 'open_png_info_file' | 'open_compare_image' | 'pnginfo_send_prompt' | 'pnginfo_generate'
  | 'pnginfo_transplant_meta'
  | 'pull_prompt_from_image' | 'explore_seed' | 'copy_to_clipboard'
  | 'gallery_open_folder' | 'gallery_send_exif_to_t2i' | 'add_favorite' | 'remove_favorite' | 'delete_image'
  | 'toggle_automation' | 'stop_automation' | 'set_automation_settings'
  | 'automation_override_next' | 'pause_automation' | 'resume_automation'
  | 'workflow_profile_list' | 'workflow_profile_save' | 'workflow_profile_load'
  | 'workflow_profile_delete' | 'workflow_profile_rename'
  | 'prompt_order_list' | 'prompt_order_save' | 'prompt_order_reset'
  | 'instant_wildcards_list' | 'instant_wildcards_save' | 'instant_wildcards_delete'
  | 'show_prompt_history' | 'show_api_manager' | 'open_url'
  | 'import_anima_from_forge' | 'reset_anima_guidance' | 'unload_model_request' | 'show_toast'
  | 'chat_send' | 'chat_stop' | 'chat_load' | 'chat_save' | 'chat_export' | 'chat_model_info'
  | 'chat_models'
  | 'get_xyz_capabilities'
  | 'comfy_compatibility_refresh' | 'comfy_compatibility_save_baseline'
  | 'comfy_controls_inspect' | 'comfy_controls_save' | 'comfy_controls_clear'
  | 'comfy_quality_preset' | 'comfy_feature_preflight'
  | 'relight_preview' | 'relight_export' | 'relight_cancel'
  | 'hand_reconstruction_generate' | 'hand_reconstruction_export' | 'hand_reconstruction_cancel'
  | 'native_tab_switch' | 'vue_tab_switch'
  | 'probe_backend' | 'select_backend' | 'pick_comfy_workflow'
  | 'creator_get_state' | 'creator_select_media' | 'creator_generate' | 'creator_cancel'
  | 'creator_h3_cache_status' | 'creator_h3_cache_clear'
  | 'model_download_status' | 'model_download_start' | 'model_download_cancel' | 'model_download_verify'
  | 'comic_plan' | 'comic_generate_all' | 'comic_animate_all'
  | 'comic_export_page' | 'comic_export_living' | 'comic_save'

/** Python이 vue_bridge에서 emit하고 Vue가 onBackendEvent()로 받는 시그널 이름. */
export type BackendEvent =
  | 'imageGenerated' | 'generationStarted' | 'generationError' | 'generationProgress' | 'generationPreview'
  | 'automationStatus' | 'automationSettingsLoaded'
  | 'searchResultsReady' | 'searchResultLineage' | 'searchStatus' | 'queueUpdated' | 'queueItemAdded' | 'queueCompleted'
  | 'uiPrefsLoaded' | 'condRulesLoaded' | 'loraStackLoaded' | 'globalWeightsLoaded'
  | 'promptOrderLoaded' | 'instantWildcardsList' | 'workflowProfilesList'
  | 'editorImageLoaded' | 'editorWatermarkImageLoaded' | 'editorResult' | 'editorSaveResult' | 'editorAutoSaveReady' | 'yoloModelUpdated' | 'i2iImageLoaded' | 'i2iJobState' | 'inpaintImageLoaded'
  | 'pngInfoImageLoaded'
  | 'compareImageLoaded' | 'galleryFolderLoaded' | 'galleryImagesReady' | 'thumbnailReady'
  | 'imageSearchTextsReady'
  | 'imageDeleteResult'
  | 'upscalersReady' | 'ollamaModelsReady' | 'adetailerModelsReady'
  | 'chatToken' | 'chatDone' | 'chatThreads'
  | 'chatGenerationEvent' | 'modelDownloadEvent' | 'creatorCacheEvent' | 'chatModelInfo'
  | 'chatModelsReady' | 'aiAssistInstructionsChanged' | 'instructionPresetsChanged'
  | 'xyzCapabilitiesReceived' | 'xyzPlotEvent'
  | 'comfyWorkflowEvent' | 'comfyCompatibilityResult' | 'relightEvent' | 'handReconstructionEvent'
  | 'batchFilesSelected' | 'batchJobState' | 'adetailerResult' | 'adetailerProgress' | 'sam3Result' | 'sam3Progress'
  | 'captionFilesSelected' | 'captionProgress' | 'captionDone' | 'captionOutDirSelected'
  | 'captionModelDirSelected' | 'captionRuntimeReady'
  | 'eventSearchProgress' | 'eventSearchResults' | 'eventImportResults' | 'eventLoadStatus'
  | 'ollamaResult' | 'genNlResult' | 'vramUpdated' | 'showNotification' | 'tabChanged'
  | 'backendSelectionRequired' | 'backendProbeResult' | 'backendSelected' | 'comfyWorkflowPicked'
  | 'backendStatus'
  | 'creatorStateChanged' | 'creatorProgress' | 'creatorResult' | 'creatorMediaSelected'
  | 'comicStoryboardReady' | 'comicDocumentChanged'
  | 'loraManagerUrlReady' | 'refineResult'
  | 'statusMessage'
  | 'lorasReady' | 'characterTagsOnlineReady' | 'compareGifReady'

/** statusMessage — 파이썬 show_status 한 줄(core/status_message.py). 하단 계기 스트립이 보인다.
 *  timeoutMs 0 = 다음 문구까지 유지. at = 보낸 시각(epoch ms, getStatusMessage 재생용). */
export interface StatusMessagePayload {
  text: string
  level: 'info' | 'success' | 'warning' | 'error'
  timeoutMs: number
  at: number
}

// ── 비동기 조회(request* 슬롯 → *Ready 이벤트) ──
// GUI 스레드를 막던 동기 슬롯의 짝. 요청 때 보낸 requestId 를 그대로 되돌려 준다
// (utils/bridgeRequest.ts 가 짝을 맞추고, 옛 요청·다른 웹 클라이언트의 응답은 버린다).

/** lorasReady — requestLoras(mode, requestId). loras 는 getLoras 와 같은 병합 카탈로그 */
export interface LorasReadyPayload {
  requestId: string
  mode: string
  loras?: Array<Record<string, unknown>>
  error?: string
}

/** characterTagsOnlineReady — requestCharacterTagsOnline(name, requestId). 이름도 되돌려 준다 */
export interface CharacterTagsOnlinePayload {
  requestId: string
  name: string
  tags?: string[]
  sampled?: number
  error?: string
}

/** compareGifReady — requestCompareGif(before, after, durationMs, loops, requestId) */
export interface CompareGifReadyPayload {
  requestId: string
  path?: string
  frames?: number
  error?: string
}

/** editorAutoSaveReady — requestEditorAutoSave(path, overlayBase64, opacity0to100, requestId).
 *  크래시 복구본을 워커에서 쓴다(core/editor_autosave.py). discarded = 쓰기 전·중에 복구본이
 *  폐기됐다(저장 성공 → editorClearAutoSave) — 남은 복구본이 없다. */
export interface EditorAutoSaveReadyPayload {
  requestId: string
  path?: string
  drawing?: boolean
  discarded?: boolean
  error?: string
}

/**
 * thumbnailReady — 청크 단위 통지(core/thumb_prefetch.py). thumb 는 file:/// URL, 실패면 ''.
 * v 는 썸네일 파일 버전(렌더 시각 ns, 문자열) — 썸네일이 있을 때만. URL 에 붙여 다시 만든 썸네일을 새로 읽는다.
 */
export interface ThumbnailReadyPayload {
  width: number
  items: Array<{ path: string; thumb: string; v?: string }>
}

// AI helper instructions are intentionally independent of Chat settings.
export type AiAssistFeature = 'expand' | 'suggest' | 'nl2tags' | 'nl_caption'
  | 'nl_scene' | 'translate' | 'creative' | 'negative' | 'auto_nl'

export interface AiAssistInstructions {
  common: string
  features: Record<AiAssistFeature, string>
}

export interface InstructionPreset {
  id: string
  name: string
  scope: 'chat' | 'assist' | 'schema'
  instructions: string | AiAssistInstructions
}

export type AiAssistInstructionsResult =
  | { ok: true; instructions: AiAssistInstructions }
  | { ok: false; error: string }

// ── Batch / Caption ──

export type CaptionEngineMode = 'caformer' | 'torii' | 'combined' | 'ollama'

export interface CaptionRuntimeSnapshot {
  clientToken: string
  requestId: number
  caformer?: { available?: boolean; modelDir?: string; error?: string }
  torii?: { available?: boolean; model?: string; error?: string }
  onnxruntime?: boolean
  error?: string
}

export interface CaptionProgressEvent {
  clientToken: string
  jobId: string
  index: number
  total: number
  path: string
  caption?: string
  txtPath?: string
  skipped?: boolean
  error?: string
}

export interface CaptionDoneEvent {
  clientToken: string
  jobId: string
  total: number
  ok: number
  failed: number
  skipped?: number
  error?: string
  status?: 'done'
}

export interface CaptionJobStatus {
  clientToken: string
  jobId: string
  status: 'idle' | 'running' | 'done'
  total?: number
  ok?: number
  failed?: number
  skipped?: number
  error?: string
  current?: number
  processed?: number
  succeeded?: number
  engine?: CaptionEngineMode
  items?: Array<CaptionProgressEvent | null>
}

export interface CaptionStartResponse {
  clientToken: string
  jobId: string
  started?: boolean
  total?: number
  error?: string
}

/** show_toast 페이로드 */
export interface ShowToastPayload { type: 'success' | 'error' | 'info' | 'warning'; msg: string }

/** batchJobState 이벤트 — Vue 일괄 처리·업스케일 진행 (core/batch_job_state.JobProgress) */
export interface BatchJobStatePayload {
  job: 'batch' | 'upscale'
  running: boolean
  total: number
  done: number
  success: number
  failed: number
  stopped: boolean
  output_dir: string
}

/** i2iJobState 이벤트 — Vue I2I 진행 (ui/i2i_actions.emit_job_state).
 *  running 이면 I2IView 가 시작 버튼을 막고 취소 버튼(cancel_i2i)을 보인다. cancelling 은 취소 요청 뒤. */
export interface I2IJobStatePayload {
  running: boolean
  cancelling: boolean
}

/** set_rating_filter 페이로드 — g/s/q/e 중 활성 등급 */
export interface SetRatingFilterPayload { ratings: string[] }

/** Search 결과 스냅숏 출처 (searchResultLineage 이벤트와 같은 모양) */
export interface SearchLineagePayload { label: string; fingerprint: string; snapshot_id: string }

/** update_prompt_deck 페이로드 (utils/searchRestore.ts buildSearchDeckUpdate · core/search_session.py).
 *  권장: 필터 결과를 base(검색 결과 전체) 배열 인덱스로 — 행 전체 직렬화 없음.
 *  하위 호환: 행 전체. lineage 가 현재 스냅숏과 다르거나 base_size 가 Python base 와
 *  다르면 Python 이 거부한다. */
export type UpdatePromptDeckPayload =
  | { indices: number[]; base_size: number; lineage: SearchLineagePayload }
  | { results: Array<Record<string, unknown>>; lineage: SearchLineagePayload }

/** import_search_results 페이로드 — 검색과 같은 결과 상한(50만, 무작위 표본). true 면 끈다. */
export interface ImportSearchResultsPayload { disable_result_cap?: boolean }

/** set_high_res_factor 페이로드 */
export interface SetHighResPayload { enabled: boolean; factor: number }

/** LoRA 스택 1개 항목. weight 단위는 경계마다 고정이다(core/lora_stack.py 문서):
 *  - 브리지(set_lora_stack / loraStackLoaded)·Python _vue_lora_entries·프로파일 v2 = 배율(0~3)
 *  - ui_prefs.loraStack·Vue loraStack(localStorage) = 정수 퍼센트(weight/100 = 배율)
 *  변환은 utils/loraUnits.ts 한 곳에서만 한다. 생성 LoRA 는 _vue_lora_entries 하나에서 파생된다. */
export interface LoraEntry {
  name: string
  weight: number
  enabled: boolean
  triggerWords: string[]
}

/** set_lora_stack 페이로드 */
export interface SetLoraStackPayload { entries: LoraEntry[] }

// ── 자동화 ──
// 화면은 '10장 / 1시간 / 무제한' 으로 말하지만 계약은 mode + limit 두 개뿐이다.
// 단위 환산(시간 → 분)은 전부 Vue 쪽에서 끝내고, 백엔드에는 예전 그대로 보낸다.

/** set_automation_settings 페이로드 — 자동화 루프 설정 한 벌.
 *  `limit` 의 단위는 mode 가 정한다: count=장 수, timer=분, unlimited=안 씀. */
export interface AutomationSettings {
  mode: 'count' | 'timer' | 'unlimited'
  limit: number
  /** 한 프롬프트로 이어서 만들 장 수 (시드만 바뀐다). */
  repeat: number
  /** 장과 장 사이에 쉬는 시간(초). */
  delay: number
  allowDupes: boolean
  autoResetDeck: boolean
  /** 생성 실패 1회당 다시 시도할 횟수. */
  maxRetries: number
  /** N장마다 백엔드의 LoRA 캐시를 비운다. 0 이면 안 비운다. */
  cleanupEveryN: number
}

/** automation_override_next 페이로드 — **다음 한 장에만** 쓸 프롬프트 전문.
 *  백엔드가 한 번 쓰고 스스로 지운다. 빈 문자열이면 덮어쓰기 취소(원래 프롬프트로). */
export interface AutomationOverrideNextPayload { prompt: string }

/** workflow_profile_save 페이로드 — 현재 설정을 프로파일로 저장.
 *  `overwrite` 는 사용자가 덮어쓰기를 확인했을 때만 true. 없거나 false 면 백엔드는 이름 규칙상 같은
 *  파일('Flux.' · 'flux' → Flux.json)이 이미 있을 때 쓰지 않고 경고한다(core/workflow_profiles). */
export interface WorkflowProfileSavePayload { name: string; overwrite?: boolean }

/** automationStatus 이벤트 — 자동화 루프가 매 단계 보내는 상태.
 *  `prompt`/`paused` 는 조종석(다음 프롬프트 편집 · 일시정지 버튼) 때문에 늘어난 필드다. */
export interface AutomationStatusEvent {
  running: boolean
  paused: boolean
  count: number
  waiting: boolean
  wait_remaining_ms: number
  wait_total_ms: number
  deck_total: number
  deck_remaining: number
  deck_used: number
  allow_duplicates: boolean
  /** 다음 생성에 나갈 프롬프트 전문. 없으면 빈 문자열. */
  prompt: string
  /** `prompt` 가 다음 덱 장에 그대로 쓰이는가. false 면 생성 중인(또는 방금 생성한) 프롬프트라
   *  편집해도 다음 장을 새로 뽑을 때 버려진다 — 조종석이 편집을 막는다(core/automation_prompt_state).
   *  옛 백엔드는 안 보낸다(없으면 true 로 본다). */
  prompt_is_next?: boolean
}

// ── 시작 백엔드 게이트 ──
// 창 위에 뜨는 Vue 오버레이. 예전의 별도 QDialog 를 대체하지만, Vue 가 못 뜨면
// 그릴 방법이 없으므로 QDialog 는 비상 경로로 남아 있다(Python 쪽 폴백).

/** probe_backend 페이로드 — 두 주소를 한 번에 감지 요청 */
export interface ProbeBackendPayload { webuiUrl: string; comfyUrl: string }

/** backendProbeResult 이벤트 — 각 백엔드의 응답 여부 */
export interface BackendProbeResult { webui: 'ok' | 'fail'; comfy: 'ok' | 'fail' }

/** backendSelectionRequired 이벤트 — 게이트를 열 때 쓸 현재 설정값 */
export interface BackendSelectionRequired {
  webuiUrl: string
  comfyUrl: string
  workflowPath: string
}

/** select_backend 페이로드 — url 은 고른 쪽 주소 하나만 보낸다 */
export interface SelectBackendPayload {
  type: 'webui' | 'comfyui'
  url: string
  workflowPath?: string
}

/** backendSelected 이벤트 — 실패해도 게이트는 떠 있어야 하므로 error 를 함께 싣는다 */
export interface BackendSelectedResult { ok: boolean; error?: string }

/** comfyWorkflowPicked 이벤트 — 경로 + analyze_workflow 요약 */
export interface ComfyWorkflowPicked {
  path: string
  info: {
    valid: boolean
    error?: string
    format?: string
    node_count?: number
    ksampler_type?: string
    width?: number
    height?: number
    classification?: string
    is_locked?: boolean
    /** 생성 컴파일러가 항상 거부하는 구조의 이유 (없으면 null) */
    generation_blocker?: string | null
    /** 앱 모델 선택이 워크플로 로더에 적용되는지 */
    model_selectable?: boolean
  }
}

// ── 액션 페이로드 맵 ──
// requestAction(widgetStore.js JSDoc)과 뷰의 action() 래퍼가 `<K extends ActionName>` 제네릭으로
// 이 맵을 따라간다 — 적힌 액션은 호출부 페이로드 모양이 type-check 로 검사된다.
// set_automation_settings 는 호출부가 AutomationSettings 밖의 필드도 함께 보내므로 넣지 않는다.

/** 액션 이름 → 페이로드 타입. 여기 없는 액션은 ActionPayload 가 `object` 로 푼다. */
export interface ActionPayloads {
  show_toast: ShowToastPayload
  set_rating_filter: SetRatingFilterPayload
  set_high_res_factor: SetHighResPayload
  set_lora_stack: SetLoraStackPayload
  update_prompt_deck: UpdatePromptDeckPayload
  import_search_results: ImportSearchResultsPayload
  automation_override_next: AutomationOverrideNextPayload
  workflow_profile_save: WorkflowProfileSavePayload
  probe_backend: ProbeBackendPayload
  select_backend: SelectBackendPayload
}

/** 액션 K 의 페이로드 타입(맵에 없으면 `object`). */
export type ActionPayload<K extends ActionName> = K extends keyof ActionPayloads ? ActionPayloads[K] : object
