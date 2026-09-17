/**
 * 브리지 계약 타입 (Vue ↔ PyQt) — App.vue 분할/TypeScript 기반(②).
 *
 * 액션/이벤트 이름과 페이로드의 강제 계약이다. 새 이름은 Python handler/signal과
 * 이 파일을 함께 갱신해야 하며 tests/test_bridge_contract.py가 양쪽 드리프트를 잡는다.
 */

/** Vue가 requestAction()/action()으로 호출하는 백엔드 액션 이름. */
export type ActionName =
  | 'generate' | 'cancel_generation' | 'random_prompt' | 'swap_resolution'
  | 'set_high_res_factor' | 'set_random_resolutions' | 'set_rating_filter' | 'update_prompt_deck'
  | 'set_lora_text' | 'set_lora_stack' | 'set_artist_locked'
  | 'save_settings' | 'save_preset' | 'load_preset' | 'save_preset_by_name'
  | 'load_preset_by_name' | 'delete_preset'
  | 'save_ui_prefs' | 'save_global_weights' | 'save_cond_rules' | 'save_tab_defaults' | 'set_tab_order'
  | 'send_to_i2i' | 'send_to_inpaint' | 'send_to_editor' | 'send_to_compare'
  | 'generate_i2i' | 'generate_inpaint' | 'start_batch' | 'start_upscale' | 'start_xyz_plot'
  | 'run_adetailer_single' | 'run_adetailer_batch' | 'stop_adetailer_batch'
  | 'run_sam3_single' | 'run_sam3_batch' | 'run_refine' | 'open_ad_files' | 'open_ad_folder'
  | 'open_batch_files' | 'open_upscale_files'
  | 'caption_pick_files' | 'caption_pick_folder' | 'caption_pick_outdir' | 'caption_pick_caformer_dir'
  | 'apply_search_result' | 'add_search_to_queue' | 'export_search_results' | 'import_search_results'
  | 'reset_prompt_deck'
  | 'search_events' | 'select_event' | 'event_add_to_queue' | 'event_generate_now'
  | 'export_event_results' | 'import_event_results'
  | 'start_queue' | 'stop_queue' | 'pause_queue' | 'resume_queue' | 'clear_queue'
  | 'remove_queue_items' | 'move_queue_item' | 'update_queue_item' | 'add_image_to_queue'
  | 'editor_open_file' | 'editor_save' | 'editor_save_as' | 'editor_add_yolo_model'
  | 'editor_clear_yolo_models' | 'editor_load_watermark_image'
  | 'open_png_info_file' | 'open_compare_image' | 'pnginfo_send_prompt' | 'pnginfo_generate'
  | 'pull_prompt_from_image' | 'explore_seed' | 'copy_to_clipboard'
  | 'gallery_open_folder' | 'gallery_send_exif_to_t2i' | 'add_favorite' | 'remove_favorite' | 'delete_image'
  | 'toggle_automation' | 'stop_automation' | 'set_automation_settings'
  | 'automation_override_next' | 'pause_automation' | 'resume_automation'
  | 'workflow_profile_list' | 'workflow_profile_save' | 'workflow_profile_load'
  | 'workflow_profile_delete' | 'workflow_profile_rename'
  | 'prompt_order_list' | 'prompt_order_save' | 'prompt_order_reset'
  | 'instant_wildcards_list' | 'instant_wildcards_save' | 'instant_wildcards_delete'
  | 'show_prompt_history' | 'open_lora_manager' | 'show_api_manager' | 'open_url'
  | 'import_anima_from_forge' | 'unload_model_request' | 'show_toast'
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
  | 'uiPrefsLoaded' | 'condRulesLoaded' | 'loraStackLoaded' | 'loraInserted' | 'globalWeightsLoaded'
  | 'promptOrderLoaded' | 'instantWildcardsList' | 'workflowProfilesList'
  | 'editorImageLoaded' | 'editorWatermarkImageLoaded' | 'editorResult' | 'yoloModelUpdated' | 'i2iImageLoaded' | 'inpaintImageLoaded'
  | 'compareImageLoaded' | 'galleryFolderLoaded' | 'galleryImagesReady' | 'thumbnailReady'
  | 'upscalersReady' | 'ollamaModelsReady' | 'adetailerModelsReady'
  | 'chatToken' | 'chatDone' | 'chatThreads'
  | 'chatGenerationEvent' | 'modelDownloadEvent' | 'creatorCacheEvent' | 'chatModelInfo'
  | 'chatModelsReady' | 'aiAssistInstructionsChanged' | 'instructionPresetsChanged'
  | 'xyzCapabilitiesReceived' | 'xyzPlotEvent'
  | 'comfyWorkflowEvent' | 'comfyCompatibilityResult' | 'relightEvent' | 'handReconstructionEvent'
  | 'batchFilesSelected' | 'adetailerResult' | 'adetailerProgress' | 'sam3Result' | 'sam3Progress'
  | 'captionFilesSelected' | 'captionProgress' | 'captionDone' | 'captionOutDirSelected'
  | 'captionModelDirSelected' | 'captionRuntimeReady'
  | 'eventSearchProgress' | 'eventSearchResults' | 'eventImportResults'
  | 'ollamaResult' | 'genNlResult' | 'vramUpdated' | 'showNotification' | 'tabChanged'
  | 'backendSelectionRequired' | 'backendProbeResult' | 'backendSelected' | 'comfyWorkflowPicked'
  | 'backendRuntimeEvent' | 'generationApiEvent' | 'backendStatus'
  | 'creatorStateChanged' | 'creatorProgress' | 'creatorResult' | 'creatorMediaSelected'
  | 'comicStoryboardReady' | 'comicDocumentChanged'
  | 'loraManagerUrlReady' | 'refineResult'

// AI helper instructions are intentionally independent of Chat settings.
export type AiAssistFeature = 'expand' | 'suggest' | 'nl2tags' | 'nl_caption'
  | 'nl_scene' | 'translate' | 'creative' | 'negative' | 'auto_nl'

export interface AiAssistInstructions {
  common: string
  features: Record<AiAssistFeature, string>
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

/** set_rating_filter 페이로드 — g/s/q/e 중 활성 등급 */
export interface SetRatingFilterPayload { ratings: string[] }

/** set_high_res_factor 페이로드 */
export interface SetHighResPayload { enabled: boolean; factor: number }

/** LoRA 스택 1개 항목 (ui_prefs.loraStack 단일 소스). weight는 set_lora_stack에선 0~3 배율,
 *  loraStack 저장 시엔 정수 퍼센트(weight/100 = 배율). */
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
  }
}
