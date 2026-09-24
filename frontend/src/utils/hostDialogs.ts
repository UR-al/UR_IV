/**
 * 호스트 PC 네이티브 대화상자 가용성 — 원격(LAN) 웹 모드에서 해당 버튼을 끈다.
 *
 * 파일 선택·저장 위치·프리셋 이름 같은 대화상자는 Python(호스트 PC)이 띄운다.
 * 데스크톱 앱이나 loopback 웹 모드(같은 PC 브라우저)에선 같은 화면에 뜨지만,
 * AISTUDIO_BIND=0.0.0.0 으로 연 원격 웹 모드에선 호스트 화면에만 떠서 원격 사용자에겐
 * 버튼이 무반응이 된다. 서버도 같은 액션을 거부하고(core/web_action_policy.py),
 * 여기선 누르기 전에 버튼을 꺼서 이유를 보여 준다.
 *
 * DESKTOP_DIALOG_ACTIONS 는 Python DESKTOP_DIALOG_ACTIONS 와 같아야 한다 —
 * tests/test_web_action_policy.py 가 두 목록을 대조한다.
 */
import type { Directive } from 'vue'

export const DESKTOP_DIALOG_ACTIONS: readonly string[] = [
  // 에디터
  'editor_open_file', 'editor_save_as', 'editor_load_watermark_image', 'editor_add_yolo_model',
  // 배치·업스케일·ADetailer
  'open_batch_files', 'open_upscale_files', 'open_ad_files', 'open_ad_folder',
  // 캡션
  'caption_pick_files', 'caption_pick_folder', 'caption_pick_outdir', 'caption_pick_caformer_dir',
  // 갤러리·PNG Info(메타 이식: 대상 선택·저장 위치)·비교
  'gallery_open_folder', 'open_png_info_file', 'pnginfo_transplant_meta', 'open_compare_image',
  // 검색·이벤트 내보내기/가져오기
  'export_search_results', 'import_search_results', 'export_event_results', 'import_event_results',
  // 프롬프트 히스토리
  'show_prompt_history',
  // 설정 백업 내보내기·생성/캐릭터 프리셋 공유
  'settings_export', 'presets_export', 'presets_import',
  'character_presets_export', 'character_presets_import',
  // Creator·대화
  'creator_select_media', 'chat_export',
]

export const HOST_DIALOG_MESSAGE =
  '원격 웹 모드에서는 호스트 PC 의 파일 대화상자·창을 열 수 없습니다 — 호스트 PC 에서 작업하세요.'

type HostWindow = { __AISTUDIO_WS_PORT__?: unknown, __AISTUDIO_WS_URL__?: unknown, __AISTUDIO_HOST_DIALOGS__?: unknown }

function currentWindow(): HostWindow | undefined {
  return typeof window === 'undefined' ? undefined : (window as unknown as HostWindow)
}

/**
 * 브라우저(웹 모드)에서 도는가 — 설정 복원·앱 재시작처럼 호스트 권한 액션은 웹 모드면 loopback
 * 이어도 서버가 항상 거부한다(core/web_action_policy.WEB_BLOCKED_ACTIONS). 버튼을 미리 끄는 데 쓴다.
 */
export function isWebHost(win: HostWindow | undefined = currentWindow()): boolean {
  if (!win) return false
  return Boolean(win.__AISTUDIO_WS_PORT__ || win.__AISTUDIO_WS_URL__)
}

/** 호스트 대화상자가 이 화면에 뜨는가. 데스크톱 앱·loopback 웹 모드 = true. */
export function hostDialogsAvailable(win: HostWindow | undefined = currentWindow()): boolean {
  if (!win) return true
  const webMode = Boolean(win.__AISTUDIO_WS_PORT__ || win.__AISTUDIO_WS_URL__)
  if (!webMode) return true
  // 웹 서버가 runtime-config.js 로 알려 준다. 값이 없으면(구버전 서버) 막지 않는다 —
  // 그 경우에도 서버가 원격 모드면 액션을 거부하고 이유를 토스트로 돌려준다.
  return win.__AISTUDIO_HOST_DIALOGS__ !== false
}

export function isDesktopDialogAction(action: unknown): boolean {
  return typeof action === 'string' && DESKTOP_DIALOG_ACTIONS.includes(action.trim().toLowerCase())
}

/** 이 액션을 지금 누를 수 있는가(= 대화상자 액션이 아니거나 호스트 대화상자를 볼 수 있음). */
export function hostDialogActionAllowed(action: unknown, win: HostWindow | undefined = currentWindow()): boolean {
  return !isDesktopDialogAction(action) || hostDialogsAvailable(win)
}

type ActivationListener = (event: Event) => void

type DisableTarget = {
  tagName?: string
  disabled?: boolean
  title?: string
  style?: { pointerEvents?: string, opacity?: string, cursor?: string }
  setAttribute?: (name: string, value: string) => void
  classList?: { add: (name: string) => void }
  addEventListener?: (type: string, listener: ActivationListener, options?: boolean) => void
  /** 이미 단 차단 리스너 — updated 가 다시 불려도 두 번 달지 않는다. */
  __hostDialogBlocker?: ActivationListener
}

const FORM_CONTROL_TAGS = new Set(['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'])
/** div 같은 클릭 영역에서 막을 '누름' 이벤트. 키보드는 Enter·Space 만(Tab 이동은 살린다). */
const BLOCKED_POINTER_EVENTS = ['click', 'dblclick', 'auxclick'] as const

function blockActivation(event: Event): void {
  if (event.type === 'keydown') {
    const key = (event as KeyboardEvent).key
    if (key !== 'Enter' && key !== ' ') return
  }
  event.preventDefault()
  event.stopImmediatePropagation()
  event.stopPropagation()
}

/**
 * disabled 가 없는 요소(div 등)의 누름을 캡처 단계에서 삼킨다.
 * pointer-events:none 으로 막으면 hover 도 사라져 title(끈 이유)이 영영 안 뜬다 —
 * 그래서 포인터는 살려 두고 활성화만 막는다.
 */
function installActivationBlocker(el: DisableTarget): void {
  if (el.__hostDialogBlocker || typeof el.addEventListener !== 'function') return
  el.__hostDialogBlocker = blockActivation
  for (const type of BLOCKED_POINTER_EVENTS) el.addEventListener(type, blockActivation, true)
  el.addEventListener('keydown', blockActivation, true)
}

/** 원격 웹 모드면 요소를 끄고 이유를 title 로 단다. 끌 필요가 없으면 false. */
export function applyHostDialogState(el: DisableTarget, action: unknown,
  win: HostWindow | undefined = currentWindow()): boolean {
  if (hostDialogActionAllowed(action, win)) return false
  const tag = String(el.tagName || '').toUpperCase()
  if (FORM_CONTROL_TAGS.has(tag)) el.disabled = true
  else {
    // div 같은 클릭 영역은 disabled 가 없다 — 누름만 막고 hover(툴팁)는 살린다.
    installActivationBlocker(el)
    if (el.style) {
      el.style.opacity = '0.45'
      el.style.cursor = 'not-allowed'
    }
  }
  el.title = HOST_DIALOG_MESSAGE
  el.setAttribute?.('aria-disabled', 'true')
  el.classList?.add('host-dialog-off')
  return true
}

/**
 * `v-host-dialog="'open_batch_files'"` — 값은 그 버튼이 부르는 액션 이름.
 * 목록에 없는 액션은 건드리지 않는다. 템플릿의 :disabled 가 다시 켜도 updated 에서 다시 끈다.
 * created 에서도 건다 — 템플릿 @click 보다 먼저 캡처 리스너를 달아, 요소 자체를 눌렀을 때
 * (at-target) 리스너 실행 순서가 등록 순서인 옛 Chromium 에서도 차단이 먼저 돈다.
 */
export const vHostDialog: Directive<HTMLElement, string> = {
  created(el, binding) { applyHostDialogState(el as unknown as DisableTarget, binding.value) },
  mounted(el, binding) { applyHostDialogState(el as unknown as DisableTarget, binding.value) },
  updated(el, binding) { applyHostDialogState(el as unknown as DisableTarget, binding.value) },
}
