/**
 * ui_prefs.json ↔ localStorage 미러 표 — 한 곳 (감사 #41).
 *
 * 주인은 파일(config/ui_prefs.json, Python load_ui_prefs/save_ui_prefs)이다. localStorage 사본은
 * **첫 렌더 캐시**다: 부팅 직후 파일 응답(uiPrefsLoaded)이 오기 전에 동기로 읽는 곳(UI 배율·탭 순서·
 * 블록 모드 등)이 깜빡이지 않게 할 뿐이고, 파일 값이 도착하면 그 값이 이긴다.
 *
 * 예전엔 이 미러가 App.vue·PromptPanel·SettingsView 에 세 벌 있었고 검증 규칙이 조금씩 달랐으며,
 * SettingsView 의 토글 두 개(블록 모드·갤러리 메타)는 localStorage 에만 써서 재시작하면 파일 값으로
 * 되돌아갔다. 이제
 *   - 파일 → 캐시: {@link mirrorPrefsToStorage} (App.vue uiPrefsLoaded · SettingsView getUiPrefs)
 *   - 화면 → 파일+캐시: composables/uiPrefs.ts persistUiPrefs (이 표로 캐시를 쓰고 save_ui_prefs)
 * 둘 다 이 표 하나를 쓴다.
 */
import { normalizePreviewThumbWidth } from '../composables/useHistoryThumbs'

type StorageLike = Pick<Storage, 'getItem' | 'setItem'>

/** 값 → localStorage 문자열. 받아들일 수 없는 값이면 null(캐시를 건드리지 않는다). */
type Encoder = (value: unknown) => string | null

export interface PrefMirror {
  /** ui_prefs.json 키 */
  pref: string
  /** localStorage 키 */
  storageKey: string
  encode: Encoder
}

const bool: Encoder = v => (typeof v === 'boolean' ? String(v) : null)
// 빈 문자열도 값이다 — Ollama 모델 '' = 백엔드가 설치 모델로 정한다, URL '' = 기본 주소(utils/ollamaPrefs)
const text: Encoder = v => (typeof v === 'string' ? v.trim() : null)
const oneOf = (allowed: readonly string[]): Encoder => v => (typeof v === 'string' && allowed.includes(v) ? v : null)
const numberIn = (min: number, max: number, integer = false): Encoder => (v) => {
  const n = Number(v)
  if (v === null || v === '' || typeof v === 'boolean' || !Number.isFinite(n)) return null
  if (integer && !Number.isInteger(n)) return null
  return n >= min && n <= max ? String(n) : null
}
const jsonArray = (length?: number): Encoder => (v) => {
  if (!Array.isArray(v)) return null
  if (length !== undefined && v.length !== length) return null
  return JSON.stringify(v)
}

export const HISTORY_JUMP_MODIFIERS = ['shiftKey', 'ctrlKey', 'altKey'] as const

export const PREF_MIRRORS: readonly PrefMirror[] = [
  { pref: 'tagBlockMode', storageKey: 'tagBlockMode', encode: bool },
  { pref: 'galleryShowMetadata', storageKey: 'galleryShowMetadata', encode: bool },
  { pref: 'autoAddCopyright', storageKey: 'autoAddCopyright', encode: bool },
  { pref: 'historyJumpModifier', storageKey: 'historyJumpModifier', encode: oneOf(HISTORY_JUMP_MODIFIERS) },
  { pref: 'historyBlinkSelected', storageKey: 'historyBlinkSelected', encode: bool },
  {
    pref: 'previewThumbWidth',
    storageKey: 'previewThumbWidth',
    encode: v => (v === undefined || v === null ? null : String(normalizePreviewThumbWidth(v))),
  },
  { pref: 'uiScale', storageKey: 'ui.scale', encode: numberIn(0.8, 1.5) },
  { pref: 'editorSidePanelWidth', storageKey: 'editorSidePanelWidth', encode: numberIn(200, 500, true) },
  {
    pref: 'tabOrder',
    storageKey: 'tabOrder',
    encode: v => (Array.isArray(v) && v.length > 0 ? JSON.stringify(v) : null),
  },
  { pref: 'ollamaUrl', storageKey: 'ollamaUrl', encode: text },
  { pref: 'ollamaModel', storageKey: 'ollamaModel', encode: text },
  { pref: 'ollamaUnloadOnGen', storageKey: 'ollamaUnloadOnGen', encode: bool },
  { pref: 'unloadModelsAfterGen', storageKey: 'unloadModelsAfterGen', encode: bool },
  { pref: 'forgeSaveOutputs', storageKey: 'forgeSaveOutputs', encode: bool },
  { pref: 'comfySam3KeepInRam', storageKey: 'comfySam3KeepInRam', encode: bool },
  { pref: 'autoNlGen', storageKey: 'autoNlGen', encode: bool },
  { pref: 'highResEnabled', storageKey: 'highRes.enabled', encode: bool },
  { pref: 'highResFactor', storageKey: 'highRes.factor', encode: numberIn(1, 4) },
  { pref: 'ratingFilter', storageKey: 'ratingFilter', encode: jsonArray(4) },
  { pref: 'loraStack', storageKey: 'loraStack', encode: jsonArray() },
]

const BY_PREF = new Map(PREF_MIRRORS.map(m => [m.pref, m]))

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null   // 차단된 저장소(웹 모드 사생활 설정 등)
  }
}

/** 이 ui_prefs 키의 localStorage 캐시 문자열 — 미러가 없거나 값이 잘못됐으면 null. */
export function encodePrefForStorage(pref: string, value: unknown): string | null {
  const mirror = BY_PREF.get(pref)
  return mirror ? mirror.encode(value) : null
}

/**
 * ui_prefs 조각을 localStorage 캐시에 옮긴다. 표의 검증을 통과한(= 화면에 적용해도 되는) ui_prefs
 * 키 목록을 돌려준다 — 저장소가 차단돼 캐시 쓰기가 실패해도 그 값은 유효하므로 목록에 든다.
 * 미러가 없는 키(themeOverrides·searchState 등)와 잘못된 값은 건너뛴다.
 */
export function mirrorPrefsToStorage(
  prefs: Record<string, unknown> | null | undefined,
  storage: StorageLike | null = browserStorage(),
): string[] {
  if (!prefs || typeof prefs !== 'object') return []
  const accepted: string[] = []
  for (const [pref, value] of Object.entries(prefs)) {
    const mirror = BY_PREF.get(pref)
    if (!mirror) continue
    const encoded = mirror.encode(value)
    if (encoded === null) continue
    accepted.push(pref)
    try {
      storage?.setItem(mirror.storageKey, encoded)
    } catch { /* 가득 찬·차단된 저장소 — 캐시일 뿐이라 무시 */ }
  }
  return accepted
}

/** 캐시된 불리언 — 없거나 읽을 수 없으면 fallback. */
export function readStoredBool(
  storageKey: string,
  fallback: boolean,
  storage: Pick<Storage, 'getItem'> | null = browserStorage(),
): boolean {
  try {
    const raw = storage?.getItem(storageKey)
    if (raw === 'true') return true
    if (raw === 'false') return false
  } catch { /* 차단된 저장소 */ }
  return fallback
}
