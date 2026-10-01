import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'
import type { ModelKind } from './anima38Card'
import {
  choiceOf, isBlocked, normalizeOverrides, optionStatus, specOf, type ForgeOptionSpec, type OptionsCapsLike,
  type Overrides,
} from './forgeOptionOverrides'

/**
 * VAE DeGrid 카드(components/params/VaeDegridCard.vue)와 Hires.fix 경고의 순수 로직.
 *
 * 파이썬 core/vae_degrid.py(위젯 키·앱 기본값·모드 키와 확장 UI 라벨·강도/타일 범위·ComfyUI 옵션)와
 * ui/vae_degrid_ui.py(`comfyModels` 속성 이름)의 거울이다 — tests/test_vae_degrid_card_mirror.py 가 이 파일을 읽어
 * 파이썬 표와 대조한다(하나라도 다르면 실패). 위젯 값은 파이썬 프록시와 같은 문자열('true'/'false', 모델 이름 — ''
 * 는 자동, 모드 키, '1', '512')이다. 보낼지(게이트·대상·백엔드)는 파이썬이 정한다 — 여기서는 카드 표시만 계산한다.
 *
 * 모델 목록: Forge 는 기능 스냅샷 `choices.degrid_models`(Forge 가 **시작할 때** 만든 목록 — 파일이 없으면 ['None']),
 * ComfyUI 는 파이썬이 연결 때(그리고 카드 ↻ 때) object_info 에서 읽어 `_degrid_model` 위젯 속성 `comfyModels` 로 보낸
 * Forge 식 이름(배열 = 노드 있음, null = Forge·모름·옛 팩). 'None' 자리 표시는 어디서든 뺀다.
 *
 * ↻ 는 백엔드와 상관없이 같은 액션(`sam_extra_capabilities_get {refresh: true}`)이다 — 파이썬이 Forge 면 기능 스냅샷을,
 * ComfyUI 면 object_info 를 다시 받는다(ui/sam_extra_capabilities_actions → ui/vae_degrid_ui.refresh_comfy_models).
 */

export type DegridWidgetKey = 'enabled' | 'model' | 'mode' | 'strength' | 'tile' | 'apply_img2img'
export type DegridValues = Record<DegridWidgetKey, string>

export interface DegridOption {
  key: string
  label: string
}

/** 위젯 스토어 키 — 파이썬 core/vae_degrid.widget_id(key) = '_degrid_<key>'. */
export const WIDGET_IDS: Readonly<Record<DegridWidgetKey, string>> = {
  enabled: '_degrid_enabled',
  model: '_degrid_model',
  mode: '_degrid_mode',
  strength: '_degrid_strength',
  tile: '_degrid_tile',
  apply_img2img: '_degrid_apply_img2img',
}

/** `_degrid_model` 위젯 속성 — ComfyUI 노드의 모델 선택지(파이썬 ui/vae_degrid_ui.COMFY_MODELS_PROPERTY). */
export const COMFY_MODELS_PROPERTY = 'comfyModels'

/** 모델이 없을 때 Forge 목록의 자리 표시(파이썬 NONE_NAME) — 파일이 아니다. */
export const NONE_NAME = 'None'

/** 모드 — 확장 UI 라벨 그대로(상류 이름이라 번역하지 않는다), 확장 라디오 순서. 키를 저장한다. */
export const MODE_OPTIONS: readonly DegridOption[] = [
  { key: 'full', label: 'Full (전체)' },
  { key: 'dark', label: 'Dark Pixels Mainly (어두운 점 위주)' },
  { key: 'bright', label: 'Bright Pixels Mainly (밝은 점 위주)' },
]

/** 요약 줄의 짧은 모드 이름 — 파이썬 SHORT_MODE. */
export const SHORT_MODE: Readonly<Record<string, string>> = {
  full: 'Full',
  dark: 'Dark',
  bright: 'Bright',
}

/** 앱 기본값 = 확장 기본값(꺼짐·자동·Full·1·512, 파이썬 widget_values(APP_DEFAULTS)). "기본값" 버튼이 스토어에 쓴다. */
export const DEFAULTS: Readonly<DegridValues> = {
  enabled: 'false',
  model: '',
  mode: 'full',
  strength: '1',
  tile: '512',
  apply_img2img: 'false',
}

/** 강도 입력 범위 — 확장 UI 슬라이더(파이썬 STRENGTH_MIN/MAX/STEP). */
export const STRENGTH_RANGE = { min: 0, max: 1.5, step: 0.05 } as const
/** 타일 입력 범위 — 0(나누지 않음) 또는 128 단위(파이썬 MAX_TILE·TILE_STEP). 1~127 은 확장이 128 로 읽는다. */
export const TILE_RANGE = { min: 0, max: 4096, step: 128 } as const
const MIN_TILE = 128

/** Forge 설정(P10 덮어쓰기) 키 — 카드의 '장치·메모리' 칸이 읽기 전용으로 보인다. 파이썬 OPT_DEVICE·OPT_GPU_PRECISION·
 *  OPT_KEEP_LOADED 순서. */
export const OPTION_KEYS = ['sam3_degrid_device', 'sam3_degrid_gpu_precision', 'sam3_degrid_keep_loaded'] as const
export type DegridOptionKey = typeof OPTION_KEYS[number]

/** ComfyUI 노드가 쓰는 값 — 확장 기본값(파이썬 COMFY_OPTIONS). Forge 설정은 ComfyUI 에 해당 없다. */
export const COMFY_OPTIONS: Readonly<Record<DegridOptionKey, string | boolean>> = {
  sam3_degrid_device: 'auto',
  sam3_degrid_gpu_precision: 'fp32',
  sam3_degrid_keep_loaded: false,
}

/** 업스케일러(Hires.fix·배치 업스케일)에 DeGrid 모델을 골랐을 때의 경고. */
export const UPSCALER_DEGRID_WARNING = 'DeGrid 모델은 업스케일러가 아닙니다 — 거의 검은 이미지가 나옵니다.'

const MODE_KEYS = new Set(MODE_OPTIONS.map(o => o.key))
const MODEL_EXTENSIONS = ['.safetensors', '.pth', '.pt', '.ckpt', '.bin']

export function labelOf(options: readonly DegridOption[], key: string): string {
  return options.find(o => o.key === key)?.label ?? key
}

export function keyOf(options: readonly DegridOption[], label: string): string {
  return options.find(o => o.label === label)?.key ?? label
}

function text(raw: unknown): string {
  return raw === undefined || raw === null ? '' : String(raw).trim()
}

/** 모델 이름 → 저장값. ''·'None'·'auto'(대소문자 무시)는 자동('') — 파이썬 normalize_model. */
export function normalizeModel(raw: unknown): string {
  const value = text(raw)
  return ['', 'none', 'auto'].includes(value.toLowerCase()) ? '' : value
}

/** 스토어 → 카드 값. 비었거나 모르는 칸은 앱 기본값(파이썬 parse_settings 와 같은 규칙 — 모델 '' 는 값이다). */
export function readValues(widgets: Record<string, unknown>): DegridValues {
  const pick = (key: DegridWidgetKey): string => text(widgets[WIDGET_IDS[key]]) || DEFAULTS[key]
  const mode = pick('mode')
  return {
    enabled: pick('enabled'),
    model: normalizeModel(widgets[WIDGET_IDS.model]),
    mode: MODE_KEYS.has(mode) ? mode : DEFAULTS.mode,
    strength: pick('strength'),
    tile: pick('tile'),
    apply_img2img: pick('apply_img2img'),
  }
}

/** 강도 → [0, 1.5], 읽을 수 없으면 1(파이썬 coerce_strength). */
export function coerceStrength(raw: unknown): number {
  const n = typeof raw === 'number' ? raw : Number(text(raw))
  if (!text(raw) || !Number.isFinite(n)) return 1
  return Math.min(Math.max(n, STRENGTH_RANGE.min), STRENGTH_RANGE.max)
}

/** 타일 → 0 또는 [128, 4096](1~127 은 128), 읽을 수 없으면 512(파이썬 coerce_tile). */
export function coerceTile(raw: unknown): number {
  const n = typeof raw === 'number' ? raw : Number(text(raw))
  if (!text(raw) || !Number.isFinite(n)) return Number(DEFAULTS.tile)
  const tile = Math.trunc(n)
  if (tile <= 0) return 0
  return Math.min(Math.max(tile, MIN_TILE), TILE_RANGE.max)
}

/** 강도 문자열 — 1 → '1', 0.85 → '0.85'(파이썬 format_strength). */
export function formatStrength(value: number): string {
  return String(Math.round(value * 1000) / 1000)
}

export function tileText(tile: number): string {
  return tile <= 0 ? '타일 없음' : String(tile)
}

/** 카드 제목 옆 요약 — 꺼짐이면 ''. 'Full 1 · 512', 'Dark 0.8 · 타일 없음 · <모델>'(파이썬 describe 와 같은 문구). */
export function summary(values: DegridValues): string {
  if (values.enabled !== 'true') return ''
  const head = `${SHORT_MODE[values.mode] ?? values.mode} ${formatStrength(coerceStrength(values.strength))} · ${tileText(coerceTile(values.tile))}`
  return values.model ? `${head} · ${values.model}` : head
}

function lastSegment(name: string): string {
  const parts = name.replace(/\\/g, '/').split('/')
  return parts[parts.length - 1] ?? name
}

/** 파일 이름 줄기 — 폴더 접두와 모델 확장자를 뗀 소문자(비교용). */
export function modelStem(name: string): string {
  let base = lastSegment(text(name))
  const lower = base.toLowerCase()
  const ext = MODEL_EXTENSIONS.find(item => lower.endsWith(item))
  if (ext) base = base.slice(0, -ext.length)
  return base.toLowerCase()
}

/** 목록(Forge 식 이름)에서 고른 이름을 찾는다 — 정확히 → 대소문자 무시 → 줄기(파이썬 resolve_name). 자동('')이면 첫 이름. */
export function resolveModel(name: string, names: readonly string[]): string | null {
  if (!names.length) return null
  const wanted = normalizeModel(name)
  if (!wanted) return names[0]!
  return names.find(item => item === wanted)
    ?? names.find(item => item.toLowerCase() === wanted.toLowerCase())
    ?? names.find(item => modelStem(item) === modelStem(wanted))
    ?? null
}

/** 목록을 읽는 스냅샷 칸 — ComfyUI 스냅샷(status 'not_applicable', choices 없음)도 받는다. */
type ChoicesCaps = {
  readonly status?: string
  readonly choices?: Readonly<Record<string, readonly string[]>>
} | null | undefined

function cleanNames(list: unknown): string[] | null {
  if (!Array.isArray(list)) return null
  const out: string[] = []
  for (const item of list) {
    const name = text(item)
    if (name && name !== NONE_NAME && !out.includes(name)) out.push(name)
  }
  return out
}

/** 지금 백엔드의 모델 목록(Forge 식 이름, 'None' 뺌) — ComfyUI 는 `comfyModels`(배열일 때), 그 밖은 Forge 스냅샷.
 *  모르면 null. */
export function liveModels(caps: ChoicesCaps, comfyModels?: unknown): string[] | null {
  if (Array.isArray(comfyModels)) return cleanNames(comfyModels)
  return cleanNames(caps?.choices?.degrid_models)
}

export interface ModelChoice {
  value: string
  label: string
}

export interface ModelChoices {
  options: ModelChoice[]
  /** 목록을 모른다(Forge 확인 전·스냅샷에 목록이 없음) */
  unverified: boolean
  /** 고른 모델이 알려진 목록에 없다 */
  missing: boolean
  /** 자동('')이 지금 고를 이름(목록 첫 이름) — 모르면 null */
  autoName: string | null
}

export const AUTO_LABEL = '자동 (가장 높은 버전)'

/**
 * 모델 칸 선택지. 맨 앞은 자동('' — 확장이 modelspec 버전이 가장 높은 파일을 고른다, 지금 고를 이름을 보인다), 그 뒤
 * 목록 이름. 지금 값이 목록에 정확히 없으면 자동 다음에 붙인다(저장값 보존) — 알려진 목록에서 찾지 못하면 '목록에 없음'.
 */
export function modelChoices(caps: ChoicesCaps, current: string, comfyModels?: unknown): ModelChoices {
  const live = liveModels(caps, comfyModels)
  const names = live ?? []
  const autoName = names[0] ?? null
  const options: ModelChoice[] = [{ value: '', label: autoName ? `자동 (가장 높은 버전 — 지금: ${autoName})` : AUTO_LABEL }]
  for (const name of names) options.push({ value: name, label: name })
  const value = normalizeModel(current)
  let missing = false
  if (value && !names.includes(value)) {
    missing = live !== null && resolveModel(value, names) === null
    options.splice(1, 0, { value, label: missing ? `${value} (목록에 없음)` : value })
  }
  return { options, unverified: live === null, missing, autoName }
}

export type DegridCardStatus =
  | 'off' | 'krea2'
  | 'comfy_ok' | 'comfy_pack_update' | 'comfy_no_model' | 'comfy_model_missing'
  | 'unknown' | 'missing' | 'no_model' | 'model_missing' | 'other_model' | 'ok'
export type DegridTone = 'info' | 'warn' | 'alert'

/** 카드 상태가 읽는 스냅샷 칸만 — useSamExtraCapabilities 의 readonly 값도 그대로 받는다. */
export type DegridCapsLike = Readonly<Pick<SamExtraCapabilitiesEvent, 'status' | 'known'>> & {
  readonly features?: Readonly<Partial<Record<SamExtraFeature, boolean>>>
  readonly choices?: Readonly<Record<string, readonly string[]>>
}

export interface DegridStatusInput {
  caps: DegridCapsLike | null | undefined
  values: DegridValues
  /** 지금 체크포인트 종류(utils/anima38Card modelKind) — 'other' 면 Anima 가 아니다 */
  kind?: ModelKind
  krea2?: boolean
  /** `_degrid_model` 의 `comfyModels` 속성(배열 = 노드 있음, 그 밖 = 모름·옛 팩) */
  comfyModels?: unknown
}

/** 모델 목록 ↻ 버튼 설명 — 지금 백엔드가 다시 받는 곳(ComfyUI 스냅샷은 status 'not_applicable'). */
export function refreshTitle(caps: { readonly status?: string } | null | undefined): string {
  return caps?.status === 'not_applicable'
    ? 'ComfyUI 에서 DeGrid 모델 목록 다시 받기 (30초에 한 번). 새로 넣은 파일도 바로 보이고, 다음 생성도 이 목록으로 만듭니다'
    : '연결된 Forge 에서 모델 목록 다시 받기 (30초에 한 번). Forge 목록은 Forge 가 시작할 때 만들어집니다'
}

const COMFY_OPTIONS_NOTE = 'Forge 설정의 장치·정밀도·VRAM 상주는 ComfyUI 에 해당 없음 — 노드는 auto · fp32 · 끔으로 돕니다.'

/**
 * 카드 아래 상태 한 줄. 파이썬 규칙과 같다: 꺼짐·Krea2 는 보내지 않는다. Forge 는 사용자가 켠 값이라 확인 전에도
 * 보내고(없으면 Forge 가 거절하고 설명), 스크립트가 없다고 확인되면 보내지 않는다. 모델이 Forge 시작 때 목록에 없어도
 * 보낸다(생성할 때 확장이 다시 찾는다 — 결과 알림이 정답). ComfyUI 는 노드·모델이 없으면 노드를 빼고 생성한다.
 * `status === 'not_applicable'` 은 ComfyUI 백엔드다.
 */
export function cardStatus(input: DegridStatusInput): { status: DegridCardStatus; tone: DegridTone; text: string } {
  const { caps, values, kind = 'unknown', krea2 = false, comfyModels } = input
  if (values.enabled !== 'true') {
    return { status: 'off', tone: 'info', text: '꺼짐 — 켜면 생성마다 모든 후처리 뒤, 저장 직전에 이미지마다 한 번 VAE 격자를 지웁니다.' }
  }
  if (krea2) {
    return { status: 'krea2', tone: 'warn', text: 'Krea2 생성에는 보내지 않습니다 — Anima(Qwen VAE) 격자용 후처리입니다.' }
  }
  const otherModel = kind === 'other'
    ? 'Anima 모델이 아닙니다 — 그래도 보냅니다. DeGrid 는 Anima(Qwen VAE) 격자용이라 다른 모델에서는 효과가 다를 수 있습니다.'
    : ''
  if (caps?.status === 'not_applicable') {
    if (!Array.isArray(comfyModels)) {
      return {
        status: 'comfy_pack_update', tone: 'alert',
        text: 'ComfyUI 에서 DeGrid 노드를 확인하지 못했습니다 — 앱 노드 팩 1.5.0(ai_studio_forge_parity)이 필요합니다. '
          + '노드가 없으면 DeGrid 없이 생성합니다. 팩을 갱신하고 ComfyUI 를 재시작했으면 ↻ 로 다시 확인하세요.',
      }
    }
    const names = liveModels(caps, comfyModels) ?? []
    if (!names.length) {
      return {
        status: 'comfy_no_model', tone: 'alert',
        text: 'ComfyUI 에 DeGrid 모델이 없습니다 — models/upscale_models 또는 models/degrid 에 두고 ↻ 로 목록을 다시 받으세요. '
          + '없으면 DeGrid 없이 생성합니다.',
      }
    }
    if (values.model && resolveModel(values.model, names) === null) {
      return {
        status: 'comfy_model_missing', tone: 'alert',
        text: `고른 모델 '${values.model}' 이(가) ComfyUI 목록에 없어 DeGrid 없이 생성합니다 — 파일을 넣었으면 ↻ 로 목록을 `
          + '다시 받고, 아니면 다시 고르거나 자동으로 두세요.',
      }
    }
    if (otherModel) return { status: 'other_model', tone: 'warn', text: `${otherModel} ${COMFY_OPTIONS_NOTE}` }
    return { status: 'comfy_ok', tone: 'info', text: `ComfyUI 노드로 적용합니다. ${COMFY_OPTIONS_NOTE}` }
  }
  if (!caps || !caps.known) {
    return {
      status: 'unknown', tone: 'warn',
      text: '확장 확인 전 — 켠 값은 그대로 보냅니다(스크립트가 없는 Forge 는 요청을 거절하고 이유를 알려 줍니다).',
    }
  }
  if (caps.features?.degrid !== true) {
    return { status: 'missing', tone: 'alert', text: '연결된 Forge 의 sam-extra 에 VAE DeGrid 스크립트가 없어 보내지 않습니다(확장 업데이트 필요).' }
  }
  const names = liveModels(caps)
  if (names !== null && !names.length) {
    return {
      status: 'no_model', tone: 'alert',
      text: 'Forge 가 시작할 때 DeGrid 모델이 없었습니다 — models/ESRGAN 또는 models/DeGrid 에 두세요(생성할 때 확장이 다시 찾습니다).',
    }
  }
  if (names !== null && values.model && resolveModel(values.model, names) === null) {
    return {
      status: 'model_missing', tone: 'warn',
      text: `'${values.model}' 이(가) Forge 시작 때 목록에 없습니다 — 그대로 보내고, 생성할 때 확장이 다시 찾습니다(못 찾으면 결과 알림).`,
    }
  }
  if (otherModel) return { status: 'other_model', tone: 'warn', text: otherModel }
  return { status: 'ok', tone: 'info', text: '생성마다 모든 후처리 뒤, 저장 직전에 이미지마다 한 번 적용합니다.' }
}

function valueText(spec: ForgeOptionSpec, value: boolean | string): string {
  if (typeof value === 'boolean') return value ? '켬' : '끔'
  return spec.choices.find(([item]) => item === value)?.[1] ?? value
}

export interface EffectiveOption {
  key: DegridOptionKey
  label: string
  /** 'app' = 앱이 이 요청에 덮어씀, 'forge' = Forge 설정값, 'unknown' = 확인 전, 'comfy' = ComfyUI 노드 고정값 */
  source: 'app' | 'forge' | 'unknown' | 'comfy'
  text: string
  /** 앱 값을 골랐지만 이 Forge 에 보낼 수 없는 상태(경고 색) */
  blocked: boolean
}

/**
 * 카드의 '장치·메모리 (Forge 설정)' 읽기 전용 한 줄 — 앱 덮어쓰기(설정 › Forge, P10)가 있으면 그 값, 없으면 Forge 설정값.
 * ComfyUI 는 노드 고정값(확장 기본값)이다. 보낼지는 파이썬(core/forge_override_settings)이 정한다 — 여기는 표시만.
 */
export function effectiveOption(
  key: DegridOptionKey,
  overrides: Readonly<Overrides>,
  caps: OptionsCapsLike | null | undefined,
): EffectiveOption {
  const spec = specOf(key)!
  if (caps?.status === 'not_applicable') {
    return { key, label: spec.label, source: 'comfy', text: `ComfyUI 노드: ${valueText(spec, COMFY_OPTIONS[key])}`, blocked: false }
  }
  const status = optionStatus(caps, key)
  const chosen = normalizeOverrides(overrides)[key]
  if (chosen !== undefined) {
    const blocked = isBlocked(status, choiceOf(overrides, key))
    const tail = blocked ? ' — 지금은 보내지 못함(설정 › Forge 참고)' : ''
    return { key, label: spec.label, source: 'app', text: `앱 설정: ${valueText(spec, chosen)}${tail}`, blocked }
  }
  if (status.forgeValue === null) {
    const hint = status.state === 'missing' ? '이 Forge 에 없음' : '값 확인 전'
    return { key, label: spec.label, source: 'unknown', text: `Forge 설정 따름 (${hint})`, blocked: false }
  }
  return { key, label: spec.label, source: 'forge', text: `Forge 설정: ${valueText(spec, status.forgeValue)}`, blocked: false }
}

/**
 * 고른 업스케일러가 DeGrid 모델인가 — Forge 는 models/ESRGAN 의 NAFNet 을, ComfyUI 는 upscale_models 의 것을 업스케일러
 * 목록에도 올린다. 업스케일러 선택 칸 전부(Hires.fix 카드·배치 업스케일)가 같은 규칙을 쓴다 — composables/
 * useUpscalerDegridWarning. Forge 스냅샷 목록과 ComfyUI `comfyModels` 모두와 줄기(폴더·확장자 뗀 이름, 대소문자
 * 무시)로 비교한다. 'None' 은 무시. 경고 문구 또는 ''.
 */
export function upscalerDegridWarning(upscaler: unknown, caps: ChoicesCaps, comfyModels?: unknown): string {
  const name = text(upscaler)
  if (!name || name === NONE_NAME) return ''
  const stems = new Set<string>()
  for (const list of [cleanNames(caps?.choices?.degrid_models), cleanNames(comfyModels)]) {
    for (const item of list ?? []) stems.add(modelStem(item))
  }
  return stems.has(modelStem(name)) ? UPSCALER_DEGRID_WARNING : ''
}
