import type { SamExtraCapabilitiesEvent } from '../types/bridge'

/**
 * Anima SPEED 카드(components/params/AnimaSpeedCard.vue)의 순수 로직.
 *
 * 파이썬 core/anima_speed.py(위젯 키·앱 기본값·선택지·δ/σ 나눔값 범위·카드 값 정규화 parse_settings·설정 검사 validate
 * 와 그 한국어 문구)의 거울이다 — tests/test_anima_speed_card_mirror.py 가 이 파일을 읽어 파이썬 표와 대조하고, 정규화와
 * 검사는 두 구현이 같은 골든 파일(animaSpeed.cases.json — 파이썬이 만든 기대값)을 읽는다. 위젯 값은 파이썬 프록시와 같은
 * 문자열('true'/'false', 선택지 키, 친 그대로의 해상도 비율·수동 σ, 숫자 '0.01')이다. 보낼지(대상·백엔드·게이트)는
 * 파이썬이 정한다 — 여기서는 카드 표시(보일 칸·요약·검사 줄·안내)만 계산한다.
 *
 * 검사는 확장 sam-extra 0.32.1 의 규칙이다: coerce_settings 의 오류(해상도 비율·수동 σ·δ·σ 나눔값) → build_plan 이 설정만
 * 보고 거절하는 것(수동 σ 개수, σ 나눔값, custom A ≤ 0, dwt 비율). 해상도 비율·수동 σ 칸의 'a/0' 분수는 확장이 예외로
 * 죽는다(스크립트 오류) — 숫자는 파이썬 float() 문법으로 읽는다('1_0' = 10, '0x1'·'' 은 오류, 'inf'·'nan' 은 유한하지 않다).
 */

export type SpeedKey =
  | 'enabled' | 'mode' | 'threshold' | 'preset' | 'scales' | 'delta' | 'sigma_divisor' | 'manual_sigmas'
  | 'adaptive' | 'hires' | 'apply_img2img' | 'transform' | 'spectrum_A' | 'spectrum_beta' | 'seed'
export type SpeedValues = Record<SpeedKey, string>
export type Mode = 'transition' | 'respace'
export type Threshold = 'neo_shift' | 'delta_optimal' | 'manual'
export type Transform = 'dct' | 'dwt' | 'fft'
export type ErrorCode =
  | 'script_error' | 'choice' | 'scales' | 'manual' | 'delta' | 'divisor' | 'manual_count' | 'spectrum_A'
  | 'spectrum_beta' | 'dwt_ratio'

/** 카드 값(파이썬 SpeedSettings) — 텍스트 칸은 친 그대로, 숫자는 유한, 선택지는 확장 선택지. */
export interface SpeedSettings {
  enabled: boolean
  mode: Mode
  threshold: Threshold
  preset: string
  scales: string
  delta: number
  sigma_divisor: number
  manual_sigmas: string
  adaptive: boolean
  transform: Transform
  spectrum_A: number
  spectrum_beta: number
  seed: number
  hires: boolean
  apply_img2img: boolean
}

export interface SpeedOption<K extends string = string> {
  key: K
  label: string
}

/** 위젯 스토어 키 — 파이썬 core/anima_speed.widget_id(key) = '_aspeed_<key>'(WIDGET_KEYS 순서). */
export const WIDGET_IDS: Readonly<Record<SpeedKey, string>> = {
  enabled: '_aspeed_enabled',
  mode: '_aspeed_mode',
  threshold: '_aspeed_threshold',
  preset: '_aspeed_preset',
  scales: '_aspeed_scales',
  delta: '_aspeed_delta',
  sigma_divisor: '_aspeed_sigma_divisor',
  manual_sigmas: '_aspeed_manual_sigmas',
  adaptive: '_aspeed_adaptive',
  hires: '_aspeed_hires',
  apply_img2img: '_aspeed_apply_img2img',
  transform: '_aspeed_transform',
  spectrum_A: '_aspeed_spectrum_A',
  spectrum_beta: '_aspeed_spectrum_beta',
  seed: '_aspeed_seed',
}

/** 앱 기본값 = 확장 기본값 + 'I2I에도 적용' 끔(파이썬 widget_values(APP_DEFAULTS)). "기본값" 버튼이 스토어에 쓴다. */
export const DEFAULTS: Readonly<SpeedValues> = {
  enabled: 'false',
  mode: 'transition',
  threshold: 'neo_shift',
  preset: 'anima',
  scales: '0.5,1.0',
  delta: '0.01',
  sigma_divisor: '1.03',
  manual_sigmas: '0.7',
  adaptive: 'true',
  hires: 'false',
  apply_img2img: 'false',
  transform: 'dct',
  spectrum_A: '203.615097',
  spectrum_beta: '1.915461',
  seed: '-1',
}

/** 확장 선택지(파이썬 MODES·THRESHOLDS·TRANSFORMS·PRESET_NAMES — 드롭다운 순서). */
export const MODES: readonly Mode[] = ['transition', 'respace']
export const THRESHOLDS: readonly Threshold[] = ['neo_shift', 'delta_optimal', 'manual']
export const TRANSFORMS: readonly Transform[] = ['dct', 'dwt', 'fft']
export const PRESET_NAMES: readonly string[] = ['anima', 'flux', 'flux2', 'krea-2', 'krea-2-raw', 'z-image', 'wan21', 'custom']
/** 프리셋별 Spectrum A(파이썬 PRESETS — custom 은 카드의 A·β). 검사(custom A ≤ 0)만 쓴다. */
export const PRESET_A: Readonly<Record<string, number | null>> = {
  anima: 8664.998524, flux: 1963.48906, flux2: 520.495848, 'krea-2': 2357.960557, 'krea-2-raw': 887.402358,
  'z-image': 877.001476, wan21: 219.484718, custom: null,
}

/** 입력 범위 — 확장 UI 슬라이더(파이썬 DELTA_MIN/MAX/STEP·DIVISOR_MIN/MAX/STEP, 레지스트리 픽스처 핀 speed_ui_*).
 *  API 가 받는 범위는 더 넓다(δ 는 (0, 1), 나눔값 > 0) — 카드는 슬라이더 범위로 입력받는다. */
export const DELTA_RANGE = { min: 0.0001, max: 0.5, step: 0.0001 } as const
export const DIVISOR_RANGE = { min: 0.5, max: 2, step: 0.01 } as const

export const LABELS: Readonly<Record<SpeedKey, string>> = {
  enabled: 'SPEED 사용 (실험)',
  mode: '방식',
  threshold: '전환 σ',
  preset: '스펙트럼 프리셋',
  scales: '해상도 비율',
  delta: 'δ (노이즈 지배 허용치)',
  sigma_divisor: 'σ 나눔값',
  manual_sigmas: '수동 σ',
  adaptive: '적응형 δ (1024 px 기준 고정)',
  hires: 'Hires 패스에도 적용',
  apply_img2img: 'I2I에도 적용',
  transform: '변환',
  spectrum_A: 'Spectrum A',
  spectrum_beta: 'Spectrum β',
  seed: '스펙트럼 노이즈 시드 (-1 = 이미지 시드)',
}

export const MODE_OPTIONS: readonly SpeedOption<Mode>[] = [
  { key: 'transition', label: 'transition — 전환 스텝 σ만 정렬 (공식·aoleg, 기본)' },
  { key: 'respace', label: 'respace — 남은 σ를 비율로 다시 배치, 짝수 격자 (sorryhyun)' },
]
export const THRESHOLD_OPTIONS: readonly SpeedOption<Threshold>[] = [
  { key: 'neo_shift', label: 'neo_shift — 프리셋 σ* ÷ 나눔값 (기본)' },
  { key: 'delta_optimal', label: 'delta_optimal — 프리셋 σ* 그대로' },
  { key: 'manual', label: 'manual — 직접 적은 σ' },
]
export const TRANSFORM_OPTIONS: readonly SpeedOption<Transform>[] = [
  { key: 'dct', label: 'dct (기본)' },
  { key: 'dwt', label: 'dwt' },
  { key: 'fft', label: 'fft' },
]

export const HINTS: Readonly<Partial<Record<SpeedKey, string>>> = {
  scales: '1.0으로 끝나는 오름차순, 쉼표 구분 (예: 0.5,1.0 · 0.25,0.5,1.0)',
  delta: '작을수록 일찍 전환 — 품질↑ 속도↓',
  sigma_divisor: '1.0 = delta_optimal과 같음 · 클수록 늦게 전환(빠름·위험)',
  manual_sigmas: '전환마다 하나 · transition 방식은 내림차순 (예: 0.7)',
  adaptive: '해상도가 커져도 저해상도 구간이 늘지 않게 1024 px 기준 σ*·스텝 비율에 고정',
  transform: 'dct(기본)·fft = 아무 비율, dwt = 단계마다 정확히 2배',
  hires: '끄면 Hires 패스는 원래대로 샘플링합니다',
  apply_img2img: 'Forge 의 img2img 탭에는 따로 있는 칸 — 기본은 T2I 계열에만. 인페인트·마스크에서는 늘 건너뜁니다',
}

export const LEAD_NOTE = '초반의 노이즈가 지배적인 스텝을 DCT로 줄인 저해상도 latent에서 돌린 뒤, 전환 σ에서 원래 크기로 넓혀(κ 보정) '
  + '고른 샘플러로 이어서 샘플링합니다. 빨라지지만 이미지가 달라집니다(무손실 가속 아님). flow 모델 전용 · 기본 끔 · Anima '
  + '화질은 아직 검증 전입니다.'
export const SKIP_NOTE = '인페인트·마스크·ControlNet·레퍼런스 latent·Spectrum(ComfyUI)과 함께면 SPEED는 건너뜁니다 — 이유는 결과 정보 '
  + "'Anima SPEED status' 에 남습니다(ComfyUI 는 알림·콘솔)."
export const RESET_TITLE = '확장 기본값으로 — 꺼짐 · transition · neo_shift · anima · 0.5,1.0 · δ 0.01 · 1.03 · 0.7 · 적응형 켬 · '
  + 'dct · 시드 -1 · Hires 끔, I2I 끔'

/** 검사 줄(파이썬 VALIDATION_MESSAGES — 코드는 core/anima_speed ERR_*). */
export const VALIDATION_MESSAGES: Readonly<Record<ErrorCode, string>> = {
  script_error: '해상도 비율·수동 σ 에 0으로 나누는 분수(a/0)가 있습니다 — Forge 에서 SPEED 스크립트 오류가 납니다',
  choice: '방식·전환 σ·스펙트럼 프리셋·변환 중 확장에 없는 값이 있습니다',
  scales: '해상도 비율은 (0, 1] 사이 오름차순이고 1.0으로 끝나야 합니다',
  manual: '수동 σ는 0과 1 사이, transition 방식은 내림차순이어야 합니다',
  delta: 'δ는 0과 1 사이여야 합니다',
  divisor: 'σ 나눔값은 0보다 커야 합니다',
  manual_count: '수동 σ 개수는 전환 수(해상도 비율 개수 − 1)와 같아야 합니다 (respace 는 하나도 됩니다)',
  spectrum_A: 'Spectrum A 는 0보다 커야 합니다',
  spectrum_beta: 'Spectrum β 는 유한한 수여야 합니다',
  dwt_ratio: 'dwt 변환은 해상도 비율이 단계마다 정확히 2배여야 합니다 (dct·fft 는 아무 비율)',
}
/** 검사 오류가 있을 때 덧붙이는 줄(파이썬 INVALID_NOTE). */
export const INVALID_NOTE = '이대로 생성하면 SPEED 는 적용되지 않습니다(설정 오류)'

// ── 숫자 읽기 — 파이썬 float() 문법(밑줄은 숫자 사이에만, inf·nan 은 읽히지만 유한하지 않다) ─────────────────
const PY_FLOAT = /^[+-]?(?:(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.?)(?:e[+-]?\d(?:_?\d)*)?|inf(?:inity)?|nan)$/i

/** 파이썬 ``float(text)`` — 읽을 수 없으면 null(ValueError). 앞뒤 공백은 무시한다. */
export function pyFloat(text: unknown): number | null {
  const value = String(text ?? '').trim()
  if (!PY_FLOAT.test(value)) return null
  const lower = value.toLowerCase()
  if (lower.includes('inf')) return lower.startsWith('-') ? -Infinity : Infinity
  if (lower.includes('nan')) return NaN
  return Number(value.replace(/_/g, ''))
}

type Parsed = { values: number[] } | { error: 'value' | 'zero' }

/** 확장 schedule._numbers — ',' 와 ';', 'a/b' 분수, 유한한 수만. 첫 실패에서 멈춘다: 'value'(ValueError — 확장은 오류
 *  목록으로) 또는 'zero'(0으로 나눔 — 확장은 예외로 죽는다). */
export function numbers(text: unknown): Parsed {
  const out: number[] = []
  for (const item of String(text ?? '').replace(/;/g, ',').split(',')) {
    const token = item.trim()
    if (!token) continue
    let value: number
    if (token.includes('/')) {
      const at = token.indexOf('/')
      const num = pyFloat(token.slice(0, at))
      if (num === null) return { error: 'value' }
      const den = pyFloat(token.slice(at + 1))
      if (den === null) return { error: 'value' }
      if (den === 0) return { error: 'zero' }
      value = num / den
    } else {
      const parsed = pyFloat(token)
      if (parsed === null) return { error: 'value' }
      value = parsed
    }
    if (!Number.isFinite(value)) return { error: 'value' }
    out.push(value)
  }
  return { values: out }
}

function scalesError(scales: readonly number[]): boolean {
  if (scales.length === 0) return true
  if (scales.some(s => s <= 0 || s > 1)) return true
  if (Math.abs(scales[scales.length - 1] - 1) > 1e-6) return true
  return scales.slice(1).some((b, i) => !(scales[i] < b))
}

function sigmasError(values: readonly number[], strictlyDecreasing: boolean): boolean {
  if (values.length === 0) return true
  if (values.some(v => !(v > 0 && v < 1))) return true
  return strictlyDecreasing && values.slice(1).some((b, i) => !(values[i] > b))
}

// ── 카드 값 정규화 — 파이썬 parse_settings(raw, base=APP_DEFAULTS) ─────────────────────────────────────
function text(value: unknown): string {
  return String(value ?? '').trim()
}

function truthy(value: unknown): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  return ['1', 'true', 'yes', 'on'].includes(text(value).toLowerCase())
}

function finite(value: unknown): number | null {
  if (typeof value === 'boolean' || value === null || value === undefined) return null
  const number = typeof value === 'number' ? value : pyFloat(value)
  return number !== null && Number.isFinite(number) ? number : null
}

function choice<T extends string>(value: unknown, choices: readonly T[]): T | null {
  const key = text(value).toLowerCase()
  return (choices as readonly string[]).includes(key) ? key as T : null
}

/** 파이썬 SpeedSettings 기본값(APP_DEFAULTS). */
export const DEFAULT_SETTINGS: Readonly<SpeedSettings> = {
  enabled: false, mode: 'transition', threshold: 'neo_shift', preset: 'anima', scales: '0.5,1.0', delta: 0.01,
  sigma_divisor: 1.03, manual_sigmas: '0.7', adaptive: true, transform: 'dct', spectrum_A: 203.615097,
  spectrum_beta: 1.915461, seed: -1, hires: false, apply_img2img: false,
}

/** 원값(키 = SpeedKey) → 카드 값. 없거나 읽을 수 없는 칸(모르는 선택지·유한하지 않은 수)은 기본값, 텍스트 칸은 친 그대로
 *  (앞뒤 공백만 뗀다), 시드는 음수면 -1·소수는 버린다(파이썬 parse_settings). */
export function parseSettings(raw: Partial<Record<SpeedKey, unknown>> | null | undefined): SpeedSettings {
  const r = raw ?? {}
  const out: SpeedSettings = { ...DEFAULT_SETTINGS }
  for (const key of ['enabled', 'adaptive', 'hires', 'apply_img2img'] as const) {
    const value = r[key]
    if (key in r && value !== null && value !== undefined && value !== '') out[key] = truthy(value)
  }
  out.mode = choice(r.mode, MODES) ?? out.mode
  out.threshold = choice(r.threshold, THRESHOLDS) ?? out.threshold
  out.transform = choice(r.transform, TRANSFORMS) ?? out.transform
  out.preset = choice(r.preset, PRESET_NAMES) ?? out.preset
  for (const key of ['scales', 'manual_sigmas'] as const) {
    if (r[key] !== null && r[key] !== undefined) out[key] = text(r[key])
  }
  for (const key of ['delta', 'sigma_divisor', 'spectrum_A', 'spectrum_beta'] as const) {
    const number = finite(r[key])
    if (number !== null) out[key] = number
  }
  const seed = finite(r.seed)
  if (seed !== null) out.seed = seed >= 0 ? Math.trunc(seed) : -1
  return out
}

/** 위젯 스토어(키 = '_aspeed_*') → 카드 값. */
export function readSettings(widgets: Record<string, unknown> | null | undefined): SpeedSettings {
  const w = widgets ?? {}
  const raw: Partial<Record<SpeedKey, unknown>> = {}
  for (const key of Object.keys(WIDGET_IDS) as SpeedKey[]) {
    if (WIDGET_IDS[key] in w) raw[key] = w[WIDGET_IDS[key]]
  }
  return parseSettings(raw)
}

// ── 검사 — 파이썬 validate(settings) ──────────────────────────────────────────────────────────
/** 카드 설정의 오류 코드(없으면 빈 배열). 확장 coerce_settings 의 오류 → build_plan 이 설정만 보고 거절하는 것 순. */
export function validate(settings: SpeedSettings): ErrorCode[] {
  const scales = numbers(settings.scales)
  if ('error' in scales && scales.error === 'zero') return ['script_error']
  const manual = numbers(settings.manual_sigmas)          // threshold 와 상관없이 늘 읽힌다(확장 그대로)
  if ('error' in manual && manual.error === 'zero') return ['script_error']
  const codes: ErrorCode[] = []
  const scaleValues = 'values' in scales && !scalesError(scales.values) ? scales.values : null
  if (scaleValues === null) codes.push('scales')
  const manualValues = 'values' in manual && !sigmasError(manual.values, settings.mode === 'transition')
    ? manual.values : null
  if (settings.threshold === 'manual' && manualValues === null) codes.push('manual')
  if (settings.threshold !== 'manual') {
    if (!(settings.delta > 0 && settings.delta < 1)) codes.push('delta')
    if (settings.threshold === 'neo_shift' && !(settings.sigma_divisor > 0)) codes.push('divisor')
  }
  if (codes.length) return codes
  // build_plan 이 설정만 보고 거절하는 것(scaleValues 는 여기서 늘 있다)
  const ladder = scaleValues ?? [0.5, 1]
  const transitions = ladder.length - 1
  if (transitions < 1) return []
  if (settings.threshold === 'manual') {
    let values = manualValues ?? [0.7]
    if (settings.mode === 'respace' && values.length === 1 && transitions > 1) values = Array(transitions).fill(values[0])
    if (values.length !== transitions) codes.push('manual_count')
  } else {
    const presetA = PRESET_A[settings.preset]
    const A = presetA === null || presetA === undefined ? settings.spectrum_A : presetA
    if (!(settings.sigma_divisor > 0)) codes.push('divisor')
    else if (!(Number.isFinite(A) && A > 0)) codes.push('spectrum_A')
    else if (!Number.isFinite(settings.spectrum_beta)) codes.push('spectrum_beta')
  }
  if (!codes.length && settings.transform === 'dwt') {
    if (ladder.slice(1).some((b, i) => Math.abs(b / ladder[i] - 2) > 1e-6)) codes.push('dwt_ratio')
  }
  return codes
}

export function validationMessages(settings: SpeedSettings): string[] {
  return validate(settings).map(code => VALIDATION_MESSAGES[code])
}

// ── 표시 ─────────────────────────────────────────────────────────────────────────────────
export type VisibleField = Exclude<SpeedKey, 'enabled'>

/** 지금 보일 칸(파이썬 확장이 읽지 않는 칸은 숨긴다 — 값은 그대로 두고 보낸다). transform·seed 는 '세부값' 묶음. */
export function visibleFields(settings: SpeedSettings): VisibleField[] {
  const manual = settings.threshold === 'manual'
  const fields: VisibleField[] = ['mode', 'threshold']
  if (!manual) fields.push('preset')
  fields.push('scales')
  if (!manual) fields.push('delta')
  if (settings.threshold === 'neo_shift') fields.push('sigma_divisor')
  if (manual) fields.push('manual_sigmas')
  if (!manual) fields.push('adaptive')
  fields.push('hires', 'apply_img2img', 'transform')
  if (!manual && settings.preset === 'custom') fields.push('spectrum_A', 'spectrum_beta')
  fields.push('seed')
  return fields
}

function short(value: number): string {
  return String(Number(value.toPrecision(10)))
}

/** 카드 요약 배지 — 꺼짐: '', 오류: '⚠ 설정 오류', 그 밖: 'transition · neo_shift · 0.5→1' / 'respace · manual σ 0.7'. */
export function summary(settings: SpeedSettings): string {
  if (!settings.enabled) return ''
  if (validate(settings).length) return '⚠ 설정 오류'
  if (settings.threshold === 'manual') return `${settings.mode} · manual σ ${settings.manual_sigmas}`
  const parsed = numbers(settings.scales)
  const ladder = 'values' in parsed ? parsed.values.map(short).join('→') : settings.scales
  return `${settings.mode} · ${settings.threshold} · ${ladder}`
}

/** 기본값과 같은가("기본값" 버튼을 흐리게). 숫자는 값으로, 텍스트는 공백을 뗀 글자로 비교한다. */
export function isDefault(settings: SpeedSettings): boolean {
  return (Object.keys(DEFAULT_SETTINGS) as SpeedKey[]).every(key => settings[key] === DEFAULT_SETTINGS[key])
}

export const FORGE_NOTE = "Forge 확장 'Anima SPEED' 로 보냅니다"
export const OLD_FORGE_NOTE = '연결된 Forge 의 sam-extra 에 Anima SPEED 가 없어(0.31.0 이상 필요) 보내지 않습니다.'
export const COMFY_PENDING_NOTE = 'ComfyUI 는 앱 노드 팩 1.7.0 단계 뒤에 같은 계산으로 돌립니다 — 지금은 ComfyUI 로 가지 않습니다.'
export const KREA2_NOTE = 'Krea2 생성에는 보내지 않습니다.'

/** 카드 아래 백엔드 줄: Krea2 → 안 보냄, ComfyUI → 팩 1.7.0 단계 뒤(계획 R16), 확장 확인됐는데 없음 → 옛 Forge, 그 밖 → Forge. */
export function cardNote(caps: Pick<SamExtraCapabilitiesEvent, 'status' | 'known' | 'features'> | null | undefined,
  krea2 = false): string {
  if (krea2) return KREA2_NOTE
  if (caps?.status === 'not_applicable') return COMFY_PENDING_NOTE
  if (caps?.known && caps.features?.speed !== true) return OLD_FORGE_NOTE
  return FORGE_NOTE
}
