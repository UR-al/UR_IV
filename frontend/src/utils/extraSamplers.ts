import type { SamExtraCapabilitiesEvent } from '../types/bridge'
import { flag, passLabel } from './extraSchedulers'

/**
 * Extra Samplers ER SDE 칸(components/params/ExtraSamplersPanel.vue — 파라미터 카드의 Sampler/Scheduler 줄 아래)의 순수 로직.
 *
 * 파이썬 core/extra_samplers.py(위젯 키·앱 기본값·차수/η/잡음 구간 범위·샘플러 18개의 표기/ComfyUI 이름/종류·값을 읽는
 * 샘플러·패스 판정 위젯·값 정규화·요약 문구·옛 sam-extra 의 잡음 구간 판정)의 거울이다 — tests/test_extra_samplers_panel_mirror.py
 * 가 이 파일을 읽어 파이썬 표·확장 슬라이더(script-info 픽스처)와 대조하고, 이름 풀이·패스 판정·정규화·요약·구간 판정은 두
 * 구현이 같은 골든 파일(extraSamplers.cases.json — 파이썬이 만든 기대값)을 읽는다. 위젯 값은 파이썬 프록시와 같은 문자열(차수
 * '3', η '1', 구간 체크 '0'/'1', 시작 '0.2', 끝 '0.8')이다. 보낼지(스냅샷·패스·백엔드·게이트)는 파이썬이 정한다 — 여기서는
 * 칸을 보일지와 안내 문구만 계산한다.
 *
 * 확장(sam-extra 'Extra Samplers', v0.33.0)은 생성마다 값 다섯을 요청에 붙이고 이 확장의 ER SDE 셋만 그것을 읽는다(차수는
 * 셋 모두, η·잡음 구간은 Reverse-time·Tunable) — 그래서 칸은 다음 요청의 어느 패스(메인·Hires·ADetailer·SAM3)가 ER SDE 셋
 * 중 하나일 때만 보인다(한 생성에 값은 한 벌 — 여러 패스가 같은 값을 쓴다).
 */

export type XsampKey = 'max_stage' | 'eta' | 'noise_window' | 'noise_start' | 'noise_end'
export type XsampValues = Record<XsampKey, string>
/** 파이썬 sampler_kind — 칸·힌트 종류. '' = 그 밖의 이 확장 샘플러(4M SDE·flow ODE·IPNDM 등). */
export type SamplerKind = 'er_sde' | 'dy' | 'smea' | 'cfg_pp' | 'restart' | ''

export interface SamplerSettings {
  max_stage: number
  eta: number
  noise_window: boolean
  noise_start: number
  noise_end: number
}

export interface SamplerRow {
  label: string             // Forge 샘플러 표기(/sdapi/v1/samplers 이름 = 앱이 보내는 값)
  name: string              // ComfyUI 이름(확장의 첫 별칭)
  kind: SamplerKind
  alias?: string            // 확장의 둘째 별칭(reForge k_* — 그 셋만)
}

export interface PassUse {
  pass: string              // 'main' | 'hires' | 'adetailer_1' | 'adetailer_2' | 'sam3'
  sampler: string           // ER_SDE_LABELS 의 표기
}

/** 기능 스냅샷 scripts 의 키(소문자 스크립트 제목 — 파이썬 TITLE_EXTRA_SAMPLERS). */
export const SCRIPT_TITLE = 'extra samplers'
/** 앱이 늘 보내는 위치 인자 수(파이썬 len(ARG_NAMES) — 계획 X10). 스냅샷의 인자 수가 이보다 적으면 잡음 구간이 빠진다. */
export const ARG_COUNT = 5

/** 위젯 스토어 키 — 파이썬 core/extra_samplers.widget_id(key) = '_xsamp_' + 확장 인자 이름. */
export const WIDGET_IDS: Readonly<Record<XsampKey, string>> = {
  max_stage: '_xsamp_max_stage',
  eta: '_xsamp_eta',
  noise_window: '_xsamp_noise_window',
  noise_start: '_xsamp_noise_start',
  noise_end: '_xsamp_noise_end',
}

/** 앱 기본값 = 확장 기본값(파이썬 widget_values(APP_DEFAULTS)). "기본값" 버튼이 스토어에 쓴다. */
export const DEFAULTS: Readonly<XsampValues> = {
  max_stage: '3',
  eta: '1',
  noise_window: '0',
  noise_start: '0.2',
  noise_end: '0.8',
}

/** 같은 기본값을 값으로(파이썬 APP_DEFAULTS). */
export const DEFAULT_SETTINGS: Readonly<SamplerSettings> = {
  max_stage: 3,
  eta: 1,
  noise_window: false,
  noise_start: 0.2,
  noise_end: 0.8,
}

/** 입력 범위 — 확장 UI 슬라이더(파이썬 MAX_STAGE_* · ETA_* · NOISE_PERCENT_*, script-info 픽스처 0·1·3·4번). */
export const STAGE_RANGE = { min: 1, max: 3, step: 1 } as const
export const ETA_RANGE = { min: 0, max: 10, step: 0.01 } as const
export const NOISE_RANGE = { min: 0, max: 1, step: 0.01 } as const
/** 차수 버튼(1·2·3). */
export const STAGE_OPTIONS: readonly string[] = ['1', '2', '3']

/** 확장이 Forge 목록에 등록하는 샘플러 18개(파이썬 LABELS·ALIASES·sampler_kind — registry.SPECS 순서). */
export const SAMPLERS: readonly SamplerRow[] = [
  { label: 'ER SDE (Reverse-time)', name: 'er_sde_reverse_time', kind: 'er_sde' },
  { label: 'ER SDE (ODE)', name: 'er_sde_ode', kind: 'er_sde' },
  { label: 'ER SDE (Tunable)', name: 'er_sde_tunable', kind: 'er_sde' },
  { label: 'DPM++ 4M SDE', name: 'dpmpp_4m_sde', kind: '' },
  { label: 'Euler Dy CFG++', name: 'euler_dy_cfg_pp', kind: 'dy' },
  { label: 'Euler SMEA Dy CFG++', name: 'euler_smea_dy_cfg_pp', kind: 'smea' },
  { label: 'Euler Dy', name: 'euler_dy', kind: 'dy', alias: 'k_euler_dy' },
  { label: 'Euler SMEA Dy', name: 'euler_smea_dy', kind: 'smea', alias: 'k_euler_smea_dy' },
  { label: 'DPM++ 2M SDE Heun', name: 'dpmpp_2m_sde_heun', kind: '', alias: 'k_dpmpp_2m_sde_heun' },
  { label: 'DPM++ 2M (flow ODE)', name: 'dpmpp_2m_flow_ode', kind: '' },
  { label: 'DPM++ 2M Heun (flow ODE)', name: 'dpmpp_2m_heun_flow_ode', kind: '' },
  { label: 'DPM++ 3M (flow ODE)', name: 'dpmpp_3m_flow_ode', kind: '' },
  { label: 'UniPC bh2', name: 'uni_pc_bh2', kind: '' },
  { label: 'CFG++ UD10 AB', name: 'cfgpp_ud10_ab', kind: 'cfg_pp' },
  { label: 'IPNDM', name: 'ipndm', kind: '' },
  { label: 'IPNDM_V', name: 'ipndm_v', kind: '' },
  { label: 'DEIS', name: 'deis', kind: '' },
  { label: 'Restart (flow)', name: 'restart_flow', kind: 'restart' },
]

/** max stage 를 읽는 셋(파이썬 ER_SDE_LABELS). */
export const ER_SDE_LABELS: readonly string[] = ['ER SDE (Reverse-time)', 'ER SDE (ODE)', 'ER SDE (Tunable)']
/** η 와 잡음 구간을 읽는 둘(파이썬 ETA_LABELS = WINDOW_LABELS). */
export const ETA_LABELS: readonly string[] = ['ER SDE (Reverse-time)', 'ER SDE (Tunable)']
/** CFG 1~2 권장(파이썬 CFG_PP_LABELS). */
export const CFG_PP_LABELS: readonly string[] = ['Euler Dy CFG++', 'Euler SMEA Dy CFG++', 'CFG++ UD10 AB']
/** 잡음 없는 DPM-Solver++ 셋(파이썬 FLOW_ODE_LABELS). */
export const FLOW_ODE_LABELS: readonly string[] = ['DPM++ 2M (flow ODE)', 'DPM++ 2M Heun (flow ODE)', 'DPM++ 3M (flow ODE)']

/** Forge 의 '같은 샘플러'(파이썬 USE_SAME_SAMPLER). */
export const USE_SAME_SAMPLER = 'Use same sampler'

/** 패스 판정에 쓰는 위젯 스토어 키 — 파이썬 MAIN_SAMPLER_WIDGET·…·ADETAILER_SLOT_WIDGETS·SAM3_*(pass_widget_ids 순서). */
export const PASS_WIDGETS = {
  main: 'sampler_combo',
  hiresGroup: 'hires_options_group',
  hiresSampler: 'hires_sampler_combo',
  adetailerGroup: 'adetailer_group',
  adetailerSlots: [
    { index: 1, enabled: 'ad_slot1_group', useSampler: '_ad_s1_use_sampler', sampler: '_ad_s1_sampler' },
    { index: 2, enabled: 'ad_slot2_group', useSampler: '_ad_s2_use_sampler', sampler: '_ad_s2_sampler' },
  ],
  sam3Group: 'sam3_group',
  sam3Mode: '_sam3_mode',
  sam3UseSampler: '_sam3_use_sampler',
  sam3Sampler: '_sam3_sampler',
} as const

/** CFG 칸(파라미터 카드) — CFG++ 경고가 본다. */
export const CFG_WIDGET = 'cfg_input'
/** Hires CFG 칸(Hires 카드 'CFG (0=off)') — 0 이하·빈 값이면 앱이 hr_cfg 를 보내지 않는다(ui/generator_generation). */
export const HIRES_CFG_WIDGET = 'hires_cfg_input'
/** hr_cfg 가 없을 때 Forge 하이레스 패스의 CFG(파이썬 core/sam_extra_notices.FORGE_DEFAULT_HR_CFG — processing.py
 *  hr_cfg = 1.0). ComfyUI 는 메인 CFG 다(컴파일러 hr_cfg → cfg_scale). */
export const FORGE_DEFAULT_HR_CFG = 1

const MASK_ONLY = 'mask only'   // core/sam3_args.MODE_MASK_ONLY 를 그 판정(strip·소문자)대로

function folded(text: unknown): string {
  return String(text ?? '').split(/\s+/).filter(Boolean).join(' ').toLowerCase()
}

const BY_FOLDED: ReadonlyMap<string, string> = new Map(
  SAMPLERS.flatMap(row => [row.label, row.name, ...(row.alias ? [row.alias] : [])]
    .map(text => [folded(text), row.label] as const)),
)
const ROW_BY_LABEL: ReadonlyMap<string, SamplerRow> = new Map(SAMPLERS.map(row => [row.label, row] as const))

/** 샘플러 표기·ComfyUI 이름·reForge 별칭(대소문자·앞뒤·겹친 공백 무시) → 이 확장의 Forge 표기. 'Use same sampler'·다른
 *  샘플러·빈 값은 null(파이썬 resolve_sampler). */
export function resolveSampler(text: unknown): string | null {
  return BY_FOLDED.get(folded(text)) ?? null
}

/** 칸·힌트 종류(파이썬 sampler_kind) — 이 확장 샘플러가 아니면 null. */
export function samplerKind(text: unknown): SamplerKind | null {
  const label = resolveSampler(text)
  return label === null ? null : ROW_BY_LABEL.get(label)!.kind
}

export function isCfgPp(text: unknown): boolean {
  return CFG_PP_LABELS.includes(resolveSampler(text) ?? '')
}

function usesSame(text: unknown): boolean {
  const value = folded(text)
  return !value || value === USE_SAME_SAMPLER.toLowerCase()
}

/** 다음 T2I 요청에서 ER SDE 셋을 쓰는 패스(파이썬 passes_from_widgets — 같은 골든 파일로 묶인다). */
export function passesUsing(widgets: Record<string, unknown>): PassUse[] {
  const w = widgets ?? {}
  const main = w[PASS_WIDGETS.main]
  const entries: Array<[string, unknown]> = [['main', main]]
  if (flag(w[PASS_WIDGETS.hiresGroup])) {
    const hires = w[PASS_WIDGETS.hiresSampler]
    entries.push(['hires', usesSame(hires) ? main : hires])
  }
  if (flag(w[PASS_WIDGETS.adetailerGroup])) {
    for (const slot of PASS_WIDGETS.adetailerSlots) {
      if (!flag(w[slot.enabled])) continue
      const text = flag(w[slot.useSampler]) ? w[slot.sampler] : main
      entries.push([`adetailer_${slot.index}`, usesSame(text) ? main : text])
    }
  }
  if (flag(w[PASS_WIDGETS.sam3Group]) && String(w[PASS_WIDGETS.sam3Mode] ?? '').trim().toLowerCase() !== MASK_ONLY) {
    const text = flag(w[PASS_WIDGETS.sam3UseSampler]) ? w[PASS_WIDGETS.sam3Sampler] : main
    entries.push(['sam3', usesSame(text) ? main : text])
  }
  const out: PassUse[] = []
  for (const [pass, text] of entries) {
    const label = resolveSampler(text)
    if (label !== null && ER_SDE_LABELS.includes(label)) out.push({ pass, sampler: label })
  }
  return out
}

/** η·잡음 구간을 읽는 패스가 있나(Reverse-time·Tunable). ER SDE (ODE) 만 쓰면 거짓 — 그 칸을 끈다. */
export function usesEta(passes: readonly PassUse[]): boolean {
  return passes.some(p => ETA_LABELS.includes(p.sampler))
}

/** 칸 부제 — '사용: 메인 · Hires'. 패스가 없으면 ''. */
export function subtitle(passes: readonly PassUse[]): string {
  const labels = [...new Set(passes.map(p => passLabel(p.pass)))]
  return labels.length ? `사용: ${labels.join(' · ')}` : ''
}

/** 스토어 → 칸 값(없거나 null 이면 기본값, 그 밖은 글자 그대로 — 정규화는 parseValues). */
export function readValues(widgets: Record<string, unknown>): XsampValues {
  const w = widgets ?? {}
  const out = { ...DEFAULTS }
  for (const key of Object.keys(DEFAULTS) as XsampKey[]) {
    const value = w[WIDGET_IDS[key]]
    if (value !== undefined && value !== null) out[key] = String(value)
  }
  return out
}

const DECIMAL = /^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/i
const TRUE_TEXTS = ['true', 'on', 'yes', '1']

/** 파이썬 float(value) 가 유한한 수로 읽는 글자인가 → 그 수, 아니면 null(파이썬 _readable_number). */
function readable(text: string): number | null {
  const trimmed = String(text ?? '').trim()
  const value = DECIMAL.test(trimmed) ? Number(trimmed) : NaN
  return Number.isFinite(value) ? value : null
}

/** 파이썬 round() — 한가운데는 짝수 쪽(2.5 → 2). */
function roundHalfEven(value: number): number {
  const lower = Math.floor(value)
  const diff = value - lower
  if (diff < 0.5) return lower
  if (diff > 0.5) return lower + 1
  return lower % 2 === 0 ? lower : lower + 1
}

function clamp(value: number, range: { readonly min: number; readonly max: number }): number {
  return Math.min(range.max, Math.max(range.min, value))
}

/** 칸 글자 → 파이썬 parse_settings 가 보낼 값(읽을 수 없거나 유한하지 않으면 기본값, 범위 밖은 자른다, 차수는 반올림 —
 *  확장 coerce_*; 구간 체크는 true/on/yes/1 만 켬). */
export function parseValues(values: XsampValues): SamplerSettings {
  const stage = readable(values.max_stage)
  const eta = readable(values.eta)
  const start = readable(values.noise_start)
  const end = readable(values.noise_end)
  return {
    max_stage: stage === null ? DEFAULT_SETTINGS.max_stage : clamp(roundHalfEven(stage), STAGE_RANGE),
    eta: eta === null ? DEFAULT_SETTINGS.eta : clamp(eta, ETA_RANGE),
    noise_window: TRUE_TEXTS.includes(String(values.noise_window ?? '').trim().toLowerCase()),
    noise_start: start === null ? DEFAULT_SETTINGS.noise_start : clamp(start, NOISE_RANGE),
    noise_end: end === null ? DEFAULT_SETTINGS.noise_end : clamp(end, NOISE_RANGE),
  }
}

/** 기본값과 같은가(숫자 표기 무시 — '1.0' = '1'; 구간 체크는 켬/끔으로). "기본값" 버튼을 흐리게 할 때. */
export function isDefault(values: XsampValues): boolean {
  return Number(values.max_stage) === DEFAULT_SETTINGS.max_stage
    && Number(values.eta) === DEFAULT_SETTINGS.eta
    && parseValues(values).noise_window === DEFAULT_SETTINGS.noise_window
    && Number(values.noise_start) === DEFAULT_SETTINGS.noise_start
    && Number(values.noise_end) === DEFAULT_SETTINGS.noise_end
}

/** 구간을 켰지만 시작 ≥ 끝 — 어느 스텝에도 잡음을 넣지 않는다(파이썬 ExtraSamplersSettings.window_empty). */
export function windowEmpty(settings: SamplerSettings): boolean {
  return settings.noise_window && settings.noise_start >= settings.noise_end
}

/** 위젯·요약 숫자 — 파이썬 widget_number(repr, 끝의 '.0' 없음, 0 → '0', 1e-05 처럼 지수는 두 자리). */
export function widgetNumber(value: number): string {
  if (value === 0) return '0'
  const size = Math.abs(value)
  if (size < 1e-4 || size >= 1e16) {
    const [mantissa, exponent = '0'] = value.toExponential().split('e')
    const sign = exponent.startsWith('-') ? '-' : '+'
    return `${mantissa}e${sign}${exponent.replace(/^[+-]/, '').padStart(2, '0')}`
  }
  return String(value)
}

/** 잡음 구간 요약 — '잡음 0.2–0.8'(빈 구간이면 ' (빈 구간)'), 끔이면 ''. 파이썬 describe 의 그 부분. */
export function windowSummary(settings: SamplerSettings): string {
  if (!settings.noise_window) return ''
  const window = `잡음 ${widgetNumber(settings.noise_start)}–${widgetNumber(settings.noise_end)}`
  return windowEmpty(settings) ? `${window} (빈 구간)` : window
}

/** 한 줄 요약 — 'ER SDE 차수 2 · η 0.5 · 잡음 0.2–0.8'(파이썬 describe). ``samplers``(표기·ComfyUI 이름)를 주면 그 샘플러가
 *  읽는 칸만(ODE 는 차수만, ER SDE 가 없으면 ''), 없으면 모두. */
export function describe(settings: SamplerSettings, samplers?: readonly unknown[]): string {
  const labels = samplers === undefined ? null : new Set(samplers.map(resolveSampler))
  if (labels !== null && !ER_SDE_LABELS.some(label => labels.has(label))) return ''
  const parts = [`ER SDE 차수 ${settings.max_stage}`]
  if (labels === null || ETA_LABELS.some(label => labels.has(label))) {
    parts.push(`η ${widgetNumber(settings.eta)}`)
    const window = windowSummary(settings)
    if (window) parts.push(window)
  }
  return parts.join(' · ')
}

/** 스냅샷에서 구간 판정이 읽는 것만(SamExtraCapabilitiesEvent·그 readonly 판 모두 맞는다). */
interface Caps {
  readonly known?: boolean
  readonly scripts?: { readonly [title: string]: { readonly present?: boolean; readonly live_argc?: number | null } | undefined }
}

/** 연결된 Forge 의 sam-extra 가 잡음 구간(인자 3~5)을 받나 — 스냅샷이 본 인자 수가 다섯 이상이면 true, 적으면 false
 *  (0.31·0.32), 모르면 null(파이썬 window_supported — Forge API 는 스크립트가 가진 수만큼만 넘긴다). */
export function windowSupported(caps: Caps | null | undefined): boolean | null {
  if (caps?.known !== true) return null
  const detail = caps.scripts?.[SCRIPT_TITLE]
  if (!detail || detail.present !== true) return null
  const argc = detail.live_argc
  if (typeof argc !== 'number' || !Number.isInteger(argc)) return null
  return argc >= ARG_COUNT
}

/** 구간을 켰는데 연결된 sam-extra 가 그 인자를 버린다(확실할 때만) — 파이썬 window_unsupported(계획 X10). η 0 이면 거짓 —
 *  구간은 어느 판에서도 아무 일도 하지 않는다(확장 _er_sde_window: η ≤ 0 이면 구간 없음). */
export function windowUnsupported(settings: SamplerSettings, caps: Caps | null | undefined): boolean {
  return settings.noise_window && settings.eta > 0 && windowSupported(caps) === false
}

/** 메인·Hires 의 샘플러(이 확장 것만, 표기) — 힌트가 본다. ADetailer·SAM3 는 보지 않는다(작은 영역). */
function mainAndHires(widgets: Record<string, unknown>): string[] {
  const w = widgets ?? {}
  const labels = [resolveSampler(w[PASS_WIDGETS.main])]
  if (flag(w[PASS_WIDGETS.hiresGroup]) && !usesSame(w[PASS_WIDGETS.hiresSampler])) {
    labels.push(resolveSampler(w[PASS_WIDGETS.hiresSampler]))
  }
  return labels.filter((label): label is string => label !== null)
}

// 한 줄 힌트(계획 PLAN_amend_v033 §5·design_samplers §3.2, 확장 HELP 문구)
export const TUNABLE_HINT = 'ER SDE (Tunable): Forge 내장 ER SDE 와 같은 풀이·잡음 척도에 η(잡음 세기)와 잡음 구간을 더한 것 — '
  + 'η 1·구간 끔이면 내장 ER SDE 와 같습니다.'
export const DY_CFGPP_HINT = 'Euler Dy CFG++: 2·3번째 스텝에 절반 해상도로 모델을 한 번 더 부릅니다. CFG++ 샘플러 — CFG 1~2 를 '
  + '권장합니다.'
export const SMEA_CFGPP_HINT = 'Euler SMEA Dy CFG++: 0번째 스텝(×1.25 해상도 — VRAM 을 더 씁니다)·1번째 스텝에 모델을 한 번 더 '
  + '부릅니다. CFG++ 샘플러 — CFG 1~2 를 권장합니다.'
export const DY_PLAIN_HINT = 'Euler Dy · Euler SMEA Dy: CFG++ 판과 같은 보조 스텝, 보통 CFG 로 씁니다. churn 0(기본)이면 다시 '
  + '잡음을 넣지 않습니다.'
export const UD10_HINT = 'CFG++ UD10 AB: ComfyUI 가 Anima 용으로 만든 CFG++ 샘플러 — CFG 2 권장(1~2). CFG 1 이면 이력을 섞은 '
  + 'Euler 와 같습니다.'
export const FLOW_ODE_HINT = 'DPM++ 2M · 2M Heun · 3M (flow ODE): 잡음 없는 DPM-Solver++ — flow 의 λ 좌표에서 풉니다(Res '
  + 'Multistep 의 flow 판 비교용). Eta 를 쓰지 않습니다.'
/** 10-05 Anima 확인(GPU part 2 O2)에서 본 것만 — 권장하지 않는다(다른 스케줄·모델은 확인하지 않았다). */
export const IPNDM_HINT = 'IPNDM · IPNDM_V · UniPC bh2: 10-05 Anima 확인(Linear Quadratic, 28 스텝)에서 IPNDM 은 정상이었고 '
  + 'IPNDM_V(녹색 잡음)·UniPC bh2(무너짐)는 깨졌습니다 — 다른 스케줄은 확인하지 않았습니다.'
/** 그 힌트를 보이는 샘플러(파이썬 LABEL_IPNDM·LABEL_IPNDM_V·LABEL_UNIPC_BH2). */
export const IPNDM_HINT_LABELS: readonly string[] = ['IPNDM', 'IPNDM_V', 'UniPC bh2']
export const RESTART_HINT = 'Restart (flow): 스케줄을 유지한 채 σ 0.09~0.67 구간을 한 번(36 스텝부터 두 번) 다시 밟습니다 — 모델 '
  + '호출이 Heun 의 약 1.4배(28 스텝).'
/** CFG++ 샘플러인데 CFG 가 2 를 넘을 때(Forge 도 'CFG between 1.0 ~ 2.0 is recommended' 를 로그에 남긴다). */
export const CFG_PP_MAX = 2
export function cfgWarning(cfg: number): string {
  return `CFG++ 샘플러는 CFG 1~2 를 권장합니다 (지금 ${widgetNumber(cfg)})`
}
/** Hires 패스의 CFG++ 샘플러 — 그 패스의 CFG(Hires CFG, 끔이면 Forge 1·ComfyUI 메인 CFG)로 본다. */
export function hiresCfgWarning(cfg: number): string {
  return `Hires 의 CFG++ 샘플러는 CFG 1~2 를 권장합니다 (Hires 패스 CFG 지금 ${widgetNumber(cfg)})`
}

/** 메인·Hires 패스의 CFG++ 경고 — 각 패스의 샘플러를 그 패스가 실제로 쓰는 CFG 와 대조한다. 메인은 CFG 칸, Hires 는
 *  Hires CFG 칸(> 0 일 때만 앱이 hr_cfg 를 보낸다), 끔이면 Forge 기본 1(``FORGE_DEFAULT_HR_CFG``)·ComfyUI 는 메인 CFG.
 *  Hires 샘플러가 'Use same' 이면 메인 샘플러다. */
function cfgWarnings(widgets: Record<string, unknown>, comfy: boolean): string[] {
  const w = widgets ?? {}
  const cfg = readable(String(w[CFG_WIDGET] ?? ''))
  const out: string[] = []
  const over = (value: number | null): value is number => value !== null && value > CFG_PP_MAX
  if (isCfgPp(w[PASS_WIDGETS.main]) && over(cfg)) out.push(cfgWarning(cfg))
  if (flag(w[PASS_WIDGETS.hiresGroup])) {
    const hiresSampler = usesSame(w[PASS_WIDGETS.hiresSampler]) ? w[PASS_WIDGETS.main] : w[PASS_WIDGETS.hiresSampler]
    const hiresCfg = readable(String(w[HIRES_CFG_WIDGET] ?? ''))
    const effective = hiresCfg !== null && hiresCfg > 0 ? hiresCfg : (comfy ? cfg : FORGE_DEFAULT_HR_CFG)
    if (isCfgPp(hiresSampler) && over(effective)) out.push(hiresCfgWarning(effective))
  }
  return out
}

/** Sampler 줄 아래 힌트 — v0.33.0 샘플러 설명과 CFG++ 의 CFG 경고(패스마다 그 패스의 CFG). ``comfy``: ComfyUI 로 보낸다
 *  (Hires CFG 가 끔이면 메인 CFG 로 돈다). 해당 없으면 빈 배열. */
export function samplerHints(widgets: Record<string, unknown>, comfy = false): string[] {
  const labels = mainAndHires(widgets)
  const has = (label: string) => labels.includes(label)
  const hints: string[] = []
  if (has('ER SDE (Tunable)')) hints.push(TUNABLE_HINT)
  if (has('Euler Dy CFG++')) hints.push(DY_CFGPP_HINT)
  if (has('Euler SMEA Dy CFG++')) hints.push(SMEA_CFGPP_HINT)
  if (has('Euler Dy') || has('Euler SMEA Dy')) hints.push(DY_PLAIN_HINT)
  if (has('CFG++ UD10 AB')) hints.push(UD10_HINT)
  if (labels.some(label => FLOW_ODE_LABELS.includes(label))) hints.push(FLOW_ODE_HINT)
  if (labels.some(label => IPNDM_HINT_LABELS.includes(label))) hints.push(IPNDM_HINT)
  if (has('Restart (flow)')) hints.push(RESTART_HINT)
  hints.push(...cfgWarnings(widgets, comfy))
  return hints
}

// 칸 안 문구(design_samplers §3.2, 확장 HELP)
export const STAGE_HELP = '1 = 지수 Euler, 2·3 = 다단계 보정(기본 3). Forge 내장 ER SDE 는 이 값을 쓰지 않습니다.'
export const ETA_HELP = 'η 는 잡음 척도의 지수 — 1(기본), 0 = ODE 와 같음, 클수록 스텝마다 잡음이 늘고 너무 크면 결과가 깨질 수 '
  + '있습니다. Reverse-time h(λ)=λ^(η+1)(η 1 = 고전 역시간 SDE), Tunable h(λ)=λ·(e^(λ^0.3)+10)^η.'
export const ODE_NOTE = 'ER SDE (ODE) 는 잡음을 넣지 않아 η·잡음 구간을 쓰지 않습니다.'
export const WINDOW_HELP = '켜면 시작~끝(샘플링 진행률 0–1, 기본 0.2–0.8 — Anima shift 3 에서 σ 0.92~0.43)에서만 SDE 잡음을 넣고 '
  + '나머지 스텝은 ODE 로 풉니다(Reverse-time·Tunable).'
export const WINDOW_EMPTY_WARNING = '시작 ≥ 끝 — 잡음을 넣지 않습니다'
/** 파이썬 WINDOW_UNSUPPORTED_TEXT(계획 X10). */
export const WINDOW_UNSUPPORTED_TEXT = '잡음 구간은 sam-extra 0.33.0 이상에서만 적용됩니다'
export const FORGE_PARAMS_NOTE = 'Forge 설정의 Sampler Parameters(sigma noise·Eta·churn)는 Forge 값을 따릅니다(앱은 보내지 '
  + '않습니다).'
/** 칸이 안 보이는데 값이 기본값이 아닐 때(값은 그대로 남는다 — 다음에 ER SDE 를 고르면 그 값으로). */
export function hiddenNote(settings: SamplerSettings): string {
  return `ER SDE 값(${describe(settings).replace(/^ER SDE /, '')})은 ER SDE (Reverse-time)·(ODE)·(Tunable) 을 고를 때만 쓰입니다.`
}

export const OLD_FORGE_NOTE = '연결된 Forge 의 sam-extra 에 Extra Samplers 가 없어(0.31.0 이상 필요) 이 값은 전달되지 않습니다.'
/** ComfyUI 일 때 FORGE_PARAMS_NOTE 자리(계획 C6 — 값은 늘 보내고 컴파일러가 기본값과 다를 때만 노드를 넣는다, D6). */
export const COMFY_PARAMS_NOTE = 'ComfyUI 는 앱 노드 팩 1.7.0 의 같은 계산으로 돌립니다(값이 기본값과 다를 때만 노드로 '
  + '보냅니다). Forge 설정의 Sampler Parameters(sigma noise·Eta·churn)는 ComfyUI 에 없어 기본값으로 돕니다.'
export const KREA2_NOTE = 'Krea2 생성에는 보내지 않습니다.'

/** 칸 아래 경고 한 줄(없으면 ''): Krea2 → 안 보냄, 확장 확인됐는데 없음 → 옛 Forge. ComfyUI 는 경고가 아니다(paramsNote). */
export function cardNote(caps: Pick<SamExtraCapabilitiesEvent, 'status' | 'known' | 'features'> | null | undefined,
  krea2 = false): string {
  if (krea2) return KREA2_NOTE
  if (caps?.known && caps.features?.extra_samplers !== true) return OLD_FORGE_NOTE
  return ''
}

/** 경고가 없을 때의 안내 줄 — Forge 는 FORGE_PARAMS_NOTE, ComfyUI(스냅샷 'not_applicable')는 COMFY_PARAMS_NOTE. */
export function paramsNote(caps: Pick<SamExtraCapabilitiesEvent, 'status'> | null | undefined): string {
  return caps?.status === 'not_applicable' ? COMFY_PARAMS_NOTE : FORGE_PARAMS_NOTE
}
