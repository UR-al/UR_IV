import type { SamExtraCapabilitiesEvent } from '../types/bridge'

/**
 * Extra Schedulers 값 칸(components/params/ExtraSchedulerFields.vue — 파라미터 카드의 Sampler/Scheduler 줄 아래)의 순수 로직.
 *
 * 파이썬 core/extra_schedulers.py(위젯 키·앱 기본값·μ/β 범위·스케줄러 이름/라벨/README 별칭·값을 읽는 스케줄러·패스 판정
 * 위젯)의 거울이다 — tests/test_extra_schedulers_card_mirror.py 가 이 파일을 읽어 파이썬 표와 대조하고, 패스 판정은
 * 두 구현이 같은 골든 파일(extraSchedulers.cases.json)을 읽는다. 위젯 값은 파이썬 프록시와 같은 문자열(모드 키
 * 'expression'·'sigmas', 식·목록 그대로, 숫자 '0'·'0.5')이다. 보낼지(스냅샷·패스·백엔드·게이트)는 파이썬이 정한다 —
 * 여기서는 칸을 보일지와 안내 문구만 계산한다.
 *
 * 확장(sam-extra 'Extra Schedulers (sam-extra)')은 생성마다 값 다섯을 전역 상태에 두고 이 확장의 custom·Laplace 만 그것을
 * 읽는다 — 그래서 칸은 다음 요청의 어느 패스(메인·Hires·ADetailer·SAM3)가 laplace·custom 일 때만 보인다(한 생성에 값은
 * 한 벌 — 여러 패스가 같은 값을 쓴다).
 */

export type XschedKey = 'custom_mode' | 'custom_expression' | 'custom_sigmas' | 'laplace_mu' | 'laplace_beta'
export type XschedValues = Record<XschedKey, string>
export type SchedulerName = 'cosine' | 'cosine_exponential' | 'phi' | 'laplace' | 'karras_dynamic' | 'custom'
export type ValueScheduler = 'laplace' | 'custom'

export interface PassUse {
  pass: string              // 'main' | 'hires' | 'adetailer_1' | 'adetailer_2' | 'sam3'
  scheduler: ValueScheduler
}

export interface SchedulerRow {
  name: SchedulerName       // Forge API·콤보 값 = ComfyUI 팩 등록 이름
  label: string             // infotext 'Schedule type'
  alias: string             // aoleg/Neo_ExtraSchedulers README 표기(소문자)
}

/** 위젯 스토어 키 — 파이썬 core/extra_schedulers.widget_id(key) = '_xsched_' + 확장 인자 이름. */
export const WIDGET_IDS: Readonly<Record<XschedKey, string>> = {
  custom_mode: '_xsched_custom_mode',
  custom_expression: '_xsched_custom_expression',
  custom_sigmas: '_xsched_custom_sigmas',
  laplace_mu: '_xsched_laplace_mu',
  laplace_beta: '_xsched_laplace_beta',
}

/** 앱 기본값 = 확장 기본값(파이썬 widget_values(APP_DEFAULTS)). "기본값" 버튼이 스토어에 쓴다. */
export const DEFAULTS: Readonly<XschedValues> = {
  custom_mode: 'expression',
  custom_expression: 'M * (m / M) ** x',
  custom_sigmas: '[1.0, 0.6, 0.25, 0.1, 0.0]',
  laplace_mu: '0',
  laplace_beta: '0.5',
}

/** μ·β 입력 범위 — 확장 UI 슬라이더(파이썬 LAPLACE_MU_* · LAPLACE_BETA_*, ComfyUI LaplaceScheduler 와 같다). */
export const MU_RANGE = { min: -10, max: 10, step: 0.1 } as const
export const BETA_RANGE = { min: 0, max: 10, step: 0.1 } as const

/** custom 방식 — 키를 저장한다(파이썬 MODES 순서). 라벨은 앱 말(확장 라디오는 'Expression'·'Sigma list'). */
export const MODE_OPTIONS: readonly { key: 'expression' | 'sigmas'; label: string }[] = [
  { key: 'expression', label: '식' },
  { key: 'sigmas', label: '시그마 목록' },
]

/** 확장이 Forge 목록에 등록하는 스케줄러 6개(파이썬 SCHEDULER_NAMES·SCHEDULER_LABELS·README_LABELS). */
export const SCHEDULERS: readonly SchedulerRow[] = [
  { name: 'cosine', label: 'Cosine', alias: 'cosine' },
  { name: 'cosine_exponential', label: 'CosineExponential blend', alias: 'cosine-exponential blend' },
  { name: 'phi', label: 'Phi', alias: 'phi' },
  { name: 'laplace', label: 'Laplace', alias: 'laplace' },
  { name: 'karras_dynamic', label: 'Karras Dynamic', alias: 'karras dynamic' },
  { name: 'custom', label: 'custom', alias: 'custom' },
]

/** 이 칸의 값을 읽는 스케줄러(파이썬 VALUE_SCHEDULERS). */
export const VALUE_SCHEDULERS: readonly ValueScheduler[] = ['laplace', 'custom']
/** flow 모델(Anima 등)에서 뿌옇게 나오는 SD·SDXL 용(파이썬 FLOW_UNSUITABLE) — 힌트만, 막지 않는다. */
export const FLOW_UNSUITABLE: readonly SchedulerName[] = ['cosine', 'cosine_exponential', 'phi', 'karras_dynamic']

/** Forge 의 '같은 스케줄러'(파이썬 USE_SAME_SCHEDULER). */
export const USE_SAME_SCHEDULER = 'Use same scheduler'

/** 식·목록 오류를 칸 아래에 보일 위젯 속성(파이썬 계획 C7 이 pushWidgetProperty 로 보낸다). */
export const ERROR_PROPERTY = 'error'

/** 패스 판정에 쓰는 위젯 스토어 키 — 파이썬 MAIN_SCHEDULER_WIDGET·…·ADETAILER_SLOT_WIDGETS·SAM3_*(pass_widget_ids 순서). */
export const PASS_WIDGETS = {
  main: 'scheduler_combo',
  hiresGroup: 'hires_options_group',
  hiresScheduler: 'hires_scheduler_combo',
  adetailerGroup: 'adetailer_group',
  adetailerSlots: [
    { index: 1, enabled: 'ad_slot1_group', useSampler: '_ad_s1_use_sampler', scheduler: '_ad_s1_scheduler' },
    { index: 2, enabled: 'ad_slot2_group', useSampler: '_ad_s2_use_sampler', scheduler: '_ad_s2_scheduler' },
  ],
  sam3Group: 'sam3_group',
  sam3Mode: '_sam3_mode',
  sam3UseScheduler: '_sam3_use_scheduler',
  sam3Scheduler: '_sam3_scheduler',
} as const

const MASK_ONLY = 'mask only'   // core/sam3_args.MODE_MASK_ONLY 를 그 판정(strip·소문자)대로

function folded(text: unknown): string {
  return String(text ?? '').split(/\s+/).filter(Boolean).join(' ').toLowerCase()
}

const BY_FOLDED: ReadonlyMap<string, SchedulerName> = new Map(
  SCHEDULERS.flatMap(row => [row.name, row.label, row.alias].map(text => [folded(text), row.name] as const)),
)

/** 스케줄러 이름·라벨·README 별칭(대소문자·공백 무시) → 이 확장의 이름. 'Use same'·다른 스케줄러·빈 값은 null
 *  (파이썬 resolve_scheduler). */
export function resolveScheduler(text: unknown): SchedulerName | null {
  return BY_FOLDED.get(folded(text)) ?? null
}

/** 위젯 체크 값('true'/'false' 문자열, bool, 숫자) — 파이썬 _truthy(기본 false). */
export function flag(value: unknown): boolean {
  if (typeof value === 'boolean') return value
  if (typeof value === 'number') return value !== 0
  return ['1', 'true', 'yes', 'on'].includes(String(value ?? '').trim().toLowerCase())
}

function usesSame(text: unknown): boolean {
  const value = folded(text)
  return !value || value === USE_SAME_SCHEDULER.toLowerCase()
}

/** 다음 T2I 요청에서 laplace·custom 을 쓰는 패스(파이썬 passes_from_widgets — 같은 골든 파일로 묶인다). */
export function passesUsing(widgets: Record<string, unknown>): PassUse[] {
  const w = widgets ?? {}
  const main = w[PASS_WIDGETS.main]
  const entries: Array<[string, unknown]> = [['main', main]]
  if (flag(w[PASS_WIDGETS.hiresGroup])) {
    const hires = w[PASS_WIDGETS.hiresScheduler]
    entries.push(['hires', usesSame(hires) ? main : hires])
  }
  if (flag(w[PASS_WIDGETS.adetailerGroup])) {
    for (const slot of PASS_WIDGETS.adetailerSlots) {
      if (!flag(w[slot.enabled])) continue
      const text = flag(w[slot.useSampler]) ? w[slot.scheduler] : main
      entries.push([`adetailer_${slot.index}`, usesSame(text) ? main : text])
    }
  }
  if (flag(w[PASS_WIDGETS.sam3Group]) && String(w[PASS_WIDGETS.sam3Mode] ?? '').trim().toLowerCase() !== MASK_ONLY) {
    const text = flag(w[PASS_WIDGETS.sam3UseScheduler]) ? w[PASS_WIDGETS.sam3Scheduler] : main
    entries.push(['sam3', usesSame(text) ? main : text])
  }
  const out: PassUse[] = []
  for (const [pass, text] of entries) {
    const name = resolveScheduler(text)
    if (name === 'laplace' || name === 'custom') out.push({ pass, scheduler: name })
  }
  return out
}

/** 패스 이름 — 메인 · Hires · ADetailer 1 · SAM3 (파이썬 pass_label). */
export function passLabel(pass: string): string {
  if (pass === 'main') return '메인'
  if (pass === 'hires') return 'Hires'
  if (pass === 'sam3') return 'SAM3'
  if (pass.startsWith('adetailer_')) return `ADetailer ${pass.slice('adetailer_'.length)}`
  return pass
}

/** 칸 부제 — '사용: 메인 · Hires'. 패스가 없으면 ''. */
export function subtitle(passes: readonly PassUse[]): string {
  const labels = [...new Set(passes.map(p => passLabel(p.pass)))]
  return labels.length ? `사용: ${labels.join(' · ')}` : ''
}

export function usesLaplace(passes: readonly PassUse[]): boolean {
  return passes.some(p => p.scheduler === 'laplace')
}

export function usesCustom(passes: readonly PassUse[]): boolean {
  return passes.some(p => p.scheduler === 'custom')
}

/** 스토어 → 칸 값(없거나 빈 모드는 기본값; 식·목록은 그대로 — 빈 글자도 값이다, 파이썬 parse_settings). */
export function readValues(widgets: Record<string, unknown>): XschedValues {
  const w = widgets ?? {}
  const out = { ...DEFAULTS }
  for (const key of Object.keys(DEFAULTS) as XschedKey[]) {
    const value = w[WIDGET_IDS[key]]
    if (value !== undefined && value !== null) out[key] = String(value)
  }
  out.custom_mode = ['sigmas', 'sigma list', 'sigma_list', 'list', 'sigma'].includes(out.custom_mode.trim().toLowerCase())
    ? 'sigmas' : 'expression'
  return out
}

/** 기본값과 같은가(공백·숫자 표기 무시 — '0.50' = '0.5'). "기본값" 버튼을 흐리게 할 때. */
export function isDefault(values: XschedValues): boolean {
  return values.custom_mode === DEFAULTS.custom_mode
    && values.custom_expression.trim() === DEFAULTS.custom_expression
    && values.custom_sigmas.trim() === DEFAULTS.custom_sigmas
    && Number(values.laplace_mu) === Number(DEFAULTS.laplace_mu)
    && Number(values.laplace_beta) === Number(DEFAULTS.laplace_beta)
}

/** 메인·Hires 의 스케줄러(이 확장 것만) — 힌트가 본다. ADetailer·SAM3 는 보지 않는다(작은 영역). */
function mainAndHires(widgets: Record<string, unknown>): SchedulerName[] {
  const w = widgets ?? {}
  const names = [resolveScheduler(w[PASS_WIDGETS.main])]
  if (flag(w[PASS_WIDGETS.hiresGroup]) && !usesSame(w[PASS_WIDGETS.hiresScheduler])) {
    names.push(resolveScheduler(w[PASS_WIDGETS.hiresScheduler]))
  }
  return names.filter((name): name is SchedulerName => name !== null)
}

export const FLOW_HINT = 'Cosine · CosineExponential blend · Phi · Karras Dynamic 은 SD·SDXL 용 — flow 모델(Anima 등)에서는 '
  + '뿌옇게 나옵니다.'
export const KARRAS_DYNAMIC_HINT = 'ρ 는 Forge 설정 → Sampler Parameters → rho(기본 7, Karras 와 공유)를 따릅니다. '
  + 'ComfyUI 는 7 고정.'

/** Scheduler 줄 아래 힌트 — flow 에 맞지 않는 스케줄러, Karras Dynamic 의 ρ. 해당 없으면 빈 배열. */
export function schedulerHints(widgets: Record<string, unknown>): string[] {
  const names = mainAndHires(widgets)
  const hints: string[] = []
  if (names.some(name => FLOW_UNSUITABLE.includes(name))) hints.push(FLOW_HINT)
  if (names.includes('karras_dynamic')) hints.push(KARRAS_DYNAMIC_HINT)
  return hints
}

export const OLD_FORGE_NOTE = '연결된 Forge 의 sam-extra 에 Extra Schedulers 가 없어(0.31.0 이상 필요) 이 값은 전달되지 않습니다.'
export const COMFY_PENDING_NOTE = 'ComfyUI 는 앱 노드 팩 1.7.0 단계 뒤에 이 값을 보냅니다 — 지금은 ComfyUI 로 가지 않습니다'
  + '(ComfyUI 에서는 아직 laplace·custom 스케줄러를 고를 수 없습니다).'
export const KREA2_NOTE = 'Krea2 생성에는 보내지 않습니다.'

/** 칸 아래 상태 한 줄(없으면 ''): Krea2 → 안 보냄, ComfyUI → 팩 1.7.0 단계 뒤(계획 R16), 확장 확인됐는데 없음 → 옛 Forge. */
export function cardNote(caps: Pick<SamExtraCapabilitiesEvent, 'status' | 'known' | 'features'> | null | undefined,
  krea2 = false): string {
  if (krea2) return KREA2_NOTE
  if (caps?.status === 'not_applicable') return COMFY_PENDING_NOTE
  if (caps?.known && caps.features?.extra_schedulers !== true) return OLD_FORGE_NOTE
  return ''
}
