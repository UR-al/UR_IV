/**
 * sam-extra Forge 옵션 요청별 덮어쓰기(P10) — 파이썬 core/forge_override_settings.py `SPECS` 의 거울(표시·저장 모양).
 *
 * 보낼지(키 존재 확인·병합·거절 시 빼고 다시)는 파이썬 백엔드가 정한다 — 여기는 카드 표시와 ui_prefs 값만.
 * 값은 `ui_prefs.forgeOptionOverrides` = { 옵션 키: boolean }. 키가 없으면 'Forge 설정 따름'(보내지 않음, 기본).
 * tests/test_forge_override_settings.py 가 키·라벨·묶음·기본값·infotext 가 파이썬 표와 같은지 정규식으로 본다 —
 * `SPECS` 한 줄 모양을 바꾸면 그 정규식도 고친다.
 */
import type { SamExtraCapabilitiesEvent } from '../types/bridge'

export const PREF_KEY = 'forgeOptionOverrides'

export type OptionGroup = 'memory' | 'result_minor' | 'result'
export type OverrideChoice = 'follow' | 'on' | 'off'
export type Overrides = Record<string, boolean>

export interface ForgeOptionSpec {
  readonly key: string
  readonly label: string
  readonly group: OptionGroup
  readonly default: boolean
  readonly infotext: string
}

export const GROUPS: readonly { readonly id: OptionGroup; readonly title: string }[] = [
  { id: 'memory', title: '메모리·속도 (결과 같음)' },
  { id: 'result_minor', title: '재현용 (결과가 아주 미세하게 다름 · infotext 에 기록)' },
  { id: 'result', title: '결과가 달라짐' },
]

// 순서 = 카드 순서 = 파이썬 SPECS 순서
export const SPECS: readonly ForgeOptionSpec[] = [
  { key: 'sam3_anima38_keep_resident', label: 'Anima 3.8B 모델 VRAM 상주', group: 'memory', default: true, infotext: '' },
  { key: 'sam3_anima38_connector_fp32', label: 'Anima 3.8B 커넥터 fp32 상주', group: 'memory', default: true, infotext: '' },
  { key: 'sam3_anima38_connector_run_cache', label: 'Anima 3.8B 커넥터 실행 캐시', group: 'memory', default: true, infotext: '' },
  { key: 'sam3_unload_keep_in_ram', label: 'SAM3 모델 RAM 보관', group: 'memory', default: true, infotext: '' },
  { key: 'sam3_guidance_pag_prefix_dedup', label: 'PAG 앞 블록 중복 계산 건너뛰기', group: 'result_minor', default: true, infotext: 'Anima PAG prefix dedup' },
  { key: 'sam3_guidance_seg_separable_blur', label: 'SEG 블러 1D 분리 계산', group: 'result_minor', default: true, infotext: 'Anima SEG separable blur' },
  { key: 'sam3_anima_sparse_lora_forge_guess', label: '부분 LoRA 순정 추측 변환', group: 'result', default: false, infotext: 'Anima sparse LoRA' },
  { key: 'sam3_guidance_dave_pre_dd_sigma', label: 'DAVE 판정을 Detail Daemon 전 σ 로 (우회)', group: 'result', default: true, infotext: 'Anima DAVE pre-DD sigma' },
]

/** 줄마다 한두 문장 — 확장 설명을 요약한 것(복사하지 않는다). */
export const DESCRIPTIONS: Readonly<Record<string, string>> = {
  sam3_anima38_keep_resident: 'TE·Qwen3.5·커넥터를 생성 사이 VRAM 에 남깁니다(최대 약 6~8 GB). 끄면 생성이 끝날 때 내리고 다음 생성이 1~2초 더 걸릴 수 있습니다.',
  sam3_anima38_connector_fp32: '커넥터를 샘플링 동안 fp32 로 상주시킵니다(VRAM 약 +1.5 GB, 장당 약 1~2초 빨라짐).',
  sam3_anima38_connector_run_cache: '커넥터가 스텝마다 되풀이하는 계산을 프롬프트 줄마다 한 번만 합니다(VRAM 줄당 약 15~70 MB, 생성이 끝나면 버림).',
  sam3_unload_keep_in_ram: "SAM3 '사용 후 언로드' 뒤 모델(약 3.4 GB)을 CPU RAM 에 두었다가 다음 검출 때 옮기기만 합니다. 끔이면 그 요청의 언로드에서 RAM 까지 비웁니다. ComfyUI 는 일반 탭의 'ComfyUI SAM3 모델을 RAM 에 보관'을 씁니다.",
  sam3_guidance_pag_prefix_dedup: 'PAG·SEG·SLG 에서 첫 대상 블록 앞의 중복 계산을 건너뜁니다(PAG 한 장에 약 9% 빠름). v0.30 이전 결과를 비트 단위로 재현하려면 끔.',
  sam3_guidance_seg_separable_blur: 'SEG 블러를 2D 한 번 대신 가로·세로 1D 두 번으로 계산합니다. 반올림 순서만 다릅니다.',
  sam3_anima_sparse_lora_forge_guess: '일부 블록만 담은 Anima LoRA 를 순정 판정 레이아웃에서 지금 모델로 추측 변환해 로드합니다(끔이면 건너뜀). 블록 대응이 틀릴 수 있습니다.',
  sam3_guidance_dave_pre_dd_sigma: 'DAVE 와 Detail Daemon 을 함께 켰을 때 DAVE 적용 스텝을 Detail Daemon 이 σ 를 바꾸기 전 값으로 판정합니다. Detail Daemon 을 끈 생성은 켜고 끔이 같습니다.',
}

export const CHOICES: readonly { readonly id: OverrideChoice; readonly label: string }[] = [
  { id: 'follow', label: 'Forge 설정 따름' },
  { id: 'on', label: '켬' },
  { id: 'off', label: '끔' },
]

const SPEC_KEYS = new Set(SPECS.map(spec => spec.key))

export function specsOf(group: OptionGroup): ForgeOptionSpec[] {
  return SPECS.filter(spec => spec.group === group)
}

/** ui_prefs 값 → 스펙 키 중 값이 진짜 boolean 인 것만(스펙 순서). 파이썬 normalize_overrides 와 같은 규칙. */
export function normalizeOverrides(raw: unknown): Overrides {
  const out: Overrides = {}
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out
  const source = raw as Record<string, unknown>
  for (const spec of SPECS) {
    const value = source[spec.key]
    if (typeof value === 'boolean') out[spec.key] = value
  }
  return out
}

export function choiceOf(overrides: Readonly<Overrides>, key: string): OverrideChoice {
  const value = overrides[key]
  return value === true ? 'on' : value === false ? 'off' : 'follow'
}

/** 새 객체 — 'follow' 는 키를 지운다(보내지 않음). 스펙 밖 키는 바꾸지 않는다. */
export function withChoice(overrides: Readonly<Overrides>, key: string, choice: OverrideChoice): Overrides {
  const next = normalizeOverrides(overrides)
  if (!SPEC_KEYS.has(key)) return next
  if (choice === 'follow') delete next[key]
  else next[key] = choice === 'on'
  return normalizeOverrides(next)
}

export function choiceLabel(choice: OverrideChoice): string {
  return CHOICES.find(item => item.id === choice)?.label ?? CHOICES[0]!.label
}

export function choiceFromLabel(label: string | number): OverrideChoice {
  return CHOICES.find(item => item.label === String(label))?.id ?? 'follow'
}

export type OptionsCapsLike = Readonly<Pick<SamExtraCapabilitiesEvent, 'status' | 'known'>> & {
  readonly options_known?: boolean
  readonly options?: Readonly<Record<string, unknown>>
}
export type OptionState = 'comfy' | 'unknown' | 'unverified' | 'missing' | 'ok'

export interface OptionStatus {
  readonly state: OptionState
  /** Forge 설정 화면을 만들 때의 값(Gradio /config) — 표시용. 모르면 null */
  readonly forgeValue: boolean | null
  readonly text: string
}

/**
 * 줄 상태. 파이썬 plan_overrides 와 같은 규칙: 스냅샷을 모르거나 /config 를 못 읽었으면 보내지 않는다(모르는 키는
 * 요청 전체가 500), 키가 없으면 보내지 않는다. `status === 'not_applicable'` 은 ComfyUI 백엔드다.
 */
export function optionStatus(caps: OptionsCapsLike | null | undefined, key: string): OptionStatus {
  if (caps?.status === 'not_applicable') {
    return { state: 'comfy', forgeValue: null, text: 'ComfyUI 에는 해당 없음' }
  }
  if (!caps || !caps.known) {
    return { state: 'unknown', forgeValue: null, text: 'Forge 확인 전 — 확인될 때까지 보내지 않음' }
  }
  if (!caps.options_known) {
    return { state: 'unverified', forgeValue: null, text: 'Forge 설정 목록을 읽지 못함(로그인·--nowebui) — 보내지 않음' }
  }
  const options = caps.options || {}
  if (!Object.prototype.hasOwnProperty.call(options, key)) {
    return { state: 'missing', forgeValue: null, text: '이 Forge 의 sam-extra 에 없음 — 보내지 않음(확장 업데이트 필요)' }
  }
  const value = options[key]
  const forgeValue = typeof value === 'boolean' ? value : null
  const shown = forgeValue === null ? '알 수 없음' : forgeValue ? '켬' : '끔'
  return { state: 'ok', forgeValue, text: `Forge 값: ${shown} (Forge 시작 시점)` }
}

/** 고른 값이 보내지지 않는 상태인가(경고 색) — 'Forge 설정 따름'이면 원래 보내지 않으므로 아니다. */
export function isBlocked(status: OptionStatus, choice: OverrideChoice): boolean {
  return choice !== 'follow' && (status.state === 'unknown' || status.state === 'unverified' || status.state === 'missing')
}

/** 줄별 주의 문구(없으면 ''). */
export function rowWarning(key: string, choice: OverrideChoice, forgeValue: boolean | null): string {
  if (key === 'sam3_unload_keep_in_ram' && choice === 'on' && forgeValue === false) {
    return 'Forge 설정이 끔이면 요청이 끝날 때 보관본을 버립니다 — 한 요청 안의 여러 장에서만 효과가 있습니다.'
  }
  if (key === 'sam3_anima38_keep_resident' && choice === 'on') {
    return '같은 GPU 에서 학습 중이면 끔을 권장합니다 — 켜 두면 생성 사이에 최대 6~8 GB 를 남깁니다.'
  }
  if (key === 'sam3_anima_sparse_lora_forge_guess' && choice !== 'follow') {
    return '값이 Forge 설정과 다르면 그 LoRA 를 다시 합칩니다(Forge UI 와 번갈아 쓸 때마다 몇 초).'
  }
  if (key === 'sam3_guidance_dave_pre_dd_sigma' && choice === 'off') {
    return '끄면 DAVE 와 Detail Daemon 을 함께 켠 생성이 원본 ComfyUI 노드 조합처럼 무너집니다(재현용).'
  }
  return ''
}

/** 카드 머리 한 줄 요약. */
export function summary(overrides: Readonly<Overrides>): string {
  const count = Object.keys(normalizeOverrides(overrides)).length
  return count ? `${count}개 덮어씀` : '모두 Forge 설정 따름'
}
