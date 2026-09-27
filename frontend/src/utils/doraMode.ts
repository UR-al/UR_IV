import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'

/**
 * DoRA 추론 방식 카드(components/params/DoraModeCard.vue)의 순수 로직.
 *
 * 파이썬 core/dora_infer_mode.py 표의 거울이다 — 키·라벨(확장 UI 라벨 그대로, 상류 이름이라 번역하지 않는다)·
 * 순서·앱 기본값. tests/test_dora_infer_mode.py 가 이 파일을 읽어 파이썬 표와 대조한다(하나라도 다르면 실패).
 * 위젯 값은 파이썬 프록시와 같은 문자열('true'/'false', 키, '0.12')이다. 판정(보낼지)은 파이썬이 한다 —
 * 여기서는 카드 표시만 계산한다.
 */

export type DoraWidgetKey = 'enabled' | 'mode' | 'inserted' | 'weak_strength' | 'weak_scope' | 'apply_img2img'
export type DoraValues = Record<DoraWidgetKey, string>

export interface DoraOption {
  key: string
  label: string
}

/** 위젯 스토어 키 — 파이썬 core/dora_infer_mode.widget_id(key) = '_dora_<key>'. */
export const WIDGET_IDS: Readonly<Record<DoraWidgetKey, string>> = {
  enabled: '_dora_enabled',
  mode: '_dora_mode',
  inserted: '_dora_inserted',
  weak_strength: '_dora_weak_strength',
  weak_scope: '_dora_weak_scope',
  apply_img2img: '_dora_apply_img2img',
}

/** 계산 방식 — 확장 scripts/dora_infer_mode.py MODE_CHOICES 순서. */
export const MODE_OPTIONS: readonly DoraOption[] = [
  { key: 'lycoris', label: 'LyCORIS (학습과 동일 · fp32)' },
  { key: 'forge_fp32', label: 'Forge/Comfy 공식 · fp32' },
  { key: 'forge', label: 'Forge/Comfy (순정)' },
  { key: 'no_magnitude', label: 'DoRA 끔 (크기 보정 없이 ΔW만 · 일반 LoKr처럼)' },
]

/** 끼워 넣은 블록 — 확장 DUP_CHOICES 순서. */
export const INSERT_OPTIONS: readonly DoraOption[] = [
  { key: 'keep', label: '그대로 복제 (순정 · Forge 기본)' },
  { key: 'additive', label: '덧셈형 (끼워 넣은 블록만 DoRA 크기 보정 끔)' },
  { key: 'skip', label: '넣지 않음 (원래 블록에만 · 모든 LoRA)' },
  { key: 'weak', label: '약한 복사 (브리지식 · 강도·범위 조절)' },
]

/** 약한 복사 범위 — 확장 SCOPE_CHOICES 순서. */
export const SCOPE_OPTIONS: readonly DoraOption[] = [
  { key: 'attn', label: '어텐션만 (브리지 기본)' },
  { key: 'attn_mlp', label: '어텐션+MLP' },
  { key: 'all', label: '전체 (모듈레이션·노름 포함)' },
]

/** 요약 줄에 쓰는 짧은 이름 — 파이썬 SHORT_LABELS. */
export const SHORT_LABELS: Readonly<Record<string, string>> = {
  lycoris: 'LyCORIS', forge_fp32: 'Forge fp32', forge: '순정', no_magnitude: 'DoRA 끔',
  keep: '그대로 복제', additive: '덧셈형', skip: '넣지 않음', weak: '약한 복사',
  attn: '어텐션만', attn_mlp: '어텐션+MLP', all: '전체',
}

/** 앱 기본값 — 사용자 Forge ui-config 의 txt2img 값(파이썬 APP_DEFAULTS). "기본값" 버튼이 스토어에 쓴다. */
export const DEFAULTS: Readonly<DoraValues> = {
  enabled: 'true',
  mode: 'lycoris',
  inserted: 'keep',
  weak_strength: '0.12',
  weak_scope: 'attn',
  apply_img2img: 'false',
}

/** 약한 복사 강도 입력 범위 — 확장 UI 슬라이더(파이썬 WEAK_STRENGTH_RANGE). */
export const WEAK_STRENGTH_RANGE = { min: 0, max: 1, step: 0.01 } as const

export function labelOf(options: readonly DoraOption[], key: string): string {
  return options.find(o => o.key === key)?.label ?? key
}

export function keyOf(options: readonly DoraOption[], label: string): string {
  return options.find(o => o.label === label)?.key ?? label
}

/** 스토어 → 카드 값. 비었거나 모르는 칸은 앱 기본값(파이썬 parse_settings 와 같은 규칙). */
export function readValues(widgets: Record<string, unknown>): DoraValues {
  const pick = (key: DoraWidgetKey, known?: readonly DoraOption[]): string => {
    const raw = widgets[WIDGET_IDS[key]]
    const text = raw === undefined || raw === null ? '' : String(raw)
    if (!text) return DEFAULTS[key]
    if (known && !known.some(o => o.key === text)) return DEFAULTS[key]
    return text
  }
  return {
    enabled: pick('enabled'),
    mode: pick('mode', MODE_OPTIONS),
    inserted: pick('inserted', INSERT_OPTIONS),
    weak_strength: pick('weak_strength'),
    weak_scope: pick('weak_scope', SCOPE_OPTIONS),
    apply_img2img: pick('apply_img2img'),
  }
}

/** 실효 순정인가 — 꺼짐 또는 순정 방식 + 그대로 복제(블록을 보내지 않는 상태). */
export function isStock(values: DoraValues): boolean {
  return values.enabled !== 'true' || (values.mode === 'forge' && values.inserted === 'keep')
}

/** 약한 복사 강도·범위 칸을 보일지. */
export function showWeakFields(values: DoraValues): boolean {
  return values.inserted === 'weak'
}

function formatStrength(value: string): string {
  const n = Number(value)
  return Number.isFinite(n) ? String(Math.round(n * 1000) / 1000) : value
}

/** 카드 제목 옆 요약 — 순정이면 ''. 예: 'LyCORIS · 그대로 복제', '순정 · 약한 복사 0.12 어텐션만'. */
export function summary(values: DoraValues): string {
  if (isStock(values)) return ''
  let text = `${SHORT_LABELS[values.mode] ?? values.mode} · ${SHORT_LABELS[values.inserted] ?? values.inserted}`
  if (values.inserted === 'weak') {
    text += ` ${formatStrength(values.weak_strength)} ${SHORT_LABELS[values.weak_scope] ?? values.weak_scope}`
  }
  return text
}

export type DoraCardStatus = 'off' | 'comfy' | 'unknown' | 'missing' | 'ok'

/** cardStatus 가 읽는 스냅샷 칸만 — useSamExtraCapabilities 의 readonly 값도 그대로 받는다. */
export type DoraCapsLike = Readonly<Pick<SamExtraCapabilitiesEvent, 'status' | 'known'>> & {
  readonly features?: Readonly<Partial<Record<SamExtraFeature, boolean>>>
}

/**
 * 카드 아래 상태 한 줄. 파이썬 게이트와 같은 규칙: 모르면 보내지 않는다(순정), 없으면 보내지 않는다.
 * `status === 'not_applicable'` 은 ComfyUI 백엔드다(types/bridge.d.ts SamExtraStatus).
 */
export function cardStatus(
  caps: DoraCapsLike | null | undefined,
  values: DoraValues,
): { status: DoraCardStatus; text: string } {
  if (isStock(values)) {
    return { status: 'off', text: '꺼짐 — Forge/ComfyUI 순정 공식으로 합칩니다.' }
  }
  if (caps?.status === 'not_applicable') {
    return { status: 'comfy', text: 'ComfyUI 는 순정으로만 합칩니다 — 이 설정은 Forge 에만 적용됩니다.' }
  }
  if (!caps || !caps.known) {
    return { status: 'unknown', text: '확장 확인 전 — 확인될 때까지의 생성은 순정으로 보냅니다.' }
  }
  if (caps.features?.dora !== true) {
    return { status: 'missing', text: '연결된 Forge 에 DoRA 추론 방식 스크립트가 없어 순정으로 보냅니다.' }
  }
  return { status: 'ok', text: '방식을 바꾸면 다음 생성에서 LoRA 를 한 번 다시 합칩니다(1~2초).' }
}
