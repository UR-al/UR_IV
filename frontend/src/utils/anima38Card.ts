import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'

/**
 * Anima 3.8B 카드(components/params/Anima38Card.vue)의 순수 로직.
 *
 * 파이썬 core/anima38.py(앱 기본값·위젯 키·강도 범위)와 ui/anima38_ui.py(모델 종류 속성 이름)의 거울이다 —
 * tests/test_anima38_payload.py 가 이 파일을 읽어 파이썬 표와 대조한다(하나라도 다르면 실패). 위젯 값은 파이썬
 * 프록시와 같은 문자열('true'/'false', 어댑터 이름, '1')이다. 보낼지(모델 종류·게이트)는 파이썬이 정한다 —
 * 여기서는 카드 표시만 계산한다. 모델 종류는 파이썬이 연결 때 체크포인트 헤더를 읽어
 * `model_combo` 위젯 속성 `animaKinds`({모델 이름: 'v2'|'anima'|'other'|'unknown'})로 보낸다.
 */

export type Anima38WidgetKey =
  | 'enabled' | 'adapter' | 'strength' | 'negative' | 'negative_strength' | 'bypass' | 'apply_img2img'
export type Anima38Values = Record<Anima38WidgetKey, string>
export type ModelKind = 'v2' | 'anima' | 'other' | 'unknown'

/** 위젯 스토어 키 — 파이썬 core/anima38.widget_id(key) = '_a38_<key>'. */
export const WIDGET_IDS: Readonly<Record<Anima38WidgetKey, string>> = {
  enabled: '_a38_enabled',
  adapter: '_a38_adapter',
  strength: '_a38_strength',
  negative: '_a38_negative',
  negative_strength: '_a38_negative_strength',
  bypass: '_a38_bypass',
  apply_img2img: '_a38_apply_img2img',
}

/** model_combo 위젯 속성 이름 — 파이썬 ui/anima38_ui.KIND_PROPERTY. */
export const KIND_PROPERTY = 'animaKinds'

/** 확장 기본 v1 어댑터 — 파이썬 core/anima38.DEFAULT_ADAPTER(확장 DEFAULT_ADAPTER 핀). */
export const DEFAULT_ADAPTER = 'Anima-3.8B-expanded_adapter.safetensors'

/** `_a38_adapter` 위젯 속성 — ComfyUI 의 v1 어댑터 선택지(파이썬 ui/anima38_ui.COMFY_ADAPTER_PROPERTY). Forge 면 null. */
export const COMFY_ADAPTER_PROPERTY = 'comfyAdapters'

/** ComfyUI 팩이 어댑터를 하나도 못 찾을 때 폴백하는 이름(파이썬 core/anima38.COMFY_ADAPTER_FALLBACK). 파일이 아닌
 *  자리표시자라 파이썬이 `comfyAdapters` 에서 뺀다 — 이 이름이 오면 같은 이름의 파일이 text_encoders 에 있는 것이다. */
export const COMFY_ADAPTER_FALLBACK = 'qwen35_expanded_adapter.safetensors'

/**
 * 앱 기본값 — 사용자 Forge ui-config txt2img(부정 커넥터 켬, v1 끔 — 사용자 결정 D1=B, 파이썬 APP_T2I_DEFAULTS).
 * "기본값" 버튼이 스토어에 쓴다.
 */
export const DEFAULTS: Readonly<Anima38Values> = {
  enabled: 'false',
  adapter: DEFAULT_ADAPTER,
  strength: '1',
  negative: 'true',
  negative_strength: '1',
  bypass: 'false',
  apply_img2img: 'false',
}

/** 어댑터·부정 강도 슬라이더 — 확장 UI(0–2, step 0.05, 파이썬 STRENGTH_RANGE). */
export const STRENGTH_RANGE = { min: 0, max: 2, step: 0.05 } as const

const KINDS: readonly ModelKind[] = ['v2', 'anima', 'other', 'unknown']

/** 스토어 → 카드 값. 비었거나 없는 칸은 앱 기본값(파이썬 settings_from_widgets 와 같은 규칙). */
export function readValues(widgets: Record<string, unknown>): Anima38Values {
  const out = { ...DEFAULTS }
  for (const key of Object.keys(WIDGET_IDS) as Anima38WidgetKey[]) {
    const raw = widgets[WIDGET_IDS[key]]
    const text = raw === undefined || raw === null ? '' : String(raw)
    if (text) out[key] = text
  }
  return out
}

/** 현재 모델의 종류 — 속성이 없거나 모르는 값이면 'unknown'. */
export function modelKind(kinds: unknown, model: unknown): ModelKind {
  if (!kinds || typeof kinds !== 'object') return 'unknown'
  const value = (kinds as Record<string, unknown>)[String(model ?? '')]
  return KINDS.includes(value as ModelKind) ? (value as ModelKind) : 'unknown'
}

const on = (value: string) => value === 'true'

function formatStrength(value: string): string {
  const n = Number(value)
  return Number.isFinite(n) ? String(Math.round(n * 100) / 100) : value
}

/**
 * 카드 제목 옆 요약 — 이 모델에서 실제로 쓰이는 값만(파이썬 effective 와 같은 규칙). 블록 없음과 같으면 ''.
 * v2: Bypass 또는 부정 커넥터 / 비 번들 Anima: v1 을 켰을 때만(부정은 v1 설치 안에서만 쓰인다) / 모름: 켠 것 모두.
 */
export function summary(kind: ModelKind, values: Anima38Values): string {
  const negative = on(values.negative)
  const v1 = on(values.enabled)
  const bypass = on(values.bypass)
  if (kind === 'other') return ''
  if (kind === 'v2') return bypass ? 'Bypass' : (negative ? '부정 커넥터' : '')
  if (kind === 'anima') {
    if (bypass || !v1) return ''
    const text = `v1 ${formatStrength(values.strength)}`
    return negative ? `${text} · 부정 ${formatStrength(values.negative_strength)}` : text
  }
  if (bypass) return 'Bypass'
  const parts: string[] = []
  if (negative) parts.push('부정 커넥터')
  if (v1) parts.push(`v1 ${formatStrength(values.strength)}`)
  return parts.join(' · ')
}

export interface Anima38CardView {
  /** 카드를 보일지 — 비 Anima(other)·Krea2 면 숨긴다(설정은 저장된다). */
  visible: boolean
  /** 부정 강도 칸(v1 에서만 쓰인다 — v2 는 1.0 고정 문구). */
  showNegativeStrength: boolean
  /** v2 번들은 부정 강도가 1.0 고정이라는 안내. */
  v2FixedNote: boolean
  /** Bypass — v2 번들 자동 켜짐을 끄는 칸(v2·모름). 비 번들 Anima 에서도 켜져 있으면 보인다 — 확장은 Bypass 면
   *  v1 설치 전에 돌아가므로(process_batch) v1 을 막는데, 칸이 숨으면 되돌릴 수 없다. */
  showBypass: boolean
  /** 비 번들 Anima 에서 v1 을 켰지만 Bypass 때문에 쓰이지 않는다는 안내. */
  v1BypassedNote: boolean
  /** v1 어댑터 묶음(비 번들 Anima·모름). */
  showV1: boolean
}

export function cardView(kind: ModelKind, values: Anima38Values, krea2 = false): Anima38CardView {
  const visible = !krea2 && kind !== 'other'
  return {
    visible,
    showNegativeStrength: visible && kind !== 'v2' && on(values.negative),
    v2FixedNote: visible && kind === 'v2' && on(values.negative),
    showBypass: visible && (kind !== 'anima' || on(values.bypass)),
    v1BypassedNote: visible && kind === 'anima' && on(values.bypass) && on(values.enabled),
    showV1: visible && kind !== 'v2',
  }
}

/** v1 어댑터 선택지 — Forge 스냅샷의 choices.anima38_adapters, ComfyUI 면 `comfyAdapters`(object_info 의
 *  ForgeNeoAnimaQwen35Prompt.adapter_name — 파이썬이 연결 때 `_a38_adapter` 속성으로 보낸다. 배열이 아니면 Forge).
 *  확장·팩은 아무것도 없어도 이름 하나로 폴백하므로 그 이름 하나뿐이면 설치 여부를 모른다(unverified).
 *  ComfyUI 팩의 폴백은 파일이 아닌 자리표시자라 파이썬(core/anima38.comfy_adapter_choices)이 빼고 빈 배열을 보낸다 —
 *  그때는 기본 이름(+저장값)만 두고 미확인으로 표시한다(없는 파일을 고르게 하지 않는다 — P9 리뷰 2차 1). 폴백 이름이
 *  그대로 오면 그 이름의 파일이 있다(팩이 v1 으로 받았는지는 모름 — 미확인).
 *  지금 값이 목록에 없으면 앞에 붙인다(저장값 보존). */
export function adapterChoices(
  caps: { readonly choices?: Readonly<Record<string, readonly string[]>> } | null | undefined,
  current: string,
  comfyAdapters?: unknown,
): { options: string[]; unverified: boolean } {
  const comfy = Array.isArray(comfyAdapters)
  const listed = comfy ? comfyAdapters : caps?.choices?.anima38_adapters
  const live: string[] = Array.isArray(listed) ? listed.map(String).filter(Boolean) : []
  const fallback = comfy ? COMFY_ADAPTER_FALLBACK : DEFAULT_ADAPTER
  const unverified = live.length === 0 || (live.length === 1 && live[0] === fallback)
  const options = live.length ? live : [DEFAULT_ADAPTER]
  if (current && !options.includes(current)) options.unshift(current)
  return { options, unverified }
}

export type Anima38CardStatus = 'comfy' | 'missing' | 'unknown_caps' | 'v2' | 'anima' | 'unknown_model'

/** 카드 상태 줄이 읽는 스냅샷 칸만 — useSamExtraCapabilities 의 readonly 값도 그대로 받는다. */
export type Anima38CapsLike = Readonly<Pick<SamExtraCapabilitiesEvent, 'status' | 'known'>> & {
  readonly features?: Readonly<Partial<Record<SamExtraFeature, boolean>>>
}

/**
 * 카드 아래 상태 한 줄. 파이썬 규칙과 같다: 스크립트가 없으면 보내지 않는다, 확인 전에는 앱 기본값을 보내지 않고
 * 바꾼 값만 보낸다(A2), 모델 종류를 모르면 바꾼 값만 보낸다. `status === 'not_applicable'` 은 ComfyUI 다.
 */
export function cardStatus(
  caps: Anima38CapsLike | null | undefined,
  kind: ModelKind,
): { status: Anima38CardStatus; text: string } {
  if (caps?.status === 'not_applicable') {
    return {
      status: 'comfy',
      text: 'ComfyUI — 같은 설정을 워크플로 컴파일러가 씁니다. v1 은 켤 때만, 어댑터는 이 카드에서 고른 것만 쓰고'
        + '(모듈 목록의 어댑터는 쓰지 않음), Qwen3.5·어댑터 파일이 없으면 생성 전에 오류로 멈춥니다(Forge 는 순정으로 계속).',
    }
  }
  if (caps?.known && caps.features?.anima38 !== true) {
    return { status: 'missing', text: '연결된 Forge 의 sam-extra 에 Anima 3.8B 스크립트가 없어 이 설정을 보내지 않습니다.' }
  }
  if (!caps || !caps.known) {
    return { status: 'unknown_caps', text: '확장 확인 전 — 확인될 때까지는 바꾼 값만 보냅니다(기본값은 확인 뒤부터).' }
  }
  if (kind === 'v2') {
    return { status: 'v2', text: 'v2 번들 — Forge 가 자동으로 켭니다(끄려면 Bypass). 어댑터·강도는 쓰이지 않습니다.' }
  }
  if (kind === 'anima') {
    return { status: 'anima', text: '구형 v1 어댑터를 쓸 수 있는 Anima 모델 — v1 은 켤 때만 씁니다.' }
  }
  return { status: 'unknown_model', text: '모델 종류를 확인하지 못했습니다 — 바꾼 값만 보냅니다.' }
}
