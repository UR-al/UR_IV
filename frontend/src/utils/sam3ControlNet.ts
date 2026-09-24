/**
 * SAM3 ControlNet 13필드 — widget id · 확장 인자 키 · 기본값 표 (순수 로직).
 *
 * T2I 는 이 widget id(`_sam3_cn_*`)를 위젯 스토어로 Python 프록시와 동기화하고
 * (ui/generator_ui_setup.py → _build_sam3_settings), SAM3 Refine·배치 SAM3 는 같은 키로
 * 로컬 상태를 만든 뒤 `sam3CnSettings` 로 `sam3_cn_*` 설정에 옮긴다.
 *
 * 기본값은 Python core/sam3_controlnet.default_values()(= core/sam3_args.SAM3_SPEC)와
 * 같아야 한다 — tests/test_sam3_controlnet.py 가 이 파일을 읽어 대조한다.
 * 선택지(전처리기·control/resize mode)는 여기 두지 않는다: Python 이 프록시 items 로
 * 보내고(`getProperty(id, 'items')`) 그 목록 하나만 쓴다.
 */

export type Sam3CnKind = 'bool' | 'number' | 'text'

export interface Sam3CnField {
  /** 위젯·저장 키 (예: cn_weight) — Python CN_FIELDS 와 같다 */
  key: string
  kind: Sam3CnKind
  /** 위젯 스토어에 들어가는 문자열 기본값 */
  def: string
}

export const SAM3_CN_FIELDS: readonly Sam3CnField[] = [
  { key: 'cn_enable', kind: 'bool', def: 'false' },
  { key: 'cn_override_external', kind: 'bool', def: 'false' },
  { key: 'cn_model', kind: 'text', def: 'None' },
  { key: 'cn_module', kind: 'text', def: 'inpaint_only' },
  { key: 'cn_weight', kind: 'number', def: '1.0' },
  { key: 'cn_guidance_start', kind: 'number', def: '0.0' },
  { key: 'cn_guidance_end', kind: 'number', def: '1.0' },
  { key: 'cn_pixel_perfect', kind: 'bool', def: 'true' },
  { key: 'cn_control_mode', kind: 'text', def: 'Balanced' },
  { key: 'cn_resize_mode', kind: 'text', def: 'Crop and Resize' },
  { key: 'cn_processor_res', kind: 'number', def: '512' },
  { key: 'cn_threshold_a', kind: 'number', def: '-1.0' },
  { key: 'cn_threshold_b', kind: 'number', def: '-1.0' },
] as const

/** cn_weight → _sam3_cn_weight (위젯 스토어 id) */
export function sam3CnId(key: string): string {
  return `_sam3_${key}`
}

/** cn_weight → sam3_cn_weight (확장 인자 키) */
export function sam3CnSettingKey(key: string): string {
  return `sam3_${key}`
}

/** 스토어는 불리언을 'true'/'false' 문자열로 들고 있다 — 백엔드 coercion 과 같은 규칙. */
export function isTrue(value: unknown): boolean {
  if (typeof value === 'boolean') return value
  return String(value ?? '').trim().toLowerCase() === 'true'
}

/** widget id → 기본 문자열. Refine·배치 SAM3 의 로컬 상태 초기값. */
export function sam3CnDefaults(): Record<string, string> {
  const out: Record<string, string> = {}
  for (const field of SAM3_CN_FIELDS) out[sam3CnId(field.key)] = field.def
  return out
}

/** 로컬 상태(widget id → 값) → 백엔드 설정(`sam3_cn_*`). 잘못된 숫자는 기본값으로. */
export function sam3CnSettings(values: Record<string, unknown>): Record<string, boolean | number | string> {
  const out: Record<string, boolean | number | string> = {}
  for (const field of SAM3_CN_FIELDS) {
    const raw = values[sam3CnId(field.key)]
    const settingKey = sam3CnSettingKey(field.key)
    if (field.kind === 'bool') {
      out[settingKey] = raw === undefined ? isTrue(field.def) : isTrue(raw)
    } else if (field.kind === 'number') {
      const text = String(raw ?? '').trim()
      const parsed = text === '' ? Number.NaN : Number(text)
      out[settingKey] = Number.isFinite(parsed) ? parsed : Number(field.def)
    } else {
      const text = String(raw ?? '').trim()
      out[settingKey] = text || field.def
    }
  }
  return out
}

/** 선택지가 아직 안 왔으면(개발 모드·연결 전) 현재 값 하나만 보여 준다 — 목록을 복제하지 않는다. */
export function choiceOptions(items: unknown, current: unknown, fallback: string): string[] {
  if (Array.isArray(items) && items.length) return items.map((item) => String(item))
  const value = String(current ?? '').trim() || fallback
  return [value]
}
