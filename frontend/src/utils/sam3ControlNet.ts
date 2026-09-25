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
 *
 * 전처리기·모델은 **연결된 Forge 의 라이브 목록**이 먼저다(P3): sam-extra 기능 스냅샷의
 * `choices.controlnet_modules` / `controlnet_models` (= Forge /controlnet/module_list · model_list,
 * 모델에는 확장이 등록한 models/sam3 LLLite 포함). 스냅샷을 모르면 전처리기는 Python 정적 폴백 items,
 * 모델은 자유 입력이다. 모델 목록은 연결 때 것이라 목록을 알아도 직접 입력을 막지 않는다(cnModelOptions).
 * Forge 는 이름을 대소문자까지 그대로 찾으므로(소문자 'none' 은 KeyError)
 * 대소문자만 다른 값은 목록 표기로 고친다 — 파이썬 core/sam3_cn_names.py 와 같은 규칙.
 * Anima ControlNet-LLLite 모델이면 전처리기를 원본처럼 맞춘다(sam3CnLlliteModule — Tile & Repair 는 'None',
 * 그 밖의 Anima LLLite 는 inpaint_* 만 'None').
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

/** Forge 의 '없음' 표기 — 전처리기·모델 모두 대문자 N (소문자 'none' 은 Forge 에서 KeyError). */
export const CN_NONE = 'None'

/** 연결된 Forge 의 라이브 CN 목록. null = 모름(스냅샷 없음·확인 실패·ComfyUI·CN 확장 없음). */
export interface Sam3CnLiveLists {
  modules: string[] | null
  models: string[] | null
}

/** 스냅샷에서 읽는 부분만 — readonly 스냅샷(useSamExtraCapabilities)도 받는다. */
export interface Sam3CnCapabilitiesLike {
  readonly known?: boolean
  readonly choices?: Readonly<Record<string, unknown>>
}

function nameList(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null
  const names = [...new Set(value.filter((item): item is string => typeof item === 'string' && item.trim() !== ''))]
  return names.length ? names : null
}

/** 기능 스냅샷 → 라이브 CN 목록. known 이 아니면 둘 다 null (파이썬 sam3_cn_names.live_lists 와 같은 규칙). */
export function sam3CnLiveLists(caps: Sam3CnCapabilitiesLike | null | undefined): Sam3CnLiveLists {
  if (!caps || !caps.known) return { modules: null, models: null }
  return {
    modules: nameList(caps.choices?.controlnet_modules),
    models: nameList(caps.choices?.controlnet_models),
  }
}

/** 전처리기 드롭다운 선택지: 라이브 목록 → 정적 폴백(Python 프록시 items) → 현재 값 하나. */
export function cnModuleOptions(live: readonly string[] | null, fallbackItems: unknown, current: unknown): string[] {
  if (live && live.length) return [...live]
  return choiceOptions(fallbackItems, current, 'inpaint_only')
}

/**
 * 목록 표기로 맞춘 이름 — 정확히 같으면 그대로, 대소문자만 다르면 목록 쪽, 'none' 은 'None'.
 * 빈 값과 목록 밖 이름은 받은 그대로 돌려준다(다른 이름으로 바꿔치지 않는다).
 */
export function canonicalCnName(value: unknown, options: readonly string[]): unknown {
  const text = String(value ?? '').trim()
  if (!text) return value
  if (options.includes(text)) return text
  const folded = text.toLowerCase()
  const match = options.find((option) => option.toLowerCase() === folded)
  if (match !== undefined) return match
  return folded === CN_NONE.toLowerCase() ? CN_NONE : value
}

/**
 * 모델 드롭다운 선택지: 라이브 목록 + (목록 밖이면) 지금 값.
 *
 * 라이브 모델 목록은 **연결 때** 받은 Forge `controlnet_names` 라 모델을 다 담지 못한다. 확장은 SAM3 패스마다
 * models/sam3 를 다시 스캔해 `controlnet_filename_dict` 에만 더하므로(sam3ext/inpaint_core.inject_controlnet_unit),
 * Forge 시작 뒤 models/sam3 에 넣은 LLLite, Forge 의 ControlNet 새로고침 뒤의 models/sam3 파일은 목록에 없어도
 * 생성 때 찾는다. 그래서 직접 입력한 이름을 목록 밖이라고 지우지 않고 선택지에 남긴다.
 */
export function cnModelOptions(live: readonly string[] | null, current: unknown): string[] {
  const options = live && live.length ? [...live] : [CN_NONE]
  const text = String(current ?? '').trim()
  if (text && !options.includes(text)) options.push(text)
  return options
}

/** Anima ControlNet-LLLite cond 채널 — 3 = 표준(Tile & Repair·lineart·canny·depth 등), 4 = 인페인트(RGB+마스크). */
export type Sam3CnLlliteChannels = 3 | 4

function flatName(value: unknown): string {
  return String(value ?? '').trim().toLowerCase().replace(/[^0-9a-z]/g, '')
}

/**
 * 모델 이름 → Anima LLLite 채널 추정. Anima LLLite 로 보이지 않으면 null.
 *
 * 파이썬 core/sam3_cn_names.lllite_channels_from_name · 확장 sam3ext/sam3_cn_lllite 의 이름 폴백과 같은 표다
 * (확장은 모델 파일 헤더로 먼저 판별한다 — 이름을 바꾼 파일도 생성 때 다시 잡는다). 구분자·대소문자는 무시.
 *  - 'anima' + 'lllite' + 'inpaint'         → 4 (anima-lllite-inpainting-v2)
 *  - 'anima' + ('lllite' 또는 'tilerepair') → 3 (animaTileRepair_v20, anima_lllite_lineart_v1)
 * 'anima' 가 없으면 null — SDXL kohya_controllllite_xl_* 는 'lllite' 가 들어 있어도 Anima LLLite 가 아니다
 * (확장은 헤더가 Anima 가 아니면 전처리기를 그대로 둔다).
 */
export function sam3CnLlliteChannels(model: unknown): Sam3CnLlliteChannels | null {
  const flat = flatName(model)
  if (!flat || flat === CN_NONE.toLowerCase() || !flat.includes('anima')) return null
  if (flat.includes('lllite') && flat.includes('inpaint')) return 4
  if (flat.includes('lllite') || flat.includes('tilerepair')) return 3
  return null
}

/**
 * 이름으로 본 3채널 Anima LLLite 가 Tile & Repair 인가 — 이름에 'tile' (animaTileRepair_*, anima_tiled_lllite_*).
 * 확장은 헤더 modelspec.title(v1.0 'anima_tiled_lllite_v1', v2.0 'anima_tile_multitask_v1')로 먼저 본다.
 * 파이썬 lllite_tile_repair_from_name 과 같다.
 */
export function sam3CnLlliteTileRepair(model: unknown): boolean {
  return sam3CnLlliteChannels(model) === 3 && flatName(model).includes('tile')
}

export interface Sam3CnLlliteGuard {
  /** 보낼(쓸) 전처리기 */
  module: string
  /** 모델의 LLLite 채널(아니면 null) */
  channels: Sam3CnLlliteChannels | null
  /** Tile & Repair 인가 (3채널만) */
  tileRepair: boolean
  /** 전처리기를 바꿔야 하는가 */
  forced: boolean
}

/**
 * Anima LLLite 전처리기 가드 — 원본(kohya sd-scripts · ComfyUI-Anima-LLLite)은 LLLite 에 사용자가 준 제어
 * 이미지를 그대로 준다. SAM3 CN 유닛의 제어 이미지는 인페인트 입력 이미지라:
 *  - Tile & Repair → 전처리기는 언제나 'None' (그 그림 자체가 제어 이미지, inpaint_only 는 고칠 영역을 비운다)
 *  - 그 밖의 Anima LLLite(3채널 lineart·canny·depth, 4채널 인페인트) → inpaint_* 만 'None'. 3채널은 원본이
 *    마스크를 쓰지 않고, 4채널은 inpaint_* 가 마스크를 버려 LLLite 가 실패한다. lineart_anime·canny 등은 그대로.
 * 파이썬 lllite_module_override · 확장 forced_cn_module 과 같다.
 */
export function sam3CnLlliteModule(module: unknown, model: unknown): Sam3CnLlliteGuard {
  const current = String(module ?? '')
  const channels = sam3CnLlliteChannels(model)
  const tileRepair = sam3CnLlliteTileRepair(model)
  const forced = channels !== null && current !== CN_NONE && (tileRepair || current.startsWith('inpaint'))
  return { module: forced ? CN_NONE : current, channels, tileRepair, forced }
}

/** LLLite 모델일 때 고를 수 있는 전처리기: Tile & Repair = 'None' 하나, 그 밖의 Anima LLLite = inpaint_* 뺀 목록. */
export function cnModuleOptionsForModel(options: readonly string[], model: unknown): string[] {
  if (sam3CnLlliteChannels(model) === null) return [...options]
  if (sam3CnLlliteTileRepair(model)) return [CN_NONE]
  const allowed = options.filter((option) => !option.startsWith('inpaint'))
  return allowed.length ? allowed : [CN_NONE]
}

/**
 * 라이브 목록이 있는데 그 안에 없는 이름. 모르면 false.
 * 전처리기는 Forge 시작 때 정해지므로 목록 밖 = KeyError(SAM3 패스 실패)다. 모델은 연결 때 목록이라
 * models/sam3 파일이면 생성 때 여전히 찾는다 — 패널이 문구를 나눠 쓴다(cnModelOptions 주석).
 */
export function cnNameMissing(value: unknown, live: readonly string[] | null): boolean {
  if (!live || !live.length) return false
  const text = String(value ?? '').trim()
  return text !== '' && !live.includes(text)
}
