/**
 * SAM3 '얼굴 안에서만 찾기'(sam-extra 0.34.0) — widget id · 확장 인자 키 · 기본값 표 (순수 로직).
 *
 * 확장은 SAM3 Mask 아코디언과 Refine 패널에 한 줄(켜기 · 얼굴 검출 모델 · 신뢰도)을 더했다 — face_yolov8 얼굴 상자를
 * 넓힌 정사각 영역마다 SAM3 를 돌려 작은 얼굴의 눈·입을 얼굴마다 따로 잡는다. 앱은 T2I SAM3 카드·SAM3 Refine·배치
 * SAM3 에 같은 줄(components/Sam3FaceCropFields.vue)을 둔다.
 *
 * T2I 는 이 widget id(`_sam3_face_crop*`)를 위젯 스토어로 Python 프록시와 동기화하고(ui/generator_ui_setup.py →
 * _build_sam3_settings), Refine·배치 SAM3 는 같은 키로 로컬 상태를 만든 뒤 `sam3FaceCropSettings` 로
 * `sam3_face_crop*` 설정에 옮긴다. 기본값은 Python core/sam3_face_crop.default_values()(= core/sam3_args.SAM3_SPEC —
 * 확장 sam3ext/face_detect.py DEFAULT_MODEL·DEFAULT_CONFIDENCE)와 같아야 한다 — tests/test_sam3_face_crop.py 가 이
 * 파일을 읽어 대조한다. 끈 설정은 Python build_state 가 세 키를 빼서 보낸다(요청이 0.34.0 전과 같다).
 *
 * 얼굴 검출 모델 목록은 Python 이 확장 드롭다운과 같은 규칙으로 골라 `_sam3_face_crop_model` 의 items 로 보낸다
 * (core/sam3_face_crop.model_choices — Forge 는 ADetailer 모델 목록, ComfyUI 는 앱 Forge models/adetailer 폴더). 칸은
 * 선택만 받는다(확장 드롭다운처럼). 목록 밖의 지금 값(다른 PC 에서 온 설정 등)은 지우지 않고 그대로 보낸다(없는 파일이면
 * Forge 결과 알림이 'FaceDetectorUnavailable' 을 알리고, ComfyUI 노드는 마스크 없이 끝낸다).
 *
 * ComfyUI 는 앱 노드 팩 1.8.0 의 ForgeNeoSAM3Mask 가 같은 계산을 한다(sam-extra 얼굴 crop + 0.34.1 crop 키우기). 얼굴 검출은
 * ultralytics 라 ComfyUI 의 Python 에 있어야 한다 — Python 이 /object_info 를 받을 때마다(연결·카드 ↻·팩 설치 뒤 생성)
 * 판단해(core/sam3_face_crop.comfy_support) `_sam3_face_crop` 의 `comfySupport` 속성으로 보낸다. 검출기가 없으면 카드가
 * 켜기를 막고 이유를 보인다. 팩이 옛것이면 경고만 한다 — 앱이 관리하는 ComfyUI 는 다음 생성이 번들 팩을 먼저 설치하고,
 * 컴파일러가 새 스키마로 다시 판단한다.
 */

export type Sam3FaceCropKind = 'bool' | 'number' | 'text'

export interface Sam3FaceCropField {
  /** 위젯·저장 키 (예: face_crop_model) — Python FACE_CROP_FIELDS 와 같다 */
  key: string
  kind: Sam3FaceCropKind
  /** 위젯 스토어에 들어가는 문자열 기본값 */
  def: string
}

/** 확장 sam3ext/face_detect.py DEFAULT_MODEL — 빈 모델 칸도 이 이름으로 보낸다(확장 _normalise_face_crop_model) */
export const FACE_CROP_DEFAULT_MODEL = 'face_yolov8n.pt'
/** 확장 sam3ext/face_detect.py DEFAULT_CONFIDENCE — 0–1 로 잘린다(확장 _NUMERIC_BOUNDS) */
export const FACE_CROP_DEFAULT_CONFIDENCE = 0.3

export const SAM3_FACE_CROP_FIELDS: readonly Sam3FaceCropField[] = [
  { key: 'face_crop', kind: 'bool', def: 'false' },
  { key: 'face_crop_model', kind: 'text', def: 'face_yolov8n.pt' },
  { key: 'face_crop_confidence', kind: 'number', def: '0.3' },
] as const

/** face_crop → _sam3_face_crop (위젯 스토어 id) */
export function sam3FaceCropId(key: string): string {
  return `_sam3_${key}`
}

/** face_crop → sam3_face_crop (확장 인자 키) */
export function sam3FaceCropSettingKey(key: string): string {
  return `sam3_${key}`
}

function isTrue(value: unknown): boolean {
  if (typeof value === 'boolean') return value
  return String(value ?? '').trim().toLowerCase() === 'true'
}

/** widget id → 기본 문자열. Refine·배치 SAM3 의 로컬 상태 초기값. */
export function sam3FaceCropDefaults(): Record<string, string> {
  const out: Record<string, string> = {}
  for (const field of SAM3_FACE_CROP_FIELDS) out[sam3FaceCropId(field.key)] = field.def
  return out
}

export interface Sam3FaceCropSettings {
  sam3_face_crop: boolean
  sam3_face_crop_model: string
  sam3_face_crop_confidence: number
}

/** 로컬 상태(widget id → 값) → 백엔드 설정(`sam3_face_crop*`). 빈 모델은 기본 모델, 신뢰도는 0–1(잘못된 값은 기본값). */
export function sam3FaceCropSettings(values: Record<string, unknown>): Sam3FaceCropSettings {
  const model = String(values[sam3FaceCropId('face_crop_model')] ?? '').trim()
  const text = String(values[sam3FaceCropId('face_crop_confidence')] ?? '').trim()
  const parsed = text === '' ? Number.NaN : Number(text)
  const confidence = Number.isFinite(parsed) ? Math.min(1, Math.max(0, parsed)) : FACE_CROP_DEFAULT_CONFIDENCE
  return {
    sam3_face_crop: isTrue(values[sam3FaceCropId('face_crop')]),
    sam3_face_crop_model: model || FACE_CROP_DEFAULT_MODEL,
    sam3_face_crop_confidence: confidence,
  }
}

/**
 * 얼굴 검출 모델 드롭다운 선택지: Python 이 보낸 추천 목록(items) — 아직 없으면 기본 모델 하나 — 에 목록 밖의 지금 값을
 * 더한다(직접 넣은 파일 이름·다른 PC 에서 온 설정을 지우지 않는다).
 */
export function faceCropModelOptions(items: unknown, current: unknown): string[] {
  const options = Array.isArray(items)
    ? [...new Set(items.map((item) => String(item ?? '').trim()).filter((item) => item !== ''))]
    : []
  if (!options.length) options.push(FACE_CROP_DEFAULT_MODEL)
  const text = String(current ?? '').trim()
  if (text && !options.includes(text)) options.push(text)
  return options
}

/** `_sam3_face_crop` 위젯 속성 — Python core/sam3_face_crop.COMFY_SUPPORT_PROPERTY */
export const COMFY_SUPPORT_PROPERTY = 'comfySupport'
/** Python core/sam3_face_crop.comfy_support 의 값 — 팩 1.8.0 + ultralytics · ultralytics 없음 · 1.8.0 전 팩 */
export type ComfyFaceCropSupport = 'ready' | 'no_detector' | 'old_pack'
/** ComfyUI 에서 얼굴 crop 이 처음 들어온 앱 노드 팩 — Python core/sam3_face_crop.COMFY_MIN_PACK_VERSION */
export const COMFY_MIN_PACK_VERSION = '1.8.0'

/**
 * 연결된 백엔드에서 이 기능을 쓸 수 있나 — 'unknown' 은 판단 재료가 없을 때(연결 전·확인 실패·SAM3 스크립트 없음).
 * ComfyUI 는 'comfy_*'(연결 때 받은 노드 정보), 그 정보가 없으면 'comfy'.
 */
export type FaceCropSupport = 'unknown' | 'supported' | 'unsupported' | 'comfy'
  | 'comfy_ready' | 'comfy_no_detector' | 'comfy_old_pack'

/** 스냅샷에서 읽는 부분만 — readonly 스냅샷(useSamExtraCapabilities)도 받는다. */
export interface FaceCropCapabilitiesLike {
  readonly status?: string
  readonly known?: boolean
  readonly features?: Readonly<Record<string, boolean>>
  readonly sam3_keys?: { readonly missing_in_extension?: readonly string[] }
}

/**
 * 기능 스냅샷 → 지원 여부. ComfyUI(status 'not_applicable')는 연결 때 받은 노드 정보(`comfy` — `comfySupport` 속성)로
 * 'comfy_ready' · 'comfy_no_detector' · 'comfy_old_pack', 정보가 없으면 'comfy'(컴파일러가 생성 때 다시 판단한다). Forge 는
 * script-info SAM3 state 에 'sam3_face_crop' 이 없으면(sam-extra ≤ 0.33.x — 키를 조용히 버리고 전체 이미지에서 찾는다)
 * 'unsupported'. 파이썬 core/sam_extra_capabilities 가 이 키를 연결 경고에서는 뺀다(켤 때만 보내는 키).
 */
export function faceCropSupport(caps: FaceCropCapabilitiesLike | null | undefined, comfy?: unknown): FaceCropSupport {
  if (!caps) return 'unknown'
  if (caps.status === 'not_applicable') {
    if (comfy === 'ready') return 'comfy_ready'
    if (comfy === 'no_detector') return 'comfy_no_detector'
    if (comfy === 'old_pack') return 'comfy_old_pack'
    return 'comfy'
  }
  if (!caps.known || !caps.features?.sam3) return 'unknown'
  const missing = caps.sam3_keys?.missing_in_extension
  return Array.isArray(missing) && missing.includes(sam3FaceCropSettingKey('face_crop')) ? 'unsupported' : 'supported'
}

/**
 * 이 백엔드에서 켤 수 없음 — ComfyUI 에 얼굴 검출기(ultralytics)가 없을 때만. 카드가 켜기를 막는다(이미 켜진 설정은 끌 수 있고,
 * 켠 채 생성하면 컴파일러가 멈춘다). 옛 팩은 막지 않는다(`faceCropPackNote` — 다음 생성이 팩을 먼저 설치할 수 있다).
 */
export function faceCropBlocked(support: FaceCropSupport): boolean {
  return support === 'comfy_no_detector'
}

/** 카드가 상태를 다시 묻는 버튼(↻ — 기능 스냅샷 새로고침이 ComfyUI 에선 /object_info 를 다시 받는다)을 보일 상태 */
export function faceCropRecheckable(support: FaceCropSupport): boolean {
  return support === 'comfy_no_detector' || support === 'comfy_old_pack'
}

/** ComfyUI 에서 켤 수 없는 이유(카드의 경고 줄) — 쓸 수 있거나 Forge 면 ''. `enabled` = 이미 켜져 있다(저장·프리셋에서). */
export function faceCropBlockedReason(support: FaceCropSupport, enabled = false): string {
  if (support !== 'comfy_no_detector') return ''
  const reason = '연결된 ComfyUI 의 Python 에 얼굴 검출기(ultralytics)가 없어 켤 수 없습니다 — ComfyUI 의 Python 에 ultralytics 를 '
    + '설치하고 재시작한 뒤 ↻ 로 다시 확인하거나(다시 연결해도 됩니다) Forge(sam-extra)에서 쓰세요.'
  return enabled ? `${reason} 켜 둔 채 생성하면 이 이유로 멈춥니다 — 끄세요.` : reason
}

/** 옛 팩(1.8.0 전)의 경고 줄 — 켤 수는 있다. 그 밖이면 ''. */
export function faceCropPackNote(support: FaceCropSupport): string {
  if (support !== 'comfy_old_pack') return ''
  return `ComfyUI 의 앱 노드 팩이 ${COMFY_MIN_PACK_VERSION} 보다 오래됐습니다 — 앱이 시작한 ComfyUI 면 다음 생성 때 번들 노드 `
    + '팩을 설치하고 다시 시작합니다. 직접 띄운 ComfyUI 는 팩을 갱신하고 재시작한 뒤 ↻ 로 다시 확인하세요. 그 뒤에도 face_crop '
    + '입력이나 얼굴 검출기(ultralytics)가 없으면 켠 채 생성할 때 이유와 함께 멈춥니다.'
}

/** 'major.minor.patch' 가 `want` 이상인가 — 숫자가 아닌 판(없음 포함)은 false. */
function versionAtLeast(version: unknown, want: readonly number[]): boolean {
  const match = /^\s*v?(\d+)\.(\d+)(?:\.(\d+))?/.exec(String(version ?? ''))
  if (!match) return false
  const have = [Number(match[1]), Number(match[2]), Number(match[3] ?? 0)]
  for (let i = 0; i < want.length; i += 1) {
    if (have[i] !== want[i]) return have[i] > want[i]
  }
  return true
}

/** sam-extra 가 crop 가장자리에 잘린 머리카락을 crop 을 키워 다시 찾기 시작한 판 */
export const FACE_CROP_GROWTH_VERSION = '0.34.1'

/**
 * 영역 밖으로 뻗은 머리카락을 어떻게 하는지 — 백엔드마다 다르다. ComfyUI 팩은 sam-extra 0.34.1 의 crop 키우기를 하고,
 * Forge 는 연결된 sam-extra 판(기능 스냅샷 version.extension_version)에 따른다: 0.34.1 이상이면 키우고 그 전(0.34.0)은 자른다.
 */
export function faceCropHairNote(caps: { readonly version?: unknown } | null | undefined, support: FaceCropSupport): string {
  if (support.startsWith('comfy')) {
    return `crop 가장자리에 잘린 머리카락은 crop 을 키워 다시 찾습니다(sam-extra ${FACE_CROP_GROWTH_VERSION} 방식 — Forge 에 그 전 `
      + 'sam-extra 가 있으면 Forge 는 자릅니다).'
  }
  const version = (caps?.version as { extension_version?: unknown } | undefined)?.extension_version
  if (versionAtLeast(version, FACE_CROP_GROWTH_VERSION.split('.').map(Number))) {
    return 'crop 가장자리에 잘린 머리카락은 crop 을 키워 다시 찾습니다.'
  }
  return `영역 밖으로 뻗은 머리카락은 잘립니다(sam-extra ${FACE_CROP_GROWTH_VERSION} 부터는 crop 을 키워 다시 찾습니다).`
}
