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
 * 얼굴 검출 모델 목록은 Python 이 ADetailer 모델 목록에서 확장 드롭다운과 같은 규칙으로 골라
 * `_sam3_face_crop_model` 의 items 로 보낸다(core/sam3_face_crop.model_choices). 확장은 아무것도 받지 않으므로 목록
 * 밖 이름도 그대로 보낸다(없는 파일이면 결과 알림이 'FaceDetectorUnavailable' 을 알린다).
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

/** 연결된 백엔드에서 이 기능을 쓸 수 있나 — 'unknown' 은 판단 재료가 없을 때(연결 전·확인 실패·SAM3 스크립트 없음). */
export type FaceCropSupport = 'unknown' | 'supported' | 'unsupported' | 'comfy'

/** 스냅샷에서 읽는 부분만 — readonly 스냅샷(useSamExtraCapabilities)도 받는다. */
export interface FaceCropCapabilitiesLike {
  readonly status?: string
  readonly known?: boolean
  readonly features?: Readonly<Record<string, boolean>>
  readonly sam3_keys?: { readonly missing_in_extension?: readonly string[] }
}

/**
 * 기능 스냅샷 → 지원 여부. ComfyUI(status 'not_applicable')는 팩에 아직 없어 'comfy'(컴파일러가 거절한다). Forge 는
 * script-info SAM3 state 에 'sam3_face_crop' 이 없으면(sam-extra ≤ 0.33.x — 키를 조용히 버리고 전체 이미지에서 찾는다)
 * 'unsupported'. 파이썬 core/sam_extra_capabilities 가 이 키를 연결 경고에서는 뺀다(켤 때만 보내는 키).
 */
export function faceCropSupport(caps: FaceCropCapabilitiesLike | null | undefined): FaceCropSupport {
  if (!caps) return 'unknown'
  if (caps.status === 'not_applicable') return 'comfy'
  if (!caps.known || !caps.features?.sam3) return 'unknown'
  const missing = caps.sam3_keys?.missing_in_extension
  return Array.isArray(missing) && missing.includes(sam3FaceCropSettingKey('face_crop')) ? 'unsupported' : 'supported'
}
