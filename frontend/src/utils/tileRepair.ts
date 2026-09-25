import type { TileRepairOptions, TileRepairResultEvent, TileRepairSettings } from '../types/bridge'

/**
 * Anima Tile & Repair 카드(components/TileRepairPanel.vue)의 순수 로직 — 기본값·범위·이벤트 읽기.
 *
 * 기본값·범위는 원본 그대로이고 확장 패널·라우트(sam-extra sam3ext/tile_repair_api.py)와 같다.
 * 파이썬 core/tile_repair_request.py 의 DEFAULTS·RANGES 와도 같아야 한다(tests/test_tile_repair_request.py 가
 * 이 파일을 읽어 대조한다).
 *  - origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134
 *    (--negative_prompt "", --infer_steps 50, --guidance_scale 3.5, --flow_shift 5.0)
 *  - origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:167-170 (--lllite_multiplier 1.0)
 *  - origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:130 (strength −10..10 step .01)
 */

/** 확장 패널의 프롬프트 기본값(sam3ext/ui_anima.py) — 라우트도 같은 값을 쓴다. */
export const TILE_REPAIR_DEFAULT_PROMPT =
  'repair the low-quality anime image, reduce blur and compression artifacts, preserve the original composition'

export type TileRepairNumberKey = 'steps' | 'cfg_scale' | 'flow_shift' | 'multiplier' | 'short_side'

export const TILE_REPAIR_RANGES: Readonly<Record<TileRepairNumberKey, readonly [number, number]>> = {
  steps: [1, 150],
  cfg_scale: [0, 20],
  flow_shift: [0, 30],
  multiplier: [-10, 10],
  short_side: [256, 4096],
}

export const TILE_REPAIR_INCREMENTS: Readonly<Record<TileRepairNumberKey, number>> = {
  steps: 1, cfg_scale: 0.1, flow_shift: 0.1, multiplier: 0.01, short_side: 32,
}

export function defaultTileRepairSettings(): TileRepairSettings {
  return {
    model: '',
    prompt: TILE_REPAIR_DEFAULT_PROMPT,
    negative_prompt: '',
    steps: 50,
    cfg_scale: 3.5,
    flow_shift: 5.0,
    multiplier: 1.0,
    short_side: 1024,
    seed: -1,
    dit: '',
    text_encoder: '',
    vae: '',
    unload_forge_before: true,
  }
}

const INTEGER_KEYS: ReadonlySet<TileRepairNumberKey> = new Set(['steps', 'short_side'])

/** 입력칸 값 → 범위 안의 숫자(정수 칸은 반올림). 숫자가 아니면 null. */
export function clampTileRepairNumber(key: TileRepairNumberKey, value: unknown): number | null {
  const number = typeof value === 'number' ? value : Number(String(value ?? '').trim())
  if (!Number.isFinite(number) || String(value ?? '').trim() === '') return null
  const [low, high] = TILE_REPAIR_RANGES[key]
  const rounded = INTEGER_KEYS.has(key) ? Math.round(number) : number
  return Math.min(high, Math.max(low, rounded))
}

/** 시드 입력 → -1(랜덤) 또는 0 이상의 정수. 알아볼 수 없으면 -1. */
export function normalizeTileRepairSeed(value: unknown): number {
  const text = String(value ?? '').trim()
  if (!/^-?\d+$/.test(text)) return -1
  const seed = Number(text)
  return Number.isSafeInteger(seed) && seed >= 0 ? seed : -1
}

/** 보낼 원본: 로컬 경로가 있으면 경로, 없으면 업로드 data URL. 둘 다 없으면 null. */
export function tileRepairSource(imagePath: string, imageSrc: string): { image_path: string; image: string } | null {
  const path = String(imagePath || '').trim()
  if (path) return { image_path: path, image: '' }
  const src = String(imageSrc || '')
  if (/^data:image\/(png|jpeg|webp);base64,/.test(src)) return { image_path: '', image: src }
  return null
}

/** tileRepairResult 페이로드(JSON 문자열) → 이벤트. 깨졌거나 모양이 다르면 null. */
export function parseTileRepairEvent(json: unknown): TileRepairResultEvent | null {
  let value: any = json
  if (typeof json === 'string') {
    try { value = JSON.parse(json) } catch { return null }
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  if (!['tile_repair_options', 'tile_repair_run', 'tile_repair_cancel'].includes(value.action)) return null
  if (typeof value.requestId !== 'string' || typeof value.ok !== 'boolean') return null
  return value as TileRepairResultEvent
}

/** 옵션 응답의 모양 확인 — 목록이 배열이 아니면 null. */
export function readTileRepairOptions(value: unknown): TileRepairOptions | null {
  const options = value as TileRepairOptions | null
  if (!options || typeof options !== 'object') return null
  for (const key of ['models', 'dit', 'text_encoder', 'vae'] as const) {
    if (!Array.isArray(options[key]) || options[key].some(item => typeof item !== 'string')) return null
  }
  return options
}

/** 모델 드롭다운의 첫 칸('' = 확장 기본값) 이름. */
export function defaultChoiceLabel(defaultName: string | null | undefined): string {
  return defaultName ? `기본값 (${defaultName})` : '기본값 (확장이 고름)'
}
