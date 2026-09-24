/**
 * Settings '기본값' 패널(config/tab_defaults.json)의 프론트 규칙 — 파이썬 core/tab_defaults.py 와 같은 키.
 *
 * 누가 읽나(audit #140 — 예전엔 대부분 아무도 안 읽었다):
 *  - steps/cfg/width/height/seed/sampler/scheduler → 시작 시 T2I 빈 칸(첫 실행이면 전부)
 *  - denoising → I2I Denoising 초기값
 *  - brushSize/effectStrength/yoloConf/snapRadius → 에디터 브러시·효과 세기·YOLO 신뢰도·스냅 반경
 *  - hires/ad/sam3_enabled → 첫 실행(prompt_settings.json 없음)에만
 *
 * 저장은 바뀐 키만 보낸다(diffTabDefaults) — keep-alive Settings 가 들고 있던 옛 값이 '전역 저장'이
 * 갱신한 T2I 기본값을 되덮던 양방향 덮어쓰기를 막는다. 파이썬이 기존 파일 위에 병합한다.
 * 값이 바뀌면 TAB_DEFAULTS_EVENT 로 알린다 — 에디터·I2I 는 사용자가 건드리지 않은 값만 따라간다.
 */

export interface TabDefaults {
  steps: number
  cfg: number
  width: number
  height: number
  seed: string
  sampler: string
  scheduler: string
  denoising: number
  brushSize: number
  effectStrength: number
  yoloConf: number
  snapRadius: number
  hires_enabled: boolean
  ad_enabled: boolean
  sam3_enabled: boolean
}

export const FACTORY_TAB_DEFAULTS: Readonly<TabDefaults> = Object.freeze({
  steps: 20, cfg: 7, width: 1024, height: 1024, seed: '-1',
  sampler: '', scheduler: '', denoising: 0.75,
  brushSize: 20, effectStrength: 15, yoloConf: 0.25, snapRadius: 12,
  hires_enabled: false, ad_enabled: false, sam3_enabled: false,
})

export const TAB_DEFAULT_KEYS = Object.keys(FACTORY_TAB_DEFAULTS) as (keyof TabDefaults)[]

/** 창 이벤트 — detail 은 정규화된 TabDefaults(저장 직후 값). */
export const TAB_DEFAULTS_EVENT = 'tabDefaultsChanged'

type Range = readonly [number, number]
const NUMBER_RANGES: Partial<Record<keyof TabDefaults, { range: Range; int: boolean }>> = {
  steps: { range: [1, 500], int: true },
  cfg: { range: [0, 100], int: false },
  width: { range: [64, 8192], int: true },
  height: { range: [64, 8192], int: true },
  denoising: { range: [0, 1], int: false },
  brushSize: { range: [1, 500], int: true },
  effectStrength: { range: [1, 100], int: true },
  yoloConf: { range: [0.01, 1], int: false },
  snapRadius: { range: [1, 100], int: true },
}
const BOOL_KEYS = new Set<keyof TabDefaults>(['hires_enabled', 'ad_enabled', 'sam3_enabled'])

function normalizeValue(key: keyof TabDefaults, raw: unknown): unknown {
  const spec = NUMBER_RANGES[key]
  if (spec) {
    const n = typeof raw === 'number' ? raw : Number(String(raw ?? '').trim())
    if (!Number.isFinite(n) || (typeof raw === 'string' && !raw.trim())) return undefined
    const clamped = Math.min(spec.range[1], Math.max(spec.range[0], n))
    return spec.int ? Math.round(clamped) : Math.round(clamped * 10000) / 10000
  }
  if (BOOL_KEYS.has(key)) {
    if (typeof raw === 'string') return ['1', 'true', 'yes', 'on'].includes(raw.trim().toLowerCase())
    return !!raw
  }
  if (key === 'seed') {
    const s = String(raw ?? '').trim()
    return s || '-1'
  }
  return String(raw ?? '').trim()
}

/** 파일/브리지 값 → 완전한 TabDefaults(없거나 깨진 키는 공장값). 폐기된 키는 버린다. */
export function normalizeTabDefaults(raw: unknown): TabDefaults {
  const out: TabDefaults = { ...FACTORY_TAB_DEFAULTS }
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out
  const src = raw as Record<string, unknown>
  for (const key of TAB_DEFAULT_KEYS) {
    if (!(key in src)) continue
    const v = normalizeValue(key, src[key])
    if (v !== undefined) (out as any)[key] = v
  }
  return out
}

/** synced(마지막으로 파일과 맞춘 값)와 다른 키만 — 저장 요청에 실을 부분 값. 바뀐 게 없으면 null. */
export function diffTabDefaults(current: TabDefaults, synced: TabDefaults): Partial<TabDefaults> | null {
  const out: Partial<TabDefaults> = {}
  for (const key of TAB_DEFAULT_KEYS) {
    const a = normalizeValue(key, current[key])
    if (a === undefined) continue   // 입력 중인 빈 숫자 칸 — 저장하지 않는다
    if (a !== normalizeValue(key, synced[key])) (out as any)[key] = a
  }
  return Object.keys(out).length ? out : null
}

/** 에디터가 쓰는 네 값 — YOLO 신뢰도는 EffectPanel 슬라이더 단위(%)로. */
export function editorDefaultsFrom(d: TabDefaults): { brushSize: number; effectStrength: number; detectConf: number; snapRadius: number } {
  return {
    brushSize: d.brushSize,
    effectStrength: d.effectStrength,
    detectConf: Math.min(100, Math.max(1, Math.round(d.yoloConf * 100))),
    snapRadius: d.snapRadius,
  }
}

/**
 * 기본값이 바뀌었을 때 화면 값을 따라가게 할지 — 사용자가 이전 기본값에서 손대지 않았을 때만.
 * (이미 바꾼 값을 설정 화면 저장이 덮어쓰면 안 된다.)
 */
export function followDefault<T>(current: T, previousDefault: T, nextDefault: T): T {
  return current === previousDefault ? nextDefault : current
}

type BackendLike = { getTabDefaults?: (cb: (json: string) => void) => void }

/** 브리지에서 읽기 — 실패하면 공장값. */
export function readTabDefaults(backend: BackendLike | null | undefined): Promise<TabDefaults> {
  return new Promise(resolve => {
    if (!backend || typeof backend.getTabDefaults !== 'function') { resolve({ ...FACTORY_TAB_DEFAULTS }); return }
    try {
      backend.getTabDefaults((json: string) => {
        try { resolve(normalizeTabDefaults(JSON.parse(json || '{}'))) } catch { resolve({ ...FACTORY_TAB_DEFAULTS }) }
      })
    } catch {
      resolve({ ...FACTORY_TAB_DEFAULTS })
    }
  })
}

/** 저장 직후 다른 화면에 알린다. */
export function announceTabDefaults(d: TabDefaults, target: Pick<Window, 'dispatchEvent'> | null =
  (typeof window === 'undefined' ? null : window)): void {
  try { target?.dispatchEvent(new CustomEvent(TAB_DEFAULTS_EVENT, { detail: { ...d } })) } catch {}
}
