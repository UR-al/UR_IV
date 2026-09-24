/**
 * 자동화 설정(모드별 파일 config/automation/automation_settings_<mode>.json)을 화면에 싣는 규칙 (감사 #42).
 *
 * 부팅 순서: Vue 는 getAutomationSettings 로 현재 모드의 파일 값을 **먼저** 당겨 채운 뒤에
 * set_automation_settings 로 동기화한다. 예전엔 하드코딩 기본값(limit 10 · repeat 1 · delay 1.0 …)을
 * 먼저 보내 Python 이 곧바로 파일에 써서, 사용자 값이 부팅마다 기본값으로 바뀌었다.
 *
 * 응답이 늦거나(3초 초과) 오기 전에 사용자가 한 칸을 고쳐도 같은 일이 났다 — 동기화가 객체 전체를
 * 보내 모르는 칸의 기본값까지 저장됐다(R2b#1). 그래서 동기화는 **화면이 실제로 아는 키**(서버에서
 * 받았거나 사용자가 고친 키)만 보내고(automationSyncPayload), Python 도 보낸 키만 합친다
 * (core/mode_aware_automation.merge_automation_payload). 시간 초과 뒤 늦게 온 응답도 고치지 않은
 * 칸에는 적용한다.
 */
import type { AutomationSettings } from '../types/bridge'

const MODES: ReadonlyArray<AutomationSettings['mode']> = ['count', 'timer', 'unlimited']

export type AutomationKey = keyof AutomationSettings
/** 파일에 영속되는 자동화 키 — Python _DEFAULT_AUTOMATION_SETTINGS 와 같은 목록 */
export const AUTOMATION_KEYS: readonly AutomationKey[] = [
  'mode', 'limit', 'repeat', 'delay', 'allowDupes', 'autoResetDeck', 'maxRetries', 'cleanupEveryN',
]

/** 패치의 키 중 자동화 키만 집합에 더한다(서버 응답·사용자 수정 기록용). */
export function markAutomationKeys(target: Set<AutomationKey>, patch: object | null | undefined): void {
  if (!patch) return
  for (const key of Object.keys(patch)) {
    if ((AUTOMATION_KEYS as readonly string[]).includes(key)) target.add(key as AutomationKey)
  }
}

/**
 * set_automation_settings 로 보낼 자동화 값 — `known`(서버에서 받았거나 사용자가 고친 키)만 싣는다.
 * 숫자가 잘못됐으면(입력 중인 빈 칸 등) 예전과 같은 안전값으로 보낸다.
 */
export function automationSyncPayload(settings: AutomationSettings, known: ReadonlySet<AutomationKey>): Partial<AutomationSettings> {
  const num = (v: unknown, ok: (n: number) => boolean, fallback: number) => {
    const n = Number(v)
    return Number.isFinite(n) && ok(n) ? n : fallback
  }
  const full: AutomationSettings = {
    mode: settings.mode,
    limit: num(settings.limit, n => n > 0, 10),
    repeat: num(settings.repeat, n => n > 0, 1),
    delay: num(settings.delay, n => n >= 0, 1),
    allowDupes: !!settings.allowDupes,
    autoResetDeck: !!settings.autoResetDeck,
    maxRetries: num(settings.maxRetries, n => n >= 0, 2),
    cleanupEveryN: num(settings.cleanupEveryN, n => n >= 0, 0),
  }
  const out: Partial<AutomationSettings> = {}
  for (const key of AUTOMATION_KEYS) {
    if (known.has(key)) (out as Record<string, unknown>)[key] = full[key]
  }
  return out
}

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** 서버 응답(automationSettingsLoaded · getAutomationSettings) → 화면에 합칠 조각. 잘못된 값은 뺀다. */
export function automationPatchFromServer(payload: unknown): Partial<AutomationSettings> {
  let d: unknown = payload
  if (typeof payload === 'string') {
    try { d = JSON.parse(payload || '{}') } catch { return {} }
  }
  if (!d || typeof d !== 'object' || Array.isArray(d)) return {}
  const s = d as Record<string, unknown>
  const patch: Partial<AutomationSettings> = {}
  if (typeof s.mode === 'string' && (MODES as readonly string[]).includes(s.mode)) patch.mode = s.mode as AutomationSettings['mode']
  if (finite(s.limit) && s.limit > 0) patch.limit = s.limit
  if (finite(s.repeat) && s.repeat > 0) patch.repeat = s.repeat
  if (finite(s.delay) && s.delay >= 0) patch.delay = s.delay
  if (typeof s.allowDupes === 'boolean') patch.allowDupes = s.allowDupes
  if (typeof s.autoResetDeck === 'boolean') patch.autoResetDeck = s.autoResetDeck
  if (finite(s.maxRetries) && s.maxRetries >= 0) patch.maxRetries = s.maxRetries
  if (finite(s.cleanupEveryN) && s.cleanupEveryN >= 0) patch.cleanupEveryN = s.cleanupEveryN
  return patch
}

/**
 * 부팅 hydrate 한 번 — 동기화는 onSettled 에서 한다.
 *  - 서버가 답하면 `onSettled(true)` 를 한 번 부른다. 이제 화면 값(파일 값 + 사용자가 고친 값)이 기준이다.
 *  - 답이 오기 전에 시간이 다 되면 먼저 `onSettled(false)` 를 한 번 부른다 — 호출자는 모르는 키를 보내지
 *    않는다. 자동 NL·Ollama 값을 Python 에 전하는 경로가 이것뿐이라 동기화 자체는 한다.
 *    그 뒤에 늦게 온 답도 버리지 않는다(적용 후 `onSettled(true)`).
 * 사용자가 이미 고친 키(`editedKeys`)는 파일 값으로 되돌리지 않고 나머지 키에만 적용한다 — 예전엔 한 칸만
 * 고쳐도 응답 전체를 버렸고, 시간 초과 뒤의 답도 버려 화면이 기본값으로 남았다.
 */
export function hydrateAutomationSettings(opts: {
  request: ((callback: (json: string) => void) => void) | null | undefined
  /** 사용자가 고친 자동화 키 — 응답이 올 때 읽는다 */
  editedKeys: () => ReadonlySet<string>
  apply: (patch: Partial<AutomationSettings>) => void
  onSettled: (hydrated: boolean) => void
  timeoutMs?: number
  setTimer?: (fn: () => void, ms: number) => unknown
}): void {
  const { request, editedKeys, apply, onSettled, timeoutMs = 3000, setTimer = (fn, ms) => setTimeout(fn, ms) } = opts
  let answered = false
  let gaveUp = false
  const giveUp = () => {
    if (answered || gaveUp) return
    gaveUp = true
    onSettled(false)
  }
  const receive = (json: string) => {
    if (answered) return
    answered = true
    const edited = editedKeys()
    const patch = automationPatchFromServer(json)
    for (const key of Object.keys(patch)) {
      if (edited.has(key)) delete (patch as Record<string, unknown>)[key]
    }
    if (Object.keys(patch).length) apply(patch)
    onSettled(true)
  }
  if (typeof request !== 'function') { giveUp(); return }
  setTimer(giveUp, timeoutMs)
  try {
    request((json: string) => receive(typeof json === 'string' ? json : JSON.stringify(json ?? {})))
  } catch {
    giveUp()
  }
}
