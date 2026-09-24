/**
 * 크래시 복구 백업(cache/session/session_backup.json)의 순수 규칙 (감사 #160).
 *
 * 예전 조건 '현재 메인 프롬프트가 비었을 때만 제안'은 부팅의 load_settings 가 prompt_settings.json
 * 으로 프롬프트를 늘 채우므로 사실상 성립하지 않았고, 곧이어 2.5초/30초 저장이 백업을 낡은 값으로
 * 덮어 크래시 직전 편집을 영구히 지웠다. 이제
 * - 정상 종료는 백업에 ``clean: true`` 를 남긴다(Python core/session_backup.mark_session_clean).
 * - ``clean`` 이 아니고, 백업 프롬프트가 있으며, 지금 화면 값(메인·네거티브)과 다르면 제안한다.
 * - 사용자가 고르기 전엔 백업을 덮지 않는다(composables/useSessionRestore.ts).
 * mtime 은 쓰지 않는다 — 부팅 중 연결 경로의 save_settings 가 prompt_settings.json 을 새로 쓴다.
 */

export interface SessionBackup {
  tab?: string
  prompt?: string
  negative?: string
  /** 정상 종료 표시 — true 면 복구할 것이 없다 */
  clean?: boolean
  /** 저장 시각(ms) — 표시·진단용 */
  savedAt?: number
  [key: string]: unknown
}

export interface SessionPrompts {
  prompt?: string | null
  negative?: string | null
}

const text = (value: unknown): string => (typeof value === 'string' ? value.trim() : '')

/** JSON 응답 → 백업 dict. 깨졌거나 오류 응답({error})이면 null. */
export function parseSessionBackup(json: unknown): SessionBackup | null {
  let data: unknown = json
  if (typeof json === 'string') {
    try { data = JSON.parse(json || '{}') } catch { return null }
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null
  if ('error' in (data as Record<string, unknown>)) return null
  return data as SessionBackup
}

/** 복구를 제안할까 — 정상 종료가 아니고, 백업 프롬프트가 있고, 지금 화면과 다를 때. */
export function shouldOfferSessionRestore(
  backup: SessionBackup | null | undefined,
  current: SessionPrompts,
): boolean {
  if (!backup || backup.clean === true) return false
  const savedPrompt = text(backup.prompt)
  if (!savedPrompt) return false
  return savedPrompt !== text(current.prompt) || text(backup.negative) !== text(current.negative)
}

/** 저장할 백업 한 벌 — clean 은 Python 이 늘 false 로 쓴다. */
export function buildSessionBackup(
  tab: unknown,
  prompts: SessionPrompts,
  now: number = Date.now(),
): SessionBackup {
  return {
    tab: typeof tab === 'string' && tab ? tab : 't2i',
    prompt: typeof prompts.prompt === 'string' ? prompts.prompt : '',
    negative: typeof prompts.negative === 'string' ? prompts.negative : '',
    savedAt: now,
  }
}
