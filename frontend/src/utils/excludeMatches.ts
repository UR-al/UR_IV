// 제외 규칙 매니저 — 규칙이 지우는 태그 목록 조회를 규칙마다 한 번만 한다.
//
// 백엔드 getExcludeMatches 는 GUI 스레드에서 태그 사전(약 77만 개)을 훑는다. 같은 규칙을
// 다시 누를 때마다 부르면 그때마다 창이 100ms 넘게 멈칫한다. 태그 사전은 세션 동안
// 바뀌지 않으므로 받은 결과는 규칙 문자열로 캐시하고, 조회 중인 규칙은 다시 보내지 않는다.

/** 조회 중 표시가 이보다 오래되면(콜백 유실) 다시 보낼 수 있게 풀어 준다. */
export const EXCLUDE_MATCH_PENDING_TTL_MS = 15_000

export interface ExcludeMatchRequests {
  /** 캐시에 없고 조회 중도 아니면 true — 호출자가 백엔드를 부르고 끝나면 end() 한다. */
  begin(rule: string, cache: Record<string, unknown>, now?: number): boolean
  end(rule: string): void
  isPending(rule: string, now?: number): boolean
}

export function createExcludeMatchRequests(ttlMs: number = EXCLUDE_MATCH_PENDING_TTL_MS): ExcludeMatchRequests {
  const pending = new Map<string, number>()

  function isPending(rule: string, now: number = Date.now()): boolean {
    const startedAt = pending.get(rule)
    if (startedAt === undefined) return false
    if (now - startedAt > ttlMs) {
      pending.delete(rule)
      return false
    }
    return true
  }

  return {
    begin(rule, cache, now = Date.now()) {
      if (!rule) return false
      if (Object.prototype.hasOwnProperty.call(cache, rule)) return false
      if (isPending(rule, now)) return false
      pending.set(rule, now)
      return true
    },
    end(rule) {
      pending.delete(rule)
    },
    isPending,
  }
}

/** 백엔드 응답(JSON 배열) → 태그 목록. 오류 객체·깨진 JSON 은 null(캐시하지 않음). */
export function parseExcludeMatches(json: unknown): string[] | null {
  let value: unknown = json
  if (typeof json === 'string') {
    try { value = JSON.parse(json) } catch { return null }
  }
  if (!Array.isArray(value)) return null
  return value.filter((tag): tag is string => typeof tag === 'string')
}
