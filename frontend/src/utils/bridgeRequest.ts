/**
 * 요청 id 로 짝을 맞추는 '마지막 요청만 유효' 브리지 요청 — `backend.requestX(..., id)` + `xReady` 이벤트.
 *
 * GUI 스레드를 막던 동기 슬롯(getLoras·fetchCharacterTagsOnline·exportCompareGif)을 비동기로 바꾸면서
 * 쓴다. Python 은 결과 JSON 에 `requestId` 를 그대로 되돌려 준다. 이 헬퍼는
 *  - 요청마다 고유 id 를 만든다(웹 모드는 시그널이 모든 클라이언트에 방송되므로 무작위 꼬리를 붙인다),
 *  - 새 요청을 시작하면 앞선 대기를 null 로 끝낸다(늦게 온 옛 결과가 새 화면을 덮지 않게),
 *  - 다른 id 의 응답(옛 요청·다른 클라이언트)은 무시하고,
 *  - 응답이 오지 않으면 timeoutMs 뒤 null 로 끝낸다('불러오는 중'이 영영 켜져 있지 않게).
 *
 * null 로 끝난 이유는 요청마다 `outcome()` 으로 가린다 — 'timeout' 만 사용자에게 알릴 실패다.
 * 'superseded'(더 새 요청이 이어받음)·'cancelled'(화면이 닫힘)는 조용히 끝내야 한다. 예전에는
 * cancel() 이 시간 초과와 똑같이 null 로 끝나, 조회 중 모달을 닫으면 버린 조회에 대해
 * '응답 없음' 오류 토스트가 떴다.
 */

export interface BridgeReply { requestId?: string }

/** 한 요청이 어떻게 끝났는가. */
export type RequestOutcome = 'pending' | 'replied' | 'timeout' | 'superseded' | 'cancelled'

export interface LatestRequestTicket<T> {
  id: string
  /** 응답이면 그 객체, 아니면 null(시간 초과·새 요청·취소 — 이유는 outcome()) */
  done: Promise<T | null>
  /** 이 요청이 끝난 이유. done 이 풀리기 전에 정해진다. */
  outcome(): RequestOutcome
}

/** 사용자가 버렸거나(취소) 더 새 요청이 이어받은(대체) 요청 — 결과·오류를 보여 주지 않는다. */
export function wasAbandoned(outcome: RequestOutcome): boolean {
  return outcome === 'superseded' || outcome === 'cancelled'
}

export interface LatestRequestOptions {
  timeoutMs: number
  /** id 앞머리 — 로그에서 어느 요청인지 보이게 */
  prefix?: string
  /** 결과보다 시간 초과가 먼저 왔을 때 (null 로 끝나기 직전에 한 번) */
  onTimeout?: () => void
  setTimer?: (callback: () => void, ms: number) => unknown
  clearTimer?: (handle: any) => void
  /** 테스트용 id 꼬리 생성기 */
  randomSuffix?: () => string
}

export interface LatestRequest<T extends BridgeReply> {
  /** 새 요청을 시작한다. 앞선 대기는 null 로 끝난다(그 요청의 outcome 은 'superseded'). */
  begin(): LatestRequestTicket<T>
  /** 이벤트 페이로드(JSON 문자열 또는 객체)를 받는다. 지금 요청의 응답이면 끝내고 true. */
  receive(payload: string | T | null | undefined): boolean
  /** 대기를 버린다(화면 닫힘 등). 그 요청의 outcome 은 'cancelled' — 시간 초과와 구별된다. */
  cancel(): void
  readonly pending: boolean
  readonly currentId: string
}

/** 브리지 이벤트 JSON → 객체. 깨졌거나 객체가 아니면 null. */
export function parseBridgeReply<T extends object = Record<string, unknown>>(payload: unknown): T | null {
  if (payload && typeof payload === 'object') return payload as T
  if (typeof payload !== 'string' || !payload) return null
  try {
    const value = JSON.parse(payload)
    return value && typeof value === 'object' && !Array.isArray(value) ? value as T : null
  } catch {
    return null
  }
}

export function createLatestRequest<T extends BridgeReply>(options: LatestRequestOptions): LatestRequest<T> {
  const setTimer = options.setTimer ?? ((callback: () => void, ms: number) => setTimeout(callback, ms))
  const clearTimer = options.clearTimer ?? ((handle: any) => clearTimeout(handle))
  const randomSuffix = options.randomSuffix ?? (() => Math.random().toString(36).slice(2, 8))
  const prefix = options.prefix || 'req'
  let counter = 0
  let currentId = ''
  let resolve: ((value: T | null) => void) | null = null
  // 지금 대기 중인 요청의 결말을 적는 곳 — 요청(티켓)마다 따로라, 뒤이은 begin() 이 앞 요청의
  // 결말을 덮지 않는다.
  let markOutcome: ((outcome: RequestOutcome) => void) | null = null
  let timer: unknown = null

  function settle(value: T | null, outcome: RequestOutcome): void {
    if (!resolve) return
    const done = resolve
    const mark = markOutcome
    resolve = null
    markOutcome = null
    currentId = ''
    if (timer !== null) { clearTimer(timer); timer = null }
    mark?.(outcome)   // done 이 풀리기 전에 — await 뒤 outcome() 이 늘 최종 값을 본다
    done(value)
  }

  function begin(): LatestRequestTicket<T> {
    settle(null, 'superseded')
    counter += 1
    const id = `${prefix}-${Date.now().toString(36)}-${counter}-${randomSuffix()}`
    let outcome: RequestOutcome = 'pending'
    const done = new Promise<T | null>((res) => { resolve = res })
    markOutcome = (value) => { outcome = value }
    currentId = id
    timer = setTimer(() => {
      if (currentId !== id || !resolve) return   // 이미 끝났거나 새 요청으로 바뀌었다
      timer = null
      options.onTimeout?.()
      settle(null, 'timeout')
    }, options.timeoutMs)
    return { id, done, outcome: () => outcome }
  }

  function receive(payload: string | T | null | undefined): boolean {
    if (!resolve) return false
    const reply = parseBridgeReply<T>(payload)
    if (!reply || reply.requestId !== currentId) return false
    settle(reply, 'replied')
    return true
  }

  return {
    begin,
    receive,
    cancel: () => settle(null, 'cancelled'),
    get pending() { return resolve !== null },
    get currentId() { return currentId },
  }
}
