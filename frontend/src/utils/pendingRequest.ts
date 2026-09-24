/**
 * 결과 이벤트나 시간 초과 중 먼저 온 쪽으로 끝나는 '한 번에 하나' 대기 요청.
 *
 * 태그→자연어 변환(App.vue, genNlResult)이 쓴다. 대기마다 자기 타이머를 소유하고 끝날 때
 * 해제한다 — 예전엔 65초 타이머 id 를 버리고 전역 resolve 만 확인해서, 일찍 끝난 변환 A 의
 * 타이머가 A 시작 65초 뒤에 아직 대기 중인 변환 B 를 '시간 초과'로 끝냈다(B 결과는 버려짐).
 */

export interface PendingRequestOptions {
  timeoutMs: number
  /** 대기 시작(true)·종료(false) — 버튼의 '변환 중' 표시 등 */
  onPendingChange?: (pending: boolean) => void
  /** 결과보다 시간 초과가 먼저 왔을 때 (종료 직전에 한 번) */
  onTimeout?: () => void
  setTimer?: (callback: () => void, ms: number) => unknown
  clearTimer?: (handle: any) => void
}

export interface PendingRequest<T> {
  /** 새 대기를 시작한다. 앞선 대기가 남아 있으면 null 로 먼저 끝낸다(타이머도 해제). */
  start(): Promise<T | null>
  /** 현재 대기를 끝낸다. 대기 중이 아니면 무시하고 false. */
  finish(value: T | null): boolean
  readonly pending: boolean
}

export function createPendingRequest<T>(options: PendingRequestOptions): PendingRequest<T> {
  const setTimer = options.setTimer ?? ((callback: () => void, ms: number) => setTimeout(callback, ms))
  const clearTimer = options.clearTimer ?? ((handle: any) => clearTimeout(handle))
  let resolve: ((value: T | null) => void) | null = null
  let timer: unknown = null
  let generation = 0

  function finish(value: T | null): boolean {
    if (!resolve) return false
    const settle = resolve
    resolve = null
    if (timer !== null) { clearTimer(timer); timer = null }
    options.onPendingChange?.(false)
    settle(value)
    return true
  }

  function start(): Promise<T | null> {
    finish(null)
    const mine = ++generation
    return new Promise<T | null>((res) => {
      resolve = res
      options.onPendingChange?.(true)
      timer = setTimer(() => {
        // 해제된 타이머가 늦게 와도(환경에 따라) 다른 대기를 끝내지 않는다
        if (mine !== generation || !resolve) return
        timer = null
        options.onTimeout?.()
        finish(null)
      }, options.timeoutMs)
    })
  }

  return {
    start,
    finish,
    get pending() { return resolve !== null },
  }
}
