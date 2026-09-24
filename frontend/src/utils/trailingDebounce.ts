/**
 * 뒤쪽(trailing) 디바운스 — 마지막 호출 뒤 waitMs 동안 잠잠하면 그 마지막 인자로 한 번 실행한다.
 *
 * 슬라이더 드래그처럼 틱마다 들어오는 영속 저장용(감사 #130): LoRA 가중치·고해상도 배율 슬라이더는
 * input 이벤트마다 save_ui_prefs 를 보내 GUI 스레드가 ui_prefs.json 을 읽고-고치고-다시 썼다(회당 ~7ms).
 * 생성 경로가 곧바로 읽는 Python 런타임 값(set_lora_stack·set_high_res_factor)은 디바운스하지 않는다 —
 * 드래그 직후 누른 생성이 옛 값으로 나가면 안 된다. 이건 '파일 저장'만 모은다.
 *
 * flush() 는 기다리는 호출을 지금 실행한다(페이지를 떠나기 직전 등). cancel() 은 버린다.
 */
export interface TrailingDebounced<A extends unknown[]> {
  (...args: A): void
  /** 기다리는 호출이 있으면 지금 실행하고 true */
  flush(): boolean
  /** 기다리는 호출을 버린다 */
  cancel(): void
  /** 기다리는 호출이 있는가 */
  pending(): boolean
}

export interface TimerApi {
  set: (fn: () => void, ms: number) => unknown
  clear: (handle: unknown) => void
}

const defaultTimers: TimerApi = {
  set: (fn, ms) => setTimeout(fn, ms),
  clear: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
}

export function createTrailingDebounce<A extends unknown[]>(
  fn: (...args: A) => void,
  waitMs: number,
  timers: TimerApi = defaultTimers,
): TrailingDebounced<A> {
  let handle: unknown = null
  let lastArgs: A | null = null

  const run = () => {
    handle = null
    const args = lastArgs
    lastArgs = null
    if (args) fn(...args)
  }

  const debounced = ((...args: A) => {
    lastArgs = args
    if (handle !== null) timers.clear(handle)
    handle = timers.set(run, waitMs)
  }) as TrailingDebounced<A>

  debounced.flush = () => {
    if (handle === null) return false
    timers.clear(handle)
    run()
    return true
  }
  debounced.cancel = () => {
    if (handle !== null) timers.clear(handle)
    handle = null
    lastArgs = null
  }
  debounced.pending = () => handle !== null
  return debounced
}

/**
 * 페이지를 떠날 때(창 닫기·새로고침·웹 탭 닫기) 기다리는 저장을 흘려보낸다. 등록 해제 함수를 돌려준다.
 * pagehide 가 없는 환경(테스트)에선 아무것도 하지 않는다.
 */
export function flushOnPageHide(flush: () => unknown): () => void {
  try {
    if (typeof window === 'undefined' || typeof window.addEventListener !== 'function') return () => {}
    const handler = () => { try { flush() } catch { /* 떠나는 중 — 무시 */ } }
    window.addEventListener('pagehide', handler)
    window.addEventListener('beforeunload', handler)
    return () => {
      window.removeEventListener('pagehide', handler)
      window.removeEventListener('beforeunload', handler)
    }
  } catch {
    return () => {}
  }
}
