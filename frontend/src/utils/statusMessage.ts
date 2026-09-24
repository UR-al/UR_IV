/**
 * 하단 계기 스트립의 상태 한 줄 — Python `show_status` → `statusMessage` 이벤트 페이로드.
 *
 * 페이로드(core/status_message.py): `{ text, level, timeoutMs, at }`
 *  - level: info | success | warning | error (모르면 info)
 *  - timeoutMs: 0 이면 다음 문구가 올 때까지 남긴다(sticky)
 *  - at: 보낸 시각(epoch ms). 늦게 붙은 화면이 `getStatusMessage` 로 마지막 문구를 읽을 때만
 *    남은 시간을 계산하는 데 쓴다 — 실시간 이벤트는 받은 순간부터 timeoutMs 를 센다
 *    (웹 모드의 다른 기기는 시계가 다를 수 있다).
 */

export type StatusLevel = 'info' | 'success' | 'warning' | 'error'

export interface StatusMessage {
  text: string
  level: StatusLevel
  timeoutMs: number
  at: number
}

const LEVELS: ReadonlySet<string> = new Set(['info', 'success', 'warning', 'error'])

/** 이벤트/슬롯 JSON → StatusMessage. 비었거나 깨졌으면 null. */
export function parseStatusMessage(payload: unknown): StatusMessage | null {
  let value: any = payload
  if (typeof payload === 'string') {
    if (!payload) return null
    try { value = JSON.parse(payload) } catch { return null }
  }
  if (!value || typeof value !== 'object') return null
  const text = typeof value.text === 'string' ? value.text.trim() : ''
  if (!text) return null
  const level = LEVELS.has(value.level) ? value.level as StatusLevel : 'info'
  const timeout = Number(value.timeoutMs)
  const at = Number(value.at)
  return {
    text,
    level,
    timeoutMs: Number.isFinite(timeout) && timeout > 0 ? timeout : 0,
    at: Number.isFinite(at) && at > 0 ? at : 0,
  }
}

/**
 * 표시할 남은 시간(ms). sticky 면 Infinity, 이미 지났으면 0.
 * `live` 면(방금 받은 이벤트) 받은 순간부터 센다. 아니면(늦게 읽은 마지막 문구) `at` 기준이고,
 * 시계가 어긋나도 timeoutMs 를 넘지 않게 자른다.
 */
export function statusRemainingMs(message: StatusMessage, now: number, live: boolean): number {
  if (message.timeoutMs <= 0) return Number.POSITIVE_INFINITY
  if (live || !message.at) return message.timeoutMs
  const remaining = message.at + message.timeoutMs - now
  return Math.max(0, Math.min(message.timeoutMs, remaining))
}

export interface StatusLineOptions {
  /** 보이는 한 줄이 바뀔 때마다(지워질 때는 null) */
  onChange: (message: StatusMessage | null) => void
  now?: () => number
  setTimer?: (callback: () => void, ms: number) => unknown
  clearTimer?: (handle: any) => void
}

export interface StatusLine {
  /** 실시간 `statusMessage` 이벤트 — 받은 순간부터 timeoutMs 를 센다. */
  live(payload: unknown): void
  /** 마운트 때 `getStatusMessage` 로 읽은 마지막 문구. 실시간 문구가 이미 왔으면 더 새로우니 버린다. */
  replay(payload: unknown): void
  /** 화면이 사라질 때 — 타이머를 풀고 이후 입력을 무시한다. */
  dispose(): void
  readonly current: StatusMessage | null
}

/**
 * 상태 한 줄의 표시 규칙(Vue 비의존) — useStatusMessage 가 onBackendEvent/getStatusMessage 에 잇는다.
 * - 새 문구가 옛 문구를 덮고, 옛 문구의 만료 타이머는 새 문구를 지우지 않는다.
 * - sticky(timeoutMs 0)는 다음 문구가 올 때까지 남는다.
 * - 늦게 읽은 문구(replay)는 남은 시간만큼만 보이고, 이미 지났으면 보이지 않는다.
 */
export function createStatusLine(options: StatusLineOptions): StatusLine {
  const now = options.now ?? (() => Date.now())
  const setTimer = options.setTimer ?? ((callback: () => void, ms: number) => setTimeout(callback, ms))
  const clearTimer = options.clearTimer ?? ((handle: any) => clearTimeout(handle))
  let current: StatusMessage | null = null
  let timer: unknown = null
  let receivedLive = false
  let disposed = false

  function stopTimer(): void {
    if (timer !== null) { clearTimer(timer); timer = null }
  }

  function show(next: StatusMessage, isLive: boolean): void {
    const remaining = statusRemainingMs(next, now(), isLive)
    if (remaining <= 0) return
    stopTimer()
    current = next
    options.onChange(next)
    if (Number.isFinite(remaining)) {
      timer = setTimer(() => {
        timer = null
        if (current !== next) return
        current = null
        options.onChange(null)
      }, remaining)
    }
  }

  return {
    live(payload: unknown) {
      if (disposed) return
      const next = parseStatusMessage(payload)
      if (!next) return
      receivedLive = true
      show(next, true)
    },
    replay(payload: unknown) {
      if (disposed || receivedLive) return
      const last = parseStatusMessage(payload)
      if (last) show(last, false)
    },
    dispose() {
      disposed = true
      stopTimer()
    },
    get current() { return current },
  }
}
