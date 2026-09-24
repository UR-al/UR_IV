import { onMounted, onUnmounted } from 'vue'
import { getBackend } from '../bridge.js'
import {
  FACTORY_TAB_DEFAULTS,
  TAB_DEFAULTS_EVENT,
  normalizeTabDefaults,
  readTabDefaults,
  type TabDefaults,
} from '../utils/tabDefaults'

/** next 기본값을 화면에 반영한다. prev 는 직전에 반영한 기본값(처음엔 공장값). */
export type TabDefaultsApply = (next: TabDefaults, prev: TabDefaults) => void

type EventTargetLike = Pick<Window, 'addEventListener' | 'removeEventListener'>

/**
 * 기본값을 읽어 한 번 반영하고, Settings 가 저장할 때마다(TAB_DEFAULTS_EVENT) 다시 반영한다.
 * 반영 함수는 followDefault 로 '사용자가 손대지 않은 값'만 바꿔야 한다. 반환: 정지 함수.
 */
export function startTabDefaultsFollower(apply: TabDefaultsApply, deps: {
  loadBackend: () => Promise<unknown>
  target?: EventTargetLike | null
}): () => void {
  let prev: TabDefaults = { ...FACTORY_TAB_DEFAULTS }
  let stopped = false
  const target = deps.target === undefined ? (typeof window === 'undefined' ? null : window) : deps.target
  const push = (next: TabDefaults) => {
    if (stopped) return
    apply(next, prev)
    prev = next
  }
  const onEvent = (event: Event) => push(normalizeTabDefaults((event as CustomEvent).detail))
  target?.addEventListener(TAB_DEFAULTS_EVENT, onEvent)
  void deps.loadBackend()
    .then(backend => readTabDefaults(backend as any))
    .then(push)
    .catch(() => {})
  return () => {
    stopped = true
    target?.removeEventListener(TAB_DEFAULTS_EVENT, onEvent)
  }
}

/** 컴포넌트용 — 마운트 때 시작하고 언마운트 때 멈춘다. */
export function useTabDefaultsFollower(apply: TabDefaultsApply): void {
  let stop: (() => void) | null = null
  onMounted(() => { stop = startTabDefaultsFollower(apply, { loadBackend: getBackend }) })
  onUnmounted(() => { stop?.(); stop = null })
}
