import { getCurrentScope, onScopeDispose, ref, shallowRef, watch, type Ref } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction, useWidgetStore } from '../stores/widgetStore.js'
import type {
  RelatedTagsBuildEvent, RelatedTagsBuildStatus, RelatedTagsQueryEvent, RelatedTagsQueryState, RelatedTagsShard,
  RelatedTagSuggestion,
} from '../types/bridge'
import { createLatestRequest, type LatestRequestOptions } from '../utils/bridgeRequest'
import { buildRelatedTagsQuery, parseRelatedTagsEvent, querySignature } from '../utils/relatedTags'
import { createTrailingDebounce, type TimerApi } from '../utils/trailingDebounce'
import type { OnBackendEventFn, RequestActionFn } from './managerDeps'
import { persistUiPrefs } from './uiPrefs'
import { ratingFilterRevision } from './useRatingFilter.js'

/**
 * 연관 태그 패널 상태 — 프롬프트가 바뀌면(패널이 열려 있을 때만) 잠깐 기다렸다 한 번 묻는다.
 *
 * 브리지(tests/test_bridge_contract.py 가 이름을 지킨다): related_tags_query · related_tags_build ·
 * related_tags_cancel, 시그널 relatedTagsEvent(ui/related_tags_actions.py). 질의는 requestId 로 마지막 것만
 * 받고(createLatestRequest), 표 만들기 진행은 방송이라 어느 id 든 받는다. 표는 사용자가 버튼을 눌러야 만든다.
 * 열림 상태는 ui_prefs.relatedTagsOpen 에 저장한다(기본 닫힘).
 */
export interface RelatedTagsDeps {
  onBackendEvent: OnBackendEventFn
  requestAction: RequestActionFn
  /** 위젯 스토어(반응형) — 프롬프트 칸 값 */
  widgets: Record<string, unknown>
  persistUiPrefs: (payload: Record<string, unknown>) => void
  /** 테스트용 build id 생성기 */
  newId?: () => string
  debounceMs?: number
  timers?: TimerApi
  /** 질의 응답 대기(테스트용 타이머 포함) */
  request?: Partial<LatestRequestOptions>
  now?: () => number
  /** 등급 필터를 Python 에 보낼 때마다 바뀌는 값 — 바뀌면 같은 프롬프트라도 다시 묻는다(등급은 Python 이 정한다) */
  ratingRevision?: Readonly<Ref<number>>
}

export type RelatedTagsViewState = 'idle' | RelatedTagsQueryState

export const RELATED_TAGS_TIMEOUT_MESSAGE = '연관 태그 응답이 없습니다. 패널을 닫았다 다시 열어 주세요.'
/** 진행 이벤트(0.5초마다)가 이만큼 끊긴 뒤에야 '진행 없음' 질의 답이 진행 표시를 지운다. */
export const BUILD_QUIET_MS = 3000

let counter = 0
function defaultId(): string {
  counter += 1
  return `rtb-${Date.now().toString(36)}-${counter}-${Math.random().toString(36).slice(2, 8)}`
}

export function createRelatedTags(deps: RelatedTagsDeps) {
  const newId = deps.newId ?? defaultId
  const now = deps.now ?? (() => Date.now())
  const ratingRevision = deps.ratingRevision ?? ref(0)
  const open = ref(false)
  const loading = ref(false)
  const state = ref<RelatedTagsViewState>('idle')
  const tags = shallowRef<RelatedTagSuggestion[]>([])
  const context = shallowRef<string[]>([])
  const ratings = shallowRef<RelatedTagsShard[]>([])
  const missing = shallowRef<RelatedTagsShard[]>([])
  const stale = shallowRef<RelatedTagsShard[]>([])
  const unavailable = shallowRef<RelatedTagsShard[]>([])
  const estimateSec = ref(0)
  const build = ref<RelatedTagsBuildStatus | null>(null)
  const cancelling = ref(false)
  const error = ref('')
  const feedback = ref('')
  const latest = createLatestRequest<RelatedTagsQueryEvent>({ timeoutMs: 20000, prefix: 'rtq', ...deps.request })
  let lastSignature = ''
  let buildSeenAt = 0
  let touched = false
  let disposed = false

  function apply(event: RelatedTagsQueryEvent) {
    state.value = event.state
    tags.value = event.tags
    context.value = event.context
    ratings.value = event.ratings
    missing.value = event.missing
    stale.value = event.stale
    unavailable.value = event.unavailable
    estimateSec.value = event.estimateSec
    error.value = event.error || ''
    if (event.build) {
      build.value = event.build
      buildSeenAt = now()
    } else if (build.value && now() - buildSeenAt > BUILD_QUIET_MS) {
      // 진행 이벤트보다 먼저 계산된 답이 막 시작한 표 만들기를 지우지 않게 — 한동안 조용할 때만 지운다
      build.value = null
      cancelling.value = false
    }
  }

  async function refresh(force = false) {
    if (!open.value || disposed) return
    debounced.cancel()
    const payload = buildRelatedTagsQuery(deps.widgets, '')
    const signature = currentSignature()
    if (!force && signature === lastSignature) return
    lastSignature = signature
    const ticket = latest.begin()
    loading.value = true
    deps.requestAction('related_tags_query', { ...payload, requestId: ticket.id })
    const reply = await ticket.done
    if (disposed) return
    if (reply) {
      loading.value = false
      apply(reply)
    } else if (ticket.outcome() === 'timeout') {
      loading.value = false
      lastSignature = ''        // 같은 프롬프트로 다시 물을 수 있게
      state.value = 'error'
      error.value = RELATED_TAGS_TIMEOUT_MESSAGE
    } else if (ticket.outcome() === 'cancelled') {
      loading.value = false
    }
  }

  function currentSignature(): string {
    return `${querySignature(buildRelatedTagsQuery(deps.widgets, ''))}|r${ratingRevision.value}`
  }

  const debounced = createTrailingDebounce(() => { void refresh(false) }, deps.debounceMs ?? 500, deps.timers)

  function setOpen(value: boolean, persist = true) {
    if (disposed || open.value === value) return
    if (persist) touched = true
    open.value = value
    if (persist) deps.persistUiPrefs({ relatedTagsOpen: value })
    if (value) {
      void refresh(true)
    } else {
      debounced.cancel()
      latest.cancel()
      loading.value = false
    }
  }

  /** ui_prefs(JSON 문자열 또는 객체)에서 열림 상태를 한 번 복원 — 사용자가 이미 열고 닫았으면 두지 않는다. */
  function restoreOpen(raw: unknown) {
    if (touched || disposed) return
    try {
      const prefs = typeof raw === 'string' ? JSON.parse(raw) : raw
      if (prefs && typeof prefs === 'object' && typeof (prefs as any).relatedTagsOpen === 'boolean') {
        setOpen((prefs as any).relatedTagsOpen, false)
      }
    } catch { /* 깨진 옛 설정이 프롬프트 편집을 막지 않게 */ }
  }

  /** 표 만들기 — 기본은 없는 등급, 없으면 낡은 등급을 다시. */
  function startBuild(shards?: readonly RelatedTagsShard[]) {
    if (disposed || build.value) return
    const target = shards && shards.length ? [...shards] : missing.value.length ? [...missing.value] : [...stale.value]
    if (!target.length) return
    const id = newId()
    build.value = { requestId: id, shard: target[0]!, index: 0, count: target.length, progress: 0, overall: 0 }
    buildSeenAt = now()
    cancelling.value = false
    error.value = ''
    feedback.value = ''
    deps.requestAction('related_tags_build', { requestId: id, shards: target })
  }

  function cancelBuild() {
    if (!build.value || cancelling.value) return
    cancelling.value = true
    feedback.value = '취소하는 중…'
    deps.requestAction('related_tags_cancel', { requestId: build.value.requestId })
  }

  function onBuildEvent(event: RelatedTagsBuildEvent) {
    if (event.state === 'building') {
      if (event.progress) build.value = { ...event.progress, requestId: event.requestId }
      buildSeenAt = now()
      return
    }
    build.value = null
    cancelling.value = false
    feedback.value = event.state === 'done' ? '연관 태그 표를 만들었습니다.'
      : event.state === 'canceled' ? '표 만들기를 취소했습니다.'
        : event.error || '표를 만들지 못했습니다.'
    void refresh(true)       // 만든 등급(취소 전까지 만든 것 포함)을 곧바로 반영
  }

  function receive(raw: unknown) {
    if (disposed) return
    const event = parseRelatedTagsEvent(raw)
    if (!event) return
    if (event.action === 'related_tags_build') onBuildEvent(event)
    else latest.receive(event)
  }

  /** 칩을 누른 뒤 — 다음 질의가 오기 전에 그 칩을 먼저 뺀다. */
  function dropSuggestion(tag: string) {
    tags.value = tags.value.filter(item => item.tag !== tag)
  }

  const stopWatch = watch(currentSignature, () => {
    if (open.value) debounced()
  })
  const off = deps.onBackendEvent('relatedTagsEvent', receive)

  function dispose() {
    if (disposed) return
    disposed = true
    stopWatch()
    debounced.cancel()
    latest.cancel()
    try { (off as (() => void) | void)?.() } catch { /* 이미 끊김 */ }
  }

  return {
    open, loading, state, tags, context, ratings, missing, stale, unavailable, estimateSec, build, cancelling,
    error, feedback,
    setOpen, restoreOpen, refresh, startBuild, cancelBuild, dropSuggestion, receive, dispose,
  }
}

export type RelatedTagsStore = ReturnType<typeof createRelatedTags>

/** 컴포넌트용 — 진짜 의존성으로 만들고, 그 컴포넌트(스코프)가 사라지면 정리한다. */
export function useRelatedTags(): RelatedTagsStore {
  const store = createRelatedTags({
    onBackendEvent, requestAction, widgets: useWidgetStore().widgets, persistUiPrefs,
    ratingRevision: ratingFilterRevision,
  })
  if (getCurrentScope()) onScopeDispose(store.dispose)
  return store
}
