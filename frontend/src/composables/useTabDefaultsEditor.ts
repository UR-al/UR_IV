import { nextTick, reactive, watch } from 'vue'
import {
  FACTORY_TAB_DEFAULTS,
  TAB_DEFAULT_KEYS,
  announceTabDefaults,
  diffTabDefaults,
  normalizeTabDefaults,
  readTabDefaults,
  type TabDefaults,
} from '../utils/tabDefaults'

type SendAction = (name: any, payload?: Record<string, unknown>) => void

/**
 * Settings '기본값' 패널 상태 — 파일(config/tab_defaults.json)과 맞춘 값(synced)을 기억하고,
 * 사용자가 바꾼 키만 1.5초 뒤 저장한다(audit #140).
 *
 * 예전엔 onMounted 에서 한 번 읽고 deep watch 가 객체 전체를 통째로 저장해서
 *  - 마운트 때 Object.assign 이 watch 를 건드려 불필요한 저장·토스트가 한 번 났고
 *  - keep-alive 로 남은 옛 값이 '전역 저장'이 방금 T2I 값으로 갱신한 steps/cfg/… 를 되덮었다.
 * 지금은 코드가 값을 대입하는 동안(reload) 자동 저장을 막고, 저장은 diff 만 보낸다.
 */
export function useTabDefaultsEditor(deps: {
  /** 브리지 액션 전송 — widgetStore 의 requestAction 을 넘긴다 */
  sendAction: SendAction
  getBackend: () => Promise<any>
  debounceMs?: number
}) {
  const defaults = reactive<TabDefaults>({ ...FACTORY_TAB_DEFAULTS })
  let synced: TabDefaults = { ...FACTORY_TAB_DEFAULTS }
  let applying = false
  let timer: ReturnType<typeof setTimeout> | null = null
  const debounceMs = deps.debounceMs ?? 1500
  // 재조회 요청 번호 — 겹친 재조회에서 앞선 응답은 버린다(전송은 FIFO 라 방어용)
  let reloadSeq = 0
  // 재조회를 기다리는 사이 저장 요청으로 보낸 값 — 응답(요청 시점의 파일)이 이보다 옛값일 수 있다
  let savedWhileLoading: Partial<TabDefaults> | null = null

  function flush(): boolean {
    if (timer) { clearTimeout(timer); timer = null }
    const patch = diffTabDefaults({ ...defaults }, synced)
    if (!patch) return false
    deps.sendAction('save_tab_defaults', patch as Record<string, unknown>)
    if (savedWhileLoading) Object.assign(savedWhileLoading, patch)
    synced = normalizeTabDefaults({ ...synced, ...patch })
    announceTabDefaults(synced)
    return true
  }

  watch(defaults, () => {
    if (applying) return
    if (timer) clearTimeout(timer)
    timer = setTimeout(flush, debounceMs)
  }, { deep: true })

  /**
   * 파일에서 다시 읽는다(Settings 재활성·전역 저장 뒤). 대기 중인 사용자 변경은 먼저 저장한다.
   * 응답을 기다리는 사이 사용자가 고친 칸은 늦게 온 파일 값으로 되돌리지 않는다 — 예전엔 그 칸이
   * 파일 값으로 튀어 돌아가고 대기 중인 저장은 diff 가 없어 사라졌다(또는 이미 저장됐는데 화면만
   * 옛값으로 돌아가 파일과 어긋났다). 나머지 칸은 파일 값을 따른다.
   */
  async function reload(): Promise<void> {
    flush()
    const seq = ++reloadSeq
    const before: TabDefaults = { ...defaults }
    savedWhileLoading = {}
    const backend = await deps.getBackend()
    const loaded = await readTabDefaults(backend)
    if (seq !== reloadSeq) return   // 더 새 재조회가 있다 — 그 응답이 적용한다
    const sent = savedWhileLoading || {}
    savedWhileLoading = null
    const merged: TabDefaults = { ...loaded }
    for (const key of TAB_DEFAULT_KEYS) {
      if (!Object.is(defaults[key], before[key])) (merged as any)[key] = defaults[key]
    }
    applying = true
    Object.assign(defaults, merged)
    // 기다리는 사이 보낸 저장은 응답보다 뒤에 파일에 들어간다 — synced 에 반영해 같은 값을 또 보내지 않는다
    synced = normalizeTabDefaults({ ...loaded, ...sent })
    await nextTick()   // deep watch(pre-flush) 가 대입을 본 뒤에 가드를 푼다
    applying = false
    // 지킨 칸이 아직 저장 전이면(디바운스 대기 중이 아니면) 저장을 예약한다
    if (!timer && diffTabDefaults({ ...defaults }, synced)) timer = setTimeout(flush, debounceMs)
  }

  /** '기본값 저장' 버튼 — 대기 중 변경을 바로 보낸다. 바뀐 게 없으면 알려만 준다. */
  function saveNow(): void {
    if (!flush()) deps.sendAction('show_toast', { type: 'info', msg: '바뀐 기본값이 없습니다 — 이미 저장돼 있습니다' })
  }

  function resetToFactory(): void {
    Object.assign(defaults, FACTORY_TAB_DEFAULTS)
  }

  function dispose(): void {
    flush()
  }

  return { defaults, reload, saveNow, resetToFactory, flush, dispose }
}
