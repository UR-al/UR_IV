import { ref, computed, watch, nextTick } from 'vue'
import { requestAction } from '../stores/widgetStore.js'
import { createTrailingDebounce, flushOnPageHide } from '../utils/trailingDebounce'

/** 배율 슬라이더 드래그 중 ui_prefs.json 저장을 모으는 간격(ms) — Python 배율은 즉시 (감사 #130) */
export const HIGH_RES_SAVE_DEBOUNCE_MS = 300

/**
 * 고해상도(단일 패스) 모드 — 입력 해상도 × 배율로 처음부터 크게 생성 (App.vue 분할 ④).
 * Hires.fix(hires_* 위젯)와는 별개. 단일 소스: config/ui_prefs.json (highResEnabled/highResFactor).
 * localStorage 는 첫 렌더 캐시 — 부팅 시 파일 값(restoreFromPrefs)이 이긴다. 예전엔 마운트 400ms 뒤
 * localStorage 값을 파일에 무조건 써서 타이밍에 따라 캐시가 파일을 덮었다(감사 #107).
 *
 * @param {object} deps
 * @param {object} deps.storeWidgets  위젯 스토어 reactive (width_input/height_input 읽기)
 * @param {Function} deps.saveUiPrefs (payload) ui_prefs 영속
 */
export function useHighRes({ storeWidgets, saveUiPrefs }) {
  const highResEnabled = ref(localStorage.getItem('highRes.enabled') === 'true')
  const highResFactor = ref(parseFloat(localStorage.getItem('highRes.factor') || '1.5') || 1.5)

  // 미리보기 — 8 배수 정렬된 실제 생성 해상도
  const hrActualW = computed(() => {
    const w = parseInt(storeWidgets.width_input || 0) || 0
    return Math.max(8, Math.floor((w * highResFactor.value) / 8) * 8)
  })
  const hrActualH = computed(() => {
    const h = parseInt(storeWidgets.height_input || 0) || 0
    return Math.max(8, Math.floor((h * highResFactor.value) / 8) * 8)
  })

  let _restoring = false

  function _prefsPayload() {
    return { highResEnabled: highResEnabled.value, highResFactor: highResFactor.value }
  }
  function _cacheHighRes() {
    try {
      localStorage.setItem('highRes.enabled', highResEnabled.value ? 'true' : 'false')
      localStorage.setItem('highRes.factor', String(highResFactor.value))
    } catch {}
  }
  // 생성이 곧바로 읽는 Python 배율(_high_res_factor) — 디바운스하지 않는다
  function _pushHighRes() {
    requestAction('set_high_res_factor', {
      enabled: highResEnabled.value,
      factor: highResFactor.value,
    })
  }
  // 파일 저장만 모은다 — 슬라이더 input 틱마다 GUI 스레드가 ui_prefs.json 을 다시 쓰지 않게
  const _persistHighRes = createTrailingDebounce(() => saveUiPrefs(_prefsPayload()), HIGH_RES_SAVE_DEBOUNCE_MS)
  flushOnPageHide(() => _persistHighRes.flush())

  // 토글/배율 변경 → 캐시·Python 즉시, 파일은 디바운스. 파일에서 복원 중이면 아무것도 하지 않는다
  // (복원 뒤 nextTick 에서 캐시·Python 을 한 번에 맞춘다 — 같은 값을 파일에 되쓰지 않는다).
  watch([highResEnabled, highResFactor], () => {
    if (_restoring) return
    _cacheHighRes()
    _pushHighRes()
    _persistHighRes()
  })

  // 단일 소스(ui_prefs) 복원 — App.vue uiPrefsLoaded 핸들러에서 호출. 복원 뒤 Python 에도 보낸다
  // (useLoraStack·useRatingFilter 와 같은 계약). ui_prefs 에 키가 한 번도 저장된 적 없으면 화면의
  // localStorage 캐시 값이 사용자가 보던 값이므로 파일로 한 번 이관한다.
  function restoreFromPrefs(prefs) {
    if (!prefs || typeof prefs !== 'object') return
    const hasFactor = typeof prefs.highResFactor === 'number' && Number.isFinite(prefs.highResFactor)
    const hasEnabled = typeof prefs.highResEnabled === 'boolean'
    if (!hasFactor && !hasEnabled) {
      _pushHighRes()
      saveUiPrefs(_prefsPayload())
      return
    }
    _persistHighRes.cancel()   // 파일 값이 이긴다 — 복원 전 화면 값의 늦은 저장이 덮지 않게
    _restoring = true
    if (hasFactor) highResFactor.value = prefs.highResFactor
    if (hasEnabled) highResEnabled.value = prefs.highResEnabled
    nextTick(() => {
      _restoring = false
      _cacheHighRes()
      _pushHighRes()
    })
  }

  return { highResEnabled, highResFactor, hrActualW, hrActualH, restoreFromPrefs }
}
