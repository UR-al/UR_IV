/**
 * 위젯 상태 저장소 — Python 프록시와 실시간 2방향 동기화
 */
import { reactive } from 'vue'

/** @type {{ values: Record<string, any>, properties: Record<string, Record<string, any>> }} */
const state = reactive({
  values: {},      // { widget_id: value_string }
  properties: {},  // { widget_id: { items, enabled, ... } }
})

let _backend = null
let _disconnectBackend = null
let _applyingFromBackend = false
const _prevSnapshot = Object.create(null)
// 첫 getAllWidgetValues 적용 여부 — 이 시점을 기다려야 하는 쪽(PromptPanel 의 Undo 기준점)이
// '800ms 뒤면 왔겠지' 같은 추측 타이머를 쓰지 않게 한다.
let _initialValuesLoaded = false
const _initialValueWaiters = new Set()

function _markInitialValuesLoaded() {
  if (_initialValuesLoaded) return
  _initialValuesLoaded = true
  const waiters = [..._initialValueWaiters]
  _initialValueWaiters.clear()
  for (const cb of waiters) {
    try { cb() } catch (e) { console.error('[Store] initial-values waiter failed:', e) }
  }
}

/**
 * 첫 초기값(getAllWidgetValues)이 스토어에 적용된 직후 ``cb`` 를 한 번 부른다.
 * 이미 적용됐으면 즉시 부른다. 반환값은 대기 해제 함수.
 * @param {() => void} cb
 * @returns {() => void}
 */
export function whenWidgetValuesLoaded(cb) {
  if (_initialValuesLoaded) { cb(); return () => {} }
  _initialValueWaiters.add(cb)
  return () => { _initialValueWaiters.delete(cb) }
}

// Vue에서 발생한 키 단위 변경만 Python으로 전송한다.
// reactive 객체 전체를 deep-watch하면 입력 한 글자마다 모든 위젯을 순회하고,
// Python에서 받은 초기값까지 다시 echo하게 된다.
const widgetValues = new Proxy(state.values, {
  set(target, id, val) {
    if (target[id] === val) return true
    target[id] = val
    if (!_applyingFromBackend && _backend) {
      const strVal = String(val)
      if (_prevSnapshot[id] !== strVal) {
        _prevSnapshot[id] = strVal
        _backend.onWidgetChanged(String(id), strVal)
      }
    }
    return true
  },
})

function applyBackendValues(data) {
  _applyingFromBackend = true
  try {
    for (const [id, val] of Object.entries(data || {})) {
      _prevSnapshot[id] = String(val)
      state.values[id] = val
    }
  } finally {
    _applyingFromBackend = false
  }
}

function applyBackendProperty(id, prop, value) {
  if (!state.properties[id]) state.properties[id] = {}
  state.properties[id][prop] = value
}

/** getAllWidgetProperties 스냅숏 `{widget_id: {prop: value}}` 을 적용한다. */
function applyBackendProperties(data) {
  if (!data || typeof data !== 'object') return
  for (const [id, props] of Object.entries(data)) {
    if (!props || typeof props !== 'object') continue
    for (const [prop, value] of Object.entries(props)) applyBackendProperty(id, prop, value)
  }
}

/**
 * 브릿지 연결
 */
export function connectStore(backend) {
  if (_disconnectBackend) _disconnectBackend()
  _backend = backend

  // 1. Python -> Vue: 개별 값 수신
  const onValueChanged = (id, val) => {
    if (state.values[id] !== val) applyBackendValues({ [id]: val })
  }
  backend.widgetValueChanged.connect(onValueChanged)

  // 2. Python -> Vue: 속성 수신
  const onPropertyChanged = (id, prop, valJson) => {
    let value
    try {
      value = JSON.parse(valJson)
    } catch {
      value = valJson
    }
    applyBackendProperty(id, prop, value)
  }
  backend.widgetPropertyChanged.connect(onPropertyChanged)

  // 3. Python -> Vue: 배치 업데이트
  const onBatchUpdate = (json) => {
    try {
      const data = JSON.parse(json)
      applyBackendValues(data)
    } catch (e) { console.error('[Store] Batch Error:', e) }
  }
  backend.batchUpdate.connect(onBatchUpdate)

  // 4. 초기값 로드
  backend.getAllWidgetValues((json) => {
    if (_backend !== backend) return
    try {
      const data = JSON.parse(json)
      applyBackendValues(data)
      _markInitialValuesLoaded()
    } catch (e) { console.error('[Store] Init Error:', e) }
  })

  // 5. 초기 속성 로드 — 속성(콤보 선택지·enabled …)은 push 로만 와서, 페이지가 뜨기 전에
  //    Python 이 채운 것(예: SAM3 ControlNet 전처리기·control/resize mode 목록은 _setup_ui
  //    에서 한 번 채운다)이나 웹 클라이언트가 붙기 전에 보낸 것은 여기서만 받는다.
  //    채널이 순서를 지키므로 이 스냅숏은 앞서 받은 push 보다 오래되지 않는다.
  //    (웹 capability 에 없거나 목 백엔드면 건너뛴다)
  if (typeof backend.getAllWidgetProperties === 'function') {
    backend.getAllWidgetProperties((json) => {
      if (_backend !== backend) return
      try {
        applyBackendProperties(JSON.parse(json))
      } catch (e) { console.error('[Store] Property Init Error:', e) }
    })
  }

  const disconnect = () => {
    try { backend.widgetValueChanged.disconnect(onValueChanged) } catch {}
    try { backend.widgetPropertyChanged.disconnect(onPropertyChanged) } catch {}
    try { backend.batchUpdate.disconnect(onBatchUpdate) } catch {}
    if (_backend === backend) _backend = null
    if (_disconnectBackend === disconnect) _disconnectBackend = null
  }
  _disconnectBackend = disconnect
  return disconnect
}

export function getValue(id) { return state.values[id] ?? '' }
export function getProperty(id, prop, def = '') { return state.properties[id]?.[prop] ?? def }
// 값 쓰기는 `useWidgetStore().widgets[id] = v` 한 길뿐이다 — Proxy 의 set 트랩이 Python 으로 보낸다.

// 액션 요청
/**
 * Vue → Python 액션 요청. (이름 정합성 강제는 tests/test_bridge_contract.py)
 * 페이로드는 bridge.d.ts 의 ActionPayloads 맵을 따라 TS 호출부에서 type-check 된다
 * (맵에 없는 액션은 `object`). 이 .js 파일 자체와 .js 호출부는 checkJs:false 라 검사 밖이다.
 * @template {import('../types/bridge').ActionName} K
 * @param {K} action  백엔드 액션 이름(에디터 자동완성)
 * @param {import('../types/bridge').ActionPayload<K>} [payload]
 */
export function requestAction(action, payload = {}) {
  if (_backend) {
    // 개발 빌드에서만 — 배포 빌드의 콘솔 줄은 javaScriptConsoleMessage 로 Python print 까지 가서
    // 슬라이더 드래그(set_lora_stack·set_high_res_factor 가 틱마다)마다 로그를 찍었다(감사 #130).
    // 액션 기록은 Python 쪽 core.action_log 가 남긴다. 페이로드 객체를 콘솔에 넘기면 콘솔 버퍼가
    // 그 객체(update_prompt_deck·export 의 수만 행, chat_save 의 base64 이미지)를 계속 붙잡는다.
    if (import.meta.env.DEV) console.log(`[Vue -> Python] Action: ${action}`, payload)
    _backend.onAction(action, JSON.stringify(payload))
  }
}

/**
 * Composable 래퍼 — PromptPanel 등에서 useWidgetStore()로 사용
 */
export function useWidgetStore() {
  return {
    widgets: widgetValues,
    getProperty: (id, prop, def = '') => state.properties[id]?.[prop] ?? def,
    getValue,
    requestAction,
  }
}

export { state }
