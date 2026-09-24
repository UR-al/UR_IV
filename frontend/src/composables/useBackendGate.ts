import { ref } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import type {
  BackendProbeResult,
  BackendSelectedResult,
  BackendSelectionRequired,
  ComfyWorkflowPicked,
  ProbeBackendPayload,
  SelectBackendPayload,
} from '../types/bridge'
import { toGateWorkflowInfo, type GateWorkflowInfo } from '../utils/gateWorkflowInfo'

export type { GateWorkflowInfo }

/**
 * 시작 백엔드 게이트의 배선 — `components/BackendGate.vue`(그림)와 파이썬
 * `WebUIMixin`(동작) 사이를 잇는 상태만 담는다. (App.vue 분할 ④)
 *
 * **왜 App.vue 가 아니라 여기인가**: 게이트는 이벤트 4개 · 액션 3개 · 상태 8개가
 * 서로만 물고 도는 닫힌 묶음이다. App.vue(2700줄)에 풀어 놓으면 다음 사람이
 * 게이트 하나를 고치려고 파일 전체를 읽어야 한다.
 */

/** 감지 결과. 'checking' 은 요청은 나갔는데 답이 안 온 상태. */
type ProbeState = 'ok' | 'fail' | 'checking'

// BackendGate 가 받는 워크플로 요약(GateWorkflowInfo)과 파이썬 snake_case → 게이트
// 이름 변환(toGateWorkflowInfo)은 utils/gateWorkflowInfo.ts 의 순수 로직이다.

/** 시그널 페이로드는 전부 JSON 문자열이다. 깨진 JSON 하나로 시작이 막히면 안 된다. */
function parseEvent<T>(json: string): T | null {
  try {
    return JSON.parse(json || 'null') as T | null
  } catch {
    console.error('[backendGate] 페이로드 파싱 실패', json)
    return null
  }
}

export function useBackendGate() {
  const gateOpen = ref(false)
  const gateWebuiUrl = ref('')
  const gateComfyUrl = ref('')
  const gateWorkflowPath = ref('')
  // undefined = 아직 결과 없음. 게이트가 이걸 '확인 중'으로 그린다.
  const gateProbe = ref<{ webui?: ProbeState; comfy?: ProbeState } | undefined>(undefined)
  const gateWorkflowInfo = ref<GateWorkflowInfo | undefined>(undefined)
  const gateBusy = ref(false)
  const gateError = ref('')
  const gateDismissible = ref(false)

  // ── Python → Vue ────────────────────────────────────────────────────────
  function bindBackendGate() {
    onBackendEvent('backendSelectionRequired', (json: string) => {
      // 파이썬은 이 신호를 0.3/0.7/1.2초 간격으로 되풀이 보낸다(QWebChannel 핸드셰이크
      // 와 Vue 마운트 경합에 대한 보험). 이미 떠 있는데 다시 열면 사용자가 고치던
      // 주소와 감지 결과가 초기값으로 되돌아간다 — 그래서 첫 신호만 받는다.
      if (gateOpen.value) return
      const cfg = parseEvent<BackendSelectionRequired>(json)
      gateWebuiUrl.value = cfg?.webuiUrl || ''
      gateComfyUrl.value = cfg?.comfyUrl || ''
      gateWorkflowPath.value = cfg?.workflowPath || ''
      gateProbe.value = undefined
      gateWorkflowInfo.value = undefined
      gateBusy.value = false
      gateError.value = ''
      // 시작 게이트는 닫을 수 없다. 백엔드를 안 고르면 뒤에 쓸 수 있는 화면이 없다.
      gateDismissible.value = false
      // 마지막에 연다 — 게이트의 open watch 가 곧바로 probe 를 emit 하므로, 그 전에
      // 주소가 채워져 있어야 빈 주소로 감지를 나간다.
      gateOpen.value = true
    })

    onBackendEvent('backendProbeResult', (json: string) => {
      const r = parseEvent<BackendProbeResult>(json)
      if (!r) return
      gateProbe.value = { webui: r.webui, comfy: r.comfy }
    })

    onBackendEvent('backendSelected', (json: string) => {
      const r = parseEvent<BackendSelectedResult>(json)
      gateBusy.value = false
      if (r?.ok) {
        gateError.value = ''
        gateOpen.value = false
        return
      }
      // 실패해도 닫지 않는다. 닫으면 백엔드 없는 빈 앱만 남고 다시 고를 길이 없다.
      gateError.value = r?.error || '백엔드에 연결하지 못했습니다.'
    })

    onBackendEvent('comfyWorkflowPicked', (json: string) => {
      const r = parseEvent<ComfyWorkflowPicked>(json)
      if (!r) return
      // 경로를 prop 으로 돌려주면 게이트의 지역 입력값이 watch 로 따라온다.
      gateWorkflowPath.value = r.path || ''
      gateWorkflowInfo.value = toGateWorkflowInfo(r.info)
    })
  }

  // ── Vue → Python ────────────────────────────────────────────────────────
  // 페이로드 모양은 types/bridge.d.ts 한 벌(ProbeBackendPayload/SelectBackendPayload)을 따른다 —
  // 여기 따로 적으면 d.ts 와 어긋난다(예전: workflowPath 필수/선택 불일치).
  function onGateProbe(payload: ProbeBackendPayload) {
    // 답이 오기 전까지 '확인 중'. 옛 결과를 남겨 두면 방금 고친 주소의 상태인 척한다.
    gateProbe.value = { webui: 'checking', comfy: 'checking' }
    requestAction('probe_backend', payload)
  }

  function onGateSelect(payload: SelectBackendPayload) {
    if (gateBusy.value) return   // 연결 중 두 번 누르면 파이썬이 워커를 겹쳐 띄운다
    gateBusy.value = true
    gateError.value = ''         // 새 시도이니 지난 실패 문구는 치운다
    requestAction('select_backend', payload)
  }

  function onGatePickWorkflow() {
    // 파일 대화상자는 파이썬 몫(QFileDialog + analyze_workflow). 결과는
    // comfyWorkflowPicked 로 돌아온다 — 브라우저 file input 으로는 절대경로를 못 얻는다.
    requestAction('pick_comfy_workflow')
  }

  function onGateDismiss() {
    if (!gateDismissible.value) return
    gateOpen.value = false
  }

  return {
    gateOpen, gateWebuiUrl, gateComfyUrl, gateWorkflowPath,
    gateProbe, gateWorkflowInfo, gateBusy, gateError, gateDismissible,
    bindBackendGate,
    onGateProbe, onGateSelect, onGatePickWorkflow, onGateDismiss,
  }
}
