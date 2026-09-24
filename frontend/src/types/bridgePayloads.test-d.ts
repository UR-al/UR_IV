/**
 * 타입 수준 계약 테스트 — `npm run type-check`(vue-tsc)만 이 파일을 본다.
 * (번들에 import 되지 않고, vitest 는 `*.test.ts` 만 돌리므로 실행되지도 않는다.)
 *
 * bridge.d.ts 의 ActionPayloads 맵이 requestAction 의 `<K extends ActionName>` 제네릭(JSDoc)을
 * 거쳐 실제로 강제되는지 확인한다. 아래 `@ts-expect-error` 가 '쓰이지 않음'이 되면(= 잘못된
 * 페이로드가 통과) type-check 가 실패한다 — 맵이나 JSDoc 이 조용히 느슨해지는 회귀를 잡는다.
 */
import { requestAction } from '../stores/widgetStore.js'
import type { ActionName, ActionPayload, SelectBackendPayload } from './bridge'

// 없는 액션 이름은 호출식 안에 리터럴로 적지 않는다 — tests/test_bridge_contract.py 의 정방향
// 검사가 requestAction(<따옴표 리터럴> 모양을 실제 호출로 읽는다(주석 안이라도).
const unknownAction = 'no_such_action' as const

export function actionPayloadContractCases(): void {
  requestAction('show_toast', { type: 'warning', msg: 'ok' })
  // @ts-expect-error 토스트 종류는 success/error/info/warning 넷뿐
  requestAction('show_toast', { type: 'warn', msg: 'x' })
  // @ts-expect-error msg 누락
  requestAction('show_toast', { type: 'info' })

  requestAction('select_backend', { type: 'comfyui', url: 'http://127.0.0.1:8188' })
  requestAction('select_backend', { type: 'webui', url: '', workflowPath: 'C:/wf.json' })
  // @ts-expect-error 백엔드 종류 오타
  requestAction('select_backend', { type: 'comfy', url: '' })

  requestAction('probe_backend', { webuiUrl: 'http://127.0.0.1:7860', comfyUrl: 'http://127.0.0.1:8188' })
  // @ts-expect-error comfyUrl 누락
  requestAction('probe_backend', { webuiUrl: '' })

  requestAction('set_high_res_factor', { enabled: true, factor: 1.5 })
  // @ts-expect-error factor 는 숫자
  requestAction('set_high_res_factor', { enabled: true, factor: '1.5' })

  requestAction('set_rating_filter', { ratings: ['g', 's'] })
  requestAction('set_lora_stack', { entries: [{ name: 'a', weight: 0.8, enabled: true, triggerWords: [] }] })
  // @ts-expect-error weight 는 배율(숫자)
  requestAction('set_lora_stack', { entries: [{ name: 'a', weight: '80', enabled: true, triggerWords: [] }] })
  requestAction('automation_override_next', { prompt: '' })

  requestAction('workflow_profile_save', { name: 'Flux' })
  requestAction('workflow_profile_save', { name: 'Flux', overwrite: true })
  // @ts-expect-error overwrite 는 불리언 — 백엔드는 true 만 덮어쓰기 확인으로 친다
  requestAction('workflow_profile_save', { name: 'Flux', overwrite: 'true' })

  // 맵에 없는 액션은 여전히 느슨하다(object) — 페이로드 없이도 부를 수 있다.
  requestAction('generate')
  requestAction('save_ui_prefs', { anyKey: 1 })

  // @ts-expect-error ActionName 에 없는 이름
  requestAction(unknownAction)
}

// 래퍼가 쓰는 제네릭 모양 — 맵의 타입을 그대로 풀어야 한다.
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2) ? true : false
export const selectBackendPayloadIsMapped: Equals<ActionPayload<'select_backend'>, SelectBackendPayload> = true
export const unmappedPayloadIsObject: Equals<ActionPayload<'generate'>, object> = true
export const unionNameStaysPermissive: ActionPayload<ActionName> = { any: 'shape' }
