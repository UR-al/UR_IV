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

  requestAction('memo_save', { id: 'm1', title: '', text: '본문', base_updated_at: null })
  // @ts-expect-error base_updated_at 은 빠뜨리지 않는다(새 메모는 null) — 충돌 판정의 근거다
  requestAction('memo_save', { id: 'm1', title: '', text: '본문' })
  requestAction('memo_delete', { id: 'm1' })
  requestAction('memo_list', {})
  // @ts-expect-error memo_sync 는 페이로드가 없다
  requestAction('memo_sync', { force: true })

  const tileSettings = {
    model: '', prompt: 'repair', negative_prompt: '', steps: 50, cfg_scale: 3.5, flow_shift: 5,
    multiplier: 1, short_side: 1024, seed: -1, dit: '', text_encoder: '', vae: '', unload_forge_before: true,
  }
  requestAction('tile_repair_run', { requestId: 'tile_run_1', image_path: 'C:/a.png', image: '', settings: tileSettings })
  // @ts-expect-error 원본은 image_path 와 image 를 둘 다 보낸다(안 쓰는 쪽은 '') — 파이썬이 경로를 먼저 본다
  requestAction('tile_repair_run', { requestId: 'tile_run_1', image_path: 'C:/a.png', settings: tileSettings })
  // @ts-expect-error 배율은 숫자(−10..10) — 문자열이면 확장이 400 으로 거절한다
  requestAction('tile_repair_run', { requestId: 'r', image_path: '', image: '', settings: { ...tileSettings, multiplier: '1' } })
  // @ts-expect-error 확장 라우트는 모르는 키를 거절한다 — lllite_multiplier 가 아니라 multiplier
  requestAction('tile_repair_run', { requestId: 'r', image_path: '', image: '', settings: { ...tileSettings, lllite_multiplier: 1 } })
  requestAction('tile_repair_options', { requestId: 'tile_options_1' })
  requestAction('tile_repair_cancel', { requestId: 'tile_run_1' })
  // @ts-expect-error cancel 은 멈출 run 의 requestId 가 필요하다
  requestAction('tile_repair_cancel', {})

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
