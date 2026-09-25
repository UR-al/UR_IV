import { reactive, ref } from 'vue'
import { onBackendEvent } from '../bridge.js'
import { requestAction } from '../stores/widgetStore.js'
import type { TileRepairOptions, TileRepairResultEvent, TileRepairSettings } from '../types/bridge'
import {
  defaultTileRepairSettings, parseTileRepairEvent, readTileRepairOptions, tileRepairSource,
} from '../utils/tileRepair'
import type { OnBackendEventFn, RequestActionFn } from './managerDeps'

/**
 * Anima Tile & Repair 상태 — **모듈 싱글턴**(I2I 카드를 닫거나 탭을 옮겨도 설정·진행·결과가 남는다).
 *
 * 브리지(tests/test_bridge_contract.py 가 이름을 지킨다): tile_repair_options · tile_repair_run ·
 * tile_repair_cancel, 시그널 tileRepairResult. 파이썬 ui/tile_repair_actions.py 가 Forge sam-extra
 * POST /sam-extra/tile-repair 로 보낸다. 요청마다 requestId 를 붙이고, 지금 기다리는 id 의 답만 받는다
 * (옛 요청 · 다른 웹 클라이언트의 답은 버린다).
 */
export interface TileRepairDeps {
  onBackendEvent: OnBackendEventFn
  requestAction: RequestActionFn
  /** 테스트용 id 생성기 */
  newId?: (kind: 'options' | 'run') => string
}

export interface TileRepairOutput {
  path: string
  width: number | null
  height: number | null
  seed: number | null
  info: string
  model: string
}

let counter = 0
function defaultId(kind: 'options' | 'run'): string {
  counter += 1
  return `tile_${kind}_${Date.now().toString(36)}_${counter}_${Math.random().toString(36).slice(2, 8)}`
}

export function createTileRepair(deps: TileRepairDeps) {
  const newId = deps.newId ?? defaultId
  const settings = reactive<TileRepairSettings>(defaultTileRepairSettings())
  const options = ref<TileRepairOptions | null>(null)
  const optionsError = ref('')
  const loadingOptions = ref(false)
  const busy = ref(false)
  const cancelling = ref(false)
  const error = ref('')
  const notice = ref('')
  const result = ref<TileRepairOutput | null>(null)
  let optionsId = ''
  let runId = ''

  function loadOptions() {
    optionsId = newId('options')
    loadingOptions.value = true
    optionsError.value = ''
    deps.requestAction('tile_repair_options', { requestId: optionsId })
  }

  /** 복원 1회. 원본이 없으면 false(오류 문구를 채운다). */
  function run(imagePath: string, imageSrc: string): boolean {
    if (busy.value) return false
    const source = tileRepairSource(imagePath, imageSrc)
    error.value = ''
    notice.value = ''
    if (!source) {
      error.value = 'I2I 원본 이미지를 먼저 올리세요 (PNG/JPEG/WebP).'
      return false
    }
    if (!settings.prompt.trim()) {
      error.value = '프롬프트가 비어 있습니다.'
      return false
    }
    runId = newId('run')
    busy.value = true
    cancelling.value = false
    deps.requestAction('tile_repair_run', { requestId: runId, ...source, settings: { ...settings } })
    return true
  }

  function cancel() {
    if (!busy.value || !runId || cancelling.value) return
    cancelling.value = true
    deps.requestAction('tile_repair_cancel', { requestId: runId })
  }

  function resetSettings() {
    Object.assign(settings, defaultTileRepairSettings())
  }

  /** 결과의 시드를 고정한다(-1 → 쓴 시드). */
  function reuseSeed() {
    if (result.value?.seed != null) settings.seed = result.value.seed
  }

  function finishRun(event: TileRepairResultEvent) {
    runId = ''
    busy.value = false
    cancelling.value = false
    if (event.ok && typeof event.path === 'string' && event.path) {
      result.value = {
        path: event.path,
        width: event.width ?? null,
        height: event.height ?? null,
        seed: event.seed ?? null,
        info: event.info || '',
        model: event.model || '',
      }
      notice.value = '복원 결과를 새 파일로 저장했습니다. 원본은 그대로입니다.'
    } else if (event.canceled) {
      notice.value = event.error || 'Tile & Repair 를 취소했습니다.'
    } else {
      error.value = event.error || 'Tile & Repair 가 실패했습니다.'
    }
  }

  function receive(json: string) {
    const event = parseTileRepairEvent(json)
    if (!event) return
    if (event.action === 'tile_repair_options') {
      if (!optionsId || event.requestId !== optionsId) return
      optionsId = ''
      loadingOptions.value = false
      const read = event.ok ? readTileRepairOptions(event.options) : null
      if (read) options.value = read
      else optionsError.value = event.error || 'Forge Tile & Repair 선택지를 읽지 못했습니다.'
      return
    }
    if (!runId || event.requestId !== runId) return
    if (event.action === 'tile_repair_run') {
      finishRun(event)
      return
    }
    // tile_repair_cancel — 결과는 run 의 답으로 온다(취소면 canceled). Forge 는 요청이 도착한 순간부터
    // (큐 대기 포함) 멈추고, 파이썬이 잠깐씩 다시 물은 뒤에도 stopped=false 면 요청을 못 찾은 것이다 —
    // 방금 끝났으면 run 의 답이 곧 온다. 아니면 다시 누를 수 있게 취소 버튼을 풀어 둔다.
    if (!event.ok) {
      cancelling.value = false
      error.value = event.error || '취소 요청이 실패했습니다.'
    } else if (event.stopped === false) {
      cancelling.value = false
      notice.value = 'Forge 에서 멈출 작업을 찾지 못했습니다. 결과가 오지 않으면 취소를 다시 누르세요.'
    }
  }

  let unbind: (() => void) | null = null
  function bind() {
    if (unbind) return
    const off = deps.onBackendEvent('tileRepairResult', receive)
    unbind = off || (() => {})
  }
  function unbindAll() {
    try { unbind?.() } catch {}
    unbind = null
  }

  return {
    settings, options, optionsError, loadingOptions, busy, cancelling, error, notice, result,
    loadOptions, run, cancel, resetSettings, reuseSeed, receive, bind, unbind: unbindAll,
  }
}

export type TileRepairStore = ReturnType<typeof createTileRepair>

let _app: TileRepairStore | null = null
/** 앱 전체가 함께 쓰는 Tile & Repair 상태. 처음 부를 때 tileRepairResult 를 듣기 시작한다. */
export function useTileRepair(): TileRepairStore {
  if (!_app) {
    _app = createTileRepair({ onBackendEvent, requestAction })
    _app.bind()
  }
  return _app
}
