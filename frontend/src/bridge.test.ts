import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/**
 * onBackendBound — 백엔드가 붙을 때마다(첫 연결 · 웹 재접속) 부르는 훅.
 *
 * 회귀: 대기열 패널은 onMounted 에서 곧바로 requestAction('sync_queue_state') 를 보냈다. 자식의
 * onMounted 는 App 의 onMounted(await initBridge())보다 먼저 돌아 스토어에 백엔드가 없었고, 요청은
 * 조용히 버려졌다(시작 때 복구된 대기열이 Vue 에 안 보임). 웹 재접속은 패널을 다시 마운트하지 않아
 * 다시 당겨 오지도 않았다.
 */

// 웹 모드 transport — npm qwebchannel 대신: 소켓에 붙여 둔 facade 를 채널 객체로 돌려준다
vi.mock('qwebchannel', () => ({
  QWebChannel: class {
    constructor(socket: any, ready: (channel: any) => void) {
      ready({ objects: { backend: socket.facade } })
    }
  },
}))

type Listener = (...args: any[]) => void

function signal() {
  const listeners = new Set<Listener>()
  return {
    connect: (fn: Listener) => { listeners.add(fn) },
    disconnect: (fn: Listener) => { listeners.delete(fn) },
    emit: (...args: any[]) => { for (const fn of [...listeners]) fn(...args) },
  }
}

/** 웹 facade 대역 — getCapabilities/invoke/event 만. invoke 한 이름과 인자를 기록한다. */
function fakeFacade(log: Array<[string, any[]]>) {
  const event = signal()
  return {
    event,
    getCapabilities: (cb: (json: string) => void) => cb(JSON.stringify({
      methods: ['onAction', 'onWidgetChanged', 'getAllWidgetValues'],
      signals: ['widgetValueChanged', 'widgetPropertyChanged', 'batchUpdate', 'queueUpdated'],
    })),
    invoke: (name: string, argsJson: string, cb?: (json: string) => void) => {
      log.push([name, JSON.parse(argsJson)])
      if (cb) cb(JSON.stringify({ ok: true, value: name === 'getAllWidgetValues' ? '{}' : null }))
    },
  }
}

class FakeSocket {
  static OPEN = 1
  static instances: FakeSocket[] = []
  readyState = FakeSocket.OPEN
  onopen: (() => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  onerror: (() => void) | null = null
  facade: any
  constructor(public url: string) {
    this.facade = fakeFacade(FakeSocket.log)
    FakeSocket.instances.push(this)
  }
  static log: Array<[string, any[]]> = []
  drop() {
    this.readyState = 3
    this.onclose?.({ code: 1006 })
  }
}

async function freshBridge() {
  vi.resetModules()
  const bridge = await import('./bridge.js')
  // 구조 분해로 꺼낸다 — 계약 스캐너(test_web_mode_security)는 모듈 멤버 접근을 같은 이름의 옛 슬롯 호출로 본다
  const { requestAction } = await import('./stores/widgetStore.js')
  return { bridge, requestAction }
}

describe('onBackendBound', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.instances = []
    FakeSocket.log = []
    vi.spyOn(console, 'log').mockImplementation(() => {})
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('fires after the store can send actions, and again after every web reconnect', async () => {
    vi.stubGlobal('window', { __AISTUDIO_WS_URL__: 'ws://bridge.test' })
    vi.stubGlobal('location', { protocol: 'http:', hostname: 'bridge.test' })
    vi.stubGlobal('WebSocket', FakeSocket)
    const { bridge, requestAction } = await freshBridge()

    const actions = () => FakeSocket.log.filter(([name]) => name === 'onAction').map(([, args]) => args[0])
    const off = bridge.onBackendBound(() => requestAction('sync_queue_state'))
    requestAction('sync_queue_state')                    // 붙기 전 요청은 버려진다(그래서 훅이 필요하다)

    void bridge.initBridge()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(actions()).toEqual([])                        // 아직 채널이 없다
    FakeSocket.instances[0].onopen!()
    expect(actions()).toEqual(['sync_queue_state'])      // 훅이 보낸 한 번 — 스토어가 이미 보낼 수 있다

    FakeSocket.instances[0].drop()                       // 웹 재접속 — 패널은 다시 마운트되지 않는다
    vi.advanceTimersByTime(1000)
    expect(FakeSocket.instances).toHaveLength(2)
    FakeSocket.instances[1].onopen!()
    expect(actions()).toEqual(['sync_queue_state', 'sync_queue_state'])

    off()
    FakeSocket.instances[1].drop()
    vi.advanceTimersByTime(2000)
    FakeSocket.instances[2].onopen!()
    expect(actions()).toEqual(['sync_queue_state', 'sync_queue_state'])   // 해제 뒤엔 부르지 않는다
  })

  it('calls a late subscriber right away when the backend is already bound', async () => {
    vi.stubGlobal('window', { __AISTUDIO_WS_URL__: 'ws://bridge.test' })
    vi.stubGlobal('location', { protocol: 'http:', hostname: 'bridge.test' })
    vi.stubGlobal('WebSocket', FakeSocket)
    const { bridge } = await freshBridge()
    void bridge.initBridge()
    FakeSocket.instances[0].onopen!()

    const seen: unknown[] = []
    bridge.onBackendBound((backend: unknown) => seen.push(backend))
    expect(seen).toHaveLength(1)
    expect(seen[0]).toBe(await bridge.getBackend())
  })

  it('keeps notifying the others when one listener throws', async () => {
    vi.stubGlobal('window', { __AISTUDIO_WS_URL__: 'ws://bridge.test' })
    vi.stubGlobal('location', { protocol: 'http:', hostname: 'bridge.test' })
    vi.stubGlobal('WebSocket', FakeSocket)
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { bridge } = await freshBridge()
    const calls: string[] = []
    bridge.onBackendBound(() => { throw new Error('boom') })
    bridge.onBackendBound(() => { calls.push('second') })
    void bridge.initBridge()
    FakeSocket.instances[0].onopen!()
    expect(calls).toEqual(['second'])
  })
})

/**
 * 개발 모드 목(vite dev — Qt·WebSocket transport 없음).
 *
 * 회귀: 목에 위젯 시그널(widgetValueChanged/widgetPropertyChanged/batchUpdate)이 없어 스토어
 * connectStore 가 바인딩 즉시 TypeError 를 냈고 initBridge 가 거부돼 화면 초기화가 멈췄다(P13c).
 */
describe('dev mock transport', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(console, 'log').mockImplementation(() => {})
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('binds the store and routes actions to the mock after the Qt wait times out', async () => {
    vi.stubGlobal('window', {})
    const { bridge, requestAction } = await freshBridge()
    const bound = vi.fn()
    bridge.onBackendBound(bound)

    const ready = bridge.initBridge()
    await vi.advanceTimersByTimeAsync(5200)              // waitForQWebChannel 5초 → 목 모드
    const backend: any = await ready

    expect(backend._mock).toBe(true)
    expect(bound).toHaveBeenCalledTimes(1)
    for (const name of ['widgetValueChanged', 'widgetPropertyChanged', 'batchUpdate']) {
      expect(typeof backend[name].connect).toBe('function')
    }
    // 화면들의 `if (backend.requestX)` 가드가 목 모드를 건너뛸 수 있게 조회 슬롯은 없다
    expect(backend.requestGalleryImages).toBeUndefined()

    requestAction('sync_queue_state')
    expect(console.log).toHaveBeenCalledWith('[mock] action sync_queue_state', expect.anything())
  })
})
