import { describe, expect, it, vi } from 'vitest'
import { connectStore, getProperty, getValue } from './widgetStore.js'

/**
 * 위젯 스토어 초기 동기화 (#47).
 * 속성(콤보 선택지 등)은 Python 이 push 로만 보내서, Vue 페이지가 뜨기 전에 채운 SAM3
 * ControlNet 선택지가 스토어에 닿지 않았다. connectStore 는 값뿐 아니라 속성 스냅숏
 * (getAllWidgetProperties)도 받아야 한다.
 */

type Listener = (...args: any[]) => void

function signal() {
  const listeners = new Set<Listener>()
  return {
    connect: (fn: Listener) => { listeners.add(fn) },
    disconnect: (fn: Listener) => { listeners.delete(fn) },
    emit: (...args: any[]) => { for (const fn of [...listeners]) fn(...args) },
    get size() { return listeners.size },
  }
}

function fakeBackend(values: Record<string, string>, properties?: Record<string, Record<string, unknown>>) {
  const pending: Array<() => void> = []
  const backend: Record<string, any> = {
    widgetValueChanged: signal(),
    widgetPropertyChanged: signal(),
    batchUpdate: signal(),
    onWidgetChanged: () => {},
    // QWebChannel 응답은 비동기 — flush() 로 도착시킨다.
    getAllWidgetValues: (cb: (json: string) => void) => { pending.push(() => cb(JSON.stringify(values))) },
  }
  if (properties) {
    backend.getAllWidgetProperties = (cb: (json: string) => void) => {
      pending.push(() => cb(JSON.stringify(properties)))
    }
  }
  const flush = () => { while (pending.length) pending.shift()!() }
  return { backend, flush }
}

const MODULES = ['inpaint_only', 'inpaint_only+lama', 'depth_anything']

describe('connectStore 초기 동기화', () => {
  it('페이지 로드 전에 push 된 콤보 선택지를 스냅숏으로 받는다', () => {
    const { backend, flush } = fakeBackend(
      { _sam3_cn_module: 'inpaint_only' },
      {
        _sam3_cn_module: { items: MODULES },
        _sam3_cn_control_mode: { items: ['Balanced', 'My prompt is more important', 'ControlNet is more important'] },
        some_line_edit: { enabled: false, placeholder: 'None' },
      },
    )
    const disconnect = connectStore(backend)
    expect(getProperty('_sam3_cn_module', 'items')).toBe('')   // 응답 전
    flush()
    expect(getValue('_sam3_cn_module')).toBe('inpaint_only')
    expect(getProperty('_sam3_cn_module', 'items')).toEqual(MODULES)
    expect(getProperty('_sam3_cn_control_mode', 'items')).toHaveLength(3)
    expect(getProperty('some_line_edit', 'enabled')).toBe(false)
    expect(getProperty('some_line_edit', 'placeholder')).toBe('None')
    disconnect()
  })

  it('스냅숏 뒤에 온 push 가 최신 값이 된다', () => {
    const { backend, flush } = fakeBackend({}, { te_main_input: { items: ['a.safetensors'] } })
    const disconnect = connectStore(backend)
    flush()
    backend.widgetPropertyChanged.emit('te_main_input', 'items', JSON.stringify(['b.safetensors']))
    expect(getProperty('te_main_input', 'items')).toEqual(['b.safetensors'])
    disconnect()
  })

  it('스냅숏 슬롯이 없는 백엔드(목·옛 웹 capability)에서도 값 동기화는 된다', () => {
    const { backend, flush } = fakeBackend({ plain_widget: 'x' })
    const disconnect = connectStore(backend)
    flush()
    expect(getValue('plain_widget')).toBe('x')
    backend.widgetPropertyChanged.emit('plain_widget', 'items', JSON.stringify(['x', 'y']))
    expect(getProperty('plain_widget', 'items')).toEqual(['x', 'y'])
    disconnect()
  })

  it('끊긴 백엔드의 늦은 스냅숏 응답은 버린다', () => {
    const stale = fakeBackend({}, { reconnect_combo: { items: ['old'] } })
    connectStore(stale.backend)
    const fresh = fakeBackend({}, { reconnect_combo: { items: ['new'] } })
    const disconnect = connectStore(fresh.backend)
    fresh.flush()
    stale.flush()   // 이전 연결의 응답이 뒤늦게 도착
    expect(getProperty('reconnect_combo', 'items')).toEqual(['new'])
    expect(stale.backend.widgetPropertyChanged.size).toBe(0)
    disconnect()
  })
})

describe('whenWidgetValuesLoaded — 첫 초기값 적용 시점', () => {
  it('fires once right after the first getAllWidgetValues is applied (or immediately afterwards)', async () => {
    vi.resetModules()
    const store = await import('./widgetStore.js')
    const seen: string[] = []
    store.whenWidgetValuesLoaded(() => { seen.push(store.getValue('main_prompt_text')) })
    const cancelled = store.whenWidgetValuesLoaded(() => { seen.push('cancelled') })
    cancelled()
    const { backend, flush } = fakeBackend({ main_prompt_text: '1girl, smile' })
    const disconnect = store.connectStore(backend)
    expect(seen).toEqual([])                   // 응답 전에는 부르지 않는다
    flush()
    expect(seen).toEqual(['1girl, smile'])     // 값이 들어간 뒤에 한 번
    // 재연결(웹 모드 재접속)의 두 번째 초기값에는 다시 부르지 않는다
    const again = fakeBackend({ main_prompt_text: 'other' })
    store.connectStore(again.backend)
    again.flush()
    expect(seen).toEqual(['1girl, smile'])
    // 이미 적용된 뒤 등록하면 즉시
    store.whenWidgetValuesLoaded(() => { seen.push('late') })
    expect(seen).toEqual(['1girl, smile', 'late'])
    disconnect()
  })
})
