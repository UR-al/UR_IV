import { afterEach, expect, it, vi } from 'vitest'
import { createPromptOrder } from './usePromptOrder'

const SECTIONS = [{ key: 'count', label: '인물수' }, { key: 'char', label: '캐릭터' }, { key: 'main', label: '메인' }]

function setup() {
  const handlers = new Map<string, (...args: any[]) => void>()
  const requestAction = vi.fn()
  const addToast = vi.fn()
  const mgr = createPromptOrder({ requestAction, addToast, onBackendEvent: (name, cb) => { handlers.set(name, cb) } })
  mgr.bind()
  handlers.get('promptOrderLoaded')!(JSON.stringify(SECTIONS))
  return { mgr, handlers, requestAction, addToast }
}

afterEach(() => { vi.unstubAllGlobals() })

it('opening asks for the current order; the list follows promptOrderLoaded', () => {
  const { mgr, handlers, requestAction } = setup()
  mgr.openOrderManager()
  expect(mgr.showOrderManager.value).toBe(true)
  expect(requestAction).toHaveBeenCalledWith('prompt_order_list', {})
  expect(mgr.promptOrderList.value.map(s => s.key)).toEqual(['count', 'char', 'main'])
  handlers.get('promptOrderLoaded')!('oops')
  expect(mgr.promptOrderList.value).toHaveLength(3)
})

it('moves sections up and down within bounds, then saves the keys and closes', () => {
  const { mgr, requestAction, addToast } = setup()
  mgr.moveOrderUp(0)                  // 맨 위 — 그대로
  mgr.moveOrderDown(2)                // 맨 아래 — 그대로
  mgr.moveOrderDown(0)
  mgr.moveOrderUp(2)
  expect(mgr.promptOrderList.value.map(s => s.key)).toEqual(['char', 'main', 'count'])
  mgr.openOrderManager()
  mgr.saveOrderAndClose()
  expect(requestAction).toHaveBeenLastCalledWith('prompt_order_save', { order: ['char', 'main', 'count'] })
  expect(mgr.showOrderManager.value).toBe(false)
  expect(addToast).toHaveBeenCalledWith('success', '프롬프트 순서 저장됨')
})

it('reset asks first', () => {
  const { mgr, requestAction, addToast } = setup()
  vi.stubGlobal('window', { confirm: () => false })
  mgr.resetPromptOrder()
  expect(requestAction).not.toHaveBeenCalled()
  vi.stubGlobal('window', { confirm: () => true })
  mgr.resetPromptOrder()
  expect(requestAction).toHaveBeenCalledWith('prompt_order_reset', {})
  expect(addToast).toHaveBeenCalledWith('info', '기본 순서로 복원')
})
