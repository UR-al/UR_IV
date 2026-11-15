import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// 첫 Studio 연결이 한 번 실패해도 업데이트 패널이 세션 내내 실패로 굳지 않는다(UR-02):
// 실패한 초기화는 캐시하지 않고 다음 호출이 다시 시도하며, 복구되면 실패 문구도 지운다.
const mocks = vi.hoisted(() => ({
  getStudioClient: vi.fn(),
  actionSpy: vi.fn(),
}))
vi.mock('../studio/client', () => ({
  getStudioClient: mocks.getStudioClient,
  replyData: (reply: any) => reply?.data,
}))
vi.mock('./widgetStore.js', () => ({ requestAction: mocks.actionSpy }))
vi.mock('../utils/externalUrl', () => ({ openExternalUrl: vi.fn() }))

function fakeClient(snapshot: () => Promise<unknown>) {
  const unsubscribe = vi.fn()
  return {
    unsubscribe,
    supports: () => true,
    subscribe: vi.fn(() => unsubscribe),
    invoke: vi.fn(async (name: string) => {
      if (name === 'app_update.snapshot') return { status: 'ok', data: await snapshot() }
      return { status: 'ok', data: {} }
    }),
  }
}

async function load() {
  vi.resetModules()
  return import('./appUpdateStore')
}

beforeEach(() => {
  mocks.getStudioClient.mockReset()
  mocks.actionSpy.mockReset()
  vi.stubGlobal('window', {
    localStorage: { getItem: () => null, setItem: () => {} },
    setTimeout: () => 0,
  })
})
afterEach(() => { vi.unstubAllGlobals() })

it('retries after a failed handshake instead of caching the failure', async () => {
  const good = fakeClient(async () => ({ currentVersion: '1.0.0' }))
  mocks.getStudioClient
    .mockRejectedValueOnce(new Error('replaced during handshake'))
    .mockResolvedValueOnce(good)
  const store = await load()

  await store.initialiseAppUpdates(false)
  expect(store.appUpdateState.available).toBe(false)

  await store.initialiseAppUpdates(false)
  expect(mocks.getStudioClient).toHaveBeenCalledTimes(2)
  expect(store.appUpdateState.available).toBe(true)
  expect(store.appUpdateState.currentVersion).toBe('1.0.0')
  expect(store.appUpdateState.statusMessage).toBe('')
})

it('a failed snapshot drops its subscription and the retry subscribes once on the new client', async () => {
  const broken = fakeClient(async () => { throw new Error('snapshot failed') })
  const good = fakeClient(async () => ({ currentVersion: '2.0.0' }))
  mocks.getStudioClient.mockResolvedValueOnce(broken).mockResolvedValueOnce(good)
  const store = await load()

  await store.initialiseAppUpdates(false)
  expect(store.appUpdateState.available).toBe(false)
  expect(broken.unsubscribe).toHaveBeenCalledTimes(1)

  await store.runAppUpdateAction('check')
  expect(good.subscribe).toHaveBeenCalledTimes(1)
  expect(good.invoke).toHaveBeenCalledWith('app_update.execute', { action: 'check', payload: {} })
  expect(store.appUpdateState.available).toBe(true)
  expect(store.appUpdateState.statusMessage).not.toBe('snapshot failed')
})

it('a later message is kept: only the init failure text is cleared', async () => {
  const good = fakeClient(async () => ({ currentVersion: '4.0.0' }))
  mocks.getStudioClient
    .mockRejectedValueOnce(new Error('handshake failed'))
    .mockResolvedValueOnce(good)
  const store = await load()

  await store.initialiseAppUpdates(false)
  store.appUpdateState.statusMessage = 'checking...'
  await store.initialiseAppUpdates(false)
  expect(store.appUpdateState.statusMessage).toBe('checking...')
})

it('a successful initialisation is still shared, not repeated', async () => {
  const good = fakeClient(async () => ({ currentVersion: '3.0.0' }))
  mocks.getStudioClient.mockResolvedValue(good)
  const store = await load()

  await store.initialiseAppUpdates(false)
  await store.initialiseAppUpdates(false)
  expect(mocks.getStudioClient).toHaveBeenCalledTimes(1)
})
