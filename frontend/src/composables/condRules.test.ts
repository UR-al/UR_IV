import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

// 조건식: 부팅 복원은 파일을 다시 쓰지 않고(거짓 '저장되었습니다' 토스트 없음), 파일이 이긴다.
// 캐시가 더 최신 편집이면 조용히 파일로 올린다. 편집은 800ms 뒤 조용히, 버튼만 _manual (감사 #108).
const mocks = vi.hoisted(() => ({
  actionSpy: vi.fn(),
  handlers: new Map<string, (...args: any[]) => void>(),
}))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))
vi.mock('../bridge.js', () => ({
  onBackendEvent: (name: string, cb: (...args: any[]) => void) => { mocks.handlers.set(name, cb); return () => {} },
}))

const rule = (condition: string, target: string) => ({ enabled: true, condition, exists: true, target, action: 'add', location: 'main' })
let storage: Map<string, string>

async function load() {
  vi.resetModules()
  return import('./condRules.js')
}
const saves = () => mocks.actionSpy.mock.calls.filter(([n]) => n === 'save_cond_rules').map(([, p]) => p)
const fileEvent = (payload: unknown) => mocks.handlers.get('condRulesLoaded')!(JSON.stringify(payload))

beforeEach(() => {
  vi.useFakeTimers()
  mocks.actionSpy.mockReset()
  mocks.handlers.clear()
  storage = new Map()
  vi.stubGlobal('window', {
    localStorage: {
      getItem: (k: string) => storage.get(k) ?? null,
      setItem: (k: string, v: string) => { storage.set(k, String(v)) },
    },
  })
})
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

it('boot restore of identical cache and file writes nothing', async () => {
  const saved = { enabled: true, positive: [rule('a', 'b')], negative: [], updatedAt: 100 }
  storage.set('searchCondRules', JSON.stringify(saved))
  const m = await load()
  m.loadCondRules()
  fileEvent(saved)
  await nextTick(); await nextTick()
  vi.advanceTimersByTime(5000)
  expect(saves()).toEqual([])
  expect(m.condPositive.map(r => r.condition)).toEqual(['a'])
})

it('the file wins over an older cache (manual edit / other web client) without a write back', async () => {
  storage.set('searchCondRules', JSON.stringify({ enabled: true, positive: [rule('old', 'x')], negative: [], updatedAt: 100 }))
  const m = await load()
  m.loadCondRules()
  expect(m.condPositive.map(r => r.condition)).toEqual(['old'])   // 첫 렌더 캐시
  fileEvent({ enabled: false, positive: [rule('new', 'y')], negative: [], updatedAt: 200 })
  await nextTick(); await nextTick()
  vi.advanceTimersByTime(5000)
  expect(m.condPositive.map(r => r.condition)).toEqual(['new'])
  expect(m.condEnabled.value).toBe(false)
  expect(saves()).toEqual([])
  expect(JSON.parse(storage.get('searchCondRules')!).positive[0].condition).toBe('new')
})

it('a newer cached edit (lost to the exit debounce) is uploaded silently', async () => {
  storage.set('searchCondRules', JSON.stringify({ enabled: true, positive: [rule('edited', 'z')], negative: [], updatedAt: 300 }))
  const m = await load()
  m.loadCondRules()
  fileEvent({ enabled: true, positive: [], negative: [], updatedAt: 200 })
  await nextTick()
  expect(m.condPositive.map(r => r.condition)).toEqual(['edited'])
  expect(saves()).toHaveLength(1)
  expect(saves()[0]).toMatchObject({ positive: [{ condition: 'edited' }], updatedAt: 300 })
  expect(saves()[0]).not.toHaveProperty('_manual')
})

it('edits autosave once after the debounce, silently; empty rules alone do not save', async () => {
  const m = await load()
  m.loadCondRules()
  fileEvent({ enabled: true, positive: [], negative: [] })
  await nextTick()
  m.addCondRule('pos')   // 빈 규칙 — 파일 내용은 그대로
  await nextTick()
  vi.advanceTimersByTime(2000)
  expect(saves()).toEqual([])
  m.condPositive[0].condition = 'hair'
  m.condPositive[0].target = 'long hair'
  await nextTick()
  vi.advanceTimersByTime(m.COND_RULES_SAVE_DELAY_MS - 1)
  expect(saves()).toEqual([])
  vi.advanceTimersByTime(1)
  expect(saves()).toHaveLength(1)
  expect(saves()[0]).not.toHaveProperty('_manual')
  expect(saves()[0].positive).toEqual([expect.objectContaining({ condition: 'hair', target: 'long hair' })])
  expect(saves()[0].updatedAt).toBeGreaterThan(0)
})

it('the master toggle saves immediately; the save button is the only manual (toast) save', async () => {
  const m = await load()
  m.loadCondRules()
  fileEvent({ enabled: true, positive: [rule('a', 'b')], negative: [] })
  await nextTick()
  m.condEnabled.value = false
  await nextTick()
  expect(saves()).toHaveLength(1)
  expect(saves()[0]).toMatchObject({ enabled: false })
  m.saveCondRules()
  expect(saves()).toHaveLength(2)
  expect(saves()[1]).toMatchObject({ _manual: true, enabled: false })
})

it('edits before the file arrives stay in the cache and are uploaded once it does', async () => {
  const base = { enabled: true, positive: [rule('keep', 'me')], negative: [], updatedAt: 1 }
  storage.set('searchCondRules', JSON.stringify(base))
  const m = await load()
  m.loadCondRules()
  m.condNegative.push(rule('nsfw', 'censored'))
  await nextTick()
  vi.advanceTimersByTime(2000)
  expect(saves()).toEqual([])   // 파일 상태를 모르면 보내지 않는다(캐시에만, 더 최신 updatedAt 으로)
  expect(JSON.parse(storage.get('searchCondRules')!).updatedAt).toBeGreaterThan(1)
  fileEvent(base)
  await nextTick()
  expect(saves()).toHaveLength(1)
  expect(saves()[0].positive).toEqual([expect.objectContaining({ condition: 'keep' })])
  expect(saves()[0].negative).toEqual([expect.objectContaining({ condition: 'nsfw' })])
})

it('after a settings-backup import the restamped file wins on the restart boot and nothing is written back', async () => {
  // 가져오기 전에 한 편집(캐시 5000)이 백업을 내보낸 때(1000)보다 새것이어도, 가져오기가 찍은 시각이 이긴다.
  storage.set('searchCondRules', JSON.stringify({ enabled: true, positive: [rule('pre-import', 'x')], negative: [], updatedAt: 5_000 }))
  const m = await load()
  m.loadCondRules()
  fileEvent({ enabled: true, positive: [rule('imported', 'y')], negative: [], updatedAt: 9_000 })
  await nextTick(); await nextTick()
  vi.advanceTimersByTime(5000)
  expect(m.condPositive.map(r => r.condition)).toEqual(['imported'])
  expect(saves()).toEqual([])   // 옛 캐시를 파일로 올리지 않는다
  const cached = JSON.parse(storage.get('searchCondRules')!)
  expect(cached.positive[0].condition).toBe('imported')
  expect(cached.updatedAt).toBe(9_000)
})
