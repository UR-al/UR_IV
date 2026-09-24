import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// 블록 모드·갤러리 메타는 모듈 전역 ref — 쓰는 곳(Settings)과 읽는 곳(PromptPanel·Gallery)이 같은 ref 를
// 보고, 바꾸면 곧바로 ui_prefs 에 저장한다(예전엔 localStorage 에만 써서 재시작하면 되돌아갔다).
const mocks = vi.hoisted(() => ({ actionSpy: vi.fn() }))
vi.mock('../stores/widgetStore.js', () => ({ requestAction: mocks.actionSpy }))

let storage: Map<string, string>
let storageListener: ((event: { key: string; newValue: string | null }) => void) | null

async function load() {
  vi.resetModules()
  return import('./uiPrefs')
}

beforeEach(() => {
  mocks.actionSpy.mockReset()
  storage = new Map([['tagBlockMode', 'true'], ['galleryShowMetadata', 'false']])
  storageListener = null
  const localStorage = {
    getItem: (k: string) => storage.get(k) ?? null,
    setItem: (k: string, v: string) => { storage.set(k, String(v)) },
  }
  vi.stubGlobal('localStorage', localStorage)
  vi.stubGlobal('window', {
    localStorage,
    addEventListener: (name: string, cb: any) => { if (name === 'storage') storageListener = cb },
  })
})
afterEach(() => { vi.unstubAllGlobals() })

it('starts from the localStorage cache (first render only)', async () => {
  const m = await load()
  expect(m.tagBlockMode.value).toBe(true)
  expect(m.galleryShowMetadata.value).toBe(false)
})

it('setters update the shared ref, the cache and ui_prefs immediately', async () => {
  const m = await load()
  m.setTagBlockMode(false)
  m.setGalleryShowMetadata(true)
  expect(m.tagBlockMode.value).toBe(false)
  expect(m.galleryShowMetadata.value).toBe(true)
  expect(storage.get('tagBlockMode')).toBe('false')
  expect(storage.get('galleryShowMetadata')).toBe('true')
  expect(mocks.actionSpy.mock.calls).toEqual([
    ['save_ui_prefs', { tagBlockMode: false }],
    ['save_ui_prefs', { galleryShowMetadata: true }],
  ])
})

it('file values win on restore without echoing a save', async () => {
  const m = await load()
  m.restoreUiFlagsFromPrefs({ tagBlockMode: false, galleryShowMetadata: true, other: 1 })
  expect(m.tagBlockMode.value).toBe(false)
  expect(m.galleryShowMetadata.value).toBe(true)
  expect(mocks.actionSpy).not.toHaveBeenCalled()
  m.restoreUiFlagsFromPrefs({ tagBlockMode: 'yes' } as any)   // 잘못된 값은 무시
  expect(m.tagBlockMode.value).toBe(false)
})

it('follows other browser tabs through the storage event (web mode)', async () => {
  const m = await load()
  storageListener!({ key: 'tagBlockMode', newValue: 'false' })
  storageListener!({ key: 'galleryShowMetadata', newValue: 'true' })
  expect(m.tagBlockMode.value).toBe(false)
  expect(m.galleryShowMetadata.value).toBe(true)
  storageListener!({ key: 'tagBlockMode', newValue: null })   // 지워진 키 — 무시
  expect(m.tagBlockMode.value).toBe(false)
})

it('persistUiPrefs mirrors the cache and saves once; empty payloads are skipped', async () => {
  const m = await load()
  m.persistUiPrefs({ uiScale: 1.2, searchState: { q: 'x' } })
  expect(storage.get('ui.scale')).toBe('1.2')
  expect(mocks.actionSpy).toHaveBeenCalledWith('save_ui_prefs', { uiScale: 1.2, searchState: { q: 'x' } })
  mocks.actionSpy.mockReset()
  m.persistUiPrefs({})
  expect(mocks.actionSpy).not.toHaveBeenCalled()
})
