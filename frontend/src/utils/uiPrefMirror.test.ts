import { describe, expect, it } from 'vitest'
import { PREF_MIRRORS, encodePrefForStorage, mirrorPrefsToStorage, readStoredBool } from './uiPrefMirror'

function memoryStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial))
  return {
    data,
    getItem: (k: string) => (data.has(k) ? data.get(k)! : null),
    setItem: (k: string, v: string) => { data.set(k, String(v)) },
  }
}

describe('uiPrefMirror', () => {
  it('mirrors valid prefs to their storage keys and reports what it accepted', () => {
    const storage = memoryStorage()
    const accepted = mirrorPrefsToStorage({
      tagBlockMode: true,
      galleryShowMetadata: false,
      uiScale: 1.25,
      tabOrder: ['T2I', 'Gallery'],
      historyJumpModifier: 'ctrlKey',
      highResFactor: 1.5,
      ratingFilter: [true, true, false, false],
      themeOverrides: { accent: '#fff' },   // 미러 없는 키 — 건너뛴다
    }, storage)
    expect(accepted.sort()).toEqual(
      ['galleryShowMetadata', 'highResFactor', 'historyJumpModifier', 'ratingFilter', 'tabOrder', 'tagBlockMode', 'uiScale'].sort(),
    )
    expect(storage.data.get('tagBlockMode')).toBe('true')
    expect(storage.data.get('galleryShowMetadata')).toBe('false')
    expect(storage.data.get('ui.scale')).toBe('1.25')
    expect(storage.data.get('tabOrder')).toBe('["T2I","Gallery"]')
    expect(storage.data.get('highRes.factor')).toBe('1.5')
    expect(storage.data.get('ratingFilter')).toBe('[true,true,false,false]')
    expect(storage.data.has('themeOverrides')).toBe(false)
  })

  it('rejects invalid values without touching the cache', () => {
    const storage = memoryStorage({ 'ui.scale': '1.1', historyJumpModifier: 'shiftKey' })
    const accepted = mirrorPrefsToStorage({
      uiScale: 9,                       // 범위 밖
      editorSidePanelWidth: 250.5,      // 정수 아님
      historyJumpModifier: 'metaKey',   // 허용 목록 밖
      tagBlockMode: 'true',             // 문자열은 불리언이 아니다
      tabOrder: [],                     // 빈 순서
      ratingFilter: [true, false],      // 길이 4 아님
    }, storage)
    expect(accepted).toEqual([])
    expect(storage.data.get('ui.scale')).toBe('1.1')
    expect(storage.data.get('historyJumpModifier')).toBe('shiftKey')
  })

  it('treats an empty Ollama model as a value (file wins over the cache)', () => {
    const storage = memoryStorage({ ollamaModel: 'gemma3:4b' })
    expect(mirrorPrefsToStorage({ ollamaModel: '' }, storage)).toEqual(['ollamaModel'])
    expect(storage.data.get('ollamaModel')).toBe('')
  })

  it('normalizes the preview thumbnail width to a preset', () => {
    expect(encodePrefForStorage('previewThumbWidth', 384)).toBe('384')
    expect(encodePrefForStorage('previewThumbWidth', 999)).toBe('256')
    expect(encodePrefForStorage('previewThumbWidth', undefined)).toBeNull()
  })

  it('still reports valid prefs when the storage is blocked', () => {
    const blocked = { getItem: () => null, setItem: () => { throw new Error('denied') } }
    expect(mirrorPrefsToStorage({ tagBlockMode: false }, blocked)).toEqual(['tagBlockMode'])
    expect(mirrorPrefsToStorage({ tagBlockMode: false }, null)).toEqual(['tagBlockMode'])
  })

  it('reads cached booleans with a fallback', () => {
    const storage = memoryStorage({ a: 'true', b: 'false', c: 'garbage' })
    expect(readStoredBool('a', false, storage)).toBe(true)
    expect(readStoredBool('b', true, storage)).toBe(false)
    expect(readStoredBool('c', true, storage)).toBe(true)
    expect(readStoredBool('missing', false, storage)).toBe(false)
    expect(readStoredBool('a', false, { getItem: () => { throw new Error('x') } })).toBe(false)
  })

  it('has one mirror per pref and per storage key', () => {
    const prefs = PREF_MIRRORS.map(m => m.pref)
    const keys = PREF_MIRRORS.map(m => m.storageKey)
    expect(new Set(prefs).size).toBe(prefs.length)
    expect(new Set(keys).size).toBe(keys.length)
  })
})
