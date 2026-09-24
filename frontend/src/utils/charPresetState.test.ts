import { describe, expect, it } from 'vitest'
import {
  CHAR_STATE_KEY,
  applyCharState,
  diffCharState,
  loadCharState,
  normTag,
  restoreCharState,
  sanitizeCharState,
  storeCharState,
  type ChipGroup,
  type ChipLike,
  type CustomChip,
} from './charPresetState'

function memStorage(init: Record<string, string> = {}) {
  const data: Record<string, string> = { ...init }
  return {
    data,
    getItem: (k: string) => (k in data ? data[k] : null),
    setItem: (k: string, v: string) => { data[k] = v },
    removeItem: (k: string) => { delete data[k] },
  }
}

// localStorage 처럼 용량 한도가 있는 저장소 — 키+값 길이 합이 한도를 넘는 쓰기는 막힌다(같은 키 덮어쓰기는 차이만 센다)
function quotaStorage(limit: number, init: Record<string, string> = {}) {
  const st = memStorage(init)
  const usedAfter = (k: string, v: string) => Object.entries(st.data)
    .reduce((n, [key, val]) => n + (key === k ? 0 : key.length + val.length), k.length + v.length)
  return {
    ...st,
    setItem: (k: string, v: string) => {
      if (usedAfter(k, v) > limit) throw new Error('QuotaExceededError')
      st.data[k] = v
    },
  }
}

// 전역 설정 흉내: etc 카테고리 OFF, 'smile' 전역 제외, existing 은 항상 false
function makeDef(catOff: string[] = ['etc'], wordOff: string[] = []) {
  return (cat: string, t: ChipLike) => {
    if (t.existing) return false
    if (catOff.includes(cat)) return false
    if (wordOff.includes(normTag(t.tag))) return false
    return true
  }
}

function groups(core: ChipLike[], etc: ChipLike[] = []): ChipGroup[] {
  return [{ cat: 'core', tags: core }, { cat: 'etc', tags: etc }]
}

describe('diffCharState', () => {
  it('stores nothing when every chip is at its default', () => {
    const g = groups([{ tag: 'blue_eyes', checked: true }], [{ tag: 'smile', checked: false }])
    expect(diffCharState(g, [], makeDef(), [], null)).toBeNull()
  })

  it('never records existing chips (their false would freeze features OFF later)', () => {
    const g = groups([{ tag: 'blue eyes', existing: true, checked: false }, { tag: 'long hair', checked: true }])
    expect(diffCharState(g, [], makeDef(), [], null)).toBeNull()
  })

  it('records only user deviations from the default', () => {
    const g = groups(
      [{ tag: 'blue_eyes', checked: false }, { tag: 'long hair', checked: true }],
      [{ tag: 'smile', checked: true }],
    )
    const s = diffCharState(g, [], makeDef(), [], null)
    expect(s).toEqual({ checked: { 'blue eyes': false, smile: true }, custom: [] })
  })

  it('keeps earlier overrides for tags that are absent or existing now', () => {
    const prev = { checked: { 'blue eyes': false, 'twin tails': true }, custom: [] }
    const g = groups([{ tag: 'blue eyes', existing: true, checked: false }, { tag: 'long hair', checked: true }])
    const s = diffCharState(g, [], makeDef(), [], prev)
    expect(s).toEqual({ checked: { 'blue eyes': false, 'twin tails': true }, custom: [] })
  })

  it('drops an override once the chip is back to its default', () => {
    const prev = { checked: { 'blue eyes': false }, custom: [] }
    const g = groups([{ tag: 'blue eyes', checked: true }])
    expect(diffCharState(g, [], makeDef(), [], prev)).toBeNull()
  })

  it('stores user-added customs and disabled preset customs only', () => {
    const custom: CustomChip[] = [
      { tag: 'preset_a', checked: true },
      { tag: 'preset_b', checked: false },
      { tag: 'mine', checked: true },
    ]
    const s = diffCharState(groups([]), custom, makeDef(), ['preset a', 'preset_b'], null)
    expect(s).toEqual({ checked: {}, custom: [{ tag: 'preset_b', checked: false }, { tag: 'mine', checked: true }] })
  })
})

describe('applyCharState', () => {
  it('overlays saved values on defaults but leaves existing chips alone', () => {
    const core: ChipLike[] = [
      { tag: 'blue eyes', checked: true },
      { tag: 'long hair', existing: true, checked: false },
    ]
    const custom: CustomChip[] = [{ tag: 'preset a', checked: true }]
    applyCharState(groups(core), custom, {
      checked: { 'blue eyes': false, 'long hair': true },
      custom: [{ tag: 'preset_a', checked: false }, { tag: 'mine', checked: true }],
    })
    expect(core.map(t => t.checked)).toEqual([false, false])
    expect(custom).toEqual([{ tag: 'preset a', checked: false }, { tag: 'mine', checked: true }])
  })

  it('lets global category changes apply to characters that were only opened', () => {
    // 열어 보기만 한 캐릭터 → 저장 없음 → 전역 기본값이 그대로 적용된다
    const g = groups([{ tag: 'blue eyes', checked: true }])
    const saved = diffCharState(g, [], makeDef(), [], null)
    const reopened = groups([{ tag: 'blue eyes', checked: makeDef(['core', 'etc'])('core', { tag: 'blue eyes' }) }])
    applyCharState(reopened, [], saved)
    expect(reopened[0].tags[0].checked).toBe(false)
  })
})

describe('storage helpers', () => {
  it('round-trips per character, deletes empty entries and clears the legacy key', () => {
    const st = memStorage({ cpmCharState: '{"x":{"checked":{"a":false}}}' })
    storeCharState('hatsune miku', { checked: { 'blue eyes': false }, custom: [] }, st)
    expect(st.data.cpmCharState).toBeUndefined()
    expect(loadCharState('hatsune miku', st)).toEqual({ checked: { 'blue eyes': false }, custom: [] })
    storeCharState('hatsune miku', null, st)
    expect(JSON.parse(st.data[CHAR_STATE_KEY])).toEqual({})
    expect(loadCharState('hatsune miku', st)).toBeNull()
  })

  it('ignores the polluted legacy snapshot and broken data', () => {
    const st = memStorage({ cpmCharState: '{"miku":{"checked":{"a":false}}}', [CHAR_STATE_KEY]: 'not json' })
    expect(loadCharState('miku', st)).toBeNull()
    expect(sanitizeCharState({ checked: { A_b: 1 }, custom: [{ tag: '' }, { tag: 'x' }, 3] }))
      .toEqual({ checked: { 'a b': true }, custom: [{ tag: 'x', checked: true }] })
    expect(sanitizeCharState([1, 2])).toBeNull()
  })

  it('survives a storage that throws', () => {
    const bad = {
      getItem: () => { throw new Error('denied') },
      setItem: () => { throw new Error('denied') },
      removeItem: () => { throw new Error('denied') },
    }
    expect(loadCharState('k', bad)).toBeNull()
    expect(() => storeCharState('k', { checked: { a: true }, custom: [] }, bad)).not.toThrow()
  })
})

// 옛 키(cpmCharState)의 checked 는 오염돼 버리지만, custom 은 그 키에만 있던 사용자 입력이다 —
// 예전엔 첫 저장 때 옛 키를 통째로 지워 모든 캐릭터의 커스텀 태그가 조용히 사라졌다.
describe('legacy snapshot migration', () => {
  const legacy = () => JSON.stringify({
    'hatsune miku': {
      checked: { 'blue eyes': false, twintails: false },
      custom: [{ tag: 'my_miku_outfit', checked: true }, { tag: 'glowing eyes', checked: false }],
    },
    'kagamine rin': {
      checked: { 'short hair': false },
      custom: [{ tag: 'rin_bow', checked: true }, { tag: 'sailor collar', checked: false }],
    },
    'only checked': { checked: { a: false } },
  })

  it('carries custom tags (with their on/off) into v2 on the first read, dropping the polluted checked', () => {
    const st = memStorage({ cpmCharState: legacy() })
    // 옛 custom 엔 프리셋 기본 커스텀도 섞여 있다 — 정리 대기(legacyCustom)로 옮긴다
    expect(loadCharState('kagamine rin', st)).toEqual({
      checked: {},
      custom: [],
      legacyCustom: [{ tag: 'rin_bow', checked: true }, { tag: 'sailor collar', checked: false }],
    })
    expect(st.data.cpmCharState).toBeUndefined()
    const v2 = JSON.parse(st.data[CHAR_STATE_KEY])
    expect(Object.keys(v2).sort()).toEqual(['hatsune miku', 'kagamine rin'])
    expect(loadCharState('only checked', st)).toBeNull()
  })

  it('keeps every other character when one character is stored first', () => {
    const st = memStorage({ cpmCharState: legacy() })
    const x = { checked: { smile: true }, custom: [{ tag: 'mine', checked: true }] }
    storeCharState('hatsune miku', x, st)
    expect(st.data.cpmCharState).toBeUndefined()
    expect(loadCharState('hatsune miku', st)).toEqual(x)
    expect(loadCharState('kagamine rin', st)?.legacyCustom?.map(c => c.tag)).toEqual(['rin_bow', 'sailor collar'])
  })

  it('survives the modal flow (restore → unchanged diff → store) for the opened character', () => {
    const st = memStorage({ cpmCharState: legacy() })
    const custom: CustomChip[] = []                // 프리셋 기본 커스텀 없음
    const g = groups([{ tag: 'blue eyes', checked: true }])
    restoreCharState('hatsune miku', g, custom, makeDef(), [], st)
    expect(custom).toEqual([{ tag: 'my_miku_outfit', checked: true }, { tag: 'glowing eyes', checked: false }])
    expect(g[0].tags[0].checked).toBe(true)         // 옛 checked(false) 는 버렸다
    // 기본 커스텀이 없으니 옛 커스텀은 전부 사용자 태그 — 열자마자 정리돼 custom 으로 저장된다
    expect(loadCharState('hatsune miku', st)).toEqual({ checked: {}, custom })
    const next = diffCharState(g, custom, makeDef(), [], loadCharState('hatsune miku', st))
    storeCharState('hatsune miku', next, st)
    expect(loadCharState('hatsune miku', st)?.custom).toEqual(custom)
    expect(loadCharState('kagamine rin', st)?.legacyCustom?.length).toBe(2)   // 아직 안 연 캐릭터는 정리 대기
  })

  // 옛 스냅샷은 프리셋이 준 기본 커스텀(ON)까지 저장했다 — 그대로 사용자 커스텀으로 두면, 프리셋에서 그
  // 태그가 빠지거나 프리셋을 지운 뒤 '사용자가 켠 커스텀'으로 되살아나 다음 저장 때 굳었다
  it('drops preset-provided customs carried over from the old snapshot when the character is first opened', () => {
    const st = memStorage({
      cpmCharState: JSON.stringify({
        miku: {
          checked: { 'blue eyes': false },
          custom: [
            { tag: 'preset_t', checked: true }, { tag: 'preset_off', checked: false }, { tag: 'mine', checked: true },
          ],
        },
      }),
    })
    const custom: CustomChip[] = [{ tag: 'preset_t', checked: true }, { tag: 'preset off', checked: true }]
    const base = ['preset_t', 'preset off']
    restoreCharState('miku', groups([]), custom, makeDef(), base, st)
    expect(custom).toEqual([
      { tag: 'preset_t', checked: true }, { tag: 'preset off', checked: false }, { tag: 'mine', checked: true },
    ])
    expect(JSON.parse(st.data[CHAR_STATE_KEY]).miku).toEqual({
      checked: {}, custom: [{ tag: 'preset off', checked: false }, { tag: 'mine', checked: true }],
    })
    // 프리셋을 지운 뒤(기본 커스텀 없음) 다시 열어도 preset_t 는 사용자 커스텀으로 되살아나지 않는다
    const reopened: CustomChip[] = []
    restoreCharState('miku', groups([]), reopened, makeDef(), [], st)
    expect(reopened.map(c => c.tag)).toEqual(['preset off', 'mine'])
    expect(diffCharState(groups([]), reopened, makeDef(), [], loadCharState('miku', st))?.custom.map(c => c.tag))
      .toEqual(['preset off', 'mine'])
  })

  it('keeps an entry without old customs as it is on open (no rewrite)', () => {
    const v2 = JSON.stringify({ miku: { checked: { smile: true }, custom: [{ tag: 'now in preset', checked: true }] } })
    const st = memStorage({ [CHAR_STATE_KEY]: v2 })
    const custom: CustomChip[] = [{ tag: 'now in preset', checked: true }]
    restoreCharState('miku', groups([{ tag: 'smile', checked: true }]), custom, makeDef(), ['now in preset'], st)
    expect(st.data[CHAR_STATE_KEY]).toBe(v2)
  })

  it('a save made before the old customs were applied (features failed to load) keeps them', () => {
    const prev = { checked: {}, custom: [], legacyCustom: [{ tag: 'mine', checked: true }, { tag: 'typed', checked: false }] }
    const custom: CustomChip[] = [{ tag: 'new tag', checked: true }, { tag: 'Typed', checked: true }]
    expect(diffCharState(groups([]), custom, makeDef(), [], prev)).toEqual({
      checked: {},
      custom: [{ tag: 'new tag', checked: true }, { tag: 'Typed', checked: true }],
      legacyCustom: [{ tag: 'mine', checked: true }],
    })
  })

  it('does not override a v2 entry for the same tag (any spelling) but adds the missing ones', () => {
    const st = memStorage({
      cpmCharState: legacy(),
      [CHAR_STATE_KEY]: JSON.stringify({
        'hatsune miku': { checked: { smile: true }, custom: [{ tag: 'My Miku Outfit', checked: false }] },
      }),
    })
    expect(loadCharState('hatsune miku', st)).toEqual({
      checked: { smile: true },
      custom: [{ tag: 'My Miku Outfit', checked: false }],
      legacyCustom: [{ tag: 'glowing eyes', checked: false }],
    })
    // 다시 읽어도 늘어나지 않는다
    expect(loadCharState('hatsune miku', st)?.legacyCustom?.length).toBe(1)
  })

  it('keeps the legacy key when v2 cannot be written, but still shows the tags', () => {
    const st = memStorage({ cpmCharState: legacy() })
    const blocked = { ...st, setItem: () => { throw new Error('quota') } }
    expect(loadCharState('kagamine rin', blocked)?.legacyCustom?.map(c => c.tag)).toEqual(['rin_bow', 'sailor collar'])
    expect(() => storeCharState('kagamine rin', null, blocked)).not.toThrow()
    expect(st.data.cpmCharState).toBe(legacy())
    expect(st.data[CHAR_STATE_KEY]).toBeUndefined()
  })

  it('drops an unreadable legacy value (nothing to recover)', () => {
    const st = memStorage({ cpmCharState: 'not json' })
    expect(loadCharState('x', st)).toBeNull()
    expect(st.data.cpmCharState).toBeUndefined()
  })
})

// 저장소가 거의 찬 상태(같은 origin 의 historyImagesCache 등) — 예전엔 옛 키를 v2 쓰기가 성공한 뒤에만
// 지워서, 옛 스냅샷이 자리를 차지한 채 v2 쓰기가 계속 막혀 이후 캐릭터 상태가 영영 저장되지 않았다
describe('legacy migration under a full storage (quota)', () => {
  const customs = [{ tag: 'rin_bow', checked: true }, { tag: 'sailor collar', checked: false }]
  const filler = { historyImagesCache: 'x'.repeat(5000) }
  const size = (d: Record<string, string>) => Object.entries(d).reduce((n, [k, v]) => n + k.length + v.length, 0)
  const compact = JSON.stringify({ 'kagamine rin': { custom: customs } })
  const migratedV2 = JSON.stringify({ 'kagamine rin': { checked: {}, custom: [], legacyCustom: customs } })

  it('shrinks the old snapshot (drops its polluted checked) to make room, so saving is not stuck', () => {
    const polluted: Record<string, boolean> = {}
    for (let i = 0; i < 300; i++) polluted[`tag number ${i}`] = false
    const init = { ...filler, cpmCharState: JSON.stringify({ 'kagamine rin': { checked: polluted, custom: customs } }) }
    const st = quotaStorage(size(init) + 50, init)   // 50자만 남았다 — 옛 키가 그대로면 v2 가 들어갈 자리가 없다
    storeCharState('hatsune miku', { checked: { smile: true }, custom: [] }, st)
    expect(st.data.cpmCharState).toBeUndefined()
    expect(loadCharState('hatsune miku', st)).toEqual({ checked: { smile: true }, custom: [] })
    expect(loadCharState('kagamine rin', st)?.legacyCustom).toEqual(customs)
    storeCharState('hatsune miku', { checked: { smile: false }, custom: [] }, st)   // 이후 저장도 막히지 않는다
    expect(loadCharState('hatsune miku', st)?.checked).toEqual({ smile: false })
  })

  it('empties the old key to make room when shrinking is not enough, without losing its tags', () => {
    const init = { ...filler, cpmCharState: compact }                       // 이미 작은 모양 — 줄일 게 없다
    const limit = size(filler) + CHAR_STATE_KEY.length + migratedV2.length   // 옛 키를 비워야만 v2 가 들어간다
    expect(size(init)).toBeLessThanOrEqual(limit)
    const st = quotaStorage(limit, init)
    expect(loadCharState('kagamine rin', st)?.legacyCustom).toEqual(customs)
    expect(st.data.cpmCharState).toBeUndefined()
    expect(st.data[CHAR_STATE_KEY]).toBe(migratedV2)
  })

  it('puts the shrunk old key back when even the freed room is too small, and still shows the tags', () => {
    const init = { ...filler, cpmCharState: JSON.stringify({ 'kagamine rin': { checked: { a: false }, custom: customs } }) }
    const limit = size(filler) + CHAR_STATE_KEY.length + migratedV2.length - 1
    expect(size(init)).toBeLessThanOrEqual(limit)
    const st = quotaStorage(limit, init)
    expect(loadCharState('kagamine rin', st)?.legacyCustom).toEqual(customs)
    expect(st.data.cpmCharState).toBe(compact)            // 커스텀만 남긴 모양 — 다음에 다시 옮긴다
    expect(st.data[CHAR_STATE_KEY]).toBeUndefined()
    storeCharState('hatsune miku', null, st)
    expect(st.data.cpmCharState).toBe(compact)
    expect(loadCharState('kagamine rin', st)?.legacyCustom).toEqual(customs)
  })
})
