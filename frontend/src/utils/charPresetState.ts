/**
 * 캐릭터 프리셋 모달의 per-캐릭터 작업 상태(칩 ON/OFF + 커스텀 태그) 영속 규칙.
 *
 * 예전엔 칩 목록 전체(기본값 포함, existing 칩의 false 까지)를 스냅샷으로 저장해서
 *  - 한 번이라도 열어 본 캐릭터엔 전역 카테고리 OFF·전역 제외 단어가 더는 적용되지 않았고
 *  - '특징 적용' 뒤 다시 열면 existing(이미 프롬프트에 있던) 칩의 false 가 저장돼,
 *    이후 빈 프롬프트에서 그 캐릭터의 특징이 전부 OFF 로 굳었다.
 * 그래서 **기본값(_defChecked)과 다른 항목만** 저장한다(diff). existing 칩은 사용자 의도를 알 수
 * 없으므로 새로 기록하지 않고 이전 저장값을 그대로 둔다. 키도 v2 로 올려 오염된 옛 `checked` 스냅샷은
 * 버린다. 단, 옛 키의 `custom`(사용자가 직접 넣은 태그 · 끈 프리셋 태그)은 그 키에만 있던 사용자
 * 입력이라 v2 로 옮긴다 — 예전엔 첫 저장 때 옛 키를 통째로 지워 모든 캐릭터의 커스텀 태그가 사라졌다.
 * 옛 `custom` 엔 프리셋이 준 기본 커스텀(ON)도 섞여 있어 바로 `custom` 에 넣지 않고 `legacyCustom` 에
 * 둔다 — 기본 커스텀을 아는 곳(모달이 캐릭터를 열 때, restoreCharState)에서 diff 규칙대로 정리한다.
 */

export interface ChipLike { tag: string; existing?: boolean; checked?: boolean }
export interface CustomChip { tag: string; checked: boolean }
export interface CharState {
  checked: Record<string, boolean>
  custom: CustomChip[]
  /**
   * 옛 스냅샷에서 옮겨 와 아직 정리하지 않은 커스텀 — 사용자 태그와 프리셋 기본 커스텀(ON)이 섞여 있다.
   * 정리 전까지는 `custom` 처럼 적용·보존하고, restoreCharState 가 기본 커스텀과 같은 것은 버린다.
   * 비어 있으면 키 자체를 두지 않는다.
   */
  legacyCustom?: CustomChip[]
}
export interface ChipGroup<T extends ChipLike = ChipLike> { cat: string; tags: T[] }

export const CHAR_STATE_KEY = 'cpmCharState.v2'
/** 전체 스냅샷을 담던 옛 키 — `checked` 는 오염돼 있어 버리고 `custom` 만 v2 로 옮긴 뒤 지운다. */
export const LEGACY_CHAR_STATE_KEYS: readonly string[] = ['cpmCharState']

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

export function normTag(s: string): string {
  return (s || '').trim().toLowerCase().replace(/_/g, ' ')
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === 'object' && !Array.isArray(v)
}

function sanitizeCustoms(raw: unknown): CustomChip[] {
  const out: CustomChip[] = []
  if (!Array.isArray(raw)) return out
  for (const c of raw) {
    if (!isRecord(c) || typeof c.tag !== 'string' || !c.tag.trim()) continue
    out.push({ tag: c.tag, checked: c.checked !== false })
  }
  return out
}

/** checked·custom·legacyCustom 으로 CharState 를 만든다 — 빈 legacyCustom 은 싣지 않고, 모두 비면 null. */
function makeCharState(
  checked: Record<string, boolean>, custom: CustomChip[], legacyCustom: CustomChip[] = [],
): CharState | null {
  if (!Object.keys(checked).length && !custom.length && !legacyCustom.length) return null
  return legacyCustom.length ? { checked, custom, legacyCustom } : { checked, custom }
}

/** 저장값을 안전한 모양으로 정규화 — 깨진 항목은 버린다. */
export function sanitizeCharState(raw: unknown): CharState | null {
  if (!isRecord(raw)) return null
  const checked: Record<string, boolean> = {}
  if (isRecord(raw.checked)) {
    for (const [k, v] of Object.entries(raw.checked)) {
      const n = normTag(k)
      if (n) checked[n] = !!v
    }
  }
  return makeCharState(checked, sanitizeCustoms(raw.custom), sanitizeCustoms(raw.legacyCustom))
}

/**
 * 현재 칩 상태에서 저장할 diff 를 만든다.
 * - 칩: existing 이 아니고 기본값과 다른 것만 기록.
 * - 지금 목록에 없거나 existing 인 태그의 이전 저장값은 보존(danbooru 교체·프롬프트 상태에 따라 사라지지 않게).
 * - 커스텀: 캐릭터 프리셋 기본 커스텀(baseCustom, 기본 ON)과 다른 것만 — 사용자가 추가한 태그, 끈 프리셋 태그.
 * - 아직 정리하지 않은 옛 커스텀(prev.legacyCustom)은 지금 커스텀 목록에 없는 것만 그대로 둔다 — 적용된
 *   것은 위 규칙으로 이미 다시 계산됐고, 적용되지 못한 것(특징 조회 실패 등)은 잃지 않는다.
 * 기록할 것이 없으면 null(=항목 삭제).
 */
export function diffCharState<T extends ChipLike>(
  groups: readonly ChipGroup<T>[],
  custom: readonly CustomChip[],
  defChecked: (cat: string, t: T) => boolean,
  baseCustom: readonly string[],
  prev: CharState | null,
): CharState | null {
  const checked: Record<string, boolean> = {}
  const decided = new Set<string>()
  for (const g of groups) {
    for (const t of g.tags) {
      if (t.existing) continue
      const n = normTag(t.tag)
      if (!n) continue
      decided.add(n)
      const on = !!t.checked
      if (on !== defChecked(g.cat, t)) checked[n] = on
    }
  }
  if (prev) {
    for (const [n, v] of Object.entries(prev.checked)) {
      if (!decided.has(n) && !(n in checked)) checked[n] = !!v
    }
  }
  const base = new Set(baseCustom.map(normTag))
  const seen = new Set<string>()
  const outCustom: CustomChip[] = []
  for (const c of custom) {
    const n = normTag(c.tag)
    if (!n || seen.has(n)) continue
    seen.add(n)
    const on = c.checked !== false
    if (!base.has(n) || !on) outCustom.push({ tag: c.tag, checked: on })
  }
  const outLegacy: CustomChip[] = []
  for (const c of prev?.legacyCustom || []) {
    const n = normTag(c.tag)
    if (!n || seen.has(n)) continue
    seen.add(n)
    outLegacy.push({ tag: c.tag, checked: c.checked !== false })
  }
  return makeCharState(checked, outCustom, outLegacy)
}

/**
 * 저장된 diff 를 기본값이 채워진 칩 위에 덮어쓴다. existing 칩은 건드리지 않는다.
 * 아직 정리하지 않은 옛 커스텀(legacyCustom)도 `custom` 과 똑같이 덮는다 — 기본 커스텀과 같은 것은
 * 이미 그 모양이라 달라지는 게 없다. 저장소를 정리하는 것은 restoreCharState 다.
 */
export function applyCharState<T extends ChipLike>(
  groups: readonly ChipGroup<T>[],
  custom: CustomChip[],
  saved: CharState | null,
): void {
  if (!saved) return
  const ck = saved.checked || {}
  for (const g of groups) {
    for (const t of g.tags) {
      const n = normTag(t.tag)
      if (!t.existing && n in ck) t.checked = !!ck[n]
    }
  }
  for (const c of [...(saved.custom || []), ...(saved.legacyCustom || [])]) {
    const found = custom.find(t => normTag(t.tag) === normTag(c.tag))
    if (found) found.checked = c.checked !== false
    else custom.push({ tag: c.tag, checked: c.checked !== false })
  }
}

function browserStorage(): StorageLike | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage
  } catch {
    return null
  }
}

function readV2Map(storage: StorageLike | null): Record<string, unknown> {
  try {
    const parsed = JSON.parse(storage?.getItem(CHAR_STATE_KEY) || '{}')
    return isRecord(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

/**
 * 옛 스냅샷 맵의 캐릭터별 `custom` 을 v2 맵 `m` 의 `legacyCustom` 에 합친다(`checked` 는 버린다).
 * v2 에 이미 있는 태그(normTag 기준, custom·legacyCustom)는 v2 값을 그대로 둔다. 바뀐 게 있으면 true.
 */
function mergeLegacyCustoms(m: Record<string, unknown>, legacy: Record<string, unknown>): boolean {
  let changed = false
  for (const [charKey, entry] of Object.entries(legacy)) {
    const old = sanitizeCharState(entry)
    if (!charKey || !old || !old.custom.length) continue
    const cur = sanitizeCharState(m[charKey])
    const legacyCustom = [...(cur?.legacyCustom || [])]
    const seen = new Set([...(cur?.custom || []), ...legacyCustom].map(c => normTag(c.tag)))
    let added = false
    for (const c of old.custom) {
      const n = normTag(c.tag)
      if (!n || seen.has(n)) continue
      seen.add(n)
      legacyCustom.push({ tag: c.tag, checked: c.checked !== false })
      added = true
    }
    if (added) { m[charKey] = makeCharState(cur?.checked || {}, cur?.custom || [], legacyCustom); changed = true }
  }
  return changed
}

/** 옛 키 하나 — `m` 에 합쳤지만 아직 지우지 못한 것. `compact` 는 커스텀만 남긴 작은 모양. */
interface PendingLegacy { key: string; compact: string }

/** 옛 맵에서 캐릭터별 `custom` 만 남긴다 — 자리를 많이 차지하는 오염된 `checked` 는 버린다. */
function compactLegacy(legacy: Record<string, unknown>): string {
  const out: Record<string, { custom: CustomChip[] }> = {}
  for (const [charKey, entry] of Object.entries(legacy)) {
    const s = sanitizeCharState(entry)
    if (charKey && s && s.custom.length) out[charKey] = { custom: s.custom }
  }
  return JSON.stringify(out)
}

/**
 * v2 맵 `m` 을 쓰고, 그 내용이 이미 `m` 에 합쳐진 옛 키(`pending`)를 지운다. 실패하면 throw.
 *
 * 용량이 모자라면 옛 키가 자리를 차지한 채 남아 이후의 모든 저장이 같은 이유로 실패했다(영영 저장 안 됨).
 * 그래서 막히면 ① 옛 키를 커스텀만 남긴 작은 모양으로 줄여 다시 쓰고 ② 그래도 안 되면 옛 키를 비워
 * 자리를 만든 뒤 다시 쓴다 — 이것도 실패하면 방금 비운 자리에 작은 모양을 되돌려 둔다(다음에 다시 옮긴다).
 * ①의 줄이기조차 막히는 저장소(쓰기 차단)에선 옛 키를 원래대로 둔다 — 지우면 되돌릴 수 없다.
 */
function commitV2(storage: StorageLike, m: Record<string, unknown>, pending: readonly PendingLegacy[]): void {
  const json = JSON.stringify(m)
  try {
    storage.setItem(CHAR_STATE_KEY, json)
  } catch (err) {
    if (!pending.length) throw err
    for (const p of pending) storage.setItem(p.key, p.compact)   // ① — 실패하면 옛 키는 원래대로다
    try {
      storage.setItem(CHAR_STATE_KEY, json)
    } catch {
      for (const p of pending) storage.removeItem(p.key)         // ②
      try {
        storage.setItem(CHAR_STATE_KEY, json)
      } catch (err2) {
        for (const p of pending) {
          try { storage.setItem(p.key, p.compact) } catch { /* 방금 비운 자리 — 들어간다 */ }
        }
        throw err2
      }
    }
  }
  for (const p of pending) storage.removeItem(p.key)
}

/**
 * 옛 키가 남아 있으면 그 `custom` 을 v2 맵 `m` 으로 옮긴다(여러 번 불러도 같다).
 * v2 쓰기가 성공한 뒤에만 옛 키를 지운다(commitV2) — 끝내 막히면 옛 키(또는 그 작은 모양)를 남겨
 * 다음에 다시 옮기고, 이번 화면에는 메모리의 `m` 으로 보여 준다. 읽을 수 없는(깨진) 옛 값과 옮길 것이
 * 없는 옛 값은 살릴 게 없어 지운다.
 * 반환값: `m` 에는 합쳤지만 아직 지우지 못한 옛 키 — 호출자가 `m` 을 저장할 때 commitV2 로 넘긴다.
 */
function migrateLegacyCharState(m: Record<string, unknown>, storage: StorageLike | null): PendingLegacy[] {
  const pending: PendingLegacy[] = []
  if (!storage) return pending
  for (const legacyKey of LEGACY_CHAR_STATE_KEYS) {
    let entry: PendingLegacy | null = null
    try {
      const raw = storage.getItem(legacyKey)
      if (raw == null) continue
      let parsed: unknown = null
      try { parsed = JSON.parse(raw) } catch { parsed = null }
      if (isRecord(parsed) && mergeLegacyCustoms(m, parsed)) {
        entry = { key: legacyKey, compact: compactLegacy(parsed) }
        commitV2(storage, m, [entry])
      } else {
        storage.removeItem(legacyKey)
      }
    } catch {
      // 저장이 막혔다 — 옛 키를 남겨 둔다
      if (entry) pending.push(entry)
    }
  }
  return pending
}

/** v2 맵 — 옛 키의 커스텀 태그가 남아 있으면 먼저 옮겨 합친다. */
function readMap(storage: StorageLike | null): Record<string, unknown> {
  const m = readV2Map(storage)
  migrateLegacyCharState(m, storage)
  return m
}

export function loadCharState(key: string, storage: StorageLike | null = browserStorage()): CharState | null {
  if (!key) return null
  return sanitizeCharState(readMap(storage)[key])
}

/** state 가 null 이면 그 캐릭터 항목을 지운다. 옛 스냅샷 키는 커스텀 태그를 옮긴 뒤 정리한다. */
export function storeCharState(key: string, state: CharState | null, storage: StorageLike | null = browserStorage()): void {
  if (!key || !storage) return
  try {
    const m = readV2Map(storage)
    const pending = migrateLegacyCharState(m, storage)
    if (state) m[key] = state
    else delete m[key]
    // 옮긴 커스텀 태그가 함께 저장되면 옛 키를 지운다 — 용량이 모자라면 옛 키부터 줄이거나 비워 자리를 만든다
    commitV2(storage, m, pending)
  } catch {
    // 차단된 저장소 — 영속만 포기한다
  }
}

/**
 * 모달이 캐릭터를 열 때 — 저장값을 기본값 칩 위에 덮고(applyCharState), 옛 스냅샷에서 옮겨 온
 * 커스텀(legacyCustom)이 있으면 지금 프리셋 기본 커스텀(baseCustom)을 기준으로 정리해 곧바로 저장한다:
 * 기본 커스텀과 같은(ON) 것은 사용자 태그가 아니라 버리고, 나머지는 `custom` 으로 옮긴다.
 * 미뤄 두면 그 사이 프리셋에서 그 태그가 빠지거나 프리셋이 지워졌을 때, 옛 스냅샷에 딸려 온 프리셋
 * 태그가 '사용자가 켠 커스텀'으로 되살아나 다음 저장 때 굳었다. 칩 ON/OFF(checked)는 건드리지 않는다.
 * 반환값: 적용한 저장값.
 */
export function restoreCharState<T extends ChipLike>(
  key: string,
  groups: readonly ChipGroup<T>[],
  custom: CustomChip[],
  defChecked: (cat: string, t: T) => boolean,
  baseCustom: readonly string[],
  storage: StorageLike | null = browserStorage(),
): CharState | null {
  if (!key) return null
  const saved = loadCharState(key, storage)
  applyCharState(groups, custom, saved)
  if (saved?.legacyCustom?.length) {
    const cleaned = diffCharState(groups, custom, defChecked, baseCustom, null)
    storeCharState(key, makeCharState(saved.checked, cleaned?.custom || []), storage)
  }
  return saved
}
