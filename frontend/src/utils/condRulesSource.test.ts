import { describe, expect, it } from 'vitest'
import { condRulesContent, condRulesContentKey, pickCondRulesSource } from './condRulesSource'

const rule = (condition: string, target: string) => ({ enabled: true, condition, exists: true, target, action: 'add', location: 'main' })

describe('condRulesSource', () => {
  it('file wins when both copies are equally old or the file is newer', () => {
    const local = { positive: [rule('a', 'b')], updatedAt: 100 }
    expect(pickCondRulesSource(local, { positive: [rule('c', 'd')], updatedAt: 100 })).toBe('file')
    expect(pickCondRulesSource(local, { positive: [rule('c', 'd')], updatedAt: 200 })).toBe('file')
  })

  it('keeps a strictly newer local edit (lost to the exit debounce) so it can be uploaded', () => {
    expect(pickCondRulesSource({ positive: [rule('a', 'b')], updatedAt: 300 }, { positive: [], updatedAt: 200 })).toBe('local')
    expect(pickCondRulesSource({ positive: [rule('a', 'b')], updatedAt: 300 }, { positive: [] })).toBe('local')
  })

  it('legacy copies without timestamps: file wins unless the file is empty', () => {
    expect(pickCondRulesSource({ positive: [rule('a', 'b')] }, { positive: [rule('c', 'd')] })).toBe('file')
    expect(pickCondRulesSource({ positive: [rule('a', 'b')] }, { positive: [], negative: [] })).toBe('local')
    expect(pickCondRulesSource({ positive: [] }, { positive: [] })).toBe('file')
  })

  it('a file restamped by a settings-backup import beats a cache edited before the import', () => {
    // 백업 속 시각(내보낸 때 1000, 옛 백업은 없음)만 보면 그 뒤 편집한 캐시(5000)가 이겨 가져온 규칙을 되덮는다.
    // Python(core/settings_backup → cond_rules_store.stamp_cond_rules_file)이 가져온 시각을 찍는다.
    const cacheEditedAfterExport = { positive: [rule('mine', 'x')], updatedAt: 5_000 }
    expect(pickCondRulesSource(cacheEditedAfterExport, { positive: [rule('backup', 'y')], updatedAt: 1_000 })).toBe('local')
    expect(pickCondRulesSource(cacheEditedAfterExport, { positive: [rule('backup', 'y')] })).toBe('local')
    const importedAt = 9_000   // 가져온 시각 > 가져오기 전의 어떤 편집
    expect(pickCondRulesSource(cacheEditedAfterExport, { positive: [rule('backup', 'y')], updatedAt: importedAt })).toBe('file')
    expect(pickCondRulesSource(cacheEditedAfterExport, { positive: [], updatedAt: importedAt })).toBe('file')   // 빈 규칙 백업도
  })

  it('an unstamped legacy migration loses only to a stamped (newer) cache', () => {
    const migrated = { enabled: true, positive: [rule('legacy', 'z')], negative: [] }   // _migrate_legacy_cond_rules
    expect(pickCondRulesSource({ positive: [rule('old-cache', 'x')] }, migrated)).toBe('file')
    expect(pickCondRulesSource({ positive: [rule('recent', 'x')], updatedAt: 5_000 }, migrated)).toBe('local')
  })

  it('uses whichever copy exists', () => {
    expect(pickCondRulesSource(null, { positive: [] })).toBe('file')
    expect(pickCondRulesSource({ positive: [] }, null)).toBe('local')
    expect(pickCondRulesSource(null, null)).toBe('none')
  })

  it('content key ignores timestamps, control flags, empty rules and key order', () => {
    const a = { enabled: true, positive: [rule('x', 'y')], negative: [], updatedAt: 1, _manual: true }
    const reordered = { negative: [], positive: [{ location: 'main', action: 'add', target: 'y', exists: true, condition: 'x', enabled: true }], updatedAt: 999 }
    const withEmpty = { enabled: true, positive: [rule('x', 'y'), rule('', '')], negative: [] }
    expect(condRulesContentKey(a)).toBe(condRulesContentKey(reordered))
    expect(condRulesContentKey(a)).toBe(condRulesContentKey(withEmpty))
    expect(condRulesContentKey(a)).not.toBe(condRulesContentKey({ ...a, enabled: false }))
  })

  it('content defaults the master toggle to on', () => {
    expect(condRulesContent({ positive: [] }).enabled).toBe(true)
    expect(condRulesContent({ enabled: false }).enabled).toBe(false)
    expect(condRulesContent(null)).toEqual({ enabled: true, positive: [], negative: [] })
  })
})
