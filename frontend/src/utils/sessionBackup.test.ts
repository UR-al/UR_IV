import { describe, expect, it } from 'vitest'
import { buildSessionBackup, parseSessionBackup, shouldOfferSessionRestore } from './sessionBackup'

describe('sessionBackup', () => {
  it('offers a crash backup whose prompt differs from what load_settings restored', () => {
    // 부팅은 prompt_settings.json 으로 프롬프트를 늘 채운다 — '비어 있을 때만' 이면 제안이 안 뜬다
    const backup = { prompt: 'edited after last save', negative: 'bad', clean: false }
    expect(shouldOfferSessionRestore(backup, { prompt: 'saved prompt', negative: 'bad' })).toBe(true)
    expect(shouldOfferSessionRestore(backup, { prompt: 'edited after last save', negative: 'other' })).toBe(true)
  })

  it('does not offer after a clean shutdown, for an identical or empty backup', () => {
    expect(shouldOfferSessionRestore({ prompt: 'x', clean: true }, { prompt: 'y' })).toBe(false)
    expect(shouldOfferSessionRestore({ prompt: ' same ', negative: 'n' }, { prompt: 'same', negative: 'n ' })).toBe(false)
    expect(shouldOfferSessionRestore({ prompt: '   ' }, { prompt: '' })).toBe(false)
    expect(shouldOfferSessionRestore(null, { prompt: '' })).toBe(false)
  })

  it('treats a legacy backup without the clean flag as unclean', () => {
    expect(shouldOfferSessionRestore({ tab: 't2i', prompt: 'a', negative: '' }, { prompt: 'b', negative: '' })).toBe(true)
  })

  it('parses bridge responses defensively', () => {
    expect(parseSessionBackup('{"prompt":"a"}')).toEqual({ prompt: 'a' })
    expect(parseSessionBackup('{"error":"boom"}')).toBeNull()
    expect(parseSessionBackup('not json')).toBeNull()
    expect(parseSessionBackup('[]')).toBeNull()
    expect(parseSessionBackup({ prompt: 'b' })).toEqual({ prompt: 'b' })
  })

  it('builds the payload saved to the backend', () => {
    expect(buildSessionBackup('i2i', { prompt: 'p', negative: 'n' }, 42)).toEqual({ tab: 'i2i', prompt: 'p', negative: 'n', savedAt: 42 })
    expect(buildSessionBackup(undefined, { prompt: null, negative: undefined }, 1)).toEqual({ tab: 't2i', prompt: '', negative: '', savedAt: 1 })
  })
})
