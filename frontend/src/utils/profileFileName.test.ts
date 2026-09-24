import { describe, expect, it } from 'vitest'
import { profileFileKey, profileSavedName } from './profileFileName'

// 입력 → [기록되는 이름, 실제 파일 이름] — tests/test_workflow_profiles.py 의 PROFILE_NAME_GOLDEN 과
// 같은 표(값은 파이썬 core/file_naming.sanitize_filename 으로 뽑았다). 한쪽 규칙만 바뀌면 둘 중 하나가 깨진다.
const GOLDEN: [string, string, string][] = [
  ['Flux', 'Flux', 'Flux'],
  ['  Flux.', 'Flux', 'Flux'],
  ['FLUX..', 'FLUX', 'FLUX'],
  ['Fl/ux', 'Flux', 'Flux'],
  ['Flux?', 'Flux', 'Flux'],
  ['a\\b:c*d"e<f>g|h', 'abcdefgh', 'abcdefgh'],
  ['tab\there', 'tabhere', 'tabhere'],
  ['con', '_con', '_con'],
  ['CON.json', '_CON.json', '_CON.json'],
  ['lpt9.x', '_lpt9.x', '_lpt9.x'],
  ['console', 'console', 'console'],
  ['???', '프로파일', '프로파일'],
  ['   ', '프로파일', '프로파일'],
  ['. .', '프로파일', '프로파일'],
  ['A'.repeat(64) + 'x', 'A'.repeat(64), 'A'.repeat(64)],
  ['A'.repeat(63) + '. b', 'A'.repeat(63) + '.', 'A'.repeat(63)],
  ['A'.repeat(63) + ' b', 'A'.repeat(63) + ' ', 'A'.repeat(63)],
  ['/ Flux', ' Flux', 'Flux'],
  ['/ con', ' con', '_con'],
  ['한글 프로파일', '한글 프로파일', '한글 프로파일'],
  ['\u{1F600}'.repeat(70), '\u{1F600}'.repeat(64), '\u{1F600}'.repeat(64)],
  ['con' + 'x'.repeat(70), 'con' + 'x'.repeat(61), 'con' + 'x'.repeat(61)],
]

describe('profile file name rule (mirror of core/file_naming.sanitize_filename)', () => {
  it.each(GOLDEN)('%j → saved %j, file %j', (typed, saved, stem) => {
    expect(profileSavedName(typed)).toBe(saved)
    expect(profileFileKey(typed)).toBe(stem.toLowerCase())
  })

  it('names that end up in the same file share a key; different files do not', () => {
    const key = profileFileKey('Flux')
    for (const typed of ['Flux ', 'Flux.', 'flux', 'FLUX..', 'Fl/ux', 'Flux?', 'F:lux', 'Flux"', '/ Flux', ': Flux.']) {
      expect(profileFileKey(typed)).toBe(key)
    }
    expect(profileFileKey('A'.repeat(64) + 'x')).toBe(profileFileKey('A'.repeat(64) + 'y'))
    expect(profileFileKey('Flux 2')).not.toBe(key)
    expect(profileFileKey('Flu x')).not.toBe(key)
  })
})
