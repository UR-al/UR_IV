/**
 * 워크플로 프로파일의 파일 이름 규칙 — 파이썬 core/file_naming.sanitize_filename(name, fallback='프로파일')
 * 의 사본(core/workflow_profiles 가 config/profiles/<이름>.json 을 이 규칙으로 만든다).
 *
 * 저장 때 금지 문자(\ / : * ? " < > |)·제어 문자·끝의 점과 공백이 사라지고 64자(코드 포인트)에서
 * 잘리며, Windows 는 대소문자를 가리지 않는다. 그래서 'Flux.' · 'flux' · 'Fl/ux' · 'Flux?' 는 모두
 * Flux.json 이다 — 목록과 글자 그대로만 비교하면 덮어쓰기 확인 없이 기존 프로파일을 바꾼다.
 * 백엔드도 확인받지 않은 덮어쓰기는 거절한다(최종 방어선) — 여기는 확인 창을 띄울지 고르는 쪽이다.
 * 두 구현이 갈라지지 않게 같은 표를 profileFileName.test.ts 와 tests/test_workflow_profiles.py 가 지킨다.
 */

const RESERVED = new Set<string>([
  'con', 'prn', 'aux', 'nul',
  ...Array.from({ length: 9 }, (_, i) => `com${i + 1}`),
  ...Array.from({ length: 9 }, (_, i) => `lpt${i + 1}`),
])

const MAX_LEN = 64
const FALLBACK = '프로파일'

/** 이 이름으로 저장하면 기록되는 프로파일 이름(= 파이썬 profile_saved_name). */
export function profileSavedName(name: string): string {
  let s = String(name ?? '').trim()
  s = s.replace(/[\\/:*?"<>|]/g, '')
  s = s.replace(/[\x00-\x1f]/g, '')
  s = s.replace(/[. ]+$/, '')
  if (RESERVED.has(s.split('.')[0].toLowerCase())) s = '_' + s
  // 파이썬 s[:64] 는 코드 포인트 단위다(이모지 한 글자 = 1)
  return Array.from(s).slice(0, MAX_LEN).join('') || FALLBACK
}

/**
 * 같은 파일을 가리키는가를 가르는 열쇠 — 실제 파일 이름(저장 경로는 기록되는 이름을 한 번 더 정규화한다:
 * 64자에서 잘려 드러난 끝의 점은 그때 지워진다)을 소문자로.
 */
export function profileFileKey(name: string): string {
  return profileSavedName(profileSavedName(name)).toLowerCase()
}
