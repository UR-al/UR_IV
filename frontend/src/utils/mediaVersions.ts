import { reactive } from 'vue'
import { imagePathKey } from './imageDeleteResult'

/**
 * 갤러리·즐겨찾기 카드 URL 의 내용 버전 — 같은 경로에 덮어쓴 파일을 옛 그림으로 보여 주지 않게.
 *
 * 카드 썸네일 URL(Qt `aithumb:thumb?path=&width=` / 웹 `/thumbnail?path=&width=`)은 경로·폭만 담는다.
 * '다른 이름으로 저장'·에디터 저장이 파일을 덮어쓰면 백엔드는 썸네일을 다시 만들지만(원본 서명 비교,
 * core/thumb_cache.py) 브라우저는 같은 URL 을 다시 묻지 않는다 — Qt 는 페이지 안 이미지 캐시,
 * 웹은 `Cache-Control: max-age=3600`. 히스토리 스트립은 썸네일 파일 버전(v)을 붙여 풀었고
 * (composables/useHistoryThumbs), 카드는 여기서 정한 버전을 `v=` 로 붙인다(media.js withUrlVersion).
 *
 * 버전 = `목록 버전.저장 버전` (있는 것만)
 *  - 목록 버전: 갤러리 목록(galleryImagesReady `versions`)이 준 원본 서명(mtime·크기). 목록을 다시
 *    읽을 때마다(탭 재진입·폴더 변경·F5) 앱 밖에서 덮어쓴 파일도 URL 이 바뀐다.
 *  - 저장 버전: 이 앱이 덮어쓴 경로(에디터 저장·다른 이름으로 저장, 같은 경로의 생성 결과)에 올리는
 *    시각 기반 카운터. 목록을 다시 읽지 않는 즐겨찾기도 바로 새로 읽고, 저장 전에 만든 목록이 늦게
 *    도착해도 이미 캐시된 옛 URL 로 돌아가지 않는다.
 * 키는 imagePathKey — 구분자(`\`·`/`)·드라이브 대소문자가 달라도 같은 파일이다.
 */

/** 목록 버전을 이만큼 넘게 쌓으면 비우고 새 목록부터 다시 담는다(폴더를 많이 돌아다닌 긴 세션). */
export const MAX_LISTED_VERSIONS = 50_000

const listed = reactive(new Map<string, string>())
const saved = reactive(new Map<string, number>())
let lastSaved = 0

/** 목록 버전과 저장 버전을 이은 카드 URL 버전(둘 다 없으면 ''). */
export function composeMediaVersion(listedVersion?: string | null, savedVersion?: number | null): string {
  const parts: string[] = []
  if (listedVersion) parts.push(String(listedVersion))
  if (savedVersion) parts.push(String(savedVersion))
  return parts.join('.')
}

/** 갤러리 목록 페이로드의 `files`·`versions`(같은 순서) → 경로별 목록 버전. 모양이 틀린 항목은 건너뛴다. */
export function listedVersionEntries(files: unknown, versions: unknown): Array<[string, string]> {
  if (!Array.isArray(files) || !Array.isArray(versions)) return []
  const out: Array<[string, string]> = []
  const count = Math.min(files.length, versions.length)
  for (let i = 0; i < count; i++) {
    const path = files[i]
    const version = versions[i]
    if (typeof path !== 'string' || !path) continue
    if (typeof version !== 'string' && typeof version !== 'number') continue
    const text = String(version)
    if (text) out.push([imagePathKey(path), text])
  }
  return out
}

/** 갤러리 목록이 준 원본 서명을 기록한다(목록마다 합친다 — 다른 폴더·히스토리 목록이 서로 지우지 않게). */
export function recordListedMediaVersions(files: unknown, versions: unknown): void {
  const entries = listedVersionEntries(files, versions)
  if (!entries.length) return
  if (listed.size + entries.length > MAX_LISTED_VERSIONS) listed.clear()
  for (const [key, version] of entries) {
    if (listed.get(key) !== version) listed.set(key, version)
  }
}

/** 이 앱이 `path` 를 덮어썼다 — 그 카드 URL 을 바꿔 새로 읽게 한다. */
export function bumpMediaVersion(path: string | null | undefined): void {
  if (!path) return
  const key = imagePathKey(path)
  if (!key) return
  // 시각 기반 — 새로고침 뒤 카운터가 0 부터 다시 세어 지난 세션의 (캐시된) URL 과 겹치지 않게
  lastSaved = Math.max(Date.now(), lastSaved + 1)
  saved.set(key, lastSaved)
}

/** 카드 URL 에 붙일 버전('' = 아는 버전 없음). 반응형 — 목록·저장 버전이 바뀌면 다시 그린다. */
export function mediaVersion(path: string | null | undefined): string {
  if (!path) return ''
  const key = imagePathKey(path)
  return composeMediaVersion(listed.get(key), saved.get(key))
}

/** 테스트 전용 — 모듈 상태를 비운다. */
export function resetMediaVersionsForTest(): void {
  listed.clear()
  saved.clear()
  lastSaved = 0
}
