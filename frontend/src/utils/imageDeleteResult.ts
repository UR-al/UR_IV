/**
 * delete_image 결과(imageDeleteResult) 처리 — 실제로 파일이 없어졌을 때만 목록에서 뺀다.
 *
 * 예전 갤러리·히스토리는 삭제 액션을 보내자마자 목록에서 먼저 지웠다. 휴지통 이동이
 * 실패하면(send2trash 없음·권한·잠김) '휴지통으로 옮기지 못했습니다' 가 뜨는데도 파일은
 * 목록과 폴더 캐시에서 사라졌고, 폴더를 다시 훑기 전까지 돌아오지 않았다.
 * 백엔드(core/image_delete.py)는 경로마다 {path, ok, removed, level, message} 를 돌려준다.
 *  - removed = 그 경로에 더는 파일이 없다(옮겼거나 원래 없었다) → 목록에서 뺀다.
 *  - 그 밖(실패·거부)은 목록을 건드리지 않는다. 알림은 showNotification 이 따로 띄운다.
 */
import { stripFileUrl } from './fileUrl'

export type ImageDeleteLevel = 'info' | 'warning' | 'error'

export interface ImageDeleteResult {
  /** 프론트가 보낸 경로 원문(백엔드가 그대로 돌려준다). */
  path: string
  /** 이번 요청으로 휴지통에 옮겼다. */
  ok: boolean
  /** 그 경로에 더는 파일이 없다 — 목록에서 빼도 된다. */
  removed: boolean
  level: ImageDeleteLevel
  message: string
}

/** 시그널 인자(JSON 문자열 또는 객체)를 검증해 읽는다. 모양이 틀리면 null — 목록을 건드리지 않는다. */
export function parseImageDeleteResult(raw: unknown): ImageDeleteResult | null {
  let data: unknown = raw
  if (typeof raw === 'string') {
    try { data = JSON.parse(raw) } catch { return null }
  }
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null
  const d = data as Record<string, unknown>
  if (typeof d.path !== 'string' || !d.path.trim()) return null
  const level: ImageDeleteLevel = d.level === 'info' || d.level === 'warning' ? d.level : 'error'
  return {
    path: d.path,
    ok: d.ok === true,
    removed: d.removed === true,
    level,
    message: typeof d.message === 'string' ? d.message : '',
  }
}

/**
 * 같은 파일을 가리키는 표기 차이('/' vs '\\', file:///, 중복 구분자)를 하나로 모은 비교 키.
 * 갤러리는 '/' 로, 히스토리는 백엔드가 준 표기 그대로 들고 있어 문자열 비교로는 어긋난다.
 * Windows 경로(드라이브·UNC)만 대소문자를 무시한다 — POSIX 경로는 대소문자가 다른 파일일 수 있다.
 * file URL 은 퍼센트 인코딩을 풀고 원시 경로의 '%' 는 글자 그대로 둔다(utils/fileUrl.ts —
 * 백엔드 strip_file_url 과 같은 규칙). 예전엔 스킴만 떼어 'file:///D:/a%20b.png' 가 'a b.png' 와
 * 어긋나고, 이름에 '%20' 이 글자 그대로 있는 형제 파일과 같은 키가 됐다.
 */
export function imagePathKey(path: string): string {
  let p = stripFileUrl(String(path ?? '').trim()).replace(/\\/g, '/')
  const unc = p.startsWith('//')
  p = (unc ? '//' : '') + p.slice(unc ? 2 : 0).replace(/\/{2,}/g, '/')
  if (unc || /^[a-z]:\//i.test(p)) p = p.toLowerCase()
  return p
}

export function isSameImagePath(a: string | null | undefined, b: string | null | undefined): boolean {
  if (!a || !b) return false
  return imagePathKey(a) === imagePathKey(b)
}

/** path 를 뺀 새 목록. 빠진 게 없으면 null(호출부가 불필요한 재할당·반응성 갱신을 건너뛰게). */
export function withoutImagePath(list: readonly string[], path: string): string[] | null {
  const key = imagePathKey(path)
  const next = list.filter(item => imagePathKey(item) !== key)
  return next.length === list.length ? null : next
}

/**
 * 히스토리(App.vue)에 삭제 결과를 반영한다. 바꿀 게 없으면 null.
 * 보고 있던 이미지가 지워졌으면 남은 목록의 첫 장으로 옮긴다(예전 동작과 같다).
 */
export function applyDeleteToHistory(
  history: readonly string[],
  current: string,
  result: ImageDeleteResult | null,
): { history: string[], current: string } | null {
  if (!result || !result.removed) return null
  const next = withoutImagePath(history, result.path)
  const currentGone = isSameImagePath(current, result.path)
  if (!next && !currentGone) return null
  const list = next ?? [...history]
  return { history: list, current: currentGone ? (list[0] || '') : current }
}
