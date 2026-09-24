/**
 * 갤러리 확대 뷰 'EXIF 저장' 응답을 화면에 어떻게 반영할지.
 *
 * 예전 콜백은 응답이 오면 무조건 dirty 를 끄고 `{...현재, ...응답 info}` 로 덮었다.
 *  - 저장을 보낸 뒤(응답 전) 프롬프트를 또 고치고 포커스를 옮기면, 응답의 옛 텍스트가 그 편집을
 *    되돌리고 dirty 까지 꺼서 다시 저장해도 '바뀐 내용이 없습니다' 였다(두 번째 편집 유실).
 *  - A 를 저장하고 B 를 열어 고친 뒤 A 의 응답이 오면 B 의 dirty 를 꺼서 B 를 저장할 수 없었다.
 * 웹 모드(WebSocket 브리지)에서는 왕복·GUI 대기 동안 입력이 계속 들어와 실제로 일어난다.
 *
 * 규칙
 *  - 확대 뷰 세대(이미지를 열거나 닫으면 오름)가 다르거나 뷰가 닫혔으면 아무것도 건드리지 않는다.
 *    (같은 뷰에서 이름만 바뀐 경우는 같은 파일이다 — 경로 문자열이 아니라 세대로 본다)
 *  - 같은 뷰이고 보낸 뒤 고친 것이 없으면: 응답 info 로 갱신하고 저장됨(dirty=false).
 *  - 같은 뷰인데 보낸 뒤 또 고쳤으면: info 의 다른 필드(params_line 등)는 받되 지금 보이는
 *    프롬프트/네거티브는 지키고 dirty=true 로 남긴다.
 */

export interface ExifViewLike {
  path: string
  filename: string
  mediaType?: string
  prompt?: string
  negative?: string
  [k: string]: any
}

/** 저장을 보낼 때의 상태 */
export interface ExifSaveRequest {
  /** 확대 뷰 세대 — 이미지를 열거나 닫으면 오른다 */
  viewGen: number
  path: string
  /** 보낸 프롬프트/네거티브 원문 */
  prompt: string
  negative: string
}

export interface ExifSaveOutcome<T> {
  /** 새로 보일 데이터. null 이면 확대 뷰·사이드바를 건드리지 않는다 */
  view: T | null
  /** 새 dirty 값. null 이면 건드리지 않는다(다른 이미지의 편집 표시를 지우지 않게) */
  dirty: boolean | null
  /** 저장을 보낸 뒤 또 고쳤다 — 그 편집은 아직 저장되지 않았다 */
  editedSince: boolean
}

export function resolveExifSaveResponse<T extends ExifViewLike>(args: {
  sent: ExifSaveRequest
  currentViewGen: number
  current: T | null | undefined
  info: unknown
}): ExifSaveOutcome<T> {
  const { sent, currentViewGen, current, info } = args
  if (!current || currentViewGen !== sent.viewGen) {
    return { view: null, dirty: null, editedSince: false }
  }
  const editedSince = (current.prompt || '') !== sent.prompt || (current.negative || '') !== sent.negative
  let view: T | null = null
  if (info && typeof info === 'object' && !Array.isArray(info)) {
    // 경로·파일명·종류는 목록 표기를 유지한다(삭제·이름 변경·검색 캐시가 같은 키를 쓴다)
    const merged = {
      ...current, ...(info as Record<string, unknown>),
      path: current.path, filename: current.filename, mediaType: current.mediaType,
    } as T
    if (editedSince) {
      merged.prompt = current.prompt
      merged.negative = current.negative
    }
    view = merged
  }
  return { view, dirty: editedSince, editedSince }
}
