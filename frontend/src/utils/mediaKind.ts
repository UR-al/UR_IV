/**
 * 갤러리·즐겨찾기 미디어 판별과 목록 정렬 — 확장자 기준 순수 함수.
 *
 * 예전엔 Gallery 에만 있어서 Favorites 는 영상·오디오도 <img> 로 그리고 모든 경로를
 * getImageExif(Pillow)로 보냈다. 두 뷰가 이 모듈 하나를 쓴다.
 */
import { imagePathKey } from './imageDeleteResult'

export type MediaKind = 'image' | 'video' | 'audio'
export type MediaSortKey = 'date' | 'name'

export const VIDEO_EXTENSIONS: ReadonlySet<string> = new Set(['mp4', 'webm', 'mov', 'mkv', 'm4v', 'avi', 'ogv'])
export const AUDIO_EXTENSIONS: ReadonlySet<string> = new Set(['wav', 'mp3', 'ogg', 'flac', 'm4a', 'aac', 'opus'])
/** 썸네일(정지 JPEG)로 줄이면 움직임이 사라지는 확장자 — 카드도 원본을 쓴다. */
export const ANIMATED_EXTENSIONS: ReadonlySet<string> = new Set(['gif', 'apng', 'webp'])

/** 쿼리·프래그먼트가 뜻을 갖는 진짜 URL — 로컬 경로(드라이브·UNC·상대·file:///)는 아니다. */
const URL_WITH_FRAGMENT = /^(https?|blob|aithumb|data):/i

/**
 * 확장자(소문자). 로컬 경로의 `#` 는 Windows 파일·폴더 이름에 쓸 수 있는 글자라 자르지 않는다 —
 * 예전엔 `C:/art#1/clip.mp4` 를 `C:/art` 로 잘라 영상·오디오를 이미지로, PNG 를 PNG 아님으로 봤다.
 * `?` 는 Windows 경로에 못 쓰므로 캐시 무력화 쿼리(`?t=`)로 보고 자른다.
 */
export function mediaExtension(path: string): string {
  const s = String(path || '')
  const clean = URL_WITH_FRAGMENT.test(s) ? s.split(/[?#]/, 1)[0] : s.split('?', 1)[0]
  const filename = clean.replace(/\\/g, '/').split('/').pop() || ''
  const dot = filename.lastIndexOf('.')
  return dot >= 0 ? filename.slice(dot + 1).toLowerCase() : ''
}

export function mediaKind(path: string): MediaKind {
  const ext = mediaExtension(path)
  if (VIDEO_EXTENSIONS.has(ext)) return 'video'
  if (AUDIO_EXTENSIONS.has(ext)) return 'audio'
  return 'image'
}

export const isVideo = (path: string): boolean => mediaKind(path) === 'video'
export const isAudio = (path: string): boolean => mediaKind(path) === 'audio'
export const isImage = (path: string): boolean => mediaKind(path) === 'image'
export const isAnimated = (path: string): boolean => ANIMATED_EXTENSIONS.has(mediaExtension(path))
/** 'EXIF 저장'은 PNG parameters 청크만 고친다. */
export const isPng = (path: string): boolean => mediaExtension(path) === 'png'

export function filenameOf(path: string): string {
  return String(path || '').replace(/\\/g, '/').split('/').pop() || String(path || '')
}

export function mediaLabel(path: string): string {
  const kind = mediaKind(path)
  if (kind === 'video') return 'VIDEO'
  if (kind === 'audio') return 'AUDIO'
  return isAnimated(path) ? 'ANIMATED' : 'IMAGE'
}

/**
 * 표시 순서. 'date' 는 백엔드가 준 순서(수정 시각 내림차순) 그대로, 'name' 은 파일 이름순.
 * 원본 배열을 바꾸지 않는다 — 예전 Gallery 는 제자리 정렬이라 새로고침·탭 재진입 뒤 목록은
 * 날짜순인데 '이름' 칩만 켜져 있었다.
 */
export function sortMediaPaths(list: readonly string[], sortBy: MediaSortKey | string): readonly string[] {
  if (sortBy !== 'name') return list
  return [...list].sort((a, b) => filenameOf(a).localeCompare(filenameOf(b)))
}

/** 이름 변경 반영 — oldPath 를 newPath 로 바꾼 새 목록. 바뀐 게 없으면 null. */
export function replaceMediaPath(list: readonly string[], oldPath: string, newPath: string): string[] | null {
  const key = imagePathKey(oldPath)
  let changed = false
  const next = list.map(item => {
    if (imagePathKey(item) !== key) return item
    changed = true
    return newPath
  })
  return changed ? next : null
}
