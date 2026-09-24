/**
 * 드래그 앤 드롭으로 받은 이미지의 로컬 경로 (순수 로직).
 *
 * 일괄 처리·업스케일·ADetailer·SAM3 는 파일 **경로**가 필요하다(Python 워커가 파일을 연다).
 * 예전엔 `(file as any).path` 를 그대로 넣었는데, 그건 Electron 전용 속성이라
 * QtWebEngine·브라우저에는 없다 — OS 에서 끌어 온 파일마다 목록에 `undefined` 가 들어갔다.
 *
 *  - File 에 문자열 path 가 있으면 그 경로를 쓴다(있는 환경에서만).
 *  - 앱 안의 히스토리·갤러리 카드는 경로를 `text/plain` 으로 싣는다(App.vue onDragStart).
 *  - 경로를 알 수 없는 OS 파일은 `unresolved` 로 세어 호출자가 안내하게 한다.
 */
import { stripFileUrl } from './fileUrl'

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp)$/i

export interface DroppedPaths {
  paths: string[]
  /** 이미지 파일이지만 경로를 알 수 없어 건너뛴 개수 */
  unresolved: number
}

interface FileLike { name?: string; type?: string; path?: unknown }
interface TransferLike {
  files?: ArrayLike<FileLike> | null
  getData?: (format: string) => string
}

function normalizePath(raw: string): string {
  // file URL 만 디코드, 원시 경로의 '%' 는 이름의 일부(utils/fileUrl.ts — 백엔드와 같은 규칙)
  return stripFileUrl(raw.trim()).replace(/\\/g, '/')
}

function isImageFile(file: FileLike): boolean {
  return String(file.type || '').startsWith('image/') || IMAGE_EXT.test(String(file.name || ''))
}

export function droppedImagePaths(transfer: TransferLike | null | undefined): DroppedPaths {
  const paths: string[] = []
  let unresolved = 0
  const seen = new Set<string>()
  const add = (raw: string) => {
    const path = normalizePath(raw)
    if (!path || seen.has(path)) return
    seen.add(path)
    paths.push(path)
  }

  const files = Array.from(transfer?.files ?? [])
  if (files.length) {
    for (const file of files) {
      if (!isImageFile(file)) continue
      if (typeof file.path === 'string' && file.path.trim()) add(file.path)
      else unresolved += 1
    }
    return { paths, unresolved }
  }

  let text = ''
  try { text = transfer?.getData?.('text/plain') || '' } catch { text = '' }
  for (const line of text.split(/\r?\n/)) {
    const candidate = line.trim()
    if (/[\\/]/.test(candidate) && IMAGE_EXT.test(candidate.split('?')[0])) add(candidate)
  }
  return { paths, unresolved }
}

/** 이미 목록에 있는 경로를 뺀 새 항목만 (순서 유지). */
export function newPaths(existing: readonly string[], incoming: readonly string[]): string[] {
  const seen = new Set(existing)
  const out: string[] = []
  for (const path of incoming) {
    if (!path || seen.has(path)) continue
    seen.add(path)
    out.push(path)
  }
  return out
}
