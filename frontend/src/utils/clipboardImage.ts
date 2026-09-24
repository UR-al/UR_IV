/**
 * 에디터 클립보드 이미지 붙여넣기.
 *
 * 예전 EditorView 는 `btoa(String.fromCharCode(...new Uint8Array(buf)))` 로 base64 를 만들었다.
 * 인자 스프레드는 ~120KB 를 넘으면 'Maximum call stack size exceeded' RangeError 라, 1~3MB 인
 * 보통 스크린샷·생성 PNG 는 전부 '클립보드 접근 실패' 토스트로 끝났다. 게다가 데스크톱
 * QtWebEngine 페이지에는 navigator.clipboard.read() 권한이 없다.
 *
 * - 데스크톱: 백엔드가 Qt 로 시스템 클립보드를 직접 읽는다(editorPasteFromSystemClipboard).
 *   탐색기에서 복사한 이미지 파일도 여기서 잡힌다.
 * - 웹 모드(그 슬롯이 없음): 브라우저 클립보드 API → 청크 단위 base64 → editorPasteImage.
 */
import { getBackend } from '../bridge.js'

/** 백엔드(core/clipboard_paste.PASTE_EXTENSIONS)와 같은 5종 — HEIC/AVIF 등은 거부 */
export const CLIPBOARD_IMAGE_TYPES: readonly string[] = [
  'image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp',
]

/** 한 번에 String.fromCharCode 에 넘길 바이트 수 — 인자 개수 한도(~12만)보다 한참 작게 */
const BASE64_CHUNK = 0x8000

/** 바이트 → base64. 크기와 무관하게 스택을 넘지 않는다. */
export function bytesToBase64(bytes: Uint8Array, chunk: number = BASE64_CHUNK): string {
  const step = Math.max(1, Math.floor(chunk))
  const parts: string[] = []
  for (let i = 0; i < bytes.length; i += step) {
    parts.push(String.fromCharCode.apply(null, bytes.subarray(i, i + step) as unknown as number[]))
  }
  return btoa(parts.join(''))
}

export type ClipboardTypePick = { type: string } | { unsupported: string } | null

/**
 * 클립보드 항목의 MIME 목록에서 붙여넣을 형식을 고른다.
 * 허용 형식이 있으면 그것, 이미지이긴 한데 허용 밖이면 `unsupported`, 이미지가 없으면 null.
 */
export function pickClipboardImageType(types: readonly string[]): ClipboardTypePick {
  const found = types.find(t => CLIPBOARD_IMAGE_TYPES.includes(t.toLowerCase()))
  if (found) return { type: found }
  const other = types.find(t => t.toLowerCase().startsWith('image/'))
  return other ? { unsupported: other } : null
}

/** 글자를 붙여넣을 수 있는 입력칸인지 — 여기선 Ctrl+V 를 이미지 붙여넣기로 가로채지 않는다. */
const TEXT_INPUT_TYPES = new Set(['', 'text', 'search', 'url', 'email', 'tel', 'password', 'number'])
export function acceptsTextPaste(el: unknown): boolean {
  if (!el || typeof el !== 'object') return false
  const node = el as { tagName?: string; type?: string; isContentEditable?: boolean; readOnly?: boolean; disabled?: boolean }
  if (node.isContentEditable) return true
  const tag = String(node.tagName || '').toUpperCase()
  if (tag === 'TEXTAREA') return !node.readOnly && !node.disabled
  if (tag === 'INPUT') {
    return TEXT_INPUT_TYPES.has(String(node.type ?? '').toLowerCase()) && !node.readOnly && !node.disabled
  }
  return false
}

export type PasteOutcome =
  | { kind: 'path'; path: string }
  | { kind: 'info' | 'warning' | 'error'; msg: string }

export const NO_CLIPBOARD_IMAGE = '클립보드에 이미지가 없습니다'

/** 백엔드 붙여넣기 슬롯의 JSON 응답 → 결과. */
export function parsePasteResponse(json: unknown): PasteOutcome {
  let data: any
  try { data = typeof json === 'string' ? JSON.parse(json) : json } catch { data = null }
  if (data && typeof data.path === 'string' && data.path) return { kind: 'path', path: data.path }
  if (data && data.empty) return { kind: 'info', msg: String(data.error || NO_CLIPBOARD_IMAGE) }
  return { kind: 'error', msg: String((data && data.error) || '붙여넣기 실패') }
}

function callSlot(fn: (...args: any[]) => void, ...args: unknown[]): Promise<string> {
  return new Promise(resolve => fn(...args, (result: string) => resolve(result)))
}

function isDesktopHost(): boolean {
  return typeof window !== 'undefined' && !!(window as any).qt?.webChannelTransport
}

/** 클립보드 이미지를 임시 파일로 저장해(또는 복사한 원본 파일을 골라) 열 경로를 돌려준다. */
export async function pasteClipboardImage(): Promise<PasteOutcome> {
  const backend: any = await getBackend()
  if (isDesktopHost() && typeof backend?.editorPasteFromSystemClipboard === 'function') {
    return parsePasteResponse(await callSlot(backend.editorPasteFromSystemClipboard.bind(backend)))
  }
  if (!navigator.clipboard || !navigator.clipboard.read) {
    return { kind: 'info', msg: '브라우저가 클립보드 API를 지원하지 않습니다 — 파일로 열어주세요' }
  }
  const items = await navigator.clipboard.read()
  for (const item of items) {
    const pick = pickClipboardImageType(item.types)
    if (!pick) continue
    if ('unsupported' in pick) {
      return { kind: 'warning', msg: `지원 안 하는 이미지 포맷: ${pick.unsupported} (PNG/JPG/BMP/WEBP만 가능)` }
    }
    if (typeof backend?.editorPasteImage !== 'function') {
      return { kind: 'error', msg: '이 연결에서는 붙여넣기를 쓸 수 없습니다' }
    }
    const blob = await item.getType(pick.type)
    const b64 = bytesToBase64(new Uint8Array(await blob.arrayBuffer()))
    return parsePasteResponse(await callSlot(backend.editorPasteImage.bind(backend), b64, pick.type))
  }
  return { kind: 'info', msg: NO_CLIPBOARD_IMAGE }
}
