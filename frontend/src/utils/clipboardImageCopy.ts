/**
 * 갤러리·즐겨찾기 이미지 복사 — 누구의 클립보드에 쓰는가.
 *
 * - 데스크톱(Qt 창): 백엔드가 Qt 로 시스템 클립보드에 이미지를 넣는다(copy_to_clipboard 액션,
 *   결과 토스트도 백엔드가 보낸다).
 * - 웹 모드: **이 브라우저의** 클립보드에 쓴다. 예전엔 웹 복사 버튼도 copy_to_clipboard 를 보내
 *   원격 기기가 아니라 호스트 PC 사용자의 클립보드를 덮어쓰고 '복사했습니다'로 답했다. 이제 서버도
 *   웹 모드면 그 액션을 거부한다(core/web_action_policy.WEB_BLOCKED_ACTIONS) — 텍스트 복사
 *   (utils/clipboard.ts)와 같은 규칙. 원격 LAN http 는 보안 컨텍스트가 아니라 브라우저가
 *   Clipboard API 를 주지 않는다 → 'unsupported' 로 알린다.
 *
 * ClipboardItem 은 클릭 처리 안에서 **동기로** 만든다(Safari 는 사용자 동작 안에서 만든 항목에
 * Promise<Blob> 을 넘겨야 받아 준다). 크롬은 이미지로 image/png 만 쓰므로 JPEG·WebP 는 PNG 로 바꾼다.
 */
import { requestAction } from '../stores/widgetStore.js'
import { isWebMode, mediaUrl } from './media.js'

export type ImageCopyOutcome = 'host' | 'browser' | 'unsupported' | 'failed'

type ToastType = 'success' | 'error' | 'info' | 'warning'

type ClipboardLike = { write?: (items: ClipboardItem[]) => Promise<void> }
type ClipboardItemCtor = new (items: Record<string, Blob | PromiseLike<Blob>>) => ClipboardItem

export interface ImageCopyDeps {
  webMode?: boolean
  secureContext?: boolean
  clipboard?: ClipboardLike | null
  ClipboardItem?: ClipboardItemCtor | null
  fetchImpl?: (url: string, init?: RequestInit) => Promise<Response>
  toPng?: (blob: Blob) => Promise<Blob>
  sendAction?: (name: 'copy_to_clipboard', payload: { path: string }) => void
  toast?: (type: ToastType, msg: string) => void
}

export const IMAGE_COPY_MESSAGES = {
  browser: '이미지를 복사했습니다',
  unsupported:
    '이 연결(원격 HTTP)에서는 브라우저가 이미지 복사를 허용하지 않습니다. 이미지를 길게 누르거나 우클릭해 복사하세요.',
  failed: '이미지를 클립보드에 복사하지 못했습니다',
} as const

/** 이미지 Blob → PNG Blob(이미 PNG 면 그대로). */
export async function toPngBlob(blob: Blob): Promise<Blob> {
  if (blob.type === 'image/png') return blob
  const bitmap = await createImageBitmap(blob)
  try {
    if (typeof OffscreenCanvas !== 'undefined') {
      const canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('canvas 2d context unavailable')
      ctx.drawImage(bitmap, 0, 0)
      return await canvas.convertToBlob({ type: 'image/png' })
    }
    const canvas = document.createElement('canvas')
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('canvas 2d context unavailable')
    ctx.drawImage(bitmap, 0, 0)
    return await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(png => (png ? resolve(png) : reject(new Error('PNG encode failed'))), 'image/png')
    })
  } finally {
    bitmap.close?.()
  }
}

function defaultClipboard(): ClipboardLike | null {
  return typeof navigator !== 'undefined' && navigator.clipboard ? navigator.clipboard : null
}

function defaultClipboardItem(): ClipboardItemCtor | null {
  return typeof ClipboardItem !== 'undefined' ? ClipboardItem : null
}

/**
 * 이미지를 복사한다. 데스크톱이면 호스트('host'), 웹이면 이 브라우저('browser').
 * 웹인데 브라우저가 이미지 복사를 못 하면 'unsupported', 읽기·쓰기가 실패하면 'failed'.
 */
export async function copyImageToClipboard(path: string, deps: ImageCopyDeps = {}): Promise<ImageCopyOutcome> {
  const webMode = deps.webMode ?? isWebMode()
  if (!webMode) {
    // 리터럴 호출 — tests/test_bridge_contract.py 가 이 액션의 프론트 호출부로 읽는다.
    if (deps.sendAction) deps.sendAction('copy_to_clipboard', { path })
    else requestAction('copy_to_clipboard', { path })
    return 'host'
  }
  const secure = deps.secureContext ?? (typeof window !== 'undefined' && window.isSecureContext === true)
  const clipboard = deps.clipboard === undefined ? defaultClipboard() : deps.clipboard
  const Item = deps.ClipboardItem === undefined ? defaultClipboardItem() : deps.ClipboardItem
  if (!secure || !clipboard || typeof clipboard.write !== 'function' || !Item) return 'unsupported'

  const fetchImpl = deps.fetchImpl ?? ((url: string, init?: RequestInit) => fetch(url, init))
  const toPng = deps.toPng ?? toPngBlob
  // 첫 await 전에(= 클릭 처리 안에서 동기로) 항목을 만든다.
  // 캐시를 쓰지 않고 지금 파일을 읽는다(no-store). `/file?path=` 는 버전 없는 URL 이고 서버가 60초 동안
  // 신선하다고 답해(max-age=60), 같은 경로를 덮어쓴 뒤(에디터 재저장·같은 경로 생성) 다시 복사하면 옛 그림이
  // 클립보드에 들어가고도 '복사했습니다'가 떴다. 카드는 mediaVersion 으로 새 그림을 보여 주는데 복사만 옛것.
  // no-cache 가 아니라 no-store 인 이유: 조건부 요청(If-Modified-Since)은 1초 안에 덮어쓴 파일에 304 를
  // 받을 수 있다. 복사는 사용자가 한 번 누르는 동작이라 매번 읽어도 싸다.
  const png = fetchImpl(mediaUrl(path), { cache: 'no-store' }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response.blob()
  }).then(toPng)
  png.catch(() => { /* write 가 같은 실패로 거부된다 — 처리 안 된 거부로 남기지 않는다 */ })
  try {
    await clipboard.write([new Item({ 'image/png': png })])
    // 브라우저 구현이 Promise 를 끝까지 기다리지 않아도 실제 실패를 놓치지 않게 확인한다.
    await png
    return 'browser'
  } catch {
    return 'failed'
  }
}

/** 결과를 알린다 — 'host' 는 백엔드가 이미 알리므로 조용히 둔다. */
export function imageCopyToast(outcome: ImageCopyOutcome): { type: ToastType, msg: string } | null {
  if (outcome === 'browser') return { type: 'success', msg: IMAGE_COPY_MESSAGES.browser }
  if (outcome === 'unsupported') return { type: 'warning', msg: IMAGE_COPY_MESSAGES.unsupported }
  if (outcome === 'failed') return { type: 'error', msg: IMAGE_COPY_MESSAGES.failed }
  return null
}

/** 갤러리·즐겨찾기 복사 버튼 — 복사하고 결과를 토스트로. */
export async function copyImageAndNotify(path: string, deps: ImageCopyDeps = {}): Promise<ImageCopyOutcome> {
  const outcome = await copyImageToClipboard(path, deps)
  const note = imageCopyToast(outcome)
  if (note) {
    const toast = deps.toast ?? ((type: ToastType, msg: string) => requestAction('show_toast', { type, msg }))
    toast(note.type, note.msg)
  }
  return outcome
}
