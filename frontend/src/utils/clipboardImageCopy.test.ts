import { afterEach, describe, expect, it, vi } from 'vitest'
vi.mock('../stores/widgetStore.js', () => ({ requestAction: vi.fn() }))
import { requestAction } from '../stores/widgetStore.js'
import {
  IMAGE_COPY_MESSAGES,
  copyImageAndNotify,
  copyImageToClipboard,
  imageCopyToast,
  toPngBlob,
  type ImageCopyDeps,
} from './clipboardImageCopy'

afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks() })

class FakeItem {
  constructor(public items: Record<string, Blob | PromiseLike<Blob>>) {}
}

function webDeps(overrides: Partial<ImageCopyDeps> = {}) {
  const png = new Blob(['png-bytes'], { type: 'image/png' })
  const written: FakeItem[][] = []
  const clipboard = {
    write: vi.fn(async (items: ClipboardItem[]) => {
      const fakes = items as unknown as FakeItem[]
      written.push(fakes)
      for (const item of fakes) await item.items['image/png']   // 브라우저처럼 Promise 를 기다린다
    }),
  }
  const fetchImpl = vi.fn(async (_url: string) => ({ ok: true, status: 200, blob: async () => png }) as unknown as Response)
  const deps: ImageCopyDeps = {
    webMode: true,
    secureContext: true,
    clipboard,
    ClipboardItem: FakeItem as unknown as new (items: Record<string, Blob | PromiseLike<Blob>>) => ClipboardItem,
    fetchImpl,
    toPng: async (blob: Blob) => blob,
    ...overrides,
  }
  return { written, clipboard, fetchImpl, deps }
}

describe('copyImageToClipboard — 웹 클라이언트는 호스트 PC 클립보드를 쓰지 않는다 (Codex R3 #2)', () => {
  it('desktop asks the host to copy and never touches the browser clipboard', async () => {
    const { deps, clipboard, fetchImpl } = webDeps({ webMode: false, sendAction: undefined })
    expect(await copyImageToClipboard('C:/out/a.png', deps)).toBe('host')
    expect(requestAction).toHaveBeenCalledWith('copy_to_clipboard', { path: 'C:/out/a.png' })
    expect(clipboard.write).not.toHaveBeenCalled()
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('web mode copies into this browser from the /file endpoint', async () => {
    vi.stubGlobal('window', { __AISTUDIO_WS_PORT__: 8765 })
    const { deps, written, fetchImpl } = webDeps({ webMode: undefined })
    expect(await copyImageToClipboard('C:/out/그림 #1.png', deps)).toBe('browser')
    expect(requestAction).not.toHaveBeenCalled()
    expect(fetchImpl).toHaveBeenCalledWith('/file?path=' + encodeURIComponent('C:/out/그림 #1.png'), { cache: 'no-store' })
    expect(written).toHaveLength(1)
    expect(Object.keys(written[0][0].items)).toEqual(['image/png'])
    expect(await written[0][0].items['image/png']).toBeInstanceOf(Blob)
  })

  it('insecure LAN http or a browser without image clipboard reports unsupported without any request', async () => {
    for (const overrides of [
      { secureContext: false },
      { clipboard: null },
      { clipboard: {} },
      { ClipboardItem: null },
    ]) {
      const { deps, fetchImpl } = webDeps(overrides)
      expect(await copyImageToClipboard('C:/out/a.png', deps)).toBe('unsupported')
      expect(fetchImpl).not.toHaveBeenCalled()
    }
    expect(requestAction).not.toHaveBeenCalled()
  })

  it('reports failure when the image cannot be fetched or the browser refuses', async () => {
    const notFound = webDeps({
      fetchImpl: vi.fn(async () => ({ ok: false, status: 404, blob: async () => new Blob() }) as unknown as Response),
    })
    expect(await copyImageToClipboard('C:/gone.png', notFound.deps)).toBe('failed')

    const offline = webDeps({ fetchImpl: vi.fn(async () => { throw new Error('network') }) })
    expect(await copyImageToClipboard('C:/a.png', offline.deps)).toBe('failed')

    const denied = webDeps()
    denied.clipboard.write.mockRejectedValueOnce(new Error('NotAllowedError'))
    expect(await copyImageToClipboard('C:/a.png', denied.deps)).toBe('failed')
    expect(requestAction).not.toHaveBeenCalled()
  })

  it('re-copying a file overwritten in place puts the new image on the clipboard (no HTTP cache reuse)', async () => {
    vi.stubGlobal('window', { __AISTUDIO_WS_PORT__: 8765 })
    // 브라우저 HTTP 캐시 흉내: 서버가 max-age=60 으로 답한 /file?path= 는 60초 동안 요청 없이 캐시 본문을
    // 돌려준다 — no-store/no-cache/reload 만 네트워크로 간다.
    let onDisk = new Blob(['OLD-PIXELS'], { type: 'image/png' })
    const httpCache = new Map<string, Blob>()
    const fetchImpl = vi.fn(async (url: string, init?: RequestInit) => {
      const bypass = init?.cache === 'no-store' || init?.cache === 'no-cache' || init?.cache === 'reload'
      const cached = httpCache.get(url)
      const body = !bypass && cached ? cached : onDisk
      if (init?.cache !== 'no-store') httpCache.set(url, body)
      return { ok: true, status: 200, blob: async () => body } as unknown as Response
    })
    // 다른 화면(PNG 정보·채팅 첨부 등)이 같은 버전 없는 URL 을 먼저 읽어 캐시를 채웠다
    await fetchImpl('/file?path=' + encodeURIComponent('C:/out/a_edited.png'))
    const { deps, written } = webDeps({ webMode: undefined, fetchImpl })

    expect(await copyImageToClipboard('C:/out/a_edited.png', deps)).toBe('browser')
    expect(await (await written[0][0].items['image/png']).text()).toBe('OLD-PIXELS')

    onDisk = new Blob(['NEW-PIXELS'], { type: 'image/png' })   // 에디터가 같은 사본을 다시 저장했다
    expect(await copyImageToClipboard('C:/out/a_edited.png', deps)).toBe('browser')
    expect(await (await written[1][0].items['image/png']).text()).toBe('NEW-PIXELS')
  })

  it('creates the clipboard item synchronously inside the click (before any await)', () => {
    const { deps, clipboard } = webDeps()
    void copyImageToClipboard('C:/a.png', deps)
    expect(clipboard.write).toHaveBeenCalledOnce()
  })
})

describe('copyImageAndNotify', () => {
  it('toasts browser results and leaves host results to the backend', async () => {
    const toast = vi.fn()
    expect(await copyImageAndNotify('C:/a.png', { ...webDeps().deps, toast })).toBe('browser')
    expect(toast).toHaveBeenCalledWith('success', IMAGE_COPY_MESSAGES.browser)

    toast.mockClear()
    await copyImageAndNotify('C:/a.png', { webMode: false, sendAction: vi.fn(), toast })
    expect(toast).not.toHaveBeenCalled()

    await copyImageAndNotify('C:/a.png', { ...webDeps({ secureContext: false }).deps, toast })
    expect(toast).toHaveBeenCalledWith('warning', IMAGE_COPY_MESSAGES.unsupported)
  })

  it('uses the show_toast action by default', async () => {
    await copyImageAndNotify('C:/a.png', webDeps({ secureContext: false }).deps)
    expect(requestAction).toHaveBeenCalledWith('show_toast', { type: 'warning', msg: IMAGE_COPY_MESSAGES.unsupported })
  })

  it('maps every outcome', () => {
    expect(imageCopyToast('host')).toBeNull()
    expect(imageCopyToast('failed')).toEqual({ type: 'error', msg: IMAGE_COPY_MESSAGES.failed })
  })
})

describe('toPngBlob', () => {
  it('passes PNG through untouched', async () => {
    const png = new Blob(['x'], { type: 'image/png' })
    expect(await toPngBlob(png)).toBe(png)
  })

  it('re-encodes other formats to PNG', async () => {
    const close = vi.fn()
    vi.stubGlobal('createImageBitmap', vi.fn(async () => ({ width: 3, height: 2, close })))
    const drawImage = vi.fn()
    const converted = new Blob(['p'], { type: 'image/png' })
    const convertToBlob = vi.fn(async () => converted)
    vi.stubGlobal('OffscreenCanvas', class {
      constructor(public width: number, public height: number) {}
      getContext() { return { drawImage } }
      convertToBlob = convertToBlob
    })
    const out = await toPngBlob(new Blob(['j'], { type: 'image/jpeg' }))
    expect(out).toBe(converted)
    expect(convertToBlob).toHaveBeenCalledWith({ type: 'image/png' })
    expect(drawImage).toHaveBeenCalledOnce()
    expect(close).toHaveBeenCalledOnce()
  })
})
