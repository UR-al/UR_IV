import { describe, expect, it } from 'vitest'
import {
  CLIPBOARD_IMAGE_TYPES, NO_CLIPBOARD_IMAGE, acceptsTextPaste, bytesToBase64, parsePasteResponse,
  pickClipboardImageType,
} from './clipboardImage'

/** 느리지만 확실한 기준 구현 — 바이트마다 한 글자씩 */
function referenceBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

function pseudoRandomBytes(n: number): Uint8Array {
  const out = new Uint8Array(n)
  let x = 12345
  for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) >>> 0; out[i] = x >>> 24 }
  return out
}

describe('bytesToBase64', () => {
  it('encodes multi-megabyte buffers without a stack overflow (spread btoa failed past ~120KB)', () => {
    const bytes = pseudoRandomBytes(3 * 1024 * 1024 + 7)
    const b64 = bytesToBase64(bytes)
    expect(b64).toBe(referenceBase64(bytes))
    // 예전 방식은 이 크기에서 스택을 넘는다 — 회귀 테스트가 실제로 그 경계를 넘는지 확인
    expect(() => btoa(String.fromCharCode(...bytes))).toThrow(RangeError)
  })

  it('matches the reference for small inputs and odd chunk sizes', () => {
    for (const n of [0, 1, 2, 3, 4, 5, 100, 0x8000, 0x8001]) {
      const bytes = pseudoRandomBytes(n)
      expect(bytesToBase64(bytes, 7)).toBe(referenceBase64(bytes))
      expect(bytesToBase64(bytes)).toBe(referenceBase64(bytes))
    }
  })
})

describe('pickClipboardImageType', () => {
  it('prefers a supported image type', () => {
    expect(pickClipboardImageType(['text/html', 'image/png'])).toEqual({ type: 'image/png' })
    expect(pickClipboardImageType(['IMAGE/JPEG'])).toEqual({ type: 'IMAGE/JPEG' })
  })

  it('reports unsupported image formats and ignores non-images', () => {
    expect(pickClipboardImageType(['image/heic'])).toEqual({ unsupported: 'image/heic' })
    expect(pickClipboardImageType(['text/plain'])).toBeNull()
    expect(pickClipboardImageType([])).toBeNull()
  })

  it('whitelist matches the backend PASTE_EXTENSIONS keys', () => {
    expect([...CLIPBOARD_IMAGE_TYPES].sort()).toEqual(
      ['image/bmp', 'image/jpeg', 'image/jpg', 'image/png', 'image/webp'],
    )
  })
})

describe('parsePasteResponse', () => {
  it('maps backend replies to outcomes', () => {
    expect(parsePasteResponse('{"path":"C:/tmp/clipboard_x.png"}')).toEqual({ kind: 'path', path: 'C:/tmp/clipboard_x.png' })
    expect(parsePasteResponse('{"empty":true,"error":"클립보드에 이미지가 없습니다"}'))
      .toEqual({ kind: 'info', msg: '클립보드에 이미지가 없습니다' })
    expect(parsePasteResponse('{"empty":true}')).toEqual({ kind: 'info', msg: NO_CLIPBOARD_IMAGE })
    expect(parsePasteResponse('{"error":"너무 큽니다"}')).toEqual({ kind: 'error', msg: '너무 큽니다' })
    expect(parsePasteResponse('not json')).toEqual({ kind: 'error', msg: '붙여넣기 실패' })
    expect(parsePasteResponse({ path: '' })).toEqual({ kind: 'error', msg: '붙여넣기 실패' })
  })
})

describe('acceptsTextPaste', () => {
  it('lets text fields keep Ctrl+V for text', () => {
    expect(acceptsTextPaste({ tagName: 'INPUT', type: 'text' })).toBe(true)
    expect(acceptsTextPaste({ tagName: 'input', type: '' })).toBe(true)
    expect(acceptsTextPaste({ tagName: 'TEXTAREA' })).toBe(true)
    expect(acceptsTextPaste({ tagName: 'DIV', isContentEditable: true })).toBe(true)
  })

  it('still pastes an image while a slider, checkbox or button has focus', () => {
    expect(acceptsTextPaste({ tagName: 'INPUT', type: 'range' })).toBe(false)
    expect(acceptsTextPaste({ tagName: 'INPUT', type: 'checkbox' })).toBe(false)
    expect(acceptsTextPaste({ tagName: 'INPUT', type: 'text', readOnly: true })).toBe(false)
    expect(acceptsTextPaste({ tagName: 'BUTTON' })).toBe(false)
    expect(acceptsTextPaste(null)).toBe(false)
  })
})
