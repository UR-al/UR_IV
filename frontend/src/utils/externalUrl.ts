/**
 * 외부 링크 열기 — 데스크톱은 Python open_url(호스트 기본 브라우저), 웹 모드는 이 브라우저.
 *
 * 웹 모드에서 open_url 을 보내면 링크가 원격 기기가 아니라 호스트 PC 에서 열린다(그리고
 * Windows 에선 os.startfile 이라 서버가 거부한다 — core/url_safety.py). 그래서 웹 모드는
 * window.open 으로 직접 연다. 허용 기준(호스트가 있는 http/https)은 Python
 * is_safe_external_url 과 같다.
 */
import { requestAction } from '../stores/widgetStore.js'
import { isWebMode } from './media.js'

const MAX_EXTERNAL_URL_LENGTH = 8192

export function isSafeExternalUrl(raw: unknown): boolean {
  if (typeof raw !== 'string') return false
  const url = raw.trim()
  if (!url || url.length > MAX_EXTERNAL_URL_LENGTH) return false
  // 제어문자·공백·역슬래시는 브라우저와 셸이 다르게 해석하는 틈이 된다.
  if (/[\s\\\u0000-\u001f\u007f]/.test(url)) return false
  // 권한부(호스트)가 '//' 바로 뒤에 있어야 한다. WHATWG URL 은 'https:///host' 의 빈
  // 권한부를 건너뛰고 host 를 찾아 주지만, 파이썬 urlsplit 은 호스트 없음으로 본다 — 파이썬 기준을 따른다.
  const match = /^([a-z][a-z0-9+.-]*):\/\/(?!\/)/i.exec(url)
  if (!match || !['http', 'https'].includes(match[1].toLowerCase())) return false
  try {
    return Boolean(new URL(url).hostname)
  } catch {
    return false
  }
}

export interface OpenExternalDeps {
  webMode?: boolean
  openWindow?: (url: string, target: string, features: string) => unknown
  sendAction?: (name: string, payload: Record<string, unknown>) => void
}

/** 링크를 연다. 연 방법('window' | 'host') 또는 안전하지 않아 거부했으면 null. */
export function openExternalUrl(url: unknown, deps: OpenExternalDeps = {}): 'window' | 'host' | null {
  if (!isSafeExternalUrl(url)) return null
  const target = (url as string).trim()
  const webMode = deps.webMode ?? isWebMode()
  if (webMode) {
    const open = deps.openWindow ?? ((u: string, t: string, f: string) => window.open(u, t, f))
    open(target, '_blank', 'noopener,noreferrer')
    return 'window'
  }
  const send = deps.sendAction ?? ((name: string, payload: Record<string, unknown>) => requestAction(name as any, payload))
  send('open_url', { url: target })
  return 'host'
}
