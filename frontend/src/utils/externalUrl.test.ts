import { describe, expect, it, vi } from 'vitest'
vi.mock('../stores/widgetStore.js', () => ({ requestAction: vi.fn() }))
import { requestAction } from '../stores/widgetStore.js'
import { isSafeExternalUrl, openExternalUrl } from './externalUrl'

describe('isSafeExternalUrl (mirrors core/url_safety.py)', () => {
  it('accepts http/https with a host', () => {
    for (const url of [
      'https://github.com/UR-al/UR_IV/releases',
      'http://example.com/release-notes?x=1#y',
      'HTTPS://Example.com:8443/path',
      '  https://example.com  ',
    ]) expect(isSafeExternalUrl(url), url).toBe(true)
  })

  it('rejects anything the host shell could execute or misread', () => {
    for (const url of [
      'file:///C:/Windows/System32/calc.exe',
      'C:\\Windows\\System32\\calc.exe',
      '\\\\server\\share\\run.exe',
      '//server/share/run.exe',
      'search-ms:query=x',
      'javascript:alert(1)',
      'data:text/html,<script>1</script>',
      'http:evil.example',
      'https:///missing-host',
      'https://exa mple.com',
      'https://example.com/\\..\\x',
      'https://example.com/\u0000',
      'mailto:someone@example.com',
      '',
      'https://' + 'a'.repeat(9000) + '.com',
    ]) expect(isSafeExternalUrl(url), url).toBe(false)
    expect(isSafeExternalUrl(undefined)).toBe(false)
    expect(isSafeExternalUrl(42)).toBe(false)
  })
})

describe('openExternalUrl', () => {
  it('web mode opens in this browser without asking the host PC', () => {
    const openWindow = vi.fn()
    const sendAction = vi.fn()
    expect(openExternalUrl(' https://example.com/r ', { webMode: true, openWindow, sendAction })).toBe('window')
    expect(openWindow).toHaveBeenCalledWith('https://example.com/r', '_blank', 'noopener,noreferrer')
    expect(sendAction).not.toHaveBeenCalled()
  })

  it('desktop hands the link to Python open_url', () => {
    const openWindow = vi.fn()
    const sendAction = vi.fn()
    expect(openExternalUrl('https://example.com', { webMode: false, openWindow, sendAction })).toBe('host')
    expect(sendAction).toHaveBeenCalledWith('open_url', { url: 'https://example.com' })
    expect(openWindow).not.toHaveBeenCalled()
  })

  it('refuses unsafe links in both modes', () => {
    const openWindow = vi.fn()
    const sendAction = vi.fn()
    expect(openExternalUrl('file:///C:/x.exe', { webMode: true, openWindow, sendAction })).toBeNull()
    expect(openExternalUrl('file:///C:/x.exe', { webMode: false, openWindow, sendAction })).toBeNull()
    expect(openWindow).not.toHaveBeenCalled()
    expect(sendAction).not.toHaveBeenCalled()
  })

  it('defaults to the widget store bridge on desktop', () => {
    vi.stubGlobal('window', {})
    try {
      openExternalUrl('https://example.com/default')
      expect(requestAction).toHaveBeenCalledWith('open_url', { url: 'https://example.com/default' })
    } finally {
      vi.unstubAllGlobals()
    }
  })
})
