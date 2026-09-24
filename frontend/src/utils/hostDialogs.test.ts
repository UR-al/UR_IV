import { describe, expect, it, vi } from 'vitest'
import {
  DESKTOP_DIALOG_ACTIONS,
  HOST_DIALOG_MESSAGE,
  applyHostDialogState,
  hostDialogActionAllowed,
  hostDialogsAvailable,
  isDesktopDialogAction,
  isWebHost,
  vHostDialog,
} from './hostDialogs'

const DESKTOP = {}
const LOOPBACK_WEB = { __AISTUDIO_WS_PORT__: 7801, __AISTUDIO_HOST_DIALOGS__: true }
const REMOTE_WEB = { __AISTUDIO_WS_PORT__: 7801, __AISTUDIO_HOST_DIALOGS__: false }

function fakeElement(tagName: string) {
  return {
    tagName,
    disabled: false,
    title: '',
    style: {} as { pointerEvents?: string, opacity?: string },
    attrs: {} as Record<string, string>,
    classes: [] as string[],
    setAttribute(name: string, value: string) { this.attrs[name] = value },
    classList: { add: vi.fn() },
  }
}

/** node 환경용 최소 DOM 요소 — 실제 EventTarget 이라 dispatch·stopImmediatePropagation 이 진짜로 돈다. */
function domElement(tagName: string) {
  const target = new EventTarget() as EventTarget & ReturnType<typeof fakeElement>
  return Object.assign(target, {
    tagName,
    disabled: false,
    title: '',
    style: {} as { pointerEvents?: string, opacity?: string, cursor?: string },
    attrs: {} as Record<string, string>,
    classes: [] as string[],
    setAttribute(name: string, value: string) { (this as any).attrs[name] = value },
    classList: { add: vi.fn() },
  })
}

describe('host dialog availability', () => {
  it('desktop and loopback web mode can show host dialogs, remote web mode cannot', () => {
    expect(hostDialogsAvailable(DESKTOP)).toBe(true)
    expect(hostDialogsAvailable(LOOPBACK_WEB)).toBe(true)
    expect(hostDialogsAvailable(REMOTE_WEB)).toBe(false)
    expect(hostDialogsAvailable({ __AISTUDIO_WS_URL__: 'ws://x', __AISTUDIO_HOST_DIALOGS__: false })).toBe(false)
  })

  it('older web servers without the flag are not blocked client-side (server still refuses)', () => {
    expect(hostDialogsAvailable({ __AISTUDIO_WS_PORT__: 7801 })).toBe(true)
  })

  it('knows which actions open host dialogs', () => {
    expect(isDesktopDialogAction('open_batch_files')).toBe(true)
    expect(isDesktopDialogAction(' Editor_Open_File ')).toBe(true)
    expect(isDesktopDialogAction('generate')).toBe(false)
    expect(isDesktopDialogAction('editor_save')).toBe(false)   // 저장은 대화상자 없음(원본 옆 _edited 사본)
    expect(isDesktopDialogAction(undefined)).toBe(false)
    expect(new Set(DESKTOP_DIALOG_ACTIONS).size).toBe(DESKTOP_DIALOG_ACTIONS.length)
  })

  it('knows when it runs in any web mode (host-authority actions are always refused there)', () => {
    expect(isWebHost(DESKTOP)).toBe(false)
    expect(isWebHost(LOOPBACK_WEB)).toBe(true)
    expect(isWebHost(REMOTE_WEB)).toBe(true)
    expect(isWebHost({ __AISTUDIO_WS_URL__: 'ws://x' })).toBe(true)
    expect(isWebHost(undefined)).toBe(false)
  })

  it('settings backup export and preset sharing are host-dialog actions', () => {
    for (const action of ['settings_export', 'presets_export', 'presets_import', 'character_presets_export', 'character_presets_import']) {
      expect(isDesktopDialogAction(action)).toBe(true)
    }
    expect(isDesktopDialogAction('save_preset')).toBe(false)   // 사문 액션 제거
  })

  it('only dialog actions are gated, and only in remote web mode', () => {
    expect(hostDialogActionAllowed('open_batch_files', REMOTE_WEB)).toBe(false)
    expect(hostDialogActionAllowed('generate', REMOTE_WEB)).toBe(true)
    expect(hostDialogActionAllowed('open_batch_files', LOOPBACK_WEB)).toBe(true)
    expect(hostDialogActionAllowed('open_batch_files', DESKTOP)).toBe(true)
  })
})

describe('applyHostDialogState', () => {
  it('disables buttons in remote web mode and explains why', () => {
    const el = fakeElement('BUTTON')
    expect(applyHostDialogState(el, 'caption_pick_folder', REMOTE_WEB)).toBe(true)
    expect(el.disabled).toBe(true)
    expect(el.title).toBe(HOST_DIALOG_MESSAGE)
    expect(el.attrs['aria-disabled']).toBe('true')
    expect(el.classList.add).toHaveBeenCalledWith('host-dialog-off')
  })

  it('blocks clicks on non-form elements such as the gallery folder bar but keeps the tooltip', () => {
    const el = domElement('DIV')
    const opened = vi.fn()
    applyHostDialogState(el, 'gallery_open_folder', REMOTE_WEB)
    el.addEventListener('click', opened)          // 템플릿 @click (directive created 뒤에 붙는다)
    // hover 가 살아 있어야 title(끈 이유)이 뜬다 — pointer-events 를 끄지 않는다.
    expect(el.style.pointerEvents).toBeUndefined()
    expect(el.title).toBe(HOST_DIALOG_MESSAGE)
    expect(el.style.opacity).toBe('0.45')
    expect(el.style.cursor).toBe('not-allowed')
    expect(el.disabled).toBe(false)
    expect(el.attrs['aria-disabled']).toBe('true')

    const click = new Event('click', { cancelable: true })
    el.dispatchEvent(click)
    expect(click.defaultPrevented).toBe(true)
    expect(opened).not.toHaveBeenCalled()

    for (const type of ['dblclick', 'auxclick']) {
      const ev = new Event(type, { cancelable: true })
      el.dispatchEvent(ev)
      expect(ev.defaultPrevented).toBe(true)
    }
  })

  it('blocks keyboard activation only (Tab navigation still works)', () => {
    const el = domElement('DIV')
    applyHostDialogState(el, 'gallery_open_folder', REMOTE_WEB)
    for (const [key, blocked] of [['Enter', true], [' ', true], ['Tab', false], ['a', false]] as const) {
      const ev = Object.assign(new Event('keydown', { cancelable: true }), { key })
      el.dispatchEvent(ev)
      expect(ev.defaultPrevented).toBe(blocked)
    }
  })

  it('re-applying (directive updated) never stacks duplicate blockers', () => {
    const el = domElement('DIV')
    const add = vi.spyOn(el, 'addEventListener')
    applyHostDialogState(el, 'gallery_open_folder', REMOTE_WEB)
    const first = add.mock.calls.length
    applyHostDialogState(el, 'gallery_open_folder', REMOTE_WEB)
    applyHostDialogState(el, 'gallery_open_folder', REMOTE_WEB)
    expect(first).toBe(4)   // click·dblclick·auxclick·keydown, 모두 캡처 단계
    expect(add.mock.calls.every(call => call[2] === true)).toBe(true)
    expect(add.mock.calls.length).toBe(first)
  })

  it('directive installs the blocker in created, before the template click handler', () => {
    vi.stubGlobal('window', REMOTE_WEB)
    try {
      const el = domElement('DIV')
      const opened = vi.fn()
      const binding = { value: 'gallery_open_folder' } as any
      ;(vHostDialog as any).created(el, binding)
      el.addEventListener('click', opened)   // Vue 가 created 뒤에 @click 을 붙인다
      ;(vHostDialog as any).mounted(el, binding)
      el.dispatchEvent(new Event('click', { cancelable: true }))
      expect(opened).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
    }
  })

  it('does not block non-form elements outside remote web mode', () => {
    const el = domElement('DIV')
    const opened = vi.fn()
    expect(applyHostDialogState(el, 'gallery_open_folder', LOOPBACK_WEB)).toBe(false)
    el.addEventListener('click', opened)
    el.dispatchEvent(new Event('click', { cancelable: true }))
    expect(opened).toHaveBeenCalledTimes(1)
    expect(el.title).toBe('')
  })

  it('leaves elements alone on desktop, loopback web mode and for unknown actions', () => {
    for (const [action, win] of [['open_batch_files', DESKTOP], ['open_batch_files', LOOPBACK_WEB], ['generate', REMOTE_WEB]] as const) {
      const el = fakeElement('BUTTON')
      expect(applyHostDialogState(el, action, win)).toBe(false)
      expect(el.disabled).toBe(false)
      expect(el.title).toBe('')
    }
  })

  it('directive re-applies after re-render (template :disabled cannot re-enable it)', () => {
    vi.stubGlobal('window', REMOTE_WEB)
    try {
      const el = fakeElement('BUTTON') as any
      const binding = { value: 'chat_export' } as any
      ;(vHostDialog as any).mounted(el, binding)
      expect(el.disabled).toBe(true)
      el.disabled = false   // Vue 가 :disabled 를 다시 그린 상황
      ;(vHostDialog as any).updated(el, binding)
      expect(el.disabled).toBe(true)
    } finally {
      vi.unstubAllGlobals()
    }
  })
})
