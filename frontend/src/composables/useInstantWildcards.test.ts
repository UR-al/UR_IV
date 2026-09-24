import { afterEach, expect, it, vi } from 'vitest'
import { createInstantWildcards, INSTANT_WILDCARD_NAME } from './useInstantWildcards'

function setup() {
  const handlers = new Map<string, (...args: any[]) => void>()
  const requestAction = vi.fn()
  const addToast = vi.fn()
  const mgr = createInstantWildcards({ requestAction, addToast, onBackendEvent: (name, cb) => { handlers.set(name, cb) } })
  mgr.bind()
  handlers.get('instantWildcardsList')!(JSON.stringify([{ name: 'mood', lines: ['happy', '{100}:sad'] }]))
  return { mgr, handlers, requestAction, addToast }
}

afterEach(() => { vi.unstubAllGlobals() })

it('names allow only word characters, dots and slashes', () => {
  expect(INSTANT_WILDCARD_NAME.test('hair.color/v2_1')).toBe(true)
  expect(INSTANT_WILDCARD_NAME.test('bad name')).toBe(false)
  expect(INSTANT_WILDCARD_NAME.test('$$x$$')).toBe(false)
})

it('opening asks for the list; selecting copies the lines into the editor', () => {
  const { mgr, requestAction } = setup()
  mgr.openInstantWcManager()
  expect(requestAction).toHaveBeenCalledWith('instant_wildcards_list', {})
  mgr.selectInstantWc('mood')
  expect(mgr.iwEditLines.value).toEqual(['happy', '{100}:sad'])
  mgr.iwEditLines.value.push('x')
  expect(mgr.selectedInstantWcData.value?.lines).toEqual(['happy', '{100}:sad'])   // 복사본
})

it('create validates the name, rejects duplicates and saves an empty first line', () => {
  const { mgr, requestAction, addToast } = setup()
  vi.stubGlobal('window', { prompt: () => 'bad name' })
  mgr.createNewInstantWc()
  vi.stubGlobal('window', { prompt: () => 'mood' })
  mgr.createNewInstantWc()
  expect(addToast).toHaveBeenCalledWith('error', '이미 존재함')
  vi.stubGlobal('window', { prompt: () => 'pose' })
  mgr.createNewInstantWc()
  expect(requestAction).toHaveBeenCalledTimes(1)
  expect(requestAction).toHaveBeenCalledWith('instant_wildcards_save', { name: 'pose', lines: [''] })
  expect(mgr.selectedInstantWc.value).toBe('pose')
  expect(mgr.iwEditLines.value).toEqual([''])
})

it('save sends the edited lines and mirrors them locally', () => {
  const { mgr, requestAction, addToast } = setup()
  mgr.saveCurrentInstantWc()                          // 선택 없음 — 아무것도 안 한다
  expect(requestAction).not.toHaveBeenCalled()
  mgr.selectInstantWc('mood')
  mgr.appendIwLine()
  mgr.iwEditLines.value[2] = 'calm'
  mgr.removeIwLine(0)
  mgr.saveCurrentInstantWc()
  expect(requestAction).toHaveBeenCalledWith('instant_wildcards_save', { name: 'mood', lines: ['{100}:sad', 'calm'] })
  expect(mgr.selectedInstantWcData.value?.lines).toEqual(['{100}:sad', 'calm'])
  expect(addToast).toHaveBeenCalledWith('success', '인스턴트 와일드카드 저장: $$mood$$')
})

it('delete confirms and clears the editor when the open entry goes', () => {
  const { mgr, requestAction } = setup()
  mgr.selectInstantWc('mood')
  vi.stubGlobal('window', { confirm: () => false })
  mgr.deleteInstantWc('mood')
  expect(requestAction).not.toHaveBeenCalled()
  vi.stubGlobal('window', { confirm: () => true })
  mgr.deleteInstantWc('mood')
  expect(requestAction).toHaveBeenCalledWith('instant_wildcards_delete', { name: 'mood' })
  expect(mgr.instantWildcards.value).toEqual([])
  expect(mgr.selectedInstantWc.value).toBe('')
  expect(mgr.iwEditLines.value).toEqual([])
})
