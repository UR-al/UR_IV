import { afterEach, expect, it, vi } from 'vitest'
import { createWorkflowProfiles } from './useWorkflowProfiles'

function setup() {
  const handlers = new Map<string, (...args: any[]) => void>()
  const requestAction = vi.fn()
  const mgr = createWorkflowProfiles({
    requestAction,
    onBackendEvent: (name, cb) => { handlers.set(name, cb) },
  })
  mgr.bind()
  handlers.get('workflowProfilesList')!(JSON.stringify([{ name: 'ANIMA', model: 'anima.safetensors' }, { name: 'Flux' }]))
  return { mgr, handlers, requestAction }
}

function stubDialogs(promptValue: string | null, confirmValue = true) {
  vi.stubGlobal('window', { prompt: vi.fn(() => promptValue), confirm: vi.fn(() => confirmValue) })
}

afterEach(() => { vi.unstubAllGlobals() })

it('keeps the dropdown names in sync with workflowProfilesList and ignores non-arrays', () => {
  const { mgr, handlers } = setup()
  expect(mgr.profileNames.value).toEqual(['ANIMA', 'Flux'])
  handlers.get('workflowProfilesList')!('{"not":"a list"}')
  handlers.get('workflowProfilesList')!('broken')
  expect(mgr.profileNames.value).toEqual(['ANIMA', 'Flux'])
})

it('the gear opens the manager and asks for a fresh list', () => {
  const { mgr, requestAction } = setup()
  mgr.openProfileManager()
  expect(mgr.showProfileManager.value).toBe(true)
  expect(requestAction).toHaveBeenCalledWith('workflow_profile_list', {})
  mgr.closeProfileManager()
  expect(mgr.showProfileManager.value).toBe(false)
})

it('the dropdown loads a picked profile but ignores the empty placeholder', () => {
  const { mgr, requestAction } = setup()
  mgr.loadWorkflowProfile('')
  expect(requestAction).not.toHaveBeenCalled()
  mgr.loadWorkflowProfile('Flux')
  expect(requestAction).toHaveBeenCalledWith('workflow_profile_load', { name: 'Flux' })
})

it('saving trims the name and asks before overwriting an existing profile', () => {
  const { mgr, requestAction } = setup()
  stubDialogs('  New  ')
  mgr.saveCurrentAsProfile()
  expect(window.confirm).not.toHaveBeenCalled()
  expect(requestAction).toHaveBeenLastCalledWith('workflow_profile_save', { name: 'New', overwrite: false })
  stubDialogs('ANIMA', false)
  mgr.saveCurrentAsProfile()
  expect(requestAction).toHaveBeenCalledTimes(1)
  stubDialogs('ANIMA', true)
  mgr.saveCurrentAsProfile()
  expect(window.confirm).toHaveBeenCalledWith("'ANIMA' 이미 존재합니다. 덮어쓸까요?")
  expect(requestAction).toHaveBeenLastCalledWith('workflow_profile_save', { name: 'ANIMA', overwrite: true })
  stubDialogs('   ')
  mgr.saveCurrentAsProfile()
  stubDialogs(null)
  mgr.saveCurrentAsProfile()
  expect(requestAction).toHaveBeenCalledTimes(2)
})

// 저장 파일 이름은 규칙(core/file_naming.sanitize_filename)을 거친다 — 'Flux.' · 'flux' · 'Flux?' 는
// 모두 Flux.json 이다. 예전엔 목록과 글자 그대로만 비교해 확인 없이 기존 'Flux' 를 덮었다.
it.each(['Flux.', 'flux', 'Flux?', 'Fl/ux', 'FLUX..', ' Flux '])('%j is the same file as Flux — asks first, names the profile it replaces', (typed) => {
  const { mgr, requestAction } = setup()
  stubDialogs(typed, false)
  mgr.saveCurrentAsProfile()
  expect(window.confirm).toHaveBeenCalledTimes(1)
  expect(requestAction).not.toHaveBeenCalled()

  stubDialogs(typed, true)
  mgr.saveCurrentAsProfile()
  const trimmed = typed.trim()
  const asked = vi.mocked(window.confirm).mock.calls[0][0]
  if (trimmed !== 'Flux') expect(asked).toContain("기존 프로파일 'Flux'")
  expect(requestAction).toHaveBeenCalledWith('workflow_profile_save', { name: trimmed, overwrite: true })
})

it('a name that only shares a prefix is a new file — no confirm, no overwrite', () => {
  const { mgr, requestAction } = setup()
  stubDialogs('Flux 2')
  mgr.saveCurrentAsProfile()
  expect(window.confirm).not.toHaveBeenCalled()
  expect(requestAction).toHaveBeenCalledWith('workflow_profile_save', { name: 'Flux 2', overwrite: false })
})

it('delete confirms and drops the row; rename skips unchanged names', () => {
  const { mgr, requestAction } = setup()
  stubDialogs(null, true)
  mgr.deleteWorkflowProfile('Flux')
  expect(requestAction).toHaveBeenCalledWith('workflow_profile_delete', { name: 'Flux' })
  expect(mgr.profileNames.value).toEqual(['ANIMA'])
  stubDialogs('ANIMA')
  mgr.renameWorkflowProfile('ANIMA')
  expect(requestAction).toHaveBeenCalledTimes(1)
  stubDialogs(' ANIMA v2 ')
  mgr.renameWorkflowProfile('ANIMA')
  expect(requestAction).toHaveBeenLastCalledWith('workflow_profile_rename', { old: 'ANIMA', new: 'ANIMA v2' })
})
