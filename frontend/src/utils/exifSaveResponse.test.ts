import { describe, expect, it } from 'vitest'
import { resolveExifSaveResponse, type ExifSaveRequest, type ExifViewLike } from './exifSaveResponse'

const view = (over: Partial<ExifViewLike> = {}): ExifViewLike => ({
  path: 'C:/gen/a.png', filename: 'a.png', mediaType: 'PNG', source: 'webui',
  prompt: 'first edit', negative: 'lowres', params_line: 'Steps: 20', ...over,
})
const sent = (over: Partial<ExifSaveRequest> = {}): ExifSaveRequest =>
  ({ viewGen: 4, path: 'C:/gen/a.png', prompt: 'first edit', negative: 'lowres', ...over })
// 백엔드가 저장한 파일을 다시 읽은 결과(read_metadata_for_ui)
const info = { prompt: 'first edit', negative: 'lowres', params_line: 'Steps: 20, Seed: 42', raw: 'first edit\n...' }

describe('resolveExifSaveResponse', () => {
  it('merges the saved metadata and clears dirty when nothing changed since the request', () => {
    const out = resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current: view(), info })
    expect(out.dirty).toBe(false)
    expect(out.editedSince).toBe(false)
    expect(out.view).toMatchObject({ prompt: 'first edit', params_line: 'Steps: 20, Seed: 42', path: 'C:/gen/a.png' })
  })

  it('keeps a prompt edited after the request and stays dirty', () => {
    // 저장을 보낸 뒤 응답 전에 프롬프트를 또 고치고 포커스를 옮겼다
    const current = view({ prompt: 'second edit' })
    const out = resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current, info })
    expect(out.editedSince).toBe(true)
    expect(out.dirty).toBe(true)
    expect(out.view?.prompt).toBe('second edit')
    // 다른 필드는 저장된 파일 기준으로 갱신된다
    expect(out.view?.params_line).toBe('Steps: 20, Seed: 42')
  })

  it('keeps a negative edited after the request and stays dirty', () => {
    const current = view({ negative: 'lowres, blurry' })
    const out = resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current, info })
    expect(out.dirty).toBe(true)
    expect(out.view?.negative).toBe('lowres, blurry')
    expect(out.view?.prompt).toBe('first edit')
  })

  it('leaves another image untouched — its dirty flag is not cleared', () => {
    // A 를 저장하고 B 를 열어 고친 뒤 A 의 응답이 왔다
    const b = view({ path: 'C:/gen/b.png', filename: 'b.png', prompt: 'B edit' })
    const out = resolveExifSaveResponse({ sent: sent(), currentViewGen: 5, current: b, info })
    expect(out).toEqual({ view: null, dirty: null, editedSince: false })
  })

  it('leaves a reopened view of the same image untouched', () => {
    const out = resolveExifSaveResponse({ sent: sent(), currentViewGen: 6, current: view(), info })
    expect(out.view).toBeNull()
    expect(out.dirty).toBeNull()
  })

  it('does nothing when the large view was closed', () => {
    expect(resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current: null, info }))
      .toEqual({ view: null, dirty: null, editedSince: false })
  })

  it('treats a rename during the save as the same image and keeps the new path', () => {
    const renamed = view({ path: 'C:\\gen\\renamed.png', filename: 'renamed.png' })
    const out = resolveExifSaveResponse({
      sent: sent(), currentViewGen: 4, current: renamed, info: { ...info, path: 'C:/gen/a.png', filename: 'a.png' },
    })
    expect(out.dirty).toBe(false)
    expect(out.view).toMatchObject({ path: 'C:\\gen\\renamed.png', filename: 'renamed.png' })
  })

  it('without info only resolves the dirty flag', () => {
    expect(resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current: view(), info: undefined }))
      .toEqual({ view: null, dirty: false, editedSince: false })
    expect(resolveExifSaveResponse({ sent: sent(), currentViewGen: 4, current: view({ prompt: 'x' }), info: ['bad'] }))
      .toEqual({ view: null, dirty: true, editedSince: true })
  })

  it('treats a missing prompt as empty like the request does', () => {
    const current = view({ prompt: undefined })
    const out = resolveExifSaveResponse({ sent: sent({ prompt: '' }), currentViewGen: 4, current, info })
    expect(out.dirty).toBe(false)
  })
})
