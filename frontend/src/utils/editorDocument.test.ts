import { describe, expect, it } from 'vitest'
import {
  ImageSizeCache, aliasesFromSaveResult, editorResultForDoc, editorResultKind, hasPendingSaveFor, initialDocGen,
  isEditorDirty, normalizeDrawOpacity, parseSaveResult, referencesAutosaveFile, replacePath,
  saveResultAction, saveToastMessage, writtenPathFromSaveResult,
  type EditorDocState, type PendingSave, type SavedMarker,
} from './editorDocument'

describe('writtenPathFromSaveResult', () => {
  it('names the file a save actually wrote (so gallery/favorites cards drop their cached picture)', () => {
    expect(writtenPathFromSaveResult({ ok: true, path: 'C:/out/a_edited.png' })).toBe('C:/out/a_edited.png')
    expect(writtenPathFromSaveResult({ ok: true, mode: 'save_as', path: 'C:/out/a.png', request_id: 99 }))
      .toBe('C:/out/a.png')   // 다른 창의 요청이어도 디스크의 파일은 바뀌었다
  })

  it('ignores failures, cancels and unchanged saves', () => {
    expect(writtenPathFromSaveResult({ ok: false, path: 'C:/out/a.png', error: 'x' })).toBe('')
    expect(writtenPathFromSaveResult({ ok: true, cancelled: true, path: 'C:/out/a.png' })).toBe('')
    expect(writtenPathFromSaveResult({ ok: true, unchanged: true, path: 'C:/out/a.png' })).toBe('')
    expect(writtenPathFromSaveResult({ ok: true })).toBe('')
    expect(writtenPathFromSaveResult(null)).toBe('')
  })
})

const saved = (imagePath: string, drawRevision = 0, drawHadContent = false, drawOpacity = 100): SavedMarker =>
  ({ imagePath, drawRevision, drawHadContent, drawOpacity })
const cur = (imagePath: string, drawRevision = 0, drawHasContent = false, drawOpacity = 100): EditorDocState =>
  ({ imagePath, drawRevision, drawHasContent, drawOpacity })

describe('isEditorDirty', () => {
  it('no image is never dirty', () => {
    expect(isEditorDirty(cur('', 5, true), null)).toBe(false)
  })

  it('an image with no saved marker is dirty', () => {
    expect(isEditorDirty(cur('a.png'), null)).toBe(true)
  })

  it('undoing back to the saved path clears the dirty mark', () => {
    const marker = saved('orig.png', 3)
    expect(isEditorDirty(cur('edited_1.png', 3), marker)).toBe(true)
    expect(isEditorDirty(cur('orig.png', 3), marker)).toBe(false)
  })

  it('clearing an already empty layer (new image load) is not a change', () => {
    expect(isEditorDirty(cur('a.png', 4), saved('a.png', 3))).toBe(false)
  })

  it('unmerged strokes after the save are unsaved changes', () => {
    expect(isEditorDirty(cur('a.png', 4, true), saved('a.png', 3))).toBe(true)
  })

  it('strokes that were saved stay clean until the layer changes', () => {
    const marker = saved('a.png', 7, true)
    expect(isEditorDirty(cur('a.png', 7, true), marker)).toBe(false)
    // 저장한 그림을 지웠다 — 디스크(그림 있음)와 달라졌다
    expect(isEditorDirty(cur('a.png', 8, false), marker)).toBe(true)
  })

  it('changing only the opacity of a saved layer is an unsaved change', () => {
    // 레이어를 100% 로 합성해 저장한 뒤 슬라이더만 50 으로 — 저장본(100%)과 화면(50%)이 다르다
    const marker = saved('a.png', 7, true, 100)
    expect(isEditorDirty(cur('a.png', 7, true, 50), marker)).toBe(true)
    // 저장 때 값으로 되돌리면 다시 깨끗하다
    expect(isEditorDirty(cur('a.png', 7, true, 100), marker)).toBe(false)
  })

  it('the opacity of an empty layer does not matter', () => {
    expect(isEditorDirty(cur('a.png', 3, false, 20), saved('a.png', 3, false, 100))).toBe(false)
    // 새로 연 이미지(마커는 빈 레이어) — 예전 문서에서 바꾼 슬라이더 값이 변경으로 잡히면 안 된다
    expect(isEditorDirty(cur('a.png', 4, false, 35), saved('a.png', 3, false, 100))).toBe(false)
  })

  it('a marker built from a pending save keeps the opacity that was sent', () => {
    const pending: PendingSave = {
      id: 1, docGen: 1, imagePath: 'a.png', drawRevision: 5, drawHadContent: true, drawOpacity: 100, startedAt: 0,
    }
    // onEditorSaveResult 가 하는 것처럼 요청 시점의 상태를 마커로 옮긴다
    const marker = saved(pending.imagePath, pending.drawRevision, pending.drawHadContent, pending.drawOpacity)
    // 응답을 기다리는 사이 슬라이더를 50 으로 바꿨다 — 여전히 미저장이다
    expect(isEditorDirty(cur('a.png', 5, true, 50), marker)).toBe(true)
  })

  it('opacity comparison clamps and rounds like the backend', () => {
    expect(normalizeDrawOpacity(49.6)).toBe(50)
    expect(normalizeDrawOpacity(150)).toBe(100)
    expect(normalizeDrawOpacity(-3)).toBe(0)
    expect(normalizeDrawOpacity('70')).toBe(70)
    expect(normalizeDrawOpacity(undefined)).toBe(100)
    expect(isEditorDirty(cur('a.png', 7, true, 100.2), saved('a.png', 7, true, 100))).toBe(false)
  })
})

describe('editorResultForDoc', () => {
  // 문서 A(세대 11)에서 느린 작업을 시작하고 B 를 열었다(세대 12) — B 는 아직 작업을 안 했다
  it('accepts results for the open document', () => {
    expect(editorResultForDoc({ doc_gen: 11, path: 'C:/cache/edited_a.png', job_id: 4 }, 11)).toBe(true)
  })

  it('rejects an image result of a document that was replaced by loadImage', () => {
    expect(editorResultForDoc({ doc_gen: 11, path: 'C:/cache/edited_a.png', job_id: 4 }, 12)).toBe(false)
  })

  it('rejects a result that arrives after the editor was closed', () => {
    // resetEditor 도 세대를 올린다 — 늦은 결과가 닫은 에디터를 다시 열면 안 된다
    expect(editorResultForDoc({ doc_gen: 11, path: 'C:/cache/edited_a.png' }, 12)).toBe(false)
  })

  it('rejects mask, error and preview results of another document the same way', () => {
    expect(editorResultForDoc({ doc_gen: 11, mask_base64: 'iVBOR', operation: 'auto_detect' }, 12)).toBe(false)
    expect(editorResultForDoc({ doc_gen: 11, error: 'rembg 실패', operation: 'remove_bg' }, 12)).toBe(false)
    expect(editorResultForDoc({ doc_gen: 11, preview: true, preview_request: true, preview_token: 3 }, 12)).toBe(false)
  })

  it('accepts results without a document token (older backend) and drops malformed ones', () => {
    expect(editorResultForDoc({ path: 'x.png' } as any, 12)).toBe(true)
    expect(editorResultForDoc({ doc_gen: '11', path: 'x.png' } as any, 12)).toBe(true)
    expect(editorResultForDoc(null, 12)).toBe(false)
  })
})

describe('editorResultKind', () => {
  it('treats an auto_detect result as a mask even though it also carries the source path', () => {
    // ui/vue_bridge.py 의 auto_detect 결과 모양 — path 를 먼저 보면 원본을 다시 불러 마스크를 버렸다
    const detect = { mask_base64: 'data:image/png;base64,iVBOR', detect_count: 2, path: 'C:/img/a.png',
      operation: 'auto_detect', job_id: 7 }
    expect(editorResultKind(detect)).toBe('mask')
  })

  it('routes file results, errors and empty payloads', () => {
    expect(editorResultKind({ path: 'C:/cache/edited_a.png', operation: 'auto_censor' })).toBe('image')
    expect(editorResultKind({ error: 'rembg 실패', operation: 'remove_bg' })).toBe('error')
    expect(editorResultKind({ mask_base64: '', path: '', error: '' })).toBe('none')
    expect(editorResultKind(null)).toBe('none')
  })
})

describe('initialDocGen', () => {
  it('starts each window at a different, JSON-safe generation', () => {
    const a = initialDocGen(1_790_000_000_000, 0.1)
    const b = initialDocGen(1_790_000_000_000, 0.7)
    const c = initialDocGen(1_790_000_000_001, 0.1)
    expect(new Set([a, b, c]).size).toBe(3)
    for (const g of [a, b, c]) {
      expect(Number.isSafeInteger(g)).toBe(true)
      expect(Number.isSafeInteger(g + 1_000_000)).toBe(true)
      expect(JSON.parse(JSON.stringify({ g })).g).toBe(g)
    }
  })
})

describe('replacePath', () => {
  it('replaces every occurrence and returns a new array', () => {
    const list = ['a', 'b', 'a']
    const out = replacePath(list, 'a', 'z')
    expect(out).toEqual(['z', 'b', 'z'])
    expect(list).toEqual(['a', 'b', 'a'])
  })
})

describe('save result helpers', () => {
  it('parses only well-formed results', () => {
    expect(parseSaveResult('{"ok":true,"path":"C:/x.png","request_id":3}')?.path).toBe('C:/x.png')
    expect(parseSaveResult('{"path":"C:/x.png"}')).toBeNull()
    expect(parseSaveResult('not json')).toBeNull()
  })

  it('maps replaced history paths to the snapshot', () => {
    expect(aliasesFromSaveResult({
      ok: true, snapshot_path: 'C:/cache/saved_orig_1.png', replaced_paths: ['C:/gen/a.png', 'c:\\gen\\a.png'],
    })).toEqual([
      ['C:/gen/a.png', 'C:/cache/saved_orig_1.png'],
      ['c:\\gen\\a.png', 'C:/cache/saved_orig_1.png'],
    ])
    expect(aliasesFromSaveResult({ ok: false, snapshot_path: 's', replaced_paths: ['a'] })).toEqual([])
    expect(aliasesFromSaveResult({ ok: true, replaced_paths: ['a'] })).toEqual([])
  })
})

describe('saveResultAction', () => {
  const pending = (over: Partial<PendingSave> = {}): PendingSave => ({
    id: 7, docGen: 3, imagePath: 'C:/cache/edited_1.png', drawRevision: 0, drawHadContent: false, drawOpacity: 100,
    startedAt: 1000, ...over,
  })

  it('applies a result for the document that is still open', () => {
    expect(saveResultAction(pending(), { ok: true, request_id: 7, path: 'C:/gen/a_edited.png' }, 3)).toBe('apply')
  })

  it('only notifies when the document changed while saving', () => {
    // 저장 중 Ctrl+V 로 새 문서를 열었다(docGen 3 → 4) — sourcePath 등을 새 문서에 덮으면 안 된다
    expect(saveResultAction(pending(), { ok: true, request_id: 7, path: 'C:/gen/a_edited.png' }, 4)).toBe('stale')
  })

  it('still reports failures and cancellations of an older document', () => {
    expect(saveResultAction(pending(), { ok: false, request_id: 7, error: 'disk full' }, 4)).toBe('failed')
    expect(saveResultAction(pending(), { ok: false, request_id: 7, cancelled: true }, 4)).toBe('cancelled')
  })

  it('ignores results of requests this window did not send', () => {
    expect(saveResultAction(undefined, { ok: true, request_id: 7 }, 3)).toBe('ignore')
    expect(saveResultAction(pending(), { ok: true, request_id: 8 }, 3)).toBe('ignore')
  })
})

describe('hasPendingSaveFor', () => {
  const p = (docGen: number, startedAt: number): PendingSave =>
    ({ id: docGen, docGen, imagePath: 'x', drawRevision: 0, drawHadContent: false, drawOpacity: 100, startedAt })

  it('blocks a second save of the same document only while it is fresh', () => {
    expect(hasPendingSaveFor([p(2, 1000)], 2, 1500, 120_000)).toBe(true)
    expect(hasPendingSaveFor([p(2, 1000)], 2, 1000 + 120_000, 120_000)).toBe(false)
  })

  it('lets a newly opened document save while the old one is still writing', () => {
    expect(hasPendingSaveFor([p(2, 1000)], 3, 1500, 120_000)).toBe(false)
  })
})

describe('referencesAutosaveFile', () => {
  it('spots the crash-recovery file in any spelling', () => {
    expect(referencesAutosaveFile(['C:\\Users\\me\\AppData\\Local\\Temp\\AIStudioPro_editor\\_autosave_session.png'])).toBe(true)
    expect(referencesAutosaveFile(['file:///C:/Temp/aistudiopro_editor/_AUTOSAVE_SESSION.PNG'])).toBe(true)
  })

  it('does not flag the recovered working copy or ordinary files', () => {
    expect(referencesAutosaveFile([
      'C:/app/image_cache/editor_temp/recovered_ab12.png',
      'C:/gen/_autosave_session.png',
      '',
    ])).toBe(false)
  })
})

describe('saveToastMessage', () => {
  it('names the written file and explains format changes', () => {
    expect(saveToastMessage({ ok: true, path: 'C:\\gen\\a_edited.png' })).toBe('저장됨: a_edited.png')
    expect(saveToastMessage({ ok: true, path: 'C:/gen/a.png', unchanged: true })).toBe('변경 사항 없음: a.png')
    expect(saveToastMessage({ ok: true, path: 'C:/gen/p_edited.png', alpha_png: true }))
      .toContain('PNG')
  })
})

describe('ImageSizeCache', () => {
  it('stores, renames and ignores invalid sizes', () => {
    const cache = new ImageSizeCache()
    cache.set('a', 832, 1216)
    cache.set('bad', 0, 10)
    expect(cache.get('a')).toEqual({ w: 832, h: 1216 })
    expect(cache.get('bad')).toBeUndefined()
    cache.rename('a', 'snap')
    expect(cache.get('snap')).toEqual({ w: 832, h: 1216 })
  })

  it('drops the oldest entry past the limit', () => {
    const cache = new ImageSizeCache(2)
    cache.set('a', 1, 1)
    cache.set('b', 2, 2)
    cache.set('c', 3, 3)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.size).toBe(2)
    cache.clear()
    expect(cache.size).toBe(0)
  })
})
