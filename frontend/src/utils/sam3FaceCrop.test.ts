import { describe, expect, it } from 'vitest'
import {
  COMFY_MIN_PACK_VERSION, COMFY_SUPPORT_PROPERTY, FACE_CROP_DEFAULT_CONFIDENCE, FACE_CROP_DEFAULT_MODEL,
  FACE_CROP_GROWTH_VERSION, SAM3_FACE_CROP_FIELDS, faceCropBlocked, faceCropBlockedReason, faceCropHairNote,
  faceCropModelOptions, faceCropPackNote, faceCropRecheckable, faceCropSupport, sam3FaceCropDefaults, sam3FaceCropId,
  sam3FaceCropSettings,
} from './sam3FaceCrop'

describe('sam3FaceCrop table', () => {
  it('has the three extension fields with the extension defaults (off · face_yolov8n.pt · 0.3)', () => {
    expect(SAM3_FACE_CROP_FIELDS.map((field) => field.key)).toEqual(['face_crop', 'face_crop_model', 'face_crop_confidence'])
    expect(sam3FaceCropDefaults()).toEqual({
      _sam3_face_crop: 'false',
      _sam3_face_crop_model: 'face_yolov8n.pt',
      _sam3_face_crop_confidence: '0.3',
    })
    expect(sam3FaceCropId('face_crop')).toBe('_sam3_face_crop')
  })
})

describe('sam3FaceCropSettings', () => {
  it('defaults are off with the extension model and confidence', () => {
    expect(sam3FaceCropSettings(sam3FaceCropDefaults())).toEqual({
      sam3_face_crop: false, sam3_face_crop_model: FACE_CROP_DEFAULT_MODEL, sam3_face_crop_confidence: FACE_CROP_DEFAULT_CONFIDENCE,
    })
  })

  it('reads store strings and booleans, trims the model and clamps the confidence like the extension', () => {
    expect(sam3FaceCropSettings({ _sam3_face_crop: 'true', _sam3_face_crop_model: ' face_yolov8s.pt ', _sam3_face_crop_confidence: '0.45' }))
      .toEqual({ sam3_face_crop: true, sam3_face_crop_model: 'face_yolov8s.pt', sam3_face_crop_confidence: 0.45 })
    expect(sam3FaceCropSettings({ _sam3_face_crop: true, _sam3_face_crop_model: '', _sam3_face_crop_confidence: '7' }))
      .toEqual({ sam3_face_crop: true, sam3_face_crop_model: FACE_CROP_DEFAULT_MODEL, sam3_face_crop_confidence: 1 })
    expect(sam3FaceCropSettings({ _sam3_face_crop_confidence: '-1' }).sam3_face_crop_confidence).toBe(0)
    expect(sam3FaceCropSettings({ _sam3_face_crop_confidence: 'abc' }).sam3_face_crop_confidence).toBe(FACE_CROP_DEFAULT_CONFIDENCE)
    expect(sam3FaceCropSettings({ _sam3_face_crop_confidence: '' }).sam3_face_crop_confidence).toBe(FACE_CROP_DEFAULT_CONFIDENCE)
    expect(sam3FaceCropSettings({ _sam3_face_crop: 'yes' }).sam3_face_crop).toBe(false)
  })
})

describe('faceCropModelOptions', () => {
  it('uses the Python items, falls back to the default model and keeps a typed value', () => {
    expect(faceCropModelOptions(['face_yolov8n.pt', 'face_yolov8s.pt'], 'face_yolov8s.pt'))
      .toEqual(['face_yolov8n.pt', 'face_yolov8s.pt'])
    expect(faceCropModelOptions(undefined, '')).toEqual([FACE_CROP_DEFAULT_MODEL])
    expect(faceCropModelOptions('', 'face_yolov8n.pt')).toEqual([FACE_CROP_DEFAULT_MODEL])
    expect(faceCropModelOptions(['face_yolov8n.pt', ' ', 'face_yolov8n.pt'], 'C:/models/my_face.pt'))
      .toEqual(['face_yolov8n.pt', 'C:/models/my_face.pt'])
  })
})

describe('faceCropSupport', () => {
  const forge = (missing: string[], sam3 = true) => ({
    status: 'ok', known: true, features: { sam3 }, sam3_keys: { missing_in_extension: missing },
  })
  it('tells an old sam-extra (no sam3_face_crop key) from a 0.34.0 one', () => {
    expect(faceCropSupport(forge([]))).toBe('supported')
    expect(faceCropSupport(forge(['sam3_face_crop', 'sam3_face_crop_confidence', 'sam3_face_crop_model']))).toBe('unsupported')
  })
  it('is unknown without a snapshot, before the check or without the SAM3 script, and comfy on ComfyUI', () => {
    expect(faceCropSupport(null)).toBe('unknown')
    expect(faceCropSupport({ status: 'unknown', known: false })).toBe('unknown')
    expect(faceCropSupport(forge(['sam3_face_crop'], false))).toBe('unknown')
    expect(faceCropSupport({ status: 'not_applicable', known: false })).toBe('comfy')
  })
  it('reads the ComfyUI node info (pack 1.8.0, ultralytics) only on ComfyUI', () => {
    const comfy = { status: 'not_applicable', known: false }
    expect(faceCropSupport(comfy, 'ready')).toBe('comfy_ready')
    expect(faceCropSupport(comfy, 'no_detector')).toBe('comfy_no_detector')
    expect(faceCropSupport(comfy, 'old_pack')).toBe('comfy_old_pack')
    expect(faceCropSupport(comfy, null)).toBe('comfy')
    expect(faceCropSupport(comfy, 'weird')).toBe('comfy')
    expect(faceCropSupport(forge([]), 'no_detector')).toBe('supported')
    expect(COMFY_SUPPORT_PROPERTY).toBe('comfySupport')
    expect(COMFY_MIN_PACK_VERSION).toBe('1.8.0')
  })
  it('blocks switching on only for a ComfyUI without the face detector, with the reason', () => {
    for (const support of ['unknown', 'supported', 'unsupported', 'comfy', 'comfy_ready', 'comfy_old_pack'] as const) {
      expect(faceCropBlocked(support)).toBe(false)
      expect(faceCropBlockedReason(support)).toBe('')
    }
    expect(faceCropBlocked('comfy_no_detector')).toBe(true)
    expect(faceCropBlockedReason('comfy_no_detector')).toContain('ultralytics')
    expect(faceCropBlockedReason('comfy_no_detector')).toContain('↻')
    expect(faceCropBlockedReason('comfy_no_detector')).not.toContain('멈춥니다')
    expect(faceCropBlockedReason('comfy_no_detector', true)).toContain('켜 둔 채 생성하면 이 이유로 멈춥니다')
  })
  it('only warns about an old pack (the next generation can install it) and offers a re-check there', () => {
    expect(faceCropPackNote('comfy_old_pack')).toContain('1.8.0')
    expect(faceCropPackNote('comfy_old_pack')).toContain('다음 생성 때 번들 노드 팩을 설치')
    for (const support of ['unknown', 'supported', 'unsupported', 'comfy', 'comfy_ready', 'comfy_no_detector'] as const) {
      expect(faceCropPackNote(support)).toBe('')
    }
    expect(['comfy_no_detector', 'comfy_old_pack'].map((s) => faceCropRecheckable(s as never))).toEqual([true, true])
    expect(['comfy', 'comfy_ready', 'supported', 'unknown'].map((s) => faceCropRecheckable(s as never)))
      .toEqual([false, false, false, false])
  })
})

describe('faceCropHairNote', () => {
  const forge = (version: unknown) => ({ version: { extension_version: version } })
  it('says the ComfyUI pack regrows cut hair (sam-extra 0.34.1) and never claims parity with an older Forge', () => {
    for (const support of ['comfy', 'comfy_ready', 'comfy_old_pack', 'comfy_no_detector'] as const) {
      const note = faceCropHairNote(null, support)
      expect(note).toContain('crop 을 키워 다시 찾습니다')
      expect(note).toContain(FACE_CROP_GROWTH_VERSION)
      expect(note).not.toContain('잘립니다')
    }
  })
  it('follows the connected sam-extra on Forge: 0.34.1 and later regrow, 0.34.0 (or unknown) cuts', () => {
    expect(faceCropHairNote(forge('0.34.0'), 'supported')).toContain('잘립니다')
    expect(faceCropHairNote(forge(null), 'unknown')).toContain('잘립니다')
    expect(faceCropHairNote({ version: {} }, 'supported')).toContain('잘립니다')
    for (const version of ['0.34.1', '0.35.0', 'v1.0', '0.34.10']) {
      const note = faceCropHairNote(forge(version), 'supported')
      expect(note).toContain('crop 을 키워 다시 찾습니다')
      expect(note).not.toContain('잘립니다')
    }
    expect(faceCropHairNote(forge('0.33.9'), 'unsupported')).toContain('잘립니다')
  })
})
