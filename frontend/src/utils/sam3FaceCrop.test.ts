import { describe, expect, it } from 'vitest'
import {
  FACE_CROP_DEFAULT_CONFIDENCE, FACE_CROP_DEFAULT_MODEL, SAM3_FACE_CROP_FIELDS,
  faceCropModelOptions, faceCropSupport, sam3FaceCropDefaults, sam3FaceCropId, sam3FaceCropSettings,
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
})
