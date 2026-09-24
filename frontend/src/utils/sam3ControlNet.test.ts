import { describe, expect, it } from 'vitest'
import {
  SAM3_CN_FIELDS, choiceOptions, isTrue, sam3CnDefaults, sam3CnId, sam3CnSettings,
} from './sam3ControlNet'

describe('SAM3 ControlNet 필드 표', () => {
  it('13필드이고 키가 겹치지 않는다', () => {
    const keys = SAM3_CN_FIELDS.map((field) => field.key)
    expect(keys).toHaveLength(13)
    expect(new Set(keys).size).toBe(13)
    for (const key of keys) expect(key).toMatch(/^cn_[a-z_]+$/)
  })

  it('widget id 는 T2I 프록시(_sam3_cn_*) 와 같은 규칙', () => {
    expect(sam3CnId('cn_weight')).toBe('_sam3_cn_weight')
    expect(Object.keys(sam3CnDefaults())).toContain('_sam3_cn_threshold_b')
  })
})

describe('sam3CnSettings', () => {
  it('기본값 → 확장 기본값 (Sam3Args)', () => {
    expect(sam3CnSettings(sam3CnDefaults())).toEqual({
      sam3_cn_enable: false,
      sam3_cn_override_external: false,
      sam3_cn_model: 'None',
      sam3_cn_module: 'inpaint_only',
      sam3_cn_weight: 1,
      sam3_cn_guidance_start: 0,
      sam3_cn_guidance_end: 1,
      sam3_cn_pixel_perfect: true,
      sam3_cn_control_mode: 'Balanced',
      sam3_cn_resize_mode: 'Crop and Resize',
      sam3_cn_processor_res: 512,
      sam3_cn_threshold_a: -1,
      sam3_cn_threshold_b: -1,
    })
  })

  it('스토어 문자열을 타입에 맞게 바꾸고 잘못된 숫자는 기본값', () => {
    const values = {
      ...sam3CnDefaults(),
      _sam3_cn_enable: 'true',
      _sam3_cn_weight: '0.65',
      _sam3_cn_processor_res: 'abc',
      _sam3_cn_model: '  ',
      _sam3_cn_threshold_a: '',
    }
    const out = sam3CnSettings(values)
    expect(out.sam3_cn_enable).toBe(true)
    expect(out.sam3_cn_weight).toBe(0.65)
    expect(out.sam3_cn_processor_res).toBe(512)
    expect(out.sam3_cn_model).toBe('None')
    expect(out.sam3_cn_threshold_a).toBe(-1)
  })

  it('isTrue 는 불리언과 문자열 둘 다', () => {
    expect(isTrue(true)).toBe(true)
    expect(isTrue('TRUE')).toBe(true)
    expect(isTrue('false')).toBe(false)
    expect(isTrue(undefined)).toBe(false)
  })
})

describe('choiceOptions', () => {
  it('Python items 가 있으면 그 목록, 없으면 현재 값 하나만 (목록을 복제하지 않는다)', () => {
    expect(choiceOptions(['a', 'b'], 'b', 'a')).toEqual(['a', 'b'])
    expect(choiceOptions([], 'depth_zoe', 'inpaint_only')).toEqual(['depth_zoe'])
    expect(choiceOptions(undefined, '', 'inpaint_only')).toEqual(['inpaint_only'])
  })
})
