import { expect, it } from 'vitest'
import { AD_SLOT_FIELDS, adSlotGroupKey, adSlotKey } from './adSlotKeys'

// 분할 전 App.vue 가 슬롯 1 에 적어 두었던 위젯 키 23개(순서 그대로) — 슬롯 컴포넌트로 합치며 하나도 잃지 않았는지
const ORIGINAL_SLOT1_KEYS = [
  '_ad_s1_model', '_ad_s1_prompt', '_ad_s1_neg', '_ad_s1_confidence', '_ad_s1_denoise',
  '_ad_s1_mask_blur', '_ad_s1_padding', '_ad_s1_dilate_erode', '_ad_s1_mask_merge',
  '_ad_s1_use_inp_size', '_ad_s1_inp_w', '_ad_s1_inp_h', '_ad_s1_use_steps', '_ad_s1_steps',
  '_ad_s1_use_cfg', '_ad_s1_cfg', '_ad_s1_use_sampler', '_ad_s1_sampler', '_ad_s1_scheduler',
  '_ad_s1_use_ckpt', '_ad_s1_ckpt', '_ad_s1_use_vae', '_ad_s1_vae',
]

it('slot 1 keys are exactly the ones the old template bound', () => {
  expect(AD_SLOT_FIELDS.map(f => adSlotKey(1, f))).toEqual(ORIGINAL_SLOT1_KEYS)
})

it('slot 2 differs only by the prefix', () => {
  expect(AD_SLOT_FIELDS.map(f => adSlotKey(2, f))).toEqual(ORIGINAL_SLOT1_KEYS.map(k => k.replace('_ad_s1_', '_ad_s2_')))
})

it('the enable checkbox is ad_slot{n}_group', () => {
  expect(adSlotGroupKey(1)).toBe('ad_slot1_group')
  expect(adSlotGroupKey(2)).toBe('ad_slot2_group')
})
