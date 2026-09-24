/**
 * ADetailer 슬롯 위젯 키 — Python `ui/generator_ui_setup.py` 의 `_ad_slot(prefix)` 가 만드는 프록시
 * widget_id(`_ad_s1_model` …)와 같은 규칙. 슬롯 1·2 는 접두만 다르다.
 * 필드 목록이 Python 과 같은지는 tests/test_ad_slot_fields_contract.py 가 정적으로 본다.
 */
export const AD_SLOT_FIELDS = [
  'model', 'prompt', 'neg',
  'confidence', 'denoise',
  'mask_blur', 'padding', 'dilate_erode',
  'mask_merge',
  'use_inp_size', 'inp_w', 'inp_h',
  'use_steps', 'steps',
  'use_cfg', 'cfg',
  'use_sampler', 'sampler', 'scheduler',
  'use_ckpt', 'ckpt',
  'use_vae', 'vae',
] as const

export type AdSlotField = typeof AD_SLOT_FIELDS[number]
export type AdSlotNo = 1 | 2

/** `_ad_s{n}_{field}` */
export function adSlotKey(slot: AdSlotNo, field: AdSlotField): string {
  return `_ad_s${slot}_${field}`
}

/** 슬롯 활성 체크박스(CheckBoxProxy) — `ad_slot{n}_group` */
export function adSlotGroupKey(slot: AdSlotNo): string {
  return `ad_slot${slot}_group`
}
