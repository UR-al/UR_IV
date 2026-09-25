import { describe, expect, it } from 'vitest'
import {
  CN_NONE, SAM3_CN_FIELDS, canonicalCnName, choiceOptions, cnModelOptions, cnModuleOptions,
  cnModuleOptionsForModel, cnNameMissing, isTrue, sam3CnDefaults, sam3CnId, sam3CnLiveLists,
  sam3CnLlliteChannels, sam3CnLlliteModule, sam3CnLlliteTileRepair, sam3CnSettings,
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

// 녹화된 Forge classic(sam-extra 0.30.0) /controlnet/module_list · model_list 의 일부
// (전체는 tests/fixtures/sam_extra_live_cn_*.json).
const LIVE_MODULES = ['None', 'canny', 'inpaint_noobai', 'inpaint_only', 'inpaint_only+lama', 'tile_resample']
const LIVE_MODELS = ['None', 'anima-lllite-inpainting-v2', 'animaTileRepair_v10', 'animaTileRepair_v20']
const STATIC_ITEMS = ['inpaint_only', 'inpaint_only+lama', 'canny', 'None']

describe('sam3CnLiveLists (기능 스냅샷 → 라이브 CN 목록)', () => {
  it('known 스냅샷이면 /controlnet/module_list · model_list', () => {
    const caps = { known: true, choices: { controlnet_modules: LIVE_MODULES, controlnet_models: LIVE_MODELS } }
    expect(sam3CnLiveLists(caps)).toEqual({ modules: LIVE_MODULES, models: LIVE_MODELS })
  })

  it('모르면(확인 전·ComfyUI·실패) 둘 다 null — 정적 폴백을 쓰게', () => {
    expect(sam3CnLiveLists(null)).toEqual({ modules: null, models: null })
    expect(sam3CnLiveLists({ known: false, choices: { controlnet_modules: LIVE_MODULES } }))
      .toEqual({ modules: null, models: null })
  })

  it('CN 확장이 없어 목록이 빠졌거나 비었으면 그 목록만 null', () => {
    expect(sam3CnLiveLists({ known: true, choices: { controlnet_models: [] } }))
      .toEqual({ modules: null, models: null })
    expect(sam3CnLiveLists({ known: true, choices: { controlnet_modules: ['None', '', 'None', 3] } }).modules)
      .toEqual(['None'])
  })
})

describe('cnModuleOptions (전처리기 드롭다운)', () => {
  it('라이브 목록이 먼저 — 정적 목록에 없는 inpaint_noobai 도 고를 수 있다', () => {
    expect(cnModuleOptions(LIVE_MODULES, STATIC_ITEMS, 'inpaint_only')).toEqual(LIVE_MODULES)
  })

  it('라이브를 모르면 Python 정적 폴백 items, 그것도 없으면 현재 값 하나', () => {
    expect(cnModuleOptions(null, STATIC_ITEMS, 'canny')).toEqual(STATIC_ITEMS)
    expect(cnModuleOptions(null, '', 'depth_zoe')).toEqual(['depth_zoe'])
    expect(cnModuleOptions([], undefined, '')).toEqual(['inpaint_only'])
  })

  it('정적 폴백의 없음 표기는 대문자 None (Forge 는 소문자 none 을 모른다)', () => {
    expect(cnModuleOptions(null, STATIC_ITEMS, '')).toContain(CN_NONE)
    expect(cnModuleOptions(null, STATIC_ITEMS, '')).not.toContain('none')
  })
})

describe('canonicalCnName (저장된 대소문자 맞추기)', () => {
  it('소문자 none → None, 대소문자만 다른 이름 → 목록 표기', () => {
    expect(canonicalCnName('none', LIVE_MODULES)).toBe('None')
    expect(canonicalCnName('none', [])).toBe('None')
    expect(canonicalCnName('INPAINT_NOOBAI', LIVE_MODULES)).toBe('inpaint_noobai')
    expect(canonicalCnName('Anima-LLLite-Inpainting-V2', LIVE_MODELS)).toBe('anima-lllite-inpainting-v2')
  })

  it('정확히 같은 이름·빈 값·목록 밖 이름은 그대로 (바꿔치지 않는다)', () => {
    expect(canonicalCnName('canny', LIVE_MODULES)).toBe('canny')
    expect(canonicalCnName('', LIVE_MODULES)).toBe('')
    expect(canonicalCnName(undefined, LIVE_MODULES)).toBeUndefined()
    expect(canonicalCnName('my_custom', LIVE_MODULES)).toBe('my_custom')
  })
})

describe('cnModelOptions (모델 드롭다운 — 연결 때 목록 + 목록 밖 지금 값)', () => {
  it('목록 안 이름이면 라이브 목록 그대로', () => {
    expect(cnModelOptions(LIVE_MODELS, 'anima-lllite-inpainting-v2')).toEqual(LIVE_MODELS)
    expect(cnModelOptions(LIVE_MODELS, '')).toEqual(LIVE_MODELS)
    expect(cnModelOptions(LIVE_MODELS, undefined)).toEqual(LIVE_MODELS)
  })

  it('시작 뒤 models/sam3 에 넣은 LLLite 처럼 목록 밖 이름은 지우지 않고 끝에 남긴다', () => {
    // Forge ControlNet 새로고침 뒤 스냅샷엔 models/sam3 파일이 빠지지만 확장은 생성 때 다시 찾는다
    const afterCnRefresh = ['None', 'animaTileRepair_v10']
    expect(cnModelOptions(afterCnRefresh, 'anima-lllite-inpainting-v2'))
      .toEqual(['None', 'animaTileRepair_v10', 'anima-lllite-inpainting-v2'])
    expect(cnModelOptions(LIVE_MODELS, ' new-lllite ')).toEqual([...LIVE_MODELS, 'new-lllite'])
  })

  it('라이브 목록을 복사한다(스냅샷을 건드리지 않는다), 목록이 없으면 None + 지금 값', () => {
    const live = [...LIVE_MODELS]
    cnModelOptions(live, 'x').push('y')
    expect(live).toEqual(LIVE_MODELS)
    expect(cnModelOptions(null, 'x')).toEqual([CN_NONE, 'x'])
    expect(cnModelOptions([], '')).toEqual([CN_NONE])
  })
})

describe('cnNameMissing (Forge 에 없는 이름 표시)', () => {
  it('라이브 목록이 있을 때만, 목록 밖 이름이면 true', () => {
    expect(cnNameMissing('depth_zoe', LIVE_MODULES)).toBe(true)
    expect(cnNameMissing('inpaint_noobai', LIVE_MODULES)).toBe(false)
    expect(cnNameMissing('none', LIVE_MODULES)).toBe(true)   // 대소문자도 다르면 없는 이름
    expect(cnNameMissing('', LIVE_MODELS)).toBe(false)
    expect(cnNameMissing('depth_zoe', null)).toBe(false)     // 모르면 경고하지 않는다
  })
})

describe('Anima LLLite 전처리기 가드 (plan §7.2 #7·#14 — 원본처럼 사용자가 준 제어 이미지)', () => {
  // 원본은 LLLite 에 사용자가 준 RGB 제어 이미지를 그대로 준다:
  // origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:65-74
  //   Image.open(path).convert("RGB") → resize(BICUBIC) → arr / 127.5 - 1.0
  // origin: kohya-ss/ComfyUI-Anima-LLLite@b7495bd8:nodes.py:168-183
  //   cond_in_channels = int(meta.get("lllite.cond_in_channels", 3)); 4ch 는 MASK 필수, 3ch 는 MASK 를 버린다
  // origin: kohya-ss/sd-scripts@690ea7f9:docs/anima_train_control_net_lllite.md:355 ('"3" standard, "4" inpaint'),
  //   :77 (conditioning images e.g. lineart, canny, depth) — 3채널은 Tile & Repair 전용이 아니다.
  // 표 (이름, 채널, Tile & Repair) 는 파이썬 tests/test_sam3_cn_names.py LLLITE_NAME_CASES ·
  // 확장 tests/test_sam3_cn_lllite.py NameTests.CASES 와 같다.
  const CASES: Array<[unknown, 3 | 4 | null, boolean]> = [
    ['animaTileRepair_v20', 3, true],
    ['animaTileRepair_v10', 3, true],
    ['anima_tile-repair_v3', 3, true],
    ['anima_tiled_lllite_v1', 3, true],
    ['anima_lllite_lineart_v1', 3, false],
    ['anima-lllite-canny', 3, false],
    ['Anima_LLLite_Depth', 3, false],
    ['anima-lllite-inpainting-v2', 4, false],
    ['Anima-LLLite-Inpainting-V2', 4, false],
    ['new-lllite-inpaint', null, false],
    ['kohya_controllllite_xl_inpaint', null, false],
    ['kohya_controllllite_xl_canny_anime', null, false],
    ['controlnet_tile_sdxl', null, false],
    ['TileRepair_sdxl', null, false],
    ['my_lineart_cn', null, false],
    ['None', null, false],
    ['', null, false],
    [null, null, false],
  ]

  it('이름 → 채널·Tile & Repair 표', () => {
    for (const [name, want, tile] of CASES) {
      expect(sam3CnLlliteChannels(name), String(name)).toBe(want)
      expect(sam3CnLlliteTileRepair(name), String(name)).toBe(tile)
    }
  })

  it('녹화된 라이브 모델 목록 분류', () => {
    expect(LIVE_MODELS.map((model) => sam3CnLlliteChannels(model))).toEqual([null, 4, 3, 3])
    expect(LIVE_MODELS.map((model) => sam3CnLlliteTileRepair(model))).toEqual([false, false, true, true])
  })

  it('Tile & Repair: 저장값·기본값 inpaint_only 를 포함해 어떤 전처리기든 None', () => {
    const def = sam3CnDefaults()['_sam3_cn_module']
    expect(def).toBe('inpaint_only')
    for (const module of [def, 'tile_resample', 'canny', 'none']) {
      expect(sam3CnLlliteModule(module, 'animaTileRepair_v20'))
        .toEqual({ module: CN_NONE, channels: 3, tileRepair: true, forced: true })
    }
    expect(sam3CnLlliteModule('None', 'animaTileRepair_v20'))
      .toEqual({ module: 'None', channels: 3, tileRepair: true, forced: false })
  })

  it('3채널 lineart·canny·depth Anima LLLite: 고른 전처리기 그대로, inpaint_* 만 None', () => {
    expect(sam3CnLlliteModule('lineart_anime', 'anima_lllite_lineart_v1'))
      .toEqual({ module: 'lineart_anime', channels: 3, tileRepair: false, forced: false })
    expect(sam3CnLlliteModule('canny', 'anima-lllite-canny').forced).toBe(false)
    expect(sam3CnLlliteModule('depth_anything_v2', 'Anima_LLLite_Depth').module).toBe('depth_anything_v2')
    for (const module of ['inpaint_only', 'inpaint_global_harmonious', 'inpaint_only+lama']) {
      expect(sam3CnLlliteModule(module, 'anima_lllite_lineart_v1'))
        .toEqual({ module: CN_NONE, channels: 3, tileRepair: false, forced: true })
    }
  })

  it('4채널 인페인트: inpaint_* 만 None, 다른 전처리기는 그대로', () => {
    expect(sam3CnLlliteModule('inpaint_only', 'anima-lllite-inpainting-v2').module).toBe(CN_NONE)
    expect(sam3CnLlliteModule('inpaint_noobai', 'anima-lllite-inpainting-v2').forced).toBe(true)
    expect(sam3CnLlliteModule('depth_leres', 'anima-lllite-inpainting-v2'))
      .toEqual({ module: 'depth_leres', channels: 4, tileRepair: false, forced: false })
  })

  it('Anima LLLite 가 아닌 모델(SDXL controllllite 포함)은 건드리지 않는다', () => {
    const untouched = { module: 'inpaint_only', channels: null, tileRepair: false, forced: false }
    expect(sam3CnLlliteModule('inpaint_only', 'None')).toEqual(untouched)
    expect(sam3CnLlliteModule('inpaint_only', '')).toEqual(untouched)
    expect(sam3CnLlliteModule('inpaint_only', 'kohya_controllllite_xl_inpaint')).toEqual(untouched)
  })

  it('드롭다운: Tile & Repair 는 None 하나, 그 밖의 Anima LLLite 는 inpaint_* 를 뺀 목록, 그 밖은 그대로(복사)', () => {
    expect(cnModuleOptionsForModel(LIVE_MODULES, 'animaTileRepair_v20')).toEqual([CN_NONE])
    const four = cnModuleOptionsForModel(LIVE_MODULES, 'anima-lllite-inpainting-v2')
    expect(four).toContain(CN_NONE)
    expect(four.some((name) => name.startsWith('inpaint'))).toBe(false)
    expect(four).toEqual(['None', 'canny', 'tile_resample'])
    expect(cnModuleOptionsForModel(LIVE_MODULES, 'anima_lllite_lineart_v1')).toEqual(['None', 'canny', 'tile_resample'])
    const plain = cnModuleOptionsForModel(LIVE_MODULES, 'None')
    expect(plain).toEqual(LIVE_MODULES)
    expect(plain).not.toBe(LIVE_MODULES)
    expect(cnModuleOptionsForModel(LIVE_MODULES, 'kohya_controllllite_xl_inpaint')).toEqual(LIVE_MODULES)
    expect(cnModuleOptionsForModel(['inpaint_only'], 'anima-lllite-inpainting-v2')).toEqual([CN_NONE])
  })
})
