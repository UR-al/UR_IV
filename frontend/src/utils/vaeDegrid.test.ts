import { describe, expect, it } from 'vitest'
import {
  AUTO_LABEL, COMFY_OPTIONS, DEFAULTS, UPSCALER_DEGRID_WARNING, MODE_OPTIONS, OPTION_KEYS, WIDGET_IDS,
  cardStatus, coerceStrength, coerceTile, effectiveOption, upscalerDegridWarning, keyOf, labelOf, liveModels,
  modelChoices, modelStem, normalizeModel, readValues, refreshTitle, resolveModel, summary, type DegridValues,
} from './vaeDegrid'
import { SPECS } from './forgeOptionOverrides'

const V11 = 'qwenVAEDegridNafnet_v11'
const V10 = 'NAFNet-QwenVAE-DeGrid'
const on = (patch: Partial<DegridValues> = {}): DegridValues => ({ ...DEFAULTS, enabled: 'true', ...patch })
const forge = (models: string[] | undefined, degrid = true) => {
  const choices: Record<string, string[]> = models ? { degrid_models: models } : {}
  return { status: 'ok', known: true, features: { degrid }, choices } as const
}
const comfy = { status: 'not_applicable', known: false } as const

describe('table and values', () => {
  it('widget ids, defaults, mode labels', () => {
    expect(Object.values(WIDGET_IDS).every(id => id.startsWith('_degrid_'))).toBe(true)
    expect(DEFAULTS).toEqual({ enabled: 'false', model: '', mode: 'full', strength: '1', tile: '512', apply_img2img: 'false' })
    expect(MODE_OPTIONS.map(o => o.key)).toEqual(['full', 'dark', 'bright'])
    for (const o of MODE_OPTIONS) expect(keyOf(MODE_OPTIONS, labelOf(MODE_OPTIONS, o.key))).toBe(o.key)
    expect(OPTION_KEYS.every(key => SPECS.some(spec => spec.key === key))).toBe(true)
    expect(COMFY_OPTIONS).toEqual({ sam3_degrid_device: 'auto', sam3_degrid_gpu_precision: 'fp32', sam3_degrid_keep_loaded: false })
  })

  it('readValues: empty → defaults, unknown mode → full, model "" / None / auto = auto', () => {
    expect(readValues({})).toEqual(DEFAULTS)
    expect(readValues({ [WIDGET_IDS.mode]: 'weird', [WIDGET_IDS.model]: 'None' }).mode).toBe('full')
    expect(readValues({ [WIDGET_IDS.model]: 'None' }).model).toBe('')
    expect(readValues({ [WIDGET_IDS.model]: 'AUTO' }).model).toBe('')
    expect(readValues({ [WIDGET_IDS.model]: ` ${V11} ` }).model).toBe(V11)
    expect(normalizeModel(null)).toBe('')
  })

  it('coercion mirrors the extension rules (python coerce_strength / coerce_tile)', () => {
    for (const [raw, want] of [[-1, 0], [0, 0], [1.5, 1.5], [2, 1.5], ['x', 1], ['', 1], [Number.NaN, 1], ['0.85', 0.85]] as const) {
      expect(coerceStrength(raw)).toBe(want)
    }
    for (const [raw, want] of [[0, 0], [-5, 0], [1, 128], [127, 128], [128, 128], [300, 300], [300.9, 300], [4096, 4096], [5000, 4096], ['x', 512], ['', 512]] as const) {
      expect(coerceTile(raw)).toBe(want)
    }
  })

  it('summary = python describe wording', () => {
    expect(summary(DEFAULTS)).toBe('')
    expect(summary(on())).toBe('Full 1 · 512')
    expect(summary(on({ mode: 'dark', strength: '0.8', tile: '0', model: V11 }))).toBe(`Dark 0.8 · 타일 없음 · ${V11}`)
    expect(summary(on({ mode: 'bright', strength: '9', tile: '100' }))).toBe('Bright 1.5 · 128')
  })
})

describe('model list', () => {
  it('drops the None placeholder everywhere; ComfyUI list wins when it is an array', () => {
    expect(liveModels(forge(['None']))).toEqual([])
    expect(liveModels(forge([V11, 'None', V10, V11]))).toEqual([V11, V10])
    expect(liveModels(forge(undefined))).toBeNull()
    expect(liveModels(null)).toBeNull()
    expect(liveModels(forge([V11]), ['ESRGAN/x', 'None'])).toEqual(['ESRGAN/x'])
    expect(liveModels(forge([V11]), null)).toEqual([V11])
  })

  it('auto first with the name it resolves to now, then the list', () => {
    const got = modelChoices(forge([V11, V10]), '')
    expect(got.options).toEqual([
      { value: '', label: `자동 (가장 높은 버전 — 지금: ${V11})` }, { value: V11, label: V11 }, { value: V10, label: V10 },
    ])
    expect(got).toMatchObject({ unverified: false, missing: false, autoName: V11 })
    expect(modelChoices(null, '').options).toEqual([{ value: '', label: AUTO_LABEL }])
    expect(modelChoices(null, '').unverified).toBe(true)
  })

  it('keeps the saved value; flags it missing only against a known list', () => {
    const missing = modelChoices(forge([V11]), 'gone')
    expect(missing.options[1]).toEqual({ value: 'gone', label: 'gone (목록에 없음)' })
    expect(missing.missing).toBe(true)
    const unknown = modelChoices(null, 'gone')
    expect(unknown.options[1]).toEqual({ value: 'gone', label: 'gone' })
    expect(unknown.missing).toBe(false)
    const cased = modelChoices(forge([V11]), V11.toUpperCase())
    expect(cased.missing).toBe(false)                  // 확장처럼 대소문자·줄기로 찾는다
    expect(cased.options.map(o => o.value)).toEqual(['', V11.toUpperCase(), V11])
    const comfyList = modelChoices(comfy, V10, [V10])
    expect(comfyList.options.map(o => o.value)).toEqual(['', V10])
  })

  it('resolveModel: exact → case-insensitive → stem', () => {
    expect(resolveModel('', [V11, V10])).toBe(V11)
    expect(resolveModel(V10.toLowerCase(), [V11, V10])).toBe(V10)
    expect(resolveModel(`${V10}.safetensors`, [V11, V10])).toBe(V10)
    expect(resolveModel(`ESRGAN/${V11}`, [V11])).toBe(V11)
    expect(resolveModel('x', [])).toBeNull()
    expect(modelStem('upscale_models\\Foo.PTH')).toBe('foo')
  })
})

describe('cardStatus', () => {
  it('off and krea2 first', () => {
    expect(cardStatus({ caps: forge([V11]), values: DEFAULTS }).status).toBe('off')
    expect(cardStatus({ caps: forge([V11]), values: on(), krea2: true }).status).toBe('krea2')
    expect(cardStatus({ caps: comfy, values: on(), krea2: true, comfyModels: [V11] }).status).toBe('krea2')
  })

  it('Forge: unknown sends, missing script does not, list states, other model note', () => {
    expect(cardStatus({ caps: null, values: on() })).toMatchObject({ status: 'unknown', tone: 'warn' })
    expect(cardStatus({ caps: { status: 'unknown', known: false }, values: on() }).text).toContain('그대로 보냅니다')
    expect(cardStatus({ caps: forge([V11], false), values: on() })).toMatchObject({ status: 'missing', tone: 'alert' })
    expect(cardStatus({ caps: forge(['None']), values: on() })).toMatchObject({ status: 'no_model', tone: 'alert' })
    expect(cardStatus({ caps: forge([V11]), values: on({ model: 'gone' }) })).toMatchObject({ status: 'model_missing', tone: 'warn' })
    expect(cardStatus({ caps: forge(undefined), values: on({ model: 'gone' }) }).status).toBe('ok')   // 목록 모름 — 막지 않는다
    expect(cardStatus({ caps: forge([V11]), values: on(), kind: 'other' })).toMatchObject({ status: 'other_model', tone: 'warn' })
    expect(cardStatus({ caps: forge([V11]), values: on(), kind: 'v2' })).toMatchObject({ status: 'ok', tone: 'info' })
  })

  it('ComfyUI: pack update, no model, model missing, ok — texts say Forge options do not apply', () => {
    expect(cardStatus({ caps: comfy, values: on(), comfyModels: null })).toMatchObject({ status: 'comfy_pack_update', tone: 'alert' })
    expect(cardStatus({ caps: comfy, values: on(), comfyModels: null }).text).toContain('1.5.0')
    expect(cardStatus({ caps: comfy, values: on(), comfyModels: [] }).status).toBe('comfy_no_model')
    expect(cardStatus({ caps: comfy, values: on(), comfyModels: ['None'] }).status).toBe('comfy_no_model')
    expect(cardStatus({ caps: comfy, values: on({ model: 'gone' }), comfyModels: [V11] }).status).toBe('comfy_model_missing')
    // ComfyUI 의 '없음' 상태는 ↻ 로 목록을 다시 받을 수 있다고 말한다(↻ 가 ComfyUI object_info 를 다시 읽는다)
    for (const comfyModels of [null, [], [V11]]) {
      const status = cardStatus({ caps: comfy, values: on({ model: comfyModels?.length ? 'gone' : '' }), comfyModels })
      expect(status.text).toContain('↻')
    }
    const ok = cardStatus({ caps: comfy, values: on({ model: V11 }), comfyModels: [V11] })
    expect(ok.status).toBe('comfy_ok')
    expect(ok.text).toContain('ComfyUI 에 해당 없음')
    expect(ok.text).toContain('auto · fp32 · 끔')
    const other = cardStatus({ caps: comfy, values: on(), comfyModels: [V11], kind: 'other' })
    expect(other.status).toBe('other_model')
    expect(other.text).toContain('auto · fp32 · 끔')
  })
})

describe('effectiveOption (read-only mirror of the Forge options)', () => {
  const caps = (options: Record<string, unknown>, optionsKnown = true) =>
    ({ status: 'ok', known: true, options_known: optionsKnown, options }) as const

  it('app override wins, else the Forge value, labels from the P10 spec', () => {
    const forgeCpu = caps({ sam3_degrid_device: 'cpu', sam3_degrid_keep_loaded: false, sam3_degrid_gpu_precision: 'fp32' })
    expect(effectiveOption('sam3_degrid_device', {}, forgeCpu)).toMatchObject({ source: 'forge', text: 'Forge 설정: CPU (VRAM 안 씀, 느림)' })
    expect(effectiveOption('sam3_degrid_keep_loaded', {}, forgeCpu).text).toBe('Forge 설정: 끔')
    const app = effectiveOption('sam3_degrid_gpu_precision', { sam3_degrid_gpu_precision: 'fp16' }, forgeCpu)
    expect(app).toMatchObject({ source: 'app', text: '앱 설정: fp16 autocast', blocked: false })
    expect(effectiveOption('sam3_degrid_device', { sam3_degrid_device: 'bogus' }, forgeCpu).source).toBe('forge')
  })

  it('blocked and unknown states; ComfyUI shows the fixed node values', () => {
    const blocked = effectiveOption('sam3_degrid_device', { sam3_degrid_device: 'cpu' }, caps({}))
    expect(blocked).toMatchObject({ source: 'app', blocked: true })
    expect(blocked.text).toContain('보내지 못함')
    expect(effectiveOption('sam3_degrid_device', {}, null)).toMatchObject({ source: 'unknown', text: 'Forge 설정 따름 (값 확인 전)' })
    expect(effectiveOption('sam3_degrid_device', {}, caps({})).text).toContain('이 Forge 에 없음')
    const comfyRow = effectiveOption('sam3_degrid_device', { sam3_degrid_device: 'cpu' }, comfy)
    expect(comfyRow).toMatchObject({ source: 'comfy', text: 'ComfyUI 노드: GPU (Forge 장치)' })
    expect(effectiveOption('sam3_degrid_keep_loaded', {}, comfy).text).toBe('ComfyUI 노드: 끔')
    expect(effectiveOption('sam3_degrid_gpu_precision', {}, comfy).text).toBe('ComfyUI 노드: fp32 (ComfyUI 노드와 같음)')
  })
})

describe('upscalerDegridWarning', () => {
  it('matches the Forge list and the ComfyUI list by stem, case-insensitive; None ignored', () => {
    expect(upscalerDegridWarning(V11, forge([V11]))).toBe(UPSCALER_DEGRID_WARNING)
    expect(upscalerDegridWarning(V11.toLowerCase(), forge([`ESRGAN/${V11}`]))).toBe(UPSCALER_DEGRID_WARNING)
    expect(upscalerDegridWarning(`${V11}.safetensors`, comfy, [V11])).toBe(UPSCALER_DEGRID_WARNING)
    expect(upscalerDegridWarning(`upscale_models/${V10}.PTH`, null, [V10])).toBe(UPSCALER_DEGRID_WARNING)
    expect(upscalerDegridWarning('R-ESRGAN 4x+', forge([V11]), [V10])).toBe('')
    expect(upscalerDegridWarning('None', forge(['None']), ['None'])).toBe('')
    expect(upscalerDegridWarning('', forge([V11]))).toBe('')
    expect(upscalerDegridWarning(V11, null, null)).toBe('')
  })
})

describe('refreshTitle', () => {
  it('names the backend the refresh button actually reloads', () => {
    expect(refreshTitle(comfy)).toContain('ComfyUI')
    expect(refreshTitle(comfy)).not.toContain('Forge')
    for (const caps of [forge([V11]), null, undefined, { status: 'unknown' }]) {
      expect(refreshTitle(caps)).toContain('Forge')
      expect(refreshTitle(caps)).not.toContain('ComfyUI')
    }
  })
})
