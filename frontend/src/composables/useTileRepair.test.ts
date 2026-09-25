import { describe, expect, it, vi } from 'vitest'
import { createTileRepair } from './useTileRepair'

function fixture() {
  const handlers = new Map<string, (json: string) => void>()
  const requestAction = vi.fn()
  let n = 0
  const store = createTileRepair({
    onBackendEvent: (name, cb) => { handlers.set(name, cb); return () => handlers.delete(name) },
    requestAction,
    newId: kind => `${kind}_${++n}`,
  })
  store.bind()
  const emit = (event: object) => handlers.get('tileRepairResult')!(JSON.stringify(event))
  return { store, handlers, requestAction, emit }
}

const OPTIONS = {
  version: 1, available: true,
  models: ['animaTileRepair_v10.safetensors', 'animaTileRepair_v20.safetensors'],
  default_model: 'animaTileRepair_v20.safetensors',
  dit: ['Use Forge current'], text_encoder: ['Use Forge current', 'qwen_3_06b_base.safetensors'],
  vae: ['Use Forge current', 'qwen_image_vae.safetensors'],
  defaults: { steps: 50, text_encoder: 'qwen_3_06b_base.safetensors' },
  ranges: { multiplier: [-10, 10] }, increments: { multiplier: 0.01 },
}

describe('createTileRepair', () => {
  it('starts from the original defaults (model fields empty = extension default)', () => {
    const { store } = fixture()
    // origin: kohya-ss/sd-scripts@690ea7f9:anima_minimal_inference_control_net_lllite.py:127-134, :167-170
    expect({ ...store.settings }).toMatchObject({
      model: '', negative_prompt: '', steps: 50, cfg_scale: 3.5, flow_shift: 5, multiplier: 1,
      short_side: 1024, seed: -1, dit: '', text_encoder: '', vae: '', unload_forge_before: true,
    })
    expect(store.settings.prompt).toContain('repair the low-quality anime image')
  })

  it('listens once and stops listening on unbind', () => {
    const { store, handlers } = fixture()
    store.bind()
    expect(handlers.has('tileRepairResult')).toBe(true)
    store.unbind()
    expect(handlers.has('tileRepairResult')).toBe(false)
  })

  it('loads options and ignores answers to other requests', () => {
    const { store, requestAction, emit } = fixture()
    store.loadOptions()
    expect(requestAction).toHaveBeenCalledWith('tile_repair_options', { requestId: 'options_1' })
    expect(store.loadingOptions.value).toBe(true)
    emit({ action: 'tile_repair_options', requestId: 'someone_else', ok: true, options: OPTIONS })
    expect(store.options.value).toBeNull()
    emit({ action: 'tile_repair_options', requestId: 'options_1', ok: true, options: OPTIONS })
    expect(store.options.value?.default_model).toBe('animaTileRepair_v20.safetensors')
    expect(store.loadingOptions.value).toBe(false)
  })

  it('reports an options failure and a malformed options body', () => {
    const { store, emit } = fixture()
    store.loadOptions()
    emit({ action: 'tile_repair_options', requestId: 'options_1', ok: false, error: '확장을 업데이트하세요' })
    expect(store.optionsError.value).toBe('확장을 업데이트하세요')
    store.loadOptions()
    emit({ action: 'tile_repair_options', requestId: 'options_2', ok: true, options: { models: 'x' } })
    expect(store.options.value).toBeNull()
    expect(store.optionsError.value).toBeTruthy()
  })

  it('sends the local path first, else the uploaded data URL, with a settings snapshot', () => {
    const { store, requestAction, emit } = fixture()
    store.settings.multiplier = -2.5
    expect(store.run('C:/out/a.png', 'file:///C:/out/a.png')).toBe(true)
    expect(requestAction).toHaveBeenLastCalledWith('tile_repair_run', {
      requestId: 'run_1', image_path: 'C:/out/a.png', image: '',
      settings: expect.objectContaining({ multiplier: -2.5, steps: 50 }),
    })
    const sent = requestAction.mock.calls[requestAction.mock.calls.length - 1][1].settings
    store.settings.steps = 20                         // later edits do not change what was sent
    expect(sent.steps).toBe(50)
    expect(store.run('C:/b.png', '')).toBe(false)     // one run at a time
    emit({ action: 'tile_repair_run', requestId: 'run_1', ok: false, error: 'x' })
    store.run('', 'data:image/png;base64,AAAA')
    expect(requestAction).toHaveBeenLastCalledWith('tile_repair_run', expect.objectContaining({
      image_path: '', image: 'data:image/png;base64,AAAA',
    }))
  })

  it('refuses to run without a usable source or prompt', () => {
    const { store, requestAction } = fixture()
    expect(store.run('', 'https://example.com/a.png')).toBe(false)
    expect(store.error.value).toContain('원본')
    store.settings.prompt = '  '
    expect(store.run('C:/a.png', '')).toBe(false)
    expect(store.error.value).toContain('프롬프트')
    expect(requestAction).not.toHaveBeenCalled()
  })

  it('keeps the result, then reuses its seed', () => {
    const { store, emit } = fixture()
    store.run('C:/a.png', '')
    expect(store.busy.value).toBe(true)
    emit({ action: 'tile_repair_run', requestId: 'stale', ok: true, path: 'C:/old.png' })
    expect(store.busy.value).toBe(true)
    emit({
      action: 'tile_repair_run', requestId: 'run_1', ok: true, path: 'C:/out/tile_repair/t.png',
      width: 1024, height: 1472, seed: 987, info: 'repair\nSteps: 50', model: 'animaTileRepair_v20.safetensors',
    })
    expect(store.busy.value).toBe(false)
    expect(store.result.value).toEqual({
      path: 'C:/out/tile_repair/t.png', width: 1024, height: 1472, seed: 987, info: 'repair\nSteps: 50',
      model: 'animaTileRepair_v20.safetensors',
    })
    store.reuseSeed()
    expect(store.settings.seed).toBe(987)
    store.resetSettings()
    expect(store.settings.seed).toBe(-1)
  })

  it('cancels the running request and treats the run answer as the end', () => {
    const { store, requestAction, emit } = fixture()
    store.cancel()                                     // nothing running → nothing sent
    expect(requestAction).not.toHaveBeenCalled()
    store.run('C:/a.png', '')
    store.cancel(); store.cancel()
    expect(requestAction.mock.calls.filter(c => c[0] === 'tile_repair_cancel')).toEqual([
      ['tile_repair_cancel', { requestId: 'run_1' }],
    ])
    expect(store.cancelling.value).toBe(true)
    emit({ action: 'tile_repair_cancel', requestId: 'run_1', ok: true, stopped: true })
    expect(store.cancelling.value).toBe(true)          // stopped — the run's answer ends it
    emit({ action: 'tile_repair_run', requestId: 'run_1', ok: false, canceled: true, error: 'Tile & Repair 를 취소했습니다.' })
    expect(store.busy.value).toBe(false)
    expect(store.cancelling.value).toBe(false)
    expect(store.notice.value).toContain('취소')
    expect(store.error.value).toBe('')
  })

  it('lets the user cancel again when Forge found nothing to stop', () => {
    // e.g. the run request had not reached Forge yet: stopped=false must not leave the button disabled
    // (the run then went on in full) nor claim the job is already finishing.
    const { store, requestAction, emit } = fixture()
    store.run('C:/a.png', '')
    store.cancel()
    emit({ action: 'tile_repair_cancel', requestId: 'run_1', ok: true, stopped: false })
    expect(store.busy.value).toBe(true)
    expect(store.cancelling.value).toBe(false)
    expect(store.notice.value).not.toContain('이미 끝나는 중')
    expect(store.notice.value).toContain('다시')
    store.cancel()
    expect(requestAction.mock.calls.filter(c => c[0] === 'tile_repair_cancel')).toHaveLength(2)
    expect(store.cancelling.value).toBe(true)
  })

  it('shows run errors and survives broken payloads', () => {
    const { store, emit, handlers } = fixture()
    store.run('C:/a.png', '')
    handlers.get('tileRepairResult')!('not json')
    emit({ action: 'unknown', requestId: 'run_1', ok: true })
    expect(store.busy.value).toBe(true)
    emit({ action: 'tile_repair_run', requestId: 'run_1', ok: false, error: 'Forge Tile & Repair 준비가 안 됐습니다' })
    expect(store.error.value).toContain('준비')
    expect(store.result.value).toBeNull()
  })
})
