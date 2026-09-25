import { describe, expect, it } from 'vitest'
import type { SamExtraCapabilitiesEvent, SamExtraFeature } from '../types/bridge'
import {
  mayUseFeature, missingEnabledFeatures, missingFeaturesMessage, parseSamExtraCapabilities,
} from './samExtraCapabilities'

const FLAGS: SamExtraFeature[] = [
  'sam3', 'anima_guidance', 'skimmed_cfg', 'detail_daemon', 'anima38', 'dora', 'vae2x',
  'lora_manager', 'memo_routes', 'tipo_route', 'reference_route', 'contract_route',
  'tile_repair_route',
]

function snapshot(over: Partial<SamExtraCapabilitiesEvent> = {}, on: SamExtraFeature[] = FLAGS): SamExtraCapabilitiesEvent {
  const features = Object.fromEntries(FLAGS.map(f => [f, on.includes(f)])) as Record<SamExtraFeature, boolean>
  return {
    status: 'ok', known: true, checked_at: '2026-09-25T05:00:00Z', installed: true, features,
    anima_guidance_argc: 62, detail_daemon_hires: true, version: {}, scripts: {}, sam3_keys: {},
    options: {}, options_known: false, gradio_api: [], choices: {}, warnings: [], errors: {},
    ...over,
  }
}

const widgets = (values: Record<string, string>) => (id: string) => values[id]

describe('parseSamExtraCapabilities', () => {
  it('reads a snapshot and rejects broken payloads', () => {
    expect(parseSamExtraCapabilities(JSON.stringify(snapshot()))?.status).toBe('ok')
    expect(parseSamExtraCapabilities('{not json')).toBeNull()
    expect(parseSamExtraCapabilities('null')).toBeNull()
    expect(parseSamExtraCapabilities(JSON.stringify({ status: 'ok' }))).toBeNull()
  })
})

describe('mayUseFeature', () => {
  it('never blocks when the snapshot is unknown', () => {
    expect(mayUseFeature(null, 'sam3')).toBe(true)
    expect(mayUseFeature(snapshot({ status: 'unreachable', known: false }, []), 'sam3')).toBe(true)
  })
  it('follows the flag when known', () => {
    expect(mayUseFeature(snapshot({}, ['sam3']), 'sam3')).toBe(true)
    expect(mayUseFeature(snapshot({}, ['sam3']), 'dora')).toBe(false)
  })
})

describe('missingEnabledFeatures', () => {
  it('lists enabled features the backend lacks', () => {
    const caps = snapshot({ installed: false }, [])
    const read = widgets({ sam3_group: 'true', _guid_dcw_enabled: 'true', _dd_enabled: 'false' })
    expect(missingEnabledFeatures(caps, read)).toEqual(['sam3', 'anima_guidance'])
  })
  it('does not count the RDC switch alone — RDC only runs inside DCW (Forge import sets it true)', () => {
    const caps = snapshot({ installed: false }, [])
    expect(missingEnabledFeatures(caps, widgets({ _guid_rdc_enabled: 'true', _guid_rdc_tau: '0.2' }))).toEqual([])
    expect(missingEnabledFeatures(caps, widgets({ _guid_rdc_enabled: 'true', _guid_dcw_enabled: 'true' })))
      .toEqual(['anima_guidance'])
  })
  it('stays quiet when nothing is enabled, the feature exists, or the snapshot is unknown', () => {
    expect(missingEnabledFeatures(snapshot({ installed: false }, []), widgets({}))).toEqual([])
    expect(missingEnabledFeatures(snapshot(), widgets({ sam3_group: 'true' }))).toEqual([])
    expect(missingEnabledFeatures(snapshot({ status: 'not_applicable', known: false }, []),
      widgets({ sam3_group: 'true' }))).toEqual([])
  })
})

describe('missingFeaturesMessage', () => {
  it('distinguishes a missing extension from a missing script', () => {
    expect(missingFeaturesMessage(snapshot({ installed: false }, []), ['sam3'])).toContain('확장이 없습니다')
    expect(missingFeaturesMessage(snapshot({}, ['sam3']), ['detail_daemon'])).toContain('Detail Daemon')
    expect(missingFeaturesMessage(snapshot(), [])).toBe('')
  })
})
