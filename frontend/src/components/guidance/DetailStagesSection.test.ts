import { describe, expect, it, vi } from 'vitest'
import { createSSRApp, reactive, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import type { SamExtraCapabilitiesEvent } from '../../types/bridge'

const caps = ref<Partial<SamExtraCapabilitiesEvent> | null>(null)
vi.mock('../../composables/useSamExtraCapabilities', () => ({
  useSamExtraCapabilities: () => ({ capabilities: caps, refresh: () => {}, mayUse: () => true }),
}))

import DetailStagesSection from './DetailStagesSection.vue'

/**
 * 디테일 단계(TSR · Momentum · HiGS · HiFlow — 확장 인자 71-90). 칸 범위는 tests/test_guidance_detail_inputs.py 가
 * 확장 슬라이더(픽스처)와 대조하고, 여기서는 스위치마다 칸이 숨고 보이는지와 안내만 본다.
 */
async function render(values: Record<string, string>) {
  const widgets = reactive<Record<string, any>>(values)
  return renderToString(createSSRApp(DetailStagesSection, { widgets }))
}

function numberInputs(html: string): string[] {
  return html.match(/<input[^>]*type="number"[^>]*>/g) ?? []
}

const STAGES: Array<[string, number]> = [
  // 스위치 → 그 스위치가 보여 주는 숫자 칸 수(HiGS 세부값 포함)
  ['_guid_tsr_enabled', 2], ['_guid_mg_enabled', 4], ['_guid_higs_enabled', 6], ['_guid_hiflow_enabled', 3],
]

describe('DetailStagesSection', () => {
  it('shows the four switches and no number field while they are off', async () => {
    const html = await render({})
    for (const label of ['Enable TSR', 'Enable Momentum Guidance', 'Enable HiGS', 'Enable HiFlow (Hires.fix 전용)']) {
      expect(html).toContain(label)
    }
    expect(numberInputs(html)).toHaveLength(0)
  })

  it("opens each stage's fields with its own switch", async () => {
    for (const [key, count] of STAGES) {
      expect(numberInputs(await render({ [key]: 'true' })), key).toHaveLength(count)
    }
    const all = Object.fromEntries(STAGES.map(([key]) => [key, 'true']))
    expect(numberInputs(await render(all))).toHaveLength(STAGES.reduce((sum, [, n]) => sum + n, 0))
  })

  it('warns about HiGS w 1.75 on multistep samplers (measured on Anima 3.8B)', async () => {
    const html = await render({ _guid_higs_enabled: 'true' })
    expect(html).toContain('Res Multistep')
    expect(html).toContain('0.5 이하부터')
  })

  it('says HiFlow does nothing while Hires.fix is off', async () => {
    const note = 'Hires.fix 가 꺼져 있어 HiFlow 는 아무것도 하지 않습니다'
    expect(await render({ _guid_hiflow_enabled: 'true', hires_options_group: 'false' })).toContain(note)
    expect(await render({ _guid_hiflow_enabled: 'true', hires_options_group: 'true' })).not.toContain(note)
  })

  it('tells when the connected sam-extra has no detail stages, once a stage is on', async () => {
    caps.value = { known: true, anima_guidance_argc: 62 }
    expect(await render({})).not.toContain('인자 62개 빌드')
    expect(await render({ _guid_mg_enabled: 'true' })).toContain('인자 62개 빌드')
    caps.value = { known: true, anima_guidance_argc: 91 }
    expect(await render({ _guid_mg_enabled: 'true' })).not.toContain('인자 62개 빌드')
    caps.value = null
  })
})
