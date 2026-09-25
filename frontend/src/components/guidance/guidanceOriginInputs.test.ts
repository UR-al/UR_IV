import { describe, expect, it } from 'vitest'
import { createSSRApp, reactive, type Component } from 'vue'
import { renderToString } from '@vue/server-renderer'
import DcwSection from './DcwSection.vue'
import CwmSmcSection from './CwmSmcSection.vue'
import RdcSection from './RdcSection.vue'
import DaveSection from './DaveSection.vue'
import CnsSection from './CnsSection.vue'
import SkimSection from './SkimSection.vue'
import PagSection from './PagSection.vue'
import pagSource from './PagSection.vue?raw'
import dcwSource from './DcwSection.vue?raw'
import cwmSmcSource from './CwmSmcSection.vue?raw'
import rdcSource from './RdcSection.vue?raw'
import daveSource from './DaveSection.vue?raw'
import cnsSource from './CnsSection.vue?raw'
import skimSource from './SkimSection.vue?raw'

/**
 * 가이던스 칸의 범위·step = 원본 노드 입력(APP-UI). 기본값·전송은 core/anima_guidance.py 스펙이 단일 출처라
 * 여기서는 칸의 min/max/step 과 원본 뜻을 알리는 라벨·안내만 지킨다. 표는 원본 INPUT_TYPES 에서 숫자만 옮겼다
 * (DCW·CNS 는 GPL-3.0 — 코드는 옮기지 않는다).
 *
 * 위젯 키 → [min, max, step]
 */
type Range = [number, number, number]
interface OriginSection {
  name: string
  component: Component
  source: string
  /** 칸을 보이게 하는 스위치 · 선택 값 */
  show: Record<string, string>
  inputs: Record<string, Range>
  /** 원본 노드에 없는 확장 전용 숫자 칸 — 범위 표에서 빠진다(확장 범위를 그대로 둔다) */
  hostOnly?: string[]
}

const SECTIONS: OriginSection[] = [
  {
    // origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201-207
    // (scale 0~100 / .1, perturbation_strength 0~1 / .01, start·end_percent 0~1 / .001, rescale 0~1 / .01)
    name: 'PagSection', component: PagSection, source: pagSource,
    show: { _guid_enabled: 'true' },
    inputs: {
      _guid_scale: [0, 100, 0.1],
      _guid_official_strength: [0, 1, 0.01],
      _guid_start_percent: [0, 1, 0.001],
      _guid_end_percent: [0, 1, 0.001],
      _guid_rescale: [0, 1, 0.01],
    },
    // SEG query blur sigma · SLG scale · Legacy strength — 원본 PAG 노드에 없는 확장 기능
    hostOnly: ['_guid_seg_sigma', '_guid_slg_scale', '_guid_legacy_strength'],
  },
  {
    // origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:637-667 (lambda_l ±0.5 / .005, lambda_h ±0.3 / .001)
    name: 'DcwSection', component: DcwSection, source: dcwSource,
    show: { _guid_dcw_enabled: 'true' },
    inputs: {
      _guid_dcw_lambda_low: [-0.5, 0.5, 0.005],
      _guid_dcw_lambda_high: [-0.3, 0.3, 0.001],
    },
  },
  {
    // origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:675-705 (alpha_l/alpha_h), :732-755 (smc_lambda, smc_k)
    name: 'CwmSmcSection', component: CwmSmcSection, source: cwmSmcSource,
    show: { _guid_cwm_enabled: 'true', _guid_smc_preset: 'Custom' },
    inputs: {
      _guid_cwm_alpha_low: [-1, 2, 0.01],
      _guid_cwm_alpha_high: [-1, 2, 0.01],
      _guid_smc_lambda: [0.5, 30, 0.1],
      _guid_smc_k: [0, 5, 0.01],
    },
  },
  {
    // origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-815 (rdc_tau — 0 = 끔, 따로 스위치 없음 — rdc_alpha_ll/hh)
    name: 'RdcSection', component: RdcSection, source: rdcSource,
    show: { _guid_rdc_enabled: 'true' },
    inputs: {
      _guid_rdc_tau: [0, 0.5, 0.01],
      _guid_rdc_alpha_ll: [0, 0.3, 0.005],
      _guid_rdc_alpha_hh: [0, 0.1, 0.001],
    },
  },
  {
    // origin: sorryhyun/ComfyUI-Anima-DAVE@83143e8d:nodes.py:121-147 (strength, tau)
    name: 'DaveSection', component: DaveSection, source: daveSource,
    show: { _guid_dave_enabled: 'true' },
    inputs: {
      _guid_dave_strength: [0, 1, 0.01],
      _guid_dave_tau: [0, 1, 0.01],
    },
  },
  {
    // origin: namemechan/comfyui-cns_sampler_patch@42278b13:cns_sampler_patch.py:396-437
    name: 'CnsSection', component: CnsSection, source: cnsSource,
    show: { _guid_cns_enabled: 'true' },
    inputs: {
      _guid_cns_strength: [0, 1, 0.05],
      _guid_cns_gamma_power: [0.1, 2, 0.05],
      _guid_cns_gamma_scale: [0.1, 25, 0.1],
    },
  },
  {
    // origin: Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py:5-6 (MAX_SCALE 10, STEP_STEP 2), :93-134.
    // skimming_cfg 의 min 은 원본 0 → 칸 −1: Clean Skim / Timed flip 노드가 넘기는 −1(:204-282)을 칸 하나로 대신한다.
    name: 'SkimSection', component: SkimSection, source: skimSource,
    show: { _skim_enabled: 'true' },
    inputs: {
      _skim_skimming_cfg: [-1, 10, 0.5],
      _skim_start_percent: [0, 1, 0.01],
      _skim_end_percent: [0, 1, 0.01],
      _skim_flip_at: [0, 1, 0.01],
    },
  },
]

async function render(component: Component, values: Record<string, string>) {
  const widgets = reactive<Record<string, any>>(values)
  return renderToString(createSSRApp(component, { widgets }))
}

function attrOf(tag: string, name: string): string {
  const m = tag.match(new RegExp(`\\s${name}="([^"]*)"`))
  expect(m, `${name} in ${tag}`).not.toBeNull()
  return m![1]
}

/** 칸마다 다른 표식 값 — 렌더된 <input> 을 그 값으로 찾는다(범위와 상관없이 value 로만 쓰인다). */
function marks(section: OriginSection): Record<string, string> {
  return Object.fromEntries(Object.keys(section.inputs).map((key, i) => [key, `0.0${i + 1}37`]))
}

/** 소스에서 v-model="w.<key>" 칸의 <input> 태그 — 렌더와 따로, 칸이 두 번 묶이지 않았는지 본다. */
function sourceInputs(source: string, key: string): string[] {
  return [...source.matchAll(/<input\b[^>]*>/g)].map(m => m[0]).filter(tag => tag.includes(`v-model="w.${key}"`))
}

function visibleText(html: string): string {
  return html.replace(/<!--[\s\S]*?-->/g, '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ')
}

describe('guidance sections use the original node input ranges', () => {
  for (const section of SECTIONS) {
    it(`${section.name}: min / max / step of every original input`, async () => {
      const values = marks(section)
      const html = await render(section.component, { ...section.show, ...values })
      for (const [key, [min, max, step]] of Object.entries(section.inputs)) {
        const tags = html.match(new RegExp(`<input[^>]*value="${values[key].replace('.', '\\.')}"[^>]*>`, 'g')) ?? []
        expect(tags, key).toHaveLength(1)
        const tag = tags[0]!
        expect(attrOf(tag, 'type'), key).toBe('number')
        expect([attrOf(tag, 'min'), attrOf(tag, 'max'), attrOf(tag, 'step')].map(Number), key)
          .toEqual([min, max, step])
        expect(sourceInputs(section.source, key), key).toHaveLength(1)
      }
    })
  }

  it('every number input of these sections is in the origin table (or listed as extension-only)', () => {
    for (const section of SECTIONS) {
      const bound = [...section.source.matchAll(/<input\b[^>]*type="number"[^>]*>/g)]
        .map(m => m[0].match(/v-model="w\.(\w+)"/)?.[1])
      expect(bound.sort(), section.name)
        .toEqual([...Object.keys(section.inputs), ...(section.hostOnly ?? [])].sort())
    }
  })
})

describe('guidance section labels and notes follow the originals', () => {
  it('DCW: shows the original defaults and the flow-model starting hint', async () => {
    const text = visibleText(await render(DcwSection, { _guid_dcw_enabled: 'true' }))
    expect(text).toContain('원본 기본 0.05')
    expect(text).toContain('원본 기본 0.01')
    expect(text).toMatch(/Flow 계열\(Anima 포함\)/)
  })

  it('CWM: says 0 is standard CFG (the original default)', async () => {
    const text = visibleText(await render(CwmSmcSection, { _guid_cwm_enabled: 'true' }))
    expect(text).toContain('0 = 표준 CFG')
    expect(text).toContain('원본 기본 0 / 0')
  })

  it('RDC: tau 0 means off, and RDC needs DCW (warned only while DCW is off)', async () => {
    const off = visibleText(await render(RdcSection, { _guid_rdc_enabled: 'true', _guid_dcw_enabled: 'false' }))
    expect(off).toContain('0 = 끔')
    expect(off).toContain('tau 0 이면')
    expect(off).toContain('DCW 가 꺼져 있어 RDC 가 적용되지 않습니다')
    const on = visibleText(await render(RdcSection, { _guid_rdc_enabled: 'true', _guid_dcw_enabled: 'true' }))
    expect(on).not.toContain('DCW 가 꺼져 있어')
    expect(on).not.toMatch(/tau 0\.15/)
  })

  it('DAVE: a blank block field means the original 8-18 mask; tau ≤ 0.10 hint', async () => {
    const html = await render(DaveSection, { _guid_dave_enabled: 'true', _guid_dave_blocks: '' })
    const blocks = html.match(/<input[^>]*type="text"[^>]*>/)
    expect(blocks).not.toBeNull()
    expect(attrOf(blocks![0], 'placeholder')).toBe('8-18')
    const text = visibleText(html)
    expect(text).toContain('빈칸 = 8-18')
    expect(text).toContain('≤ 0.10 권장')
    expect(text).toContain('0 = 모든 스텝')
  })

  it("CNS: default 2.0 with the README's Anima+cfg_pp 3.0 recommendation (not 'Anima 3.0' as the default)", async () => {
    const text = visibleText(await render(CnsSection, { _guid_cns_enabled: 'true' }))
    expect(text).toContain('CNS gamma scale (기본 2.0 · Anima+cfg_pp 권장 3.0)')
    expect(text).not.toContain('(Anima 3.0)')
    expect(text).toContain('ODE 샘플러')
  })

  it('CNS / RDC tooltips state facts only (range, original default, direction), not the GPL tooltip wording', async () => {
    const titles = (html: string) => [...html.matchAll(/\stitle="([^"]*)"/g)].map(m => m[1]).join(' | ')
    const cns = titles(await render(CnsSection, { _guid_cns_enabled: 'true' }))
    for (const fact of ['원본 기본 1.0', '범위 0~1 step 0.05', '원본 기본 0.5', '범위 0.1~2 step 0.05',
      '원본 기본 2.0', '범위 0.1~25 step 0.1']) expect(cns).toContain(fact)
    const rdc = titles(await render(RdcSection, { _guid_rdc_enabled: 'true' }))
    expect(rdc).toContain('0 = RDC 끔(원본 기본)')
    expect(rdc).toContain('범위 0~0.5 step 0.01')
    // 원본 툴팁 문장의 번역이던 조각(권장 구간 · 37% 감쇠 · '잡티가 보일 때만')이 돌아오지 않게
    for (const phrase of ['0.75–1.0', '0.25–0.4', '2–5', '잡티', '37%', '무관']) {
      expect(cns + rdc, phrase).not.toContain(phrase)
    }
  })

  it("Skimmed: percents are σ-based; Anima's first step is kept only on a denoise 1 pass from Start 0", async () => {
    const text = visibleText(await render(SkimSection, { _skim_enabled: 'true' }))
    for (const label of ['Start at (σ 기준 %)', 'End at (σ 기준 %)', 'Flip at (σ 기준 %) · 0 = 사용 안 함']) {
      expect(text).toContain(label)
    }
    expect(text).toContain('스텝 수가 아니라 노이즈 스케줄(σ)')
    // origin: Extraltodeus/Skimmed_CFG@d8300583:skimmed_CFG.py:153, :167-172 — start_σ = percent_to_sigma(0) = 1.0
    // (flow), σ >= start_σ 면 그대로 둔다. denoise < 1 의 잘린 스케줄은 첫 σ < 1.0 이라 첫 스텝도 깎인다.
    expect(text).toContain('Start 0 · denoise 1')
    expect(text).toContain('첫 스텝은 깎지 않습니다')
    expect(text).toContain('denoise 가 1 보다 작은 패스는 첫 σ 가 그보다 작아 첫 스텝부터 깎습니다')
  })

  it('keeps the number inputs of each section hidden while its switch is off', async () => {
    for (const section of SECTIONS) {
      const html = await render(section.component, marks(section))
      expect(html.match(/<input[^>]*type="number"[^>]*>/g) ?? [], section.name).toHaveLength(0)
    }
  })
})
