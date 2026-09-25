import { describe, expect, it, vi } from 'vitest'
import { createSSRApp, reactive, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import panelSource from '../AnimaGuidancePanel.vue?raw'
import { useGuidanceWidgets } from './guidanceWidgets'

vi.mock('../../composables/useSamExtraCapabilities', () => ({
  useSamExtraCapabilities: () => ({ capabilities: ref(null), refresh: () => {}, mayUse: () => true }),
}))
vi.mock('../../stores/widgetStore.js', () => ({ requestAction: vi.fn() }))

import AnimaGuidancePanel from '../AnimaGuidancePanel.vue'

/**
 * AnimaGuidancePanel 분할(P0-B) — 기능별 칸은 components/guidance/*Section.vue 에 있고, 패널은 카드 ·
 * 그룹 아코디언 · 요약 배지 · 버튼만 든다. 섹션은 감싸는 요소 없는 조각이라 패널의 그룹 <details> 안에
 * 그대로 놓인다(분할 전과 같은 DOM).
 */
const SECTIONS = import.meta.glob<string>('./*Section.vue', { query: '?raw', import: 'default', eager: true })

function sectionName(path: string): string {
  return path.replace(/^\.\//, '').replace(/\.vue$/, '')
}

function template(source: string): string {
  return source.slice(source.indexOf('<template>'), source.lastIndexOf('</template>'))
}

describe('useGuidanceWidgets', () => {
  it("reads and writes toggles as the store's 'true'/'false' strings on the same object", () => {
    const widgets = reactive<Record<string, any>>({ _a: 'true', _b: 'false', _c: true, _d: '' })
    const { w, b, setB } = useGuidanceWidgets({ widgets })
    expect(w).toBe(widgets)
    expect([b('a'), b('b'), b('c'), b('d'), b('missing')]).toEqual([true, false, true, false, false])
    setB('b', true)
    setB('a', false)
    expect(widgets._b).toBe('true')
    expect(widgets._a).toBe('false')
  })
})

describe('AnimaGuidancePanel sections', () => {
  it('places every section in the panel exactly once, bound to the same widget store', () => {
    const names = Object.keys(SECTIONS).map(sectionName)
    expect(names.length).toBeGreaterThanOrEqual(8)
    for (const name of names) {
      expect(panelSource).toContain(`import ${name} from './guidance/${name}.vue'`)
      const uses = template(panelSource).match(new RegExp(`<${name}\\b[^>]*>`, 'g')) ?? []
      expect(uses, name).toEqual([`<${name} :widgets="widgets" />`])
    }
  })

  it("keeps panel-only scoped rules off section markup (sections get the panel's look through :deep)", () => {
    // 섹션 요소는 패널의 data-v 속성을 받지 않는다 — :deep 이 아닌 패널 규칙의 클래스를 섹션에서 쓰면 모양이 빠진다.
    const style = panelSource.slice(panelSource.indexOf('<style scoped>'))
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/:deep\([^)]*\)/g, '')
    const panelOnly = new Set([...style.matchAll(/\.([a-z][\w-]*)/g)].map(m => m[1]))
    expect(panelOnly.has('ag-group')).toBe(true)
    expect(panelSource).toMatch(/:deep\(\.ag-sub\)/)
    expect(panelSource).toMatch(/:deep\(\.ext-note\)/)
    for (const [path, source] of Object.entries(SECTIONS)) {
      expect(source, path).not.toMatch(/<style\b/)
      const classes = [...template(source).matchAll(/class="([^"]+)"/g)].flatMap(m => m[1].split(/\s+/))
      expect(classes.filter(cls => panelOnly.has(cls)), path).toEqual([])
    }
  })

  it('renders each section inside its group in the original order', async () => {
    const toggles = [
      'guid_enabled', 'guid_slg_on', 'guid_legacy_attn', 'guid_apg_enabled', 'guid_cwm_enabled',
      'guid_smc_master_enabled', 'skim_enabled', 'guid_dcw_enabled', 'guid_rdc_enabled', 'guid_dave_enabled',
      'guid_cns_enabled', 'dd_enabled', 'guid_adg_enabled', 'guid_mod_enabled',
    ]
    const widgets = reactive<Record<string, any>>(Object.fromEntries(toggles.map(key => [`_${key}`, 'true'])))
    const html = await renderToString(createSSRApp(AnimaGuidancePanel, { widgets }))
    const order = [
      'PAG / SEG / SLG — Attention perturbation</summary>', 'Enable Perturbation Guidance', 'Enable SLG',
      'Legacy Soft/Approx 호환',
      'APG / CWM / SMC — CFG base</summary>', 'Enable APG', 'Enable CWM', 'Enable SMC', 'Legacy CFG base 라디오',
      'Skimmed CFG — anti-burn</summary>', 'Enable Skimmed CFG',
      'DCW / RDC / DAVE / CNS</summary>', 'Enable DCW', 'Enable RDC', 'Enable DAVE', 'Enable CNS',
      'Detail Daemon</summary>', 'Enable Detail Daemon',
      'Adaptive Guidance / CLIP Modulation</summary>', 'Enable Adaptive Guidance', 'Enable Anima Modulation Guidance',
      'Forge에서 가져오기',
    ]
    const positions = order.map(text => html.indexOf(text))
    order.forEach((text, i) => expect(positions[i], text).toBeGreaterThanOrEqual(0))
    expect(positions).toEqual([...positions].sort((a, b) => a - b))
    expect(html.split('<details class="ag-group"')).toHaveLength(7)
  })
})

describe('AnimaGuidancePanel summary badge', () => {
  async function badge(values: Record<string, string>): Promise<string[]> {
    const widgets = reactive<Record<string, any>>(values)
    const html = await renderToString(createSSRApp(AnimaGuidancePanel, { widgets }))
    const text = html.match(/class="ag-badge"[^>]*>([^<]*)</)?.[1] ?? ''
    return text ? text.split(' · ') : []
  }

  it('shows RDC only when the backend sends it on: switch + DCW + tau > 0 (core/anima_guidance.py _rdc_on)', async () => {
    // origin: namemechan/ComfyUI-DCW@66aaf9dd:dcw_node.py:757-760 — rdc_tau 0 = 끔(기본), RDC 는 DCW 훅 안에서만 돈다
    const on = { _guid_rdc_enabled: 'true', _guid_dcw_enabled: 'true', _guid_rdc_tau: '0.1' }
    expect(await badge(on)).toEqual(['DCW', 'RDC'])
    expect(await badge({ ...on, _guid_rdc_tau: '0' })).toEqual(['DCW'])
    expect(await badge({ ...on, _guid_rdc_tau: '0.0' })).toEqual(['DCW'])
    expect(await badge({ ...on, _guid_rdc_tau: '' })).toEqual(['DCW'])
    expect(await badge({ ...on, _guid_dcw_enabled: 'false' })).toEqual([])
    expect(await badge({ ...on, _guid_rdc_enabled: 'false' })).toEqual(['DCW'])
    // 다른 칸의 순서는 그대로: CWM · DCW · RDC · DAVE
    expect(await badge({ ...on, _guid_cwm_enabled: 'true', _guid_dave_enabled: 'true' }))
      .toEqual(['CWM', 'DCW', 'RDC', 'DAVE'])
  })
})
