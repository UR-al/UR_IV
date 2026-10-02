import { describe, expect, it, vi } from 'vitest'
import { createSSRApp, reactive, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import type { SamExtraCapabilitiesEvent } from '../../types/bridge'

const caps = ref<Partial<SamExtraCapabilitiesEvent> | null>(null)
vi.mock('../../composables/useSamExtraCapabilities', () => ({
  useSamExtraCapabilities: () => ({ capabilities: caps, refresh: () => {}, mayUse: () => true }),
}))

import PagSection from './PagSection.vue'

/**
 * PAG / SEG Attn Scale 칸 = 원본 노드 scale 입력 범위 0~100, step 0.1.
 * 확장(scripts/anima_safe_pag.py)의 슬라이더·클램프와 앱 스펙(core/anima_guidance.py guid_scale)도 같은 범위다.
 *
 * origin: iljung1106/comfyui-anima-safe-pag@905b0107:__init__.py:201 (scale: default 4.0, min 0, max 100, step 0.1)
 */
const ORIGIN_SCALE = { min: '0', max: '100', step: '0.1' }

async function render(values: Record<string, string>) {
  const widgets = reactive<Record<string, any>>(values)
  return renderToString(createSSRApp(PagSection, { widgets }))
}

function inputWithValue(html: string, value: string): string {
  const tag = html.match(new RegExp(`<input[^>]*value="${value.replace('.', '\\.')}"[^>]*>`))
  expect(tag, value).not.toBeNull()
  return tag![0]
}

function attrOf(tag: string, name: string): string {
  const m = tag.match(new RegExp(`\\s${name}="([^"]*)"`))
  expect(m, `${name} in ${tag}`).not.toBeNull()
  return m![1]
}

function boundInputs(html: string): string[] {
  return [...html.matchAll(/<input[^>]*>/g)].map(m => m[0])
}

describe('PagSection', () => {
  it('lets Attn Scale use the original node range 0..100', async () => {
    // 고유한 값으로 칸을 찾는다(SLG scale 은 별도 칸이라 값이 다르다)
    const html = await render({ _guid_enabled: 'true', _guid_scale: '42.5', _guid_slg_on: 'true', _guid_slg_scale: '3' })
    const scale = inputWithValue(html, '42.5')
    expect({ min: attrOf(scale, 'min'), max: attrOf(scale, 'max'), step: attrOf(scale, 'step') }).toEqual(ORIGIN_SCALE)
    // SLG 는 원본 노드에 없는 확장 기능이라 범위를 그대로 둔다
    expect(attrOf(inputWithValue(html, '3'), 'max')).toBe('15')
  })

  it('swaps the fixed SLG fields for the S² fields when SLG mode is Stochastic (S²)', async () => {
    const base = { _guid_enabled: 'true', _guid_slg_on: 'true', _guid_slg_scale: '3.3', _guid_s2_scale: '0.37' }
    const fixed = await render(base)
    expect(fixed).toContain('SLG mode')
    expect(boundInputs(fixed).some(tag => tag.includes('value="3.3"'))).toBe(true)
    expect(fixed).not.toContain('S² scale ω')

    const s2 = await render({ ...base, _guid_slg_mode: 'Stochastic (S²)' })
    expect(s2).toContain('S² scale ω')
    expect(s2).toContain('S² eligible blocks (빈칸 = 1~마지막 · 블록 0 제외)')
    expect(boundInputs(s2).some(tag => tag.includes('value="3.3"'))).toBe(false)
    expect(attrOf(inputWithValue(s2, '0.37'), 'max')).toBe('5')
  })

  it('keeps SLG mode and S² hidden while SLG is off', async () => {
    const html = await render({ _guid_enabled: 'true', _guid_slg_mode: 'Stochastic (S²)' })
    expect(html).not.toContain('SLG mode')
    expect(html).not.toContain('S² scale ω')
  })

  it('tells when the connected sam-extra does not know S² (62-arg build), and only in S² mode', async () => {
    const values = { _guid_enabled: 'true', _guid_slg_on: 'true', _guid_slg_mode: 'Stochastic (S²)' }
    caps.value = { known: true, anima_guidance_argc: 62 }
    expect(await render(values)).toContain('인자 62개 빌드')
    expect(await render({ ...values, _guid_slg_mode: 'Fixed' })).not.toContain('인자 62개 빌드')
    caps.value = { known: true, anima_guidance_argc: 91 }
    expect(await render(values)).not.toContain('인자 62개 빌드')
    caps.value = null
  })
})
