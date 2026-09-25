import { describe, expect, it } from 'vitest'
import { createSSRApp, reactive } from 'vue'
import { renderToString } from '@vue/server-renderer'
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

describe('PagSection', () => {
  it('lets Attn Scale use the original node range 0..100', async () => {
    // 고유한 값으로 칸을 찾는다(SLG scale 은 별도 칸이라 값이 다르다)
    const html = await render({ _guid_enabled: 'true', _guid_scale: '42.5', _guid_slg_on: 'true', _guid_slg_scale: '3' })
    const scale = inputWithValue(html, '42.5')
    expect({ min: attrOf(scale, 'min'), max: attrOf(scale, 'max'), step: attrOf(scale, 'step') }).toEqual(ORIGIN_SCALE)
    // SLG 는 원본 노드에 없는 확장 기능이라 범위를 그대로 둔다
    expect(attrOf(inputWithValue(html, '3'), 'max')).toBe('15')
  })
})
