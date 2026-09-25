import { describe, expect, it, vi } from 'vitest'
import { createSSRApp, ref } from 'vue'
import { renderToString } from '@vue/server-renderer'
import TagBlockField from './TagBlockField.vue'
import { splitExcludeRules } from '../utils/excludeRules'

// 블록 경계만 본다 — 자동완성(브리지)은 대역으로
vi.mock('../composables/useTagAutocomplete', () => ({
  useTagAutocomplete: () => ({
    items: ref([]), index: ref(-1), queryHangul: ref(false),
    close: () => {}, request: () => {}, move: () => {}, selected: () => undefined,
  }),
}))

async function blockTexts(props: Record<string, unknown>): Promise<string[]> {
  const app = createSSRApp(TagBlockField, props)
  app.component('Icon', { render: () => null })   // 앱에서는 전역 등록 — 여기선 빈 대역
  const html = await renderToString(app)
  return [...html.matchAll(/<span class="tbf-text"[^>]*>([\s\S]*?)<\/span>/g)].map(m => m[1])
}

/**
 * 제외 규칙 칸(블록 모드)은 적용 쪽(core/exclude_rules)과 같은 경계로 블록을 나눠야 한다 —
 * 줄바꿈으로 나눈 규칙이 블록 하나로 뭉쳐 보이면 개수 배지·관리 창 목록과 어긋난다.
 */
describe('TagBlockField split prop', () => {
  it('without split: prompt tag rules (commas only, bracket commas protected)', async () => {
    expect(await blockTexts({ modelValue: 'a, {b, c}, d' })).toEqual(['a', '{b, c}', 'd'])
    expect(await blockTexts({ modelValue: 'a\nb, c' })).toEqual(['a\nb', 'c'])
  })

  it('with the exclude splitter: same blocks as the applied rules', async () => {
    const text = '_short\n~_tank_top, long hair,, *blue_hair'
    const blocks = await blockTexts({ modelValue: text, split: splitExcludeRules })
    expect(blocks).toEqual(['_short', '~_tank_top', 'long hair', '*blue_hair'])
    expect(blocks).toEqual(splitExcludeRules(text))
  })
})
