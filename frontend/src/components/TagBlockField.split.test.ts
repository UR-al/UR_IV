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

  it('prompt comments: comment pieces dimmed, comment-free blocks render as before', async () => {
    const app = createSSRApp(TagBlockField, { modelValue: 'a, smile // why\nb, // only\nc /* x' })
    app.component('Icon', { render: () => null })
    const html = await renderToString(app)
    expect(await blockTexts({ modelValue: 'a, smile // why\nb' })).toEqual(['a', 'b'])
    // 태그에 붙은 주석 조각만 흐리게(tbf-cmt) — 태그 글자는 그대로
    expect(html).toMatch(/<span[^>]*>smile <\/span><span class="tbf-cmt" title="주석 — 생성에 보내지 않음"[^>]*>\/\/ why<\/span>/)
    // 주석만 있는 블록은 블록째 흐리게(comment)
    expect(html).toMatch(/class="comment tbf-block" title="주석 — 생성에 보내지 않음"[^>]*><!---->[^]*?\/\/ only/)
    // 닫히지 않은 /* 는 open 표시와 안내
    expect(html).toMatch(/class="open tbf-block" title="닫히지 않은 \/\* — 이 줄 끝까지만 주석"/)
    expect(html).toMatch(/<span class="tbf-cmt open"[^>]*>\/\* x<\/span>/)
  })

  it('with the exclude splitter: same blocks as the applied rules', async () => {
    const text = '_short\n~_tank_top, long hair,, *blue_hair'
    const blocks = await blockTexts({ modelValue: text, split: splitExcludeRules })
    expect(blocks).toEqual(['_short', '~_tank_top', 'long hair', '*blue_hair'])
    expect(blocks).toEqual(splitExcludeRules(text))
  })
})
