import { describe, expect, it } from 'vitest'
import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import PromptSections from './PromptSections.vue'

const slots = { main: () => h('i', 'main'), prefix: () => h('i', 'prefix'), suffix: () => h('i', 'suffix') }

async function textOrder(order: string[]) {
  const html = await renderToString(createSSRApp({ render: () => h(PromptSections, { order } as any, slots) }))
  return [...html.matchAll(/<i>(\w+)<\/i>/g)].map((match) => match[1])
}

describe('PromptSections', () => {
  it('draws the slots in the given order', async () => {
    expect(await textOrder(['prefix', 'main', 'suffix'])).toEqual(['prefix', 'main', 'suffix'])
    expect(await textOrder(['suffix', 'prefix', 'main'])).toEqual(['suffix', 'prefix', 'main'])
  })
})
