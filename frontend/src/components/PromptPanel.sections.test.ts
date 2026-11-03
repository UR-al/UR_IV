import { describe, expect, it } from 'vitest'
import source from './PromptPanel.vue?raw'
import appSource from '../App.vue?raw'
import modalSource from './managers/PromptOrderModal.vue?raw'
import { DEFAULT_PROMPT_SECTIONS } from '../utils/promptSections'

/**
 * 입력 칸이 프롬프트 섹션 순서를 따르는지 — 칸이 `<PromptSections>` 슬롯 밖에 남으면 저장한 순서와
 * 상관없이 제자리에 있고, 슬롯 이름이 어긋나면 그 칸이 조용히 사라진다.
 */
const template = source.slice(0, source.indexOf('<script')).replace(/<!--[\s\S]*?-->/g, '')
const list = template.slice(template.indexOf('<PromptSections'), template.indexOf('</PromptSections>'))

describe('the prompt panel follows the section order', () => {
  it('has one slot per section', () => {
    for (const key of DEFAULT_PROMPT_SECTIONS) expect(list.split(`<template #${key}>`)).toHaveLength(2)
  })

  it('keeps every section box inside the list', () => {
    for (const widget of ['char_count_input', 'character_input', 'copyright_input', 'artist_input', 'prefix_prompt_text', 'main_prompt_text', 'suffix_prompt_text']) {
      const outside = template.replace(list, '')
      expect(outside).not.toContain(`v-model="widgets.${widget}"`)
      expect(list).toContain(`v-model="widgets.${widget}"`)
    }
  })

  it('names the two boxes 선행 고정 and 후행 고정', () => {
    expect(list).toContain('<label>선행 고정 ')
    expect(list).toContain('<label>후행 고정 ')
    expect(list).not.toMatch(/<label>접[두미]/)
  })

  it('asks for the saved order when the app starts, after its listener', () => {
    const bound = appSource.indexOf('bindPromptOrder()')
    const asked = appSource.indexOf('loadPromptOrder()')
    expect(bound).toBeGreaterThan(0)
    expect(asked).toBeGreaterThan(bound)
  })

  it('tells that the order moves the input boxes too', () => {
    expect(modalSource).toContain('입력 칸')
  })
})
