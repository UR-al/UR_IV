import { describe, expect, it } from 'vitest'
import { DEFAULT_PROMPT_SECTIONS, orderedSections } from './promptSections'

describe('orderedSections', () => {
  it('keeps a saved order', () => {
    const saved = ['prefix', 'char_count', 'character', 'copyright', 'artist', 'main', 'suffix']
    expect(orderedSections(saved)).toEqual(saved)
  })

  it('gives the default order when nothing is known yet', () => {
    expect(orderedSections([])).toEqual([...DEFAULT_PROMPT_SECTIONS])
  })

  it('never loses an input box: a missing section goes to the end', () => {
    expect(orderedSections(['suffix', 'main'])).toEqual(['suffix', 'main', 'char_count', 'character', 'copyright', 'artist', 'prefix'])
  })

  it('drops unknown keys and repeats', () => {
    expect(orderedSections(['main', 'bogus', 'main', 5, null])).toEqual(['main', 'char_count', 'character', 'copyright', 'artist', 'prefix', 'suffix'])
  })
})
