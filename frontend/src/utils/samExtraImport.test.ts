import { describe, expect, it } from 'vitest'
import { IMPORT_KEYS, IMPORT_LABELS, importSamExtra, importText, importableFeatures } from './samExtraImport'

describe('sam-extra import helper', () => {
  it('finds the card features an image carries', () => {
    expect(importableFeatures(null)).toEqual([])
    expect(importableFeatures({ Steps: 28 })).toEqual([])
    expect(importableFeatures({ 'SAM Extra Colorcraft': 'v1;mods=I' })).toEqual(['colorcraft'])
    expect(importableFeatures({ Colorcraft: 'MOD(I:none)', 'Anima SPEED': 'mode=transition' })).toEqual(['colorcraft', 'speed'])
    expect(importableFeatures({ 'Anima SPEED': null })).toEqual([])
    expect(Object.keys(IMPORT_LABELS)).toEqual(Object.keys(IMPORT_KEYS))
  })

  it('writes one quoted Key: value line per key so commas and semicolons survive', () => {
    const params = { Steps: 28, 'SAM Extra Colorcraft': 'v1;mods=I;I.exposure=0.3', 'Anima SPEED': 'scales=0.5,1; seed=7' }
    expect(importText(params)).toBe('SAM Extra Colorcraft: "v1;mods=I;I.exposure=0.3"\nAnima SPEED: "scales=0.5,1; seed=7"')
    expect(importText(params, ['speed'])).toBe('Anima SPEED: "scales=0.5,1; seed=7"')
  })

  it('sends the action (only when given) and never an empty text', () => {
    const sent: unknown[] = []
    const send = (name: string, payload: unknown) => { sent.push([name, payload]) }
    expect(importSamExtra('  v1;mods=I  ', undefined, send)).toBe(true)
    expect(importSamExtra('x', 'colorcraft', send)).toBe(true)
    expect(importSamExtra('   ', 'speed', send)).toBe(false)
    expect(sent).toEqual([['sam_extra_import', { text: 'v1;mods=I' }],
      ['sam_extra_import', { text: 'x', only: 'colorcraft' }]])
  })
})
