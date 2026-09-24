import { describe, expect, it } from 'vitest'
import {
  GENERATION_FAMILY_ITEMS, familyRestoreAction, isKrea2Family, storedFamilyLabel, toFamilyLabel,
} from './generationFamily'

describe('generation family combo', () => {
  it('uses the exact labels the Python ComboBoxProxy items use', () => {
    // core/generation_family.py GENERATION_FAMILY_ITEMS 와 글자 그대로 같아야 한다.
    expect([...GENERATION_FAMILY_ITEMS]).toEqual(['Standard', 'Krea2'])
  })

  it('maps stored uppercase values and junk to display labels', () => {
    expect(toFamilyLabel('KREA2')).toBe('Krea2')
    expect(toFamilyLabel('krea2')).toBe('Krea2')
    expect(toFamilyLabel('STANDARD')).toBe('Standard')
    expect(toFamilyLabel(null)).toBe('Standard')
    expect(toFamilyLabel('Anima')).toBe('Standard')
  })

  it('detects Krea2 regardless of case', () => {
    expect(isKrea2Family('Krea2')).toBe(true)
    expect(isKrea2Family('KREA2')).toBe(true)
    expect(isKrea2Family(' krea2 ')).toBe(true)
    expect(isKrea2Family('Standard')).toBe(false)
    expect(isKrea2Family(undefined)).toBe(false)
  })

  it('restores the Standard snapshot for a saved label (was dead: label never equalled STANDARD)', () => {
    expect(familyRestoreAction(toFamilyLabel('STANDARD'), 'Standard', true)).toBe('restore-standard')
    expect(familyRestoreAction('Standard', 'Standard', false)).toBe('none')
  })

  it('never resets live Krea2 steps/CFG when Python already shows Krea2 (page refresh)', () => {
    // Python 이 새로고침을 넘어 Krea2 상태를 유지 → steps/CFG 는 사용자가 맞춘 현재값
    expect(familyRestoreAction(toFamilyLabel('KREA2'), 'Krea2', false)).toBe('none')
    expect(familyRestoreAction('Krea2', 'KREA2', true)).toBe('none')
    // 콤보가 Standard 면 라벨 대입이 watch 를 발동시켜 스냅샷·기본값(8/1)을 처리한다
    expect(familyRestoreAction('Krea2', 'Standard', false)).toBe('none')
  })

  it('keeps the Python combo when this browser never stored a family (fresh web client)', () => {
    expect(storedFamilyLabel(null)).toBeNull()
    expect(storedFamilyLabel(undefined)).toBeNull()
    expect(storedFamilyLabel('  ')).toBeNull()
    expect(storedFamilyLabel('KREA2')).toBe('Krea2')
    expect(storedFamilyLabel('STANDARD')).toBe('Standard')
    expect(storedFamilyLabel('Anima')).toBe('Standard')   // 모르는 옛 값은 Standard
  })
})
