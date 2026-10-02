import { describe, expect, it } from 'vitest'
import type { SamExtraCapabilitiesEvent } from '../types/bridge'
import { PAG_DETAIL_SUITE_FROM, detailSuiteNote } from './guidanceDetailSuite'

function caps(known: boolean, argc: number | null): SamExtraCapabilitiesEvent {
  return { known, anima_guidance_argc: argc } as SamExtraCapabilitiesEvent
}

describe('detailSuiteNote', () => {
  it('warns when the connected sam-extra has no detail-suite arguments (62-arg build)', () => {
    const note = detailSuiteNote(caps(true, PAG_DETAIL_SUITE_FROM))
    expect(note).toContain('인자 62개 빌드')
    expect(note).toContain('v0.30.0')
    expect(detailSuiteNote(caps(true, 57))).toContain('인자 57개 빌드')
  })

  it('stays silent when the build has the arguments, or when nothing is known (ComfyUI, before the check)', () => {
    expect(detailSuiteNote(caps(true, 91))).toBe('')
    expect(detailSuiteNote(caps(true, null))).toBe('')
    expect(detailSuiteNote(caps(false, 62))).toBe('')
    expect(detailSuiteNote(null)).toBe('')
    expect(detailSuiteNote(undefined)).toBe('')
  })
})
