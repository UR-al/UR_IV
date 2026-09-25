import { describe, expect, it } from 'vitest'
import { IDLE_I2I_JOB, i2iGenerateLabel, parseI2IJobState } from './i2iJobState'

describe('parseI2IJobState', () => {
  it('Python emit_job_state JSON 을 읽는다', () => {
    expect(parseI2IJobState('{"running": true, "cancelling": false}')).toEqual({ running: true, cancelling: false })
    expect(parseI2IJobState('{"running": true, "cancelling": true}')).toEqual({ running: true, cancelling: true })
    expect(parseI2IJobState({ running: false, cancelling: false })).toEqual(IDLE_I2I_JOB)
  })

  it('끝난 작업은 취소 중일 수 없다', () => {
    expect(parseI2IJobState({ running: false, cancelling: true })).toEqual({ running: false, cancelling: false })
  })

  it('깨진 JSON·running 이 없는 값은 null (상태를 바꾸지 않는다)', () => {
    expect(parseI2IJobState('{')).toBeNull()
    expect(parseI2IJobState(null)).toBeNull()
    expect(parseI2IJobState({ running: 'yes' })).toBeNull()
    expect(parseI2IJobState('[]')).toBeNull()
  })
})

describe('i2iGenerateLabel', () => {
  it('상태별 문구', () => {
    expect(i2iGenerateLabel(IDLE_I2I_JOB, false, false)).toBe('이미지를 먼저 올리세요')
    expect(i2iGenerateLabel(IDLE_I2I_JOB, true, false)).toBe('I2I 생성 시작')
    expect(i2iGenerateLabel(IDLE_I2I_JOB, true, true)).toBe('Krea2 아이덴티티 편집 시작')
    expect(i2iGenerateLabel({ running: true, cancelling: false }, true, false)).toBe('I2I 생성 중…')
    expect(i2iGenerateLabel({ running: true, cancelling: false }, true, true)).toBe('Krea2 아이덴티티 편집 중…')
    expect(i2iGenerateLabel({ running: true, cancelling: true }, true, false)).toBe('취소하는 중…')
  })
})
