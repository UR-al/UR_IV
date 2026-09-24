import { describe, expect, it } from 'vitest'
import { createExifWarningNotice } from './exifWarningNotice'

describe('createExifWarningNotice', () => {
  it('announces only the first warning of a run and summarises the rest', () => {
    const notice = createExifWarningNotice('SAM3')
    expect(notice.note({ before: 'C:/a/ok.png' })).toBeNull()
    expect(notice.note({ before: 'C:/a/x.jpg', exif_warning: '메타 없음' })).toBe('SAM3 · x.jpg: 메타 없음')
    expect(notice.note({ before: 'C:/a/y.jpg', exif_warning: '메타 없음' })).toBeNull()
    expect(notice.count).toBe(2)
    expect(notice.summary()).toContain('2장')
  })

  it('reset starts a fresh run and a single warning needs no summary', () => {
    const notice = createExifWarningNotice('ADetailer')
    notice.note({ exif_warning: 'w' })
    notice.reset()
    expect(notice.note({ path: 'b.png', exif_warning: 'w2' })).toBe('ADetailer · b.png: w2')
    expect(notice.summary()).toBeNull()
  })
})
