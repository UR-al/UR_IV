import { describe, expect, it } from 'vitest'
import { vramLevel, vramTooltipText } from './vramStatus'

describe('vramLevel', () => {
  it('uses strict 70/90 thresholds', () => {
    expect(vramLevel(0)).toBe('ok')
    expect(vramLevel(70)).toBe('ok')
    expect(vramLevel(70.1)).toBe('warn')
    expect(vramLevel(90)).toBe('warn')
    expect(vramLevel(90.1)).toBe('critical')
  })
})

describe('vramTooltipText', () => {
  it('is empty until the total is known', () => {
    expect(vramTooltipText({ used: 0, total: 0, pct: 0, source: '' })).toBe('')
  })

  it('describes whole-GPU numbers with the free amount and the click hint', () => {
    expect(vramTooltipText({ used: 6.5, total: 16, pct: 40, source: 'nvml' })).toBe(
      '사용: 6.5GB / 전체: 16GB (여유: 9.5GB)\nGPU 전체 (모든 프로세스 · 5초마다 갱신)\n\n클릭하여 백엔드 모델 unload 요청')
  })

  it('says when only the backend memory is counted and warns by level', () => {
    const warn = vramTooltipText({ used: 12, total: 16, pct: 75, source: 'backend' })
    expect(warn).toContain('백엔드가 잡은 메모리만')
    expect(warn).toContain('▲ 70% 초과')
    const critical = vramTooltipText({ used: 15.5, total: 16, pct: 97, source: 'nvml' })
    expect(critical).toContain('⚠ VRAM 부족')
    expect(critical).not.toContain('▲ 70% 초과')
  })
})
