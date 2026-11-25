/**
 * 색 계산이 파이썬과 갈라지지 않는지 못박는다.
 *
 * 두 벌이 있는 이유는 `presets.ts` 의 머리말에 있다 — PyQt 시작 다이얼로그(Python)와
 * Vue 화면이 **같은 색**이어야 한다. 계산이 조금이라도 다르면 앱이 두 개처럼 보인다.
 * golden 값은 `core/theme_presets.py` 가 직접 만들어 `__golden__.json` 에 넣는다
 * (`core/curves.py` ↔ `utils/curves.ts` 와 같은 방식).
 */
import { describe, it, expect } from 'vitest'
import golden from './__golden__.json'
import { PRESETS, EDITABLE_KEYS } from './presets'
import {
  deriveAccent,
  contrastRatio,
  shiftLightness,
  resolveTheme,
  isValidHex,
  normalizeHex,
} from './applyTheme'

describe('파이썬과 같은 결과', () => {
  it('deriveAccent', () => {
    for (const [key, expected] of Object.entries(golden.derive)) {
      const [color, mode] = key.split('|')
      expect(deriveAccent(color, mode), key).toEqual(expected)
    }
  })

  it('contrastRatio', () => {
    for (const [key, expected] of Object.entries(golden.contrast)) {
      const [a, b] = key.split('|')
      expect(contrastRatio(a, b), key).toBeCloseTo(expected as number, 6)
    }
  })

  it('shiftLightness', () => {
    for (const [key, expected] of Object.entries(golden.shift)) {
      const [color, delta] = key.split('|')
      expect(shiftLightness(color, Number(delta)), key).toBe(expected)
    }
  })

  it('resolveTheme — 프리셋별 최종 색표', () => {
    for (const [name, expected] of Object.entries(golden.resolve)) {
      expect(resolveTheme(name), name).toEqual(expected)
    }
  })

  it('resolveTheme — 덮어쓰기는 허용된 키만, 잘못된 값은 무시', () => {
    expect(
      resolveTheme('light', { accent: '#2563eb', 'state-ok': 'bad', 'bg-primary': '#FF0000' }),
    ).toEqual(golden.resolve_override)
  })

  it('resolveTheme — 채움색을 바꾸면 글자용 변형이 따라온다', () => {
    expect(resolveTheme('default', { 'state-info': '#1E3A8A', 'state-alert': '#7F1D1D' }))
      .toEqual(golden.resolve_state_override.dark)
    expect(resolveTheme('light', { 'state-ok': '#86EFAC' }))
      .toEqual(golden.resolve_state_override.light)
  })
})

describe('상태색을 바꿔도 문구가 읽힌다', () => {
  it('어떤 채움색을 골라도 그 뜻의 글자색은 배경에서 4.5:1 이상', () => {
    // 배지(채움)만 바뀌고 문구가 프리셋 색으로 남으면 둘이 따로 논다.
    // 파랑 배지 옆 초록 글자 같은 상태가 되지 않는지 색상환을 돌며 확인한다.
    for (const preset of ['default', 'dark', 'light']) {
      for (let h = 0; h < 360; h += 11) {
        for (const l of [18, 45, 82]) {
          const fill = hslHex(h, 65, l)
          const colors = resolveTheme(preset, { 'state-info': fill })
          const ratio = contrastRatio(colors['state-info-fg'], colors['bg-primary'])
          expect(ratio, `${preset} / 채움 ${fill} → 글자 ${colors['state-info-fg']}`)
            .toBeGreaterThanOrEqual(4.5)
        }
      }
    }
  })

  it('이미 읽히는 색은 그대로 쓴다 — 괜히 밀지 않는다', () => {
    const colors = resolveTheme('default', { 'state-info': '#8FB8E6' })
    expect(colors['state-info-fg']).toBe('#8FB8E6')
  })

  it('라이트 테마에서 상태색을 덮어써도 bg-button-hover 위에서 4.5:1 이상 유지된다 (U-06)', () => {
    for (const role of ['state-info', 'state-alert', 'state-ok', 'state-warn'] as const) {
      for (const fill of ['#86EFAC', '#93C5FD', '#FCA5A5', '#FDE047', '#2563EB']) {
        const colors = resolveTheme('light', { [role]: fill })
        const fg = colors[`${role}-fg`]
        expect(contrastRatio(fg, colors['bg-button-hover']), `light ${role}=${fill} → ${fg}`)
          .toBeGreaterThanOrEqual(4.5)
        expect(contrastRatio(fg, colors['bg-primary']), `light ${role}=${fill} → ${fg} on primary`)
          .toBeGreaterThanOrEqual(4.5)
      }
    }
  })
})

describe('사용자가 아무 색이나 골라도 읽을 수 있어야 한다', () => {
  it('on-accent 는 강조색 밝기에서 정해진다', () => {
    // 밝은 노랑 위에는 검은 글자, 짙은 남색 위에는 흰 글자
    expect(deriveAccent('#FACC15', 'dark')['on-accent']).toBe('#0A0A0A')
    expect(deriveAccent('#1E3A8A', 'dark')['on-accent']).toBe('#FFFFFF')
  })

  it('어떤 강조색이든 주 버튼 글자는 4.5:1 이상', () => {
    // 색상환을 한 바퀴 돌며 명도·채도를 바꿔 가며 확인.
    // 기준은 `accent-fill` — 사용자의 색 그대로는 보장할 수 없다(중간 밝기 색은
    // 흰 글자도 검은 글자도 4.4 대다). 그래서 글자를 얹는 면만 민다.
    for (let h = 0; h < 360; h += 7) {
      for (const s of [30, 60, 100]) {
        for (const l of [20, 40, 55, 70, 88]) {
          const hex = hslHex(h, s, l)
          const derived = deriveAccent(hex, 'dark')
          const ratio = contrastRatio(derived['on-accent'], derived['accent-fill'])
          expect(ratio, `${hex} → 면 ${derived['accent-fill']} 위 ${derived['on-accent']}`)
            .toBeGreaterThanOrEqual(4.5)
        }
      }
    }
  })

  it('보통은 면을 밀지 않는다 — 사용자가 고른 색 그대로', () => {
    for (const accent of ['#C9A227', '#FACC15', '#2563EB', '#1E3A8A']) {
      const derived = deriveAccent(accent, 'dark')
      expect(derived['accent-fill'], accent).toBe(normalizeHex(accent))
    }
  })
})

describe('입력 방어', () => {
  it('isValidHex', () => {
    expect(isValidHex('#abc')).toBe(true)
    expect(isValidHex('123456')).toBe(true)
    expect(isValidHex('#12345')).toBe(false)
    expect(isValidHex('rgb(1,2,3)')).toBe(false)
    expect(isValidHex(null)).toBe(false)
    expect(isValidHex(undefined)).toBe(false)
  })

  it('normalizeHex — 3자리를 6자리로, 대문자로', () => {
    expect(normalizeHex('#abc')).toBe('#AABBCC')
    expect(normalizeHex('c9a227')).toBe('#C9A227')
  })

  it('없는 프리셋이면 기본으로 떨어진다', () => {
    expect(resolveTheme('없는테마')).toEqual(resolveTheme('default'))
  })
})

describe('프리셋 자체의 대비', () => {
  it('본문·라벨·경계가 배경에서 읽힌다', () => {
    for (const [name, colors] of Object.entries(PRESETS)) {
      const bg = colors['bg-primary']
      expect(contrastRatio(colors['text-primary'], bg), `${name} text-primary`).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(colors['text-secondary'], bg), `${name} text-secondary`).toBeGreaterThanOrEqual(4.5)
      expect(contrastRatio(colors['text-muted'], bg), `${name} text-muted`).toBeGreaterThanOrEqual(3)
      expect(contrastRatio(colors['edge'], bg), `${name} edge`).toBeGreaterThanOrEqual(3)
    }
  })

  it('상태색은 글자용이 배경에서, 채움용이 흰 글자를 받친다', () => {
    for (const [name, colors] of Object.entries(PRESETS)) {
      const bg = colors['bg-primary']
      for (const role of ['info', 'alert', 'ok', 'warn']) {
        expect(contrastRatio(colors[`state-${role}-fg`], bg), `${name} state-${role}-fg`)
          .toBeGreaterThanOrEqual(4.5)
        expect(contrastRatio(colors[`state-${role}`], '#FFFFFF'), `${name} state-${role} 채움`)
          .toBeGreaterThanOrEqual(4.5)
      }
    }
  })

  it('태그 7색이 배경에서 읽힌다', () => {
    for (const [name, colors] of Object.entries(PRESETS)) {
      const bg = colors['bg-primary']
      for (const tag of ['person', 'scene', 'pose', 'wear', 'fx', 'nsfw', 'neutral']) {
        expect(contrastRatio(colors[`tag-${tag}`], bg), `${name} tag-${tag}`).toBeGreaterThanOrEqual(4.5)
      }
    }
  })

  it('바꿀 수 있는 색은 모든 프리셋에 있다', () => {
    for (const [name, colors] of Object.entries(PRESETS)) {
      for (const key of EDITABLE_KEYS) {
        expect(colors[key], `${name}.${key}`).toBeTruthy()
      }
    }
  })
})

/** 테스트용 HSL→hex (colorsys 와 같은 정의). */
function hslHex(h: number, s: number, l: number): string {
  const sN = s / 100
  const lN = l / 100
  const c = (1 - Math.abs(2 * lN - 1)) * sN
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = lN - c / 2
  const t: number[] =
    h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
      : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x]
  return (
    '#' +
    t.map((v) => Math.round((v + m) * 255).toString(16).padStart(2, '0')).join('').toUpperCase()
  )
}
