/**
 * 테마 적용 — 프리셋 + 사용자 덮어쓰기 → `:root` 의 CSS 커스텀 속성.
 *
 * **첫 페인트 전에 넣어야 한다.** 영속 설정(`config/ui_prefs.json`)은 브리지로
 * `uiPrefsLoaded` 가 도착해야 읽히는데 그건 Vue 가 뜬 뒤다. 그때 색을 넣으면
 * 기본 테마가 한 번 번쩍인 뒤 바뀐다. 그래서 **localStorage 를 부팅 캐시로** 쓰고
 * (동기 읽기라 페인트 전에 끝난다), 나중에 도착한 영속 설정으로 다시 맞춘다.
 * 이 앱이 `ollamaUrl`·`tabOrder`·`autoNlGen` 에 이미 쓰는 방식과 같다.
 *
 * 색 계산은 `core/theme_presets.py` 와 **같은 식**이어야 한다 — PyQt 시작
 * 다이얼로그와 Vue 가 다른 색을 쓰면 앱이 두 개처럼 보인다.
 * `presets.test.ts` 가 파이썬과 같은 golden 값으로 못박는다.
 */

import { PRESETS, DEFAULT_PRESET, EDITABLE_KEYS, type ThemeColors } from './presets'

const STORAGE_PRESET = 'theme.preset'
const STORAGE_OVERRIDES = 'theme.overrides'

/** 색이 아닌 메타데이터 — CSS 변수로 내보내지 않는다. */
const META_KEYS = new Set(['mode', 'label'])

export interface ThemeState {
  preset: string
  overrides: Record<string, string>
}

// ── 색 계산 (theme_presets.py 의 대응 함수와 같은 결과여야 한다) ──────────

export function isValidHex(value: unknown): value is string {
  if (typeof value !== 'string') return false
  const h = value.trim().replace(/^#/, '')
  return (h.length === 3 || h.length === 6) && /^[0-9a-fA-F]+$/.test(h)
}

export function normalizeHex(value: string): string {
  let h = value.trim().replace(/^#/, '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  return '#' + h.toUpperCase()
}

function hexToRgb(value: string): [number, number, number] {
  const h = normalizeHex(value).slice(1)
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as [number, number, number]
}

function rgbToHex(r: number, g: number, b: number): string {
  const to = (c: number) => Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16).padStart(2, '0')
  return ('#' + to(r) + to(g) + to(b)).toUpperCase()
}

export function relativeLuminance(value: string): number {
  const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))
  const [r, g, b] = hexToRgb(value).map(lin)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

/** WCAG 대비비. 1.0(같은 색) ~ 21.0(검정↔흰색). */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a)
  const lb = relativeLuminance(b)
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05)
}

/** colorsys.rgb_to_hls / hls_to_rgb 와 같은 정의(HLS, H 는 0~1). */
function rgbToHls(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b)
  const min = Math.min(r, g, b)
  const l = (max + min) / 2
  if (max === min) return [0, l, 0]
  const d = max - min
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
  let h: number
  if (max === r) h = ((g - b) / d) % 6
  else if (max === g) h = (b - r) / d + 2
  else h = (r - g) / d + 4
  h /= 6
  if (h < 0) h += 1
  return [h, l, s]
}

function hlsToRgb(h: number, l: number, s: number): [number, number, number] {
  if (s === 0) return [l, l, l]
  const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s
  const m1 = 2 * l - m2
  const hue = (t: number): number => {
    let x = t % 1
    if (x < 0) x += 1
    if (x < 1 / 6) return m1 + (m2 - m1) * 6 * x
    if (x < 0.5) return m2
    if (x < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - x) * 6
    return m1
  }
  return [hue(h + 1 / 3), hue(h), hue(h - 1 / 3)]
}

export function shiftLightness(value: string, delta: number): string {
  const [r, g, b] = hexToRgb(value)
  const [h, l, s] = rgbToHls(r, g, b)
  return rgbToHex(...hlsToRgb(h, Math.max(0, Math.min(1, l + delta)), s))
}

function rgba(value: string, alpha: number): string {
  const [r, g, b] = hexToRgb(value).map((c) => Math.round(c * 255))
  return `rgba(${r},${g},${b},${alpha})`
}

/** 글자를 얹는 면이 지켜야 할 최소 대비. */
export const MIN_ON_ACCENT_CONTRAST = 4.5

/**
 * `on` 색 글자가 4.5:1 로 읽힐 때까지 면 색의 명도만 민다.
 *
 * **왜 필요한가**: 중간 밝기 색(예: `#D14747`)은 흰 글자도 검은 글자도 4.5:1 이
 * 안 나온다 — 둘 다 4.4 대다. 강조색을 그대로 버튼 배경에 쓰면 사용자가 그런 색을
 * 고르는 순간 주 버튼 글자가 안 읽힌다. 그래서 **사용자의 색은 그대로 두고**
 * (테두리·표시등은 그 색을 쓴다) 글자를 얹는 면만 필요한 만큼 민다.
 */
export function readableFill(color: string, on: string): string {
  if (contrastRatio(on, color) >= MIN_ON_ACCENT_CONTRAST) return color
  const step = relativeLuminance(on) > 0.5 ? -0.02 : 0.02
  let candidate = color
  for (let i = 0; i < 50; i++) {
    candidate = shiftLightness(candidate, step)
    if (contrastRatio(on, candidate) >= MIN_ON_ACCENT_CONTRAST) return candidate
  }
  return candidate
}

/**
 * 강조색 하나에서 파생색을 만든다.
 *
 * 사용자가 아무 색이나 고를 수 있으므로 `on-accent`(주 버튼의 글자색)를 **밝기에서
 * 계산**한다. 흰색으로 고정하면 노란 버튼 위 흰 글자처럼 안 읽히는 조합이 나온다.
 * 그래도 모자라면 `accent-fill` 이 면을 밀어 준다.
 */
/**
 * `background` 위에서 읽히도록 색의 명도만 민다 — 색상·채도는 그대로.
 *
 * **왜 필요한가**: 사용자가 '선택'·'알림'·'연결' 색을 바꾸면 배지(채움)만 바뀌고
 * **같은 뜻의 글자는 프리셋 값 그대로** 남는다. 파란 배지 옆에 초록 글자가 남는 식이라
 * 편집이 반쪽이 된다. 그래서 채움색을 바꾸면 글자용 변형(`state-x-fg`)을 여기서 만든다.
 */
export function readableText(
  color: string,
  background: string,
  minimum = MIN_ON_ACCENT_CONTRAST,
): string {
  if (contrastRatio(color, background) >= minimum) return color
  const step = relativeLuminance(background) < 0.5 ? 0.02 : -0.02
  let candidate = color
  for (let i = 0; i < 60; i++) {
    candidate = shiftLightness(candidate, step)
    if (contrastRatio(candidate, background) >= minimum) return candidate
  }
  return candidate
}

/**
 * 글자를 얹는 면의 호버색 — **글자색에서 멀어지는** 쪽으로 민다.
 *
 * **왜 방향을 글자에서 정하는가**: 예전에는 다크 테마면 무조건 밝게 밀었다.
 * 그런데 사용자가 중간 밝기 색(#D14747 류)을 고르면 `on-accent` 가 흰색이 되는데,
 * 그 면을 더 밝히면 흰 글자가 오히려 안 읽힌다 — `accent-fill` 이 4.5:1 을 맞춰
 * 놔도 **호버에서만 2.9:1 로 떨어졌다**. 면이 지켜야 할 건 테마의 밝기가 아니라
 * 그 위에 얹힌 글자다. 미는 폭만 테마가 정한다.
 */
export function hoverFill(fill: string, on: string, mode: string): string {
  const amount = mode === 'dark' ? 0.10 : 0.08
  const delta = relativeLuminance(on) > 0.5 ? -amount : amount
  const shifted = shiftLightness(fill, delta)
  // 이미 순백/순검이라 그쪽으로 더 못 밀면 반대로 민다 — 호버가 원색과 같으면
  // 버튼이 마우스에 반응하지 않는 것처럼 보인다.
  return shifted !== fill ? shifted : shiftLightness(fill, -delta)
}

export function deriveAccent(accent: string, mode: string): Record<string, string> {
  const hoverDelta = mode === 'dark' ? 0.10 : -0.08
  const on = contrastRatio('#FFFFFF', accent) >= contrastRatio('#0A0A0A', accent)
    ? '#FFFFFF'
    : '#0A0A0A'
  const fill = readableFill(accent, on)
  return {
    accent,
    // 글자를 얹지 않는 쪽(테두리·표시등)이라 테마 밝기대로 민다.
    'accent-hover': shiftLightness(accent, hoverDelta),
    'accent-dim': rgba(accent, mode === 'dark' ? 0.14 : 0.12),
    // 글자를 얹는 면 — 주 버튼 배경. 보통은 accent 와 같다.
    'accent-fill': fill,
    'accent-fill-hover': hoverFill(fill, on, mode),
    'on-accent': on,
  }
}

/** 프리셋 + 덮어쓰기 → 최종 색 표. 잘못된 값은 조용히 무시한다. */
export function resolveTheme(preset?: string, overrides?: Record<string, string>): ThemeColors {
  const name = preset && PRESETS[preset] ? preset : DEFAULT_PRESET
  const colors: ThemeColors = { ...PRESETS[name] }
  for (const key of EDITABLE_KEYS) {
    const value = overrides?.[key]
    if (!isValidHex(value)) continue
    colors[key] = normalizeHex(value)
    // 채움색을 바꿨으면 같은 뜻의 글자색도 따라와야 한다 — 안 그러면
    // 배지만 바뀌고 문구는 프리셋 색으로 남아 둘이 따로 논다.
    if (key.startsWith('state-')) {
      colors[`${key}-fg`] = readableText(colors[key], colors['bg-button-hover'])
    }
  }
  return { ...colors, ...deriveAccent(colors.accent, colors.mode) }
}

// ── 적용 ──────────────────────────────────────────────────────────────────

/**
 * 색 표를 `:root` 에 넣는다.
 *
 * `data-theme-mode` 도 같이 세운다 — 밝기에 따라 갈라져야 하는 규칙(그림자·오버레이
 * 처럼 색 하나로는 안 되는 것)이 `[data-theme-mode="light"]` 로 붙을 수 있게.
 */
export function applyColors(colors: ThemeColors): void {
  const root = document.documentElement
  for (const [key, value] of Object.entries(colors)) {
    if (!META_KEYS.has(key)) root.style.setProperty(`--${key}`, value)
  }
  root.setAttribute('data-theme-mode', colors.mode === 'light' ? 'light' : 'dark')
  root.style.colorScheme = colors.mode === 'light' ? 'light' : 'dark'
}

export function applyTheme(preset?: string, overrides?: Record<string, string>): ThemeColors {
  const colors = resolveTheme(preset, overrides)
  applyColors(colors)
  return colors
}

// ── 영속 ──────────────────────────────────────────────────────────────────

function readStored(): ThemeState {
  let preset = DEFAULT_PRESET
  let overrides: Record<string, string> = {}
  try {
    const saved = window.localStorage.getItem(STORAGE_PRESET)
    if (saved && PRESETS[saved]) preset = saved
    const raw = window.localStorage.getItem(STORAGE_OVERRIDES)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (parsed && typeof parsed === 'object') overrides = parsed
    }
  } catch {
    /* 사생활 모드 등 — 기본값으로 간다 */
  }
  return { preset, overrides }
}

function writeStored(state: ThemeState): void {
  try {
    window.localStorage.setItem(STORAGE_PRESET, state.preset)
    window.localStorage.setItem(STORAGE_OVERRIDES, JSON.stringify(state.overrides))
  } catch {
    /* 무시 — 화면은 이미 바뀌었고, 다음 실행에 기본값으로 뜰 뿐이다 */
  }
}

/** 현재 상태 — 설정 화면이 읽고 쓴다. */
let current: ThemeState = { preset: DEFAULT_PRESET, overrides: {} }

export function getThemeState(): ThemeState {
  return { preset: current.preset, overrides: { ...current.overrides } }
}

/**
 * 부팅 — `main.js` 가 mount 전에 부른다. localStorage 만 읽으므로 동기다.
 */
export function bootTheme(): ThemeColors {
  current = readStored()
  return applyTheme(current.preset, current.overrides)
}

/**
 * 사용자가 설정에서 바꿨을 때. 화면에 즉시 반영하고 저장한다.
 * `persist` 는 영속 저장(ui_prefs.json)으로 보낼 페이로드를 받는 콜백 —
 * 이 모듈이 브리지를 직접 알지 않게 해서 테스트에서 그냥 부를 수 있다.
 */
export function setTheme(
  next: Partial<ThemeState>,
  persist?: (payload: { theme: string; themeOverrides: Record<string, string> }) => void,
): ThemeColors {
  if (next.preset && PRESETS[next.preset]) current.preset = next.preset
  if (next.overrides) {
    const clean: Record<string, string> = {}
    for (const key of EDITABLE_KEYS) {
      const value = next.overrides[key]
      if (isValidHex(value)) clean[key] = normalizeHex(value)
    }
    current.overrides = clean
  }
  writeStored(current)
  persist?.({ theme: current.preset, themeOverrides: current.overrides })
  return applyTheme(current.preset, current.overrides)
}

/**
 * 영속 설정이 도착했을 때 다시 맞춘다(`uiPrefsLoaded`).
 *
 * 디스크가 단일 출처다 — 다른 기기/프로필에서 바꾼 값이 localStorage 보다 우선한다.
 * 실제로 달라졌을 때만 다시 칠해서 불필요한 리페인트를 피한다.
 */
export function reconcileTheme(prefs: {
  theme?: unknown
  themeOverrides?: unknown
}): ThemeColors | null {
  const preset = typeof prefs.theme === 'string' && PRESETS[prefs.theme] ? prefs.theme : null
  const overrides =
    prefs.themeOverrides && typeof prefs.themeOverrides === 'object'
      ? (prefs.themeOverrides as Record<string, string>)
      : null
  if (!preset && !overrides) return null

  const nextPreset = preset ?? current.preset
  const nextOverrides: Record<string, string> = {}
  for (const key of EDITABLE_KEYS) {
    const value = (overrides ?? current.overrides)[key]
    if (isValidHex(value)) nextOverrides[key] = normalizeHex(value)
  }
  const same =
    nextPreset === current.preset &&
    JSON.stringify(nextOverrides) === JSON.stringify(current.overrides)
  if (same) return null

  current = { preset: nextPreset, overrides: nextOverrides }
  writeStored(current)
  return applyTheme(current.preset, current.overrides)
}
