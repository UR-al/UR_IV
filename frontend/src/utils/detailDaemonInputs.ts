/**
 * Detail Daemon 숫자 칸 — 원본 노드 입력 그대로(기본값·범위·step). 화면(components/guidance/DetailDaemonSection.vue)용.
 *
 * origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:320-356 (DetailDaemonSamplerNode INPUT_TYPES)
 * 토글(smooth 기본 켬, Hires Pass 기본 끔 — muerrilla/sd-webui-detail-daemon@19479998:scripts/detail_daemon.py:104)의
 * 기본값은 백엔드 default_settings 가 채운다.
 *
 * 값은 노드 단위 그대로 저장·전송된다 — 앱은 변환하지 않고 엔진이 ×0.1×CFG 를 곱한다. 백엔드
 * (core/anima_guidance.py DETAIL_DAEMON_SPEC)도 같은 기본값·범위를 쓴다: 빈 칸·유한하지 않은 값은 기본값을,
 * 범위 밖 값은 노드 범위로 자른 값을 보낸다. 그래서 화면은
 *   - placeholder 로 기본값을 보여 주고(빈 칸 = 그 값이 나간다),
 *   - 입력을 확정(change)할 때 범위 밖 값만 노드 범위로 자른다(ComfyUI 숫자 위젯과 같다). 범위 안의 유한한 값은
 *     step 에 맞추지 않고 입력한 그대로 둔다.
 */

export type DdFloatKey =
  | 'dd_amount' | 'dd_start' | 'dd_end' | 'dd_bias' | 'dd_exponent'
  | 'dd_start_offset' | 'dd_end_offset' | 'dd_fade'

export interface DdFloatInput {
  readonly default: number
  readonly min: number
  readonly max: number
  readonly step: number
}

/** 노드 입력(detail_amount, start, end, bias, exponent, start_offset, end_offset, fade) — 이름만 앱 키(dd_*). */
export const DD_FLOAT_INPUTS: Readonly<Record<DdFloatKey, DdFloatInput>> = Object.freeze({
  dd_amount:       { default: 0.1, min: -5.0, max: 5.0,  step: 0.01 },
  dd_start:        { default: 0.2, min: 0.0,  max: 1.0,  step: 0.01 },
  dd_end:          { default: 0.8, min: 0.0,  max: 1.0,  step: 0.01 },
  dd_bias:         { default: 0.5, min: 0.0,  max: 1.0,  step: 0.01 },
  dd_exponent:     { default: 1.0, min: 0.0,  max: 10.0, step: 0.05 },
  dd_start_offset: { default: 0.0, min: -1.0, max: 1.0,  step: 0.01 },
  dd_end_offset:   { default: 0.0, min: -1.0, max: 1.0,  step: 0.01 },
  dd_fade:         { default: 0.0, min: 0.0,  max: 1.0,  step: 0.05 },
})

function stepDecimals(step: number): number {
  return (String(step).split('.')[1] ?? '').length
}

/** 빈 칸에 보일 기본값 — step 자릿수로('0.10', '1.00'). 빈 칸이면 백엔드가 이 값을 보낸다. */
export function ddPlaceholder(key: DdFloatKey): string {
  const spec = DD_FLOAT_INPUTS[key]
  return spec.default.toFixed(stepDecimals(spec.step))
}

export type DdWidgetValue = string | number | boolean | null | undefined

/**
 * 입력 확정 때 값 — 백엔드가 보낼 값과 화면이 같도록.
 *  - 빈 칸(없음·공백)은 그대로 둔다: 백엔드가 기본값을 보내고 placeholder 가 그 값을 보여 준다.
 *  - 숫자로 읽을 수 없거나 유한하지 않으면 '' (백엔드도 기본값을 보낸다).
 *  - 범위 밖이면 노드 min/max 로 자른 숫자.
 *  - 범위 안이면 입력한 값을 그대로(같은 참조) 돌려준다 — step 에 맞추지 않고 타입도 바꾸지 않는다.
 */
export function normalizeDdInput(key: DdFloatKey, raw: DdWidgetValue): DdWidgetValue {
  if (raw === null || raw === undefined) return raw
  if (typeof raw === 'string' && raw.trim() === '') return raw
  const n = typeof raw === 'number' ? raw : Number(String(raw).trim())
  if (!Number.isFinite(n)) return ''
  const { min, max } = DD_FLOAT_INPUTS[key]
  if (n < min) return min
  if (n > max) return max
  return raw
}

/**
 * 칸 확정(change) — 위젯 스토어의 `_<key>` 값을 normalizeDdInput 결과로 바꾼다. 값이 그대로면 쓰지 않는다
 * (범위 안의 값은 입력한 그대로 남고, 없는 키를 새로 만들지도 않는다). DetailDaemonSection 의 @change 가 부른다.
 */
export function commitDdInput(widgets: Record<string, any>, key: DdFloatKey): void {
  const id = `_${key}`
  const next = normalizeDdInput(key, widgets[id])
  if (!Object.is(next, widgets[id])) widgets[id] = next
}
