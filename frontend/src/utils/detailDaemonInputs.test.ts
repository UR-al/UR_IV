import { describe, expect, it } from 'vitest'
import {
  DD_FLOAT_INPUTS, commitDdInput, ddPlaceholder, normalizeDdInput,
  type DdFloatKey,
} from './detailDaemonInputs'

/**
 * 원본 노드 입력표 — [default, min, max, step].
 * origin: Jonseed/ComfyUI-Detail-Daemon@3394e44:detail_daemon_node.py:324-356 (DetailDaemonSamplerNode INPUT_TYPES)
 */
const NODE_INPUTS: Record<DdFloatKey, [number, number, number, number]> = {
  dd_amount: [0.1, -5.0, 5.0, 0.01],          // detail_amount :324-327
  dd_start: [0.2, 0.0, 1.0, 0.01],            // start :328-331
  dd_end: [0.8, 0.0, 1.0, 0.01],              // end :332-335
  dd_bias: [0.5, 0.0, 1.0, 0.01],             // bias :336-339
  dd_exponent: [1.0, 0.0, 10.0, 0.05],        // exponent :340-343
  dd_start_offset: [0.0, -1.0, 1.0, 0.01],    // start_offset :344-347
  dd_end_offset: [0.0, -1.0, 1.0, 0.01],      // end_offset :348-351
  dd_fade: [0.0, 0.0, 1.0, 0.05],             // fade :352-355
}
const KEYS = Object.keys(NODE_INPUTS) as DdFloatKey[]

describe('DD_FLOAT_INPUTS', () => {
  it('equals the original node inputs', () => {
    expect(Object.keys(DD_FLOAT_INPUTS).sort()).toEqual([...KEYS].sort())
    for (const key of KEYS) {
      const { default: def, min, max, step } = DD_FLOAT_INPUTS[key]
      expect([def, min, max, step], key).toEqual(NODE_INPUTS[key])
    }
  })

  it('is frozen', () => {
    expect(Object.isFrozen(DD_FLOAT_INPUTS)).toBe(true)
  })
})

describe('ddPlaceholder', () => {
  it('shows the node default in step precision', () => {
    expect(ddPlaceholder('dd_amount')).toBe('0.10')
    expect(ddPlaceholder('dd_exponent')).toBe('1.00')
    expect(ddPlaceholder('dd_start_offset')).toBe('0.00')
    for (const key of KEYS) expect(Number(ddPlaceholder(key)), key).toBe(NODE_INPUTS[key][0])
  })
})

describe('normalizeDdInput', () => {
  it('keeps any finite in-range value exactly as typed (no step snapping, same type)', () => {
    for (const raw of [0.1, '0.1', 0.123, '0.1234', -4.99, 5, -5, '5', 0, '-0']) {
      expect(normalizeDdInput('dd_amount', raw)).toBe(raw)
    }
    expect(normalizeDdInput('dd_exponent', 7.33)).toBe(7.33)       // step .05 이지만 그대로
    expect(normalizeDdInput('dd_start_offset', '-0.37')).toBe('-0.37')
  })

  it('clamps out-of-range values to the node range (what the backend sends)', () => {
    expect(normalizeDdInput('dd_amount', 7)).toBe(5)
    expect(normalizeDdInput('dd_amount', '-12.5')).toBe(-5)
    expect(normalizeDdInput('dd_start_offset', 1.5)).toBe(1)
    expect(normalizeDdInput('dd_end_offset', '-2')).toBe(-1)
    expect(normalizeDdInput('dd_exponent', 11)).toBe(10)
    expect(normalizeDdInput('dd_start', -0.1)).toBe(0)
    expect(normalizeDdInput('dd_fade', 1.01)).toBe(1)
  })

  it('leaves an empty field empty (the backend sends the default shown as placeholder)', () => {
    for (const raw of ['', '   ', null, undefined]) {
      expect(normalizeDdInput('dd_amount', raw)).toBe(raw)
    }
  })

  it('turns unreadable or non-finite values into an empty field (the backend sends the default)', () => {
    for (const raw of ['abc', 'NaN', 'Infinity', NaN, Infinity, -Infinity, true]) {
      expect(normalizeDdInput('dd_amount', raw), String(raw)).toBe('')
    }
  })
})

describe('commitDdInput', () => {
  /** 위젯 스토어 대역 — 쓰기(set)를 센다. */
  function store(values: Record<string, unknown>) {
    const writes: string[] = []
    const widgets = new Proxy({ ...values } as Record<string, any>, {
      set(target, prop, value) { writes.push(String(prop)); target[prop as string] = value; return true },
    })
    return { widgets, writes }
  }

  it('writes the clamped value under `_<key>` (the node range)', () => {
    const { widgets, writes } = store({ _dd_amount: 7, _dd_end_offset: '-2', _dd_exponent: '11' })
    commitDdInput(widgets, 'dd_amount')
    commitDdInput(widgets, 'dd_end_offset')
    commitDdInput(widgets, 'dd_exponent')
    expect({ ...widgets }).toEqual({ _dd_amount: 5, _dd_end_offset: -1, _dd_exponent: 10 })
    expect(writes).toEqual(['_dd_amount', '_dd_end_offset', '_dd_exponent'])
  })

  it('does not write in-range or empty values (kept exactly as typed)', () => {
    const { widgets, writes } = store({ _dd_amount: '0.123', _dd_start: 0.37, _dd_fade: '', _dd_bias: -0 })
    for (const key of ['dd_amount', 'dd_start', 'dd_fade', 'dd_bias'] as const) commitDdInput(widgets, key)
    expect({ ...widgets }).toEqual({ _dd_amount: '0.123', _dd_start: 0.37, _dd_fade: '', _dd_bias: -0 })
    expect(writes).toEqual([])
  })

  it('does not create a missing key (the backend fills the default)', () => {
    const { widgets, writes } = store({})
    commitDdInput(widgets, 'dd_start_offset')
    expect(writes).toEqual([])
    expect('_dd_start_offset' in widgets).toBe(false)
  })

  it('turns unreadable or non-finite values into an empty field', () => {
    const { widgets, writes } = store({ _dd_amount: NaN, _dd_start: 'abc', _dd_end: Infinity })
    for (const key of ['dd_amount', 'dd_start', 'dd_end'] as const) commitDdInput(widgets, key)
    expect({ ...widgets }).toEqual({ _dd_amount: '', _dd_start: '', _dd_end: '' })
    expect(writes).toEqual(['_dd_amount', '_dd_start', '_dd_end'])
  })

  it('touches only the committed key', () => {
    const { widgets, writes } = store({ _dd_amount: 9, _dd_start: 9 })
    commitDdInput(widgets, 'dd_start')
    expect({ ...widgets }).toEqual({ _dd_amount: 9, _dd_start: 1 })
    expect(writes).toEqual(['_dd_start'])
  })
})
