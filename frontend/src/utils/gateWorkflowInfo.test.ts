import { describe, expect, it } from 'vitest'
import { describeWorkflowLine, toGateWorkflowInfo } from './gateWorkflowInfo'

describe('toGateWorkflowInfo', () => {
  it('maps analyze_workflow snake_case and absorbs python None', () => {
    const info = toGateWorkflowInfo({
      valid: true, format: 'api', node_count: 7, width: 832, height: 1216,
      classification: 'native_checkpoint', is_locked: false,
      generation_blocker: null, model_selectable: true,
    })
    expect(info).toEqual({
      valid: true, format: 'api', nodeCount: 7, width: 832, height: 1216,
      locked: false, classification: 'native_checkpoint', error: undefined,
      generationBlocker: undefined,
    })
  })

  it('keeps a non-empty generation blocker', () => {
    const info = toGateWorkflowInfo({ valid: true, generation_blocker: '  샘플러 여러 개  ' })
    expect(info?.generationBlocker).toBe('샘플러 여러 개')
    expect(toGateWorkflowInfo(undefined)).toBeUndefined()
  })
})

describe('describeWorkflowLine', () => {
  it('asks for a workflow when no path is set and waits while unanalysed', () => {
    expect(describeWorkflowLine('  ', undefined)?.tone).toBe('muted')
    expect(describeWorkflowLine('C:/wf.json', undefined)).toBeNull()
  })

  it('shows web-format rejection as an unreadable workflow', () => {
    const line = describeWorkflowLine('C:/wf.json', {
      valid: false, error: 'Export (API)로 저장한 JSON 파일을 선택하세요.',
    })
    expect(line).toEqual({ tone: 'alert', text: '읽을 수 없음 — Export (API)로 저장한 JSON 파일을 선택하세요.' })
  })

  it('summarises a usable workflow and its model lock', () => {
    expect(describeWorkflowLine('wf.json', {
      valid: true, format: 'api', nodeCount: 7, width: 512, height: 768,
      classification: 'native_checkpoint', locked: false,
    })).toEqual({ tone: 'muted', text: 'API 형식 · 노드 7개 · 512×768 · Checkpoint 로더 · 모델 선택 가능' })
    expect(describeWorkflowLine('wf.json', {
      valid: true, classification: 'locked_unknown', locked: true,
    })).toEqual({ tone: 'warn', text: '커스텀 로더 · 워크플로가 모델을 고정' })
  })

  it('warns that a valid-looking workflow cannot generate', () => {
    const line = describeWorkflowLine('wf.json', {
      valid: true, format: 'api', locked: false,
      generationBlocker: 'custom workflow에 sampler 노드가 여러 개이므로 자동 삽입 대상이 불명확합니다: 5, 8',
    })
    expect(line?.tone).toBe('alert')
    expect(line?.text).toContain('생성 불가 — custom workflow에 sampler 노드가 여러 개')
    expect(line?.text.startsWith('API 형식')).toBe(true)
  })
})
