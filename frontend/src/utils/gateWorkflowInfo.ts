import type { ComfyWorkflowPicked } from '../types/bridge'

/**
 * 백엔드 게이트의 워크플로 요약 — 파이썬 `analyze_workflow()` 결과를 게이트가 읽는
 * 이름으로 옮긴 것과, 그것을 한 줄 문구로 만드는 순수 로직.
 * (그림은 components/BackendGate.vue, 배선은 composables/useBackendGate.ts)
 */
export interface GateWorkflowInfo {
  valid: boolean
  format?: string
  nodeCount?: number
  width?: number
  height?: number
  locked?: boolean
  classification?: string
  error?: string
  /**
   * 필수 노드는 다 있지만 앱 생성이 항상 거부하는 구조의 이유
   * (샘플러 여러 개, SamplerCustom 등). 있으면 '생성 불가'로 알린다.
   */
  generationBlocker?: string
}

export type WorkflowLineTone = 'muted' | 'warn' | 'alert'

export interface WorkflowLine {
  tone: WorkflowLineTone
  text: string
}

/** 파이썬 `WorkflowClassification` 값 → 사람이 읽는 말. */
export const WORKFLOW_KIND: Record<string, string> = {
  native_checkpoint: 'Checkpoint 로더',
  native_unet: 'UNet 로더',
  locked_unknown: '커스텀 로더',
  no_sampler: '샘플러 없음',
  unknown: '알 수 없는 구성',
}

/**
 * 파이썬은 '없음'을 None(→ null) 으로 보내는데 게이트는 optional(undefined) 로 본다.
 * `?? undefined` 가 그 경계를 흡수한다 — null 이 그대로 새면 '0×0' 같은 헛것이 찍힌다.
 */
export function toGateWorkflowInfo(
  info: ComfyWorkflowPicked['info'] | undefined,
): GateWorkflowInfo | undefined {
  if (!info) return undefined
  const blocker = typeof info.generation_blocker === 'string' ? info.generation_blocker.trim() : ''
  return {
    valid: !!info.valid,
    format: info.format ?? undefined,
    nodeCount: typeof info.node_count === 'number' ? info.node_count : undefined,
    width: info.width ?? undefined,
    height: info.height ?? undefined,
    locked: !!info.is_locked,
    classification: info.classification ?? undefined,
    error: info.error ?? undefined,
    generationBlocker: blocker || undefined,
  }
}

/**
 * 워크플로 한 줄. 경로가 없으면 안내, 분석 전이면 null(빈 줄이 가짜 정보보다 낫다).
 * 뜻은 글자가 나르고 색(tone)은 거든다.
 */
export function describeWorkflowLine(
  path: string,
  info: GateWorkflowInfo | undefined,
): WorkflowLine | null {
  if (!path.trim()) {
    return { tone: 'muted', text: '워크플로 JSON 이 있어야 ComfyUI 가 생성을 실행합니다.' }
  }
  if (!info) return null
  if (!info.valid) {
    return { tone: 'alert', text: `읽을 수 없음 — ${info.error || '알 수 없는 오류'}` }
  }
  const parts: string[] = []
  if (info.format) parts.push(`${info.format.toUpperCase()} 형식`)
  if (typeof info.nodeCount === 'number') parts.push(`노드 ${info.nodeCount}개`)
  if (info.width && info.height) parts.push(`${info.width}×${info.height}`)
  if (info.classification) parts.push(WORKFLOW_KIND[info.classification] ?? info.classification)
  // 모델 콤보가 잠기는지는 사용자가 시작 전에 알아야 한다 — 나중에 회색 콤보를
  // 보고 고장으로 오해하는 게 이 화면의 단골 문의였다.
  parts.push(info.locked ? '워크플로가 모델을 고정' : '모델 선택 가능')
  if (info.generationBlocker) {
    // 필수 노드는 있어 '읽을 수는' 있지만, 생성하면 매번 같은 이유로 거부된다.
    return { tone: 'alert', text: `${parts.join(' · ')} · 생성 불가 — ${info.generationBlocker}` }
  }
  return { tone: info.locked ? 'warn' : 'muted', text: parts.join(' · ') }
}
