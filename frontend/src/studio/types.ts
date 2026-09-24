export const STUDIO_PROTOCOL_VERSION = 1 as const

export type StudioOperation =
  | 'sync.bootstrap'
  | 'runtime.snapshot'
  | 'runtime.execute'
  | 'generation_api.snapshot'
  | 'generation_api.execute'
  | 'app_update.snapshot'
  | 'app_update.execute'
  | 'model_paths.snapshot'
  | 'model_paths.save'
  | 'model_paths.reset'
  | 'model_paths.refresh'
  | 'native.pick_directory'

export type StudioTopic = 'runtime' | 'generation_api' | 'app_update' | 'model_paths'

export interface StudioOperationInputMap {
  /** includeAppUpdate=false 면 git 을 실행하는 앱 업데이트 스냅샷을 빼고 돌려준다. */
  'sync.bootstrap': { includeAppUpdate?: boolean }
  'runtime.snapshot': Record<string, never>
  'runtime.execute': {
    engine: string
    action: string
    payload?: Record<string, unknown>
  }
  'generation_api.snapshot': Record<string, never>
  'generation_api.execute': {
    action: string
    payload?: Record<string, unknown>
  }
  'app_update.snapshot': Record<string, never>
  'app_update.execute': {
    action: 'check' | 'configure' | 'skip' | 'install'
    payload?: Record<string, unknown>
  }
  'model_paths.snapshot': Record<string, never>
  'model_paths.save': {
    paths: Record<string, string>
  }
  'model_paths.reset': Record<string, never>
  'model_paths.refresh': Record<string, never>
  'native.pick_directory': {
    purpose: 'runtime_install' | 'runtime_extension' | 'model_path'
    engine?: string
    key?: string
  }
}

export interface StudioCommandEnvelope<O extends StudioOperation = StudioOperation> {
  version: typeof STUDIO_PROTOCOL_VERSION
  requestId: string
  operation: O
  input: StudioOperationInputMap[O]
}

export interface StudioJob {
  id: string
  [key: string]: unknown
}

export interface StudioErrorPayload {
  code: string
  message: string
  retryable: boolean
  fields?: Record<string, string>
  details?: Record<string, unknown>
  [key: string]: unknown
}

export interface StudioOkReply<T = unknown> {
  version: typeof STUDIO_PROTOCOL_VERSION
  requestId: string
  status: 'ok'
  data: T
}

export interface StudioAcceptedReply {
  version: typeof STUDIO_PROTOCOL_VERSION
  requestId: string
  status: 'accepted'
  job: StudioJob
}

export interface StudioErrorReply {
  version: typeof STUDIO_PROTOCOL_VERSION
  requestId: string
  status: 'error'
  error: StudioErrorPayload
}

export type StudioReply<T = unknown> = StudioOkReply<T> | StudioAcceptedReply | StudioErrorReply

export interface StudioEvent<T = unknown> {
  version: typeof STUDIO_PROTOCOL_VERSION
  eventEpoch: string
  seq: number
  topic: string
  type: string
  operation?: string
  jobId?: string
  data: T
}

export interface StudioDescription {
  version: typeof STUDIO_PROTOCOL_VERSION
  eventEpoch: string
  operations: string[]
  topics: string[]
  [key: string]: unknown
}

export interface StudioSignal {
  connect(callback: (eventJson: string) => void): void
  disconnect(callback: (eventJson: string) => void): void
}

export interface StudioTransport {
  readonly bindingGeneration: number
  describe(callback: (descriptionJson: string) => void): void
  invoke(envelopeJson: string, callback: (replyJson: string) => void): void
  resume(afterSeq: number, callback: (replyJson: string) => void): void
  acknowledge(sequence: number): void
  replaceCursor(sequence: number, eventEpoch?: string, generation?: number): boolean
  confirmEpoch(eventEpoch: string, generation?: number): {
    changed: boolean
    resumedPending: boolean
    stale?: boolean
  }
  releaseEvents(eventEpoch: string, generation?: number): boolean
  event: StudioSignal
  resumeError: StudioSignal
  reconnected: {
    connect(callback: (generation: number) => void): void
    disconnect(callback: (generation: number) => void): void
  }
}
