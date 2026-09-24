import { getStudioTransport } from '../bridge.js'
import {
  CursorRecoveryController,
  MonotonicEventCursor,
  finishResumeWithRecovery,
  selectStudioClient,
} from './resumableTransport.js'
import {
  STUDIO_PROTOCOL_VERSION,
  type StudioAcceptedReply,
  type StudioCommandEnvelope,
  type StudioDescription,
  type StudioErrorPayload,
  type StudioEvent,
  type StudioOperation,
  type StudioOperationInputMap,
  type StudioReply,
  type StudioTopic,
  type StudioTransport,
} from './types'

type EventHandler = (event: StudioEvent) => void
type BootstrapRecovery = {
  eventEpoch: string
  eventCursor: number
  description: StudioDescription
  snapshots: {
    runtime: unknown
    generationApi: unknown
    appUpdate: unknown
    modelPaths: unknown
  }
}

let fallbackRequestSequence = 0

function requestId(): string {
  const randomUuid = globalThis.crypto?.randomUUID?.bind(globalThis.crypto)
  if (randomUuid) return randomUuid()
  fallbackRequestSequence += 1
  return `studio-${Date.now().toString(36)}-${fallbackRequestSequence.toString(36)}`
}

function parseJsonValue(raw: unknown): unknown {
  let value = raw
  for (let depth = 0; depth < 3 && typeof value === 'string'; depth += 1) {
    value = JSON.parse(value || '{}')
  }
  return value
}

function objectValue(raw: unknown, label: string): Record<string, any> {
  const value = parseJsonValue(raw)
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw protocolError(`${label}이(가) JSON 객체가 아닙니다.`)
  }
  return value as Record<string, any>
}

function responseGeneration(raw: unknown, label: string): number {
  const value = objectValue(raw, label)
  const generation = Number(value.transportGeneration)
  if (!Number.isSafeInteger(generation) || generation < 0) {
    throw protocolError(`${label} transportGeneration이 없습니다.`)
  }
  return generation
}

function protocolError(message: string): StudioClientError {
  return new StudioClientError({ code: 'PROTOCOL_ERROR', message, retryable: false })
}

function unavailableError(message: string): StudioClientError {
  return new StudioClientError({ code: 'UNAVAILABLE', message, retryable: true })
}

function operationList(raw: Record<string, any>): string[] {
  const source = raw.operations ?? raw.capabilities?.operations ?? raw.methods
  if (Array.isArray(source)) {
    return source.flatMap((item: unknown) => {
      if (typeof item === 'string') return [item]
      if (item && typeof item === 'object') {
        const descriptor = item as Record<string, unknown>
        if (descriptor.available === false) return []
        const name = String(descriptor.name || '')
        return name ? [name] : []
      }
      return []
    })
  }
  if (source && typeof source === 'object') return Object.keys(source)
  return []
}

function topicList(raw: Record<string, any>): string[] {
  const source = raw.topics ?? raw.events ?? raw.capabilities?.topics
  if (Array.isArray(source)) return source.map(String)
  if (source && typeof source === 'object') return Object.keys(source)
  return []
}

function parseDescription(raw: unknown): StudioDescription {
  const value = objectValue(raw, 'Studio description')
  const version = Number(value.version ?? value.protocolVersion)
  if (version !== STUDIO_PROTOCOL_VERSION) {
    throw protocolError(`지원하지 않는 Studio protocol version입니다: ${String(value.version ?? value.protocolVersion)}`)
  }
  if (value.status === 'error') {
    const error = value.error && typeof value.error === 'object' ? value.error : {}
    throw new StudioClientError({
      ...error,
      code: String(error.code || 'INTERNAL'),
      message: String(error.message || 'Studio handshake가 실패했습니다.'),
      retryable: Boolean(error.retryable),
    })
  }
  const eventEpoch = String(value.eventEpoch || '').trim()
  if (!eventEpoch) throw protocolError('Studio description eventEpoch이 없습니다.')
  return {
    ...value,
    version: STUDIO_PROTOCOL_VERSION,
    eventEpoch,
    operations: operationList(value),
    topics: topicList(value),
  }
}

function parseReply<T>(raw: unknown, expectedRequestId: string): StudioReply<T> {
  const value = objectValue(raw, 'Studio reply')
  if (Number(value.version) !== STUDIO_PROTOCOL_VERSION) {
    throw protocolError('잘못된 Studio reply version입니다.')
  }
  if (String(value.requestId || '') !== expectedRequestId) {
    throw protocolError('Studio reply requestId가 요청과 다릅니다.')
  }
  if (value.status === 'ok') {
    return {
      version: STUDIO_PROTOCOL_VERSION,
      requestId: expectedRequestId,
      status: 'ok',
      data: value.data as T,
    }
  }
  if (value.status === 'accepted') {
    const fallbackData = value.data && typeof value.data === 'object' ? value.data : {}
    const rawJob = value.job && typeof value.job === 'object'
      ? value.job
      : fallbackData.jobId
        ? { ...fallbackData, id: fallbackData.jobId }
        : null
    if (!rawJob || !String(rawJob.id || '')) {
      throw protocolError('accepted reply에 job.id가 없습니다.')
    }
    return {
      version: STUDIO_PROTOCOL_VERSION,
      requestId: expectedRequestId,
      status: 'accepted',
      job: { ...rawJob, id: String(rawJob.id) },
    }
  }
  if (value.status === 'error') {
    const error = value.error && typeof value.error === 'object' ? value.error : {}
    return {
      version: STUDIO_PROTOCOL_VERSION,
      requestId: expectedRequestId,
      status: 'error',
      error: {
        ...error,
        code: String(error.code || 'INTERNAL'),
        message: String(error.message || 'Studio 요청이 실패했습니다.'),
        retryable: Boolean(error.retryable),
      },
    }
  }
  throw protocolError(`알 수 없는 Studio reply status입니다: ${String(value.status)}`)
}

function parseEvent(raw: unknown): StudioEvent {
  const value = objectValue(raw, 'Studio event')
  if (Number(value.version) !== STUDIO_PROTOCOL_VERSION) throw protocolError('잘못된 Studio event version입니다.')
  if (!Number.isSafeInteger(Number(value.seq)) || Number(value.seq) < 0) {
    throw protocolError('Studio event seq가 올바르지 않습니다.')
  }
  if (!String(value.topic || '') || !String(value.type || '')) throw protocolError('Studio event topic/type이 없습니다.')
  const eventEpoch = String(value.eventEpoch || '').trim()
  if (!eventEpoch) throw protocolError('Studio event eventEpoch이 없습니다.')
  return {
    version: STUDIO_PROTOCOL_VERSION,
    eventEpoch,
    seq: Number(value.seq),
    topic: String(value.topic),
    type: String(value.type),
    jobId: value.jobId == null ? undefined : String(value.jobId),
    data: value.data,
  }
}

function callQt(target: any, method: 'describe' | 'invoke' | 'resume', args: unknown[]): Promise<unknown> {
  const fn = target?.[method]
  if (typeof fn !== 'function') return Promise.reject(unavailableError(`Studio transport에 ${method}가 없습니다.`))
  return new Promise((resolve, reject) => {
    try {
      fn(...args, (raw: unknown) => resolve(raw))
    } catch (error) {
      reject(unavailableError(error instanceof Error ? error.message : String(error)))
    }
  })
}

function topicMatches(subscription: string, actual: string): boolean {
  if (actual === subscription || actual.startsWith(`${subscription}.`)) return true
  return subscription === 'model_paths'
    && (actual === 'models' || actual.startsWith('models.'))
}

export class StudioClientError extends Error {
  readonly code: string
  readonly retryable: boolean
  readonly fields?: Record<string, string>
  readonly details?: Record<string, unknown>

  constructor(readonly detail: StudioErrorPayload, readonly requestId = '') {
    super(detail.message)
    this.name = 'StudioClientError'
    this.code = detail.code
    this.retryable = detail.retryable
    this.details = detail.details
    const nestedFields = detail.details?.fields
    const fields = detail.fields || (nestedFields && typeof nestedFields === 'object'
      ? Object.fromEntries(Object.entries(nestedFields).map(([key, value]) => [key, String(value)]))
      : undefined)
    this.fields = fields
  }
}

function parseResumeReply(raw: unknown, expectedAfterSeq: number, expectedEpoch: string): void {
  const value = objectValue(raw, 'Studio resume reply')
  if (Number(value.version) !== STUDIO_PROTOCOL_VERSION) {
    throw protocolError('잘못된 Studio resume reply version입니다.')
  }
  if (value.status === 'error') {
    const error = value.error && typeof value.error === 'object' ? value.error : {}
    throw new StudioClientError({
      ...error,
      code: String(error.code || 'INTERNAL'),
      message: String(error.message || 'Studio event 구독을 복구하지 못했습니다.'),
      retryable: Boolean(error.retryable),
      details: {
        ...(error.details && typeof error.details === 'object' ? error.details : {}),
        eventEpoch: String(value.eventEpoch || ''),
      },
    })
  }
  if (value.status !== 'ok' || Number(value.afterSeq) !== expectedAfterSeq) {
    throw protocolError('Studio event cursor가 요청한 위치에서 복구되지 않았습니다.')
  }
  const actualEpoch = String(value.eventEpoch || '').trim()
  if (!actualEpoch) throw protocolError('Studio resume reply eventEpoch이 없습니다.')
  if (actualEpoch !== expectedEpoch) {
    throw new StudioClientError({
      code: 'EPOCH_MISMATCH',
      message: 'Studio event journal instance가 변경되었습니다.',
      retryable: true,
      details: { expectedEpoch, actualEpoch },
    })
  }
}

function isRecoverableCursorError(error: unknown): error is StudioClientError {
  return error instanceof StudioClientError
    && (error.code === 'CURSOR_EXPIRED' || error.code === 'EPOCH_MISMATCH')
}

function isRecoveryCancelled(error: unknown): boolean {
  return error instanceof Error
    && 'code' in error
    && error.code === 'RECOVERY_CANCELLED'
}

export interface StudioClient {
  readonly kind: 'v1' | 'unavailable'
  readonly description: StudioDescription
  supports(operation: StudioOperation): boolean
  invoke<O extends StudioOperation, T = unknown>(
    operation: O,
    input: StudioOperationInputMap[O],
  ): Promise<StudioReply<T>>
  subscribe(topic: StudioTopic, handler: EventHandler): () => void
  dispose(): void
}

class NativeStudioClient implements StudioClient {
  readonly kind = 'v1' as const
  description: StudioDescription
  private readonly listeners = new Map<string, Set<EventHandler>>()
  private readonly cursor = new MonotonicEventCursor()
  private readonly recovery: CursorRecoveryController
  private disposed = false
  private eventEpoch: string
  private bindingGeneration: number
  private reconnectGeneration = 0
  private reconnectPromise: Promise<void> | null = null
  private latestReconnect: Promise<void> | null = null
  private activeResume: Promise<void> | null = null
  private readonly eventHandler = (raw: string) => {
    try {
      const event = parseEvent(raw)
      if (event.eventEpoch !== this.eventEpoch) {
        void this.beginReconnect(false, this.transport.bindingGeneration)
        return
      }
      if (!this.cursor.accept(event.seq)) return
      this.transport.acknowledge(event.seq)
      this.dispatchEvent(event)
    } catch (error) {
      console.error('[studio] malformed event ignored', error)
    }
  }
  private readonly resumeErrorHandler = (raw: string) => {
    if (this.disposed) return
    try {
      parseResumeReply(raw, this.cursor.value, this.eventEpoch)
    } catch (error) {
      if (!isRecoverableCursorError(error)) {
        console.error('[studio] automatic event resume failed', error)
        return
      }
      void this.recoverCursor().catch((recoveryError) => {
        if (!this.disposed) console.error('[studio] expired cursor recovery failed', recoveryError)
      })
    }
  }
  private readonly reconnectedHandler = (generation: number) => {
    this.recovery.reset()
    void this.beginReconnect(true, generation).catch((error) => {
      if (!this.disposed) console.error('[studio] reconnect handshake failed', error)
    })
  }

  constructor(
    private readonly transport: StudioTransport,
    description: StudioDescription,
  ) {
    this.description = description
    this.eventEpoch = description.eventEpoch
    this.bindingGeneration = Number(description.transportGeneration)
    if (!Number.isSafeInteger(this.bindingGeneration) || this.bindingGeneration < 0) {
      throw protocolError('Studio description transportGeneration이 없습니다.')
    }
    const initialConfirmation = this.transport.confirmEpoch(
      this.eventEpoch,
      this.bindingGeneration,
    )
    if (initialConfirmation.stale) throw unavailableError('Studio 연결이 handshake 중 교체되었습니다.')
    this.recovery = new CursorRecoveryController({
      bootstrap: () => this.loadBootstrapRecovery(),
      reconcile: (state: BootstrapRecovery) => this.reconcileSnapshots(state),
      resume: (eventCursor: number, state: BootstrapRecovery) => (
        this.resumeAt(eventCursor, state.eventEpoch)
      ),
      isExpired: (error: unknown) => isRecoverableCursorError(error),
      maxAttempts: 2,
    })
    this.transport.event.connect(this.eventHandler)
    this.transport.resumeError.connect(this.resumeErrorHandler)
    this.transport.reconnected.connect(this.reconnectedHandler)
  }

  supports(operation: StudioOperation): boolean {
    return this.description.operations.includes(operation)
  }

  async startEventStream(): Promise<void> {
    try {
      await this.resumeAt(this.cursor.value, this.eventEpoch)
    } catch (error) {
      if (!isRecoverableCursorError(error)) throw error
      try {
        await this.recoverCursor()
      } catch (recoveryError) {
        const reconnect = this.latestReconnect
        if (!isRecoveryCancelled(recoveryError) || !reconnect || this.disposed) {
          throw recoveryError
        }
        await reconnect
      }
    }
  }

  async invoke<O extends StudioOperation, T = unknown>(
    operation: O,
    input: StudioOperationInputMap[O],
  ): Promise<StudioReply<T>> {
    if (!this.supports(operation)) throw unavailableError(`Studio operation을 지원하지 않습니다: ${operation}`)
    const id = requestId()
    const envelope: StudioCommandEnvelope<O> = {
      version: STUDIO_PROTOCOL_VERSION,
      requestId: id,
      operation,
      input,
    }
    const raw = await callQt(this.transport, 'invoke', [JSON.stringify(envelope)])
    const generation = responseGeneration(raw, 'Studio reply')
    if (generation !== this.transport.bindingGeneration) {
      throw new StudioClientError({
        code: 'RESULT_UNKNOWN',
        message: '연결이 바뀌어 Studio 요청 결과를 현재 상태에 적용하지 않았습니다.',
        retryable: false,
      }, id)
    }
    const reply = parseReply<T>(raw, id)
    if (reply.status === 'error') throw new StudioClientError(reply.error, id)
    return reply
  }

  subscribe(topic: StudioTopic, handler: EventHandler): () => void {
    if (!this.listeners.has(topic)) this.listeners.set(topic, new Set())
    const handlers = this.listeners.get(topic)!
    handlers.add(handler)
    return () => {
      handlers.delete(handler)
      if (handlers.size === 0) this.listeners.delete(topic)
    }
  }

  dispose(): void {
    if (this.disposed) return
    this.disposed = true
    this.recovery.dispose()
    this.listeners.clear()
    this.reconnectGeneration += 1
    try { this.transport.event.disconnect(this.eventHandler) } catch {}
    try { this.transport.resumeError.disconnect(this.resumeErrorHandler) } catch {}
    try { this.transport.reconnected.disconnect(this.reconnectedHandler) } catch {}
  }

  private dispatchEvent(event: StudioEvent): void {
    for (const [topic, handlers] of this.listeners) {
      if (!topicMatches(topic, event.topic)) continue
      for (const handler of [...handlers]) {
        try { handler(event) } catch (error) { console.error('[studio] event handler failed', error) }
      }
    }
  }

  private resumeAt(afterSeq: number, expectedEpoch: string): Promise<void> {
    const operation = this.performResumeAt(afterSeq, expectedEpoch)
    let tracked: Promise<void>
    tracked = operation.then(
      () => {
        if (this.activeResume === tracked) this.activeResume = null
      },
      (error) => {
        if (this.activeResume === tracked) this.activeResume = null
        throw error
      },
    )
    this.activeResume = tracked
    return tracked
  }

  private async performResumeAt(afterSeq: number, expectedEpoch: string): Promise<void> {
    const raw = await callQt(this.transport, 'resume', [afterSeq])
    const generation = responseGeneration(raw, 'Studio resume reply')
    parseResumeReply(raw, afterSeq, expectedEpoch)
    if (!this.transport.releaseEvents(expectedEpoch, generation)) {
      throw unavailableError('Studio event gate를 현재 연결에서 열지 못했습니다.')
    }
  }

  private recoverCursor(): Promise<void> {
    if (this.disposed) return Promise.reject(unavailableError('Studio client가 종료되었습니다.'))
    return this.recovery.recover()
  }

  private async loadBootstrapRecovery(): Promise<BootstrapRecovery> {
    const reply = await this.invoke('sync.bootstrap', {})
    if (reply.status !== 'ok') throw protocolError('sync.bootstrap이 즉시 완료되지 않았습니다.')
    const data = objectValue(reply.data, 'sync.bootstrap data')
    const description = parseDescription(data.description)
    const eventEpoch = String(data.eventEpoch || description.eventEpoch).trim()
    if (!eventEpoch || eventEpoch !== description.eventEpoch) {
      throw protocolError('sync.bootstrap eventEpoch이 description과 다릅니다.')
    }
    const eventCursor = Number(description.eventCursor)
    if (!Number.isSafeInteger(eventCursor) || eventCursor < 0) {
      throw protocolError('sync.bootstrap description.eventCursor가 올바르지 않습니다.')
    }
    for (const key of ['runtime', 'generationApi', 'appUpdate', 'modelPaths']) {
      if (!data[key] || typeof data[key] !== 'object' || Array.isArray(data[key])) {
        throw protocolError(`sync.bootstrap ${key} snapshot이 올바르지 않습니다.`)
      }
    }
    return {
      eventEpoch,
      eventCursor,
      description,
      snapshots: {
        runtime: data.runtime,
        generationApi: data.generationApi,
        appUpdate: data.appUpdate,
        modelPaths: data.modelPaths,
      },
    }
  }

  private reconcileSnapshots(state: BootstrapRecovery): void {
    this.description = state.description
    this.eventEpoch = state.eventEpoch
    this.bindingGeneration = this.transport.bindingGeneration
    const confirmation = this.transport.confirmEpoch(
      state.eventEpoch,
      this.bindingGeneration,
    )
    if (confirmation.stale) throw unavailableError('Studio 연결이 snapshot 복구 중 교체되었습니다.')
    const base = {
      version: STUDIO_PROTOCOL_VERSION,
      eventEpoch: state.eventEpoch,
      seq: state.eventCursor,
      type: 'reconciled',
      operation: 'sync.bootstrap',
    } as const
    this.dispatchEvent({
      ...base,
      topic: 'runtime.operation',
      data: { snapshot: state.snapshots.runtime, source: 'sync.bootstrap' },
    })
    this.dispatchEvent({
      ...base,
      topic: 'generation_api.operation',
      data: { snapshot: state.snapshots.generationApi, source: 'sync.bootstrap' },
    })
    this.dispatchEvent({
      ...base,
      topic: 'app_update.operation',
      data: { snapshot: state.snapshots.appUpdate, source: 'sync.bootstrap' },
    })
    this.dispatchEvent({
      ...base,
      topic: 'model_paths.changed',
      data: { snapshot: state.snapshots.modelPaths, source: 'sync.bootstrap' },
    })
    this.cursor.replace(state.eventCursor)
    if (!this.transport.replaceCursor(
      state.eventCursor,
      state.eventEpoch,
      this.bindingGeneration,
    )) {
      throw unavailableError('Studio cursor를 현재 연결에 적용하지 못했습니다.')
    }
  }

  private beginReconnect(force: boolean, bindingGeneration: number): Promise<void> {
    if (this.disposed) return Promise.reject(unavailableError('Studio client가 종료되었습니다.'))
    if (!force && this.reconnectPromise) return this.reconnectPromise
    const generation = ++this.reconnectGeneration
    const operation = this.performReconnect(generation, bindingGeneration)
    let tracked: Promise<void>
    tracked = operation.then(
      () => {
        if (this.reconnectPromise === tracked) this.reconnectPromise = null
      },
      (error) => {
        if (this.reconnectPromise === tracked) this.reconnectPromise = null
        throw error
      },
    )
    this.reconnectPromise = tracked
    this.latestReconnect = tracked
    return tracked
  }

  private async performReconnect(
    generation: number,
    bindingGeneration: number,
  ): Promise<void> {
    const guard = () => {
      if (
        this.disposed
        || generation !== this.reconnectGeneration
        || bindingGeneration !== this.transport.bindingGeneration
      ) {
        throw unavailableError('더 최신 Studio 연결이 이 handshake를 대체했습니다.')
      }
    }
    const raw = await callQt(this.transport, 'describe', [])
    guard()
    if (responseGeneration(raw, 'Studio description') !== bindingGeneration) {
      throw unavailableError('Studio description이 이전 연결에서 도착했습니다.')
    }
    const description = parseDescription(raw)
    const inheritedResume = this.activeResume
    const confirmation = this.transport.confirmEpoch(
      description.eventEpoch,
      bindingGeneration,
    )
    guard()
    if (confirmation.stale) throw unavailableError('Studio 연결이 handshake 중 교체되었습니다.')
    this.description = description
    this.eventEpoch = description.eventEpoch
    this.bindingGeneration = bindingGeneration
    if (confirmation.changed) {
      await this.recoverCursor()
      guard()
      return
    }
    const resume = confirmation.resumedPending
      ? inheritedResume
      : this.resumeAt(this.cursor.value, this.eventEpoch)
    if (!resume) {
      throw protocolError('재발행된 Studio resume의 완료 상태를 찾지 못했습니다.')
    }
    await finishResumeWithRecovery({
      resume,
      recover: () => this.recoverCursor(),
      isRecoverable: (error: unknown) => isRecoverableCursorError(error),
      guard,
    })
  }
}

/**
 * studio transport 가 없는 경우(vite 목 모드 등)의 클라이언트 — 아무 작업도 지원하지 않는다.
 *
 * 데스크톱·웹 호스트는 'studio' 객체를 항상 등록하므로 운영에서는 선택되지 않는다. 예전엔 여기서
 * 레거시 VueBridge 설정 슬롯으로 폴백했지만 그 슬롯들은 도달 불가라 제거했다(audit #176).
 */
class UnavailableStudioClient implements StudioClient {
  readonly kind = 'unavailable' as const
  readonly description: StudioDescription = {
    version: STUDIO_PROTOCOL_VERSION,
    eventEpoch: 'unavailable',
    operations: [],
    topics: [],
  }

  supports(_operation: StudioOperation): boolean {
    return false
  }

  async invoke<O extends StudioOperation, T = unknown>(
    operation: O,
    _input: StudioOperationInputMap[O],
  ): Promise<StudioReply<T>> {
    throw unavailableError(`Studio 연결이 없어 ${operation}을(를) 실행할 수 없습니다.`)
  }

  subscribe(_topic: StudioTopic, _handler: EventHandler): () => void {
    return () => {}
  }

  dispose(): void {}
}

async function createNativeClient(transport: StudioTransport): Promise<StudioClient> {
  const raw = await callQt(transport, 'describe', [])
  const generation = responseGeneration(raw, 'Studio description')
  if (generation !== transport.bindingGeneration) {
    throw unavailableError('Studio 연결이 초기 handshake 중 교체되었습니다.')
  }
  const client = new NativeStudioClient(transport, parseDescription(raw))
  try {
    await client.startEventStream()
    return client
  } catch (error) {
    client.dispose()
    throw error
  }
}

async function createUnavailableClient(): Promise<StudioClient> {
  return new UnavailableStudioClient()
}

let studioClientPromise: Promise<StudioClient> | null = null

export function getStudioClient(): Promise<StudioClient> {
  if (!studioClientPromise) {
    studioClientPromise = getStudioTransport()
      .then((transport: StudioTransport | null) => (
        selectStudioClient(transport, createNativeClient, createUnavailableClient)
      ))
      .catch((error) => {
        studioClientPromise = null
        throw error
      })
  }
  return studioClientPromise
}

export function acceptedData(reply: StudioAcceptedReply): Record<string, unknown> {
  return {
    accepted: true,
    operationId: reply.job.id,
    job: reply.job,
  }
}

export function replyData<T>(reply: StudioReply<T>): T | Record<string, unknown> {
  if (reply.status === 'accepted') return acceptedData(reply)
  if (reply.status === 'error') throw new StudioClientError(reply.error, reply.requestId)
  return reply.data
}

