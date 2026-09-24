/** Stable Studio transport state shared across QWebChannel reconnects. */

export function createSignal() {
  const handlers = new Set()
  const emit = (...args) => {
    for (const handler of [...handlers]) {
      try { handler(...args) } catch (error) { console.error('[studio] signal handler failed', error) }
    }
  }
  return {
    connect(handler) {
      if (typeof handler === 'function') handlers.add(handler)
    },
    disconnect(handler) { handlers.delete(handler) },
    emit,
    _emit: emit,
    get size() { return handlers.size },
  }
}

function validSequence(value) {
  return Number.isSafeInteger(value) && value >= 0
}

function controlError(code, message, {
  requestId = '',
  eventEpoch = '',
  retryable = false,
} = {}) {
  const reply = {
    version: 1,
    requestId,
    status: 'error',
    error: { code, message, retryable },
    seq: 0,
  }
  if (eventEpoch) reply.eventEpoch = eventEpoch
  return JSON.stringify({
    ...reply,
  })
}

export class MonotonicEventCursor {
  constructor(initial = 0) {
    if (!validSequence(initial)) throw new RangeError('event cursor must be a non-negative safe integer')
    this._value = initial
  }

  get value() { return this._value }

  accept(sequence) {
    if (!validSequence(sequence)) throw new RangeError('event sequence must be a non-negative safe integer')
    if (sequence <= this._value) return false
    this._value = sequence
    return true
  }

  replace(sequence) {
    if (!validSequence(sequence)) throw new RangeError('event sequence must be a non-negative safe integer')
    this._value = sequence
  }
}

export class ResumableStudioTransport {
  constructor({ onResumeError } = {}) {
    this._raw = null
    this._binding = null
    this._bindingGeneration = 0
    this._active = false
    this._lastSeq = 0
    this._eventEpoch = ''
    this._resumeAttempt = 0
    this._pendingResume = null
    this._pendingCalls = new Set()
    this._eventsReady = false
    this._captureEvents = false
    this._bufferedEvents = []
    this._event = createSignal()
    this._resumeError = createSignal()
    this._reconnected = createSignal()
    this._onResumeError = typeof onResumeError === 'function'
      ? onResumeError
      : (error) => console.error('[studio] event resume failed', error)

    this.proxy = new Proxy({}, {
      get: (_target, property) => {
        if (property === '_rawStudio') return this._raw
        if (property === 'bindingGeneration') return this._bindingGeneration
        if (property === 'event') return this._event
        if (property === 'resumeError') return this._resumeError
        if (property === 'reconnected') return this._reconnected
        if (property === 'describe' || property === 'invoke') {
          return (...args) => this._requestCall(String(property), args)
        }
        if (property === 'resume') {
          return (afterSeq, callback) => this.resume(afterSeq, callback)
        }
        if (property === 'acknowledge') {
          return (sequence) => this.acknowledge(sequence)
        }
        if (property === 'replaceCursor') {
          return (sequence, eventEpoch, generation) => this.replaceCursor(sequence, eventEpoch, generation)
        }
        if (property === 'confirmEpoch') {
          return (eventEpoch, generation) => this.confirmEpoch(eventEpoch, generation)
        }
        if (property === 'releaseEvents') {
          return (eventEpoch, generation) => this.releaseEvents(eventEpoch, generation)
        }
        const value = this._raw?.[property]
        if (typeof value !== 'function') return value
        return (...args) => {
          const current = this._raw
          const fn = current?.[property]
          if (typeof fn !== 'function') {
            throw new Error(`Studio transport disconnected: ${String(property)}`)
          }
          return fn.apply(current, args)
        }
      },
      ownKeys: () => {
        const rawKeys = this._raw ? Reflect.ownKeys(this._raw) : []
        return [...new Set([
          ...rawKeys,
          'bindingGeneration',
          'event',
          'resumeError',
          'reconnected',
          'describe',
          'invoke',
          'resume',
          'acknowledge',
          'replaceCursor',
          'confirmEpoch',
          'releaseEvents',
        ])]
      },
      getOwnPropertyDescriptor: () => ({ enumerable: true, configurable: true }),
    })
  }

  get connected() { return this._raw !== null }
  get lastSeq() { return this._lastSeq }
  get eventEpoch() { return this._eventEpoch }

  bind(raw) {
    this.detach()
    this._raw = raw || null
    const generation = this._bindingGeneration
    const signal = this._raw?.event
    if (signal && typeof signal.connect === 'function') {
      const handler = (...args) => this._handleRawEvent(generation, args)
      signal.connect(handler)
      this._binding = { signal, handler }
    }
    if (this._raw) this._reconnected.emit(generation)
  }

  detach() {
    const detachedRaw = this._raw
    this._bindingGeneration += 1
    this._resumeAttempt += 1
    if (this._binding) {
      try { this._binding.signal.disconnect(this._binding.handler) } catch {}
      this._binding = null
    }
    this._raw = null
    this._eventsReady = false
    this._captureEvents = false
    this._bufferedEvents = []
    for (const pending of [...this._pendingCalls]) {
      if (pending.raw !== detachedRaw) continue
      const resultUnknown = pending.method === 'invoke' && pending.dispatched
      this._settleCall(pending, controlError(
        resultUnknown ? 'RESULT_UNKNOWN' : 'UNAVAILABLE',
        resultUnknown
          ? '연결이 끊겨 요청 처리 결과를 확인할 수 없습니다. 요청은 자동 재실행되지 않습니다.'
          : 'Studio transport 연결이 끊겼습니다.',
        {
          requestId: pending.requestId,
          eventEpoch: this._eventEpoch,
          retryable: false,
        },
      ))
    }
  }

  acknowledge(sequence) {
    if (!validSequence(sequence)) throw new RangeError('event sequence must be a non-negative safe integer')
    this._lastSeq = Math.max(this._lastSeq, sequence)
  }

  replaceCursor(sequence, eventEpoch = '', generation = this._bindingGeneration) {
    if (!validSequence(sequence)) throw new RangeError('event sequence must be a non-negative safe integer')
    if (generation !== this._bindingGeneration) return false
    this._lastSeq = sequence
    if (eventEpoch) this._eventEpoch = this._requireEpoch(eventEpoch)
    return true
  }

  confirmEpoch(eventEpoch, generation = this._bindingGeneration) {
    const nextEpoch = this._requireEpoch(eventEpoch)
    if (generation !== this._bindingGeneration) {
      return { changed: false, resumedPending: false, stale: true }
    }
    if (!this._raw) throw new Error('Studio transport disconnected')
    const previousEpoch = this._eventEpoch
    const changed = Boolean(previousEpoch && previousEpoch !== nextEpoch)
    this._eventEpoch = nextEpoch
    this._eventsReady = false
    this._captureEvents = false
    this._bufferedEvents = []

    if (changed) {
      if (this._pendingResume && !this._pendingResume.settled) {
        this._pendingResume.generation = this._bindingGeneration
        this._settlePending(
          this._pendingResume,
          controlError(
            'EPOCH_MISMATCH',
            'Studio event journal instance가 변경되었습니다.',
            { eventEpoch: nextEpoch, retryable: true },
          ),
        )
      }
      return { changed: true, resumedPending: false, stale: false }
    }

    const pending = this._pendingResume
    if (pending && !pending.settled) {
      this._requestResume(pending)
      return { changed: false, resumedPending: true, stale: false }
    }
    return { changed: false, resumedPending: false, stale: false }
  }

  releaseEvents(eventEpoch, generation = this._bindingGeneration) {
    const expectedEpoch = this._requireEpoch(eventEpoch)
    if (
      generation !== this._bindingGeneration
      || !this._raw
      || expectedEpoch !== this._eventEpoch
    ) return false
    const boundRaw = this._raw
    this._eventsReady = false
    this._captureEvents = true
    while (this._bufferedEvents.length) {
      const batch = this._bufferedEvents.splice(0)
      batch.sort((left, right) => this._eventSequence(left) - this._eventSequence(right))
      for (const args of batch) {
        this._event.emit(...args)
        if (
          generation !== this._bindingGeneration
          || boundRaw !== this._raw
        ) return false
      }
    }
    if (generation !== this._bindingGeneration || boundRaw !== this._raw) return false
    this._captureEvents = false
    this._eventsReady = true
    return true
  }

  resume(afterSeq, callback) {
    if (!validSequence(afterSeq)) {
      const response = this._tagResponse(
        controlError('INVALID_CURSOR', 'event cursor must be a non-negative safe integer'),
        this._bindingGeneration,
      )
      if (typeof callback === 'function') callback(response)
      else throw new RangeError('event cursor must be a non-negative safe integer')
      return
    }
    this._active = true
    this._lastSeq = Math.max(this._lastSeq, afterSeq)
    if (typeof callback === 'function') {
      if (this._pendingResume && !this._pendingResume.settled) {
        this._settlePending(
          this._pendingResume,
          controlError('SUPERSEDED', 'a newer event resume request replaced this request'),
        )
      }
      const pending = { afterSeq, callback, settled: false }
      this._pendingResume = pending
      this._requestResume(pending)
      return
    }
    this._requestResume()
  }

  _requestResume(pending = null) {
    const raw = this._raw
    const fn = raw?.resume
    if (pending) pending.generation = this._bindingGeneration
    if (typeof fn !== 'function') {
      const error = new Error('Studio transport에 resume이 없습니다.')
      if (pending) {
        this._settlePending(pending, controlError('UNAVAILABLE', error.message))
      } else {
        this._reportResumeError(controlError('UNAVAILABLE', error.message), error)
      }
      return
    }
    const requestedSeq = pending ? pending.afterSeq : this._lastSeq
    const attempt = ++this._resumeAttempt
    const generation = this._bindingGeneration
    this._eventsReady = false
    this._captureEvents = true
    this._bufferedEvents = []
    try {
      fn.call(raw, requestedSeq, (response) => {
        if (raw !== this._raw || attempt !== this._resumeAttempt) return
        let value = null
        try { value = typeof response === 'string' ? JSON.parse(response) : response } catch {}
        if (!value || value.status !== 'ok') this._discardBufferedEvents()
        if (pending) {
          this._settlePending(pending, response)
          return
        }
        try {
          if (!value || value.status !== 'ok') {
            this._reportResumeError(
              typeof response === 'string' ? response : JSON.stringify(response),
              new Error(value?.error?.message || 'Studio event resume failed'),
            )
          } else if (value.eventEpoch) {
            this.releaseEvents(value.eventEpoch, generation)
          }
        } catch (error) {
          this._reportResumeError(
            controlError('PROTOCOL_ERROR', error instanceof Error ? error.message : String(error)),
            error,
          )
        }
      })
    } catch (error) {
      if (raw !== this._raw || attempt !== this._resumeAttempt) return
      this._discardBufferedEvents()
      if (pending) {
        this._settlePending(
          pending,
          controlError('UNAVAILABLE', error instanceof Error ? error.message : String(error)),
        )
      } else {
        this._reportResumeError(
          controlError('UNAVAILABLE', error instanceof Error ? error.message : String(error)),
          error,
        )
      }
    }
  }

  _settlePending(pending, response) {
    if (!pending || pending.settled || this._pendingResume !== pending) return
    pending.settled = true
    this._pendingResume = null
    pending.callback(this._tagResponse(
      response,
      Number.isSafeInteger(pending.generation)
        ? pending.generation
        : this._bindingGeneration,
    ))
  }

  _requestCall(method, callArgs) {
    const args = [...callArgs]
    const callback = typeof args[args.length - 1] === 'function' ? args.pop() : null
    const requestId = method === 'invoke' ? this._requestIdFromEnvelope(args[0]) : ''
    const raw = this._raw
    const fn = raw?.[method]
    if (typeof callback !== 'function') {
      if (typeof fn !== 'function') throw new Error(`Studio transport disconnected: ${method}`)
      return fn.apply(raw, args)
    }
    if (typeof fn !== 'function') {
      callback(this._tagResponse(
        controlError('UNAVAILABLE', `Studio transport에 ${method}이(가) 없습니다.`, {
          requestId,
          eventEpoch: this._eventEpoch,
          retryable: true,
        }),
        this._bindingGeneration,
      ))
      return undefined
    }

    const pending = {
      method,
      requestId,
      callback,
      raw,
      generation: this._bindingGeneration,
      dispatched: false,
      settled: false,
    }
    this._pendingCalls.add(pending)
    try {
      pending.dispatched = true
      return fn.apply(raw, [...args, (response) => {
        if (
          !this._pendingCalls.has(pending)
          || raw !== this._raw
          || pending.generation !== this._bindingGeneration
        ) return
        this._settleCall(pending, response)
      }])
    } catch (error) {
      this._settleCall(pending, controlError(
        'UNAVAILABLE',
        error instanceof Error ? error.message : String(error),
        { requestId, eventEpoch: this._eventEpoch, retryable: true },
      ))
      return undefined
    }
  }

  _settleCall(pending, response) {
    if (!pending || pending.settled || !this._pendingCalls.has(pending)) return
    pending.settled = true
    this._pendingCalls.delete(pending)
    pending.callback(this._tagResponse(response, pending.generation))
  }

  _handleRawEvent(generation, args) {
    if (!this._raw || generation !== this._bindingGeneration) return
    const rawEvent = args[0]
    if (this._eventEpoch && typeof rawEvent === 'string') {
      try {
        const event = JSON.parse(rawEvent)
        if (event?.eventEpoch && event.eventEpoch !== this._eventEpoch) return
      } catch {}
    }
    if (this._eventsReady) {
      this._event.emit(...args)
    } else if (this._captureEvents) {
      this._bufferedEvents.push(args)
    }
  }

  _discardBufferedEvents() {
    this._captureEvents = false
    this._eventsReady = false
    this._bufferedEvents = []
  }

  _requireEpoch(value) {
    const eventEpoch = typeof value === 'string' ? value.trim() : ''
    if (!eventEpoch) throw new TypeError('event epoch must be a non-empty string')
    return eventEpoch
  }

  _requestIdFromEnvelope(raw) {
    try {
      const value = typeof raw === 'string' ? JSON.parse(raw) : raw
      return value && typeof value === 'object' ? String(value.requestId || '') : ''
    } catch {
      return ''
    }
  }

  _eventSequence(args) {
    try {
      const value = typeof args?.[0] === 'string' ? JSON.parse(args[0]) : args?.[0]
      const sequence = Number(value?.seq)
      return validSequence(sequence) ? sequence : Number.MAX_SAFE_INTEGER
    } catch {
      return Number.MAX_SAFE_INTEGER
    }
  }

  _tagResponse(response, generation) {
    try {
      const value = typeof response === 'string' ? JSON.parse(response) : response
      if (!value || typeof value !== 'object' || Array.isArray(value)) return response
      const tagged = { ...value, transportGeneration: generation }
      return typeof response === 'string' ? JSON.stringify(tagged) : tagged
    } catch {
      return response
    }
  }

  _reportResumeError(response, error) {
    this._resumeError.emit(response)
    if (this._resumeError.size === 0) this._onResumeError(error)
  }
}

/**
 * studio transport 가 있으면 native 클라이언트, 없으면(목 모드) 아무것도 지원하지 않는 클라이언트.
 * native handshake 실패는 그대로 전파한다 — 조용히 다른 경로로 떨어지지 않는다(fail closed).
 */
export function selectStudioClient(transport, createNative, createUnavailable) {
  return transport == null ? createUnavailable() : createNative(transport)
}

export async function recoverExpiredCursor({
  bootstrap,
  reconcile,
  resume,
  isExpired,
  maxAttempts = 2,
}) {
  if (![bootstrap, reconcile, resume, isExpired].every((value) => typeof value === 'function')) {
    throw new TypeError('cursor recovery callbacks are required')
  }
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1 || maxAttempts > 10) {
    throw new RangeError('maxAttempts must be an integer between 1 and 10')
  }

  let lastError = null
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const state = await bootstrap()
    if (!state || !validSequence(state.eventCursor)) {
      throw new Error('sync.bootstrap description.eventCursor가 올바르지 않습니다.')
    }
    await reconcile(state)
    try {
      await resume(state.eventCursor, state)
      return state
    } catch (error) {
      lastError = error
      if (!isExpired(error) || attempt + 1 >= maxAttempts) throw error
    }
  }
  throw lastError || new Error('event cursor recovery failed')
}

export async function finishResumeWithRecovery({
  resume,
  recover,
  isRecoverable,
  guard = () => {},
}) {
  if (!resume || typeof resume.then !== 'function') {
    throw new TypeError('an in-flight resume promise is required')
  }
  if (![recover, isRecoverable, guard].every((value) => typeof value === 'function')) {
    throw new TypeError('resume recovery callbacks are required')
  }
  try {
    await resume
  } catch (error) {
    guard()
    if (!isRecoverable(error)) throw error
    await recover()
  }
  guard()
}

function recoveryCancelledError() {
  return Object.assign(new Error('cursor recovery was cancelled'), {
    code: 'RECOVERY_CANCELLED',
  })
}

export class CursorRecoveryController {
  constructor(options) {
    this._options = { ...options }
    this._running = null
    this._generation = 0
    this._disposed = false
  }

  recover() {
    if (this._disposed) return Promise.reject(recoveryCancelledError())
    if (this._running) return this._running
    const generation = this._generation
    const guard = () => {
      if (this._disposed || generation !== this._generation) {
        throw recoveryCancelledError()
      }
    }
    const guarded = {
      ...this._options,
      bootstrap: async () => {
        guard()
        const state = await this._options.bootstrap()
        guard()
        return state
      },
      reconcile: async (state) => {
        guard()
        await this._options.reconcile(state)
        guard()
      },
      resume: async (cursor, state) => {
        guard()
        await this._options.resume(cursor, state)
        guard()
      },
    }
    const operation = recoverExpiredCursor(guarded)
    let tracked
    tracked = operation.then(
      () => {
        if (this._running === tracked) this._running = null
      },
      (error) => {
        if (this._running === tracked) this._running = null
        throw error
      },
    )
    this._running = tracked
    return tracked
  }

  reset() {
    if (this._disposed) return
    this._generation += 1
    this._running = null
  }

  dispose() {
    if (this._disposed) return
    this._disposed = true
    this._generation += 1
  }
}
