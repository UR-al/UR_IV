import assert from 'node:assert/strict'
import test from 'node:test'

import {
  CursorRecoveryController,
  MonotonicEventCursor,
  ResumableStudioTransport,
  createSignal,
  finishResumeWithRecovery,
  recoverExpiredCursor,
  selectStudioClient,
} from './resumableTransport.js'

function eventJson(epoch, seq, type = 'progress') {
  return JSON.stringify({
    version: 1,
    eventEpoch: epoch,
    seq,
    topic: 'runtime.operation',
    type,
    data: {},
  })
}

function rawTransport(epoch = 'epoch-a') {
  const event = createSignal()
  const resumeCalls = []
  const describeCalls = []
  const invokeCalls = []
  return {
    event,
    resumeCalls,
    describeCalls,
    invokeCalls,
    describe(callback) {
      describeCalls.push(callback)
      callback(JSON.stringify({
        version: 1,
        eventEpoch: epoch,
        eventCursor: 0,
        operations: [],
        topics: [],
      }))
    },
    invoke(envelope, callback) {
      invokeCalls.push({ envelope, callback })
      callback(JSON.stringify({
        version: 1,
        requestId: JSON.parse(envelope).requestId,
        status: 'ok',
        data: {},
      }))
    },
    resume(afterSeq, callback) {
      resumeCalls.push(afterSeq)
      callback(JSON.stringify({ version: 1, status: 'ok', afterSeq, eventEpoch: epoch }))
    },
  }
}

function resumeAndRelease(transport, afterSeq, epoch) {
  return new Promise((resolve) => {
    transport.proxy.resume(afterSeq, (raw) => {
      const reply = JSON.parse(raw)
      if (reply.status === 'ok') {
        assert.equal(transport.proxy.releaseEvents(epoch, reply.transportGeneration), true)
      }
      resolve(reply)
    })
  })
}

test('same-epoch reconnect gates live events, sorts replay, and drops duplicates by cursor', async () => {
  const transport = new ResumableStudioTransport()
  const cursor = new MonotonicEventCursor(5)
  const received = []
  let second = null
  transport.proxy.event.connect((raw) => {
    const event = JSON.parse(raw)
    if (!cursor.accept(event.seq)) return
    transport.proxy.acknowledge(event.seq)
    received.push(event.seq)
    if (event.seq === 6) second.event.emit(eventJson('epoch-a', 8))
  })

  const first = rawTransport('epoch-a')
  transport.bind(first)
  transport.proxy.confirmEpoch('epoch-a', transport.proxy.bindingGeneration)
  transport.proxy.replaceCursor(5, 'epoch-a', transport.proxy.bindingGeneration)
  await resumeAndRelease(transport, 5, 'epoch-a')

  transport.detach()
  second = rawTransport('epoch-a')
  second.resume = (afterSeq, callback) => {
    second.resumeCalls.push(afterSeq)
    second.event.emit(eventJson('epoch-a', 10))
    for (const seq of [6, 7, 8, 9, 10]) second.event.emit(eventJson('epoch-a', seq))
    callback(JSON.stringify({
      version: 1,
      status: 'ok',
      afterSeq,
      eventEpoch: 'epoch-a',
    }))
  }
  transport.bind(second)
  second.event.emit(eventJson('epoch-a', 11))
  assert.deepEqual(received, [])
  assert.deepEqual(second.resumeCalls, [])

  const confirmation = transport.proxy.confirmEpoch(
    'epoch-a',
    transport.proxy.bindingGeneration,
  )
  assert.deepEqual(confirmation, { changed: false, resumedPending: false, stale: false })
  await resumeAndRelease(transport, 5, 'epoch-a')

  assert.deepEqual(second.resumeCalls, [5])
  assert.deepEqual(received, [6, 7, 8, 9, 10])
  second.event.emit(eventJson('epoch-a', 11, 'completed'))
  assert.deepEqual(received, [6, 7, 8, 9, 10, 11])
})

test('epoch mismatch never resumes an old cursor and only releases the new journal', async () => {
  const transport = new ResumableStudioTransport()
  const first = rawTransport('epoch-a')
  let oldResumeCallback
  first.resume = (afterSeq, callback) => {
    first.resumeCalls.push(afterSeq)
    oldResumeCallback = callback
  }
  transport.bind(first)
  transport.proxy.confirmEpoch('epoch-a', transport.proxy.bindingGeneration)
  transport.proxy.replaceCursor(50, 'epoch-a', transport.proxy.bindingGeneration)
  const pending = new Promise((resolve) => transport.proxy.resume(50, (raw) => resolve(JSON.parse(raw))))
  assert.deepEqual(first.resumeCalls, [50])

  transport.detach()
  const second = rawTransport('epoch-b')
  transport.bind(second)
  const generation = transport.proxy.bindingGeneration
  const confirmation = transport.proxy.confirmEpoch('epoch-b', generation)
  assert.deepEqual(confirmation, { changed: true, resumedPending: false, stale: false })
  assert.deepEqual(second.resumeCalls, [])
  const oldReply = await pending
  assert.equal(oldReply.error.code, 'EPOCH_MISMATCH')
  assert.equal(oldReply.eventEpoch, 'epoch-b')
  assert.equal(oldReply.transportGeneration, generation)

  const delivered = []
  transport.proxy.event.connect((raw) => delivered.push(JSON.parse(raw).seq))
  assert.equal(transport.proxy.replaceCursor(60, 'epoch-b', generation), true)
  await resumeAndRelease(transport, 60, 'epoch-b')
  second.event.emit(eventJson('epoch-b', 61))
  assert.deepEqual(second.resumeCalls, [60])
  assert.deepEqual(delivered, [61])

  oldResumeCallback(JSON.stringify({
    version: 1,
    status: 'ok',
    afterSeq: 50,
    eventEpoch: 'epoch-a',
  }))
  assert.deepEqual(delivered, [61])
})

test('detach settles pending describe and invoke once without replaying either call', () => {
  const transport = new ResumableStudioTransport()
  const first = rawTransport()
  let describeCallback
  let invokeCallback
  first.describe = (callback) => {
    first.describeCalls.push(callback)
    describeCallback = callback
  }
  first.invoke = (envelope, callback) => {
    first.invokeCalls.push({ envelope, callback })
    invokeCallback = callback
  }
  transport.bind(first)

  const descriptions = []
  const invocations = []
  transport.proxy.describe((raw) => descriptions.push(JSON.parse(raw)))
  transport.proxy.invoke(JSON.stringify({ requestId: 'r1' }), (raw) => invocations.push(JSON.parse(raw)))
  transport.detach()

  assert.equal(descriptions.length, 1)
  assert.equal(descriptions[0].error.code, 'UNAVAILABLE')
  assert.equal(invocations.length, 1)
  assert.equal(invocations[0].error.code, 'RESULT_UNKNOWN')
  assert.equal(invocations[0].requestId, 'r1')
  assert.equal(invocations[0].error.retryable, false)

  const second = rawTransport('epoch-b')
  transport.bind(second)
  assert.equal(second.describeCalls.length, 0)
  assert.equal(second.invokeCalls.length, 0)
  describeCallback(JSON.stringify({ version: 1 }))
  invokeCallback(JSON.stringify({ version: 1, requestId: 'r1', status: 'ok', data: {} }))
  assert.equal(descriptions.length, 1)
  assert.equal(invocations.length, 1)
})

test('failed resume discards captured events and keeps the gate closed until recovery succeeds', async () => {
  const transport = new ResumableStudioTransport()
  const raw = rawTransport('epoch-a')
  let resumeCount = 0
  raw.resume = (afterSeq, callback) => {
    raw.resumeCalls.push(afterSeq)
    resumeCount += 1
    raw.event.emit(eventJson('epoch-a', resumeCount === 1 ? 6 : 7))
    callback(JSON.stringify(resumeCount === 1
      ? {
          version: 1,
          status: 'error',
          eventEpoch: 'epoch-a',
          error: { code: 'CURSOR_EXPIRED', message: 'expired', retryable: true },
        }
      : { version: 1, status: 'ok', afterSeq, eventEpoch: 'epoch-a' }))
  }
  const delivered = []
  transport.proxy.event.connect((value) => delivered.push(JSON.parse(value).seq))
  transport.bind(raw)
  const generation = transport.proxy.bindingGeneration
  transport.proxy.confirmEpoch('epoch-a', generation)

  const failed = await resumeAndRelease(transport, 5, 'epoch-a')
  assert.equal(failed.error.code, 'CURSOR_EXPIRED')
  raw.event.emit(eventJson('epoch-a', 99))
  assert.deepEqual(delivered, [])

  transport.proxy.replaceCursor(6, 'epoch-a', generation)
  await resumeAndRelease(transport, 6, 'epoch-a')
  assert.deepEqual(delivered, [7])
})

test('same-epoch reconnect inherits a cancelled recovery resume and starts fresh recovery on expiry', async () => {
  const transport = new ResumableStudioTransport()
  const first = rawTransport('epoch-a')
  let heldResumeCallback
  first.resume = (afterSeq, callback) => {
    first.resumeCalls.push(afterSeq)
    heldResumeCallback = callback
  }
  transport.bind(first)
  transport.proxy.confirmEpoch('epoch-a', transport.proxy.bindingGeneration)

  const delivered = []
  transport.proxy.event.connect((raw) => delivered.push(JSON.parse(raw).seq))
  let bootstrapCalls = 0
  let activeResume = null
  const resumeThroughTransport = (afterSeq) => {
    const operation = new Promise((resolve, reject) => {
      transport.proxy.resume(afterSeq, (raw) => {
        const reply = JSON.parse(raw)
        if (reply.status === 'error') {
          reject(Object.assign(new Error(reply.error.message), { code: reply.error.code }))
          return
        }
        if (!transport.proxy.releaseEvents('epoch-a', reply.transportGeneration)) {
          reject(Object.assign(new Error('event gate stayed closed'), { code: 'UNAVAILABLE' }))
          return
        }
        resolve()
      })
    })
    activeResume = operation
    return operation
  }
  const controller = new CursorRecoveryController({
    bootstrap: async () => ({
      eventEpoch: 'epoch-a',
      eventCursor: ++bootstrapCalls === 1 ? 10 : 20,
    }),
    reconcile: async (state) => {
      transport.proxy.replaceCursor(
        state.eventCursor,
        state.eventEpoch,
        transport.proxy.bindingGeneration,
      )
    },
    resume: (cursor) => resumeThroughTransport(cursor),
    isExpired: (error) => error?.code === 'CURSOR_EXPIRED',
    maxAttempts: 2,
  })

  const oldRecovery = controller.recover()
  const oldCancelled = assert.rejects(
    oldRecovery,
    (error) => error?.code === 'RECOVERY_CANCELLED',
  )
  await new Promise((resolve) => setImmediate(resolve))
  assert.deepEqual(first.resumeCalls, [10])
  assert.ok(heldResumeCallback)
  const inheritedResume = activeResume

  transport.detach()
  const second = rawTransport('epoch-a')
  second.resume = (afterSeq, callback) => {
    second.resumeCalls.push(afterSeq)
    if (afterSeq === 10) {
      second.event.emit(eventJson('epoch-a', 11))
      callback(JSON.stringify({
        version: 1,
        status: 'error',
        eventEpoch: 'epoch-a',
        error: { code: 'CURSOR_EXPIRED', message: 'expired', retryable: true },
      }))
      return
    }
    second.event.emit(eventJson('epoch-a', 21))
    callback(JSON.stringify({
      version: 1,
      status: 'ok',
      afterSeq,
      eventEpoch: 'epoch-a',
    }))
  }
  transport.bind(second)
  controller.reset()
  const confirmation = transport.proxy.confirmEpoch(
    'epoch-a',
    transport.proxy.bindingGeneration,
  )
  assert.equal(confirmation.resumedPending, true)
  assert.deepEqual(delivered, [])

  await finishResumeWithRecovery({
    resume: inheritedResume,
    recover: () => controller.recover(),
    isRecoverable: (error) => error?.code === 'CURSOR_EXPIRED',
  })
  await oldCancelled

  assert.equal(bootstrapCalls, 2)
  assert.deepEqual(second.resumeCalls, [10, 20])
  assert.deepEqual(delivered, [21])
  second.event.emit(eventJson('epoch-a', 22, 'completed'))
  assert.deepEqual(delivered, [21, 22])
})

test('a reentrant bind aborts the old drain without opening the new binding gate', async () => {
  const transport = new ResumableStudioTransport()
  const first = rawTransport('epoch-a')
  const second = rawTransport('epoch-a')
  first.resume = (afterSeq, callback) => {
    first.resumeCalls.push(afterSeq)
    first.event.emit(eventJson('epoch-a', 1))
    first.event.emit(eventJson('epoch-a', 2))
    callback(JSON.stringify({
      version: 1,
      status: 'ok',
      afterSeq,
      eventEpoch: 'epoch-a',
    }))
  }

  const seen = []
  transport.proxy.event.connect((raw) => {
    const sequence = JSON.parse(raw).seq
    seen.push(sequence)
    if (sequence === 1) {
      transport.bind(second)
      second.event.emit(eventJson('epoch-a', 99))
    }
  })
  transport.bind(first)
  transport.proxy.confirmEpoch('epoch-a', transport.proxy.bindingGeneration)
  const releaseResult = await new Promise((resolve) => {
    transport.proxy.resume(0, (raw) => {
      const reply = JSON.parse(raw)
      resolve(transport.proxy.releaseEvents('epoch-a', reply.transportGeneration))
    })
  })

  assert.equal(releaseResult, false)
  assert.deepEqual(seen, [1])
  second.event.emit(eventJson('epoch-a', 100))
  assert.deepEqual(seen, [1])
})

test('stale binding generations cannot confirm, replace, or release a newer gate', () => {
  const transport = new ResumableStudioTransport()
  transport.bind(rawTransport('epoch-a'))
  const staleGeneration = transport.proxy.bindingGeneration
  transport.proxy.confirmEpoch('epoch-a', staleGeneration)
  transport.detach()
  transport.bind(rawTransport('epoch-b'))

  assert.equal(transport.proxy.confirmEpoch('epoch-a', staleGeneration).stale, true)
  assert.equal(transport.proxy.replaceCursor(99, 'epoch-a', staleGeneration), false)
  assert.equal(transport.proxy.releaseEvents('epoch-a', staleGeneration), false)
  assert.equal(transport.lastSeq, 0)
  assert.equal(transport.eventEpoch, 'epoch-a')
})

test('native handshake errors propagate and only an absent transport is unavailable', async () => {
  const handshakeError = new Error('invalid Studio description')
  let unavailableCalls = 0
  const unavailable = async () => {
    unavailableCalls += 1
    return { kind: 'unavailable' }
  }

  await assert.rejects(
    selectStudioClient({}, async () => { throw handshakeError }, unavailable),
    handshakeError,
  )
  assert.equal(unavailableCalls, 0)
  assert.deepEqual(
    await selectStudioClient(null, async () => ({ kind: 'v1' }), unavailable),
    { kind: 'unavailable' },
  )
  assert.equal(unavailableCalls, 1)
})

test('cursor recovery reconciles before resume and passes the whole bootstrap state', async () => {
  const order = []
  const bootstrap = {
    eventEpoch: 'epoch-b',
    eventCursor: 42,
    snapshots: { runtime: {}, generationApi: {}, modelPaths: {} },
  }
  const recovered = await recoverExpiredCursor({
    bootstrap: async () => {
      order.push('bootstrap')
      return bootstrap
    },
    reconcile: async (value) => {
      assert.equal(value, bootstrap)
      order.push('reconcile')
    },
    resume: async (cursor, state) => {
      assert.equal(state, bootstrap)
      order.push(`resume:${cursor}:${state.eventEpoch}`)
    },
    isExpired: (error) => error?.code === 'CURSOR_EXPIRED',
  })
  assert.equal(recovered, bootstrap)
  assert.deepEqual(order, ['bootstrap', 'reconcile', 'resume:42:epoch-b'])
})

test('cursor recovery is bounded, single-flight, resettable, and disposal-safe', async () => {
  let releaseBootstrap
  let bootstrapCalls = 0
  let resumeCalls = 0
  const controller = new CursorRecoveryController({
    bootstrap: async () => {
      bootstrapCalls += 1
      await new Promise((resolve) => { releaseBootstrap = resolve })
      return { eventCursor: 10 }
    },
    reconcile: async () => {},
    resume: async () => { resumeCalls += 1 },
    isExpired: () => false,
  })

  const first = controller.recover()
  assert.equal(controller.recover(), first)
  controller.reset()
  releaseBootstrap()
  await assert.rejects(first, (error) => error?.code === 'RECOVERY_CANCELLED')
  assert.equal(resumeCalls, 0)

  const second = controller.recover()
  controller.dispose()
  releaseBootstrap()
  await assert.rejects(second, (error) => error?.code === 'RECOVERY_CANCELLED')
  assert.equal(bootstrapCalls, 2)
})

test('expired cursor recovery retries only to its configured bound', async () => {
  let bootstrapCalls = 0
  let resumeCalls = 0
  const expired = Object.assign(new Error('expired again'), { code: 'CURSOR_EXPIRED' })
  await assert.rejects(recoverExpiredCursor({
    bootstrap: async () => ({ eventCursor: ++bootstrapCalls }),
    reconcile: async () => {},
    resume: async () => {
      resumeCalls += 1
      throw expired
    },
    isExpired: (error) => error?.code === 'CURSOR_EXPIRED',
    maxAttempts: 2,
  }), expired)
  assert.equal(bootstrapCalls, 2)
  assert.equal(resumeCalls, 2)
})
