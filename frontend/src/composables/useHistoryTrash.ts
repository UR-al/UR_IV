import { computed, ref, shallowRef, type Ref } from 'vue'
import { applyDeleteToHistory, isSameImagePath, parseImageDeleteResult } from '../utils/imageDeleteResult'

export interface HistoryDeleteRequest {
  path: string
  undoable: true
  request_id: string
}

export interface HistoryRestoreRequest {
  undo_token: string
  request_id: string
}

export interface HistoryTrashDeps {
  history: Ref<string[]>
  current: Ref<string>
  page: Ref<number>
  perPage: number
  sendDelete: (payload: HistoryDeleteRequest) => void
  sendRestore: (payload: HistoryRestoreRequest) => void
  /** 선택·EXIF 갱신. 빈 경로는 마지막 이미지가 삭제되었다는 뜻이다. */
  onSelection: (path: string) => void
  onChanged?: () => void
  /** 연결 단절로 결과가 오지 않을 때 요청 잠금을 푸는 시간. 기본 30초. */
  timeoutMs?: number
  onTimeout?: () => void
}

interface HistoryPosition {
  path: string
  index: number
  before?: string
  after?: string
}

interface UndoEntry extends HistoryPosition {
  token: string
  order: number
}

type PendingRequest =
  | { kind: 'delete'; id: string; position: HistoryPosition; order: number }
  | { kind: 'restore'; id: string; entry: UndoEntry }

function requestNonce(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}

function resultObject(raw: unknown): Record<string, unknown> | null {
  try {
    const value: unknown = typeof raw === 'string' ? JSON.parse(raw) : raw
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch { return null }
}

/**
 * 휴지통 요청의 결과가 온 뒤에만 히스토리를 바꾼다. Undo 는 이 히스토리에서 요청하여
 * 성공한 삭제만 추적하며, 실제 파일 복구는 백엔드의 불투명 토큰으로 요청한다.
 * 갤러리 등 다른 화면의 삭제 결과도 반영하되 그 화면의 Undo 소유권은 가져오지 않는다.
 */
export function useHistoryTrash(deps: HistoryTrashDeps) {
  const active = shallowRef<PendingRequest | null>(null)
  const undoStack = ref<UndoEntry[]>([])
  const requestPrefix = `history-trash:${requestNonce()}:`
  const pageSize = Math.max(1, Math.floor(deps.perPage) || 1)
  const timeoutMs = Number.isFinite(deps.timeoutMs) && Number(deps.timeoutMs) > 0 ? Number(deps.timeoutMs) : 30_000
  // 타임아웃은 연결 실패의 증거가 아니므로 늦은 성공 결과도 받아야 한다.
  // 무한 연결 단절 중 메모리가 커지지 않도록 최근 미해결 요청만 보존한다.
  const requests = new Map<string, PendingRequest>()
  const restoredTokens = new Set<string>()
  let timer: ReturnType<typeof setTimeout> | null = null
  let disposed = false
  let sequence = 0
  const pending = computed(() => active.value !== null)
  const canUndo = computed(() => !disposed && !pending.value && undoStack.value.length > 0)

  function beginRequest(request: PendingRequest): void {
    active.value = request
    requests.set(request.id, request)
    if (requests.size > 60) requests.delete(requests.keys().next().value!)
    timer = setTimeout(() => {
      if (disposed || active.value?.id !== request.id) return
      timer = null
      active.value = null
      deps.onTimeout?.()
    }, timeoutMs)
  }

  function finishRequest(id: string): void {
    requests.delete(id)
    if (active.value?.id !== id) return
    if (timer !== null) clearTimeout(timer)
    timer = null
    active.value = null
  }

  function removeUndoToken(token: string): void {
    undoStack.value = undoStack.value.filter(entry => entry.token !== token)
  }

  function dispose(): void {
    disposed = true
    if (timer !== null) clearTimeout(timer)
    timer = null
    active.value = null
    requests.clear()
    undoStack.value = []
    restoredTokens.clear()
  }

  function deleteImage(path: string): boolean {
    if (disposed || pending.value || !path.trim()) return false
    const index = deps.history.value.findIndex(item => isSameImagePath(item, path))
    if (index < 0) return false
    const position: HistoryPosition = {
      path: deps.history.value[index], index,
      before: deps.history.value[index - 1], after: deps.history.value[index + 1],
    }
    const request: PendingRequest = { kind: 'delete', id: requestPrefix + requestNonce(), position, order: ++sequence }
    beginRequest(request)
    try {
      deps.sendDelete({ path: position.path, undoable: true, request_id: request.id })
    } catch (error) {
      finishRequest(request.id)
      throw error
    }
    return true
  }

  function undo(): boolean {
    if (!canUndo.value) return false
    const entry = undoStack.value[undoStack.value.length - 1]
    const request: PendingRequest = { kind: 'restore', id: requestPrefix + requestNonce(), entry }
    beginRequest(request)
    try {
      deps.sendRestore({ undo_token: entry.token, request_id: request.id })
    } catch (error) {
      finishRequest(request.id)
      throw error
    }
    return true
  }

  function onDeleteResult(raw: unknown): void {
    if (disposed) return
    const data = resultObject(raw)
    if (!data || typeof data.ok !== 'boolean' || typeof data.removed !== 'boolean') return
    const result = parseImageDeleteResult(data)
    if (!result) return
    const requestId = typeof data.request_id === 'string' ? data.request_id : ''
    const request = requests.get(requestId)
    const own = request?.kind === 'delete'
    // 중복·지연 응답으로 복구한 파일을 다시 숨기거나 다른 요청의 pending 을 풀지 않는다.
    if (requestId.startsWith(requestPrefix) && !own) return
    if (own) {
      if (!isSameImagePath(result.path, request.position.path)) return
      finishRequest(request.id)
      if (result.ok && result.removed && typeof data.undo_token === 'string' && data.undo_token.trim()) {
        if (restoredTokens.has(data.undo_token)) return
        removeUndoToken(data.undo_token)
        undoStack.value.push({ ...request.position, token: data.undo_token, order: request.order })
        undoStack.value.sort((left, right) => left.order - right.order)
        if (undoStack.value.length > 30) undoStack.value.shift()
      }
    }
    const next = applyDeleteToHistory(deps.history.value, deps.current.value, result)
    if (!next) return
    deps.history.value = next.history
    const selectionChanged = next.current !== deps.current.value
    deps.current.value = next.current
    const maxPage = Math.max(0, Math.ceil(next.history.length / pageSize) - 1)
    deps.page.value = Math.max(0, Math.min(deps.page.value, maxPage))
    if (selectionChanged) {
      const index = next.history.findIndex(item => isSameImagePath(item, next.current))
      deps.page.value = index < 0 ? 0 : Math.floor(index / pageSize)
      deps.onSelection(next.current)
    }
    deps.onChanged?.()
  }

  function onRestoreResult(raw: unknown): void {
    if (disposed) return
    const data = resultObject(raw)
    if (!data || typeof data.request_id !== 'string') return
    const request = requests.get(data.request_id)
    if (request?.kind !== 'restore') return
    if (data.undo_token !== request.entry.token || typeof data.ok !== 'boolean' || typeof data.restored !== 'boolean') return
    if (!data.ok || !data.restored) {
      finishRequest(request.id)
      // 이름 충돌·일시적 실패는 다시 시도할 수 있다. 만료·소실된 항목만 제거한다.
      // 토큰이 만료되면 백엔드도 원래 경로를 모르므로 실패의 path 는 비어 있을 수 있다.
      if (data.retryable !== true) removeUndoToken(request.entry.token)
      return
    }
    if (typeof data.path !== 'string' || !isSameImagePath(data.path, request.entry.path)) return
    finishRequest(request.id)
    removeUndoToken(request.entry.token)
    if (restoredTokens.has(request.entry.token)) return
    restoredTokens.add(request.entry.token)
    if (restoredTokens.size > 60) restoredTokens.delete(restoredTokens.values().next().value!)
    const entry = request.entry
    const list = [...deps.history.value]
    let index = list.findIndex(item => isSameImagePath(item, entry.path))
    if (index < 0) {
      const before = list.findIndex(item => isSameImagePath(item, entry.before))
      const after = list.findIndex(item => isSameImagePath(item, entry.after))
      // 삭제 후 새 생성물이 맨 앞에 추가되어도 원래 이웃 사이에 복원한다.
      index = before >= 0 ? before + 1 : after >= 0 ? after : Math.min(entry.index, list.length)
      list.splice(index, 0, entry.path)
      deps.history.value = list
    }
    const selected = list[index]
    deps.current.value = selected
    deps.page.value = Math.floor(index / pageSize)
    deps.onSelection(selected)
    deps.onChanged?.()
  }

  return { deleteImage, undo, onDeleteResult, onRestoreResult, pending, canUndo, dispose }
}
