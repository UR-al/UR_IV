// utils/queueLocks.ts
// 대기열 행 잠금 규칙 — 지금 생성 중인 항목(processing_index)은 지우거나 옮기지 못한다.
// 백엔드(core/queue_model.py QueueModel)가 같은 규칙으로 막는다. 여기서는 버튼을 미리 끄고,
// '선택 삭제'에서 생성 중인 항목을 빼 사용자가 거부 토스트를 보지 않게 한다.
// processingIdx 가 -1 이면 생성 중인 대기열 항목이 없다.

export function isRowLocked(i: number, processingIdx: number): boolean {
  return processingIdx >= 0 && i === processingIdx
}

/** i 번 행을 한 칸 위로 — 첫 행이 아니고, 자신도 바로 위 행도 생성 중이 아니어야 한다. */
export function canMoveUp(i: number, processingIdx: number): boolean {
  if (i <= 0) return false
  return !isRowLocked(i, processingIdx) && !isRowLocked(i - 1, processingIdx)
}

/** i 번 행을 한 칸 아래로 — 마지막 행이 아니고, 자신도 바로 아래 행도 생성 중이 아니어야 한다. */
export function canMoveDown(i: number, length: number, processingIdx: number): boolean {
  if (i < 0 || i >= length - 1) return false
  return !isRowLocked(i, processingIdx) && !isRowLocked(i + 1, processingIdx)
}

/** 삭제 요청에 보낼 id — 생성 중인 항목의 id 는 뺀다. */
export function removableIds(ids: Iterable<string>, processingId: string | null | undefined): string[] {
  const out: string[] = []
  for (const id of ids) {
    if (id && id !== processingId) out.push(id)
  }
  return out
}

/** 생성 중인 행만 남아 '전체' 비우기로 지울 것이 없을 때 알림 */
export const ONLY_RUNNING_ROW_NOTICE = '생성 중인 항목은 지울 수 없습니다 — 생성이 끝난 뒤 지우세요'

/**
 * '전체' 비우기 확인 문구 — 생성 중인 항목은 백엔드가 남기므로 개수에서 뺀다
 * (예전엔 N개를 모두 지운다고 묻고 한 줄을 조용히 남겼다). 지울 것이 없으면 null.
 */
export function clearConfirmMessage(length: number, processingIdx: number): string | null {
  const locked = processingIdx >= 0 && processingIdx < length ? 1 : 0
  const removable = Math.max(0, length - locked)
  if (removable === 0) return null
  return locked
    ? `생성 중인 1개를 뺀 ${removable}개 항목을 삭제할까요?`
    : `대기열 ${removable}개 항목을 모두 삭제할까요?`
}
