"""메모 동기화 한 번(pass) — 로컬 저장소 ↔ Forge 메모 라우트. Qt 를 모르는 순수 모듈.

순서:
  1. 서버 목록을 삭제 표시까지 받는다(GET ?include_deleted=1).
  2. 메모마다 :func:`core.memo_store.plan_memo_merge` 로 할 일을 정한다.
  3. 네트워크가 필요 없는 일(서버 것 들이기·보낼 것 없는 삭제 표시 정리)을 먼저 한꺼번에 하고
     파일에 한 번 쓴다 — 첫 동기화에 메모 수백 개를 하나씩 fsync 하지 않게.
  4. PUT/DELETE 를 하나씩 보내고, 응답마다 저장소에 반영한다(중간에 끊겨도 한 일은 남는다).

저장소 잠금은 네트워크 요청 동안 잡지 않는다 — 그동안 사용자가 같은 메모를 저장할 수 있다.
그래서 반영할 때마다 '계획 당시 사본(expected)'과 지금 로컬을 대조한다(core.memo_store 참조).
409(서버 것이 더 새것)는 둘 다 남기는 충돌로 처리하고, 만든 충돌 사본은 같은 pass 안에서 올린다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.forge_memo_client import (
    MemoConflictError, MemoNotFoundError, MemoRejectedError, MemoRemoteError,
)
from core.memo_store import (
    ADOPT_DELETE, ADOPT_REMOTE, CONFLICT_COPY, MARK_CLEAN, PUSH, PUSH_DELETE,
    MemoStore, iter_ids, plan_memo_merge,
)

# 충돌 사본을 올리다 또 충돌하는 일이 되풀이돼도 pass 가 끝나게 하는 상한.
_MAX_CONFLICT_ROUNDS = 3


@dataclass
class MemoSyncReport:
    revision: int = 0
    pulled: int = 0
    pushed: int = 0
    deleted: int = 0
    conflicts: list = field(default_factory=list)   # 충돌 사본을 만든 원래 메모 id
    # 서버가 받지 않은 메모 (id, 사유) — 이 메모만 dirty 로 남고 나머지 동기화는 계속한다.
    rejected: list = field(default_factory=list)
    synced_at: Optional[str] = None


def _push(store: MemoStore, client, server: str, memo: dict, remote: Optional[dict],
          report: MemoSyncReport, rounds: int = 0) -> None:
    """로컬 메모 하나를 PUT. 409 면 서버 것을 들이고 로컬 내용을 충돌 사본으로 올린다."""
    try:
        revision, saved = client.put_memo(
            memo["id"], title=memo.get("title", ""), text=memo.get("text", ""),
            updated_at=memo.get("updated_at") or None,
            base_updated_at=(remote or {}).get("updated_at") or None,
            created_at=memo.get("created_at") or None)
    except MemoRejectedError as exc:
        report.rejected.append((memo["id"], str(exc)))
        return
    except MemoConflictError as exc:
        if exc.memo is None or exc.memo.get("id") != memo["id"]:
            raise MemoRemoteError("Forge 가 충돌을 알렸지만 서버 메모를 보내지 않았습니다") from exc
        _resolve_conflict(store, client, server, exc.memo, report, rounds + 1)
        return
    report.revision = max(report.revision, revision)
    store.record_push(server, saved, expected=memo)
    report.pushed += 1


def _resolve_conflict(store: MemoStore, client, server: str, remote: dict,
                      report: MemoSyncReport, rounds: int) -> None:
    copy_memo = store.resolve_conflict(server, remote)
    if copy_memo is None:
        current = store.get(remote["id"])
        if current is not None and current.get("deleted") and not remote.get("deleted"):
            # 보내는 동안 지웠고 그 삭제가 서버 편집보다 새것이다 — 저장소가 삭제 표시를 지켰다. 서버 판 위에 보낸다.
            _push_delete(store, client, server, current, remote, report)
        return
    report.conflicts.append(remote["id"])
    if rounds > _MAX_CONFLICT_ROUNDS:
        return   # 사본은 로컬에 dirty 로 남아 다음 동기화가 올린다
    _push(store, client, server, copy_memo, None, report, rounds)


def _push_delete(store: MemoStore, client, server: str, local: dict, remote: Optional[dict],
                 report: MemoSyncReport, rounds: int = 0) -> None:
    """로컬 삭제를 DELETE — 계획 때 본 서버 판을 base 로 붙인다.

    GET 뒤에 다른 곳(확장 UI)에서 고쳤으면 확장이 409 로 거절한다. 그 판으로 다시 정해, 삭제가 더
    새것이면 다시 지우고 편집이 더 새것이면 들인다(삭제 표시는 더 새것일 때만 이긴다).
    """
    try:
        revision, tombstone = client.delete_memo(
            local["id"], base_updated_at=(remote or {}).get("updated_at") or None)
    except MemoNotFoundError:
        store.mark_clean(local["id"], expected=local)
        return
    except MemoRejectedError as exc:
        report.rejected.append((local["id"], str(exc)))
        return
    except MemoConflictError as exc:
        newer = exc.memo
        if newer is None or newer.get("id") != local["id"]:
            raise MemoRemoteError("Forge 가 충돌을 알렸지만 서버 메모를 보내지 않았습니다") from exc
        base = (remote or {}).get("updated_at")
        if plan_memo_merge(local, newer, base) == PUSH_DELETE:
            if rounds < _MAX_CONFLICT_ROUNDS:
                _push_delete(store, client, server, local, newer, report, rounds + 1)
            return   # 삭제 표시는 로컬에 dirty 로 남아 다음 동기화가 다시 보낸다
        if store.adopt_remote(server, newer, expected=local):
            report.pulled += 1
        return
    report.revision = max(report.revision, revision)
    store.record_push(server, tombstone, expected=local)
    report.deleted += 1


def run_memo_sync(store: MemoStore, client, server: str) -> MemoSyncReport:
    """한 번 동기화한다. 네트워크·서버 오류는 core.forge_memo_client 의 예외로 그대로 올린다."""
    remote = client.list_memos(include_deleted=True)
    report = MemoSyncReport(revision=remote.revision)
    remote_map = {memo["id"]: memo for memo in remote.memos}
    local_memos = store.all_memos()
    local_map = {memo["id"]: memo for memo in local_memos}
    bases = store.sync_bases(server)
    may_prune = store.remote_may_have_pruned(
        server, remote_count=len(remote.memos), remote_revision=remote.revision)

    network = []
    touched = False
    for memo_id in iter_ids(local_memos, remote.memos):
        local, theirs = local_map.get(memo_id), remote_map.get(memo_id)
        action = plan_memo_merge(local, theirs, bases.get(memo_id), remote_may_prune=may_prune)
        if action == ADOPT_REMOTE:
            if store.adopt_remote(server, theirs, expected=local, persist=False):
                report.pulled += 1
                touched = True
        elif action == ADOPT_DELETE:
            if store.adopt_remote_delete(server, memo_id, expected=local, persist=False):
                report.pulled += 1
                touched = True
        elif action == MARK_CLEAN:
            touched = store.mark_clean(memo_id, expected=local, persist=False) or touched
        elif action in (PUSH, PUSH_DELETE, CONFLICT_COPY):
            network.append((action, local, theirs))
    if touched:
        store.flush()

    for action, local, theirs in network:
        if action == PUSH:
            _push(store, client, server, local, theirs, report)
        elif action == CONFLICT_COPY:
            _resolve_conflict(store, client, server, theirs, report, 1)
        else:
            _push_delete(store, client, server, local, theirs, report)

    report.synced_at = store.finish_sync(server, revision=report.revision)
    return report


__all__ = ["MemoSyncReport", "run_memo_sync"]
