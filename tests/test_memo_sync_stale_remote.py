"""메모 동기화 — Forge 쪽이 계획할 때 본 판과 달라진 경우.

- 옛 판(.bak 복구 등): 서버 updated_at 이 마지막으로 맞춘 기준보다 옛것이면 들이지 않고 로컬 것을 다시 올린다.
- 정리(prune): 확장은 저장소가 MAX_MEMOS(삭제 표시 포함)를 넘을 때 오래된 삭제 표시를 지운다. 가득 찬 서버에서
  맞췄던 메모가 사라졌으면 지운 것으로 본다(다시 올리면 지운 메모가 되살아난다). 덜 찼거나 revision 이
  줄었으면 서버가 잃어버린 것이라 전처럼 다시 올린다.
- GET 과 DELETE 사이의 편집: DELETE 에 계획 때 본 판(base_updated_at)을 붙여, 그 뒤 고친 메모는 확장이
  409 로 지키고 앱은 그 판으로 다시 정한다.
"""
import copy
import unittest

from core.forge_memo_client import ForgeMemoClient, MemoConflictError, MemoRemoteError
from core.memo_store import (
    ADOPT_DELETE, MARK_CLEAN, MAX_MEMOS, NOOP, PUSH, PUSH_DELETE, parse_timestamp, plan_memo_merge,
)
from tests.test_forge_memo_client import TS, FakeRequest, Response
from tests.test_forge_memo_client import memo as wire_memo
from tests.test_memo_store import memo, ts
from tests.test_memo_sync import SERVER, FakeForge, SyncCase


class RollbackPlanTests(unittest.TestCase):
    def test_remote_older_than_base_loses_to_local(self):
        base = ts(20)
        # 깨끗한 로컬(마지막으로 맞춘 판) — 옛 판을 들이지 않고 다시 올린다
        self.assertEqual(plan_memo_merge(memo("a", updated=20), memo("a", text="old", updated=10), base), PUSH)
        # 그 위에 고친 로컬도 충돌 사본 없이 올린다
        self.assertEqual(plan_memo_merge(memo("a", text="mine", updated=30, dirty=True),
                                         memo("a", text="old", updated=10), base), PUSH)
        # 로컬은 지웠는데 서버는 지우기 전 판으로 돌아갔다
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True),
                                         memo("a", updated=10), ts(30)), PUSH_DELETE)
        # 둘 다 삭제 표시 — 보낼 것이 없다
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True),
                                         memo("a", updated=10, deleted=True), ts(30)), NOOP)
        self.assertEqual(plan_memo_merge(memo("a", updated=30, deleted=True, dirty=True),
                                         memo("a", updated=10, deleted=True), base), MARK_CLEAN)

    def test_missing_remote_is_a_pruned_delete_only_when_the_server_may_prune(self):
        self.assertEqual(plan_memo_merge(memo("a"), None, ts(10), remote_may_prune=True), ADOPT_DELETE)
        self.assertEqual(plan_memo_merge(memo("a"), None, ts(10)), PUSH)
        # 고친 메모·이 서버와 맞춘 적 없는 메모는 잃지 않게 올린다
        self.assertEqual(plan_memo_merge(memo("a", dirty=True), None, ts(10), remote_may_prune=True), PUSH)
        self.assertEqual(plan_memo_merge(memo("a"), None, None, remote_may_prune=True), PUSH)


def _fill_with_tombstones(server, count, prefix="old"):
    for index in range(count):
        memo_id = f"{prefix}-{index}"
        server.memos[memo_id] = {"id": memo_id, "title": "", "text": "", "created_at": ts(1),
                                 "updated_at": ts(2), "deleted": True}


class PrunedTombstoneSyncTests(SyncCase):
    def test_memo_deleted_in_forge_whose_tombstone_was_pruned_stays_deleted(self):
        self.store.save("m1", "t", "지울 메모")
        self.sync()
        self.server.remove("m1")                 # Forge 에서 지웠고
        del self.server.memos["m1"]              # 가득 차서 그 삭제 표시가 정리됐다
        _fill_with_tombstones(self.server, MAX_MEMOS)
        self.server.revision += MAX_MEMOS
        writes = list(self.server.writes)

        self.sync()

        self.assertNotIn("m1", self.server.memos)
        self.assertEqual(self.server.writes, writes)
        self.assertEqual(self.texts(), {})
        self.assertTrue(self.local("m1")["deleted"])
        self.assertFalse(self.local("m1")["dirty"])
        self.assertNotIn("m1", self.store.sync_bases(SERVER))
        # 다음 동기화도 되살리지 않는다
        self.sync()
        self.assertNotIn("m1", self.server.memos)

    def test_edit_made_offline_after_the_prune_is_still_uploaded(self):
        self.store.save("m1", "t", "처음")
        self.sync()
        self.store.save("m1", "t", "오프라인에서 고침", self.local("m1")["updated_at"])
        del self.server.memos["m1"]
        _fill_with_tombstones(self.server, MAX_MEMOS)
        self.server.revision += MAX_MEMOS

        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "오프라인에서 고침")

    def test_reset_server_with_lower_revision_gets_the_memo_back(self):
        self.server.revision = 50
        self.store.save("m1", "t", "keep me")
        self.sync()
        # 데이터 폴더를 새로 만들었고 그사이 가득 찼다 — revision 이 지난 동기화보다 작다
        self.server.memos.clear()
        _fill_with_tombstones(self.server, MAX_MEMOS)
        self.server.revision = 10

        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "keep me")
        self.assertEqual(self.texts(), {"m1": "keep me"})


class RolledBackServerSyncTests(SyncCase):
    def test_server_restored_from_backup_gets_the_newer_copy_back(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        older, older_revision = copy.deepcopy(self.server.memos["m1"]), self.server.revision
        self.store.save("m1", "t", "v2", self.local("m1")["updated_at"])
        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "v2")

        # Forge 의 memos.json 이 깨져 한 판 전의 .bak 을 내준다
        self.server.memos["m1"] = older
        self.server.revision = older_revision

        self.sync()
        self.assertEqual(self.texts(), {"m1": "v2"})
        self.assertEqual(self.server.memos["m1"]["text"], "v2")
        self.assertFalse(self.local("m1")["dirty"])
        # 다시 맞춰졌으니 그다음은 조용하다
        writes = list(self.server.writes)
        self.sync()
        self.assertEqual(self.server.writes, writes)


class DeleteGuardClientTests(unittest.TestCase):
    def test_delete_sends_the_version_it_planned_from(self):
        tomb = wire_memo("m1", text="", deleted=True)
        fake = FakeRequest(Response(200, {"revision": 4, "memo": tomb}),
                           Response(200, {"revision": 5, "memo": tomb}))
        client = ForgeMemoClient("http://127.0.0.1:7860", request=fake)
        client.delete_memo("m1", base_updated_at=TS)
        self.assertEqual(fake.calls[0][2]["params"], {"base_updated_at": TS})
        client.delete_memo("m1")
        self.assertNotIn("params", fake.calls[1][2])

    def test_409_on_delete_carries_the_newer_memo(self):
        fake = FakeRequest(Response(409, {"detail": "changed", "memo": wire_memo("m1", text="newer")}),
                           Response(409, {"detail": "Unsupported memo schema version 2"}))
        client = ForgeMemoClient("http://127.0.0.1:7860", request=fake)
        with self.assertRaises(MemoConflictError) as caught:
            client.delete_memo("m1", base_updated_at=TS)
        self.assertEqual(caught.exception.memo["text"], "newer")
        # 메모 없는 409 는 더 새 schema — 원격 오류
        with self.assertRaises(MemoRemoteError):
            client.delete_memo("m1", base_updated_at=TS)


class RacingForge(FakeForge):
    """DELETE 직전 다른 곳(확장 UI)의 쓰기를 흉내 낸다."""

    before_delete = None

    def delete_memo(self, memo_id, *, base_updated_at=None):
        if self.before_delete is not None:
            hook, self.before_delete = self.before_delete, None
            hook(memo_id)
        return super().delete_memo(memo_id, base_updated_at=base_updated_at)


class DeleteRaceSyncTests(SyncCase):
    def setUp(self):
        super().setUp()
        self.server = RacingForge(self.clock)

    def test_forge_edit_between_list_and_delete_is_kept(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        self.store.delete("m1")
        self.server.before_delete = lambda memo_id: self.server.edit(memo_id, "raced forge edit")

        self.sync()

        self.assertFalse(self.server.memos["m1"]["deleted"])
        self.assertEqual(self.server.memos["m1"]["text"], "raced forge edit")
        self.assertEqual(self.texts(), {"m1": "raced forge edit"})
        self.assertFalse(self.local("m1")["dirty"])

    def test_newer_local_delete_is_sent_again_on_the_newer_version(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        planned_from = self.server.memos["m1"]["updated_at"]
        self.store.delete("m1")

        def older_forge_edit(memo_id):
            # 계획 때 본 판보다 새것이지만, 로컬 삭제보다는 옛것인 편집
            moment = parse_timestamp(planned_from).replace(microsecond=500_000)
            self.server.memos[memo_id].update(text="older forge edit", updated_at=moment.isoformat())

        self.server.before_delete = older_forge_edit
        report = self.sync()

        self.assertTrue(self.server.memos["m1"]["deleted"])
        self.assertEqual(report.deleted, 1)
        self.assertEqual(self.texts(), {})
        self.assertFalse(self.local("m1")["dirty"])


if __name__ == "__main__":
    unittest.main()
