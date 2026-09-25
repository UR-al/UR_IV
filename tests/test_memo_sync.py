"""core.memo_sync — 로컬 저장소 ↔ (계약대로 움직이는 가짜) Forge 메모 서버 동기화."""
import copy
import tempfile
import unittest
from pathlib import Path

from core.forge_memo_client import (
    MemoConflictError, MemoNotFoundError, MemoRejectedError, MemoRoutesUnavailable, RemoteMemos,
)
from core.memo_store import (
    CONFLICT_SUFFIX, MemoStore, is_newer, parse_timestamp, timestamps_equal,
)
from core.memo_sync import run_memo_sync
from tests.test_memo_store import Clock, Ids

SERVER = "http://127.0.0.1:7860"


class FakeForge:
    """공유 메모 계약의 서버 쪽을 메모리에서 흉내 낸다(409·삭제 표시·404 포함)."""

    def __init__(self, clock):
        self.clock = clock
        self.memos = {}
        self.revision = 0
        self.available = True
        self.writes = []
        self.before_put = None   # (memo_id) -> None — PUT 직전 다른 클라이언트의 쓰기를 흉내
        self.reject = set()      # 이 id 의 PUT 은 413/400 처럼 거절한다

    def list_memos(self, *, include_deleted=True):
        if not self.available:
            raise MemoRoutesUnavailable("old extension")
        memos = [copy.deepcopy(m) for m in self.memos.values() if include_deleted or not m["deleted"]]
        return RemoteMemos(self.revision, memos)

    def put_memo(self, memo_id, *, title, text, updated_at=None, base_updated_at=None, created_at=None):
        if memo_id in self.reject:
            raise MemoRejectedError("too large")
        if self.before_put is not None:
            hook, self.before_put = self.before_put, None
            hook(memo_id)
        stored = self.memos.get(memo_id)
        if (base_updated_at is not None and stored is not None
                and not timestamps_equal(stored["updated_at"], base_updated_at)
                and is_newer(stored["updated_at"], base_updated_at)):
            raise MemoConflictError("newer", copy.deepcopy(stored))
        now = updated_at if parse_timestamp(updated_at) else self.clock()
        self.memos[memo_id] = {
            "id": memo_id, "title": title, "text": text,
            "created_at": stored["created_at"] if stored else (created_at or now),
            "updated_at": now, "deleted": False,
        }
        self.revision += 1
        self.writes.append(("PUT", memo_id))
        return self.revision, copy.deepcopy(self.memos[memo_id])

    def delete_memo(self, memo_id, *, base_updated_at=None):
        stored = self.memos.get(memo_id)
        if stored is None:
            raise MemoNotFoundError("unknown")
        # 확장처럼: 본 판(base) 뒤에 고친 살아 있는 메모는 지우지 않는다(409)
        if (base_updated_at is not None and not stored["deleted"]
                and is_newer(stored["updated_at"], base_updated_at)):
            raise MemoConflictError("newer", copy.deepcopy(stored))
        stored.update(deleted=True, text="", updated_at=self.clock())
        self.revision += 1
        self.writes.append(("DELETE", memo_id))
        return self.revision, copy.deepcopy(stored)

    # 다른 기기(확장 UI)에서의 편집
    def edit(self, memo_id, text, title="t"):
        now = self.clock()
        old = self.memos.get(memo_id)
        self.memos[memo_id] = {"id": memo_id, "title": title, "text": text,
                               "created_at": old["created_at"] if old else now,
                               "updated_at": now, "deleted": False}
        self.revision += 1

    def remove(self, memo_id):
        self.memos[memo_id].update(deleted=True, text="", updated_at=self.clock())
        self.revision += 1


class SyncCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()   # 저장소와 서버가 같은 시계를 쓴다(선후가 분명하게)
        self.server = FakeForge(self.clock)
        self.store = self.make_store("a")

    def make_store(self, name):
        return MemoStore(Path(self.tmp.name) / f"{name}.json", clock=self.clock, id_factory=Ids())

    def sync(self, store=None):
        return run_memo_sync(store or self.store, self.server, SERVER)

    def local(self, memo_id, store=None):
        return (store or self.store).get(memo_id)

    def texts(self, store=None):
        return {m["id"]: m["text"] for m in (store or self.store).list_public()}


class BasicSyncTests(SyncCase):
    def test_first_sync_pushes_new_memos_and_marks_clean(self):
        self.store.save("m1", "제목", "본문")
        report = self.sync()
        self.assertEqual(report.pushed, 1)
        self.assertEqual(self.server.memos["m1"]["text"], "본문")
        # 클라이언트 updated_at 을 서버가 그대로 쓴다
        self.assertEqual(self.server.memos["m1"]["updated_at"], self.local("m1")["updated_at"])
        self.assertFalse(self.local("m1")["dirty"])
        self.assertEqual(self.store.sync_bases(SERVER)["m1"], self.server.memos["m1"]["updated_at"])
        self.assertIsNotNone(self.store.last_synced_at(SERVER))
        self.assertEqual(report.revision, self.server.revision)

    def test_second_sync_without_changes_writes_nothing(self):
        self.store.save("m1", "t", "x")
        self.sync()
        writes = list(self.server.writes)
        report = self.sync()
        self.assertEqual(self.server.writes, writes)
        self.assertEqual((report.pushed, report.pulled, report.deleted), (0, 0, 0))

    def test_remote_only_memo_is_adopted_but_unknown_tombstone_is_not(self):
        self.server.edit("r1", "서버 메모")
        self.server.edit("r2", "곧 지움")
        self.server.remove("r2")
        report = self.sync()
        self.assertEqual(report.pulled, 1)
        self.assertEqual(self.texts(), {"r1": "서버 메모"})
        self.assertIsNone(self.local("r2"))
        self.assertEqual(self.server.writes, [])

    def test_remote_edit_with_clean_local_wins(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        self.server.edit("m1", "v2 from forge")
        self.sync()
        self.assertEqual(self.texts(), {"m1": "v2 from forge"})
        self.assertFalse(self.local("m1")["dirty"])

    def test_local_edit_with_unchanged_remote_is_pushed(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        self.store.save("m1", "t", "v2 local", first["updated_at"])
        report = self.sync()
        self.assertEqual(report.pushed, 1)
        self.assertEqual(self.server.memos["m1"]["text"], "v2 local")

    def test_local_delete_becomes_remote_tombstone(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        self.store.delete("m1")
        report = self.sync()
        self.assertEqual(report.deleted, 1)
        self.assertTrue(self.server.memos["m1"]["deleted"])
        self.assertFalse(self.local("m1")["dirty"])
        self.assertEqual(self.texts(), {})
        self.assertEqual(self.sync().deleted, 0)

    def test_offline_create_then_delete_never_reaches_the_server(self):
        self.store.save("tmp", "t", "x")
        self.store.delete("tmp")
        self.sync()
        self.assertEqual(self.server.writes, [])
        self.assertFalse(self.local("tmp")["dirty"])

    def test_memo_lost_by_the_server_is_uploaded_again(self):
        self.store.save("m1", "t", "keep me")
        self.sync()
        self.server.memos.clear()   # Forge 데이터 폴더를 새로 만든 경우
        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "keep me")

    def test_created_at_travels_with_a_new_memo(self):
        self.store.save("m1", "t", "x")
        self.clock()   # 서버 시각이 앞서 있어도
        self.sync()
        self.assertEqual(self.server.memos["m1"]["created_at"], self.local("m1")["created_at"])

    def test_rejected_memo_stays_dirty_and_others_still_sync(self):
        self.store.save("big", "t", "x")
        self.store.save("ok", "t", "y")
        self.server.reject.add("big")
        report = self.sync()
        self.assertEqual([memo_id for memo_id, _ in report.rejected], ["big"])
        self.assertEqual(self.server.memos["ok"]["text"], "y")
        self.assertNotIn("big", self.server.memos)
        self.assertTrue(self.local("big")["dirty"])
        self.assertFalse(self.local("ok")["dirty"])
        self.assertIsNotNone(self.store.last_synced_at(SERVER))

    def test_old_extension_leaves_local_untouched(self):
        self.store.save("m1", "t", "x")
        self.server.available = False
        with self.assertRaises(MemoRoutesUnavailable):
            self.sync()
        self.assertTrue(self.local("m1")["dirty"])
        self.assertIsNone(self.store.last_synced_at(SERVER))


class ConflictTests(SyncCase):
    def test_both_edited_keeps_both(self):
        first = self.store.save("m1", "회의", "v1").memo
        self.sync()
        self.server.edit("m1", "forge edit", title="회의")
        self.store.save("m1", "회의", "app edit", first["updated_at"])
        report = self.sync()
        self.assertEqual(report.conflicts, ["m1"])
        self.assertEqual(self.local("m1")["text"], "forge edit")
        copies = [m for m in self.store.list_public() if m["id"] != "m1"]
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0]["title"], "회의" + CONFLICT_SUFFIX)
        self.assertEqual(copies[0]["text"], "app edit")
        # 충돌 사본은 같은 pass 에서 서버에도 올라간다
        self.assertEqual(self.server.memos[copies[0]["id"]]["text"], "app edit")
        self.assertFalse(self.store.has_dirty())
        self.assertEqual(self.sync().conflicts, [])

    def test_409_between_list_and_put_keeps_both(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        self.store.save("m1", "t", "app edit", first["updated_at"])
        self.server.before_put = lambda memo_id: self.server.edit(memo_id, "raced forge edit")
        report = self.sync()
        self.assertEqual(report.conflicts, ["m1"])
        self.assertEqual(self.server.memos["m1"]["text"], "raced forge edit")
        self.assertEqual(sorted(self.texts().values()), ["app edit", "raced forge edit"])
        self.assertFalse(self.store.has_dirty())

    def test_newer_remote_tombstone_beats_older_local_edit(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        self.store.save("m1", "t", "local edit", first["updated_at"])
        self.server.remove("m1")          # 시계상 로컬 편집보다 나중
        self.sync()
        self.assertEqual(self.texts(), {})
        self.assertTrue(self.local("m1")["deleted"])
        self.assertTrue(self.server.memos["m1"]["deleted"])

    def test_newer_local_edit_resurrects_remote_tombstone(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        self.server.remove("m1")
        self.store.save("m1", "t", "local edit wins", first["updated_at"])   # 삭제보다 나중
        self.sync()
        self.assertEqual(self.texts(), {"m1": "local edit wins"})
        self.assertFalse(self.server.memos["m1"]["deleted"])
        self.assertEqual(self.server.memos["m1"]["text"], "local edit wins")

    def test_newer_remote_edit_resurrects_local_tombstone(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        self.store.delete("m1")
        self.server.edit("m1", "forge kept it")
        self.sync()
        self.assertEqual(self.texts(), {"m1": "forge kept it"})

    def test_newer_local_tombstone_beats_older_remote_edit(self):
        self.store.save("m1", "t", "v1")
        self.sync()
        self.server.edit("m1", "forge edit")
        self.store.delete("m1")
        self.sync()
        self.assertTrue(self.server.memos["m1"]["deleted"])
        self.assertEqual(self.texts(), {})

    def test_user_typing_during_push_is_not_overwritten(self):
        first = self.store.save("m1", "t", "v1").memo
        self.sync()
        second = self.store.save("m1", "t", "v2", first["updated_at"]).memo

        def typed_while_sending(memo_id):
            self.store.save(memo_id, "t", "v3 typed meanwhile", second["updated_at"])
        self.server.before_put = typed_while_sending
        self.sync()
        self.assertEqual(self.server.memos["m1"]["text"], "v2")
        self.assertEqual(self.local("m1")["text"], "v3 typed meanwhile")
        self.assertTrue(self.local("m1")["dirty"])
        report = self.sync()                 # 다음 동기화는 충돌 없이 이어서 올린다
        self.assertEqual(report.conflicts, [])
        self.assertEqual(self.server.memos["m1"]["text"], "v3 typed meanwhile")


class TwoClientTests(SyncCase):
    def test_two_apps_converge_through_forge(self):
        other = self.make_store("b")
        self.store.save("a1", "A", "from A")
        self.sync()
        run_memo_sync(other, self.server, SERVER)
        self.assertEqual(self.texts(other), {"a1": "from A"})
        base = other.get("a1")["updated_at"]
        other.save("a1", "A", "edited on B", base)
        other.save("b1", "B", "from B")
        run_memo_sync(other, self.server, SERVER)
        self.sync()
        self.assertEqual(self.texts(), {"a1": "edited on B", "b1": "from B"})
        self.store.delete("b1")
        self.sync()
        run_memo_sync(other, self.server, SERVER)
        self.assertEqual(self.texts(other), {"a1": "edited on B"})
        self.assertEqual(self.texts(), self.texts(other))


if __name__ == "__main__":
    unittest.main()
